"""Benchmark: how many CPU threads per NUTS chain make the likelihood gradient fastest on this machine.

NUTS runs its chains side by side, each taking one gradient per leapfrog step, so what matters is (a) the time of one
gradient for a chain (it sets the wall time of the run) and (b) the gradients per second of all chains together.  This
script measures both for every (chains, threads per chain) layout that fits the machine, for the system-block and the
all-block covariances.  Each chain is a separate worker process pinned to its own cores (Linux), with XLA's thread pool
and the BLAS pools set to that many threads before JAX starts; all workers compile first, then time together.

It needs a saved init of the cluster (a run directory with init/artifacts, from `xonsh run.xsh ... --stage init`):

    python scripts/bench_threads_per_chain.py results/<campaign>/m0416_canucs2025/<run>
    python scripts/bench_threads_per_chain.py <run> --chains 4 8 --threads 1 2 4 8 --blocks system all --seconds 30

Results: a table on screen and a CSV (default <run>/bench_threads_per_chain.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------- worker (one chain)
def worker(args) -> None:
    threads, blocks = int(args.threads[0]), str(args.blocks[0])
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else []
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)                  # XLA sizes its thread pool from the affinity mask
    flags = ["--xla_cpu_multi_thread_eigen=" + ("true" if threads > 1 else "false"), f"intra_op_parallelism_threads={threads}"]
    os.environ["XLA_FLAGS"] = " ".join(flags + [os.environ.get("XLA_FLAGS", "")]).strip()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["JAX_NUM_CPU_DEVICES"] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace
    from lenscluster import artifacts
    from lenscluster.evaluator import Evaluator
    from lenscluster.state import build_state, with_member_classes

    init = Path(args.run) / "init" / "artifacts"
    config = artifacts.load_config(init)
    config = replace(config, likelihood=replace(config.likelihood, gp_blocks=blocks))
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered = set(classes.get("centered_members", []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids], dtype=bool))
    theta = jnp.asarray(artifacts.load_start(init))
    gradient = jax.jit(jax.value_and_grad(Evaluator(state, config.likelihood.likelihood_mode).log_likelihood))
    value, grad = gradient(theta)
    grad.block_until_ready()
    if not np.isfinite(float(value)):
        raise SystemExit(f"non-finite log-likelihood at the saved start under {blocks} blocks")

    sync = Path(args.sync)
    (sync / f"ready_{args.index}").touch()
    while not (sync / "go").exists():                  # wait until every worker has compiled
        time.sleep(0.05)
    times = []
    deadline = time.perf_counter() + float(args.seconds)
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        gradient(theta)[1].block_until_ready()
        times.append(time.perf_counter() - t0)
    (sync / f"result_{args.index}.json").write_text(json.dumps({"times": times}))


# ----------------------------------------------------------------------------------------------- driver
def core_groups(n_chains: int, threads: int) -> list[list[int]]:
    """Disjoint CPU sets, one per chain: physical cores first (one logical CPU per core), then SMT siblings."""
    if not hasattr(os, "sched_getaffinity"):
        return [[] for _ in range(n_chains)]           # no pinning on this platform (macOS): the OS schedules
    sys.path.insert(0, str(ROOT / "src"))
    from lenscluster.cpu_runtime import _physical_core_order
    order = list(_physical_core_order(sorted(os.sched_getaffinity(0))))
    return [order[k * threads:(k + 1) * threads] for k in range(n_chains)]


def run_layout(run: Path, blocks: str, chains: int, threads: int, seconds: float, python: str) -> dict:
    groups = core_groups(chains, threads)
    with tempfile.TemporaryDirectory(prefix="bench_threads_") as sync:
        procs = [subprocess.Popen([python, __file__, str(run), "--worker", "--blocks", blocks, "--threads", str(threads),
                                   "--cpus", ",".join(map(str, g)), "--index", str(k), "--sync", sync, "--seconds", str(seconds)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                 for k, g in enumerate(groups)]
        while sum((Path(sync) / f"ready_{k}").exists() for k in range(chains)) < chains:
            if any(p.poll() not in (None, 0) for p in procs):
                errors = [p.stderr.read()[-2000:] for p in procs if p.poll() not in (None, 0)]
                for p in procs:
                    p.kill()
                raise RuntimeError(f"a worker failed ({blocks}, {chains} x {threads}):\n{errors[0]}")
            time.sleep(0.2)
        (Path(sync) / "go").touch()
        for p in procs:
            p.wait()
        import numpy as np
        per_chain = [json.loads((Path(sync) / f"result_{k}.json").read_text())["times"] for k in range(chains)]
    medians = [float(np.median(t)) for t in per_chain]
    counts = [len(t) for t in per_chain]
    return dict(blocks=blocks, chains=chains, threads_per_chain=threads, cores_used=chains * threads,
                ms_per_gradient=1e3 * float(np.median(medians)), ms_worst_chain=1e3 * float(max(medians)),
                gradients_per_second=float(sum(counts) / seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="run directory containing init/artifacts")
    parser.add_argument("--chains", type=int, nargs="+", default=[4, 8], help="concurrent chains (default 4 8)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="threads per chain (default 1 2 4 8)")
    parser.add_argument("--blocks", nargs="+", default=["system", "all"], choices=["system", "all", "family"])
    parser.add_argument("--seconds", type=float, default=20.0, help="timing window per layout after compilation (default 20)")
    parser.add_argument("--out", type=Path, default=None, help="CSV path (default <run>/bench_threads_per_chain.csv)")
    # worker mode (internal)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("--index", "--sync", "--cpus"):
        parser.add_argument(name, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.run is None or not (args.run / "init" / "artifacts").is_dir():
        parser.error("give a run directory that contains init/artifacts")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    layouts = [(c, t) for c in args.chains for t in args.threads if c * t <= available]
    skipped = [(c, t) for c in args.chains for t in args.threads if c * t > available]
    print(f"{available} CPUs available; {len(layouts)} layouts per covariance" + (f" (skipped, too many cores: {skipped})" if skipped else ""))
    rows = []
    for blocks in args.blocks:
        for chains, threads in layouts:
            started = time.perf_counter()
            row = run_layout(args.run, blocks, chains, threads, args.seconds, sys.executable)
            rows.append(row)
            print(f"  {blocks:6s} {chains:3d} chains x {threads:2d} threads: {row['ms_per_gradient']:7.1f} ms per gradient "
                  f"(worst chain {row['ms_worst_chain']:.1f}), {row['gradients_per_second']:7.1f} gradients/s in total "
                  f"[{time.perf_counter() - started:.0f} s]", flush=True)
    # Speed-up of each layout against one thread per chain at the same number of chains.
    base = {(r["blocks"], r["chains"]): r["ms_per_gradient"] for r in rows if r["threads_per_chain"] == 1}
    for r in rows:
        ref = base.get((r["blocks"], r["chains"]))
        r["speedup_vs_1_thread"] = (ref / r["ms_per_gradient"]) if ref else float("nan")
    out = args.out or (args.run / "bench_threads_per_chain.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{'blocks':7s} {'chains':>6s} {'thr/chain':>9s} {'ms/grad':>8s} {'speedup':>8s} {'grad/s total':>13s}")
    for r in rows:
        print(f"{r['blocks']:7s} {r['chains']:6d} {r['threads_per_chain']:9d} {r['ms_per_gradient']:8.1f} "
              f"{r['speedup_vs_1_thread']:8.2f} {r['gradients_per_second']:13.1f}")
    print(f"\nwrote {out}.  The run's wall time follows ms/grad (each chain steps on its own); pick the threads per chain\n"
          f"with the lowest ms/grad for the number of chains you run, as long as chains x threads fits the machine.")


if __name__ == "__main__":
    main()
