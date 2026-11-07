"""Quick screen of the NUTS geometry changes of 2026-11-04 on AS1063 and AS1063_JWST: short chains (150 + 100 draws, 4
chains, no plots), every variant against its baseline with the same seed.  It tells which changes may help, not the final
posterior.

    python scripts/geometry_quick_tests.py                 # run all variants, at most --parallel at a time
    python scripts/geometry_quick_tests.py --summarize     # table of the sampler diagnostics from the logs

Each run claims chains x 2 cores on Linux (run.xsh pins them, skipping cores other runs hold), so on 32 cores four
runs fit side by side; starts are staggered so that each run sees the cores the previous one claimed.  Outputs go to
results/<campaign>/<cluster>/<label>/, logs to results/<campaign>/logs/<label>.log.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# label, cluster, extra run.xsh flags, what it tests
VARIANTS = [
    # Defaults since 2026-11-04: dense block = globals + centered members, Student-t tails, log-normal BCG cut prior.
    ("h0_baseline", "AS1063", ["--dense-offsets", "all", "--offset-tail", "normal"], "old defaults: all offsets dense, normal tails"),
    ("h1_dense_centered", "AS1063", ["--dense-offsets", "centered", "--offset-tail", "normal"], "dense block: globals + centered (data-pinned) members only"),
    ("h4_dense_exact", "AS1063", ["--dense-offsets", "exact"], "far-field offsets sampled outside the dense block"),
    ("h5_far_fixed", "AS1063", ["--far-offsets", "fixed"], "far-field offsets pinned at the scaling relation"),
    ("h2_tail_student", "AS1063", ["--offset-tail", "student_t"], "Student-t(4) deviations against normal ones"),
    # A scale-mixture form of the Student-t (normal x sqrt(lambda)) was tested on 2026-11-04 and dropped: per-member
    # (offset, lambda) funnels, 911-dimensional dense block, chain disagreement 55 sd against 36.
    ("h7_centered_far_fixed", "AS1063", ["--dense-offsets", "centered", "--far-offsets", "fixed"],
     "centered-only dense block and far-field offsets pinned"),
    ("j0_bcg_uniform", "AS1063_JWST", ["--bcg-cut-prior", "uniform"], "BCG 1809 cut radius: old bounded uniform"),
    ("j1_bcg_lognormal", "AS1063_JWST", [], "BCG 1809 cut radius: unbounded log-normal around the relation"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", default="nov4d_quick")
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--parallel", type=int, default=4, help="runs at the same time (each takes chains x 2 cores on Linux)")
    p.add_argument("--stagger", type=float, default=90.0, help="seconds between launches, so each run sees the cores already claimed")
    p.add_argument("--only", default="", help="comma list of labels to run (default all)")
    p.add_argument("--xonsh", default=str(Path(sys.executable).with_name("xonsh")), help="xonsh of the environment")
    p.add_argument("--summarize", action="store_true", help="only print the comparison table from existing logs")
    return p.parse_args()


def run_all(args):
    logs = ROOT / "results" / args.campaign / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    queue = [v for v in VARIANTS if wanted is None or v[0] in wanted]
    running: list[tuple[str, subprocess.Popen]] = []
    last_start = 0.0
    while queue or running:
        for label, proc in list(running):
            if proc.poll() is not None:
                print(f"[{time.strftime('%H:%M')}] {label} finished with exit code {proc.returncode}", flush=True)
                running.remove((label, proc))
        if queue and len(running) < args.parallel and time.time() - last_start >= args.stagger:
            label, cluster, extra, what = queue.pop(0)
            command = [args.xonsh, "run.xsh", cluster, "--campaign", args.campaign, "--run-label", label,
                       "--warmup", str(args.warmup), "--samples", str(args.samples), "--skip-plots", *extra]
            log = open(logs / f"{label}.log", "w")
            running.append((label, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
            last_start = time.time()
            print(f"[{time.strftime('%H:%M')}] started {label}: {what}", flush=True)
        time.sleep(5)
    summarize(args)


def final_log(campaign, cluster, label):
    return ROOT / "results" / campaign / cluster.lower() / label / "final" / "logs" / "final.log"


def numbers(pattern, text, cast=float):
    return [cast(v) for v in re.findall(pattern, text)]


def bcg_chains(campaign, cluster, label, name="bcg_1809.cut_radius_kpc"):
    """Per-chain medians of a parameter from the saved posterior (physical units)."""
    try:
        import numpy as np
        sys.path.insert(0, str(ROOT / "scripts"))
        from fit_arc_source import load_model                         # the state with the run's dropped members and classes
        _, state, _, post = load_model(ROOT / "results" / campaign / cluster.lower() / label)
        names = [s.name for s in state.parameter_specs]
        values = np.asarray(post.physical, float)[:, names.index(name)].reshape(int(post.num_chains), -1)
        return "/".join(f"{np.median(v):.0f}" for v in values)
    except Exception as error:                                         # the table must not fail on one run
        return f"({type(error).__name__})"


def summarize(args):
    rows = []
    for label, cluster, _, what in VARIANTS:
        path = final_log(args.campaign, cluster, label)
        if not path.exists():
            rows.append(dict(label=label, note="no final log"))
            continue
        text = path.read_text(errors="replace")
        row = dict(label=label)
        done = re.search(r"\[nuts\] done in ([\d.]+)s accept_mean=([\d.]+) divergences=(\d+) mean_steps=([\d.]+)", text)
        if done:
            row.update(seconds=float(done.group(1)), accept=float(done.group(2)), divergences=int(done.group(3)), steps=float(done.group(4)))
        steps = numbers(r"step size ([\d.e+-]+), E-BFMI", text)
        cap = numbers(r"(\d+)% of transitions at the tree-depth cap", text)
        ebfmi = numbers(r"E-BFMI ([\d.]+)", text)
        row.update(step=min(steps) if steps else None, at_cap=max(cap) if cap else None, ebfmi=min(ebfmi) if ebfmi else None)
        dense = re.search(r"mass seed global_dim=(\d+)", text)
        row["dense_dim"] = int(dense.group(1)) if dense else None
        cond = numbers(r"condition ([\d.e+-]+), half an orbit", text)
        orbit = numbers(r"half an orbit needs (\d+) leapfrog", text)
        frac = numbers(r"= ([\d.]+) of it; stiffest", text)
        row.update(condition=max(cond) if cond else None, half_orbit=max(orbit) if orbit else None, step_frac=max(frac) if frac else None)
        slow = re.search(r"slow mode 1: chain means ([\d.]+) within-chain sd apart", text)
        row["slow_sd"] = float(slow.group(1)) if slow else None
        for kind in ("globals", "exact scatter offset", "far-field scatter offset"):
            m = re.search(rf"convergence {re.escape(kind)}\s+n=\s*\d+: R-hat median ([\d.]+) max ([\d.]+), ESS min (\d+) median (\d+)", text)
            if m:
                short = {"globals": "glob", "exact scatter offset": "exact", "far-field scatter offset": "far"}[kind]
                row[f"rhat_{short}"] = float(m.group(2)); row[f"ess_{short}"] = int(m.group(4))
        row["bounds"] = "; ".join(re.findall(r"bound pressure: (\S+) has", text)) or ""
        if cluster == "AS1063_JWST":
            row["bcg1809_rcut"] = bcg_chains(args.campaign, cluster, label)
        rows.append(row)
    try:
        import pandas as pd
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
        print(pd.DataFrame(rows).set_index("label").round(3).to_string())
    except ImportError:
        for r in rows:
            print(r)
    print("\nRead: ESS per second and slow_sd (chain disagreement, smaller is better) are the headline; at_cap and half_orbit"
          "\nshow whether trajectories are still too short; dense_dim the size of the dense mass block; rhat_* and ess_* by group.")


def main():
    args = parse_args()
    if args.summarize:
        summarize(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
