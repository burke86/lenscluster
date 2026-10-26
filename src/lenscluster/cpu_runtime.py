"""Bound CPU pools before JAX initializes its backend.

The CPU budget is shared by JAX's parallel chains, not a private thread pool
or a pinning assignment for each chain. Linux affinity bounds the worker pools;
BLAS stays single-threaded to avoid nested parallelism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
import os
from pathlib import Path
import sys

_BLAS_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
_ACTIVE_PLAN = None
_THREADPOOL_LIMITER = None


@dataclass(frozen=True)
class CPUPlan:
    chains: int
    chain_method: str
    threads_per_chain: int | None
    jax_cpu_devices: int
    available_cpu_ids: tuple[int, ...]
    selected_cpu_ids: tuple[int, ...]
    affinity_supported: bool
    blas_threads: int | None

    def to_dict(self):
        return asdict(self)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _core_key(cpu):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        return ((root / "physical_package_id").read_text().strip(), (root / "core_id").read_text().strip())
    except OSError:
        return ("unknown", str(cpu))


def _physical_core_order(cpu_ids):
    """Use one allowed logical CPU per physical core before SMT siblings."""
    groups = {}
    for cpu in sorted(cpu_ids):
        groups.setdefault(_core_key(cpu), []).append(cpu)
    return tuple(group[i] for i in range(max(map(len, groups.values()), default=0))
                 for group in groups.values() if i < len(group))


def _parse_cpu_list(text):
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpus_claimed_by_other_runs(markers=("run.xsh", "lenscluster")):
    """Logical CPUs that other solver processes on this machine have pinned themselves to (Linux /proc)."""
    claimed = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return claimed
    me = os.getpid()
    total = os.cpu_count() or 1
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if not any(marker in cmdline for marker in markers):
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus = _parse_cpu_list(line.split(":", 1)[1])
                    if len(cpus) < total:                 # a pinned process, not one allowed everywhere
                        claimed |= cpus
        except OSError:
            continue
    return claimed


def cpu_busy_fractions(interval=0.25):
    """Fraction of the last ``interval`` seconds each logical CPU was busy, from two /proc/stat samples (Linux only)."""
    import time

    def sample():
        rows = {}
        try:
            for line in Path("/proc/stat").read_text().splitlines():
                if line.startswith("cpu") and line[3:4].isdigit():
                    fields = [int(v) for v in line.split()[1:]]
                    rows[int(line[3:].split()[0])] = (fields[3] + fields[4], sum(fields))     # idle + iowait, total
        except OSError:
            pass
        return rows

    first = sample()
    if not first:
        return {}
    time.sleep(interval)
    second = sample()
    busy = {}
    for cpu, (idle0, total0) in first.items():
        idle1, total1 = second.get(cpu, (idle0, total0))
        elapsed = total1 - total0
        busy[cpu] = 1.0 - (idle1 - idle0) / elapsed if elapsed > 0 else 0.0
    return busy


def resolve_cpu_plan(chains, chain_method="parallel", threads_per_chain=None, *, avoid=None, load=None):
    """Choose the CPUs of this run.  XLA sizes its thread pool from the affinity mask, so with ``threads_per_chain``
    the run is pinned to concurrent * threads_per_chain physical cores.  Those are the least busy cores at launch
    (``load``: busy fraction per CPU, by default sampled from /proc/stat), excluding cores already claimed by other
    solver runs (``avoid``; by default read from /proc), so that several fits and other programs share the machine
    without stacking on the same cores."""
    chains = _positive_integer("chains", chains)
    if chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("Unsupported NUTS chain method for CPU setup.")
    if threads_per_chain is not None:
        threads_per_chain = _positive_integer("cpu_threads_per_chain", threads_per_chain)
    # A repeated request for the configured plan returns it: the core choice depends on the load at launch and on
    # our own narrowed affinity mask, so it must not be recomputed once applied.
    if _ACTIVE_PLAN is not None and (_ACTIVE_PLAN.chains, _ACTIVE_PLAN.chain_method, _ACTIVE_PLAN.threads_per_chain) == (chains, chain_method, threads_per_chain):
        return _ACTIVE_PLAN
    supported = hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
    available = tuple(sorted(os.sched_getaffinity(0))) if supported else tuple(range(os.cpu_count() or 1))
    if _ACTIVE_PLAN is not None and available == tuple(sorted(_ACTIVE_PLAN.selected_cpu_ids)):
        available = _ACTIVE_PLAN.available_cpu_ids
    concurrent = 1 if chain_method == "sequential" else chains
    count = len(available) if threads_per_chain is None else min(len(available), concurrent * threads_per_chain)
    if threads_per_chain is None:
        selected = available
    else:
        claimed = set(cpus_claimed_by_other_runs() if avoid is None else avoid)
        free = tuple(cpu for cpu in available if cpu not in claimed)
        pool = free if len(_physical_core_order(free)) >= count else available   # not enough unclaimed cores: share
        busy = cpu_busy_fractions() if load is None else dict(load)
        # A physical core is as busy as its busiest logical CPU, siblings pinned by other runs included; one logical CPU
        # per core first (as _physical_core_order lists them), least busy core first, stable for equal loads.
        siblings = {}
        for cpu in available:
            siblings.setdefault(_core_key(cpu), []).append(cpu)
        core_load = lambda cpu: max(busy.get(c, 0.0) for c in siblings[_core_key(cpu)])
        ordered = _physical_core_order(pool)
        primary = ordered[:len({_core_key(cpu) for cpu in pool})]
        primary = tuple(sorted(primary, key=lambda cpu: (round(core_load(cpu), 2), cpu)))
        selected = (primary + tuple(cpu for cpu in ordered if cpu not in primary))[:count]
    return CPUPlan(chains, chain_method, threads_per_chain,
                   chains if chain_method == "parallel" else 1,
                   available, selected, supported, 1 if threads_per_chain is not None else None)


def configure_cpu_runtime(chains, chain_method="parallel", threads_per_chain=None):
    """Apply once before backend initialization; spawned stages inherit it.

    Existing JAX backends cannot resize their pools. Reject late configuration
    instead of silently presenting ineffective environment settings as tuning.
    """
    global _ACTIVE_PLAN, _THREADPOOL_LIMITER
    plan = resolve_cpu_plan(chains, chain_method, threads_per_chain)
    if _ACTIVE_PLAN == plan:
        return plan
    if _ACTIVE_PLAN is not None:
        raise RuntimeError("CPU setup is already configured differently; start a fresh process to change it.")
    jax = sys.modules.get("jax")
    if jax is not None:
        from jax._src import xla_bridge
        if xla_bridge.backends_are_initialized():
            raise RuntimeError(
                "CPU setup must run before JAX backend initialization. Start a fresh process "
                "and call configure_cpu_runtime with the resolved runtime configuration "
                "before cosmology calculations or inference."
            )
    if threads_per_chain is not None and not plan.affinity_supported:
        raise RuntimeError("cpu_threads_per_chain requires CPU affinity support; use None on this platform.")
    if threads_per_chain is not None:
        os.sched_setaffinity(0, plan.selected_cpu_ids)
        for name in _BLAS_VARIABLES:
            os.environ[name] = "1"
        # NumPy/SciPy may already be imported while constructing typed configs.
        # Limit those loaded libraries as well as future spawned processes.
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
        if "numexpr" in sys.modules:
            sys.modules["numexpr"].set_num_threads(1)
    os.environ["JAX_NUM_CPU_DEVICES"] = str(plan.jax_cpu_devices)
    if jax is not None:
        jax.config.update("jax_num_cpu_devices", plan.jax_cpu_devices)
    _ACTIVE_PLAN = plan
    return plan
