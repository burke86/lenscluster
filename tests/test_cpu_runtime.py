"""CPU planning and fresh-process backend/worker setup, without lens inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lenscluster import cpu_runtime as cpu


@pytest.mark.parametrize("chains", [1, 2, 4, 8])
@pytest.mark.parametrize("method", ["parallel", "sequential", "vectorized"])
def test_budget_follows_concurrent_chain_count(monkeypatch, chains, method):
    monkeypatch.setattr(cpu, "_ACTIVE_PLAN", None)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {8, 9, 10, 11, 12, 13})
    monkeypatch.setattr(cpu, "_physical_core_order", lambda ids: tuple(ids))
    plan = cpu.resolve_cpu_plan(chains, method, 2)
    expected = min(6, 2 * (1 if method == "sequential" else chains))
    assert len(plan.selected_cpu_ids) == expected
    assert set(plan.selected_cpu_ids) <= {8, 9, 10, 11, 12, 13}
    assert plan.chains == chains  # Never silently reduce chains to fit CPUs.
    assert plan.jax_cpu_devices == (chains if method == "parallel" else 1)
    assert plan.blas_threads == 1


def test_none_preserves_available_allocation(monkeypatch):
    monkeypatch.setattr(cpu, "_ACTIVE_PLAN", None)
    monkeypatch.setattr(cpu, "affinity_supported", lambda: True)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {5, 7, 19}, raising=False)
    plan = cpu.resolve_cpu_plan(2, threads_per_chain=None)
    assert plan.selected_cpu_ids == (5, 7, 19)
    assert plan.blas_threads == 1


@pytest.mark.parametrize("chains,method,expected", [(4, "parallel", 1), (1, "parallel", None), (4, "sequential", None)])
def test_parallel_blas_limit_without_affinity(monkeypatch, chains, method, expected):
    monkeypatch.setattr(cpu, "_ACTIVE_PLAN", None)
    monkeypatch.setattr(cpu, "affinity_supported", lambda: False)
    plan = cpu.resolve_cpu_plan(chains, method, None)
    assert plan.selected_cpu_ids == plan.available_cpu_ids
    assert plan.blas_threads == expected


def test_parallel_backend_without_affinity_limits_loaded_and_future_blas():
    code = """
import os
import numpy as np
from threadpoolctl import threadpool_info
from lenscluster import cpu_runtime as cpu
cpu.affinity_supported = lambda: False
plan = cpu.configure_cpu_runtime(4, 'parallel', None)
assert plan.blas_threads == 1
assert all(os.environ[name] == '1' for name in cpu._BLAS_VARIABLES)
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
a = np.array([[4., 1.], [1., 3.]])
result = jax.pmap(jnp.linalg.cholesky)(jnp.asarray(np.stack([a] * 4)))
np.testing.assert_allclose(result, np.stack([np.linalg.cholesky(a)] * 4), rtol=1e-12)
blas = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
assert blas and set(blas) == {1}
assert cpu.configure_cpu_runtime(4, 'parallel', None) == plan
"""
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=60)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_reject_invalid_thread_budgets(value):
    with pytest.raises(ValueError, match="positive integer"):
        cpu.resolve_cpu_plan(4, threads_per_chain=value)


def test_prefer_physical_cores_within_allowed_mask(monkeypatch):
    def read(path):
        logical = int(str(path).split('/cpu/cpu')[1].split('/')[0])
        return '0' if path.name == 'physical_package_id' else str(logical % 2)
    monkeypatch.setattr(Path, "read_text", read)
    assert cpu._physical_core_order([0, 1, 2, 3]) == (0, 1, 2, 3)
    assert cpu._physical_core_order([0, 2, 3]) == (0, 3, 2)


@pytest.mark.parametrize("chains", [1, 2, 4])
def test_fresh_backend_and_spawned_worker_follow_resolved_chains(tmp_path, chains):
    script = tmp_path / "cpu_spawn_check.py"
    script.write_text('''
import json, multiprocessing, os
from lenscluster.cpu_runtime import configure_cpu_runtime
def inspect_backend():
    import jax
    import jax.numpy as jnp
    from threadpoolctl import threadpool_info
    n = len(jax.devices('cpu'))
    values = jax.pmap(lambda x: x*x + 1)(jnp.arange(n))
    return dict(devices=n, cpus=sorted(os.sched_getaffinity(0)),
                values=values.tolist(), blas=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'])
def child(q):
    q.put(inspect_backend())
if __name__ == '__main__':
    import sys
    chains = int(sys.argv[1])
    plan = configure_cpu_runtime(chains, 'parallel', 1)
    parent = inspect_backend()
    assert configure_cpu_runtime(chains, 'parallel', 1) == plan
    try:
        configure_cpu_runtime(chains + 1, 'parallel', 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('reconfiguration should require a fresh process')
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue(); p = ctx.Process(target=child, args=(q,))
    p.start(); result = q.get(timeout=45); p.join(timeout=45)
    assert p.exitcode == 0
    print(json.dumps(dict(plan=plan.to_dict(), parent=parent, child=result)))
''')
    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    # Deliberately start with a different device count: typed config wins.
    env["JAX_NUM_CPU_DEVICES"] = "7"
    result = subprocess.run([sys.executable, str(script), str(chains)], env=env,
                            capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(result.stdout)
    assert report["parent"] == report["child"]
    assert report["parent"]["devices"] == chains
    assert report["parent"]["values"] == [i*i+1 for i in range(chains)]
    assert len(report["parent"]["cpus"]) == min(chains, len(report["plan"]["available_cpu_ids"]))
    assert report["parent"]["blas"] and set(report["parent"]["blas"]) == {1}


def test_late_backend_setup_fails_actionably():
    code = """
import jax
jax.devices('cpu')
from lenscluster.cpu_runtime import configure_cpu_runtime
try:
    configure_cpu_runtime(1, 'parallel', 1)
except RuntimeError as error:
    assert 'before JAX backend initialization' in str(error)
else:
    raise AssertionError('late setup was silently accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45,
                   env=dict(os.environ, PYTHONPATH=str(Path("src").resolve())))


@pytest.mark.parametrize("cluster", ["HERA", "AS1063"])
def test_launcher_uses_resolved_runtime_before_cosmology(cluster):
    code = """
from dataclasses import replace
from pathlib import Path
import os, sys
path = Path('run.xsh').resolve()
namespace = {'__file__': str(path), '__name__': 'cpu_launcher_test'}
source = '\\n'.join(line for line in path.read_text().splitlines() if not line.startswith('$'))
exec(compile(source, str(path), 'exec'), namespace)
from jax._src import xla_bridge
assert not xla_bridge.backends_are_initialized(), 'launcher initialized JAX before CPU setup'
build = namespace['build_runtime']
namespace['build_runtime'] = lambda cores: replace(build(cores), chains=2, cpu_threads_per_chain=2)
def check_run(config, stage="both"):
    import jax
    assert config.runtime.chains == 2
    assert len(jax.devices('cpu')) == 2
    assert len(os.sched_getaffinity(0)) <= 4
import lenscluster.pipeline as pipeline
pipeline.run = check_run
assert namespace['main'](['run.xsh', sys.argv[1]]) == 0
"""
    subprocess.run([sys.executable, "-c", code, cluster], check=True, timeout=45,
                   env=dict(os.environ, PYTHONPATH=str(Path("src").resolve())))


def test_plan_prefers_idle_cores():
    import lenscluster.cpu_runtime as cpu
    idle = cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(), load={})
    if len(idle.available_cpu_ids) < 2 * len(idle.selected_cpu_ids):
        pytest.skip("not enough CPUs to test load-aware pinning")
    busy = {c: 1.0 for c in idle.selected_cpu_ids}                 # the default choice is fully loaded
    shifted = cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(), load=busy)
    assert not set(shifted.selected_cpu_ids) & set(idle.selected_cpu_ids)
    assert len(shifted.selected_cpu_ids) == len(idle.selected_cpu_ids)
    assert cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(), load={}) == idle   # deterministic on an idle machine


def test_plan_skips_cores_claimed_by_other_runs():
    import lenscluster.cpu_runtime as cpu
    base = cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(), load={})
    if len(base.available_cpu_ids) < 2 * len(base.selected_cpu_ids):
        pytest.skip("not enough CPUs to test disjoint pinning")
    other = cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(base.selected_cpu_ids), load={})
    assert not set(other.selected_cpu_ids) & set(base.selected_cpu_ids)
    assert len(other.selected_cpu_ids) == len(base.selected_cpu_ids)
    # When every core is claimed the plan falls back to sharing rather than failing.
    crowded = cpu.resolve_cpu_plan(4, "parallel", 2, avoid=set(base.available_cpu_ids), load={})
    assert len(crowded.selected_cpu_ids) == len(base.selected_cpu_ids)
    assert cpu._parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
