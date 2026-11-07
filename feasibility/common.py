"""Serialization, synchronized timing, and reproducible selections."""
from pathlib import Path
import contextlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "feasibility"
REVISION = "m0416-feasibility-v1"
SETTINGS = dict(seed=20261031, warmup=500, samples=1000, chains=2,
                target_accept=.90, max_tree_depth=8, starts=3, maxiter=2000,
                laplace_draws=10000, field_draws=100, area_draws=32,
                bright_fraction=.30, near_arcsec=10., sigma_obs=.3,
                stationarity_tolerance=1e-3, polishing_gtol=5e-4,
                reference_family="0141", z_lens=.396, H0=70., Om0=.3, w=-1.,
                trial_redshifts=[.5, 1., 1.5, 2., 3., 4., 7., 10.])

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2) + "\n")
    tmp.replace(path)

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default

def write_npz(path,**arrays):
    """Atomic numerical checkpoint, including endpoints before reporting."""
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)

def ready(x):
    import jax
    return jax.block_until_ready(x)

def timed_call(fn, *args):
    t = time.perf_counter(); result = ready(fn(*args))
    return result, time.perf_counter() - t

def compile_call(fn, *args):
    """Explicit compile timing, separate from first synchronized execution."""
    t = time.perf_counter(); executable = fn.lower(*args).compile()
    compile_seconds = time.perf_counter() - t
    value, seconds = timed_call(executable, *args)
    return executable, value, dict(compile_seconds=compile_seconds, execution_seconds=seconds)

def indices(n, count):
    return np.linspace(0, n - 1, min(n, count), dtype=int)

def require_spd(matrix, relative_tol=1e-10):
    a = np.asarray(matrix, float); a = (a + a.T) / 2
    if not np.isfinite(a).all(): raise ValueError("Nonfinite curvature")
    eigen = np.linalg.eigvalsh(a)
    if eigen[0] <= relative_tol * max(abs(eigen[-1]), 1.):
        raise ValueError(f"Curvature not numerically positive definite: min={eigen[0]:.6g}, max={eigen[-1]:.6g}")
    chol = np.linalg.cholesky(a)
    inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(a))))
    return inv, eigen

def correlation(cov):
    scale = np.sqrt(np.diag(cov))
    return cov / np.outer(scale, scale)

@contextlib.contextmanager
def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(fname=ROOT / "src/lenscluster/style.mplstyle",
                        rc={"axes.grid": False}):
        yield plt
