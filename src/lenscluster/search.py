"""Multi-start MAP of the global block in NumPyro's unconstrained coordinates, with mode clustering.

Three optimizers share the same reduced objective (member deviations and source offsets held at
zero): scipy L-BFGS-B per start, vmapped Adam, and vmapped mean-field SVI.  The best modes are
then polished in full dimension with L-BFGS-B so the Hessian mass seed is taken at the MAP.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from numpyro.infer import Predictive
from numpyro.infer.util import constrain_fn, potential_energy
from scipy.optimize import minimize

from .config import SearchConfig
from .evaluator import Evaluator
from .model import ModelState
from .utils import progress
from .sampling import (LATENT_FAMILIES, posterior_model, prior_model, theta_from_site_values,
                       unconstrained_from_theta)

HYPERPARAMETER_FAMILIES = frozenset({"member_velocity_scatter", "source_scatter"})
LATENT_SITES = ("member_velocity_offsets", "member_radius_offsets", "source_position_beta")
BAD_POTENTIAL = 1.0e29                      # potentials at or above this come from the invalid-model sentinel


def prior_starts(state: ModelState, n_starts: int, seed: int) -> np.ndarray:
    """Latent starts: lens parameters drawn from the prior; offsets, sources and hyperparameters at their centres."""
    specs = state.parameter_specs
    draws = Predictive(prior_model(specs), num_samples=int(n_starts))(jax.random.PRNGKey(int(seed)))
    theta = theta_from_site_values({k: np.asarray(v) for k, v in draws.items()}, specs)
    reference = np.asarray(state.initial_theta, dtype=float)
    for idx, spec in enumerate(specs):
        if spec.component_family in LATENT_FAMILIES:
            theta[:, idx] = 0.0
        elif spec.component_family in HYPERPARAMETER_FAMILIES:
            theta[:, idx] = reference[idx]
    theta[0] = reference                     # the configured model is always one of the starts
    return theta


def pinned_scatter_values(state: ModelState, scatter: float | None) -> dict[str, float]:
    """Unconstrained value per member-scatter site when the scatter is pinned (the site holds ln sigma with a normal prior)."""
    if scatter is None:
        return {}
    return {(spec.sample_site_name or spec.sample_name): float(np.log(scatter))
            for spec in state.parameter_specs if spec.component_family == "member_velocity_scatter"}


class Objective:
    """Potential energy on the flat unconstrained vector and on its active block.

    The active block excludes the latent sites unless ``include_latent`` and excludes pinned sites, whose
    coordinates are held at the given unconstrained values.
    """

    def __init__(self, state: ModelState, evaluator: Evaluator, reference_theta: np.ndarray, *, include_latent: bool = False,
                 pinned: dict[str, float] | None = None, active_sites: set[str] | None = None) -> None:
        self.specs = state.parameter_specs
        self.model = posterior_model(self.specs, evaluator.log_likelihood)
        sites = unconstrained_from_theta(self.specs, np.asarray(reference_theta))
        sites = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in sites.items()}
        pinned = pinned or {}
        for name, value in pinned.items():
            if name in sites:
                sites[name] = jnp.full_like(sites[name], value)
        self.base, self.unravel = ravel_pytree(sites)
        # ravel_pytree flattens a dict in sorted key order, not insertion order: walk the leaves the same way.
        offset, mask = 0, np.ones(self.base.size, dtype=bool)
        for path, value in jax.tree_util.tree_flatten_with_path(sites)[0]:
            name, n = path[0].key, int(np.size(value))
            inactive = (name not in active_sites) if active_sites is not None else (name in LATENT_SITES and not include_latent)
            if inactive or name in pinned:
                mask[offset:offset + n] = False
            offset += n
        assert offset == self.base.size
        self.global_index = jnp.asarray(np.flatnonzero(mask))
        self.n_global = int(mask.sum())
        self.potential = jax.jit(lambda u: potential_energy(self.model, (), {}, self.unravel(u)))
        self.value_and_grad = jax.jit(jax.value_and_grad(lambda u: potential_energy(self.model, (), {}, self.unravel(u))))
        self.reduced = jax.jit(lambda g: potential_energy(self.model, (), {}, self.unravel(self.base.at[self.global_index].set(g))))
        self.reduced_value_and_grad = jax.jit(jax.value_and_grad(
            lambda g: potential_energy(self.model, (), {}, self.unravel(self.base.at[self.global_index].set(g)))))

    def flatten(self, theta: np.ndarray) -> np.ndarray:
        """Flat unconstrained vector of ``theta`` with the pinned coordinates replaced by their pinned values."""
        sites = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in unconstrained_from_theta(self.specs, np.asarray(theta)).items()}
        flat = np.array(ravel_pytree(sites)[0], dtype=float)
        base = np.asarray(self.base)
        fixed = np.ones(flat.size, dtype=bool)
        fixed[np.asarray(self.global_index)] = False
        flat[fixed] = base[fixed]
        return flat

    def expand(self, g: np.ndarray) -> np.ndarray:
        return np.asarray(self.base.at[self.global_index].set(jnp.asarray(g)))

    def theta(self, u: np.ndarray) -> np.ndarray:
        values = constrain_fn(prior_model(self.specs), (), {}, self.unravel(jnp.asarray(u)))
        return theta_from_site_values({k: np.asarray(v) for k, v in values.items()}, self.specs)


def _scipy_lbfgs(fun: Callable, x0: np.ndarray, max_iter: int, trace: list[float] | None = None) -> tuple[np.ndarray, float, int, int]:
    last = [np.inf]

    def objective(x):
        value, grad = fun(jnp.asarray(x))
        value = float(value)
        if not np.isfinite(value) or value >= BAD_POTENTIAL:
            return 1.0e30, np.zeros_like(x)
        last[0] = value
        return value, np.asarray(grad, dtype=float)

    def callback(_x):
        if trace is not None:
            trace.append(last[0])
    result = minimize(objective, np.asarray(x0, dtype=float), jac=True, method="L-BFGS-B", callback=callback,
                      options={"maxiter": int(max_iter), "maxcor": 20, "ftol": 1e-12, "gtol": 1e-6})
    return np.asarray(result.x), float(result.fun), int(result.nit), int(result.nfev)


def run_lbfgs(objective: Objective, starts_g: np.ndarray, config: SearchConfig) -> dict[str, Any]:
    """scipy L-BFGS-B on the active block, one start at a time (1 to 2 evaluations per iteration)."""
    def one(g):
        trace: list[float] = [float(objective.reduced(jnp.asarray(g)))]
        x, value, nit, nfev = _scipy_lbfgs(objective.reduced_value_and_grad, g, config.lbfgs_max_iter, trace)
        return x, value, nit, nfev, np.asarray(trace)
    results = [one(g) for g in progress(starts_g, desc="L-BFGS starts")]
    final, potentials, iterations, evaluations, traces = zip(*results)
    return dict(final=np.stack(final), potential=np.asarray(potentials), iterations=np.asarray(iterations), evaluations=np.asarray(evaluations),
                traces=_pad_traces(list(traces)))


def _pad_traces(traces: list[np.ndarray]) -> np.ndarray:
    """Per-start potential traces padded with NaN to a (starts, iterations) array."""
    width = max(len(t) for t in traces)
    out = np.full((len(traces), width), np.nan)
    for i, t in enumerate(traces):
        out[i, :len(t)] = t
    return out


def run_adam(objective: Objective, starts_g: np.ndarray, steps: int, learning_rate: float) -> dict[str, Any]:
    """Vmapped Adam on the active block with a cosine-decayed learning rate; keeps the best iterate per start."""
    steps = int(steps)
    schedule = optax.cosine_decay_schedule(float(learning_rate), steps)
    optimizer = optax.adam(schedule)
    vg = jax.value_and_grad(objective.reduced)

    def step(carry, _):
        g, opt_state, best_g, best_value = carry
        value, grad = vg(g)
        bad = ~jnp.isfinite(value) | (value >= BAD_POTENTIAL) | ~jnp.all(jnp.isfinite(grad))
        grad = jnp.where(bad, 0.0, grad)
        improved = (value < best_value) & ~bad
        best_g = jnp.where(improved, g, best_g)
        best_value = jnp.where(improved, value, best_value)
        updates, opt_state = optimizer.update(grad, opt_state, g)
        return (optax.apply_updates(g, updates), opt_state, best_g, best_value), value

    chunk, chunks = _chunking(steps)
    run_chunk = jax.jit(jax.vmap(lambda carry: jax.lax.scan(step, carry, None, length=chunk)))
    g0 = jnp.asarray(starts_g)
    carry = (g0, jax.vmap(optimizer.init)(g0), g0, jnp.full(len(starts_g), jnp.inf))
    values = []
    with progress(total=steps, desc=f"Adam x{len(starts_g)}") as bar:
        for _ in range(chunks):
            carry, chunk_values = run_chunk(carry)
            values.append(np.asarray(chunk_values))
            bar.update(chunk)
    traces = np.concatenate(values, axis=1)[:, :steps]
    return dict(final=np.asarray(carry[2]), potential=np.asarray(carry[3]), iterations=np.full(len(starts_g), steps),
                evaluations=np.full(len(starts_g), steps), traces=traces)


def run_optax_lbfgs(objective: Objective, starts_g: np.ndarray, max_iter: int, memory: int = 20) -> dict[str, Any]:
    """Vmapped optax L-BFGS with the zoom line search on the active block, every start in one jitted scan; keeps the best
    iterate per start.  Invalid models (the BAD_POTENTIAL sentinel) are given a finite large value and a zero gradient so
    the line search backs off instead of stepping through them."""
    max_iter = int(max_iter)
    optimizer = optax.lbfgs(memory_size=int(memory), linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=25))

    def safe_value(g):
        value = objective.reduced(g)
        return jnp.where(jnp.isfinite(value) & (value < BAD_POTENTIAL), value, 1.0e12)

    value_and_grad = optax.value_and_grad_from_state(safe_value)

    def step(carry, _):
        g, opt_state, best_g, best_value = carry
        value, grad = value_and_grad(g, state=opt_state)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        improved = value < best_value
        best_g = jnp.where(improved, g, best_g)
        best_value = jnp.where(improved, value, best_value)
        updates, opt_state = optimizer.update(grad, opt_state, g, value=value, grad=grad, value_fn=safe_value)
        return (optax.apply_updates(g, updates), opt_state, best_g, best_value), value

    chunk, chunks = _chunking(max_iter, target=50)
    run_chunk = jax.jit(jax.vmap(lambda carry: jax.lax.scan(step, carry, None, length=chunk)))
    g0 = jnp.asarray(starts_g)
    carry = (g0, jax.vmap(optimizer.init)(g0), g0, jnp.full(len(starts_g), jnp.inf))
    values = []
    with progress(total=max_iter, desc=f"optax L-BFGS x{len(starts_g)}") as bar:
        for _ in range(chunks):
            carry, chunk_values = run_chunk(carry)
            values.append(np.asarray(chunk_values))
            bar.update(chunk)
    traces = np.concatenate(values, axis=1)[:, :max_iter]
    potential = np.where(np.asarray(carry[3]) >= 1.0e12, np.inf, np.asarray(carry[3]))
    return dict(final=np.asarray(carry[2]), potential=potential, iterations=np.full(len(starts_g), max_iter),
                evaluations=np.full(len(starts_g), max_iter), traces=traces)


def _chunking(steps: int, target: int = 100) -> tuple[int, int]:
    """Chunk length and count so that a progress bar can tick between jitted scans."""
    chunk = max(1, min(int(target), int(steps)))
    return chunk, -(-int(steps) // chunk)


def run_svi(objective: Objective, starts_g: np.ndarray, steps: int, learning_rate: float, seed: int, particles: int = 1) -> dict[str, Any]:
    """Vmapped mean-field Gaussian guide on the active block trained with Adam on the Monte-Carlo ELBO.

    The point estimate is the guide mean; the reported potential is evaluated there.
    """
    steps = int(steps)
    schedule = optax.cosine_decay_schedule(float(learning_rate), steps)
    optimizer = optax.adam(schedule)
    n = objective.n_global

    def negative_elbo(params, key):
        mu, log_sigma = params
        eps = jax.random.normal(key, (int(particles), n))
        values = jax.vmap(lambda e: objective.reduced(mu + jnp.exp(log_sigma) * e))(eps)
        values = jnp.where(jnp.isfinite(values) & (values < BAD_POTENTIAL), values, 1.0e6)
        return jnp.mean(values) - jnp.sum(log_sigma)            # -E_q[log p] - H[q] up to a constant
    vg = jax.value_and_grad(negative_elbo)

    def step(carry, k):
        params, opt_state = carry
        value, grad = vg(params, k)
        grad = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), value

    chunk, chunks = _chunking(steps)
    run_chunk = jax.jit(jax.vmap(lambda carry, keys: jax.lax.scan(step, carry, keys)))
    g0 = jnp.asarray(starts_g)
    params = (g0, jnp.full((len(starts_g), n), -3.0))
    carry = (params, jax.vmap(optimizer.init)(params))
    key = jax.random.PRNGKey(int(seed) + 7)
    values = []
    with progress(total=steps, desc=f"SVI x{len(starts_g)}") as bar:
        for _ in range(chunks):
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, len(starts_g) * chunk).reshape(len(starts_g), chunk, 2)
            carry, chunk_values = run_chunk(carry, keys)
            values.append(np.asarray(chunk_values))
            bar.update(chunk)
    mu, log_sigma = carry[0]
    potential = jax.jit(jax.vmap(objective.reduced))(mu)
    traces = np.concatenate(values, axis=1)[:, :steps]
    return dict(final=np.asarray(mu), potential=np.asarray(potential), iterations=np.full(len(starts_g), steps),
                evaluations=np.full(len(starts_g), steps), guide_sigma_median=float(np.median(np.exp(np.asarray(log_sigma)))), traces=traces)


def cluster_modes(final_g: np.ndarray, potential: np.ndarray, finite: np.ndarray, starts_g: np.ndarray, tolerance: float):
    """Group end points closer than ``tolerance`` (in units of the spread of the starts) into modes, best first."""
    scale = np.where(finite.sum() > 1, np.std(starts_g[finite], axis=0), 1.0)
    scale = np.where(scale > 0, scale, 1.0)
    order = np.argsort(np.where(finite, potential, np.inf))
    modes: list[dict[str, Any]] = []
    assignment = np.full(len(final_g), -1, dtype=int)
    for index in order:
        if not finite[index]:
            continue
        for mode_index, mode in enumerate(modes):
            if np.max(np.abs(final_g[index] - final_g[mode["representative"]]) / scale) < float(tolerance):
                assignment[index] = mode_index
                mode["members"].append(int(index))
                break
        else:
            assignment[index] = len(modes)
            modes.append({"representative": int(index), "members": [int(index)], "potential": float(potential[index])})
    return modes, assignment


def multistart_map(state: ModelState, evaluator: Evaluator, starts: np.ndarray, config: SearchConfig, *, seed: int = 0,
                   log=print) -> dict[str, Any]:
    """Optimize the global block from every start, cluster the end points into modes, polish the best modes."""
    pinned = pinned_scatter_values(state, config.polish_scatter)
    objective = Objective(state, evaluator, starts[0], include_latent=bool(config.search_latent), pinned=pinned)
    polish_objective = objective if config.search_latent else Objective(state, evaluator, starts[0], include_latent=True, pinned=pinned)
    starts_g = np.stack([objective.flatten(s)[np.asarray(objective.global_index)] for s in starts])
    started = time.perf_counter()
    if config.method == "lbfgs":
        out = run_lbfgs(objective, starts_g, config)
    elif config.method == "adam":
        out = run_adam(objective, starts_g, config.adam_steps, config.adam_learning_rate)
    elif config.method == "svi":
        out = run_svi(objective, starts_g, config.svi_steps, config.svi_learning_rate, seed, config.svi_particles)
    else:
        raise ValueError(f"unknown search method {config.method!r}")
    search_seconds = time.perf_counter() - started
    final_g, potential = out["final"], out["potential"]
    finite = np.isfinite(potential) & np.isfinite(final_g).all(axis=1) & (potential < BAD_POTENTIAL)
    if not finite.any():
        raise RuntimeError("Every multi-start optimization ended at a non-finite or invalid potential.")
    modes, assignment = cluster_modes(final_g, potential, finite, starts_g, config.mode_parameter_tolerance)
    # Polish the modes that can seed chains in full dimension (offsets and sources now free).
    started = time.perf_counter()
    best_potential = modes[0]["potential"]
    for mode in progress(modes, desc="polish modes"):
        mode["within_margin"] = bool(mode["potential"] - best_potential <= float(config.mode_margin))
        u = objective.expand(final_g[mode["representative"]])
        if mode["within_margin"] and len([m for m in modes if m.get("polished")]) < int(config.max_chain_modes):
            active = polish_objective.flatten(objective.theta(u))[np.asarray(polish_objective.global_index)]
            active, value, nit, _ = _scipy_lbfgs(polish_objective.reduced_value_and_grad, active, config.polish_max_iter)
            u = polish_objective.expand(active)
            mode.update(polished=True, polish_iterations=nit, potential_polished=float(value))
        else:
            mode.update(polished=False, potential_polished=float(objective.potential(jnp.asarray(u))))
        mode["unconstrained"] = u
        mode["theta"] = objective.theta(u).tolist()
        mode["log_posterior"] = -mode["potential_polished"]
        mode["count"] = len(mode["members"])
    polish_seconds = time.perf_counter() - started
    modes.sort(key=lambda m: m["potential_polished"])
    best = modes[0]
    log(f"[search] {config.method}: {len(starts)} starts, {int(finite.sum())} finite, {len(modes)} modes, "
        f"{sum(m['within_margin'] for m in modes)} within margin; active block {objective.n_global} of {objective.base.size} "
        f"(latent {'free' if config.search_latent else 'at zero'}, scatter {'pinned at %.3g' % config.polish_scatter if config.polish_scatter else 'free'}); "
        f"parameters; search {search_seconds:.1f}s, polish {polish_seconds:.1f}s; best log posterior {-best['potential_polished']:.2f} "
        f"(before polish {-best['potential']:.2f}); mean iterations {out['iterations'].mean():.0f}, mean evaluations {out['evaluations'].mean():.0f}")
    theta_best = np.asarray(best["theta"], dtype=float)
    for mode in modes:
        mode.pop("unconstrained", None)
    final_theta = np.stack([objective.theta(objective.expand(g)) if ok else np.full(theta_best.size, np.nan)
                            for g, ok in zip(final_g, finite)])
    starts_in_basin = int(sum(m["count"] for m in modes if m["within_margin"]))
    configured_in_basin = any(0 in m["members"] and m["within_margin"] for m in modes)
    return dict(theta=theta_best, modes=modes, assignment=assignment.tolist(), potential=potential.tolist(),
                iterations=out["iterations"].tolist(), evaluations=out["evaluations"].tolist(), method=config.method,
                starts_in_basin=starts_in_basin, configured_in_basin=bool(configured_in_basin),
                traces=out["traces"], final_theta=final_theta, global_index=np.asarray(objective.global_index).tolist(),
                search_seconds=float(search_seconds), polish_seconds=float(polish_seconds), n_finite=int(finite.sum()),
                n_global=objective.n_global, starts=starts.tolist(), **{k: v for k, v in out.items() if k == "guide_sigma_median"})


def chain_starts(search: dict[str, Any], chains: int, max_modes: int) -> np.ndarray:
    """One latent start per chain: the best mode first, then other modes within the margin."""
    candidates = [np.asarray(m["theta"], dtype=float) for m in search["modes"] if m["within_margin"]][:int(max_modes)]
    if not candidates:
        candidates = [np.asarray(search["theta"], dtype=float)]
    return np.stack([candidates[i % len(candidates)] for i in range(int(chains))])
