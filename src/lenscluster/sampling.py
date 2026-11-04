"""Prior sites, the NumPyro posterior, the block mass seed, and the NUTS driver."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import potential_energy, unconstrain_fn

from .evaluator import Evaluator
from .model import ModelState, ParameterSpec, PosteriorResults

LATENT_FAMILIES = frozenset({"source_position", "member_scatter_offset"})
EIGENVALUE_FLOOR = 1.0
FINITE_DIFFERENCE_STEP = 1.0e-4


@dataclass(frozen=True)
class SampleSite:
    name: str
    indices: tuple[int, ...]
    vector: bool


def sample_sites(specs: list[ParameterSpec]) -> list[SampleSite]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for idx, spec in enumerate(specs):
        name = spec.sample_site_name or spec.sample_name
        grouped.setdefault(name, []).append((idx, idx if spec.sample_site_index is None else int(spec.sample_site_index)))
    sites, seen = [], set()
    for spec in specs:
        name = spec.sample_site_name or spec.sample_name
        if name in seen:
            continue
        seen.add(name)
        ordered = sorted(grouped[name], key=lambda item: item[1])
        if len(ordered) > 1 and [item[1] for item in ordered] != list(range(len(ordered))):
            raise ValueError(f"Vector sample site {name!r} has non-contiguous indices.")
        vector = len(ordered) > 1 or any(specs[item[0]].sample_site_index is not None for item in ordered)
        sites.append(SampleSite(name=name, indices=tuple(int(item[0]) for item in ordered), vector=vector))
    return sites


class _LowerInclusiveTruncatedNormal(dist.LeftTruncatedDistribution):
    """Include beta_radius=0 without changing the normalized interior density."""

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return dist.constraints.greater_than_eq(self.low)


def distribution_for_site(site: SampleSite, specs: list[ParameterSpec], sampled: dict[str, Any] | None = None):
    """Prior of one site. Centered deviation specs get N(0, exp(scatter site value)) using the already sampled scatter."""
    members = [specs[i] for i in site.indices]
    kinds = {spec.prior_kind for spec in members}
    if len(kinds) != 1:
        raise ValueError(f"Sample site {site.name!r} mixes prior kinds {sorted(kinds)}.")
    kind = kinds.pop()
    if any(spec.centered for spec in members):
        if sampled is None:
            raise ValueError(f"Site {site.name!r} has centered deviations and needs the sampled scatter values.")
        scale = jnp.stack([jnp.exp(sampled[spec.scale_site_name]) if spec.centered else jnp.asarray(float(spec.std), dtype=jnp.float64)
                           for spec in members])
        loc = jnp.asarray([float(spec.mean) for spec in members], dtype=jnp.float64)
        if kind == "student_t":
            df = jnp.asarray([float(spec.dof) for spec in members], dtype=jnp.float64)
            return dist.Independent(dist.StudentT(df, loc, scale), 1) if site.vector else dist.StudentT(df[0], loc[0], scale[0])
        return dist.Independent(dist.Normal(loc, scale), 1) if site.vector else dist.Normal(loc[0], scale[0])
    if not site.vector:
        spec = members[0]
        if kind == "normal":
            return dist.Normal(float(spec.mean), float(spec.std))
        if kind == "student_t":
            return dist.StudentT(float(spec.dof), float(spec.mean), float(spec.std))
        if kind == "truncated_normal":
            high = None if not np.isfinite(float(spec.upper)) else float(spec.upper)
            if spec.field == "beta_radius" and high is None:
                return _LowerInclusiveTruncatedNormal(dist.Normal(float(spec.mean), float(spec.std)), low=float(spec.lower), validate_args=True)
            return dist.TruncatedNormal(float(spec.mean), float(spec.std), low=float(spec.lower), high=high, validate_args=True)
        return dist.Uniform(float(spec.lower), float(spec.upper))
    as_array = lambda attr: jnp.asarray([float(getattr(spec, attr)) for spec in members], dtype=jnp.float64)
    if kind == "normal":
        return dist.Independent(dist.Normal(as_array("mean"), as_array("std")), 1)
    if kind == "student_t":
        return dist.Independent(dist.StudentT(as_array("dof"), as_array("mean"), as_array("std")), 1)
    if kind == "truncated_normal":
        high = jnp.asarray([np.inf if not np.isfinite(float(spec.upper)) else float(spec.upper) for spec in members], dtype=jnp.float64)
        return dist.Independent(dist.TruncatedNormal(as_array("mean"), as_array("std"), low=as_array("lower"), high=high, validate_args=True), 1)
    if kind == "uniform":
        return dist.Independent(dist.Uniform(as_array("lower"), as_array("upper")), 1)
    raise ValueError(f"Unsupported prior kind {kind!r} for site {site.name!r}.")


def site_values_from_theta(theta: jnp.ndarray, sites: list[SampleSite]) -> dict[str, jnp.ndarray]:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return {site.name: (jnp.stack([theta[i] for i in site.indices]) if site.vector else theta[site.indices[0]]) for site in sites}


def theta_from_site_values(values: dict[str, Any], specs: list[ParameterSpec]) -> np.ndarray:
    """Latent vector (…, n_params) from per-site arrays."""
    sites = sample_sites(specs)
    columns = []
    for site in sites:
        array = np.asarray(values[site.name], dtype=float)
        if site.vector:
            for position, _ in enumerate(site.indices):
                columns.append((site.indices[position], array[..., position]))
        else:
            columns.append((site.indices[0], array))
    columns.sort(key=lambda item: item[0])
    return np.stack([column for _, column in columns], axis=-1)


def prior_model(specs: list[ParameterSpec]):
    sites = sample_sites(specs)

    def model():
        sampled: dict[str, Any] = {}
        for site in sites:
            sampled[site.name] = numpyro.sample(site.name, distribution_for_site(site, specs, sampled))
    return model


def posterior_model(specs: list[ParameterSpec], loglike: Callable[[jnp.ndarray], jnp.ndarray]):
    sites = sample_sites(specs)
    n = len(specs)

    def model():
        values = [None] * n
        sampled: dict[str, Any] = {}
        for site in sites:
            value = numpyro.sample(site.name, distribution_for_site(site, specs, sampled))
            sampled[site.name] = value
            if site.vector:
                for position, idx in enumerate(site.indices):
                    values[idx] = value[..., position]
            else:
                values[site.indices[0]] = value
        theta = jnp.stack(values)
        numpyro.factor("position_loglike", loglike(theta))
    return model


def prior_log_prob(specs: list[ParameterSpec], theta: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=jnp.float64)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    values = site_values_from_theta(theta, sample_sites(specs))
    for site in sample_sites(specs):
        total = total + distribution_for_site(site, specs, values).log_prob(values[site.name])
    return total


def unconstrained_from_theta(specs: list[ParameterSpec], theta: np.ndarray) -> dict[str, jnp.ndarray]:
    sites = sample_sites(specs)
    constrained = site_values_from_theta(jnp.asarray(theta, dtype=jnp.float64), sites)
    return unconstrain_fn(prior_model(specs), (), {}, constrained)


HESSIAN_CHUNK = 16                      # Hessian columns per batched forward-over-reverse product (memory bound on all blocks)


def block_mass_seed(model, specs: list[ParameterSpec], reference_theta: np.ndarray, *,
                    dense_families: tuple[str, ...] = ()) -> tuple[list, dict, dict[str, Any]]:
    """Dense inverse mass for the global block from the Hessian at the reference; unit diagonal for latent sites.
    ``dense_families`` moves latent families into the dense block (SamplingConfig.dense_member_offsets: the member
    offsets, whose bright members trade off against the relation's sigma*, core and slopes, which a diagonal cannot
    follow)."""
    started = time.perf_counter()
    theta = np.asarray(reference_theta, dtype=float)
    if not np.all(np.isfinite(theta)):
        raise ValueError("reference theta must be finite")
    sites = sample_sites(specs)
    global_names, latent_names = [], []
    for site in sites:
        latent = [specs[i].component_family in LATENT_FAMILIES and specs[i].component_family not in dense_families for i in site.indices]
        if any(latent) and not all(latent):
            raise ValueError(f"Sample site {site.name!r} mixes global and latent parameters")
        (latent_names if all(latent) else global_names).append(site.name)
    global_names, latent_names = tuple(sorted(global_names)), tuple(sorted(latent_names))
    unconstrained = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in unconstrained_from_theta(specs, theta).items()}
    if any(not np.all(np.isfinite(np.asarray(value))) for value in unconstrained.values()):
        raise ValueError("The reference lies on or outside a prior support boundary")
    global_vector, unravel = ravel_pytree({name: unconstrained[name] for name in global_names})
    latent_dim = sum(int(unconstrained[name].size) for name in latent_names)
    inverse_mass: dict = {}
    if latent_names:
        # Unit mass for unit offsets; centered deviations get the scatter^2 at the reference as their scale.
        site_by_name = {site.name: site for site in sites}
        scales = []
        for name in latent_names:
            for idx in site_by_name[name].indices:
                spec = specs[idx]
                scales.append(float(np.exp(theta[[s_.sample_name for s_ in specs].index(spec.scale_site_name)])) ** 2 if spec.centered else 1.0)
        inverse_mass[latent_names] = jnp.asarray(scales, dtype=jnp.float64)
    diagnostics: dict[str, Any] = dict(global_dim=int(global_vector.size), latent_dim=latent_dim,
                                       global_sites=list(global_names), latent_sites=list(latent_names))
    if not global_names:
        return [], inverse_mass, diagnostics

    def global_potential(vector):
        current = dict(unconstrained)
        current.update(unravel(vector))
        return potential_energy(model, (), {}, current)

    try:
        # Forward-over-reverse Hessian-vector products, HESSIAN_CHUNK columns at a time: one jacfwd over a few hundred
        # dimensions would hold that many copies of every intermediate (606 x 606 blocks under all-image covariances).
        gradient_fn = jax.grad(global_potential)
        products = jax.jit(jax.vmap(lambda tangent: jax.jvp(gradient_fn, (global_vector,), (tangent,))[1]))
        basis = np.eye(int(global_vector.size))
        hessian = np.concatenate([np.asarray(products(jnp.asarray(basis[k:k + HESSIAN_CHUNK])), dtype=float)
                                  for k in range(0, basis.shape[0], HESSIAN_CHUNK)], axis=0).T
    except Exception as error:  # forward-over-reverse unsupported: central differences of the gradient
        diagnostics["autodiff_failure"] = repr(error)
        gradient = jax.jit(jax.grad(global_potential))
        vector = np.asarray(global_vector)
        columns = []
        for index in range(vector.size):
            delta = np.zeros(vector.size); delta[index] = FINITE_DIFFERENCE_STEP
            columns.append((np.asarray(gradient(jnp.asarray(vector + delta))) - np.asarray(gradient(jnp.asarray(vector - delta)))) / (2.0 * FINITE_DIFFERENCE_STEP))
        hessian = np.stack(columns, axis=1)
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("Hessian mass seeding produced non-finite curvature at the reference")
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    floored = np.maximum(eigenvalues, EIGENVALUE_FLOOR)
    seed = (eigenvectors * (1.0 / floored)) @ eigenvectors.T
    inverse_mass[global_names] = jnp.asarray(0.5 * (seed + seed.T), dtype=jnp.float64)
    positive = eigenvalues[eigenvalues > 0]
    diagnostics.update(seconds=float(time.perf_counter() - started), n_negative_eigenvalues=int(np.sum(eigenvalues < 0)),
                       n_floored_eigenvalues=int(np.sum(eigenvalues < EIGENVALUE_FLOOR)),
                       condition_raw=float(positive.max() / positive.min()) if positive.size else float("nan"),
                       condition_floored=float(floored.max() / floored.min()))
    return [global_names], inverse_mass, diagnostics


def print_global_summary(grouped: dict[str, Any], specs: list[ParameterSpec], extra: dict[str, Any] | None = None) -> None:
    """NumPyro's posterior summary table for the global sites only: halos, shear, scaling relation, scatters and error
    scales.  The member offsets and the source positions (LATENT_FAMILIES, hundreds of rows) are left out."""
    from numpyro.diagnostics import print_summary as numpyro_print_summary
    latent = {site.name for site in sample_sites(specs) if all(specs[i].component_family in LATENT_FAMILIES for i in site.indices)}
    numpyro_print_summary({name: value for name, value in grouped.items() if name not in latent}, group_by_chain=True)
    if extra is not None and "diverging" in extra:
        print(f"Number of divergences: {int(np.sum(np.asarray(extra['diverging'])))}")


def stiffest_direction(potential, x0, whiten, *, iterations: int = 30, seed: int = 0) -> tuple[float, np.ndarray]:
    """Largest curvature of ``potential`` at ``x0`` in the sampler's whitened coordinates, by power iteration on
    W^T H W (W = ``whiten``, a linear map from whitened to flat coordinates; H from Hessian-vector products), and its
    direction in the flat coordinates (unit norm).  The leapfrog step is stable below 2 / sqrt(|lambda|)."""
    gradient = jax.grad(potential)
    hvp = jax.jit(lambda v: jax.jvp(gradient, (x0,), (v,))[1])
    xi = jax.random.normal(jax.random.PRNGKey(int(seed)), jnp.shape(x0), dtype=jnp.float64)
    xi = xi / jnp.linalg.norm(xi)
    value = 0.0
    for _ in range(int(iterations)):
        hx = hvp(whiten(xi))
        w = whiten(hx, transpose=True)
        value = float(jnp.vdot(xi, w))
        norm = jnp.linalg.norm(w)
        if not bool(jnp.isfinite(norm)) or float(norm) == 0.0:
            break
        xi = w / norm
    direction = np.asarray(whiten(xi), dtype=float)
    return value, direction / max(np.linalg.norm(direction), 1e-300)


def stiffness_diagnostic(model, specs: list[ParameterSpec], mcmc, step_size: np.ndarray, chains: int, *, top: int = 5,
                         iterations: int = 30) -> list[dict[str, Any]]:
    """Per chain, at its last state and with the mass matrix it adapted: the largest curvature of the potential in the
    sampler's coordinates (power iteration with Hessian-vector products), the step size it allows (2 / sqrt(lambda)),
    the adapted step size against it, and the parameters that make up that stiffest direction."""
    state = mcmc.last_state
    z_all, inverse_all = state.z, state.adapt_state.inverse_mass_matrix
    sites = {site.name: site for site in sample_sites(specs)}
    batched = int(chains) > 1
    out = []
    for c in range(int(chains)):
        z = {k: jnp.asarray(v[c] if batched else v) for k, v in z_all.items()}
        inverse = {k: jnp.asarray(v[c] if batched else v) for k, v in inverse_all.items()}
        order = [name for key in inverse for name in key]                  # NumPyro flattens each block in its key's order
        sizes = [int(jnp.size(z[name])) for name in order]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        x0 = jnp.concatenate([jnp.ravel(z[name]) for name in order])

        def unflatten(vector):
            return {name: jnp.reshape(vector[offsets[i]:offsets[i + 1]], jnp.shape(z[name])) for i, name in enumerate(order)}

        factors = []                                                       # per block: (slice, cholesky or sqrt-diagonal)
        start = 0
        for key, block in inverse.items():
            n = sum(int(jnp.size(z[name])) for name in key)
            factors.append((slice(start, start + n), jnp.linalg.cholesky(block) if block.ndim == 2 else jnp.sqrt(block)))
            start += n

        def whiten(vector, transpose=False):
            parts = []
            for span, factor in factors:
                piece = vector[span]
                parts.append((factor.T @ piece if transpose else factor @ piece) if factor.ndim == 2 else factor * piece)
            return jnp.concatenate(parts)

        potential = lambda vector: potential_energy(model, (), {}, unflatten(vector))
        value, direction = stiffest_direction(potential, x0, whiten, iterations=iterations, seed=c)
        labels = [specs[i].name for name in order for i in sites[name].indices]
        heavy = np.argsort(-np.abs(direction))[:int(top)]
        limit = 2.0 / np.sqrt(abs(value)) if value != 0 and np.isfinite(value) else float("nan")
        out.append(dict(chain=c, curvature=float(value), step_limit=float(limit), step_size=float(step_size[c]),
                        step_fraction=float(step_size[c] / limit) if np.isfinite(limit) else float("nan"),
                        direction=[(labels[j], float(direction[j])) for j in heavy]))
    return out


def chain_diagnostics(grouped_theta: np.ndarray) -> dict[str, Any]:
    """Split-R-hat and effective sample size per parameter (NaN with a single chain)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    chains, draws, n = grouped_theta.shape
    if chains < 2 or draws < 4:
        return {"r_hat": [float("nan")] * n, "n_eff": [float("nan")] * n, "max_r_hat": float("nan"), "min_n_eff": float("nan")}
    r_hat = np.asarray(split_gelman_rubin(grouped_theta), dtype=float)
    n_eff = np.asarray(effective_sample_size(grouped_theta), dtype=float)
    return {"r_hat": r_hat.tolist(), "n_eff": n_eff.tolist(), "max_r_hat": float(np.nanmax(r_hat)), "min_n_eff": float(np.nanmin(n_eff))}


HESSIAN_NEGATIVE_FRACTION_WARNING = 0.10   # at a mode the global-block Hessian is positive definite; runs started at a
                                            # correct mode show 0-3 of 35 negative eigenvalues, a misplaced start 13-20 of 37


def hessian_seed_warning(mass_diagnostics: dict[str, Any]) -> str | None:
    """A warning when the Hessian of the global block at the first chain start has many negative eigenvalues: the start
    is then not at a posterior mode, and the floored seed gives NUTS a poor mass matrix."""
    negative, dim = mass_diagnostics.get("n_negative_eigenvalues"), mass_diagnostics.get("global_dim")
    if not negative or not dim or negative / dim <= HESSIAN_NEGATIVE_FRACTION_WARNING:
        return None
    return (f"WARNING: the Hessian at the first chain start has {negative} negative eigenvalues out of {dim} in the global block "
            f"(more than {100 * HESSIAN_NEGATIVE_FRACTION_WARNING:.0f}%): the start is not at a posterior mode, so the mass seed is floored and "
            f"warmup starts from a poor point.  Check that the chain starts reproduce the saved init start.")


def e_bfmi(energy: np.ndarray) -> float:
    """Estimated Bayesian fraction of missing information of one chain, sum (E_t - E_t-1)^2 / sum (E_t - mean E)^2;
    values below about 0.3 mean the momentum resampling cannot move the energy across the marginal energy distribution
    (heavy tails or a funnel)."""
    energy = np.asarray(energy, dtype=float)
    energy = energy[np.isfinite(energy)]
    if energy.size < 3:
        return float("nan")
    denominator = float(np.sum((energy - energy.mean()) ** 2))
    return float(np.sum(np.diff(energy) ** 2) / denominator) if denominator > 0 else float("nan")


def adaptation_diagnostics(step_size: np.ndarray, energy: np.ndarray, accept: np.ndarray, diverging: np.ndarray,
                           num_steps: np.ndarray, max_tree_depth: int) -> dict[str, Any]:
    """Per chain: the adapted step size, E-BFMI, energy spread, acceptance, divergences and how often the trajectory
    hit the tree-depth cap (2^depth - 1 leapfrog steps).  Every per-draw input has shape (chains, draws)."""
    cap = 2 ** int(max_tree_depth) - 1
    per_chain = [{"chain": int(c), "step_size": float(step_size[c]), "e_bfmi": e_bfmi(energy[c]),
                  "energy_mean": float(np.nanmean(energy[c])), "energy_std": float(np.nanstd(energy[c])),
                  "accept_mean": float(np.mean(accept[c])), "divergences": int(np.sum(diverging[c])),
                  "mean_steps": float(np.mean(num_steps[c])), "tree_depth_saturation": float(np.mean(num_steps[c] >= cap))}
                 for c in range(energy.shape[0])]
    return {"per_chain": per_chain, "max_tree_depth": int(max_tree_depth),
            "min_e_bfmi": float(np.nanmin([c["e_bfmi"] for c in per_chain])) if per_chain else float("nan"),
            "max_tree_depth_saturation": float(max(c["tree_depth_saturation"] for c in per_chain)) if per_chain else float("nan"),
            "step_size_range": [float(np.min(step_size)), float(np.max(step_size))] if len(step_size) else []}


def run_nuts(state: ModelState, evaluator: Evaluator, starts: np.ndarray, *, warmup: int, samples: int, chains: int,
             chain_method: str, max_tree_depth: int, target_accept: float, initial_step_size: float, seed: int,
             thin: int = 1, print_summary: bool = False, mass_matrix: str = "hessian", dense_member_offsets: bool = False,
             stiffness: bool = True, log=print) -> PosteriorResults:
    """Run NUTS from one latent start per chain (starts has shape (chains, n_params))."""
    specs = state.parameter_specs
    model = posterior_model(specs, evaluator.log_likelihood)
    starts = np.asarray(starts, dtype=float).reshape(chains, -1)
    dense_families = ("member_scatter_offset",) if dense_member_offsets else ()
    dense_mass, inverse_mass, mass_diagnostics = block_mass_seed(model, specs, starts[0], dense_families=dense_families)
    if mass_matrix == "hessian":
        log(f"[nuts] mass seed global_dim={mass_diagnostics.get('global_dim')} latent_dim={mass_diagnostics.get('latent_dim')} "
            f"cond_raw={mass_diagnostics.get('condition_raw', float('nan')):.3e} negative={mass_diagnostics.get('n_negative_eigenvalues')}")
    elif mass_matrix == "dense":
        inverse_mass = None
        log(f"[nuts] dense mass for the global block {mass_diagnostics.get('global_sites')}, adapted from the identity")
    else:
        dense_mass, inverse_mass = False, None
        log("[nuts] diagonal mass matrix, adapted from the identity")
    mass_diagnostics["mass_matrix"] = mass_matrix
    warning = hessian_seed_warning(mass_diagnostics)
    if warning:
        log(f"[nuts] {warning}")
    mass_diagnostics["warning"] = warning
    if chain_method == "parallel" and jax.device_count() < chains:
        raise RuntimeError(f"nuts_chain_method='parallel' needs {chains} JAX devices; {jax.device_count()} are visible. "
                           "Set JAX_NUM_CPU_DEVICES or use 'sequential'.")
    init = [unconstrained_from_theta(specs, start) for start in starts]
    init_params = jax.tree.map(lambda *values: jnp.stack(values), *init) if chains > 1 else init[0]
    kernel = NUTS(model, target_accept_prob=float(target_accept), max_tree_depth=int(max_tree_depth),
                  step_size=float(initial_step_size), dense_mass=dense_mass, inverse_mass_matrix=inverse_mass)
    mcmc = MCMC(kernel, num_warmup=int(warmup), num_samples=int(samples), num_chains=int(chains),
                chain_method=chain_method, progress_bar=True, thinning=int(thin))
    started = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)), extra_fields=("accept_prob", "diverging", "num_steps", "potential_energy", "energy",
                                                           "adapt_state.step_size"), init_params=init_params)
    grouped = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    # The step size is frozen after warmup, so the first sampled draw of each chain carries the adapted value.
    step_size = np.asarray(extra["adapt_state.step_size"], dtype=float).reshape(int(chains), -1)[:, 0]
    energy_by_chain = np.asarray(extra["energy"], dtype=float).reshape(int(chains), -1)
    jax.block_until_ready(grouped)
    elapsed = time.perf_counter() - started
    if print_summary:
        print_global_summary(grouped, specs, extra)
    grouped_theta = theta_from_site_values({k: np.asarray(v) for k, v in grouped.items()}, specs)   # (chains, draws, n)
    flat = grouped_theta.reshape(-1, grouped_theta.shape[-1])
    log_prob = -np.asarray(extra["potential_energy"], dtype=float).reshape(-1)
    finite = np.isfinite(flat).all(axis=1) & np.isfinite(log_prob)
    if not finite.all():
        log(f"[nuts] dropping {int((~finite).sum())} non-finite draws")
    reported = jax.vmap(jax.jit(evaluator.reported_physical))
    physical = np.asarray(reported(jnp.asarray(flat)))
    accept = np.asarray(extra["accept_prob"], dtype=float).reshape(-1)
    diverging = np.asarray(extra["diverging"], dtype=bool).reshape(-1)
    num_steps = np.asarray(extra["num_steps"], dtype=float).reshape(-1)
    adaptation = adaptation_diagnostics(step_size, energy_by_chain, accept.reshape(int(chains), -1), diverging.reshape(int(chains), -1),
                                        num_steps.reshape(int(chains), -1), max_tree_depth)
    log(f"[nuts] done in {elapsed:.1f}s accept_mean={accept.mean():.3f} divergences={int(diverging.sum())} "
        f"mean_steps={num_steps.mean():.1f} draws={flat.shape[0]}")
    for c in adaptation["per_chain"]:
        log(f"[nuts] chain {c['chain']}: step size {c['step_size']:.3g}, E-BFMI {c['e_bfmi']:.2f}, accept {c['accept_mean']:.3f}, "
            f"{c['divergences']} divergences, mean steps {c['mean_steps']:.0f}, {100 * c['tree_depth_saturation']:.0f}% of transitions at the "
            f"tree-depth cap ({2 ** int(max_tree_depth) - 1} steps)")
    if adaptation["min_e_bfmi"] < 0.3:
        log(f"[nuts] WARNING: E-BFMI below 0.3 (min {adaptation['min_e_bfmi']:.2f}): the energy does not mix, typical of a funnel or heavy tails")
    stiff = None
    if stiffness:
        # What sets the step size: the stiffest direction at each chain's last state, with the mass matrix it adapted.
        try:
            started_stiffness = time.perf_counter()
            stiff = stiffness_diagnostic(model, specs, mcmc, step_size, int(chains))
            for row in stiff:
                parts = ", ".join(f"{name} {weight:+.2f}" for name, weight in row["direction"])
                log(f"[nuts] chain {row['chain']} stiffness: step limit {row['step_limit']:.3g} (2/sqrt of curvature {row['curvature']:.3g}), "
                    f"adapted step {row['step_size']:.3g} = {row['step_fraction']:.2f} of it; stiffest direction: {parts}")
            fractions = [r["step_fraction"] for r in stiff if np.isfinite(r["step_fraction"])]
            if fractions and min(fractions) > 0.3 and adaptation.get("max_tree_depth_saturation", 0.0) > 0.5:
                log("[nuts] the step size sits near the stability limit of that direction and trajectories hit the tree-depth cap: "
                    "the stiffness, not the posterior's width, sets the trajectory length")
            log(f"[nuts] stiffness diagnostic: {time.perf_counter() - started_stiffness:.0f}s")
        except Exception as error:                                     # a diagnostic must never fail a run
            log(f"[nuts] WARNING: stiffness diagnostic failed ({type(error).__name__}: {error})")
    best = int(np.nanargmax(np.where(finite, log_prob, -np.inf)))
    diagnostics = dict(mass_seed=mass_diagnostics, adaptation=adaptation, elapsed_seconds=float(elapsed), chain_method=chain_method,
                       n_nonfinite_draws=int((~finite).sum()), stiffness=stiff, **chain_diagnostics(grouped_theta))
    return PosteriorResults(samples=flat, physical=physical, log_prob=log_prob, accept_prob=accept, diverging=diverging,
                           num_steps=num_steps, warmup_steps=int(warmup), sample_steps=int(samples), num_chains=int(chains),
                           energy=energy_by_chain.reshape(-1), step_size=step_size,
                           grouped_samples=grouped_theta, diagnostics=diagnostics,
                           median_fit=np.median(physical[finite], axis=0) if finite.any() else None,
                           maximum_likelihood_fit=physical[best])
