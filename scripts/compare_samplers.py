"""Compare samplers on one saved init, per gradient: NUTS (the baseline), adjusted MCLMC, and adjusted MCLMC from
Pathfinder starts.

Every arm samples the same target as the pipeline's NUTS (the init's posterior on the flat unconstrained vector, source
transport included), whitened by the same dense Hessian-seeded inverse mass the pipeline uses (global parameters and
member offsets dense, sources diagonal): u = u_start + L v with L L^T = M^-1, so each sampler sees an identity mass and
its own adaptation works on top.  Every log-density evaluation is counted by a host callback, so the arms are compared
per evaluation, tuning included.  Chains run in parallel, one host device each.

Usage:
    python scripts/compare_samplers.py results/<campaign>/m0416_canucs2025/<run> [--blocks all|system] [--chains 4]
        [--arms nuts mclmc mclmc_pf] [--nuts-warmup 200 --nuts-samples 100] [--mclmc-tune 1000 --mclmc-samples 1000]
        [--start-from <previous out>/nuts_draws.npz] [--out DIR]

Writes <out>/compare_samplers.json (per arm: evaluations, wall time, ESS of the global parameters per 1000 evaluations,
R-hat, acceptance, divergences, step sizes, trajectory lengths, medians of key parameters) and <out>/<arm>_draws.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory with init/artifacts")
    p.add_argument("--blocks", default="all", choices=["all", "system", "family"], help="GP covariance blocks (default all)")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=["nuts", "mclmc", "mclmc_pf"], choices=["nuts", "mclmc", "mclmc_pf"])
    p.add_argument("--nuts-warmup", type=int, default=200)
    p.add_argument("--nuts-samples", type=int, default=100)
    p.add_argument("--nuts-depth", type=int, default=8, help="max tree depth (default 8, as the pipeline)")
    p.add_argument("--mclmc-tune", type=int, default=1000, help="MCLMC tuning steps per stage (unadjusted, then adjusted)")
    p.add_argument("--mclmc-samples", type=int, default=1000)
    p.add_argument("--mclmc-target", type=float, default=0.9, help="target acceptance of adjusted MCLMC")
    p.add_argument("--pathfinder-iter", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-from", type=Path, default=None,
                   help="a previous arm's <arm>_draws.npz: every arm starts each chain at that file's last draw of the chain, and "
                        "the whitening mass is the Hessian at the best of those draws (an equilibrated start: the comparison "
                        "measures mixing, not burn-in); Pathfinder starts from these draws instead of the search modes")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


args = parse()
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={args.chains}").strip()
os.environ["JAX_NUM_CPU_DEVICES"] = str(args.chains)
sys.path.insert(0, str(ROOT / "src"))

import jax                                                                     # noqa: E402
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_num_cpu_devices", args.chains)
import jax.numpy as jnp                                                        # noqa: E402
import numpy as np                                                             # noqa: E402
import blackjax                                                                # noqa: E402
from dataclasses import replace                                               # noqa: E402
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin     # noqa: E402

from lenscluster import artifacts                                              # noqa: E402
from lenscluster.evaluator import Evaluator                                    # noqa: E402
from lenscluster.sampling import LATENT_FAMILIES, block_mass_seed, sample_sites   # noqa: E402
from lenscluster.search import Objective                                       # noqa: E402
from lenscluster.state import build_state, with_member_classes                 # noqa: E402

KEY_PARAMETERS = ("4.x_centre", "4.y_centre", "4.v_disp", "1.v_disp", "2.v_disp", "potfile_1.sigma", "potfile_1.cutkpc",
                  "potfile_1.velocity_log_scatter", "potfile_1.radius_log_scatter", "images.gp_sigma_alpha", "images.gp_knee",
                  "images.gp_ell", "shear.gamma1", "shear.gamma2")


# ----------------------------------------------------------------------------------------------- the target
def build_target(run: Path, blocks: str):
    init = run / "init" / "artifacts"
    config = artifacts.load_config(init)
    likelihood = config.likelihood
    if blocks != likelihood.gp_blocks:
        modes = ("translation", "convergence", "shear") if blocks == "all" and likelihood.gp_kernel == "knee" else ()
        likelihood = replace(likelihood, gp_blocks=blocks, gp_knee_project_modes=modes)
        config = replace(config, likelihood=likelihood)
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered = set(classes.get("centered_members", []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids], dtype=bool))
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    start = artifacts.load_start(init)
    objective = Objective(state, evaluator, start, include_latent=True, log_likelihood=evaluator.log_likelihood)
    try:
        found = [m for m in artifacts.load_search(init).get("modes", []) if m.get("theta") is not None]
        found = sorted(found, key=lambda m: -float(m.get("log_posterior", -np.inf)))[:4]          # the 4 best search modes
        modes = [np.asarray(m["theta"], dtype=float) for m in found]
    except Exception:                                                          # an init without a saved search
        modes = []
    return config, state, evaluator, objective, start, modes


def flat_layout(state, objective):
    """Site offsets and the parameter name of every entry of the objective's flat vector (ravel_pytree: sorted sites)."""
    specs = state.parameter_specs
    sites = {s.name: s for s in sample_sites(specs)}
    shapes = {name: np.shape(v) for name, v in objective.unravel(objective.base).items()}
    offsets, names, start = {}, [], 0
    for name in sorted(shapes):
        n = int(np.prod(shapes[name])) if shapes[name] else 1
        offsets[name] = (start, start + n)
        names += [specs[i].name for i in sites[name].indices]
        start += n
    families = {s.name: s.component_family for s in specs}
    return offsets, names, families


def whitening(state, objective, start, offsets):
    """Cholesky factor of the pipeline's inverse mass (dense global + member offsets, diagonal sources) in flat order."""
    dense, inverse, diagnostics = block_mass_seed(objective.model, state.parameter_specs, start, dense_families=("member_scatter_offset",))
    n = int(objective.base.size)
    matrix = np.zeros((n, n))
    for key, block in inverse.items():
        index = np.concatenate([np.arange(*offsets[name]) for name in key])
        block = np.asarray(block, dtype=float)
        matrix[np.ix_(index, index)] = block if block.ndim == 2 else np.diag(block)
    return np.linalg.cholesky(0.5 * (matrix + matrix.T)), diagnostics


# ----------------------------------------------------------------------------------------------- counting
COUNT = {"n": 0}


def _tick(_):
    COUNT["n"] += 1


def make_logdensity(objective, centre, factor):
    centre, factor = jnp.asarray(centre), jnp.asarray(factor)

    def logdensity(v):
        jax.debug.callback(_tick, v[0])
        value = -objective.potential(centre + factor @ v)
        return jnp.where(jnp.isfinite(value), value, -jnp.inf)
    return logdensity


# ----------------------------------------------------------------------------------------------- samplers
def run_nuts(logdensity, starts, keys, warmup, samples, depth):
    def one(key, position):
        warm_key, run_key = jax.random.split(key)
        adapt = blackjax.window_adaptation(blackjax.nuts, logdensity, is_mass_matrix_diagonal=True, target_acceptance_rate=0.8,
                                           max_num_doublings=depth)
        (state, params), _ = adapt.run(warm_key, position, num_steps=warmup)
        kernel = blackjax.nuts(logdensity, **{**params, "max_num_doublings": depth})

        def step(state, k):
            state, info = kernel.step(k, state)
            return state, (state.position, info.acceptance_rate, info.is_divergent, info.num_integration_steps)
        _, (positions, accept, divergent, steps) = jax.lax.scan(step, state, jax.random.split(run_key, samples))
        return positions, accept, divergent, steps, params["step_size"], jnp.zeros(())
    return jax.pmap(one)(keys, starts)


def run_mclmc(logdensity, starts, keys, tune, samples, target):
    integrator = blackjax.mcmc.integrators.isokinetic_mclachlan

    def steps_fn(random_arg, avg_steps):                                       # a jittered trajectory length around the tuned mean
        return jnp.maximum(1, jnp.ceil(jax.random.uniform(random_arg) * 2.0 * avg_steps)).astype(jnp.int32)

    def one(key, position):
        unadjusted_key, init_key, tune_key, run_key = jax.random.split(key, 4)
        # Stage 1, unadjusted MCLMC tuning: the trajectory length L from the chain's autocorrelation (frac_tune3), the
        # step size and a diagonal preconditioner on top of the whitening.  Only the tuning uses the unadjusted dynamics.
        unadjusted = blackjax.mcmc.mclmc.init(position=position, logdensity_fn=logdensity, rng_key=unadjusted_key)
        unadjusted_kernel = blackjax.mcmc.mclmc.build_kernel(integrator=integrator)
        unadjusted, guess, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=unadjusted_kernel, logdensity_fn=logdensity, num_steps=tune, state=unadjusted, rng_key=init_key,
            frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1, diagonal_preconditioning=True)
        n_steps = jnp.clip(guess.L / guess.step_size, 2.0, 255.0)
        # Stage 2, adjusted MCLMC: the step size calibrated by dual averaging at that many integration steps per
        # trajectory (this BlackJAX pins L = target_num_integration_steps * step), then Metropolis-adjusted sampling.
        state = blackjax.mcmc.adjusted_mclmc_dynamic.init(position=unadjusted.position, logdensity_fn=logdensity,
                                                          random_generator_arg=init_key)
        kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(integration_steps_fn=steps_fn, integrator=integrator)
        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, logdensity_fn=logdensity, num_steps=tune, state=state, rng_key=tune_key, target=target,
            frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, diagonal_preconditioning=True, params=guess,
            target_num_integration_steps=n_steps)
        avg = params.L / params.step_size

        def step(state, k):
            state, info = kernel(rng_key=k, state=state, logdensity_fn=logdensity, step_size=params.step_size,
                                 inverse_mass_matrix=params.inverse_mass_matrix, integration_steps_params=(avg,))
            return state, (state.position, info.acceptance_rate, info.is_divergent, info.num_integration_steps)
        _, (positions, accept, divergent, steps) = jax.lax.scan(step, state, jax.random.split(run_key, samples))
        return positions, accept, divergent, steps, params.step_size, params.L
    return jax.pmap(one)(keys, starts)


def pathfinder_starts(logdensity, initial_positions, key, chains, maxiter):
    """Pathfinder from each initial position (compiled once); the chain starts are draws from the approximation with the
    best ELBO."""
    approximate = jax.jit(lambda k, position: blackjax.vi.pathfinder.approximate(k, logdensity, position, num_samples=50,
                                                                                 maxiter=maxiter)[0])
    best = None
    for k, position in zip(jax.random.split(key, len(initial_positions)), initial_positions):
        state = approximate(k, position)
        print(f"[mclmc_pf] pathfinder ELBO {float(state.elbo):.1f}", flush=True)
        if best is None or float(state.elbo) > float(best.elbo):
            best = state
    draws, _ = blackjax.vi.pathfinder.sample(jax.random.fold_in(key, 1), best, num_samples=chains)
    return draws, float(best.elbo)


# ----------------------------------------------------------------------------------------------- scoring
def score(name, positions, accept, divergent, steps, step_size, length, evaluations, seconds, centre, factor, objective, evaluator,
          names, families, out):
    chains, draws, _ = positions.shape
    u = np.asarray(centre)[None, None, :] + np.einsum("ij,cdj->cdi", np.asarray(factor), np.asarray(positions))
    theta = np.stack([np.stack([objective.theta(u[c, d]) for d in range(draws)]) for c in range(chains)])
    physical = np.asarray(jax.vmap(jax.vmap(jax.jit(evaluator.physical)))(jnp.asarray(theta)))
    global_index = [i for i, n in enumerate(names) if families[n] not in LATENT_FAMILIES]
    spec_names = [s.name for s in evaluator.state.parameter_specs]
    g_cols = [spec_names.index(names[i]) for i in global_index]
    ess = np.asarray(effective_sample_size(theta[:, :, g_cols]))
    rhat = np.asarray(split_gelman_rubin(theta[:, :, g_cols])) if chains > 1 else np.full(len(g_cols), np.nan)
    worst = np.argsort(ess)[:5]
    medians = {k: [float(np.median(physical[c, :, spec_names.index(k)])) for c in range(chains)] for k in KEY_PARAMETERS if k in spec_names}
    result = dict(arm=name, chains=int(chains), draws_per_chain=int(draws), evaluations=int(evaluations), seconds=float(seconds),
                  sampling_gradients=int(np.sum(steps)), mean_steps_per_draw=float(np.mean(steps)),
                  ess_global_min=float(ess.min()), ess_global_median=float(np.median(ess)),
                  ess_min_per_1000_evaluations=float(1000.0 * ess.min() / max(evaluations, 1)),
                  ess_median_per_1000_evaluations=float(1000.0 * np.median(ess) / max(evaluations, 1)),
                  max_rhat=float(np.nanmax(rhat)), worst_ess=[(names[global_index[i]], float(ess[i])) for i in worst],
                  acceptance=float(np.mean(accept)), divergences=int(np.sum(divergent)),
                  step_size=[float(x) for x in np.atleast_1d(step_size)], trajectory_length=[float(x) for x in np.atleast_1d(length)],
                  medians_per_chain=medians)
    np.savez_compressed(out / f"{name}_draws.npz", theta=theta, physical=physical, names=np.asarray(spec_names))
    return result


# ----------------------------------------------------------------------------------------------- main
def main():
    out = args.out or (args.run / "compare_samplers")
    out.mkdir(parents=True, exist_ok=True)
    print(f"devices: {jax.devices()}", flush=True)
    config, state, evaluator, objective, start, modes = build_target(args.run, args.blocks)
    offsets, names, families = flat_layout(state, objective)
    reference, chain_starts, start_note = start, [start] * args.chains, "the saved init start"
    if args.start_from is not None:
        previous = np.load(args.start_from, allow_pickle=True)["theta"]          # (chains, draws, n) latent theta
        last = [np.asarray(previous[c, -1], dtype=float) for c in range(previous.shape[0])]
        values = [-float(objective.potential(jnp.asarray(objective.flatten(t)))) for t in last]
        reference = last[int(np.argmax(values))]
        chain_starts = [last[c % len(last)] for c in range(args.chains)]
        start_note = (f"the last draws of {args.start_from} (log posterior {', '.join(f'{v:.1f}' for v in values)}); "
                      f"mass at the best, {max(values):.1f}")
    print(f"starts: {start_note}", flush=True)
    centre = np.asarray(objective.flatten(reference), dtype=float)
    t0 = time.perf_counter()
    factor, mass = whitening(state, objective, reference, offsets)
    print(f"target: {len(names)} parameters, blocks {config.likelihood.gp_blocks}, projection {config.likelihood.gp_knee_project_modes}; "
          f"whitening from the Hessian seed ({mass['global_dim']} dense, {mass['latent_dim']} diagonal) in {time.perf_counter() - t0:.0f}s",
          flush=True)
    logdensity = make_logdensity(objective, centre, factor)
    value = float(jax.jit(logdensity)(jnp.zeros(len(names))))
    print(f"log density at the start: {value:.2f}", flush=True)
    key = jax.random.PRNGKey(args.seed)
    factor_inv = np.linalg.inv(factor)
    starts_v = jnp.asarray(np.stack([factor_inv @ (np.asarray(objective.flatten(t)) - centre) for t in chain_starts]))
    results = {"run": str(args.run), "blocks": config.likelihood.gp_blocks, "n_parameters": len(names), "starts": start_note, "arms": []}

    for arm in args.arms:
        COUNT["n"] = 0
        started = time.perf_counter()
        extra = {}
        if arm == "nuts":
            keys = jax.random.split(jax.random.fold_in(key, 1), args.chains)
            out_arm = run_nuts(logdensity, starts_v, keys, args.nuts_warmup, args.nuts_samples, args.nuts_depth)
        elif arm == "mclmc":
            keys = jax.random.split(jax.random.fold_in(key, 2), args.chains)
            out_arm = run_mclmc(logdensity, starts_v, keys, args.mclmc_tune, args.mclmc_samples, args.mclmc_target)
        else:
            if args.start_from is not None:
                initial = [starts_v[c] for c in range(starts_v.shape[0])]
            else:
                initial = [jnp.asarray(factor_inv @ (np.asarray(objective.flatten(m)) - centre)) for m in (modes or [start])]
            starts, elbo = pathfinder_starts(logdensity, initial, jax.random.fold_in(key, 3), args.chains, args.pathfinder_iter)
            extra = dict(pathfinder_elbo=elbo, pathfinder_modes=len(initial), pathfinder_evaluations=int(COUNT["n"]))
            print(f"[mclmc_pf] pathfinder from {len(initial)} starting points: best ELBO {elbo:.1f}, {COUNT['n']} evaluations", flush=True)
            keys = jax.random.split(jax.random.fold_in(key, 4), args.chains)
            out_arm = run_mclmc(logdensity, starts, keys, args.mclmc_tune, args.mclmc_samples, args.mclmc_target)
        positions, accept, divergent, steps, step_size, length = (np.asarray(x) for x in jax.block_until_ready(out_arm))
        seconds = time.perf_counter() - started
        result = score(arm, positions, accept, divergent, steps, step_size, length, COUNT["n"], seconds, centre, factor, objective,
                       evaluator, names, families, out)
        result.update(extra)
        results["arms"].append(result)
        print(f"[{arm}] {result['evaluations']} evaluations in {seconds / 60:.1f} min; ESS global min {result['ess_global_min']:.1f}, "
              f"median {result['ess_global_median']:.1f} ({result['ess_min_per_1000_evaluations']:.2f} / "
              f"{result['ess_median_per_1000_evaluations']:.2f} per 1000 evaluations); max R-hat {result['max_rhat']:.2f}; "
              f"accept {result['acceptance']:.2f}; divergences {result['divergences']}; mean steps per draw {result['mean_steps_per_draw']:.1f}",
              flush=True)
        (out / "compare_samplers.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out / 'compare_samplers.json'}")


if __name__ == "__main__":
    main()
