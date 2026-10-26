"""Staged init: fit the big halos, then also the free (independent member) halos, then everything, with Adam or SVI."""
from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from .config import LensModelConfig, SearchConfig
from .evaluator import Evaluator
from .model import ModelState
from .search import BAD_POTENTIAL, Objective, cluster_modes, pinned_scatter_values, run_adam, run_svi
from .utils import progress

STAGE_NAMES = ("big_halos", "free_halos", "all")


def stage_site_sets(state: ModelState, model: LensModelConfig) -> dict[str, set[str] | None]:
    """Sample sites active in each stage: configured large halos; plus independent member halos; everything."""
    big_ids = {str(h.id) for h in model.large_halos}
    big, free = set(), set()
    for spec in state.parameter_specs:
        if spec.component_family != "large":
            continue
        site = spec.sample_site_name or spec.sample_name
        (big if spec.name.split(".", 1)[0] in big_ids else free).add(site)
    return {"big_halos": big, "free_halos": big | free, "all": None}


def staged_map(state: ModelState, evaluator: Evaluator, starts: np.ndarray, config: SearchConfig, model: LensModelConfig, *,
               seed: int = 0, log=print) -> dict[str, Any]:
    """Three optimizations of growing scope; the best ``max_chain_modes`` starts of a stage seed the next."""
    pinned = pinned_scatter_values(state, config.polish_scatter) if config.staged_pin_scatter else {}
    sites = stage_site_sets(state, model)
    steps, rate = int(config.staged_steps), float(config.staged_learning_rate)

    def optimize(objective: Objective, starts_g: np.ndarray) -> dict[str, Any]:
        if config.staged_method == "adam":
            return run_adam(objective, starts_g, steps, rate)
        return run_svi(objective, starts_g, steps, rate, seed, config.svi_particles)

    started = time.perf_counter()
    stage_log: list[dict[str, Any]] = []
    stage_traces: dict[str, np.ndarray] = {}
    current_theta = np.asarray(starts, dtype=float)[:int(config.staged_starts)]   # latent vectors carried between stages
    first = None
    for name in STAGE_NAMES:
        objective = Objective(state, evaluator, current_theta[0], pinned=pinned, active_sites=sites[name], include_latent=sites[name] is None)
        if objective.n_global == 0:
            log(f"[staged] {name}: no active parameters, skipped")
            continue
        starts_g = np.stack([objective.flatten(t)[np.asarray(objective.global_index)] for t in current_theta])
        t0 = time.perf_counter()
        out = optimize(objective, starts_g)
        potential = np.asarray(out["potential"], dtype=float)
        finite = np.isfinite(potential) & np.isfinite(out["final"]).all(axis=1) & (potential < BAD_POTENTIAL)
        if not finite.any():
            raise RuntimeError(f"staged init: every start of stage {name!r} ended at an invalid potential.")
        stage_traces[name] = np.asarray(out["traces"], dtype=float)
        order = np.argsort(np.where(finite, potential, np.inf))
        keep = [int(i) for i in order[:int(config.max_chain_modes)] if finite[i]]
        if first is None:                                       # stage-1 bookkeeping for the mode plots
            first = dict(objective=objective, starts_g=starts_g, final_g=np.asarray(out["final"]), potential=potential, finite=finite,
                         iterations=np.asarray(out["iterations"]), evaluations=np.asarray(out["evaluations"]), traces=stage_traces[name])
        current_theta = np.stack([objective.theta(objective.expand(out["final"][i])) for i in keep])
        stage_log.append({"stage": name, "active": objective.n_global, "starts": len(starts_g), "finite": int(finite.sum()),
                          "best_log_posterior": float(-potential[keep[0]]), "kept": len(keep), "seconds": time.perf_counter() - t0})
        log(f"[staged] {name}: {objective.n_global} active parameters, {len(starts_g)} starts, {int(finite.sum())} finite, "
            f"best log posterior {-potential[keep[0]]:.2f}, {len(keep)} carried, {time.perf_counter() - t0:.1f}s")
    search_seconds = time.perf_counter() - started
    final_objective = Objective(state, evaluator, current_theta[0], pinned=pinned, active_sites=None, include_latent=True)
    final_u = np.stack([final_objective.flatten(t) for t in current_theta])
    final_potential = np.asarray([float(final_objective.potential(jnp.asarray(u))) for u in final_u])
    finite = np.isfinite(final_potential) & (final_potential < BAD_POTENTIAL)
    modes, assignment = cluster_modes(final_u, final_potential, finite, final_u, config.mode_parameter_tolerance)
    best_potential = modes[0]["potential"]
    for mode in modes:
        u = final_u[mode["representative"]]
        mode.update(within_margin=bool(mode["potential"] - best_potential <= float(config.mode_margin)), polished=False,
                    potential_polished=float(mode["potential"]), theta=final_objective.theta(u).tolist(),
                    log_posterior=float(-mode["potential"]), count=len(mode["members"]))
    best = modes[0]
    log(f"[search] staged {config.staged_method}: {len(starts)} starts, {len(modes)} modes, {sum(m['within_margin'] for m in modes)} within margin; "
        f"{final_objective.n_global} of {final_objective.base.size} parameters active in the last stage "
        f"(scatter {'pinned at %.3g' % config.polish_scatter if pinned else 'free'}); search {search_seconds:.1f}s; "
        f"best log posterior {-best['potential']:.2f}; steps {steps}, learning rate {rate:g}, particles {config.svi_particles if config.staged_method == 'svi' else 1}")
    final_theta = np.stack([first["objective"].theta(first["objective"].expand(g)) if ok else np.full(len(starts[0]), np.nan)
                            for g, ok in zip(first["final_g"], first["finite"])])
    starts = np.asarray(starts, dtype=float)[:int(config.staged_starts)]
    assignment_first = np.full(len(starts), -1, dtype=int)
    return dict(theta=np.asarray(best["theta"], dtype=float), modes=modes, assignment=assignment_first.tolist(),
                potential=first["potential"].tolist(), iterations=first["iterations"].tolist(), evaluations=first["evaluations"].tolist(),
                method=f"staged-{config.staged_method}", traces=first["traces"], final_theta=final_theta,
                global_index=np.asarray(first["objective"].global_index).tolist(), search_seconds=float(search_seconds), polish_seconds=0.0,
                n_finite=int(first["finite"].sum()), n_global=int(final_objective.n_global), starts=starts.tolist(),
                stages=stage_log, stage_traces={k: v.tolist() for k, v in stage_traces.items()})


def hybrid_map(state: ModelState, evaluator: Evaluator, starts: np.ndarray, config: SearchConfig, *, seed: int = 0, log=print) -> dict[str, Any]:
    """Adam on the lens block (halos, shear, free halos, scaling relation, sigma_src) from every prior start, then
    L-BFGS on everything (deviations and sources free, member scatter pinned) from each mode within the margin."""
    from .search import _scipy_lbfgs
    pinned = pinned_scatter_values(state, config.polish_scatter)
    started = time.perf_counter()
    # Stage A: Adam, members on the relation, sources at their conditional optimum.
    stage_a = Objective(state, evaluator, starts[0], pinned=pinned, include_latent=False)
    starts_g = np.stack([stage_a.flatten(t)[np.asarray(stage_a.global_index)] for t in starts])
    t0 = time.perf_counter()
    out = run_adam(stage_a, starts_g, config.staged_steps, config.staged_learning_rate)
    potential = np.asarray(out["potential"], dtype=float)
    finite = np.isfinite(potential) & np.isfinite(out["final"]).all(axis=1) & (potential < BAD_POTENTIAL)
    if not finite.any():
        raise RuntimeError("hybrid init: every Adam start ended at an invalid potential.")
    modes_a, assignment = cluster_modes(np.asarray(out["final"]), potential, finite, starts_g, config.mode_parameter_tolerance)
    best_a = modes_a[0]["potential"]
    carried = [m for m in modes_a if m["potential"] - best_a <= float(config.mode_margin)][:max(int(config.max_chain_modes), 8)]
    seconds_a = time.perf_counter() - t0
    log(f"[hybrid] adam: {len(starts)} starts, {int(finite.sum())} finite, {len(modes_a)} modes, {len(carried)} within margin; "
        f"{stage_a.n_global} of {stage_a.base.size} parameters; best log posterior {-best_a:.2f}; {seconds_a:.1f}s")
    # Stage B: L-BFGS on everything from each carried mode.
    stage_b = Objective(state, evaluator, starts[0], pinned=pinned, include_latent=True)
    t0 = time.perf_counter()
    finals, potentials_b, traces_b, iterations_b = [], [], [], []
    for mode in progress(carried, desc="L-BFGS modes"):
        theta = stage_a.theta(stage_a.expand(np.asarray(out["final"])[mode["representative"]]))
        active = stage_b.flatten(theta)[np.asarray(stage_b.global_index)]
        trace: list[float] = [float(stage_b.reduced(jnp.asarray(active)))]
        active, value, nit, _ = _scipy_lbfgs(stage_b.reduced_value_and_grad, active, config.lbfgs_max_iter, trace)
        finals.append(stage_b.expand(active)); potentials_b.append(value); traces_b.append(np.asarray(trace)); iterations_b.append(nit)
    finals = np.stack(finals); potentials_b = np.asarray(potentials_b)
    finite_b = np.isfinite(potentials_b) & (potentials_b < BAD_POTENTIAL)
    modes, _ = cluster_modes(finals, potentials_b, finite_b, finals, config.mode_parameter_tolerance)
    best = modes[0]["potential"]
    for mode in modes:
        mode.update(within_margin=bool(mode["potential"] - best <= float(config.mode_margin)), polished=True,
                    polish_iterations=int(iterations_b[mode["representative"]]), potential_polished=float(mode["potential"]),
                    theta=stage_b.theta(finals[mode["representative"]]).tolist(), log_posterior=float(-mode["potential"]), count=len(mode["members"]))
    seconds_b = time.perf_counter() - t0
    from .search import _pad_traces
    log(f"[hybrid] lbfgs: {len(carried)} starts, {len(modes)} modes, {sum(m['within_margin'] for m in modes)} within margin; "
        f"{stage_b.n_global} of {stage_b.base.size} parameters (scatter {'pinned at %.3g' % config.polish_scatter if pinned else 'free'}); "
        f"best log posterior {-best:.2f}; mean iterations {np.mean(iterations_b):.0f}; {seconds_b:.1f}s")
    log(f"[search] hybrid: adam {seconds_a:.1f}s + lbfgs {seconds_b:.1f}s; best log posterior {-best:.2f}")
    final_theta = np.stack([stage_a.theta(stage_a.expand(g)) if ok else np.full(len(starts[0]), np.nan) for g, ok in zip(out["final"], finite)])
    starts_in_basin = int(sum(len(m["members"]) for m in carried))          # stage-A end points within the margin of the Adam best
    return dict(theta=np.asarray(modes[0]["theta"], dtype=float), modes=modes, assignment=assignment.tolist(), potential=potential.tolist(),
                iterations=out["iterations"].tolist(), evaluations=out["evaluations"].tolist(), method="hybrid-adam-lbfgs",
                starts_in_basin=starts_in_basin, configured_in_basin=bool(any(0 in m["members"] for m in carried)),
                traces=np.asarray(out["traces"]), final_theta=final_theta, global_index=np.asarray(stage_a.global_index).tolist(),
                search_seconds=float(seconds_a), polish_seconds=float(seconds_b), n_finite=int(finite.sum()), n_global=int(stage_b.n_global),
                starts=np.asarray(starts).tolist(),
                stages=[{"stage": "adam_lens_block", "active": stage_a.n_global, "starts": len(starts), "finite": int(finite.sum()),
                         "best_log_posterior": float(-best_a), "kept": len(carried), "seconds": seconds_a},
                        {"stage": "lbfgs_all", "active": stage_b.n_global, "starts": len(carried), "finite": int(finite_b.sum()),
                         "best_log_posterior": float(-best), "kept": len(modes), "seconds": seconds_b}],
                stage_traces={"adam_lens_block": np.asarray(out["traces"]).tolist(), "lbfgs_all": _pad_traces(traces_b).tolist()})
