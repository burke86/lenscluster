"""The init search: the large-scale model first (halos, shear, independent halos and the member scaling relation, with
every member on the relation), then the members with the large-scale parameters frozen, then a full-dimension polish from
the best modes."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from dataclasses import replace
from typing import Any

import numpy as np

from .config import LensClusterSolverConfig
from .evaluator import Evaluator
from .model import ModelState
from .search import multistart_map, prior_starts
from .state import build_state


def transfer_by_name(source: ModelState, source_theta: np.ndarray, target: ModelState, target_theta: np.ndarray) -> np.ndarray:
    """``target_theta`` with every parameter that also exists in ``source`` (same spec name) set to its source value."""
    index = {spec.name: i for i, spec in enumerate(target.parameter_specs)}
    out = np.array(target_theta, dtype=float, copy=True)
    for i, spec in enumerate(source.parameter_specs):
        if spec.name in index:
            out[index[spec.name]] = source_theta[i]
    return out


def search_blocks(state: ModelState) -> ModelState:
    """The state the init search optimizes: under grf_gp the covariance on system blocks (all knots of a galaxy) whatever
    gp_blocks the final stage samples.  A covariance coupling all images lets a long-range error mode absorb what the
    incomplete lens of the search leaves out (w1000s500_prod: a 40\" image-plane GP over 137\" and massless members).
    The blocks are also grouped by size, which leaves every block term unchanged (checked to 1e-11 on CANUCS)."""
    if getattr(state, "gp_kernel", "") != "knee" or getattr(state, "gp_search_layout", False):
        return state
    out = copy.copy(state)
    out.gp_blocks = "system"
    out.gp_group_blocks = True              # blocks padded per size group (gp_knee.bucket_layouts): the same likelihood, cheaper
    out.gp_search_layout = True
    return out


def warm_start_theta(run: str | Path, target: ModelState) -> tuple[np.ndarray, int, bool]:
    """The saved init start of ``run`` (a run directory or its init/artifacts) on ``target``'s parameters, by name, with
    the run's centered member deviations converted back to unit offsets (the search samples unit offsets).  Returns the
    theta, the number of parameters taken from the run, and whether that run's init was not robust (or has no verdict)."""
    from . import artifacts
    directory = Path(run)
    if not (directory / artifacts.START_FILE).exists():
        directory = directory / "init" / "artifacts"
    if not (directory / artifacts.START_FILE).exists():
        raise FileNotFoundError(f"warm start: no saved init start under {run}")
    saved = json.loads((directory / artifacts.STATE_FILE).read_text())
    specs = saved["parameter_specs"]
    theta = np.asarray(artifacts.load_start(directory), dtype=float)
    by_sample = {s["sample_name"]: i for i, s in enumerate(specs)}
    for i, spec in enumerate(specs):
        if spec.get("centered") and spec.get("scale_site_name") in by_sample:
            theta[i] /= np.exp(theta[by_sample[spec["scale_site_name"]]])       # deviation -> unit offset
    index = {spec.name: i for i, spec in enumerate(target.parameter_specs)}
    out = np.array(target.initial_theta, dtype=float, copy=True)
    shared = 0
    for i, spec in enumerate(specs):
        if spec["name"] in index:
            out[index[spec["name"]]] = theta[i]
            shared += 1
    summary = directory / artifacts.SUMMARY_FILE
    verdict = json.loads(summary.read_text()).get("verdict", "") if summary.exists() else ""
    return out, shared, not str(verdict).startswith("verdict: robust")


STAGE_A_INACTIVE = frozenset({"error_scale", "member_velocity_scatter", "member_scatter_offset", "member_shape_offset"})
LARGE_GEOMETRY_FIELDS = frozenset({"x_centre", "y_centre", "e1", "e2", "gamma1", "gamma2"})


def _with_relation(theta: np.ndarray, draw: np.ndarray, relation_index: list[int]) -> np.ndarray:
    out = np.array(theta, dtype=float, copy=True)
    out[relation_index] = draw[relation_index]
    return out


def diverse_modes(stage_a: dict[str, Any], specs, carried: list[np.ndarray], count: int, margin: float) -> list[dict[str, Any]]:
    """Up to ``count`` stage-A modes outside ``carried``, non-degenerate and within ``margin`` of the best, chosen by
    farthest-point sampling in the large family's coordinates scaled by their spread over the candidates."""
    if count <= 0:
        return []
    large = [i for i, spec in enumerate(specs) if spec.component_family == "large"]
    best = max(float(m["log_posterior"]) for m in stage_a["modes"] if not m["degenerate"])
    pool = [m for m in stage_a["modes"] if not m["degenerate"] and best - float(m["log_posterior"]) <= margin
            and not any(np.allclose(np.asarray(m["theta"], dtype=float), c) for c in carried)]
    if not pool or not large:
        return []
    points = np.stack([np.asarray(m["theta"], dtype=float)[large] for m in pool])
    chosen_points = np.stack([np.asarray(c, dtype=float)[large] for c in carried]) if carried else points[:1]
    scale = np.std(np.vstack([points, chosen_points]), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    picked: list[dict[str, Any]] = []
    distance = np.min(np.linalg.norm((points[:, None, :] - chosen_points[None]) / scale, axis=-1), axis=1)
    for _ in range(min(count, len(pool))):
        index = int(np.argmax(distance))
        if distance[index] <= 0:
            break
        picked.append(pool[index])
        distance = np.minimum(distance, np.linalg.norm((points - points[index]) / scale, axis=1))
    return picked


def split_map(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator, *, seed: int = 0, log=print) -> dict[str, Any]:
    """Stage A fits the large-scale model and the member scaling relation from ``n_starts`` prior starts (the configured
    model among them): halos, shear, sources and the relation's normalizations and slopes move, every member sits on the
    relation (offsets zero), and the error hyperparameters are held at their starts in the search and in its polish (a lens
    that leaves a large residual lets free error scales absorb it).  Drawing the relation from its prior per start varies
    the members' size while the halos are fitted; with the members removed instead (until 2026-11-04) the halos absorbed
    the member mass and the polish never gave it back (M0416's southern halo, 20 in log L).  Stage B starts each carried mode
    with ``member_starts`` prior draws of the relation (the first keeps the carried relation) and optimizes the sites outside
    the large family (members, sources and error scales); the polish then frees everything.  Under grf_gp both stages use
    system blocks (``search_blocks``)."""
    search = config.search
    started = time.perf_counter()
    search_state = search_blocks(state)
    if search_state is not state:
        log(f"[init] the search uses the covariance on system blocks grouped by size; the final stage samples {state.gp_blocks} blocks")
        evaluator = Evaluator(search_state, config.likelihood.likelihood_mode)
    if search.warm_start:
        theta, shared, fragile = warm_start_theta(search.warm_start, search_state)
        log(f"[init] warm start from {search.warm_start}: {shared} of {search_state.n_parameters} parameters taken from its saved start; "
            f"stages A and B skipped, L-BFGS and the polish from it")
        result = multistart_map(search_state, evaluator, theta[None], search, method="lbfgs", seed=seed, log=log, label="warm start")
        stages = [{"stage": "warm start (L-BFGS + polish)", "method": "lbfgs", "active": int(result["n_global"]), "starts": 1,
                   "finite": int(result["n_finite"]), "modes": len(result["modes"]), "within_margin": int(sum(m["within_margin"] for m in result["modes"])),
                   "best_log_posterior": float(result["modes"][0]["log_posterior"]), "mean_iterations": float(result["mean_iterations"]),
                   "seconds": float(result["search_seconds"] + result["polish_seconds"]), "kept": 1}]
        result.update(method="warm-start", stages=stages, stage_traces={"warm_start": np.asarray(result["traces"]).tolist()},
                      total_seconds=float(time.perf_counter() - started), warm_start=str(search.warm_start), warm_start_shared=int(shared),
                      warm_start_fragile=bool(fragile))
        from .report import search_summary_table
        log(search_summary_table(result))
        return result
    halo_state = search_state
    halo_evaluator = Evaluator(halo_state, config.likelihood.likelihood_mode)      # its kernels are frozen below; stage B keeps ``evaluator``
    halo_starts = prior_starts(halo_state, int(search.n_starts), seed)              # halos, shear and the relation from the prior; offsets zero
    halo_starts[0] = np.asarray(state.initial_theta, dtype=float)
    n_members = len(state.members.member_ids)
    specs_a = halo_state.parameter_specs
    error_index = [i for i, spec in enumerate(specs_a) if spec.component_family == "error_scale"]
    if getattr(halo_evaluator, "gp_knee", False) and error_index:
        # The error scales do not move in stage A, so the knee kernel blocks are computed once (exact: checked to 1e-15
        # in the gradient on CANUCS); every start carries the same error scales.
        halo_starts[:, error_index] = halo_starts[0, error_index]
        halo_evaluator.freeze_gp_kernels(halo_starts[0])
    lens_sites = {spec.sample_site_name or spec.sample_name for spec in specs_a if spec.component_family not in STAGE_A_INACTIVE}
    n_active_a = sum(spec.component_family not in STAGE_A_INACTIVE for spec in specs_a)
    n_error = sum(spec.component_family == "error_scale" for spec in specs_a)
    n_relation = sum(spec.component_family == "scaling" for spec in specs_a)
    log(f"[init] search stage A, the large-scale model with the members on their relation: {n_active_a} of {halo_state.n_parameters} parameters "
        f"(halos, shear, sources and the {n_relation} relation parameters; the {n_error} error scales held at their starts, the {n_members} members' "
        f"offsets and scatters at zero), {len(halo_starts)} prior starts including the configured model, {search.halo_method}")
    stage_a = multistart_map(halo_state, halo_evaluator, halo_starts, search, method=search.halo_method, seed=seed, log=log,
                             active_sites=lens_sites, polish_sites=lens_sites, label="stage A")
    carried = [np.asarray(m["theta"], dtype=float) for m in stage_a["modes"] if m["within_margin"]][:int(search.max_chain_modes)]
    seconds_a = time.perf_counter() - started
    member_starts = prior_starts(search_state, int(search.member_starts), seed + 1)
    relation_index = [i for i, spec in enumerate(specs_a) if spec.component_family == "scaling"]
    starts_b = []
    for halo_theta in carried:
        for k, member_theta in enumerate(member_starts):
            start = np.array(halo_theta, dtype=float, copy=True)
            if k > 0:                                                                # the first start keeps stage A's relation
                start[relation_index] = member_theta[relation_index]
            starts_b.append(start)
    starts_b = np.stack(starts_b)
    # The large family is frozen in stage B, except (search.stage_b_free_amplitudes) the halos' amplitudes and radii: a
    # member solution with large deviations needs a smaller halo next to it, which a frozen halo cannot give (M0416's
    # southern halo, 2026-11-05).  Centres, shapes and shear stay frozen.
    free_amplitudes = bool(getattr(search, "stage_b_free_amplitudes", False))
    frozen = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs
              if spec.component_family == "large" and not (free_amplitudes and spec.field not in LARGE_GEOMETRY_FIELDS)}
    active = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs} - frozen
    n_frozen = sum(spec.component_family == "large" and not (free_amplitudes and spec.field not in LARGE_GEOMETRY_FIELDS)
                   for spec in state.parameter_specs)
    log(f"[init] search stage B, the members with the large-scale model frozen: {len(carried)} carried mode(s) x {len(member_starts)} prior draws of "
        f"the member block = {len(starts_b)} starts; {state.n_parameters - n_frozen} of {state.n_parameters} parameters active "
        f"(relation, scatters, deviations, sources, error scales{', halo amplitudes and radii' if free_amplitudes else ''}), "
        f"{search.member_method}; then an L-BFGS polish of everything")
    t0 = time.perf_counter()
    stage_b = multistart_map(search_state, evaluator, starts_b, search, method=search.member_method, seed=seed, log=log, active_sites=active, label="stage B")
    diverse = diverse_modes(stage_a, specs_a, carried, int(getattr(search, "diverse_carry", 0)), float(getattr(search, "diverse_carry_margin", 0.0)))
    if diverse:
        log(f"[init] diverse carry: {len(diverse)} more stage-A mode(s), up to {float(search.diverse_carry_margin):g} behind the best, each "
            f"with its own stage B and polish; the chain-start step keeps them only within the margin on the NUTS target")
        single = replace(search, max_chain_modes=1)
        for k, mode in enumerate(diverse):
            starts_k = np.stack([np.array(mode["theta"], dtype=float) if j == 0 else
                                 _with_relation(np.asarray(mode["theta"], dtype=float), member_starts[j], relation_index) for j in range(len(member_starts))])
            extra = multistart_map(search_state, evaluator, starts_k, single, method=search.member_method, seed=seed + 2 + k, log=log,
                                   active_sites=active, label=f"stage B, diverse {k + 1} (stage A {float(mode['log_posterior']):.1f})")
            best = extra["modes"][0]
            if not best["degenerate"]:
                best.update(diverse_origin=True, within_margin=True)
                stage_b["modes"].append(best)
        stage_b["modes"].sort(key=lambda m: (m["degenerate"], m["potential_polished"]))
        stage_b["theta"] = np.asarray(stage_b["modes"][0]["theta"], dtype=float)
    seconds_b = time.perf_counter() - t0
    total = time.perf_counter() - started
    stages = [
        {"stage": "A  large-scale model + member relation (offsets zero, errors held)", "method": search.halo_method, "active": int(stage_a["n_global"]), "starts": len(halo_starts),
         "finite": int(stage_a["n_finite"]), "modes": len(stage_a["modes"]), "within_margin": int(sum(m["within_margin"] for m in stage_a["modes"])),
         "best_log_posterior": float(stage_a["best_before_polish"]), "mean_iterations": float(stage_a["mean_iterations"]), "seconds": seconds_a, "kept": len(carried)},
        {"stage": "B  members (large-scale model frozen)", "method": search.member_method, "active": int(stage_b["n_global"]), "starts": len(starts_b),
         "finite": int(stage_b["n_finite"]), "modes": len(stage_b["modes"]), "within_margin": int(sum(m["within_margin"] for m in stage_b["modes"])),
         "best_log_posterior": float(stage_b["best_before_polish"]), "mean_iterations": float(stage_b["mean_iterations"]), "seconds": float(stage_b["search_seconds"]),
         "kept": int(sum(m.get("polished", False) for m in stage_b["modes"]))},
        {"stage": "polish (everything free)", "method": "lbfgs", "active": int(stage_b["n_polish"]), "starts": int(sum(m.get("polished", False) for m in stage_b["modes"])),
         "finite": None, "modes": len(stage_b["modes"]), "within_margin": int(sum(m["within_margin"] for m in stage_b["modes"])),
         "best_log_posterior": float(stage_b["modes"][0]["log_posterior"]), "mean_iterations": float(stage_b["polish_iterations"]),
         "seconds": float(stage_b["polish_seconds"]), "kept": int(sum(m["within_margin"] for m in stage_b["modes"]))},
    ]
    stage_b.update(method=f"{search.halo_method}+{search.member_method}", stages=stages,
                   stage_traces={"large_scale": np.asarray(stage_a["traces"]).tolist(), "members": np.asarray(stage_b["traces"]).tolist()},
                   search_seconds=float(seconds_a + stage_b["search_seconds"]), total_seconds=float(total))
    from .report import search_summary_table
    log(search_summary_table(stage_b))
    return stage_b
