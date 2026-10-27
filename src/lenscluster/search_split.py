"""The init search: the large-scale model first (halos, shear and independent halos, no member population), then the
members with the large-scale parameters frozen, then a full-dimension polish from the best modes."""
from __future__ import annotations

import time
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


def split_map(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator, *, seed: int = 0, log=print) -> dict[str, Any]:
    """Stage A fits the model without the member population from ``n_starts`` prior starts (the configured model among
    them); stage B starts each carried large-scale mode with ``member_starts`` prior draws of the member block and
    optimizes only the sites outside the large family; the polish inside multistart_map then frees everything."""
    search = config.search
    started = time.perf_counter()
    halo_config = replace(config, model=replace(config.model, member_populations=()))
    halo_state = build_state(halo_config)
    halo_evaluator = Evaluator(halo_state, config.likelihood.likelihood_mode)
    halo_starts = prior_starts(halo_state, int(search.n_starts), seed)
    halo_starts[0] = transfer_by_name(state, np.asarray(state.initial_theta, dtype=float), halo_state, halo_starts[0])
    n_members = len(state.members.member_ids)
    log(f"[init] search stage A, the large-scale model alone: {halo_state.n_parameters} parameters (halos, shear, sources, error scales; "
        f"the {n_members} members removed), {len(halo_starts)} prior starts including the configured model, {search.halo_method}")
    stage_a = multistart_map(halo_state, halo_evaluator, halo_starts, search, method=search.halo_method, seed=seed, log=log, label="stage A")
    carried = [np.asarray(m["theta"], dtype=float) for m in stage_a["modes"] if m["within_margin"]][:int(search.max_chain_modes)]
    seconds_a = time.perf_counter() - started
    member_starts = prior_starts(state, int(search.member_starts), seed + 1)
    starts_b = np.stack([transfer_by_name(halo_state, halo_theta, state, member_theta) for halo_theta in carried for member_theta in member_starts])
    frozen = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs if spec.component_family == "large"}
    active = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs} - frozen
    n_frozen = sum(spec.component_family == "large" for spec in state.parameter_specs)
    log(f"[init] search stage B, the members with the large-scale model frozen: {len(carried)} carried mode(s) x {len(member_starts)} prior draws of "
        f"the member block = {len(starts_b)} starts; {state.n_parameters - n_frozen} of {state.n_parameters} parameters active "
        f"(relation, scatters, deviations, sources, error scales), {search.member_method}; then an L-BFGS polish of everything")
    t0 = time.perf_counter()
    stage_b = multistart_map(state, evaluator, starts_b, search, method=search.member_method, seed=seed, log=log, active_sites=active, label="stage B")
    seconds_b = time.perf_counter() - t0
    total = time.perf_counter() - started
    stages = [
        {"stage": "A  large-scale model (members removed)", "method": search.halo_method, "active": int(stage_a["n_global"]), "starts": len(halo_starts),
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
