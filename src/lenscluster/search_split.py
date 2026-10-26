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
    stage_a = multistart_map(halo_state, halo_evaluator, halo_starts, search, method=search.halo_method, seed=seed, log=log)
    carried = [np.asarray(m["theta"], dtype=float) for m in stage_a["modes"] if m["within_margin"]][:int(search.max_chain_modes)]
    seconds_a = time.perf_counter() - started
    log(f"[init] large-scale stage ({search.halo_method}): {halo_state.n_parameters} parameters, {len(halo_starts)} starts, "
        f"{len(carried)} modes carried, best log posterior {stage_a['modes'][0]['log_posterior']:.2f}, {seconds_a:.1f}s")
    member_starts = prior_starts(state, int(search.member_starts), seed + 1)
    starts_b = np.stack([transfer_by_name(halo_state, halo_theta, state, member_theta) for halo_theta in carried for member_theta in member_starts])
    frozen = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs if spec.component_family == "large"}
    active = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs} - frozen
    t0 = time.perf_counter()
    stage_b = multistart_map(state, evaluator, starts_b, search, method=search.member_method, seed=seed, log=log, active_sites=active)
    seconds_b = time.perf_counter() - t0
    log(f"[init] member stage ({search.member_method}): {len(active)} of {len(active) + len(frozen)} sites active, {len(starts_b)} starts, "
        f"best log posterior {stage_b['modes'][0]['log_posterior']:.2f} after the polish, {seconds_b:.1f}s; search total {time.perf_counter() - started:.1f}s")
    stage_b.update(method=f"{search.halo_method}+{search.member_method}",
                   stages=[{"stage": "large_scale", "active": int(stage_a["n_global"]), "starts": len(halo_starts), "finite": int(stage_a["n_finite"]),
                            "best_log_posterior": float(stage_a["modes"][0]["log_posterior"]), "kept": len(carried), "seconds": seconds_a},
                           {"stage": "members", "active": int(stage_b["n_global"]), "starts": len(starts_b), "finite": int(stage_b["n_finite"]),
                            "best_log_posterior": float(stage_b["modes"][0]["log_posterior"]), "kept": len(stage_b["modes"]), "seconds": seconds_b}],
                   stage_traces={"large_scale": np.asarray(stage_a["traces"]).tolist(), "members": np.asarray(stage_b["traces"]).tolist()},
                   search_seconds=float(seconds_a + stage_b["search_seconds"]))
    return stage_b
