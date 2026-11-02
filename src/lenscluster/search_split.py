"""The init search: the large-scale model first (halos, shear and independent halos, no member population), then the
members with the large-scale parameters frozen, then a full-dimension polish from the best modes."""
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


def split_map(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator, *, seed: int = 0, log=print) -> dict[str, Any]:
    """Stage A fits the large-scale model without the member population from ``n_starts`` prior starts (the configured
    model among them), with the error hyperparameters held at their starts in the search and in its polish: a lens without
    its members leaves a large residual that free error scales absorb.  Stage B starts each carried large-scale mode with
    ``member_starts`` prior draws of the member block and optimizes the sites outside the large family (members, sources and
    error scales); the polish then frees everything.  Under grf_gp both stages use system blocks (``search_blocks``)."""
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
    halo_config = replace(config, model=replace(config.model, member_populations=()))
    halo_state = search_blocks(build_state(halo_config))
    halo_evaluator = Evaluator(halo_state, config.likelihood.likelihood_mode)
    halo_starts = prior_starts(halo_state, int(search.n_starts), seed)
    halo_starts[0] = transfer_by_name(state, np.asarray(state.initial_theta, dtype=float), halo_state, halo_starts[0])
    n_members = len(state.members.member_ids)
    error_index = [i for i, spec in enumerate(halo_state.parameter_specs) if spec.component_family == "error_scale"]
    if getattr(halo_evaluator, "gp_knee", False) and error_index:
        # The error scales do not move in stage A, so the knee kernel blocks are computed once (exact: checked to 1e-15
        # in the gradient on CANUCS); every start carries the same error scales.
        halo_starts[:, error_index] = halo_starts[0, error_index]
        halo_evaluator.freeze_gp_kernels(halo_starts[0])
    error_sites = {spec.sample_site_name or spec.sample_name for spec in halo_state.parameter_specs if spec.component_family == "error_scale"}
    lens_sites = {spec.sample_site_name or spec.sample_name for spec in halo_state.parameter_specs} - error_sites
    n_error = sum(spec.component_family == "error_scale" for spec in halo_state.parameter_specs)
    log(f"[init] search stage A, the large-scale model alone: {halo_state.n_parameters - n_error} of {halo_state.n_parameters} parameters "
        f"(halos, shear, sources; the {n_error} error scales held at their starts, the {n_members} members removed), "
        f"{len(halo_starts)} prior starts including the configured model, {search.halo_method}")
    stage_a = multistart_map(halo_state, halo_evaluator, halo_starts, search, method=search.halo_method, seed=seed, log=log,
                             active_sites=lens_sites, polish_sites=lens_sites, label="stage A")
    carried = [np.asarray(m["theta"], dtype=float) for m in stage_a["modes"] if m["within_margin"]][:int(search.max_chain_modes)]
    seconds_a = time.perf_counter() - started
    member_starts = prior_starts(search_state, int(search.member_starts), seed + 1)
    starts_b = np.stack([transfer_by_name(halo_state, halo_theta, state, member_theta) for halo_theta in carried for member_theta in member_starts])
    frozen = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs if spec.component_family == "large"}
    active = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs} - frozen
    n_frozen = sum(spec.component_family == "large" for spec in state.parameter_specs)
    log(f"[init] search stage B, the members with the large-scale model frozen: {len(carried)} carried mode(s) x {len(member_starts)} prior draws of "
        f"the member block = {len(starts_b)} starts; {state.n_parameters - n_frozen} of {state.n_parameters} parameters active "
        f"(relation, scatters, deviations, sources, error scales), {search.member_method}; then an L-BFGS polish of everything")
    t0 = time.perf_counter()
    stage_b = multistart_map(search_state, evaluator, starts_b, search, method=search.member_method, seed=seed, log=log, active_sites=active, label="stage B")
    seconds_b = time.perf_counter() - t0
    total = time.perf_counter() - started
    stages = [
        {"stage": "A  large-scale model (members removed, errors held)", "method": search.halo_method, "active": int(stage_a["n_global"]), "starts": len(halo_starts),
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
