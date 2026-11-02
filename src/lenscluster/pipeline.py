"""Two stages: init (classify, multi-start MAP, reclassify) finds the start; final (NUTS, audit, outputs) samples from it."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import artifacts
from .audit import audit_posterior
from .classification import (classify_members, far_field_threshold_rows, conservative_cut_radii_arcsec, member_cut_radii_arcsec, member_max_deflection_arcsec,
                             prior_quantile_beta, prior_quantile_cut_kpc)
from .config import LensClusterSolverConfig
from .evaluator import Evaluator
from .init_plots import write_init_plots
from .model import ModelState, PosteriorResults
from .outputs import write_outputs
from .report import audit_report, classification_report, far_field_threshold_table, run_summary_table
from .sampling import run_nuts
from .search import chain_starts
from .search_split import split_map
from .state import build_state, deviation_constraint_ratio, with_member_classes, convert_member_centering

STAGES = ("init", "final", "both")


@dataclass(frozen=True)
class StagePaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def fits(self) -> Path:
        return self.root / "fits"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def init(self) -> StagePaths:
        return StagePaths(self.root / "init")

    @property
    def final(self) -> StagePaths:
        return StagePaths(self.root / "final")


@dataclass
class InitResult:
    paths: StagePaths
    state: ModelState
    evaluator: Evaluator
    search: dict[str, Any]
    theta: np.ndarray
    resumed: bool
    elapsed_seconds: float = 0.0


@dataclass
class RunResult:
    paths: RunPaths
    init: InitResult
    state: ModelState
    evaluator: Evaluator
    posterior: PosteriorResults
    audit: dict[str, Any]
    resumed: bool
    summary: dict[str, Any] = field(default_factory=dict)      # what write_outputs returned (fit numbers under "fit")
    timing: dict[str, float] = field(default_factory=dict)     # seconds: init, sampling, outputs, final, total


class RunLog:
    def __init__(self, path: Path, quiet: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")
        self.quiet = quiet
        self.started = time.perf_counter()

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{time.perf_counter() - self.started:8.1f}s] {message}"
        self.handle.write(line + "\n")
        self.handle.flush()
        if not self.quiet:
            print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def run_paths(config: LensClusterSolverConfig) -> RunPaths:
    return RunPaths(Path(str(config.paths.output_dir)) / str(config.paths.run_name))


def _scaling_value(state: ModelState, physical: np.ndarray, field: str) -> float:
    for index, spec in enumerate(state.parameter_specs):
        if spec.component_family == "scaling" and spec.field == field:
            return float(physical[index])
    raise KeyError(field)


def classify(state: ModelState, evaluator: Evaluator, theta: np.ndarray, config: LensClusterSolverConfig, *,
             reference_cut_kpc: float | None, beta_radius: float | None, log) -> ModelState:
    """Classify members at ``theta`` with a conservative cut radius: the larger of ``cut_radius_inflation`` times the
    current-model value and the prior upper quantile of cutkpc (when a quantile is configured), with beta spanning the
    current value and the prior quantiles.  Either route leaves the sampled r_cut* room to grow before a far-field member
    reaches the guard; they are not multiplied."""
    physical = np.asarray(evaluator.physical(jnp.asarray(theta)))
    if reference_cut_kpc is None:
        reference_cut_kpc = max(float(p["cutkpc_nominal"]) for p in state.potfiles) if state.potfiles else 0.0
    if beta_radius is None:
        beta_radius = float(config.scaling.beta_radius_prior_mean)
    quantile = config.members.classification_prior_quantile
    betas = [float(beta_radius)]
    inflated = float(reference_cut_kpc) * float(config.members.cut_radius_inflation)
    if quantile is not None and state.potfiles:
        inflated = max(inflated, max(prior_quantile_cut_kpc(p, float(quantile)) for p in state.potfiles))
        betas += [prior_quantile_beta(config.scaling, float(quantile)), prior_quantile_beta(config.scaling, 1.0 - float(quantile))]
    cut = conservative_cut_radii_arcsec(state, inflated, betas)
    exact, diagnostics = classify_members(state, evaluator, physical, cut_radius_arcsec=cut,
                                          guard_cut_radii=float(config.members.far_field_cut_radii),
                                          tolerance=float(config.members.template_tolerance), top_exact=config.members.top_exact)
    diagnostics.update(reference_cut_kpc=float(reference_cut_kpc), inflated_cut_kpc=inflated, beta_radius=float(beta_radius),
                       beta_range=[float(min(betas)), float(max(betas))], prior_quantile=None if quantile is None else float(quantile),
                       threshold_study=far_field_threshold_rows(state, evaluator, physical, cut_radius_arcsec=cut,
                                                                tolerance=float(config.members.template_tolerance),
                                                                selected_guard=float(config.members.far_field_cut_radii),
                                                                top_exact=config.members.top_exact))
    diagnostics.update(guard_headroom(state, evaluator, physical, exact, guard=float(config.members.far_field_cut_radii)))
    log(f"[classify] members={diagnostics['member_count']} exact={diagnostics['exact_count']} "
        f"mandatory={diagnostics['mandatory_count']} cut_kpc={inflated:.1f} beta in [{min(betas):.2f}, {max(betas):.2f}] "
        f"guard headroom x{diagnostics['guard_headroom']:.2f}")
    if diagnostics["guard_headroom"] < MIN_GUARD_HEADROOM:
        log(f"[classify] WARNING: the nearest far-field member ({diagnostics['guard_headroom_member']}) sits at only "
            f"{diagnostics['guard_headroom']:.2f} times the far-field guard at the current model; the sampler will be pinned "
            f"against the guard as soon as r_cut* grows by that factor.  Raise members.cut_radius_inflation or the prior quantile.")
    return with_member_classes(state, exact, diagnostics)


MIN_GUARD_HEADROOM = 2.0


def guard_headroom(state: ModelState, evaluator: Evaluator, physical: np.ndarray, exact: np.ndarray, *, guard: float) -> dict[str, Any]:
    """By what factor the sampled r_cut* can grow before the nearest far-field member reaches the guard, at this model:
    min over far-field members of (R / r_cut_current) / guard, with r_cut_current from the current relation."""
    members = state.members
    n = len(members.member_ids)
    far = np.ones(n, dtype=bool)
    far[np.asarray(exact, dtype=int)] = False
    if not far.any() or not state.potfiles:
        return {"guard_headroom": float("inf"), "guard_headroom_member": None}
    cut_now = member_cut_radii_arcsec(state, _scaling_value(state, physical, "cutkpc"), _scaling_value(state, physical, "beta_radius"))
    separation = np.asarray(evaluator.geometry.separation).min(axis=1)        # (M,) nearest image per member
    ratio = separation[far] / np.maximum(cut_now[far], 1e-12) / float(guard)
    worst = int(np.argmin(ratio))
    return {"guard_headroom": float(ratio[worst]), "guard_headroom_member": str(np.asarray(members.member_ids)[far][worst])}


def reclassification_at_median(state: ModelState, evaluator: Evaluator, samples: np.ndarray, config: LensClusterSolverConfig) -> dict[str, Any]:
    """Would classifying at the posterior median change the exact set? Reports the members that would move."""
    theta = np.median(np.asarray(samples, dtype=float), axis=0)
    physical = np.asarray(evaluator.physical(jnp.asarray(theta)))
    trial = classify(state, evaluator, theta, config, reference_cut_kpc=_scaling_value(state, physical, "cutkpc"),
                     beta_radius=_scaling_value(state, physical, "beta_radius"), log=lambda _: None)
    ids = state.members.member_ids
    current = {ids[i] for i in state.members.exact_positions.tolist()}
    median = {ids[i] for i in trial.members.exact_positions.tolist()}
    return {"exact_count": len(median), "current_exact_count": len(current), "would_add": sorted(median - current),
            "would_remove": sorted(current - median), "changed": bool(median != current)}


def _seed(config: LensClusterSolverConfig) -> None:
    np.random.seed(config.runtime.seed)
    random.seed(config.runtime.seed)


def choose_centering(state: ModelState, evaluator: Evaluator, config: LensClusterSolverConfig, *, log) -> ModelState:
    """Decide which members sample a centered deviation, per MemberConfig.centering, and convert the start."""
    members = config.members
    n = len(state.members.member_ids)
    if members.centering == "none" or n == 0:
        centered = np.zeros(n, dtype=bool)
    elif members.centering == "exact":
        centered = np.zeros(n, dtype=bool)
        centered[state.members.exact_positions] = True
    else:
        scatter = float(config.search.polish_scatter) if config.search.polish_scatter else float(config.scaling.member_velocity_scatter_prior.median)
        ratio = deviation_constraint_ratio(state, evaluator, state.initial_theta, scatter)
        centered = ratio > float(members.centering_ratio)
        log(f"[classify] centering: data/prior curvature of the deviations at scatter {scatter:g}: median {np.median(ratio):.3g}, "
            f"{int(centered.sum())} members above {members.centering_ratio:g} -> centered"
            + (f" ({', '.join(state.members.member_ids[i] for i in np.flatnonzero(centered)[:12])})" if centered.any() else ""))
    return with_member_classes(state, state.members.exact_positions, state.members.classification, centered=centered)


def convert_search_centering(search: dict[str, Any], before: ModelState, after: ModelState) -> dict[str, Any]:
    """The search result moved to the member centering of ``after``: every mode's theta, the best theta and the end
    points, so that the saved modes, the chain starts and the Hessian seed all live in the coordinates NUTS samples."""
    centered = _member_centering(after)
    out = dict(search)
    out["modes"] = [dict(m, theta=convert_member_centering(before, np.asarray(m["theta"], dtype=float), centered).tolist()) for m in search["modes"]]
    out["theta"] = convert_member_centering(before, np.asarray(search["theta"], dtype=float), centered)
    if search.get("final_theta") is not None:
        out["final_theta"] = convert_member_centering(before, np.asarray(search["final_theta"], dtype=float), centered)
    return out


def _member_centering(state: ModelState) -> np.ndarray:
    return np.asarray(state.members.centered, dtype=bool) if len(state.members.centered) else np.zeros(len(state.members.member_ids), dtype=bool)


START_LOGLIKE_TOLERANCE = 1.0e-6     # the first chain start must reproduce the saved init start to rounding


def chain_start_check(evaluator: Evaluator, starts: np.ndarray, reference_theta: np.ndarray, *, margin: float) -> dict[str, Any]:
    """Compare the NUTS starts with the init's saved start under the model NUTS samples.  The first start must be the
    saved start; every start must be finite and within ``5 x margin`` in log-likelihood of it (the modes that seed chains
    are within ``margin`` of the best in log posterior, and the final classification can move them a little)."""
    reference = float(evaluator.loglike(jnp.asarray(reference_theta)))
    values = [float(evaluator.loglike(jnp.asarray(start))) for start in np.asarray(starts, dtype=float)]
    warnings = []
    if not np.array_equal(np.asarray(starts[0], dtype=float), np.asarray(reference_theta, dtype=float)) or abs(values[0] - reference) > START_LOGLIKE_TOLERANCE * max(1.0, abs(reference)):
        warnings.append(f"WARNING: the first chain start is not the saved init start (log-likelihood {values[0]:.3f} against {reference:.3f}); "
                        "the starts and the init are in different coordinates")
    for chain, value in enumerate(values):
        if not np.isfinite(value) or value < reference - 5.0 * float(margin):
            warnings.append(f"WARNING: chain {chain} starts at log-likelihood {value:.3f}, {reference - value:.3g} below the saved init start "
                            f"({reference:.3f}); more than 5 x the mode margin ({float(margin):g})")
    return {"reference_log_likelihood": reference, "start_log_likelihoods": values, "warnings": warnings}


def search_is_fragile(search: dict[str, Any]) -> bool:
    """True when at most one start reaches the best basin: the result rests on a single optimization, whether that was
    the configured model or one lucky prior draw."""
    if "starts_in_basin" in search:
        return int(search["starts_in_basin"]) <= 1
    modes = search["modes"]
    return sum(int(m.get("count", 1)) for m in modes if m["within_margin"]) <= 1


def init_verdict(search: dict[str, Any], *, fit: dict[str, Any] | None) -> str:
    """One line: did independent starts reach the best basin, or only the configured model?"""
    modes = search["modes"]
    n_starts = len(search.get("starts", []))
    in_basin = int(search.get("starts_in_basin", sum(int(m.get("count", 1)) for m in modes if m["within_margin"])))
    rms = f"; whitened rms at the start {fit['whitened_residual_rms']:.2f}" if fit else ""
    finite = search.get("n_finite")
    invalid = f"; {n_starts - int(finite)} of {n_starts} starts invalid" if finite is not None and int(finite) < n_starts else ""
    tag = f" [{search['method']}]"
    if search_is_fragile(search):
        return f"verdict: fragile{tag} ({in_basin} of {n_starts} starts reach the best basin){invalid}{rms}"
    return f"verdict: robust{tag} ({in_basin} of {n_starts} starts reach the best basin within the margin){invalid}{rms}"


def run_init(config: LensClusterSolverConfig) -> InitResult:
    """Init stage: provisional classification, multi-start MAP, final classification at the MAP, saved start."""
    started = time.perf_counter()
    result = _run_init(config)
    result.elapsed_seconds = time.perf_counter() - started
    return result


def _run_init(config: LensClusterSolverConfig) -> InitResult:
    config = config.validate()
    paths = run_paths(config).init
    for directory in (paths.artifacts, paths.tables, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    log = RunLog(paths.logs / "init.log", quiet=config.runtime.quiet)
    _seed(config)
    try:
        signature = artifacts.init_signature(config)
        if config.runtime.resume and artifacts.is_init_complete(paths.artifacts):
            saved = json.loads((paths.artifacts / artifacts.INIT_SIGNATURE_FILE).read_text())
            if saved != artifacts._jsonable(signature):
                raise ValueError("The saved init stage was made with a different model, likelihood, member or search configuration; use a new run name.")
            log("[init] complete init stage found; loading the saved start")
            saved_classes = json.loads((paths.artifacts / artifacts.CLASSIFICATION_FILE).read_text())
            state = build_state(config, drop_member_ids=saved_classes.get("dropped_members", ()))
            centered = np.asarray([m in set(saved_classes.get("centered_members", [])) for m in state.members.member_ids], dtype=bool)
            state = with_member_classes(state, artifacts.load_exact_positions(paths.artifacts), saved_classes, centered=centered)
            theta = artifacts.load_start(paths.artifacts)
            state.initial_theta = theta
            evaluator, search = Evaluator(state, config.likelihood.likelihood_mode), artifacts.load_search(paths.artifacts)
            if not config.runtime.skip_plots and not (paths.figures / "init_fit.pdf").exists() and not any(paths.figures.glob("*init_fit.pdf")):
                write_init_plots(state, evaluator, search, theta, paths.figures, margin=config.search.mode_margin,
                                 prefix=config.paths.figure_prefix, z_source=float(config.truth.caustic_source_redshift), log=log)
            return InitResult(paths, state, evaluator, search, theta, resumed=True)

        artifacts.write_json(paths.artifacts / artifacts.INIT_SIGNATURE_FILE, signature)
        artifacts.save_config(paths.artifacts, config)
        log(f"[init] building model for run {config.paths.run_name}")
        state = build_state(config)
        log(f"[init] parameters={state.n_parameters} images={state.images.n_images} families={state.images.n_families} "
            f"members={len(state.members.member_ids)} components={len(state.base_components)}")
        evaluator = Evaluator(state, config.likelihood.likelihood_mode)

        # 0. Optional impact cut: members that deflect every image by less than the threshold leave the model.
        floor = config.members.min_deflection_arcsec
        if floor:
            deflection = member_max_deflection_arcsec(state, evaluator, state.initial_theta)
            dropped = tuple(mid for mid, value in zip(state.members.member_ids, deflection) if value < float(floor))
            if dropped:
                log(f"[classify] impact cut: dropping {len(dropped)} of {len(deflection)} members whose deflection at every image is "
                    f"below {floor:g}\" at the initial model (summed maxima {float(np.sum(deflection[deflection < floor])):.3f}\"); "
                    f"faintest kept member {max(float(r['catalog_mag']) for r, d in zip(state.scaling_component_records, deflection) if d >= floor):.1f} mag")
                state = build_state(config, drop_member_ids=dropped)
                evaluator = Evaluator(state, config.likelihood.likelihood_mode)
                log(f"[init] parameters={state.n_parameters} members={len(state.members.member_ids)} after the impact cut")

        # 1. Provisional classification from the prior-inflated cut radii at the initial model.
        state = classify(state, evaluator, state.initial_theta, config, reference_cut_kpc=None, beta_radius=None, log=log)
        evaluator = Evaluator(state, config.likelihood.likelihood_mode)

        # 2. The search: large-scale model, then members with the large family frozen, then a polish of everything.
        search = split_map(config, state, evaluator, seed=config.runtime.seed, log=log)
        theta = np.asarray(search["theta"], dtype=float)
        physical = np.asarray(evaluator.physical(jnp.asarray(theta)))

        # 3. Final classification at the best mode; this is the model the final stage samples.
        provisional = state.members.classification
        state = classify(state, evaluator, theta, config, reference_cut_kpc=_scaling_value(state, physical, "cutkpc"),
                         beta_radius=_scaling_value(state, physical, "beta_radius"), log=log)
        state.initial_theta = theta
        before_centering = state
        state = choose_centering(state, Evaluator(state, config.likelihood.likelihood_mode), config, log=log)
        search = convert_search_centering(search, before_centering, state)
        theta = np.asarray(state.initial_theta, dtype=float)
        state.members.classification["dropped_members"] = list(state.dropped_member_ids)
        state.members.classification["min_deflection_arcsec"] = config.members.min_deflection_arcsec
        report = classification_report(state, provisional=provisional) + "\n" + far_field_threshold_table(state)
        log(report)
        (paths.tables / "classification_report.txt").write_text(report + "\n")
        evaluator = Evaluator(state, config.likelihood.likelihood_mode)
        value = float(evaluator.loglike(jnp.asarray(theta)))
        log(f"[init] log-likelihood at the start with final classes: {value:.3f}")
        if not np.isfinite(value):
            raise RuntimeError("The best mode is rejected by the likelihood after reclassification.")
        if evaluator.gp_mode:
            # Pre-run check of the shallow fold pairs at the start; sigma_fl is not changed here, the report says when to lower it.
            from .gp_image_plane import shallow_fold_pair_report, shallow_fold_pairs
            if getattr(evaluator, "gp_knee", False):
                scales = {k: float(v) for k, v in evaluator.error_scales(evaluator.physical(jnp.asarray(theta))).items()
                          if k in ("gp_sigma_alpha", "gp_knee", "gp_knee_gamma", "sigma_shared", "gp_ell", "sigma_white")}
                header = (f"grf_gp likelihood (GRF of unmodelled mass with a knee spectrum + image-plane GP) on {evaluator.gp_blocks} blocks at the start: sigma_alpha {scales['gp_sigma_alpha']:.3f}\" knee {scales['gp_knee']:.2f}\" slope {scales['gp_knee_gamma']:.2f} "
                          f"sigma_shared {scales['sigma_shared']:.3f}\" at {scales['gp_ell']:.2f}\" sigma_white {scales['sigma_white']:.3f}\"")
                pairs = shallow_fold_pairs(state, evaluator, theta)
                log(shallow_fold_pair_report(pairs, evaluator.gp_floor, header=header, potential=True))
                artifacts.write_json(paths.tables / "shallow_fold_pairs.json", {"pairs": pairs, "model": "init start", "likelihood": "grf_gp", "gp_kernel": "knee",
                                                                                 "gp_blocks": evaluator.gp_blocks, **scales})
            elif evaluator.gp_potential:
                scales = {k: float(v) for k, v in evaluator.error_scales(evaluator.physical(jnp.asarray(theta))).items()
                          if k in ("gp_u_small", "gp_ell_small", "gp_sigma_small", "gp_sigma_large", "gp_ell_large", "sigma_extra")}
                header = (f"{evaluator.gp_kernel_family} potential kernel at the start: sigma_S {scales['gp_sigma_small']:.3f}\" ell_S {scales['gp_ell_small']:.2f}\" "
                          f"sigma_L {scales['gp_sigma_large']:.3f}\" ell_L {scales['gp_ell_large']:.1f}\" sigma_extra {scales['sigma_extra']:.3f}\"")
                pairs = shallow_fold_pairs(state, evaluator, theta)
                log(shallow_fold_pair_report(pairs, evaluator.gp_floor, header=header, potential=True))
                artifacts.write_json(paths.tables / "shallow_fold_pairs.json", {"pairs": pairs, "model": "init start", "gp_kernel": "potential",
                                                                                 "gp_kernel_family": evaluator.gp_kernel_family, **scales})
            else:
                shared, white = float(config.likelihood.sigma_shared_start_arcsec), float(config.likelihood.sigma_white_start_arcsec)
                pairs = shallow_fold_pairs(state, evaluator, theta, sigma_shared_arcsec=shared, sigma_white_arcsec=white)
                log(shallow_fold_pair_report(pairs, evaluator.gp_floor, shared, white))
                artifacts.write_json(paths.tables / "shallow_fold_pairs.json", {"pairs": pairs, "model": "init start", "sigma_shared_arcsec": shared, "sigma_white_arcsec": white})
        artifacts.save_state(paths.artifacts, state)
        artifacts.save_search(paths.artifacts, search)
        artifacts.save_start(paths.artifacts, theta, {"log_likelihood": value, "log_posterior": search["modes"][0]["log_posterior"],
                                                       "modes_within_margin": int(sum(m["within_margin"] for m in search["modes"]))})
        summary = {"run_name": config.paths.run_name, "n_parameters": state.n_parameters, "images": state.images.n_images,
                   "families": state.images.n_families, "members": len(state.members.member_ids),
                   "exact_members": int(len(state.members.exact_positions)), "search_method": search["method"],
                   "modes": len(search["modes"]), "modes_within_margin": int(sum(m["within_margin"] for m in search["modes"])),
                   "best_log_posterior": search["modes"][0]["log_posterior"], "search_seconds": search["search_seconds"],
                   "polish_seconds": search["polish_seconds"]}
        verdict = init_verdict(search, fit=None)
        log(f"[init] {verdict}")
        summary["verdict"] = verdict
        if not config.runtime.skip_plots:
            summary["fit_at_start"] = write_init_plots(state, evaluator, search, theta, paths.figures, margin=config.search.mode_margin,
                                                       prefix=config.paths.figure_prefix, z_source=float(config.truth.caustic_source_redshift), log=log)
        artifacts.write_json(paths.artifacts / artifacts.SUMMARY_FILE, summary)
        return InitResult(paths, state, evaluator, search, theta, resumed=False)
    finally:
        log.close()


def run_final(config: LensClusterSolverConfig, init: InitResult) -> RunResult:
    """Final stage: NUTS from the init start, posterior audit, tables and figures."""
    config = config.validate()
    paths = run_paths(config)
    stage = paths.final
    for directory in (stage.artifacts, stage.figures, stage.tables, stage.fits, stage.logs):
        directory.mkdir(parents=True, exist_ok=True)
    log = RunLog(stage.logs / "final.log", quiet=config.runtime.quiet)
    runtime = config.runtime
    _seed(config)
    state, evaluator = init.state, init.evaluator
    try:
        if runtime.resume and artifacts.is_final_complete(stage.artifacts):
            saved = artifacts.load_config(stage.artifacts)
            if saved.to_json_dict() != config.to_json_dict():
                raise ValueError("The saved final stage was made with a different configuration; use a new run name.")
            log("[final] complete posterior found; loading artifacts")
            posterior = artifacts.load_posterior(stage.artifacts)
            audit = json.loads((stage.artifacts / artifacts.AUDIT_FILE).read_text())
            outputs_started = time.perf_counter()
            summary = write_outputs(config, state, evaluator, posterior, audit, stage, log=log) if not runtime.skip_plots else {}
            timing = {"sampling": 0.0, "outputs": time.perf_counter() - outputs_started}
            return RunResult(paths, init, state, evaluator, posterior, audit, resumed=True, summary=summary, timing=timing)

        artifacts.save_config(stage.artifacts, config)
        starts = chain_starts(init.search, runtime.chains, config.search.max_chain_modes)
        sampling = config.sampling
        log(f"[final] NUTS from {len({tuple(s) for s in starts.tolist()})} distinct start(s), chains={runtime.chains} "
            f"warmup={sampling.warmup} samples={sampling.samples}")
        start_check = chain_start_check(evaluator, starts, init.theta, margin=config.search.mode_margin)
        log(f"[final] chain-start log-likelihoods {', '.join(f'{v:.2f}' for v in start_check['start_log_likelihoods'])} "
            f"(saved init start {start_check['reference_log_likelihood']:.2f})")
        for warning in start_check["warnings"]:
            log(f"[final] {warning}")
        sampling_started = time.perf_counter()
        posterior = run_nuts(state, evaluator, starts, warmup=sampling.warmup, samples=sampling.samples, chains=runtime.chains,
                             chain_method=runtime.nuts_chain_method, max_tree_depth=sampling.max_tree_depth,
                             target_accept=sampling.target_accept, initial_step_size=sampling.initial_step_size,
                             seed=runtime.seed, thin=sampling.thin, print_summary=runtime.numpyro_print_summary,
                             mass_matrix=sampling.mass_matrix, log=log)
        posterior.diagnostics["start_check"] = start_check
        artifacts.save_posterior(stage.artifacts, posterior)
        sampling_seconds = time.perf_counter() - sampling_started

        audit = audit_posterior(state, evaluator, posterior.samples, tolerance=config.members.template_tolerance,
                                guard_cut_radii=config.members.far_field_cut_radii)
        audit["reclassification_at_median"] = reclassification_at_median(state, evaluator, posterior.samples, config)
        artifacts.write_json(stage.artifacts / artifacts.AUDIT_FILE, audit)
        diagnostics = posterior.diagnostics or {}
        report = audit_report(state, audit, posterior_summary={
            "divergences": int(posterior.diverging.sum()), "mean accept": f"{float(posterior.accept_prob.mean()):.3f}",
            "mean steps": f"{float(posterior.num_steps.mean()):.1f}", "max R-hat": f"{diagnostics.get('max_r_hat', float('nan')):.3f}",
            "min ESS": f"{diagnostics.get('min_n_eff', float('nan')):.0f}"})
        log(report)
        (stage.tables / "audit_report.txt").write_text(report + "\n")
        summary = {"run_name": config.paths.run_name, "n_parameters": state.n_parameters, "images": state.images.n_images,
                   "families": state.images.n_families, "members": len(state.members.member_ids),
                   "exact_members": int(len(state.members.exact_positions)), "nuts": posterior.diagnostics,
                   "divergences": int(posterior.diverging.sum()), "mean_steps": float(posterior.num_steps.mean()),
                   "audit": audit["approximation_validation"]}
        artifacts.write_json(stage.artifacts / artifacts.SUMMARY_FILE, summary)
        outputs_started = time.perf_counter()
        outputs = write_outputs(config, state, evaluator, posterior, audit, stage, log=log) if not runtime.skip_plots else {}
        timing = {"sampling": sampling_seconds, "outputs": time.perf_counter() - outputs_started}
        return RunResult(paths, init, state, evaluator, posterior, audit, resumed=False, summary=outputs, timing=timing)
    finally:
        log.close()


def run(config: LensClusterSolverConfig, stage: str = "both") -> RunResult | InitResult:
    """Run the init stage, the final stage (which needs a completed init), or both."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}; got {stage!r}")
    if stage == "final":
        paths = run_paths(config).init
        if not artifacts.is_init_complete(paths.artifacts):
            raise FileNotFoundError(f"The final stage needs a completed init stage under {paths.artifacts}")
    started = time.perf_counter()
    init = run_init(config)
    if stage == "init":
        return init
    result = run_final(config, init)
    result.timing.update(init=init.elapsed_seconds, total=time.perf_counter() - started)
    from .results_tex import write_run_results
    # Preserve original full-run timing when only reloading/replotting a run.
    if not result.resumed and not init.resumed:
        artifacts.write_json(result.paths.final.tables / "runtime.json", result.timing)
    write_run_results(result.paths.final.root, name=config.paths.figure_prefix)
    table = run_summary_table(result, config)
    (result.paths.final.tables / "run_summary.txt").write_text(table + "\n")
    if not config.runtime.quiet:
        print(table, flush=True)
    return result


def dry_run(config: LensClusterSolverConfig) -> dict[str, Any]:
    """Describe both stages without running them: model size, provisional classification, search and sampling sizes."""
    config = config.validate()
    state = build_state(config)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    floor = config.members.min_deflection_arcsec
    if floor:
        deflection = member_max_deflection_arcsec(state, evaluator, state.initial_theta)
        dropped = tuple(mid for mid, value in zip(state.members.member_ids, deflection) if value < float(floor))
        if dropped:
            state = build_state(config, drop_member_ids=dropped)
            evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    state = classify(state, evaluator, state.initial_theta, config, reference_cut_kpc=None, beta_radius=None, log=lambda _: None)
    classification = state.members.classification
    classification["dropped_members"] = list(state.dropped_member_ids)
    classification["min_deflection_arcsec"] = floor
    print(classification_report(state))
    print(far_field_threshold_table(state))
    return {
        "stages": {"init": f"classify -> large-scale search from {config.search.n_starts} prior starts -> members with the large family frozen -> polish -> reclassify at the MAP -> start.json",
                   "final": "NUTS from the start -> posterior audit -> tables and figures"},
        "output": {"init": str(run_paths(config).init.root), "final": str(run_paths(config).final.root)},
        "likelihood": {"smoothed": "smoothed-local-image-plane",
                       "gp_image_plane": ("gp-image-plane (correlated image-plane displacements per family)" if config.likelihood.gp_kernel == "image_plane"
                                          else f"grf_gp (GRF of unmodelled mass with a knee spectrum + image-plane GP, {config.likelihood.gp_blocks} blocks)" if config.likelihood.gp_kernel == "knee"
                                          else f"gp-potential ({config.likelihood.gp_kernel_family} deflection kernel, two scales, per family)")}.get(
            config.likelihood.likelihood_mode, config.likelihood.likelihood_mode),
        "parameters": int(state.n_parameters), "images": int(state.images.n_images), "families": int(state.images.n_families),
        "members": int(len(state.members.member_ids)), "components": int(len(state.base_components)),
        "provisional_classification": {"exact": classification.get("exact_count"), "mandatory": classification.get("mandatory_count"),
                                       "far_field": int(len(state.members.member_ids)) - int(classification.get("exact_count", 0)),
                                       "inflated_cut_kpc": classification.get("inflated_cut_kpc")},
        "search": {"halo_method": config.search.halo_method, "member_method": config.search.member_method, "n_starts": int(config.search.n_starts),
                   "member_starts": int(config.search.member_starts), "lbfgs_max_iter": int(config.search.lbfgs_max_iter),
                   "polish_max_iter": int(config.search.polish_max_iter), "polish_scatter": config.search.polish_scatter,
                   "max_chain_modes": int(config.search.max_chain_modes)},
        "sampling": {"chains": int(config.runtime.chains), "warmup": int(config.sampling.warmup), "samples": int(config.sampling.samples),
                     "max_tree_depth": int(config.sampling.max_tree_depth)},
        "config": config.to_json_dict(),
    }
