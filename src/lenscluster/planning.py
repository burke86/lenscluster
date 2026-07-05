from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import LensClusterSolverConfig, PriorConfig


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int
    chains: int
    resume: str | bool
    plots_only: bool
    skip_validation: bool
    skip_plots: bool
    quiet: bool
    jax_default_device: str
    smc_device: str


@dataclass(frozen=True)
class OutputPlan:
    run_name: str
    output_dir: Path
    stage0_minimal_outputs: bool


@dataclass(frozen=True)
class DiagnosticsPlan:
    truth_recovery_enabled: bool
    image_recovery_enabled: bool
    image_catalog_cutouts_enabled: bool


@dataclass(frozen=True)
class StagePlan:
    name: str
    fit_mode: str
    fit_method: str
    sampling_engine: str
    sample_likelihood_mode: str
    likelihood_family: str
    source_position_policy: str
    svi_steps: int
    refresh_every: int | None
    warmup: int
    samples: int
    sampling_refresh_runs: int
    max_tree_depth: int
    output_plan: OutputPlan


@dataclass(frozen=True)
class RunPlan:
    config: LensClusterSolverConfig
    runtime_args: "SolverRuntime" = field(repr=False, compare=False)
    runtime: RuntimeSettings
    output: OutputPlan
    diagnostics: DiagnosticsPlan
    stages: tuple[StagePlan, ...]
    run_metadata: dict[str, Any]


@dataclass(frozen=True)
class SolverRuntime:
    values: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            values = object.__getattribute__(self, "values")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getstate__(self) -> dict[str, Any]:
        return {"values": object.__getattribute__(self, "values")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        values = state.get("values")
        if not isinstance(values, dict):
            raise TypeError("SolverRuntime pickle state must contain a values dictionary.")
        object.__setattr__(self, "values", dict(values))


def _positive_lognormal_prior_payload(prefix: str, prior: PriorConfig) -> dict[str, float]:
    return {
        f"{prefix}_lower": float(prior.lower),
        f"{prefix}_upper": float(prior.upper),
        f"{prefix}_prior_median": float(prior.mean),
        f"{prefix}_prior_log_sigma": float(prior.std),
    }


SOLVER_RUNTIME_DEFAULTS: dict[str, Any] = {
    "active_blocked_nuts_block_library_size": None,
    "active_blocked_nuts_block_size": None,
    "active_blocked_nuts_cycles": None,
    "active_blocked_nuts_global_period": 1,
    "arc_aware_curve_step_arcsec": 0.1,
    "arc_aware_max_arclength_arcsec": 5.0,
    "arc_recovery_p_arc_threshold": 0.1,
    "blocked_nuts_cycles": None,
    "blocked_nuts_pilot_warmup": None,
    "cab_curvature_sigma_floor_arcsec_inv": 1.0e-4,
    "cab_finite_difference_step_arcsec": 1.0e-3,
    "cab_likelihood_weight": None,
    "cab_tangent_sigma_floor_rad": 1.0e-3,
    "caustic_plot_grid_scale_arcsec": 0.2,
    "critical_arc_base_prob": 0.1,
    "critical_arc_anisotropic_covariance": True,
    "critical_arc_critical_direction_sigma_arcsec": 5.0,
    "critical_arc_lm_damping_absolute": 1.0e-6,
    "critical_arc_lm_damping_relative": 1.0e-3,
    "critical_arc_lm_trust_radius_arcsec": 20.0,
    "critical_arc_max_prob": 0.8,
    "critical_arc_singular_softness": 0.02,
    "critical_arc_singular_softness_lower": 0.005,
    "critical_arc_singular_softness_prior_log_sigma": 0.5,
    "critical_arc_singular_softness_prior_median": 0.05,
    "critical_arc_singular_softness_upper": 0.2,
    "critical_arc_singular_threshold": 0.05,
    "critical_arc_singular_threshold_lower": 0.03,
    "critical_arc_singular_threshold_prior_log_sigma": 0.5,
    "critical_arc_singular_threshold_prior_median": 0.15,
    "critical_arc_singular_threshold_upper": 0.4,
    "critical_det_diagnostic_threshold": 0.01,
    "evidence_likelihood_mode": "linearized-forward-beta-image-plane",
    "evidence_source_prior_mean_x_arcsec": 0.0,
    "evidence_source_prior_mean_y_arcsec": 0.0,
    "evidence_source_prior_sigma_arcsec": None,
    "fit_cosmology_flat_wcdm": False,
    "fix_image_sigma_int_arcsec": None,
    "image_plane_mode": "none",
    "image_plane_scatter_prior_log_sigma": 0.5,
    "image_plane_scatter_prior_median_arcsec": 0.3,
    "likelihood_stabilizer_max_gain": 0.0,
    "likelihood_stabilizer_max_residual_arcsec": 0.0,
    "likelihood_stabilizer_residual_loss": "gaussian",
    "likelihood_stabilizer_student_t_nu": 4.0,
    "magnitude_min_reliability": 1.0e-3,
    "magnitude_mu_floor": 1.0e-3,
    "magnitude_sigma_floor": 0.05,
    "mchmc_divergence_threshold": 1000.0,
    "mchmc_l_estimator": "avg",
    "mchmc_l_proposal_factor": float("inf"),
    "mchmc_num_windows": 1,
    "mchmc_random_trajectory_length": True,
    "mchmc_target_accept": 0.9,
    "mchmc_tuning_factor": 1.3,
    "mclmc_desired_energy_var": 5.0e-4,
    "mclmc_lfactor": 0.4,
    "mclmc_num_effective_samples": 150,
    "mclmc_trust_in_estimate": 1.5,
    "microcanonical_diagonal_preconditioning": True,
    "microcanonical_tune_frac1": 0.1,
    "microcanonical_tune_frac2": 0.1,
    "microcanonical_tune_frac3": 0.1,
    "ns_dlogz": 1.0e-4,
    "ns_max_samples": None,
    "ns_num_live_points": None,
    "nuts_init_boundary_frac": 0.02,
    "nuts_init_jitter_frac": 0.02,
    "nuts_init_strategy": "svi",
    "refresh_param_drift_frac": 0.25,
    "sample_critical_arc_singular_softness": False,
    "sample_critical_arc_singular_threshold": False,
    "sample_likelihood_mode": "source",
    "critical_arc_source_position_policy": "sampled",
    "sampler": "numpyro_nuts",
    "skip_critical_det_diagnostic": False,
    "source_plane_outlier_sigma_arcsec": 10.0,
    "stage1_run_dir": None,
    "thin": 1,
    "truth": None,
    "use_magnitude_likelihood": False,
}


def compile_run_plan(config: LensClusterSolverConfig) -> RunPlan:
    config.validate()
    payload = _runtime_payload(config)
    run_name = _resolved_run_name(config)
    payload["run_name"] = run_name
    runtime_args = SolverRuntime(payload)
    output = OutputPlan(
        run_name=run_name,
        output_dir=Path(str(config.paths.output_dir)),
        stage0_minimal_outputs=config.workflow.fit_mode == "sequential",
    )
    runtime = RuntimeSettings(
        seed=config.runtime.seed,
        chains=config.runtime.chains,
        resume=config.runtime.resume,
        plots_only=config.runtime.plots_only,
        skip_validation=config.runtime.skip_validation,
        skip_plots=config.runtime.skip_plots,
        quiet=config.runtime.quiet,
        jax_default_device=config.runtime.jax_default_device,
        smc_device=config.runtime.smc_device,
    )
    diagnostics = DiagnosticsPlan(
        truth_recovery_enabled=bool(config.truth.kappa_true_fits),
        image_recovery_enabled=not config.runtime.skip_validation,
        image_catalog_cutouts_enabled=bool(config.image_catalog.image_dir and config.image_catalog.cutouts),
    )
    return RunPlan(
        config=config,
        runtime_args=runtime_args,
        runtime=runtime,
        output=output,
        diagnostics=diagnostics,
        stages=_stage_plans(config, output),
        run_metadata=config.to_run_dict(),
    )


def _stage_plans(config: LensClusterSolverConfig, output: OutputPlan) -> tuple[StagePlan, ...]:
    schedule = config.schedule
    workflow = config.workflow
    if workflow.fit_mode != "sequential":
        return (
            StagePlan(
                name="stage2",
                fit_mode=workflow.fit_mode,
                fit_method=schedule.fit_method[0],
                sampling_engine=workflow.sampling_engine,
                sample_likelihood_mode="source",
                likelihood_family="source",
                source_position_policy="sampled",
                svi_steps=schedule.svi_steps[0],
                refresh_every=schedule.refresh_every[0],
                warmup=schedule.warmup[0],
                samples=schedule.samples[0],
                sampling_refresh_runs=schedule.sampling_refresh_runs[0],
                max_tree_depth=schedule.max_tree_depth[0],
                output_plan=output,
            ),
        )
    production_index_stage1 = 0
    production_index_stage2 = 1
    stages = [
        StagePlan(
            name="stage0_fast_initializer",
            fit_mode="stage0_fast_initializer",
            fit_method="svi",
            sampling_engine="full_flat",
            sample_likelihood_mode=_stage_likelihood_sample_mode(workflow.stage0_likelihood, field_name="stage0_likelihood"),
            likelihood_family=_stage_likelihood_family(workflow.stage0_likelihood, field_name="stage0_likelihood"),
            source_position_policy=_stage_likelihood_source_position_policy(workflow.stage0_likelihood),
            svi_steps=schedule.svi_steps[0],
            refresh_every=schedule.refresh_every[0],
            warmup=schedule.warmup[production_index_stage1],
            samples=schedule.samples[production_index_stage1],
            sampling_refresh_runs=schedule.sampling_refresh_runs[production_index_stage1],
            max_tree_depth=schedule.max_tree_depth[production_index_stage1],
            output_plan=OutputPlan(output.run_name, output.output_dir, stage0_minimal_outputs=True),
        ),
        StagePlan(
            name="stage1_backprojected_centroid_fit",
            fit_mode="stage1_backprojected_centroid_fit",
            fit_method=schedule.fit_method[production_index_stage1],
            sampling_engine=workflow.stage1_sampling_engine,
            sample_likelihood_mode=_stage1_sample_likelihood_mode(workflow.stage1_likelihood),
            likelihood_family=_stage1_likelihood_family(workflow.stage1_likelihood),
            source_position_policy=_stage1_source_position_policy(workflow.stage1_likelihood),
            svi_steps=schedule.svi_steps[1],
            refresh_every=schedule.refresh_every[1],
            warmup=schedule.warmup[production_index_stage1],
            samples=schedule.samples[production_index_stage1],
            sampling_refresh_runs=schedule.sampling_refresh_runs[production_index_stage1],
            max_tree_depth=schedule.max_tree_depth[production_index_stage1],
            output_plan=output,
        ),
    ]
    if workflow.stage2_forward_mode != "none":
        stages.append(
            StagePlan(
                name="stage2_free_source_forward_fit",
                fit_mode="stage2_free_source_forward_fit",
                fit_method=schedule.fit_method[production_index_stage2],
                sampling_engine=workflow.stage2_sampling_engine,
                sample_likelihood_mode=_stage2_sample_likelihood_mode(workflow.stage2_forward_mode),
                likelihood_family=_stage2_likelihood_family(workflow.stage2_forward_mode),
                source_position_policy=_stage2_source_position_policy(workflow.stage2_forward_mode),
                svi_steps=schedule.svi_steps[2],
                refresh_every=schedule.refresh_every[2],
                warmup=schedule.warmup[production_index_stage2],
                samples=schedule.samples[production_index_stage2],
                sampling_refresh_runs=schedule.sampling_refresh_runs[production_index_stage2],
                max_tree_depth=schedule.max_tree_depth[production_index_stage2],
                output_plan=output,
            )
        )
    return tuple(stages)


def _stage_likelihood_sample_mode(likelihood: str, *, field_name: str) -> str:
    if likelihood in {"source", "local-jacobian"}:
        return likelihood
    if likelihood == "critical-arc":
        return "critical-arc-mixture-image-plane"
    if likelihood == "critical-arc-anisotropic":
        return "critical-arc-anisotropic-image-plane"
    raise ValueError(f"Unsupported {field_name}={likelihood!r}.")


def _stage_likelihood_family(likelihood: str, *, field_name: str) -> str:
    if likelihood in {"source", "local-jacobian"}:
        return likelihood
    if likelihood in {"critical-arc", "critical-arc-anisotropic"}:
        return "critical-arc"
    raise ValueError(f"Unsupported {field_name}={likelihood!r}.")


def _stage_likelihood_source_position_policy(likelihood: str) -> str:
    return "centroid-fixed" if likelihood in {"critical-arc", "critical-arc-anisotropic"} else "sampled"


def _stage1_sample_likelihood_mode(stage1_likelihood: str) -> str:
    return _stage_likelihood_sample_mode(stage1_likelihood, field_name="stage1_likelihood")


def _stage1_likelihood_family(stage1_likelihood: str) -> str:
    return _stage_likelihood_family(stage1_likelihood, field_name="stage1_likelihood")


def _stage1_source_position_policy(stage1_likelihood: str) -> str:
    return _stage_likelihood_source_position_policy(stage1_likelihood)


def _stage2_sample_likelihood_mode(stage2_forward_mode: str) -> str:
    if stage2_forward_mode == "linearized":
        return "linearized-forward-beta-image-plane"
    if stage2_forward_mode == "critical-arc":
        return "critical-arc-mixture-image-plane"
    if stage2_forward_mode == "critical-arc-anisotropic":
        return "critical-arc-anisotropic-image-plane"
    raise ValueError(f"Unsupported stage2_forward_mode={stage2_forward_mode!r}.")


def _stage2_likelihood_family(stage2_forward_mode: str) -> str:
    if stage2_forward_mode == "linearized":
        return "linearized"
    if stage2_forward_mode in {"critical-arc", "critical-arc-anisotropic"}:
        return "critical-arc"
    raise ValueError(f"Unsupported stage2_forward_mode={stage2_forward_mode!r}.")


def _stage2_source_position_policy(stage2_forward_mode: str) -> str:
    del stage2_forward_mode
    return "sampled"


def _resolved_run_name(config: LensClusterSolverConfig) -> str:
    if config.paths.run_name:
        return config.paths.run_name
    return "cluster_solver"


def _runtime_payload(config: LensClusterSolverConfig) -> dict[str, Any]:
    rgb = config.image_catalog.rgb
    payload: dict[str, Any] = {
        **SOLVER_RUNTIME_DEFAULTS,
        "model_config": config.model,
        "output_dir": config.paths.output_dir,
        "run_name": config.paths.run_name,
        "corner_overlay_bayes_dat": config.paths.corner_overlay_bayes_dat,
        "seed": config.runtime.seed,
        "chains": config.runtime.chains,
        "resume": config.runtime.resume,
        "plots_only": config.runtime.plots_only,
        "skip_validation": config.runtime.skip_validation,
        "skip_plots": config.runtime.skip_plots,
        "plot_numpyro_model": config.runtime.plot_numpyro_model,
        "show_plots": config.runtime.show_plots,
        "quick_diagnostics": config.runtime.quick_diagnostics,
        "quiet": config.runtime.quiet,
        "debug_sampler_diagnostics": config.runtime.debug_sampler_diagnostics,
        "numpyro_print_summary": config.runtime.numpyro_print_summary,
        "jax_default_device": config.runtime.jax_default_device,
        "smc_device": config.runtime.smc_device,
        "nuts_chain_method": config.runtime.nuts_chain_method,
        "dense_mass": config.runtime.dense_mass,
        "jax_clear_caches_after_svi_refresh": config.runtime.jax_clear_caches_after_svi_refresh,
        "fit_mode": config.workflow.fit_mode,
        "sampling_engine": config.workflow.sampling_engine,
        "stage1_sampling_engine": config.workflow.stage1_sampling_engine,
        "stage0_likelihood": config.workflow.stage0_likelihood,
        "stage1_likelihood": config.workflow.stage1_likelihood,
        "stage2_forward_mode": config.workflow.stage2_forward_mode,
        "stage2_sampling_engine": config.workflow.stage2_sampling_engine,
        "stage2_fresh_process": config.workflow.stage2_fresh_process,
        "exact_image_diagnostics_stage2": config.workflow.exact_image_diagnostics_stage2,
        "exact_image_diagnostics_stage3": config.workflow.exact_image_diagnostics_stage3,
        "best_value": config.workflow.best_value,
        "image_plane_mode": config.workflow.image_plane_mode,
        "image_plane_newton_steps": config.workflow.image_plane_newton_steps,
        "linearized_beta_prior_sigma_arcsec": config.workflow.linearized_beta_prior_sigma_arcsec,
        "source_position_parameterization": config.workflow.source_position_parameterization,
        "fit_method": config.schedule.fit_method,
        "svi_steps": config.schedule.svi_steps,
        "refresh_every": config.schedule.refresh_every,
        "warmup": config.schedule.warmup,
        "samples": config.schedule.samples,
        "sampling_refresh_runs": config.schedule.sampling_refresh_runs,
        "max_tree_depth": config.schedule.max_tree_depth,
        "target_accept": config.schedule.target_accept,
        "z_bin_efficiency_tol": config.schedule.z_bin_efficiency_tol,
        "initial_step_size": config.schedule.initial_step_size,
        "svi_learning_rate": config.schedule.svi_learning_rate,
        "fov_limit_radius": config.members.fov_limit_radius,
        "fov_limit_x": config.members.fov_limit_x,
        "fov_limit_y": config.members.fov_limit_y,
        "potfile_member_brightest_n": config.members.potfile_member_brightest_n,
        "potfile_member_mag_max": config.members.potfile_member_mag_max,
    }
    payload.update(
        {
            "perturbation_discovery_jacobian_weight": config.perturbation.perturbation_discovery_jacobian_weight,
            "perturbation_discovery_top_k": config.perturbation.perturbation_discovery_top_k,
            "independent_scaling_free_log_sigma_offset_width": (
                config.scaling.independent_scaling_free_log_sigma_offset_width
            ),
            "independent_scaling_free_log_mass_offset_width": (
                config.scaling.independent_scaling_free_log_mass_offset_width
            ),
            "scaling_relation_mode": config.scaling.scaling_relation_mode,
            "potfile_alpha_sigma_prior_mean": config.scaling.potfile_alpha_sigma_prior_mean,
            "potfile_alpha_sigma_prior_std": config.scaling.potfile_alpha_sigma_prior_std,
            "potfile_alpha_sigma_prior_lower": config.scaling.potfile_alpha_sigma_prior_lower,
            "potfile_alpha_sigma_prior_upper": config.scaling.potfile_alpha_sigma_prior_upper,
            "potfile_gamma_ml_prior_mean": config.scaling.potfile_gamma_ml_prior_mean,
            "potfile_gamma_ml_prior_std": config.scaling.potfile_gamma_ml_prior_std,
            "potfile_gamma_ml_prior_lower": config.scaling.potfile_gamma_ml_prior_lower,
            "potfile_gamma_ml_prior_upper": config.scaling.potfile_gamma_ml_prior_upper,
            "scaling_scatter": config.scaling.scaling_scatter,
            "softening_length_kpc": config.scaling.softening_length_kpc,
            "softening_length_prior_log_sigma": config.scaling.softening_length_prior_log_sigma,
            "pos_sigma_arcsec": config.likelihood.pos_sigma_arcsec,
            "source_plane_covariance_floor": config.likelihood.source_plane_covariance_floor,
            "source_plane_covariance_mode": config.likelihood.source_plane_covariance_mode,
            "critical_arc_anisotropic_covariance": config.likelihood.critical_arc_anisotropic_covariance,
            "image_presence_penalty_weight": config.likelihood.image_presence_penalty_weight,
            "image_presence_match_radius_arcsec": config.likelihood.image_presence_match_radius_arcsec,
            "image_presence_temperature_arcsec": config.likelihood.image_presence_temperature_arcsec,
            "image_presence_count_softness": config.likelihood.image_presence_count_softness,
            "image_presence_count_margin": config.likelihood.image_presence_count_margin,
            "image_plane_scatter_prior": config.likelihood.image_plane_scatter_prior,
            "image_plane_scatter_floor_arcsec": config.likelihood.image_plane_scatter_floor_arcsec,
            "image_plane_scatter_upper_arcsec": config.likelihood.image_plane_scatter_upper_arcsec,
            "fix_image_sigma_int_arcsec": config.likelihood.fix_image_sigma_int_arcsec,
            "use_magnitude_likelihood": config.likelihood.use_magnitude_likelihood,
            "magnitude_sigma_floor": config.likelihood.magnitude_sigma_floor,
            "magnitude_mu_floor": config.likelihood.magnitude_mu_floor,
            "magnitude_min_reliability": config.likelihood.magnitude_min_reliability,
            **_positive_lognormal_prior_payload(
                "magnitude_base_scatter",
                config.likelihood.magnitude_base_scatter_prior,
            ),
            **_positive_lognormal_prior_payload(
                "magnitude_arc_scatter",
                config.likelihood.magnitude_arc_scatter_prior,
            ),
            "posterior_image_diagnostic_draws": config.image_diagnostics.posterior_image_diagnostic_draws,
            "posterior_image_diagnostic_mode": config.image_diagnostics.posterior_image_diagnostic_mode,
            "arc_recovery_p_arc_threshold": config.image_diagnostics.arc_recovery_p_arc_threshold,
            "critical_arc_singular_threshold": config.image_diagnostics.critical_arc_singular_threshold,
            "exact_image_min_distance_arcsec": config.image_diagnostics.exact_image_min_distance_arcsec,
            "exact_image_precision_limit": config.image_diagnostics.exact_image_precision_limit,
            "exact_image_num_iter_max": config.image_diagnostics.exact_image_num_iter_max,
            "exact_image_finder": config.image_diagnostics.exact_image_finder,
            "exact_image_displacement_tol_arcsec": config.image_diagnostics.exact_image_displacement_tol_arcsec,
            "exact_image_identification_tol_arcsec": config.image_diagnostics.exact_image_identification_tol_arcsec,
            "exact_image_lm_max_iter": config.image_diagnostics.exact_image_lm_max_iter,
            "exact_image_lm_trust_radius_arcsec": config.image_diagnostics.exact_image_lm_trust_radius_arcsec,
            "exact_image_adaptive_max_levels": config.image_diagnostics.exact_image_adaptive_max_levels,
            "match_tolerance_arcsec": config.image_diagnostics.match_tolerance_arcsec,
            "kappa_true_fits": config.truth.kappa_true_fits,
            "gammax_true_fits": config.truth.gammax_true_fits,
            "gammay_true_fits": config.truth.gammay_true_fits,
            "truth_grid_mode": config.truth.truth_grid_mode,
            "posterior_truth_recovery_draws": config.truth.posterior_truth_recovery_draws,
            "truth_grid_size": config.truth.truth_grid_size,
            "caustic_source_redshift": config.truth.caustic_source_redshift,
            "caustic_plot_grid_scale_arcsec": config.truth.caustic_plot_grid_scale_arcsec,
            "skip_grid_diagnostics": config.truth.skip_grid_diagnostics,
            "image_catalog_family_cutout_image_dir": config.image_catalog.image_dir,
            "image_catalog_family_cutout_image_scale": config.image_catalog.image_scale,
            "image_catalog_family_cutout_bands": config.image_catalog.bands,
            "image_catalog_family_cutout_rgb_q": rgb.q,
            "image_catalog_family_cutout_rgb_stretch": rgb.stretch,
            "image_catalog_family_cutout_rgb_minimum": rgb.minimum,
            "image_catalog_family_cutout_rgb_red_gain": rgb.red_gain,
            "image_catalog_family_cutout_rgb_green_gain": rgb.green_gain,
            "image_catalog_family_cutout_rgb_blue_gain": rgb.blue_gain,
            "image_catalog_family_cutout_mode": config.image_catalog.mode,
            "image_catalog_family_cutout_dpi": config.image_catalog.dpi,
            "image_catalog_family_cutout_max_side_pixels": config.image_catalog.max_side_pixels,
            "image_catalog_family_cutout_critical_lines": config.image_catalog.critical_lines,
            "image_catalog_family_cutouts": config.image_catalog.cutouts,
        }
    )
    return {
        key: _runtime_value(
            SOLVER_RUNTIME_DEFAULTS[key]
            if value is None and SOLVER_RUNTIME_DEFAULTS.get(key) is not None
            else value
        )
        for key, value in payload.items()
    }


def _runtime_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _runtime_value(item) for key, item in value.items()}
    return value
