from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np

DEFAULT_SEARCH_PADDING = 8.0
DEFAULT_SAMPLER = "numpyro_nuts"


class ParameterTransformSpec(Protocol):
    transform_kind: str
    transform_offset: float
    transform_scale: float


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    sample_name: str
    potential_id: str
    profile_type: int
    field: str
    prior_kind: str
    lower: float
    upper: float
    step: float
    mean: float | None = None
    std: float | None = None
    component_family: str = "large"
    transform_kind: str = "identity"
    physical_lower: float | None = None
    physical_upper: float | None = None
    physical_mean: float | None = None
    physical_std: float | None = None
    transform_offset: float = 0.0
    transform_scale: float = 1.0
    parent_sample_name: str | None = None
    sample_site_name: str | None = None
    sample_site_index: int | None = None


@dataclass
class PackedLensSpec:
    profile_type: np.ndarray
    component_family: np.ndarray
    x_center_base: np.ndarray
    y_center_base: np.ndarray
    e1_base: np.ndarray
    e2_base: np.ndarray
    core_radius_kpc_base: np.ndarray
    cut_radius_kpc_base: np.ndarray
    v_disp_base: np.ndarray
    gamma1_base: np.ndarray
    gamma2_base: np.ndarray
    x_center_param_index: np.ndarray
    y_center_param_index: np.ndarray
    e1_param_index: np.ndarray
    e2_param_index: np.ndarray
    core_radius_param_index: np.ndarray
    cut_radius_param_index: np.ndarray
    v_disp_param_index: np.ndarray
    gamma1_param_index: np.ndarray
    gamma2_param_index: np.ndarray
    luminosity_ratio: np.ndarray
    sigma_ref_base: np.ndarray
    cut_ref_base: np.ndarray
    core_ref_base: np.ndarray
    sigma_ref_param_index: np.ndarray
    cut_ref_param_index: np.ndarray
    core_ref_param_index: np.ndarray
    sigma_log_scatter_param_index: np.ndarray
    mass_log_scatter_param_index: np.ndarray
    cut_radius_excess_above_effective_core: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    alpha_sigma_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    gamma_ml_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    alpha_sigma_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    gamma_ml_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    independent_branch_role: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    independent_magnitude_feature: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    independent_free_log_sigma_delta_unit_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    independent_free_log_mass_delta_unit_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    active_gate_intercept_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    active_gate_mag_slope_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    active_gate_logit_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    active_magnitude_feature: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))


class PackedLensState(NamedTuple):
    profile_type: jnp.ndarray
    sigma0: jnp.ndarray
    Ra: jnp.ndarray
    Rs: jnp.ndarray
    e1: jnp.ndarray
    e2: jnp.ndarray
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray


class PackedLensDetails(NamedTuple):
    is_dpie: jnp.ndarray
    is_shear: jnp.ndarray
    is_scaling: jnp.ndarray
    sigma0: jnp.ndarray
    ra_raw: jnp.ndarray
    rs_raw: jnp.ndarray
    core_radius_kpc: jnp.ndarray
    core_radius_effective_kpc: jnp.ndarray
    softening_length_kpc: jnp.ndarray
    log_softening_length_kpc: jnp.ndarray
    v_disp: jnp.ndarray
    alpha_sigma: jnp.ndarray
    beta_radius: jnp.ndarray
    gamma_ml: jnp.ndarray
    x_center: jnp.ndarray
    y_center: jnp.ndarray
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray
    e1: jnp.ndarray
    e2: jnp.ndarray
    factor_array: jnp.ndarray
    independent_branch_weight: jnp.ndarray


class PackedLensValidity(NamedTuple):
    is_valid: jnp.ndarray
    reason_flags: jnp.ndarray


@dataclass
class SurrogateBinCache:
    effective_z_source: float
    inactive_alpha_x: np.ndarray
    inactive_alpha_y: np.ndarray
    # Derivative rows follow the evaluator's surrogate-parameter basis: scaling parameters,
    # plus sampled cosmology parameters when flat-wCDM cosmology is fitted.
    inactive_alpha_dx_dparams: np.ndarray
    inactive_alpha_dy_dparams: np.ndarray
    inactive_jacobian_delta_a00: np.ndarray | None = None
    inactive_jacobian_delta_a01: np.ndarray | None = None
    inactive_jacobian_delta_a10: np.ndarray | None = None
    inactive_jacobian_delta_a11: np.ndarray | None = None
    inactive_jacobian_delta_da00_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da01_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da10_dparams: np.ndarray | None = None
    inactive_jacobian_delta_da11_dparams: np.ndarray | None = None
    fold_regularized_kappa_eff: np.ndarray | None = None
    fold_regularized_near_indices: np.ndarray | None = None
    fold_regularized_far_indices: np.ndarray | None = None
    catastrophe_kappa: np.ndarray | None = None
    catastrophe_rho: np.ndarray | None = None
    catastrophe_kappa_tangent_x: np.ndarray | None = None
    catastrophe_kappa_tangent_y: np.ndarray | None = None


@dataclass
class ActiveInferenceFlatCache:
    component_indices: np.ndarray
    reference_alpha_x: jnp.ndarray
    reference_alpha_y: jnp.ndarray
    reference_jacobian_delta_a00: jnp.ndarray | None = None
    reference_jacobian_delta_a01: jnp.ndarray | None = None
    reference_jacobian_delta_a10: jnp.ndarray | None = None
    reference_jacobian_delta_a11: jnp.ndarray | None = None


@dataclass
class FamilyData:
    family_id: str
    z_source: float
    effective_z_source: float
    sigma_arcsec: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray
    reliability: np.ndarray
    catalog_mag: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    catalog_mag_err: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    catalog_mag_band_names: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.image_labels)

    @property
    def x_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.x_obs) + np.max(self.x_obs)))

    @property
    def y_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.y_obs) + np.max(self.y_obs)))

    @property
    def search_window(self) -> float:
        span_x = np.ptp(self.x_obs) if self.n_images > 1 else 0.0
        span_y = np.ptp(self.y_obs) if self.n_images > 1 else 0.0
        return float(max(span_x, span_y) + DEFAULT_SEARCH_PADDING)


@dataclass
class BinData:
    effective_z_source: float
    family_ids: list[str]
    family_index_per_image: np.ndarray
    x_obs: np.ndarray
    y_obs: np.ndarray
    sigma_per_image: np.ndarray
    reliability_per_image: np.ndarray
    magnitude_per_image: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    magnitude_error_per_image: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))


@dataclass
class ArcConstraintData:
    arc_ids: list[str]
    z_arc: np.ndarray
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    tangent_angle_rad: np.ndarray
    curvature_arcsec_inv: np.ndarray
    sigma_tangent_angle_rad: np.ndarray
    sigma_curvature_arcsec_inv: np.ndarray
    reliability: np.ndarray

    @property
    def n_arcs(self) -> int:
        return len(self.arc_ids)


@dataclass(frozen=True)
class GeometryCache:
    effective_z_source_values: list[float]
    exact_z_source_values: list[float]
    family_z_source_map: dict[str, float]
    family_effective_z_source_map: dict[str, float]
    dpie_sigma0_factor_by_effective_z: dict[float, float]
    dpie_sigma0_factor_by_exact_z: dict[float, float]
    family_redshift_binning_sec: float = 0.0
    geometry_cache_build_sec: float = 0.0
    flat_wcdm_quadrature_order: int = 64
    lens_quadrature_z: list[float] | None = None
    lens_quadrature_weights: list[float] | None = None
    effective_z_quadrature_z: list[list[float]] | None = None
    effective_z_quadrature_weights: list[list[float]] | None = None
    exact_z_quadrature_z: list[list[float]] | None = None
    exact_z_quadrature_weights: list[list[float]] | None = None


@dataclass
class FamilyValidationCache:
    exact_validation_count: int = 0
    multiplicity_mismatch_count: int = 0
    match_failure_count: int = 0
    source_plane_rms: float | None = None
    exact_image_rms: float | None = None
    last_source_x: float | None = None
    last_source_y: float | None = None


@dataclass
class EvaluationResult:
    loglike: float
    family_predictions: dict[str, dict[str, Any]]


@dataclass
class PosteriorResults:
    samples: np.ndarray
    log_prob: np.ndarray
    accept_prob: np.ndarray
    diverging: np.ndarray
    num_steps: np.ndarray
    warmup_steps: int
    sample_steps: int
    num_chains: int
    init_diagnostics: dict[str, Any] | None = None
    grouped_samples: np.ndarray | None = None
    grouped_log_prob: np.ndarray | None = None
    sampler: str = DEFAULT_SAMPLER
    sample_weights: np.ndarray | None = None
    temperature_schedule: np.ndarray | None = None
    ess_history: np.ndarray | None = None
    move_acceptance_history: np.ndarray | None = None
    ns_diagnostics: dict[str, np.ndarray] | None = None
    map_fit: np.ndarray | None = None
    maximum_likelihood_fit: np.ndarray | None = None
    median_fit: np.ndarray | None = None


@dataclass(frozen=True)
class ChainSeed:
    values: np.ndarray
    source_label: str


@dataclass(frozen=True)
class NUTSInitialization:
    init_params: dict[str, jnp.ndarray]
    chain_seeds: list[ChainSeed]
    diagnostics: dict[str, Any]
    reference_theta: np.ndarray


@dataclass
class Stage1PriorSummary:
    map_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]


@dataclass
class BuildState:
    run_name: str
    par_path: str
    cosmo_config: dict[str, Any]
    z_lens: float
    sigma_arcsec: float
    parsed: dict[str, Any]
    parameter_specs: list[ParameterSpec]
    base_components: list[dict[str, Any]]
    packed_lens_spec: PackedLensSpec
    family_data: list[FamilyData]
    bin_data: list[BinData]
    arc_data: ArcConstraintData | None
    lens_model_list: list[str]
    reference: tuple[int, float, float]
    fit_mode: str
    potfiles: list[dict[str, Any]]
    scaling_component_records: list[dict[str, Any]]
    geometry_cache: GeometryCache | None = None
    svi_init_values: dict[str, float] | None = None
    previous_stage_best_values: dict[str, float] | None = None
    fit_cosmology_flat_wcdm: bool = False
    source_position_parameterization: str = "direct"
    infer_active_scaling: bool = False
    active_scaling_inference_likelihood: str = "population"
    active_scaling_freeze_threshold: float = 0.5
    independent_scaling_model: str = "log_displacement"
    scaling_relation_mode: str = "direct-exponents"
    independent_scaling_free_log_sigma_offset_width: float = 0.10
    independent_scaling_free_log_mass_offset_width: float = 0.20
    potfile_alpha_sigma_prior_mean: float = 0.25
    potfile_alpha_sigma_prior_std: float = 0.04
    potfile_alpha_sigma_prior_lower: float = 0.10
    potfile_alpha_sigma_prior_upper: float = 0.40
    potfile_gamma_ml_prior_mean: float = 0.00
    potfile_gamma_ml_prior_std: float = 0.12
    potfile_gamma_ml_prior_lower: float = -0.40
    potfile_gamma_ml_prior_upper: float = 0.40
    softening_length_kpc: float = 0.0
    softening_length_prior_log_sigma: float = 0.15
    log_softening_length_param_index: int = -1
    frozen_active_scaling_component_indices: np.ndarray | None = None
    active_scaling_frozen_from_previous_stage: bool = False
    active_scaling_frozen_source_run_dir: str | None = None
    active_scaling_frozen_source_path: str | None = None
    perturbation_discovery_stage0: bool = False


def positive_lognormal_parameters(mean: float, std: float, *, floor: float = 1.0e-6) -> tuple[float, float]:
    """Convert physical positive-normal moments into latent log-normal moments."""
    mean = max(float(mean), float(floor))
    std = max(float(std), 1.0e-9)
    variance = std * std
    sigma2 = np.log1p(variance / (mean * mean))
    return float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2))


def physical_to_latent(value: float, spec: ParameterTransformSpec) -> float:
    value = float(value)
    offset = float(spec.transform_offset)
    scale = float(getattr(spec, "transform_scale", 1.0))
    if spec.transform_kind == "log_positive":
        return float(np.log(max(value, 1.0e-12)))
    if spec.transform_kind == "log_offset_positive":
        return float(np.log(max(value - offset, 1.0e-12)))
    if spec.transform_kind == "affine":
        return float((value - offset) / scale)
    return value


def latent_to_physical(value: float, spec: ParameterTransformSpec) -> float:
    value = float(value)
    if spec.transform_kind == "log_positive":
        return float(np.exp(value))
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset + np.exp(value))
    if spec.transform_kind == "affine":
        return float(spec.transform_offset + float(getattr(spec, "transform_scale", 1.0)) * value)
    return value


def latent_array_to_physical(values: np.ndarray, spec: ParameterTransformSpec) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if spec.transform_kind == "log_positive":
        return np.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return float(spec.transform_offset) + np.exp(array)
    if spec.transform_kind == "affine":
        return float(spec.transform_offset) + float(getattr(spec, "transform_scale", 1.0)) * array
    return array


def latent_jax_to_physical(values: jnp.ndarray, spec: ParameterTransformSpec) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=jnp.float64)
    if spec.transform_kind == "log_positive":
        return jnp.exp(array)
    if spec.transform_kind == "log_offset_positive":
        return jnp.asarray(spec.transform_offset, dtype=jnp.float64) + jnp.exp(array)
    if spec.transform_kind == "affine":
        return jnp.asarray(spec.transform_offset, dtype=jnp.float64) + jnp.asarray(
            getattr(spec, "transform_scale", 1.0),
            dtype=jnp.float64,
        ) * array
    return array


def display_lower(spec: ParameterSpec) -> float:
    return float(spec.physical_lower if spec.physical_lower is not None else spec.lower)


def display_upper(spec: ParameterSpec) -> float:
    return float(spec.physical_upper if spec.physical_upper is not None else spec.upper)


def convert_theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[idx] = latent_to_physical(theta_array[idx], spec)
    return converted


def convert_theta_to_latent(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.size == 0:
        return theta_array.copy()
    converted = theta_array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[idx] = physical_to_latent(theta_array[idx], spec)
    return converted


def convert_sample_matrix_to_latent(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = np.asarray(
            [physical_to_latent(value, spec) for value in array[..., idx].reshape(-1)],
            dtype=float,
        ).reshape(array[..., idx].shape)
    return converted


def apply_parameter_transforms_jax(
    params: jnp.ndarray,
    log_positive_mask: jnp.ndarray,
    log_offset_positive_mask: jnp.ndarray,
    transform_offset_array: jnp.ndarray,
    affine_mask: jnp.ndarray | None = None,
    transform_scale_array: jnp.ndarray | None = None,
) -> jnp.ndarray:
    params_array = jnp.asarray(params, dtype=jnp.float64)
    if affine_mask is None:
        affine_mask = jnp.zeros_like(log_positive_mask, dtype=bool)
    if transform_scale_array is None:
        transform_scale_array = jnp.ones_like(transform_offset_array, dtype=jnp.float64)
    safe_log_positive_params = jnp.where(log_positive_mask, params_array, 0.0)
    physical_params = jnp.where(
        log_positive_mask,
        jnp.exp(safe_log_positive_params),
        params_array,
    )
    safe_log_offset_params = jnp.where(log_offset_positive_mask, params_array, 0.0)
    physical_params = jnp.where(
        log_offset_positive_mask,
        transform_offset_array + jnp.exp(safe_log_offset_params),
        physical_params,
    )
    physical_params = jnp.where(
        affine_mask,
        transform_offset_array + transform_scale_array * params_array,
        physical_params,
    )
    return physical_params


def convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = latent_array_to_physical(array[..., idx], spec)
    return converted
