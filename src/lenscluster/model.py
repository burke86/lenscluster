"""Data model for the single-stage smoothed-local-image-plane solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np

DP_IE_PROFILE = 81          # Lenstool profile codes
SHEAR_PROFILE = 14
NFW_PROFILE = 12
MASS_SHEET_PROFILE = 101    # not a Lenstool profile: uniform external convergence at the lens redshift
DPIE_PROFILE_NAME = "DPIE_NIE_E"        # lenstronomy names (exact-image solver)
SHEAR_PROFILE_NAME = "SHEAR"
NFW_PROFILE_NAME = "NFW_ELLIPSE_CSE"
MASS_SHEET_PROFILE_NAME = "CONVERGENCE"
COMPONENT_FAMILY_LARGE = 0
COMPONENT_FAMILY_SCALING = 1
DEFAULT_SEARCH_PADDING = 8.0
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
SAFE_RADIUS_MARGIN_KPC = 1.0e-3


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
    sample_site_name: str | None = None
    sample_site_index: int | None = None
    centered: bool = False                   # deviation sampled directly with prior N(0, scatter) instead of a unit offset
    scale_site_name: str | None = None       # sample site whose exp(value) is that scatter


@dataclass
class PackedLensSpec:
    """Fixed per-component arrays plus the parameter index that overrides each field."""
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
    alpha_sigma_base: np.ndarray
    beta_radius_base: np.ndarray
    sigma_ref_param_index: np.ndarray
    cut_ref_param_index: np.ndarray
    core_ref_param_index: np.ndarray
    alpha_sigma_param_index: np.ndarray
    beta_radius_param_index: np.ndarray
    cut_radius_excess_above_effective_core: np.ndarray
    # Sampled member deviations (per component) and their population scatters; -1 where absent.
    velocity_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    velocity_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    deviation_centered: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    deviation_mode: str = "sigma"
    # Components whose radius deviation acts on the model; far-field members keep the relation radius when
    # radius_deviation_members is "exact" (their offsets are then prior-only).
    radius_deviation_active: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    radius_deviation_members: str = "exact"
    # Elliptical NFW halos: M200 and scale radius.
    scale_radius_kpc_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    log10_m200_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    scale_radius_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    log10_m200_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    # External convergence: the sheet's kappa for a source at infinite redshift (D_ls / D_s = 1).
    kappa_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    kappa_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))


class PackedLensState(NamedTuple):
    """Per-component kernel parameters.  Normalizations (sigma0, alpha_rs) carry the lensing efficiency of the source
    and have shape (component, image) in the likelihood or (component,) for a single source redshift."""
    profile_type: jnp.ndarray
    sigma0: jnp.ndarray          # dPIE
    Ra: jnp.ndarray
    Rs: jnp.ndarray
    e1: jnp.ndarray              # dPIE and NFW
    e2: jnp.ndarray
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    gamma1: jnp.ndarray          # external shear
    gamma2: jnp.ndarray
    nfw_rs: jnp.ndarray          # NFW: scale radius (arcsec) and deflection at the scale radius (arcsec)
    alpha_rs: jnp.ndarray
    kappa: jnp.ndarray           # mass sheet: convergence seen by each source (efficiency applied)


class PackedLensValidity(NamedTuple):
    is_valid: jnp.ndarray
    reason_flags: jnp.ndarray


@dataclass
class FamilyData:
    family_id: str
    z_source: float
    sigma_inst_arcsec: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray
    reliability: np.ndarray

    @property
    def effective_z_source(self) -> float:
        return self.z_source

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


@dataclass(frozen=True)
class ImageData:
    """All constrained images of every family, flattened, with per-family bookkeeping."""
    family_ids: tuple[str, ...]
    n_families: int
    x_obs: jnp.ndarray
    y_obs: jnp.ndarray
    sigma_inst_per_image: jnp.ndarray
    parity: jnp.ndarray
    z_index_per_image: jnp.ndarray
    family_index_per_image: jnp.ndarray
    source_x_param_index: jnp.ndarray
    source_y_param_index: jnp.ndarray
    image_labels: tuple[str, ...]

    @property
    def n_images(self) -> int:
        return int(self.x_obs.shape[0])


@dataclass(frozen=True)
class MemberModel:
    """Population members: identities, component rows, parameter indices, and classes."""
    member_ids: tuple[str, ...]
    component_indices: np.ndarray
    scatter_param_index: np.ndarray          # per member -> population scatter parameter
    offset_param_index: np.ndarray           # per member -> sampled velocity offset parameter
    exact_positions: np.ndarray              # positions into member arrays evaluated exactly
    far_positions: np.ndarray                # positions evaluated by the far-field series
    source_prior_mean: tuple[float, float]
    source_prior_sigma: float
    far_field_guard_cut_radii: float
    classification: dict[str, Any] = field(default_factory=dict)
    far_field_barrier_width: float = 0.05     # width of the soft guard barrier in ln(R / (guard r_cut))
    far_field_barrier_strength: float = 25.0  # log-likelihood penalty scale
    radius_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    centered: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))


@dataclass
class ModelState:
    run_name: str
    cosmo_config: dict[str, Any]
    z_lens: float
    kpc_per_arcsec: float
    reference: tuple[int, float, float]
    parameter_specs: list[ParameterSpec]
    base_components: list[dict[str, Any]]
    packed_lens_spec: PackedLensSpec
    lens_model_list: list[str]
    family_data: list[FamilyData]
    images: ImageData
    potfiles: list[dict[str, Any]]
    scaling_component_records: list[dict[str, Any]]
    members: MemberModel
    z_source_values: list[float]
    dpie_sigma0_factor_by_z: dict[float, float]
    log_softening_length_param_index: int = -1
    scaling_priors: dict[str, float] = field(default_factory=dict)
    sigma_def_param_index: int = -1              # residual deflection scatter, isotropic in the source plane
    sigma_extra_param_index: int = -1            # extra measurement scatter added in quadrature to sigma_inst; -1 when absent
    sigma_sel_param_index: int = -1              # feature-selection scatter; -1 when the term is disabled
    lambda_sel_param_index: int = -1             # stretching scale of the selection weight R = (I + A A^T / lambda_sel^2)^-1; -1 with the term
    freeze_determinant_gradient: bool = False    # hold grad det A fixed in the backward pass (LikelihoodConfig)
    dropped_member_ids: tuple[str, ...] = ()  # catalog members removed from the lens model (members.min_deflection_arcsec)
    initial_theta: np.ndarray | None = None

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_specs)


# Plotting code written against the previous solver uses this name.
BuildState = ModelState


@dataclass
class PosteriorResults:
    samples: np.ndarray                      # (draws, n_params) latent
    physical: np.ndarray                     # (draws, n_params) reported physical
    log_prob: np.ndarray
    accept_prob: np.ndarray
    diverging: np.ndarray
    num_steps: np.ndarray
    warmup_steps: int
    sample_steps: int
    num_chains: int
    grouped_samples: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    median_fit: np.ndarray | None = None
    maximum_likelihood_fit: np.ndarray | None = None


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
            getattr(spec, "transform_scale", 1.0), dtype=jnp.float64) * array
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


def convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.size == 0:
        return array.copy()
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = latent_array_to_physical(array[..., idx], spec)
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
    physical_params = jnp.where(log_positive_mask, jnp.exp(safe_log_positive_params), params_array)
    safe_log_offset_params = jnp.where(log_offset_positive_mask, params_array, 0.0)
    physical_params = jnp.where(
        log_offset_positive_mask, transform_offset_array + jnp.exp(safe_log_offset_params), physical_params)
    physical_params = jnp.where(
        affine_mask, transform_offset_array + transform_scale_array * params_array, physical_params)
    return physical_params


def clip_value_to_safe_bounds(value: float, spec: ParameterSpec, boundary_frac: float = 0.0) -> float:
    lower = float(spec.lower)
    upper = float(spec.upper)
    clipped = float(np.clip(float(value), lower, upper))
    if spec.prior_kind == "uniform":
        width = float(upper - lower)
        margin = min(max(float(boundary_frac) * width, 0.0), 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = float(np.clip(clipped, lower + margin, upper - margin))
    elif spec.prior_kind == "truncated_normal" and np.isfinite(lower):
        margin = max(float(boundary_frac) * max(abs(lower), 1.0e-12), 0.0)
        if spec.field == "beta_radius":
            margin = max(margin, float(boundary_frac) * float(spec.std))
        if np.isfinite(upper):
            width = float(upper - lower)
            margin = min(margin, 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = max(clipped, lower + margin)
    return float(np.clip(clipped, lower, upper))


def clip_theta_to_support(theta: np.ndarray, parameter_specs: list[ParameterSpec], boundary_frac: float = 0.0) -> np.ndarray:
    clipped = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        clipped[idx] = clip_value_to_safe_bounds(clipped[idx], spec, boundary_frac=boundary_frac)
    return clipped


def default_theta(parameter_specs: list[ParameterSpec], boundary_frac: float = 0.02) -> np.ndarray:
    """Latent vector at prior centres: the configured values for lens parameters, zero offsets."""
    values = np.asarray(
        [0.5 * (spec.lower + spec.upper) if spec.prior_kind == "uniform" else float(spec.mean) for spec in parameter_specs],
        dtype=float)
    return clip_theta_to_support(values, parameter_specs, boundary_frac=boundary_frac)


def member_log_deviations(spec: PackedLensSpec, physical) -> tuple[Any, Any]:
    """Per-component (d ln sigma, d ln radius) implied by the sampled deviations in ``physical``.

    Centered members hold the deviation itself; non-centered members hold a unit offset times the population
    scatter.  In "mass" mode one deviation moves dispersion and radii together along the relation
    (d ln sigma : d ln r = alpha : beta); in "sigma_rcut" mode the radius has its own offset and scatter.
    Works with NumPy or JAX arrays.
    """
    import jax.numpy as jnp
    physical = jnp.asarray(physical)
    centered = jnp.asarray(spec.deviation_centered, dtype=bool)

    def pick(offset_index, scatter_index):
        offset_index, scatter_index = jnp.asarray(offset_index), jnp.asarray(scatter_index)
        present = offset_index >= 0
        offset = jnp.where(present, physical[jnp.maximum(offset_index, 0)], 0.0)
        scatter = jnp.where(scatter_index >= 0, physical[jnp.maximum(scatter_index, 0)], 1.0)
        return jnp.where(present, jnp.where(centered, offset, offset * scatter), 0.0)

    delta_v = pick(spec.velocity_offset_param_index, spec.velocity_scatter_param_index)
    active = np.asarray(spec.radius_deviation_active, dtype=bool)
    radius_mask = jnp.asarray(active, dtype=jnp.float64) if active.size else 1.0

    def pick_radius():
        return radius_mask * pick(spec.radius_offset_param_index, spec.radius_scatter_param_index)

    if spec.deviation_mode == "sigma":
        return delta_v, jnp.zeros_like(delta_v)
    if spec.deviation_mode in ("mass", "mass_rcut"):
        alpha = jnp.where(jnp.asarray(spec.alpha_sigma_param_index) >= 0,
                          physical[jnp.maximum(jnp.asarray(spec.alpha_sigma_param_index), 0)], jnp.asarray(spec.alpha_sigma_base))
        beta = jnp.where(jnp.asarray(spec.beta_radius_param_index) >= 0,
                         physical[jnp.maximum(jnp.asarray(spec.beta_radius_param_index), 0)], jnp.asarray(spec.beta_radius_base))
        total = 2.0 * alpha + beta
        extra = pick_radius() if spec.deviation_mode == "mass_rcut" else 0.0
        return delta_v * alpha / total, delta_v * beta / total + extra
    if spec.deviation_mode == "sigma_rcut":
        return delta_v, pick_radius()
    raise ValueError(f"unknown member deviation mode {spec.deviation_mode!r}")
