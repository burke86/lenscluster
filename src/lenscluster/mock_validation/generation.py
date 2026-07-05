from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_lenscluster_validation")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.getuid()}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.special import gammainc, gammaincinv
from skimage.measure import find_contours

from .. import jax_lensing
from ..jax_cosmology import (
    dpie_sigma0_from_vel_disp as _jax_dpie_sigma0_from_vel_disp,
    flat_wcdm_config,
    kpc_per_arcsec_from_config,
)

ORIGINAL_DPIE_PROFILE_NAME = "DPIE_NIE"
DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC = 160.0
DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC = 0.2
DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2 = 1.0e-5
DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC = 0.5
# 1" resolves merging image pairs near critical curves down to mu ~ 50; coarser grids
# silently drop exactly the high-magnification pairs the validation is most sensitive to.
DEFAULT_PRIMARY_SEED_GRID_PIXEL_ARCSEC = 1.0
# Fine seed grid overlaid on each subhalo contour's critical curve so image multiplets
# split below the global seed-grid pixel (seed_grid_pixel_arcsec) are still resolved.
MOCK_LOCAL_SEED_GRID_POINTS = 24
MOCK_LOCAL_SEED_GRID_MARGIN = 1.5
MOCK_LOCAL_SEED_GRID_MIN_HALF_ARCSEC = 0.25
DEFAULT_SUBHALO_SEED_GRID_PIXEL_ARCSEC = 1.0
DEFAULT_BCG_POSITION_PRIOR_HALF_WIDTH_ARCSEC = 8.0
SUBHALO_SELECTION_MAX_ATTEMPTS = 5
MockProgressCallback = Callable[[str, dict[str, Any]], None]


def _emit_mock_progress(
    progress_callback: MockProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event, payload)


@dataclass(frozen=True)
class DPIETruth:
    potential_id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_arcsec: float
    cut_radius_arcsec: float
    v_disp: float


@dataclass(frozen=True)
class SourceTruth:
    family_id: str
    beta_x: float
    beta_y: float
    z_source: float


@dataclass(frozen=True)
class CausticContour:
    caustic_index: int
    caustic_class: str
    beta_x: np.ndarray
    beta_y: np.ndarray
    critical_x: np.ndarray
    critical_y: np.ndarray
    caustic_area_arcsec2: float
    critical_area_arcsec2: float


@dataclass(frozen=True)
class SingleBCGMockConfig:
    z_lens: float = 0.396
    reference_ra_deg: float = 64.0381417
    reference_dec_deg: float = -24.0674722
    pos_sigma_arcsec: float = 0.15
    seed: int = 12345
    source_redshift: float = 2.0
    primary_source_redshifts: tuple[float, ...] = (1.5, 2.0, 3.0)
    subhalo_source_redshifts: tuple[float, ...] = (1.5, 2.0, 3.0)
    source_sigma_int_arcsec: float = 0.05
    n_primary_families: int = 20
    n_subhalo_families: int = 0
    min_images_per_family: int = 3
    max_images_per_family: int | None = None
    primary_seed_grid_pixel_arcsec: float = DEFAULT_PRIMARY_SEED_GRID_PIXEL_ARCSEC
    subhalo_seed_grid_pixel_arcsec: float = DEFAULT_SUBHALO_SEED_GRID_PIXEL_ARCSEC
    bcg_position_prior_half_width_arcsec: float = DEFAULT_BCG_POSITION_PRIOR_HALF_WIDTH_ARCSEC
    max_sources_to_try: int = 400
    caustic_compute_window_arcsec: float = DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC
    caustic_grid_scale_arcsec: float = DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC
    caustic_min_area_arcsec2: float = DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2
    caustic_boundary_margin_arcsec: float = DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC
    mock_caustic_grid_chunk_memory_gb: float = 1.0
    mock_image_candidate_batch_size: int = 256
    mock_image_seed_cap: int = 32
    mock_image_search_window_arcsec: float = 80.0
    mock_image_lm_max_iter: int = 100
    mock_image_precision_limit: float = 1.0e-8
    mock_generation_workers: int = 1
    n_subhalos: int = 0
    subhalo_schechter_alpha: float = -0.7
    subhalo_parent_factor: int = 1000
    subhalo_mag_faint_limit: float = 24.0
    subhalo_mass_min: float = 1.0e9
    subhalo_mass_max: float = 1.0e13
    subhalo_mass_ref: float = 1.0e12
    subhalo_spatial_distribution: str = "compact_gamma"
    subhalo_field_radius_arcsec: float = 65.0
    subhalo_spatial_core_radius_arcsec: float = 25.0
    subhalo_spatial_cut_radius_arcsec: float = 180.0
    subhalo_force_core_count: int = 4
    subhalo_force_core_min_arcsec: float = 4.0
    subhalo_force_core_max_arcsec: float = 18.0
    subhalo_mag0: float = 17.0
    subhalo_sigma_ref: float = 245.0
    subhalo_sigma_ref_std: float = 35.0
    subhalo_sigma_scatter_dex: float = 0.0
    subhalo_core_radius_arcsec: float = 0.0001
    subhalo_cut_radius_arcsec: float = 35.0
    subhalo_cut_lower_arcsec: float = 8.0
    subhalo_cut_upper_arcsec: float = 80.0
    subhalo_cut_scatter_dex: float = 0.0
    subhalo_alpha_sigma: float = 0.25
    subhalo_gamma_ml: float = 0.0
    source_magnitude_min: float = 24.0
    source_magnitude_max: float = 27.0
    image_magnitude_sigma_mag: float = 0.10
    image_magnitude_mu_floor: float = 1.0e-3
    halo: DPIETruth = DPIETruth(
        potential_id="halo",
        x_centre=0.0,
        y_centre=0.0,
        ellipticite=0.28,
        angle_pos=35.0,
        core_radius_arcsec=5.0,
        cut_radius_arcsec=220.0,
        v_disp=760.0,
    )
    bcg: DPIETruth = DPIETruth(
        potential_id="bcg",
        x_centre=0.35,
        y_centre=-0.22,
        ellipticite=0.12,
        angle_pos=10.0,
        core_radius_arcsec=0.15,
        cut_radius_arcsec=35.0,
        v_disp=285.0,
    )

    @property
    def n_families(self) -> int:
        return int(self.n_primary_families) + int(self.n_subhalo_families)

    def with_updates(self, **updates: Any) -> "SingleBCGMockConfig":
        return replace(self, **updates)

    def validate(self) -> "SingleBCGMockConfig":
        validate_single_bcg_mock_config(self)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MockClusterPaths:
    root: Path
    par_path: Path
    image_catalog_path: Path
    truth_path: Path
    mock_images_path: Path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_single_bcg_mock_config(config: SingleBCGMockConfig) -> None:
    if isinstance(config.seed, bool) or int(config.seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not math.isfinite(float(config.z_lens)) or float(config.z_lens) <= 0.0:
        raise ValueError("z_lens must be positive and finite.")
    if int(config.n_primary_families) < 0 or int(config.n_subhalo_families) < 0:
        raise ValueError("source-family counts must be nonnegative.")
    if config.n_families <= 0:
        raise ValueError("at least one source family is required.")
    if int(config.min_images_per_family) < 2:
        raise ValueError("min_images_per_family must be at least 2.")
    if config.max_images_per_family is not None and int(config.max_images_per_family) < int(config.min_images_per_family):
        raise ValueError("max_images_per_family must be at least min_images_per_family.")
    if not math.isfinite(float(config.pos_sigma_arcsec)) or float(config.pos_sigma_arcsec) < 0.0:
        raise ValueError("pos_sigma_arcsec must be finite and nonnegative.")
    for name in (
        "source_redshift",
        "source_sigma_int_arcsec",
        "primary_seed_grid_pixel_arcsec",
        "subhalo_seed_grid_pixel_arcsec",
        "bcg_position_prior_half_width_arcsec",
        "caustic_compute_window_arcsec",
        "caustic_grid_scale_arcsec",
        "caustic_min_area_arcsec2",
        "mock_caustic_grid_chunk_memory_gb",
        "mock_image_search_window_arcsec",
        "mock_image_precision_limit",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    for name in ("mock_image_candidate_batch_size", "mock_image_seed_cap", "mock_image_lm_max_iter", "mock_generation_workers"):
        if isinstance(getattr(config, name), bool) or int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if float(config.caustic_boundary_margin_arcsec) < 0.0:
        raise ValueError("caustic_boundary_margin_arcsec must be nonnegative.")
    if int(config.n_subhalos) < 0:
        raise ValueError("n_subhalos must be nonnegative.")
    if int(config.subhalo_parent_factor) <= 0:
        raise ValueError("subhalo_parent_factor must be positive.")
    if not math.isfinite(float(config.subhalo_schechter_alpha)) or float(config.subhalo_schechter_alpha) <= -1.0:
        raise ValueError("subhalo_schechter_alpha must be greater than -1.")
    if float(config.subhalo_mass_min) <= 0.0 or float(config.subhalo_mass_max) <= float(config.subhalo_mass_min):
        raise ValueError("subhalo_mass bounds must be positive and ordered.")
    if float(config.subhalo_mass_ref) <= 0.0:
        raise ValueError("subhalo_mass_ref must be positive.")
    spatial_distribution = str(config.subhalo_spatial_distribution).strip().lower()
    if spatial_distribution not in {"compact_gamma", "dpie"}:
        raise ValueError("subhalo_spatial_distribution must be one of: compact_gamma, dpie.")
    for name in (
        "subhalo_field_radius_arcsec",
        "subhalo_spatial_core_radius_arcsec",
        "subhalo_spatial_cut_radius_arcsec",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    if float(config.subhalo_spatial_cut_radius_arcsec) <= float(config.subhalo_spatial_core_radius_arcsec):
        raise ValueError("subhalo_spatial_cut_radius_arcsec must exceed subhalo_spatial_core_radius_arcsec.")
    if int(config.subhalo_force_core_count) < 0:
        raise ValueError("subhalo_force_core_count must be nonnegative.")
    force_min = float(config.subhalo_force_core_min_arcsec)
    force_max = float(config.subhalo_force_core_max_arcsec)
    if not math.isfinite(force_min) or force_min < 0.0:
        raise ValueError("subhalo_force_core_min_arcsec must be finite and nonnegative.")
    if not math.isfinite(force_max) or force_max <= force_min:
        raise ValueError("subhalo_force_core_max_arcsec must exceed subhalo_force_core_min_arcsec.")
    if force_max > float(config.subhalo_field_radius_arcsec):
        raise ValueError("subhalo_force_core_max_arcsec must not exceed subhalo_field_radius_arcsec.")
    for name in ("subhalo_alpha_sigma", "subhalo_gamma_ml"):
        if not math.isfinite(float(getattr(config, name))):
            raise ValueError(f"{name} must be finite.")
    _subhalo_beta_radius_from_config(config)
    source_mag_min = float(config.source_magnitude_min)
    source_mag_max = float(config.source_magnitude_max)
    if not math.isfinite(source_mag_min) or not math.isfinite(source_mag_max) or source_mag_max <= source_mag_min:
        raise ValueError("source_magnitude_min/source_magnitude_max must be finite and ordered.")
    for name in ("image_magnitude_sigma_mag", "image_magnitude_mu_floor"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")


def _lenstool_ellipticite_to_axis_ratio(ellipticite: float) -> float:
    safe_e = min(max(float(ellipticite), 0.0), 1.0 - 1.0e-9)
    q = math.sqrt((1.0 - safe_e) / (1.0 + safe_e))
    return min(max(q, 1.0e-3), 1.0)


def _axis_ratio_to_lenstool_ellipticite(q: float) -> float:
    safe_q = min(max(float(q), 1.0e-3), 1.0)
    return float((1.0 - safe_q * safe_q) / (1.0 + safe_q * safe_q))


def _config_to_jsonable(config: SingleBCGMockConfig, kpc_per_arcsec: float) -> dict[str, Any]:
    payload = asdict(config)
    payload["n_families"] = int(config.n_families)
    payload["primary_source_redshifts"] = [float(value) for value in config.primary_source_redshifts]
    payload["subhalo_source_redshifts"] = [float(value) for value in config.subhalo_source_redshifts]
    payload["subhalo_sigma_log_scatter"] = _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)
    payload["subhalo_cut_log_scatter"] = _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)
    for component_name in ("halo", "bcg"):
        component = payload[component_name]
        component["core_radius_kpc"] = float(component["core_radius_arcsec"]) * float(kpc_per_arcsec)
        component["cut_radius_kpc"] = float(component["cut_radius_arcsec"]) * float(kpc_per_arcsec)
    return payload


def _dex_scatter_to_ln(scatter_dex: float) -> float:
    return float(np.log(10.0) * max(float(scatter_dex), 0.0))


def _subhalo_beta_radius_from_config(config: SingleBCGMockConfig) -> float:
    alpha_sigma = float(config.subhalo_alpha_sigma)
    gamma_ml = float(config.subhalo_gamma_ml)
    beta_radius = 1.0 + gamma_ml - 2.0 * alpha_sigma
    if not np.isfinite(beta_radius) or beta_radius <= 0.0:
        raise ValueError("Derived subhalo beta_radius = 1 + gamma_ml - 2 * alpha_sigma must be positive and finite.")
    return float(beta_radius)


def _sample_truncated_schechter_luminosities(
    rng: np.random.Generator,
    n_samples: int,
    *,
    luminosity_min: float,
    luminosity_max: float,
    alpha: float,
) -> np.ndarray:
    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.empty(0, dtype=float)
    luminosity_min = float(luminosity_min)
    luminosity_max = float(luminosity_max)
    alpha = float(alpha)
    if (
        not np.isfinite(luminosity_min)
        or not np.isfinite(luminosity_max)
        or luminosity_min <= 0.0
        or luminosity_max <= luminosity_min
    ):
        raise ValueError("Schechter luminosity bounds must satisfy 0 < luminosity_min < luminosity_max.")
    if not np.isfinite(alpha) or alpha <= -1.0:
        raise ValueError("Schechter alpha must be finite and greater than -1 for a log-space peak.")
    shape = alpha + 1.0
    cdf_min = float(gammainc(shape, luminosity_min))
    cdf_max = float(gammainc(shape, luminosity_max))
    if not np.isfinite(cdf_min) or not np.isfinite(cdf_max) or cdf_max <= cdf_min:
        raise ValueError("Schechter luminosity bounds produce an invalid truncated CDF range.")
    u = rng.random(n_samples)
    return np.asarray(gammaincinv(shape, cdf_min + u * (cdf_max - cdf_min)), dtype=float)


def _subhalo_mass_luminosity_exponent(config: SingleBCGMockConfig) -> float:
    alpha_sigma = float(config.subhalo_alpha_sigma)
    beta_radius = _subhalo_beta_radius_from_config(config)
    exponent = 2.0 * alpha_sigma + beta_radius
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("Subhalo mass-luminosity exponent must be positive and finite.")
    return float(exponent)


def _subhalo_luminosity_ratio_from_mass(
    mass_msun: float | np.ndarray,
    config: SingleBCGMockConfig,
) -> np.ndarray:
    mass = np.asarray(mass_msun, dtype=float)
    mass_ref = float(config.subhalo_mass_ref)
    if not np.isfinite(mass_ref) or mass_ref <= 0.0:
        raise ValueError("Subhalo reference mass must be positive and finite.")
    exponent = _subhalo_mass_luminosity_exponent(config)
    return np.power(mass / mass_ref, 1.0 / exponent)


def _subhalo_mass_from_luminosity_ratio(
    luminosity_ratio: float | np.ndarray,
    config: SingleBCGMockConfig,
) -> np.ndarray:
    luminosity = np.asarray(luminosity_ratio, dtype=float)
    mass_ref = float(config.subhalo_mass_ref)
    if not np.isfinite(mass_ref) or mass_ref <= 0.0:
        raise ValueError("Subhalo reference mass must be positive and finite.")
    exponent = _subhalo_mass_luminosity_exponent(config)
    return mass_ref * np.power(luminosity, exponent)


def _subhalo_magnitude_from_luminosity_ratio(
    luminosity_ratio: float | np.ndarray,
    config: SingleBCGMockConfig,
) -> np.ndarray:
    luminosity = np.asarray(luminosity_ratio, dtype=float)
    return float(config.subhalo_mag0) - 2.5 * np.log10(luminosity)


def _subhalo_magnitude_from_mass(
    mass_msun: float | np.ndarray,
    config: SingleBCGMockConfig,
) -> np.ndarray:
    luminosity_ratio = _subhalo_luminosity_ratio_from_mass(mass_msun, config)
    return _subhalo_magnitude_from_luminosity_ratio(luminosity_ratio, config)


def _subhalo_selection_payload(
    config: SingleBCGMockConfig,
    masses: np.ndarray,
    magnitudes: np.ndarray,
    luminosities: np.ndarray,
    parent_rank: np.ndarray,
    passes_mag_cut: np.ndarray,
    selected_indices: list[int],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_member_ids = {
        int(candidate_index): f"member{member_index + 1:03d}"
        for member_index, candidate_index in enumerate(selected_indices)
    }
    candidates: list[dict[str, Any]] = []
    for idx in range(int(masses.size)):
        selected = idx in selected_member_ids
        row = {
            "subhalo_candidate_index": int(idx),
            "subhalo_mass_msun": float(masses[idx]),
            "catalog_mag": float(magnitudes[idx]),
            "luminosity_ratio": float(luminosities[idx]),
            "subhalo_parent_rank": int(parent_rank[idx]),
            "passes_mag_cut": bool(passes_mag_cut[idx]),
            "selected": bool(selected),
        }
        if selected:
            row["selected_member_id"] = selected_member_ids[idx]
        candidates.append(row)
    payload = {
        "schechter_alpha": float(config.subhalo_schechter_alpha),
        "luminosity_star": 1.0,
        "mass_min": float(config.subhalo_mass_min),
        "mass_max": float(config.subhalo_mass_max),
        "mass_ref": float(config.subhalo_mass_ref),
        "mag_faint_limit": float(config.subhalo_mag_faint_limit),
        "parent_count": int(masses.size),
        "selected_count": int(len(selected_indices)),
        "mass_luminosity_exponent": _subhalo_mass_luminosity_exponent(config),
        "candidates": candidates,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def _subhalo_schechter_luminosity_bounds(config: SingleBCGMockConfig) -> tuple[float, float]:
    mass_min = float(config.subhalo_mass_min)
    mass_max = float(config.subhalo_mass_max)
    if not np.isfinite(mass_min) or not np.isfinite(mass_max) or mass_min <= 0.0 or mass_max <= mass_min:
        raise ValueError("Schechter mass bounds must satisfy 0 < mass_min < mass_max.")
    mass_ref = float(config.subhalo_mass_ref)
    if not np.isfinite(mass_ref) or mass_ref <= 0.0:
        raise ValueError("Subhalo reference mass must be positive and finite.")
    exponent = _subhalo_mass_luminosity_exponent(config)
    luminosity_min = float((mass_min / mass_ref) ** (1.0 / exponent))
    luminosity_max = float((mass_max / mass_ref) ** (1.0 / exponent))
    if not np.isfinite(luminosity_min) or not np.isfinite(luminosity_max) or luminosity_max <= luminosity_min:
        raise ValueError("Schechter luminosity bounds derived from mass bounds are invalid.")
    return luminosity_min, luminosity_max


def _generate_schechter_subhalo_candidates(
    config: SingleBCGMockConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    n_subhalos = int(config.n_subhalos)
    if n_subhalos <= 0:
        return [], None
    parent_factor = int(config.subhalo_parent_factor)
    if parent_factor <= 0:
        raise ValueError("subhalo_parent_factor must be positive.")
    alpha = float(config.subhalo_schechter_alpha)
    if not np.isfinite(alpha) or alpha <= -1.0:
        raise ValueError("subhalo_schechter_alpha must be finite and greater than -1.")
    base_parent_count = max(n_subhalos, parent_factor * n_subhalos)
    faint_limit = float(config.subhalo_mag_faint_limit)
    if not np.isfinite(faint_limit):
        raise ValueError("subhalo_mag_faint_limit must be finite.")
    luminosity_min, luminosity_max = _subhalo_schechter_luminosity_bounds(config)
    luminosity_peak = alpha + 1.0
    mass_peak = float(_subhalo_mass_from_luminosity_ratio(luminosity_peak, config))
    last_parent_count = base_parent_count
    for attempt in range(SUBHALO_SELECTION_MAX_ATTEMPTS):
        parent_count = base_parent_count * (2**attempt)
        last_parent_count = parent_count
        luminosities = _sample_truncated_schechter_luminosities(
            rng,
            parent_count,
            luminosity_min=luminosity_min,
            luminosity_max=luminosity_max,
            alpha=alpha,
        )
        masses = np.asarray(_subhalo_mass_from_luminosity_ratio(luminosities, config), dtype=float)
        magnitudes = np.asarray(_subhalo_magnitude_from_luminosity_ratio(luminosities, config), dtype=float)
        finite = np.isfinite(masses) & np.isfinite(magnitudes) & np.isfinite(luminosities) & (luminosities > 0.0)
        observable = finite & (magnitudes <= faint_limit)
        brightness_order = np.argsort(magnitudes, kind="stable")
        parent_rank = np.empty(parent_count, dtype=int)
        parent_rank[brightness_order] = np.arange(1, parent_count + 1)
        observable_indices = np.flatnonzero(observable)
        selected_indices = (
            [int(idx) for idx in rng.choice(observable_indices, size=n_subhalos, replace=False)]
            if observable_indices.size >= n_subhalos
            else []
        )
        if len(selected_indices) == n_subhalos:
            selection_payload = _subhalo_selection_payload(
                config,
                masses,
                magnitudes,
                luminosities,
                parent_rank,
                observable,
                selected_indices,
                extra_metadata={
                    "luminosity_min": float(luminosity_min),
                    "luminosity_max": float(luminosity_max),
                    "luminosity_peak": float(luminosity_peak),
                    "mass_peak": float(mass_peak),
                    "schechter_shape": float(alpha + 1.0),
                },
            )
            return [
                {
                    "subhalo_candidate_index": int(idx),
                    "catalog_mag": float(magnitudes[idx]),
                    "luminosity_ratio": float(luminosities[idx]),
                    "subhalo_mass_msun": float(masses[idx]),
                    "subhalo_parent_rank": int(parent_rank[idx]),
                    "selected_by_mag_cut": True,
                }
                for idx in selected_indices
            ], selection_payload
    raise RuntimeError(
        "Failed to select "
        f"{n_subhalos} Schechter subhalos brighter than mag {faint_limit:.6g} "
        f"after drawing up to {last_parent_count} parent candidates. "
        "Increase --subhalo-parent-factor, relax --subhalo-mag-faint-limit, "
        "or adjust the Schechter mass bounds."
    )


def _generate_subhalo_candidates(
    config: SingleBCGMockConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    return _generate_schechter_subhalo_candidates(config, rng)


def _scaled_subhalo_params(row: dict[str, Any], config: SingleBCGMockConfig) -> DPIETruth:
    luminosity_ratio = float(row["luminosity_ratio"])
    sigma_log_offset = float(row.get("sigma_log_offset", 0.0))
    cut_log_offset = float(row.get("cut_log_offset", 0.0))
    beta_radius = _subhalo_beta_radius_from_config(config)
    return DPIETruth(
        potential_id=str(row["id"]),
        x_centre=float(row["x_arcsec"]),
        y_centre=float(row["y_arcsec"]),
        ellipticite=float(row["ellipticite"]),
        angle_pos=float(row["angle_pos"]),
        core_radius_arcsec=float(config.subhalo_core_radius_arcsec * luminosity_ratio ** beta_radius),
        cut_radius_arcsec=float(
            config.subhalo_cut_radius_arcsec
            * luminosity_ratio ** beta_radius
            * np.exp(cut_log_offset)
        ),
        v_disp=float(config.subhalo_sigma_ref * luminosity_ratio ** config.subhalo_alpha_sigma * np.exp(sigma_log_offset)),
    )


def _subhalo_spatial_metadata(config: SingleBCGMockConfig) -> dict[str, Any]:
    return {
        "spatial_distribution": str(config.subhalo_spatial_distribution).strip().lower(),
        "field_radius_arcsec": float(config.subhalo_field_radius_arcsec),
        "spatial_core_radius_arcsec": float(config.subhalo_spatial_core_radius_arcsec),
        "spatial_cut_radius_arcsec": float(config.subhalo_spatial_cut_radius_arcsec),
        "force_core_count": int(config.subhalo_force_core_count),
        "force_core_min_arcsec": float(config.subhalo_force_core_min_arcsec),
        "force_core_max_arcsec": float(config.subhalo_force_core_max_arcsec),
    }


def _sample_dpie_subhalo_radii(config: SingleBCGMockConfig, rng: np.random.Generator, count: int) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=float)
    r_max = float(config.subhalo_field_radius_arcsec)
    r_core = float(config.subhalo_spatial_core_radius_arcsec)
    r_cut = float(config.subhalo_spatial_cut_radius_arcsec)
    grid_size = 8192
    radius_grid = np.linspace(0.0, r_max, grid_size, dtype=float)
    surface_density = 1.0 / np.sqrt(radius_grid * radius_grid + r_core * r_core) - 1.0 / np.sqrt(
        radius_grid * radius_grid + r_cut * r_cut
    )
    pdf = np.maximum(radius_grid * surface_density, 0.0)
    increments = 0.5 * (pdf[1:] + pdf[:-1]) * np.diff(radius_grid)
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("dPIE subhalo spatial distribution produced a nonpositive normalization.")
    cdf /= total
    return np.asarray(np.interp(rng.random(count), cdf, radius_grid), dtype=float)


def _sample_subhalo_radii(config: SingleBCGMockConfig, rng: np.random.Generator, count: int) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=float)
    distribution = str(config.subhalo_spatial_distribution).strip().lower()
    if distribution == "dpie":
        return _sample_dpie_subhalo_radii(config, rng, count)
    if distribution != "compact_gamma":
        raise ValueError(f"Unsupported subhalo_spatial_distribution={config.subhalo_spatial_distribution!r}.")
    radii = np.empty(count, dtype=float)
    forced_count = min(int(config.subhalo_force_core_count), count)
    if forced_count > 0:
        radii[:forced_count] = rng.uniform(
            float(config.subhalo_force_core_min_arcsec),
            float(config.subhalo_force_core_max_arcsec),
            size=forced_count,
        )
    if forced_count < count:
        radii[forced_count:] = np.minimum(
            float(config.subhalo_field_radius_arcsec),
            rng.gamma(shape=2.0, scale=14.0, size=count - forced_count) + 3.0,
        )
    return radii


def _generate_subhalo_catalog_payload(
    config: SingleBCGMockConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if config.n_subhalos <= 0:
        return [], None
    candidates, subhalo_selection = _generate_subhalo_candidates(config, rng)
    if subhalo_selection is not None:
        subhalo_selection.update(_subhalo_spatial_metadata(config))
    rows: list[dict[str, Any]] = []
    radii = _sample_subhalo_radii(config, rng, len(candidates))
    for idx, candidate in enumerate(candidates):
        radius = float(radii[idx])
        theta = rng.uniform(0.0, 2.0 * np.pi)
        x_arcsec = float(radius * np.cos(theta))
        y_arcsec = float(radius * np.sin(theta))
        q = float(rng.uniform(0.65, 1.0))
        ellipticite = _axis_ratio_to_lenstool_ellipticite(q)
        angle_pos = float(rng.uniform(-90.0, 90.0))
        mag = float(candidate["catalog_mag"])
        luminosity_ratio = float(candidate["luminosity_ratio"])
        sigma_log_offset = float(rng.normal(0.0, _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)))
        cut_log_offset = float(rng.normal(0.0, _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)))
        row = {
            "id": f"member{idx + 1:03d}",
            "x_arcsec": x_arcsec,
            "y_arcsec": y_arcsec,
            "catalog_a": 0.3734,
            "catalog_b": 0.3734 * q,
            "catalog_theta": angle_pos,
            "catalog_mag": mag,
            "catalog_lum": luminosity_ratio,
            "ellipticite": ellipticite,
            "angle_pos": angle_pos,
            "luminosity_ratio": luminosity_ratio,
            "sigma_log_offset": sigma_log_offset,
            "cut_log_offset": cut_log_offset,
        }
        for key in ("subhalo_candidate_index", "subhalo_mass_msun", "subhalo_parent_rank", "selected_by_mag_cut"):
            if key in candidate:
                row[key] = candidate[key]
        rows.append(row)
    return rows, subhalo_selection


def _generate_subhalo_catalog(config: SingleBCGMockConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows, _subhalo_selection = _generate_subhalo_catalog_payload(config, rng)
    return rows


def _component_kwargs(
    component: DPIETruth,
    config: SingleBCGMockConfig,
    z_source: float,
    cosmo_config: dict[str, Any],
) -> dict[str, float]:
    sigma0 = _dpie_sigma0_from_vel_disp_local(
        component.v_disp,
        component.core_radius_arcsec,
        component.cut_radius_arcsec,
        config.z_lens,
        z_source,
        cosmo_config,
    )
    q = _lenstool_ellipticite_to_axis_ratio(component.ellipticite)
    phi = math.radians(component.angle_pos)
    fac = (1.0 - q) / (1.0 + q)
    return {
        "sigma0": float(sigma0),
        "Ra": float(component.core_radius_arcsec),
        "Rs": float(component.cut_radius_arcsec),
        "e1": float(fac * math.cos(2.0 * phi)),
        "e2": float(fac * math.sin(2.0 * phi)),
        "center_x": float(component.x_centre),
        "center_y": float(component.y_centre),
    }


def _dpie_sigma0_from_vel_disp_local(
    vel_disp: float,
    ra_arcsec: float,
    rs_arcsec: float,
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any],
) -> float:
    return _jax_dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        rs_arcsec,
        z_lens,
        z_source,
        cosmo_config,
    )


def _truth_parameter_values(config: SingleBCGMockConfig, kpc_per_arcsec: float) -> dict[str, float]:
    truth: dict[str, float] = {}
    for component in (config.halo, config.bcg):
        prefix = component.potential_id
        truth[f"{prefix}.x_centre"] = float(component.x_centre)
        truth[f"{prefix}.y_centre"] = float(component.y_centre)
        truth[f"{prefix}.ellipticite"] = float(component.ellipticite)
        truth[f"{prefix}.angle_pos"] = float(component.angle_pos)
        truth[f"{prefix}.core_radius_kpc"] = float(component.core_radius_arcsec) * float(kpc_per_arcsec)
        truth[f"{prefix}.cut_radius_kpc"] = float(component.cut_radius_arcsec) * float(kpc_per_arcsec)
        truth[f"{prefix}.v_disp"] = float(component.v_disp)
    # No truth entry for source.sigma_int: the mock never injects intrinsic source-plane
    # scatter (astrometric noise is injected via pos_sigma_arcsec and is already accounted
    # for by the likelihood's catalog sigma), so the sampled image.sigma_int has no finite
    # truth to score against. Recording config.source_sigma_int_arcsec here previously
    # produced a spurious fractional bias of -1 in the recovery tables.
    if config.n_subhalos > 0:
        truth["potfile.sigma"] = float(config.subhalo_sigma_ref)
        truth["potfile.cutkpc"] = float(config.subhalo_cut_radius_arcsec) * float(kpc_per_arcsec)
        truth["potfile.alpha_sigma"] = float(config.subhalo_alpha_sigma)
        truth["potfile.gamma_ml"] = float(config.subhalo_gamma_ml)
        truth["potfile.beta_radius"] = _subhalo_beta_radius_from_config(config)
        if config.subhalo_sigma_scatter_dex > 0.0:
            truth["potfile.sigma_log_scatter"] = _dex_scatter_to_ln(config.subhalo_sigma_scatter_dex)
        if config.subhalo_cut_scatter_dex > 0.0:
            truth["potfile.mass_log_scatter"] = _dex_scatter_to_ln(config.subhalo_cut_scatter_dex)
    return truth


def _write_member_catalog(path: Path, subhalos: list[dict[str, Any]]) -> None:
    path.write_text(
        "#REFERENCE 3\n"
        + "\n".join(
            (
                f"{row['id']:>10s} {row['x_arcsec']: .8f} {row['y_arcsec']: .8f} "
                f"{row['catalog_a']:.6f} {row['catalog_b']:.6f} {row['catalog_theta']:.4f} "
                f"{row['catalog_mag']:.6f} {row['catalog_lum']:.8e} nan"
            )
            for row in subhalos
        )
        + ("\n" if subhalos else ""),
        encoding="utf-8",
    )


def _single_bcg_champ_dmax(images: pd.DataFrame, subhalos: list[dict[str, Any]], padding_arcsec: float = 10.0) -> int:
    radii: list[float] = []
    if not images.empty and {"x_obs_arcsec", "y_obs_arcsec"}.issubset(images.columns):
        image_positions = images.loc[:, ["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(dtype=float)
        finite_images = image_positions[np.isfinite(image_positions).all(axis=1)]
        radii.extend(float(radius) for radius in np.hypot(finite_images[:, 0], finite_images[:, 1]))

    for subhalo in subhalos:
        x_arcsec = float(subhalo["x_arcsec"])
        y_arcsec = float(subhalo["y_arcsec"])
        if np.isfinite(x_arcsec) and np.isfinite(y_arcsec):
            radii.append(float(math.hypot(x_arcsec, y_arcsec)))

    max_radius_arcsec = max(radii, default=0.0)
    return int(math.ceil(max_radius_arcsec + float(padding_arcsec)))


def _write_single_bcg_par(
    path: Path,
    config: SingleBCGMockConfig,
    images: pd.DataFrame,
    subhalos: list[dict[str, Any]] | None = None,
) -> None:
    def potential_block(component: DPIETruth, position_prior_half_width_arcsec: float) -> str:
        half_width = float(position_prior_half_width_arcsec)
        return f"""
potentiel {component.potential_id}
    profil      81
    x_centre    {component.x_centre:.8f}
    y_centre    {component.y_centre:.8f}
    ellipticite {component.ellipticite:.8f}
    angle_pos   {component.angle_pos:.8f}
    core_radius {component.core_radius_arcsec:.8f}
    cut_radius  {component.cut_radius_arcsec:.8f}
    v_disp      {component.v_disp:.8f}
    z_lens      {config.z_lens:.8f}
    end

limit {component.potential_id}
    x_centre    1 {component.x_centre - half_width:.8f} {component.x_centre + half_width:.8f} 0.05
    y_centre    1 {component.y_centre - half_width:.8f} {component.y_centre + half_width:.8f} 0.05
    ellipticite 1 0.0 0.75 0.02
    angle_pos   1 -90.0 90.0 0.5
    core_radius 1 {max(0.001, component.core_radius_arcsec * 0.2):.8f} {component.core_radius_arcsec * 3.0:.8f} 0.02
    cut_radius  1 {component.cut_radius_arcsec * 0.4:.8f} {component.cut_radius_arcsec * 2.0:.8f} 0.5
    v_disp      1 {component.v_disp * 0.55:.8f} {component.v_disp * 1.45:.8f} 1.0
    end
"""

    subhalos = subhalos or []
    large_scale_lens_count = 2
    nlens = large_scale_lens_count + len(subhalos)
    nlens_opt = large_scale_lens_count
    champ_dmax = _single_bcg_champ_dmax(images, subhalos)
    potfile_block = ""
    if subhalos:
        potfile_block = f"""
potfile
    filein 3 members.cat
    type 81
    mag0 {config.subhalo_mag0:.8f}
    core {config.subhalo_core_radius_arcsec:.8f}
    z_lens {config.z_lens:.8f}
    sigma 3 {config.subhalo_sigma_ref:.8f} {config.subhalo_sigma_ref_std:.8f}
    cut 1 {config.subhalo_cut_lower_arcsec:.8f} {config.subhalo_cut_upper_arcsec:.8f}
    end
"""

    content = f"""runmode
    reference 3 {config.reference_ra_deg:.8f} {config.reference_dec_deg:.8f}
    inverse   3 0.5 100
    end

image
    multfile    1 {path.with_name("obs_arcs.cat").name}
    sigposArcsec {config.pos_sigma_arcsec:.8f}
    forme       0
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

grille
    nlens {nlens}
    nlens_opt {nlens_opt}
    nombre 256
    end
{potential_block(config.halo, DEFAULT_BCG_POSITION_PRIOR_HALF_WIDTH_ARCSEC)}
{potential_block(config.bcg, config.bcg_position_prior_half_width_arcsec)}
{potfile_block}
champ
    dmax {champ_dmax}
    end

finish
"""
    path.write_text(content, encoding="utf-8")


def _deduplicate_images(x: np.ndarray, y: np.ndarray, tolerance: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    for idx, (x_i, y_i) in enumerate(zip(x, y)):
        if not np.isfinite(x_i) or not np.isfinite(y_i):
            continue
        if all(math.hypot(float(x_i - x[j]), float(y_i - y[j])) > tolerance for j in keep):
            keep.append(idx)
    return np.asarray(x[keep], dtype=float), np.asarray(y[keep], dtype=float)


def _lowercase_image_suffix(index: int) -> str:
    value = int(index)
    if value < 1:
        raise ValueError("Image suffix index must be positive.")
    value -= 1
    letters: list[str] = []
    while True:
        letters.append(chr(ord("a") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def _closed_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < 3:
        return 0.0
    if math.hypot(float(x_arr[0] - x_arr[-1]), float(y_arr[0] - y_arr[-1])) > 0.0:
        x_arr = np.concatenate([x_arr, x_arr[:1]])
        y_arr = np.concatenate([y_arr, y_arr[:1]])
    return float(0.5 * abs(np.sum(x_arr[:-1] * y_arr[1:] - x_arr[1:] * y_arr[:-1])))


def _curve_hits_boundary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    compute_window_arcsec: float,
    boundary_margin_arcsec: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> bool:
    half_window = 0.5 * float(compute_window_arcsec)
    margin = max(float(boundary_margin_arcsec), 0.0)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return bool(
        np.any(x_arr <= center_x - half_window + margin)
        or np.any(x_arr >= center_x + half_window - margin)
        or np.any(y_arr <= center_y - half_window + margin)
        or np.any(y_arr >= center_y + half_window - margin)
    )


def _primary_contour_position(
    contours: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]
) -> int:
    if not contours:
        raise ValueError("Cannot choose a primary contour from an empty contour list.")
    return int(np.argmax([item[4] for item in contours]))


def _compute_tangential_caustic_contours(
    lens_state: jax_lensing.StaticLensState,
    config: SingleBCGMockConfig,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> list[CausticContour]:
    compute_window = float(config.caustic_compute_window_arcsec)
    grid_scale = float(config.caustic_grid_scale_arcsec)
    if compute_window <= 0.0 or grid_scale <= 0.0:
        raise ValueError("Caustic compute window and grid scale must be positive.")
    num_pix = max(16, int(math.ceil(compute_window / grid_scale)) + 1)
    if num_pix % 2 == 0:
        num_pix += 1
    x_axis = np.linspace(center_x - 0.5 * compute_window, center_x + 0.5 * compute_window, num_pix)
    y_axis = np.linspace(center_y - 0.5 * compute_window, center_y + 0.5 * compute_window, num_pix)
    xx, yy = np.meshgrid(x_axis, y_axis)
    chunk_size = jax_lensing.estimated_alpha_hessian_chunk_size(float(config.mock_caustic_grid_chunk_memory_gb))
    _alpha_x, _alpha_y, f_xx, f_xy, f_yx, f_yy = jax_lensing.alpha_and_hessian_tiled(
        jnp.asarray(xx.ravel(), dtype=jnp.float64),
        jnp.asarray(yy.ravel(), dtype=jnp.float64),
        lens_state,
        chunk_size=chunk_size,
    )
    f_xx = np.asarray(f_xx, dtype=float).reshape(xx.shape)
    f_yy = np.asarray(f_yy, dtype=float).reshape(xx.shape)
    f_xy = np.asarray(f_xy, dtype=float).reshape(xx.shape)
    f_yx = np.asarray(f_yx, dtype=float).reshape(xx.shape)
    kappa = 0.5 * (f_xx + f_yy)
    gamma1 = 0.5 * (f_xx - f_yy)
    gamma2 = 0.5 * (f_xy + f_yx)
    lambda_tan = 1.0 - kappa - np.hypot(gamma1, gamma2)
    critical_contours: list[tuple[np.ndarray, np.ndarray, float]] = []
    for vertices in find_contours(lambda_tan, 0.0):
        if vertices.shape[0] < 4:
            continue
        row = vertices[:, 0]
        col = vertices[:, 1]
        crit_x = center_x - 0.5 * compute_window + col * (compute_window / (num_pix - 1))
        crit_y = center_y - 0.5 * compute_window + row * (compute_window / (num_pix - 1))
        if not np.all(np.isfinite(crit_x)) or not np.all(np.isfinite(crit_y)):
            continue
        if _curve_hits_boundary(
            crit_x,
            crit_y,
            compute_window_arcsec=compute_window,
            boundary_margin_arcsec=float(config.caustic_boundary_margin_arcsec),
            center_x=center_x,
            center_y=center_y,
        ):
            continue
        close_gap = math.hypot(float(crit_x[0] - crit_x[-1]), float(crit_y[0] - crit_y[-1]))
        if close_gap > 1.5 * grid_scale:
            continue
        crit_area = _closed_polygon_area(crit_x, crit_y)
        critical_contours.append((crit_x, crit_y, crit_area))
    contours: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    if critical_contours:
        vertex_counts = [int(crit_x.size) for crit_x, _crit_y, _crit_area in critical_contours]
        all_crit_x = np.concatenate([crit_x for crit_x, _crit_y, _crit_area in critical_contours])
        all_crit_y = np.concatenate([crit_y for _crit_x, crit_y, _crit_area in critical_contours])
        all_beta_x, all_beta_y = jax_lensing.ray_shooting_tiled(
            jnp.asarray(all_crit_x, dtype=jnp.float64),
            jnp.asarray(all_crit_y, dtype=jnp.float64),
            lens_state,
            chunk_size=chunk_size,
        )
        all_beta_x_np = np.asarray(all_beta_x, dtype=float)
        all_beta_y_np = np.asarray(all_beta_y, dtype=float)
    else:
        vertex_counts = []
        all_beta_x_np = np.asarray([], dtype=float)
        all_beta_y_np = np.asarray([], dtype=float)
    offset = 0
    for (crit_x, crit_y, crit_area), vertex_count in zip(critical_contours, vertex_counts):
        beta_x = all_beta_x_np[offset : offset + vertex_count]
        beta_y = all_beta_y_np[offset : offset + vertex_count]
        offset += vertex_count
        if not np.all(np.isfinite(beta_x)) or not np.all(np.isfinite(beta_y)):
            continue
        caustic_area = _closed_polygon_area(beta_x, beta_y)
        if caustic_area < float(config.caustic_min_area_arcsec2):
            continue
        contours.append((crit_x, crit_y, beta_x, beta_y, crit_area, caustic_area))
    if not contours:
        return []
    primary_position = _primary_contour_position(contours)
    results: list[CausticContour] = []
    for index, (crit_x, crit_y, beta_x, beta_y, crit_area, caustic_area) in enumerate(contours):
        results.append(
            CausticContour(
                caustic_index=index,
                caustic_class="primary" if index == primary_position else "subhalo",
                beta_x=beta_x,
                beta_y=beta_y,
                critical_x=crit_x,
                critical_y=crit_y,
                caustic_area_arcsec2=float(caustic_area),
                critical_area_arcsec2=float(crit_area),
            )
        )
    return results


def _sample_point_in_caustic(contour: CausticContour, rng: np.random.Generator, *, max_tries: int = 1000) -> tuple[float, float]:
    vertices = np.column_stack([np.asarray(contour.beta_x, dtype=float), np.asarray(contour.beta_y, dtype=float)])
    path = MplPath(vertices)
    x_min, y_min = np.nanmin(vertices, axis=0)
    x_max, y_max = np.nanmax(vertices, axis=0)
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"Cannot sample from degenerate caustic contour {contour.caustic_index}.")
    for _ in range(max(1, int(max_tries))):
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        if path.contains_point((x, y), radius=0.0):
            return x, y
    raise RuntimeError(f"Failed to sample a point inside caustic contour {contour.caustic_index}.")


def _sample_point_near_caustic(contour: CausticContour, rng: np.random.Generator) -> tuple[float, float]:
    vertices = np.column_stack([np.asarray(contour.beta_x, dtype=float), np.asarray(contour.beta_y, dtype=float)])
    x_min, y_min = np.nanmin(vertices, axis=0)
    x_max, y_max = np.nanmax(vertices, axis=0)
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        raise RuntimeError(f"Cannot sample near degenerate caustic contour {contour.caustic_index}.")
    span_x = max(float(x_max - x_min), 0.1)
    span_y = max(float(y_max - y_min), 0.1)
    return (
        float(rng.uniform(x_min - 0.5 * span_x, x_max + 0.5 * span_x)),
        float(rng.uniform(y_min - 0.5 * span_y, y_max + 0.5 * span_y)),
    )


def _sample_caustic_source_candidate(
    contours: list[CausticContour],
    caustic_class: str,
    rng: np.random.Generator,
    *,
    inside_caustic: bool = True,
) -> tuple[float, float, CausticContour]:
    class_contours = [contour for contour in contours if contour.caustic_class == caustic_class]
    if not class_contours:
        raise RuntimeError(f"No valid {caustic_class} caustics were found for source placement.")
    weights = np.asarray([max(contour.caustic_area_arcsec2, 0.0) for contour in class_contours], dtype=float)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(class_contours), dtype=float)
    probabilities = weights / np.sum(weights)
    contour = class_contours[int(rng.choice(len(class_contours), p=probabilities))]
    if inside_caustic:
        beta_x, beta_y = _sample_point_in_caustic(contour, rng)
    else:
        beta_x, beta_y = _sample_point_near_caustic(contour, rng)
    return beta_x, beta_y, contour


def _caustic_contour_to_json(contour: CausticContour) -> dict[str, Any]:
    return {
        "caustic_index": int(contour.caustic_index),
        "caustic_class": str(contour.caustic_class),
        "caustic_area_arcsec2": float(contour.caustic_area_arcsec2),
        "critical_area_arcsec2": float(contour.critical_area_arcsec2),
        "critical_x": np.asarray(contour.critical_x, dtype=float).tolist(),
        "critical_y": np.asarray(contour.critical_y, dtype=float).tolist(),
        "caustic_beta_x": np.asarray(contour.beta_x, dtype=float).tolist(),
        "caustic_beta_y": np.asarray(contour.beta_y, dtype=float).tolist(),
    }


def _caustic_contours_by_z_from_truth(truth: dict[str, Any]) -> dict[str, list[CausticContour]]:
    parsed: dict[str, list[CausticContour]] = {}
    raw_by_z = truth.get("caustics_by_source_redshift", {})
    if not isinstance(raw_by_z, dict):
        return parsed
    for z_key, raw_contours in raw_by_z.items():
        contours: list[CausticContour] = []
        if not isinstance(raw_contours, list):
            continue
        for raw in raw_contours:
            if not isinstance(raw, dict):
                continue
            required = {"critical_x", "critical_y", "caustic_beta_x", "caustic_beta_y"}
            if not required.issubset(raw):
                continue
            critical_x = np.asarray(raw["critical_x"], dtype=float)
            critical_y = np.asarray(raw["critical_y"], dtype=float)
            beta_x = np.asarray(raw["caustic_beta_x"], dtype=float)
            beta_y = np.asarray(raw["caustic_beta_y"], dtype=float)
            if (
                critical_x.ndim != 1
                or critical_y.ndim != 1
                or beta_x.ndim != 1
                or beta_y.ndim != 1
                or critical_x.size != critical_y.size
                or beta_x.size != beta_y.size
                or critical_x.size < 3
                or beta_x.size < 3
                or not np.all(np.isfinite(critical_x))
                or not np.all(np.isfinite(critical_y))
                or not np.all(np.isfinite(beta_x))
                or not np.all(np.isfinite(beta_y))
            ):
                continue
            contours.append(
                CausticContour(
                    caustic_index=int(raw.get("caustic_index", len(contours))),
                    caustic_class=str(raw.get("caustic_class", "unknown")),
                    beta_x=beta_x,
                    beta_y=beta_y,
                    critical_x=critical_x,
                    critical_y=critical_y,
                    caustic_area_arcsec2=float(raw.get("caustic_area_arcsec2", _closed_polygon_area(beta_x, beta_y))),
                    critical_area_arcsec2=float(raw.get("critical_area_arcsec2", _closed_polygon_area(critical_x, critical_y))),
                )
            )
        if contours:
            parsed[str(z_key)] = contours
    return parsed


def _valid_source_redshifts(values: tuple[float, ...], config: SingleBCGMockConfig) -> tuple[float, ...]:
    parsed = tuple(
        float(value)
        for value in values
        if np.isfinite(float(value)) and float(value) > float(config.z_lens)
    )
    return parsed if parsed else (float(config.source_redshift),)


def _family_source_redshifts(config: SingleBCGMockConfig, caustic_class: str) -> tuple[float, ...]:
    raw_values = (
        config.subhalo_source_redshifts
        if str(caustic_class) == "subhalo"
        else config.primary_source_redshifts
    )
    return _valid_source_redshifts(raw_values, config)


def _mock_seed_grid_pixel_arcsec(config: SingleBCGMockConfig, caustic_class: str) -> float:
    caustic_class = str(caustic_class)
    if caustic_class == "primary":
        value = float(config.primary_seed_grid_pixel_arcsec)
    elif caustic_class == "subhalo":
        value = float(config.subhalo_seed_grid_pixel_arcsec)
    else:
        raise ValueError(f"Unknown mock caustic class {caustic_class!r}.")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Mock image min_distance for {caustic_class} families must be positive and finite.")
    return value


def _ordered_unique_source_redshifts(*groups: tuple[float, ...]) -> tuple[float, ...]:
    seen: set[float] = set()
    ordered: list[float] = []
    for group in groups:
        for value in group:
            z_source = float(value)
            if z_source in seen:
                continue
            seen.add(z_source)
            ordered.append(z_source)
    return tuple(ordered)


def _mock_family_source_redshifts(
    config: SingleBCGMockConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    primary_source_redshifts = _family_source_redshifts(config, "primary")
    subhalo_source_redshifts = _family_source_redshifts(config, "subhalo")
    primary_family_redshifts = tuple(
        primary_source_redshifts[index % len(primary_source_redshifts)]
        for index in range(int(config.n_primary_families))
    )
    subhalo_family_redshifts = tuple(
        subhalo_source_redshifts[index % len(subhalo_source_redshifts)]
        for index in range(int(config.n_subhalo_families))
    )
    needed_source_redshifts = _ordered_unique_source_redshifts(
        primary_family_redshifts,
        subhalo_family_redshifts,
    )
    return primary_family_redshifts, subhalo_family_redshifts, needed_source_redshifts


def _image_count_requirement_text(min_images: int, max_images: int | None) -> str:
    if max_images is None:
        return f"at least {int(min_images)} images"
    if int(max_images) == int(min_images):
        return f"exactly {int(min_images)} images"
    return f"between {int(min_images)} and {int(max_images)} images"


def _image_count_within_requirement(n_images: int, min_images: int, max_images: int | None) -> bool:
    return int(n_images) >= int(min_images) and (max_images is None or int(n_images) <= int(max_images))


@dataclass(frozen=True)
class MockMappedGrid:
    x_grid: np.ndarray
    y_grid: np.ndarray
    beta_x_grid: np.ndarray
    beta_y_grid: np.ndarray
    num_pix: int
    pixel_width: float


@dataclass(frozen=True)
class MockImageCandidateResult:
    attempt_index: int
    beta_x: float
    beta_y: float
    contour: CausticContour
    x_img: np.ndarray
    y_img: np.ndarray


def _mock_static_lens_state(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    z_source: float,
    cosmo_config: dict[str, Any],
) -> jax_lensing.StaticLensState:
    lens_model_list, kwargs_lens = _mock_lens_model_list_and_kwargs(
        config,
        subhalo_components,
        float(z_source),
        cosmo_config,
    )
    return jax_lensing.static_lens_state_from_kwargs(lens_model_list, kwargs_lens)


def _mock_mapped_grid(
    lens_state: jax_lensing.StaticLensState,
    *,
    search_window_arcsec: float,
    seed_grid_pixel_arcsec: float,
    x_center: float = 0.0,
    y_center: float = 0.0,
) -> MockMappedGrid:
    num_pix = int(search_window_arcsec / seed_grid_pixel_arcsec) + 1
    x_axis = np.linspace(
        -0.5 * float(search_window_arcsec) + float(x_center),
        0.5 * float(search_window_arcsec) + float(x_center),
        num_pix,
    )
    y_axis = np.linspace(
        -0.5 * float(search_window_arcsec) + float(y_center),
        0.5 * float(search_window_arcsec) + float(y_center),
        num_pix,
    )
    xx, yy = np.meshgrid(x_axis, y_axis)
    beta_x, beta_y = jax_lensing.ray_shooting(
        jnp.asarray(xx.ravel(), dtype=jnp.float64),
        jnp.asarray(yy.ravel(), dtype=jnp.float64),
        lens_state,
    )
    pixel_width = float(x_axis[1] - x_axis[0]) if num_pix > 1 else float(search_window_arcsec)
    return MockMappedGrid(
        x_grid=xx.ravel().astype(float),
        y_grid=yy.ravel().astype(float),
        beta_x_grid=np.asarray(beta_x, dtype=float),
        beta_y_grid=np.asarray(beta_y, dtype=float),
        num_pix=int(num_pix),
        pixel_width=float(pixel_width),
    )


def _mock_candidate_seed_grid(
    beta_x_grid: jnp.ndarray,
    beta_y_grid: jnp.ndarray,
    source_x: jnp.ndarray,
    source_y: jnp.ndarray,
    *,
    num_pix: int,
    seed_cap: int,
    pixel_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    source_x = jnp.asarray(source_x, dtype=jnp.float64).reshape(-1, 1)
    source_y = jnp.asarray(source_y, dtype=jnp.float64).reshape(-1, 1)
    dist2 = jnp.square(beta_x_grid[jnp.newaxis, :] - source_x) + jnp.square(beta_y_grid[jnp.newaxis, :] - source_y)
    maps = jnp.reshape(dist2, (dist2.shape[0], int(num_pix), int(num_pix)))
    padded = jnp.pad(maps, ((0, 0), (1, 1), (1, 1)), constant_values=jnp.inf)
    local = jax.lax.reduce_window(
        padded,
        jnp.inf,
        jax.lax.min,
        window_dimensions=(1, 3, 3),
        window_strides=(1, 1, 1),
        padding="VALID",
    )
    is_min = (maps <= local) & (maps <= float(pixel_width) ** 2)
    masked = jnp.where(is_min.reshape(dist2.shape), dist2, jnp.inf)
    order = jnp.argsort(masked, axis=1)[:, : int(seed_cap)]
    values = jnp.take_along_axis(masked, order, axis=1)
    valid = jnp.isfinite(values)
    return order, values, valid


def _refine_mock_image_seeds_jax(
    lens_state: jax_lensing.StaticLensState,
    source_x: float,
    source_y: float,
    start_x: np.ndarray,
    start_y: np.ndarray,
    *,
    max_iter: int,
    precision_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_x = jnp.asarray(start_x, dtype=jnp.float64).reshape(-1)
    current_y = jnp.asarray(start_y, dtype=jnp.float64).reshape(-1)
    source_x_j = jnp.asarray(float(source_x), dtype=jnp.float64)
    source_y_j = jnp.asarray(float(source_y), dtype=jnp.float64)

    def step(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _idx: jnp.ndarray) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        x, y, active = carry
        beta_x, beta_y, h_xx, h_xy, h_yx, h_yy = jax_lensing.alpha_and_hessian(x, y, lens_state)
        beta_x = x - beta_x
        beta_y = y - beta_y
        r0 = beta_x - source_x_j
        r1 = beta_y - source_y_j
        a00 = 1.0 - h_xx
        a01 = -h_xy
        a10 = -h_yx
        a11 = 1.0 - h_yy
        det = a00 * a11 - a01 * a10
        finite = active & jnp.isfinite(det) & (jnp.abs(det) > 1.0e-14)
        dx = -(a11 * r0 - a01 * r1) / det
        dy = -(-a10 * r0 + a00 * r1) / det
        step_norm = jnp.hypot(dx, dy)
        scale = jnp.minimum(1.0, 5.0 / jnp.maximum(step_norm, 1.0e-12))
        dx = jnp.where(finite, dx * scale, 0.0)
        dy = jnp.where(finite, dy * scale, 0.0)
        x_next = x + dx
        y_next = y + dy
        source_resid = jnp.hypot(r0, r1)
        still_active = finite & (source_resid > float(precision_limit)) & (jnp.hypot(dx, dy) > 0.1 * float(precision_limit))
        return (x_next, y_next, still_active), None

    active0 = jnp.isfinite(current_x) & jnp.isfinite(current_y)
    (final_x, final_y, _active), _unused = jax.lax.scan(step, (current_x, current_y, active0), jnp.arange(int(max_iter)))
    beta_x, beta_y = jax_lensing.ray_shooting(final_x, final_y, lens_state)
    residual = jnp.hypot(beta_x - source_x_j, beta_y - source_y_j)
    ok = jnp.isfinite(final_x) & jnp.isfinite(final_y) & jnp.isfinite(residual) & (residual <= float(precision_limit))
    return np.asarray(final_x, dtype=float), np.asarray(final_y, dtype=float), np.asarray(ok, dtype=bool)


def _mock_magnification(
    lens_state: jax_lensing.StaticLensState,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    _alpha_x, _alpha_y, h_xx, h_xy, h_yx, h_yy = jax_lensing.alpha_and_hessian(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
        lens_state,
    )
    a00 = 1.0 - np.asarray(h_xx, dtype=float)
    a01 = -np.asarray(h_xy, dtype=float)
    a10 = -np.asarray(h_yx, dtype=float)
    a11 = 1.0 - np.asarray(h_yy, dtype=float)
    det = a00 * a11 - a01 * a10
    return np.where(np.isfinite(det) & (det != 0.0), 1.0 / det, np.nan)


def _mock_lens_model_list_and_kwargs(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    z_source: float,
    cosmo_config: dict[str, Any],
) -> tuple[list[str], list[dict[str, float]]]:
    lens_model_list = [ORIGINAL_DPIE_PROFILE_NAME, ORIGINAL_DPIE_PROFILE_NAME] + [
        ORIGINAL_DPIE_PROFILE_NAME for _ in subhalo_components
    ]
    kwargs_lens = [
        _component_kwargs(config.halo, config, float(z_source), cosmo_config),
        _component_kwargs(config.bcg, config, float(z_source), cosmo_config),
    ] + [_component_kwargs(component, config, float(z_source), cosmo_config) for component in subhalo_components]
    return lens_model_list, kwargs_lens


def _current_truth_model_and_kwargs(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    z_source: float,
    cosmo_config: dict[str, Any],
) -> tuple[list[str], list[dict[str, float]]]:
    lens_model_list = [ORIGINAL_DPIE_PROFILE_NAME, ORIGINAL_DPIE_PROFILE_NAME] + [
        ORIGINAL_DPIE_PROFILE_NAME for _ in subhalo_components
    ]
    kwargs_lens = [
        _component_kwargs(config.halo, config, float(z_source), cosmo_config),
        _component_kwargs(config.bcg, config, float(z_source), cosmo_config),
    ] + [_component_kwargs(component, config, float(z_source), cosmo_config) for component in subhalo_components]
    return lens_model_list, kwargs_lens


def _current_truth_component_records(
    config: SingleBCGMockConfig,
    subhalo_components: list[DPIETruth],
    source_redshifts: list[float],
    cosmo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    def kwargs_by_z(component: DPIETruth) -> dict[str, dict[str, float]]:
        return {
            f"{float(z_source):.8f}": _component_kwargs(component, config, float(z_source), cosmo_config)
            for z_source in source_redshifts
        }

    records: list[dict[str, Any]] = [
        {
            "component_id": "halo",
            "component_role": "halo",
            "profile_name": ORIGINAL_DPIE_PROFILE_NAME,
            "kwargs_by_source_redshift": kwargs_by_z(config.halo),
        },
        {
            "component_id": "bcg",
            "component_role": "bcg",
            "profile_name": ORIGINAL_DPIE_PROFILE_NAME,
            "kwargs_by_source_redshift": kwargs_by_z(config.bcg),
        },
    ]
    for component in subhalo_components:
        records.append(
            {
                "component_id": str(component.potential_id),
                "component_role": "subhalo",
                "catalog_id": str(component.potential_id),
                "profile_name": ORIGINAL_DPIE_PROFILE_NAME,
                "kwargs_by_source_redshift": kwargs_by_z(component),
            }
        )
    return records


def generate_single_bcg_mock(
    output_dir: str | Path,
    config: SingleBCGMockConfig | None = None,
    *,
    overwrite: bool = True,
    progress_callback: MockProgressCallback | None = None,
) -> tuple[MockClusterPaths, pd.DataFrame, dict[str, Any]]:
    """Generate a deterministic single-BCG mock cluster on disk."""
    config = config or SingleBCGMockConfig()
    config.validate()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    par_path = root / "single_bcg_mock.par"
    image_catalog_path = root / "obs_arcs.cat"
    member_catalog_path = root / "members.cat"
    truth_path = root / "truth.json"
    mock_images_path = root / "mock_images.json"
    if not overwrite and any(path.exists() for path in (par_path, image_catalog_path, member_catalog_path, truth_path, mock_images_path)):
        raise FileExistsError(f"Mock output already exists under {root}.")

    rng = np.random.default_rng(config.seed)
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    kpc_per_arcsec = kpc_per_arcsec_from_config(config.z_lens, cosmo_config)
    _emit_mock_progress(
        progress_callback,
        "subhalos_start",
        requested_subhalos=int(config.n_subhalos),
        parent_factor=int(config.subhalo_parent_factor),
        mag_faint_limit=float(config.subhalo_mag_faint_limit),
    )
    subhalos, subhalo_selection = _generate_subhalo_catalog_payload(config, rng)
    observable_count = 0
    parent_count = 0
    retry_count = 0
    if subhalo_selection is not None:
        parent_count = int(subhalo_selection.get("parent_count", 0))
        observable_count = int(
            sum(1 for candidate in subhalo_selection.get("candidates", []) if candidate.get("passes_mag_cut"))
        )
        base_parent_count = max(int(config.n_subhalos), int(config.subhalo_parent_factor) * int(config.n_subhalos))
        if parent_count > 0 and base_parent_count > 0:
            retry_count = max(0, int(round(math.log2(parent_count / base_parent_count))))
    _emit_mock_progress(
        progress_callback,
        "subhalos_complete",
        requested_subhalos=int(config.n_subhalos),
        selected_subhalos=int(len(subhalos)),
        parent_count=parent_count,
        observable_count=observable_count,
        retry_count=retry_count,
        mag_faint_limit=float(config.subhalo_mag_faint_limit),
    )
    subhalo_components = [_scaled_subhalo_params(row, config) for row in subhalos]
    primary_family_redshifts, subhalo_family_redshifts, needed_source_redshifts = _mock_family_source_redshifts(config)
    redshift_index_by_value = {float(z): index + 1 for index, z in enumerate(needed_source_redshifts)}
    caustic_num_pix = 0
    if float(config.caustic_compute_window_arcsec) > 0.0 and float(config.caustic_grid_scale_arcsec) > 0.0:
        caustic_num_pix = max(
            16,
            int(math.ceil(float(config.caustic_compute_window_arcsec) / float(config.caustic_grid_scale_arcsec))) + 1,
        )
        if caustic_num_pix % 2 == 0:
            caustic_num_pix += 1
    caustic_grid_points = int(caustic_num_pix * caustic_num_pix)
    caustic_chunk_points = min(
        caustic_grid_points,
        jax_lensing.estimated_alpha_hessian_chunk_size(float(config.mock_caustic_grid_chunk_memory_gb)),
    )
    caustic_chunk_count = int(math.ceil(caustic_grid_points / max(caustic_chunk_points, 1))) if caustic_grid_points else 0
    lens_state_cache: dict[float, jax_lensing.StaticLensState] = {}
    caustic_cache: dict[float, list[CausticContour]] = {}
    mapped_grid_cache: dict[tuple[float, float], MockMappedGrid] = {}

    def get_lens_state(z_source: float) -> jax_lensing.StaticLensState:
        z_key = float(z_source)
        if z_key not in lens_state_cache:
            lens_state_cache[z_key] = _mock_static_lens_state(config, subhalo_components, z_key, cosmo_config)
        return lens_state_cache[z_key]

    def get_caustics(z_source: float) -> list[CausticContour]:
        z_key = float(z_source)
        if z_key not in caustic_cache:
            start_time = time.perf_counter()
            _emit_mock_progress(
                progress_callback,
                "redshift_start",
                z_source=z_key,
                redshift_index=int(redshift_index_by_value.get(z_key, len(caustic_cache) + 1)),
                redshift_count=int(len(needed_source_redshifts)),
                lens_component_count=int(2 + len(subhalo_components)),
                caustic_compute_window_arcsec=float(config.caustic_compute_window_arcsec),
                caustic_grid_scale_arcsec=float(config.caustic_grid_scale_arcsec),
                caustic_grid_pixels=int(caustic_num_pix),
                caustic_grid_points=int(caustic_grid_points),
                caustic_grid_chunk_memory_gb=float(config.mock_caustic_grid_chunk_memory_gb),
                caustic_grid_chunk_points=int(caustic_chunk_points),
                caustic_grid_chunk_count=int(caustic_chunk_count),
                mock_generation_workers=int(config.mock_generation_workers),
            )
            contours = _compute_tangential_caustic_contours(get_lens_state(z_key), config)
            caustic_cache[z_key] = contours
            elapsed = time.perf_counter() - start_time
            _emit_mock_progress(
                progress_callback,
                "redshift_complete",
                z_source=z_key,
                redshift_index=int(redshift_index_by_value.get(z_key, len(caustic_cache))),
                redshift_count=int(len(needed_source_redshifts)),
                lens_component_count=int(2 + len(subhalo_components)),
                caustic_compute_window_arcsec=float(config.caustic_compute_window_arcsec),
                caustic_grid_scale_arcsec=float(config.caustic_grid_scale_arcsec),
                caustic_grid_pixels=int(caustic_num_pix),
                caustic_grid_points=int(caustic_grid_points),
                caustic_grid_chunk_memory_gb=float(config.mock_caustic_grid_chunk_memory_gb),
                caustic_grid_chunk_points=int(caustic_chunk_points),
                caustic_grid_chunk_count=int(caustic_chunk_count),
                elapsed_s=float(elapsed),
                mock_generation_workers=int(config.mock_generation_workers),
                caustic_count=int(len(contours)),
                primary_caustic_count=int(sum(1 for contour in contours if contour.caustic_class == "primary")),
                subhalo_caustic_count=int(sum(1 for contour in contours if contour.caustic_class == "subhalo")),
            )
        return caustic_cache[z_key]

    def get_mapped_grid(z_source: float, seed_grid_pixel_arcsec: float) -> MockMappedGrid:
        key = (float(z_source), float(seed_grid_pixel_arcsec))
        if key not in mapped_grid_cache:
            start_time = time.perf_counter()
            mapped_grid_cache[key] = _mock_mapped_grid(
                get_lens_state(float(z_source)),
                search_window_arcsec=float(config.mock_image_search_window_arcsec),
                seed_grid_pixel_arcsec=float(seed_grid_pixel_arcsec),
            )
            _emit_mock_progress(
                progress_callback,
                "mapped_grid_complete",
                z_source=float(z_source),
                mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
                seed_grid_pixel_arcsec=float(seed_grid_pixel_arcsec),
                mapped_grid_pixels=int(mapped_grid_cache[key].num_pix),
                mapped_grid_points=int(mapped_grid_cache[key].x_grid.size),
                elapsed_s=float(time.perf_counter() - start_time),
            )
        return mapped_grid_cache[key]

    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    target_families = [("primary", z_source) for z_source in primary_family_redshifts] + [
        ("subhalo", z_source) for z_source in subhalo_family_redshifts
    ]

    @dataclass
    class FamilySearchState:
        family_index: int
        family_id: str
        caustic_class: str
        z_source: float
        contours: list[CausticContour]
        seed_grid_pixel_arcsec: float
        max_attempts: int
        inside_caustic: bool
        candidate_rng: np.random.Generator
        observation_rng: np.random.Generator
        next_attempt_index: int = 1
        accepted: tuple[float, float, CausticContour, np.ndarray, np.ndarray] | None = None
        accepted_attempt: int | None = None
        seed_cap_saturations: int = 0
        focused_retry: bool = False
        drop_reason: str | None = None

    caustic_workers = max(1, min(int(config.mock_generation_workers), len(needed_source_redshifts)))
    if caustic_workers > 1:
        with ThreadPoolExecutor(max_workers=caustic_workers) as executor:
            future_by_z = {
                executor.submit(get_caustics, float(z_source)): float(z_source)
                for z_source in needed_source_redshifts
            }
            for future in as_completed(future_by_z):
                future.result()
    else:
        for z_source in needed_source_redshifts:
            get_caustics(float(z_source))

    progress_lock = threading.Lock()
    accepted_family_lock = threading.Lock()
    accepted_family_count = 0

    def emit_mock_progress(event: str, **payload: Any) -> None:
        with progress_lock:
            _emit_mock_progress(progress_callback, event, **payload)

    max_attempts = max(1, int(config.max_sources_to_try))
    family_count = int(len(target_families))
    if family_count:
        family_seed_values = rng.integers(
            0,
            np.iinfo(np.int64).max,
            size=(family_count, 2),
            dtype=np.int64,
        )
    else:
        family_seed_values = np.empty((0, 2), dtype=np.int64)
    family_states: list[FamilySearchState] = []
    for family_index, (caustic_class, z_source) in enumerate(target_families, start=1):
        contours = get_caustics(float(z_source))
        inside_caustic = config.max_images_per_family is None or int(config.max_images_per_family) > 3
        family_state = FamilySearchState(
            family_index=int(family_index),
            family_id=str(family_index),
            caustic_class=str(caustic_class),
            z_source=float(z_source),
            contours=contours,
            seed_grid_pixel_arcsec=_mock_seed_grid_pixel_arcsec(config, str(caustic_class)),
            max_attempts=max_attempts,
            inside_caustic=inside_caustic,
            candidate_rng=np.random.default_rng(int(family_seed_values[family_index - 1, 0])),
            observation_rng=np.random.default_rng(int(family_seed_values[family_index - 1, 1])),
        )
        family_states.append(family_state)
        emit_mock_progress(
            "family_start",
            family_index=int(family_index),
            family_count=int(family_count),
            caustic_class=str(caustic_class),
            z_source=float(z_source),
            max_attempts=int(max_attempts),
            seed_grid_pixel_arcsec=float(family_state.seed_grid_pixel_arcsec),
            mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
            image_candidate_batch_size=int(config.mock_image_candidate_batch_size),
            mock_image_seed_cap=int(config.mock_image_seed_cap),
        )

    for z_source, seed_grid_pixel_arcsec in sorted(
        {
            (float(state.z_source), float(state.seed_grid_pixel_arcsec))
            for state in family_states
        }
    ):
        get_mapped_grid(z_source, seed_grid_pixel_arcsec)

    family_workers = max(1, min(int(config.mock_generation_workers), len(family_states)))

    emit_mock_progress(
        "family_queue_start",
        family_count=int(family_count),
        image_solver_workers=int(family_workers),
        queued_families=int(len(family_states)),
        mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
        image_candidate_batch_size=int(config.mock_image_candidate_batch_size),
        mock_image_seed_cap=int(config.mock_image_seed_cap),
        max_attempts=int(max_attempts),
    )

    def sample_candidate_attempt(state: FamilySearchState) -> tuple[int, float, float, CausticContour]:
        attempt_index = int(state.next_attempt_index)
        beta_x, beta_y, contour = _sample_caustic_source_candidate(
            state.contours,
            state.caustic_class,
            state.candidate_rng,
            inside_caustic=state.inside_caustic,
        )
        state.next_attempt_index += 1
        return int(attempt_index), float(beta_x), float(beta_y), contour

    def fail_family(state: FamilySearchState) -> RuntimeError:
        emit_mock_progress(
            "family_fail",
            family_index=int(state.family_index),
            family_count=int(family_count),
            caustic_class=str(state.caustic_class),
            z_source=float(state.z_source),
            max_attempts=int(state.max_attempts),
            seed_grid_pixel_arcsec=float(state.seed_grid_pixel_arcsec),
            mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
            image_candidate_batch_size=int(config.mock_image_candidate_batch_size),
            mock_image_seed_cap=int(config.mock_image_seed_cap),
            image_solver_workers=int(family_workers),
        )
        return RuntimeError(
            f"Failed to generate a {state.caustic_class} source with "
            f"{_image_count_requirement_text(config.min_images_per_family, config.max_images_per_family)} "
            f"at z_source={state.z_source:.6g} after "
            f"{config.max_sources_to_try} attempts."
        )

    # Subhalo image multiplets are split by ~2 theta_E (often well below the global seed-grid
    # pixel of seed_grid_pixel_arcsec), so subhalo candidates get an additional fine seed
    # grid centred on their contour's critical curve; without it the global grid cannot place
    # seeds in the distinct image basins and families exhaust their attempts spuriously.
    local_seed_grid_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    local_seed_grid_lock = threading.Lock()

    def get_local_seed_grid(
        z_source: float,
        contour: CausticContour,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        key = (float(z_source), int(contour.caustic_index))
        with local_seed_grid_lock:
            cached = local_seed_grid_cache.get(key)
        if cached is not None:
            return cached
        critical_x = np.asarray(contour.critical_x, dtype=float)
        critical_y = np.asarray(contour.critical_y, dtype=float)
        center_x = 0.5 * (float(critical_x.min()) + float(critical_x.max()))
        center_y = 0.5 * (float(critical_y.min()) + float(critical_y.max()))
        extent = max(
            float(critical_x.max()) - float(critical_x.min()),
            float(critical_y.max()) - float(critical_y.min()),
            2.0 * MOCK_LOCAL_SEED_GRID_MIN_HALF_ARCSEC,
        )
        half = 0.5 * float(MOCK_LOCAL_SEED_GRID_MARGIN) * extent
        n = int(MOCK_LOCAL_SEED_GRID_POINTS)
        axis_x = np.linspace(center_x - half, center_x + half, n)
        axis_y = np.linspace(center_y - half, center_y + half, n)
        grid_x, grid_y = np.meshgrid(axis_x, axis_y)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        lens_state = get_lens_state(float(z_source))
        beta_x_grid, beta_y_grid = jax_lensing.ray_shooting(
            jnp.asarray(grid_x, dtype=jnp.float64),
            jnp.asarray(grid_y, dtype=jnp.float64),
            lens_state,
        )
        entry = (
            grid_x,
            grid_y,
            np.asarray(beta_x_grid, dtype=float),
            np.asarray(beta_y_grid, dtype=float),
            2.0 * half / max(n - 1, 1),
        )
        with local_seed_grid_lock:
            local_seed_grid_cache[key] = entry
        return entry

    def local_seed_points(
        state: FamilySearchState,
        contour: CausticContour,
        beta_x: float,
        beta_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        grid_x, grid_y, beta_x_grid, beta_y_grid, pixel = get_local_seed_grid(state.z_source, contour)
        dist2 = np.square(beta_x_grid - float(beta_x)) + np.square(beta_y_grid - float(beta_y))
        selected = dist2 <= (2.0 * pixel) ** 2
        if not np.any(selected):
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        order = np.argsort(dist2[selected])[: int(config.mock_image_seed_cap)]
        return grid_x[selected][order], grid_y[selected][order]

    def evaluate_candidate_batch(
        state: FamilySearchState,
        candidates: list[tuple[int, float, float, CausticContour]],
    ) -> list[MockImageCandidateResult]:
        if not candidates:
            return []
        mapped = get_mapped_grid(state.z_source, state.seed_grid_pixel_arcsec)
        source_x = np.asarray([candidate[1] for candidate in candidates], dtype=float)
        source_y = np.asarray([candidate[2] for candidate in candidates], dtype=float)
        seed_indices, _seed_values, valid_seeds = _mock_candidate_seed_grid(
            jnp.asarray(mapped.beta_x_grid, dtype=jnp.float64),
            jnp.asarray(mapped.beta_y_grid, dtype=jnp.float64),
            jnp.asarray(source_x, dtype=jnp.float64),
            jnp.asarray(source_y, dtype=jnp.float64),
            num_pix=int(mapped.num_pix),
            seed_cap=int(config.mock_image_seed_cap),
            pixel_width=float(mapped.pixel_width),
        )
        seed_indices_np = np.asarray(seed_indices, dtype=int)
        valid_seeds_np = np.asarray(valid_seeds, dtype=bool)
        lens_state = get_lens_state(state.z_source)
        results: list[MockImageCandidateResult] = []
        for row_index, candidate in enumerate(candidates):
            attempt_index, beta_x, beta_y, contour = candidate
            valid = valid_seeds_np[row_index]
            if int(np.count_nonzero(valid)) >= int(config.mock_image_seed_cap):
                if state.seed_cap_saturations == 0:
                    emit_mock_progress(
                        "family_seed_cap_saturated",
                        family_index=int(state.family_index),
                        caustic_class=str(state.caustic_class),
                        z_source=float(state.z_source),
                        seed_cap=int(config.mock_image_seed_cap),
                    )
                state.seed_cap_saturations += 1
            indices = seed_indices_np[row_index][valid]
            start_x = mapped.x_grid[indices]
            start_y = mapped.y_grid[indices]
            if str(state.caustic_class) == "subhalo":
                local_x, local_y = local_seed_points(state, contour, float(beta_x), float(beta_y))
                if local_x.size:
                    start_x = np.concatenate([start_x, local_x])
                    start_y = np.concatenate([start_y, local_y])
            if start_x.size == 0:
                results.append(MockImageCandidateResult(attempt_index, beta_x, beta_y, contour, np.asarray([]), np.asarray([])))
                continue
            refined_x, refined_y, ok = _refine_mock_image_seeds_jax(
                lens_state,
                beta_x,
                beta_y,
                start_x,
                start_y,
                max_iter=int(config.mock_image_lm_max_iter),
                precision_limit=float(config.mock_image_precision_limit),
            )
            x_arr, y_arr = _deduplicate_images(refined_x[ok], refined_y[ok], tolerance=1.0e-3)
            results.append(MockImageCandidateResult(attempt_index, beta_x, beta_y, contour, x_arr, y_arr))
        return results

    def run_attempt_loop(state: FamilySearchState) -> None:
        nonlocal accepted_family_count
        while state.accepted is None and state.next_attempt_index <= state.max_attempts:
            batch: list[tuple[int, float, float, CausticContour]] = []
            try:
                while state.next_attempt_index <= state.max_attempts and len(batch) < int(config.mock_image_candidate_batch_size):
                    batch.append(sample_candidate_attempt(state))
            except RuntimeError:
                # No caustics of this class exist at this source redshift; a subhalo
                # family is then dropped below instead of failing the whole seed.
                if str(state.caustic_class) == "subhalo":
                    state.drop_reason = "no_usable_caustics"
                    break
                raise
            for result in evaluate_candidate_batch(state, batch):
                accepted_attempt = _image_count_within_requirement(
                    len(result.x_img),
                    int(config.min_images_per_family),
                    config.max_images_per_family,
                )
                with accepted_family_lock:
                    current_accepted_family_count = int(accepted_family_count)
                emit_mock_progress(
                    "family_attempt",
                    family_index=int(state.family_index),
                    family_count=int(family_count),
                    caustic_class=str(state.caustic_class),
                    z_source=float(state.z_source),
                    attempt=int(result.attempt_index),
                    max_attempts=int(state.max_attempts),
                    image_count=int(len(result.x_img)),
                    accepted=bool(accepted_attempt),
                    seed_grid_pixel_arcsec=float(state.seed_grid_pixel_arcsec),
                    mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
                    image_candidate_batch_size=int(config.mock_image_candidate_batch_size),
                    mock_image_seed_cap=int(config.mock_image_seed_cap),
                    image_solver_workers=int(family_workers),
                    accepted_families=int(current_accepted_family_count),
                )
                if accepted_attempt:
                    state.accepted = (
                        float(result.beta_x),
                        float(result.beta_y),
                        result.contour,
                        result.x_img,
                        result.y_img,
                    )
                    state.accepted_attempt = int(result.attempt_index)
                    with accepted_family_lock:
                        accepted_family_count += 1
                        current_accepted_family_count = int(accepted_family_count)
                    emit_mock_progress(
                        "family_accept",
                        family_index=int(state.family_index),
                        family_count=int(family_count),
                        caustic_class=str(state.caustic_class),
                        z_source=float(state.z_source),
                        attempt=int(result.attempt_index),
                        max_attempts=int(state.max_attempts),
                        image_count=int(len(result.x_img)),
                        seed_grid_pixel_arcsec=float(state.seed_grid_pixel_arcsec),
                        mock_image_search_window_arcsec=float(config.mock_image_search_window_arcsec),
                        caustic_index=int(result.contour.caustic_index),
                        caustic_area_arcsec2=float(result.contour.caustic_area_arcsec2),
                        image_candidate_batch_size=int(config.mock_image_candidate_batch_size),
                        mock_image_seed_cap=int(config.mock_image_seed_cap),
                        image_solver_workers=int(family_workers),
                        accepted_families=int(current_accepted_family_count),
                    )
                    break

    def search_family(state: FamilySearchState) -> FamilySearchState:
        class_contours = [contour for contour in state.contours if contour.caustic_class == state.caustic_class]
        if str(state.caustic_class) == "subhalo":
            max_area = max((float(contour.caustic_area_arcsec2) for contour in class_contours), default=0.0)
            if max_area < float(config.caustic_min_area_arcsec2):
                # No subhalo caustic can plausibly host a multiply imaged source; skip
                # the attempt budget entirely and drop the family.
                state.drop_reason = "no_usable_caustics"
                fail_family(state)
                return state
        run_attempt_loop(state)
        if (
            state.accepted is None
            and state.drop_reason is None
            and str(state.caustic_class) == "subhalo"
            and class_contours
        ):
            # Focused retry: area-weighted sampling spreads attempts across all subhalo
            # caustics; concentrate one more budget round on the largest before dropping.
            largest = max(class_contours, key=lambda contour: float(contour.caustic_area_arcsec2))
            state.contours = [largest]
            state.focused_retry = True
            state.max_attempts = int(state.max_attempts) + int(max_attempts)
            emit_mock_progress(
                "family_focused_retry",
                family_index=int(state.family_index),
                caustic_class=str(state.caustic_class),
                z_source=float(state.z_source),
                caustic_index=int(largest.caustic_index),
                caustic_area_arcsec2=float(largest.caustic_area_arcsec2),
                max_attempts=int(state.max_attempts),
            )
            run_attempt_loop(state)
        if state.accepted is None:
            if str(state.caustic_class) == "subhalo":
                # Subhalo caustics can be too small to multiply-image any source within
                # the attempt budget. Dropping the family (loudly) is preferable to
                # failing the whole seed; primary families remain fatal.
                if state.drop_reason is None:
                    state.drop_reason = "attempts_exhausted"
                fail_family(state)
                return state
            raise fail_family(state)
        return state

    if family_workers > 1:
        executor = ThreadPoolExecutor(max_workers=family_workers)
        futures = [executor.submit(search_family, state) for state in family_states]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        for state in family_states:
            search_family(state)

    dropped_families: list[dict[str, Any]] = []
    for state in sorted(family_states, key=lambda item: item.family_index):
        if state.accepted is None:
            if str(state.caustic_class) != "subhalo":
                raise RuntimeError(
                    f"Primary family {state.family_id} at z_source={state.z_source:.6g} "
                    "was not generated; primary families are required."
                )
            dropped_families.append(
                {
                    "family_id": str(state.family_id),
                    "caustic_class": str(state.caustic_class),
                    "z_source": float(state.z_source),
                    "max_attempts": int(state.max_attempts),
                    "attempts_used": int(state.next_attempt_index) - 1,
                    "focused_retry": bool(state.focused_retry),
                    "reason": str(state.drop_reason or "attempts_exhausted"),
                }
            )
            continue
        beta_x, beta_y, contour, x_arr, y_arr = state.accepted
        family_id = state.family_id
        source_rows.append(
            {
                "family_id": family_id,
                "beta_x": float(beta_x),
                "beta_y": float(beta_y),
                "z_source": float(state.z_source),
                "n_images": int(len(x_arr)),
                "caustic_class": str(contour.caustic_class),
                "caustic_index": int(contour.caustic_index),
                "caustic_area_arcsec2": float(contour.caustic_area_arcsec2),
                "critical_area_arcsec2": float(contour.critical_area_arcsec2),
            }
        )
        family_lens_state = get_lens_state(state.z_source)
        magnification = _mock_magnification(family_lens_state, x_arr, y_arr)
        x_obs_arr = x_arr + state.observation_rng.normal(0.0, config.pos_sigma_arcsec, size=x_arr.shape)
        y_obs_arr = y_arr + state.observation_rng.normal(0.0, config.pos_sigma_arcsec, size=y_arr.shape)
        # The recovery metric evaluates the model magnification at the OBSERVED positions,
        # so record the truth magnification there too; near critical curves the two differ.
        magnification_at_obs = _mock_magnification(family_lens_state, x_obs_arr, y_obs_arr)
        source_magnitude_true = float(
            state.observation_rng.uniform(float(config.source_magnitude_min), float(config.source_magnitude_max))
        )
        source_rows[-1]["source_magnitude_true"] = source_magnitude_true
        image_magnitude_true = source_magnitude_true - 2.5 * np.log10(
            np.abs(magnification_at_obs) + float(config.image_magnitude_mu_floor)
        )
        image_magnitude_noise = state.observation_rng.normal(
            0.0,
            float(config.image_magnitude_sigma_mag),
            size=image_magnitude_true.shape,
        )
        image_magnitude_obs = image_magnitude_true + image_magnitude_noise
        for image_index, (x_true, y_true, x_obs, y_obs, mu_true, mu_obs, mag_true, mag_obs) in enumerate(
            zip(
                x_arr,
                y_arr,
                x_obs_arr,
                y_obs_arr,
                magnification,
                magnification_at_obs,
                image_magnitude_true,
                image_magnitude_obs,
            ),
            start=1,
        ):
            image_id = _lowercase_image_suffix(image_index)
            image_rows.append(
                {
                    "image_label": f"{family_id}.{image_id}",
                    "family_id": family_id,
                    "image_id": image_id,
                    "z_source": float(state.z_source),
                    "x_true_arcsec": float(x_true),
                    "y_true_arcsec": float(y_true),
                    "x_obs_arcsec": float(x_obs),
                    "y_obs_arcsec": float(y_obs),
                    "magnification_true": float(mu_true),
                    "magnification_true_at_obs": float(mu_obs),
                    "source_magnitude_true": source_magnitude_true,
                    "image_magnitude_true": float(mag_true),
                    "image_magnitude_obs": float(mag_obs),
                    "image_magnitude_err": float(config.image_magnitude_sigma_mag),
                    "caustic_class": str(contour.caustic_class),
                    "caustic_index": int(contour.caustic_index),
                }
            )

    if len(source_rows) + len(dropped_families) < config.n_families:
        raise RuntimeError(
            f"Generated {len(source_rows)} multiply imaged families "
            f"(+{len(dropped_families)} dropped subhalo families), fewer than requested {config.n_families}."
        )
    if len(source_rows) < int(config.n_primary_families):
        raise RuntimeError(
            f"Generated {len(source_rows)} families, fewer than the {config.n_primary_families} "
            "required primary families."
        )

    images = pd.DataFrame(image_rows)
    _emit_mock_progress(
        progress_callback,
        "outputs_start",
        output_dir=str(root),
        family_count=int(len(source_rows)),
        image_count=int(len(image_rows)),
        subhalo_count=int(len(subhalos)),
    )
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        + "\n".join(
            (
                f"{row.image_label} {row.x_obs_arcsec:.8f} {row.y_obs_arcsec:.8f} "
                f"0.3734 0.3734 90.0 {row.z_source:.8f} "
                f"{row.image_magnitude_obs:.8f} 1.0 {row.image_magnitude_err:.8f}"
            )
            for row in images.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )
    if subhalos:
        _write_member_catalog(member_catalog_path, subhalos)
    _write_single_bcg_par(par_path, config, images, subhalos)

    parameter_truth = _truth_parameter_values(config, kpc_per_arcsec)
    parameter_truth["cosmology_Om0"] = 0.3
    parameter_truth["cosmology_w0"] = -1.0
    for source_row in source_rows:
        family_id = str(source_row["family_id"])
        parameter_truth[f"source.{family_id}.beta_x"] = float(source_row["beta_x"])
        parameter_truth[f"source.{family_id}.beta_y"] = float(source_row["beta_y"])
    truth_lens_model_list, truth_kwargs_lens = _current_truth_model_and_kwargs(
        config,
        subhalo_components,
        float(config.source_redshift),
        cosmo_config,
    )
    truth_source_redshifts = _ordered_unique_source_redshifts(
        needed_source_redshifts,
        (float(config.source_redshift),),
    )
    truth_kwargs_lens_by_z = {
        f"{z:.8f}": _current_truth_model_and_kwargs(config, subhalo_components, float(z), cosmo_config)[1]
        for z in truth_source_redshifts
    }
    truth_lens_components = _current_truth_component_records(
        config,
        subhalo_components,
        truth_source_redshifts,
        cosmo_config,
    )

    truth_payload = {
        "mock": "single-bcg-subhalos" if subhalos else "single-bcg",
        "profile_convention": "lenstool_profile_81",
        "config": _config_to_jsonable(config, kpc_per_arcsec),
        "kpc_per_arcsec": kpc_per_arcsec,
        "parameter_truth": parameter_truth,
        "sources": source_rows,
        "images": image_rows,
        "subhalos": subhalos,
        "subhalo_components": [asdict(component) for component in subhalo_components],
        "caustics_by_source_redshift": {
            f"{z:.8f}": [_caustic_contour_to_json(contour) for contour in get_caustics(float(z))]
            for z in needed_source_redshifts
        },
        "lens_model_list": truth_lens_model_list,
        "kwargs_lens": truth_kwargs_lens,
        "kwargs_lens_by_source_redshift": truth_kwargs_lens_by_z,
        "lens_components": truth_lens_components,
    }
    if subhalo_selection is not None:
        truth_payload["subhalo_selection"] = subhalo_selection
    if dropped_families:
        truth_payload["dropped_families"] = dropped_families
    truth_payload["generation_stats"] = {
        "families": [
            {
                "family_id": str(state.family_id),
                "caustic_class": str(state.caustic_class),
                "z_source": float(state.z_source),
                "accepted": state.accepted is not None,
                "accepted_attempt": state.accepted_attempt,
                "attempts_used": int(state.next_attempt_index) - 1,
                "focused_retry": bool(state.focused_retry),
                "seed_cap_saturations": int(state.seed_cap_saturations),
                "drop_reason": state.drop_reason,
            }
            for state in sorted(family_states, key=lambda item: item.family_index)
        ],
        "dropped_family_count": int(len(dropped_families)),
        "seed_cap_saturation_total": int(sum(state.seed_cap_saturations for state in family_states)),
    }
    truth_path.write_text(json.dumps(truth_payload, indent=2), encoding="utf-8")
    mock_images_path.write_text(json.dumps(image_rows, indent=2), encoding="utf-8")
    _emit_mock_progress(
        progress_callback,
        "outputs_complete",
        output_dir=str(root),
        par_path=str(par_path),
        image_catalog_path=str(image_catalog_path),
        member_catalog_path=str(member_catalog_path) if subhalos else None,
        truth_path=str(truth_path),
        mock_images_path=str(mock_images_path),
        family_count=int(len(source_rows)),
        image_count=int(len(image_rows)),
        subhalo_count=int(len(subhalos)),
    )
    return MockClusterPaths(root, par_path, image_catalog_path, truth_path, mock_images_path), images, truth_payload


def _source_redshift_tuple_from_config(raw: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    try:
        if isinstance(raw, str):
            values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
        else:
            values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return default
    return values if values else default


def _caustic_config_from_truth(truth: dict[str, Any]) -> SingleBCGMockConfig:
    raw = truth.get("config", {})
    config = raw if isinstance(raw, dict) else {}
    if "subhalo_beta_radius" in config:
        raise ValueError("Legacy mock config field subhalo_beta_radius is unsupported; regenerate the mock with subhalo_gamma_ml.")
    defaults = SingleBCGMockConfig()
    legacy_source_redshifts_raw = config.get("source_redshifts", defaults.primary_source_redshifts)
    primary_source_redshifts = _source_redshift_tuple_from_config(
        config.get("primary_source_redshifts", legacy_source_redshifts_raw),
        defaults.primary_source_redshifts,
    )
    subhalo_source_redshifts = _source_redshift_tuple_from_config(
        config.get("subhalo_source_redshifts", legacy_source_redshifts_raw),
        defaults.subhalo_source_redshifts,
    )
    max_images_raw = config.get("max_images_per_family", defaults.max_images_per_family)
    max_images_per_family = None if max_images_raw is None else int(max_images_raw)
    return SingleBCGMockConfig(
        z_lens=float(config.get("z_lens", defaults.z_lens)),
        source_redshift=float(config.get("source_redshift", defaults.source_redshift)),
        primary_source_redshifts=primary_source_redshifts,
        subhalo_source_redshifts=subhalo_source_redshifts,
        n_primary_families=int(config.get("n_primary_families", defaults.n_primary_families)),
        n_subhalo_families=int(config.get("n_subhalo_families", defaults.n_subhalo_families)),
        min_images_per_family=int(config.get("min_images_per_family", defaults.min_images_per_family)),
        max_images_per_family=max_images_per_family,
        primary_seed_grid_pixel_arcsec=float(
            config.get(
                "primary_seed_grid_pixel_arcsec",
                config.get("primary_image_min_distance_arcsec", defaults.primary_seed_grid_pixel_arcsec),
            )
        ),
        subhalo_seed_grid_pixel_arcsec=float(
            config.get(
                "subhalo_seed_grid_pixel_arcsec",
                config.get("subhalo_image_min_distance_arcsec", defaults.subhalo_seed_grid_pixel_arcsec),
            )
        ),
        bcg_position_prior_half_width_arcsec=float(
            config.get("bcg_position_prior_half_width_arcsec", defaults.bcg_position_prior_half_width_arcsec)
        ),
        n_subhalos=int(config.get("n_subhalos", defaults.n_subhalos)),
        subhalo_schechter_alpha=float(
            config.get("subhalo_schechter_alpha", defaults.subhalo_schechter_alpha)
        ),
        subhalo_parent_factor=int(config.get("subhalo_parent_factor", defaults.subhalo_parent_factor)),
        subhalo_mag_faint_limit=float(config.get("subhalo_mag_faint_limit", defaults.subhalo_mag_faint_limit)),
        subhalo_mass_min=float(config.get("subhalo_mass_min", defaults.subhalo_mass_min)),
        subhalo_mass_max=float(config.get("subhalo_mass_max", defaults.subhalo_mass_max)),
        subhalo_mass_ref=float(config.get("subhalo_mass_ref", defaults.subhalo_mass_ref)),
        subhalo_spatial_distribution=str(
            config.get("subhalo_spatial_distribution", defaults.subhalo_spatial_distribution)
        ),
        subhalo_field_radius_arcsec=float(
            config.get("subhalo_field_radius_arcsec", defaults.subhalo_field_radius_arcsec)
        ),
        subhalo_spatial_core_radius_arcsec=float(
            config.get("subhalo_spatial_core_radius_arcsec", defaults.subhalo_spatial_core_radius_arcsec)
        ),
        subhalo_spatial_cut_radius_arcsec=float(
            config.get("subhalo_spatial_cut_radius_arcsec", defaults.subhalo_spatial_cut_radius_arcsec)
        ),
        subhalo_force_core_count=int(config.get("subhalo_force_core_count", defaults.subhalo_force_core_count)),
        subhalo_force_core_min_arcsec=float(
            config.get("subhalo_force_core_min_arcsec", defaults.subhalo_force_core_min_arcsec)
        ),
        subhalo_force_core_max_arcsec=float(
            config.get("subhalo_force_core_max_arcsec", defaults.subhalo_force_core_max_arcsec)
        ),
        subhalo_sigma_scatter_dex=float(
            config.get("subhalo_sigma_scatter_dex", defaults.subhalo_sigma_scatter_dex)
        ),
        subhalo_cut_scatter_dex=float(config.get("subhalo_cut_scatter_dex", defaults.subhalo_cut_scatter_dex)),
        subhalo_gamma_ml=float(config.get("subhalo_gamma_ml", defaults.subhalo_gamma_ml)),
        source_magnitude_min=float(config.get("source_magnitude_min", defaults.source_magnitude_min)),
        source_magnitude_max=float(config.get("source_magnitude_max", defaults.source_magnitude_max)),
        image_magnitude_sigma_mag=float(
            config.get("image_magnitude_sigma_mag", defaults.image_magnitude_sigma_mag)
        ),
        image_magnitude_mu_floor=float(config.get("image_magnitude_mu_floor", defaults.image_magnitude_mu_floor)),
        caustic_compute_window_arcsec=float(
            config.get("caustic_compute_window_arcsec", defaults.caustic_compute_window_arcsec)
        ),
        caustic_grid_scale_arcsec=float(config.get("caustic_grid_scale_arcsec", defaults.caustic_grid_scale_arcsec)),
        caustic_min_area_arcsec2=float(config.get("caustic_min_area_arcsec2", defaults.caustic_min_area_arcsec2)),
        caustic_boundary_margin_arcsec=float(
            config.get("caustic_boundary_margin_arcsec", defaults.caustic_boundary_margin_arcsec)
        ),
        mock_caustic_grid_chunk_memory_gb=float(
            config.get("mock_caustic_grid_chunk_memory_gb", defaults.mock_caustic_grid_chunk_memory_gb)
        ),
        mock_image_candidate_batch_size=int(
            config.get("mock_image_candidate_batch_size", defaults.mock_image_candidate_batch_size)
        ),
        mock_image_seed_cap=int(config.get("mock_image_seed_cap", defaults.mock_image_seed_cap)),
        mock_image_search_window_arcsec=float(
            config.get("mock_image_search_window_arcsec", defaults.mock_image_search_window_arcsec)
        ),
        mock_image_lm_max_iter=int(config.get("mock_image_lm_max_iter", defaults.mock_image_lm_max_iter)),
        mock_image_precision_limit=float(
            config.get("mock_image_precision_limit", defaults.mock_image_precision_limit)
        ),
        mock_generation_workers=int(config.get("mock_generation_workers", defaults.mock_generation_workers)),
    )
