from __future__ import annotations

import importlib.metadata as importlib_metadata
import sys
import types
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def _install_pkg_resources_shim() -> None:
    if "pkg_resources" in sys.modules:
        return
    shim = types.ModuleType("pkg_resources")
    shim.DistributionNotFound = Exception

    def get_distribution(name: str) -> Any:
        normalized = str(name).replace("_", "-")
        return types.SimpleNamespace(version=importlib_metadata.version(normalized))

    shim.get_distribution = get_distribution
    sys.modules["pkg_resources"] = shim


try:
    import jax_cosmo as jc
    import jax_cosmo.background as jc_background
except ModuleNotFoundError as exc:  # pragma: no cover - depends on external package metadata behavior
    if exc.name != "pkg_resources":
        raise
    _install_pkg_resources_shim()
    import jax_cosmo as jc
    import jax_cosmo.background as jc_background


DEFAULT_JAX_COSMO_DISTANCE_STEPS = 256
DEFAULT_FLAT_WCDM_QUADRATURE_ORDER = DEFAULT_JAX_COSMO_DISTANCE_STEPS
C_LIGHT_KM_S = 299792.458
C_LIGHT_M_S = 299792458.0
G_SI = 6.67384e-11
MPC_M = 3.08567758e22
M_SUN_KG = 1.9891e30
ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


def flat_wcdm_config(
    *,
    h0: float = 70.0,
    om0: float = 0.3,
    ode0: float | None = None,
    w0: float = -1.0,
) -> dict[str, float | str]:
    ode0_value = 1.0 - float(om0) if ode0 is None else float(ode0)
    class_name = "FlatLambdaCDM" if abs(float(w0) + 1.0) < 1.0e-10 else "FlatwCDM"
    return {
        "class": class_name,
        "H0": float(h0),
        "Om0": float(om0),
        "Ode0": float(ode0_value),
        "w0": float(w0),
    }


def h0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return 70.0
    return float(cosmo_config.get("H0", 70.0))


def om0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return 0.3
    return float(cosmo_config.get("Om0", cosmo_config.get("omegaM", cosmo_config.get("omega", 0.3))))


def ode0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    om0 = om0_from_config(cosmo_config)
    if not isinstance(cosmo_config, dict):
        return 1.0 - om0
    return float(cosmo_config.get("Ode0", cosmo_config.get("omegaX", cosmo_config.get("lambda", 1.0 - om0))))


def w0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return -1.0
    return float(cosmo_config.get("w0", cosmo_config.get("wX", cosmo_config.get("w", -1.0))))


def flat_wcdm_cosmology(
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
) -> Any:
    h = jnp.asarray(h0, dtype=jnp.float64) / 100.0
    return jc.Cosmology(
        Omega_c=jnp.asarray(omega_m, dtype=jnp.float64),
        Omega_b=jnp.asarray(0.0, dtype=jnp.float64),
        h=h,
        n_s=jnp.asarray(0.96, dtype=jnp.float64),
        sigma8=jnp.asarray(0.8, dtype=jnp.float64),
        Omega_k=jnp.asarray(0.0, dtype=jnp.float64),
        w0=jnp.asarray(w0, dtype=jnp.float64),
        wa=jnp.asarray(0.0, dtype=jnp.float64),
    )


def scale_factor_from_redshift(z: float | jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64))


def _reshape_distance(distance: jnp.ndarray, z: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(jnp.asarray(distance, dtype=jnp.float64), jnp.asarray(z, dtype=jnp.float64).shape)


def flat_wcdm_comoving_distance_mpc(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    cosmo = flat_wcdm_cosmology(h0, omega_m, w0)
    distance_mpc_over_h = jc_background.radial_comoving_distance(
        cosmo,
        scale_factor_from_redshift(z),
        steps=int(steps),
    )
    return _reshape_distance(distance_mpc_over_h / cosmo.h, z)


def flat_wcdm_angular_diameter_distance_mpc(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    return flat_wcdm_comoving_distance_mpc(z, h0, omega_m, w0, steps=steps) * scale_factor_from_redshift(z)


def flat_wcdm_angular_diameter_distance_z1z2_mpc(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    chi_lens = flat_wcdm_comoving_distance_mpc(z_lens, h0, omega_m, w0, steps=steps)
    chi_source = flat_wcdm_comoving_distance_mpc(z_source, h0, omega_m, w0, steps=steps)
    distance = (chi_source - chi_lens) / (1.0 + jnp.asarray(z_source, dtype=jnp.float64))
    return jnp.where((chi_source > chi_lens) & (distance > 0.0), distance, jnp.nan)


def flat_wcdm_lensing_efficiency(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    chi_lens = flat_wcdm_comoving_distance_mpc(z_lens, h0, omega_m, w0, steps=steps)
    chi_source = flat_wcdm_comoving_distance_mpc(z_source, h0, omega_m, w0, steps=steps)
    efficiency = (chi_source - chi_lens) / chi_source
    return jnp.where((chi_source > 0.0) & (efficiency > 0.0), efficiency, jnp.nan)


def flat_wcdm_lens_geometry_factors(
    z_lens: float | jnp.ndarray,
    z_sources: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return lens-plane scale and per-source lensing factors from one distance table."""
    cosmo = flat_wcdm_cosmology(h0, omega_m, w0)
    z_lens_array = jnp.reshape(jnp.asarray(z_lens, dtype=jnp.float64), (1,))
    z_sources_array = jnp.ravel(jnp.asarray(z_sources, dtype=jnp.float64))
    z_all = jnp.concatenate([z_lens_array, z_sources_array], axis=0)
    chi_all = jc_background.radial_comoving_distance(
        cosmo,
        scale_factor_from_redshift(z_all),
        steps=int(steps),
    ) / cosmo.h
    chi_lens = chi_all[0]
    chi_sources = chi_all[1:]
    angular_diameter_lens_mpc = chi_lens * scale_factor_from_redshift(z_lens)
    kpc_per_arcsec = angular_diameter_lens_mpc * 1000.0 * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)
    efficiency = (chi_sources - chi_lens) / chi_sources
    efficiency = jnp.where((chi_sources > 0.0) & (efficiency > 0.0), efficiency, jnp.nan)
    return kpc_per_arcsec, efficiency, dpie_sigma0_factor_from_lensing_efficiency(efficiency)


def flat_wcdm_kpc_per_arcsec(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    angular_diameter_mpc = flat_wcdm_angular_diameter_distance_mpc(z, h0, omega_m, w0, steps=steps)
    return angular_diameter_mpc * 1000.0 * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)


def kpc_per_arcsec_from_config(
    z: float,
    cosmo_config: dict[str, Any] | None,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> float:
    return float(
        np.asarray(
            flat_wcdm_kpc_per_arcsec(
                float(z),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                steps=steps,
            )
        )
    )


def dpie_sigma0_factor_from_lensing_efficiency(efficiency: float | jnp.ndarray) -> jnp.ndarray:
    return (
        (1.0 / jnp.asarray(C_LIGHT_KM_S, dtype=jnp.float64)) ** 2
        * 3.0
        * jnp.pi
        * jnp.asarray(efficiency, dtype=jnp.float64)
        / jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)
    )


def dpie_sigma0_factor(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    return dpie_sigma0_factor_from_lensing_efficiency(
        flat_wcdm_lensing_efficiency(z_lens, z_source, h0, omega_m, w0, steps=steps)
    )


def dpie_sigma0_factor_from_config(
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> float:
    return float(
        np.asarray(
            dpie_sigma0_factor(
                float(z_lens),
                float(z_source),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                steps=steps,
            )
        )
    )


def dpie_sigma0_from_vel_disp(
    vel_disp: float,
    ra_arcsec: float,
    rs_arcsec: float,
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> float:
    if float(ra_arcsec) <= 0.0 or float(rs_arcsec) <= float(ra_arcsec):
        return float("nan")
    factor = dpie_sigma0_factor_from_config(z_lens, z_source, cosmo_config, steps=steps)
    sigma0 = float(vel_disp) ** 2 * factor / float(ra_arcsec)
    return float(sigma0)


def critical_surface_density_angle_msun_per_arcsec2(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> jnp.ndarray:
    dd = flat_wcdm_angular_diameter_distance_mpc(z_lens, h0, omega_m, w0, steps=steps)
    ds = flat_wcdm_angular_diameter_distance_mpc(z_source, h0, omega_m, w0, steps=steps)
    dds = flat_wcdm_angular_diameter_distance_z1z2_mpc(z_lens, z_source, h0, omega_m, w0, steps=steps)
    factor = (
        jnp.asarray(C_LIGHT_M_S, dtype=jnp.float64) ** 2
        / (4.0 * jnp.pi * jnp.asarray(G_SI, dtype=jnp.float64))
        * jnp.asarray(MPC_M / M_SUN_KG, dtype=jnp.float64)
    )
    sigma_crit = ds / (dd * dds) * factor * (dd * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)) ** 2
    return jnp.where((dd > 0.0) & (ds > 0.0) & (dds > 0.0), sigma_crit, jnp.nan)


def critical_surface_density_angle_from_config(
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    steps: int = DEFAULT_JAX_COSMO_DISTANCE_STEPS,
) -> float:
    return float(
        np.asarray(
            critical_surface_density_angle_msun_per_arcsec2(
                float(z_lens),
                float(z_source),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                steps=steps,
            )
        )
    )
