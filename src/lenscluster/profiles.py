"""Registry of the lens profiles the solver knows: Lenstool-style code, lenstronomy name, the fields a potential
definition must carry, which pair of polar shape fields becomes a Cartesian pair, and which fields are sampled in
log space.  Adding a profile means one entry here, a kernel in ``jax_lensing``, the packing in ``Evaluator.packed_state``
and the per-component branch of ``Evaluator.component_rows`` and ``exact_images.packed_to_kwargs_lens``."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from .model import (DP_IE_PROFILE, DPIE_PROFILE_NAME, MASS_SHEET_PROFILE, MASS_SHEET_PROFILE_NAME, NFW_PROFILE, NFW_PROFILE_NAME,
                    SHEAR_PROFILE, SHEAR_PROFILE_NAME)

G_KPC_KMS2_PER_MSUN = 4.30091e-6
DELTA_200 = 200.0


@dataclass(frozen=True)
class LensProfile:
    code: int
    name: str                                   # lenstronomy model name (exact-image solver)
    required_fields: tuple[str, ...]            # numeric fields of a potential definition, besides id, profil and z_lens
    polar_shape: tuple[str, str] | None         # (amplitude, angle) pair sampled together ...
    cartesian_shape: tuple[str, str] | None     # ... as this Cartesian pair
    log_positive_fields: frozenset[str]         # sampled as log of a positive quantity
    label: str


PROFILES: dict[int, LensProfile] = {
    DP_IE_PROFILE: LensProfile(
        code=DP_IE_PROFILE, name=DPIE_PROFILE_NAME, label="dPIE",
        required_fields=("x_centre", "y_centre", "ellipticite", "angle_pos", "core_radius_kpc", "cut_radius_kpc", "v_disp"),
        polar_shape=("ellipticite", "angle_pos"), cartesian_shape=("e1", "e2"),
        log_positive_fields=frozenset({"core_radius_kpc", "cut_radius_kpc", "v_disp"})),
    NFW_PROFILE: LensProfile(
        code=NFW_PROFILE, name=NFW_PROFILE_NAME, label="elliptical NFW",
        required_fields=("x_centre", "y_centre", "ellipticite", "angle_pos", "log10_m200", "scale_radius_kpc"),
        polar_shape=("ellipticite", "angle_pos"), cartesian_shape=("e1", "e2"),
        log_positive_fields=frozenset({"scale_radius_kpc"})),
    SHEAR_PROFILE: LensProfile(
        code=SHEAR_PROFILE, name=SHEAR_PROFILE_NAME, label="external shear",
        required_fields=("gamma", "angle_pos"),
        polar_shape=("gamma", "angle_pos"), cartesian_shape=("gamma1", "gamma2"),
        log_positive_fields=frozenset()),
    MASS_SHEET_PROFILE: LensProfile(
        code=MASS_SHEET_PROFILE, name=MASS_SHEET_PROFILE_NAME, label="external convergence",
        required_fields=("kappa",), polar_shape=None, cartesian_shape=None, log_positive_fields=frozenset()),
}
SUPPORTED_PROFILES = frozenset(PROFILES)
ELLIPTICAL_PROFILES = frozenset({DP_IE_PROFILE, NFW_PROFILE})


def profile(code: int) -> LensProfile:
    try:
        return PROFILES[int(code)]
    except KeyError:
        raise ValueError(f"Unsupported lens profile code {code}; known: {sorted(PROFILES)}") from None


# ----------------------------------------------------------------------------- NFW physics
def critical_density_msun_kpc3(z: float, cosmo_config: dict[str, Any] | None) -> float:
    """rho_crit(z) = 3 H(z)^2 / (8 pi G) for the flat (w)CDM cosmology of the run."""
    from .jax_cosmology import h0_from_config, ode0_from_config, om0_from_config, w0_from_config
    h0, om0, ode0, w0 = (f(cosmo_config) for f in (h0_from_config, om0_from_config, ode0_from_config, w0_from_config))
    e2 = om0 * (1.0 + z) ** 3 + ode0 * (1.0 + z) ** (3.0 * (1.0 + w0))
    hubble_kms_per_kpc = h0 / 1000.0 * math.sqrt(e2)
    return 3.0 * hubble_kms_per_kpc ** 2 / (8.0 * math.pi * G_KPC_KMS2_PER_MSUN)


def nfw_scale_density(log10_m200, scale_radius_kpc, rho_crit_msun_kpc3):
    """rho_s [Msun/kpc^3] and the concentration of an NFW halo of mass M200 and scale radius r_s (JAX-traceable)."""
    m200 = jnp.power(10.0, log10_m200)
    r200 = jnp.cbrt(3.0 * m200 / (4.0 * jnp.pi * DELTA_200 * rho_crit_msun_kpc3))
    c = r200 / scale_radius_kpc
    rho_s = DELTA_200 / 3.0 * rho_crit_msun_kpc3 * c ** 3 / (jnp.log1p(c) - c / (1.0 + c))
    return rho_s, c


# alpha_Rs[arcsec] = 4 rho_s r_s^2 (1 - ln 2) x (4 pi G / c^2) (D_ls / D_s) / arcsec; the dPIE sigma0 factor of this
# code is 3 pi (D_ls / D_s) / (c^2 arcsec), so the NFW factor is that times 4 G / 3.
NFW_FACTOR_OVER_DPIE_FACTOR = 4.0 * G_KPC_KMS2_PER_MSUN / 3.0


def nfw_alpha_rs_arcsec(rho_s_msun_kpc3, scale_radius_kpc, dpie_sigma0_factor):
    """Deflection at the scale radius in arcsec from rho_s, r_s and the per-source dPIE sigma0 factor."""
    return 4.0 * rho_s_msun_kpc3 * scale_radius_kpc ** 2 * (1.0 - jnp.log(2.0)) * NFW_FACTOR_OVER_DPIE_FACTOR * dpie_sigma0_factor


def nfw_lenstronomy_parameters(log10_m200: float, scale_radius_kpc: float, *, z_lens: float, z_source: float,
                               cosmo_config: dict[str, Any], kpc_per_arcsec: float) -> tuple[float, float]:
    """(Rs in arcsec, alpha_Rs in arcsec) for one halo: the quantities lenstronomy's NFW profiles take."""
    from .jax_cosmology import dpie_sigma0_factor_from_config
    rho_s, _ = nfw_scale_density(float(log10_m200), float(scale_radius_kpc), critical_density_msun_kpc3(z_lens, cosmo_config))
    factor = dpie_sigma0_factor_from_config(z_lens, z_source, cosmo_config)
    return float(scale_radius_kpc) / float(kpc_per_arcsec), float(nfw_alpha_rs_arcsec(rho_s, float(scale_radius_kpc), factor))


def lenstronomy_kwargs(packed, index: int) -> dict[str, float]:
    """Keyword arguments of component ``index`` for lenstronomy, from a packed state with scalar normalizations."""
    code = int(np.asarray(packed.profile_type)[index])
    f = lambda name: float(np.asarray(getattr(packed, name))[index])
    if code == DP_IE_PROFILE:
        return {"sigma0": f("sigma0"), "Ra": f("Ra"), "Rs": f("Rs"), "e1": f("e1"), "e2": f("e2"), "center_x": f("center_x"), "center_y": f("center_y")}
    if code == NFW_PROFILE:
        return {"Rs": f("nfw_rs"), "alpha_Rs": f("alpha_rs"), "e1": f("e1"), "e2": f("e2"), "center_x": f("center_x"), "center_y": f("center_y")}
    if code == SHEAR_PROFILE:
        return {"gamma1": f("gamma1"), "gamma2": f("gamma2")}
    if code == MASS_SHEET_PROFILE:
        return {"kappa": f("kappa"), "ra_0": 0.0, "dec_0": 0.0}
    raise ValueError(f"Unsupported profile type {code}.")
