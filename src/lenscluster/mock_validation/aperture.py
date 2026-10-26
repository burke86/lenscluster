"""Projected dPIE aperture masses about the cluster centre, per component group."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN

APERTURE_RADII_KPC = tuple(float(v) for v in np.geomspace(3.0, 500.0, 36))
COMPONENT_GROUPS = ("halo", "bcg", "members")


def dpie_sigma0_msun_per_kpc2(v_disp: np.ndarray, core_kpc: np.ndarray, cut_kpc: np.ndarray) -> np.ndarray:
    """Central surface density in the lenstool sigma convention, Sigma0 = (3/4) sigma^2 (s^2 - a^2) / (G a s^2)."""
    a, s, v = (np.asarray(x, dtype=float) for x in (core_kpc, cut_kpc, v_disp))
    return 0.75 * np.square(v) * (np.square(s) - np.square(a)) / (DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN * a * np.square(s))


def dpie_surface_density(distance_kpc: np.ndarray, v_disp: np.ndarray, core_kpc: np.ndarray, cut_kpc: np.ndarray) -> np.ndarray:
    """Circularized dPIE surface density at distance (..., C) from each of C component centres."""
    a, s = np.asarray(core_kpc, dtype=float), np.asarray(cut_kpc, dtype=float)
    d = np.asarray(distance_kpc, dtype=float)
    return dpie_sigma0_msun_per_kpc2(v_disp, a, s) * (a * s / (s - a)) * (1.0 / np.sqrt(a * a + d * d) - 1.0 / np.sqrt(s * s + d * d))


def aperture_profiles(radii_kpc: np.ndarray, groups: dict[str, tuple[np.ndarray, ...]], *, centre_kpc: tuple[float, float] = (0.0, 0.0),
                      integration_radii: int = 240, azimuth_samples: int = 64) -> dict[str, np.ndarray]:
    """M2D(<R) about ``centre_kpc`` for each group of (v_disp, core_kpc, cut_kpc, x_kpc, y_kpc), plus their total.

    Off-centre components are azimuthally averaged on a polar grid; ellipticity is neglected.
    """
    radii = np.asarray(radii_kpc, dtype=float)
    r_grid = np.geomspace(max(1.0e-2, radii.min() / 20.0), float(radii.max()), int(integration_radii))
    phi = np.linspace(0.0, 2.0 * np.pi, int(azimuth_samples), endpoint=False)
    px = float(centre_kpc[0]) + r_grid[:, None] * np.cos(phi)[None, :]
    py = float(centre_kpc[1]) + r_grid[:, None] * np.sin(phi)[None, :]
    profiles, total = {}, np.zeros_like(r_grid)
    for group, (v, core, cut, x, y) in groups.items():
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            profiles[group] = np.zeros_like(radii)
            continue
        distance = np.sqrt(np.square(px[:, :, None] - np.asarray(x, dtype=float)) + np.square(py[:, :, None] - np.asarray(y, dtype=float)))
        mean_sigma = dpie_surface_density(distance, v, core, cut).sum(axis=-1).mean(axis=1)
        total += mean_sigma
        profiles[group] = _cumulative(radii, r_grid, mean_sigma)
    profiles["total"] = _cumulative(radii, r_grid, total)
    return profiles


def _cumulative(radii: np.ndarray, r_grid: np.ndarray, mean_sigma: np.ndarray) -> np.ndarray:
    integrand = 2.0 * np.pi * r_grid * mean_sigma
    cumulative = np.pi * r_grid[0] ** 2 * mean_sigma[0] + np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_grid))])
    return np.interp(radii, r_grid, cumulative)


def truth_groups(truth: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    """Truth components of a realization grouped as the solver groups them (modelled members only)."""
    kpc = float(truth["kpc_per_arcsec"])

    def large(entry: dict[str, Any]) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray([float(entry[key]) * scale]) for key, scale in
                     (("v_disp", 1.0), ("core_radius_kpc", 1.0), ("cut_radius_kpc", 1.0), ("x_centre", kpc), ("y_centre", kpc)))

    members = [m for m in truth.get("members", []) if m.get("modelled", True)]
    return {"halo": large(truth["halo"]), "bcg": large(truth["bcg"]),
            "members": tuple(np.asarray([float(m[key]) * scale for m in members]) for key, scale in
                             (("v_disp", 1.0), ("core_radius_kpc", 1.0), ("cut_radius_kpc", 1.0), ("x_centre", kpc), ("y_centre", kpc)))}
