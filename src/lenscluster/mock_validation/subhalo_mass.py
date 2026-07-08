from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..utils import log10_dpie_mass_msun


def positive_finite_values(values: list[Any] | np.ndarray) -> np.ndarray:
    finite_values: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            finite_values.append(number)
    return np.asarray(finite_values, dtype=float)


def dpie_truth_scaling_values(truth: dict[str, Any]) -> tuple[float, float, float, float] | None:
    parameter_truth = truth.get("parameter_truth", {})
    if not isinstance(parameter_truth, dict):
        return None
    try:
        sigma_ref = float(parameter_truth["potfile.sigma"])
        cut_ref_kpc = float(parameter_truth["potfile.cutkpc"])
        alpha_sigma = float(parameter_truth["potfile.alpha_sigma"])
        beta_radius = float(parameter_truth["potfile.beta_radius"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (sigma_ref, cut_ref_kpc, alpha_sigma, beta_radius)
    if not all(np.isfinite(values)) or sigma_ref <= 0.0 or cut_ref_kpc <= 0.0:
        return None
    return values


def dpie_truth_subhalo_masses_from_scaling(truth: dict[str, Any]) -> np.ndarray:
    """Recompute physical dPIE member masses from current potfile truth."""
    scaling = dpie_truth_scaling_values(truth)
    if scaling is None:
        return np.empty((0,), dtype=float)
    sigma_ref, cut_ref_kpc, alpha_sigma, beta_radius = scaling
    subhalos = truth.get("subhalos", [])
    masses: list[float] = []
    for record in subhalos if isinstance(subhalos, list) else []:
        if not isinstance(record, dict):
            continue
        try:
            luminosity = float(record.get("luminosity_ratio"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(luminosity) and luminosity > 0.0):
            continue
        v_disp = (
            sigma_ref
            * luminosity**alpha_sigma
            * math.exp(float(record.get("sigma_log_offset", 0.0)))
        )
        cut_kpc = (
            cut_ref_kpc
            * luminosity**beta_radius
            * math.exp(float(record.get("cut_log_offset", 0.0)))
        )
        masses.append(float(10.0 ** log10_dpie_mass_msun(v_disp, cut_kpc)))
    return positive_finite_values(masses)


def dpie_mass_anchor_from_truth(truth: dict[str, Any]) -> float | None:
    """Physical dPIE mass of an L=1, no-scatter member."""
    scaling = dpie_truth_scaling_values(truth)
    if scaling is None:
        return None
    sigma_ref, cut_ref_kpc, _alpha_sigma, _beta_radius = scaling
    return float(10.0 ** log10_dpie_mass_msun(sigma_ref, cut_ref_kpc))
