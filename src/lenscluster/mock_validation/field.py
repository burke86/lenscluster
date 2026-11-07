"""Injected correlated errors of the mock: the mass-field error of unmodelled mass and the image-plane displacement.

Mass field.  A realization of the solver's knee field (gp_knee.knee_tables): a curl-free deflection on the lens plane with
convergence spectrum P_kappa(k) = 1 / (1 + (k / k_t)^gamma) between 2 pi / scale_max and 2 pi / scale_min, k_t = 2 pi /
knee, rms deflection sigma_alpha per component for a source at the reference redshift.  It is a sum of N plane waves in
the potential,

    psi(theta) = sum_n (a / k_n) cos(k_n . theta + phi_n),   a = 2 sigma_alpha / sqrt(N),

with |k_n| drawn in ln k with weight P_kappa dln k (the weights of knee_tables), uniform directions and phases: each mode
gives alpha = -a sin(.) k_hat (variance a^2 / 4 per component) and Hessian -a |k| k_hat k_hat^T cos(.), exact and smooth
everywhere, so caustics and highly magnified images see the field.  A source at z_s sees w alpha with w the ratio of
D_ls / D_s to its value at the reference redshift.

Image-plane displacement.  One Gaussian draw over all observed images, per axis, with covariance sigma^2 exp(-r^2 / 2 ell^2)
(the solver's image-plane GP kernel), added to the observed positions.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

FIELD_CHUNK = 4096


def _log_uniform(rng: np.random.Generator, bounds) -> float:
    lo, hi = (float(b) for b in bounds)
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def draw_mass_field(rng: np.random.Generator, config) -> dict[str, Any] | None:
    """The field's hyperparameters (sigma_alpha, knee log-uniform, gamma uniform in their ranges) and its modes, or None."""
    if config.field_sigma_alpha_range_arcsec is None:
        return None
    sigma = _log_uniform(rng, config.field_sigma_alpha_range_arcsec)
    knee = _log_uniform(rng, config.field_knee_range_arcsec)
    gamma = float(rng.uniform(*config.field_gamma_range))
    scale_min, scale_max = (float(s) for s in config.field_scales_arcsec)
    n = int(config.field_n_modes)
    ln_k = np.linspace(math.log(2.0 * math.pi / scale_max), math.log(2.0 * math.pi / scale_min), 8192)
    weight = 1.0 / (1.0 + (np.exp(ln_k) / (2.0 * math.pi / knee)) ** gamma)              # P_kappa per dln k
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(ln_k))])
    k = np.exp(np.interp(rng.uniform(0.0, cdf[-1], n), cdf, ln_k))
    direction, phase = rng.uniform(0.0, 2.0 * math.pi, n), rng.uniform(0.0, 2.0 * math.pi, n)
    return {"sigma_alpha_arcsec": sigma, "knee_arcsec": knee, "gamma": gamma, "scales_arcsec": [scale_min, scale_max],
            "reference_redshift": float(config.field_reference_redshift), "n_modes": n, "amplitude": 2.0 * sigma / math.sqrt(n),
            "kx": (k * np.cos(direction)).tolist(), "ky": (k * np.sin(direction)).tolist(), "phase": phase.tolist()}


def field_arrays(field: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    return (jnp.asarray(field["kx"], dtype=jnp.float64), jnp.asarray(field["ky"], dtype=jnp.float64),
            jnp.asarray(field["phase"], dtype=jnp.float64), float(field["amplitude"]))


def field_alpha_hessian(x, y, arrays, weight):
    """Deflection and Hessian (ax, ay, hxx, hxy, hyx, hyy) of the field times ``weight`` at the points (traceable)."""
    kx, ky, phase, amplitude = arrays
    argument = jnp.asarray(x)[..., None] * kx + jnp.asarray(y)[..., None] * ky + phase
    s, c = jnp.sin(argument), jnp.cos(argument)
    k = jnp.hypot(kx, ky)
    a = weight * amplitude
    ax, ay = -a * jnp.sum(s * (kx / k), -1), -a * jnp.sum(s * (ky / k), -1)
    hxx, hxy, hyy = (-a * jnp.sum(c * v, -1) for v in (kx * kx / k, kx * ky / k, ky * ky / k))
    return ax, ay, hxx, hxy, hxy, hyy


_field_chunk = jax.jit(field_alpha_hessian)


def field_alpha_hessian_tiled(x: np.ndarray, y: np.ndarray, arrays, weight: float) -> tuple[np.ndarray, ...]:
    """The same over many points, in chunks of FIELD_CHUNK (points x modes stays small), as numpy arrays."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    out = [np.empty(x.size) for _ in range(6)]
    for start in range(0, x.size, FIELD_CHUNK):
        stop = min(start + FIELD_CHUNK, x.size)
        xs, ys = np.pad(x[start:stop], (0, FIELD_CHUNK - (stop - start)), mode="edge"), np.pad(y[start:stop], (0, FIELD_CHUNK - (stop - start)), mode="edge")
        for target, value in zip(out, _field_chunk(jnp.asarray(xs), jnp.asarray(ys), arrays, jnp.asarray(float(weight)))):
            target[start:stop] = np.asarray(value)[:stop - start]
    return tuple(out)


def draw_image_plane_error(rng: np.random.Generator, config) -> dict[str, float] | None:
    """The image-plane GP's sigma and ell (log-uniform in their ranges), or None."""
    if config.image_plane_sigma_range_arcsec is None:
        return None
    return {"sigma_arcsec": _log_uniform(rng, config.image_plane_sigma_range_arcsec),
            "ell_arcsec": _log_uniform(rng, config.image_plane_ell_range_arcsec)}


def image_plane_displacements(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, error: dict[str, float]) -> np.ndarray:
    """One correlated draw (N, 2) at the points: per axis N(0, sigma^2 exp(-r^2 / 2 ell^2))."""
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    r2 = np.sum(np.square(xy[:, None, :] - xy[None, :, :]), axis=-1)
    covariance = float(error["sigma_arcsec"]) ** 2 * np.exp(-r2 / (2.0 * float(error["ell_arcsec"]) ** 2))
    factor = np.linalg.cholesky(covariance + 1.0e-10 * float(error["sigma_arcsec"]) ** 2 * np.eye(len(xy)))
    return factor @ rng.standard_normal((len(xy), 2))
