"""Per-image Gaussian likelihood with a measurement-smoothed determinant.

All source and image coordinates, and positional scatters, use the same units.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc, log_ndtr, logsumexp

_LOG_2PI = float(np.log(2.0 * np.pi))
# The negative normal tail is an exponentially weighted, positive integral.
# Sixteen Gauss--Laguerre nodes give double precision for |mean / sigma| >= 8.
_LAGUERRE_NODES, _LAGUERRE_WEIGHTS = np.polynomial.laguerre.laggauss(16)


def _log_positive_normal_moment(z):
    """log(phi(z) + z Phi(z)), including the cancelling negative tail."""
    z = jnp.asarray(z)
    log_phi = -0.5 * z**2 - 0.5 * _LOG_2PI
    positive_z = jnp.where(z > 0, z, 1.0)
    positive = jnp.logaddexp(log_phi, jnp.log(positive_z) + log_ndtr(z))
    positive = jnp.where(z > 0, positive, log_phi)
    central_z = jnp.where(jnp.abs(z) <= 8.0, z, 0.0)
    central_log_phi = -0.5 * central_z**2 - 0.5 * _LOG_2PI
    central = central_log_phi + jnp.log1p(central_z * jnp.exp(log_ndtr(central_z) - central_log_phi))
    tail_a = jnp.maximum(-z, 8.0)
    nodes = jnp.asarray(_LAGUERRE_NODES, dtype=z.dtype)
    weights = jnp.asarray(_LAGUERRE_WEIGHTS, dtype=z.dtype)
    tail = log_phi - 2.0 * jnp.log(tail_a) + logsumexp(
        jnp.log(weights * nodes) - 0.5 * (nodes / tail_a[..., None])**2, axis=-1)
    return jnp.where(jnp.abs(z) <= 8.0, central, jnp.where(z > 0, positive, tail))


@jax.custom_jvp
def _log_absolute_normal(mean, variance):
    """Closed-form log E|X| for finite mean and strictly positive variance.

    For |z| <= 8, h/sd = 2 phi(z) + z erf(z/sqrt(2)).  In the tail, use
    log|mean| + log1p(2 phi(|z|)/|z| - erfc(|z|/sqrt(2))), which retains tiny
    log moments even when h rounds to one.
    """
    sd = jnp.sqrt(variance)
    central = jnp.abs(mean) <= 8.0 * sd
    bounded = jnp.abs(mean) <= 128.0 * sd
    z = jnp.where(bounded, mean, 0.0) / sd
    scaled_h = jnp.sqrt(2.0 / jnp.pi) * jnp.exp(-0.5*z*z) + z*erf(z/jnp.sqrt(2.0))
    center = 0.5*jnp.log(variance) + jnp.log(scaled_h)
    tail_z = jnp.where(central | ~bounded, 8.0, jnp.abs(z))
    correction = (2.0*jnp.exp(-0.5*tail_z*tail_z - 0.5*_LOG_2PI)/tail_z - erfc(tail_z/jnp.sqrt(2.0)))
    correction = jnp.where(bounded, jnp.maximum(correction, 0.0), 0.0)
    tail = jnp.log(jnp.where(central, 1.0, jnp.abs(mean))) + jnp.log1p(correction)
    return jnp.where(central, center, tail)


@_log_absolute_normal.defjvp
def _log_absolute_normal_jvp(primals, tangents):
    mean, variance = primals
    mean_dot, variance_dot = tangents
    value = _log_absolute_normal(mean, variance)
    sd = jnp.sqrt(variance)
    bounded = jnp.abs(mean) <= 128.0 * sd
    z = jnp.where(bounded, mean, 0.0) / sd
    signed_probability = jnp.where(bounded, erf(z/jnp.sqrt(2.0)), jnp.sign(mean))
    # Exact Gaussian identities: d log h/dm = erf(z/sqrt2)/h and d log h/dv = phi(z)/(sd h).
    d_mean = signed_probability * jnp.exp(-value)
    log_d_variance = -0.5*z*z - 0.5*_LOG_2PI - jnp.log(sd) - value
    d_variance = jnp.exp(jnp.where(bounded, log_d_variance, -jnp.inf))
    return value, d_mean*mean_dot + d_variance*variance_dot


def log_expected_absolute_normal(mean, variance, parity=None):
    """Log E[|X| 1(parity X > 0)] for X ~ N(mean, variance).

    ``parity=0`` or ``None`` sums both signs.  A zero variance uses the exact
    deterministic limit, including -inf for an impossible branch or X = 0.
    Negative variances and parity values other than 0, -1, +1 return NaN.
    """
    dtype = jnp.result_type(mean, variance, 1.0)
    mean, variance = jnp.broadcast_arrays(jnp.asarray(mean, dtype=dtype), jnp.asarray(variance, dtype=dtype))
    finite = jnp.isfinite(mean) & jnp.isfinite(variance)
    positive_variance = (variance > 0) & finite
    safe_mean = jnp.where(finite, mean, 0.0)
    safe_variance = jnp.where(positive_variance, variance, 1.0)
    if parity is None or (isinstance(parity, (int, np.integer)) and parity == 0):
        stochastic = _log_absolute_normal(safe_mean, safe_variance)
        deterministic_h = jnp.abs(safe_mean)
        valid = finite & (variance >= 0)
    else:
        parity = jnp.broadcast_to(jnp.asarray(parity), mean.shape)
        unknown = parity == 0
        two_sided = _log_absolute_normal(jnp.where(unknown, safe_mean, 0.0), jnp.where(unknown, safe_variance, 1.0))
        sd = jnp.sqrt(jnp.where(unknown, 1.0, safe_variance))
        z = jnp.where(unknown, 0.0, safe_mean) / sd
        log_plus = jnp.log(sd) + _log_positive_normal_moment(z)
        log_minus = jnp.log(sd) + _log_positive_normal_moment(-z)
        stochastic = jnp.where(unknown, two_sided, jnp.where(parity > 0, log_plus, log_minus))
        deterministic_h = jnp.where(unknown, jnp.abs(safe_mean), jnp.maximum(parity*safe_mean, 0.0))
        valid = finite & (variance >= 0) & (unknown | (jnp.abs(parity) == 1))
    deterministic = jnp.where(deterministic_h > 0, jnp.log(jnp.where(deterministic_h > 0, deterministic_h, 1.0)), -jnp.inf)
    return jnp.where(valid, jnp.where(positive_variance, stochastic, deterministic), jnp.nan)


def expected_absolute_normal(mean, variance, parity=None):
    return jnp.exp(log_expected_absolute_normal(mean, variance, parity))


def source_covariance(jacobian, sigma_meas, sigma_def, sigma_sel=0.0, lambda_sel=1.0):
    """Per-image source-plane covariance C = sigma_meas^2 A A^T + sigma_def^2 I + sigma_sel^2 R, its Cholesky factor and
    log det.  R = (I + A A^T / lambda_sel^2)^-1 is the stretching weight of the selection term: the selection variance
    is halved along a principal direction whose absolute Jacobian eigenvalue equals lambda_sel; sigma_sel = 0 drops the
    term."""
    a = jnp.asarray(jacobian)
    n = a.shape[0]
    eye = jnp.eye(2, dtype=a.dtype)
    meas = jnp.broadcast_to(jnp.asarray(sigma_meas), (n,))
    deflection = jnp.broadcast_to(jnp.asarray(sigma_def), (n,))
    sel = jnp.broadcast_to(jnp.asarray(sigma_sel), (n,))
    lam = jnp.broadcast_to(jnp.asarray(lambda_sel), (n,))
    aat = a @ jnp.swapaxes(a, -1, -2)
    measurement = meas[:, None, None]**2 * aat
    safe_lam2 = jnp.where(lam > 0, lam**2, 1.0)
    weight = jnp.linalg.solve(eye + aat / safe_lam2[:, None, None], jnp.broadcast_to(eye, aat.shape))
    selection = jnp.where((sel > 0)[:, None, None], sel[:, None, None]**2 * weight, 0.0)
    covariance = deflection[:, None, None]**2 * eye + measurement + selection
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    chol = jnp.linalg.cholesky(covariance)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    valid = (jnp.all(jnp.isfinite(covariance), axis=(-2, -1)) & jnp.all(jnp.isfinite(chol), axis=(-2, -1))
             & (meas >= 0) & (deflection >= 0) & (sel >= 0) & ((sel == 0) | (lam > 0)) & jnp.all(jnp.isfinite(a), axis=(-2, -1)))
    return dict(jacobian=a, sigma_meas=meas, sigma_def=deflection, sigma_sel=sel, lambda_sel=lam, measurement=measurement,
                covariance=covariance, cholesky=chol, logdet=logdet, valid=valid)


def statistics(backprojected, source, jacobian, determinant_gradient, sigma_meas, sigma_def, sigma_sel=0.0, lambda_sel=1.0, *, parity=None):
    """Normalized local image-plane log-likelihood per image and its diagnostics.

    ``determinant_gradient`` is grad_theta(det A) with shape (N, 2).
    """
    result = source_covariance(jacobian, sigma_meas, sigma_def, sigma_sel, lambda_sel)
    a, meas = result["jacobian"], result["sigma_meas"]
    covariance, chol = result["covariance"], result["cholesky"]
    residual = jnp.asarray(backprojected) - jnp.asarray(source)
    g = jnp.asarray(determinant_gradient)
    determinant = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    # Measurement error e: F = A e + eta, X = D - g.e, so Cov(F, X) = -sigma_meas^2 A g; the deflection and selection
    # scatters enter F only.
    cross_covariance = -meas[:, None]**2 * jnp.einsum("nij,nj->ni", a, g)
    precision_residual = jnp.linalg.solve(covariance, residual[..., None])[..., 0]
    precision_cross = jnp.linalg.solve(covariance, cross_covariance[..., None])[..., 0]
    determinant_mean = determinant + jnp.sum(cross_covariance * precision_residual, axis=-1)
    # Residual latent coefficients give the Schur complement without subtraction: with p = C^-1 c,
    # sigma_meas^2 |g|^2 - c^T C^-1 c = sigma_meas^2 |g + A^T p|^2 + p^T (C - sigma_meas^2 A A^T) p.
    measurement_remainder = g + jnp.einsum("nji,nj->ni", a, precision_cross)
    rest = covariance - result["measurement"]
    determinant_variance = (meas**2 * jnp.sum(measurement_remainder**2, axis=-1)
                            + jnp.einsum("ni,nij,nj->n", precision_cross, rest, precision_cross))
    log_h = log_expected_absolute_normal(determinant_mean, determinant_variance, parity)
    whitened_residual = jnp.linalg.solve(chol, residual[..., None])[..., 0]
    quadratic = jnp.sum(whitened_residual**2, axis=-1)
    gaussian_loglike = -0.5 * (quadratic + result["logdet"] + 2.0 * _LOG_2PI)
    loglike = gaussian_loglike + log_h
    degenerate = (determinant_variance == 0) & (determinant_mean == 0)
    valid = (result["valid"] & jnp.all(jnp.isfinite(residual), axis=-1) & jnp.isfinite(determinant_mean)
             & jnp.isfinite(determinant_variance) & ~jnp.isnan(log_h) & ~degenerate)
    result.update(residual=residual, determinant=determinant, determinant_gradient=g,
                  cross_covariance=cross_covariance, determinant_mean=determinant_mean,
                  determinant_variance=determinant_variance, log_h=log_h, quadratic=quadratic,
                  gaussian_loglike=gaussian_loglike, loglike=loglike, degenerate=degenerate, valid=valid)
    return result


def source_whitening(backprojected, covariance, family_index, n_families):
    """Gaussian positional centroid and scale per family for a bijective source transport."""
    backprojected, covariance = jnp.asarray(backprojected), jnp.asarray(covariance)
    family_index = jnp.asarray(family_index, dtype=int)
    eye = jnp.eye(2, dtype=covariance.dtype)
    precision = jnp.linalg.solve(covariance, jnp.broadcast_to(eye, covariance.shape))
    family_precision = jnp.zeros((n_families, 2, 2), dtype=covariance.dtype).at[family_index].add(precision)
    weighted_beta = jnp.einsum("nij,nj->ni", precision, backprojected)
    family_weighted_beta = jnp.zeros((n_families, 2), dtype=backprojected.dtype).at[family_index].add(weighted_beta)
    source_covariance_matrix = jnp.linalg.solve(family_precision, jnp.broadcast_to(eye, family_precision.shape))
    source_covariance_matrix = 0.5 * (source_covariance_matrix + jnp.swapaxes(source_covariance_matrix, -1, -2))
    chol = jnp.linalg.cholesky(source_covariance_matrix)
    centroid = jnp.linalg.solve(family_precision, family_weighted_beta[..., None])[..., 0]
    logdet_precision = -2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    valid = jnp.all(jnp.isfinite(chol), axis=(-2, -1)) & jnp.all(jnp.isfinite(centroid), axis=-1)
    return dict(centroid=centroid, covariance=source_covariance_matrix, cholesky=chol,
                precision=family_precision, logdet_precision=logdet_precision, valid=valid)


def transport_sources(offsets, whitening, prior_mean, prior_sigma):
    """Transport standard-normal sites to physical sources and correct to the physical Gaussian prior.

    Add ``log_correction`` to a target whose source sites already contribute
    independent standard-normal log densities.
    """
    offsets = jnp.asarray(offsets)
    source = whitening["centroid"] + jnp.einsum("fij,fj->fi", whitening["cholesky"], offsets)
    sigma = jnp.asarray(prior_sigma)
    if sigma.ndim == 1:                       # one width per family, shared by both coordinates
        sigma = sigma[:, None]
    physical_log_prior = -0.5 * jnp.sum(((source - jnp.asarray(prior_mean)) / sigma)**2 + 2.0 * jnp.log(sigma) + _LOG_2PI, axis=-1)
    base_log_prior = -0.5 * jnp.sum(offsets**2 + _LOG_2PI, axis=-1)
    log_jacobian = -0.5 * whitening["logdet_precision"]
    correction = physical_log_prior + log_jacobian - base_log_prior
    valid = whitening["valid"] & jnp.all(jnp.isfinite(source), axis=-1)
    return dict(source=source, log_correction_per_family=correction, log_correction=jnp.sum(correction),
                physical_log_prior=physical_log_prior, log_jacobian=log_jacobian, valid=valid)


def inverse_source_transport(source, whitening):
    """Standard-coordinate offsets corresponding to physical sources."""
    return jnp.linalg.solve(whitening["cholesky"], (jnp.asarray(source) - whitening["centroid"])[..., None])[..., 0]
