"""The residual projection of the GRF's low-order modes (gp_knee.project_low_order): it removes exactly the modes, stays
positive semidefinite, matches (I - P) K (I - P) built by hand, leaves the flat-prior marginal over translation and shear
unchanged, and the image solver's mean deflection follows the projected field."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import gp_knee
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from test_gp_knee import knee_config, two_knot_config

MODES = ("translation", "shear")


def _matrices(evaluator, theta):
    """Weighted mass-field covariance (2N, 2N) in slot order, the weights per row, and image positions, for one block."""
    physical = evaluator.physical(jnp.asarray(theta))
    k = np.asarray(evaluator.gp_kernel_blocks(physical, evaluator.error_scales(physical))["k_alpha"])[0]
    n = k.shape[0]
    flat = k.transpose(0, 2, 1, 3).reshape(2 * n, 2 * n)
    return flat, np.repeat(np.asarray(evaluator.gp["weights"])[0], 2), np.asarray(evaluator.gp["xy"])[0]


def _basis(xy, modes):
    centre = xy.mean(0)
    u = (xy - centre) / gp_knee.PROJECTION_SCALE_ARCSEC
    cols = []
    for mode in modes:
        if mode == "translation":
            cols += [np.stack([np.ones(len(u)), np.zeros(len(u))], 1), np.stack([np.zeros(len(u)), np.ones(len(u))], 1)]
        elif mode == "convergence":
            cols += [u.copy()]
        elif mode == "shear":
            cols += [np.stack([u[:, 0], -u[:, 1]], 1), np.stack([u[:, 1], u[:, 0]], 1)]
    return np.stack([c.ravel() for c in cols], 1)                       # (2N, n), rows (x_i, y_i)


@pytest.fixture
def evaluators(tmp_path):
    config, _ = two_knot_config(tmp_path)
    plain = build_state(knee_config(config, "all"))
    projected = build_state(knee_config(config, "all", gp_knee_project_modes=MODES))
    return Evaluator(plain, "gp_image_plane"), Evaluator(projected, "gp_image_plane"), np.asarray(plain.initial_theta, dtype=float)


def test_projection_removes_the_modes_and_matches_the_explicit_projector(evaluators):
    plain, projected, theta = evaluators
    k, w, xy = _matrices(plain, theta)
    kb, _, _ = _matrices(projected, theta)
    phi = _basis(xy, MODES)
    # explicit: W (I - P) W^-1 K W^-1 (I - P) W with P the least-squares projector onto the modes
    p = phi @ np.linalg.solve(phi.T @ phi, phi.T)
    kref = k / np.outer(w, w)
    expected = np.outer(w, w) * ((np.eye(len(w)) - p) @ kref @ (np.eye(len(w)) - p))
    np.testing.assert_allclose(kb, expected, atol=1e-12 * np.abs(k).max())
    np.testing.assert_allclose((phi / w[:, None]).T @ kb, 0.0, atol=1e-12 * np.abs(k).max())
    eigenvalues = np.linalg.eigvalsh(0.5 * (kb + kb.T))
    assert eigenvalues.min() > -1e-12 * eigenvalues.max()
    assert np.sum(eigenvalues < 1e-9 * eigenvalues.max()) >= phi.shape[1]


def test_flat_prior_marginal_over_translation_and_shear_is_unchanged(evaluators):
    """With flat priors on a translation and a shear (weighted by the redshift factors), removing those modes from the
    field changes nothing: the marginal likelihood of any data vector is the same."""
    plain, projected, theta = evaluators
    k, w, xy = _matrices(plain, theta)
    kb, _, _ = _matrices(projected, theta)
    design = w[:, None] * _basis(xy, MODES)
    rng = np.random.default_rng(3)
    noise = np.diag(rng.uniform(1e-4, 4e-4, len(w)))

    def marginal(covariance, data):
        inverse = np.linalg.inv(covariance)
        gram = design.T @ inverse @ design
        r = inverse @ data
        quadratic = data @ r - r @ design @ np.linalg.solve(gram, design.T @ r)
        return -0.5 * quadratic - 0.5 * np.linalg.slogdet(covariance)[1] - 0.5 * np.linalg.slogdet(gram)[1]
    for _ in range(3):
        data = np.linalg.cholesky(k + noise) @ rng.normal(size=len(w)) + design @ rng.normal(size=design.shape[1]) * 0.05
        np.testing.assert_allclose(marginal(kb + noise, data), marginal(k + noise, data), rtol=1e-9, atol=1e-8)


def test_mean_deflection_at_the_images_follows_the_projected_field(evaluators):
    for evaluator, theta in ((evaluators[0], evaluators[2]), (evaluators[1], evaluators[2])):
        k, _, _ = _matrices(evaluator, theta)
        v, _ = evaluator._knee_mean_inputs(jnp.asarray(theta))
        expected = (k @ np.asarray(v)[0].ravel()).reshape(-1, 2)
        correction = evaluator.gp_deflection_correction(theta)
        slots = np.asarray(evaluator.gp["slots"])[0]
        images = evaluator.state.images
        dx, dy, *_ = correction(np.asarray(images.x_obs)[slots], np.asarray(images.y_obs)[slots], np.asarray(images.family_index_per_image)[slots])
        np.testing.assert_allclose(np.stack([dx, dy], 1), expected, rtol=1e-6, atol=1e-9)


def test_projection_keeps_the_likelihood_finite_and_differentiable(evaluators):
    _, projected, theta = evaluators
    value, grad = jax.jit(jax.value_and_grad(projected.log_likelihood))(jnp.asarray(theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))


def test_projection_modes_are_validated(tmp_path):
    config, _ = two_knot_config(tmp_path)
    with pytest.raises(ValueError, match="gp_blocks"):
        knee_config(config, "system", gp_knee_project_modes=("shear",))
    with pytest.raises(ValueError, match="distinct names"):
        knee_config(config, "all", gp_knee_project_modes=("flexion",))
