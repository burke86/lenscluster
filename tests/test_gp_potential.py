"""The potential (curl-free) deflection kernel under gp_image_plane: kernel blocks against the autodiff Hessian of the scalar
kernel, the white limit against the smoothed likelihood, redshift weights, the critical curve with no floor, finite
gradients for both families, the fold-pair check and the speed against the image-plane kernel."""
from __future__ import annotations

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import synthetic_cluster_config
from lenscluster import gp_image_plane as gp, smoothed_likelihood as sl
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from test_gp_image_plane import fold_config, set_scale

FAMILIES = ("gaussian", "matern52")


def potential_config(config, family="gaussian", **likelihood):
    return replace(config, likelihood=replace(config.likelihood, likelihood_mode="gp_image_plane", gp_kernel="potential",
                                              gp_kernel_family=family, **likelihood)).validate()


def scalar_kernel(family, sigma_alpha, ell):
    """k_psi(theta, theta') parameterized by sigma_alpha, so that -d_a d_b k_psi has sigma_alpha^2 I on the diagonal."""
    if family == "gaussian":
        return lambda t, tp: sigma_alpha ** 2 * ell ** 2 * jnp.exp(-jnp.sum((t - tp) ** 2) / (2 * ell ** 2))
    a = jnp.sqrt(5.0) / ell
    sigma_psi2 = 3 * ell ** 2 * sigma_alpha ** 2 / 5

    def k(t, tp):
        r = jnp.sqrt(jnp.sum((t - tp) ** 2))
        return sigma_psi2 * (1 + a * r + a ** 2 * r ** 2 / 3) * jnp.exp(-a * r)
    return k


# ----------------------------------------------------------------------------- 1. the kernel blocks
@pytest.mark.parametrize("family", FAMILIES)
def test_deflection_kernel_is_the_hessian_of_the_scalar_kernel(family):
    sigma_alpha, ell = 0.7, 2.3
    k = scalar_kernel(family, sigma_alpha, ell)
    hessian = jax.jacfwd(jax.jacrev(k, 0), 1)                              # d/dtheta_a d/dtheta'_b k
    rng = np.random.default_rng(0)
    points = rng.normal(size=(1, 5, 2)) * 3.0
    separations = points[:, :, None, :] - points[:, None, :, :]
    d2 = np.sum(separations ** 2, -1)
    outer = separations[..., :, None] * separations[..., None, :]
    mask = np.ones((1, 5), dtype=bool)
    blocks = np.asarray(gp.deflection_kernel(d2, outer, mask, sigma_alpha ** 2, ell, family))
    for i in range(5):
        for j in range(5):
            expected = np.asarray(hessian(jnp.asarray(points[0, i]), jnp.asarray(points[0, j]))) if i != j else sigma_alpha ** 2 * np.eye(2)
            np.testing.assert_allclose(blocks[0, i, j], expected, rtol=1e-10, atol=1e-12)
    # symmetric, positive definite as a whole and padding-neutral
    full = blocks.transpose(0, 1, 3, 2, 4).reshape(10, 10)
    np.testing.assert_allclose(full, full.T, atol=1e-14)
    assert np.linalg.eigvalsh(full).min() > 0
    mask[0, -1] = False
    padded = np.asarray(gp.deflection_kernel(d2, outer, mask, sigma_alpha ** 2, ell, family))
    assert np.all(padded[0, -1] == 0) and np.all(padded[0, :, -1] == 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_kernel_profile_matches_the_blocks(family):
    profile = gp.kernel_profile(family)
    ell, r = 1.7, np.array([0.3, 1.0, 1.9])
    d = np.stack([r * ell, np.zeros_like(r)], -1)                           # separations along x
    d2, outer = np.sum(d ** 2, -1)[None, None], (d[..., :, None] * d[..., None, :])[None, None]
    blocks = np.asarray(gp.deflection_kernel(d2, outer, np.ones((1, 1), bool), 1.0, ell, family))[0, 0]
    np.testing.assert_allclose(blocks[:, 0, 0], profile["longitudinal"](r), rtol=1e-12)
    np.testing.assert_allclose(blocks[:, 1, 1], profile["transverse"](r), rtol=1e-12)
    # the longitudinal correlation changes sign: zero at ell (gaussian) or 0.72 ell (matern52)
    zero = {"gaussian": 1.0, "matern52": (1 + np.sqrt(5)) / 2 / np.sqrt(5)}[family]
    assert abs(profile["longitudinal"](zero)) < 1e-12 and profile["longitudinal"](1.5 * zero) < 0 < profile["transverse"](1.5 * zero)


# ----------------------------------------------------------------------------- 2. the white limit is the smoothed model
@pytest.mark.parametrize("family", FAMILIES)
def test_white_limit_matches_the_smoothed_likelihood(synthetic_config, family):
    """ell_S -> 0 at fixed sigma_S, no large scale: C is block diagonal with sigma_def^2 = sigma_S^2 + sigma_fl^2 per image,
    which is the smoothed model with sigma_meas^2 = sigma_inst^2 + sigma_extra^2."""
    sigma_small, ell_small, sigma_extra, floor = 0.03, 1e-4, 0.05, 0.002
    config = potential_config(synthetic_config, family, gp_sigma_large_prior_scale_arcsec=None, gp_floor_arcsec=floor)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    assert evaluator.gp_potential and state.gp_sigma_large_param_index == -1
    theta = set_scale(state, state.initial_theta, "images.gp_ell_small", ell_small)
    theta = set_scale(state, theta, "images.gp_u_small", sigma_small / ell_small)
    theta = set_scale(state, theta, "images.sigma_extra", sigma_extra)
    stats = evaluator.statistics(theta)
    images, fields = state.images, stats["fields"]
    w = np.asarray(evaluator.gp_weights_per_image)
    sigma_meas = np.sqrt(np.asarray(images.sigma_inst_per_image) ** 2 + sigma_extra ** 2)
    sigma_def = np.sqrt(w ** 2 * sigma_small ** 2 + floor ** 2)
    old = sl.statistics(fields["backprojected"], stats["source_by_family"][images.family_index_per_image], fields["jacobian"],
                        fields["determinant_gradient"], sigma_meas, sigma_def, 0.0, 1.0, parity=images.parity)
    family_index = np.asarray(images.family_index_per_image)
    for f in range(images.n_families):
        np.testing.assert_allclose(float(stats["family_gaussian_loglike"][f]), float(np.sum(np.asarray(old["gaussian_loglike"])[family_index == f])), rtol=1e-9)
    whitening = sl.source_whitening(fields["backprojected"], old["covariance"], images.family_index_per_image, images.n_families)
    np.testing.assert_allclose(np.asarray(stats["whitening"]["centroid"]), np.asarray(whitening["centroid"]), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(stats["log_h"]), np.asarray(old["log_h"]), rtol=1e-9)
    total = float(np.sum(np.asarray(old["gaussian_loglike"]) + np.asarray(old["log_h"]))) + float(stats["source_correction"])
    np.testing.assert_allclose(float(evaluator.loglike(theta)), total, rtol=1e-9)


# ----------------------------------------------------------------------------- 3. redshift weights and the diagonal block
def test_redshift_weights_scale_each_family(synthetic_config):
    config = potential_config(synthetic_config, gp_reference_redshift=2.0)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    z = np.asarray(state.z_source_values)[np.asarray(state.images.z_index_per_image)]
    w = np.asarray(evaluator.gp_weights_per_image)
    np.testing.assert_allclose(w[z == 2.0], 1.0, rtol=1e-12)                 # the reference redshift has weight one
    assert np.all(w[z > 2.0] > 1.0)                                         # farther sources are lensed more
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    scales = evaluator.error_scales(physical)
    blocks = np.asarray(evaluator.gp_kernel_blocks(physical)["k_alpha"])
    slots, mask = np.asarray(evaluator.gp["slots"]), np.asarray(evaluator.gp["mask"])
    total = float(scales["gp_sigma_small"]) ** 2 + float(scales["gp_sigma_large"]) ** 2
    for f in range(slots.shape[0]):
        for m in np.flatnonzero(mask[f]):
            i = slots[f, m]
            np.testing.assert_allclose(blocks[f, m, m], w[i] ** 2 * total * np.eye(2), rtol=1e-12)
            np.testing.assert_allclose(float(np.asarray(scales["sigma_def"])[i]) ** 2, w[i] ** 2 * total + evaluator.gp_floor ** 2, rtol=1e-12)
    # the family covariance of the evaluator is the diagonal blocks plus the measurement term plus the floor
    stats = evaluator.statistics(jnp.asarray(state.initial_theta))
    per_image = evaluator.per_image_covariance(physical, stats["fields"]["jacobian"])["covariance"]
    c = np.asarray(stats["family_covariance"])
    for f in range(slots.shape[0]):
        for m in np.flatnonzero(mask[f]):
            np.testing.assert_allclose(c[f, 2 * m:2 * m + 2, 2 * m:2 * m + 2], np.asarray(per_image)[slots[f, m]], rtol=1e-10, atol=1e-14)


# ----------------------------------------------------------------------------- 4. the critical curve without a floor
@pytest.mark.parametrize("family", FAMILIES)
def test_fold_pair_on_the_critical_curve_without_floor(tmp_path, family):
    """A pair 0.1 mas deep (|lambda_t| ~ 1e-3) with sigma_fl = 0: the measurement term has no tangential rank there and the
    image-plane kernel would need the floor; the potential kernel keeps C positive definite and the gradient finite."""
    config, _ = fold_config(tmp_path, depth=1e-4)
    config = potential_config(config, family, gp_floor_arcsec=0.0)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    theta = jnp.asarray(state.initial_theta)
    jac = np.asarray(evaluator.fields(evaluator.physical(theta))["jacobian"])[:2]
    assert max(abs(gp.tangential_eigenpair(a)[0]) for a in jac) < 5e-3
    stats = evaluator.statistics(theta)
    assert bool(jnp.all(jnp.isfinite(stats["family_cholesky"])))
    value, grad = evaluator.value_and_grad(theta)
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))


# ----------------------------------------------------------------------------- 5. gradients of the hyperparameters
@pytest.mark.parametrize("family", FAMILIES)
def test_hyperparameter_gradients_match_finite_differences(tmp_path, family):
    """Both families, near a fold (so the kernel sees separations from 0 to the family size) and with single-image
    families, where the Matern square root sits exactly at d = 0."""
    config, _ = fold_config(tmp_path, depth=0.01)
    state = build_state(potential_config(config, family))
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    theta = np.asarray(state.initial_theta, dtype=float)
    theta[names.index("source.1.beta_x")] = 0.3
    value, grad = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))
    for name, step in (("images.gp_u_small", 1e-5), ("images.gp_ell_small", 1e-5), ("images.gp_sigma_large", 1e-6),
                       ("images.sigma_extra", 1e-5), ("halo.v_disp", 1e-6), ("source.1.beta_x", 1e-5)):
        i = names.index(name)
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        fd = (float(evaluator.loglike(jnp.asarray(plus))) - float(evaluator.loglike(jnp.asarray(minus)))) / (2 * step)
        np.testing.assert_allclose(float(grad[i]), fd, rtol=2e-5, atol=1e-6, err_msg=name)


# ----------------------------------------------------------------------------- 6. the fold-pair check
def test_shallow_pair_check_uses_the_source_plane_errors(tmp_path):
    config, fold = fold_config(tmp_path, depth=0.004)
    state = build_state(potential_config(config))
    evaluator = Evaluator(state, "gp_image_plane")
    rows = gp.shallow_fold_pairs(state, evaluator, state.initial_theta)
    assert len(rows) == 1 and sorted(rows[0]["labels"]) == ["1.a", "1.b"]
    row = rows[0]
    scales = evaluator.error_scales(evaluator.physical(jnp.asarray(state.initial_theta)))
    w = float(np.asarray(evaluator.gp_weights_per_image)[0])
    total = w ** 2 * (float(scales["gp_sigma_small"]) ** 2 + float(scales["gp_sigma_large"]) ** 2)
    # the common mode of a close pair is nearly the full deflection error
    assert 0.7 * np.sqrt(total) < row["depth_error_arcsec"] < 1.05 * np.sqrt(total + evaluator.gp_floor ** 2)
    # the difference error is e_t^T (K_ii + K_jj - K_ij - K_ji + sigma_meas^2 (A_i A_i^T + A_j A_j^T)) e_t plus the floor
    xy = np.column_stack([state.images.x_obs, state.images.y_obs])
    labels = list(state.images.image_labels)
    i, j = labels.index("1.a"), labels.index("1.b")
    d = xy[i] - xy[j]
    e_t = np.asarray(row["e_t"])
    jac = np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(state.initial_theta)), force_exact=True, derivative=False)["jacobian"])
    sigma_meas2 = np.asarray(scales["sigma_meas"]) ** 2
    expected = 2 * evaluator.gp_floor ** 2 + float(e_t @ (sigma_meas2[i] * jac[i] @ jac[i].T + sigma_meas2[j] * jac[j] @ jac[j].T) @ e_t)
    for sigma, ell in ((float(scales["gp_sigma_small"]), float(scales["gp_ell_small"])), (float(scales["gp_sigma_large"]), evaluator.gp_ell_large)):
        d2 = np.array([[[0.0, d @ d], [d @ d, 0.0]]])
        outer = np.array([[[np.zeros((2, 2)), np.outer(d, d)], [np.outer(d, d), np.zeros((2, 2))]]])
        pair = np.asarray(gp.deflection_kernel(d2, outer, np.ones((1, 2), bool), sigma ** 2, ell, "gaussian"))[0]
        expected += w ** 2 * float(e_t @ (pair[0, 0] + pair[1, 1] - pair[0, 1] - pair[1, 0]) @ e_t)
    np.testing.assert_allclose(row["sigma_delta_arcsec"], np.sqrt(expected), rtol=2e-3)
    np.testing.assert_allclose(row["depth_arcsec"], 0.004, rtol=0.05)
    report = gp.shallow_fold_pair_report(rows, evaluator.gp_floor, header="potential kernel", potential=True)
    assert "potential kernel" in report and "1.a/1.b" in report
    assert "gp_floor_arcsec" not in report and "common-mode error for 1.a/1.b" in report   # 4 mas depth against a ~30 mas error


# ----------------------------------------------------------------------------- 7. speed
def test_potential_kernel_cost_is_comparable(tmp_path):
    base = synthetic_cluster_config(tmp_path)
    timings = {}
    for label, config in (("image_plane", replace(base, likelihood=replace(base.likelihood, likelihood_mode="gp_image_plane")).validate()),
                          ("gaussian", potential_config(base, "gaussian")), ("matern52", potential_config(base, "matern52"))):
        state = build_state(config)
        evaluator = Evaluator(state, "gp_image_plane")
        theta = jnp.asarray(state.initial_theta)
        evaluator.value_and_grad(theta)[0].block_until_ready()
        best = np.inf
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(20):
                evaluator.value_and_grad(theta)[0].block_until_ready()
            best = min(best, (time.perf_counter() - start) / 20)
        timings[label] = best
    assert timings["gaussian"] < 2.0 * timings["image_plane"] and timings["matern52"] < 2.0 * timings["image_plane"], timings


# ----------------------------------------------------------------------------- 8. the GP mean deflection for the image solver
def _shift_offsets(state, theta, family_index, ux, uy):
    theta = np.array(theta, dtype=float)
    theta[np.asarray(state.images.source_x_param_index)[family_index]] = ux
    theta[np.asarray(state.images.source_y_param_index)[family_index]] = uy
    return theta


@pytest.mark.parametrize("family", FAMILIES)
def test_mean_deflection_matches_the_dense_conditional_and_its_jacobian(tmp_path, family):
    """E[delta_alpha(theta) | F] = K_*^T C^-1 F from the explicit family covariance, and d/dtheta by finite differences,
    including exactly at an observed image (the solver's starting point)."""
    config, _ = fold_config(tmp_path, depth=0.01)
    state = build_state(potential_config(config, family))
    evaluator = Evaluator(state, "gp_image_plane")
    theta = _shift_offsets(state, state.initial_theta, 0, 0.7, -0.4)
    mean = evaluator.gp_deflection_correction(theta)
    stats = evaluator.statistics(jnp.asarray(theta))
    scales = evaluator.error_scales(stats["physical"])
    layout = evaluator.gp
    slots, mask = np.asarray(layout["slots"]), np.asarray(layout["mask"])
    xy = np.column_stack([state.images.x_obs, state.images.y_obs])
    w = np.asarray(evaluator.gp_weights_per_image)
    b, src = np.asarray(stats["fields"]["backprojected"]), np.asarray(stats["source_by_family"])
    terms = [(float(scales["gp_sigma_small"]) ** 2, float(scales["gp_ell_small"])), (float(scales["gp_sigma_large"]) ** 2, evaluator.gp_ell_large)]

    def block(d):
        d2, outer = (d @ d)[None, None, None], np.outer(d, d)[None, None, None]
        return sum(np.asarray(gp.deflection_kernel(d2, outer, np.ones((1, 1), bool), s2, ell, family))[0, 0, 0] for s2, ell in terms)
    rng = np.random.default_rng(3)
    for f in range(slots.shape[0]):
        rows = slots[f][mask[f]]
        c = np.asarray(stats["family_covariance"])[f][:2 * len(rows), :2 * len(rows)]
        v = np.linalg.solve(c, (b[rows] - src[f]).ravel())
        points = np.vstack([xy[rows[0]], xy[rows[0]] + rng.normal(size=(3, 2))])
        dx, dy, *_ = mean(points[:, 0], points[:, 1], np.full(len(points), f))
        for p, got in zip(points, np.column_stack([dx, dy])):
            k_star = np.hstack([w[rows[0]] * w[j] * block(p - xy[j]) for j in rows])          # (2, 2M)
            np.testing.assert_allclose(got, k_star @ v, rtol=1e-9, atol=1e-13)
        # Jacobian against central differences, at the observed image and off it
        h = 1e-6
        _, _, j00, j01, j10, j11 = mean(points[:, 0], points[:, 1], np.full(len(points), f))
        jac = np.stack([np.stack([j00, j01], -1), np.stack([j10, j11], -1)], -2)
        for axis in range(2):
            step = np.zeros(2); step[axis] = h
            plus = np.column_stack(mean(*(points + step).T, np.full(len(points), f))[:2])
            minus = np.column_stack(mean(*(points - step).T, np.full(len(points), f))[:2])
            np.testing.assert_allclose(jac[:, :, axis], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-9)
        assert np.all(np.isfinite(jac))


def test_corrected_solver_recovers_a_fold_pair_whatever_the_sampled_source(tmp_path):
    """A 4 mas fold pair and a source offset four conditional sigmas from the centroid: the lens alone loses the pair for
    some directions of the offset; the lens plus the GP mean deflection keeps it for all of them."""
    from lenscluster.diagnostics import predictions_at
    from lenscluster.exact_images import ExactImageSolver
    config, _ = fold_config(tmp_path, depth=0.004)
    config = potential_config(config)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    family_by_id = {f.family_id: f for f in state.family_data}
    families = [family_by_id[f] for f in state.images.family_ids]
    pair = list(state.images.family_ids).index("1")
    lost, kept = 0, 0
    for ux, uy in ((4, 4), (4, -4), (-4, 4), (-4, -4)):
        theta = _shift_offsets(state, state.initial_theta, pair, ux, uy)
        physical = evaluator.physical(jnp.asarray(theta))
        sources = np.asarray(evaluator.source_positions(jnp.asarray(theta)))
        plain = solver.predict_all_local(state.images, families, evaluator.packed_state(physical)[0], sources)
        fixed = predictions_at(solver, state, evaluator, theta, adaptive=False)
        lost += plain[pair]["recovered_image_count"] < 2
        kept += fixed["count_rows"][pair]["recovered_image_count"] == 2
    assert lost >= 1 and kept == 4


def test_corrected_roots_sit_on_the_observed_images_without_measurement_error(tmp_path):
    """With the measurement term and the floor near zero the GP mean reproduces each residual, so the corrected lens maps
    every observed image onto its family's source and the roots are the observed positions."""
    from lenscluster.diagnostics import predictions_at
    from lenscluster.exact_images import ExactImageSolver
    config, _ = fold_config(tmp_path, depth=0.01)
    config = potential_config(config, gp_floor_arcsec=1e-6)
    config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, sigma_inst_arcsec=1e-5))).validate()
    state = build_state(config)
    assert np.allclose(np.asarray(state.images.sigma_inst_per_image), 1e-5)
    evaluator = Evaluator(state, "gp_image_plane")
    theta = set_scale(state, _shift_offsets(state, state.initial_theta, 0, 1.5, 0.5), "images.sigma_extra", 1e-5)
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    rows = predictions_at(solver, state, evaluator, theta, adaptive=False)["image_rows"]
    residual = np.array([r["image_residual_arcsec"] for r in rows if np.isfinite(r["image_residual_arcsec"])])
    assert len(residual) == state.images.n_images and residual.max() < 1e-3
