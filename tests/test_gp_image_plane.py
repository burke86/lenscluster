"""The gp_image_plane likelihood: the old model untouched, the independent limit, padding, gradients, whitening, the
fold-pair protection, stress values and speed."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from conftest import synthetic_cluster_config
from lenscluster import gp_image_plane as gp, smoothed_likelihood as sl
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state

ELL, FLOOR = 7.0, 0.0015
SHARED2, WHITE2 = 0.99, 0.01          # the fractions of sigma_img^2 that reproduce the earlier rho = 0.99 model


def gp_config(config, **likelihood):
    return replace(config, likelihood=replace(config.likelihood, likelihood_mode="gp_image_plane", **likelihood)).validate()


def set_scale(state, theta, name, value):
    theta = np.array(theta, dtype=float)
    index = [s.name for s in state.parameter_specs].index(name)
    theta[index] = np.log(value) if state.parameter_specs[index].transform_kind == "log_positive" else value
    return jnp.asarray(theta)


def set_image_scales(state, theta, sigma_img, shared2=SHARED2, white2=WHITE2):
    """sigma_shared^2 = shared2 sigma_img^2 and sigma_white^2 = white2 sigma_img^2."""
    theta = set_scale(state, theta, "images.sigma_shared", np.sqrt(shared2) * sigma_img)
    return set_scale(state, theta, "images.sigma_white", np.sqrt(white2) * sigma_img)


# ----------------------------------------------------------------------------- the fold of the synthetic lens
def fold_geometry(evaluator, physical, z_source, angle=0.7):
    """Outer critical point of the synthetic halo along a ray, its caustic image and the local fold expansion
    beta(theta_c + u e_t) = beta_c + u^2 v.  At the critical point A e_r is parallel to e_r, so the caustic runs along e_r
    and its normal is e_t; the depth of a source is its offset along e_t, u^2 (v . e_t) = c u^2 / 2."""
    packed = evaluator.packed_state_for_z(physical, z_source)
    components = np.arange(evaluator.n_components, dtype=np.int32)

    def fields(points):
        alpha, u, du = evaluator.component_rows(jnp.asarray(np.atleast_2d(points)), packed, components, derivative=True)
        return (np.atleast_2d(points) - np.asarray(jnp.sum(alpha, 0)), np.eye(2) - np.asarray(jnp.sum(u, 0)), -np.asarray(jnp.sum(du, 0)))

    det = lambda r: np.linalg.det(fields([r * np.cos(angle), r * np.sin(angle)])[1][0])
    radii = np.linspace(2.0, 40.0, 200)
    k = np.flatnonzero(np.diff(np.sign([det(r) for r in radii])))[-1]
    rc = brentq(det, radii[k], radii[k + 1])
    theta_c = np.array([rc * np.cos(angle), rc * np.sin(angle)])
    beta_c, a, da = (x[0] for x in fields(theta_c))
    _, e_t = gp.tangential_eigenpair(a)
    v = 0.5 * np.einsum("abk,b,k->a", da, e_t, e_t)
    c = 2.0 * float(v @ e_t)
    normal = np.sign(c) * e_t
    c = abs(c)

    def images(depth):
        """Source at ``depth`` inside the caustic along the fold normal and its two exact images."""
        beta = beta_c + depth * normal
        roots = []
        for sign in (1.0, -1.0):
            theta = theta_c + sign * np.sqrt(2.0 * depth / c) * e_t
            for _ in range(60):
                b, jac, _ = fields(theta)
                theta = theta - np.linalg.solve(jac[0], b[0] - beta)
            assert np.abs(fields(theta)[0][0] - beta).max() < 1e-12
            roots.append(theta)
        return beta, np.array(roots), fields(np.array(roots))[1]

    return dict(theta_c=theta_c, beta_c=beta_c, e_t=e_t, normal=normal, c=c, images=images, fields=fields)


def fold_config(tmp_path: Path, depth: float = 0.01, extra_offset: float | None = None):
    """The synthetic cluster with family 1 replaced by the exact fold pair at ``depth`` (z = 2), optionally with a third
    image ``extra_offset`` arcsec from the critical point along e_t."""
    base = synthetic_cluster_config(tmp_path)
    state = build_state(gp_config(base))
    evaluator = Evaluator(state, "gp_image_plane")
    fold = fold_geometry(evaluator, evaluator.physical(jnp.asarray(state.initial_theta)), 2.0)
    _, images, _ = fold["images"](depth)
    rows = [f"1.{label} {x:.12f} {y:.12f} 1 1 0 2 24 1 .05" for label, (x, y) in zip("ab", images)]
    if extra_offset is not None:
        x, y = fold["theta_c"] + extra_offset * fold["e_t"]
        rows.append(f"1.c {x:.12f} {y:.12f} 1 1 0 2 24 1 .05")
    rows += ["2.a 8 -6 1 1 0 3 24 1 .05", "2.b -7 9 1 1 0 3 24 1 .05"]
    Path(base.model.image_constraints.catalog_path).write_text("#REFERENCE 3\n" + "\n".join(rows) + "\n")
    return gp_config(base), fold


# ----------------------------------------------------------------------------- 1. the existing model is untouched
BASELINE = {
    "init": (-872.0390336318344,
             [6.3828442167327974, 4.5855598294114515, 2.2594962448735196, -1.8971199848858813, 0.27, 0.65, -2.3025850929940455, -1.2039728043259361,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.3025850929940455, -2.995732273553991, -3.2188758248682006, -0.35667494393873245, 0.0, 0.0, 0.0, 0.0],
             [-9010.025819880258, -82.77412373742811, -39.43229521488714, -0.12545565682903617, 13.440384269714643, 5.820419712677817, 0.0, 0.0,
              -6.816875291277276, -1.4654993579333893, 0.004329324609624853, 0.0006329508582267723, -9.969600706483137, -1.9050771599980973,
              0.00640638879185415, 0.000946216174526195, 974.1258645285537, 126.37147778684704, 78.72055741262074, 35.95984468382649,
              -0.002530780534922646, -0.006295381349949845, -0.10590429465191872, 0.08211008181790769]),
    "perturbed": (-956.8690869432104,
                  [6.389130727787467, 4.578954586246886, 2.291517377395684, -1.8918749790282292, 0.24321653134194446, 0.6680797527454743,
                   -2.2373850907375386, -1.156618756169474, -0.03518676179034963, -0.06327107355230263, -0.03116372312686761, 0.00206629896736218,
                   -0.11625153873194172, -0.010939583196627287, -0.06229554736265326, -0.03661336773517258, -2.329798042136911, -3.0115472813724486,
                   -3.1982942980494937, -0.30454927546659855, -0.006426733147201713, 0.0683231735274843, -0.03325973367433068, 0.017575503504650987],
                  [-10003.71074104369, -87.3990960159081, -42.505683325618314, -0.15202484666530847, 15.610915495042015, 6.761689538561891,
                   0.37930288790369276, 1.316038727039001, -7.51807377546797, -1.8162060833694946, 0.004771015274097208, 0.0007698576426837218,
                   -11.106407738413754, -2.3195482009964405, 0.006930846967878674, 0.0011305147911845966, 1046.249019016055, 138.60868183887956,
                   97.41553443173167, 42.81011119976618, -0.0028386518844669066, -0.0065057418576004306, -0.10940074695355131, 0.08489125723795038]),
}


def test_existing_model_unchanged(synthetic_config):
    """Log-target and gradient of the default model at two points, recorded before gp_image_plane existed."""
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    assert state.sigma_shared_param_index == -1 and not any(s.name in ("images.sigma_shared", "images.sigma_white") for s in state.parameter_specs)
    for value, theta, grad in BASELINE.values():
        v, g = evaluator.value_and_grad(jnp.asarray(theta))
        np.testing.assert_allclose(float(v), value, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(np.asarray(g), grad, rtol=1e-12, atol=1e-12 * np.abs(grad).max())


# ----------------------------------------------------------------------------- 2. the independent limit
def test_independent_limit_matches_per_image_model(synthetic_config):
    """sigma_shared -> 0 and sigma_fl = 0: the family likelihood and the source conditional are those of independent images
    with sigma_meas^2 = sigma_inst^2 + sigma_white^2 and no source-plane floor."""
    config = gp_config(synthetic_config, gp_floor_arcsec=0.0)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    theta = set_scale(state, set_scale(state, state.initial_theta, "images.sigma_white", 0.07), "images.sigma_shared", 1e-9)
    stats = evaluator.statistics(theta)
    images, fields = state.images, stats["fields"]
    sigma_meas = np.sqrt(np.asarray(images.sigma_inst_per_image) ** 2 + 0.07 ** 2)
    old = sl.statistics(fields["backprojected"], stats["source_by_family"][images.family_index_per_image], fields["jacobian"],
                        fields["determinant_gradient"], sigma_meas, 0.0, 0.0, 1.0, parity=images.parity)
    family_index = np.asarray(images.family_index_per_image)
    for f in range(images.n_families):
        np.testing.assert_allclose(float(stats["family_gaussian_loglike"][f]), float(np.sum(np.asarray(old["gaussian_loglike"])[family_index == f])), rtol=1e-9)
    whitening = sl.source_whitening(fields["backprojected"], old["covariance"], images.family_index_per_image, images.n_families)
    np.testing.assert_allclose(np.asarray(stats["whitening"]["centroid"]), np.asarray(whitening["centroid"]), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(stats["whitening"]["covariance"]), np.asarray(whitening["covariance"]), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(np.asarray(stats["log_h"]), np.asarray(old["log_h"]), rtol=1e-9)
    total = float(np.sum(np.asarray(old["gaussian_loglike"]) + np.asarray(old["log_h"]))) + float(stats["source_correction"])
    np.testing.assert_allclose(float(evaluator.loglike(theta)), total, rtol=1e-9)


# ----------------------------------------------------------------------------- 3. positive definiteness and padding
def test_family_covariance_positive_definite_and_padding_neutral(synthetic_config):
    state = build_state(gp_config(synthetic_config))
    evaluator = Evaluator(state, "gp_image_plane")
    theta = jnp.asarray(state.initial_theta)
    stats = evaluator.statistics(theta)
    layout = evaluator.gp
    for f, cov in enumerate(np.asarray(stats["family_covariance"])):
        real = np.repeat(np.asarray(layout["mask"][f]), 2)
        assert np.linalg.eigvalsh(cov[np.ix_(real, real)]).min() >= FLOOR ** 2 * (1 - 1e-9)
    assert np.all(np.isfinite(np.asarray(stats["family_cholesky"])))
    # One more padded slot per family changes nothing.
    jacobian, backprojected = stats["fields"]["jacobian"], stats["fields"]["backprojected"]
    pad = lambda x, fill: jnp.concatenate([x, jnp.full((x.shape[0], 1) + x.shape[2:], fill, dtype=x.dtype)], axis=1)
    slots = jnp.concatenate([layout["slots"], jnp.zeros((layout["slots"].shape[0], 1), dtype=layout["slots"].dtype)], axis=1)
    mask = pad(layout["mask"], False)
    kernel = jnp.pad(layout["kernel"], ((0, 0), (0, 1), (0, 1)))
    _, chol = gp.family_covariance(jacobian[slots], kernel, pad(layout["sigma_inst2"], 0.0), mask, jnp.square(stats["sigma_shared"]),
                                   jnp.square(stats["sigma_white"]), FLOOR ** 2)
    whitening = gp.family_source_whitening(chol, backprojected[slots], mask)
    source = stats["source_by_family"]
    loglike, _, _ = gp.family_gaussian_loglike(chol, backprojected[slots] - source[:, None, :], mask, layout["counts"])
    np.testing.assert_allclose(np.asarray(loglike), np.asarray(stats["family_gaussian_loglike"]), rtol=1e-12)
    for key in ("centroid", "covariance", "logdet_precision"):
        np.testing.assert_allclose(np.asarray(whitening[key]), np.asarray(stats["whitening"][key]), rtol=1e-12, atol=1e-14)
    assert np.isfinite(float(evaluator.loglike(theta)))
    assert np.all(np.isfinite(np.asarray(stats["family_cholesky"])))


# ----------------------------------------------------------------------------- 4. gradients at a near-critical image
def test_gradient_matches_finite_differences_near_critical(tmp_path):
    config, fold = fold_config(tmp_path, depth=0.01)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    theta = np.asarray(state.initial_theta, dtype=float)
    theta[names.index("source.1.beta_x")] = 0.3
    theta[names.index("members.g1.velocity_offset")] = 0.2
    lambda_t = [gp.tangential_eigenpair(a)[0] for a in np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(theta)))["jacobian"])[:2]]
    assert max(abs(l) for l in lambda_t) < 0.05            # the exact pair of a 10 mas source: |lambda_t| about 0.02
    value, grad = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value))
    for name, step in (("halo.v_disp", 1e-6), ("members.sigma", 1e-6), ("members.g1.velocity_offset", 1e-5), ("images.sigma_shared", 1e-5),
                       ("images.sigma_white", 1e-6), ("source.1.beta_x", 1e-5)):
        i = names.index(name)
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        fd = (float(evaluator.loglike(jnp.asarray(plus))) - float(evaluator.loglike(jnp.asarray(minus)))) / (2 * step)
        np.testing.assert_allclose(float(grad[i]), fd, rtol=2e-5, atol=1e-6, err_msg=name)


# ----------------------------------------------------------------------------- 5. whitening
def test_source_offsets_are_standard_normal_at_fixed_lens(synthetic_config):
    """Hessian of the log-target (likelihood, transport correction and the standard-normal prior of the offsets, which the
    sampling model adds) in u_f on a family without near-critical images is minus the identity to 10%."""
    state = build_state(gp_config(synthetic_config))
    evaluator = Evaluator(state, "gp_image_plane")
    theta = jnp.asarray(state.initial_theta)
    images = state.images
    determinant = np.abs(np.linalg.det(np.asarray(evaluator.fields(evaluator.physical(theta))["jacobian"])))
    family_index = np.asarray(images.family_index_per_image)
    families = [f for f in range(images.n_families) if determinant[family_index == f].min() > 0.3]
    assert families
    for family in families:
        idx = jnp.asarray([int(images.source_x_param_index[family]), int(images.source_y_param_index[family])])
        target = lambda u: evaluator.log_likelihood(theta.at[idx].set(u)) - 0.5 * jnp.sum(u ** 2)
        hessian = np.asarray(jax.hessian(target)(theta[idx]))
        np.testing.assert_allclose(hessian, -np.eye(2), atol=0.1)


# ----------------------------------------------------------------------------- 6. the fold pair
def family_loglike_at(fold, depth, floor, sigma_img=0.3, sigma_inst=0.05):
    """Gaussian family log-likelihood of the exact pair as a function of the source offset along the fold normal."""
    beta, images, jacobian = fold["images"](depth)
    mask = np.ones((1, 2), dtype=bool)
    kernel = gp.displacement_kernel(images[:, 0], images[:, 1], np.array([[0, 1]]), mask, ELL)
    _, chol = gp.family_covariance(jacobian[None], kernel, np.full((1, 2), sigma_inst ** 2), mask, SHARED2 * sigma_img ** 2, WHITE2 * sigma_img ** 2, floor ** 2)
    assert np.all(np.isfinite(np.asarray(chol)))

    def value(shift):
        residual = np.broadcast_to(shift * fold["normal"], (1, 2, 2))          # b_i = beta for both images; beta_test = beta + shift n
        return float(gp.family_gaussian_loglike(chol, residual, mask, np.array([2.0]))[0][0])

    def old(shift, sigma_def=0.012, sigma_meas=0.17, sigma_sel=0.0, lambda_sel=1.0):
        """The per-image model at the floors fitted to M0416 (oct29a: sigma_def 12 mas, sigma_extra 0.14" over sigma_inst 0.1")."""
        residual = np.broadcast_to(shift * fold["normal"], (2, 2))
        stats = sl.statistics(beta + residual, beta, jacobian, np.zeros((2, 2)), sigma_meas, sigma_def, sigma_sel, lambda_sel)
        return float(np.sum(np.asarray(stats["gaussian_loglike"])))

    return value, old


def test_fold_pair_stays_inside_the_caustic(synthetic_config):
    state = build_state(gp_config(synthetic_config))
    evaluator = Evaluator(state, "gp_image_plane")
    fold = fold_geometry(evaluator, evaluator.physical(jnp.asarray(state.initial_theta)), 2.0)
    assert 0.01 < fold["c"] < 0.1
    value, old = family_loglike_at(fold, 0.01, FLOOR)
    to_caustic, outside = value(0.0) - value(-0.01), value(0.0) - value(-0.02)
    assert to_caustic >= 5.0 and outside >= 20.0, (to_caustic, outside)
    old_outside = old(0.0) - old(-0.02)
    print(f"fold toy: gp drops {to_caustic:.1f} / {outside:.1f} nats, old model {old_outside:.2f} nats")
    assert old_outside < 0.1 * outside
    value_1p5, _ = family_loglike_at(fold, 0.003, FLOOR)
    value_0p5, _ = family_loglike_at(fold, 0.003, 0.0005)
    assert value_1p5(0.0) - value_1p5(-0.003) > 2.0
    assert value_0p5(0.0) - value_0p5(-0.003) > 5.0


def test_shallow_pair_check_reports_the_pair(tmp_path):
    config, fold = fold_config(tmp_path, depth=0.01)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    rows = gp.shallow_fold_pairs(state, evaluator, state.initial_theta)
    assert len(rows) == 1 and rows[0]["labels"] == ["1.a", "1.b"]
    row = rows[0]
    np.testing.assert_allclose(row["c_per_arcsec"], fold["c"], rtol=0.05)
    np.testing.assert_allclose(row["depth_arcsec"], 0.01, rtol=0.1)          # own depth recovers the injected depth
    assert row["lambda_t"][0] * row["lambda_t"][1] < 0 and row["depth_error_arcsec"] > 0
    assert "1.a/1.b" in gp.shallow_fold_pair_report(rows, FLOOR)


# ----------------------------------------------------------------------------- 7. stress
def test_stress_values_and_gradients_finite(tmp_path):
    config, fold = fold_config(tmp_path, depth=0.01, extra_offset=0.01)
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    jacobian = np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(state.initial_theta)))["jacobian"])
    assert abs(gp.tangential_eigenpair(jacobian[2])[0]) < 5e-4            # 1.c sits 0.01" from the critical curve
    for sigma_img in (0.01, 0.3, 3.0):
        theta = set_image_scales(state, state.initial_theta, sigma_img)
        value, grad = evaluator.value_and_grad(theta)
        assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad))), sigma_img
        stats = evaluator.statistics(theta)
        assert bool(np.all(np.asarray(stats["valid"]))) and bool(np.all(np.asarray(stats["transport_valid"])))


# ----------------------------------------------------------------------------- the earlier one-scale model
def test_two_scales_reproduce_the_rho_model(synthetic_config):
    """sigma_shared^2 = 0.99 sigma^2 and sigma_white^2 = 0.01 sigma^2 is the earlier rho = 0.99 model at sigma_img = sigma:
    S = sigma^2 (rho K + (1 - rho) I) kron I_2 + sigma_inst^2 I.  The log-target agrees to 1e-10.  ell is fixed so that
    the precomputed kernel is the one the likelihood reads."""
    state = build_state(gp_config(synthetic_config, gp_correlation_length_prior=None))
    evaluator = Evaluator(state, "gp_image_plane")
    sigma = 0.23
    theta = set_image_scales(state, state.initial_theta, sigma)
    new_value = float(evaluator.loglike(theta))
    layout = evaluator.gp
    mask = np.asarray(layout["mask"])
    eye = np.eye(mask.shape[1])[None] * (mask[:, :, None] & mask[:, None, :])
    rho_kernel = jnp.asarray(0.99 * np.asarray(layout["kernel"]) + 0.01 * eye)
    original = dict(layout)
    try:
        layout["kernel"] = rho_kernel
        old_theta = set_scale(state, set_scale(state, state.initial_theta, "images.sigma_shared", sigma), "images.sigma_white", 0.0)
        stats = evaluator.statistics(old_theta)                      # sigma_white = 0 bypasses the positivity check of log_likelihood
        old_value = float(jnp.sum(stats["loglike"]) + stats["source_correction"] + evaluator.far_field_log_barrier(stats["fields"]["far_cut_radii"]))
    finally:
        layout.update(original)
    np.testing.assert_allclose(new_value, old_value, rtol=1e-10, atol=0.0)


# ----------------------------------------------------------------------------- 8. speed
def big_config(tmp_path: Path, n_families: int = 40, per_family: int = 4):
    base = synthetic_cluster_config(tmp_path)
    rng = np.random.default_rng(3)
    rows = []
    for f in range(n_families):
        z = 2.0 if f % 2 else 3.0
        for label in "abcdef"[:per_family]:
            x, y = rng.uniform(-15, 15, 2)
            rows.append(f"{f + 1}.{label} {x:.6f} {y:.6f} 1 1 0 {z:g} 24 1 .05")
    Path(base.model.image_constraints.catalog_path).write_text("#REFERENCE 3\n" + "\n".join(rows) + "\n")
    return base


def timed_gradient(evaluator, theta, repeats=30):
    value, grad = evaluator.value_and_grad(theta)
    grad.block_until_ready()
    best = np.inf
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(repeats):
            value, grad = evaluator.value_and_grad(theta)
        grad.block_until_ready()
        best = min(best, (time.perf_counter() - started) / repeats)
    return best


def test_gradient_overhead_below_five_percent(tmp_path):
    base = big_config(tmp_path)
    old_state, new_state = build_state(base), build_state(gp_config(base))
    old, new = Evaluator(old_state), Evaluator(new_state, "gp_image_plane")
    old_time = timed_gradient(old, jnp.asarray(old_state.initial_theta))
    new_time = timed_gradient(new, jnp.asarray(new_state.initial_theta))
    assert new_time < 1.05 * old_time, (old_time, new_time)


def test_sampled_ell_matches_the_fixed_kernel_and_has_a_gradient(synthetic_config):
    """ell sampled (default) and ell fixed at the same value give the same likelihood; None removes the site."""
    from lenscluster.config import LogNormalPriorConfig
    sampled_state = build_state(gp_config(synthetic_config))
    fixed_state = build_state(gp_config(synthetic_config, gp_correlation_length_prior=None, gp_correlation_length_arcsec=3.0))
    assert fixed_state.gp_ell_param_index == -1 and not any(s.name == "images.gp_ell" for s in fixed_state.parameter_specs)
    spec = sampled_state.parameter_specs[sampled_state.gp_ell_param_index]
    assert spec.name == "images.gp_ell" and spec.transform_kind == "log_positive" and spec.physical_upper == 30.0
    assert np.isclose(np.exp(sampled_state.initial_theta[sampled_state.gp_ell_param_index]), 7.0)
    sampled, fixed = Evaluator(sampled_state, "gp_image_plane"), Evaluator(fixed_state, "gp_image_plane")
    theta = set_scale(sampled_state, sampled_state.initial_theta, "images.gp_ell", 3.0)
    names = [s.name for s in sampled_state.parameter_specs]
    fixed_theta = jnp.asarray([float(theta[names.index(s.name)]) for s in fixed_state.parameter_specs])
    assert np.isclose(float(sampled.log_likelihood(theta)), float(fixed.log_likelihood(fixed_theta)), rtol=0, atol=1e-9)
    # The kernel changes the likelihood, and the gradient with respect to ln ell matches central differences.
    other = set_scale(sampled_state, theta, "images.gp_ell", 12.0)
    assert abs(float(sampled.log_likelihood(other)) - float(sampled.log_likelihood(theta))) > 1e-3
    index = sampled_state.gp_ell_param_index
    gradient = float(jax.grad(sampled.log_likelihood)(theta)[index])
    h = 1e-5
    plus, minus = theta.at[index].add(h), theta.at[index].add(-h)
    numerical = (float(sampled.log_likelihood(plus)) - float(sampled.log_likelihood(minus))) / (2 * h)
    assert np.isclose(gradient, numerical, rtol=1e-5, atol=1e-6)
    assert LogNormalPriorConfig(median=7.0, log_sigma=0.5, upper=30.0) == gp_config(synthetic_config).likelihood.gp_correlation_length_prior
