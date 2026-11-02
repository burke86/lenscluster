"""The knee-spectrum mass field + image-plane GP (gp_knee.py): tables against quadrature, the clamp, positive definiteness,
the no-mass-field limit, several knots per block, gradients, a critical line sweeping across an image, a fold pair in a
two-knot system with the mean-deflection solver, and speed."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from scipy.special import jv

from conftest import synthetic_cluster_config, write_offset_catalog
from lenscluster import gp_knee
from lenscluster.evaluator import Evaluator
from lenscluster.image_catalog import write_image_catalog
from lenscluster.sky import offsets_to_radec
from lenscluster.state import build_state
from test_gp_image_plane import big_config, fold_config, set_scale

BLOCKS = ("family", "system", "all")


def knee_config(config, blocks="all", **likelihood):
    return replace(config, likelihood=replace(config.likelihood, likelihood_mode="gp_image_plane", gp_kernel="knee", gp_blocks=blocks,
                                              **likelihood)).validate()


def set_sites(state, theta, **values):
    for name, value in values.items():
        theta = set_scale(state, theta, f"images.{name}", value)
    return theta


# ----------------------------------------------------------------------------- 1. the tables
def direct_curves(r, lam, *, scale_min=0.2, scale_max=100.0, gamma=4.0, nodes=3000):
    k = np.geomspace(2 * np.pi / scale_max, 2 * np.pi / scale_min, nodes)
    w = np.log(k[1] / k[0]) / (1 + (k / (2 * np.pi / lam)) ** gamma)
    w /= w.sum()
    return (jv(0, np.outer(r, k)) - jv(2, np.outer(r, k))) @ w, (jv(0, np.outer(r, k)) + jv(2, np.outer(r, k))) @ w


@pytest.fixture(scope="module")
def tables():
    return gp_knee.knee_tables(r_max=100.0, r_step=0.02, lam_min=0.5, lam_max=80.0, n_lam=64, scale_min=0.2, scale_max=100.0, gamma=4.0)


def test_tables_match_direct_quadrature_off_grid(tables):
    r = np.array([0.0, 0.37, 1.3, 4.1, 9.7, 23.0, 51.0])
    for lam in (0.83, 3.7, 6.0, 21.5):
        index, fraction = gp_knee.r_grid_position(tables, r)
        c_l, c_t = (np.asarray(v) for v in gp_knee.knee_coefficients(tables, index, fraction, jnp.asarray(lam)))
        d_l, d_t = direct_curves(r, lam)
        np.testing.assert_allclose(c_l, d_l, atol=1e-4)
        np.testing.assert_allclose(c_t, d_t, atol=1e-4)
        assert c_l[0] == pytest.approx(1.0, abs=1e-12) and c_t[0] == pytest.approx(1.0, abs=1e-12)
    # the curl-free signature: c_L drops below zero past the knee, c_T stays positive longer, and c_L = d(r c_T)/dr
    c_l, c_t = (np.asarray(v) for v in gp_knee.knee_curves(tables, jnp.asarray(6.0)))
    rr = np.asarray(tables["r_grid"])
    assert c_l[np.searchsorted(rr, 12.0)] < 0.01 < c_t[np.searchsorted(rr, 12.0)]
    np.testing.assert_allclose(np.gradient(rr * c_t, rr)[5:-5], c_l[5:-5], atol=2e-3)


def test_knee_wavelength_is_clamped_to_the_grid(tables):
    r = np.array([0.5, 2.0, 8.0])
    index, fraction = gp_knee.r_grid_position(tables, r)
    f = lambda ln_lam: gp_knee.knee_coefficients(tables, index, fraction, jnp.exp(ln_lam))[0]
    for lam, edge in ((0.1, 0.5), (500.0, 80.0)):
        np.testing.assert_allclose(np.asarray(f(jnp.log(lam))), np.asarray(f(jnp.log(edge))), rtol=1e-12)
        grad = np.asarray(jax.jacfwd(f)(jnp.log(lam)))
        assert np.all(np.isfinite(grad)) and np.all(grad == 0.0)
    inside = np.asarray(jax.jacfwd(f)(jnp.log(6.0)))
    assert np.all(np.isfinite(inside)) and np.any(inside != 0.0)


def test_deflection_blocks_stay_positive_definite_over_the_knee_grid(tables):
    rng = np.random.default_rng(5)
    xy = rng.uniform(-30, 30, (40, 2))
    d = xy[:, None] - xy[None]
    r = np.hypot(d[..., 0], d[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(r[..., None] > 0, d / np.maximum(r, 1e-300)[..., None], 0.0)
    nn = jnp.asarray((n[..., :, None] * n[..., None, :])[None])
    index, fraction = gp_knee.r_grid_position(tables, r[None])
    worst = 0.0
    for lam in np.geomspace(0.5, 80.0, 200):
        c_l, c_t = gp_knee.knee_coefficients(tables, index, fraction, jnp.asarray(lam))
        k = np.asarray(gp_knee.deflection_blocks(c_l, c_t, nn, jnp.ones((1, 40, 40), bool), 1.0))[0]
        full = k.transpose(0, 2, 1, 3).reshape(80, 80)
        worst = min(worst, float(np.linalg.eigvalsh(full).min()))
    assert worst > -1e-10, worst
    with pytest.raises(ValueError, match="r_max"):                   # a separation beyond the tables is an error, never clipped
        gp_knee.r_grid_position(tables, np.array([150.0]))


def test_natural_spline_operator_matches_scipy():
    from scipy.interpolate import CubicSpline
    x = np.linspace(2.0, 6.0, 17)
    y = np.random.default_rng(0).normal(size=x.size)
    expected = CubicSpline(x, y, bc_type="natural").derivative(2)(x)
    np.testing.assert_allclose(gp_knee._natural_second_derivative_operator(x) @ y, expected, atol=1e-10)


@pytest.fixture(scope="module")
def slope_tables():
    return gp_knee.knee_tables(r_max=60.0, r_step=0.02, lam_min=0.5, lam_max=80.0, n_lam=64, scale_min=0.2, scale_max=100.0,
                               gamma_grid=(2.0, 6.0, 17))


def test_slope_tables_match_direct_quadrature_between_slope_nodes(slope_tables):
    """The sampled slope: a cubic spline through 17 slopes reproduces the quadrature at slopes between the nodes."""
    r = np.array([0.0, 0.37, 1.3, 4.1, 9.7, 23.0, 51.0])
    index, fraction = gp_knee.r_grid_position(slope_tables, r)
    for gamma in (2.0, 2.37, 3.1, 4.0, 4.62, 5.9):
        for lam in (0.83, 6.0, 21.5):
            c_l, c_t = (np.asarray(v) for v in gp_knee.knee_coefficients(slope_tables, index, fraction, jnp.asarray(lam), jnp.asarray(gamma)))
            d_l, d_t = direct_curves(r, lam, gamma=gamma)
            np.testing.assert_allclose(c_l, d_l, atol=2e-4, err_msg=f"gamma {gamma} lambda {lam}")
            np.testing.assert_allclose(c_t, d_t, atol=2e-4, err_msg=f"gamma {gamma} lambda {lam}")
    # differentiable in the slope, and a steeper fall-off beyond the knee keeps more correlation at small separations
    f = lambda g: gp_knee.knee_coefficients(slope_tables, index, fraction, jnp.asarray(6.0), g)[1][3]
    grad = float(jax.grad(f)(jnp.asarray(3.3)))
    fd = (float(f(jnp.asarray(3.3 + 1e-4))) - float(f(jnp.asarray(3.3 - 1e-4)))) / 2e-4
    assert np.isfinite(grad) and grad == pytest.approx(fd, rel=1e-4, abs=1e-8) and grad > 0
    with pytest.raises(ValueError):
        gp_knee.knee_curves(slope_tables, jnp.asarray(6.0))
    # the tables restricted to the rows the pairs use give the same coefficients
    restricted, local = gp_knee.pair_tables(slope_tables, index)
    for a, b in zip(gp_knee.pair_coefficients(restricted, local, fraction, jnp.asarray(6.0), jnp.asarray(3.3)),
                    gp_knee.knee_coefficients(slope_tables, index, fraction, jnp.asarray(6.0), jnp.asarray(3.3))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-14)


def test_sampled_slope_at_four_matches_the_fixed_slope(synthetic_config):
    sampled = build_state(knee_config(synthetic_config, "all"))
    fixed = build_state(knee_config(synthetic_config, "all", gp_knee_gamma_prior=None))
    names = [p.name for p in sampled.parameter_specs]
    assert "images.gp_knee_gamma" in names and "images.gp_knee_gamma" not in [p.name for p in fixed.parameter_specs]
    spec = sampled.parameter_specs[names.index("images.gp_knee_gamma")]
    assert (spec.prior_kind, spec.lower, spec.upper) == ("uniform", 2.0, 6.0)
    theta = np.asarray(sampled.initial_theta, dtype=float)
    assert theta[names.index("images.gp_knee_gamma")] == 4.0
    keep = [i for i, n in enumerate(names) if n != "images.gp_knee_gamma"]
    a = float(Evaluator(sampled, "gp_image_plane").loglike(jnp.asarray(theta)))
    b = float(Evaluator(fixed, "gp_image_plane").loglike(jnp.asarray(theta[keep])))
    np.testing.assert_allclose(a, b, rtol=1e-9)


# ----------------------------------------------------------------------------- 2. limits and layouts
def test_without_the_mass_field_the_knee_path_is_the_image_plane_gp(synthetic_config):
    state_ip = build_state(replace(synthetic_config, likelihood=replace(synthetic_config.likelihood, likelihood_mode="gp_image_plane")).validate())
    state_k = build_state(knee_config(synthetic_config, "family", gp_sigma_alpha_prior=None))
    assert state_k.gp_sigma_alpha_param_index == -1 and "images.gp_knee" not in [p.name for p in state_k.parameter_specs]
    values = dict(sigma_shared=0.15, sigma_white=0.03, gp_ell=5.0)
    a = float(Evaluator(state_ip, "gp_image_plane").loglike(jnp.asarray(set_sites(state_ip, state_ip.initial_theta, **values))))
    b = float(Evaluator(state_k, "gp_image_plane").loglike(jnp.asarray(set_sites(state_k, state_k.initial_theta, **values))))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_family_and_system_blocks_agree_when_every_system_has_one_knot(synthetic_config):
    values = [float(Evaluator(build_state(knee_config(synthetic_config, b)), "gp_image_plane").loglike(jnp.asarray(build_state(knee_config(synthetic_config, b)).initial_theta)))
              for b in ("family", "system")]
    np.testing.assert_allclose(values[0], values[1], rtol=1e-12)


def two_knot_config(tmp_path: Path, depth: float = 0.004, knot_offset: float = 0.3):
    """The synthetic cluster with system 1 = a fold pair (knot A) plus a second knot B offset by ``knot_offset`` arcsec in both
    images, and system 2 a wide pair; built with the fold geometry of the gp tests."""
    config, fold = fold_config(tmp_path, depth=depth)
    _, images, _ = fold["images"](depth)
    def entry(x, y):
        ra, dec = offsets_to_radec(float(x), float(y), 0.0, 0.0)
        return {"ra": float(ra), "dec": float(dec)}
    two = lambda p: {"knots": {"A": entry(*p), "B": entry(p[0] + knot_offset, p[1] + knot_offset)}}
    systems = {"1": {"z": 2.0, "knots": ["A", "B"], "images": {letter: two(p) for letter, p in zip("ab", images)}},
               "2": {"z": 3.0, "knots": ["A", "B"], "images": {"a": two((8, -6)), "b": two((-7, 9))}}}
    write_image_catalog(Path(config.model.image_constraints.catalog_path), systems, defaults={"mag": 24.0, "mag_err": 0.05})
    return config, fold


def test_two_knot_system_layout_and_source_transport(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(knee_config(config, "system"))
    evaluator = Evaluator(state, "gp_image_plane")
    layout = evaluator.gp
    assert tuple(np.asarray(layout["mask"]).shape) == (2, 4) and layout["n_knots"] == 2
    assert sorted(state.images.family_ids) == ["1A", "1B", "2A", "2B"] and len(state.images.system_ids) == 2
    assert np.asarray(layout["knot_mask"]).sum(1).tolist() == [2, 2]
    # the whitened source offsets of a two-knot block are standard normal at fixed lens: the Hessian of the log target in
    # the four offsets (likelihood + transport correction + standard-normal prior) is minus the identity to 10%.  System 2,
    # away from the critical curve: at the fold pair the determinant term makes the target non-Gaussian in the source.
    names = [p.name for p in state.parameter_specs]
    idx = jnp.asarray([names.index(f"source.{f}.beta_{a}") for f in ("2A", "2B") for a in ("x", "y")])
    theta0 = jnp.asarray(state.initial_theta)
    def target(u):
        theta = theta0.at[idx].set(u)
        return evaluator.log_likelihood(theta) - 0.5 * jnp.sum(u ** 2)
    hessian = np.asarray(jax.hessian(target)(jnp.zeros(4)))
    np.testing.assert_allclose(hessian, -np.eye(4), atol=0.1)


# ----------------------------------------------------------------------------- 3. gradients and the critical line
@pytest.mark.parametrize("blocks", BLOCKS)
def test_gradients_match_finite_differences(tmp_path, blocks):
    config, _ = fold_config(tmp_path, depth=0.01)
    state = build_state(knee_config(config, blocks))
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    theta = np.asarray(state.initial_theta, dtype=float)
    theta[names.index("source.1.beta_x")] = 0.3
    value, grad = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))
    for name, step in (("images.gp_sigma_alpha", 1e-5), ("images.gp_knee", 1e-5), ("images.sigma_shared", 1e-5), ("images.gp_ell", 1e-5),
                       ("images.sigma_white", 1e-6), ("images.gp_knee_gamma", 1e-5), ("halo.v_disp", 1e-6), ("source.1.beta_x", 1e-5)):
        i = names.index(name)
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        fd = (float(evaluator.loglike(jnp.asarray(plus))) - float(evaluator.loglike(jnp.asarray(minus)))) / (2 * step)
        np.testing.assert_allclose(float(grad[i]), fd, rtol=2e-5, atol=1e-6, err_msg=name)


def test_critical_line_sweeping_across_an_image_keeps_the_target_smooth(tmp_path):
    """Scale the halo so the critical curve moves across image 1.a of a 10 mas fold pair (det A changes sign): the log
    target and its gradient stay finite and vary continuously, with sigma_fl = 0 (the mass field keeps C positive
    definite on the critical curve)."""
    config, _ = fold_config(tmp_path, depth=0.01)
    state = build_state(knee_config(config, "all", gp_floor_arcsec=0.0))
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    i_v = names.index("halo.v_disp")
    theta0 = np.asarray(state.initial_theta, dtype=float)
    grid = np.linspace(-0.12, 0.12, 49)                        # in the log coordinate of v_disp: +-12%
    values, grads, dets, fd = [], [], [], []
    step = 1e-6
    for d in grid:
        theta = theta0.copy(); theta[i_v] += d
        value, grad = evaluator.value_and_grad(jnp.asarray(theta))
        values.append(float(value)); grads.append(np.asarray(grad))
        det = np.linalg.det(np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(theta)), force_exact=True, derivative=False)["jacobian"]))
        dets.append(det[0])
        plus, minus = theta.copy(), theta.copy(); plus[i_v] += step; minus[i_v] -= step
        fd.append((float(evaluator.loglike(jnp.asarray(plus))) - float(evaluator.loglike(jnp.asarray(minus)))) / (2 * step))
    values, grads, dets, fd = np.array(values), np.array(grads), np.array(dets), np.array(fd)
    assert np.sign(dets).min() < 0 < np.sign(dets).max(), "the sweep did not cross the critical curve"
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(grads))
    # differentiable everywhere along the path, including where det A changes sign: autodiff equals the local finite
    # difference at every point
    np.testing.assert_allclose(grads[:, i_v], fd, rtol=2e-4, atol=1e-3)


def test_fold_pair_in_a_two_knot_system_survives_a_displaced_source(tmp_path):
    """Knot A is a 4 mas fold pair.  With its source sampled four conditional sigmas off the centroid, the lens alone can
    lose the pair; the lens plus the GP mean deflection keeps both images for every direction of the offset."""
    from lenscluster.diagnostics import predictions_at
    from lenscluster.exact_images import ExactImageSolver
    config, _ = two_knot_config(tmp_path, depth=0.004)
    config = knee_config(config, "system")
    state = build_state(config)
    evaluator = Evaluator(state, "gp_image_plane")
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    names = [p.name for p in state.parameter_specs]
    pair = list(state.images.family_ids).index("1A")
    # The solver corrects the mass field's mean deflection only; an image-plane displacement d is not applied to the
    # recovered positions (as for the image-plane GP today), so keep the image-plane terms small here.
    start = set_sites(state, state.initial_theta, sigma_shared=1e-3, sigma_white=1e-3)
    kept = 0
    for ux, uy in ((4, 4), (4, -4), (-4, 4), (-4, -4)):
        theta = np.array(start, dtype=float)
        theta[names.index("source.1A.beta_x")], theta[names.index("source.1A.beta_y")] = ux, uy
        fixed = predictions_at(solver, state, evaluator, theta, adaptive=False)
        kept += fixed["count_rows"][pair]["recovered_image_count"] == 2
    assert kept == 4
    assert evaluator.gp_deflection_correction(state.initial_theta) is not None


# ----------------------------------------------------------------------------- 4. speed
def test_knee_on_all_images_costs_under_three_times_the_family_gp(tmp_path):
    base = big_config(tmp_path)                                  # 40 families, 160 images
    timings = {}
    for label, config in (("image_plane", replace(base, likelihood=replace(base.likelihood, likelihood_mode="gp_image_plane")).validate()),
                          ("knee/all", knee_config(base, "all")), ("knee/system", knee_config(base, "system"))):
        state = build_state(config)
        evaluator = Evaluator(state, "gp_image_plane")
        theta = jnp.asarray(state.initial_theta)
        evaluator.value_and_grad(theta)[0].block_until_ready()
        best = np.inf
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(10):
                evaluator.value_and_grad(theta)[0].block_until_ready()
            best = min(best, (time.perf_counter() - start) / 10)
        timings[label] = best
    # The toy lens costs nothing, so the covariance dominates here; at M0416 size the measured total is 10.7 ms against about
    # 4 ms for the lens alone.  Bounds: the coupled covariance of 160 images under 8 ms, system blocks under 2x the family GP.
    assert timings["knee/all"] < 8e-3 and timings["knee/system"] < 2.0 * timings["image_plane"], timings
