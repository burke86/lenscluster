"""Elliptical NFW cluster halos: kernel, physical normalization and the end-to-end model."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jaxtronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE

from lenscluster import jax_lensing
from lenscluster.config import NFWHaloConfig, PriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.exact_images import packed_to_kwargs_lens
from lenscluster.jax_cosmology import flat_wcdm_config, kpc_per_arcsec_from_config
from lenscluster.profiles import nfw_lenstronomy_parameters
from lenscluster.state import build_state


def test_nfw_kernel_matches_jaxtronomy_and_finite_differences():
    rng = np.random.default_rng(5)
    x, y = (jnp.asarray(rng.uniform(-40, 40, 30)) for _ in range(2))
    rs, e1, e2, cx, cy = 60.0, 0.12, -0.2, 3.0, -1.5
    alpha_rs = jnp.asarray(rng.uniform(5, 10, 30))                      # per-point normalization, as per image
    alpha, u, du = jax_lensing.nfw_fields(x, y, rs, alpha_rs, e1, e2, cx, cy)
    ref = NFW_ELLIPSE_CSE(high_accuracy=True)
    ax, ay = ref.derivatives(x, y, rs, 1.0, e1, e2, cx, cy)
    np.testing.assert_allclose(alpha, np.stack((ax, ay), -1) * alpha_rs[:, None], rtol=1e-12)
    fxx, fxy, fyx, fyy = ref.hessian(x, y, rs, 1.0, e1, e2, cx, cy)
    np.testing.assert_allclose(u[:, 0, 0], fxx * alpha_rs, rtol=1e-12)
    np.testing.assert_allclose(u[:, 0, 1], fxy * alpha_rs, rtol=1e-12)
    np.testing.assert_allclose(u[:, 1, 1], fyy * alpha_rs, rtol=1e-12)
    h = 1e-4
    for l, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        plus = jax_lensing.nfw_fields(x + dx, y + dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[1]
        minus = jax_lensing.nfw_fields(x - dx, y - dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[1]
        np.testing.assert_allclose(du[..., l], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-9)
    alpha_only, u_only, none = jax_lensing.nfw_fields(x, y, rs, alpha_rs, e1, e2, cx, cy, derivative=False)
    assert none is None
    np.testing.assert_array_equal(alpha_only, alpha)
    # Gradients with respect to the ellipticity are finite at e = 0 and match finite differences away from it.
    total = lambda e: jnp.sum(jax_lensing.nfw_fields(x, y, rs, alpha_rs, e[0], e[1], cx, cy, derivative=False)[0] ** 2)
    assert np.all(np.isfinite(np.asarray(jax.grad(total)(jnp.zeros(2)))))
    e = jnp.array([0.12, -0.2])
    g = np.asarray(jax.grad(total)(e))
    for k in range(2):
        step = jnp.zeros(2).at[k].set(1e-6)
        np.testing.assert_allclose(g[k], (total(e + step) - total(e - step)) / 2e-6, rtol=1e-5)


def test_nfw_physical_normalization_matches_lenstronomy():
    from astropy.cosmology import FlatLambdaCDM
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    z_lens, z_source, m200, c = 0.5, 2.0, 1.0e15, 4.0
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3))
    _, rs_mpc, _ = lens_cosmo.nfwParam_physical(m200, c)
    rs_angle_ref, alpha_rs_ref = lens_cosmo.nfw_physical2angle(m200, c)
    cosmo = flat_wcdm_config(h0=70.0, om0=0.3)
    rs_angle, alpha_rs = nfw_lenstronomy_parameters(np.log10(m200), rs_mpc * 1000.0, z_lens=z_lens, z_source=z_source,
                                                    cosmo_config=cosmo, kpc_per_arcsec=kpc_per_arcsec_from_config(z_lens, cosmo))
    np.testing.assert_allclose(rs_angle, rs_angle_ref, rtol=3e-3)
    np.testing.assert_allclose(alpha_rs, alpha_rs_ref, rtol=3e-3)


def test_nfw_cluster_halo_end_to_end(synthetic_config):
    halo = NFWHaloConfig(id="halo", x_centre=0.0, y_centre=0.0, ellipticite=0.2, angle_pos=0.0, log10_m200=14.6, scale_radius_kpc=300.0, z_lens=0.3,
                         priors={"log10_m200": PriorConfig("uniform", lower=14.0, upper=15.2),
                                 "scale_radius_kpc": PriorConfig("truncated_normal", mean=300.0, std=100.0, lower=100.0, upper=600.0),
                                 "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.6), "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0)})
    config = replace(synthetic_config, model=replace(synthetic_config.model, large_halos=(halo,))).validate()
    state = build_state(config)
    names = [s.name for s in state.parameter_specs]
    assert {"halo.log10_m200", "halo.scale_radius_kpc", "halo.e1", "halo.e2"} <= set(names)
    assert state.lens_model_list[0] == "NFW_ELLIPSE_CSE"
    evaluator = Evaluator(state)
    theta = jnp.asarray(state.initial_theta)
    value, grad = evaluator.value_and_grad(theta)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad)))
    for name in ("halo.log10_m200", "halo.scale_radius_kpc"):
        assert float(grad[names.index(name)]) != 0.0
    # The exact-image solver's lenstronomy model reproduces the evaluator's deflections.
    from lenstronomy.LensModel.lens_model import LensModel
    z_source = float(state.z_source_values[0])
    packed = evaluator.packed_state_for_z(evaluator.physical(theta), z_source)
    kwargs = packed_to_kwargs_lens(packed)
    assert set(kwargs[0]) == {"Rs", "alpha_Rs", "e1", "e2", "center_x", "center_y"} and kwargs[0]["alpha_Rs"] > 0
    points = np.array([[3.0, -2.0], [-8.0, 5.0], [15.0, 12.0]])
    alpha, _ = evaluator.total_fields_at(points, packed)
    ax, ay = LensModel(lens_model_list=state.lens_model_list).alpha(points[:, 0], points[:, 1], kwargs)
    np.testing.assert_allclose(alpha, np.stack((ax, ay), -1), rtol=1e-8, atol=1e-10)
