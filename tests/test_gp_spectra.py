"""The alternative mass-field spectra (LikelihoodConfig.gp_knee_spectrum): the broken power law with beta_low = 0 is the
knee, two knees with a vanishing second amplitude are one knee, the power law does not depend on the knee axis, the
foreground field carries redshift weight 1, every variant gives a finite likelihood and gradient, and the configurations
that would mislabel or misplace a component are refused."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import gp_knee
from lenscluster.config import LogNormalPriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from test_gp_knee import knee_config, set_sites, two_knot_config

WIDE = dict(gp_knee_scales_arcsec=(0.2, 100.0), gp_knee_prior=LogNormalPriorConfig(median=20.0, log_sigma=1.0, upper=80.0),
            gp_knee_start_arcsec=20.0)
TABLE = dict(r_max=30.0, r_step=0.05, lam_min=0.5, lam_max=80.0, n_lam=24, scale_min=0.2, scale_max=100.0, n_nodes=1500)


def _variant(config, spectrum, **likelihood):
    extra = {"two_knee": dict(gp_knee_gamma_prior=None, gp_knee_prior=LogNormalPriorConfig(median=20.0, log_sigma=1.0, upper=80.0, lower=10.0)),
             "knee_foreground": dict(gp_knee_gamma_prior=None),
             "broken_power_law": dict(gp_knee_gamma_prior=None),
             "power_law": dict(gp_knee_gamma_prior=(0.5, 6.0), gp_knee_gamma=4.0),
             "knee": dict(gp_knee_gamma_prior=None)}[spectrum]
    return knee_config(config, "all", gp_knee_spectrum=spectrum, **{**WIDE, **extra, **likelihood})


def _kernel(evaluator, theta):
    physical = evaluator.physical(jnp.asarray(theta))
    return np.asarray(evaluator.gp_kernel_blocks(physical, evaluator.error_scales(physical))["k_alpha"])


def _set(state, theta, name, value, log=True):
    out = np.array(theta, dtype=float)
    out[[s.name for s in state.parameter_specs].index(name)] = np.log(value) if log else value
    return out


def test_broken_power_law_with_flat_low_end_is_the_knee():
    knee = gp_knee.knee_tables(**TABLE, gamma=4.0)
    broken = gp_knee.knee_tables(**TABLE, gamma=4.0, shape="broken_power_law", beta_low_grid=(-1.0, 3.0, 17))
    zero = int(np.argmin(np.abs(np.asarray(broken["gamma_grid"]))))
    assert float(broken["gamma_grid"][zero]) == 0.0
    for key in ("c_l", "c_t"):
        np.testing.assert_allclose(np.asarray(broken[key])[:, zero, :], np.asarray(knee[key])[:, 0, :], rtol=0, atol=1e-12)
    # and through the evaluator's spline in the low-k slope, at a knee between the nodes
    lam = jnp.asarray(13.7)
    for a, b in zip(gp_knee.knee_curves(broken, lam, jnp.asarray(0.0)), gp_knee.knee_curves(knee, lam)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-10)


def test_power_law_tables_do_not_depend_on_the_knee_axis():
    tables = gp_knee.knee_tables(**TABLE, gamma_grid=(1.0, 5.0, 9), shape="power_law")
    c_l = np.asarray(tables["c_l"])
    np.testing.assert_allclose(c_l, np.broadcast_to(c_l[:, :, :1], c_l.shape), atol=1e-12)
    np.testing.assert_allclose(c_l[0], 1.0, atol=1e-12)


def test_two_knees_with_a_vanishing_second_field_are_one_knee(tmp_path):
    config, _ = two_knot_config(tmp_path)
    one, two = build_state(_variant(config, "knee")), build_state(_variant(config, "two_knee"))
    e1, e2 = Evaluator(one, "gp_image_plane"), Evaluator(two, "gp_image_plane")
    t1 = _set(one, one.initial_theta, "images.gp_knee", 20.0)
    t2 = _set(two, _set(two, two.initial_theta, "images.gp_knee", 20.0), "images.gp_sigma_alpha2", 1.0e-4)
    np.testing.assert_allclose(_kernel(e2, t2), _kernel(e1, t1), rtol=0, atol=2e-8)      # the second field's variance is 1e-8
    # with a real second amplitude the kernel is the sum of the two fields
    t3 = _set(two, t2, "images.gp_sigma_alpha2", 0.05)
    lone = _set(one, _set(one, one.initial_theta, "images.gp_knee", 3.0), "images.gp_sigma_alpha", 0.05)
    np.testing.assert_allclose(_kernel(e2, t3), _kernel(e1, t1) + _kernel(e1, lone), rtol=1e-6, atol=1e-12)


def test_the_foreground_field_has_redshift_weight_one(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(_variant(config, "knee_foreground"))
    evaluator = Evaluator(state, "gp_image_plane")
    theta = _set(state, state.initial_theta, "images.gp_sigma_alpha", 1.0e-4)
    k = _kernel(evaluator, theta)
    knee_only = Evaluator(build_state(_variant(config, "knee")), "gp_image_plane")
    reference = knee_only.state
    alone = _set(reference, _set(reference, reference.initial_theta, "images.gp_knee", 3.0), "images.gp_sigma_alpha", 0.05)
    w = np.asarray(knee_only.gp["weights"])
    unweighted = _kernel(knee_only, alone) / (w[:, :, None] * w[:, None, :])[..., None, None]
    mask = np.asarray(knee_only.gp["pair_mask"])[..., None, None] > 0
    np.testing.assert_allclose(np.where(mask, k, 0.0), np.where(mask, unweighted, 0.0), rtol=0, atol=2e-8)   # the main field adds 1e-8


@pytest.mark.parametrize("spectrum", ["two_knee", "broken_power_law", "power_law", "knee_foreground"])
def test_every_variant_gives_a_finite_likelihood_and_gradient(tmp_path, spectrum):
    config, _ = two_knot_config(tmp_path)
    state = build_state(_variant(config, spectrum))
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    assert ("images.gp_knee" in names) == (spectrum != "power_law")
    assert ("images.gp_sigma_alpha2" in names) == (spectrum in ("two_knee", "knee_foreground"))
    assert ("images.gp_knee_beta_low" in names) == (spectrum == "broken_power_law")
    value, grad = jax.jit(jax.value_and_grad(evaluator.log_likelihood))(jnp.asarray(state.initial_theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))
    assert evaluator.gp_deflection_correction(state.initial_theta) is None


def test_configurations_that_would_mix_up_the_components_are_refused(tmp_path):
    config, _ = two_knot_config(tmp_path)
    with pytest.raises(ValueError, match="cannot swap"):
        knee_config(config, "all", gp_knee_spectrum="two_knee", gp_knee_gamma_prior=None, **WIDE)          # no lower bound on the large knee
    with pytest.raises(ValueError, match="fixes the high-k slope"):
        knee_config(config, "all", gp_knee_spectrum="broken_power_law", **WIDE)
    with pytest.raises(ValueError, match="one redshift weighting"):
        knee_config(config, "all", gp_knee_spectrum="knee_foreground", gp_knee_gamma_prior=None, gp_knee_project_modes=("shear",), **WIDE)


def test_a_fixed_knee_and_length_equal_the_sampled_model_at_those_values(tmp_path):
    config, _ = two_knot_config(tmp_path)
    free = build_state(knee_config(config, "all", gp_knee_gamma_prior=None))
    fixed = build_state(knee_config(config, "all", gp_knee_gamma_prior=None, gp_knee_fixed_arcsec=6.0, gp_knee_ell_fixed_arcsec=1.0))
    names_free, names_fixed = [s.name for s in free.parameter_specs], [s.name for s in fixed.parameter_specs]
    assert sorted(set(names_free) - set(names_fixed)) == ["images.gp_ell", "images.gp_knee"]
    theta = _set(free, _set(free, free.initial_theta, "images.gp_knee", 6.0), "images.gp_ell", 1.0)
    reduced = np.array([theta[names_free.index(n)] for n in names_fixed])
    values = [float(Evaluator(s, "gp_image_plane").log_likelihood(jnp.asarray(t))) for s, t in ((free, theta), (fixed, reduced))]
    assert values[0] == values[1]
    with pytest.raises(ValueError, match="no knee"):
        knee_config(config, "all", gp_knee_spectrum="power_law", gp_knee_fixed_arcsec=6.0, **WIDE)
