"""Member deviation modes and the centered/non-centered switch leave the physical model unchanged."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.evaluator import Evaluator
from lenscluster.model import member_log_deviations
from lenscluster.sampling import prior_log_prob
from lenscluster.state import build_state, with_member_classes


@pytest.mark.parametrize("mode", ["sigma", "mass", "sigma_rcut", "mass_rcut"])
def test_centered_conversion_preserves_the_likelihood(synthetic_config, mode):
    config = replace(synthetic_config, scaling=replace(synthetic_config.scaling, member_deviation=mode))
    state = build_state(config)
    state = with_member_classes(state, state.members.exact_positions, centered=np.ones(4, dtype=bool))   # all centered
    assert state.members.centered.all()
    theta = np.asarray(state.initial_theta) + np.random.default_rng(0).normal(0, 0.1, state.n_parameters)
    state.initial_theta = theta
    value = float(Evaluator(state).loglike(jnp.asarray(theta)))
    switched = with_member_classes(state, state.members.exact_positions, centered=np.asarray([True, True, False, False]))
    assert switched.members.centered.tolist() == [True, True, False, False]
    assert np.isfinite(value)
    np.testing.assert_allclose(float(Evaluator(switched).loglike(jnp.asarray(switched.initial_theta))), value, rtol=1e-10, atol=1e-8)
    assert np.isfinite(float(prior_log_prob(switched.parameter_specs, jnp.asarray(switched.initial_theta))))
    expected = {"sigma": 19, "mass": 19, "sigma_rcut": 24, "mass_rcut": 24}[mode]   # incl. sigma_def, sigma_extra, sigma_sel, lambda_sel
    assert state.n_parameters == expected


def test_mass_mode_moves_radii_with_the_dispersion(synthetic_config):
    from lenscluster.model import member_log_deviations
    config = replace(synthetic_config, scaling=replace(synthetic_config.scaling, member_deviation="mass"))
    state = build_state(config)
    state = with_member_classes(state, state.members.exact_positions, centered=np.ones(4, dtype=bool))
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta).copy()
    theta[state.members.offset_param_index[0]] = 0.3                 # centered: a 0.3 deviation in ln mass
    physical = evaluator.physical(jnp.asarray(theta))
    dv, dr = member_log_deviations(state.packed_lens_spec, physical)
    c = int(state.members.component_indices[0])
    alpha = float(physical[int(state.packed_lens_spec.alpha_sigma_param_index[c])])
    beta = float(physical[int(state.packed_lens_spec.beta_radius_param_index[c])])
    np.testing.assert_allclose(float(dv[c]), 0.3 * alpha / (2 * alpha + beta))
    np.testing.assert_allclose(float(dr[c]), 0.3 * beta / (2 * alpha + beta))
    np.testing.assert_allclose(2 * float(dv[c]) + float(dr[c]), 0.3)


def test_far_field_members_keep_the_relation_radius(synthetic_config):
    config = replace(synthetic_config, scaling=replace(synthetic_config.scaling, member_deviation="sigma_rcut"))
    state = with_member_classes(build_state(config), np.asarray([0, 1], dtype=np.int32))
    theta = np.asarray(state.initial_theta).copy()
    theta[np.asarray(state.members.radius_offset_param_index)] = 2.0
    evaluator = Evaluator(state)
    _, delta_r = member_log_deviations(state.packed_lens_spec, evaluator.physical(jnp.asarray(theta)))
    delta_r = np.asarray(delta_r)[np.asarray(state.members.component_indices)]
    assert np.all(delta_r[:2] > 0) and np.all(delta_r[2:] == 0)
