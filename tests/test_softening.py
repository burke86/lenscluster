"""The softening length blurs member cores without a validity wall and without changing member masses."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state


def _evaluator(config, softening_kpc):
    scaling = replace(config.scaling, softening_length_kpc=softening_kpc)
    population = replace(config.model.member_populations[0], cutkpc=1.0,
                         cutkpc_prior=replace(config.model.member_populations[0].cutkpc_prior, mean=1.0, std=0.3, lower=0.3, upper=3.0))
    model = replace(config.model, member_populations=(population,))
    state = build_state(replace(config, scaling=scaling, model=model).validate())
    return state, Evaluator(state)


def test_softened_members_stay_valid_and_keep_their_mass(synthetic_config):
    state0, plain = _evaluator(synthetic_config, 0.0)
    state1, soft = _evaluator(synthetic_config, 3.0)           # softening 3 kpc > member cut radii of ~1 kpc
    theta = jnp.asarray(state1.initial_theta)
    physical = soft.physical(theta)
    packed, valid = soft.packed_state(physical)
    assert bool(valid.is_valid), np.asarray(valid.reason_flags)
    members = np.asarray(state1.packed_lens_spec.component_family) == 1
    assert np.all(np.asarray(packed.Rs)[members] > np.asarray(packed.Ra)[members])
    value, grad = soft.value_and_grad(theta)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad)))
    # Far from a member its deflection depends only on the total mass pi sigma^2 r_cut / G, so softening must not change it.
    names0 = [s.name for s in state0.parameter_specs]
    theta0 = np.asarray(state0.initial_theta)
    shared = [i for i, s in enumerate(state1.parameter_specs) if s.name in names0]
    theta0[[names0.index(state1.parameter_specs[i].name) for i in shared]] = np.asarray(theta)[shared]
    z = float(state1.z_source_values[0])
    packed0 = plain.packed_state_for_z(plain.physical(jnp.asarray(theta0)), z)
    packed1 = soft.packed_state_for_z(physical, z)
    member = int(np.flatnonzero(members)[0])
    centre = np.array([state1.packed_lens_spec.x_center_base[member], state1.packed_lens_spec.y_center_base[member]])
    points = centre + np.array([[40.0, 0.0], [0.0, -60.0], [30.0, 30.0]])
    a0, _ = plain.total_fields_at(points, packed0, np.asarray([member], dtype=np.int32))
    a1, _ = soft.total_fields_at(points, packed1, np.asarray([member], dtype=np.int32))
    np.testing.assert_allclose(a1, a0, rtol=3e-3, atol=1e-6)   # tangential parts are ~1e-7; the radial part is the mass
