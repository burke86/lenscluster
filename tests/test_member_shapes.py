"""Sampled member shapes (ScalingModelConfig.member_shape_scatter): e = e_light + s u on exact members only."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state, with_member_classes

SCATTER = 0.1


def _states(config):
    fixed = build_state(config)
    sampled = build_state(replace(config, scaling=replace(config.scaling, member_shape_scatter=SCATTER)).validate())
    n = len(fixed.members.member_ids)
    exact = np.arange(n)[:2]                                    # the two members near the images (conftest)
    fixed = with_member_classes(fixed, exact, dict(fixed.members.classification))
    sampled = with_member_classes(sampled, exact, dict(sampled.members.classification))
    return fixed, sampled, exact


def _packed_shapes(evaluator, theta):
    packed, _ = evaluator.packed_state(evaluator.physical(jnp.asarray(theta)), factor_per_image=1.0)
    return np.asarray(packed.e1), np.asarray(packed.e2)


def test_two_offsets_per_member_and_only_exact_members_active(synthetic_config):
    fixed, sampled, exact = _states(synthetic_config)
    shape_specs = [s for s in sampled.parameter_specs if s.component_family == "member_shape_offset"]
    n = len(sampled.members.member_ids)
    assert len(shape_specs) == 2 * n and len(sampled.parameter_specs) == len(fixed.parameter_specs) + 2 * n
    assert {s.sample_site_name for s in shape_specs} == {"member_shape_e1_offset", "member_shape_e2_offset"}
    active = np.asarray(sampled.packed_lens_spec.shape_active)
    components = np.asarray(sampled.members.component_indices)
    assert active[components[exact]].all() and not np.delete(active[components], exact).any()
    assert active.sum() == len(exact)


def test_zero_offsets_reproduce_the_fixed_shapes(synthetic_config):
    fixed, sampled, _ = _states(synthetic_config)
    a, b = Evaluator(fixed), Evaluator(sampled)
    np.testing.assert_allclose(float(b.log_likelihood(jnp.asarray(sampled.initial_theta))),
                               float(a.log_likelihood(jnp.asarray(fixed.initial_theta))), rtol=1e-12)


def test_offsets_move_exact_members_by_the_scatter_and_leave_far_members(synthetic_config):
    _, sampled, exact = _states(synthetic_config)
    evaluator = Evaluator(sampled)
    names = [s.name for s in sampled.parameter_specs]
    components = np.asarray(sampled.members.component_indices)
    ids = sampled.members.member_ids
    e1_0, e2_0 = _packed_shapes(evaluator, sampled.initial_theta)
    theta = np.array(sampled.initial_theta, dtype=float)
    near, far = int(exact[0]), int(len(ids) - 1)
    packed_spec = sampled.packed_lens_spec
    index = {"e1": np.asarray(packed_spec.shape_e1_offset_param_index), "e2": np.asarray(packed_spec.shape_e2_offset_param_index)}
    pick = lambda position, comp: int(index[comp][components[position]])
    assert names[pick(near, "e1")].endswith("shape_e1_offset") and names[pick(far, "e2")].endswith("shape_e2_offset")
    theta[pick(near, "e1")] = 1.5
    theta[pick(near, "e2")] = -0.5
    theta[pick(far, "e1")] = 2.0
    e1, e2 = _packed_shapes(evaluator, theta)
    assert e1[components[near]] == pytest.approx(e1_0[components[near]] + 1.5 * SCATTER)
    assert e2[components[near]] == pytest.approx(e2_0[components[near]] - 0.5 * SCATTER)
    assert e1[components[far]] == pytest.approx(e1_0[components[far]])        # far-field: the light shape
    assert float(evaluator.log_likelihood(jnp.asarray(theta))) != pytest.approx(float(evaluator.log_likelihood(jnp.asarray(sampled.initial_theta))))
