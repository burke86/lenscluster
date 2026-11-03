"""Heavy-tailed member deviations (scaling.member_deviation_tail = "student_t") and the smooth ellipticity cap of a halo
(DPIEHaloConfig.ellipticity_soft_cap): the Student-t prior is the one sampled and drawn, a large dof gives back the
normal, and the cap is flat below its modulus, falls smoothly above it and keeps the likelihood and gradient finite."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from lenscluster.config import PriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.inputs import lenstool_ellipticite_to_axis_ratio
from lenscluster.sampling import distribution_for_site, sample_sites
from lenscluster.state import build_state


def _tail(config, tail="student_t", dof=4.0):
    return replace(config, scaling=replace(config.scaling, member_deviation_tail=tail, member_deviation_dof=dof))


def _offset_site(state):
    sites = sample_sites(state.parameter_specs)
    return next(s for s in sites if all(state.parameter_specs[i].component_family == "member_scatter_offset" for i in s.indices))


def test_student_t_deviations_are_the_sampled_prior(synthetic_config):
    state = build_state(_tail(synthetic_config).validate())
    offsets = [s for s in state.parameter_specs if s.component_family == "member_scatter_offset"]
    assert offsets and all(s.prior_kind == "student_t" and s.dof == 4.0 for s in offsets)
    site = _offset_site(state)
    value = jnp.linspace(-6.0, 6.0, len(site.indices))
    prior = distribution_for_site(site, state.parameter_specs)
    expected = stats.t(df=4.0).logpdf(np.asarray(value)).sum()
    np.testing.assert_allclose(float(prior.log_prob(value)), expected, rtol=1e-10)
    evaluator = Evaluator(state)
    value, grad = jax.value_and_grad(evaluator.log_likelihood)(jnp.asarray(state.initial_theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad)))


def test_a_large_dof_gives_back_the_normal(synthetic_config):
    normal = build_state(synthetic_config.validate())
    wide = build_state(_tail(synthetic_config, dof=1.0e7).validate())
    a, b = _offset_site(normal), _offset_site(wide)
    value = jnp.linspace(-3.0, 3.0, len(a.indices))
    np.testing.assert_allclose(float(distribution_for_site(b, wide.parameter_specs).log_prob(value)),
                               float(distribution_for_site(a, normal.parameter_specs).log_prob(value)), rtol=1e-5)


def _capped(config, cap=0.9, upper=0.94):
    halo = config.model.large_halos[0]
    priors = {**halo.priors, "ellipticite": PriorConfig("uniform", lower=0.0, upper=upper), "angle_pos": PriorConfig("uniform", lower=-90.0, upper=90.0)}
    halo = replace(halo, priors=priors, ellipticity_soft_cap=cap, ellipticity_soft_width=0.02)
    return replace(config, model=replace(config.model, large_halos=(halo,)))


def test_the_ellipticity_cap_is_flat_below_and_smooth_above(synthetic_config):
    state = build_state(_capped(synthetic_config).validate())
    evaluator = Evaluator(state)
    q = float(lenstool_ellipticite_to_axis_ratio(0.9))
    cap = (1 - q) / (1 + q)
    np.testing.assert_allclose(state.packed_lens_spec.ellipticity_soft_cap[0], cap)
    names = [s.name for s in state.parameter_specs]
    i1 = names.index("halo.e1")

    def prior_at(e1):
        theta = np.array(state.initial_theta, dtype=float)
        theta[names.index("halo.e2")] = 0.0
        theta[i1] = e1
        packed, _ = evaluator.packed_state(evaluator.physical(jnp.asarray(theta)))
        return float(evaluator.ellipticity_log_prior(packed)), theta

    low, _ = prior_at(0.2)
    high, theta = prior_at(cap + 0.04)
    assert abs(low) < 1e-6                                                 # flat well below the cap
    np.testing.assert_allclose(high, -np.logaddexp(0.0, 2.0), rtol=1e-8)  # 2 widths above: -softplus(2)
    value, grad = jax.value_and_grad(evaluator.log_likelihood)(jnp.asarray(theta))
    assert np.isfinite(float(value)) and bool(jnp.all(jnp.isfinite(grad))) and float(grad[i1]) < 0.0   # pushed back below the cap


def test_a_cap_above_the_box_is_refused(synthetic_config):
    with pytest.raises(ValueError, match="ellipticity_soft_cap"):
        _capped(synthetic_config, cap=0.95, upper=0.94).validate()
