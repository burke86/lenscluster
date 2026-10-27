"""The sampled error scales: sigma_extra in quadrature with sigma_inst, the caps, and the optional sigma_sel."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from lenscluster.config import LogNormalPriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state


def test_sigma_extra_adds_in_quadrature_to_sigma_inst(synthetic_config):
    likelihood = replace(synthetic_config.likelihood, sigma_extra_prior=LogNormalPriorConfig(median=0.05, log_sigma=1.0))
    config = replace(synthetic_config, likelihood=likelihood).validate()
    assert build_state(replace(synthetic_config, likelihood=replace(likelihood, sigma_extra_prior=None))).sigma_extra_param_index == -1
    state = build_state(config)
    names = [s.name for s in state.parameter_specs]
    index = names.index("images.sigma_extra")
    assert state.sigma_extra_param_index == index
    spec = state.parameter_specs[index]
    assert spec.prior_kind == "normal" and spec.transform_kind == "log_positive" and np.isclose(spec.mean, np.log(0.05))   # no cap requested here
    capped = build_state(replace(config, likelihood=replace(likelihood, sigma_extra_prior=LogNormalPriorConfig(median=0.05, log_sigma=1.0, upper=1.0))))
    cap = capped.parameter_specs[[s.name for s in capped.parameter_specs].index("images.sigma_extra")]
    assert cap.prior_kind == "truncated_normal" and np.isclose(cap.upper, 0.0) and cap.physical_upper == 1.0 and cap.physical_lower > 0
    src = capped.parameter_specs[[s.name for s in capped.parameter_specs].index("images.sigma_def")]
    assert src.prior_kind == "truncated_normal" and src.physical_upper == 1.0                                              # the library default caps sigma_def at 1"
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta).copy()

    sigma_catalog = float(state.images.sigma_inst_per_image[0])
    doubling = np.sqrt(3.0) * sigma_catalog                      # sqrt(sigma^2 + s^2) = 2 sigma

    def at(scatter):
        t = theta.copy(); t[index] = np.log(scatter); return jnp.asarray(t)

    stats = evaluator.statistics(at(doubling))
    np.testing.assert_allclose(np.asarray(stats["sigma_meas"]), 2.0 * np.asarray(state.images.sigma_inst_per_image))
    assert np.all(np.asarray(evaluator.statistics(at(1e-6))["sigma_meas"]) >= np.asarray(state.images.sigma_inst_per_image))   # the instrumental error is a floor
    value, grad = evaluator.value_and_grad(at(0.05))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad))) and float(grad[index]) != 0.0
    # Scale 2 is exactly the fixed-error model with the catalog errors doubled.
    import dataclasses
    fixed_state = build_state(replace(synthetic_config, likelihood=replace(synthetic_config.likelihood, sigma_extra_prior=None)))
    images = fixed_state.images
    doubled = images._replace(sigma_inst_per_image=2.0 * images.sigma_inst_per_image) if hasattr(images, "_replace") else dataclasses.replace(images, sigma_inst_per_image=2.0 * images.sigma_inst_per_image)
    fixed_state = dataclasses.replace(fixed_state, images=doubled)
    fixed = Evaluator(fixed_state)
    fixed_names = [s.name for s in fixed_state.parameter_specs]
    fixed_theta = np.asarray(fixed_state.initial_theta).copy()
    for i, name in enumerate(names):
        if name != "images.sigma_extra":
            fixed_theta[fixed_names.index(name)] = theta[i]
    np.testing.assert_allclose(float(evaluator.loglike(at(doubling))), float(fixed.loglike(jnp.asarray(fixed_theta))), rtol=1e-12)
    assert abs(float(evaluator.loglike(at(doubling))) - float(evaluator.loglike(at(0.05)))) > 1.0


def test_sigma_sel_is_absent_by_default_and_sampled_when_configured(synthetic_config):
    state = build_state(synthetic_config)
    assert state.sigma_sel_param_index == -1 and not any(s.field == "sigma_sel" for s in state.parameter_specs)
    enabled = replace(synthetic_config, likelihood=replace(synthetic_config.likelihood, sigma_sel_prior=LogNormalPriorConfig(median=0.04, log_sigma=1.0, upper=1.0),
                                                           sigma_res_arcsec=0.03)).validate()
    state = build_state(enabled)
    index = state.sigma_sel_param_index
    assert state.parameter_specs[index].name == "images.sigma_sel" and state.sigma_res_arcsec == 0.03
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta).copy()
    stats = evaluator.statistics(jnp.asarray(theta))
    assert float(stats["sigma_sel"]) > 0 and np.isfinite(float(evaluator.loglike(jnp.asarray(theta))))
    value, grad = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and float(grad[index]) != 0.0
