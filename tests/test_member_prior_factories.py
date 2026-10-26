"""The member-relation prior factories: Bergamini 2019 and the neutral set, and how they reach the sampled specs."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from lenscluster.config import MemberPopulationConfig, ScalingModelConfig
from lenscluster.state import build_state


def _population(synthetic_config, factory):
    base = synthetic_config.model.member_populations[0]
    return factory("members", base.catalog_path, base.z_lens, prior_mag0=20.0)


def test_neutral_factories_give_log_uniform_cut_and_scaling_slopes(synthetic_config):
    population = _population(synthetic_config, MemberPopulationConfig.neutral)
    config = replace(synthetic_config, model=replace(synthetic_config.model, member_populations=(population,)), scaling=ScalingModelConfig.neutral())
    state = build_state(config)
    specs = {spec.field: spec for spec in state.parameter_specs if spec.component_family == "scaling"}
    cut = specs["cutkpc"]
    assert cut.prior_kind == "uniform" and cut.transform_kind == "log_positive"
    factor = float(state.potfiles[0]["pivot"]["cutkpc_factor"])               # the bounds move to the pivot with the rest of the prior
    assert math.isclose(math.exp(cut.lower), 2.0 * factor, rel_tol=1e-6) and math.isclose(math.exp(cut.upper), 100.0 * factor, rel_tol=1e-6)
    sigma = specs["sigma"]
    assert sigma.prior_kind == "truncated_normal" and sigma.physical_lower > 0 and sigma.physical_upper < np.inf
    assert specs["alpha_sigma"].mean == 0.25 and specs["alpha_sigma"].std == 0.10 and specs["beta_radius"].mean == 0.5 and specs["beta_radius"].std == 0.25


def test_bergamini2019_factories_match_the_defaults(synthetic_config):
    population = _population(synthetic_config, MemberPopulationConfig.bergamini2019)
    assert population.sigma_prior.mean == 200.0 and population.cutkpc_prior.mean == 12.0 and population.cutkpc_prior.upper == 60.0
    assert ScalingModelConfig.bergamini2019() == ScalingModelConfig()
    assert ScalingModelConfig.neutral(fit_corekpc=False).fit_corekpc is False
