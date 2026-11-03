"""Guards against the w1000s500_prod failure (the error model absorbing the lens): stage A of the init search holds the
error scales, the search runs on system blocks under grf_gp, the grf_gp hyperparameters have upper bounds that fit the
knee tables.  The init no longer stops a run: the start-parameter table colours a value near its prior bound red."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lenscluster import search_split
from lenscluster.config import LikelihoodConfig, LogNormalPriorConfig, SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from test_gp_knee import knee_config


def test_stage_a_holds_the_error_scales_and_the_search_uses_system_blocks(synthetic_config, monkeypatch):
    config = replace(knee_config(synthetic_config, "all"), search=SearchConfig(n_starts=3, member_starts=2, lbfgs_max_iter=20, polish_max_iter=10))
    state = build_state(config)
    calls, original = [], search_split.multistart_map

    def recording(state_, evaluator_, starts, search, **kwargs):
        result = original(state_, evaluator_, starts, search, **kwargs)
        calls.append(dict(label=kwargs["label"], state=state_, evaluator=evaluator_, starts=np.asarray(starts), result=result,
                          active=kwargs.get("active_sites"), polish=kwargs.get("polish_sites")))
        return result

    monkeypatch.setattr(search_split, "multistart_map", recording)
    search_split.split_map(config, state, Evaluator(state, "gp_image_plane"), seed=1, log=lambda *a, **k: None)
    stage_a, stage_b = calls
    assert stage_a["label"] == "stage A" and stage_b["label"] == "stage B"
    assert state.gp_blocks == "all"
    assert stage_a["state"].gp_blocks == stage_b["state"].gp_blocks == "system"
    assert stage_a["evaluator"].gp_blocks == stage_b["evaluator"].gp_blocks == "system"
    specs = stage_a["state"].parameter_specs
    error = [i for i, s in enumerate(specs) if s.component_family == "error_scale"]
    error_sites = {specs[i].sample_site_name or specs[i].sample_name for i in error}
    assert {"images_gp_sigma_alpha", "images_gp_knee", "images_gp_knee_gamma", "images_sigma_shared"} <= error_sites
    assert not error_sites & stage_a["active"] and stage_a["polish"] == stage_a["active"]
    # every stage A mode, polished or not, keeps the error scales at the configured starts
    for mode in stage_a["result"]["modes"]:
        np.testing.assert_allclose(np.asarray(mode["theta"])[error], stage_a["starts"][0][error], rtol=0, atol=1e-12)
    # stage B frees them (members, sources and error scales; the large family frozen)
    assert error_sites <= stage_b["active"]


def test_grf_gp_hyperparameter_priors_have_physical_upper_bounds():
    likelihood = LikelihoodConfig()
    bounds = {"gp_sigma_alpha_prior": 0.3, "gp_knee_prior": 30.0, "gp_knee_sigma_shared_prior": 0.3, "gp_knee_ell_prior": 5.0}
    for name, upper in bounds.items():
        assert getattr(likelihood, name).upper == upper, name
    assert likelihood.gp_knee_prior.upper <= likelihood.gp_knee_lambda_grid[1]


def test_knee_prior_upper_must_lie_inside_the_tables(synthetic_config):
    for prior in (LogNormalPriorConfig(median=6.0, log_sigma=0.5), LogNormalPriorConfig(median=6.0, log_sigma=0.5, upper=100.0)):
        with pytest.raises(ValueError, match="gp_knee_prior.upper"):
            knee_config(synthetic_config, "all", gp_knee_prior=prior)
