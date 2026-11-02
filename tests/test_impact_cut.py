"""Dropping members that deflect no image measurably."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from lenscluster import artifacts, pipeline
from lenscluster.classification import member_max_deflection_arcsec
from lenscluster.config import ImageDiagnosticsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from conftest import SMOKE_SAMPLING_GATES, SMOKE_SEARCH_GATES


def test_drop_member_ids_removes_members_but_keeps_the_pivot(synthetic_config):
    full = build_state(synthetic_config)
    cut = build_state(synthetic_config, drop_member_ids=("members:g3", "members:g4"))
    assert len(full.members.member_ids) == 4 and len(cut.members.member_ids) == 2
    assert cut.dropped_member_ids == ("members:g3", "members:g4")
    assert cut.potfiles[0]["mag0"] == full.potfiles[0]["mag0"]
    deflection = member_max_deflection_arcsec(full, Evaluator(full), full.initial_theta)
    assert deflection.shape == (4,) and np.all(np.isfinite(deflection)) and np.all(deflection > 0)


def test_init_stage_applies_and_persists_the_impact_cut(synthetic_config):
    state = build_state(synthetic_config)
    deflection = np.sort(member_max_deflection_arcsec(state, Evaluator(state), state.initial_theta))
    threshold = float(np.sqrt(deflection[1] * deflection[2]))          # drops the two weakest members
    config = replace(synthetic_config, members=replace(synthetic_config.members, min_deflection_arcsec=threshold),
                     runtime=RuntimeConfig(seed=3, chains=1, resume=True, nuts_chain_method="sequential", skip_plots=True),
                     search=SearchConfig(**SMOKE_SEARCH_GATES, n_starts=2, lbfgs_max_iter=20, polish_max_iter=5),
                     sampling=SamplingConfig(**SMOKE_SAMPLING_GATES, warmup=5, samples=5, max_tree_depth=3),
                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=0)).validate()
    init = pipeline.run(config, stage="init")
    assert len(init.state.members.member_ids) == 2
    saved = json.loads((init.paths.artifacts / artifacts.CLASSIFICATION_FILE).read_text())
    assert len(saved["dropped_members"]) == 2 and saved["min_deflection_arcsec"] == threshold
    again = pipeline.run(config, stage="init")
    assert again.resumed and len(again.state.members.member_ids) == 2
