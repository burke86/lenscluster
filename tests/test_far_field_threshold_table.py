"""The far-field threshold study: counts move with the threshold, the configured one is marked, the table renders."""
from __future__ import annotations

import numpy as np

from lenscluster.evaluator import Evaluator
from lenscluster.pipeline import classify
from lenscluster.report import far_field_threshold_table
from lenscluster.state import build_state


def test_threshold_study_rows_and_table(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    state = classify(state, evaluator, state.initial_theta, synthetic_config, reference_cut_kpc=None, beta_radius=None, log=lambda _: None)
    rows = state.members.classification["threshold_study"]
    guards = [r["guard_cut_radii"] for r in rows]
    assert guards == sorted(guards) and float(synthetic_config.members.far_field_cut_radii) in guards
    assert sum(r["selected"] for r in rows) == 1
    n = len(state.members.member_ids)
    assert all(r["exact"] + r["far_field"] == n for r in rows)
    assert all(a["exact"] <= b["exact"] for a, b in zip(rows, rows[1:]))          # a larger threshold never makes fewer members exact
    selected = next(r for r in rows if r["selected"])
    assert selected["exact"] == state.members.classification["exact_count"]
    assert all(np.isfinite([r["max_error_mas"], r["median_error_mas"], r["max_whitened_error"]]).all() for r in rows)
    d = state.members.classification
    assert d["guard_headroom"] >= synthetic_config.members.cut_radius_inflation * 0.999 or d["guard_headroom"] == float("inf")   # classified at the inflated radius
    text = far_field_threshold_table(state)
    assert "Far-field threshold study" in text and "*" in text and f"{selected["guard_cut_radii"]:g}" in text
