"""The batched local exact-image solve used for posterior draws agrees with the per-family solver."""
from __future__ import annotations

import numpy as np

from lenscluster.diagnostics import predictions_at
from lenscluster.evaluator import Evaluator
from lenscluster.exact_images import ExactImageSolver
from lenscluster.state import build_state


def test_batched_local_solve_matches_per_family_solver(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    solver = ExactImageSolver(evaluator, synthetic_config.image_diagnostics)
    theta = np.asarray(state.initial_theta)
    per_family = predictions_at(solver, state, evaluator, theta)
    batched = predictions_at(solver, state, evaluator, theta, adaptive=False)
    assert len(per_family["image_rows"]) == len(batched["image_rows"]) == state.images.n_images
    both = 0
    for a, b in zip(per_family["image_rows"], batched["image_rows"]):
        assert a["image_label"] == b["image_label"]
        np.testing.assert_allclose(a["magnification_model"], b["magnification_model"], rtol=1e-9)
        if a["image_recovery_status"] == "recovered" and b["image_recovery_status"] == "recovered":
            np.testing.assert_allclose([a["x_model_arcsec"], a["y_model_arcsec"]], [b["x_model_arcsec"], b["y_model_arcsec"]], atol=1e-6)
            np.testing.assert_allclose(a["magnification_model_at_model_position"], b["magnification_model_at_model_position"], rtol=1e-6)
            assert np.isfinite(a["magnification_model_at_model_position"])
            both += 1
        else:
            assert not np.isfinite(a["magnification_model_at_model_position"]) or a["image_recovery_status"] == "recovered"
    assert both > 0
