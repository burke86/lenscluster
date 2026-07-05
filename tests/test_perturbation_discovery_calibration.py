import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lenscluster import cluster_solver as cs


def test_calibrated_tolerances_reference_values() -> None:
    alpha_tol, jacobian_tol = cs._perturbation_discovery_calibrated_tolerances(0.05, 0.10)
    assert alpha_tol == pytest.approx(0.25)
    assert jacobian_tol == pytest.approx(0.25)
    # alpha_tol scales linearly with pos sigma and inversely with the offset width.
    assert cs._perturbation_discovery_calibrated_tolerances(0.10, 0.10)[0] == pytest.approx(0.5)
    assert cs._perturbation_discovery_calibrated_tolerances(0.05, 0.20)[0] == pytest.approx(0.125)


@pytest.mark.parametrize("pos_sigma, width", [(0.0, 0.1), (-1.0, 0.1), (0.05, 0.0), (np.nan, 0.1), (0.05, np.inf)])
def test_calibrated_tolerances_reject_invalid_inputs(pos_sigma: float, width: float) -> None:
    with pytest.raises(ValueError):
        cs._perturbation_discovery_calibrated_tolerances(pos_sigma, width)


def test_reference_draws_even_spacing_and_fallbacks() -> None:
    reference = np.zeros(3)
    samples = np.arange(300, dtype=float).reshape(100, 3)
    draws = cs._perturbation_discovery_reference_draws(SimpleNamespace(samples=samples), reference, max_draws=5)
    assert draws is not None and draws.shape == (5, 3)
    np.testing.assert_array_equal(draws[0], samples[0])
    np.testing.assert_array_equal(draws[-1], samples[-1])
    assert cs._perturbation_discovery_reference_draws(SimpleNamespace(samples=None), reference) is None
    assert cs._perturbation_discovery_reference_draws(SimpleNamespace(samples=samples[:1]), reference) is None
    wrong_dim = np.zeros((10, 4))
    assert cs._perturbation_discovery_reference_draws(SimpleNamespace(samples=wrong_dim), reference) is None


def _union_state() -> SimpleNamespace:
    # sigma_arcsec=0.02, width=0.10 -> alpha_tol = jacobian_tol = 0.1
    return SimpleNamespace(
        sigma_arcsec=0.02,
        independent_scaling_free_log_sigma_offset_width=0.10,
        potfiles=[{"id": "members"}],
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g1",
                "catalog_row_index": 0,
                "component_index": 10,
                "free_component_index": 20,
            },
        ],
        family_data=[SimpleNamespace(family_id="1", image_labels=["a"])],
    )


class _DrawSequenceEvaluator:
    """Returns a different alpha row per call: crossing in some draws only."""

    perturbation_discovery_jacobian_weight = 1.0
    perturbation_discovery_top_k = None

    def __init__(self, alpha_per_draw: list[float]) -> None:
        self._alpha_per_draw = list(alpha_per_draw)
        self._call_index = 0
        self.scaling_component_indices = np.asarray([10], dtype=np.int32)
        self.flat_critical_arc_data = SimpleNamespace(
            x_obs=np.asarray([0.0], dtype=float),
            y_obs=np.asarray([0.0], dtype=float),
            family_ids=("1",),
            global_family_index_per_image=np.asarray([0], dtype=np.int32),
            local_image_index_per_image=np.asarray([0], dtype=np.int32),
        )

    def _physical_parameter_vector(self, reference):
        return reference

    def _build_flat_packed_lens_state_with_validity_from_physical(self, _physical, _flat, *, stop_gradient):
        del stop_gradient
        return SimpleNamespace(), SimpleNamespace(is_valid=np.asarray(True), reason_flags=np.asarray([], dtype=bool))

    def _flat_component_alpha_and_jacobian_delta_rows_for_components(self, _x, _y, _packed, components):
        np.testing.assert_array_equal(np.asarray(components, dtype=np.int32), np.asarray([20], dtype=np.int32))
        value = self._alpha_per_draw[min(self._call_index, len(self._alpha_per_draw) - 1)]
        self._call_index += 1
        zeros = np.zeros((1, 1), dtype=float)
        return (np.asarray([[value]], dtype=float), zeros, zeros, zeros, zeros, zeros)


def test_draw_median_outlier_draw_does_not_select() -> None:
    # alpha_tol = 0.1: candidate crosses only in 1 of 5 draws -> median 0.05 -> not selected.
    evaluator = _DrawSequenceEvaluator([0.05, 0.05, 0.5, 0.05, 0.05])
    draws = np.zeros((5, 2), dtype=float)
    selected, diagnostics, table = cs._perturbation_discovery_union_from_evaluator(
        _union_state(), evaluator, np.zeros(2), reference_draws=draws
    )
    assert selected == [set()]
    assert diagnostics["score_draws_valid"] == 5
    assert diagnostics["count"] == 0
    assert float(table.loc[0, "score_q84"]) > 1.0  # the outlier draw is visible in the spread


def test_draw_median_majority_draws_select() -> None:
    # Crosses in 3 of 5 draws -> median 0.5 -> selected.
    evaluator = _DrawSequenceEvaluator([0.5, 0.05, 0.5, 0.05, 0.5])
    draws = np.zeros((5, 2), dtype=float)
    selected, diagnostics, _table = cs._perturbation_discovery_union_from_evaluator(
        _union_state(), evaluator, np.zeros(2), reference_draws=draws
    )
    assert selected == [{0}]
    assert diagnostics["score_draws_valid"] == 5
    assert diagnostics["count"] == 1


def _patch_audit_dependencies(monkeypatch: pytest.MonkeyPatch, audit_selected: list[set[int]]) -> None:
    state = SimpleNamespace(parameter_specs=[])
    arrays = {"best_fit": np.zeros(2), "samples": np.zeros((4, 2))}
    monkeypatch.setattr(cs, "_load_artifacts", lambda _dir: (state, {}, arrays, {}))
    monkeypatch.setattr(
        cs, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda a, _specs, _diag: (a, False)
    )
    evaluator = SimpleNamespace(
        reported_physical_to_latent_parameter_vector=lambda vector: np.asarray(vector, dtype=float)
    )
    monkeypatch.setattr(cs, "_build_cluster_evaluator_from_args", lambda _args, _state: evaluator)
    monkeypatch.setattr(
        cs,
        "_perturbation_discovery_union_from_evaluator",
        lambda _state, _evaluator, _best, reference_draws=None: (
            audit_selected,
            {"score_draws_valid": 4},
            pd.DataFrame(),
        ),
    )


def test_stage1_audit_writes_csv_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit_dependencies(monkeypatch, audit_selected=[{0, 2}])
    (tmp_path / "artifacts").mkdir()
    cs._audit_frozen_independent_selection(argparse.Namespace(quiet=True), tmp_path, [{0, 1}])
    audit_path = tmp_path / "tables" / "perturbation_discovery_stage1_audit.csv"
    assert audit_path.is_file()
    audit = pd.read_csv(audit_path)
    assert set(zip(audit["status"], audit["catalog_row_index"])) == {("would_add", 2), ("would_remove", 1)}


def test_stage1_audit_silent_on_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audit_dependencies(monkeypatch, audit_selected=[{0, 1}])
    (tmp_path / "artifacts").mkdir()
    cs._audit_frozen_independent_selection(argparse.Namespace(quiet=True), tmp_path, [{0, 1}])
    assert not (tmp_path / "tables" / "perturbation_discovery_stage1_audit.csv").exists()


def test_stage1_audit_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "_load_artifacts", lambda _dir: (_ for _ in ()).throw(RuntimeError("boom")))
    cs._audit_frozen_independent_selection(argparse.Namespace(quiet=True), tmp_path, [set()])
