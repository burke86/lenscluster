from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lenscluster import cluster_solver as cs


def _member_potfile(n_members: int = 3) -> dict:
    return {
        "id": "potfile",
        "type": cs.DP_IE_PROFILE,
        "catalog_df": pd.DataFrame(
            {
                "id": [f"g{i}" for i in range(n_members)],
                "catalog_a": [2.0] * n_members,
                "catalog_b": [1.0] * n_members,
                "catalog_theta": [0.0] * n_members,
            }
        ),
    }


def test_top_k_zero_is_accepted_and_selects_no_members() -> None:
    evaluator = SimpleNamespace(perturbation_discovery_top_k=0)
    assert cs._perturbation_discovery_top_k_from_evaluator(evaluator) == 0
    score = np.asarray([[3.0, 1.0], [2.0, 5.0], [1.0, 4.0]], dtype=float)
    selected, selected_pair, _rs, _rp, _irp = cs._perturbation_discovery_top_k_selected(score, 0)
    assert not selected.any()
    assert not selected_pair.any()


def test_top_k_negative_still_rejected() -> None:
    evaluator = SimpleNamespace(perturbation_discovery_top_k=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        cs._perturbation_discovery_top_k_from_evaluator(evaluator)


def test_independent_scaling_specs_have_no_tau_parameters() -> None:
    specs, indices = cs._build_independent_scaling_parameter_specs(
        [_member_potfile()],
        [{0, 2}],
        start_index=0,
        log_sigma_offset_width=0.15,
        log_mass_offset_width=0.25,
    )
    fields = {spec.field for spec in specs}
    assert "independent_free_log_sigma_tau" not in fields
    assert "independent_free_log_mass_tau" not in fields
    delta_specs = [spec for spec in specs if "delta_unit" in spec.field]
    assert len(delta_specs) == 4
    assert all(spec.prior_kind == "normal" and spec.std == 1.0 for spec in delta_specs)
    assert all("tau" not in key for lookup in indices[0].values() for key in lookup)


def test_independent_scaling_width_validation() -> None:
    with pytest.raises(ValueError, match="log_sigma_offset_width"):
        cs._build_independent_scaling_parameter_specs(
            [_member_potfile()],
            [{0}],
            start_index=0,
            log_sigma_offset_width=0.0,
        )


def test_offset_widths_read_from_state() -> None:
    evaluator = SimpleNamespace(
        state=SimpleNamespace(
            independent_scaling_free_log_sigma_offset_width=0.17,
            independent_scaling_free_log_mass_offset_width=0.29,
        )
    )
    widths = cs.ClusterJAXEvaluator._independent_scaling_offset_widths(evaluator)
    assert widths == (0.17, 0.29)


def test_offset_widths_default_when_state_missing_fields() -> None:
    evaluator = SimpleNamespace(state=SimpleNamespace())
    widths = cs.ClusterJAXEvaluator._independent_scaling_offset_widths(evaluator)
    assert widths == (
        cs.DEFAULT_INDEPENDENT_SCALING_FREE_LOG_SIGMA_OFFSET_WIDTH,
        cs.DEFAULT_INDEPENDENT_SCALING_FREE_LOG_MASS_OFFSET_WIDTH,
    )


def test_scaling_scatter_guard_raises_when_all_members_promoted() -> None:
    with pytest.raises(ValueError, match="cannot affect the likelihood"):
        cs._ensure_scaling_scatter_has_cached_members([_member_potfile(3)], [{0, 1, 2}])


def test_scaling_scatter_guard_passes_with_cached_members() -> None:
    cs._ensure_scaling_scatter_has_cached_members([_member_potfile(3)], [{0}])
    cs._ensure_scaling_scatter_has_cached_members([_member_potfile(3)], [set()])
    cs._ensure_scaling_scatter_has_cached_members([_member_potfile(3)], [])
