import numpy as np
import pandas as pd

from lenscluster.mock_validation.aggregate_plots import binned_magnification_summary, parameter_recovery_bins


def test_binned_means_weight_mocks_not_images():
    data = pd.DataFrame({"seed": [1, 1, 1, 2],
                         "magnification_true_at_obs": [2., 2., 2., -2.],
                         "magnification_model_q50": [2., 4., 6., -8.]})
    row = binned_magnification_summary(data, bins=1).iloc[0]
    assert row.truth_mean == 2
    assert row.recovered_mean == 6
    np.testing.assert_allclose(row.recovered_sem, 2)
    assert row.n_mocks == 2 and row.n_images == 4


def test_single_mock_and_nonfinite_pairs():
    data = pd.DataFrame({"seed": [1, 2, 3],
                         "magnification_true_at_obs": [2., np.inf, 0.],
                         "magnification_model_q50": [3., 4., 5.]})
    row = binned_magnification_summary(data).iloc[0]
    assert row.n_mocks == 1
    assert np.isnan(row.recovered_sem)


def test_parameter_bins_use_independent_mock_means():
    panel = pd.DataFrame({"seed": [1, 1, 1, 2], "truth": [-2., -2., -2., -2.],
                          "median": [-2., 0., 2., 4.]})
    row = parameter_recovery_bins(panel).iloc[0]
    assert row.truth_mean == -2
    assert row.recovered_mean == 2
    np.testing.assert_allclose(row.recovered_sem, 2)
