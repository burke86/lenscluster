"""Posterior histogram counts use all draws; the legacy summary helper conditions on recovery count."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lenscluster.plots import plot_image_residual_histogram, posterior_residual_spread


def _draws(counts):
    rows = []
    for d, n in enumerate(counts):
        for k in range(4):
            rows.append({"draw_index": d, "image_label": f"1.{k}", "recovered": k < n, "image_residual_arcsec": 0.1 * (k + 1) + 0.01 * d if k < n else np.nan})
    return pd.DataFrame(rows)


def test_spread_uses_only_draws_with_the_best_count_and_flags_more():
    spread = posterior_residual_spread(_draws([3, 3, 3, 2, 4]), best_count=3)
    assert spread["draws_total"] == 5 and spread["draws_used"] == 3 and spread["draws_more_than_best"] == 1
    assert 0.1 < spread["median_q16"] <= spread["median_q84"] < 0.3 and spread["rms_q16"] <= spread["rms_q84"]
    empty = posterior_residual_spread(None, best_count=3)
    assert empty["draws_used"] == 0 and np.isnan(empty["rms_q16"])


def test_histogram_renders_with_error_bars_and_warning(tmp_path):
    image_df = pd.DataFrame({"image_residual_arcsec": [0.1, 0.2, 0.3, np.nan], "image_recovery_status": ["recovered"] * 3 + ["not_recovered"]})
    messages = []
    path = plot_image_residual_histogram(image_df, tmp_path / "h.pdf", draws=_draws([3, 3, 3, 2, 4]), log=messages.append)
    assert path.is_file() and any("recovered more images than the best fit" in m for m in messages)


def test_posterior_bin_counts_include_empty_draws_and_exclude_unrecovered():
    from lenscluster.plots import posterior_residual_histogram_counts
    draws = pd.DataFrame({'draw_index': [0, 0, 1, 1, 2, 2],
                          'recovered': [True, False, True, True, False, False],
                          'image_residual_arcsec': [.1, .1, .1, .3, .1, np.nan]})
    counts = posterior_residual_histogram_counts(draws, np.array([0., .2, .4]))
    np.testing.assert_array_equal(counts, [[1, 0], [1, 1], [0, 0]])
    assert posterior_residual_histogram_counts(None, np.array([0., .2, .4])).shape == (0, 2)


def test_histogram_band_keeps_equal_or_better_recovery_counts():
    from lenscluster.plots import posterior_residual_histogram_counts
    bins = np.array([0., .25, .5])
    draws = _draws([2, 3, 4])
    counts = posterior_residual_histogram_counts(draws, bins, min_recovered=3)
    np.testing.assert_array_equal(counts, [[2, 1], [2, 2]])
    assert posterior_residual_histogram_counts(draws, bins, min_recovered=5).shape == (0, 2)
