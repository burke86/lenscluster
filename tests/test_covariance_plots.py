from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lenscluster.covariance_plots import write_covariance_plots
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state


@pytest.mark.parametrize('selection', [True, False])
def test_covariance_outputs_and_best_only(synthetic_config, tmp_path, selection):
    config = replace(synthetic_config,
                     image_diagnostics=replace(synthetic_config.image_diagnostics, posterior_diagnostic_draws=0))
    if not selection:
        config = replace(config, likelihood=replace(config.likelihood, sigma_sel_prior=None, sigma_extra_prior=None))
    state = build_state(config)
    evaluator = Evaluator(state)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(42).normal(0, .03, (40, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))),
                                log_prob=np.arange(40.), grouped_samples=samples.reshape(2, 20, -1))
    paths = SimpleNamespace(figures=tmp_path / 'figures', tables=tmp_path / 'tables')
    style = dict(plt.rcParams)
    outputs = write_covariance_plots(config, state, evaluator, posterior, paths, log=lambda _: None)
    assert len(outputs) == 2 and all(p.is_file() for p in outputs)
    assert dict(plt.rcParams) == style
    metadata = json.loads((paths.tables / 'covariance_errors_both_planes.json').read_text())
    assert metadata['posterior_draw_indices'] == [39]
    assert metadata['plane'] == 'both'
    table = pd.read_csv(paths.tables / 'covariance_errors_both_planes.csv')
    source = table[table.plane == 'source'].reset_index(drop=True)
    image = table[table.plane == 'image'].reset_index(drop=True)
    for component in ['measurement', 'selection', 'model', 'total']:
        np.testing.assert_allclose(image[component+'_arcsec'] * image.singular_value_best,
                                   source[component+'_arcsec'], atol=1e-14)
    np.testing.assert_allclose(table.total_p16_arcsec, table.total_p84_arcsec)
    if not selection:
        assert (table.selection_arcsec == 0).all()
    corner = json.loads((paths.tables / 'covariance_corner.json').read_text())
    assert ('images.lambda_sel' in corner['parameters']) == selection


def test_normal_output_calls_covariance_suite(monkeypatch, synthetic_config, tmp_path):
    from lenscluster import outputs
    class ReachedCovariance(Exception):
        pass
    def reached(*args, **kwargs):
        raise ReachedCovariance
    monkeypatch.setattr(outputs, 'write_covariance_plots', reached)
    monkeypatch.setattr(outputs, 'posterior_summary_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(outputs, 'classification_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(outputs, 'write_chain_traces', lambda *a, **k: None)
    monkeypatch.setattr(outputs, 'write_potfile_plots', lambda *a, **k: None)
    monkeypatch.setattr(outputs.plots, 'member_scaling_table', lambda *a: pd.DataFrame())
    state = SimpleNamespace(parameter_specs=[])
    posterior = SimpleNamespace(samples=np.zeros((1, 1)), physical=np.ones((1, 1)), log_prob=np.zeros(1))
    evaluator = SimpleNamespace(physical=lambda p: p)
    paths = SimpleNamespace(tables=tmp_path/'tables', figures=tmp_path/'figures')
    with pytest.raises(ReachedCovariance):
        outputs.write_outputs(synthetic_config, state, evaluator, posterior, {}, paths)
