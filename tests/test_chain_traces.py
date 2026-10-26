from types import SimpleNamespace
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

from lenscluster.chain_traces import trace_data, write_chain_traces


def fixture(chains=2, draws=30):
    grouped = np.random.default_rng(91).normal(size=(chains, draws, 3))
    physical = grouped.reshape(-1, 3).copy()
    physical[:, 0] = np.exp(physical[:, 0])
    specs = [SimpleNamespace(name=n, component_family=f) for n, f in
             [('halo.sigma', 'large'), ('member.offset', 'member_scatter_offset'), ('source.x', 'source_position')]]
    p = SimpleNamespace(grouped_samples=grouped, samples=grouped.reshape(-1, 3), physical=physical,
                        log_prob=np.zeros(chains*draws), accept_prob=np.ones(chains*draws),
                        num_steps=np.ones(chains*draws), diverging=np.arange(chains*draws) % 7 == 0)
    return p, specs


def test_diagnostics_match_plotted_coordinates():
    p, specs = fixture()
    d = trace_data(p, specs)
    x = p.physical.reshape(2, 30, 3)
    np.testing.assert_array_equal(d['chains'], x)
    np.testing.assert_allclose(d['rhat'], split_gelman_rubin(x))
    np.testing.assert_allclose(d['ess'], effective_sample_size(x))
    assert sorted(d['selected']) == [0, 1, 2]
    p.grouped_samples = p.grouped_samples[::-1]
    with pytest.raises(ValueError, match='disagree'):
        trace_data(p, specs)


@pytest.mark.parametrize('chains,draws', [(1, 30), (2, 2)])
def test_insufficient_chains_or_draws(chains, draws):
    p, specs = fixture(chains, draws)
    d = trace_data(p, specs)
    assert np.isnan(d['rhat']).all() and np.isnan(d['ess']).all()


def test_constant_parameter_and_missing_chains(tmp_path):
    p, specs = fixture()
    p.physical[:, 1] = 0
    d = trace_data(p, specs)
    assert np.isnan(d['rhat'][1])
    p.grouped_samples = None
    assert write_chain_traces(p, specs, tmp_path/'fig', tmp_path/'tables', log=lambda _: None) is None
    assert not (tmp_path/'fig').exists()


def test_pdf_provenance_and_style_isolation(tmp_path):
    p, specs = fixture()
    style = plt.rcParams.copy()
    path = write_chain_traces(p, specs, tmp_path/'figures', tmp_path/'tables', 'hera', log=lambda _: None)
    assert path.name == 'hera_chain_traces.pdf'
    assert path.read_bytes().startswith(b'%PDF')
    meta = json.loads((tmp_path/'tables/chain_traces.json').read_text())
    assert meta['chains'] == 2 and meta['draws_per_chain'] == 30
    assert sum(meta['divergences_per_chain']) == p.diverging.sum()
    assert (tmp_path/'tables/chain_convergence.csv').is_file()
    assert dict(style) == dict(plt.rcParams)


def test_normal_output_hook(tmp_path, monkeypatch):
    from lenscluster import outputs
    p, specs = fixture()
    state = SimpleNamespace(parameter_specs=specs)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix='test'))
    paths = SimpleNamespace(tables=tmp_path/'tables', figures=tmp_path/'figures')
    monkeypatch.setattr(outputs, 'posterior_summary_table', lambda *a: __import__('pandas').DataFrame())
    monkeypatch.setattr(outputs, 'classification_table', lambda *a: __import__('pandas').DataFrame())
    class StopAfterTrace(Exception):
        pass
    monkeypatch.setattr(outputs.plots, 'member_scaling_table', lambda *a: (_ for _ in ()).throw(StopAfterTrace()))
    with pytest.raises(StopAfterTrace):
        outputs.write_outputs(config, state, SimpleNamespace(physical=lambda x: x), p, {}, paths, log=lambda _: None)
    assert (paths.figures/'test_chain_traces.pdf').is_file()
