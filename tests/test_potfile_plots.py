from types import SimpleNamespace
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lenscluster import outputs, potfile_plots


def fixture_draws(population="potfile_members"):
    specs = [SimpleNamespace(name=f"{population}.{field}", field=field,
                             component_family="scaling", transform_kind=transform,
                             prior_kind="normal", mean=mean, std=std)
             for field, transform, mean, std in (("sigma", "log_positive", 4.6, .3),
                                                 ("alpha_sigma", "identity", .25, .05))]
    rng = np.random.default_rng(42)
    samples = np.column_stack((np.exp(rng.normal(4.5, .2, 300)), rng.normal(.28, .04, 300)))
    return samples, specs


def test_run_outputs_create_both_potfile_figures(tmp_path, monkeypatch):
    samples, specs = fixture_draws()
    posterior = SimpleNamespace(samples=samples, physical=samples, log_prob=np.zeros(len(samples)))
    state = SimpleNamespace(parameter_specs=specs)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix="hera"), image_diagnostics=None)
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures")
    monkeypatch.setattr(outputs, "posterior_summary_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(outputs, "classification_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(outputs.plots, "member_scaling_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(outputs, "write_member_diagnostic_plots", lambda *args, **kwargs: [])
    # Stop after the actual potfile output hook, before unrelated image diagnostics.
    class ReachedImageDiagnostics(Exception):
        pass
    def stop(*args, **kwargs):
        raise ReachedImageDiagnostics
    monkeypatch.setattr(outputs, "image_recovery_tables", stop)
    with pytest.raises(ReachedImageDiagnostics):
        outputs.write_outputs(config, state, SimpleNamespace(physical=lambda p: p), posterior, {}, paths)
    for kind in ("corner", "prior_posterior"):
        path = paths.figures / f"hera_potfile_members_{kind}.pdf"
        assert path.read_bytes().startswith(b"%PDF")
        summary = json.loads(path.with_suffix(".json").read_text())
        assert summary["population"] == "potfile_members"
        assert summary["draws"] == len(samples)


def test_multiple_populations_and_isolated_style(tmp_path):
    samples, specs = fixture_draws("potfile_2")
    other, other_specs = fixture_draws("potfile_named")
    style = plt.rcParams.copy()
    results = potfile_plots.write_potfile_plots(np.column_stack((samples, other)), specs + other_specs, tmp_path)
    assert {p.name for p in results} == {f"potfile_{pop}_{kind}.pdf" for pop in ("2", "named")
                                       for kind in ("corner", "prior_posterior")}
    assert plt.rcParams["font.size"] == style["font.size"]
    assert plt.rcParams["mathtext.fontset"] == style["mathtext.fontset"]
    assert not plt.get_fignums()


def test_short_fixed_and_empty_posteriors(tmp_path):
    samples, specs = fixture_draws()
    assert potfile_plots.write_potfile_plots(samples[:2], specs, tmp_path, log=lambda _: None) == []
    assert potfile_plots.write_potfile_plots(np.empty((4, 0)), [], tmp_path) == []
    fixed = np.tile(samples[0], (10, 1))
    paths = potfile_plots.write_potfile_plots(fixed, specs, tmp_path, log=lambda _: None)
    assert [p.name for p in paths] == ["potfile_members_prior_posterior.pdf"]


def test_prior_quantiles_and_density_match_sampler():
    import jax
    from lenscluster.sampling import SampleSite, distribution_for_site
    jax.config.update("jax_enable_x64", True)
    for kind in ("normal", "truncated_normal", "uniform"):
        spec = SimpleNamespace(name="potfile_1.beta_radius", field="beta_radius", prior_kind=kind,
                               mean=.5, std=.2, lower=0., upper=1.5, centered=False)
        prior = potfile_plots.saved_prior(spec)
        values = prior.ppf(np.linspace(.001, .999, 100))
        distribution = distribution_for_site(SampleSite("beta", (0,), False), [spec])
        np.testing.assert_allclose(prior.logpdf(values), distribution.log_prob(values), atol=2e-14, rtol=2e-14)


def test_corner_traces_preserve_chain_order(tmp_path):
    samples, specs = fixture_draws()
    grouped = samples.reshape(3, 100, 2)
    path = tmp_path / 'corner_with_traces.pdf'
    potfile_plots.plot_potfile_corner(samples, specs, 'potfile_members', path, grouped_physical=grouped)
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['trace_chains'] == 3 and metadata['trace_draws_per_chain'] == 100
    with pytest.raises(ValueError, match='chain-ordered'):
        potfile_plots.plot_potfile_corner(samples, specs, 'potfile_members', tmp_path/'bad.pdf', grouped_physical=grouped[::-1])
    plt.close('all')
