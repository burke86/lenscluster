from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lenscluster.evaluator import Evaluator
from lenscluster.member_diagnostics import (compute_member_diagnostics, diagnostic_draw_indices, deviation_draws,
    neighbor_variables, prior_deviation_draws, reporting_fields, whiten)
from lenscluster.member_diagnostic_plots import write_member_diagnostic_plots
from lenscluster.model import member_log_deviations
from lenscluster.state import build_state, with_member_classes


def posterior_for(state, evaluator, n=12):
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(15).normal(0, .005, (n, state.n_parameters))
    physical = np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples)))
    return SimpleNamespace(samples=samples, physical=physical, log_prob=np.arange(n, dtype=float))


def test_selection_neighbors_and_whitening():
    np.testing.assert_array_equal(diagnostic_draw_indices([0., 4., 1., 2.], 0), [1])
    indices = diagnostic_draw_indices([0., 4., 1., 2.], 100)
    assert len(indices) == len(set(indices)) == 4 and indices[0] == 1
    assert len(diagnostic_draw_indices([0., np.nan, 2.], 32)) == 2
    xy = np.column_stack((np.arange(6), np.zeros(6)))
    nearest, density = neighbor_variables(xy)
    np.testing.assert_allclose(nearest, 1.)
    np.testing.assert_allclose(density[0], 5/(np.pi*25))
    assert np.isnan(neighbor_variables(xy[:1])[0]).all()
    assert np.isnan(neighbor_variables(xy[:5])[1]).all()
    cov = np.array([[[2., .7], [.7, .5]]])
    residual = np.array([[.2, -.3]])
    u = whiten(residual, cov)
    np.testing.assert_allclose((u*u).sum(), residual[0] @ np.linalg.solve(cov[0], residual[0]), atol=1e-14)


@pytest.mark.parametrize('mode', ['sigma', 'mass', 'sigma_rcut', 'mass_rcut'])
def test_physical_and_prior_deviations(synthetic_config, mode):
    config = replace(synthetic_config, scaling=replace(synthetic_config.scaling, member_deviation=mode))
    state = build_state(config)
    evaluator = Evaluator(state)
    posterior = posterior_for(state, evaluator)
    actual = deviation_draws(state, posterior.physical)
    for i, p in enumerate(posterior.physical):
        v, r = member_log_deviations(state.packed_lens_spec, p)
        components = state.members.component_indices
        np.testing.assert_allclose(actual[i], np.column_stack((v[components], r[components])), atol=1e-14)
    priors = prior_deviation_draws(state, power=10)
    centered = with_member_classes(state, state.members.exact_positions, centered=np.ones(4, bool))
    other = prior_deviation_draws(centered, power=10)
    np.testing.assert_allclose(priors, other, atol=1e-14)
    assert np.isfinite(priors).all()
    if mode == 'sigma':
        assert np.all(priors[..., 1] == 0)


def test_exact_reporting_preserves_sampled_sources(synthetic_config):
    state = with_member_classes(build_state(synthetic_config), np.array([0, 1]))
    evaluator = Evaluator(state)
    theta = jnp.asarray(state.initial_theta)
    report = jax.jit(lambda t: reporting_fields(evaluator, t))(theta)
    np.testing.assert_allclose(report['source'], evaluator.source_positions(theta), rtol=1e-12, atol=1e-12)
    exact = evaluator.fields(evaluator.physical(theta), force_exact=True, derivative=False)
    residual = np.asarray(exact['backprojected']) - np.asarray(report['source'])[state.images.family_index_per_image]
    np.testing.assert_allclose(report['residual'], whiten(residual, report['exact_covariance']), atol=1e-12)
    full = evaluator.fields(evaluator.physical(theta), force_exact=True, derivative=True)
    for key in ('backprojected', 'jacobian', 'member_alpha', 'member_jacobian'):
        np.testing.assert_allclose(exact[key], full[key], rtol=2e-14, atol=2e-14)
    sampled = evaluator.fields(evaluator.physical(theta), derivative=False)
    diff = np.asarray(sampled['backprojected']-exact['backprojected'])
    np.testing.assert_allclose(report['error_whitened'], np.linalg.norm(whiten(diff, report['sampled_covariance']), axis=-1), atol=1e-13)
    all_exact = with_member_classes(state, np.arange(4))
    zero = reporting_fields(Evaluator(all_exact), jnp.asarray(all_exact.initial_theta))
    np.testing.assert_array_equal(zero['error_arcsec'], 0)


def test_suite_and_best_only(synthetic_config, tmp_path):
    config = replace(synthetic_config, image_diagnostics=replace(synthetic_config.image_diagnostics, posterior_diagnostic_draws=0))
    state = build_state(config)
    evaluator = Evaluator(state)
    posterior = posterior_for(state, evaluator)
    data = compute_member_diagnostics(state, evaluator, posterior, config)
    assert data.fields['residual'].shape == (1, 4, 2)
    assert data.provenance['selected_draw_indices'] == [11]
    assert 'neighbor_density_arcsec2' in data.provenance['skipped_variables']
    assert 'environment_kappa' in data.variables
    assert data.members.member_id.tolist() == list(state.members.member_ids)
    paths = SimpleNamespace(figures=tmp_path/'figures', tables=tmp_path/'tables')
    style = plt.rcParams['font.size']
    results = write_member_diagnostic_plots(config, state, evaluator, posterior, paths)
    assert len(results) == 4
    assert not list(paths.figures.glob('*scaling_correlations.pdf'))
    for path in results:
        assert path.read_bytes().startswith(b'%PDF')
    provenance = json.loads((paths.tables/'member_diagnostic_provenance.json').read_text())
    assert provenance['source_definition'].startswith('sampled-model')
    assert plt.rcParams['font.size'] == style
    assert not plt.get_fignums()


@pytest.mark.parametrize('count', [-1, True, 1.5])
def test_diagnostic_count_validation(synthetic_config, count):
    config = replace(synthetic_config, image_diagnostics=replace(synthetic_config.image_diagnostics, posterior_diagnostic_draws=count))
    with pytest.raises(ValueError, match='posterior_diagnostic_draws'):
        config.validate()


def test_hierarchical_prior_width_against_independent_quadrature(synthetic_config):
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.stats import norm
    # The quadrature below integrates a normal deviation over the log-normal scatter, so the test asks for normal tails
    # (the default is Student-t with 4 degrees of freedom, whose 84th percentile is wider).
    config = replace(synthetic_config, scaling=replace(synthetic_config.scaling, member_deviation='sigma', member_deviation_tail='normal'))
    state = build_state(config)
    prior = config.scaling.member_velocity_scatter_prior
    def cdf(q):
        return quad(lambda z: norm.cdf(q/(prior.median*np.exp(prior.log_sigma*z)))*norm.pdf(z), -9, 9, epsabs=1e-10)[0]
    q84 = brentq(lambda q: cdf(q)-.84, 0, 10*prior.median)
    draws = prior_deviation_draws(state, power=14)
    measured = np.quantile(draws[..., 0], [.16, .84], axis=0)
    np.testing.assert_allclose(measured[0], -q84, rtol=.025)
    np.testing.assert_allclose(measured[1], q84, rtol=.025)


def test_normal_output_hook(synthetic_config, tmp_path, monkeypatch):
    from lenscluster import outputs
    import pandas as pd
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    posterior = posterior_for(state, evaluator)
    calls = []
    monkeypatch.setattr(outputs, 'posterior_summary_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(outputs, 'classification_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(outputs.plots, 'member_scaling_table', lambda *a: pd.DataFrame())
    monkeypatch.setattr(outputs, 'write_potfile_plots', lambda *a, **k: [])
    monkeypatch.setattr(outputs, 'write_member_diagnostic_plots', lambda *a, **k: calls.append(a))
    class StopAfterHook(Exception):
        pass
    def stop(*a, **k):
        raise StopAfterHook
    monkeypatch.setattr(outputs, 'image_recovery_tables', stop)
    paths = SimpleNamespace(figures=tmp_path/'figures', tables=tmp_path/'tables')
    with pytest.raises(StopAfterHook):
        outputs.write_outputs(synthetic_config, state, evaluator, posterior, {}, paths)
    assert len(calls) == 1
    assert calls[0][0] is synthetic_config and calls[0][1] is state


def test_multiple_populations_without_large_halo(synthetic_config):
    from lenscluster.config import ShearHaloConfig
    first = synthetic_config.model.member_populations[0]
    model = replace(synthetic_config.model, large_halos=(ShearHaloConfig('shear', .03, 0., .3),),
                    member_populations=(first, replace(first, id='second')))
    config = replace(synthetic_config, model=model,
                     image_diagnostics=replace(synthetic_config.image_diagnostics, posterior_diagnostic_draws=0))
    state = build_state(config)
    evaluator = Evaluator(state)
    data = compute_member_diagnostics(state, evaluator, posterior_for(state, evaluator), config)
    assert data.members.potfile_id.nunique() == 2
    assert data.members.member_id.is_unique
    assert 'halo_geometry' in data.provenance['skipped_variables']
    assert 'nearest_halo_arcsec' not in data.variables


def test_failed_budget_is_saved_without_changing_classes(synthetic_config, tmp_path):
    config = replace(synthetic_config, members=replace(synthetic_config.members, template_tolerance=1e-14),
                     image_diagnostics=replace(synthetic_config.image_diagnostics, posterior_diagnostic_draws=0))
    state = with_member_classes(build_state(config), np.array([0, 1]))
    before = state.members.exact_positions.copy()
    evaluator = Evaluator(state)
    paths = SimpleNamespace(figures=tmp_path/'figures', tables=tmp_path/'tables')
    write_member_diagnostic_plots(config, state, evaluator, posterior_for(state, evaluator), paths)
    p = json.loads((paths.tables/'member_diagnostic_provenance.json').read_text())
    assert p['failed_error_images']
    assert p['failed_error_members']
    np.testing.assert_array_equal(state.members.exact_positions, before)
    assert write_member_diagnostic_plots(config, SimpleNamespace(members=SimpleNamespace(member_ids=[])), None, None, paths) == []
