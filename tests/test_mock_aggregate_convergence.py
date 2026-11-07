import json
import pytest

from lenscluster.mock_validation.aggregate_plots import load_campaign


def test_completed_selection_does_not_shrink_when_first_mock_converges(tmp_path):
    for seed, rhat in [(1, 1.08), (2, 1.03), (3, 1.005)]:
        directory = tmp_path / f'seed_{seed:08d}' / 'recovery'
        directory.mkdir(parents=True)
        payload = dict(seed=seed, max_r_hat=rhat, min_n_eff=100,
                       parameters=[], magnification=[], image_residual_draws=[], m2d=[], subhalos={})
        (directory/'recovery.json').write_text(json.dumps(payload))
        # An unfinished realization must never enter any aggregate table.
        (tmp_path/'seed_99999999').mkdir(exist_ok=True)
        assert load_campaign(tmp_path).seeds == list(range(1,seed+1))
        if seed < 3:
            with pytest.raises(ValueError, match='No converged realization'):
                load_campaign(tmp_path, converged_only=True)
    assert load_campaign(tmp_path, converged_only=True).seeds == [3]


def test_aggregate_rechecks_saved_verdict_at_current_cut(tmp_path):
    for seed, rhat, neff, saved_verdict in [(1, 1.009, 50, False), (2, 1.01, 100, True),
                                             (3, 1.04, 100, True), (4, 1.009, 49, True)]:
        directory = tmp_path / f"seed_{seed:08d}" / "recovery"
        directory.mkdir(parents=True)
        payload = dict(seed=seed, max_r_hat=rhat, min_n_eff=neff, converged=saved_verdict,
                       convergence_reasons=['old verdict'], parameters=[], magnification=[],
                       image_residual_draws=[], m2d=[], subhalos={})
        (directory / 'recovery.json').write_text(json.dumps(payload))
    campaign = load_campaign(tmp_path, converged_only=True)
    assert campaign.seeds == [1]
    assert campaign.summary.converged.tolist() == [True, False, False, False]
    assert '1.01' in campaign.summary.iloc[1].convergence_reasons
    assert load_campaign(tmp_path, converged_only=False).seeds == [1, 2, 3, 4]
    assert load_campaign(tmp_path).seeds == [1, 2, 3, 4]


def test_correlated_error_parameters_in_aggregate_figures(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from lenscluster.mock_validation import aggregate_plots as plots

    names = ['images.gp_sigma_alpha', 'images.gp_knee', 'images.gp_knee_gamma',
             'images.sigma_shared', 'images.gp_ell', 'images.sigma_white']
    rows = [dict(parameter=name, seed=seed, truth=value, median=value * 1.1,
                 q16=value * .8, q84=value * 1.3, covered_68=True, covered_95=True, scored=True)
            for name in names for seed, value in enumerate([.1, .2, .3])]
    campaign = SimpleNamespace(parameters=pd.DataFrame(rows))
    assert {p for p, _ in plots._standardized_parameter_panels(campaign)} == set(names)
    assert plots._parameter_component(names[0]) == 'grf'
    assert plots._parameter_component('images.gp_ell') == 'gp'
    assert 'arcsec' in plots._axis_label('images.gp_ell', 'truth')
    assert 'arcsec' not in plots._axis_label('images.gp_knee_gamma', 'truth')
    for render in [plots.plot_parameter_recovery, plots.plot_standardized_residuals, plots.plot_bias_and_coverage]:
        path = tmp_path / (render.__name__ + '.pdf')
        render(campaign, path)
        assert path.read_bytes().startswith(b'%PDF')
    legacy = SimpleNamespace(parameters=campaign.parameters.assign(parameter='halo.v_disp'))
    assert [p for p, _ in plots._standardized_parameter_panels(legacy)] == ['halo.v_disp']


def test_halo_aperture_uses_inside_radius_and_preserves_unknown_coverage():
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from lenscluster.mock_validation.aggregate_plots import halo_aperture_parameters, coverage_table
    rows = [dict(seed=1, component='halo', radius_kpc=r, outer_image_radius_kpc=255,
                 truth_m2d_msun=1e14, model_q16=8e13, model_q50=1.01e14, model_q84=1.2e14)
            for r in (100, 230, 280)]
    result = halo_aperture_parameters(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row.aperture_radius_kpc == 230
    assert row.truth == 14
    assert np.isclose(row["median"], np.log10(1.01e14))
    assert row.covered_68
    assert np.isnan(row.covered_95)
    result['truth_percentile'] = np.nan
    table = coverage_table(SimpleNamespace(parameters=result))
    assert np.isnan(table.iloc[0].coverage_95)
    rows[1].update(model_q025=7e13, model_q975=1.4e14)
    assert halo_aperture_parameters(pd.DataFrame(rows)).iloc[0].covered_95
    assert halo_aperture_parameters(pd.DataFrame()).empty


def test_grouped_recovery_separates_components(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from lenscluster.mock_validation import aggregate_plots as plots
    names = ['halo.log10_mass_msun', 'halo.log10_aperture_mass_msun', 'bcg.v_disp',
             'images.gp_knee_gamma', 'potfile.alpha_sigma', 'source.001.beta_x',
             'source.001.beta_y', 'halo.x_centre', 'halo.y_centre', 'bcg.x_centre', 'bcg.y_centre']
    rows = [dict(parameter=name, seed=i, truth=value, median=value*1.05,
                 q16=value*.8, q84=value*1.2) for name in names for i,value in enumerate([1.,1.,2.,2.])]
    campaign = SimpleNamespace(parameters=pd.DataFrame(rows))
    save = plots._save
    def inspect(fig, path):
        labels = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert sorted(labels) == sorted(['Halo', 'BCG', 'GRF + image-plane GP', 'Member scaling relation', 'Source positions'])
        panels = [ax for ax in fig.axes if ax.containers]
        assert len(panels) == len(names)-3
        pooled = [ax for ax in panels if "x,y" in ax.get_xlabel() or "beta_x," in ax.get_xlabel()]
        assert len(pooled) == 3
        assert all(len(ax.containers[0].lines[0].get_xdata()) == 8 for ax in pooled)
        assert all(not line.get_visible() for ax in panels for line in ax.get_xgridlines())
        assert not any(ax.get_title() for ax in fig.axes)
        assert not fig.legends
        return save(fig,path)
    monkeypatch.setattr(plots, '_save', inspect)
    output = tmp_path/'grouped.pdf'
    assert plots.plot_parameter_recovery_grouped(campaign, output) == output
    assert output.read_bytes().startswith(b'%PDF')
    assert plots.plot_parameter_recovery_grouped(SimpleNamespace(parameters=pd.DataFrame()), output) is None
