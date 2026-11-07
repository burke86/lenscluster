"""Publication exports preserve recorded quantities and produce valid TeX names."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lenscluster.results_tex import command_prefix, duration, write_run_results, write_mock_results


@pytest.mark.parametrize('name,expected', [('ares','Ares'), ('hera','Hera'), ('MJ0416','MJZeroFourOneSix'), ('as1063','ASOneZeroSixThree'), ('cluster-2','ClusterTwo')])
def test_prefix(name, expected):
    assert command_prefix(name) == expected


def test_duration():
    assert duration(3599.6) == r'\ensuremath{1\,\mathrm{h}\;0\,\mathrm{min}\;0\,\mathrm{s}}'
    assert duration(90061) == r'\ensuremath{25\,\mathrm{h}\;1\,\mathrm{min}\;1\,\mathrm{s}}'
    for value in (None, np.nan, -1):
        assert 'not available' in duration(value)
    with pytest.raises(ValueError):
        command_prefix('---')


def test_run_export(tmp_path):
    def save(path, data):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
    save('artifacts/config.json', {'paths': {'figure_prefix': 'm0416'}})
    save('artifacts/summary.json', {'images': 3, 'members': 8, 'exact_members': 2})
    save('artifacts/posterior_diagnostics.json', {'elapsed_seconds': 61, 'max_r_hat': 1.005})
    save('tables/run_summary.json', {'fit': {'observed_image_count': 3, 'recovered_image_count': 2, 'image_rms_arcsec': .12345}})
    path = write_run_results(tmp_path)
    text = path.read_text()
    assert path.name == 'm0416_results.tex'
    assert path.parent == tmp_path / 'figures'
    assert not (tmp_path / 'tables' / path.name).exists()
    assert r'\MJZeroFourOneSixNRecovered}{2}' in text
    assert r'\MJZeroFourOneSixRMS}{0.123}' in text
    assert r'\MJZeroFourOneSixNFarFieldMembers}{6}' in text
    assert r'\MJZeroFourOneSixRuntime}{\textnormal{not available}}' in text
    commands = re.findall(r'\\newcommand\{\\([^}]+)\}', text)
    assert commands and all(re.fullmatch('[A-Za-z]+', c) for c in commands)
    save('tables/runtime.json', {'total': 3661})
    assert r'\MJZeroFourOneSixRuntimeHours}{1.017}' in write_run_results(tmp_path).read_text()


def test_posterior_rms_and_coordinate_coverage(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'artifacts/summary.json').write_text('{}')
    pd.DataFrame({'image_rms_arcsec': [.02, .07, .12, np.nan]}).to_csv(
        tmp_path / 'tables/posterior_image_rms.csv', index=False)
    pd.DataFrame({'x_obs_arcsec': [0, 2, 0], 'x_model_q16': [-1, -1, np.nan],
                  'x_model_q84': [1, 1, np.nan], 'y_obs_arcsec': [0, 0, 0],
                  'y_model_q16': [-1, -1, -1], 'y_model_q84': [1, 1, 1]}).to_csv(
        tmp_path / 'tables/image_fit_quality.csv', index=False)
    text = write_run_results(tmp_path, name='m0416').read_text()
    for suffix, value in [('PosteriorMedianRMS', '0.070'), ('NPosteriorRMSDraws', '3'),
                          ('PositionCoverageXPercent', '50'), ('NPositionCoverageX', '2'),
                          ('PositionCoverageYPercent', '100'), ('NPositionCoverageY', '3')]:
        assert rf'\MJZeroFourOneSix{suffix}}}{{{value}}}' in text


def test_field_parameter_medians(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'artifacts/summary.json').write_text('{}')
    (tmp_path / 'artifacts/config.json').write_text(json.dumps({'likelihood': {'gp_reference_redshift': 2}}))
    pd.DataFrame({'parameter': ['images.sigma_shared', 'images.gp_knee'],
                  'median': [.06805, 14.1126], 'best_fit': [.05, 13.6],
                  'p16': [.055, 13.4], 'p84': [.081, 14.6]}).to_csv(
        tmp_path / 'tables/posterior_summary.csv', index=False)
    text = write_run_results(tmp_path, name='m0416').read_text()
    assert r'\MJZeroFourOneSixGPSigmaShared}{0.068}' in text
    assert r'\MJZeroFourOneSixGPSigmaSharedInterval}{0.068_{-0.013}^{+0.013}}' in text
    assert r'\MJZeroFourOneSixGRFTurnoverScale}{14.1}' in text
    assert r'\MJZeroFourOneSixGRFReferenceRedshift}{2.0}' in text
    assert r'\MJZeroFourOneSixGPSigmaWhite}{\textnormal{not available}}' in text


def test_mock_export(tmp_path):
    campaign = SimpleNamespace(seeds=[1], summary=pd.DataFrame({'seed':[1,2], 'observed_image_count':[3,9], 'recovered_image_count':[2,8], 'missing_image_count':[1,1]}),
        residual_draws=pd.DataFrame({'seed':[1,1,1,1], 'draw_index':[0,0,1,1], 'image_residual_arcsec':[3,4,2,99], 'recovered':[True,True,True,False]}))
    text = write_mock_results(campaign, tmp_path, rhat_cut=1.01, min_ess=50).read_text()
    assert r'\MockNImages}{3}' in text
    assert r'\MockNExcluded}{1}' in text
    assert r'\MockRMS}{2.768}' in text
    assert r'\MockMedianResidual}{2.750}' in text


def test_member_table_uses_joint_deviation_draws(tmp_path):
    from lenscluster.results_tex import m0416_table_metrics, interval
    final = tmp_path / 'final'
    (final / 'tables').mkdir(parents=True)
    (final / 'artifacts').mkdir()
    (tmp_path / 'init/artifacts').mkdir(parents=True)
    names = ['sigma', 'cutkpc', 'alpha_sigma', 'beta_radius']
    physical = np.array([[100., 20., .3, .6]]*3)
    state = {'kpc_per_arcsec': 2., 'parameter_specs': [{'name':'potfile_1.'+n} for n in names]}
    (tmp_path / 'init/artifacts/state.json').write_text(json.dumps(state))
    pd.DataFrame({'parameter':['potfile_1.'+n for n in names], 'median': physical[0],
                  'p16':physical[0], 'p84':physical[0]}).to_csv(final/'tables/posterior_summary.csv',index=False)
    np.savez(final/'artifacts/posterior.npz',physical=physical)
    delta = np.tile([np.log(2.),np.log(3.)],(3,1,1))
    np.savez(final/'tables/member_diagnostic_draws.npz', deviations=delta)
    row = dict(member_id='potfile_1:8971',potfile_id='potfile_1',catalog_mag=21.,luminosity_ratio=1.)
    for prefix, value in [('velocity',np.log(2.)),('radius',np.log(3.))]:
        for suffix in ['p16','median','p84']: row[prefix+'_'+suffix]=value
    pd.DataFrame([row]).to_csv(final/'tables/member_diagnostics.csv',index=False)
    metrics = m0416_table_metrics(final)
    assert metrics['GalEightNineSevenOneSigma'] == interval(200,0,0,0)
    assert metrics['GalEightNineSevenOneCut'] == interval(30,0,0,1)
    assert metrics['ScalingSigma'] == interval(100*10**(.4*.3*(21-17.02)),0,0,0)
    assert not any(key.startswith('R') for key in metrics)
