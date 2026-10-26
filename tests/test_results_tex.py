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
    assert r'\MJZeroFourOneSixNRecovered}{2}' in text
    assert r'\MJZeroFourOneSixRMS}{0.123}' in text
    assert r'\MJZeroFourOneSixNFarFieldMembers}{6}' in text
    assert r'\MJZeroFourOneSixRuntime}{\textnormal{not available}}' in text
    commands = re.findall(r'\\newcommand\{\\([^}]+)\}', text)
    assert commands and all(re.fullmatch('[A-Za-z]+', c) for c in commands)
    save('tables/runtime.json', {'total': 3661})
    assert r'\MJZeroFourOneSixRuntimeHours}{1.017}' in write_run_results(tmp_path).read_text()


def test_mock_export(tmp_path):
    campaign = SimpleNamespace(seeds=[1], summary=pd.DataFrame({'seed':[1,2], 'observed_image_count':[3,9], 'recovered_image_count':[2,8], 'missing_image_count':[1,1]}),
        residual_draws=pd.DataFrame({'seed':[1,1,1,1], 'draw_index':[0,0,1,1], 'image_residual_arcsec':[3,4,2,99], 'recovered':[True,True,True,False]}))
    text = write_mock_results(campaign, tmp_path, rhat_cut=1.01, min_ess=50).read_text()
    assert r'\MockNImages}{3}' in text
    assert r'\MockNExcluded}{1}' in text
    assert r'\MockRMS}{2.768}' in text
    assert r'\MockMedianResidual}{2.750}' in text
