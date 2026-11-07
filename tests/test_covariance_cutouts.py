import json
import pytest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import WCS

from lenscluster import covariance_cutouts as cc
from lenscluster.evaluator import Evaluator
from lenscluster.exact_images import ExactImageSolver
from lenscluster.state import build_state


def test_quadratic_contour_linear_limit_and_curved_map():
    a = np.diag([.2, .8]); c = np.diag([.04, .09])
    segments = cc.quadratic_contours(a, np.zeros((2, 2, 2)), c, 4., grid_size=801)
    points = np.concatenate(segments)
    q = np.sum((points @ a.T)**2 / np.diag(c), axis=1)
    np.testing.assert_allclose(q, 1., atol=5e-5)
    np.testing.assert_allclose(np.max(abs(points), axis=0), [1., .375], atol=1e-5)
    # beta_y = .8 y + .15 x^2 bends the same unit contour into a curved strip.
    b = np.zeros((2, 2, 2)); b[1, 0, 0] = .3
    points = np.concatenate(cc.quadratic_contours(a, b, c, 4., grid_size=801))
    expected = np.column_stack([.2*points[:, 0], .8*points[:, 1]+.15*points[:, 0]**2])
    np.testing.assert_allclose(cc.quadratic_offset(points, a, b), expected, atol=1e-15)
    np.testing.assert_allclose(np.sum(expected**2 / np.diag(c), axis=1), 1., atol=7e-5)


def test_exact_geometry_and_derivatives(synthetic_config):
    state = build_state(synthetic_config); evaluator = Evaluator(state)
    g = cc.best_geometry(state, evaluator, state.initial_theta)
    solver = ExactImageSolver(evaluator, synthetic_config.image_diagnostics)
    for i in [0, 2]:  # different source redshifts
        family = state.family_data[int(state.images.family_index_per_image[i])]
        packed = evaluator.packed_state_for_z(g['physical'], family.z_source)
        point = np.array([state.images.x_obs[i], state.images.y_obs[i]])
        delta = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])*1e-4
        _, _, a, b, c, d = solver.ray_shoot_and_jacobian(packed, *(point+delta).T)
        jac = np.stack([a,b,c,d], axis=-1).reshape(-1,2,2)
        finite = np.stack([(jac[0]-jac[1])/2e-4, (jac[2]-jac[3])/2e-4], axis=-1)
        np.testing.assert_allclose(g['b'][i], finite, rtol=2e-6, atol=1e-9)
        inverse = np.linalg.inv(g['a'][i]); eig = np.linalg.eigvalsh(inverse@g['c'][i]@inverse.T)
        np.testing.assert_allclose(np.sort(g['widths'][i]**2), eig, rtol=1e-12)


def test_small_catalog_pdf_and_style(synthetic_config, tmp_path, monkeypatch):
    state=build_state(synthetic_config); evaluator=Evaluator(state)
    # FamilyData order is not necessarily the flattened image-family order.
    state.family_data = list(reversed(state.family_data))
    posterior=SimpleNamespace(samples=np.array([state.initial_theta]), log_prob=np.array([0.]))
    paths=SimpleNamespace(tables=tmp_path/'tables',figures=tmp_path/'figures')
    wcs=WCS(naxis=2);wcs.wcs.crpix=[1000,1000];wcs.wcs.crval=list(state.reference[1:]);wcs.wcs.cdelt=[-.05/3600,.05/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    band=SimpleNamespace(wcs=wcs,shape=(2000,2000),pixel_scale_arcsec=.05,path=tmp_path/'synthetic.fits')
    monkeypatch.setattr(cc,'rgb_cutout',lambda *args: np.full((120,120,3),.2))
    style=dict(plt.rcParams)
    path=cc.write_covariance_cutouts(synthetic_config,state,evaluator,posterior,paths,({'F814W':band},('F814W',),None,'toy'))
    assert path.read_bytes().startswith(b'%PDF')
    assert dict(plt.rcParams)==style
    metadata=json.loads((paths.tables/'small_lambda_covariance_cutouts.json').read_text())
    rows=metadata['records'];assert len(rows)==4
    assert [r['lambda_abs'] for r in rows]==sorted(r['lambda_abs'] for r in rows)
    assert {r['z_source'] for r in rows}=={2.,3.}
    assert all(r['z_source'] == {'1': 2., '2': 3.}[r['family_id']] for r in rows)
    assert all(r['sigma_meas']>=.1 for r in rows)
    assert all(r['sampled_max_exact_map_error'] is not None for r in rows)


@pytest.mark.parametrize("cluster,missing_hst,expected", [("toy",False,1),("m0416",False,2),("as1063",True,1),("hera",False,1),("ares",False,1)])
def test_output_reuses_loaded_imaging(monkeypatch, synthetic_config, tmp_path, cluster, missing_hst, expected):
    from dataclasses import replace
    from lenscluster import outputs, cutouts, mass_figure, chain_traces
    config=replace(synthetic_config,paths=replace(synthetic_config.paths,run_name=cluster),image_catalog=replace(synthetic_config.image_catalog,image_dir='already-loaded'))
    state=SimpleNamespace(parameter_specs=[],reference=(3,0,0),z_lens=.3,z_source_values=[2.])
    posterior=SimpleNamespace(samples=np.zeros((1,1)),physical=np.zeros((1,1)),log_prob=np.zeros(1))
    evaluator=SimpleNamespace(physical=lambda p:p,source_positions=lambda p:np.zeros((1,2)))
    paths=SimpleNamespace(tables=tmp_path/'tables',figures=tmp_path/'figures',fits=tmp_path/'fits',root=tmp_path)
    for name in ['write_chain_traces','write_potfile_plots','write_covariance_plots','write_member_diagnostic_plots','write_run_results','write_posterior_block_corner','write_gp_plots','write_aperture_mass_histogram']:
        monkeypatch.setattr(outputs,name,lambda *a,**k:None)
    for name in ['posterior_summary_table','classification_table']:
        monkeypatch.setattr(outputs,name,lambda *a:pd.DataFrame())
    monkeypatch.setattr(outputs.plots,'member_scaling_table',lambda *a:pd.DataFrame())
    for name in ['plot_image_residual_histogram','plot_image_recovery','plot_corner','plot_critical_curves']:
        monkeypatch.setattr(outputs.plots,name,lambda *a,**k:None)
    monkeypatch.setattr(chain_traces,'trace_data',lambda *a:None)
    monkeypatch.setattr(outputs,'compute_truth_recovery',lambda *a,**k:None)
    monkeypatch.setattr(outputs,'image_recovery_tables',lambda *a,**k:{'image_fit_quality':pd.DataFrame({'image':['a']}),'image_count_recovery':pd.DataFrame(),'image_recovery_extra_images':pd.DataFrame()})
    monkeypatch.setattr(outputs,'fit_summary',lambda *a:{'observed_image_count':1})
    imaging=cluster if cluster in {"hera","ares"} else "test-imaging"
    def load(image_config, *args):
        if missing_hst and not image_config.figure_suffix:
            raise FileNotFoundError('HST unavailable')
        return (None,None,None,imaging)
    monkeypatch.setattr(cutouts,'load_cluster_imaging',load)
    monkeypatch.setattr(mass_figure,'render_cluster_mass_figure',lambda *a,**k:None)
    calls=[];monkeypatch.setattr(outputs,'write_covariance_cutouts',lambda *a,**k:calls.append(a[5]))
    from lenscluster import image_position_contours
    monkeypatch.setattr(image_position_contours,"load_or_compute_contours",lambda *a,**k:None)
    from lenscluster import small_lambda_jwst
    lambda_calls=[]
    monkeypatch.setattr(small_lambda_jwst,'write_small_lambda_jwst',lambda *a,**k:lambda_calls.append(a))
    from lenscluster import jwst_rgb_figure
    rgb_calls=[]
    monkeypatch.setattr(jwst_rgb_figure,'render_rgb_critical',lambda *a,**k:rgb_calls.append(a))
    fold_calls=[]; fold_renders=[]
    monkeypatch.setattr(outputs,"compute_fold_diagnostics",lambda *a,**k: fold_calls.append(1) or {"pairs":[],"systems":[]})
    monkeypatch.setattr(outputs,"save_fold_data",lambda *a:None)
    monkeypatch.setattr(outputs,"render_fold_diagnostics",lambda *a,**k:fold_renders.append(a[4]))
    outputs.write_outputs(config,state,evaluator,posterior,{},paths,log=lambda _:None)
    assert len(rgb_calls)==(1 if cluster in {"hera","ares"} else 0)
    assert all(call[7].name.endswith("hst_rgb_critical_z9.pdf") for call in rgb_calls)
    assert len(lambda_calls)==(1 if cluster in {"m0416","as1063"} else 0)
    assert all(call[5][3] is imaging for call in lambda_calls)
    assert len(fold_calls)==1 and len(fold_renders)==expected
    assert len(calls)==expected and all(call[3] is imaging for call in calls)


def test_default_jwst_variants(synthetic_config):
    from dataclasses import replace
    from lenscluster.cutouts import imaging_variants
    from lenscluster.rgb import DEFAULT_JWST_RGB_BANDS
    from lenscluster.config import RGBDisplayConfig
    for name in ['m0416_bergamini2023', 'as1063']:
        config = replace(synthetic_config, paths=replace(synthetic_config.paths, run_name=name))
        primary, jwst = imaging_variants(config)
        assert primary is config.image_catalog
        assert jwst.bands == tuple(DEFAULT_JWST_RGB_BANDS)
        assert jwst.rgb == RGBDisplayConfig()
        assert jwst.figure_suffix == '_jwst'
        assert jwst.image_dir == 'data/JWST_Images'
        assert imaging_variants(replace(config, image_catalog=jwst)) == [jwst]
    assert imaging_variants(synthetic_config) == [synthetic_config.image_catalog]
