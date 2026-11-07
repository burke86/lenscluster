"""Eigenvalue ranking must select the near-singular direction, including saddles."""
import numpy as np
import pytest

from lenscluster import small_lambda_jwst as module


def test_near_zero_eigenvalue_not_most_negative():
    a=np.array([np.diag([-.9,.01]),np.diag([-.02,.8]),np.diag([0.,1.])])
    value,asymmetry=module.singular_eigenvalue(a)
    np.testing.assert_allclose(value,[.01,-.02,0.]);assert asymmetry==0
    angle=.71
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rotated=r@a@r.T
    np.testing.assert_allclose(module.singular_eigenvalue(rotated)[0],value,atol=1e-14)


def test_reject_nonpotential_or_nonfinite_jacobian():
    with pytest.raises(ValueError,match='not symmetric'):
        module.singular_eigenvalue(np.array([[[1.,.2],[0.,.1]]]))
    with pytest.raises(ValueError,match='Nonfinite'):
        module.singular_eigenvalue(np.full((1,2,2),np.nan))


def test_render_and_reuse_without_new_fields(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import json
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    state=NS(images=NS(image_labels=['a'],x_obs=[0.],y_obs=[0.],
                      family_index_per_image=[0],family_ids=['f']),
             family_data=[NS(family_id='f',z_source=2.)],reference=None)
    config=NS(paths=NS(figure_prefix='test'),image_catalog=NS(image_dir='test'))
    paths=NS(tables=tmp_path,figures=tmp_path)
    from lenscluster import cutout_critical_fields
    monkeypatch.setattr(cutout_critical_fields,'median_cutout_determinants',
        lambda *a,**k:(np.zeros((1,2,2,2)),np.ones((1,2,2)),np.arange(32),np.ones((1,2,2)),np.ones((1,2,2))))
    calls=[]
    def fields(t,**kwargs):
        calls.append(kwargs)
        return {'jacobian':jnp.array([[[.01,0.],[0.,.8]]])}
    evaluator=NS(physical=lambda t:t,fields=fields,gp_knee=False)
    posterior=NS(samples=np.zeros((32,1)))
    data=dict(recovered=np.random.default_rng(1).normal(0,.1,(32,1,2)),
              image_labels=np.array(['a']),draw_indices=np.arange(32))
    contours=dict(position_data=data,fingerprint='test-fingerprint')
    ref=NS(pixel_scale_arcsec=.03,wcs=NS(world_to_pixel_values=lambda x,y:(x/.03+50,y/.03+50)))
    imaging=({'band':ref},('band',),None,'test')
    monkeypatch.setattr(module,'rgb_reference_band',lambda _: 'band')
    monkeypatch.setattr(module,'arcsec_to_skycoord',lambda *a:None)
    monkeypatch.setattr(module,'arcsec_to_radec',lambda x,y,r:(x,y))
    monkeypatch.setattr(module,'cutout_window',lambda *a:NS(x0=0,y0=0))
    monkeypatch.setattr(module,'rgb_cutout',lambda *a:np.zeros((100,100,3)))
    before=plt.rcParams.copy()
    output=module.write_small_lambda_jwst(config,state,evaluator,posterior,paths,imaging,contours,minimum_major_sigma=0,minimum_axis_ratio=1,log=lambda _:None)
    assert output.exists()
    assert calls and calls[0]==dict(force_exact=True,derivative=False)
    assert plt.rcParams['font.size']==before['font.size']
    info=json.loads((tmp_path/'test_small_lambda_a_image_posteriors_jwst.json').read_text())
    assert info['selected'][0]['probabilities']==[.682689492,.954499736]
    assert not info['selected'][0]['contour_clipped']
    with np.load(tmp_path/'test_small_lambda_a_image_posteriors_jwst.npz') as saved:
        assert saved['posterior_axis_statistics'].shape==(1,3)
        assert saved['posterior_axis_statistics'][0,2]>=1
    evaluator.fields=lambda *a,**k:pytest.fail('Valid cache must avoid field evaluation')
    module.write_small_lambda_jwst(config,state,evaluator,posterior,paths,imaging,contours,minimum_major_sigma=0,minimum_axis_ratio=1,log=lambda _:None)
    assert module.write_small_lambda_jwst(config,state,evaluator,posterior,paths,imaging,contours,
                                         minimum_major_sigma=100,log=lambda _:None) is None
    import pandas as pd
    rejected=pd.read_csv(tmp_path/'test_small_lambda_a_image_posteriors_jwst.csv')
    assert not rejected.shown.any()
    assert rejected.skip_reason.iloc[0]=='Below minimum uncertainty extent or elongation'


def test_posterior_elongation_rotation_and_unavailable_draws():
    t=np.linspace(0,2*np.pi,100,endpoint=False)
    points=np.column_stack([4*np.cos(t),np.sin(t)])
    angle=.7
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    data=np.stack([points,points@rotation.T,np.zeros_like(points),np.full_like(points,np.nan)],axis=1)
    data[:9,3]=points[:9]
    result=module.posterior_axis_statistics(data)
    np.testing.assert_allclose(result[:2,2],4,rtol=1e-14)
    np.testing.assert_allclose(result[0],result[1],rtol=1e-14)
    assert np.isnan(result[2:]).all()
