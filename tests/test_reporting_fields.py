"""Analytic regression for combined reporting fields, independent of lens kernels."""
from types import SimpleNamespace
import numpy as np
from lenscluster.reporting_fields import shared_grf_fields
from lenscluster.truth_recovery import lens_quantities


def evaluator(monkeypatch):
    from lenscluster import jax_cosmology
    monkeypatch.setattr(jax_cosmology, 'dpie_sigma0_factor_from_config', lambda zl,z,c: z)
    def correction(x,y,f):
        return 3*x*x, 3*y*y, 6*x, 0*x, 0*x, 6*y
    return SimpleNamespace(gp_knee=True, state=SimpleNamespace(gp_blocks='all',z_lens=.3,
        gp_reference_redshift=2.,cosmo_config=None), gp_weights_per_family=np.array([3.]),
        gp_deflection_correction=lambda theta: correction, physical=lambda t:t,
        packed_state_for_z=lambda p,z:z,
        total_fields_at=lambda xy,p:(np.zeros_like(xy),np.broadcast_to(np.diag([.1,.2]),(len(xy),2,2))))


def test_shared_fields_redshift_and_jacobian(monkeypatch):
    e=evaluator(monkeypatch)
    xy=np.array([[.01,.02],[.03,.04],[.05,.06]])
    a,u=shared_grf_fields(e,None,4.,xy,chunk=2)
    np.testing.assert_allclose(a,2*xy**2)
    np.testing.assert_allclose(u[:,0,0],4*xy[:,0])
    np.testing.assert_allclose(u[:,1,1],4*xy[:,1])
    eps=1e-6
    for k in range(2):
        shift=np.eye(2)[k]*eps
        ap,_=shared_grf_fields(e,None,4.,xy+shift,chunk=2)
        am,_=shared_grf_fields(e,None,4.,xy-shift,chunk=2)
        np.testing.assert_allclose((ap-am)/(2*eps),u[:,:,k],atol=1e-12)


def test_total_magnification_uses_total_jacobian_once(monkeypatch):
    e=evaluator(monkeypatch)
    x,y=np.array([.01,.03]),np.array([.02,.04])
    q=lens_quantities(e,np.zeros(1),4.,x,y)
    det=(.9-4*x)*(.8-4*y)
    np.testing.assert_allclose(q['kappa'],.15+2*(x+y))
    np.testing.assert_allclose(q['detA'],det)
    np.testing.assert_allclose(q['mu'],1/det)
    p=lens_quantities(e,np.zeros(1),4.,x,y,include_grf=False)
    np.testing.assert_allclose(p['kappa'],.15)
    np.testing.assert_allclose(p['mu'],1/.72)


def test_no_field_and_empty_points(monkeypatch):
    e=evaluator(monkeypatch)
    assert shared_grf_fields(e,None,4.,np.empty((0,2)))[0].shape==(0,2)
    for blocks in ['family','system']:
        e.state.gp_blocks=blocks
        a,u=shared_grf_fields(e,None,4.,[[1.,2.]])
        assert not a.any() and not u.any()
    e.state.gp_blocks='all'
    e.gp_deflection_correction=lambda theta:None
    assert not shared_grf_fields(e,None,4.,[[1.,2.]])[1].any()


def test_observed_stretch_uses_corrected_jacobian():
    import pandas as pd
    from lenscluster.mass_figure import arc_direction_table
    # Parametric stretch is x, but the conditional mass field changes it to y.
    image=SimpleNamespace(image_labels=['a'],x_obs=np.array([0.]),y_obs=np.array([0.]),
                          family_index_per_image=np.array([0]))
    def correction(x,y,f):
        return x*0,y*0,x*0-.6,x*0,x*0,x*0+.7
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(images=image),physical=lambda p:p,
        fields=lambda *a,**k:{'jacobian':np.array([np.diag([.1,.8])])},
        gp_deflection_correction=lambda theta:correction)
    result=arc_direction_table(e,np.zeros(1),pd.DataFrame({'image_label':['a']}))
    np.testing.assert_allclose(result[['e_arc_x','e_arc_y']],[[0.,1.]])
    np.testing.assert_allclose(result.s_arc,.1)


def test_physical_median_reconstructs_gp_sources(monkeypatch):
    import jax.numpy as jnp
    from lenscluster import model
    from lenscluster.reporting_fields import posterior_median_theta
    monkeypatch.setattr(model, 'convert_theta_to_latent', lambda p,s:np.array(p,copy=True))
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(parameter_specs=[],images=SimpleNamespace(
        source_x_param_index=np.array([1]),source_y_param_index=np.array([2]))),
        gp=dict(knot_mask=np.array([[True]]),knot_family=np.array([[0]]),
                real_block=np.array([0]),real_knot=np.array([0]),real_family=np.array([0])))
    e._padded_evaluator=lambda:e
    e.statistics=lambda t:{'whitening':{'cholesky':jnp.array([[[2.,0.],[0.,2.]]]),
                                      'centroid':jnp.array([[t[0],t[0]]])}}
    e.reported_physical=lambda t:jnp.array([t[0],2*t[1]+t[0],2*t[2]+t[0]])
    p=SimpleNamespace(physical=np.array([[1.,10.,14.],[5.,12.,17.],[9.,20.,18.]]))
    theta=posterior_median_theta(e,p)
    np.testing.assert_allclose(theta,[5.,3.5,6.])
    np.testing.assert_allclose(e.reported_physical(theta),np.median(p.physical,axis=0))

