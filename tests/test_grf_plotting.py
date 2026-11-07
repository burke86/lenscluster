from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from lenscluster import gp_knee
from lenscluster.grf_plotting import SmoothPlotDeflection


def toy_field(monkeypatch,modes=()):
    r=np.linspace(0,12,1201);ct=np.exp(-r*r/8);cl=(1-r*r/4)*ct
    monkeypatch.setattr(gp_knee,'knee_curves',lambda *args:(jnp.asarray(cl),jnp.asarray(ct)))
    xy=np.array([[[0.,0.],[1.,.5],[-.7,1.1]]])
    layout=dict(xy=jnp.asarray(xy),mask=jnp.ones((1,3),bool),weights=jnp.ones((1,3)),block_of_family=jnp.array([0,0]))
    tables=dict(r_grid=jnp.asarray(r))
    coef=np.array([[[.2,-.1],[-.3,.4],[.1,.3]]])
    args=(layout,tables,coef,1.,.7,np.array([1.,.6]))
    return SmoothPlotDeflection(*args,modes=modes),args


def test_plot_field_matches_saved_radial_nodes(monkeypatch):
    smooth,args=toy_field(monkeypatch)
    # A one-image layout isolates exact preservation of both radial kernel values.
    layout,tables,coef,lam,amp,weights=args
    layout=dict(xy=jnp.zeros((1,1,2)),mask=jnp.ones((1,1),bool),weights=jnp.ones((1,1)),block_of_family=jnp.array([0,0]))
    coef=coef[:,:1]
    smooth=SmoothPlotDeflection(layout,tables,coef,lam,amp,weights)
    original=gp_knee.MeanDeflectionKnee(layout,tables,coef,lam,amp,weights)
    x=np.linspace(.1,3.,30);y=np.zeros_like(x);families=np.zeros(len(x),int)
    np.testing.assert_allclose(np.array(smooth(x,y,families)[:2]),np.array(original(x,y,families)[:2]),atol=1e-13)


def test_plot_derivatives_are_consistent_continuous_and_curl_free(monkeypatch):
    field,_=toy_field(monkeypatch,modes=('translation','shear'))
    points=np.array([[.04,.08],[.2,.1],[0.,0.],[1.,.5],[-.3,1.]])
    families=np.array([0,1,0,1,0]);c=field(*points.T,families)
    jac=np.stack(c[2:],axis=-1).reshape(-1,2,2)
    numerical=np.empty_like(jac)
    for axis in range(2):
        step=np.zeros(2);step[axis]=1e-6
        a=np.column_stack(field(*(points+step).T,families)[:2])
        b=np.column_stack(field(*(points-step).T,families)[:2])
        numerical[:,:,axis]=(a-b)/2e-6
    np.testing.assert_allclose(jac,numerical,atol=2e-9,rtol=1e-7)
    np.testing.assert_allclose(jac[:,0,1],jac[:,1,0],atol=1e-13)
    jumps=[]
    for eps in [1e-5,1e-6]:
        c=field(np.array([.5-eps,.5+eps]),np.zeros(2),np.zeros(2,int))
        jumps.append(np.linalg.norm(np.array(c[2:])[:,1]-np.array(c[2:])[:,0]))
    assert jumps[1]<.11*jumps[0]


def test_radial_refinement_converges_to_analytic_potential_kernel(monkeypatch):
    layout=dict(xy=jnp.zeros((1,1,2)),mask=jnp.ones((1,1),bool),weights=jnp.ones((1,1)),block_of_family=jnp.array([0]))
    u=np.array([.2,-.1]);points=np.array([[.137,.231],[.793,-.419],[1.233,.831],[2.371,-.325]])
    r2=np.sum(points**2,axis=1);ct=np.exp(-r2/8);du=points@u
    truth=ct[:,None,None]*(
        -(u[None,:,None]*points[:,None,:]+points[:,:,None]*u[None,None,:]+du[:,None,None]*np.eye(2))/4
        +du[:,None,None]*points[:,:,None]*points[:,None,:]/16)
    errors=[]
    for step in [.04,.02,.01]:
        r=np.arange(0.,12.+step,step);ct_nodes=np.exp(-r*r/8);cl_nodes=(1-r*r/4)*ct_nodes
        monkeypatch.setattr(gp_knee,'knee_curves',lambda *args:(jnp.asarray(cl_nodes),jnp.asarray(ct_nodes)))
        field=SmoothPlotDeflection(layout,dict(r_grid=jnp.asarray(r)),u[None,None],1.,1.,np.ones(1))
        actual=np.stack(field(*points.T,np.zeros(len(points),int))[2:],axis=-1).reshape(-1,2,2)
        errors.append(np.max(abs(actual-truth)))
    assert np.all(np.diff(errors)<0)
    assert errors[-1]<1e-6


def test_wide_plot_extends_tables_without_changing_conditioning():
    from lenscluster.grf_plotting import plot_deflection_correction
    tables=gp_knee.knee_tables(r_max=2.,r_step=.05,lam_min=1.,lam_max=4.,n_lam=4,
        scale_min=.5,scale_max=10.,gamma=4.)
    layout=dict(xy=jnp.zeros((1,1,2)),mask=jnp.ones((1,1),bool),
                weights=jnp.ones((1,1)),block_of_family=jnp.array([0]))
    coefficients=jnp.array([[[.2,-.1]]])
    evaluator=SimpleNamespace(gp_knee=True,gp_knee_spectrum='knee',gp_sigma_alpha_index=0,
        gp_groups=None,gp=layout,gp_tables=tables,gp_weights_per_family=np.ones(1),
        gp_knee_gamma_index=-1,_knee_mean_inputs=lambda theta:(coefficients,jnp.array([.7,2.])))
    field=plot_deflection_correction(evaluator,np.zeros(1),max_separation=4.)
    assert evaluator.gp_tables is tables
    assert float(tables['r_grid'][-1])==2.
    extended=evaluator._plot_extended_tables
    for key in ('c_l','c_t','d2_l','d2_t'):
        np.testing.assert_array_equal(extended[key][:len(tables['r_grid'])],tables[key])
    values=field(np.array([3.5]),np.array([.2]),np.array([0]))
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[3],values[4],atol=1e-12)
    plot_deflection_correction(evaluator,np.zeros(1),max_separation=3.)
    assert evaluator._plot_extended_tables is extended
