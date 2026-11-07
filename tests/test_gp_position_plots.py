from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from lenscluster.gp_position_plots import conditional_displacement,gaussian_mixture_grid,add_gp_position_diagnostics


def test_gp_sign_and_conditional_variance_for_one_observation():
    observed=np.zeros((1,2));jac=np.eye(2)[None]
    chol=np.sqrt(1.01)*np.eye(2)
    mean,cov=conditional_displacement(observed,observed,jac,chol,np.array([[1.,-.4]]),1.,1.)
    np.testing.assert_allclose(mean,[[1/1.01,-.4/1.01]],atol=1e-14)
    np.testing.assert_allclose(cov[0],np.eye(2)*(1-1/1.01),atol=1e-14)
    # F = observed - gravitational prediction. Positive d moves it toward data.
    assert np.linalg.norm(np.array([1.,-.4])-mean[0])<np.linalg.norm([1.,-.4])


def test_gp_moments_match_direct_full_covariance_calculation():
    rng=np.random.default_rng(4);observed=rng.normal(size=(3,2));query=rng.normal(size=(4,2))
    jac=rng.normal(size=(3,2,2));sigma=.2;ell=.8
    k=sigma**2*np.exp(-np.sum((observed[:,None]-observed[None])**2,axis=-1)/(2*ell**2))
    b=np.zeros((6,6))
    for i in range(3):b[2*i:2*i+2,2*i:2*i+2]=jac[i]
    grf=rng.normal(size=(6,6));c=b@np.kron(k,np.eye(2))@b.T+.1*grf@grf.T+.03*np.eye(6)
    residual=rng.normal(size=(3,2));chol=np.linalg.cholesky(c)
    means,covs=conditional_displacement(query,observed,jac,chol,residual,sigma,ell)
    kq=sigma**2*np.exp(-np.sum((query[:,None]-observed[None])**2,axis=-1)/(2*ell**2))
    cross=np.kron(kq,np.eye(2))@b.T
    expected=cross@np.linalg.solve(c,residual.ravel())
    full=sigma**2*np.eye(8)-cross@np.linalg.solve(c,cross.T)
    np.testing.assert_allclose(means,expected.reshape(4,2),atol=1e-14)
    for q in range(4):np.testing.assert_allclose(covs[q],full[2*q:2*q+2,2*q:2*q+2],atol=1e-14)


def test_gp_contours_include_latent_variance_with_identical_posterior_means():
    means=np.zeros((20,2));cov=np.broadcast_to(np.array([[.04,.01],[.01,.01]]),(20,2,2))
    grid=gaussian_mixture_grid(means,cov)
    assert grid is not None and abs(grid['integrated_mass']-1)<.002
    for level,p in zip(grid['levels'],[.954499736,.682689492]):
        actual=grid['density'][grid['density']>=level].sum()*grid['cell_area']
        assert abs(actual-p)<.003
    assert np.ptp(grid['x'])>.5


def test_gp_position_cache_and_missing_gravitational_roots():
    class Evaluator:
        gp_knee=True
        gp=dict(slots=jnp.array([[0]]))
        def _padded_evaluator(self):return self
        def statistics(self,t):
            return dict(fields=dict(jacobian=jnp.eye(2)[None],backprojected=jnp.array([[1.,0.]])),
                source_by_family=jnp.zeros((1,2)),physical=t,family_cholesky=(jnp.sqrt(1.01)*jnp.eye(2))[None])
        def error_scales(self,p):return dict(sigma_shared=jnp.array(1.),gp_ell=jnp.array(1.))
    images=SimpleNamespace(x_obs=np.array([1.]),y_obs=np.array([0.]),family_index_per_image=np.array([0]),image_labels=['A.a'])
    state=SimpleNamespace(gp_blocks='all',images=images)
    positions=np.array([[[0.,0.]],[[np.nan,np.nan]]])
    system=dict(labels=['A.a'],positions=positions,observed=np.array([[1.,0.]]))
    data=dict(systems=[system],draw_indices=[0,1])
    posterior=SimpleNamespace(samples=np.array([[0.],[1.]]))
    add_gp_position_diagnostics(state,Evaluator(),posterior,data,log=lambda _:None)
    np.testing.assert_allclose(system['gp_adjusted_mean_positions'][0,0],[np.exp(-.5)/1.01,0.],atol=1e-14)
    assert np.isnan(system['gp_adjusted_mean_positions'][1]).all()
    broken=SimpleNamespace(gp_knee=True,_padded_evaluator=lambda:(_ for _ in ()).throw(AssertionError('cache missed')))
    add_gp_position_diagnostics(state,broken,posterior,data,log=lambda _:None)
