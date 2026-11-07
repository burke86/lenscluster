"""Fixed-lens pixel inversion: profile parity, gradients, and source recovery."""
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_fold_source_light import make_renderer, fit_pixels, gaussian_psf


def test_profile_matches_lenstronomy_and_gradient():
    from lenstronomy.LightModel.Profiles.gaussian import Gaussian
    render, source=make_renderer((12,14),2,1)
    rng=np.random.default_rng(1); beta=rng.normal(0,.2,(24*28,2))
    p=np.array([np.log(2.),np.log(.16),.04,-.03,.1])
    expected=Gaussian().function(beta[:,0],beta[:,1],2.,.16,.04,-.03)
    np.testing.assert_allclose(source(p,beta),expected,rtol=2e-14,atol=1e-13)
    kernel=gaussian_psf(2.)
    analytic=np.asarray(jax.jacfwd(render)(p,beta,kernel))
    for i in range(5):
        d=np.eye(5)[i]*1e-6
        numeric=(np.asarray(render(p+d,beta,kernel))-np.asarray(render(p-d,beta,kernel)))/2e-6
        np.testing.assert_allclose(analytic[...,i],numeric,rtol=2e-6,atol=2e-7)


def test_synthetic_fold_recovers_source_with_psf():
    shape=(24,28); ss=2
    y,x=np.mgrid[:shape[0]*ss,:shape[1]*ss]
    x=(x+.5)/(shape[1]*ss)-.5; y=(y+.5)/(shape[0]*ss)-.5
    # Canonical fold mapping beta_y = theta_y^2; both images share one source.
    beta=np.c_[x.ravel(),(2*y*y).ravel()]
    render,_=make_renderer(shape,ss,1);psf=gaussian_psf(1.8)
    truth=np.array([np.log(.7),np.log(.08),.025,.12,.04])
    data=np.asarray(render(truth,beta,psf));noise=np.full(shape,.02);mask=np.ones(shape,bool)
    initial=truth+np.array([.2,.1,.01,-.02,.01])
    bounds=(np.array([-5,-5,-.3,0,-1]),np.array([4,0,.3,.4,1]))
    fit=fit_pixels(render,jnp.asarray(beta),psf,data,noise,mask,initial,bounds,100)
    assert fit.success
    np.testing.assert_allclose(fit.x,truth,atol=1e-7)
    assert 2*fit.cost < 1e-12


def test_surface_brightness_and_psf_normalization():
    render,_=make_renderer((20,20),3,1)
    p=np.array([0.,np.log(.1),0,0,.2]);beta=np.zeros((60*60,2))
    psf=gaussian_psf(2.)
    np.testing.assert_allclose(psf.sum(),1.)
    expected=1/(2*np.pi*.1**2)+.2
    np.testing.assert_allclose(np.asarray(render(p,beta,psf))[8:12,8:12],expected,rtol=1e-13)


def test_auxiliary_map_registration(tmp_path):
    import pytest
    from astropy.io import fits
    from astropy.wcs import WCS
    from fit_fold_source_light import crop_plane
    w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crval=[64.,-24.]
    w.wcs.crpix=[10,10];w.wcs.cdelt=[-1e-5,1e-5]
    a=np.arange(400.).reshape(20,20);path=tmp_path/'error.fits'
    fits.writeto(path,a,w.to_header())
    slices=(slice(3,8),slice(5,9))
    np.testing.assert_array_equal(crop_plane(path,None,slices,w,a.shape),a[slices])
    shifted=w.deepcopy();shifted.wcs.crpix[0]+=1
    with pytest.raises(AssertionError):crop_plane(path,None,slices,shifted,a.shape)


def test_pixel_source_matches_lenstronomy_interpol():
    from lenstronomy.LightModel.Profiles.interpolation import Interpol
    from fit_fold_source_light import make_pixel_renderer
    rng=np.random.default_rng(4); grid=rng.uniform(0,3,(9,11)); origin=np.array([-.2,-.3]);spacing=.04
    beta=rng.uniform(origin-.1,origin+np.array([10,8])*spacing+.1,(1000,2))
    _,source=make_pixel_renderer((10,10),1,grid.shape,origin,spacing)
    center=origin+np.array([10,8])*spacing/2
    expected=Interpol().function(beta[:,0],beta[:,1],image=grid,center_x=center[0],center_y=center[1],scale=spacing)
    actual=source(np.r_[grid.ravel(),0.,0.,0.],beta)
    np.testing.assert_allclose(actual,expected,atol=2e-14)


def test_positive_pixel_fit_reproduces_two_fold_images():
    from fit_fold_source_light import make_pixel_renderer,fit_source_grid
    shape=(22,22);ss=2;grid_shape=(12,12);origin=(-.3,-.02);spacing=.055
    y,x=np.mgrid[:shape[0]*ss,:shape[1]*ss];x=(x+.5)/(shape[1]*ss)-.5;y=(y+.5)/(shape[0]*ss)-.5
    beta=np.c_[x.ravel(),(2*y*y).ravel()]
    gy,gx=np.mgrid[:12,:12];grid=np.exp(-((gx-4)**2+(gy-3)**2)/2)+.7*np.exp(-((gx-7)**2+(gy-5)**2)/.9)
    truth=np.r_[grid.ravel(),.02,.01,-.005]
    render,_=make_pixel_renderer(shape,ss,grid_shape,origin,spacing);psf=gaussian_psf(1.7)
    data=np.asarray(render(truth,beta,psf));noise=np.full(shape,.01);mask=np.ones(shape,bool)
    fit=fit_source_grid(render,beta,psf,data,noise,mask,np.r_[np.ones(144)*.05,0,0,0],grid_shape,.001,2000)
    assert fit.cost/mask.sum() < .01
    assert fit.x[:144].min() >= 0


def test_matching_rms_and_psf_for_original_and_bgsub(tmp_path):
    from fit_fold_source_light import matching_product
    rms=tmp_path/'hlsp_test_rms.fits.gz';rms.touch()
    psf=tmp_path/'hlsp_test_psf.fits';psf.touch()
    assert matching_product(tmp_path/'hlsp_test_sci.fits','rms')==rms
    assert matching_product(tmp_path/'hlsp_test_bgsub-sci.fits.gz','rms')==rms
    assert matching_product(tmp_path/'hlsp_test_bgsub-sci.fits.gz','psf')==psf
    assert matching_product(tmp_path/'other.fits','rms') is None


def test_gp_inverse_and_conditional_mean_agreement():
    from scipy.linalg import cho_solve
    from lenscluster.gp_position_plots import conditional_displacement
    from fit_fold_source_light import inverse_gp_coordinates
    rng=np.random.default_rng(12)
    observed=rng.normal(0,.3,(5,2));jac=rng.normal(0,.2,(5,2,2))+np.eye(2)
    chol=np.eye(10)*10;residual=rng.normal(0,.3,(5,2));sigma=.2;ell=.4
    v=cho_solve((chol,True),residual.ravel()).reshape(-1,2)
    weights=np.einsum('nca,nc->na',jac,v)
    gravitational=rng.normal(0,.4,(40,2))
    displacement,_=conditional_displacement(gravitational,observed,jac,chol,residual,sigma,ell)
    observed_grid=gravitational+displacement
    inverse,info=inverse_gp_coordinates(observed_grid,observed,weights,sigma,ell,chunk=32)
    np.testing.assert_allclose(inverse,gravitational,atol=2e-14)
    assert info['minimum_warp_determinant']>0
    zero,_=inverse_gp_coordinates(observed_grid,observed,weights,0.,ell,chunk=32)
    np.testing.assert_array_equal(zero,observed_grid)


def test_sparse_projection_matches_direct_light_and_gradients():
    from fit_fold_source_light import make_pixel_renderer
    rng=np.random.default_rng(91);shape=(14,12);ss=4;gs=(10,11);origin=(-.15,-.1);spacing=.03
    beta=rng.uniform(-.25,.4,(shape[0]*shape[1]*ss*ss,2));p=rng.uniform(0,1,10*11+3);psf=gaussian_psf(2)
    direct,_=make_pixel_renderer(shape,ss,gs,origin,spacing)
    sparse,_=make_pixel_renderer(shape,ss,gs,origin,spacing,beta=beta)
    np.testing.assert_allclose(sparse(p,beta,psf),direct(p,beta,psf),atol=1e-14)
    a=jax.grad(lambda p:jnp.sum(direct(p,beta,psf)**2))(p)
    b=jax.grad(lambda p:jnp.sum(sparse(p,beta,psf)**2))(p)
    np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)
