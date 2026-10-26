"""Cartesian profiles against independent principal-axis and differential references."""
import mpmath as mp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtronomy.LensModel.Profiles.nie_e import NIE_E, _atan_sqrt_ratio
from jaxtronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E
from lenstronomy.LensModel.Profiles.nie_e import NIE_E as NumNIE
from lenstronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E as NumDPIE
from jaxtronomy.LensModel.Profiles.nie import NIE
from jaxtronomy.LensModel.Profiles.dpie_nie import DPIENIE

jax.config.update('jax_enable_x64', True)


def reference_nie(x, y, theta, e1, e2, r):
    """80-digit principal-axis equations; no q clipping."""
    x,y,theta,e1,e2,r = map(mp.mpf,(x,y,theta,e1,e2,r))
    e=mp.sqrt(e1*e1+e2*e2)
    if not e:
        R=mp.sqrt(r*r+x*x+y*y)
        ax,ay=theta*x/(R+r),theta*y/(R+r)
        return (ax,ay), x*ax+y*ay-theta*r*mp.log(R+r)
    q=(1-e)/(1+e);phi=mp.atan2(e2,e1)/2
    c,s=mp.cos(phi),mp.sin(phi)
    X,Y=c*x+s*y,-s*x+c*y
    b=theta*mp.sqrt(q);core=r/mp.sqrt(q);t=mp.sqrt(1-q*q)
    psi=mp.sqrt(q*q*(core*core+X*X)+Y*Y)
    ax=b/t*mp.atan(t*X/(psi+core));ay=b/t*mp.atanh(t*Y/(psi+q*q*core))
    pot=X*ax+Y*ay-b*core/2*mp.log((psi+core)**2+t*t*X*X)
    return (c*ax-s*ay,s*ax+c*ay),pot


def reference_dpie(x,y,sigma,a,b,e1,e2):
    a,b,sigma=map(mp.mpf,(a,b,sigma))
    theta=2*sigma*a*b/(b-a)
    ai,pi=reference_nie(x,y,theta,e1,e2,a)
    ao,po=reference_nie(x,y,theta,e1,e2,b)
    return tuple(i-o for i,o in zip(ai,ao)),pi-po


@pytest.mark.parametrize('profile,numpy_profile,reference,is_dpie',[(NIE_E,NumNIE,reference_nie,False),(DPIE_NIE_E,NumDPIE,reference_dpie,True)])
def test_values_high_precision_and_numpy(profile,numpy_profile,reference,is_dpie):
    rng=np.random.default_rng(385)
    with mp.workdps(80):
        for i in range(80):
            x,y=rng.normal(size=2)*10
            e=0. if i<3 else rng.uniform(1e-12,.7)
            angle=rng.uniform(-np.pi,np.pi);e1,e2=e*np.cos(angle),e*np.sin(angle)
            a=np.exp(rng.uniform(-4,1));b=a+np.exp(rng.uniform(-5,3))
            normalization = np.exp(rng.uniform(-3, 3))
            args=(normalization,a,b,e1,e2) if is_dpie else (normalization,e1,e2,a)
            expected,pot=reference(x,y,*args)
            np.testing.assert_allclose(profile.derivatives(x,y,*args),np.array(expected,float),rtol=3e-14,atol=2e-15)
            np.testing.assert_allclose(profile.function(x,y,*args),float(pot),rtol=3e-14,atol=1e-13)
            for method in ['derivatives','function','hessian']:
                np.testing.assert_allclose(getattr(profile,method)(x,y,*args),getattr(numpy_profile,method)(x,y,*args),rtol=3e-14,atol=2e-13)


@pytest.mark.parametrize('e1,e2',[(0.,0.),(1e-14,-1e-14),(.1,.2),(-.4,.2)])
@pytest.mark.parametrize('profile,args',[(NIE_E,(3.,)),(DPIE_NIE_E,(3.,.2,8.))])
def test_potential_deflection_hessian_and_third_derivatives(profile,args,e1,e2):
    p=jnp.array([1.2,-2.3,e1,e2])
    def parameters(p):return (*args,p[2],p[3],.2) if profile is NIE_E else (*args,p[2],p[3])
    def potential(p):return profile.function(p[0],p[1],*parameters(p))
    def alpha(p):return jnp.stack(profile.derivatives(p[0],p[1],*parameters(p)))
    h=np.asarray(profile.hessian(p[0],p[1],*parameters(p))).reshape(2,2)
    np.testing.assert_allclose(jax.grad(potential)(p)[:2],alpha(p),rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(jax.jacfwd(alpha)(p)[:,:2],h,rtol=2e-14,atol=2e-14)
    # Includes mixed ellipticity/position derivatives at exact circularity.
    third=jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(alpha))))(p)
    assert np.isfinite(third).all()
    second=jax.jit(jax.jacfwd(jax.jacfwd(alpha)))
    step=1e-5
    for k in [2,3]:
        delta=jnp.zeros_like(p).at[k].set(step)
        fd=(second(p+delta)-second(p-delta))/(2*step)
        np.testing.assert_allclose(third[:,:,:,k],fd,rtol=2e-7,atol=2e-8)


def test_extreme_geometry_and_broadcasting():
    x=jnp.array([0.,1e-12,1.,1e6]);y=jnp.array([0.,-1e-12,0.,-2e6])
    fn=lambda p:jnp.stack(DPIE_NIE_E.derivatives(x,y,*p))
    p=jnp.array([2.,1.,1.+1e-10,0.,0.,0.,0.])
    assert np.isfinite(jax.jit(jax.jacfwd(jax.jacfwd(fn)))(p)).all()
    values=jax.jit(jax.vmap(fn))(jnp.stack([p,p.at[3].set(.1)]))
    assert values.shape==(2,2,4)
    with mp.workdps(80):
        for i in range(4):
            a,_=reference_dpie(float(x[i]),float(y[i]),*map(float,p[:5]))
            np.testing.assert_allclose(values[0,:,i],np.array(a,float),rtol=3e-14,atol=1e-25)


def test_series_switch_derivatives():
    with mp.workdps(80):
        for value in [0.,.009999,.010001,-.009999,-.010001]:
            f=lambda u:jnp.real(_atan_sqrt_ratio(u+0j))
            ref=lambda u:mp.hyper([mp.mpf('.5'),1],[mp.mpf('1.5')],-u)
            for order in range(4):
                np.testing.assert_allclose(f(jnp.asarray(value)),float(mp.diff(ref,mp.mpf(value),order)),rtol=(1e-9 if order == 3 else 3e-10),atol=2e-11)
                f=jax.grad(f)


def test_registered_models_and_legacy_names():
    from jaxtronomy.LensModel.lens_model import LensModel as JaxModel
    from jaxtronomy.LensModel.lens_model_bulk import LensModelBulk
    from lenstronomy.LensModel.lens_model import LensModel as NumModel
    kwargs=[dict(sigma0=2.,Ra=.2,Rs=8.,e1=0.,e2=0.)]
    for model in [JaxModel(['DPIE_NIE_E']),NumModel(['DPIE_NIE_E'])]:
        assert np.isfinite(model.alpha(.2,.3,kwargs)).all()
        assert np.isfinite(model.hessian(.2,.3,kwargs)).all()
    for cls in [JaxModel,NumModel]:
        model=cls(['NIE_E'])
        assert np.isfinite(model.alpha(.2,.3,[dict(theta_E=2.,s_scale=.2,e1=0.,e2=0.)])).all()
    LensModelBulk(unique_lens_model_list=['DPIE_NIE_E','NIE_E'], multi_plane=False)
    # Existing names remain available, with their original circular approximation.
    assert np.isfinite(NIE().derivatives(1.,2.,3.,0.,0.,.2)).all()
    assert np.isfinite(DPIENIE.derivatives(1.,2.,3.,.2,8.,0.,0.)).all()


@pytest.mark.parametrize('profile,reference,args', [
    (NIE_E, reference_nie, (2., .15, -.2, .3)),
    (NIE_E, reference_nie, (2., .3, 0., .3)),
    (NIE_E, reference_nie, (2., 0., 0., .3)),
    (DPIE_NIE_E, reference_dpie, (2., .3, 8., .15, -.2)),
    (DPIE_NIE_E, reference_dpie, (2., .3, 8., .3, 0.)),
    (DPIE_NIE_E, reference_dpie, (2., .3, .3000000001, 0., 0.)),
])
def test_spatial_hessian_high_precision(profile, reference, args):
    with mp.workdps(80):
        for x, y in [(0., 0.), (1., 0.), (0., 1.), (2., -3.), (1e6, -2e6)]:
            expected = np.array([
                [mp.diff(lambda xx: reference(xx, y, *args)[0][i], mp.mpf(x)),
                 mp.diff(lambda yy: reference(x, yy, *args)[0][i], mp.mpf(y))]
                for i in range(2)], float)
            actual = np.array(profile.hessian(x, y, *args)).reshape(2, 2)
            np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=2e-15)
            np.testing.assert_allclose(np.trace(actual)/2, np.trace(expected)/2,
                                       rtol=4e-14, atol=2e-15)


def test_spherical_mass_conventions():
    from lenstronomy.LensModel.Profiles.nie import NIE as LegacyNIE
    from lenstronomy.LensModel.Profiles.dpie_nie import DPIENIE as LegacyDPIE
    r = np.array([0., .1, 1., 100.])
    for profile, old, args in [
        (NIE_E, LegacyNIE(), (2., .1, .2, .3)),
        (DPIE_NIE_E, LegacyDPIE(), (2., .3, 8., .1, .2)),
    ]:
        np.testing.assert_allclose(profile.mass_3d_lens(r, *args),
                                   old.mass_3d_lens(r, *args), rtol=2e-13, atol=2e-14)


def test_all_parameter_derivatives_and_centers():
    p = jnp.array([2., .3, 8., 0., 0., .2, -.4])
    alpha = lambda p: jnp.stack(DPIE_NIE_E.derivatives(1., -2., *p))
    second = jax.jit(jax.jacfwd(jax.jacfwd(alpha)))
    third = jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(alpha))))(p)
    assert np.isfinite(third).all()
    for k in range(len(p)):
        step = 1e-5 * max(1., abs(float(p[k])))
        delta = jnp.zeros_like(p).at[k].set(step)
        fd = (second(p + delta) - second(p - delta)) / (2 * step)
        np.testing.assert_allclose(third[..., k], fd, rtol=3e-7, atol=2e-8)
    np.testing.assert_allclose(alpha(p), DPIE_NIE_E.derivatives(.8, -1.6, *p[:5]),
                               rtol=2e-15, atol=2e-15)


def test_near_equal_radius_potential_derivatives():
    # Differentiating log1p(gap*t)/gap directly loses its radius derivatives.
    with mp.workdps(80):
        for a, b in [(1., 1.+1e-10), (.3, 8.)]:
            for k in [1, 2]:
                args = [2., a, b, .1, -.2]
                f = lambda radius: DPIE_NIE_E.function(1., -2., *args[:k], radius, *args[k+1:])
                ref = lambda radius: reference_dpie(1., -2., *args[:k], radius, *args[k+1:])[1]
                for order in range(4):
                    np.testing.assert_allclose(f(jnp.asarray(args[k])),
                        float(mp.diff(ref, mp.mpf(args[k]), order)), rtol=3e-12, atol=3e-12)
                    f = jax.grad(f)


@pytest.mark.parametrize('profile', [NIE_E, DPIE_NIE_E])
def test_third_derivatives_on_ellipse_axes(profile):
    def alpha(p):
        args = (2., p[2], p[3], .3) if profile is NIE_E else (2., .3, 8., p[2], p[3])
        return jnp.stack(profile.derivatives(p[0], p[1], *args))
    second = jax.jit(jax.jacfwd(jax.jacfwd(alpha)))
    third = jax.jit(jax.jacfwd(jax.jacfwd(jax.jacfwd(alpha))))
    for x, y in [(1., 0.), (0., 1.)]:
        p = jnp.array([x, y, .3, 0.])
        result = third(p)
        assert np.isfinite(result).all()
        step = jnp.array([0., 0., 0., 1e-5])
        np.testing.assert_allclose(result[..., 3],
            (second(p + step) - second(p - step)) / 2e-5, rtol=3e-7, atol=3e-8)
