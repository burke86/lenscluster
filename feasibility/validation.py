"""Numerical checks written for the user to run; never executed during authoring."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from .common import OUT, write_json, require_spd

def check_selection():
    from .catalogs import selection
    mag=np.array([20.,18.,18.,25.]);ids=np.array(['d','b','a','c'])
    keep,bright,near,lum=selection(mag,ids,np.array([30.,30.,30.,1.]))
    np.testing.assert_array_equal(keep,[False,True,True,True])
    np.testing.assert_allclose(lum[0]/lum[1],10**(-.4*2))
    order=np.array([3,1,0,2]);result=selection(mag[order],ids[order],np.array([30.,30.,30.,1.])[order])[0]
    np.testing.assert_array_equal(result,keep[order])
    from .common import read_json
    audit=read_json(OUT/'catalog_audit.json')
    assert not audit['failures']
    assert 0<audit['selected_luminosity_fraction']<=1

def check_distances():
    from astropy.cosmology import FlatwCDM
    from .distances import flat_wcdm_lensing_efficiency as efficiency
    z=jnp.array([.5,1.,2.,3.221,7.,10.]);p=jnp.array([.3,-1.])
    fn=lambda p:efficiency(.396,z,70.,p[0],p[1])
    cosmology=FlatwCDM(H0=70,Om0=.3,w0=-1.)
    expected=(cosmology.angular_diameter_distance_z1z2(.396,np.array(z))/cosmology.angular_diameter_distance(np.array(z))).value
    np.testing.assert_allclose(fn(p),expected,rtol=2e-8,atol=1e-10)
    jac=np.asarray(jax.jacfwd(fn)(p));fd=np.column_stack([(np.asarray(fn(p+jnp.eye(2)[i]*1e-5))-np.asarray(fn(p-jnp.eye(2)[i]*1e-5)))/2e-5 for i in range(2)])
    np.testing.assert_allclose(jac,fd,rtol=1e-6,atol=1e-8)

def check_likelihood():
    from .model import Model
    from lenscluster.evaluator import Evaluator
    model=Model();production=Evaluator(model.state,'smoothed');u=model.u0
    assert len(model.state.members.far_positions)==0
    from lenscluster.model import member_log_deviations
    dv,dr=member_log_deviations(model.state.packed_lens_spec,model.ev.physical(model.full(u)))
    np.testing.assert_array_equal(dv,np.zeros_like(dv));np.testing.assert_array_equal(dr,np.zeros_like(dr))
    # At zero shear the adapter must reproduce the untouched production target.
    def reference(u):
        s=production.statistics(model.full(u),force_exact=True)
        return -(jnp.sum(s['loglike'])+s['source_correction']+model.prior_u(u))
    a,ga=jax.value_and_grad(model._potential)(u);b,gb=jax.value_and_grad(reference)(u)
    # Shear derivatives differ intentionally because production shear is not
    # efficiency-scaled. Compare the shared parameter derivatives only.
    keep=np.array(['gamma' not in p.field for p in model.specs])
    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(np.asarray(ga)[keep],np.asarray(gb)[keep],rtol=1e-8,atol=1e-7)
    assert np.isfinite(a)
    # Complete transformed density agrees with physical sources plus Jacobian.
    def physical_chart(u):
        report=model._report(u);s=model.statistics(u)
        offsets=model.theta(u)[model.source]
        base_source=-.5*jnp.sum(offsets**2+jnp.log(2*jnp.pi))
        source_prior=-.5*jnp.sum((s['sources']/100.)**2+2*jnp.log(100.)+jnp.log(2*jnp.pi))
        logdet=-.5*jnp.sum(s['whitening']['logdet_precision'])
        return -(model.physical_likelihood(report)+source_prior+logdet+model.prior_u(u)-base_source)
    c,gc=jax.value_and_grad(physical_chart)(u)
    np.testing.assert_allclose(a,c,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(ga,gc,rtol=1e-8,atol=1e-7)
    return dict(likelihood_absolute_difference=float(abs(a-b)),gradient_max_absolute_difference=float(np.max(abs(np.asarray(ga)[keep]-np.asarray(gb)[keep]))))

def check_hessian():
    from .model import Model
    model=Model();u=model.u0;rng=np.random.default_rng(24)
    direction=rng.normal(size=len(u));direction/=np.linalg.norm(direction)
    hv=jax.jvp(jax.grad(model._potential),(u,),(jnp.asarray(direction),))[1]
    h=1e-4;fd=(model.vg(u+h*direction)[1]-model.vg(u-h*direction)[1])/(2*h)
    np.testing.assert_allclose(hv,fd,rtol=2e-4,atol=1e-5)
    try:require_spd(np.diag([1.,-1.]))
    except ValueError:pass
    else:raise AssertionError('Indefinite Hessian was accepted')

def check_compiled_reporting():
    """Exercise actual JIT/vmap reporting, not just eager autodiff tracing."""
    from .model import Model
    errors=[]
    for free in (False,True):
        model=Model(free);u=model.u0
        expected=np.asarray(model._report(u));actual=np.asarray(model.report(u))
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-10)
        physical,source=jax.jit(model.physical_from_report)(jnp.asarray(actual))
        np.testing.assert_allclose(source,model.statistics(u)['sources'],rtol=1e-11,atol=1e-10)
        np.testing.assert_allclose(np.asarray(physical)[model.active[model.global_indices]],
                                   actual[model.global_indices],rtol=1e-12,atol=1e-12)
        pair=jnp.stack([u,u.at[model.source[0]].add(.01)])
        batch=np.asarray(jax.jit(jax.vmap(model._report))(pair))
        np.testing.assert_allclose(batch,np.stack([np.asarray(model._report(x)) for x in pair]),rtol=1e-11,atol=1e-10)
        # Laplace uses forward derivatives; downstream field models use reverse AD.
        jac=np.asarray(jax.jit(jax.jacfwd(model._report))(u))
        gradient=np.asarray(jax.jit(jax.grad(lambda x:jnp.sum(model._report(x))))(u))
        assert np.isfinite(jac).all() and np.isfinite(gradient).all()
        np.testing.assert_allclose(jac.sum(0),gradient,rtol=1e-8,atol=1e-7)
        if free:np.testing.assert_allclose(actual[model.nbase:],model.ratios(u)[model.ratio_indices],rtol=1e-12)
        errors.append(dict(free_ratios=free,max_difference=float(np.max(abs(actual-expected)))))
    return errors

def check_fields():
    from .model import Model
    from lenscluster.model import SHEAR_PROFILE
    model=Model();physical=model.ev.physical(model.full(model.u0))
    # Set nonzero shear to test the otherwise invisible efficiency dependence.
    for i,p in zip(model.active,model.specs):
        if p.field=='gamma1':physical=physical.at[i].set(.03)
    points=jnp.asarray(model.xy[:8]);f1=.3;f2=.7
    one=model.fields(physical,f1,points,derivative=False);two=model.fields(physical,f2,points,derivative=False)
    np.testing.assert_allclose(points-two['backprojected'],(points-one['backprojected'])*f2/f1,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(jnp.eye(2)-two['jacobian'],(jnp.eye(2)-one['jacobian'])*f2/f1,rtol=1e-10,atol=1e-10)
    batch=model.fields(physical,jnp.full(len(points),f1),points,derivative=False)
    np.testing.assert_allclose(batch['backprojected'],one['backprojected'],rtol=1e-12,atol=1e-12)
    derivative=jax.jacfwd(lambda x:model.fields(physical,f1,x[None],derivative=False)['backprojected'][0])
    np.testing.assert_allclose(derivative(points[0]),one['jacobian'][0],rtol=1e-9,atol=1e-9)

def check_multiplicity():
    from .area import base_triangles,raster_roots,unique_roots
    np.testing.assert_equal(len(unique_roots([[0,0],[.001,0],[1,0]])),2)
    # beta=(x^2,y): two images for 0<beta_x<1, -1<beta_y<1.
    triangles=base_triangles((np.array([-1.,-1.]),np.array([1.,1.])),.025)
    beta=triangles.copy();beta[:,:,0]**=2
    roots=raster_roots(triangles,beta,.025,lambda x:np.all(abs(x)<=1,axis=1))
    area=sum(len(v)>=2 for v in roots.values())*.025**2
    np.testing.assert_allclose(area,2.,rtol=.02)
    identity=raster_roots(triangles,triangles,.05,lambda x:np.all(abs(x)<=1,axis=1))
    assert all(len(v)==1 for v in identity.values()),'Shared triangle edges counted as distinct images'

def check_priors_and_ratio_anchor():
    from .model import Model
    model=Model();free=Model(True)
    assert free.ids[free.ref]=='0141'
    assert len(free.ratio_indices)==len(free.ids)-1
    np.testing.assert_allclose(free.ratios(free.u0),free.fiducial_f,rtol=1e-14)
    for i,p in enumerate(model.specs):
        if p.prior_kind!='uniform' or p.transform_kind!='log_positive':continue
        x=np.linspace(float(p.physical_lower)*1.001,float(p.physical_upper)*.999,100)
        logpdf=np.asarray(model.priors[i].log_prob(jnp.log(x)))-np.log(x)
        np.testing.assert_allclose(logpdf,-np.log(float(p.physical_upper)-float(p.physical_lower)),rtol=1e-12)
        theta=np.asarray(model.transforms[i](jnp.array([-100.,100.])))
        assert (theta>=np.log(float(p.physical_lower))-1e-12).all()
        assert (theta<=np.log(float(p.physical_upper))+1e-12).all()

def run_checks():
    from .runtime import progress
    records=[]
    checks=(check_selection,check_distances,check_likelihood,check_compiled_reporting,check_hessian,check_fields,check_multiplicity,check_priors_and_ratio_anchor)
    for function in progress(checks,desc='Numerical checks',unit='check',position=1):
        begin=time.perf_counter()
        try:detail=function();record=dict(check=function.__name__,passed=True,detail=detail)
        except Exception as error:record=dict(check=function.__name__,passed=False,error=repr(error))
        record['seconds']=time.perf_counter()-begin;records.append(record);print(record,flush=True)
        write_json(OUT/'checks.json',records)
    if not all(r['passed'] for r in records):raise RuntimeError('Feasibility numerical checks failed; see checks.json')
    return records
