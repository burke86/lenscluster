"""Likelihood forecasts and posterior lens fields in a fixed angular lens chart."""
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay
from .distances import flat_wcdm_lensing_efficiency
from lenscluster import smoothed_likelihood as sl
from .common import OUT, compile_call, indices, write_json, read_json
from .model import Model
from .benchmarks import field_function
from .runtime import progress

def spread_families(z,ids,counts,n):
    x=np.log1p(z);order=np.lexsort((ids,-counts,x));chosen=[int(order[0]),int(order[-1])]
    while len(chosen)<min(n,len(z)):
        candidates=[i for i in range(len(z)) if i not in chosen]
        i=min(candidates,key=lambda i:(-float(np.min(abs(x[i]-x[chosen]))),-int(counts[i]),str(ids[i])))
        chosen.append(i)
    return np.array(chosen[:n],int)

def marginal_forecast(matrix,cosmo_indices):
    """SVD in a dimensionless diagonal-scaled chart; retain nullspace evidence."""
    h=(matrix+matrix.T)/2
    if not np.isfinite(h).all():return {'valid':False,'reason':'nonfinite matrix'},np.full_like(h,np.nan)
    scale=np.sqrt(np.maximum(np.abs(np.diag(h)),1e-30));scaled=h/np.outer(scale,scale)
    values,vectors=np.linalg.eigh(scaled);tol=1e-9*max(abs(values).max(),1.)
    if values.min() < -tol:return dict(valid=False,reason='indefinite observed curvature',min_eigenvalue=float(values.min())),np.full_like(h,np.nan)
    null=values<=tol;null_projection=np.sum(vectors[np.asarray(cosmo_indices)][:,null]**2,axis=1)
    invvalues=np.zeros_like(values);invvalues[~null]=1/values[~null]
    cov=((vectors*invvalues)@vectors.T)/np.outer(scale,scale)
    if (null_projection>1e-8).any():
        return dict(valid=False,reason='cosmology projects onto an unconstrained direction',nullity=int(null.sum()),null_projection=null_projection),cov
    sigma=np.sqrt(np.diag(cov)[cosmo_indices])
    return dict(valid=True,sigma_omega_m=float(sigma[0]),sigma_w=float(sigma[1]),nullity=int(null.sum()),
                min_eigenvalue=float(values.min()),condition=float(values[-1]/values[~null][0])),cov

def fisher():
    model=Model();base=np.load(OUT/'baseline_map.npz')['reported'];counts=np.bincount(np.asarray(model.fi))
    ordered=spread_families(model.z,model.ids,counts,20)
    selections={'all':np.arange(len(model.ids)),'10':ordered[:10],'20':ordered}
    entries=[];arrays={};gi=model.global_indices
    full_to_active={int(v):i for i,v in enumerate(model.active)}
    for label,families in progress(selections.items(),desc='Fisher selections',position=1):
        source_indices=[full_to_active[int(i)] for f in families for i in (model.state.images.source_x_param_index[f],model.state.images.source_y_param_index[f])]
        keep=np.r_[gi,source_indices];image_mask=np.isin(np.asarray(model.fi),families);rows=np.flatnonzero(image_mask)
        for free_sheet in progress((False,True),desc='Sheet treatment',position=2,leave=False):
            suffix=f'{label}_'+('free_sheet' if free_sheet else 'fixed_sheet')
            names=[model.names[i] for i in keep]+['Omega_m','w']+(['kappa_infinity'] if free_sheet else [])
            x0=jnp.asarray(np.r_[base[keep],[.3,-1.],([0.] if free_sheet else [])])
            def unpack(x):return jnp.asarray(base).at[keep].set(x[:len(keep)]),x[len(keep):len(keep)+2],x[-1] if free_sheet else 0.
            def nll(x):
                report,cosmo,sheet=unpack(x)
                return -model.physical_likelihood(report,families,cosmo,sheet)
            def local_gaussian(x):
                report,cosmo,sheet=unpack(x);physical,source=model.physical_from_report(report)
                f=flat_wcdm_lensing_efficiency(.396,jnp.asarray(model.z),70.,cosmo[0],cosmo[1])
                fields=model.fields(physical,f[model.fi],derivative=False,sheet=sheet)
                a=fields['jacobian'][rows];residual=fields['backprojected'][rows]-source[model.fi[rows]]
                inv=jnp.linalg.solve(a,jnp.broadcast_to(jnp.eye(2),a.shape))
                mu=jnp.asarray(model.xy)[rows]-jnp.einsum('nij,nj->ni',inv,residual)
                cov=sl.source_covariance(a,.3,physical[model.ev.sigma_def_index])['covariance']
                image_cov=inv@cov@jnp.swapaxes(inv,-1,-2)
                return mu,image_cov
            # Parameter-dependent covariance derivatives are part of the Gaussian Fisher.
            for kind in progress(('gaussian_fisher','observed_curvature'),desc='Curvature',position=3,leave=False):
                try:
                    if kind=='observed_curvature':
                        _,matrix,timing=compile_call(jax.jit(jax.hessian(nll)),x0)
                    else:
                        def calculate(x):
                            mu,cov=local_gaussian(x);dm,dc=jax.jacfwd(local_gaussian)(x)
                            precision=jnp.linalg.inv(cov)
                            mean_term=jnp.einsum('nip,nij,njq->pq',dm,precision,dm)
                            pdc=jnp.einsum('nij,njkp->nikp',precision,dc)
                            return mean_term+.5*jnp.einsum('nijp,njiq->pq',pdc,pdc)
                        _,matrix,timing=compile_call(jax.jit(calculate),x0)
                    matrix=np.asarray(matrix);result,cov=marginal_forecast(matrix,[len(keep),len(keep)+1])
                    key=suffix+'_'+kind;arrays[key]=matrix;arrays[key+'_covariance']=cov;arrays[key+'_names']=np.array(names)
                    arrays[key+'_family_ids']=model.ids[families]
                    if np.isfinite(matrix).all():
                        scale=np.sqrt(np.maximum(abs(np.diag(matrix)),1e-30))
                        eigen,vectors=np.linalg.eigh((matrix+matrix.T)/2/np.outer(scale,scale))
                        arrays[key+'_scaled_eigenvalues']=eigen
                        arrays[key+'_physical_eigenvectors']=vectors/scale[:,None]
                        arrays[key+'_null_mask']=abs(eigen)<=1e-9*max(abs(eigen).max(),1.)
                    if result['valid']:
                        correlations={}
                        for parameter in ('1.v_disp','2.v_disp','members.sigma','members.cutkpc'):
                            if parameter not in names:continue
                            i=names.index(parameter)
                            correlations[parameter]=[float(cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])) for j in (len(keep),len(keep)+1)]
                        result['normalization_cosmology_correlations']=correlations
                    entries.append(dict(selection=label,families=len(families),free_sheet=free_sheet,kind=kind,**timing,**result))
                except Exception as error:
                    entries.append(dict(selection=label,free_sheet=free_sheet,kind=kind,valid=False,reason=repr(error)))
                write_json(OUT/'cosmography/fisher.json',entries)
                np.savez_compressed(OUT/'cosmography/fisher.npz',**arrays)
    return entries

def map_quantities(beta,a):
    u=np.eye(2)-a;kappa=.5*(u[...,0,0]+u[...,1,1]);g1=.5*(u[...,0,0]-u[...,1,1]);g2=.5*(u[...,0,1]+u[...,1,0])
    det=np.linalg.det(a)
    with np.errstate(divide='ignore',invalid='ignore'):mu=1/det
    return np.stack([mu,kappa,g1,g2,np.hypot(g1,g2)],axis=-1),det

def fields_on_points(model,report,points,efficiency,batch=1024):
    fn=getattr(model,'_feas_field_fn',None)
    if fn is None:
        template=np.asarray(points[:batch])
        if len(template)<batch:template=np.pad(template,((0,batch-len(template)),(0,0)),mode='edge')
        start=time.perf_counter()
        fn=field_function(model).lower(jnp.asarray(report),jnp.asarray(template),jnp.asarray(efficiency)).compile()
        model._feas_field_compile_seconds=time.perf_counter()-start
        model._feas_field_fn=fn
    beta=[];jac=[]
    for start in range(0,len(points),batch):
        p=points[start:start+batch];n=len(p)
        if n<batch:p=np.pad(p,((0,batch-n),(0,0)),mode='edge')
        b,a=fn(jnp.asarray(report),jnp.asarray(p),jnp.asarray(efficiency))
        beta.append(np.asarray(b)[:n]);jac.append(np.asarray(a)[:n])
    return np.concatenate(beta),np.concatenate(jac)

def maps():
    model=Model();post=np.load(OUT/'baseline_nuts.npz')['reported'];flat=post.reshape(-1,len(model.names));chosen=indices(len(flat),100)
    best=np.load(OUT/'baseline_map.npz')['reported'];hull=Delaunay(model.xy)
    x=np.arange(model.xy[:,0].min(),model.xy[:,0].max()+.25,.5);y=np.arange(model.xy[:,1].min(),model.xy[:,1].max()+.25,.5)
    xx,yy=np.meshgrid(x,y);grid=np.column_stack([xx.ravel(),yy.ravel()]);mask=hull.find_simplex(grid)>=0
    points=np.vstack([grid[mask],model.xy]);zs=np.array([1.,2.,4.,7.]);results=[]
    for z in zs:
        t=time.perf_counter();before_compile=getattr(model,'_feas_field_compile_seconds',0.)
        f=float(flat_wcdm_lensing_efficiency(.396,z,70.,.3,-1.));values=[];determinants=[]
        for report in progress(np.vstack([best[None],flat[chosen]]),desc=f'Fields z={z:g}',unit='draw',position=1):
            beta,a=fields_on_points(model,report,points,f);v,det=map_quantities(beta,a);values.append(v);determinants.append(det)
        values=np.array(values);determinants=np.array(determinants);samples=values[1:]
        finite=np.isfinite(samples);clean=np.where(finite,samples,np.nan)
        with np.errstate(invalid='ignore'):
            q=np.nanpercentile(clean,[16,50,84],axis=0);std=np.nanstd(clean,axis=0,ddof=1)
        np.savez_compressed(OUT/f'cosmography/fields_z{z:g}.npz',xy_arcsec=points,grid_x=x,grid_y=y,grid_mask=mask,
            n_grid=int(mask.sum()),image_labels=model.state.images.image_labels,redshift=z,draw_indices=chosen,
            quantity_names=np.array(['mu_signed','kappa','gamma1','gamma2','gamma_abs']),best=values[0],draws=samples,
            p16=q[0],median=q[1],p84=q[2],std=std,finite_count=finite.sum(0),determinant=determinants,
            critical_flag=(np.min(determinants,axis=0)<=0)&(np.max(determinants,axis=0)>=0),
            convergence_valid=read_json(OUT/'baseline_nuts.json',{}).get('converged',False))
        compile_seconds=getattr(model,'_feas_field_compile_seconds',0.)-before_compile
        results.append(dict(z=float(z),points=len(points),draws=len(chosen),seconds=time.perf_counter()-t,
                            field_compile_seconds=compile_seconds,execution_and_io_seconds=time.perf_counter()-t-compile_seconds))
    write_json(OUT/'cosmography/maps.json',results);return results
