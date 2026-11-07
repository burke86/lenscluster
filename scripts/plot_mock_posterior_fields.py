"""Compute posterior field arrays and render aggregate GRF/GP image recovery.

Integrates Gaussian conditional means/variances over deterministic joint saved
posterior draws. No new lens inference. Maps show projected GRF deflection at z_ref.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.linalg import cho_solve, block_diag
from lenscluster import artifacts, gp_knee
from lenscluster.mock_validation.field import field_arrays, field_alpha_hessian_tiled
from make_shared_grf_map import _evaluator


def kernel(points, images, radii, cl, ct, amplitude):
    d=points[:,None]-images[None]; r=np.linalg.norm(d,axis=-1)
    u=d/np.maximum(r[...,None],1e-30)
    longitudinal=np.interp(r,radii,cl); transverse=np.interp(r,radii,ct)
    return (amplitude*(transverse[...,None,None]*np.eye(2)+(longitudinal-transverse)[...,None,None]*u[..., :,None]*u[...,None,:])).transpose(0,2,1,3).reshape(2*len(points),2*len(images))


def mixture(means,variances):
    means=np.asarray(means); variances=np.asarray(variances)
    return means.mean(0),variances.mean(0),means.var(0)


def run(seed,output,draw_count,step):
    print(f'{seed.name}: rebuilding saved evaluator',flush=True)
    final=seed/'solver/final'; init=seed/'solver/init/artifacts'
    config=artifacts.load_config(final/'artifacts')
    if config.likelihood.gp_blocks!='all': raise ValueError('Requires a single shared block')
    classes=json.loads((init/artifacts.CLASSIFICATION_FILE).read_text())
    evaluator=_evaluator(config,'all',None,classes,init,artifacts.load_start(init))
    posterior=artifacts.load_posterior(final/'artifacts'); e=evaluator._padded_evaluator();state=evaluator.state
    truth=json.loads((seed/'mock/truth.json').read_text())
    if not np.isclose(truth['field']['reference_redshift'],state.gp_reference_redshift): raise ValueError('Reference redshifts differ')
    ids=np.unique(np.linspace(0,len(posterior.samples)-1,min(draw_count,len(posterior.samples))).astype(int))
    xy=np.c_[state.images.x_obs,state.images.y_obs];n=len(xy);size=np.asarray(e.gp['mask']).shape[1]
    np.testing.assert_array_equal(np.asarray(e.gp['slots'])[0,:n],np.arange(n))
    w=np.repeat(np.asarray(e.gp['weights'])[0,:n],2)
    modes=config.likelihood.gp_knee_project_modes
    if modes:
        q,r,centre=map(np.asarray,gp_knee.low_order_projector(e.gp['xy'],e.gp['mask'],modes))
        q=q[0,:2*n]; transform=np.linalg.solve(r[0],q.T); centre=centre[0]
        def basis(points):
            return np.asarray(gp_knee.low_order_patterns(jnp.asarray(points),jnp.asarray(centre),modes)).reshape(2*len(points),-1)@transform
    else:
        def basis(points): return np.zeros((2*len(points),2*n))
    projection=np.eye(2*n)-basis(xy)
    x=np.arange(xy[:,0].min()-3,xy[:,0].max()+3,step);y=np.arange(xy[:,1].min()-3,xy[:,1].max()+3,step)
    xx,yy=np.meshgrid(x,y);grid=np.c_[xx.ravel(),yy.ravel()]
    true_fields=field_alpha_hessian_tiled(xy[:,0],xy[:,1],field_arrays(truth['field']),1.)
    raw_images=np.c_[true_fields[0],true_fields[1]].ravel()
    fields=field_alpha_hessian_tiled(grid[:,0],grid[:,1],field_arrays(truth['field']),1.)
    raw_grid=np.c_[fields[0],fields[1]]
    true_grid=np.empty_like(raw_grid)
    for start in range(0,len(grid),512):
        end=min(start+512,len(grid));true_grid[start:end]=raw_grid[start:end]-(basis(grid[start:end])@raw_images).reshape(-1,2)
    by_label={r['image_label']:r for r in truth['images']}
    true_gp=np.array([[by_label[str(label)]['x_image_plane_error_arcsec'],by_label[str(label)]['y_image_plane_error_arcsec']] for label in state.images.image_labels])
    target=np.r_[projection@raw_images,true_gp.ravel()]
    family=np.asarray(state.images.family_index_per_image)
    @jax.jit
    def statistics(theta):
        st=e.statistics(theta);scales=e.error_scales(st['physical']);blocks=e.gp_kernel_blocks(st['physical'],scales)
        cl,ct=gp_knee.knee_curves(e.gp_tables,scales['gp_knee'],scales['gp_knee_gamma'])
        return (st['family_cholesky'],st['fields']['backprojected']-st['source_by_family'][family],st['fields']['jacobian'],blocks['k_alpha'],blocks['k_img'],scales['gp_sigma_alpha'],cl,ct)
    grid_means=[];grid_variances=[];image_means=[];image_variances=[];mixture_draws=[]
    rng=np.random.default_rng(20261106)
    radii=np.asarray(e.gp_tables['r_grid'])
    median_physical=np.median(posterior.physical,axis=0)
    from lenscluster.model import convert_theta_to_latent
    median_theta=convert_theta_to_latent(median_physical,state.parameter_specs)
    ix=np.asarray(state.images.source_x_param_index);iy=np.asarray(state.images.source_y_param_index)
    median_theta[ix]=0.;median_theta[iy]=0.
    whitening=e.statistics(jnp.asarray(median_theta))['whitening']
    families=np.asarray(e.gp['knot_family'])[0];valid=np.asarray(e.gp['knot_mask'])[0]
    physical_sources=np.c_[median_physical[ix],median_physical[iy]]
    desired=np.asarray(whitening['centroid'])[0].copy().reshape(-1,2)
    desired[valid]=physical_sources[families[valid]]
    offsets=np.linalg.solve(np.asarray(whitening['cholesky'])[0],desired.ravel()-np.asarray(whitening['centroid'])[0]).reshape(-1,2)
    median_theta[ix[families[valid]]]=offsets[valid,0]
    median_theta[iy[families[valid]]]=offsets[valid,1]
    np.testing.assert_allclose(e.reported_physical(median_theta),median_physical,rtol=1e-8,atol=1e-8)
    parameters=[posterior.samples[index] for index in ids]+[median_theta]
    for count,theta in enumerate(parameters):
        L,residual,jac,K,S,sigma,cl,ct=map(np.asarray,statistics(jnp.asarray(theta)))
        L=L[0,:2*n,:2*n];residual=residual.ravel();A=block_diag(*jac)
        K=K[0].transpose(0,2,1,3).reshape(2*size,2*size)[:2*n,:2*n];S=np.kron(S[0,:n,:n],np.eye(2))
        cross=np.r_[K,S@A.T];mean=cross@cho_solve((L,True),residual)
        cov=block_diag(K,S)-cross@cho_solve((L,True),cross.T);cov=(cov+cov.T)/2
        convert=np.r_[1/w,np.ones(2*n)];mean*=convert;cov*=convert[:,None]*convert[None,:]
        ev,U=np.linalg.eigh(cov)
        if ev.min() < -1e-9*max(abs(ev).max(),1e-15): raise ValueError('Conditional covariance not PSD')
        samples=mean+rng.standard_normal((32,4*n))@(U*np.sqrt(np.maximum(ev,0))).T
        image_means.append(mean);image_variances.append(np.diag(cov));mixture_draws.append(samples)
        Kii=kernel(xy,xy,radii,cl,ct,float(sigma)**2)
        np.testing.assert_allclose((projection@Kii@projection.T)*w[:,None]*w[None,:],K,rtol=1e-7,atol=1e-10)
        gm=np.empty((len(grid),2));gv=np.empty_like(gm)
        for start in range(0,len(grid),512):
            end=min(start+512,len(grid));points=grid[start:end];B=basis(points)
            raw=kernel(points,xy,radii,cl,ct,float(sigma)**2)
            cross_grid=((raw-B@Kii)@projection.T)*w[None,:]
            gm[start:end]=(cross_grid@cho_solve((L,True),residual)).reshape(-1,2)
            prior=float(sigma)**2-2*np.sum(raw*B,axis=1)+np.sum((B@Kii)*B,axis=1)
            conditional=prior-np.sum(cross_grid*cho_solve((L,True),cross_grid.T).T,axis=1)
            if conditional.min() < -1e-8*float(sigma)**2: raise ValueError('Negative grid variance')
            gv[start:end]=np.maximum(conditional,0).reshape(-1,2)
        grid_means.append(gm);grid_variances.append(gv)
        print(f'{seed.name}: field evaluation {count+1}/{len(parameters)}',flush=True)
    median_grid=grid_means.pop();grid_variances.pop()
    median_images=image_means.pop();image_variances.pop();mixture_draws.pop()
    im,iw,ib=mixture(image_means,image_variances);gm,gw,gb=mixture(grid_means,grid_variances)
    samples=np.concatenate(mixture_draws);lo,med,hi=np.quantile(samples,[.16,.5,.84],axis=0)
    stem=output/f'{seed.name}_posterior_fields'
    np.savez_compressed(stem.with_suffix('.npz'),x=x,y=y,xy=xy,indices=ids,truth_grid=true_grid,mean_grid=gm,
                        within_variance_grid=gw,between_variance_grid=gb,truth_images=target,mean_images=im,
                        median_parameter_grid=median_grid,median_parameter_images=median_images,median_physical=median_physical,
                        within_variance_images=iw,between_variance_images=ib,conditional_means=image_means,
                        field_draws=samples,q16=lo,median=med,q84=hi)
    info=dict(posterior_draws=len(ids),conditional_draws_per_posterior=32,reference_redshift=float(state.gp_reference_redshift),
              projection=list(modes),grid_step_arcsec=step,variance='Mean conditional variance + variance of conditional means',
              maps='Projected GRF deflection at componentwise physical posterior-median parameters; sources re-whitened',
              gp_truth='Saved injected displacements at true images; model kernel uses observed positions',
              limitations='Posterior subsample; underlying chains fail convergence cut. No continuous injected GP saved.')
    for name,sl in [('grf',slice(0,2*n)),('gp',slice(2*n,4*n))]:
        info[name]=dict(fraction_inside_68=float(np.mean((target[sl]>=lo[sl])&(target[sl]<=hi[sl]))),
                       rms_median_parameter_residual=float(np.sqrt(np.mean((median_images[sl]-target[sl])**2))))
    stem.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print(seed.name,info,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--draws',type=int,default=32);p.add_argument('--step',type=float,default=.5)
    a=p.parse_args()
    if a.draws<2 or a.step<=0:p.error('Require draws >=2 and positive step')
    out=a.campaign/'aggregate_plots';out.mkdir(exist_ok=True)
    for path in sorted(a.campaign.glob('seed_*/recovery/recovery.json')):run(path.parents[1],out,a.draws,a.step)

    from lenscluster.mock_validation.aggregate_plots import load_campaign
    from lenscluster.mock_validation.field_recovery_plots import plot_field_recovery
    try: campaign=load_campaign(a.campaign)
    except ValueError: campaign=load_campaign(a.campaign,converged_only=False)
    print(plot_field_recovery(campaign,out))
