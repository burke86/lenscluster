"""Conditional image-GP displacement diagnostics from saved posterior draws.

These are additive displacements under the fitted linearized likelihood. The
base gravitational positions use the conditional-mean GRF. No GRF realization,
new measurement noise, refitting, or NUTS sampling is added here.
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_solve, solve_triangular

VERSION = 'image-gp-conditional-position-v1'


def conditional_displacement(query, observed, jacobian, cholesky, residual, sigma, ell):
    """E[d(q)|F] and marginal 2x2 Cov[d(q)|F], using C from the full likelihood.

    F = GRF + A d + white terms. Cov[d(q),F_j] = K(q,j) A_j^T.
    The GP is evaluated at the gravitationally predicted image positions.
    """
    query, observed = np.asarray(query), np.asarray(observed)
    kernel = sigma**2*np.exp(-np.sum((query[:,None]-observed[None])**2,axis=-1)/(2*ell**2))
    cross = np.einsum('qn,nca->qanc',kernel,jacobian).reshape(len(query),2,2*len(observed))
    v = cho_solve((cholesky,True),np.asarray(residual).reshape(-1))
    mean = np.einsum('qan,n->qa',cross,v)
    solved = solve_triangular(cholesky,cross.reshape(-1,2*len(observed)).T,lower=True)
    solved = solved.reshape(2*len(observed),len(query),2)
    covariance = sigma**2*np.eye(2)[None]-np.einsum('nqa,nqb->qab',solved,solved)
    covariance = .5*(covariance+covariance.swapaxes(-1,-2))
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues,initial=0.) < -max(1e-12,1e-8*sigma**2):
        raise ValueError('Conditional GP covariance is not positive semidefinite')
    covariance = np.einsum('qai,qi,qbi->qab',vectors,np.maximum(eigenvalues,0.),vectors)
    return mean,covariance


def add_gp_position_diagnostics(state,evaluator,posterior,data,*,log=print):
    """Cache moments at every recovered image, paired with its original draw."""
    if not (getattr(evaluator,'gp_knee',False) and state.gp_blocks=='all'):
        return data
    systems=data.get('systems',[])
    if not systems:
        return data
    indices=np.asarray(data['draw_indices'],int)
    samples=np.asarray(posterior.samples)[indices]
    digest=hashlib.sha256(samples.tobytes()+indices.tobytes()+VERSION.encode())
    for system in systems:
        digest.update(np.asarray(system['positions']).tobytes())
        digest.update('|'.join(system['labels']).encode())
    fingerprint=digest.hexdigest()
    if all(s.get('gp_position_fingerprint')==fingerprint for s in systems):
        return data
    e=evaluator._padded_evaluator()
    observed=np.column_stack([state.images.x_obs,state.images.y_obs])
    layout=e.gp
    np.testing.assert_array_equal(np.asarray(layout['slots'])[0],np.arange(len(observed)))
    if len(np.asarray(layout['slots']))!=1:
        raise ValueError('Shared GP position figures require one full covariance block')
    labels=list(map(str,state.images.image_labels))
    @jax.jit
    def inputs(theta):
        stats=e.statistics(theta)
        scales=e.error_scales(stats['physical'])
        jac=stats['fields']['jacobian']
        residual=(stats['fields']['backprojected']
                  -stats['source_by_family'][jnp.asarray(state.images.family_index_per_image)])
        return jac,stats['family_cholesky'][0],residual,scales['sigma_shared'],scales['gp_ell']
    for system in systems:
        shape=system['positions'].shape
        system['gp_displacement_mean']=np.full(shape,np.nan)
        system['gp_displacement_covariance']=np.full(shape+(2,),np.nan)
        system['gp_adjusted_mean_positions']=np.full(shape,np.nan)
    for draw,theta in enumerate(samples):
        jac,chol,residual,sigma,ell=map(np.asarray,inputs(jnp.asarray(theta)))
        # One solve across all displayed images, preserving each draw's geometry.
        queries=np.concatenate([s['positions'][draw] for s in systems])
        valid=np.isfinite(queries).all(1)
        means=np.full_like(queries,np.nan)
        covariance=np.full((len(queries),2,2),np.nan)
        if valid.any():
            means[valid],covariance[valid]=conditional_displacement(queries[valid],observed,jac,chol,residual,float(sigma),float(ell))
        start=0
        for system in systems:
            n=len(system['labels']);sl=slice(start,start+n)
            system['gp_displacement_mean'][draw]=means[sl]
            system['gp_displacement_covariance'][draw]=covariance[sl]
            system['gp_adjusted_mean_positions'][draw]=system['positions'][draw]+means[sl]
            start+=n
        if draw%16==0 or draw==len(samples)-1:
            log(f'[outputs] conditional GP position moments: {draw+1}/{len(samples)} saved draws')
    for system in systems:
        system['gp_position_fingerprint']=fingerprint
        rows=[labels.index(str(label)) for label in system['labels']]
        np.testing.assert_allclose(system['observed'],observed[rows],rtol=0,atol=1e-10)
    data['gp_position_diagnostics'] = dict(version=VERSION,
        mean='Gravitational image position plus E[d(query)|all observed images, theta], evaluated at that predicted position',
        uncertainty='Equal-weight Gaussian mixture over retained posterior draws, including each draw conditional marginal GP covariance',
        conditioning='Full fitted covariance including GRF, image GP, white image errors, and source-plane floor',
        limitations='GRF held at its conditional mean; no residual GRF realizations or new measurement noise. Additive displacement uses the fitted linearized likelihood.',
        arrows='Best-draw conditional-mean GP shifts at true angular scale; not a physical deflection field',
        contours='GP-adjusted mixture regions: 68.2689492% and 95.4499736%; no KDE bandwidth')
    return data


def gaussian_mixture_grid(means,covariances,*,grid_size=192):
    """Marginal positional density, including conditional latent GP uncertainty."""
    means=np.asarray(means,float);covariances=np.asarray(covariances,float)
    valid=np.isfinite(means).all(1)&np.isfinite(covariances).all((1,2))
    means,covariances=means[valid],covariances[valid]
    if len(means)<2:
        return None
    center=means.mean(0)
    total=np.cov(means.T,bias=True)+covariances.mean(0)
    if np.linalg.eigvalsh(total).min()<=0:
        return None
    chol=np.linalg.cholesky(total)
    inverse=np.linalg.inv(chol)
    white=(means-center)@inverse.T
    cov=np.einsum('ab,nbc,dc->nad',inverse,covariances,inverse)
    if np.linalg.eigvalsh(cov).min()<=0:
        raise ValueError('GP mixture requires positive conditional covariance')
    precision=np.linalg.inv(cov)
    normalization=1/(2*np.pi*np.sqrt(np.linalg.det(cov)))
    std=np.sqrt(np.diagonal(cov,axis1=1,axis2=2))
    lo=(white-6*std).min(0);hi=(white+6*std).max(0)
    for resolution in (grid_size,max(384,grid_size),max(512,grid_size)):
        x,y=np.linspace(lo[0],hi[0],resolution),np.linspace(lo[1],hi[1],resolution)
        xx,yy=np.meshgrid(x,y)
        points=np.column_stack([xx.ravel(),yy.ravel()]);density=np.empty(len(points))
        for start in range(0,len(points),2048):
            delta=points[start:start+2048,None]-white[None]
            exponent=np.einsum('pna,nab,pnb->pn',delta,precision,delta)
            density[start:start+len(delta)]=np.mean(np.exp(-.5*exponent)*normalization[None],axis=1)
        area=(x[1]-x[0])*(y[1]-y[0])
        ranked=np.sort(density)[::-1];mass=np.cumsum(ranked)*area
        if abs(mass[-1]-1)<.002:
            break
    else:
        raise ValueError('Conditional GP mixture density is unresolved on the display grid')
    levels=[ranked[min(np.searchsorted(mass,p),len(ranked)-1)] for p in (.954499736,.682689492)]
    physical=np.stack([xx,yy],axis=-1)@chol.T+center
    return dict(x=physical[...,0],y=physical[...,1],density=density.reshape(xx.shape),levels=levels,
                cell_area=area,integrated_mass=float(mass[-1]),components=len(means))
