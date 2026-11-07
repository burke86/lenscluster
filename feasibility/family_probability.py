"""Fast pair geometry and empirical calibration. No inference on import."""
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata

def geometry(model,physical,xy,efficiencies,sigma_obs=.3):
    """Evaluate exact profiles once; all source redshifts use linear f scaling."""
    import jax
    import jax.numpy as jnp
    from .common import compile_call
    def fields(p,points):
        result=model.fields(p,jnp.asarray(1.),xy=points,derivative=False)
        return points-result['backprojected'],jnp.eye(2)-result['jacobian']
    _,(alpha,u),timing=compile_call(jax.jit(fields),jnp.asarray(physical),jnp.asarray(xy))
    f=np.asarray(efficiencies)[:,None,None]
    beta=np.asarray(xy)[None]-f*np.asarray(alpha)[None]
    a=np.eye(2)-f[...,None]*np.asarray(u)[None]
    scatter=float(physical[model.ev.sigma_def_index])
    cov=sigma_obs**2*(a@a.swapaxes(-1,-2))+scatter**2*np.eye(2)
    if not np.isfinite(beta).all() or not np.isfinite(cov).all():raise ValueError('Nonfinite field or covariance')
    if np.any(np.linalg.eigvalsh(cov)<=0):raise ValueError('Covariance is not positive definite')
    return beta,cov,timing

def eligible_pairs(xy,min_separation):
    i,j=np.triu_indices(len(xy),1)
    keep=np.linalg.norm(xy[i]-xy[j],axis=1)>=min_separation
    return np.column_stack([i[keep],j[keep]])

def candidate_pairs(beta,cov,xy,min_separation,q_limit=36.,max_pairs=500000):
    """Conservative KD-tree prefilter; never allocate an N-by-N matrix.

    All pairs with Q <= q_limit at a grid point lie inside the search radius.
    The full covariance score is subsequently evaluated for retained pairs.
    """
    from .runtime import progress
    found=set()
    for b,c in progress(zip(beta,cov),total=len(beta),desc='Candidate search',unit='redshift',position=1,leave=False):
        radius=np.sqrt(2*q_limit*np.linalg.eigvalsh(c)[:,-1].max())
        tree=cKDTree(b)
        for i in range(len(b)):
            for j in tree.query_ball_point(b[i],radius):
                if j<=i or np.linalg.norm(xy[i]-xy[j])<min_separation:continue
                found.add((i,j))
                if len(found)>max_pairs:
                    raise RuntimeError(f'More than {max_pairs} candidate pairs: split the catalog spatially or raise --max-pairs. No probabilities were silently truncated.')
    return np.asarray(sorted(found),dtype=int).reshape(-1,2)

def pair_scores(beta,cov,pairs,block=4096):
    """Gaussian source overlap averaged over a fixed uniform-efficiency grid.

    This is a feature for calibration, not itself a membership probability.
    It includes the covariance normalization, not only the Mahalanobis distance.
    """
    from .runtime import progress
    score=np.empty(len(pairs));qmin=np.empty(len(pairs));best=np.empty(len(pairs),int)
    for start in progress(range(0,len(pairs),block),desc='Pair probabilities',unit='block',position=1,leave=False):
        i,j=pairs[start:start+block].T
        delta=beta[:,i]-beta[:,j];c=cov[:,i]+cov[:,j]
        a,b,d=c[...,0,0],c[...,0,1],c[...,1,1];det=a*d-b*b
        dx,dy=delta[...,0],delta[...,1]
        if np.any(det<=0):raise ValueError('Invalid pair covariance')
        q=(d*dx*dx-2*b*dx*dy+a*dy*dy)/det
        log_overlap=-.5*q-.5*np.log(det)-np.log(2*np.pi)
        end=start+len(i)
        score[start:end]=logsumexp(log_overlap,axis=0)-np.log(len(beta))
        qmin[start:end]=q.min(0);best[start:end]=np.argmax(log_overlap,axis=0)
    return score,qmin,best

def fit_calibrator(score,y):
    """Monotonic, weakly regularized logistic calibration, no class balancing."""
    score=np.asarray(score);y=np.asarray(y,float)
    if not np.isfinite(score).all():raise ValueError('Nonfinite calibration scores')
    if y.sum()<5 or (1-y).sum()<20:raise ValueError('Calibration needs at least 5 positive and 20 negative pairs')
    # Signed log compresses arbitrarily poor geometric matches without clipping
    # away their ordering. Positive slope makes better overlaps more probable.
    transform=lambda s:np.sign(s)*np.log1p(abs(s))
    center=float(np.median(transform(score)));scale=float(max(np.std(transform(score)),1e-6))
    x=(transform(score)-center)/scale
    def loss(v):
        eta=v[0]+v[1]*x;delta=expit(eta)-y
        return float(np.sum(np.logaddexp(0,eta)-y*eta)+.5*v[1]**2),np.array([delta.sum(),np.dot(delta,x)+v[1]])
    prevalence=float(y.mean());v0=[np.log(prevalence/(1-prevalence)),1.]
    result=minimize(loss,v0,jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None)])
    if not result.success or not np.isfinite(result.fun):raise RuntimeError(f'Calibration failed: {result.message}')
    return dict(intercept=float(result.x[0]),slope=float(result.x[1]),center=center,scale=scale,
                reference_pair_prior=prevalence,pairs=len(y),positives=int(y.sum()))

def probabilities(score,calibration,pair_prior=None):
    transformed=np.sign(score)*np.log1p(abs(score))
    logodds=calibration['intercept']+calibration['slope']*(transformed-calibration['center'])/calibration['scale']
    if pair_prior is not None:
        if not 0<pair_prior<1:raise ValueError('--pair-prior must be strictly between zero and one')
        prior=calibration['reference_pair_prior']
        logodds+=np.log(pair_prior/(1-pair_prior))-np.log(prior/(1-prior))
    return expit(logodds)

def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p);positive=int(y.sum());negative=int((~y).sum())
    clipped=np.clip(p,1e-12,1-1e-12);order=np.argsort(-p,kind='stable')
    precision=np.cumsum(y[order])/np.arange(1,len(y)+1)
    thresholds=[]
    for cutoff in (.1,.25,.5,.8,.9,.95):
        chosen=p>=cutoff;tp=int((chosen&y).sum());fp=int((chosen&~y).sum())
        thresholds.append(dict(probability=cutoff,proposals=int(chosen.sum()),true_positive=tp,false_positive=fp,
                               precision=tp/(tp+fp) if tp+fp else None,recall=tp/positive if positive else None))
    bins=[]
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(p>=low)&(p<high if high<1 else p<=high)
        if mask.any():bins.append(dict(low=float(low),high=float(high),count=int(mask.sum()),
            mean_probability=float(p[mask].mean()),observed_fraction=float(y[mask].mean())))
    auc=(rankdata(p)[y].sum()-positive*(positive+1)/2)/(positive*negative) if positive and negative else None
    return dict(pairs=len(y),positives=positive,prevalence=float(y.mean()) if len(y) else None,
                brier=float(np.mean((p-y)**2)) if len(y) else None,
                log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))) if len(y) else None,
                average_precision=float(precision[y[order]].mean()) if positive else None,
                roc_auc=float(auc) if auc is not None else None,thresholds=thresholds,reliability=bins)
