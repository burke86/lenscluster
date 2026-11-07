"""Written checks for the fast screen; not executed during implementation."""
import numpy as np
from .family_probability import eligible_pairs,candidate_pairs,pair_scores,fit_calibrator,probabilities,metrics

def test_overlap_against_independent_gaussian():
    from scipy.stats import multivariate_normal
    beta=np.array([[[0.,0.],[1.,.4]],[[.1,0.],[.5,.3]]])
    cov=np.broadcast_to(np.array([[.3,.02],[.02,.5]]),(2,2,2,2)).copy()
    score,q,best=pair_scores(beta,cov,np.array([[0,1]]))
    density=[multivariate_normal.pdf(beta[k,0]-beta[k,1],mean=[0,0],cov=cov[k,0]+cov[k,1]) for k in range(2)]
    np.testing.assert_allclose(np.exp(score[0]),np.mean(density),rtol=1e-12)
    assert best[0]==np.argmax(density)

def test_prefilter_keeps_every_small_mahalanobis_pair():
    rng=np.random.default_rng(3);beta=rng.normal(size=(4,25,2));xy=beta[0]
    a=rng.normal(size=(4,25,2,2));cov=a@a.swapaxes(-1,-2)+.05*np.eye(2)
    pairs=eligible_pairs(xy,0.);_,q,_=pair_scores(beta,cov,pairs)
    candidate=set(map(tuple,candidate_pairs(beta,cov,xy,0.,q_limit=1.,max_pairs=1000)))
    assert set(map(tuple,pairs[q<=1.]))<=candidate

def test_probability_monotonicity_and_prior_shift():
    score=np.linspace(-20,10,100);y=score>5
    fitted=fit_calibrator(score,y);p=probabilities(score,fitted)
    assert np.all(np.diff(p)>=0)
    lower=probabilities(score,fitted,pair_prior=.01)
    assert np.all(lower<p)
    np.testing.assert_allclose(probabilities(score,fitted,fitted['reference_pair_prior']),p)

def test_metrics_and_empty_candidates():
    result=metrics(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]))
    assert result['roc_auc']==1.
    row=next(r for r in result['thresholds'] if r['probability']==.5)
    assert row['precision']==row['recall']==1.
    assert eligible_pairs(np.zeros((1,2)),0.).shape==(0,2)

def test_fold_groups_do_not_leak():
    import pandas as pd
    from .screen_families import make_folds
    data=pd.DataFrame({'family_id':['0141','a','a','b','b','c','c','d','d','e','e'],
                       'catalog_z':[3.221,1,1,1,1,2,2,3,3,4,4]})
    folds=make_folds(data,3,12)
    assert folds[0]==-1
    assert len(set(folds[1:5]))==1
    for k in range(3):
        train=data.loc[(folds!=k)&(folds!=(k+1)%3),'family_id']
        test=data.loc[folds==k,'family_id']
        assert not set(train)&set(test)

def test_fast_redshift_scaling_matches_exact_fields():
    # Requires the selected catalog, but does not run an optimizer or NUTS.
    import jax.numpy as jnp
    from .model import Model
    from .family_probability import geometry
    model=Model();physical=model.ev.physical(model.full(model.u0))
    points=model.xy[:8];f=np.array([.2,.5,.8])
    beta,cov,_=geometry(model,physical,points,f)
    for k,eff in enumerate(f):
        exact=model.fields(physical,jnp.asarray(eff),xy=points,derivative=False)
        np.testing.assert_allclose(beta[k],exact['backprojected'],rtol=1e-11,atol=1e-11)
        a=np.asarray(exact['jacobian']);s=float(physical[model.ev.sigma_def_index])
        np.testing.assert_allclose(cov[k],.3**2*(a@a.swapaxes(-1,-2))+s*s*np.eye(2),rtol=1e-11,atol=1e-11)
