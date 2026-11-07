"""Optimization, unregularized Laplace approximation, and short NUTS runs."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import pandas as pd
from .common import OUT, SETTINGS, write_json, write_npz, compile_call, timed_call, require_spd, correlation, indices
from .model import Model
from .runtime import progress,worker_count

def prefix(free): return 'ratios' if free else 'baseline'

def batch_report(model, draws, batch=32):
    fn=jax.jit(jax.vmap(model._report));result=[]
    for i in progress(range(0,len(draws),batch),desc='Physical posterior',unit='batch',position=1,leave=False):
        result.append(np.asarray(fn(jnp.asarray(draws[i:i+batch]))))
    return np.concatenate(result)

def optimize(free=False):
    model=Model(free);name=prefix(free)
    if not free:model.write_configuration()
    start=np.array(model.u0,copy=True)
    if free:
        baseline=np.load(OUT/'baseline_map.npz')['u']
        start[:model.nbase]=baseline[:model.nbase]
    executable,_,timing=compile_call(model.vg,jnp.asarray(start))
    starts=[start];rng=np.random.default_rng(SETTINGS['seed']+(100 if free else 0))
    for k in range(1,SETTINGS['starts']):
        # Prior draws for globals only; source offsets zero and scatter at initial median.
        for attempt in range(100):
            trial=start.copy()
            for i in model.global_indices:
                if model.specs[i].component_family=='error_scale':
                    trial[i]=float(model.u0[i]);continue
                value=model.priors[i].sample(jax.random.PRNGKey(int(rng.integers(2**30))))
                trial[i]=float(model.transforms[i].inv(value))
            trial[model.source]=0.
            if np.isfinite(float(model.potential(trial))):break
        else:raise RuntimeError('No valid prior start after 100 deterministic attempts')
        starts.append(trial)
    rows=[];solutions=[];traces=[];beg=time.perf_counter()
    def fit_start(k,start):
        trace=[];solution=None
        def fun(u):
            value,grad=executable(jnp.asarray(u));value=float(value);grad=np.asarray(grad)
            trace.append(value)
            return value,grad
        t=time.perf_counter();bar=progress(total=SETTINGS['maxiter'],desc=f'{name} MAP {k+1}',unit='iter',position=k+1,leave=False)
        def callback(x):
            bar.update();bar.set_postfix(potential=f'{trace[-1]:.3g}',refresh=False)
        try:
            result=minimize(fun,start,jac=True,method='L-BFGS-B',callback=callback,options={'maxiter':SETTINGS['maxiter'],'ftol':1e-12,'gtol':1e-5,'maxls':30})
            row=dict(start=k,success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                     potential=float(result.fun),gradient_max=float(np.max(np.abs(result.jac))),seconds=time.perf_counter()-t)
            if np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all():
                solution=(float(result.fun),result.x)
        except Exception as error:row=dict(start=k,success=False,message=repr(error),seconds=time.perf_counter()-t)
        finally:bar.close()
        return k,row,trace,solution
    # Independent starts share one compiled JAX executable. No process copies
    # or concurrent checkpoint writers; numerical seeds were generated above.
    from concurrent.futures import ThreadPoolExecutor,as_completed
    finished={}
    with ThreadPoolExecutor(max_workers=min(len(starts),worker_count())) as pool:
        futures=[pool.submit(fit_start,k,s) for k,s in enumerate(starts)]
        for future in as_completed(futures):
            k,row,trace,solution=future.result();finished[k]=(row,trace,solution)
            if solution is not None:
                write_npz(OUT/f'{name}_start_{k}_endpoint.npz',u=solution[1],potential=solution[0],
                          start=k,iterations=row['iterations'],gradient_max=row['gradient_max'],names=np.array(model.names))
            write_json(OUT/f'{name}_optimization.json',dict(starts=[finished[i][0] for i in sorted(finished)],**timing))
    for k in sorted(finished):
        row,trace,solution=finished[k];rows.append(row);traces.append(trace)
        if solution is not None:solutions.append(solution)
    if not solutions:raise RuntimeError('No finite optimization endpoint')
    best_start=min((k for k,v in finished.items() if v[2] is not None),key=lambda k:finished[k][2][0])
    value,u=finished[best_start][2]
    # Save the best unconstrained endpoint before polishing or physical reporting.
    endpoint=OUT/f'{name}_best_endpoint.npz'
    write_npz(endpoint,u=u,potential=value,start=best_start,names=np.array(model.names))
    gradient_max=float(np.max(np.abs(np.asarray(executable(jnp.asarray(u))[1]))))
    remaining=max(0,SETTINGS['maxiter']-finished[best_start][0]['iterations'])
    polish=dict(attempted=False,remaining_iterations=remaining,gradient_before=gradient_max)
    if gradient_max>SETTINGS['stationarity_tolerance'] and remaining:
        begin_polish=time.perf_counter()
        with progress(total=remaining,desc=f'{name} polishing',unit='iter',position=1) as bar:
            def objective(x):
                v,g=executable(jnp.asarray(x));return float(v),np.asarray(g)
            try:
                polished=minimize(objective,u,jac=True,method='L-BFGS-B',callback=lambda x:bar.update(),
                    options={'maxiter':remaining,'ftol':0.,'gtol':SETTINGS['polishing_gtol'],'maxls':50,'maxcor':30})
                new_value,new_gradient=objective(polished.x)
                new_norm=float(np.max(abs(new_gradient)))
                finite=np.isfinite(new_value) and np.isfinite(new_gradient).all() and np.isfinite(polished.x).all()
                accepted=bool(finite and (new_value<value or
                    (new_value<=value+1e-12*max(1.,abs(value)) and new_norm<gradient_max)))
                write_npz(OUT/f'{name}_polishing_endpoint.npz',u=polished.x,potential=new_value,
                          gradient=new_gradient,accepted=accepted,iterations=polished.nit)
                polish.update(attempted=True,accepted=accepted,success=bool(polished.success),message=str(polished.message),
                              iterations=int(polished.nit),gradient_after=new_norm,seconds=time.perf_counter()-begin_polish)
                if accepted:value,u,gradient_max=new_value,np.array(polished.x),new_norm
            except Exception as error:polish.update(attempted=True,error=repr(error),seconds=time.perf_counter()-begin_polish)
    write_npz(endpoint,u=u,potential=value,start=best_start,gradient_max=gradient_max,names=np.array(model.names))
    result=dict(starts=rows,traces=traces,execution_seconds=time.perf_counter()-beg,**{'objective_compile_seconds':timing['compile_seconds']},
                best_potential=value,best_gradient_max=gradient_max,polishing=polish,
                stationary=gradient_max<=SETTINGS['stationarity_tolerance'],reporting_complete=False)
    write_json(OUT/f'{name}_optimization.json',result)
    # A reporting exception must not destroy the raw fitted parameters above.
    reported=np.asarray(model.report(u))
    write_npz(OUT/f'{name}_map.npz',u=u,reported=reported,names=np.array(model.names),potential=value,
              sources=np.asarray(model.statistics(u)['sources']),f=np.asarray(model.ratios(u)))
    result['reporting_complete']=True
    write_json(OUT/f'{name}_optimization.json',result)
    return result

def laplace():
    model=Model();u=np.load(OUT/'baseline_map.npz')['u']
    _,h,timing=compile_call(jax.jit(jax.hessian(model._potential)),jnp.asarray(u));h=np.asarray(h)
    np.savez_compressed(OUT/'baseline_hessian.npz',hessian=h,eigenvalues=np.linalg.eigvalsh((h+h.T)/2))
    try:
        gradient_max=float(np.max(np.abs(np.asarray(model.vg(u)[1]))))
        if not np.isfinite(gradient_max) or gradient_max>SETTINGS['stationarity_tolerance']:
            raise ValueError(f'MAP not stationary enough for Laplace: maximum gradient {gradient_max:g}')
        cov,eigen=require_spd(h)
    except ValueError as error:
        write_json(OUT/'laplace.json',dict(valid=False,reason=str(error),**timing));raise
    rng=np.random.default_rng(SETTINGS['seed']);beg=time.perf_counter()
    draws=rng.multivariate_normal(u,cov,SETTINGS['laplace_draws'])
    # Unconstrained draws can still violate coupled physical shape/shear guards.
    values=[]
    pot=jax.jit(jax.vmap(model._potential))
    for i in progress(range(0,len(draws),32),desc='Laplace validity',unit='batch',position=1):values.extend(np.asarray(pot(draws[i:i+32])))
    valid=np.isfinite(values);reported=np.full((len(draws),len(model.names)),np.nan)
    if valid.any():reported[valid]=batch_report(model,draws[valid])
    jac=np.asarray(jax.jacfwd(model._report)(jnp.asarray(u)));physical_cov=jac@cov@jac.T
    np.savez_compressed(OUT/'laplace.npz',u=draws,reported=reported,valid=valid,covariance_u=cov,
                        covariance_delta=physical_cov,correlation_delta=correlation(physical_cov),names=np.array(model.names))
    result=dict(valid=True,condition=float(eigen[-1]/eigen[0]),valid_draw_fraction=float(valid.mean()),
                transform_seconds=time.perf_counter()-beg,**timing)
    write_json(OUT/'laplace.json',result);return result

def nuts(free=False):
    from numpyro.infer import MCMC,NUTS
    model=Model(free);name=prefix(free);u=np.load(OUT/f'{name}_map.npz')['u']
    # Every chain starts from a valid, independently perturbed location.
    rng=np.random.default_rng(SETTINGS['seed']+5);starts=[]
    for _ in range(SETTINGS['chains']):
        trial=u.copy()
        for attempt in range(20):
            trial=u+rng.normal(0,1e-3,len(u))
            if np.isfinite(float(model.potential(trial))):break
        else:raise RuntimeError('Cannot construct valid NUTS starts')
        starts.append(trial)
    _,_,objective_timing=compile_call(model.vg,jnp.asarray(u))
    invmass=None;mass_note='dense adaptation from identity'
    if not free and (OUT/'laplace.npz').exists():
        invmass=jnp.asarray(np.load(OUT/'laplace.npz')['covariance_u']);mass_note='full positive-definite joint Hessian seed'
    kernel=NUTS(potential_fn=model.potential,dense_mass=True,inverse_mass_matrix=invmass,
                target_accept_prob=SETTINGS['target_accept'],max_tree_depth=SETTINGS['max_tree_depth'])
    method='parallel' if jax.local_device_count()>=2 else 'sequential'
    import os
    sampler=MCMC(kernel,num_warmup=SETTINGS['warmup'],num_samples=SETTINGS['samples'],num_chains=2,
                 chain_method=method,progress_bar=os.environ.get('FEASIBILITY_PROGRESS','1')=='1')
    extra=('potential_energy','diverging','num_steps','accept_prob','energy')
    t=time.perf_counter()
    compilation=[]
    def record_compilation(event,seconds,**kwargs):
        if event in ('/jax/compilation_time_secs','/jax/core/compile/backend_compile_duration'):
            compilation.append(float(seconds))
    jax.monitoring.register_event_duration_secs_listener(record_compilation)
    try:
        sampler.warmup(jax.random.PRNGKey(SETTINGS['seed']+20+int(free)),init_params=jnp.asarray(starts),extra_fields=extra)
        jax.block_until_ready(sampler.last_state);warm_seconds=time.perf_counter()-t
        warm_compile=sum(compilation);compilation.clear()
        sampler.post_warmup_state=sampler.last_state
        t=time.perf_counter();sampler.run(jax.random.PRNGKey(SETTINGS['seed']+21+int(free)),extra_fields=extra)
        grouped=sampler.get_samples(group_by_chain=True);jax.block_until_ready(grouped);sample_seconds=time.perf_counter()-t
        sample_compile=sum(compilation)
    finally:
        jax.monitoring.unregister_event_duration_listener(record_compilation)
    grouped=np.asarray(grouped);extras={k:np.asarray(v) for k,v in sampler.get_extra_fields(group_by_chain=True).items()}
    flat=grouped.reshape(-1,model.dimension);reported=batch_report(model,flat).reshape(grouped.shape)
    np.savez_compressed(OUT/f'{name}_nuts.npz',u=grouped,reported=reported,names=np.array(model.names),**extras)
    diagnostic=parameter_diagnostics(reported,model.names)
    diagnostic.to_csv(OUT/f'{name}_nuts_diagnostics.csv',index=False)
    ess=diagnostic.ess_bulk.to_numpy();elapsed=warm_seconds+sample_seconds
    converged=bool(np.isfinite(diagnostic.rhat).all() and (diagnostic.rhat<1.01).all()
                   and (diagnostic.ess_bulk>=100).all() and not extras['diverging'].any())
    result=dict(warmup_including_sampler_compile_seconds=warm_seconds,sampling_seconds=sample_seconds,
                total_seconds=elapsed,objective_compile_seconds=objective_timing['compile_seconds'],
                sampler_compile_seconds=warm_compile+sample_compile,warmup_backend_compile_seconds=warm_compile,
                sampling_backend_compile_seconds=sample_compile,
                warmup_non_backend_compile_seconds=warm_seconds-warm_compile,
                sampling_non_backend_compile_seconds=sample_seconds-sample_compile,
                compile_note='JAX compilation duration events separate backend compile from synchronized wall time. Remaining time includes Python tracing/dispatch; it is not a pure device-kernel timer.',
                chain_method=method,mass_seed=mass_note,converged=converged,
                divergences=int(extras['diverging'].sum()),tree_depth_fraction=float((extras['num_steps']>=2**SETTINGS['max_tree_depth']-1).mean()),
                mean_accept=float(extras['accept_prob'].mean()),max_rhat=float(diagnostic.rhat.max()),
                min_bulk_ess=float(np.min(ess)),median_bulk_ess=float(np.median(ess)),
                total_seconds_per_min_ess=float(elapsed/np.min(ess)),sampling_seconds_per_median_ess=float(sample_seconds/np.median(ess)))
    write_json(OUT/f'{name}_nuts.json',result)
    if free:
        f=np.broadcast_to(np.asarray(model.fiducial_f),(len(flat),len(model.ids))).copy()
        f[:,model.ratio_indices]=reported.reshape(-1,model.dimension)[:,model.nbase:]
        mapf=np.load(OUT/'ratios_map.npz')['f']
        np.savez_compressed(OUT/'cosmography/f_posteriors.npz',f=f,f_best=mapf,family_ids=model.ids,redshifts=model.z,
                            reference_index=model.ref,chain_id=np.repeat(np.arange(2),SETTINGS['samples']),
                            model_reported=reported,model_names=np.array(model.names),converged=converged,
                            correlation=np.corrcoef(reported.reshape(-1,model.dimension),rowvar=False))
        quant=np.percentile(f,[16,50,84],axis=0)
        rows=[]
        for i,fid in enumerate(model.ids):
            match=diagnostic[diagnostic.parameter==f'f.{fid}']
            rows.append(dict(family_id=fid,z=model.z[i],best=mapf[i],median=quant[1,i],p16=quant[0,i],p84=quant[2,i],std=np.std(f[:,i],ddof=1),
                             fixed=i==model.ref,rhat=np.nan if match.empty else match.rhat.iloc[0],ess_bulk=np.nan if match.empty else match.ess_bulk.iloc[0],
                             boundary_fraction=float(((f[:,i]<.01)|(f[:,i]>.99)).mean())))
        pd.DataFrame(rows).to_csv(OUT/'cosmography/f_summary.csv',index=False)
    return result

def parameter_diagnostics(grouped,names):
    import arviz as az
    rows=[]
    for i,name in progress(enumerate(names),total=len(names),desc='R-hat / ESS',unit='parameter',position=1):
        x=grouped[:,:,i];flat=x.ravel();q=np.percentile(flat,[16,50,84])
        rows.append(dict(parameter=name,median=q[1],p16=q[0],p84=q[2],std=np.std(flat,ddof=1),
                         rhat=float(az.rhat(x,method='rank')),ess_bulk=float(az.ess(x,method='bulk')),ess_tail=float(az.ess(x,method='tail')),
                         skew=float(skew(flat)),excess_kurtosis=float(kurtosis(flat))))
    return pd.DataFrame(rows)

def compare():
    n=np.load(OUT/'baseline_nuts.npz');l=np.load(OUT/'laplace.npz')
    diag=pd.read_csv(OUT/'baseline_nuts_diagnostics.csv');lp=l['reported'][l['valid']]
    if len(lp)<100:raise ValueError('Fewer than 100 physically valid Laplace draws')
    sigma=np.std(lp,axis=0,ddof=1);diag['laplace_std']=sigma;diag['laplace_delta_std']=np.sqrt(np.diag(l['covariance_delta']))
    diag['laplace_over_nuts']=sigma/diag['std'];diag['laplace_under_30pct']=diag.laplace_over_nuts<.7
    diag['non_gaussian']=(abs(diag['skew'])>.5)|(diag.excess_kurtosis>1)
    diag['map']=np.load(OUT/'baseline_map.npz')['reported'];diag.to_csv(OUT/'parameter_comparison.csv',index=False)
    nd=n['reported'].reshape(-1,len(diag));cov=np.cov(nd,rowvar=False)
    np.savez_compressed(OUT/'correlations.npz',laplace=correlation(np.cov(lp,rowvar=False)),nuts=correlation(cov),names=n['names'])
    pd.DataFrame(correlation(cov),index=n['names'],columns=n['names']).to_csv(OUT/'nuts_correlation.csv')
    pd.DataFrame(correlation(np.cov(lp,rowvar=False)),index=n['names'],columns=n['names']).to_csv(OUT/'laplace_correlation.csv')
    nu=n['u'].reshape(-1,len(diag));mean=nu.mean(0)
    # No arbitrary regularization of a rank-deficient empirical region.
    precision,_=require_spd(np.cov(nu,rowvar=False))
    distance=lambda a:np.einsum('ni,ij,nj->n',a-mean,precision,a-mean)
    cutoff=np.percentile(distance(nu),95);outside=distance(l['u'])>cutoff
    lo,hi=np.percentile(nd,[2.5,97.5],axis=0)
    marginal=np.mean((lp<lo)|(lp>hi),axis=0)
    diag['laplace_outside_nuts_marginal95']=marginal;diag.to_csv(OUT/'parameter_comparison.csv',index=False)
    result=dict(ellipsoid_outside_fraction=float(outside.mean()),ellipsoid_cutoff=float(cutoff),
                region='empirical 95% Mahalanobis ellipsoid in unconstrained coordinates; not a non-Gaussian HPD region',
                invalid_laplace_fraction=float(1-l['valid'].mean()),underestimated_parameters=diag.loc[diag.laplace_under_30pct,'parameter'].tolist(),
                non_gaussian_parameters=diag.loc[diag.non_gaussian,'parameter'].tolist())
    write_json(OUT/'comparison.json',result);return result

def residuals():
    from lenscluster.exact_images import ExactImageSolver
    model=Model();m=np.load(OUT/'baseline_map.npz');targets={'map':m['reported']}
    if (OUT/'baseline_nuts.npz').exists():
        values=np.load(OUT/'baseline_nuts.npz')['reported'];targets['posterior_median']=np.median(values.reshape(-1,values.shape[-1]),axis=0)
    output={};arrays={}
    for name,report in progress(targets.items(),desc='Exact image recovery',unit='model',position=1):
        physical,sources=model.physical_from_report(jnp.asarray(report))
        packed,valid=model.ev.packed_state(physical)
        if not bool(valid.is_valid):
            output[name]={'failure':'Physical posterior median is not a valid model'};continue
        solver=ExactImageSolver(model.ev,model.config.image_diagnostics);t=time.perf_counter()
        details=solver.predict_all_global(model.state.images,model.families,np.asarray(physical),np.asarray(sources))
        lookup={str(label):i for i,label in enumerate(model.state.images.image_labels)};pred=np.full_like(model.xy,np.nan);extra=0
        for family,detail in zip(model.families,details):
            rows=[lookup[str(label)] for label in family.image_labels]
            pred[rows]=np.column_stack([detail['matched_model_x_arcsec'],detail['matched_model_y_arcsec']]);extra+=detail['extra_image_count']
        residual=np.linalg.norm(pred-model.xy,axis=1);good=np.isfinite(residual)
        output[name]=dict(recovered=int(good.sum()),missed=int((~good).sum()),extras=int(extra),
            rms=float(np.sqrt(np.mean(residual[good]**2))) if good.any() else None,
            median=float(np.median(residual[good])) if good.any() else None,
            p90=float(np.percentile(residual[good],90)) if good.any() else None,seconds=time.perf_counter()-t,
            missed_labels=np.asarray(model.state.images.image_labels)[~good].tolist(),published_rms=.43)
        arrays[name+'_predicted']=pred;arrays[name+'_residual']=residual
    if len(arrays)==4:
        good=np.isfinite(arrays['map_residual'])&np.isfinite(arrays['posterior_median_residual'])
        output['common_images']=int(good.sum())
        for name in targets:output[name]['common_rms']=float(np.sqrt(np.mean(arrays[name+'_residual'][good]**2))) if good.any() else None
    np.savez_compressed(OUT/'residuals.npz',observed=model.xy,labels=model.state.images.image_labels,**arrays)
    write_json(OUT/'residuals.json',output);return output
