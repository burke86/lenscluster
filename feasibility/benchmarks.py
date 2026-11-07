"""Synchronized field and blocked pair-screen throughput measurements."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay
from .common import OUT, SETTINGS, compile_call, timed_call, indices, write_json, read_json
from .model import Model
from .runtime import progress,available_cpus
from .distances import flat_wcdm_lensing_efficiency

def field_function(model):
    def field(report,points,efficiency):
        physical,_=model.physical_from_report(report)
        result=model.fields(physical,efficiency,xy=points,derivative=False)
        return result['backprojected'],result['jacobian']
    return jax.jit(field)

def random_footprint(xy,n,rng):
    hull=Delaunay(xy);lo=xy.min(0);hi=xy.max(0);parts=[];count=0
    while count<n:
        draw=rng.uniform(lo,hi,(max(1024,n-count),2));draw=draw[hull.find_simplex(draw)>=0]
        parts.append(draw);count+=len(draw)
    return np.concatenate(parts)[:n]

@jax.jit
def pair_count(beta_a,cov_a,beta_b,cov_b):
    difference=beta_a[:,None]-beta_b[None]
    cov=cov_a[:,None]+cov_b[None]
    a,b,c=cov[...,0,0],cov[...,0,1],cov[...,1,1]
    dx,dy=difference[...,0],difference[...,1]
    score=(c*dx*dx-2*b*dx*dy+a*dy*dy)/(a*c-b*b)
    return jnp.sum(score<5.9914645471)  # descriptive 2D Gaussian threshold only

def benchmark():
    model=Model();saved=np.load(OUT/'baseline_nuts.npz');flat=saved['reported'].reshape(-1,len(model.names))
    chosen=indices(len(flat),100);draws=flat[chosen]
    if len(draws)<100:raise ValueError('Benchmark requires 100 actual posterior draws')
    rng=np.random.default_rng(SETTINGS['seed']);points=random_footprint(model.xy,10000,rng)
    redshifts=np.array(SETTINGS['trial_redshifts']);f=np.asarray(flat_wcdm_lensing_efficiency(.396,redshifts,70.,.3,-1.))
    batch=1024;fn=field_function(model)
    executable,_,timing=compile_call(fn,jnp.asarray(draws[0]),jnp.asarray(points[:batch]),jnp.asarray(f[0]))
    t=time.perf_counter();checksum=0.
    for di,report in progress(enumerate(draws),total=len(draws),desc='Field benchmark',unit='draw',position=1):
        report=jnp.asarray(report)
        for efficiency in f:
            for start in range(0,len(points),batch):
                block=points[start:start+batch];length=len(block)
                if length<batch:block=np.pad(block,((0,batch-length),(0,0)),mode='edge')
                beta,a=jax.block_until_ready(executable(report,jnp.asarray(block),jnp.asarray(efficiency)))
                checksum+=float(beta[0,0])+float(a[0,0,0])
    seconds=time.perf_counter()-t
    beta,a=executable(jnp.asarray(draws[0]),jnp.asarray(points[:batch]),jnp.asarray(f[0]))
    sd=float(draws[0,model.names.index('images.sigma_def')])
    cov=.3**2*(a@jnp.swapaxes(a,-1,-2))+sd**2*jnp.eye(2)
    kernel,_,pair_timing=compile_call(pair_count,beta,cov,beta[::-1],cov[::-1])
    repeat=[timed_call(kernel,beta,cov,beta[::-1],cov[::-1])[1] for _ in range(8)]
    pair_seconds=float(np.median(repeat));throughput=batch**2/pair_seconds
    # Same gradients, padded repeated constraints: a kernel scaling proxy, not a new fit.
    vg=model.vg;u=np.load(OUT/'baseline_map.npz')['u'];_,_,grad_timing=compile_call(vg,jnp.asarray(u))
    grad_repeat=[timed_call(vg,jnp.asarray(u))[1] for _ in range(8)]
    optimization=read_json(OUT/'baseline_optimization.json',{});nuts=read_json(OUT/'baseline_nuts.json',{})
    projections={}
    for n in (10000,50000):
        field_seconds=seconds*n/10000;pair_total=n*(n-1)//2
        projections[str(n)]=dict(unordered_pairs=pair_total,field_seconds=field_seconds,
            exhaustive_pair_seconds=pair_total*800/throughput,total_seconds=field_seconds+pair_total*800/throughput)
    result=dict(field_execution_seconds=seconds,field_compile_seconds=timing['compile_seconds'],positions=10000,
        cpu_ids=available_cpus(),jax_devices=[str(d) for d in jax.devices()],
        draws=100,redshifts=redshifts,total_rays=8000000,padded_total_rays=8192000,checksum=checksum,
        pair_block_size=batch,pair_compile_seconds=pair_timing['compile_seconds'],pair_seconds=pair_seconds,
        pair_comparisons_per_second=throughput,gradient_seconds=float(np.median(grad_repeat)),gradient_compile_seconds=grad_timing['compile_seconds'],
        projections=projections,refit_300=dict(image_ratio=300/237,
          optimization_seconds=optimization.get('execution_seconds',0)*300/237,
          nuts_seconds=nuts.get('total_seconds',0)*300/237,
          assumption='Linear image-count proxy only; fixed member set, family/source count, optimizer iterations and leapfrog counts; compilation excluded. Not an executed 300-image fit.'),
        screening_definition='Exhaustive same-trial-redshift source proximity using C_i+C_j and threshold 5.991; no photo-z, flux, morphology, or family-assignment inference.',
        storage='Field timing includes per-block synchronization/checksum transfer; excludes writing every field array. Pair timing is device compute; projections exclude candidate-list I/O.',
        posterior_converged=nuts.get('converged',False))
    np.savez_compressed(OUT/'benchmark_inputs.npz',positions=points,draw_indices=chosen,redshifts=redshifts)
    write_json(OUT/'benchmarks.json',result);return result
