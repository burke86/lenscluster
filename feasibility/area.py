"""Adaptive inverse triangular mapping, with explicit numerical status flags.

This estimates multiplicity *inside the observed-image hull*, not the total
strong-lensing cross section. It is deliberately bounded and may fail to converge.
"""
import time
import numpy as np
from scipy.spatial import Delaunay
from .distances import flat_wcdm_lensing_efficiency
from lenscluster.exact_images import ExactImageSolver
from .model import Model
from .cosmography import fields_on_points
from .common import OUT, indices, write_json, read_json

MAX_TRIANGLES = 1500000
MAX_RASTER_VISITS = 30000000

def unique_roots(roots, tolerance=.05):
    result=[]
    for root in roots:
        if not result or np.min(np.linalg.norm(np.asarray(result)-root,axis=1))>tolerance:
            result.append(root)
    return np.asarray(result).reshape(-1,2)

def base_triangles(bounds,pitch):
    lo,hi=bounds
    x=np.linspace(lo[0],hi[0],int(np.ceil((hi[0]-lo[0])/pitch))+1)
    y=np.linspace(lo[1],hi[1],int(np.ceil((hi[1]-lo[1])/pitch))+1)
    xx,yy=np.meshgrid(x,y);points=np.stack([xx,yy],-1)
    a=points[:-1,:-1].reshape(-1,2);b=points[:-1,1:].reshape(-1,2)
    c=points[1:,1:].reshape(-1,2);d=points[1:,:-1].reshape(-1,2)
    return np.concatenate([np.stack([a,b,c],1),np.stack([a,c,d],1)])

def adaptive_triangles(mapping,triangles,source_pitch,max_depth=3):
    """Refine curved mappings and parity-changing triangles using edge midpoints."""
    finished=[];mapped=[];unresolved=0
    for depth in range(max_depth+1):
        if len(triangles)+sum(map(len,finished))>MAX_TRIANGLES:
            raise RuntimeError('Area adaptive mesh exceeded triangle limit')
        beta,a=mapping(triangles.reshape(-1,2));beta=beta.reshape(-1,3,2)
        det=np.linalg.det(a).reshape(-1,3)
        mid=(triangles+np.roll(triangles,-1,axis=1))/2
        mb,_=mapping(mid.reshape(-1,2));mb=mb.reshape(-1,3,2)
        error=np.linalg.norm(mb-(beta+np.roll(beta,-1,axis=1))/2,axis=-1).max(1)
        curved=error>source_pitch*.2
        critical=(det.min(1)<0)&(det.max(1)>0)
        refine=curved|critical
        if depth==max_depth:
            unresolved=int(curved.sum());refine[:]=False
        finished.append(triangles[~refine]);mapped.append(beta[~refine])
        if not refine.any():break
        t=triangles[refine];m=mid[refine]
        triangles=np.concatenate([np.stack([t[:,0],m[:,0],m[:,2]],1),
            np.stack([m[:,0],t[:,1],m[:,1]],1),np.stack([m[:,2],m[:,1],t[:,2]],1),m])
    return np.concatenate(finished),np.concatenate(mapped),unresolved

def raster_roots(triangles,beta,pitch,inside,offset=.5):
    """Invert mapped triangles at a common source grid; merge shared-edge roots."""
    roots={};visits=0
    for t,b in zip(triangles,beta):
        matrix=np.column_stack([b[1]-b[0],b[2]-b[0]])
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-16:continue
        low=np.ceil(b.min(0)/pitch-offset).astype(int);high=np.floor(b.max(0)/pitch-offset).astype(int)
        if (high<low).any():continue
        visits+=int(np.prod(high-low+1))
        if visits>MAX_RASTER_VISITS:raise RuntimeError('Area raster exceeded visit limit; no converged area reported')
        ix,iy=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1))
        keys=np.column_stack([ix.ravel(),iy.ravel()]);source=(keys+offset)*pitch
        uv=np.linalg.solve(matrix,(source-b[0]).T).T
        good=(uv.min(1)>=-1e-10)&(uv.sum(1)<=1+1e-10)
        keys=keys[good];uv=uv[good]
        image=t[0]+uv[:,0,None]*(t[1]-t[0])+uv[:,1,None]*(t[2]-t[0])
        good=inside(image)
        for key,root in zip(keys[good],image[good]):
            k=tuple(key);old=roots.setdefault(k,[])
            if not old or np.min(np.linalg.norm(np.asarray(old)-root,axis=1))>.05:old.append(root)
    return roots

def verify_roots(model,report,efficiency,roots,pitch):
    """Refine representatives of single and multiple roots with the exact solver."""
    physical,_=model.physical_from_report(report)
    from lenscluster.jax_cosmology import dpie_sigma0_factor_from_lensing_efficiency
    packed,_=model.ev.packed_state(physical,dpie_sigma0_factor_from_lensing_efficiency(efficiency))
    solver=ExactImageSolver(model.ev,model.config.image_diagnostics);hull=Delaunay(model.xy)
    selected=[]
    for multiple in (False,True):
        keys=sorted(k for k,v in roots.items() if (len(v)>=2)==multiple)
        selected.extend(keys[i] for i in indices(len(keys),16))
    if not selected:return dict(valid=False,reason='No raster roots available for verification')
    failures=0;max_residual=0.
    for key in selected:
        starts=np.array(roots[key]);source=(np.array(key)+.5)*pitch
        r=solver._refine(packed,*source,starts[:,0],starts[:,1])
        xy=np.column_stack([r['x'],r['y']]);keep=r['identification_pass']&(hull.find_simplex(xy)>=0)
        refined=unique_roots(xy[keep]);failures+=int((len(refined)>=2)!=(len(starts)>=2))
        max_residual=max(max_residual,float(np.max(r['final_source_residual_arcsec'])))
    return dict(valid=failures==0,checked=len(selected),multiplicity_failures=failures,max_source_residual=max_residual,
        limitation='Local refinement verifies candidate roots; completeness relies on mesh and resolution convergence.')

def one_area(model,report,z):
    f=float(flat_wcdm_lensing_efficiency(.396,z,70.,.3,-1.));hull=Delaunay(model.xy)
    mapping=lambda xy:fields_on_points(model,report,xy,f)
    bounds=(model.xy.min(0),model.xy.max(0));history=[];last=None;agreement=0
    for level in range(4):
        image_pitch=1./2**level;source_pitch=.08/2**level
        t,b,unresolved=adaptive_triangles(mapping,base_triangles(bounds,image_pitch),source_pitch)
        roots=raster_roots(t,b,source_pitch,lambda x:hull.find_simplex(x)>=0)
        area=sum(len(v)>=2 for v in roots.values())*source_pitch**2
        # A second grid origin exposes source-grid aliasing near narrow caustics.
        shifted=raster_roots(t,b,source_pitch,lambda x:hull.find_simplex(x)>=0,offset=.25)
        shifted_area=sum(len(v)>=2 for v in shifted.values())*source_pitch**2
        change=np.inf if last is None else abs(area-last)/max(area,last,source_pitch**2)
        alias=abs(area-shifted_area)/max(area,shifted_area,source_pitch**2)
        history.append(dict(level=level,area=area,image_pitch=image_pitch,source_pitch=source_pitch,
                            relative_change=change,grid_shift_change=alias,triangles=len(t),unresolved=unresolved))
        agreement=agreement+1 if change<.05 and alias<.05 and unresolved==0 and area>0 else 0
        if agreement>=2:
            verification=verify_roots(model,report,f,roots,source_pitch)
            return dict(valid=verification['valid'],area=area,history=history,verification=verification)
        last=area
    return dict(valid=False,area=last,history=history,reason='Did not meet 5% resolution convergence twice; zero area is not certified')

def areas():
    from .runtime import process_pool,area_task,worker_count,progress
    from concurrent.futures import as_completed
    model=Model();post=np.load(OUT/'baseline_nuts.npz')['reported'];flat=post.reshape(-1,len(model.names))
    chosen=indices(len(flat),32);best=np.load(OUT/'baseline_map.npz')['reported']
    reports=np.vstack([best[None],flat[chosen]]);zs=np.arange(.5,10.01,.5)
    result=np.full((len(reports),len(zs)),np.nan);valid=np.zeros_like(result,bool)
    history=[];checkpoint=OUT/'cosmography/area_checkpoint.json';completed=read_json(checkpoint,{})
    jobs=[(f'draw{i:03d}_z{z:.1f}',-1 if i==0 else int(chosen[i-1]),float(z),report)
          for i,report in enumerate(reports) for z in zs if f'draw{i:03d}_z{z:.1f}' not in completed]
    with progress(total=result.size,initial=result.size-len(jobs),desc='Source areas',unit='model/z',position=1) as bar:
        def receive(answer):
            key,entry=answer;completed[key]=entry;write_json(checkpoint,completed)
            bar.update();bar.set_postfix(valid=entry['valid'],refresh=False)
        if worker_count()>1 and jobs:
            # Spawn avoids forking an initialized JAX runtime. Each process
            # owns its model/cache and a disjoint subset of the CPU budget.
            with process_pool() as pool:
                futures={pool.submit(area_task,job):job[0] for job in jobs}
                for future in as_completed(futures):receive(future.result())
        else:
            for job in jobs:receive(area_task(job))
    for i,report in enumerate(reports):
        for j,z in enumerate(zs):
            key=f'draw{i:03d}_z{z:.1f}';entry=completed[key]
            result[i,j]=entry.get('area',np.nan);valid[i,j]=entry['valid'];history.append(entry)
        np.savez_compressed(OUT/'cosmography/source_area.npz',z=zs,best=result[0],draws=result[1:],
            best_valid=valid[0],draw_valid=valid[1:],draw_indices=chosen,
            posterior_converged=read_json(OUT/'baseline_nuts.json',{}).get('converged',False))
    write_json(OUT/'cosmography/area_status.json',history)
    return dict(valid_cells=int(valid.sum()),total_cells=int(valid.size),all_converged=bool(valid.all()))
