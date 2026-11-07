"""Median critical curves at each cutout's source redshift."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .grf_plotting import PLOT_FIELD_VERSION


def median_cutout_determinants(state,evaluator,posterior,centers,fovs,redshifts,indices,path,fingerprint,*,resolution=96,log=print):
    """Use the same selected joint draws for every source-redshift cutout."""
    from .jax_cosmology import dpie_sigma0_factor_from_config
    from .jwst_rgb_figure import overview_grf_correction
    centers,fovs,redshifts=map(np.asarray,(centers,fovs,redshifts))
    indices=np.asarray(indices)[np.linspace(0,len(indices)-1,min(32,len(indices))).astype(int)]
    axis=np.linspace(-.5,.5,resolution)
    grids=np.array([np.stack(np.meshgrid(c[0]+axis*f,c[1]+axis*f),axis=-1) for c,f in zip(centers,fovs)])
    samples=np.asarray(posterior.samples)[indices]
    key=hashlib.sha256(samples.tobytes()+indices.tobytes()+grids.tobytes()+redshifts.tobytes()+
        fingerprint.encode()+PLOT_FIELD_VERSION.encode()+b'cutout-critical-median-v1').hexdigest()
    path=Path(path)
    if path.exists():
        with np.load(path) as saved:
            if str(saved['fingerprint'])==key and 'lower' in saved and 'upper' in saved:
                return grids,saved['median_determinant'].copy(),indices,saved['lower'].copy(),saved['upper'].copy()
    shared=bool(getattr(evaluator,'gp_knee',False) and state.gp_blocks=='all')
    if getattr(evaluator,'gp_knee',False) and not shared:
        raise ValueError('Cutout critical curves currently require a shared GRF (gp_blocks=all)')
    factors=np.zeros(len(redshifts))
    if shared:
        ref=float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))*float(evaluator.gp_weights_per_family[0])
        factors=np.array([float(dpie_sigma0_factor_from_config(state.z_lens,float(z),state.cosmo_config))/ref for z in redshifts])
    flat=grids.reshape(-1,2)
    geometry=dict(xlim=(flat[:,0].min(),flat[:,0].max()),ylim=(flat[:,1].min(),flat[:,1].max()))
    draws=[]
    for i,theta in enumerate(samples):
        correction=overview_grf_correction(evaluator,theta,geometry) if shared else None
        physical=evaluator.physical(theta)
        fields=[]
        for grid,z,factor in zip(grids,redshifts,factors):
            xy=grid.reshape(-1,2)
            packed=evaluator.packed_state_for_z(physical,float(z))
            _,h=evaluator.total_fields_at(xy,packed,chunk=1024)
            h=np.array(h,copy=True)
            if correction is not None:
                for start in range(0,len(xy),384):
                    q=xy[start:start+384]
                    c=correction(q[:,0],q[:,1],np.zeros(len(q),dtype=int))
                    h[start:start+len(q)]+=np.stack(c[2:],axis=-1).reshape(-1,2,2)*factor
            fields.append(np.linalg.det(np.eye(2)-h).reshape(resolution,resolution))
        draws.append(fields)
        log(f'[outputs] source-redshift cutout critical curves: {i+1}/{len(samples)} draws')
    draws=np.asarray(draws)
    if not np.isfinite(draws).all():
        raise ValueError('Nonfinite cutout determinant fields')
    lower,median,upper=np.quantile(draws,[.16,.5,.84],axis=0)
    np.savez_compressed(path,median_determinant=median,lower=lower,upper=upper,determinants=draws,draw_indices=indices,redshifts=redshifts,
                        centers=centers,fovs=fovs,fingerprint=key)
    path.with_suffix('.json').write_text(json.dumps(dict(definition='Zero contour of pointwise posterior median determinant; band where the same draws 16-84% determinant interval contains zero',
        model='parametric + conditional mean GRF' if shared else 'parametric',draw_indices=indices.tolist(),
        redshifts=redshifts.tolist(),resolution=resolution),indent=2)+'\n')
    return grids,median,indices,lower,upper
