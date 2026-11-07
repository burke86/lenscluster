"""Luminosity-scaled galaxy aperture masses for the Hera/Ares truth comparison.

Fixed illustrative F160W sizes, not sizes inferred from the lens model. Full
projected masses include the conditional GRF mean when the run supports a shared field.
"""
from pathlib import Path
import json

import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull, QhullError
from astropy.wcs.utils import proj_plane_pixel_area

from .plot_style import PAPER_STYLE
from .truth_recovery import load_truth_fits, mass_per_kappa_pixel_msun

STYLE = Path(__file__).with_name('style.mplstyle')
NORMALIZATION_KPC = 8.


def conditional_grf(evaluator, physical, z_source):
    """Mean field at the physical median, preserving its physical source positions."""
    if not (getattr(evaluator,'gp_knee',False) and evaluator.state.gp_blocks=='all'
            and evaluator.gp_sigma_alpha_index>=0 and evaluator.gp_knee_spectrum=='knee'):
        return None, 1.
    from .model import convert_theta_to_latent
    from .jax_cosmology import dpie_sigma0_factor_from_config
    evaluator=evaluator._padded_evaluator()
    state=evaluator.state
    theta=convert_theta_to_latent(physical,state.parameter_specs)
    ix=np.asarray(state.images.source_x_param_index);iy=np.asarray(state.images.source_y_param_index)
    theta[np.r_[ix,iy]]=0
    whitening=jax.jit(lambda t:evaluator.statistics(t)['whitening'])(theta)
    layout=evaluator.gp;mask=np.asarray(layout['knot_mask']);family=np.asarray(layout['knot_family'])
    sources=np.stack((physical[ix],physical[iy]),-1)
    target=np.where(mask[...,None],sources[np.where(mask,family,0)],0).reshape(mask.shape[0],-1)
    offset=np.linalg.solve(np.asarray(whitening['cholesky']),
        (target-np.asarray(whitening['centroid']))[...,None])[...,0].reshape(*mask.shape,2)
    real=offset[np.asarray(layout['real_block']),np.asarray(layout['real_knot'])]
    real_family=np.asarray(layout['real_family'])
    theta[ix[real_family]]=real[:,0];theta[iy[real_family]]=real[:,1]
    np.testing.assert_allclose(jax.jit(evaluator.reported_physical)(theta),physical,rtol=1e-10,atol=1e-10)
    correction=evaluator.gp_deflection_correction(theta)
    reference=float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))
    target=float(dpie_sigma0_factor_from_config(state.z_lens,z_source,state.cosmo_config))
    return correction,target/reference/float(evaluator.gp_weights_per_family[0])


def inside_image_footprint(centers, images):
    images = np.unique(np.asarray(images, dtype=float), axis=0)
    images = images[np.isfinite(images).all(axis=1)]
    if len(images) < 3:
        return np.zeros(len(centers), dtype=bool)
    try:
        hull = ConvexHull(images)
    except QhullError:
        return np.zeros(len(centers), dtype=bool)
    return np.all(np.asarray(centers) @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-10, axis=1)


def distribution_statistics(table):
    result = {}
    valid = table.complete_aperture & np.isfinite(table.ratio)
    for name, mask in [('all', valid), ('image_hull', valid & table.inside_image_hull)]:
        values = table.loc[mask, 'ratio'].to_numpy()
        low, median, high = np.quantile(values, [.25,.5,.75]) if len(values) else (np.nan,)*3
        product = (high-low)*abs(median-1)
        result[name] = dict(n=len(values),median=float(median),p25=float(low),p75=float(high),
                            iqr=float(high-low),Q=float(-np.log10(product)) if product>0 else np.nan)
    return result


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def render_histogram(table, path):
    selected = table[table.complete_aperture & np.isfinite(table.ratio)]
    if selected.empty:
        return None
    hull = selected[selected.inside_image_hull]
    low = min(0., np.floor(selected.ratio.min()*20)/20)
    upper = max(3., np.ceil(selected.ratio.max()*20)/20)
    bins = np.arange(low, upper+.05001, .05)
    counts, _ = np.histogram(selected.ratio, bins)
    inner, _ = np.histogram(hull.ratio, bins)
    limits = np.quantile(selected.ratio, [.005,.995])
    if limits[0] == limits[1]:
        limits = limits + np.array([-.05,.05])
    fig, ax = plt.subplots(figsize=(3.5,2.6))
    ax.stairs(counts,bins,color='tab:blue',lw=1.5,label='All members',zorder=3)
    ax.stairs(inner,bins,color='tab:orange',lw=1.5,label='Inside image footprint',zorder=4)
    ax.axvline(1,color='.25',ls='--',lw=.8)
    ax.set(xlabel=r'$M/M_{\rm true}$',ylabel='Number of galaxies',xlim=limits,
           ylim=(0,max(counts.max(),inner.max())*1.15))
    ax.legend(frameon=False,loc='upper right',fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(4,integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(which='both',top=False,right=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    return pd.DataFrame(dict(bin_left=bins[:-1],bin_right=bins[1:],
                             all_members=counts,inside_image_hull=inner))


def aperture_masses(state, evaluator, physical, members, image_xy, truth_config, *, normalization_kpc=NORMALIZATION_KPC,
                    grf_correction=None, grf_weight=1.):
    if not np.isfinite(normalization_kpc) or normalization_kpc <= 0:
        raise ValueError('Effective-radius normalization must be finite and positive')
    centers = members[['x_centre','y_centre']].to_numpy(float)
    mag = members.catalog_mag.to_numpy(float)
    re = normalization_kpc * 10**(-.24*(mag-20.))
    radius = 2*re/state.kpc_per_arcsec
    native, wcs = load_truth_fits(truth_config.kappa_true_fits)
    z = float(truth_config.caustic_source_redshift)
    _, ra0, dec0 = state.reference
    def pixels(xy):
        return wcs.world_to_pixel_values(ra0-xy[...,0]/(3600*np.cos(np.deg2rad(dec0))),
                                         dec0+xy[...,1]/3600)
    phi = np.arange(1024)*2*np.pi/1024
    edge = centers[:,None]+radius[:,None,None]*np.stack((np.cos(phi),np.sin(phi)),-1)
    px,py = pixels(edge)
    valid = np.isfinite(centers).all(1)&np.isfinite(radius)&(radius>0)
    valid &= np.all((px>=1)&(py>=1)&(px<=native.shape[1]-2)&(py<=native.shape[0]-2),axis=1)
    packed = evaluator.packed_state_for_z(jnp.asarray(physical), z)
    def model_integral(nt):
        angles=(np.arange(nt)+.5)*2*np.pi/nt
        n=np.stack((np.cos(angles),np.sin(angles)),-1)
        out=np.full(len(centers),np.nan)
        if valid.any():
            xy=centers[valid,None]+radius[valid,None,None]*n
            alpha,_=evaluator.total_fields_at(xy.reshape(-1,2),packed,chunk=4096)
            if grf_correction is not None:
                points=xy.reshape(-1,2)
                for start in range(0,len(points),1024):
                    block=points[start:start+1024];count=len(block)
                    block=np.pad(block,((0,1024-count),(0,0)),mode='edge')
                    field=grf_correction(block[:,0],block[:,1],np.zeros(1024,dtype=int))
                    alpha[start:start+count] += np.stack(field[:2],-1)[:count]*grf_weight
            out[valid]=np.pi*radius[valid]*np.mean(np.sum(alpha.reshape(-1,nt,2)*n,axis=-1),axis=1)
        return out
    def truth_integral(nr,nt):
        q,w=np.polynomial.legendre.leggauss(nr)
        r=(q+1)/2
        angles=(np.arange(nt)+.5)*2*np.pi/nt
        unit=(r[:,None,None]*np.stack((np.cos(angles),np.sin(angles)),-1)[None]).reshape(-1,2)
        weights=np.broadcast_to((w*r/2)[:,None],(nr,nt)).ravel()*2*np.pi/nt
        out=np.full(len(centers),np.nan)
        for i in np.flatnonzero(valid):
            px,py=pixels(centers[i]+radius[i]*unit)
            values=map_coordinates(native,[py,px],order=1,mode='constant',cval=np.nan)
            out[i]=radius[i]**2*(values@weights)
        return out
    def relative(a,b):
        return np.abs(a-b)/np.maximum(np.abs(a),1e-30)
    mc,model=model_integral(256),model_integral(512)
    tc,truth=truth_integral(48,192),truth_integral(96,384)
    me,te=relative(model,mc),relative(truth,tc)
    if np.any(me[valid]>.005):
        old=model;model=model_integral(1024);me=relative(model,old)
    if np.any(te[valid]>.005):
        old=truth;truth=truth_integral(192,768);te=relative(truth,old)
    usable=valid&np.isfinite(truth)&(truth>0)&np.isfinite(model)&(me<=.005)&(te<=.005)
    factor=mass_per_kappa_pixel_msun(state,wcs,z)/(float(proj_plane_pixel_area(wcs))*3600**2)
    table=members[['member_id','catalog_id','catalog_mag']].copy()
    table['re_kpc']=re;table['aperture_arcsec']=radius
    table['complete_aperture']=valid;table['usable']=usable
    table['inside_image_hull']=inside_image_footprint(centers,image_xy)
    table['model_mass_msun']=model*factor;table['truth_mass_msun']=truth*factor
    table['ratio']=np.divide(model,truth,out=np.full_like(model,np.nan),where=usable)
    table['model_quadrature_error']=me;table['truth_quadrature_error']=te
    return table


def write_aperture_mass_histogram(config,state,evaluator,posterior,paths,image_df,*,log=print):
    prefix=str(config.paths.figure_prefix or '')
    if prefix.lower() not in {'hera','ares'}:
        return None
    if not config.truth.kappa_true_fits or not len(state.members.member_ids):
        log('[aperture] skipped: truth convergence or population members unavailable')
        return None
    bands = {str(b).upper() for p in getattr(state,'potfiles',[])
             for b in p['catalog_df'].get('catalog_band',pd.Series(dtype=str)).dropna() if str(b)}
    if bands and bands != {'F160W'}:
        log(f'[aperture] skipped: fixed size relation requires F160W, catalog uses {sorted(bands)}')
        return None
    members=pd.read_csv(paths.tables/'member_diagnostics.csv',dtype={'member_id':str,'catalog_id':str})
    if members.member_id.tolist()!=list(state.members.member_ids):
        raise ValueError('Aperture catalog identities differ from saved model')
    physical=np.median(posterior.physical,axis=0)
    correction,weight=conditional_grf(evaluator,physical,float(config.truth.caustic_source_redshift))
    description='conditional GRF mean included' if correction is not None else 'parametric only; shared GRF correction unavailable'
    log(f'[aperture] exact 2Re masses: F160W size normalization {NORMALIZATION_KPC:g} kpc at magnitude 20; {description}')
    table=aperture_masses(state,evaluator,physical,members,
                          image_df[['x_obs_arcsec','y_obs_arcsec']].to_numpy(),config.truth,
                          grf_correction=correction,grf_weight=weight)
    table.to_csv(paths.tables/'member_aperture_masses.csv',index=False)
    stats=dict(normalization_kpc=NORMALIZATION_KPC,samples=distribution_statistics(table),
               equation=f'Re[kpc] = {NORMALIZATION_KPC:g} * 10**(-0.24*(m_F160W-20)); aperture radius = 2 Re',
               model=f'componentwise physical posterior median; exact parametric kernels; {description}',
               grf_included=correction is not None,
               footprint='galaxy centers within convex hull of observed multiple images',
               display='central 99% of all usable ratios; statistics use full distribution',
               skipped_member_ids=table.loc[~table.usable,'member_id'].tolist())
    (paths.tables/'aperture_mass_histogram.json').write_text(json.dumps(stats,indent=2)+'\n')
    path=paths.figures/f'{prefix}_all_vs_image_footprint_histogram.pdf'
    counts=render_histogram(table,path)
    if counts is None:
        log('[aperture] no usable apertures; no histogram written')
        return None
    counts.to_csv(paths.tables/'all_vs_image_footprint_histogram.csv',index=False)
    log(f'[aperture] wrote {path}; {int(table.usable.sum())}/{len(table)} usable members')
    return path
