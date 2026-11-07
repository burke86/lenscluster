"""RGB overviews with matched posterior z=9 critical curves and bands."""
from pathlib import Path
from .plot_style import CRITICAL_COLOR
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .grf_plotting import plot_deflection_correction, PLOT_FIELD_VERSION


def overview_grf_correction(evaluator, theta, geometry):
    """Cover the full overview with the same smooth field used by fold plots."""
    base=evaluator._padded_evaluator() if evaluator.gp_groups is not None else evaluator
    corners=np.array([(x,y) for x in geometry['xlim'] for y in geometry['ylim']])
    observed=np.asarray(base.gp['xy'])[np.asarray(base.gp['mask'],bool)]
    extent=float(np.linalg.norm(corners[:,None]-observed[None],axis=-1).max())
    return plot_deflection_correction(evaluator,theta,max_separation=extent)


def marker_photometry(catalog, reference, photpath):
    """Use measured F200W magnitudes only for matching IDs and sky positions."""
    from .cutouts import arcsec_to_radec
    ra,dec=arcsec_to_radec(catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy(),reference)
    result=pd.DataFrame({'image_label':catalog.image_label.to_numpy(),'phot_id':-1,
                         'match_separation_arcsec':np.nan,'m_F200W_AB':np.nan})
    if photpath is not None and Path(photpath).exists():
        saved=pd.read_csv(photpath).set_index('image_label').reindex(result.image_label)
        sep=np.hypot((saved.ra.to_numpy()-ra)*3600*np.cos(np.deg2rad(dec)),(saved.dec.to_numpy()-dec)*3600)
        valid=(sep<.05)&saved.valid_photometry.fillna(False).to_numpy(dtype=bool)&np.isfinite(saved.m_F200W_AB.to_numpy())
        for key in ('phot_id','match_separation_arcsec','m_F200W_AB'):
            result.loc[valid,key]=saved[key].to_numpy()[valid]
    valid=np.isfinite(result.m_F200W_AB.to_numpy())
    result['valid_photometry']=valid
    result['marker_area_pt2']=9.
    result.loc[valid,'marker_area_pt2']=np.clip(16*10**(-.2*(result.loc[valid,'m_F200W_AB']-26)),9,64)
    return result


@plt.rc_context()
def render_rgb_critical(config, state, evaluator, theta, catalog, extra, imaging,
                             output, tables, *, best_index=None, posterior=None, log=print):
    """Render a median determinant contour and interval from the same draws."""
    output,tables=Path(output),Path(tables)
    if posterior is None:
        raise ValueError("Median critical curves require the saved posterior")
    from .mass_figure import overview_geometry, aligned_rgb, grid_pixels
    from . import cutouts
    from .jax_cosmology import dpie_sigma0_factor_from_config
    import hashlib
    images,bands,display,cluster=imaging
    simulated=cluster.lower() in {'hera','ares'}
    stem='hst_rgb_critical_z9' if simulated else 'jwst_rgb_critical_z9'
    m0416=cluster.lower()=='m0416'
    reference_image=images[cutouts.rgb_reference_band(bands)]
    geometry=overview_geometry(reference_image,catalog,extra,state.reference,resolution=768)
    tables.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared_grf=bool(getattr(evaluator,'gp_knee',False) and state.gp_blocks=='all')
    band,band_info=critical_band(config,state,evaluator,posterior,best_index,geometry,tables,log=log,cache_stem=stem)
    with np.load(tables/f'{stem}_band.npz') as saved:
        lower,det,upper=critical_quantile_summary(saved['determinants'])
        fingerprint=hashlib.sha256(str(saved['fingerprint']).encode()+b'median-determinant-contour-v1').hexdigest()
    band_info['definition']='Zero contour of pointwise median determinant; shaded where its 16-84% interval contains zero; same parametric + conditional-mean GRF draws'
    band_info['selection']='Chain containing best draw; not all-chain uncertainty'
    (tables/f'{stem}_band.json').write_text(json.dumps(band_info,indent=2)+'\n')
    np.savez_compressed(tables/f'{stem}.npz',determinant=det,lower=lower,upper=upper,
                        fingerprint=fingerprint,z_source=9.,draw_indices=band_info['draw_indices'],
                        central_statistic='pointwise posterior median determinant')
    log(f'[outputs] rendering {cluster} RGB critical overview')
    rgb=cutouts.rgb_cutout(images,bands,display,geometry['center'],geometry['size'])
    rgb=aligned_rgb(rgb,reference_image,geometry)
    plt.style.use(str(Path(__file__).with_name('style.mplstyle')))
    insets=None
    if m0416:
        from .rgb_overview_cutouts import overview_cutout_data, render_overview_cutouts
        insets=overview_cutout_data(tables,config.paths.figure_prefix,log=log)
    fig=plt.figure(figsize=(11.4,11.4) if insets is not None else (9,9))
    ax=(fig.add_axes([.07,.07,.72,.72],projection=geometry['wcs']) if insets is not None
        else fig.add_subplot(111,projection=geometry['wcs']))
    n=geometry['resolution'];yellow='#FFE34D'
    ax.imshow(rgb,origin='lower',extent=(-.5,n-.5,-.5,n-.5),interpolation='none')
    if band is not None:
        draw_critical_interval(ax,lower,upper,CRITICAL_COLOR,(-.5,n-.5,-.5,n-.5))
    ax.contour(det,levels=[0.],colors=[CRITICAL_COLOR],linewidths=.65,
               origin='lower',extent=(-.5,n-.5,-.5,n-.5))
    px,py=grid_pixels(geometry,catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy())
    photpath=Path(__file__).with_name('data')/'m0416_image_f200w_photometry.csv' if m0416 else None
    marker_table=marker_photometry(catalog,state.reference,photpath)
    mags=marker_table.m_F200W_AB.to_numpy();sizes=marker_table.marker_area_pt2.to_numpy()
    valid=marker_table.valid_photometry.to_numpy()
    candidate_path=getattr(config.image_catalog,'candidate_catalog_path',None)
    candidate_path=(Path(candidate_path) if candidate_path else Path(__file__).with_name('data')/'m0416_candidates.json') if m0416 else None
    candidates=json.loads(candidate_path.read_text())['candidates'] if candidate_path is not None and candidate_path.exists() else []
    family_colors={'K66':'#40DFFF','K67':'#FF70C5','K20':'#FF6060',
                   'K28':'#BD9AFF','K34':'#609FFF','K115':'#FFFFFF','A':'#32E875'}
    systems=catalog.family_id.astype(str).str.extract(r'^(K\d+|C\d+)',expand=False)
    selected={r['system'] for r in candidates if r['kind']=='counterimage'}
    colors=[family_colors.get(system,'#FF9500') if system in selected else '#FF9500' for system in systems]
    ax.scatter(px,py,s=sizes,facecolors='none',edgecolors=colors,linewidths=.8)
    marker_table['system']=systems.to_numpy()
    marker_table['color']=colors
    overlay_rows=[]
    all_px,all_py=list(px),list(py)
    for system in dict.fromkeys(r['system'] for r in candidates):
        color=family_colors.get(system,'#42A5FF')
        label='Family A' if system=='A' else system
        rows=[r for r in candidates if r['system']==system]
        if not rows:
            continue
        ra0,dec0=state.reference[1:]
        x=-(np.array([r['ra'] for r in rows])-ra0)*3600*np.cos(np.deg2rad(dec0))
        y=(np.array([r['dec'] for r in rows])-dec0)*3600
        cx,cy=grid_pixels(geometry,x,y)
        if not np.all((cx>=0)&(cx<n)&(cy>=0)&(cy<n)):
            raise ValueError('Candidate overlay extends beyond the RGB grid')
        ax.scatter(cx,cy,s=48,marker='s',facecolors='none',edgecolors=color,linewidths=1.,zorder=8,label=label)
        all_px.extend(cx);all_py.extend(cy)
        overlay_rows.extend(rows)
    if overlay_rows:
        ax.legend(loc='lower left',fontsize=8,ncol=4,title='Circles: identified  /  squares: proposed',title_fontsize=8,frameon=True,facecolor='black',edgecolor='none',labelcolor='white',framealpha=.75)
        ax.get_legend().get_title().set_color('white')
        pd.DataFrame(overlay_rows).to_csv(tables/f'{stem}_candidate_overlays.csv',index=False)
    # Include both fitted images and proposed candidates, with 3 arcsec padding.
    pad=3./geometry['size']*n
    limits=[float(min(all_px)-pad),float(max(all_px)+pad),float(min(all_py)-pad),float(max(all_py)+pad)]
    ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:])
    assert np.all((px>limits[0])&(px<limits[1])&(py>limits[2])&(py<limits[3]))
    marker_table.to_csv(tables/('hst_rgb_markers.csv' if simulated else 'jwst_rgb_marker_magnitudes.csv'),index=False)
    log(f'[outputs] magnitude-scaled RGB markers: {valid.sum()}/{len(valid)}')
    ra,dec=ax.coords
    ra.set_axislabel('R.A. (J2000)');dec.set_axislabel('Dec. (J2000)')
    ra.set_major_formatter('hh:mm:ss');dec.set_major_formatter('dd:mm:ss')
    ra.set_ticks(number=5,color=yellow,direction='in',size=6,width=1)
    dec.set_ticks(number=5,color=yellow,direction='in',size=6,width=1)
    ra.set_ticks_position('bt');dec.set_ticks_position('lr')
    ra.set_ticklabel_position('b');dec.set_ticklabel_position('l')
    ra.display_minor_ticks(True);dec.display_minor_ticks(True)
    ax.text(.025,.975,r'$z_{\rm s}=9$',transform=ax.transAxes,ha='left',va='top',color=yellow,fontsize=14)
    inset_info=render_overview_cutouts(fig,insets,imaging,state.reference) if insets is not None else None
    fig.savefig(output,dpi=300,bbox_inches='tight',pad_inches=.06);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(run=config.paths.run_name,insets=inset_info,best_index=best_index,central_statistic='zero contour of pointwise posterior median determinant; same draws as band',z_source=9.,model='parametric + conditional mean GRF; no GP displacement' if shared_grf else 'parametric only; no shared GRF available',grf_derivatives=PLOT_FIELD_VERSION,bands=list(bands),catalog_images=len(catalog),resolution=n,determinant_resolution=int(det.shape[0]),critical_definition='median over draws of det(I - Hessian_draw) = 0',cache_fingerprint=fingerprint,marker_photometry=str(photpath) if photpath else None,magnitude_band='F200W AB' if photpath else None,matched_magnitudes=int(valid.sum()),missing_magnitude_policy='minimum marker area 9 pt^2',marker_area='clip(16*10**(-0.2*(m-26)),9,64) pt^2',crop_padding_arcsec=3.,crop_pixel_limits=limits,candidate_catalog=str(candidate_path) if candidate_path else None,candidate_overlays=[r['id'] for r in overlay_rows]),indent=2)+'\n')
    log(f"[outputs] RGB critical overview written: {output}")
    return output


def best_chain_draw_indices(posterior, best_index, count=32):
    """Evenly cover only the chain containing the displayed best draw."""
    grouped=np.asarray(posterior.grouped_samples)
    samples=np.asarray(posterior.samples)
    if grouped.ndim!=3 or not np.array_equal(grouped.reshape(samples.shape),samples):
        raise ValueError('Critical band requires verified chain-major saved samples')
    per_chain=grouped.shape[1]
    chain=int(best_index)//per_chain
    indices=np.unique(np.linspace(chain*per_chain,(chain+1)*per_chain-1,min(count,per_chain)).round().astype(int))
    if best_index not in indices:
        indices[np.argmin(abs(indices-best_index))]=best_index
    return np.sort(indices),chain


def draw_critical_interval(ax, lower, upper, color, extent=None, *, coordinates=None):
    """Shade the intersection lower<=0 and upper>=0, preserving subpixel bands.

    Interpolating min(-lower, upper) first can erase a narrow interval whose
    two boundaries cross a pixel. Intersect their separately contoured regions.
    """
    style=dict(origin='lower',extent=extent) if coordinates is None else {}
    args=() if coordinates is None else tuple(coordinates)
    if np.min(lower)>=0 or np.max(upper)<=0:
        return None
    bound=max(float(np.max(np.abs(lower))),float(np.max(np.abs(upper))),1.)+1.
    fill=ax.contourf(*args,lower,levels=[-bound,0.],colors=[color],alpha=.25,zorder=2,**style)
    clip=ax.contourf(*args,upper,levels=[0.,bound],colors='none',**style)
    fill.set_clip_path(clip.get_paths()[0],ax.transData)
    fill.set_clip_box(ax.bbox)  # Keep the interval clipped to the displayed RGB crop too.
    clip.remove()
    return fill


def critical_quantile_summary(determinants):
    """Matched 16/50/84 percentiles of the same determinant draws."""
    return np.quantile(determinants, [.16, .5, .84], axis=0)


def critical_quantile_band(determinants):
    """Pointwise region where the central 68% determinant interval straddles zero."""
    lower,_,upper=critical_quantile_summary(determinants)
    return lower,upper,(lower<0)&(upper>0)


def critical_band(config,state,evaluator,posterior,best_index,geometry,tables,log=print,*,cache_stem='jwst_rgb_critical_z9'):
    import hashlib
    from .jax_cosmology import dpie_sigma0_factor_from_config
    indices,chain=best_chain_draw_indices(posterior,best_index)
    samples=np.asarray(posterior.samples)[indices]
    resolution=512
    xe=np.linspace(*geometry['xlim'],resolution+1);ye=np.linspace(*geometry['ylim'],resolution+1)
    x,y=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2)
    xy=np.column_stack([x.ravel(),y.ravel()])
    fingerprint=hashlib.sha256(samples.tobytes()+indices.tobytes()+xy.tobytes()+
        json.dumps({'model':config.to_json_dict()['model'],'likelihood':config.to_json_dict()['likelihood']},sort_keys=True,default=str).encode()+PLOT_FIELD_VERSION.encode()+b'critical-band-z9-best-chain-smooth-v2').hexdigest()
    path=tables/f'{cache_stem}_band.npz'
    if path.exists():
        with np.load(path) as saved:
            if str(saved['fingerprint'])==fingerprint:
                log(f'[outputs] cached critical band: chain {chain}, {len(indices)} draws')
                return saved['band'].copy(),dict(chain=chain,draw_indices=indices.tolist(),resolution=resolution,cache_reused=True)
    shared=bool(getattr(evaluator,'gp_knee',False) and state.gp_blocks=='all')
    factor=float(dpie_sigma0_factor_from_config(state.z_lens,9.,state.cosmo_config))/float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))/float(evaluator.gp_weights_per_family[0]) if shared else 0.
    determinants=[]
    for number,theta in enumerate(samples):
        packed=evaluator.packed_state_for_z(evaluator.physical(theta),9.)
        _,hessian=evaluator.total_fields_at(xy,packed,chunk=1024)
        hessian=np.array(hessian,copy=True)
        if shared:
            correction=overview_grf_correction(evaluator,theta,geometry)
            for start in range(0,len(xy),384):
                q=xy[start:start+384]
                c=correction(q[:,0],q[:,1],np.zeros(len(q),dtype=int))
                hessian[start:start+len(q)]+=np.stack(c[2:],axis=-1).reshape(-1,2,2)*factor
        det=np.linalg.det(np.eye(2)-hessian).reshape(x.shape)
        if not np.isfinite(det).all():raise ValueError('Nonfinite posterior determinant grid')
        determinants.append(det)
        log(f'[outputs] critical band chain {chain}: {number+1}/{len(indices)} draws')
    lower,upper,band=critical_quantile_band(determinants)
    np.savez_compressed(path,band=band,lower=lower,upper=upper,determinants=np.asarray(determinants),
                        draw_indices=indices,chain=chain,fingerprint=fingerprint)
    return band,dict(chain=chain,draw_indices=indices.tolist(),resolution=resolution,cache_reused=False)


render_jwst_rgb_critical = render_rgb_critical
