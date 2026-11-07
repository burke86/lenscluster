"""JWST image-position posterior cutouts ranked by the near-singular eigenvalue."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
from .grf_plotting import plot_deflection_correction, PLOT_FIELD_VERSION
from .fold_diagnostics import _credible_density_grid
from .plot_style import PAPER_STYLE

def singular_eigenvalue(jacobian):
    """Signed eigenvalue nearest zero, not the algebraically smallest eigenvalue."""
    a=np.asarray(jacobian,dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite reporting Jacobian')
    asymmetry=float(np.max(np.abs(a-np.swapaxes(a,-1,-2))))
    if asymmetry>1e-10*max(1.,float(np.max(np.abs(a)))):
        raise ValueError(f'Jacobian is not symmetric: maximum asymmetry {asymmetry:g}')
    eigenvalues=np.linalg.eigvalsh(.5*(a+np.swapaxes(a,-1,-2)))
    selected=np.take_along_axis(eigenvalues,np.argmin(np.abs(eigenvalues),axis=-1)[...,None],axis=-1)[...,0]
    np.testing.assert_allclose(np.abs(selected),np.linalg.svd(a,compute_uv=False)[...,-1],rtol=1e-9,atol=1e-12)
    return selected,asymmetry


@plt.rc_context()
def write_small_lambda_jwst(config, state, evaluator, posterior, paths, imaging, position_contours,
                           *, count=24, minimum_fov=3., log=print):
    """Reuse the overview's verified joint position draws; no new image searches."""
    if count < 1 or minimum_fov <= 0:
        raise ValueError('count and minimum_fov must be positive')
    data=position_contours['position_data']
    fingerprint=position_contours['fingerprint']
    positions=np.asarray(data['recovered'])
    labels=np.asarray(data['image_labels']).astype(str)
    indices=np.asarray(data['draw_indices'],dtype=int)
    np.testing.assert_array_equal(labels,np.asarray(state.images.image_labels).astype(str))
    if len(np.unique(indices))!=len(indices):
        raise ValueError('Duplicate posterior draw indices')
    if len(indices)==0:
        log('[outputs] small-lambda JWST cutouts skipped: no posterior draws')
        return None
    xy=np.column_stack((state.images.x_obs,state.images.y_obs))
    families=np.asarray(state.images.family_index_per_image,dtype=int)
    stem=f'{config.paths.figure_prefix}_small_lambda_a_image_posteriors_jwst'
    field=jax.jit(lambda t:evaluator.fields(evaluator.physical(t),force_exact=True,derivative=False)['jacobian'])
    cache_meta=paths.tables/f'{stem}.json'
    cache_npz=paths.tables/f'{stem}.npz'
    previous=json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
    reuse_cache=(cache_npz.exists() and previous.get('posterior_cache_fingerprint')==fingerprint
                 and previous.get('grf_version')==PLOT_FIELD_VERSION)
    if reuse_cache:
        with np.load(paths.tables/f'{stem}.npz') as saved:
            np.testing.assert_array_equal(saved['draw_indices'],indices)
            np.testing.assert_array_equal(saved['image_labels'],labels)
            values=saved['lambda_a'].copy()
        asymmetries=[previous['jacobian_max_asymmetry']]
    else:
        values=[];asymmetries=[]
        for k,index in enumerate(indices):
            theta=posterior.samples[index]
            a=np.array(field(jnp.asarray(theta)),copy=True)
            correction=plot_deflection_correction(evaluator,theta) if evaluator.gp_knee else None
            if correction is not None:
                c=correction(xy[:,0],xy[:,1],families)
                a-=np.stack(c[2:],axis=-1).reshape(-1,2,2)
            eigenvalue,asymmetry=singular_eigenvalue(a)
            values.append(eigenvalue);asymmetries.append(asymmetry)
            if k%32==0 or k==len(indices)-1:log(f'[outputs] small-lambda Jacobians: {k+1}/{len(indices)} draws')
        values=np.asarray(values)
    quantiles=np.percentile(abs(values),[16,50,84],axis=0)
    z_by_family={str(f.family_id):f.z_source for f in state.family_data}
    family_labels=np.asarray(state.images.family_ids).astype(str)[families]
    rows=pd.DataFrame(dict(image_label=labels,family_id=family_labels,z_source=[z_by_family[f] for f in family_labels],
        x_arcsec=xy[:,0],y_arcsec=xy[:,1],abs_lambda_a_p16=quantiles[0],abs_lambda_a_median=quantiles[1],
        abs_lambda_a_p84=quantiles[2],recovered_draws=np.isfinite(positions).all(2).sum(0),total_draws=len(indices)))
    rows=rows.sort_values(['abs_lambda_a_median','image_label'],kind='stable').reset_index(names='image_index')
    rows['rank']=np.arange(1,len(rows)+1);rows['shown']=False;rows['skip_reason']='';rows['fov_arcsec']=np.nan
    images,bands,display,cluster=imaging
    reference=images[rgb_reference_band(bands)]
    selected=[]
    for row_index,row in rows.iterrows():
        coord=arcsec_to_skycoord(row.x_arcsec,row.y_arcsec,state.reference)
        window=cutout_window(reference,coord,minimum_fov)
        if window is None:
            rows.loc[row_index,'skip_reason']='Outside JWST footprint';continue
        grid=_credible_density_grid(positions[:,int(row.image_index)])
        if grid is None:
            rows.loc[row_index,'skip_reason']='Posterior KDE unavailable';continue
        import contourpy
        segments=contourpy.contour_generator(x=grid['x'],y=grid['y'],z=grid['density']).lines(grid['levels'][0])
        if not segments:
            rows.loc[row_index,'skip_reason']='No resolved 2-sigma contour';continue
        vertices=np.concatenate(segments)
        radius=np.max(np.abs(vertices-np.array([row.x_arcsec,row.y_arcsec])))
        fov=float(.5*np.ceil(max(minimum_fov,2.3*radius)/.5))
        window=cutout_window(reference,coord,fov)
        if window is None:
            rows.loc[row_index,'skip_reason']='Full contour outside JWST footprint';continue
        selected.append((row,coord,window,grid,fov));rows.loc[row_index,'shown']=True
        rows.loc[row_index,'fov_arcsec']=fov
        if len(selected)>=count:break
    if not selected:
        rows.to_csv(paths.tables/f'{stem}.csv',index=False)
        log('[outputs] small-lambda JWST cutouts skipped: no resolved posterior contours within imaging')
        return None
    stem=f'{config.paths.figure_prefix}_small_lambda_a_image_posteriors_jwst'
    plt.style.use(Path(__file__).with_name('style.mplstyle'));plt.rcParams.update(PAPER_STYLE)
    columns=min(3,len(selected));nrows=int(np.ceil(len(selected)/columns))
    fig,axes=plt.subplots(nrows,columns,figsize=(7.1,7.1*nrows/columns),squeeze=False)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    contour_rows=[]
    for ax,(row,coord,window,grid,fov) in zip(axes.flat,selected):
        rgb=rgb_cutout(images,bands,display,coord,fov)
        ax.imshow(rgb,origin='lower',interpolation='nearest')
        ra,dec=arcsec_to_radec(grid['x'],grid['y'],state.reference)
        px,py=reference.wcs.world_to_pixel_values(ra,dec)
        contour=ax.contour(px-window.x0,py-window.y0,grid['density'],levels=grid['levels'],
                          colors=['#56B4E9'],linewidths=[.65,.85],linestyles=['--','-'])
        segments=[seg for level in contour.allsegs for seg in level]
        clipped=any(np.any((seg[:,0]<0)|(seg[:,0]>rgb.shape[1]-1)|(seg[:,1]<0)|(seg[:,1]>rgb.shape[0]-1)) for seg in segments)
        contour_rows.append(dict(image_label=row.image_label,probabilities=[.682689492,.954499736],
            recovered_draws=int(row.recovered_draws),total_draws=len(indices),fov_arcsec=fov,contour_clipped=bool(clipped),
            kde_integrated_mass=grid['integrated_mass'],bandwidth_factor=grid['bandwidth_factor']))
        text=f'{row.image_label}'+ '\n'+rf'$z={row.z_source:.3f}$'+ '\n'+rf'$|\lambda_a|_{{50}}={row.abs_lambda_a_median:.4f}$'
        ax.text(.03,.97,text,transform=ax.transAxes,ha='left',va='top',fontsize=7,color='white',
                bbox=dict(facecolor='black',alpha=.5,edgecolor='none',pad=2))
        size=rgb.shape[0];bar=.5/reference.pixel_scale_arcsec
        ax.plot([.06*size,.06*size+bar],[.07*size]*2,color='white',lw=1)
        ax.text(.06*size+bar/2,.10*size,r'$0.5^{\prime\prime}$',color='white',ha='center',fontsize=7)
        ax.set(xlim=(-.5,rgb.shape[1]-.5),ylim=(-.5,rgb.shape[0]-.5),xticks=[],yticks=[]);ax.grid(False)
    for ax in axes.flat[len(selected):]:ax.set_visible(False)
    output=paths.figures/f'{stem}.pdf';fig.savefig(output,dpi=350,bbox_inches='tight',pad_inches=0);plt.close(fig)
    rows.to_csv(paths.tables/f'{stem}.csv',index=False)
    np.savez_compressed(paths.tables/f'{stem}.npz',lambda_a=values,draw_indices=indices,image_labels=labels,
                        abs_lambda_a_quantiles=quantiles,observed_xy=xy)
    info=dict(run=str(paths.figures.parent),image_dir=config.image_catalog.image_dir,bands=list(bands),minimum_fov_arcsec=minimum_fov,
        ranking='Ascending median absolute eigenvalue nearest zero, at catalog positions and family redshifts; ties by image label',
        model='exact parametric + conditional-mean GRF, smooth plotting derivatives',grf_version=PLOT_FIELD_VERSION,
        jacobian_max_asymmetry=max(asymmetries),singular_value_check='abs(lambda_a) checked against smallest singular value for every image/draw',
        position_contours='68.2689492% (solid) and 95.4499736% (dashed) KDE highest-density regions, conditional on successful local image recovery; no covariance ellipses',
        draw_indices=indices.tolist(),posterior_cache_fingerprint=fingerprint,selected=contour_rows)
    (paths.tables/f'{stem}.json').write_text(json.dumps(info,indent=2)+'\n')
    log(f'[outputs] small-lambda JWST cutouts: {output}')
    return output
