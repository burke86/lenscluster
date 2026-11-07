"""RGB image-plane diagnostics for selected systems, using cached posterior positions."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from _covariance_plot_cli import load_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--image-dir',default='data/imaging')
    parser.add_argument('--systems',nargs='+',default=['D36','D24','D19','D60','D37','D35','D1C','D9B'])
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from lenscluster.cutouts import (load_cluster_imaging,arcsec_to_skycoord,arcsec_to_radec,
                                    rgb_cutout,cutout_window,rgb_reference_band)
    from lenscluster.rgb import DEFAULT_JWST_RGB_BANDS
    from lenscluster.config import RGBDisplayConfig
    from lenscluster.image_catalog import load_image_catalog
    from lenscluster.fold_diagnostics import _contours
    from lenscluster.plot_style import CRITICAL_COLOR
    from lenscluster.plot_style import PAPER_STYLE
    config,state,e,post,paths=load_run(args.run_directory)
    best=int(np.nanargmax(post.log_prob));theta=post.samples[best]
    physical=e.physical(theta);correction=e.gp_deflection_correction(theta)
    cat=load_image_catalog(config.model.image_constraints.catalog_path,
        sigma_inst_arcsec=config.model.image_constraints.sigma_inst_arcsec,
        sigma_floor_arcsec=config.model.image_constraints.sigma_floor_arcsec).set_index('image_label')
    cache=np.load(paths.tables/'posterior_overview_positions.npz')
    labels=list(map(str,cache['image_labels']));positions=cache['recovered']
    if best not in cache['draw_indices']:raise ValueError('Position cache does not contain the current best draw')
    metadata=json.loads((paths.tables/'posterior_overview_positions.json').read_text())
    if metadata['posterior_draws_available'] != len(post.samples):
        raise ValueError('Posterior position cache has a different draw count; regenerate it first')
    np.testing.assert_array_equal(cache['image_labels'],np.asarray(state.images.image_labels))
    imaging_config=replace(config.image_catalog,image_dir=args.image_dir,image_scale='auto',
        bands=tuple(DEFAULT_JWST_RGB_BANDS),rgb=RGBDisplayConfig(),figure_suffix='_jwst')
    images,bands,display,_=load_cluster_imaging(imaging_config,config.model,config.paths.run_name)
    reference=images[rgb_reference_band(bands)]
    fits=pd.read_csv(paths.tables/'image_fit_quality.csv').set_index('image_label')
    family_index={str(f):i for i,f in enumerate(state.images.family_ids)}
    palette=['#56B4E9','#E69F00','#009E73','#CC79A7','#D55E00']
    output=paths.figures/f'{config.paths.figure_prefix}_selected_system_cutouts_jwst.pdf'
    report=[]
    with plt.rc_context(fname=Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle',rc=PAPER_STYLE),PdfPages(output) as pdf:
        for requested in args.systems:
            rows=cat[cat.system_id.astype(str).eq(requested)]
            if rows.empty:rows=cat[cat.family_id.astype(str).eq(requested)]
            rows=rows.loc[rows.index.intersection(labels)]
            if rows.empty:raise ValueError(f'No saved images for {requested}')
            ids=sorted(rows.image_id.astype(str).unique());families=sorted(rows.family_id.astype(str).unique())
            colors={f:palette[i%len(palette)] for i,f in enumerate(families)}
            fig,axes=plt.subplots(1,len(ids),figsize=(4.1*len(ids),4.8),squeeze=False)
            fig.subplots_adjust(left=.02,right=.985,bottom=.19,top=.98,wspace=.055)
            for ax,identity in zip(axes[0],ids):
                group=rows[rows.image_id.astype(str).eq(identity)]
                observed=fits.loc[group.index,['x_obs_arcsec','y_obs_arcsec']].to_numpy()
                center=(observed.min(0)+observed.max(0))/2
                size=max(6.,float(np.ptp(observed,axis=0).max())+4.)
                coord=arcsec_to_skycoord(*center,state.reference)
                window=cutout_window(reference,coord,size)
                if window is None:raise ValueError(f'No JWST coverage: {requested}.{identity}')
                def pixels(x,y):
                    ra,dec=arcsec_to_radec(x,y,state.reference)
                    px,py=reference.wcs.world_to_pixel_values(ra,dec)
                    return px-window.x0,py-window.y0
                rgb=rgb_cutout(images,bands,display,coord,size)
                ax.imshow(rgb,origin='lower',interpolation='nearest')
                z=float(fits.loc[group.index[0],'z_source']);fi=family_index[str(group.family_id.iloc[0])]
                grid=np.linspace(-size*.53,size*.53,170)
                xx,yy=np.meshgrid(grid+center[0],grid+center[1]);q=np.c_[xx.ravel(),yy.ravel()]
                packed=e.packed_state_for_z(physical,z)
                _,h=e.total_fields_at(q,packed,chunk=1024);h=np.asarray(h)
                det=np.linalg.det(np.eye(2)-h).reshape(xx.shape)
                full=h.copy()
                if correction is not None:
                    for start in range(0,len(q),384):
                        part=q[start:start+384];n=len(part);p=np.pad(part,((0,384-n),(0,0)),mode='edge')
                        c=correction(p[:,0],p[:,1],np.full(384,fi))
                        full[start:start+n]+=np.stack([c[2],c[3],c[4],c[5]],axis=-1)[:n].reshape(n,2,2)
                corrected=np.linalg.det(np.eye(2)-full).reshape(xx.shape)
                gx,gy=pixels(xx,yy)
                for field,color,ls in [(det,CRITICAL_COLOR,'--'),(corrected,CRITICAL_COLOR,'-')]:
                    if field.min()<0<field.max():ax.contour(gx,gy,field,levels=[0],colors=[color],linewidths=.75,linestyles=ls)
                info=[]
                for label,row in group.iterrows():
                    color=colors[str(row.family_id)];index=labels.index(str(label));draws=positions[:,index]
                    valid=np.isfinite(draws).all(1);f=fits.loc[label]
                    drawn=_contours(ax,draws,color,pixels)
                    ax.plot(*pixels(f.x_obs_arcsec,f.y_obs_arcsec),'x',ms=9,mew=.8,color=color)
                    model=f[['x_model_arcsec','y_model_arcsec']].to_numpy(float)
                    if np.isfinite(model).all():ax.plot(*pixels(*model),'o',ms=7,mew=.65,mfc='none',mec=color)
                    status=f'{float(f.image_residual_arcsec):.2f}″' if np.isfinite(f.image_residual_arcsec) else 'not recovered'
                    info.append(f'{label}: {status}; recovered {valid.sum()}/{len(valid)}')
                    report.append(dict(system=requested,image=str(label),best_residual_arcsec=float(f.image_residual_arcsec),
                        valid_draws=int(valid.sum()),total_draws=len(valid),contours_drawn=bool(drawn)))
                ax.text(.035,.97,f'{requested}.{identity}   z = {z:g}',transform=ax.transAxes,va='top',color='white',fontsize=10,
                    bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=3))
                ax.text(.035,.035,'\n'.join(info),transform=ax.transAxes,va='bottom',color='white',fontsize=7,
                    bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=3))
                bar=1/reference.pixel_scale_arcsec;n=rgb.shape[0]
                ax.plot([.07*n,.07*n+bar],[.19*n]*2,color='white',lw=1.3)
                ax.text(.07*n+bar/2,.22*n,'1″',color='white',ha='center',fontsize=9)
                ax.set(xlim=(-.5,n-.5),ylim=(-.5,n-.5));ax.set_xticks([]);ax.set_yticks([])
                print(f'{requested}.{identity}: rendered',flush=True)
            handles=[Line2D([],[],color=colors[f],lw=1.5,label=f) for f in families]
            handles += [Line2D([],[],color='black',marker='x',ls='',label='Catalog position'),
                Line2D([],[],color='black',marker='o',mfc='none',ls='',label='Best-model image'),
                Line2D([],[],color='black',lw=.6,label='68.3% position region'),
                Line2D([],[],color='black',lw=.5,ls='--',label='95.4% position region'),
                Line2D([],[],color=CRITICAL_COLOR,lw=.8,ls='--',label='Parametric critical curve'),
                Line2D([],[],color=CRITICAL_COLOR,lw=.8,label='Parametric + GRF critical curve')]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,fontsize=8,frameon=False)
            fig.text(.5,.015,'Position contours are conditional on successful local image recovery; curves use the best saved draw.',ha='center',fontsize=7)
            pdf.savefig(fig,bbox_inches=None);plt.close(fig)
    (paths.tables/'selected_system_cutouts.json').write_text(json.dumps(dict(run=args.run_directory,best_draw=best,
        systems=args.systems,posterior_position_draws=len(positions),images=report),indent=2))
    print(output,flush=True)


if __name__=='__main__':main()
