"""Publication fold cutouts from cached geometry, also used by normal run outputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from .fold_diagnostics import _contours, critical_curve_quantiles, CRITICAL_CURVE_STATISTIC
from .cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
from .plot_style import PAPER_STYLE, CRITICAL_COLOR


def render_apj_fold_cutouts(config, data, imaging, output, *, max_separation=3.,
                            critical_color=CRITICAL_COLOR, run_directory='', source_cache=''):
    images,bands,display,_=imaging
    pairs=[p for p in data['pairs'] if p['separation']<max_separation]
    systems=[s for s in data['systems'] if any(set(p['labels']).issubset(s['labels']) for p in pairs)]
    if not systems: return None
    # Keep all matched knots in each selected system/image pair.
    systems.sort(key=lambda s:(int(''.join(filter(str.isdigit,s['system_id'])) or 0),s['image_ids']))
    reference=images[rgb_reference_band(bands)]
    ref=config.model.reference;frame=(ref.reference,ref.ra0_deg,ref.dec0_deg)
    plt.style.use(Path(__file__).with_name('style.mplstyle'))
    plt.rcParams.update(PAPER_STYLE)
    columns=min(3,len(systems));rows=int(np.ceil(len(systems)/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(7.1,7.1*rows/columns),squeeze=False)
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    palette=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#F0E442','#7B61A8']
    curve_color=critical_color;shared=False
    for ax,s in zip(axes.flat,systems):
        centre=arcsec_to_skycoord(*s['center'],frame)
        window=cutout_window(reference,centre,s['fov'])
        if window is None: raise ValueError(f'No imaging coverage for {s["system_id"]}')
        def pixels(x,y):
            ra,dec=arcsec_to_radec(x,y,frame)
            px,py=reference.wcs.world_to_pixel_values(ra,dec)
            return px-window.x0,py-window.y0
        rgb=rgb_cutout(images,bands,display,centre,s['fov'])
        ax.imshow(rgb,origin='lower',interpolation='nearest')
        if 'critical_grf_best' in s:
            shared=True
            band_sets=[(s.get('critical_grf_local_draws',s['critical_grf_draws']),critical_color)]
        else:
            band_sets=[(s['critical'],curve_color)]
        for curves,band_color in band_sets:
            for low,median,high in critical_curve_quantiles(curves,s['seeds'],s['origin'],s['normal']):
                polygon=np.vstack((low,high[::-1]))
                ax.add_patch(Polygon(np.column_stack(pixels(*polygon.T)),facecolor=band_color,alpha=.18,edgecolor='none'))
                ax.plot(*pixels(*median.T),color=band_color,lw=.7)
        colors=[palette[k%len(palette)] for k in range(len(s['family_ids']))]
        label_points=[]
        for j,label in enumerate(s['labels']):
            color=colors[s['image_family_index'][j]]
            _contours(ax,s['positions'][:,j],color,pixels,outline=False)
            cross,=ax.plot(*pixels(*s['observed'][j]),'x',color=color,ms=6,mew=.8,zorder=8)
            label_points.append((label,pixels(*s['observed'][j]),color))
        panel_label = f"System {s['system_id']}"
        ax.text(.03,.97,panel_label+'\n'+rf"$z={s['z_source']:g}$",
            transform=ax.transAxes,ha='left',va='top',fontsize=8,color='white',
            bbox=dict(facecolor='black',alpha=.55,edgecolor='none',pad=2))
        size=rgb.shape[0];bar=.5/reference.pixel_scale_arcsec
        ax.plot([.06*size,.06*size+bar],[.07*size]*2,color='white',lw=1)
        ax.text(.06*size+bar/2,.10*size,r'$0.5^{\prime\prime}$',ha='center',color='white',fontsize=7)
        ax.set(xlim=(-.5,rgb.shape[1]-.5),ylim=(-.5,size-.5),xticks=[],yticks=[])
        ax.grid(False)
        label_catalog_images(ax,label_points)
    for ax in list(axes.flat)[len(systems):]:ax.set_visible(False)
    output=Path(output)
    if output.suffix.lower() != '.pdf':
        raise ValueError('Figure output must be PDF')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=400,bbox_inches='tight',pad_inches=0);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(run=str(run_directory),
        source_cache=str(source_cache),
        strict_max_separation_arcsec=max_separation,
        selected_pairs=[dict(labels=p['labels'],separation_arcsec=p['separation']) for p in pairs],
        systems=[s['system_id'] for s in systems],gp_blocks=config.likelihood.gp_blocks,
        curves='parametric + GRF' if shared else 'parametric',
        curve_statistic=CRITICAL_CURVE_STATISTIC,
        uncertainty_bands='parametric + GRF' if shared else 'parametric',
        critical_color=critical_color if shared else curve_color,
        grf_scope='system-specific' if config.likelihood.gp_blocks=='system' else 'shared',
        draw_indices=data['draw_indices']),indent=2)+'\n')
    return output


def label_catalog_images(ax, label_points):
    """Place catalog IDs beside their matching crosses, avoiding labels and markers."""
    fig=ax.figure
    # Place full catalog IDs close to their crosses without covering other labels.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    occupied=[text.get_window_extent(renderer).expanded(1.05,1.1) for text in ax.texts]
    markers=[Bbox.from_bounds(*(ax.transData.transform(point)-4),8,8)
             for _,point,_ in label_points]
    for label,point,color in label_points:
        text=ax.annotate(label,point,xytext=(5,5),textcoords='offset points',
            fontsize=6,color=color,zorder=9,annotation_clip=True)
        candidates=[]
        for radius in (5,10,16):
            for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)):
                text.set_position((radius*dx,radius*dy))
                text.set_ha('left' if dx>0 else 'right' if dx<0 else 'center')
                text.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')
                box=text.get_window_extent(renderer).expanded(1.05,1.1)
                overlaps=sum(box.overlaps(other) for other in occupied+markers)
                inside=ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)
                candidates.append((1000*(not inside)+100*overlaps+radius,
                                   text.get_position(),text.get_ha(),text.get_va()))
        _,position,ha,va=min(candidates,key=lambda candidate:candidate[0])
        text.set_position(position);text.set_ha(ha);text.set_va(va)
        occupied.append(text.get_window_extent(renderer).expanded(1.05,1.1))
