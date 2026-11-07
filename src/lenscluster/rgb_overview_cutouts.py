"""Selected MACS J0416 posterior cutouts for the RGB critical overview."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
from .fold_diagnostics import _credible_density_grid, load_fold_data, critical_curve_quantiles
from .plot_style import CRITICAL_COLOR

IMAGE_LABELS = ('K34.b', 'K34.a', 'K33.e', 'K77.b')
SYSTEM_IDS = ('K1', 'K6', 'K8', 'K12')


def overview_cutout_data(tables, prefix, *, log=print):
    """Reuse saved image draws and fold geometry, never run an image search."""
    tables=Path(tables)
    required=[tables/'posterior_overview_positions.npz',
              tables/f'{prefix}_small_lambda_a_image_posteriors_jwst.csv',
              tables/'fold_pairs_source_caustics.npz',tables/'fold_pairs_source_caustics.json',
              tables/f'{prefix}_small_lambda_a_image_posteriors_jwst_critical.npz']
    if not all(p.exists() for p in required):
        log('[outputs] RGB inset panels unavailable: generate posterior-position and fold diagnostics first')
        return None
    ranking=pd.read_csv(required[1]).set_index('image_label')
    with np.load(required[4]) as saved:
        critical={key:saved[key].copy() for key in ('centers','fovs','redshifts','median_determinant','lower','upper')}
    with np.load(required[0]) as data:
        labels=data['image_labels'].astype(str).tolist()
        positions=data['recovered'].copy()
    folds=load_fold_data(tables)
    systems=[]
    for identity in SYSTEM_IDS:
        matches=[s for s in folds['systems'] if s['system_id']==identity]
        if len(matches)!=1:
            log(f'[outputs] RGB inset panels unavailable: expected one fold system {identity}, found {len(matches)}')
            return None
        systems.append(matches[0])
    if any(label not in labels or label not in ranking.index for label in IMAGE_LABELS):
        log('[outputs] RGB inset panels unavailable: selected image IDs absent from catalog')
        return None
    images=[]
    for label in IMAGE_LABELS:
        row=ranking.loc[label]
        grid=_credible_density_grid(positions[:,labels.index(label)])
        if grid is None:
            log(f'[outputs] RGB inset panels unavailable: no resolved posterior for {label}')
            return None
        # Derive framing even when a selected image lies beyond the ranking plot's limit.
        import contourpy
        curves=contourpy.contour_generator(x=grid['x'],y=grid['y'],z=grid['density']).lines(grid['levels'][0])
        if not curves:
            return None
        center=np.array([row.x_arcsec,row.y_arcsec])
        radius=np.max(np.abs(np.concatenate(curves)-center))
        fov=float(.5*np.ceil(max(3.,2.3*radius)/.5))
        matches=np.flatnonzero(np.all(np.isclose(critical['centers'],center,rtol=0,atol=1e-10),axis=1)
                               &np.isclose(critical['redshifts'],row.z_source,rtol=0,atol=1e-10)
                               &np.isclose(critical['fovs'],fov,rtol=0,atol=1e-10))
        if len(matches)!=1:
            raise ValueError(f'Regenerate small-lambda cutout critical fields for {label}: cached geometry does not match')
        k=matches[0]
        images.append(dict(label=label,center=center,fov=fov,z=row.z_source,grid=grid,
                           critical={key:critical[key][k] for key in ('median_determinant','lower','upper')}))
    return dict(images=images,systems=systems)


def render_overview_cutouts(fig, data, imaging, frame):
    """Four panels left-to-right above; four folds top-to-bottom at right."""
    images,bands,display,_=imaging
    reference=images[rgb_reference_band(bands)]
    def panel(rect,center,fov,label,z):
        ax=fig.add_axes(rect)
        sky=arcsec_to_skycoord(*center,frame)
        window=cutout_window(reference,sky,fov)
        if window is None:
            raise ValueError(f'No imaging coverage for RGB inset {label}')
        rgb=rgb_cutout(images,bands,display,sky,fov)
        ax.imshow(rgb,origin='lower',interpolation='nearest')
        def pixels(x,y):
            ra,dec=arcsec_to_radec(x,y,frame)
            px,py=reference.wcs.world_to_pixel_values(ra,dec)
            return px-window.x0,py-window.y0
        ax.text(.03,.97,label+'\n'+rf'$z={z:g}$',transform=ax.transAxes,
                va='top',color='white',fontsize=7,
                bbox=dict(facecolor='black',alpha=.5,edgecolor='none',pad=2))
        size=rgb.shape[0];bar=.5/reference.pixel_scale_arcsec
        ax.plot([.06*size,.06*size+bar],[.07*size]*2,color='white',lw=.8)
        ax.text(.06*size+bar/2,.10*size,r'$0.5^{\prime\prime}$',ha='center',color='white',fontsize=6)
        ax.set(xlim=(-.5,rgb.shape[1]-.5),ylim=(-.5,size-.5),xticks=[],yticks=[])
        ax.grid(False)
        return ax,pixels
    def contour(ax,grid,pixels,color):
        if grid is not None:
            ax.contour(*pixels(grid['x'],grid['y']),grid['density'],
                       levels=[grid['levels'][0]],colors=[color],linewidths=.65,linestyles='-')
    for i,item in enumerate(data['images']):
        ax,pixels=panel([.07+i*.18,.80,.18,.18],item['center'],item['fov'],item['label'],item['z'])
        from .jwst_rgb_figure import draw_critical_interval
        critical=item['critical'];det=critical['median_determinant']
        axis=np.linspace(-.5,.5,det.shape[0])
        x,y=np.meshgrid(item['center'][0]+axis*item['fov'],item['center'][1]+axis*item['fov'])
        coordinates=pixels(x,y)
        draw_critical_interval(ax,critical['lower'],critical['upper'],CRITICAL_COLOR,coordinates=coordinates)
        if det.min()<0<det.max():
            ax.contour(*coordinates,det,levels=[0],colors=[CRITICAL_COLOR],linewidths=.65)
        contour(ax,item['grid'],pixels,'tab:orange')
    for i,s in enumerate(data['systems']):
        ax,pixels=panel([.80,.61-i*.18,.18,.18],s['center'],s['fov'],f"System {s['system_id']}",s['z_source'])
        curves=s.get('critical_grf_local_draws',s.get('critical_grf_draws',s['critical']))
        for low,median,high in critical_curve_quantiles(curves,s['seeds'],s['origin'],s['normal']):
            vertices=np.vstack((low,high[::-1]))
            ax.add_patch(Polygon(np.column_stack(pixels(*vertices.T)),facecolor=CRITICAL_COLOR,alpha=.18,edgecolor='none'))
            ax.plot(*pixels(*median.T),color=CRITICAL_COLOR,lw=.65)
        for j,label in enumerate(s['labels']):
            color='tab:orange'
            contour(ax,_credible_density_grid(s['positions'][:,j]),pixels,color)
            ax.plot(*pixels(*s['observed'][j]),'x',color=color,ms=3,mew=.5)
    return dict(top=list(IMAGE_LABELS),right=list(SYSTEM_IDS),
                position_probability=.954499736,position_contours='2 sigma only; conditional on recovery')
