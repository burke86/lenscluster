"""Photographic RGB views of fitted source light on a common source-plane grid.

Four-band color recipe is shared with the JWST cutouts. Source fits remain
independent in each filter, at the same fixed lens draw. No color-based shifts,
per-filter renormalization, or sharpening are applied.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lenscluster.rgb import (CalibratedRGBDisplayConfig,make_calibrated_rgb,
    DEFAULT_JWST_RGB_BANDS,DEFAULT_JWST_RGB_CHANNEL_WEIGHTS,
    DEFAULT_JWST_RGB_CHANNEL_GAINS,DEFAULT_JWST_RGB_Q,DEFAULT_JWST_RGB_STRETCH)
from lenscluster.cutouts import _jwst_display_fields
from lenscluster.plot_style import PAPER_STYLE


def display_config(exposure=1.):
    fields=_jwst_display_fields(.02)
    fields.update(unsharp_amount=0.,unsharp_sigma_pixels=0.)
    return CalibratedRGBDisplayConfig(q=DEFAULT_JWST_RGB_Q,
        stretch=DEFAULT_JWST_RGB_STRETCH/exposure,
        channel_gains=DEFAULT_JWST_RGB_CHANNEL_GAINS,
        channel_weights=DEFAULT_JWST_RGB_CHANNEL_WEIGHTS,
        # All input maps are already calibrated to surface brightness.
        band_fluxscales={b:.02**2 for b in DEFAULT_JWST_RGB_BANDS},
        minimum=0.,local_background=False,channel_floor_sigma=0.,**fields)


def load_sources(directories):
    images={};metadata={};header=None
    for band,directory in directories.items():
        metadata[band]=json.loads((directory/'summary.json').read_text())
        data,h=fits.getdata(directory/'best_source.fits',header=True)
        if h.get('BUNIT','').lower()!='njy/arcsec2':
            raise ValueError(f'{band}: expected calibrated nJy/arcsec2')
        if not metadata[band]['results'][0]['success']:
            raise ValueError(f'{band}: source fit did not converge')
        if header is not None:
            for key in ['CRVAL1','CRVAL2','CRPIX1','CRPIX2','CDELT1','CDELT2','NAXIS1','NAXIS2']:
                np.testing.assert_allclose(h[key],header[key],rtol=0,atol=1e-12,err_msg=f'{band}: unequal source grids')
            ref=metadata['F115W']
            for key in ['run','redshift','families']:
                if metadata[band][key]!=ref[key]:raise ValueError('Band fits use different source/lens models')
            if metadata[band]['results'][0]['draw']!=ref['results'][0]['draw']:raise ValueError('Unequal lens draws')
            for key in ['gp_displacement','parametric_only']:
                if metadata[band]['arguments'][key]!=ref['arguments'][key]:raise ValueError('Unequal fixed lens components')
        header=h;images[band]=np.asarray(data,float)
    return images,header,metadata


def observed_rgb(directories,display):
    reference=directories['F150W']/'input.fits'
    with fits.open(reference) as hdus:
        header=hdus[0].header;shape=hdus[0].data.shape;mask=hdus['FIT_MASK'].data.astype(bool)
    yy,xx=np.mgrid[:shape[0],:shape[1]];ra,dec=WCS(header).pixel_to_world_values(xx,yy)
    images={}
    for band,directory in directories.items():
        d,h=fits.getdata(directory/'input.fits',header=True)
        x,y=WCS(h).world_to_pixel_values(ra,dec)
        meta=json.loads((directory/'summary.json').read_text())
        images[band]=map_coordinates(np.asarray(d,float),[y,x],order=1,mode='constant',cval=np.nan)/meta['pixel_scale_arcsec']**2
    rgb=make_calibrated_rgb(images,bands=DEFAULT_JWST_RGB_BANDS,display=display)
    ys,xs=np.nonzero(mask);pad=8
    return rgb[max(0,ys.min()-pad):ys.max()+pad+1,max(0,xs.min()-pad):xs.max()+pad+1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-root',type=Path,default=Path('/tmp/k12_source_rgb'))
    p.add_argument('--reference-root',type=Path,default=Path('/tmp/fold_source_validation'))
    p.add_argument('--output',type=Path,default=Path('/tmp/k12_source_rgb/figures'))
    p.add_argument('--exposure',type=float,default=1.,help='One common display exposure factor for all bands and views')
    args=p.parse_args()
    if args.exposure<=0:p.error('Exposure must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    display=display_config(args.exposure)
    modes=['physical','gp']; products={};provenance={}
    for mode in modes:
        directories={b:args.fit_root/mode/b for b in DEFAULT_JWST_RGB_BANDS}
        directories['F150W']=args.reference_root/mode
        images,header,meta=load_sources(directories)
        rgb=make_calibrated_rgb(images,bands=DEFAULT_JWST_RGB_BANDS,display=display)
        products[mode]=(rgb,images,header,directories)
        provenance[mode]={b:dict(directory=str(directories[b].resolve()),draw=meta[b]['results'][0]['draw'],chi2_per_pixel=meta[b]['results'][0]['chi2_per_pixel']) for b in images}
        # The calibrated input maps, not just a stretched display, are retained.
        hdus=[fits.PrimaryHDU()]
        for b,a in images.items():hdus.append(fits.ImageHDU(a,header=header,name=b))
        rgb_header=header.copy();rgb_header['BUNIT']='display';rgb_header['CTYPE3']='RGB'
        hdus.append(fits.ImageHDU(np.moveaxis(rgb,-1,0),header=rgb_header,name='RGB'))
        fits.HDUList(hdus).writeto(args.output/f'k12_source_{mode}_rgb.fits',overwrite=True)
    def portrait(ax,rgb,header,label):
        fovx=header['CDELT1']*(rgb.shape[1]-1);fovy=header['CDELT2']*(rgb.shape[0]-1)
        ax.imshow(rgb,origin='lower',extent=(-fovx/2,fovx/2,-fovy/2,fovy/2),interpolation='bilinear')
        ax.set_axis_off()
        ax.text(.045,.95,label,transform=ax.transAxes,color='white',va='top',fontsize=10)
        x=-fovx/2+.055*fovx;y=-fovy/2+.07*fovy
        ax.plot([x,x+.05],[y,y],color='white',lw=1.5,solid_capstyle='butt')
        ax.text(x+.025,y+.013*fovy,'0.05″',color='white',ha='center',va='bottom',fontsize=9)
    with plt.rc_context(fname=ROOT/'src/lenscluster/style.mplstyle',rc=PAPER_STYLE):
        for mode,(rgb,images,header,directories) in products.items():
            fig,ax=plt.subplots(figsize=(5.4,5.4));fig.subplots_adjust(0,0,1,1)
            portrait(ax,rgb,header,'K12 · reconstructed source')
            fig.savefig(args.output/f'k12_source_{mode}_rgb.pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.55));fig.subplots_adjust(0,0,1,1,wspace=.015)
        for ax,mode,label in zip(axes,modes,['Parametric + GRF','With fixed GP correction']):
            rgb,_,header,_=products[mode];portrait(ax,rgb,header,label)
        fig.savefig(args.output/'k12_source_rgb_comparison.pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
        # Four actual source bands, with a shared intensity mapping.
        rgb,images,header,directories=products['physical']
        vmax=max(np.percentile(a,99.5) for a in images.values());norm=AsinhNorm(linear_width=.02*vmax,vmin=0,vmax=vmax)
        fig,axes=plt.subplots(1,4,figsize=(10,2.5));fig.subplots_adjust(0,0,1,1,wspace=.02)
        for ax,(band,a) in zip(axes,images.items()):
            ax.imshow(a,origin='lower',cmap='gray',norm=norm,interpolation='bilinear');ax.set_axis_off()
            ax.text(.04,.95,band,transform=ax.transAxes,color='white',va='top',fontsize=9)
        fig.savefig(args.output/'k12_source_four_bands.pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
        # Show the observed fold and its inferred source, each with its own scale.
        observed=observed_rgb(directories,display)
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.55));fig.subplots_adjust(0,0,1,1,wspace=.025)
        axes[0].imshow(observed,origin='lower',interpolation='bilinear');axes[0].set_axis_off()
        axes[0].text(.045,.95,'JWST · K12 fold',transform=axes[0].transAxes,color='white',va='top',fontsize=10)
        axes[0].plot([8,33],[8,8],color='white',lw=1.5);axes[0].text(20.5,10,'0.5″',color='white',ha='center',va='bottom',fontsize=9)
        portrait(axes[1],rgb,header,'Reconstructed source')
        fig.savefig(args.output/'k12_fold_and_source_rgb.pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
    record=dict(bands=list(DEFAULT_JWST_RGB_BANDS),channel_weights=DEFAULT_JWST_RGB_CHANNEL_WEIGHTS,
        exposure=args.exposure,source_units='nJy/arcsec2',reference_pixel_arcsec=.02,
        sharpening=False,per_band_renormalization=False,band_registration='Same absolute beta grid, no centroid shifts',
        note='Model-dependent source reconstructions, not direct photographs. Residual knot mismatch remains. F444W has a different intrinsic resolving power; no claim of PSF-matched color measurement.',fits=provenance)
    (args.output/'provenance.json').write_text(json.dumps(record,indent=2))
    print(args.output.resolve())


if __name__=='__main__':main()
