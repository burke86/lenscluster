"""Reconstruct a shared pixelized or Gaussian source from JWST fold pixels.

See scripts/fit_fold_source_light.md. No lens optimization or new lens sampling.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import gzip
import shutil
import os
from pathlib import Path
import time

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba-lenscluster')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-lenscluster')
from _covariance_plot_cli import load_run, ROOT
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jaxtronomy.LightModel.Profiles.gaussian import Gaussian
from scipy.optimize import least_squares, minimize
from types import SimpleNamespace
from jax.scipy.ndimage import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS


def matching_product(science, product):
    import re
    stem=re.sub(r'_(?:bgsub-)?sci\.fits(?:\.gz)?$',f'_{product}.fits',str(science))
    if stem==str(science): return None
    return next((Path(q) for q in (stem,stem+'.gz') if Path(q).exists()),None)


def gaussian_psf(fwhm_pixels):
    sigma = fwhm_pixels / np.sqrt(8*np.log(2))
    radius = max(2, int(np.ceil(5*sigma)))
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return kernel/kernel.sum()


def make_renderer(shape, supersample, n_knots):
    """Parameters per knot: log integrated amplitude, log sigma, beta_x, beta_y.

    Brightness is in native FITS units; averaging subpixels preserves surface
    brightness. The PSF is an effective kernel at the native mosaic pixel scale.
    """
    ny, nx = shape
    def source(p, beta):
        result = jnp.zeros(beta.shape[:-1])
        for k in range(n_knots):
            a, s, x, y = p[4*k:4*k+4]
            result += Gaussian.function(beta[..., 0], beta[..., 1], jnp.exp(a), jnp.exp(s), x, y)
        return result
    @jax.jit
    def render(p, beta, psf):
        fine = source(p, beta).reshape(ny, supersample, nx, supersample)
        image = fine.mean(axis=(1, 3))
        return convolve(image, psf, mode='same', method='fft') + p[-1]
    return render, jax.jit(source)


def fit_pixels(render, beta, psf, data, noise, mask, p0, bounds, max_nfev):
    ids = np.flatnonzero(mask)
    target = jnp.asarray(data.ravel()[ids]); err = jnp.asarray(noise.ravel()[ids])
    @jax.jit
    def residual(p):
        return (render(p, beta, psf).ravel()[ids]-target)/err
    jacobian = jax.jit(jax.jacfwd(residual))
    result = least_squares(lambda p: np.asarray(residual(p)), p0,
                           jac=lambda p: np.asarray(jacobian(p)), bounds=bounds,
                           x_scale='jac', max_nfev=max_nfev)
    return result


def uncompressed_fits(path):
    path=Path(path)
    if path.suffix != '.gz': return path
    token=f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
    root=Path('/tmp/fold_source_light_cache');root.mkdir(parents=True,exist_ok=True)
    target=root/(hashlib.sha256(token.encode()).hexdigest()+'.fits')
    if not target.exists():
        temporary=target.with_suffix(f'.{os.getpid()}.part')
        print(f'Decompressing once for fast FITS access: {path.name}',flush=True)
        try:
            with gzip.open(path,'rb') as src,temporary.open('wb') as dest:
                shutil.copyfileobj(src,dest,16*1024*1024)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def inverse_gp_coordinates(points,observed,weights,sigma,ell,chunk=512):
    """Invert theta_observed = theta_grav + E[d(theta_grav)|catalog].

    This is a positional nuisance warp, NOT a gravitational deflection. Newton
    inversion and positive warp determinants are checked at every query pixel.
    """
    observed=jnp.asarray(observed);weights=jnp.asarray(weights)
    @jax.jit
    def solve(target):
        def terms(q):
            offset=q[:,None,:]-observed[None,:,:]
            kernel=sigma**2*jnp.exp(-jnp.sum(offset**2,axis=-1)/(2*ell**2))
            mean=kernel@weights
            derivative=-jnp.einsum('qn,na,qnb->qab',kernel,weights,offset)/ell**2
            return mean,jnp.eye(2)+derivative
        def condition(state):
            _,error,iterations=state
            return (error>1e-10)&(iterations<12)
        def step(state):
            q,_,iterations=state
            mean,jac=terms(q);residual=q+mean-target
            delta=jnp.linalg.solve(jac,residual[...,None])[...,0]
            return q-jnp.clip(delta,-.2,.2),jnp.max(jnp.linalg.norm(residual,axis=-1)),iterations+1
        q,_,iterations=jax.lax.while_loop(condition,step,(target,jnp.inf,0))
        mean,jac=terms(q)
        return q,jnp.linalg.norm(q+mean-target,axis=-1),jnp.linalg.det(jac)
    output=np.empty_like(points);errors=[];dets=[]
    for start in range(0,len(points),chunk):
        block=points[start:start+chunk];n=len(block)
        padded=np.pad(block,((0,chunk-n),(0,0)),mode='edge')
        q,err,det=map(np.asarray,solve(jnp.asarray(padded)))
        output[start:start+n]=q[:n];errors.append(err[:n]);dets.append(det[:n])
    error=np.concatenate(errors);det=np.concatenate(dets)
    if not np.isfinite(output).all() or not np.isfinite(det).all() or error.max()>1e-7 or det.min()<=0:
        raise ValueError(f'GP warp inversion failed: max residual={error.max():.3g}, min determinant={det.min():.3g}')
    return output,dict(max_inverse_residual_arcsec=float(error.max()),minimum_warp_determinant=float(det.min()),maximum_shift_arcsec=float(np.linalg.norm(output-points,axis=1).max()))


def gp_coordinate_correction(evaluator,state,theta,points):
    from scipy.linalg import cho_solve
    e=evaluator._padded_evaluator()
    @jax.jit
    def inputs(theta):
        stats=e.statistics(theta);scales=e.error_scales(stats['physical'])
        residual=stats['fields']['backprojected']-stats['source_by_family'][jnp.asarray(state.images.family_index_per_image)]
        return stats['fields']['jacobian'],stats['family_cholesky'][0],residual,scales['sigma_shared'],scales['gp_ell']
    jac,chol,residual,sigma,ell=map(np.asarray,inputs(jnp.asarray(theta)))
    observed=np.c_[state.images.x_obs,state.images.y_obs]
    np.testing.assert_array_equal(np.asarray(e.gp['slots'])[0],np.arange(len(observed)))
    v=cho_solve((chol,True),residual.reshape(-1)).reshape(-1,2)
    weights=np.einsum('nca,nc->na',jac,v)
    return inverse_gp_coordinates(points,observed,weights,float(sigma),float(ell))


def make_pixel_renderer(shape, supersample, grid_shape, origin, spacing, beta=None):
    """JAX equivalent of lenstronomy INTERPOL: positive bilinear source pixels.

    A one-cell zero border matches INTERPOL. Coordinates here specify the first
    source pixel, instead of the center. A local background plane is separate.
    """
    ny,nx=shape; gy,gx=grid_shape; size=gy*gx
    by,bx=np.mgrid[-1:1:complex(ny),-1:1:complex(nx)]
    projection=None
    if beta is not None:
        from scipy.sparse import coo_matrix
        from jax.experimental.sparse import BCOO
        q=(np.asarray(beta)-np.asarray(origin))/spacing
        valid=np.flatnonzero((q[:,0]>-1)&(q[:,0]<gx)&(q[:,1]>-1)&(q[:,1]<gy))
        q=q[valid];index=np.floor(q).astype(int);frac=q-index
        pixels=(valid//(nx*supersample)//supersample)*nx+(valid%(nx*supersample)//supersample)
        rr=[];cc=[];ww=[]
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            cx=index[:,0]+dx;cy=index[:,1]+dy
            keep=(cx>=0)&(cx<gx)&(cy>=0)&(cy<gy)
            weight=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])/supersample**2
            rr.append(pixels[keep]);cc.append((cy*gx+cx)[keep]);ww.append(weight[keep])
        matrix=coo_matrix((np.concatenate(ww),(np.concatenate(rr),np.concatenate(cc))),shape=(ny*nx,size)).tocsr()
        matrix.eliminate_zeros()
        projection=BCOO.from_scipy_sparse(matrix)
    def source(p,beta):
        a=jnp.pad(p[:size].reshape(grid_shape),1)
        coords=jnp.stack([(beta[...,1]-origin[1])/spacing+1,(beta[...,0]-origin[0])/spacing+1])
        return map_coordinates(a,coords,order=1,mode='constant',cval=0.)
    @jax.jit
    def render(p,beta,psf):
        fine=(projection@p[:size]).reshape(ny,nx) if projection is not None else source(p,beta).reshape(ny,supersample,nx,supersample).mean((1,3))
        return convolve(fine,psf,mode='same',method='fft')+p[-3]+p[-2]*bx+p[-1]*by
    return render,jax.jit(source)


def fit_source_grid(render,beta,psf,data,noise,mask,p0,grid_shape,strength,maxiter):
    """Convex positive source inversion; no dense image-by-source Jacobian."""
    ids=np.flatnonzero(mask); target=jnp.asarray(data.ravel()[ids]);err=jnp.asarray(noise.ravel()[ids])
    scale=float(np.median(noise[mask])); size=int(np.prod(grid_shape))
    @jax.jit
    def objective(p):
        residual=(render(p,beta,psf).ravel()[ids]-target)/err
        src=p[:size].reshape(grid_shape)/scale
        curvature=jnp.sum(jnp.diff(src,n=2,axis=0)**2)+jnp.sum(jnp.diff(src,n=2,axis=1)**2)
        return .5*(jnp.sum(residual**2)+strength*curvature)
    # Scale coordinates by a positive curvature bound. J^T W J 1 measures
    # source sensitivity without forming the dense pixel-by-source Jacobian.
    @jax.jit
    def data_quadratic(p):
        m=render(p,beta,psf).ravel()[ids]/err
        return .5*jnp.sum(m*m)
    unit=jnp.r_[jnp.ones(size),jnp.zeros(3)]
    sensitivity=np.abs(np.asarray(jax.grad(data_quadratic)(unit)))
    sensitivity[:size] += 12*strength/scale**2
    for k in range(3):
        v=jnp.zeros(size+3).at[size+k].set(1.)
        sensitivity[size+k]=float(2*data_quadratic(v))
    coord_scale=jnp.asarray(1/np.sqrt(np.maximum(sensitivity,1.)))
    edge=np.zeros(grid_shape,bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
    bounds=[(0,0) if b else (0,None) for b in edge.ravel()]+[(None,None)]*3
    p0=np.array(p0,copy=True);p0[:size][edge.ravel()]=0
    value_grad=jax.jit(jax.value_and_grad(lambda u:objective(u*coord_scale)))
    result=minimize(lambda u:tuple(np.asarray(v) for v in value_grad(u)),p0/np.asarray(coord_scale),jac=True,
                    method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=maxiter,ftol=1e-8,gtol=1e-5,maxcor=20))
    result.x *= np.asarray(coord_scale)
    residual=(np.asarray(render(result.x,beta,psf))[mask]-data[mask])/noise[mask]
    return SimpleNamespace(x=result.x,success=bool(result.success),message=str(result.message),
                           cost=float(np.sum(residual**2)/2),nfev=int(result.nfev),
                           active_mask=np.r_[result.x[:size]<1e-10,[False]*3],
                           objective=float(result.fun),nit=int(result.nit))


def read_plane(path, ext):
    """Read a science plane descriptor without materializing the whole mosaic."""
    hdul = fits.open(uncompressed_fits(path), memmap=True)
    if ext is None:
        choices = [i for i, h in enumerate(hdul) if h.header.get('NAXIS') == 2 and WCS(h.header).has_celestial]
        if not choices:
            hdul.close(); raise ValueError(f'No celestial image in {path}')
        ext = choices[0]
    elif str(ext).isdigit():
        ext = int(ext)
    return hdul, hdul[ext]


def crop_plane(path, ext, slices, science_wcs, shape):
    hdul, hdu = read_plane(path, ext)
    try:
        # All auxiliary maps must be on the same pixel grid, never silently resample errors.
        w = WCS(hdu.header).celestial
        probe_x = np.array([slices[1].start,slices[1].stop-1,slices[1].start,slices[1].stop-1,(slices[1].start+slices[1].stop-1)/2]); probe_y = np.array([slices[0].start,slices[0].start,slices[0].stop-1,slices[0].stop-1,(slices[0].start+slices[0].stop-1)/2])
        if not w.has_celestial:
            raise ValueError('Auxiliary map needs celestial WCS')
        if w.has_celestial:
            ra, dec = science_wcs.pixel_to_world_values(probe_x, probe_y)
            px, py = w.world_to_pixel_values(ra, dec)
            np.testing.assert_allclose(np.c_[px, py], np.c_[probe_x, probe_y], atol=1e-4, rtol=0)
        if hdu.shape != shape:
            raise ValueError('Auxiliary map must have the full science mosaic shape and WCS')
        return np.asarray(hdu.section[slices], float)
    finally:
        hdul.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory')
    p.add_argument('--source-model', choices=['knots','pixels'],default='pixels')
    p.add_argument('--source-grid',type=int,default=96,help='Pixels on longest source-grid side')
    p.add_argument('--source-grid-reference',type=Path,help='A best-fit summary.json supplying an identical source grid for multiband colors; requires --draws 0')
    p.add_argument('--source-margin',type=float,default=.1,help='Source-grid margin around fitted source knots and backprojected observed knots, arcsec')
    p.add_argument('--regularization',type=float,default=.1,help='Source curvature penalty relative to median pixel RMS')
    p.add_argument('--pixel-maxiter',type=int,default=5000)
    p.add_argument('--system', default='K12')
    p.add_argument('--images', nargs='+', default=['a', 'b'])
    p.add_argument('--band',choices=['F115W','F150W','F200W','F444W'],default='F150W')
    p.add_argument('--science', type=Path)
    p.add_argument('--science-ext')
    p.add_argument('--error', type=Path, help='Full mosaic 1-sigma RMS map, same units/grid as science (not inverse variance)')
    p.add_argument('--error-ext')
    p.add_argument('--cache', type=Path, default=Path('/tmp/fold_source_light_cache'), help='Fingerprint-checked native RMS cutouts and lens mappings')
    p.add_argument('--estimate-noise', action='store_true', help='Exploratory sigma-clipped local sky RMS outside knot apertures; ignores drizzle covariance')
    p.add_argument('--noise', type=float, help='Explicit constant RMS in science units; alternative to an error map/ERR extension')
    p.add_argument('--exclude-mask', type=Path, help='Full mosaic mask, nonzero excludes contaminants; same WCS')
    ps = p.add_mutually_exclusive_group()
    ps.add_argument('--psf', type=Path, help='Odd-sized, centered effective PSF FITS at science pixel scale')
    ps.add_argument('--psf-fwhm', type=float, help='Explicit approximate Gaussian FWHM in arcsec')
    p.add_argument('--draws', type=int, default=16, help='Number of additional evenly spaced saved lens draws; best always included')
    p.add_argument('--draw-indices',nargs='+',type=int,help='Specific additional lens draws for short checks; overrides --draws')
    p.add_argument('--bootstrap', type=int, default=0, help='Noise refits at best lens, separately from lens-draw spread')
    p.add_argument('--gp-displacement',action='store_true',help='Apply the saved conditional-mean image-GP coordinate warp; not a mass deflection')
    p.add_argument('--parametric-only', action='store_true')
    p.add_argument('--mask-shape',choices=['box','arc','disks'],default='box',help='Box also constrains blank pixels around the fold; arc/disks are optional restricted masks')
    p.add_argument('--radius', type=float, default=.5, help='Fit union of disks around catalog knots, arcsec')
    p.add_argument('--supersample', type=int, default=4)
    p.add_argument('--sigma-min', type=float, default=.01, help='Minimum resolved source Gaussian sigma, arcsec')
    p.add_argument('--sigma-max', type=float, default=.5)
    p.add_argument('--center-range', type=float, default=.4, help='Allowed source offset from each saved knot, arcsec')
    p.add_argument('--max-nfev', type=int, default=300)
    p.add_argument('--starts', type=int, default=3)
    p.add_argument('--seed', type=int, default=817)
    p.add_argument('--output', type=Path, default=Path('/tmp/fold_source_light'))
    p.add_argument('--prepare-only', action='store_true', help='Extract data/mask and PDF, without lens mapping or light fitting')
    args = p.parse_args()
    grid_reference=None
    if args.source_grid_reference is not None:
        if args.source_model!='pixels' or args.draws!=0 or args.draw_indices is not None:
            p.error('--source-grid-reference requires pixel sources, --draws 0 and no --draw-indices')
        grid_reference=json.loads(args.source_grid_reference.read_text())
        a=grid_reference['arguments']
        if (Path(grid_reference['run']).resolve()!=Path(args.run_directory).resolve() or a['system']!=args.system or a['images']!=args.images or a['gp_displacement']!=args.gp_displacement or a['parametric_only']!=args.parametric_only):
            p.error('Source-grid reference must use the same run, system, images and fixed lens components')
    if args.science is None:
        scale=40 if args.band=='F444W' else 20
        stem=ROOT/f'data/JWST_Images/M0416/hlsp_canucs_jwst_nircam_macs0416-clu-{scale}mas_{args.band.lower()}_v1_sci.fits'
        args.science=stem if stem.exists() else Path(str(stem)+'.gz')
    if args.psf is None and args.psf_fwhm is None:
        args.psf=matching_product(args.science,'psf')
        if args.psf is None: p.error('No matching PSF; provide --psf or explicit approximate --psf-fwhm')
        print(f'Using matching PSF: {args.psf}',flush=True)
    if args.error is None and args.noise is None and not args.estimate_noise:
        args.error=matching_product(args.science,'rms')
        if args.error: print(f'Using matching RMS: {args.error}',flush=True)
    if not args.science.exists(): p.error(f'Science image missing: {args.science}')
    if args.parametric_only and args.gp_displacement: p.error('GP warp requires the full fitted parametric+GRF model')
    if min(args.radius, args.center_range, args.sigma_min) <= 0 or args.sigma_max <= args.sigma_min:
        p.error('Positive radii/widths required, with sigma-max > sigma-min')
    if args.source_margin<=0: p.error('source-margin must be positive')
    if args.source_grid < 8 or args.regularization < 0 or args.pixel_maxiter < 1:
        p.error('source-grid >=8, regularization >=0 and pixel-maxiter >=1 required')
    if args.supersample < 1 or args.draws < 0 or args.starts < 1 or args.bootstrap < 0:
        p.error('Invalid supersample/draws/starts/bootstrap')
    if args.estimate_noise and (args.noise is not None or args.error is not None):
        p.error('Use only one of --estimate-noise, --noise, --error')
    if args.noise is not None and (args.noise <= 0 or args.error is not None):
        p.error('--noise must be positive and cannot be combined with --error')
    from lenscluster.image_catalog import load_image_catalog
    config, state, evaluator, posterior, _ = load_run(args.run_directory)
    catalog = load_image_catalog(config.model.image_constraints.catalog_path,
        sigma_inst_arcsec=config.model.image_constraints.sigma_inst_arcsec,
        sigma_floor_arcsec=config.model.image_constraints.sigma_floor_arcsec)
    rows = catalog[catalog.system_id.eq(args.system) & catalog.image_id.isin(args.images)]
    rows = rows[rows.image_label.isin(list(map(str, state.images.image_labels)))]
    if rows.empty or not set(args.images).issubset(set(rows.image_id)):
        raise ValueError('Requested images are absent from fitted catalog')
    if rows.catalog_z.nunique() != 1:
        raise ValueError('One source redshift required')
    families = sorted(rows.family_id.unique()); n = len(families)
    family_indices = [list(map(str, state.images.family_ids)).index(f) for f in families]
    hdul, sci = read_plane(args.science, args.science_ext)
    wcs = WCS(sci.header).celestial
    mosaic_shape = sci.shape
    px, py = wcs.world_to_pixel_values(rows.ra.to_numpy(), rows.dec.to_numpy())
    from astropy.wcs.utils import proj_plane_pixel_scales
    scale = float(np.sqrt(np.prod(proj_plane_pixel_scales(wcs)))*3600)
    if args.psf:
        kernel = np.asarray(fits.getdata(args.psf), float)
        if kernel.ndim != 2 or any(s % 2 != 1 for s in kernel.shape) or not np.isfinite(kernel).all() or kernel.sum() <= 0 :
            raise ValueError('PSF must be finite, odd-sized with positive sum')
        psf_header=fits.getheader(args.psf)
        if psf_header.get('FILTER') and sci.header.get('FILTER') and psf_header['FILTER']!=sci.header['FILTER']: raise ValueError('PSF and science filters differ')
        import re
        match=re.search(r'-(\d+)mas_',args.psf.name)
        if match and not np.isclose(float(match[1])/1000,scale,rtol=.01): raise ValueError('PSF and mosaic pixel scales differ')
        print(f'PSF: {kernel.shape}, sum={kernel.sum():.6g}, negative absolute fraction={abs(kernel[kernel<0].sum())/kernel.sum():.3g}',flush=True)
        kernel /= kernel.sum()
    else:
        if args.psf_fwhm <= 0: raise ValueError('PSF FWHM must be positive')
        kernel = gaussian_psf(args.psf_fwhm/scale)
    pad = int(np.ceil(args.radius/scale))+max(kernel.shape)//2+3
    x0, x1 = int(np.floor(min(px)))-pad, int(np.ceil(max(px)))+pad+1
    y0, y1 = int(np.floor(min(py)))-pad, int(np.ceil(max(py)))+pad+1
    if x0 < 0 or y0 < 0 or x1 > sci.shape[1] or y1 > sci.shape[0]:
        raise ValueError('Requested padded cutout falls outside mosaic')
    slices = (slice(y0,y1),slice(x0,x1)); data = np.asarray(sci.section[slices],float)
    if args.error:
        err_file,err_hdu=read_plane(args.error,args.error_ext)
        noise_unit=str(err_hdu.header.get('BUNIT','')).strip().lower();science_unit=str(sci.header.get('BUNIT','')).strip().lower()
        err_file.close()
        if noise_unit and science_unit and noise_unit!=science_unit: raise ValueError('Science and RMS units differ')
        args.cache.mkdir(parents=True,exist_ok=True)
        token=json.dumps([str(args.error.resolve()),args.error.stat().st_size,args.error.stat().st_mtime_ns,args.error_ext,str(args.science.resolve()),args.science.stat().st_mtime_ns,x0,x1,y0,y1])
        cache=args.cache/(hashlib.sha256(token.encode()).hexdigest()+'_rms.npy')
        if cache.exists(): noise=np.load(cache)
        else:
            noise = crop_plane(args.error,args.error_ext,slices,wcs,sci.shape)
            np.save(cache,noise)
    elif args.noise:
        noise = np.full_like(data,args.noise)
    elif args.estimate_noise:
        from astropy.stats import sigma_clipped_stats
        gy,gx=np.mgrid[:data.shape[0],:data.shape[1]]
        outside=np.min((gx[...,None]+x0-px)**2+(gy[...,None]+y0-py)**2,axis=-1) > (args.radius/scale)**2
        if np.count_nonzero(outside & np.isfinite(data)) < 100: raise ValueError('Too few sky pixels for local RMS')
        _,_,rms=sigma_clipped_stats(data[outside],sigma=3,maxiters=8)
        if not np.isfinite(rms) or rms <= 0: raise ValueError('Invalid local sky RMS')
        noise=np.full_like(data,rms)
        print(f'Exploratory local sky RMS = {rms:.6g} (native units); no drizzle correction',flush=True)
    elif 'ERR' in hdul:
        if hdul['ERR'].shape != sci.shape: raise ValueError('ERR and SCI shapes differ')
        noise = np.asarray(hdul['ERR'].section[slices],float)
    else:
        raise ValueError('No ERR extension: supply --error RMS map or explicit --noise in science units')
    unit = sci.header.get('BUNIT',hdul[0].header.get('BUNIT','unspecified'))
    cut_header = wcs.slice(slices).to_header(); cut_header['BUNIT'] = unit
    hdul.close()
    yy,xx = np.mgrid[:data.shape[0],:data.shape[1]]
    mask = np.min((xx[...,None]+x0-px)**2+(yy[...,None]+y0-py)**2,axis=-1) <= (args.radius/scale)**2
    if args.mask_shape=='box':
        mask=(xx>=min(px)-x0-args.radius/scale)&(xx<=max(px)-x0+args.radius/scale)&(yy>=min(py)-y0-args.radius/scale)&(yy<=max(py)-y0+args.radius/scale)
    if args.mask_shape=='arc':
        # Follow the ordered knot sequence along its principal axis, including
        # the fold between partners. Fit diffuse light between catalog knots.
        positions=np.c_[px-x0,py-y0]
        _,_,v=np.linalg.svd(positions-positions.mean(0),full_matrices=False)
        ordered=positions[np.argsort(positions@v[0])]
        q=np.stack([xx,yy],axis=-1)
        for a,b in zip(ordered[:-1],ordered[1:]):
            delta=b-a
            t=np.clip(np.sum((q-a)*delta,axis=-1)/max(float(delta@delta),1e-12),0,1)
            mask |= np.sum((q-a-t[...,None]*delta)**2,axis=-1) <= (args.radius/scale)**2
    mask &= np.isfinite(data)&np.isfinite(noise)&(noise>0)
    if args.exclude_mask:
        mask &= crop_plane(args.exclude_mask,None,slices,wcs,mosaic_shape) == 0
    if mask.sum() <= 4*n+1: raise ValueError('Too few usable fit pixels')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'fits.npz').exists():
        raise FileExistsError('Output already contains a fit; choose a new --output directory')
    fits.HDUList([fits.PrimaryHDU(data,cut_header),fits.ImageHDU(noise,cut_header,name='RMS'),fits.ImageHDU(mask.astype('uint8'),cut_header,name='FIT_MASK')]).writeto(out/'input.fits',overwrite=True)
    def diagnostic(model=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from lenscluster.plot_style import PAPER_STYLE
        with plt.rc_context(fname=ROOT/'src/lenscluster/style.mplstyle',rc=PAPER_STYLE):
            fig, axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
            lo,hi=np.nanpercentile(data[mask],[2,99])
            ys,xs=np.nonzero(mask);view=(slice(max(0,ys.min()-8),min(data.shape[0],ys.max()+9)),slice(max(0,xs.min()-8),min(data.shape[1],xs.max()+9)))
            arrays=[data,mask.astype(float),noise] if model is None else [data,model,np.where(mask,(data-model)/noise,np.nan)]
            names=['JWST','Fit mask','RMS'] if model is None else ['JWST','Lens + GP warp + source' if args.gp_displacement else 'Fixed lens + fitted source','Residual / RMS']
            for i,(ax,a,label) in enumerate(zip(axes,arrays,names)):
                kw=dict(cmap='viridis',vmin=lo,vmax=hi) if i==0 or (model is not None and i==1) else dict(cmap='viridis')
                if model is not None and i==2:kw=dict(cmap='RdBu_r',vmin=-5,vmax=5)
                im=ax.imshow(a[view],origin='lower',**kw);ax.text(.03,.96,label,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
                ax.set_xlabel('Pixel');ax.set_ylabel('Pixel');ax.tick_params(top=True,right=True);fig.colorbar(im,ax=ax,location='top',shrink=.9)
            fig.savefig(out/('fit.pdf' if model is not None else 'input.pdf'));plt.close(fig)
    diagnostic()
    if args.prepare_only: return
    if not args.parametric_only and getattr(state,'gp_blocks',None) != 'all':
        raise ValueError('A common source requires a shared GRF (gp_blocks=all), or --parametric-only')
    ss=args.supersample; ny,nx=data.shape
    fy,fx=np.mgrid[:ny*ss,:nx*ss]
    ra,dec=wcs.pixel_to_world_values(x0+(fx+.5)/ss-.5,y0+(fy+.5)/ss-.5)
    _,ra0,dec0=state.reference
    points=np.c_[((ra0-ra)*3600*np.cos(np.deg2rad(dec0))).ravel(),((dec-dec0)*3600).ravel()]
    best=int(np.nanargmax(posterior.log_prob)); rest=np.delete(np.arange(len(posterior.samples)),best)
    chosen=rest[np.linspace(0,len(rest)-1,min(args.draws,len(rest)),dtype=int)] if args.draws else []
    if args.draw_indices is not None:
        if any(i<0 or i>=len(posterior.samples) for i in args.draw_indices): raise ValueError('Draw index outside saved posterior')
        chosen=[i for i in dict.fromkeys(args.draw_indices) if i!=best]
    indices=[best,*map(int,chosen)]
    if grid_reference is not None and grid_reference['results'][0]['draw']!=best:
        raise ValueError('Source-grid reference has a different best lens draw')
    render, source=make_renderer(data.shape,ss,n); rng=np.random.default_rng(args.seed)
    results=[]; parameters=[]; models=[]; bootstrap=[];bootstrap_reports=[]
    previous_pixel=None
    z=float(rows.catalog_z.iloc[0]); t0=time.perf_counter()
    for number, draw in enumerate(indices):
        theta=posterior.samples[draw]
        packed=evaluator.packed_state_for_z(evaluator.physical(theta),z)
        key=hashlib.sha256()
        for a in (np.asarray(theta),points,*jax.tree_util.tree_leaves(packed)): key.update(np.asarray(a).tobytes())
        key.update(str((args.parametric_only,args.gp_displacement,'gp-warp-v1')).encode())
        for path in [Path(args.run_directory)/'final/artifacts/config.pkl',
                     Path(args.run_directory)/'init/artifacts/state.json',
                     ROOT/'src/lenscluster/evaluator.py',ROOT/'src/lenscluster/gp_knee.py']:
            if path.exists(): key.update(path.read_bytes())
        key.update(str((z,family_indices)).encode())
        for path in [Path(config.model.image_constraints.catalog_path),Path(args.run_directory)/'init/artifacts/classification.json']:
            key.update(path.read_bytes())
        args.cache.mkdir(parents=True,exist_ok=True)
        cached=args.cache/(key.hexdigest()+'_beta.npy')
        if cached.exists():
            beta=jnp.asarray(np.load(cached));print(f'draw {draw}: cached lens mapping',flush=True)
        else:
            lens_points=points
            warp_info=None
            if args.gp_displacement:
                lens_points,warp_info=gp_coordinate_correction(evaluator,state,theta,points)
                print(f'draw {draw}: GP warp {warp_info}',flush=True)
            # Request deflections only: JAX removes the unused Hessian work.
            components=np.arange(evaluator.n_components,dtype=np.int32)
            @jax.jit
            def alpha_only(points,packed):
                return jnp.sum(evaluator.component_rows(points,packed,components)[0],axis=0)
            alpha=np.empty_like(lens_points)
            for start in range(0,len(lens_points),1024):
                q=lens_points[start:start+1024];count=len(q)
                padded=np.pad(q,((0,1024-count),(0,0)),mode='edge')
                alpha[start:start+count]=np.asarray(alpha_only(padded,packed))[:count]
            probe=np.linspace(0,len(lens_points)-1,16,dtype=int)
            reference,_=evaluator.total_fields_at(lens_points[probe],packed,chunk=64)
            np.testing.assert_allclose(alpha[probe],reference,rtol=1e-10,atol=1e-10)
            if not args.parametric_only:
                correction=evaluator.gp_deflection_correction(theta)
                if correction is None: raise ValueError('Saved run has no conditional GRF')
                @jax.jit
                def grf_only(q,families):
                    return correction._fields(q,families)[0]
                grf=np.empty_like(lens_points)
                for start in range(0,len(lens_points),1024):
                    q=lens_points[start:start+1024];count=len(q)
                    padded=np.pad(q,((0,1024-count),(0,0)),mode='edge')
                    if hasattr(correction,'_fields'):
                        value=grf_only(padded,np.full(1024,family_indices[0],dtype=np.int32))
                    else:
                        value=np.column_stack(correction(padded[:,0],padded[:,1],np.full(1024,family_indices[0]))[:2])
                    grf[start:start+count]=np.asarray(value)[:count]
                reference=np.column_stack(correction(lens_points[probe,0],lens_points[probe,1],np.full(16,family_indices[0]))[:2])
                np.testing.assert_allclose(grf[probe],reference,rtol=1e-10,atol=1e-10)
                alpha+=grf
            beta=jnp.asarray(lens_points-alpha)
            np.save(cached,np.asarray(beta))
            cached.with_suffix('.json').write_text(json.dumps(dict(warp=warp_info)))
            print(f'draw {draw}: lens mapping computed',flush=True)
        warp_info=json.loads(cached.with_suffix('.json').read_text()).get('warp') if cached.with_suffix('.json').exists() else None
        centers=np.asarray(evaluator.source_positions(theta))[family_indices]
        if args.source_model=='pixels':
            mapped=np.asarray(beta).reshape(ny,ss,nx,ss,2).mean((1,3))
            observed_beta=mapped[np.round(py-y0).astype(int),np.round(px-x0).astype(int)]
            anchors=np.vstack([centers,observed_beta])
            lower=anchors.min(0)-args.source_margin;upper=anchors.max(0)+args.source_margin
            spacing=float(max(upper-lower)/(args.source_grid-5))
            origin=lower-2*spacing
            grid_shape=(args.source_grid,args.source_grid)
            if grid_reference is not None:
                reference_grid=grid_reference['results'][0]
                grid_shape=tuple(reference_grid['source_grid_shape']);origin=np.asarray(reference_grid['source_origin']);spacing=float(reference_grid['source_spacing_arcsec'])
            render,source=make_pixel_renderer(data.shape,ss,grid_shape,origin,spacing,beta=beta)
            count=int(np.prod(grid_shape))
            bg=float(np.nanmedian(data[~mask]))
            p0=np.r_[np.full(count,max(float(np.median(data[mask]))-bg,0)),bg,0.,0.]
            if previous_pixel is not None:
                previous_p,previous_source,previous_center=previous_pixel
                sy0,sx0=np.mgrid[:grid_shape[0],:grid_shape[1]]
                q=np.stack([origin[0]+sx0*spacing,origin[1]+sy0*spacing],axis=-1)
                q-=centers.mean(0)-previous_center
                p0=np.r_[np.asarray(previous_source(previous_p,q)).ravel(),previous_p[-3:]]
            effective_regularization=args.regularization*(.003/spacing)**2
            fit=fit_source_grid(render,beta,kernel,data,noise,mask,p0,grid_shape,effective_regularization,args.pixel_maxiter)
            previous_pixel=(fit.x.copy(),source,centers.mean(0))
            parameter_order=[f'source_pixel_{i}' for i in range(count)]+['background','background_x','background_y']
        else:
            sigma=np.clip(.08,args.sigma_min*1.1,args.sigma_max*.9)
            bg=float(np.nanmedian(data[~mask])) if np.any(~mask) else float(np.nanmedian(data))
            peak=max(float(np.nanpercentile(data[mask],95)-bg),float(np.median(noise[mask])))
            p0=np.r_[np.c_[np.full(n,np.log(peak*2*np.pi*sigma**2)),np.full(n,np.log(sigma)),centers].ravel(),bg]
            low=np.r_[np.c_[np.full(n,p0[0]-15),np.full(n,np.log(args.sigma_min)),centers-args.center_range].ravel(),-np.inf]
            high=np.r_[np.c_[np.full(n,p0[0]+15),np.full(n,np.log(args.sigma_max)),centers+args.center_range].ravel(),np.inf]
            candidates=[]
            for attempt in range(args.starts):
                initial=p0.copy()
                if attempt:
                    initial[:-1].reshape(n,4)[:,2:]+=rng.normal(0,.025,(n,2))
                    initial[:-1].reshape(n,4)[:,1]+=rng.normal(0,.3,n)
                    initial=np.clip(initial,low+1e-8,high-1e-8)
                candidates.append(fit_pixels(render,beta,kernel,data,noise,mask,initial,(low,high),args.max_nfev))
            successful=[r for r in candidates if r.success]
            fit=min(successful or candidates,key=lambda r:r.cost)
            parameter_order=[f'{f}:{name}' for f in families for name in ['log_amp','log_sigma_arcsec','beta_x_arcsec','beta_y_arcsec']]+['constant_background']
        model=np.asarray(render(fit.x,beta,kernel)); parameters.append(fit.x);models.append(model)
        results.append(dict(draw=draw,gp_warp=warp_info,success=bool(fit.success),message=fit.message,chi2=float(2*fit.cost),dof=int(mask.sum()-len(p0)) if args.source_model=='knots' else None,chi2_per_pixel=float(2*fit.cost/mask.sum()),nfev=int(fit.nfev),bound_parameters=np.flatnonzero(fit.active_mask).tolist() if args.source_model=='knots' else [],zero_source_pixels=int(np.sum(fit.active_mask)) if args.source_model=='pixels' else None))
        print(f'{number+1}/{len(indices)} draw {draw}: chi2/Npixel={2*fit.cost/mask.sum():.3f}, success={fit.success}',flush=True)
        residual=(data-model)[mask]/noise[mask]
        results[-1]['residual_median_sigma']=float(np.median(residual))
        results[-1]['residual_robust_sd_sigma']=float(1.4826*np.median(abs(residual-np.median(residual))))
        results[-1]['fraction_abs_residual_gt3']=float(np.mean(abs(residual)>3))
        knot_metrics=[]
        for label,kx,ky in zip(rows.image_label,px-x0,py-y0):
            aperture=mask & ((xx-kx)**2+(yy-ky)**2 <= (.15/scale)**2)
            r=((data-model)/noise)[aperture]
            knot_metrics.append(dict(image_label=str(label),aperture_arcsec=.15,pixels=int(len(r)),rms_sigma=float(np.sqrt(np.mean(r*r))) if len(r) else None,fraction_abs_gt3=float(np.mean(abs(r)>3)) if len(r) else None))
        results[-1]['knot_residuals']=knot_metrics
        print('Knot RMS (sigma): '+', '.join(f"{m['image_label']}={m['rms_sigma']:.2f}" if m['rms_sigma'] is not None else f"{m['image_label']}=masked" for m in knot_metrics),flush=True)
        if args.source_model=='pixels':
            src=fit.x[:-3].reshape(grid_shape);gy,gx=np.mgrid[:grid_shape[0],:grid_shape[1]]
            coords=np.stack([origin[0]+gx*spacing,origin[1]+gy*spacing],axis=-1)
            total=src.sum()
            edge_ring=np.ones(grid_shape,bool);edge_ring[4:-4,4:-4]=False
            results[-1]['source_outer_four_pixel_flux_fraction']=float(src[edge_ring].sum()/max(total,1e-30))
            center=(src[...,None]*coords).sum((0,1))/max(total,1e-30)
            radius=np.sqrt(np.sum((coords-center)**2,axis=-1));order=np.argsort(radius.ravel())
            half_index=min(np.searchsorted(np.cumsum(src.ravel()[order]),total/2),len(order)-1)
            results[-1].update(source_flux_within_grid_native_units=float(total*spacing**2/scale**2),source_centroid_arcsec=center.tolist(),source_half_light_radius_arcsec=float(radius.ravel()[order][half_index]))
            results[-1].update(effective_regularization=effective_regularization,source_grid_shape=grid_shape,source_origin=origin.tolist(),source_spacing_arcsec=spacing,regularized_objective=fit.objective,iterations=fit.nit)
        if number==0:
            diagnostic(model)
            for _ in range(args.bootstrap):
                simulated=model+rng.normal(size=data.shape)*noise
                if args.source_model=='pixels':
                    b=fit_source_grid(render,beta,kernel,simulated,noise,mask,fit.x,grid_shape,effective_regularization,args.pixel_maxiter)
                else:
                    b=fit_pixels(render,beta,kernel,simulated,noise,mask,fit.x,(low,high),args.max_nfev)
                bootstrap_reports.append(dict(success=bool(b.success),message=b.message,nfev=b.nfev))
                if b.success: bootstrap.append(b.x)
                print(f'noise bootstrap {len(bootstrap)}/{args.bootstrap}: success={b.success}',flush=True)
            # Local Gaussian approximation, kept separate from between-lens spread.
            if args.source_model=='knots':
                results[-1]['jacobian_rank']=int(np.linalg.matrix_rank(fit.jac))
                covariance=np.linalg.pinv(fit.jac.T@fit.jac)
                np.save(out/'best_local_covariance.npy',covariance)
            if args.source_model=='pixels':
                cmin=np.asarray(origin);cmax=cmin+spacing*(np.asarray(grid_shape[::-1])-1)
            else:
                cmin=centers.min(0)-.2;cmax=centers.max(0)+.2
            sy,sx=np.mgrid[cmin[1]:cmax[1]:256j,cmin[0]:cmax[0]:256j]
            source_image=np.asarray(source(fit.x,jnp.asarray(np.stack([sx,sy],axis=-1))))
            sh=fits.Header();sh['CTYPE1']='LINEAR';sh['CTYPE2']='LINEAR';sh['CRPIX1']=1;sh['CRPIX2']=1
            sh['CRVAL1']=cmin[0];sh['CRVAL2']=cmin[1];sh['CDELT1']=(cmax[0]-cmin[0])/255;sh['CDELT2']=(cmax[1]-cmin[1])/255;sh['CUNIT1']='arcsec';sh['CUNIT2']='arcsec';sh['BUNIT']=f'{unit}/arcsec2' if unit.lower() in ['njy','ujy','jy'] else unit
            if unit.lower() in ['njy','ujy','jy']: source_image=source_image/scale**2
            fits.writeto(out/'best_source.fits',source_image,sh,overwrite=True)
            import matplotlib.pyplot as plt
            from lenscluster.plot_style import PAPER_STYLE
            with plt.rc_context(fname=ROOT/'src/lenscluster/style.mplstyle',rc=PAPER_STYLE):
                fig,ax=plt.subplots(figsize=(4,3.6),layout='constrained')
                center=(cmin+cmax)/2
                im=ax.imshow(source_image,origin='lower',extent=[cmin[0]-center[0],cmax[0]-center[0],cmin[1]-center[1],cmax[1]-center[1]],cmap='viridis')
                ax.set_xlabel(r'$\Delta\beta_x$ (arcsec)');ax.set_ylabel(r'$\Delta\beta_y$ (arcsec)')
                ax.tick_params(top=True,right=True);fig.colorbar(im,ax=ax,location='top',label=sh['BUNIT'])
                ax.text(.03,.96,'Reconstructed source',transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'),fontsize=8)
                fig.savefig(out/'source.pdf');plt.close(fig)
        np.savez_compressed(out/'fits.npz',draw_indices=indices[:number+1],parameters=parameters,models=models,noise_bootstrap_parameters=np.asarray(bootstrap),mask=mask,psf=kernel)
        metadata=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},run=str(Path(args.run_directory).resolve()),families=families,redshift=z,unit=unit,noise_bootstrap=bootstrap_reports,fit_pixels=int(mask.sum()),pixel_scale_arcsec=scale,results=results,elapsed_seconds=time.perf_counter()-t0,
            parameter_order=parameter_order,
            uncertainty='Unweighted fixed-lens refits; lens spread is not a source posterior. Local covariance and optional independent-pixel noise bootstrap are separate. No conditional GRF/GP realizations, PSF uncertainty, foreground-light uncertainty, or drizzle covariance included. Optional GP warp is a fixed nuisance correction, not a mass deflection.')
        diag=Path(args.run_directory)/'final/artifacts/posterior_diagnostics.json'
        metadata['lens_posterior_diagnostics']=str(diag.resolve())
        metadata['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        metadata['input_files']={str(f.resolve()):dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in [args.science,Path(args.run_directory)/'final/artifacts/posterior.npz',*[v for v in [args.error,args.psf,args.exclude_mask] if v is not None]]}
        (out/'summary.json').write_text(json.dumps(metadata,indent=2))
    good=np.array([r['success'] for r in results]); extra=good.copy();extra[0]=False
    if extra.sum()>=3:
        np.savez_compressed(out/'lens_spread.npz',parameter_quantiles=np.quantile(np.asarray(parameters)[extra],[.16,.5,.84],axis=0) if args.source_model=='knots' else np.empty((0,)),model_quantiles=np.quantile(np.asarray(models)[extra],[.16,.5,.84],axis=0),quantiles=[.16,.5,.84])
    metadata['successful_additional_lens_draws']=int(extra.sum())
    metadata['lens_quantiles_available']=bool(extra.sum()>=3)
    (out/'summary.json').write_text(json.dumps(metadata,indent=2))
    fits.writeto(out/'best_model.fits',models[0],cut_header,overwrite=True)
    print(f'Outputs: {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()
