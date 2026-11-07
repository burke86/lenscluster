"""Fit the light of a lensed arc with the lens fixed at our model: the source galaxy (and the nearby member light) is fit
to a JWST cutout, once per posterior draw of the lens, so the lens uncertainty shows up as a spread in the source.

The lens never enters the fit as parameters.  For each draw the deflection (halos, members, and the field's conditional
mean) is evaluated on a supersampled pixel grid and stored as a source-plane coordinate map beta(x); the fit then only
evaluates source light at beta and member light at x, bins to the pixel scale and convolves with the PSF.  Everything
after the beta maps is JAX (jaxtronomy light profiles), so a fit takes seconds.

Example:
    python scripts/fit_arc_source.py --run results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks \\
        --system K12 --band F150W --source knots --draws 16 --nuts 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.optimize import minimize

from lenscluster import artifacts, cutouts
from lenscluster.evaluator import Evaluator
from lenscluster.sky import radec_to_offsets_arcsec
from lenscluster.state import build_state, with_member_classes

NIRCAM_FWHM_ARCSEC = {"F115W": 0.040, "F150W": 0.050, "F200W": 0.066, "F277W": 0.092, "F356W": 0.116, "F444W": 0.145}
GAIN_E_PER_DN = 2.0        # NIRCam, approximate; only the Poisson part of the noise map depends on it


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory with init/ and final/ (a CANUCS M0416 run)")
    p.add_argument("--system", default="K12", help="catalog system whose fold pair a-b is fit")
    p.add_argument("--images", default="a,b", help="the images of the system inside the cutout")
    p.add_argument("--band", default="F150W")
    p.add_argument("--image-dir", default="data/JWST_Images/M0416")
    p.add_argument("--image-kind", choices=["bgsub", "sci"], default="bgsub",
                   help="bgsub: the CANUCS mosaic with the diffuse light and bright galaxies subtracted; sci: the plain mosaic")
    p.add_argument("--arc-radius", type=float, default=0.5,
                   help="pixels within this distance (arcsec) of the path through the system's catalog positions form the arc; "
                        "other sources above 5 sigma are masked")
    p.add_argument("--arc-sigma", type=float, default=3.0,
                   help="arc pixels: arc-region pixels above this many sigma; they set the source box and the arc chi2")
    p.add_argument("--size", type=float, default=7.0, help="cutout side in arcsec")
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--draws", type=int, default=16, help="posterior draws of the lens (the best draw is always included)")
    p.add_argument("--source", choices=["sersic", "knots", "shapelets", "pixels"], default="pixels",
                   help="pixels: a regularised pixelated source solved linearly (closest reproduction); the others are parametric")
    p.add_argument("--src-pixel", type=float, default=0.0015,
                   help="pixels: source pixel in arcsec (an image pixel along a fold arc of magnification 25 is 0.8 mas in the source)")
    p.add_argument("--src-margin", type=float, default=0.3, help="pixels: margin around the backprojected arc, fraction of its extent")
    p.add_argument("--reg", choices=["gradient", "curvature"], default="gradient", help="pixels: regularisation")
    p.add_argument("--lam", type=float, default=0.0, help="pixels: regularisation strength; 0 = largest lambda with chi2_nu <= 1 on the arc pixels")
    p.add_argument("--zoom-width", type=float, default=1.4, help="width in arcsec of the zoomed strip along the arc")
    p.add_argument("--grid-from", default="", help="pixels: reuse the source grid (per-draw origins, pixel, size) of another run's "
                                                    "pixel_sources.npz, e.g. the F150W fit, so several bands share one source grid")
    p.add_argument("--shapelet-nmax", type=int, default=6)
    p.add_argument("--knots", type=int, default=0, help="number of Gaussian knots (0 = one per catalog knot of image a)")
    p.add_argument("--members", default="auto", help="'auto' (catalog members within the cutout + 1.5\"), 'none', or ids 7764,...")
    p.add_argument("--member-catalog", default="catalogs/m0416/canucs2025/source/members.cat")
    p.add_argument("--mask", default="", help="extra circular masks in arcsec of the model frame: 'x,y,r;x,y,r'")
    p.add_argument("--psf-file", default="auto", help="FITS PSF stamp; 'auto' = the CANUCS *_psf.fits(.gz) next to the mosaic, '' = none")
    p.add_argument("--psf-trim", type=float, default=2.0, help="side in arcsec to which the PSF stamp is trimmed")
    p.add_argument("--psf-oversample", type=int, default=0, help="oversampling of --psf-file relative to the mosaic pixel (0 = from its header or 1)")
    p.add_argument("--psf-star", default="", help="'ra dec' (deg) of an isolated star for an empirical PSF; else a Gaussian")
    p.add_argument("--psf-size", type=float, default=1.5, help="side of the empirical PSF stamp in arcsec")
    p.add_argument("--noise-window", type=float, default=20.0, help="side in arcsec of the window that calibrates the rms map")
    p.add_argument("--noise-scale", type=float, default=0.0,
                   help="multiplies the rms map; 0 = calibrate on the empty-sky pixel scatter of the cutout (drizzle correlation)")
    p.add_argument("--adam-steps", type=int, default=600)
    p.add_argument("--nuts", type=int, default=0, help="NUTS samples for the source at the best-draw lens (0 = skip)")
    p.add_argument("--out", default="output/arc_fits")
    p.add_argument("--rebuild", action="store_true", help="recompute the beta maps even if cached")
    p.add_argument("--bands", default="", help="pixels: comma list of bands fit together (e.g. F115W,F150W,F200W,F444W). One source "
                   "shape is fit to all bands at once, then each band is that shape times an amplitude plus a residual "
                   "regularised by --lam-colour; --band is the reference band that sets the source grid")
    p.add_argument("--lam-colour", default="0",
                   help="--bands: regularisation of each band's departure from the shared shape, one value or per band "
                        "('0,F444W=100'); 0 = chosen by the arc chi2_nu scan")
    p.add_argument("--reg-weight", choices=["none", "resolution"], default="none",
                   help="--add: weight the gradient regulariser by 1/(traced image pixels per source pixel), clipped to [1, 1000], "
                        "so source pixels seen only at low magnification are smoothed more")
    p.add_argument("--mesh-nodes", type=int, default=1500,
                   help="--src-grid mesh: number of nodes, placed by k-means on the traced arc-region pixels (so their density "
                        "follows the magnification but the spacing stays locally even)")
    p.add_argument("--src-grid", choices=["square", "aligned", "mesh"], default="square",
                   help="--bands: 'aligned' = rectangular source pixels along and across the fold, each one image pixel mapped back "
                        "(times --src-res), so the regulariser works in resolution elements; 'mesh' (with --add) = Delaunay mesh "
                        "whose nodes are the traced image pixels of the arc regions, linear inside each triangle")
    p.add_argument("--src-res", type=float, default=1.0, help="--src-grid aligned: source pixel in units of the mapped image pixel")
    p.add_argument("--smooth", choices=["none", "sersic"], default="none",
                   help="--bands: 'sersic' = the shared shape is a Sersic, evaluated at every pixel's beta (so it also predicts the "
                        "light outside the arc), plus the pixelated residual")
    p.add_argument("--affine", action="store_true",
                   help="--add: the source-plane correction of each added region is affine (shift + 2x2 matrix near the identity, "
                        "a local lens error in magnification, shear and rotation), not only a shift")
    p.add_argument("--add", default="", help="--bands: extra image-plane regions of the same source fit jointly with the fold pair, "
                   "'x,y,size,radius;...' in arcsec (e.g. counter-images); each gets a fitted source-plane offset")
    p.add_argument("--at", default="", help="'x,y' in arcsec: fit the light there instead of the catalog images, with the system's "
                   "redshift (e.g. a counter-image predicted by scripts/predict_counter_images.py); --arc-radius sets its region")
    p.add_argument("--image-gp-mean", action="store_true",
                   help="also apply the image-plane GP's conditional mean offset, interpolated from the system's catalog images")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------------------------- lens
# Config fields added after a run was pickled take their default, except these: the run predates them and used the
# half-normal sigma_white prior they replace, so None keeps its parameterisation.
OLD_RUN_FIELDS = {"sigma_white_prior": None, "gp_knee_sigma_white_prior": None}


def _backfill(obj):
    """Fill the dataclass fields an old pickle lacks (recursively); returns the dotted names filled."""
    import dataclasses
    filled = []
    if not dataclasses.is_dataclass(obj):
        return filled
    for f in dataclasses.fields(obj):
        if f.name not in obj.__dict__:
            if f.name in OLD_RUN_FIELDS:
                value = OLD_RUN_FIELDS[f.name]
            elif f.default is not dataclasses.MISSING:
                value = f.default
            else:
                value = f.default_factory()
            object.__setattr__(obj, f.name, value)
            filled.append(f.name)
        else:
            filled += [f"{f.name}.{n}" for n in _backfill(obj.__dict__[f.name])]
    return filled


def load_model(run: Path):
    init = run / "init" / "artifacts"
    config = artifacts.load_config(init)
    filled = _backfill(config)
    if filled:
        print(f"[load_model] config predates {len(filled)} fields, filled: {', '.join(filled)}", flush=True)
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    saved = json.loads((init / artifacts.STATE_FILE).read_text())["parameter_specs"]
    key = ("name", "transform_kind", "transform_offset", "transform_scale")
    if [tuple(sp[k] for k in key) for sp in saved] != [tuple(getattr(sp, k) for k in key) for sp in state.parameter_specs]:
        raise ValueError(f"{run}: the rebuilt parameters differ from the run's saved state")
    centered = np.asarray([m in set(classes.get("centered_members", [])) for m in state.members.member_ids], dtype=bool)
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes, centered=centered)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    posterior = artifacts.load_posterior(run / "final" / "artifacts")
    return config, state, evaluator, posterior


def pick_draws(posterior, n):
    logp = np.asarray(posterior.log_prob)
    best = int(np.nanargmax(logp))
    spread = np.linspace(0, len(logp) - 1, max(n, 1)).astype(int)
    return [best] + [int(k) for k in spread if k != best][: max(n - 1, 0)]


def image_gp_offsets(evaluator, physical, theta, obs, obs_family, z_source, family_index):
    """The image-plane GP realisation at the catalog images of the system: the offset eps_i that moves the observed image to
    where this draw's lens maps the family's source, eps_i = A_i^-1 (beta_i - beta_source), and the GP scales."""
    packed = evaluator.packed_state_for_z(physical, z_source)
    alpha, u = evaluator.total_fields_at(obs, packed)
    a = np.eye(2)[None] - u
    mean = evaluator.gp_deflection_correction(theta)
    if mean is not None:
        dx, dy, jxx, jxy, jyx, jyy = mean(obs[:, 0], obs[:, 1], np.full(len(obs), family_index, dtype=np.int32))
        alpha = alpha + np.column_stack([dx, dy])
        a = a - np.stack([np.stack([jxx, jxy], -1), np.stack([jyx, jyy], -1)], -2)
    sources = np.asarray(evaluator.source_positions(theta))[obs_family]
    eps = np.einsum("nij,nj->ni", np.linalg.inv(a), (obs - alpha) - sources)
    phys = np.asarray(physical)
    sigma_shared, sigma_white = float(phys[evaluator.sigma_shared_index]), float(phys[evaluator.sigma_white_index])
    ell = float(phys[evaluator.gp_ell_index]) if evaluator.gp_ell_index >= 0 else float(evaluator.gp_ell_fixed)
    return eps, sigma_shared, sigma_white, ell


def beta_maps(evaluator, state, samples, draws, xy, z_source, family_index, cache: Path, rebuild: bool, image_gp=None):
    """Source-plane coordinates (draws, N, 2) of the supersampled image-plane points xy: lens + field mean, per draw.
    With ``image_gp`` = (obs, obs_family, sigma_inst) the image-plane GP's conditional mean offset is interpolated from the
    system's catalog images and the lens is evaluated at the corrected positions, beta(x) -= A(x) delta(x)."""
    if cache.exists() and not rebuild:
        with np.load(cache) as f:
            if list(f["draws"]) == list(draws) and f["beta"].shape[1] == len(xy):
                return f["beta"], f["det"]
    beta = np.empty((len(draws), len(xy), 2)); det = np.empty((len(draws), len(xy)))
    fam = np.full(len(xy), family_index, dtype=np.int32)
    for k, d in enumerate(draws):
        t0 = time.time()
        theta = jnp.asarray(samples[d])
        physical = evaluator.physical(theta)
        packed = evaluator.packed_state_for_z(physical, z_source)
        alpha, u = evaluator.total_fields_at(xy, packed)
        a = np.eye(2)[None] - u
        mean = evaluator.gp_deflection_correction(theta)
        if mean is not None:
            dx, dy, jxx, jxy, jyx, jyy = mean(xy[:, 0], xy[:, 1], fam)
            alpha = alpha + np.column_stack([dx, dy])
            a = a - np.stack([np.stack([jxx, jxy], -1), np.stack([jyx, jyy], -1)], -2)
        beta[k] = xy - alpha
        note = ""
        if image_gp is not None:
            obs, obs_family, sigma_inst = image_gp
            eps, sigma_shared, sigma_white, ell = image_gp_offsets(evaluator, physical, theta, obs, obs_family, z_source, family_index)
            k_obs = sigma_shared ** 2 * np.exp(-np.sum((obs[:, None] - obs[None]) ** 2, -1) / (2 * ell ** 2))
            weights = np.linalg.solve(k_obs + (sigma_white ** 2 + sigma_inst ** 2) * np.eye(len(obs)), eps)       # (M, 2)
            k_grid = sigma_shared ** 2 * np.exp(-np.sum((xy[:, None] - obs[None]) ** 2, -1) / (2 * ell ** 2))     # (N, M)
            delta = k_grid @ weights
            beta[k] = beta[k] - np.einsum("nij,nj->ni", a, delta)
            note = (f", image GP: eps {np.round(eps * 1000).astype(int).tolist()} mas, sigma_shared {sigma_shared * 1000:.0f} mas, "
                    f"ell {ell:.2f}\", |delta| on the grid up to {np.abs(delta).max() * 1000:.0f} mas")
        det[k] = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
        print(f"[lens] draw {d}: beta map in {time.time() - t0:.0f} s, field mean {'on' if mean is not None else 'off'}{note}", flush=True)
    np.savez_compressed(cache, beta=beta, det=det, draws=np.asarray(draws))
    return beta, det


# ----------------------------------------------------------------------------------------------------------------- data
def load_cutout(args, state, centre_xy):
    """Data, noise map, mask, pixel scale and the model-frame coordinates of every (supersampled) pixel, all from the WCS."""
    paths = cutouts.find_rgb_band_paths(Path(args.image_dir), cluster="m0416", bands=(args.band,))
    path = paths[args.band]
    if args.image_kind == "bgsub":
        stem = path.name.replace("_bgsub-sci", "_sci").replace("_sci.fits", "_bgsub-sci.fits").replace(".gz", "")
        found = [path.with_name(stem + ext) for ext in ("", ".gz") if path.with_name(stem + ext).exists()]
        if found:
            path = found[0]
        else:
            print("[data] no bgsub mosaic next to the science mosaic: using the plain one")
    band = cutouts.load_band_image(args.band, path)
    print(f"[data] mosaic {path.name}")
    coord = cutouts.arcsec_to_skycoord(*centre_xy, state.reference)
    npix = cutouts.cutout_npixels(band, args.size)
    window = cutouts.cutout_window(band, coord, args.size)
    if window is None:
        raise SystemExit("cutout falls outside the mosaic")
    with fits.open(band.path, memmap=True) as hdul:
        hdu = hdul[band.hdu_index]
        data = np.array(hdu.data[window.src_y0:window.src_y1, window.src_x0:window.src_x1], dtype=float)
        header = hdu.header
        primary = hdul[0].header
    ny, nx = data.shape
    s = args.supersample
    # Supersampled pixel centres in mosaic pixel coordinates, then world, then the model's arcsec frame.
    sub = (np.arange(s) + 0.5) / s - 0.5
    shape4 = (ny, s, nx, s)                                   # pixel row, sub-row, pixel column, sub-column
    py = np.broadcast_to((np.arange(ny)[:, None] + sub[None, :])[:, :, None, None], shape4)
    px = np.broadcast_to((np.arange(nx)[:, None] + sub[None, :])[None, None, :, :], shape4)
    ra, dec = band.wcs.pixel_to_world_values(px.ravel() + window.src_x0, py.ravel() + window.src_y0)
    _, ra0, dec0 = state.reference
    x, y = radec_to_offsets_arcsec(np.asarray(ra), np.asarray(dec), ra0, dec0)
    xy = np.column_stack([x, y])
    scale = band.pixel_scale_arcsec
    # Sky: sigma-clipped on the cutout.  Noise: the CANUCS rms map (per-pixel formal error, nJy) scaled to the empty-sky
    # pixel scatter, since drizzling correlates neighbouring pixels and the formal error underestimates it by ~2; without
    # an rms map, sky scatter plus a Poisson term from the exposure and the MJy/sr calibration.
    _, sky, sky_sigma = sigma_clipped_stats(data[np.isfinite(data)], sigma=3.0, maxiters=10)
    data = data - sky
    rms = load_rms_cutout(band, window)
    if rms is not None:
        empty = np.isfinite(data) & np.isfinite(rms) & (np.abs(data) < 3 * sky_sigma)
        factor = args.noise_scale or drizzle_factor(band, coord, args.noise_window)
        noise = factor * np.where(np.isfinite(rms), rms, np.inf)
        print(f"[data] rms map: median {np.median(rms[empty]):.3f} nJy/px on empty sky, scaled by {factor:.2f} "
              f"({'given' if args.noise_scale else 'empty-sky scatter / rms map'})")
    else:
        exposure = header.get("XPOSURE", primary.get("XPOSURE")) or header.get("EXPTIME", primary.get("EXPTIME"))
        photmjsr = header.get("PHOTMJSR", primary.get("PHOTMJSR"))
        pixel_sr = (scale / 206264.806) ** 2
        if exposure and photmjsr:
            electrons_per_njy = 1e-15 / pixel_sr / photmjsr * float(exposure) * GAIN_E_PER_DN      # nJy -> DN/s -> e-
            poisson = np.sqrt(np.clip(data, 0, None) * electrons_per_njy) / electrons_per_njy
        else:
            poisson = 0.0
        noise = (args.noise_scale or 1.0) * np.sqrt(sky_sigma ** 2 + poisson ** 2)
        print("[data] no rms map found: sky scatter plus an approximate Poisson term")
    mask = np.isfinite(data) & np.isfinite(noise)
    xc = xy[:, 0].reshape(ny, s, nx, s).mean((1, 3)); yc = xy[:, 1].reshape(ny, s, nx, s).mean((1, 3))
    for item in filter(None, args.mask.split(";")):
        mx, my, r = (float(v) for v in item.split(","))
        mask &= np.hypot(xc - mx, yc - my) > r
    data = np.where(mask, data, 0.0)
    print(f"[data] {args.band} {ny}x{nx} px at {scale * 1000:.0f} mas, sky {sky:.2f} +- {sky_sigma:.2f} nJy/px, "
          f"{(~mask).sum()} masked px")
    return dict(data=data, noise=noise, mask=mask, xy=xy, xc=xc, yc=yc, scale=scale, shape=(ny, nx), band=band, window=window,
                sky_reference=(float(ra0), float(dec0)))


def drizzle_factor(band, coord, size_arcsec):
    """Ratio of the empty-sky pixel scatter to the rms map in a wide window around the arc: the drizzle correlation
    correction.  A wide window keeps a large galaxy in the cutout from inflating the sky scatter."""
    window = cutouts.cutout_window(band, coord, size_arcsec)
    with fits.open(band.path, memmap=True) as hdul:
        data = np.array(hdul[band.hdu_index].data[window.src_y0:window.src_y1, window.src_x0:window.src_x1], dtype=float)
    rms = load_rms_cutout(band, window)
    good = np.isfinite(data) & np.isfinite(rms) & (rms > 0)
    ratio = data[good] / rms[good]
    _, centre, spread = sigma_clipped_stats(ratio, sigma=2.5, maxiters=10)
    print(f"[data] drizzle factor {spread:.2f} from the {size_arcsec:.0f}\" window (clipped scatter of data / rms)")
    return float(spread)


def load_rms_cutout(band, window):
    """The CANUCS rms map next to the science mosaic (``_sci`` -> ``_rms``), uncompressed preferred; None when absent."""
    name = band.path.name
    candidates = []
    for stem in (name.replace("_bgsub-sci.fits", "_rms.fits"), name.replace("_sci.fits", "_rms.fits")):
        stem = stem.replace(".gz", "")
        candidates += [band.path.with_name(stem), band.path.with_name(stem + ".gz")]
    for path in candidates:
        if path.exists() and path != band.path:
            if path.suffix == ".gz":
                print(f"[data] reading the gzipped rms map {path.name} (gunzip it next to the mosaic to make this instant)")
            with fits.open(path, memmap=path.suffix != ".gz") as hdul:
                hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) == 2)
                return np.array(hdu.data[window.src_y0:window.src_y1, window.src_x0:window.src_x1], dtype=float)
    return None


def load_psf(args, band, scale, shape_hint):
    psf_file = args.psf_file
    if psf_file == "auto":
        stem = band.path.name.replace("_bgsub-sci", "_sci").replace("_sci.fits", "_psf.fits").replace(".gz", "")
        found = [band.path.with_name(stem + ext) for ext in ("", ".gz") if band.path.with_name(stem + ext).exists()]
        psf_file = str(found[0]) if found else ""
        if not psf_file:
            print("[psf] no CANUCS PSF next to the mosaic (download *_psf.fits.gz from archive.stsci.edu/hlsps/canucs/dr1/macs0416/clu/)")
    if psf_file:
        with fits.open(psf_file) as hdul:
            hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2)
            stamp = np.array(hdu.data, dtype=float)
            header = hdu.header
        stamp = stamp.reshape(stamp.shape[-2:]) if stamp.ndim > 2 else stamp
        over = args.psf_oversample
        if not over:
            over = int(round(header.get("OVERSAMP", header.get("DET_SAMP", 0)) or 0)) or 1
            for key in ("PIXELSCL", "PIXSCALE", "CDELT2"):
                if key in header and header[key]:
                    pix = abs(float(header[key])) * (3600.0 if key == "CDELT2" else 1.0)
                    over = max(1, int(round(scale / pix)))
                    break
        stamp = np.where(np.isfinite(stamp), stamp, 0.0)
        if over > 1:                                          # block-average to the mosaic pixel, keeping the stamp centred
            n = (min(stamp.shape) // over) * over
            cy, cx = (np.array(stamp.shape) - 1) / 2
            y0, x0 = int(round(cy - n / 2 + 0.5)), int(round(cx - n / 2 + 0.5))
            stamp = stamp[y0:y0 + n, x0:x0 + n].reshape(n // over, over, n // over, over).mean((1, 3))
        half = int(min(args.psf_trim / scale / 2, (stamp.shape[0] - 1) // 2, (stamp.shape[1] - 1) // 2))
        iy, ix = np.unravel_index(np.argmax(stamp), stamp.shape)
        stamp = stamp[iy - half:iy + half + 1, ix - half:ix + half + 1]
        above = (stamp > stamp.max() / 2).sum()
        print(f"[psf] {Path(psf_file).name}: {stamp.shape[0]} px kept (oversampling {over} -> 1), "
              f"FWHM ~ {2 * np.sqrt(above / np.pi) * scale * 1000:.0f} mas, stars {header.get('N_STARS', '?')}")
        return stamp / stamp.sum()
    if args.psf_star:
        ra, dec = (float(v) for v in args.psf_star.split())
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        window = cutouts.cutout_window(band, SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"), args.psf_size)
        with fits.open(band.path, memmap=True) as hdul:
            stamp = np.array(hdul[band.hdu_index].data[window.src_y0:window.src_y1, window.src_x0:window.src_x1], dtype=float)
        _, sky, _ = sigma_clipped_stats(stamp[np.isfinite(stamp)], sigma=3.0)
        stamp = np.where(np.isfinite(stamp), stamp - sky, 0.0)
        stamp = np.clip(stamp, 0, None)
        # Recentre on the peak and apodise the stamp edge so the normalisation is not set by wings and neighbours.
        iy, ix = np.unravel_index(np.argmax(stamp), stamp.shape)
        h = min(iy, ix, stamp.shape[0] - 1 - iy, stamp.shape[1] - 1 - ix)
        stamp = stamp[iy - h:iy + h + 1, ix - h:ix + h + 1]
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        stamp *= np.clip(1.0 - (np.hypot(yy, xx) / h) ** 4, 0, 1)
        print(f"[psf] empirical PSF from the star at ({ra:.6f}, {dec:.6f}), {stamp.shape[0]} px")
        return stamp / stamp.sum()
    fwhm = NIRCAM_FWHM_ARCSEC.get(args.band.upper(), 0.06) / scale
    sig = fwhm / 2.3548
    h = int(np.ceil(4 * sig))
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    stamp = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sig ** 2)
    print(f"[psf] Gaussian PSF, FWHM {fwhm * scale * 1000:.0f} mas")
    return stamp / stamp.sum()


# ---------------------------------------------------------------------------------------------------------- light model
def member_list(args, state, cut, centre_xy):
    if args.members == "none":
        return []
    cat = pd.read_csv(args.member_catalog, sep=r"\s+", comment="#", header=None).iloc[:, :7]
    cat.columns = ["id", "ra", "dec", "a", "b", "theta", "mag"]
    _, ra0, dec0 = state.reference
    x, y = radec_to_offsets_arcsec(cat.ra.to_numpy(), cat.dec.to_numpy(), ra0, dec0)
    cat["x"], cat["y"] = x, y
    if args.members == "auto":
        half = args.size / 2 + 1.5
        sel = cat[(np.abs(cat.x - centre_xy[0]) < half) & (np.abs(cat.y - centre_xy[1]) < half)]
    else:
        ids = {int(v) for v in args.members.split(",")}
        sel = cat[cat.id.isin(ids)]
    rows = [dict(id=int(r.id), x=float(r.x), y=float(r.y), mag=float(r.mag)) for r in sel.itertuples()]
    print(f"[members] {len(rows)} member(s) modelled in the image plane: " + ", ".join(f"{r['id']} ({r['x']:.1f}, {r['y']:.1f}) m{r['mag']:.1f}" for r in rows))
    return rows


def ellipticity(u, v, cap=0.7):
    """Unconstrained (u, v) -> (e1, e2) with |e| < cap (axis ratio above (1 - cap) / (1 + cap) = 0.18)."""
    norm = jnp.sqrt(1.0 + u * u + v * v)
    return cap * u / norm, cap * v / norm


class ArcModel:
    """Parameter vector <-> jaxtronomy kwargs, forward model and chi^2, all jitted; the lens is a beta map argument."""

    def __init__(self, args, cut, psf, members, knots_beta, source_centre, scale):
        from jaxtronomy.LightModel.light_model import LightModel
        self.shape, self.s = cut["shape"], args.supersample
        self.scale = scale
        self.xy = jnp.asarray(cut["xy"])
        self.data, self.noise, self.mask = (jnp.asarray(cut[k]) for k in ("data", "noise", "mask"))
        self.psf = jnp.asarray(psf)
        self.members = members
        self.nmax = args.shapelet_nmax
        self.mode = args.source
        profiles = ["SERSIC_ELLIPSE"]
        if args.source == "knots":
            profiles += ["GAUSSIAN"] * len(knots_beta)
        elif args.source == "shapelets":
            profiles += ["SHAPELETS"]
        self.source = LightModel(profiles)
        self.lens_light = LightModel(["SERSIC_ELLIPSE"] * len(members)) if members else None
        self.knots_beta = knots_beta
        self.source_centre = np.asarray(source_centre)
        # Parameter layout: a flat unconstrained vector; names kept for the tables.
        names, start = [], []
        peak = float(np.nanpercentile(np.asarray(cut["data"])[np.asarray(cut["mask"])], 99.5))
        names += ["src_log_amp", "src_log_R", "src_log_n", "src_e1", "src_e2", "src_dx", "src_dy"]
        start += [np.log(max(peak, 1.0) / 5), np.log(0.15), np.log(1.0), 0.0, 0.0, 0.0, 0.0]
        if args.source == "knots":
            for k, (bx, by) in enumerate(knots_beta):
                names += [f"knot{k}_log_amp", f"knot{k}_log_sigma", f"knot{k}_dx", f"knot{k}_dy"]
                start += [np.log(max(peak, 1.0)), np.log(0.03), 0.0, 0.0]
        elif args.source == "shapelets":
            n_coeff = (self.nmax + 1) * (self.nmax + 2) // 2
            names += ["shp_log_beta"] + [f"shp_amp{k}" for k in range(n_coeff)]
            start += [np.log(0.1)] + [0.0] * n_coeff
        for m in members:
            names += [f"m{m['id']}_log_amp", f"m{m['id']}_log_R", f"m{m['id']}_log_n", f"m{m['id']}_e1", f"m{m['id']}_e2",
                      f"m{m['id']}_dx", f"m{m['id']}_dy"]
            start += [np.log(max(peak, 1.0)), np.log(0.5), np.log(4.0), 0.0, 0.0, 0.0, 0.0]
        self.names, self.start = names, np.asarray(start, dtype=float)
        self._chi2 = jax.jit(self.chi2)
        self._grad = jax.jit(jax.value_and_grad(self.chi2))
        self._render = jax.jit(self.render)

    # unconstrained -> jaxtronomy kwargs
    def unpack(self, p):
        i = 0
        def take(n):
            nonlocal i
            v = p[i:i + n]; i += n
            return v
        v = take(7)
        e1, e2 = ellipticity(v[3], v[4])
        source = [dict(amp=jnp.exp(v[0]), R_sersic=jnp.exp(v[1]), n_sersic=0.3 + 5.7 * jax.nn.sigmoid(v[2]), e1=e1, e2=e2,
                       center_x=self.source_centre[0] + v[5], center_y=self.source_centre[1] + v[6])]
        if self.mode == "knots":
            for bx, by in self.knots_beta:
                w = take(4)
                source.append(dict(amp=jnp.exp(w[0]), sigma=jnp.exp(w[1]), center_x=bx + w[2], center_y=by + w[3]))
        elif self.mode == "shapelets":
            n_coeff = (self.nmax + 1) * (self.nmax + 2) // 2
            w = take(1 + n_coeff)
            source.append(dict(amp=w[1:], beta=jnp.exp(w[0]), n_max=self.nmax, center_x=source[0]["center_x"], center_y=source[0]["center_y"]))
        lens_light = []
        for m in self.members:
            w = take(7)
            e1, e2 = ellipticity(w[3], w[4])
            lens_light.append(dict(amp=jnp.exp(w[0]), R_sersic=jnp.exp(w[1]), n_sersic=0.5 + 7.5 * jax.nn.sigmoid(w[2]),
                                   e1=e1, e2=e2, center_x=m["x"] + 0.3 * jnp.tanh(w[5]),
                                   center_y=m["y"] + 0.3 * jnp.tanh(w[6])))
        return source, lens_light

    def render(self, p, beta):
        source, lens_light = self.unpack(p)
        sb = self.source.surface_brightness(beta[:, 0], beta[:, 1], source)
        if self.lens_light is not None:
            sb = sb + self.lens_light.surface_brightness(self.xy[:, 0], self.xy[:, 1], lens_light)
        ny, nx = self.shape; s = self.s
        image = sb.reshape(ny, s, nx, s).mean(axis=(1, 3))
        return jax.scipy.signal.fftconvolve(image, self.psf, mode="same")

    def source_only(self, p, beta):
        source, _ = self.unpack(p)
        ny, nx = self.shape; s = self.s
        return self.source.surface_brightness(beta[:, 0], beta[:, 1], source).reshape(ny, s, nx, s).mean(axis=(1, 3))

    def chi2(self, p, beta):
        model = self.render(p, beta)
        r = jnp.where(self.mask, (self.data - model) / self.noise, 0.0)
        # Weak priors: unused amplitudes stay finite, and the Sersic radius stays within a decade of 0.2" (without it a
        # Sersic stretched along the arc direction at the ellipticity bound is a local optimum for the shapelet model).
        return jnp.sum(r * r) + 1e-3 * jnp.sum(p * p) + 0.5 * ((p[1] - jnp.log(0.2)) / 1.0) ** 2

    def fit(self, beta, start, adam_steps):
        beta = jnp.asarray(beta)
        p = jnp.asarray(start)
        opt = optax.adam(3e-2); st = opt.init(p)

        @jax.jit
        def step(p, st):
            value, grad = self._grad(p, beta)
            upd, st = opt.update(grad, st, p)
            return optax.apply_updates(p, upd), st, value
        for _ in range(adam_steps):
            p, st, _ = step(p, st)
        f = lambda q: tuple(np.asarray(x, dtype=float) for x in self._grad(jnp.asarray(q), beta))
        res = minimize(f, np.asarray(p), jac=True, method="L-BFGS-B", options=dict(maxiter=400))
        return res.x, float(res.fun)


class PixelModel:
    """Pixelated source on a regular grid, bilinearly interpolated at beta, binned and PSF-convolved: linear in the source,
    solved by conjugate gradients on the regularised normal equations.  Members are image-plane Sersics fit by L-BFGS,
    alternating with the linear solve."""

    def __init__(self, args, cut, psf, members, h, nx_s, ny_s, frame=None):
        from jaxtronomy.LightModel.light_model import LightModel
        self.shape, self.s = cut["shape"], args.supersample
        self.xy = jnp.asarray(cut["xy"])
        self.data, self.noise, self.mask = (jnp.asarray(cut[k]) for k in ("data", "noise", "mask"))
        self.weight = jnp.where(self.mask, 1.0 / self.noise ** 2, 0.0)
        self.psf = jnp.asarray(psf)
        self.members = members
        self.lens_light = LightModel(["SERSIC_ELLIPSE"] * len(members)) if members else None
        self.h, self.nx_s, self.ny_s = h, nx_s, ny_s
        # Source grid frame: unit vectors (columns) of its two axes and the pixel size along each; square by default.
        frame = frame or dict(axes=np.eye(2), pixel=(h, h))
        self.axes, self.pixel = jnp.asarray(frame["axes"]), jnp.asarray(frame["pixel"])
        self.reg = args.reg
        peak = float(np.nanpercentile(np.asarray(cut["data"])[np.asarray(cut["mask"])], 99.5))
        self.member_names, start = [], []
        for m in members:
            self.member_names += [f"m{m['id']}_{k}" for k in ("log_amp", "log_R", "log_n", "e1", "e2", "dx", "dy")]
            start += [np.log(max(peak, 1.0)), np.log(0.5), 0.0, 0.0, 0.0, 0.0, 0.0]
        self.member_start = np.asarray(start, dtype=float)
        self.edges = None          # mesh: (E, 2) node pairs; the "beta" argument is then (N, 6): 3 node ids, 3 weights
        self.reg_w = None          # square grid: sqrt of the edge weights of the gradient regulariser (rows, columns)
        self._forward = jax.jit(self.forward)
        self._members = jax.jit(self.render_members)
        self._solve = jax.jit(self.solve)
        self._member_loss = jax.jit(jax.value_and_grad(self.member_loss))

    def unpack_members(self, pm):
        out = []
        for j, m in enumerate(self.members):
            w = pm[7 * j:7 * j + 7]
            e1, e2 = ellipticity(w[3], w[4])
            out.append(dict(amp=jnp.exp(w[0]), R_sersic=jnp.exp(w[1]), n_sersic=0.5 + 7.5 * jax.nn.sigmoid(w[2]), e1=e1, e2=e2,
                            center_x=m["x"] + 0.3 * jnp.tanh(w[5]), center_y=m["y"] + 0.3 * jnp.tanh(w[6])))
        return out

    def bin_convolve(self, sb):
        ny, nx = self.shape; s = self.s
        return jax.scipy.signal.fftconvolve(sb.reshape(ny, s, nx, s).mean(axis=(1, 3)), self.psf, mode="same")

    def forward(self, source, beta, origin):
        if self.edges is not None:
            sb = jnp.sum(source[0][beta[:, :3].astype(jnp.int32)] * beta[:, 3:], axis=1)
            return self.bin_convolve(sb)
        uv = (beta - origin) @ self.axes / self.pixel
        coords = jnp.stack([uv[:, 1], uv[:, 0]])
        sb = jax.scipy.ndimage.map_coordinates(source, coords, order=1, mode="constant", cval=0.0)
        return self.bin_convolve(sb)

    def render_members(self, pm):
        if self.lens_light is None:
            return jnp.zeros(self.shape)
        return self.bin_convolve(self.lens_light.surface_brightness(self.xy[:, 0], self.xy[:, 1], self.unpack_members(pm)))

    def regulariser(self, source):
        if self.edges is not None:
            return source[0][self.edges[:, 0]] - source[0][self.edges[:, 1]]
        if self.reg == "gradient":
            dy, dx = source[1:, :] - source[:-1, :], source[:, 1:] - source[:, :-1]
            if self.reg_w is not None:
                dy, dx = dy * self.reg_w[0], dx * self.reg_w[1]
            return jnp.concatenate([dy.ravel(), dx.ravel()])
        lap = (-4 * source[1:-1, 1:-1] + source[:-2, 1:-1] + source[2:, 1:-1] + source[1:-1, :-2] + source[1:-1, 2:])
        return lap.ravel()

    def solve(self, pm, beta, origin, lam, maxiter=600):
        """Source minimising chi^2 + lam |R s|^2 at fixed members; returns the source and the CG residual norm."""
        residual = self.data - self._members(pm)
        forward = lambda src: self.forward(src, beta, origin)
        _, a_t = jax.vjp(forward, jnp.zeros((self.ny_s, self.nx_s)))
        _, r_t = jax.vjp(self.regulariser, jnp.zeros((self.ny_s, self.nx_s)))
        rhs = a_t(self.weight * residual)[0]
        ridge = 1e-6 * lam                                   # removes the regulariser's null space where the data say nothing
        operator = lambda src: a_t(self.weight * forward(src))[0] + lam * r_t(self.regulariser(src))[0] + ridge * src
        diag = a_t(self.weight * forward(jnp.ones((self.ny_s, self.nx_s))))[0] + 4.0 * lam + ridge      # see shape_solver
        source, _ = jax.scipy.sparse.linalg.cg(operator, rhs, tol=1e-7, maxiter=maxiter, M=lambda v: v / diag)
        return source, jnp.linalg.norm(operator(source) - rhs) / jnp.linalg.norm(rhs)

    def chi2(self, source, pm, beta, origin):
        model = self.forward(source, beta, origin) + self.render_members(pm)
        return jnp.sum(self.weight * (self.data - model) ** 2)

    def member_loss(self, pm, source, beta, origin):
        return self.chi2(source, pm, beta, origin) + 1e-3 * jnp.sum(pm * pm)

    def fit(self, beta, origin, lam, pm=None, rounds=4):
        beta, origin = jnp.asarray(beta), jnp.asarray(origin)
        pm = jnp.asarray(self.member_start if pm is None else pm)
        source = jnp.zeros((self.ny_s, self.nx_s))
        for _ in range(rounds if self.members else 1):
            source, cg = self._solve(pm, beta, origin, lam)
            if self.members:
                f = lambda q: tuple(np.asarray(v, dtype=float) for v in self._member_loss(jnp.asarray(q), source, beta, origin))
                pm = jnp.asarray(minimize(f, np.asarray(pm), jac=True, method="L-BFGS-B", options=dict(maxiter=200)).x)
        source, cg = self._solve(pm, beta, origin, lam)
        return np.asarray(source), np.asarray(pm), float(self.chi2(source, pm, beta, origin)), float(cg)

    def arc_chi2_nu(self, source, pm, beta, origin, arc):
        model = self.forward(source, beta, origin) + self.render_members(pm)
        r = (self.data - model) / self.noise
        return float(jnp.mean(jnp.where(arc, r * r, 0.0)) / max(arc.mean(), 1e-12))

    def choose_lambda(self, beta, origin, arc):
        """Largest lambda whose chi2_nu on the arc pixels is <= 1, or within 5% of the lowest value the scan reaches when that
        is above 1 (the least flexible source the arc allows; weaker lambdas then only fit noise).  The whole-cutout chi2
        is dominated by sky pixels and would accept any source."""
        rows = []
        for lam in np.logspace(-3, 3, 13):
            source, pm, chi2, _ = self.fit(beta, origin, lam, rounds=2)
            c = self.arc_chi2_nu(jnp.asarray(source), jnp.asarray(pm), jnp.asarray(beta), jnp.asarray(origin), jnp.asarray(arc))
            rows.append((lam, c, chi2 / float(self.mask.sum())))
            print(f"[lambda] {lam:9.3g}: arc chi2_nu {c:.3f}, cutout chi2_nu {rows[-1][2]:.3f}", flush=True)
        target = max(1.0, 1.05 * min(c for _, c, _ in rows))
        return max(lam for lam, c, _ in rows if c <= target), rows


def arc_region(cut, obs, radius):
    """Pixels within ``radius`` of the path through the catalog positions, ordered along their principal axis."""
    centre = obs.mean(0)
    axis = np.linalg.svd(obs - centre, full_matrices=False)[2][0] if len(obs) > 1 else np.array([1.0, 0.0])
    path = obs[np.argsort((obs - centre) @ axis)]
    px = np.column_stack([cut["xc"].ravel(), cut["yc"].ravel()])
    dist = np.full(len(px), np.inf)
    for a, b in zip(path[:-1], path[1:]) if len(path) > 1 else [(path[0], path[0])]:
        ab = b - a
        t = np.clip(((px - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        dist = np.minimum(dist, np.hypot(*(px - (a + t[:, None] * ab)).T))
    return (dist < radius).reshape(cut["xc"].shape)


def arc_pixels(cut, members):
    """Pixels above arc_sigma in the arc region and more than 1.2" from every modelled member: the arc."""
    data, noise, mask = (np.asarray(cut[k]) for k in ("data", "noise", "mask"))
    bright = mask & (data > cut["arc_sigma"] * noise) & cut["arc_region"]
    for m in members:
        bright &= np.hypot(cut["xc"] - m["x"], cut["yc"] - m["y"]) > 1.2
    return bright


def source_grid(args, cut, betas, members, max_px=400):
    """One source box per lens draw: the backprojected arc pixels of that draw plus a margin.  The box size (pixels) is
    common to all draws, the origin moves with the draw, so a lens draw shifts the source without resizing the grid."""
    bright = arc_pixels(cut, members)
    ny, nx = cut["shape"]; s = args.supersample
    boxes = []
    for b in betas:
        pts = np.column_stack([b[:, 0].reshape(ny, s, nx, s).mean((1, 3))[bright], b[:, 1].reshape(ny, s, nx, s).mean((1, 3))[bright]])
        if len(pts) < 10:
            raise SystemExit("fewer than 10 arc pixels above --arc-sigma; lower it, widen --arc-radius, or try another band")
        lo = np.percentile(pts, 0.5, axis=0); hi = np.percentile(pts, 99.5, axis=0)
        span = np.maximum(hi - lo, 0.02)
        boxes.append((lo - args.src_margin * span, hi + args.src_margin * span))
    h = args.src_pixel
    size = max(np.max(hi - lo) for lo, hi in boxes)
    if size / h > max_px:
        h = float(size / max_px)
        print(f"[source] grid capped at {max_px} px: source pixel raised to {h * 1000:.2f} mas")
    n = np.max([np.ceil((hi - lo) / h) for lo, hi in boxes], axis=0).astype(int)
    origins = [tuple(float(v) for v in (lo + hi) / 2 - n * h / 2) for lo, hi in boxes]
    print(f"[source] grid {n[1]} x {n[0]} px of {h * 1000:.2f} mas ({n[0] * h * 1000:.0f} x {n[1] * h * 1000:.0f} mas) from {bright.sum()} arc pixels; "
          f"origin of the best draw ({origins[0][0]:.3f}, {origins[0][1]:.3f}), spread over draws "
          f"{np.ptp(np.asarray(origins), axis=0).round(3)} arcsec")
    return origins, h, int(n[0]), int(n[1])


# ------------------------------------------------------------------------------------------------------------- outputs
def figure(out, args, cut, model, params, betas, det, tags, source_extent):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    rows = min(len(params), 4)
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.85 * rows), constrained_layout=True, squeeze=False)
    xc, yc = cut["xc"], cut["yc"]
    ext = [xc[0, 0], xc[0, -1], yc[0, 0], yc[-1, 0]]
    data = np.asarray(cut["data"]); vmax = np.nanpercentile(data[np.asarray(cut["mask"])], 99.7)
    ny, nx = cut["shape"]
    for r in range(rows):
        p, beta = params[r], betas[r]
        m = np.asarray(model._render(jnp.asarray(p), jnp.asarray(beta)))
        resid = np.where(cut["mask"], (data - m) / cut["noise"], np.nan)
        axes[r, 0].imshow(data, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
        axes[r, 1].imshow(m, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
        axes[r, 2].imshow(resid, origin="lower", extent=ext, vmin=-4, vmax=4, cmap="RdBu_r")
        # Source plane: the de-lensed source light on a regular grid around the source, with the caustic from det = 0.
        sx, sy = np.meshgrid(np.linspace(*source_extent[0], 160), np.linspace(*source_extent[1], 160))
        src = model.source.surface_brightness(jnp.asarray(sx.ravel()), jnp.asarray(sy.ravel()), model.unpack(jnp.asarray(p))[0])
        axes[r, 3].imshow(np.asarray(src).reshape(sx.shape), origin="lower", extent=[*source_extent[0], *source_extent[1]], cmap="magma")
        bx = beta[:, 0].reshape(ny, model.s, nx, model.s).mean((1, 3)); by = beta[:, 1].reshape(ny, model.s, nx, model.s).mean((1, 3))
        dd = det[r].reshape(ny, model.s, nx, model.s).mean((1, 3))
        try:
            axes[r, 2].contour(xc, yc, dd, levels=[0.0], colors="k", linewidths=0.5)
            cs = axes[r, 0].contour(xc, yc, dd, levels=[0.0], colors="w", linewidths=0.4)
            for path in cs.allsegs[0]:                      # map the critical curve to the caustic through the beta map
                ix = np.clip(np.round((path[:, 0] - xc[0, 0]) / (xc[0, 1] - xc[0, 0])).astype(int), 0, nx - 1)
                iy = np.clip(np.round((path[:, 1] - yc[0, 0]) / (yc[1, 0] - yc[0, 0])).astype(int), 0, ny - 1)
                axes[r, 3].plot(bx[iy, ix], by[iy, ix], ".", color="w", ms=0.6)
        except Exception:
            pass
        axes[r, 3].set_xlim(*source_extent[0]); axes[r, 3].set_ylim(*source_extent[1])
        axes[r, 0].text(0.03, 0.95, tags[r], transform=axes[r, 0].transAxes, color="w", fontsize=6, va="top")
        for ax in axes[r]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=5)
    for ax, t in zip(axes[0], ["data", "model", "(data - model)/noise", "source"]):
        ax.text(0.5, 1.03, t, transform=ax.transAxes, ha="center", fontsize=6)
    fig.supxlabel(r"$x$ [arcsec]", fontsize=7); fig.supylabel(r"$y$ [arcsec]", fontsize=7)
    fig.savefig(out)


def figure_pixels(out, cut, model, sources, members_pm, betas, origins, det, tags, extra=None):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    rows = min(len(sources), 4)
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.85 * rows), constrained_layout=True, squeeze=False)
    xc, yc = cut["xc"], cut["yc"]; ext = [xc[0, 0], xc[0, -1], yc[0, 0], yc[-1, 0]]
    data = np.asarray(cut["data"]); vmax = np.nanpercentile(data[np.asarray(cut["mask"])], 99.7)
    ny, nx = cut["shape"]; s = model.s
    for r in range(rows):
        x0, y0 = origins[r]
        sext = [x0, x0 + model.nx_s * model.h, y0, y0 + model.ny_s * model.h]
        m = np.asarray(model._forward(jnp.asarray(sources[r]), jnp.asarray(betas[r]), jnp.asarray(origins[r])) + model._members(jnp.asarray(members_pm[r])))
        if extra is not None:
            m = m + extra[r]
        resid = np.where(cut["mask"], (data - m) / cut["noise"], np.nan)
        axes[r, 0].imshow(data, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
        axes[r, 1].imshow(m, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
        axes[r, 2].imshow(resid, origin="lower", extent=ext, vmin=-4, vmax=4, cmap="RdBu_r")
        axes[r, 3].imshow(sources[r], origin="lower", extent=sext, cmap="magma", vmin=0)
        dd = det[r].reshape(ny, s, nx, s).mean((1, 3))
        bx = betas[r][:, 0].reshape(ny, s, nx, s).mean((1, 3)); by = betas[r][:, 1].reshape(ny, s, nx, s).mean((1, 3))
        try:
            axes[r, 2].contour(xc, yc, dd, levels=[0.0], colors="k", linewidths=0.5)
            cs = axes[r, 0].contour(xc, yc, dd, levels=[0.0], colors="w", linewidths=0.4)
            for path in cs.allsegs[0]:
                ix = np.clip(np.round((path[:, 0] - xc[0, 0]) / (xc[0, 1] - xc[0, 0])).astype(int), 0, nx - 1)
                iy = np.clip(np.round((path[:, 1] - yc[0, 0]) / (yc[1, 0] - yc[0, 0])).astype(int), 0, ny - 1)
                axes[r, 3].plot(bx[iy, ix], by[iy, ix], ".", color="w", ms=0.6)
        except Exception:
            pass
        axes[r, 3].set_xlim(sext[0], sext[1]); axes[r, 3].set_ylim(sext[2], sext[3])
        axes[r, 0].text(0.03, 0.95, tags[r], transform=axes[r, 0].transAxes, color="w", fontsize=6, va="top")
        for ax in axes[r]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=5)
    for ax, t in zip(axes[0], ["data", "model", "(data - model)/noise", "source"]):
        ax.text(0.5, 1.03, t, transform=ax.transAxes, ha="center", fontsize=6)
    fig.supxlabel(r"$x$ [arcsec]", fontsize=7); fig.supylabel(r"$y$ [arcsec]", fontsize=7)
    fig.savefig(out)


def figure_zoom(out, cut, model_image, knots_xy, knot_labels, width):
    """Data, model and residual along the arc at native resolution, rotated so the arc runs horizontally."""
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    data, noise, mask = (np.asarray(cut[k]) for k in ("data", "noise", "mask"))
    xc, yc, scale = cut["xc"], cut["yc"], cut["scale"]
    bright = mask & (data > 5 * noise)
    pts = np.column_stack([xc[bright], yc[bright]])
    centre = pts.mean(0); axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]; normal = np.array([-axis[1], axis[0]])
    along = (pts - centre) @ axis
    length = np.ptp(along) + 0.6
    u = np.arange(-length / 2, length / 2, scale); v = np.arange(-width / 2, width / 2, scale)
    uu, vv = np.meshgrid(u, v)
    gx = centre[0] + uu * axis[0] + vv * normal[0]; gy = centre[1] + uu * axis[1] + vv * normal[1]
    ix = (gx - xc[0, 0]) / (xc[0, 1] - xc[0, 0]); iy = (gy - yc[0, 0]) / (yc[1, 0] - yc[0, 0])
    sample = lambda img: map_coordinates(np.where(np.isfinite(img), img, 0.0), [iy, ix], order=1, cval=np.nan)
    d, m, n = sample(data), sample(model_image), sample(noise)
    vmax = np.nanpercentile(d, 99.8)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 0.7 + 3 * 7.2 * width / length), constrained_layout=True)
    ext = [u[0], u[-1], v[0], v[-1]]
    axes[0].imshow(d, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
    axes[1].imshow(m, origin="lower", extent=ext, vmin=-0.1 * vmax, vmax=vmax, cmap="magma")
    axes[2].imshow((d - m) / n, origin="lower", extent=ext, vmin=-4, vmax=4, cmap="RdBu_r")
    ku = (knots_xy - centre) @ axis; kv = (knots_xy - centre) @ normal
    for ax in axes[:2]:
        ax.scatter(ku, kv, s=28, facecolors="none", edgecolors="w", linewidths=0.5)
    for lab, a, b in zip(knot_labels, ku, kv):
        axes[0].annotate(lab, (a, b), xytext=(2, 3), textcoords="offset points", fontsize=4.5, color="w")
    for ax, t in zip(axes, ["data", "model", "(data - model)/noise"]):
        ax.text(0.01, 0.9, t, transform=ax.transAxes, color="w" if t != "(data - model)/noise" else "k", fontsize=6, va="top")
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
    fig.supxlabel("along the arc [arcsec]", fontsize=7); fig.supylabel("across [arcsec]", fontsize=7)
    fig.savefig(out)


def run_pixels(args, out, cut, psf, members, beta, det, draws, obs, src_labels):
    if args.grid_from:
        with np.load(args.grid_from) as f:
            if list(f["draws"]) != list(draws):
                raise SystemExit(f"--grid-from was made with draws {list(f['draws'])}, this run uses {list(draws)}; use the same --draws")
            origins = [tuple(float(v) for v in o) for o in f["origins"]]
            h = float(f["pixel"]); ny_s, nx_s = f["sources"].shape[1:]
        print(f"[source] grid {ny_s} x {nx_s} px of {h * 1000:.2f} mas reused from {args.grid_from}")
    else:
        origins, h, nx_s, ny_s = source_grid(args, cut, beta, members)
    model = PixelModel(args, cut, psf, members, h, nx_s, ny_s)
    n_pix = int(np.asarray(cut["mask"]).sum())
    arc = arc_pixels(cut, members)
    print(f"[fit] pixelated source {ny_s} x {nx_s} + {len(model.member_start)} member parameters, {n_pix} pixels "
          f"({arc.sum()} on the arc), {len(draws)} lens draws")
    lam = args.lam
    if lam <= 0:
        lam, scan = model.choose_lambda(jnp.asarray(beta[0]), origins[0], arc)
        pd.DataFrame(scan, columns=["lambda", "arc_chi2_nu", "chi2_nu"]).to_csv(out / "lambda_scan.csv", index=False)
        print(f"[lambda] using {lam:.3g}")
    rows, sources, pms = [], [], []
    for k, d in enumerate(draws):
        t0 = time.time()
        source, pm, chi2, cg = model.fit(beta[k], origins[k], lam, pm=None if k == 0 else pms[0])
        sources.append(source); pms.append(pm)
        m_src = np.asarray(model._forward(jnp.asarray(source), jnp.asarray(beta[k]), jnp.asarray(origins[k])))
        row = dict(draw=d, lam=lam, chi2=chi2, chi2_nu=chi2 / n_pix, cg_residual=cg, source_flux=float(source.sum() * h ** 2),
                   origin_x=origins[k][0], origin_y=origins[k][1],
                   negative_fraction=float((source < 0).mean()), seconds=time.time() - t0)
        for m, ll in zip(members, model.unpack_members(jnp.asarray(pm))):
            row.update({f"m{m['id']}_{k_}": float(v) for k_, v in ll.items()})
        for l, (ox, oy) in zip(src_labels, obs):                      # fold test: 0.5" apertures at the catalog positions
            ap = np.hypot(cut["xc"] - ox, cut["yc"] - oy) < 0.5
            floor = 3.0 * float(np.sqrt(np.sum(np.asarray(cut["noise"])[ap] ** 2)))
            row[f"flux_ratio_{l}"] = float(m_src[ap].sum() / max(np.asarray(cut["data"])[ap].sum(), floor))
        full = m_src + np.asarray(model._members(jnp.asarray(pm)))
        r = (np.asarray(cut["data"]) - full) / np.asarray(cut["noise"])
        row["arc_chi2_nu"] = float(np.mean(r[arc] ** 2)); row["arc_pixels"] = int(arc.sum())
        row["arc_rms_residual_sigma"] = float(np.sqrt(np.mean(r[arc] ** 2)))
        rows.append(row)
        print(f"[fit] draw {d}: chi2_nu {row['chi2_nu']:.3f}, arc chi2_nu {row['arc_chi2_nu']:.2f} on {arc.sum()} px, "
              f"cg {cg:.1e}, {row['seconds']:.0f} s", flush=True)
    table = pd.DataFrame(rows); table.to_csv(out / "fits_per_draw.csv", index=False)
    best_image = np.asarray(model._forward(jnp.asarray(sources[0]), jnp.asarray(beta[0]), jnp.asarray(origins[0])))
    np.savez_compressed(out / "pixel_sources.npz", sources=np.asarray(sources), members=np.asarray(pms), draws=np.asarray(draws),
                        origins=np.asarray(origins), pixel=h, member_names=np.asarray(model.member_names),
                        band=args.band.upper(), image_pixel=cut["scale"], arc_model=best_image,
                        extent=np.asarray([cut["xc"][0, 0], cut["xc"][0, -1], cut["yc"][0, 0], cut["yc"][-1, 0]]),
                        sky_reference=np.asarray(cut["sky_reference"]))
    numeric = table.drop(columns=["draw"]).select_dtypes(float)
    summary = pd.DataFrame(dict(best=numeric.iloc[0], median=numeric.median(), q16=numeric.quantile(0.16), q84=numeric.quantile(0.84)))
    summary.to_csv(out / "source_summary.csv"); print(summary.round(4).to_string())
    figure_pixels(out / "arc_fit.pdf", cut, model, sources, pms, beta, origins, det, [f"draw {d}" for d in draws])
    best_image = np.asarray(model._forward(jnp.asarray(sources[0]), jnp.asarray(beta[0]), jnp.asarray(origins[0])) + model._members(jnp.asarray(pms[0])))
    figure_zoom(out / "arc_zoom.pdf", cut, best_image, obs, src_labels, args.zoom_width)
    print(f"[done] outputs in {out}")


def shape_solver(models):
    """One source shape s0 for all bands at once: minimises sum_b chi2_b(a_b s0 + offset_b + members_b) + lam |R s0|^2, each
    band through its own beta map, binning and PSF.  The amplitudes a_b carry the colour and the pixel-area ratio; the
    offsets are model images held fixed (the smooth component)."""
    ref = models[0]
    zero = jnp.zeros((ref.ny_s, ref.nx_s))

    def solve(pms, betas, origin, amps, lam, offsets):
        _, r_t = jax.vjp(ref.regulariser, zero)
        rhs, ops = zero, []
        for m, pm, beta, a, off in zip(models, pms, betas, amps, offsets):
            fwd = lambda s, m=m, beta=beta: m.forward(s, beta, origin)
            _, a_t = jax.vjp(fwd, zero)
            rhs = rhs + a * a_t(m.weight * (m.data - m.render_members(pm) - off))[0]
            ops.append((fwd, a_t, m.weight, a))

        def operator(s):
            total = lam * r_t(ref.regulariser(s))[0] + 1e-6 * lam * s
            for fwd, a_t, w, a in ops:
                total = total + a * a * a_t(w * fwd(s))[0]
            return total
        # Jacobi-like preconditioner: row sums of A^T W A (entries are non-negative) plus the regulariser's typical degree.
        # Without it CG stalls on source pixels seen only at low magnification next to a highly magnified fold.
        ones = jnp.ones_like(zero)
        diag = 4.0 * lam + 1e-6 * lam
        for fwd, a_t, w, a in ops:
            diag = diag + a * a * a_t(w * fwd(ones))[0]
        s0, _ = jax.scipy.sparse.linalg.cg(operator, rhs, tol=1e-7, maxiter=2000, M=lambda v: v / diag)
        return s0, jnp.linalg.norm(operator(s0) - rhs) / jnp.linalg.norm(rhs)
    return jax.jit(solve)


def amplitude(model, s0, pm, beta, origin, offset=0.0):
    """Least-squares amplitude of the shape s0 (plus a fixed offset image per unit amplitude) in one band at fixed members."""
    m = model._forward(jnp.asarray(s0), jnp.asarray(beta), jnp.asarray(origin)) + offset
    d = model.data - model._members(jnp.asarray(pm))
    return float(jnp.sum(model.weight * m * d) / jnp.sum(model.weight * m * m))


class SmoothSource:
    """A Sersic in the source plane, evaluated at every supersampled pixel's beta and binned and PSF-convolved like the
    pixelated source, so it also predicts the galaxy's light outside the arc."""

    def __init__(self):
        from jaxtronomy.LightModel.light_model import LightModel
        self.light = LightModel(["SERSIC_ELLIPSE"])
        self.names = ["log_amp", "log_R", "n_raw", "e1_raw", "e2_raw", "center_x", "center_y"]

    def kwargs(self, ps):
        e1, e2 = ellipticity(ps[3], ps[4])
        return dict(amp=jnp.exp(ps[0]), R_sersic=jnp.exp(ps[1]), n_sersic=0.5 + 5.5 * jax.nn.sigmoid(ps[2]), e1=e1, e2=e2,
                    center_x=ps[5], center_y=ps[6])

    def sb(self, ps, x, y):
        return self.light.surface_brightness(x, y, [self.kwargs(ps)])

    def image(self, model, ps, beta):
        return model.bin_convolve(self.sb(ps, beta[:, 0], beta[:, 1]))


def grid_points(origin, frame, ny_s, nx_s):
    """Source-plane coordinates of the grid's pixel centres, (ny_s, nx_s, 2)."""
    jj, ii = np.meshgrid(np.arange(nx_s), np.arange(ny_s))
    uv = np.stack([jj * frame["pixel"][0], ii * frame["pixel"][1]], -1)
    return np.asarray(origin) + uv @ np.asarray(frame["axes"]).T


def aligned_grid(args, cut, betas, members, max_px=400):
    """Rectangular source pixels along and across the fold: from the Jacobian of the reference band's beta map over the
    arc pixels, the source-plane direction mapped to the stretched image direction gets pixels of (image pixel x small
    eigenvalue), the other (image pixel x large eigenvalue), both times --src-res.  One frame for all draws; the box
    (in pixels) is common, its origin moves with the draw."""
    bright = arc_pixels(cut, members)
    ny, nx = cut["shape"]; s = args.supersample; scale = cut["scale"]
    b = betas[0].reshape(ny, s, nx, s, 2).mean((1, 3))
    dbx_dy, dbx_dx = np.gradient(b[..., 0], scale)
    dby_dy, dby_dx = np.gradient(b[..., 1], scale)
    jac = np.stack([np.stack([dbx_dx, dbx_dy], -1), np.stack([dby_dx, dby_dy], -1)], -2)[bright]
    jac = 0.5 * (jac + np.swapaxes(jac, 1, 2))
    vals, vecs = np.linalg.eigh(jac)
    small = np.argmin(np.abs(vals), 1)
    rows = np.arange(len(vals))
    lam_t, lam_r = np.abs(vals[rows, small]), np.abs(vals[rows, 1 - small])
    v = vecs[rows, :, small]
    phi = 0.5 * np.angle(np.mean(np.exp(2j * np.arctan2(v[:, 1], v[:, 0]))))     # mean axis direction (sign-free)
    axes = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])     # columns: along, across
    hu = scale * float(np.median(lam_t)) * args.src_res
    hv = scale * float(np.median(lam_r)) * args.src_res
    print(f"[source] aligned grid: axis at {np.degrees(phi):.0f} deg, median eigenvalues {np.median(lam_t):.3f} and "
          f"{np.median(lam_r):.3f} (magnifications {1 / np.median(lam_t):.0f} and {1 / np.median(lam_r):.1f})")
    boxes = []
    for bb in betas:
        pts = np.column_stack([bb[:, 0].reshape(ny, s, nx, s).mean((1, 3))[bright], bb[:, 1].reshape(ny, s, nx, s).mean((1, 3))[bright]])
        uv = pts @ axes
        lo = np.percentile(uv, 0.5, axis=0); hi = np.percentile(uv, 99.5, axis=0)
        span = np.maximum(hi - lo, 0.02)
        boxes.append((lo - args.src_margin * span, hi + args.src_margin * span))
    size = np.max([hi - lo for lo, hi in boxes], axis=0)
    h = np.array([hu, hv])
    h = np.maximum(h, size / max_px)
    n = np.ceil(size / h).astype(int)
    origins = [tuple(float(x) for x in axes @ ((lo + hi) / 2 - n * h / 2)) for lo, hi in boxes]
    print(f"[source] aligned grid {n[0]} x {n[1]} px of {h[0] * 1000:.2f} x {h[1] * 1000:.2f} mas "
          f"({size[0] * 1000:.0f} x {size[1] * 1000:.0f} mas) from {bright.sum()} arc pixels")
    return origins, dict(axes=axes, pixel=(float(h[0]), float(h[1]))), int(n[0]), int(n[1])


def display_grid(origins, frame, ny_s, nx_s, smooth_ps=None, max_px=400):
    """A square-pixel grid covering the source grid (and 2.5 effective radii of the smooth component) for every draw,
    common in size, its origin per draw."""
    corners = []
    for o in origins:
        g = grid_points(o, frame, ny_s, nx_s)
        lo, hi = g.reshape(-1, 2).min(0), g.reshape(-1, 2).max(0)
        if smooth_ps is not None:
            c = np.asarray(smooth_ps[len(corners)][5:7]); r = 2.5 * float(np.exp(smooth_ps[len(corners)][1]))
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        corners.append((lo, hi))
    size = np.max([hi - lo for lo, hi in corners], axis=0)
    h = max(min(frame["pixel"]), float(size.max()) / max_px)
    n = np.ceil(size / h).astype(int)
    return [tuple((lo + hi) / 2 - n * h / 2) for lo, hi in corners], h, int(n[1]), int(n[0])


def to_display(source, origin, frame, d_origin, h, ny, nx, smooth=None, smooth_ps=None, amp=1.0):
    """Resample a grid source to the display grid; add the smooth component (times the band amplitude) if any."""
    from scipy.ndimage import map_coordinates
    pts = grid_points(d_origin, dict(axes=np.eye(2), pixel=(h, h)), ny, nx).reshape(-1, 2)
    uv = (pts - np.asarray(origin)) @ np.asarray(frame["axes"]) / np.asarray(frame["pixel"])
    out = map_coordinates(np.asarray(source), [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=0.0)
    if smooth is not None:
        out = out + amp * np.asarray(smooth.sb(jnp.asarray(smooth_ps), jnp.asarray(pts[:, 0]), jnp.asarray(pts[:, 1])))
    return out.reshape(ny, nx)


def run_colour(args, out, setup, obs, src_labels):
    """Pixelated source in several bands sharing one morphology.  (1) Each band is fit alone at the shape lambda, for its
    members and a first amplitude.  (2) One shape is fit to all bands together, alternating with the amplitudes; this is
    the deep stack, built in the source plane so each band keeps its PSF and pixel.  With --smooth sersic the shape is a
    Sersic (evaluated at beta everywhere in the cutout) plus the pixelated residual.  (3) Each band is refit as a_b times
    the shape plus a residual whose regulariser is --lam-colour, so a band departs from the shared shape only where its
    data ask for it."""
    from copy import copy
    bands = [b.strip().upper() for b in args.bands.split(",") if b.strip()]
    reference = args.band.upper()
    bands = [reference] + [b for b in bands if b != reference]
    draws = setup["draws"]
    per = {}
    for b in bands:
        a = copy(args); a.band = b
        print(f"=== {b}", flush=True)
        per[b] = dict(args=a, **dict(zip(("cut", "psf", "members", "beta", "det"), prepare_band(a, out, **setup))))
    ref = per[reference]
    if args.grid_from:
        with np.load(args.grid_from) as f:
            origins = [tuple(float(v) for v in o) for o in f["grid_origins"]]
            frame = dict(axes=f["grid_axes"], pixel=tuple(f["grid_pixel"])); ny_s, nx_s = f["grid_sources"].shape[1:]
    elif args.src_grid == "aligned":
        origins, frame, nx_s, ny_s = aligned_grid(args, ref["cut"], ref["beta"], ref["members"])
    else:
        origins, h, nx_s, ny_s = source_grid(args, ref["cut"], ref["beta"], ref["members"])
        frame = dict(axes=np.eye(2), pixel=(h, h))
    for b in bands:
        p = per[b]
        p["model"] = PixelModel(p["args"], p["cut"], p["psf"], p["members"], frame["pixel"][0], nx_s, ny_s, frame)
        p["arc"] = arc_pixels(p["cut"], p["members"])
    lam = args.lam
    if lam <= 0:
        lam, scan = ref["model"].choose_lambda(jnp.asarray(ref["beta"][0]), origins[0], ref["arc"])
        pd.DataFrame(scan, columns=["lambda", "arc_chi2_nu", "chi2_nu"]).to_csv(out / "lambda_scan_shape.csv", index=False)
    print(f"[shape] lambda {lam:.3g}")
    solve_shape = shape_solver([per[b]["model"] for b in bands])
    smooth = SmoothSource() if args.smooth == "sersic" else None
    if smooth is not None:
        def smooth_loss(ps, r, pms, betas, origin, amps):
            total = 0.0
            for m, pm, beta, a in zip([per[b]["model"] for b in bands], pms, betas, amps):
                model = a * (smooth.image(m, ps, beta) + m.forward(r, beta, origin)) + m.render_members(pm)
                total = total + jnp.sum(m.weight * (m.data - model) ** 2)
            return total + jnp.sum((ps[1:5] - jnp.asarray([np.log(0.1), 0.0, 0.0, 0.0])) ** 2)   # weak prior: R ~ 0.1", round
        smooth_vg = jax.jit(jax.value_and_grad(smooth_loss))
        smooth_images = jax.jit(lambda ps, betas: [smooth.image(per[b]["model"], ps, beta) for b, beta in zip(bands, betas)])

    rows, shapes, amps_all, smooth_all = [], [], [], []
    lam_colour = {b: 0.0 for b in bands}
    for item in args.lam_colour.split(","):
        key, _, value = item.rpartition("=")
        for b in ([key.strip().upper()] if key else bands):
            lam_colour[b] = float(value)
    for b in bands:
        per[b]["sources"], per[b]["pms"], per[b]["display"], per[b]["extra"] = [], [], [], []
    for k, d in enumerate(draws):
        t0 = time.time()
        beta = {b: jnp.asarray(per[b]["beta"][k]) for b in bands}
        blist = [beta[b] for b in bands]
        origin = jnp.asarray(origins[k])
        # (1) bands alone
        alone, pms = {}, {}
        for b in bands:
            start = None if k == 0 else per[b]["pms"][0]
            alone[b], pms[b], _, _ = per[b]["model"].fit(beta[b], origin, lam, pm=start)
        pml = [jnp.asarray(pms[b]) for b in bands]
        flux = {b: float(alone[b].sum()) for b in bands}
        amps = {b: flux[b] / flux[reference] for b in bands}
        zeros = [jnp.zeros(per[b]["model"].shape) for b in bands]
        # (2) shared shape, alternating with the amplitudes; the reference amplitude stays 1
        for _ in range(3):
            s0, cg = solve_shape(pml, blist, origin, jnp.asarray([amps[b] for b in bands]), lam, zeros)
            fitted = {b: amplitude(per[b]["model"], s0, pms[b], beta[b], origin) for b in bands}
            amps = {b: fitted[b] / fitted[reference] for b in bands}
            s0 = s0 * fitted[reference]
        ps, offsets = None, zeros
        if smooth is not None:
            # Sersic start: the light-weighted centre and size of the pixelated shape.
            g = grid_points(origins[k], frame, ny_s, nx_s); w = np.clip(np.asarray(s0), 0, None)
            c = (g * w[..., None]).sum((0, 1)) / w.sum()
            r = np.sqrt(((g - c) ** 2 * w[..., None]).sum((0, 1)) / w.sum()).mean()
            ps = np.array([np.log(np.percentile(w[w > 0], 90)), np.log(max(r, 0.02)), 0.0, 0.0, 0.0, c[0], c[1]]) if k == 0 \
                else smooth_all[0].copy()
            if k > 0:
                ps[5:7] += np.asarray(origins[k]) - np.asarray(origins[0])
            r_pix = jnp.zeros((ny_s, nx_s))
            amp_arr = jnp.asarray([amps[b] for b in bands])
            for _ in range(2):
                f = lambda q: tuple(np.asarray(v, dtype=float) for v in smooth_vg(jnp.asarray(q), r_pix, pml, blist, origin, amp_arr))
                ps = minimize(f, ps, jac=True, method="L-BFGS-B", options=dict(maxiter=400)).x
                imgs = smooth_images(jnp.asarray(ps), blist)
                offsets = [a * im for a, im in zip(amp_arr, imgs)]
                r_pix, cg = solve_shape(pml, blist, origin, amp_arr, lam, offsets)
            s0 = np.asarray(r_pix)
            kw = smooth.kwargs(jnp.asarray(ps))
            print(f"[smooth] draw {d}: Sersic R {float(kw['R_sersic']) * 1000:.0f} mas, n {float(kw['n_sersic']):.2f}, "
                  f"e ({float(kw['e1']):.2f}, {float(kw['e2']):.2f}), centre ({ps[5]:.3f}, {ps[6]:.3f})", flush=True)
            smooth_all.append(np.asarray(ps))
        s0 = np.asarray(s0)
        shapes.append(s0); amps_all.append([amps[b] for b in bands])
        print(f"[shape] draw {d}: amplitudes " + ", ".join(f"{b} {amps[b]:.3f}" for b in bands) + f", cg {float(cg):.1e}", flush=True)
        # (3) each band = a_b (smooth + s0) + residual
        for i, b in enumerate(bands):
            p = per[b]
            extra = np.asarray(offsets[i])                     # already times a_b
            shaped = amps[b] * np.asarray(p["model"]._forward(jnp.asarray(s0), beta[b], origin)) + extra
            rcut = dict(p["cut"], data=np.where(p["cut"]["mask"], np.asarray(p["cut"]["data"]) - shaped, 0.0))
            rmodel = PixelModel(p["args"], rcut, p["psf"], p["members"], frame["pixel"][0], nx_s, ny_s, frame)
            if lam_colour[b] <= 0:
                lam_colour[b], scan = rmodel.choose_lambda(beta[b], origin, p["arc"])
                pd.DataFrame(scan, columns=["lambda", "arc_chi2_nu", "chi2_nu"]).to_csv(out / f"lambda_scan_{b}.csv", index=False)
                print(f"[colour] {b}: lambda {lam_colour[b]:.3g}")
            resid, pm, _, _ = rmodel.fit(beta[b], origin, lam_colour[b], pm=pms[b])
            source = amps[b] * s0 + resid
            p["sources"].append(source); p["pms"].append(pm); p["extra"].append(extra)
            arc = jnp.asarray(p["arc"])
            c_shape = rmodel.arc_chi2_nu(jnp.zeros_like(jnp.asarray(resid)), jnp.asarray(pm), beta[b], origin, arc)
            c = rmodel.arc_chi2_nu(jnp.asarray(resid), jnp.asarray(pm), beta[b], origin, arc)
            c_alone = p["model"].arc_chi2_nu(jnp.asarray(alone[b]), jnp.asarray(pms[b]), beta[b], origin, arc)
            rows.append(dict(draw=d, band=b, lam_shape=lam, lam_colour=lam_colour[b], amplitude=amps[b],
                             arc_chi2_nu_alone=c_alone, arc_chi2_nu_shape=c_shape, arc_chi2_nu=c, arc_pixels=int(p["arc"].sum())))
            print(f"[colour] draw {d} {b}: arc chi2_nu alone {c_alone:.2f}, shape only {c_shape:.2f}, shape + residual {c:.2f}", flush=True)
        print(f"[colour] draw {d} in {time.time() - t0:.0f} s", flush=True)

    d_origins, dh, dny, dnx = display_grid(origins, frame, ny_s, nx_s, smooth_all if smooth is not None else None)
    print(f"[display] {dny} x {dnx} px of {dh * 1000:.2f} mas")
    pd.DataFrame(rows).to_csv(out / "colour_fits.csv", index=False)
    np.savez_compressed(out / "shape.npz", shapes=np.asarray(shapes), amplitudes=np.asarray(amps_all), bands=np.asarray(bands),
                        draws=np.asarray(draws), origins=np.asarray(origins), axes=np.asarray(frame["axes"]),
                        pixel=np.asarray(frame["pixel"]), smooth=np.asarray(smooth_all), smooth_names=np.asarray(smooth.names if smooth else []))
    for i, b in enumerate(bands):
        p = per[b]; cut = p["cut"]; folder = out / b; folder.mkdir(exist_ok=True)
        disp = [to_display(p["sources"][k], origins[k], frame, d_origins[k], dh, dny, dnx, smooth,
                           smooth_all[k] if smooth is not None else None, amps_all[k][i]) for k in range(len(draws))]
        best_image = np.asarray(p["model"]._forward(jnp.asarray(p["sources"][0]), jnp.asarray(p["beta"][0]), jnp.asarray(origins[0]))) + p["extra"][0]
        np.savez_compressed(folder / "pixel_sources.npz", sources=np.asarray(disp), members=np.asarray(p["pms"]),
                            draws=np.asarray(draws), origins=np.asarray(d_origins), pixel=dh, member_names=np.asarray(p["model"].member_names),
                            band=b, image_pixel=cut["scale"], arc_model=best_image,
                            extent=np.asarray([cut["xc"][0, 0], cut["xc"][0, -1], cut["yc"][0, 0], cut["yc"][-1, 0]]),
                            sky_reference=np.asarray(cut["sky_reference"]), grid_sources=np.asarray(p["sources"]),
                            grid_origins=np.asarray(origins), grid_axes=np.asarray(frame["axes"]), grid_pixel=np.asarray(frame["pixel"]))
        figure_pixels(folder / "arc_fit.pdf", cut, p["model"], p["sources"], p["pms"], p["beta"], origins, p["det"],
                      [f"draw {d}" for d in draws], extra=p["extra"])
    print(f"[done] outputs in {out}; colour image: python scripts/make_source_rgb.py "
          + " ".join(str(out / b) for b in bands) + f" --out {out / 'source_rgb.pdf'}")


def union_grid(args, groups, max_px=400):
    """Square source grid covering the backprojected arc pixels of several regions (each (cut, betas, members), betas
    already in the common frame); one box size for all draws, origin per draw."""
    s = args.supersample
    boxes = []
    for k in range(len(groups[0][1])):
        pts = []
        for cut, betas, members in groups:
            bright = arc_pixels(cut, members); ny, nx = cut["shape"]; b = betas[k]
            pts.append(np.column_stack([b[:, 0].reshape(ny, s, nx, s).mean((1, 3))[bright], b[:, 1].reshape(ny, s, nx, s).mean((1, 3))[bright]]))
        pts = np.concatenate(pts)
        lo = np.percentile(pts, 0.5, axis=0); hi = np.percentile(pts, 99.5, axis=0)
        span = np.maximum(hi - lo, 0.02)
        boxes.append((lo - args.src_margin * span, hi + args.src_margin * span))
    size = max(np.max(hi - lo) for lo, hi in boxes)
    h = max(args.src_pixel, float(size) / max_px)
    n = np.max([np.ceil((hi - lo) / h) for lo, hi in boxes], axis=0).astype(int)
    origins = [tuple(float(v) for v in (lo + hi) / 2 - n * h / 2) for lo, hi in boxes]
    print(f"[source] joint grid {n[1]} x {n[0]} px of {h * 1000:.2f} mas ({n[0] * h * 1000:.0f} x {n[1] * h * 1000:.0f} mas)")
    return origins, h, int(n[0]), int(n[1])


def counter_offset(args, fold, counter, k, lam_counter=1.0, search=0.8):
    """Source-plane offset delta (counter frame + delta = fold frame) between a counter-image and the fold pair, from the
    reference band: the counter-image's own source (8 mas pixels) is slid over the fold arc and delta minimises the
    fold's arc chi2 with a free amplitude.  Coarse grid, then a fine one around the best point."""
    from copy import copy
    a = copy(args); a.src_pixel = 0.008
    beta_c = counter["beta"][k:k + 1]
    oc, hc, nxc, nyc = source_grid(a, counter["cut"], beta_c, counter["members"])
    mc = PixelModel(a, counter["cut"], counter["psf"], counter["members"], hc, nxc, nyc)
    s_c, _, _, _ = mc.fit(beta_c[0], oc[0], lam_counter)
    mf = PixelModel(a, fold["cut"], fold["psf"], fold["members"], hc, nxc, nyc)
    w = mf.weight * jnp.asarray(fold["arc"])
    d = mf.data
    beta_f = jnp.asarray(fold["beta"][k]); s_c = jnp.asarray(s_c); o = jnp.asarray(oc[0])

    @jax.jit
    def chi2(deltas):
        def one(delta):
            m = mf.forward(s_c, beta_f, o + delta)
            amp = jnp.maximum(jnp.sum(w * m * d) / jnp.maximum(jnp.sum(w * m * m), 1e-30), 0.0)
            return jnp.sum(w * (d - amp * m) ** 2)
        return jax.vmap(one)(deltas)

    def scan(centre, half, step):
        g = np.arange(-half, half + step / 2, step)
        grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + centre
        vals = np.concatenate([np.asarray(chi2(jnp.asarray(grid[i:i + 64]))) for i in range(0, len(grid), 64)])
        return grid[np.argmin(vals)], float(vals.min()), float(np.median(vals))
    best, c1, med = scan(np.zeros(2), search, 0.02)
    best, c2, _ = scan(best, 0.03, 0.003)
    centre = np.asarray(fold["beta"][k])[np.repeat(np.asarray(fold["arc"]).ravel(), args.supersample ** 2)].mean(0) \
        if args.supersample == 1 else np.asarray(fold["beta"][k]).mean(0)
    if not args.affine:
        return best, np.zeros((2, 2)), centre, c2, med
    # Affine refinement: fold beta -> counter frame as (I + E)(beta - c) + c - delta, E = 0.5 tanh(raw) (bounded).
    c = jnp.asarray(centre)

    def loss(q):
        e = 0.5 * jnp.tanh(q[:4]).reshape(2, 2)
        mapped = (beta_f - c) @ (jnp.eye(2) + e).T + c - q[4:]
        m = mf.forward(s_c, mapped, o)
        amp = jnp.maximum(jnp.sum(w * m * d) / jnp.maximum(jnp.sum(w * m * m), 1e-30), 0.0)
        return jnp.sum(w * (d - amp * m) ** 2)
    vg = jax.jit(jax.value_and_grad(loss))
    q0 = np.concatenate([np.zeros(4), best])
    q = minimize(lambda q: tuple(np.asarray(v, dtype=float) for v in vg(jnp.asarray(q))), q0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=300)).x
    c3 = float(loss(jnp.asarray(q)))
    return q[4:], np.asarray(0.5 * np.tanh(q[:4])).reshape(2, 2), centre, c3, med


def traced_arc_points(args, u, k, stride=1):
    """Source-plane positions (draw k) of the arc-region pixels of a unit, at the image pixel (supersamples averaged)."""
    cut = u["cut"]; ny, nx = cut["shape"]; s = args.supersample
    b = u["beta"][k].reshape(ny, s, nx, s, 2).mean((1, 3))
    keep = np.zeros((ny, nx), bool); keep[::stride, ::stride] = True
    return b[cut["arc_region"] & keep]


def resolution_weights(args, ref_units, k, origin, h, ny_s, nx_s, wmax=1e3):
    """sqrt edge weights of the gradient regulariser: 1/(image pixels per source pixel), smoothed over 2 source pixels and
    clipped to [1, wmax], so the penalty is per resolution element rather than per source pixel."""
    from scipy.ndimage import gaussian_filter
    pts = np.concatenate([traced_arc_points(args, u, k) for u in ref_units])
    ex = origin[0] - h / 2 + np.arange(nx_s + 1) * h; ey = origin[1] - h / 2 + np.arange(ny_s + 1) * h
    rho = gaussian_filter(np.histogram2d(pts[:, 1], pts[:, 0], bins=[ey, ex])[0], 2.0)
    w = np.clip(1.0 / np.maximum(rho, 1.0 / wmax), 1.0, wmax)
    print(f"[reg] resolution weights: median {np.median(w):.1f}, 10-90% {np.percentile(w, 10):.1f}-{np.percentile(w, 90):.1f}")
    return jnp.asarray(np.sqrt(np.sqrt(w[1:, :] * w[:-1, :]))), jnp.asarray(np.sqrt(np.sqrt(w[:, 1:] * w[:, :-1])))


def build_mesh(args, ref_units, units, k):
    """Delaunay mesh on k-means centres of the traced arc-region pixels of the reference band, and for every unit the three
    nodes and barycentric weights of each supersampled pixel's beta (zero weight outside the mesh).  Nodes placed on the
    traced pixels themselves form dense lines along a fold with long thin triangles between them, which stripe the source."""
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import Delaunay
    pts = np.concatenate([traced_arc_points(args, u, k) for u in ref_units])
    nodes, _ = kmeans2(pts, min(args.mesh_nodes, len(pts) // 2), minit="++", seed=0)
    tri = Delaunay(nodes)
    e = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]])
    edges = np.unique(np.sort(e, axis=1), axis=0)
    packed = {}
    for u in units:
        pts = u["beta"][k]
        simp = tri.find_simplex(pts)
        tr = tri.transform[np.maximum(simp, 0)]
        b01 = np.einsum("nij,nj->ni", tr[:, :2], pts - tr[:, 2])
        w = np.column_stack([b01, 1 - b01.sum(1)])
        idx = tri.simplices[np.maximum(simp, 0)]
        w[simp < 0] = 0.0
        packed[id(u)] = np.column_stack([idx.astype(float), w])
        ny, nx = u["cut"]["shape"]; s = args.supersample
        outside = (simp < 0).reshape(ny, s, nx, s).all((1, 3))[u["arc"]].mean()
        if outside > 0.01:
            print(f"[mesh] {u['band']} {u['region']}: {outside:.0%} of the arc pixels trace outside the mesh")
    print(f"[mesh] draw index {k}: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges, packed


def mesh_display(nodes_all, values_all, max_px=400):
    """Square display grids (common size, origin per draw) for mesh sources, by linear interpolation on the nodes."""
    from scipy.interpolate import LinearNDInterpolator
    boxes = [(np.percentile(n, 0.5, axis=0), np.percentile(n, 99.5, axis=0)) for n in nodes_all]
    size = np.max([hi - lo for lo, hi in boxes], axis=0)
    h = float(size.max()) / max_px
    n = np.ceil(size / h).astype(int)
    origins = [tuple((lo + hi) / 2 - n * h / 2) for lo, hi in boxes]
    out = []
    for nodes, vals, o in zip(nodes_all, values_all, origins):
        pts = grid_points(o, dict(axes=np.eye(2), pixel=(h, h)), int(n[1]), int(n[0])).reshape(-1, 2)
        out.append(LinearNDInterpolator(nodes, np.ravel(vals), fill_value=0.0)(pts).reshape(int(n[1]), int(n[0])))
    return out, origins, h


def run_joint(args, out, setup, obs, src_labels):
    """One source behind several regions of the image plane (the fold pair plus counter-images given by --add), in several
    bands.  Every (band, region) is a unit with its own cutout, PSF, members and beta map; the shape and each band's
    residual are shared by all regions of that band.  Each added region gets a source-plane offset delta (lens error)."""
    from copy import copy
    bands = [b.strip().upper() for b in args.bands.split(",") if b.strip()]
    reference = args.band.upper()
    bands = [reference] + [b for b in bands if b != reference]
    regions = [dict(name="fold", obs=setup["obs_xy"], centre=setup["centre"], size=args.size, radius=args.arc_radius)]
    for j, spec in enumerate(args.add.split(";")):
        x, y, size, radius = (float(v) for v in spec.split(","))
        regions.append(dict(name=f"c{j + 1}", obs=np.asarray([[x, y]]), centre=(x, y), size=size, radius=radius))
    draws = setup["draws"]
    units = []
    for r in regions:
        for b in bands:
            a = copy(args); a.band = b; a.size = r["size"]; a.arc_radius = r["radius"]; a.region_tag = r["name"]
            print(f"=== {b} {r['name']}", flush=True)
            s = dict(setup, obs_xy=r["obs"], centre=r["centre"])
            u = dict(band=b, region=r["name"], args=a, **dict(zip(("cut", "psf", "members", "beta", "det"), prepare_band(a, out, **s))))
            u["beta"] = np.array(u["beta"]); u["arc"] = arc_pixels(u["cut"], u["members"])
            units.append(u)
    unit = {(u["band"], u["region"]): u for u in units}
    fold_ref = unit[(reference, "fold")]

    # Offsets of the added regions, per draw, from the reference band; applied to every band's beta map of that region.
    deltas = {r["name"]: [] for r in regions[1:]}
    affines = {r["name"]: [] for r in regions[1:]}
    for r in regions[1:]:
        c_ref = unit[(reference, r["name"])]
        a_jac = []
        for k, d in enumerate(draws):
            delta, e, centre, c_best, c_med = counter_offset(args, fold_ref, c_ref, k)
            deltas[r["name"]].append(delta); affines[r["name"]].append((e, centre))
            if args.affine:
                m = np.eye(2) + e
                g1, g2 = (m[0, 0] - m[1, 1]) / 2, (m[0, 1] + m[1, 0]) / 2
                print(f"[affine] {r['name']} draw {d}: det {np.linalg.det(m):.3f}, shear ({g1:+.3f}, {g2:+.3f}), "
                      f"rotation {np.degrees((m[1, 0] - m[0, 1]) / 2):+.1f} deg", flush=True)
            # The image-plane shift that delta corresponds to at the counter-image: A^-1 delta, A from its beta map.
            cut = c_ref["cut"]; ny, nx = cut["shape"]; s = args.supersample
            bm = c_ref["beta"][k].reshape(ny, s, nx, s, 2).mean((1, 3))
            iy, ix = np.unravel_index(np.argmin(np.hypot(cut["xc"] - r["centre"][0], cut["yc"] - r["centre"][1])), cut["xc"].shape)
            jac = np.array([[np.gradient(bm[..., 0], cut["scale"], axis=1)[iy, ix], np.gradient(bm[..., 0], cut["scale"], axis=0)[iy, ix]],
                            [np.gradient(bm[..., 1], cut["scale"], axis=1)[iy, ix], np.gradient(bm[..., 1], cut["scale"], axis=0)[iy, ix]]])
            shift = np.linalg.solve(jac, delta)
            print(f"[offset] {r['name']} draw {d}: source-plane delta ({delta[0] * 1000:+.0f}, {delta[1] * 1000:+.0f}) mas = image-plane "
                  f"({shift[0]:+.2f}, {shift[1]:+.2f})\" at the counter-image; fold arc chi2 {c_best:.0f} at delta vs median {c_med:.0f} over the search",
                  flush=True)
        for b in bands:
            u = unit[(b, r["name"])]
            for k in range(len(draws)):
                # counter frame -> fold frame: inverse of (I + E)(p_f - c) + c - delta
                e, centre = affines[r["name"]][k]
                u["beta"][k] = (u["beta"][k] + deltas[r["name"]][k] - centre) @ np.linalg.inv(np.eye(2) + e).T + centre

    ref_units = [unit[(reference, r["name"])] for r in regions]
    mesh = args.src_grid == "mesh"
    by_band = {b: [u for u in units if u["band"] == b] for b in bands}
    nodes_all = []
    if not mesh:
        origins, h, nx_s, ny_s = union_grid(args, [(u["cut"], u["beta"], u["members"]) for u in ref_units])
        frame = dict(axes=np.eye(2), pixel=(h, h))
        reg_w = resolution_weights(args, ref_units, 0, origins[0], h, ny_s, nx_s) if args.reg_weight == "resolution" else None
        for u in units:
            u["model"] = PixelModel(u["args"], u["cut"], u["psf"], u["members"], h, nx_s, ny_s)
            u["model"].reg_w = reg_w

    def models_for(k):
        """Per-draw forward inputs; the mesh (and so every model and solver) is rebuilt for each draw."""
        if mesh:
            nodes, edges, packed = build_mesh(args, ref_units, units, k)
            nodes_all.append(nodes)
            for u in units:
                u["model"] = PixelModel(u["args"], u["cut"], u["psf"], u["members"], 1.0, len(nodes), 1)
                u["model"].edges = jnp.asarray(edges)
                u["fwd"] = jnp.asarray(packed[id(u)])
            return jnp.zeros(2)
        for u in units:
            u["fwd"] = jnp.asarray(u["beta"][k])
        return jnp.asarray(origins[k])
    lam = args.lam
    solve_all = solve_band = None

    def arc_chi2_nu(us, images):
        num = sum(float(np.sum(((np.asarray(u["cut"]["data"]) - im) / np.asarray(u["cut"]["noise"]))[u["arc"]] ** 2)) for u, im in zip(us, images))
        return num / sum(int(u["arc"].sum()) for u in us)

    rows, shapes, amps_all = [], [], []
    lam_colour = {b: 0.0 for b in bands}
    for item in args.lam_colour.split(","):
        key, _, value = item.rpartition("=")
        for b in ([key.strip().upper()] if key else bands):
            lam_colour[b] = float(value)
    for u in units:
        u["sources"], u["pms"], u["images"] = [], [], []
    for k, d in enumerate(draws):
        t0 = time.time()
        origin = models_for(k)
        beta = {id(u): u["fwd"] for u in units}
        if lam <= 0:
            lam, _ = fold_ref["model"].choose_lambda(beta[id(fold_ref)], origin, fold_ref["arc"])
            print(f"[shape] lambda {lam:.3g}")
        if mesh or solve_all is None:
            solve_all = shape_solver([u["model"] for u in units])
            solve_band = {b: shape_solver([u["model"] for u in by_band[b]]) for b in bands}
        # (1) every unit alone: members and the first amplitudes (from the fold pair)
        pms, alone_flux = {}, {}
        for u in units:
            src, pm, _, _ = u["model"].fit(beta[id(u)], origin, lam, pm=None if k == 0 else u["pms"][0])
            pms[id(u)] = jnp.asarray(pm)
            c = u["model"].arc_chi2_nu(jnp.asarray(src), jnp.asarray(pm), beta[id(u)], origin, jnp.asarray(u["arc"]))
            print(f"[alone] draw {d} {u['band']} {u['region']}: arc chi2_nu {c:.2f}", flush=True)
            if u["region"] == "fold":
                alone_flux[u["band"]] = float(src.sum())
        amps = {b: alone_flux[b] / alone_flux[reference] for b in bands}
        pml = [pms[id(u)] for u in units]; bl = [beta[id(u)] for u in units]
        zeros = [jnp.zeros(u["model"].shape) for u in units]
        # (2) the shared shape, alternating with one amplitude per band (shared by its regions)
        for _ in range(3):
            s0, cg = solve_all(pml, bl, origin, jnp.asarray([amps[u["band"]] for u in units]), lam, zeros)
            num = {b: 0.0 for b in bands}; den = {b: 0.0 for b in bands}
            for u in units:
                m = u["model"]._forward(s0, beta[id(u)], origin)
                dd = u["model"].data - u["model"]._members(pms[id(u)])
                num[u["band"]] += float(jnp.sum(u["model"].weight * m * dd)); den[u["band"]] += float(jnp.sum(u["model"].weight * m * m))
            fitted = {b: num[b] / den[b] for b in bands}
            amps = {b: fitted[b] / fitted[reference] for b in bands}
            s0 = s0 * fitted[reference]
        s0 = np.asarray(s0); shapes.append(s0); amps_all.append([amps[b] for b in bands])
        print(f"[shape] draw {d}: amplitudes " + ", ".join(f"{b} {amps[b]:.3f}" for b in bands) + f", cg {float(cg):.1e}", flush=True)
        # (3) each band: a_b s0 + one residual shared by its regions
        for b in bands:
            us = by_band[b]
            shaped = [amps[b] * u["model"]._forward(jnp.asarray(s0), beta[id(u)], origin) for u in us]
            members = [np.asarray(u["model"]._members(pms[id(u)])) for u in us]
            args_b = ([pms[id(u)] for u in us], [beta[id(u)] for u in us], origin, jnp.ones(len(us)))

            def total(resid):
                return [np.asarray(sh) + np.asarray(u["model"]._forward(resid, beta[id(u)], origin)) + mb
                        for u, sh, mb in zip(us, shaped, members)]
            if lam_colour[b] <= 0:
                scan = []
                for lc in np.logspace(-3, 3, 13):
                    resid, _ = solve_band[b](*args_b, lc, shaped)
                    scan.append((lc, arc_chi2_nu(us, total(resid))))
                target = max(1.0, 1.05 * min(c for _, c in scan))
                lam_colour[b] = max(lc for lc, c in scan if c <= target)
                print(f"[colour] {b}: lambda {lam_colour[b]:.3g}")
            resid, _ = solve_band[b](*args_b, lam_colour[b], shaped)
            c_shape = arc_chi2_nu(us, [np.asarray(sh) + mb for sh, mb in zip(shaped, members)])
            c_all = arc_chi2_nu(us, total(resid))
            source = amps[b] * s0 + np.asarray(resid)
            per_region = {u["region"]: arc_chi2_nu([u], [im]) for u, im in zip(us, total(resid))}
            for u in us:
                u["sources"].append(source); u["pms"].append(np.asarray(pms[id(u)]))
                if k == 0:
                    u["best_image"] = np.asarray(u["model"]._forward(jnp.asarray(source), beta[id(u)], origin))
            rows.append(dict(draw=d, band=b, lam_shape=lam, lam_colour=lam_colour[b], amplitude=amps[b], arc_chi2_nu_shape=c_shape,
                             arc_chi2_nu=c_all, **{f"arc_chi2_nu_{rn}": v for rn, v in per_region.items()},
                             **{f"delta_{rn}_{ax}": float(deltas[rn][k][i]) for rn in deltas for i, ax in enumerate("xy")}))
            print(f"[colour] draw {d} {b}: arc chi2_nu shape only {c_shape:.2f}, shape + residual {c_all:.2f} ("
                  + ", ".join(f"{rn} {v:.2f}" for rn, v in per_region.items()) + ")", flush=True)
        print(f"[joint] draw {d} in {time.time() - t0:.0f} s", flush=True)

    pd.DataFrame(rows).to_csv(out / "joint_fits.csv", index=False)
    np.savez_compressed(out / "shape.npz", amplitudes=np.asarray(amps_all), bands=np.asarray(bands), draws=np.asarray(draws),
                        deltas=np.asarray([deltas[r["name"]] for r in regions[1:]]), regions=np.asarray([r["name"] for r in regions]))
    if not mesh:
        d_origins, dh, dny, dnx = display_grid(origins, frame, ny_s, nx_s)
    for u in units:
        cut = u["cut"]; folder = out / f"{u['band']}_{u['region']}"; folder.mkdir(exist_ok=True)
        if mesh:
            disp, d_origins, dh = mesh_display(nodes_all, u["sources"])
        else:
            disp = [to_display(u["sources"][k], origins[k], frame, d_origins[k], dh, dny, dnx) for k in range(len(draws))]
        np.savez_compressed(folder / "pixel_sources.npz", sources=np.asarray(disp), members=np.asarray(u["pms"]), draws=np.asarray(draws),
                            origins=np.asarray(d_origins), pixel=dh, member_names=np.asarray(u["model"].member_names), band=u["band"],
                            image_pixel=cut["scale"], arc_model=u["best_image"],
                            extent=np.asarray([cut["xc"][0, 0], cut["xc"][0, -1], cut["yc"][0, 0], cut["yc"][-1, 0]]),
                            sky_reference=np.asarray(cut["sky_reference"]))
        if not mesh:
            figure_pixels(folder / "arc_fit.pdf", cut, u["model"], u["sources"], u["pms"], u["beta"], origins, u["det"], [f"draw {d}" for d in draws])
    for r in regions:
        print(f"[done] {r['name']}: python scripts/make_source_rgb.py " + " ".join(str(out / f"{b}_{r['name']}") for b in bands)
              + f" --out {out / ('source_rgb_' + r['name'] + '.pdf')}")


def prepare_band(args, out, state, evaluator, samples, draws, obs_xy, centre, z_source, family_index, image_gp):
    """Cutout, PSF, members, masks and beta maps of one band (args.band)."""
    cut = load_cutout(args, state, centre)
    psf = load_psf(args, cut["band"], cut["scale"], cut["shape"])
    members = member_list(args, state, cut, centre)
    # The arc: pixels near the path through the catalog positions.  Other sources above 5 sigma (outside the arc region and
    # away from the modelled members) are masked, grown by 2 pixels, so their light does not pull on the source.
    cut["arc_region"] = arc_region(cut, obs_xy, args.arc_radius)
    cut["arc_sigma"] = args.arc_sigma
    other = np.asarray(cut["mask"]) & (np.asarray(cut["data"]) > 5 * np.asarray(cut["noise"])) & ~cut["arc_region"]
    for m in members:
        other &= np.hypot(cut["xc"] - m["x"], cut["yc"] - m["y"]) > 1.2
    from scipy.ndimage import binary_dilation
    other = binary_dilation(other, iterations=2) & ~cut["arc_region"]
    cut["mask"] = np.asarray(cut["mask"]) & ~other
    cut["data"] = np.where(cut["mask"], cut["data"], 0.0)
    print(f"[arc] arc region {cut['arc_region'].sum()} px within {args.arc_radius}\" of the catalog path; "
          f"{other.sum()} px of other sources masked")
    beta, det = beta_maps(evaluator, state, samples, draws, cut["xy"], z_source, family_index,
                          out / f"beta_maps{'_imagegp' if args.image_gp_mean else ''}{'_' + args.band.upper() if args.bands else ''}{'_' + getattr(args, 'region_tag', 'fold') if args.add else ''}.npz", args.rebuild, image_gp)
    return cut, psf, members, beta, det


def main():
    args = parse_args()
    run = Path(args.run)
    tag = (f"colour{'_' + args.src_grid if args.src_grid != 'square' else ''}{'_' + args.smooth if args.smooth != 'none' else ''}{'_regw' if args.reg_weight != 'none' else ''}{'_affine' if args.affine else ''}"
           if args.bands else f"{args.band.upper()}_{args.source}")
    out = Path(args.out) / f"{args.system}{'_at' if args.at else ''}{'_joint' if args.add else ''}_{tag}{'_imagegp' if args.image_gp_mean else ''}"
    out.mkdir(parents=True, exist_ok=True)
    config, state, evaluator, posterior = load_model(run)
    images = state.images
    labels = np.asarray(images.image_labels)
    fam = np.asarray(images.family_index_per_image)
    x_obs, y_obs = np.asarray(images.x_obs, float), np.asarray(images.y_obs, float)
    wanted = set(args.images.split(","))
    sel = np.array([l.startswith(args.system) and l.split(".")[-1] in wanted and not l[len(args.system)].isdigit() for l in labels])
    if not sel.any():
        raise SystemExit(f"no images of {args.system} {sorted(wanted)} in the catalog")
    obs_pts, obs_labels = np.column_stack([x_obs[sel], y_obs[sel]]), labels[sel]
    if args.at:
        obs_pts = np.asarray([[float(v) for v in args.at.split(",")]])
        obs_labels = np.asarray([f"{args.system}.at"])
    centre = (float(obs_pts[:, 0].mean()), float(obs_pts[:, 1].mean()))
    family_index = int(fam[sel][0])
    family_id = str(images.family_ids[family_index])
    z_source = float(next(f.z_source for f in state.family_data if str(f.family_id) == family_id))
    print(f"[arc] {args.system} {sorted(wanted)}: {sel.sum()} catalog positions, z {z_source}, centre ({centre[0]:.2f}, {centre[1]:.2f})")

    draws = pick_draws(posterior, args.draws)
    samples = np.asarray(posterior.samples)
    obs_xy = obs_pts
    image_gp = None
    if args.image_gp_mean:
        if evaluator.sigma_shared_index < 0:
            raise SystemExit("--image-gp-mean needs a gp_image_plane run (no sigma_shared parameter)")
        image_gp = (obs_xy, fam[sel], float(np.asarray(images.sigma_inst_per_image)[sel].mean()))
    setup = dict(state=state, evaluator=evaluator, samples=samples, draws=draws, obs_xy=obs_xy, centre=centre,
                 z_source=z_source, family_index=family_index, image_gp=image_gp)
    if args.bands:
        cut = psf = members = beta = det = None
    else:
        cut, psf, members, beta, det = prepare_band(args, out, **setup)

    # Source start: the catalog positions of image a backprojected through the best draw (mean over its knots).
    theta_best = jnp.asarray(samples[draws[0]])
    physical = evaluator.physical(theta_best)
    packed = evaluator.packed_state_for_z(physical, z_source)
    obs = obs_pts
    alpha, _ = evaluator.total_fields_at(obs, packed)
    mean = evaluator.gp_deflection_correction(theta_best)
    if mean is not None:
        dx, dy, *_ = mean(obs[:, 0], obs[:, 1], np.full(len(obs), family_index, dtype=np.int32))
        alpha = alpha + np.column_stack([dx, dy])
    src_obs = obs - alpha
    src_labels = obs_labels
    first = [l for l in src_labels if l.endswith(".a")] or list(src_labels)
    knots_beta = [tuple(src_obs[list(src_labels).index(l)]) for l in first]
    source_centre = src_obs.mean(0)
    if args.knots > len(knots_beta):                        # extra knots start spread along the backprojected arc direction
        d = src_obs - source_centre
        axis = np.linalg.svd(d, full_matrices=False)[2][0] if len(d) > 1 else np.array([1.0, 0.0])
        extra = args.knots - len(knots_beta)
        knots_beta += [tuple(source_centre + axis * t) for t in np.linspace(-0.1, 0.1, extra + 2)[1:-1]] if extra > 1 else [tuple(source_centre)]
    elif 0 < args.knots < len(knots_beta):
        knots_beta = knots_beta[: args.knots]
    print("[arc] backprojected catalog positions (best draw): " + ", ".join(f"{l} -> ({bx:.3f}, {by:.3f})" for l, (bx, by) in zip(src_labels, src_obs)))
    print(f"[arc] source-plane scatter of the backprojected positions: {np.std(src_obs, 0).round(3)} arcsec")

    if args.bands:
        if args.source != "pixels":
            raise SystemExit("--bands needs --source pixels")
        (run_joint if args.add else run_colour)(args, out, setup, obs, src_labels)
        return
    if args.source == "pixels":
        run_pixels(args, out, cut, psf, members, beta, det, draws, obs, src_labels)
        return
    model = ArcModel(args, cut, psf, members, knots_beta, source_centre, cut["scale"])
    n_pix = int(np.asarray(cut["mask"]).sum())
    print(f"[fit] {len(model.names)} parameters, {n_pix} pixels, {len(draws)} lens draws")

    rows, params = [], []
    for k, d in enumerate(draws):
        t0 = time.time()
        p, chi2 = model.fit(beta[k], model.start, args.adam_steps)
        if k > 0:                                           # later draws also try the best-draw solution; keep the better fit
            p2, chi2_2 = model.fit(beta[k], params[0], args.adam_steps)
            if chi2_2 < chi2:
                p, chi2 = p2, chi2_2
        params.append(p)
        source, lens_light = model.unpack(jnp.asarray(p))
        row = dict(draw=d, chi2=chi2, chi2_nu=chi2 / max(n_pix - len(p), 1), seconds=time.time() - t0)
        row.update({f"src_{k_}": float(v) for k_, v in source[0].items()})
        for j, extra in enumerate(source[1:]):
            row.update({f"src{j + 1}_{k_}": float(np.ravel(np.asarray(v))[0]) if np.ndim(v) else float(v) for k_, v in extra.items() if k_ != "n_max"})
        for m, ll in zip(members, lens_light):
            row.update({f"m{m['id']}_{k_}": float(v) for k_, v in ll.items()})
        # Fold test: flux in 0.5" apertures at each catalog image, model over data.
        for l, (ox, oy) in zip(src_labels, obs):
            ap = np.hypot(cut["xc"] - ox, cut["yc"] - oy) < 0.5
            m_img = np.asarray(model.source_only(jnp.asarray(p), jnp.asarray(beta[k])))
            m_img = np.asarray(jax.scipy.signal.fftconvolve(jnp.asarray(m_img), model.psf, mode="same"))
            d_img = np.asarray(cut["data"])
            floor = 3.0 * float(np.sqrt(np.sum(np.asarray(cut["noise"])[ap] ** 2)))      # a 3 sigma aperture flux at least
            row[f"flux_ratio_{l}"] = float(m_img[ap].sum() / max(d_img[ap].sum(), floor))
        rows.append(row)
        print(f"[fit] draw {d}: chi2_nu {row['chi2_nu']:.3f} in {row['seconds']:.0f} s; "
              + " ".join(f"{l} {row[f'flux_ratio_{l}']:.2f}" for l in src_labels), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "fits_per_draw.csv", index=False)
    np.savez_compressed(out / "fit_params.npz", params=np.asarray(params), names=np.asarray(model.names), draws=np.asarray(draws))
    numeric = table.drop(columns=["draw"]).select_dtypes(float)
    summary = pd.DataFrame(dict(best=numeric.iloc[0], median=numeric.median(), q16=numeric.quantile(0.16), q84=numeric.quantile(0.84)))
    summary.to_csv(out / "source_summary.csv")
    print(summary.round(4).to_string())

    centres = np.array([[float(v) for v in (model.unpack(jnp.asarray(q))[0][0]["center_x"], model.unpack(jnp.asarray(q))[0][0]["center_y"])] for q in params])
    half = max(3 * float(np.exp(params[0][1])), 0.3) + float(np.abs(centres - centres.mean(0)).max())
    cx, cy = centres.mean(0)
    extent = ((cx - half, cx + half), (cy - half, cy + half))
    figure(out / "arc_fit.pdf", args, cut, model, params, beta, det, [f"draw {d}" for d in draws], extent)

    if args.nuts > 0:
        import numpyro, numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        beta0 = jnp.asarray(beta[0]); p0 = jnp.asarray(params[0])

        def numpyro_model():
            p = numpyro.sample("p", dist.Normal(p0, 3.0).to_event(1))
            numpyro.factor("like", -0.5 * model._chi2(p, beta0))
        mcmc = MCMC(NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.8), num_warmup=args.nuts, num_samples=args.nuts, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), init_params={"p": p0})
        ps = np.asarray(mcmc.get_samples()["p"])
        post = pd.DataFrame(ps, columns=model.names)
        post.to_csv(out / "nuts_source_samples.csv", index=False)
        q = post.quantile([0.16, 0.5, 0.84]).T; q.columns = ["q16", "q50", "q84"]
        q.to_csv(out / "nuts_source_summary.csv")
        print("[nuts] unconstrained-parameter quantiles at the best-draw lens:\n" + q.round(3).to_string())
    print(f"[done] outputs in {out}")


if __name__ == "__main__":
    main()
