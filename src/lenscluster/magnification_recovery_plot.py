"""Magnification recovery at constrained images and across the simulated lens field."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, QhullError

from .plot_style import PAPER_STYLE

STYLE = Path(__file__).with_name('style.mplstyle')
BIN_COLUMNS = ['true_mu', 'model_p16', 'model_median', 'model_p84', 'n_pixels']


def binned_recovery(true_mu, model_mu):
    """Spatial quantiles, not errors on the median or posterior intervals."""
    rows = []
    edges = np.geomspace(1, 100, 13)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (true_mu >= lower) & (true_mu < upper)
        if selected.sum() >= 8:
            rows.append([np.median(true_mu[selected]), *np.percentile(model_mu[selected], [16, 50, 84]), selected.sum()])
    return pd.DataFrame(rows, columns=BIN_COLUMNS)


def image_truth_magnification(images, reference, z_lens, cosmology, truth_config):
    """Interpolate native kappa/shear, scale lens efficiency, then invert det A."""
    from .truth_recovery import load_truth_fits
    from .cutouts import arcsec_to_radec

    zs = images.z_source.to_numpy(float)
    if not np.all(np.isfinite(zs) & (zs > z_lens)):
        raise ValueError('Magnification recovery requires finite source redshifts behind the lens')
    ra, dec = arcsec_to_radec(images.x_obs_arcsec.to_numpy(), images.y_obs_arcsec.to_numpy(), reference)
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

    def efficiency(z):
        return (cosmology.angular_diameter_distance_z1z2(z_lens, z) / cosmology.angular_diameter_distance(z)).value

    fields, redshifts = [], []
    for path in (truth_config.kappa_true_fits, truth_config.gammax_true_fits, truth_config.gammay_true_fits):
        native, wcs = load_truth_fits(path)
        header = fits.getheader(path)
        z_ref = float(header.get('ZSOURCE', header.get('REDSHIFT', truth_config.caustic_source_redshift)))
        if not np.isfinite(z_ref) or z_ref <= z_lens:
            raise ValueError(f'Invalid truth source redshift in {path}')
        px, py = wcs.world_to_pixel(sky)
        field = map_coordinates(native, [py, px], order=1, mode='constant', cval=np.nan)
        fields.append(field * efficiency(zs) / efficiency(z_ref))
        redshifts.append(z_ref)
    kappa, gamma1, gamma2 = fields
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 1 / ((1 - kappa)**2 - gamma1**2 - gamma2**2)
    return mu, redshifts


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def render_recovery(true_mu, model_mu, bins, images, path):
    fig, ax = plt.subplots(figsize=(3.5, 3.1), layout='constrained')
    ax.hexbin(true_mu, model_mu, xscale='log', yscale='log', gridsize=45, mincnt=1,
              bins='log', cmap='Greys', linewidths=0, rasterized=True)
    ax.plot([.5, 200], [.5, 200], color='0.45', ls='--', lw=1)
    if not bins.empty:
        ax.errorbar(bins.true_mu, bins.model_median,
                    yerr=[bins.model_median-bins.model_p16, bins.model_p84-bins.model_median],
                    fmt='o', color='tab:orange', ms=3, lw=1, capsize=2,
                    label=r'Binned $|\mu|$', zorder=4)
    valid = images.finite.to_numpy(bool)
    ax.scatter(images.loc[valid, 'true_abs_mu'], images.loc[valid, 'model_abs_mu'], marker='D', s=10,
               color='tab:blue', edgecolors='none', linewidths=0, alpha=.85, label='Images', zorder=3)
    ax.set(xlim=(1, 100), ylim=(.5, 200), xlabel=r'True $|\mu|$', ylabel=r'Model $|\mu|$')
    ax.grid(False, which='both')
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == 'Images')
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, fontsize=8, loc='upper left')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def write_magnification_recovery(truth, images, reference, z_lens, cosmology, truth_config,
                                 figures, tables, prefix, *, log=print):
    """Write one figure plus its numbers/provenance; only simulations with shear truth qualify."""
    paths = (truth_config.kappa_true_fits, truth_config.gammax_true_fits, truth_config.gammay_true_fits)
    if truth is None or 'mu' not in truth['truth'] or not all(paths) or images.empty:
        log('[outputs] magnification recovery skipped: image catalog and kappa/shear truth required')
        return None
    xy = images[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float)
    xy = np.unique(xy[np.isfinite(xy).all(axis=1)], axis=0)
    if len(xy) < 3:
        log('[outputs] magnification recovery skipped: fewer than three distinct image positions')
        return None
    try:
        hull = Delaunay(xy)
    except QhullError:
        log('[outputs] magnification recovery skipped: image positions do not enclose an area')
        return None
    x, y = truth['x'], truth['y']
    inside = hull.find_simplex(np.c_[x.ravel(), y.ravel()]).reshape(x.shape) >= 0
    t, m = np.abs(truth['truth']['mu']), np.abs(truth['model']['mu']['median'])
    mask = inside & np.isfinite(t) & np.isfinite(m) & (t >= 1) & (m > 0)
    if not mask.any():
        log('[outputs] magnification recovery skipped: no finite map pixels inside the image hull')
        return None
    bins = binned_recovery(t[mask], m[mask])
    mu, reference_z = image_truth_magnification(images, reference, z_lens, cosmology, truth_config)
    points = images[['family_id', 'image_label', 'x_obs_arcsec', 'y_obs_arcsec', 'z_source']].copy()
    points['true_abs_mu'] = np.abs(mu)
    points['model_abs_mu'] = np.abs(images.magnification_model.to_numpy(float))
    points['finite'] = np.isfinite(points.true_abs_mu) & np.isfinite(points.model_abs_mu)
    points['within_display'] = points.finite & points.true_abs_mu.between(1, 100) & points.model_abs_mu.between(.5, 200)
    tables.mkdir(parents=True, exist_ok=True)
    bins.to_csv(tables / 'magnification_recovery_bins.csv', index=False)
    points.to_csv(tables / 'magnification_recovery_images.csv', index=False)
    metadata = dict(map_source_redshift=float(truth['z_source']), truth_field_reference_redshifts=reference_z,
                    truth_paths=list(map(str, paths)), region='convex hull of observed images; finite true |mu| >= 1',
                    map_model='coordinatewise median parameter vector',
                    map_grf_conditional_mean=truth.get('grf_conditional_mean', 'not recorded'), image_model='best-fit exact kernels at observed positions',
                    image_redshift='each family source redshift', truth_interpolation='native kappa/shear bilinear interpolation, then lens-efficiency scaling, then inverse determinant',
                    error_bars='spatial 16-84 percentiles, not posterior uncertainty or standard error',
                    bin_edges=np.geomspace(1, 100, 13).tolist(), min_bin_pixels=8,
                    map_pixels=int(mask.sum()), image_count=len(points), finite_images=int(points.finite.sum()),
                    images_outside_display=int((points.finite & ~points.within_display).sum()),
                    map_pixels_outside_display=int(((t[mask] > 100) | (m[mask] < .5) | (m[mask] > 200)).sum()))
    (tables / 'magnification_recovery_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    path = figures / f'{prefix + "_" if prefix else ""}magnification_recovery.pdf'
    render_recovery(t[mask], m[mask], bins, points, path)
    log(f'[outputs] magnification recovery: {path} ({points.finite.sum()}/{len(points)} finite image markers)')
    return path
