"""RGB comparison of linear and quadratic lens-map uncertainty contours."""
from pathlib import Path
import json

import contourpy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .cutouts import arcsec_to_radec, arcsec_to_skycoord, cutout_window, rgb_cutout, rgb_reference_band
from .exact_images import ExactImageSolver
from .plots import figure_path
from .smoothed_likelihood import source_covariance

STYLE = Path(__file__).with_name('style.mplstyle')
TOTAL, MEASUREMENT, QUADRATIC = '#00e5ff', '#ffb347', '#ff73df'
DEFINITION = ('Delta beta = A d + B[d,d]/2; contour Delta beta^T C^-1 Delta beta = 1. '
              'C and the expansion center beta(theta_observed) are fixed at the observed position. '
              'Geometric diagnostic, not a full likelihood or calibrated probability contour.')


def quadratic_offset(displacement, jacobian, derivative):
    return (np.einsum('ij,...j->...i', jacobian, displacement)
            + .5 * np.einsum('ijk,...j,...k->...i', derivative, displacement, displacement))


def quadratic_contours(jacobian, derivative, covariance, fov, *, grid_size=1001):
    axis = np.linspace(-fov / 2, fov / 2, grid_size)
    x, y = np.meshgrid(axis, axis)
    offset = quadratic_offset(np.stack([x, y], axis=-1), jacobian, derivative)
    q = np.einsum('...i,ij,...j->...', offset, np.linalg.inv(covariance), offset)
    return contourpy.contour_generator(x=axis, y=axis, z=q).lines(1.)


def best_geometry(state, evaluator, theta):
    """Exact fields in the observed image order, at each family's source redshift."""
    @jax.jit
    def evaluate(t):
        p = evaluator.physical(t)
        f = evaluator.fields(p, force_exact=True, derivative=True)
        extra = p[evaluator.sigma_extra_index] if evaluator.sigma_extra_index >= 0 else 0.
        sel = p[evaluator.sigma_sel_index] if evaluator.sigma_sel_index >= 0 else 0.
        lam = p[evaluator.lambda_sel_index] if evaluator.lambda_sel_index >= 0 else 1.
        meas = jnp.sqrt(state.images.sigma_inst_per_image**2 + extra**2)
        c = source_covariance(f['jacobian'], meas, p[evaluator.sigma_def_index], sel, lam)['covariance']
        return p, f['backprojected'], f['jacobian'], f['jacobian_spatial_derivative'], c, meas
    p, beta, a, b, c, meas = map(np.asarray, evaluate(jnp.asarray(theta)))
    _, singular, vh = np.linalg.svd(a)
    directions = vh[:, ::-1, :].transpose(0, 2, 1)
    singular = singular[:, ::-1]
    # Use solves, not a pseudoinverse: a singular image-plane covariance is undefined.
    valid = np.isfinite(a).all(axis=(1, 2)) & (singular[:, 0] > np.finfo(float).eps)
    widths = np.full(singular.shape, np.nan)
    for i in np.flatnonzero(valid):
        inverse = np.linalg.solve(a[i], np.eye(2))
        image_cov = inverse @ c[i] @ inverse.T
        widths[i] = np.sqrt(np.maximum(np.diag(directions[i].T @ image_cov @ directions[i]), 0.))
    return dict(physical=p, beta=beta, a=a, b=b, c=c, meas=meas, singular=singular,
                directions=directions, widths=widths, valid=valid & np.isfinite(widths).all(axis=1))


def write_covariance_cutouts(config, state, evaluator, posterior, paths, imaging, *, log=print):
    """Write up to twelve best-fit cutouts; reuse already loaded cluster imaging."""
    stem = f'small_lambda_covariance_cutouts{config.image_catalog.figure_suffix}'
    metadata_path = paths.tables / f'{stem}.json'
    paths.tables.mkdir(parents=True, exist_ok=True)
    if config.likelihood.likelihood_mode != 'smoothed' or not np.isfinite(posterior.log_prob).any():
        metadata_path.write_text(json.dumps({'skipped': 'Requires smoothed likelihood and a finite best draw'}))
        return None
    best = int(np.nanargmax(posterior.log_prob))
    g = best_geometry(state, evaluator, posterior.samples[best])
    labels = list(state.images.image_labels)
    order = sorted(np.flatnonzero(g['valid']), key=lambda i: (g['singular'][i, 0], labels[i]))[:12]
    if not order:
        metadata_path.write_text(json.dumps({'skipped': 'No finite nonsingular image-plane covariance'}))
        return None
    images, bands, display, _ = imaging
    reference = images[rgb_reference_band(bands)]
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    families = {str(f.family_id): f for f in state.family_data}
    records, contours = [], {}
    intervals = None
    width_path, info_path = paths.tables / 'covariance_errors_both_planes.csv', paths.tables / 'covariance_errors_both_planes.json'
    if width_path.exists() and info_path.exists() and json.loads(info_path.read_text()).get('best_draw') == best:
        intervals = pd.read_csv(width_path, dtype={'image_label': str}).query("plane == 'image' and direction == 'arc'").set_index('image_label')
    for i in order:
        family_id = state.images.family_ids[int(state.images.family_index_per_image[i])]
        family = families[str(family_id)]
        xy = np.array([state.images.x_obs[i], state.images.y_obs[i]], dtype=float)
        # Bound image I/O near a singular Jacobian. Clipping is reported, not hidden.
        fov = min(30., max(6., 2.5 * g['widths'][i, 0]))
        segments = quadratic_contours(g['a'][i], g['b'][i], g['c'][i], fov)
        contours[i] = segments
        error = None
        if segments:
            vertices = np.concatenate(segments)
            selected = vertices[np.linspace(0, len(vertices)-1, min(512, len(vertices))).astype(int)]
            # Fixed batch shape avoids compiling a new exact kernel for each contour.
            points = np.tile(xy, (512, 1)); points[:len(selected)] += selected
            packed = evaluator.packed_state_for_z(g['physical'], family.z_source)
            bx, by, *_ = solver.ray_shoot_and_jacobian(packed, points[:, 0], points[:, 1])
            actual = np.column_stack([bx, by])[:len(selected)] - g['beta'][i]
            approximate = quadratic_offset(selected, g['a'][i], g['b'][i])
            discrepancy = np.linalg.solve(np.linalg.cholesky(g['c'][i]), (actual-approximate).T).T
            error = float(np.linalg.norm(discrepancy, axis=1).max())
        clipped = any(np.any(np.abs(seg) >= fov/2 - 1e-8) for seg in segments)
        records.append(dict(image_label=labels[i], family_id=str(family.family_id), z_source=float(family.z_source),
                            x_arcsec=float(xy[0]), y_arcsec=float(xy[1]), lambda_abs=float(g['singular'][i, 0]),
                            sigma_arc=float(g['widths'][i, 0]), sigma_perp=float(g['widths'][i, 1]),
                            sigma_meas=float(g['meas'][i]), fov_arcsec=fov, quadratic_segments=len(segments),
                            quadratic_clipped=clipped, linear_clipped=bool(2*g['widths'][i, 0] > fov),
                            sampled_max_exact_map_error=error))
    output = figure_path(paths.figures, f'{stem}.pdf', config.paths.figure_prefix)
    angle = np.linspace(0, 2*np.pi, 361)
    with plt.rc_context(fname=STYLE, rc={'font.size': 9, 'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular'}):
        with PdfPages(output) as pdf:
            for start in range(0, len(order), 2):
                fig, axes = plt.subplots(2, 3, figsize=(12., 8.8))
                fig.subplots_adjust(left=.025, right=.975, bottom=.14, top=.94, wspace=.08, hspace=.28)
                for rr in range(2):
                    if start+rr >= len(order):
                        for ax in axes[rr]: ax.set_visible(False)
                        continue
                    i, row = order[start+rr], records[start+rr]
                    xy = np.array([row['x_arcsec'], row['y_arcsec']])
                    coord = arcsec_to_skycoord(*xy, state.reference)
                    rgb_cache = {}
                    for col, ax in enumerate(axes[rr]):
                        fov = 6. if col == 0 else row['fov_arcsec']
                        window = cutout_window(reference, coord, fov)
                        if window is None:
                            ax.set_axis_off(); ax.text(.5, .5, 'Outside imaging coverage', ha='center', transform=ax.transAxes)
                            continue
                        if fov not in rgb_cache:
                            rgb_cache[fov] = rgb_cutout(images, bands, display, coord, fov)
                        rgb = rgb_cache[fov]
                        ax.imshow(rgb, origin='lower', interpolation='nearest')
                        ax.set_xlim(-.5, rgb.shape[1]-.5); ax.set_ylim(-.5, rgb.shape[0]-.5)
                        def pixels(offset):
                            ra, dec = arcsec_to_radec(xy[0]+offset[0], xy[1]+offset[1], state.reference)
                            px, py = reference.wcs.world_to_pixel_values(ra, dec)
                            return px-window.x0, py-window.y0
                        ax.plot(*pixels(np.zeros((2, 1))), '+', color='white', ms=6, mew=.8, zorder=6)
                        if col < 2:
                            ax.plot(*pixels(row['sigma_meas']*np.array([np.cos(angle), np.sin(angle)])),
                                    color=MEASUREMENT, lw=.95, ls='--', zorder=5)
                        if col == 1:
                            ellipse = g['directions'][i] @ (g['widths'][i, :, None]*np.array([np.cos(angle), np.sin(angle)]))
                            ax.plot(*pixels(ellipse), color=TOTAL, lw=1.3, zorder=5)
                            axis = g['directions'][i, :, 0, None]*np.array([-row['sigma_arc'], row['sigma_arc']])
                            ax.plot(*pixels(axis), color=TOTAL, lw=.65, alpha=.8)
                        if col == 2:
                            for seg in contours[i]: ax.plot(*pixels(seg.T), color=QUADRATIC, lw=1.3)
                        xx, yy = .08*rgb.shape[1], .09*rgb.shape[0]
                        scale = 1/reference.pixel_scale_arcsec
                        ax.plot([xx, xx+scale], [yy, yy], color='white', lw=1.5)
                        ax.text(xx+scale/2, yy+.025*rgb.shape[0], '1″', color='white', ha='center', fontsize=9)
                        ax.set_xticks([]); ax.set_yticks([])
                        if col == 0:
                            text = (f"{row['image_label']} | z = {row['z_source']:g} | 6″ field\n"
                                    + rf"$|\lambda_{{arc}}|={row['lambda_abs']:.4f}$" + '\n'
                                    + rf"Measurement radius: $\sigma_{{meas}}={row['sigma_meas']:.4f}$″")
                        elif col == 1:
                            text = (f"Linear mapping | {fov:.1f}″ field | axis ratio {row['sigma_arc']/row['sigma_perp']:.1f}\n"
                                    + rf"$\sigma_{{arc}}={row['sigma_arc']:.2f}$″, $\sigma_\perp={row['sigma_perp']:.2f}$″")
                            if intervals is not None and row['image_label'] in intervals.index:
                                q = intervals.loc[row['image_label']]
                                text += f"\nPosterior 16–84%: {q.total_p16_arcsec:.2f}–{q.total_p84_arcsec:.2f}″ along stretch"
                            if row['linear_clipped']: text += '\nEllipse extends beyond field'
                        else:
                            err = row['sampled_max_exact_map_error']
                            text = f'Second-order mapping | {fov:.1f}″ field\n'
                            text += f'Exact-map error: {err:.2f} source σ (sampled max)' if err is not None else 'No unit contour in this field'
                            if row['quadratic_clipped']: text += '\nContour extends beyond field'
                        ax.text(.5, -.035, text, transform=ax.transAxes, ha='center', va='top', fontsize=9)
                fig.legend(handles=[Line2D([], [], color='.3', marker='+', ls='none', label='Observed position'),
                                    Line2D([], [], color=TOTAL, label='Linear covariance'),
                                    Line2D([], [], color=MEASUREMENT, ls='--', label='Measurement only'),
                                    Line2D([], [], color=QUADRATIC, label='Second-order contour')],
                           loc='upper center', ncol=4, frameon=False)
                fig.text(.5, .018, 'Contours enclose source offsets with Δβᵀ C⁻¹ Δβ ≤ 1; C is fixed at the observed position. Actual angular sizes.\n'
                         'Quadratic contours are geometric diagnostics, not calibrated probability contours. Best-posterior model; each family redshift.',
                         ha='center', fontsize=8)
                pdf.savefig(fig); plt.close(fig)
    pd.DataFrame(records).to_csv(paths.tables / f'{stem}.csv', index=False)
    np.savez(paths.tables / f'{stem}_contours.npz',
             **{f'image_{i}_segment_{k}': seg for i in order for k, seg in enumerate(contours[i])})
    metadata_path.write_text(json.dumps(dict(best_draw=best, definition=DEFINITION, selection='12 smallest absolute Jacobian singular values; ties by image label',
                                            kernel='exact', excluded_singular_or_nonfinite=int((~g['valid']).sum()), records=records,
                                            rgb_files={band: str(image.path) for band, image in images.items()}), indent=2))
    log(f'[outputs] covariance RGB cutouts: {len(order)} images -> {output}')
    return output
