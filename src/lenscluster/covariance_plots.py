"""Covariance posterior and exact-field uncertainty figures for run outputs."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

from .smoothed_likelihood import source_covariance
from .plot_style import PAPER_STYLE
from .member_diagnostics import diagnostic_draw_indices

STYLE = Path(__file__).with_name('style.mplstyle')


def plot_covariance_errors(config, state, evaluator, posterior, paths, *, draws=32, plane='both'):
    """Exact covariance components; field draw count follows diagnostic sampling policy."""
    if plane not in {'source', 'image', 'both'}:
        raise ValueError('Unknown covariance display plane')
    plane_mode = plane
    best = int(np.nanargmax(posterior.log_prob))
    @jax.jit
    def evaluate(theta):
        p = evaluator.physical(theta)
        a = evaluator.fields(p, force_exact=True, derivative=False)['jacobian']
        scales = evaluator.error_scales(p)
        sel, lam, model, meas = scales['sigma_sel'], scales['lambda_sel'], scales['sigma_def'], scales['sigma_meas']
        u, s, _ = jnp.linalg.svd(a)
        # Ascending singular values: maximum image stretch first.
        s, u = s[:, ::-1], u[:, :, ::-1]
        measurement = meas[:, None] * s
        selection = sel / jnp.sqrt(1 + (s / lam)**2)
        deflection = jnp.broadcast_to(jnp.broadcast_to(model, s.shape[:1])[:, None], s.shape)   # sigma_def is per image under the knee kernel
        total = jnp.sqrt(measurement**2 + selection**2 + deflection**2)
        cov = source_covariance(a, meas, model, sel, lam)['covariance']
        projected = jnp.einsum('nij,nik,nkl->njl', u, cov, u)
        return s, jnp.stack([measurement, selection, deflection, total]), projected

    s, widths, projected = map(np.asarray, evaluate(posterior.samples[best]))
    np.testing.assert_allclose(widths[-1]**2, np.diagonal(projected, axis1=-2, axis2=-1), rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(projected[:, 0, 1], 0, atol=1e-12)
    indices = diagnostic_draw_indices(posterior.log_prob, draws)
    source_widths = widths.copy()
    source_draws, image_draws = [], []
    for i in indices:
        draw_s, draw_widths, _ = map(np.asarray, evaluate(posterior.samples[i]))
        if np.any(draw_s <= 0):
            raise ValueError('Image-plane covariance is undefined at a singular Jacobian')
        source_draws.append(draw_widths)
        image_draws.append(draw_widths / draw_s[None, :, :])
    if np.any(s <= 0):
        raise ValueError('Image-plane covariance is undefined at a singular Jacobian')
    plane_data = {'source': (source_widths, np.stack(source_draws)),
                  'image': (source_widths / s[None, :, :], np.stack(image_draws))}
    planes = ['image', 'source'] if plane == 'both' else [plane]
    prefix = config.paths.figure_prefix or 'cluster'
    stem = 'covariance_errors' + {'image': '_image_plane', 'source': '', 'both': '_both_planes'}[plane]
    output = paths.figures / f'{prefix}_{stem}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ['Measurement', 'Selection', 'Model', 'Total']
    colors = ['tab:blue', 'tab:green', 'tab:red', 'black']
    table = []
    with plt.rc_context(fname=STYLE, rc=PAPER_STYLE):
        fig, axes = plt.subplots(len(planes), 2, figsize=(7.1, 5.5 if len(planes) == 2 else 3.25),
                                 sharey='row', sharex='col', squeeze=False)
        for row, plane in enumerate(planes):
            widths, draws = plane_data[plane]
            component_lo, component_median, component_hi = np.percentile(draws, [16, 50, 84], axis=0)
            lo, median, hi = np.percentile(draws[:, -1], [16, 50, 84], axis=0)
            for k, ax in enumerate(axes[row]):
                order = np.argsort(s[:, k])
                for j in range(3):
                    positive = widths[j, order, k] > 0
                    ax.plot(s[order, k][positive], widths[j, order, k][positive], color=colors[j], lw=1.5, label=names[j])
                ax.plot(s[order, k], widths[-1, order, k], color='black', lw=1.25, label='Total: best fit')
                ax.errorbar(s[:, k], median[:, k], yerr=[median[:, k]-lo[:, k], hi[:, k]-median[:, k]],
                            fmt='o', ms=2.5, color='.35', ecolor='.65', elinewidth=.65, capsize=0,
                            label='Total: posterior 16–84%', zorder=1)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                if row == len(planes)-1:
                    ax.set_xlabel(r'$|\lambda_{\rm arc}|$' if k == 0 else r'$|\lambda_{\perp}|$')
                ax.text(.04, .96, 'Along image stretch' if k == 0 else 'Perpendicular', transform=ax.transAxes,
                        va='top', fontsize=9)
                for i, label in enumerate(state.images.image_labels):
                    table.append(dict(plane=plane, image_label=label, direction='arc' if k == 0 else 'perpendicular',
                                      singular_value_best=s[i, k], **{name.lower()+'_arcsec': widths[j, i, k] for j, name in enumerate(names)},
                                      **{f'{name.lower()}_{q}_arcsec': values[j, i, k]
                                         for j, name in enumerate(names[:3])
                                         for q, values in [('p16', component_lo), ('median', component_median), ('p84', component_hi)]},
                                      total_p16_arcsec=lo[i,k], total_median_arcsec=median[i,k], total_p84_arcsec=hi[i,k]))
            axes[row, 0].set_ylabel(f'{plane.capitalize()} plane\nUncertainty (arcsec)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.52, 1.01))
        fig.subplots_adjust(left=.10, right=.98, bottom=.10 if len(planes) == 2 else .17,
                            top=.86 if len(planes) == 2 else .77, wspace=0, hspace=0)
        fig.savefig(output, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
    tables = paths.tables
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(tables / f'{stem}.csv', index=False)
    (tables / f'{stem}.json').write_text(json.dumps(dict(best_draw=best, posterior_draw_indices=indices.tolist(),
        plane=plane_mode, transformation={p: 'A^-1 C A^-T' if p == 'image' else 'C' for p in planes},
        kernels='exact', positions='observed images at their family redshifts',
        x_axis='singular values at best fit; held fixed for posterior interval display',
        intervals='16th–84th percentiles of component and total widths in each draw\u0027s singular basis',
        units='arcsec; standard deviations, not covariance eigenvalues'), indent=2))
    return output


def plot_covariance_corner(config, state, posterior, paths):
    """Covariance hyperparameters in the common prior-overlaid corner style."""
    from .corner_plots import plot_parameter_corner
    fields = ['sigma_extra', 'sigma_def', 'sigma_sel', 'lambda_sel', 'sigma_shared', 'sigma_white', 'gp_ell',
              'gp_sigma_alpha', 'gp_knee', 'gp_knee_gamma']
    symbols = [r'$\sigma_{\rm extra}$', r'$\sigma_{\rm def}$',
               r'$\sigma_{\rm sel}$', r'$\lambda_{\rm sel}$', r'$\sigma_{\rm shared}$', r'$\sigma_{\rm white}$', r'$\ell$',
               r'$\sigma_\alpha$', r'$\ell_{\rm knee}$', r'$\gamma_{\rm GRF}$']
    selection = [(i, spec, label) for field, label in zip(fields, symbols)
                 for i, spec in enumerate(state.parameter_specs) if spec.name == 'images.'+field]
    grouped = None
    if getattr(posterior, 'grouped_samples', None) is not None:
        np.testing.assert_array_equal(posterior.grouped_samples.reshape(posterior.samples.shape), posterior.samples)
        grouped = np.asarray(posterior.physical).reshape(posterior.grouped_samples.shape)
    path = paths.figures / f'{config.paths.figure_prefix or "cluster"}_covariance_corner.pdf'
    result = plot_parameter_corner(posterior.physical, selection, path, grouped_physical=grouped,
                                   best_index=int(np.nanargmax(posterior.log_prob)))
    if result is not None:
        paths.tables.mkdir(parents=True, exist_ok=True)
        (paths.tables / 'covariance_corner.json').write_text(path.with_suffix('.json').read_text())
    return result


def write_covariance_plots(config, state, evaluator, posterior, paths, *, log=print):
    outputs = []
    corner = plot_covariance_corner(config, state, posterior, paths)
    if corner is not None:
        outputs.append(corner)
    else:
        log('[outputs] covariance corner skipped: insufficient draws or no sampled covariance parameters')
    outputs.append(plot_covariance_errors(config, state, evaluator, posterior, paths,
                                          draws=config.image_diagnostics.posterior_diagnostic_draws))
    for path in outputs:
        log(f'[outputs] covariance plot: {path}')
    return outputs
