"""Render the member and likelihood diagnostic suite from shared numerical products."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .member_diagnostics import compute_member_diagnostics
from .potfile_plots import potfile_populations

STYLE = Path(__file__).with_name('style.mplstyle')
RC = {'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
      'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
      'axes.linewidth': .8, 'xtick.major.width': .7, 'ytick.major.width': .7,
      'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.top': False, 'ytick.right': False,
      'axes.grid': False, 'pdf.fonttype': 42, 'legend.fontsize': 8}
COLORS = {'exact': 'tab:orange', 'far-field': 'tab:blue'}
MARKERS = {'exact': 'D', 'far-field': 'o'}


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.tick_params(which='both', top=False, right=False)


def finish(fig, path):
    fig.savefig(path, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def member_points(ax, table, x, values, low=None, high=None):
    handles = []
    for kind in ('far-field', 'exact'):
        mask = (table.member_class.to_numpy() == kind) & np.isfinite(x) & np.isfinite(values)
        if not mask.any():
            continue
        error = None if low is None else np.maximum(0., np.array([values[mask]-low[mask], high[mask]-values[mask]]))
        handle = ax.errorbar(np.asarray(x)[mask], values[mask], yerr=error,
                            fmt=MARKERS[kind], color=COLORS[kind], markersize=3,
                            ecolor=matplotlib.colors.to_rgba(COLORS[kind], .23),
                            elinewidth=.55, capsize=0, label=kind.capitalize(), zorder=4)
        handles.append(handle)
    clean(ax)
    return handles


def deviation_pdf(data, path):
    t = data.members
    with PdfPages(path) as pdf:
        for key, label in data.variables.items():
            fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
            x = t[key].to_numpy(float)
            for j, (ax, name, symbol) in enumerate(zip(axes, ('velocity', 'radius'), (r'$\delta_\sigma$', r'$\delta_r$'))):
                handles = member_points(ax, t, x, t[f'{name}_median'].to_numpy(),
                                        t[f'{name}_p16'].to_numpy(), t[f'{name}_p84'].to_numpy())
                ax.axhline(0, color='black', lw=.7, ls='--', zorder=1)
                if key == 'catalog_mag':
                    for population, sub in t.groupby('potfile_id'):
                        sub = sub[sub[f'{name}_population_p84'] > sub[f'{name}_population_p16']]
                        if sub.empty:
                            continue
                        lo, hi = sub[f'{name}_population_p16'].median(), sub[f'{name}_population_p84'].median()
                        ax.fill_between([x.min(), x.max()], lo, hi, color='black', alpha=.055, linewidth=0)
                    ax.invert_xaxis()
                    ax.text(.03, .97, 'Band: population scatter (68%)', transform=ax.transAxes,
                            va='top', fontsize=7, color='.35')
                if key == 'nearest_halo_index':
                    labels = t[['nearest_halo_index', 'nearest_halo_id']].drop_duplicates().sort_values('nearest_halo_index')
                    ax.set_xticks(labels.nearest_halo_index, labels.nearest_halo_id)
                ax.set(xlabel=label, ylabel=symbol + ' (natural log)')
                if name == 'radius' and t.radius_deviation_fixed.any():
                    ax.text(.03, .03, f'Fixed radius deviations: {int(t.radius_deviation_fixed.sum())} members',
                            transform=ax.transAxes, fontsize=7, color='.4')
                if key in {'neighbor_density_arcsec2', 'deflection', 'shear'} and np.all(x[np.isfinite(x)] > 0):
                    ax.set_xscale('log')
                if j == 0 and handles:
                    ax.legend(handles=handles, loc='best', frameon=False)
            fig.tight_layout(w_pad=2.)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=.06)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
        ax = axes[0]
        for kind in ('far-field', 'exact'):
            sub = t[t.member_class == kind]
            if sub.empty:
                continue
            ax.errorbar(sub.velocity_median, sub.radius_median,
                        xerr=np.maximum(0., [sub.velocity_median-sub.velocity_p16, sub.velocity_p84-sub.velocity_median]),
                        yerr=np.maximum(0., [sub.radius_median-sub.radius_p16, sub.radius_p84-sub.radius_median]),
                        fmt=MARKERS[kind], color=COLORS[kind], ecolor=matplotlib.colors.to_rgba(COLORS[kind], .2),
                        ms=3, elinewidth=.5, label=kind.capitalize())
        ax.set(xlabel=r'$\delta_\sigma$', ylabel=r'$\delta_r$')
        limits = ax.get_xlim(), ax.get_ylim()
        xx = np.linspace(*limits[0], 100)
        ax.plot(xx, -2*xx, color='black', ls='--', lw=.8)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        ax.text(.03, .97, r'Dashed: $2\delta_\sigma+\delta_r=0$', transform=ax.transAxes, va='top', fontsize=7)
        ax.legend(frameon=False)
        member_points(axes[1], t, t.nearest_image_arcsec.to_numpy(), t.velocity_radius_correlation.to_numpy())
        axes[1].axhline(0, color='.6', lw=.7, ls='--')
        axes[1].set(xlabel='Nearest constrained image (arcsec)', ylabel=r'Joint-draw correlation $\rho(\delta_\sigma,\delta_r)$', ylim=(-1.05, 1.05))
        for ax in axes:
            clean(ax)
        fig.tight_layout(w_pad=2.)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=.06)
        plt.close(fig)


def constraint_pdf(data, path):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7))
    t = data.members
    for row, name in enumerate(('velocity', 'radius')):
        for col, key in enumerate(('catalog_mag', 'nearest_image_arcsec')):
            ax = axes[row, col]
            handles = member_points(ax, t, t[key].to_numpy(), t[f'{name}_width_ratio'].to_numpy())
            ax.axhline(1, color='black', lw=.8, ls='--')
            symbol = r'\delta_\sigma' if row == 0 else r'\delta_r'
            ax.set(xlabel=data.variables[key], ylabel=rf'$w_{{68}}({symbol})/w_{{68,\rm prior}}$')
            if key == 'catalog_mag':
                ax.invert_xaxis()
            ax.set_ylim(bottom=0)
            if row == col == 0:
                ax.legend(handles=handles, frameon=False)
            if not np.isfinite(t[f'{name}_width_ratio']).any():
                ax.text(.5, .5, 'No active deviations', ha='center', transform=ax.transAxes)
    fig.tight_layout(w_pad=2.)
    finish(fig, path)


def residual_pdf(data, path):
    fields = data.fields
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8))
    ax = axes[0]
    angle = np.linspace(0, 2*np.pi, 200)
    for radius in (1., 2.):
        ax.plot(radius*np.cos(angle), radius*np.sin(angle), color='.7', lw=.6, ls='--', zorder=1)
    families = data.images.family_id.to_numpy()
    cmap = plt.get_cmap('tab20')
    for i, family in enumerate(dict.fromkeys(families)):
        mask = families == family
        cloud = fields['residual'][1:, mask].reshape(-1, 2)
        ax.scatter(*cloud.T, s=2, alpha=.10, color=cmap(i % 20), linewidths=0, zorder=2)
        ax.scatter(*fields['residual'][0, mask].T, s=14, color=cmap(i % 20), label=str(family), zorder=3)
    ax.axhline(0, color='.85', lw=.6); ax.axvline(0, color='.85', lw=.6)
    ax.set(xlabel=r'Whitened residual $u_1$', ylabel=r'Whitened residual $u_2$', aspect='equal')
    radius = np.linalg.norm(fields['residual'], axis=-1)
    x = np.linspace(0, max(3.5, radius.max()*1.05), 250)
    axes[1].hist(radius[0], bins=16, density=True, histtype='step', color='black', lw=1.4, label='Best fit')
    if len(radius) > 1:
        axes[1].hist(radius[1:].ravel(), bins=24, density=True, color='tab:blue', alpha=.18, label='Posterior draws')
    axes[1].plot(x, x*np.exp(-x*x/2), color='tab:red', ls='--', lw=1.2, label='Gaussian reference')
    axes[1].set(xlabel=r'Whitened residual norm $\|u\|$', ylabel='Density')
    axes[1].legend(frameon=False, fontsize=7)
    # Family identity is recorded in the table; the color legend uses space outside the residual plane.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(8, len(labels)), fontsize=6, frameon=False,
               loc='lower center', bbox_to_anchor=(.5, .01), title='Family', title_fontsize=7)
    for ax in axes:
        clean(ax)
    fig.tight_layout(w_pad=2., rect=(0, .20, 1, 1))
    finish(fig, path)


def approximation_pdf(data, path):
    t, f, p = data.members, data.fields, data.provenance
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8))
    for ax, key, ylabel in ((axes[0, 0], 'error_arcsec', 'Deflection error (arcsec)'),
                            (axes[0, 1], 'error_whitened', 'Whitened total error')):
        x = np.arange(len(data.images))
        lo, med, hi = np.quantile(f[key], [.16, .5, .84], axis=0)
        ax.errorbar(x, med, yerr=np.maximum(0., [med-lo, hi-med]), fmt='o', ms=2.5,
                    color='tab:blue', ecolor='#b7cfdf', elinewidth=.7, label='Posterior 16–84%')
        ax.plot(x, f[key][0], '.', color='black', ms=3, label='Best fit')
        if key == 'error_whitened':
            ax.axhline(p['tolerance'], color='tab:red', ls='--', lw=1)
            bad = np.max(f[key], axis=0) >= p['tolerance']
            ax.scatter(x[bad], np.max(f[key], axis=0)[bad], marker='x', color='tab:red', s=18, zorder=5)
        ax.set(xlabel='Constrained image index', ylabel=ylabel, ylim=(0, None))
    far = t[t.member_class == 'far-field']
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax, key, label in ((axes[1, 0], 'member_error_whitened_best', 'Whitened member error'),
                            (axes[1, 1], 'guard_ratio_min', r'Minimum $R/r_{\rm cut}$')):
        if far.empty:
            ax.text(.5, .5, 'All members evaluated exactly', transform=ax.transAxes, ha='center')
        else:
            threshold = p['tolerance'] if key.startswith('member_error') else p['guard_cut_radii']
            values = far[key].to_numpy()
            if key.startswith('member_error'):
                med = far.member_error_whitened_median.to_numpy()
                lo, hi = far.member_error_whitened_p16.to_numpy(), far.member_error_whitened_p84.to_numpy()
                ax.errorbar(far.catalog_mag, med, yerr=np.maximum(0., [med-lo, hi-med]), fmt='o', ms=2.5,
                            color='tab:blue', ecolor='#b7cfdf', elinewidth=.6)
                bad = far.member_error_whitened_max.to_numpy() >= threshold
                ax.scatter(far.catalog_mag, values, s=10, c=np.where(bad, 'tab:red', 'black'), zorder=4)
            else:
                bad = values < threshold
                ax.scatter(far.catalog_mag, values, s=12, c=np.where(bad, 'tab:red', 'tab:blue'))
            ax.axhline(threshold, color='tab:red', ls='--', lw=1)
            ax.invert_xaxis()
        ax.set(xlabel='Catalog magnitude', ylabel=label)
    for ax in axes.flat:
        clean(ax)
    fig.tight_layout(w_pad=2.)
    finish(fig, path)


def scaling_correlations(samples, specs, population, path):
    names = {spec.name: i for i, spec in enumerate(specs)}
    def get(field):
        i = names.get(f'{population}.{field}')
        return None if i is None else np.asarray(samples)[:, i]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0))
    for ax, fields, labels in ((axes[0, 0], ('alpha_sigma', 'sigma'), (r'$\alpha_\sigma$', r'$\sigma_\star$ (km s$^{-1}$)')),
                               (axes[0, 1], ('beta_radius', 'cutkpc'), (r'$\beta_r$', r'$r_{{\rm cut},\star}$ (kpc)')),
                               (axes[1, 0], ('velocity_log_scatter', 'radius_log_scatter'), (r'$s_\sigma$', r'$s_r$'))):
        x, y = map(get, fields)
        if x is None or y is None:
            ax.text(.5, .5, 'Parameter pair not sampled', transform=ax.transAxes, ha='center', fontsize=8)
        else:
            ax.scatter(x, y, s=3, alpha=.15, color='tab:blue', linewidths=0, rasterized=True)
        ax.set(xlabel=labels[0], ylabel=labels[1])
    alpha, beta = get('alpha_sigma'), get('beta_radius')
    ax = axes[1, 1]
    if alpha is not None and beta is not None:
        exponent = 2*alpha+beta
        ax.hist(exponent, bins=30, density=True, histtype='step', lw=1.4, color='black', label=r'$2\alpha_\sigma+\beta_r$')
        ax.hist(exponent-1, bins=30, density=True, histtype='step', lw=1.4, color='tab:blue', label=r'$\gamma_{\rm ML}$')
        ax.legend(frameon=False, fontsize=8)
    ax.set(xlabel='Scaling exponent', ylabel='Density')
    for ax in axes.flat:
        clean(ax)
    fig.tight_layout(w_pad=2.)
    finish(fig, path)


def write_member_diagnostic_plots(config, state, evaluator, posterior, paths, *, log=print):
    """Normal-run and saved-run entry point; all reporting shares one numerical dataset."""
    if not len(state.members.member_ids):
        log('[diagnostics] member suite skipped: no population members')
        return []
    data = compute_member_diagnostics(state, evaluator, posterior, config, log=log)
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    data.members.to_csv(paths.tables / 'member_diagnostics.csv', index=False)
    data.images.to_csv(paths.tables / 'diagnostic_image_index.csv', index=False)
    np.savez_compressed(paths.tables / 'member_diagnostic_draws.npz', **data.fields,
                        deviations=data.deviations, prior_deviations=data.prior_deviations)
    rows = []
    for d, index in enumerate(data.provenance['selected_draw_indices']):
        frame = data.images.copy()
        frame['draw_index'] = index
        frame['best_fit'] = d == 0
        for key in ('error_arcsec', 'error_whitened'):
            frame[key] = data.fields[key][d]
        frame['whitened_u1'], frame['whitened_u2'] = data.fields['residual'][d].T
        rows.append(frame)
    pd.concat(rows).to_csv(paths.tables / 'image_field_diagnostics.csv', index=False)
    far = data.members[data.members.member_class == 'far-field']
    data.provenance['failed_guard_members'] = far.loc[far.guard_ratio_min < data.provenance['guard_cut_radii'], 'member_id'].tolist()
    data.provenance['failed_error_members'] = far.loc[far.member_error_whitened_max >= data.provenance['tolerance'], 'member_id'].tolist()
    data.provenance['failed_error_images'] = data.images.loc[np.max(data.fields['error_whitened'], axis=0) >= data.provenance['tolerance'], 'image_label'].tolist()
    log(f"[diagnostics] selected-model max whitened error={data.fields['error_whitened'].max():.5g} "
        f"(limit={data.provenance['tolerance']:g}); guard failures={len(data.provenance['failed_guard_members'])}")
    (paths.tables / 'member_diagnostic_provenance.json').write_text(json.dumps(data.provenance, indent=2)+'\n')
    prefix = config.paths.figure_prefix
    def output(name):
        return paths.figures / ((f'{prefix}_' if prefix else '') + name + '.pdf')
    products = []
    with plt.rc_context(fname=STYLE), plt.rc_context(RC):
        for name, renderer in (('member_deviations', deviation_pdf), ('member_constraints', constraint_pdf),
                               ('whitened_image_residuals', residual_pdf), ('far_field_accuracy', approximation_pdf)):
            path = output(name)
            renderer(data, path)
            products.append(path)
    log(f'[diagnostics] wrote {len(products)} PDFs; {len(data.variables)} member comparisons')
    return products
