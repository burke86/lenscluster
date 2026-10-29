"""Shared posterior corners with recorded priors in the displayed coordinates."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .prior_diagnostics import saved_prior

STYLE = Path(__file__).with_name('style.mplstyle')


class DisplayPrior:
    """Affine change from a sample-site prior to the displayed coordinate."""
    def __init__(self, spec):
        self.base = saved_prior(spec)
        self.scale = 1 / np.log(10) if spec.transform_kind.startswith('log_') else (getattr(spec, 'transform_scale', 1.) if spec.transform_kind == 'affine' else 1.)
        self.offset = getattr(spec, 'transform_offset', 0.) if spec.transform_kind == 'affine' else 0.
    def pdf(self, x):
        return self.base.pdf((x-self.offset)/self.scale)/abs(self.scale)
    def ppf(self, q):
        return self.offset + self.scale*self.base.ppf(q)


def display_values(values, spec):
    if spec.transform_kind == 'log_positive':
        return np.log10(values)
    if spec.transform_kind == 'log_offset_positive':
        return np.log10(values-getattr(spec, 'transform_offset', 0.))
    return values


def plot_parameter_corner(samples, selection, path, *, grouped_physical=None, best_index=None):
    import corner
    samples = np.asarray(samples)
    selection = [(i, spec, label) for i, spec, label in selection if np.ptp(samples[:, i]) > 0]
    if not selection or len(samples) <= len(selection)+1:
        return None
    grouped = None
    if grouped_physical is not None:
        grouped_physical = np.asarray(grouped_physical)
        if grouped_physical.ndim != 3 or not np.array_equal(grouped_physical.reshape(samples.shape), samples):
            raise ValueError('Corner traces require chain-ordered physical samples matching the corner draws')
    values = np.column_stack([display_values(samples[:,i], spec) for i,spec,_ in selection])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite posterior corner coordinates')
    if grouped_physical is not None:
        grouped = values.reshape(*grouped_physical.shape[:2], len(selection))
    best_values = values[best_index] if best_index is not None else None
    priors, ranges, labels = [], [], []
    for j, (_, spec, label) in enumerate(selection):
        prior = DisplayPrior(spec)
        priors.append(prior)
        prior_low, prior_high = prior.ppf(np.array([.001, .999]))
        low, high = min(values[:,j].min(), prior_low), max(values[:,j].max(), prior_high)
        ranges.append((low-.04*(high-low), high+.04*(high-low)))
        if spec.transform_kind.startswith('log_'):
            symbol = label.split('$')[1] if '$' in label else label.replace('_', r'\_')
            if spec.transform_kind == 'log_offset_positive':
                symbol += f'-{getattr(spec, "transform_offset", 0.):g}'
            units = label.split('$', 2)[2] if '$' in label else ''
            label = r'$\log_{10}(' + symbol + ')$' + units
        labels.append(label)
    n = len(selection)
    with plt.rc_context(fname=STYLE,
                        rc={'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'axes.labelsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11}):
        fig = plt.figure(figsize=(max(7.2, 1.8*n), max(7.2, 1.8*n)))
        corner.corner(values, fig=fig, labels=labels, range=ranges, bins=40, smooth=1, smooth1d=None,
                      color='black', levels=(.68, .95), plot_datapoints=False, plot_density=False,
                      fill_contours=True, no_fill_contours=False,
                      contourf_kwargs={'colors': ['white', '#E6E6E6', '#AAAAAA', 'black']},
                      contour_kwargs={'linewidths': .85}, quantiles=(.16, .5, .86),
                      hist_kwargs={'density': True, 'linewidth': 1.3}, max_n_ticks=3, labelpad=.08)
        axes = np.array(fig.axes).reshape(n, n)
        # These hyperpriors are independent. Integrate their product density on
        # a wider grid so displayed ranges do not renormalize the joint prior.
        for row in range(n):
            for col in range(row):
                grids = [np.linspace(*(priors[j].ppf([1e-7, 1-1e-7])), 700)
                         for j in (col, row)]
                densities = [priors[j].pdf(grid)
                             for j, grid in zip((col, row), grids)]
                joint = np.outer(densities[1], densities[0])
                area = (grids[0][1]-grids[0][0])*(grids[1][1]-grids[1][0])
                ranked = np.sort(joint.ravel())[::-1]
                mass = np.cumsum(ranked)*area
                thresholds = np.unique([ranked[np.searchsorted(mass, probability)] for probability in (.68, .95)])
                if np.allclose(joint, joint.max(), rtol=1e-12, atol=0):
                    # A uniform joint prior has no unique highest-density
                    # 68/95 percent regions. Show its support boundary instead.
                    x0, x1 = priors[col].ppf(np.array([0., 1.]))
                    y0, y1 = priors[row].ppf(np.array([0., 1.]))
                    axes[row, col].plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0],
                                        color='tab:orange', ls='--', lw=1.1, zorder=5)
                else:
                    axes[row, col].contour(*grids, joint, levels=thresholds, colors=['tab:orange'],
                                           linestyles='--', linewidths=1.1, zorder=5)
                axes[row, col].set_xlim(ranges[col])
                axes[row, col].set_ylim(ranges[row])
        for j, prior in enumerate(priors):
            x = np.linspace(*ranges[j], 1000)
            density = prior.pdf(x)
            axes[j, j].plot(x, density, color='tab:orange', ls='--', lw=1.5)
            axes[j, j].set_ylim(0, max(axes[j,j].get_ylim()[1], 1.05*density.max()))
            low, median, high = np.quantile(values[:, j], [.16, .5, .86])
            symbol = labels[j].split('$')[1] if '$' in labels[j] else labels[j].replace('_', r'\_')
            axes[j, j].text(.5, 1.04, rf'${symbol}={median:.3f}^{{+{high-median:.3f}}}_{{-{median-low:.3f}}}$',
                            transform=axes[j,j].transAxes, ha='center', fontsize=8)
            if best_index is not None:
                axes[j, j].axvline(best_values[ j], color='tab:red', lw=1, zorder=10)
            for k in range(j) if best_index is not None else ():
                axes[j, k].plot(best_values[ k], best_values[ j], 'x', color='tab:red', ms=5, mew=1.1, zorder=10)
        fig.subplots_adjust(wspace=0, hspace=0)
        if grouped is not None and n >= 3:
            middle = max(1, n // 2)
            left = axes[0, middle].get_position().x0 + .10
            right = axes[0, -1].get_position().x1
            top = axes[0, 0].get_position().y1 - .07
            bottom = axes[middle, middle].get_position().y1 + .13
            if n <= 4:
                left, right, top, bottom = .66, .96, .90, .64
            height = (top-bottom)/n
            for j in range(n):
                ax = fig.add_axes([left, top-(j+1)*height, right-left, height*.72])
                for chain in range(grouped.shape[0]):
                    ax.plot(np.arange(1, grouped.shape[1]+1), grouped[chain,:,j],
                            lw=.45, alpha=.75, rasterized=True)
                if best_index is not None:
                    ax.plot(best_index % grouped.shape[1]+1, best_values[j], 'x', color='tab:red', ms=6, zorder=10)
                trace_label = '$' + labels[j].split('$')[1] + '$' if '$' in labels[j] else labels[j]
                ax.set_ylabel(trace_label, fontsize=8, rotation=0, labelpad=24)
                ax.tick_params(labelsize=7, labelbottom=j==n-1, top=False, right=False)
                ax.minorticks_off()
                for spine in ax.spines.values():
                    spine.set_linewidth(.6)
                ax.spines[['top','right']].set_visible(False)
                if j == n-1:
                    ax.set_xlabel('Post-warmup draw per chain', fontsize=9)
        fig.legend(handles=[Line2D([], [], color='black', label='Posterior'),
                            Line2D([], [], color='tab:orange', ls='--', label='Prior'),
                            Line2D([], [], color='tab:red', marker='x', ls='none', label='Best fit')],
                   loc='upper center', bbox_to_anchor=(.76, .98), fontsize=9, ncol=3, frameon=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
    metadata = dict(draws=len(values), parameters=[spec.name for _,spec,_ in selection],
                    best_posterior_draw_index=best_index, ranges=ranges,
                    trace_chains=0 if grouped is None else grouped.shape[0],
                    trace_draws_per_chain=0 if grouped is None else grouped.shape[1],
                    contour_probabilities=[.68,.95], marginal_quantiles=[.16,.5,.86],
                    uniform_joint_prior='support boundary; highest-density regions are not unique',
                    coordinates='log10 for log-transformed parameters; physical otherwise')
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return path
