"""Shared block-colored posterior corner renderer."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BLOCK_COLORS = {'Halo': 'tab:blue', 'BCG': 'tab:purple', 'GRF + GP': 'tab:red',
                'Scaling relation': 'tab:orange', 'Sources': 'tab:green'}
STYLE = Path(__file__).with_name('style.mplstyle')
PAPER_STYLE = {'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans',
               'mathtext.default': 'regular', 'axes.grid': False, 'pdf.fonttype': 42}


def annotate_corner_summary(ax, values, weights, *, color='.2'):
    """Mark the marginal median without text or interval shading."""
    import corner
    median = corner.quantile(values, [.5], weights=weights)[0]
    ax.axvline(median, color=color, linewidth=1.2, linestyle=':')


def render_block_corner(values, weights, labels, groups, path, *, zero_reference=False,
                        summarize=False):
    import corner
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(np.ptp(values, axis=0) <= 0):
        raise ValueError('Corner requires finite nonconstant coordinates')
    n = values.shape[1]
    colors = [BLOCK_COLORS[g] for g in groups]
    ranges = [(min(0., v.min())-.04*np.ptp(v), max(0., v.max())+.04*np.ptp(v))
              if zero_reference else (v.min()-.04*np.ptp(v), v.max()+.04*np.ptp(v))
              for v in values.T]
    path = Path(path)
    with plt.rc_context(fname=STYLE, rc=PAPER_STYLE):
        fig, axes = plt.subplots(n, n, figsize=(max(4, 1.35*n), max(4, 1.35*n)), squeeze=False)
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]
                if col > row:
                    ax.set_visible(False); continue
                color = colors[row]
                if row == col:
                    hist, edges = np.histogram(values[:, row], bins=40, range=ranges[row], weights=weights)
                    ax.plot((edges[1:] + edges[:-1])/2, gaussian_filter1d(hist.astype(float), 1), color=color, lw=2.0)
                    ax.set_xlim(ranges[row]); ax.set_ylim(bottom=0); ax.set_yticks([])
                    if summarize:
                        annotate_corner_summary(ax, values[:, row], weights, color=color)
                else:
                    corner.hist2d(values[:, col], values[:, row], weights=weights,
                                  bins=40, range=[ranges[col], ranges[row]], smooth=1,
                                  color=color, levels=(.68, .95), plot_datapoints=False,
                                  plot_density=False, fill_contours=True,
                                  contour_kwargs={'linewidths': 1.8}, ax=ax)
                    if zero_reference:
                        ax.axhline(0, color='.3', ls='--', lw=1.0)
                    ax.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
                if zero_reference:
                    ax.axvline(0, color='.3', ls='--', lw=1.0)
                ax.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
                ax.tick_params(labelsize=7, labelbottom=row == n-1, labelleft=col == 0 and row > 0)
                ax.grid(False)
                if row == n-1: ax.set_xlabel(labels[col], fontsize=9, color=colors[col])
                if col == 0 and row > 0: ax.set_ylabel(labels[row], fontsize=9, color=color)
                if row == 0 or groups[row] != groups[row-1]:
                    ax.spines['top'].set_color(color); ax.spines['top'].set_linewidth(1.4)
                if col == 0 or groups[col] != groups[col-1]:
                    ax.spines['left'].set_color(colors[col]); ax.spines['left'].set_linewidth(1.4)
        unique = list(dict.fromkeys(zip(groups, colors)))
        fig.legend([Line2D([], [], color=c, lw=10) for _, c in unique],
                   [g for g, _ in unique], loc='upper right', bbox_to_anchor=(.94, .94),
                   frameon=False, fontsize=max(14, 64*n/26), handlelength=2.4, labelspacing=.7)
        fig.subplots_adjust(left=.045, right=.995, bottom=.045, top=.995, hspace=.035, wspace=.035)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
    return path


def posterior_corner_data(samples, specs):
    """Physical global parameters plus paired, standardized source displacements.

    Member-by-member latent offsets are omitted. Global draws are repeated for
    each family, preserving their association with that family's source draws.
    """
    from .corner_plots import display_values
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(specs):
        raise ValueError('Physical posterior and parameter specifications disagree')
    selected = []
    excluded = []
    sources = {}
    for i, spec in enumerate(specs):
        if spec.component_family == 'source_position':
            sources.setdefault(str(spec.potential_id), {})[spec.field] = i
            continue
        if spec.component_family in {'member_scatter_offset', 'member_shape_offset'}:
            excluded.append(spec.name)
            continue
        if not np.isfinite(samples[:, i]).all():
            raise ValueError(f'Nonfinite posterior: {spec.name}')
        if np.ptp(samples[:, i]) == 0:
            excluded.append(spec.name)
            continue
        if 'bcg' in spec.potential_id.lower():
            group = 'BCG'
        elif spec.component_family in {'scaling', 'member_velocity_scatter'}:
            group = 'Scaling relation'
        elif spec.component_family == 'error_scale':
            group = 'GRF + GP'
        else:
            group = 'Halo'
        selected.append((i, spec, group))
    selected.sort(key=lambda item: list(BLOCK_COLORS).index(item[2]))
    if not selected:
        return None
    globals_ = np.column_stack([display_values(samples[:, i], spec) for i, spec, _ in selected])
    labels = []
    for _, spec, _ in selected:
        label = spec.name.replace('_', r'\_')
        if spec.transform_kind.startswith('log_'):
            offset = f'-{spec.transform_offset:g}' if spec.transform_kind == 'log_offset_positive' else ''
            label = r'\log(\mathrm{' + label + '}' + offset + ')'
        else:
            label = r'\mathrm{' + label + '}'
        labels.append('$' + label + '$')
    source_arrays, families, source_scales = [], [], {}
    for family, coordinates in sorted(sources.items()):
        if not {'beta_x', 'beta_y'} <= coordinates.keys():
            continue
        raw = samples[:, [coordinates['beta_x'], coordinates['beta_y']]]
        lo, center, hi = np.percentile(raw, [16, 50, 84], axis=0)
        scale = (hi-lo)/2
        if not np.isfinite(raw).all() or np.any(scale <= 0):
            continue
        source_arrays.append((raw-center)/scale)
        families.append(family)
        source_scales[family] = dict(center=center.tolist(), s68=scale.tolist())
    groups = [g for _, _, g in selected]
    parameters = [spec.name for _, spec, _ in selected]
    if source_arrays:
        values = np.concatenate([np.column_stack([globals_, source]) for source in source_arrays])
        labels.extend([r'$\Delta\beta_x/s_{68,x}$', r'$\Delta\beta_y/s_{68,y}$'])
        groups.extend(['Sources', 'Sources'])
        parameters.extend(['pooled_source.beta_x', 'pooled_source.beta_y'])
    else:
        values = globals_
    return dict(values=values, labels=labels, groups=groups, parameters=parameters,
                weights=np.full(len(values), 1/len(values)), excluded=excluded,
                source_families=families, source_scales=source_scales)


def write_posterior_block_corner(samples, specs, path):
    """Write the normal-run counterpart of the aggregate truth-deviation corner."""
    import json
    data = posterior_corner_data(samples, specs)
    if data is None or len(samples) < 3:
        return None
    path = render_block_corner(data['values'], data['weights'], data['labels'], data['groups'], path)
    metadata = {key: value for key, value in data.items() if key not in {'values', 'weights'}}
    metadata.update(draws=len(samples), colors=BLOCK_COLORS,
                    coordinates='Physical globals; log10 for log-transformed parameters. Sources are median-centred and divided by their own half 16-84 interval.',
                    source_pooling='Equal family weights, paired x/y with original joint global draw alignment; not a single physical source posterior.',
                    contour_color='Row block', contour_probabilities=[.68, .95],
                    bcg_identification='Explicit bcg token in potential_id; other standalone components remain in Halo.')
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return path
