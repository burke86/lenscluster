"""Potfile-only posterior figures shared by run outputs and regeneration tools."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np
from scipy import stats

from .prior_diagnostics import saved_prior

STYLE = Path(__file__).with_name("style.mplstyle")
PARAMETERS = (
    ("sigma", r"$\sigma_\star$ (km s$^{-1}$)"),
    ("cutkpc", r"$r_{{\rm cut},\star}$ (kpc)"),
    ("corekpc", r"$r_{{\rm core},\star}$ (kpc)"),
    ("alpha_sigma", r"$\alpha_\sigma$"),
    ("beta_radius", r"$\beta_r$"),
    ("velocity_log_scatter", r"$s_\sigma$"),
    ("radius_log_scatter", r"$s_r$"),
)


def potfile_populations(specs):
    """Stable population names, including named catalogs rather than only numeric IDs."""
    fields = {field for field, _ in PARAMETERS}
    return sorted({spec.name.rsplit(".", 1)[0] for spec in specs
                   if spec.field in fields and spec.component_family in {"scaling", "member_velocity_scatter"}})


def _selection(samples, specs, population):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(specs):
        raise ValueError("Physical samples and parameter specifications disagree")
    selection = [(i, spec, label) for field, label in PARAMETERS for i, spec in enumerate(specs)
                 if spec.name == f"{population}.{field}"]
    if selection and not np.isfinite(values[:, [i for i, _, _ in selection]]).all():
        raise ValueError(f"Nonfinite posterior for {population}")
    return values, selection


def _style():
    return plt.rc_context(fname=STYLE)


@_style()
def plot_potfile_corner(samples, specs, population, path, *, grouped_physical=None, best_index=None):
    import corner
    samples, selection = _selection(samples, specs, population)
    if len(samples) < 3 or len(selection) < 2:
        return None
    selection = [(i, spec, label) for i, spec, label in selection if np.ptp(samples[:, i]) > 0]
    if len(selection) < 2 or len(samples) <= len(selection) + 1:   # corner needs more samples than dimensions
        return None
    values = samples[:, [i for i, _, _ in selection]]
    if best_index is not None and not 0 <= best_index < len(samples):
        raise ValueError("Best posterior index is outside the saved draws")
    spans = np.ptp(values, axis=0)
    # Full saved sample ranges, padded for contours. No tail trimming or draw filtering.
    ranges = []
    for j, (_, spec, _) in enumerate(selection):
        low, high = values[:, j].min(), values[:, j].max()
        low -= .04 * spans[j]
        high += .04 * spans[j]
        if spec.transform_kind == "log_positive":
            low = max(0., low)
        ranges.append((low, high))
    plt.rcParams.update({"font.family": "sans-serif",
                        "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
                        "pdf.fonttype": 42, "ps.fonttype": 42})
    contour_color = "black"
    n = len(selection)
    side = max(6., 1.8 * n)
    fig = plt.figure(figsize=(side, side))
    corner.corner(values, fig=fig, labels=[label for _, _, label in selection],
                  range=ranges, bins=32, smooth=1., smooth1d=.8,
                  color=contour_color, levels=(.68, .95), plot_datapoints=False,
                  plot_density=False, fill_contours=True, no_fill_contours=False,
                  contourf_kwargs={"colors": ["white", "#E6E6E6", "#AAAAAA", contour_color]},
                  contour_kwargs={"linewidths": .85},
                  hist_kwargs={"linewidth": 1.5},
                  quantiles=(.16, .5, .86), show_titles=False,
                  max_n_ticks=3, labelpad=.12)
    fig.subplots_adjust(wspace=0, hspace=0)
    axes = np.asarray(fig.axes).reshape(n, n)
    for row in range(n):
        for col in range(row + 1):
            axes[row, col].grid(False)
            axes[row, col].tick_params(pad=3)
    for j, (_, spec, label) in enumerate(selection):
        low, median, high = np.quantile(values[:, j], [.16, .5, .86])
        minus, plus = median - low, high - median
        precision_scale = min(v for v in (minus, plus, spans[j]) if v > 0)
        decimals = max(0, 1 - int(np.floor(np.log10(precision_scale))))
        symbol = label.split("$")[1]
        annotation = (rf"${symbol} = {median:.{decimals}f}"
                      rf"^{{+{plus:.{decimals}f}}}_{{-{minus:.{decimals}f}}}$")
        axes[j, j].set_title(annotation, fontsize=.75 * plt.rcParams["font.size"], pad=8)
    if best_index is not None:
        for row in range(n):
            axes[row, row].axvline(values[best_index, row], color="tab:red", lw=1.1, zorder=6)
            for col in range(row):
                axes[row, col].plot(values[best_index, col], values[best_index, row],
                                    marker="x", color="tab:red", ms=9,
                                    markeredgewidth=1.7, linestyle="none", zorder=6)
    if grouped_physical is not None:
        grouped = np.asarray(grouped_physical)
        if grouped.ndim != 3 or grouped.shape[2] != samples.shape[1] or not np.array_equal(grouped.reshape(samples.shape), samples):
            raise ValueError("Corner traces require chain-ordered physical samples matching the corner draws")
        middle = max(1, n // 2)
        left = axes[0, middle].get_position().x0 + .055
        right = axes[0, -1].get_position().x1 - .015
        bottom = axes[middle, middle].get_position().y1 + .065
        top = axes[0, 0].get_position().y1 - .025
        height = (top - bottom) / n
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown", "tab:pink", "tab:gray"]
        handles = []
        for j, (index, _, label) in enumerate(selection):
            ax = fig.add_axes([left, top - (j + 1) * height, right - left, height * .72])
            for chain in range(grouped.shape[0]):
                line, = ax.plot(np.arange(1, grouped.shape[1] + 1), grouped[chain, :, index],
                               color=palette[chain % len(palette)], lw=.45, alpha=.75,
                               label=f"Chain {chain + 1}", rasterized=True)
                if j == 0:
                    handles.append(line)
            if best_index is not None:
                draw = best_index % grouped.shape[1] + 1
                ax.plot(draw, samples[best_index, index], marker="x", color="tab:red", ms=8,
                        markeredgewidth=1.7, linestyle="none", zorder=6)
            ax.set_xlim(.5, grouped.shape[1] + .5)
            ax.set_ylabel('$' + label.split('$')[1] + '$', fontsize=11, rotation=0, labelpad=18)
            ax.yaxis.set_label_coords(-.075, .35)
            ax.yaxis.set_major_locator(MaxNLocator(2, prune="both"))
            ax.tick_params(axis="both", which="major", labelsize=8, length=2, width=.6,
                           top=False, right=False, labelbottom=j == n - 1)
            ax.minorticks_off()
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(.6)
            if j == n - 1:
                ax.set_xlabel("Post-warmup draw per chain", fontsize=10, labelpad=4)
        if best_index is not None:
            handles.append(Line2D([], [], marker="x", color="tab:red", linestyle="none", markersize=9, markeredgewidth=1.7,
                                  label="Best posterior"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=((left + right) / 2, top + .005),
                   ncol=min(len(handles), 5), fontsize=9, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({
        "population": population, "draws": len(samples), "sample_space": "physical",
        "best_posterior_draw_index": int(best_index) if best_index is not None else None,
        "trace_chains": int(np.asarray(grouped_physical).shape[0]) if grouped_physical is not None else 0,
        "trace_draws_per_chain": int(np.asarray(grouped_physical).shape[1]) if grouped_physical is not None else 0,
        "contour_probabilities": [.68, .95], "marginal_quantiles": [.16, .5, .86],
        "parameters": {spec.name: dict(zip(("q16", "q50", "q86"),
                       np.quantile(values[:, j], [.16, .5, .86]).tolist()))
                       for j, (_, spec, _) in enumerate(selection)}}, indent=2) + "\n")
    return path


@_style()
def plot_potfile_prior_posterior(samples, specs, population, path):
    samples, selection = _selection(samples, specs, population)
    if len(samples) < 3 or not selection:
        return None
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular", "font.size": 9, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .8,
        "xtick.major.size": 3, "xtick.major.width": .7, "xtick.top": False,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "axes.grid": False, "pdf.fonttype": 42})
    rows = (len(selection) + 4) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(7.1, 1.9 * rows), squeeze=False)
    blue, gray = "tab:blue", "#777777"
    summary = []
    for ax, (index, spec, label) in zip(axes.flat, selection):
        draws = samples[:, index]
        if not np.isfinite(draws).all():
            raise ValueError(f"Nonfinite posterior for {spec.name}")
        is_log = spec.transform_kind == "log_positive"
        if spec.transform_kind not in {"log_positive", "identity"}:
            raise ValueError("Unexpected potfile transform")
        values = np.log(draws) if is_log else draws
        prior = saved_prior(spec)
        prior_q = prior.ppf([.16, .5, .84])
        post_q = np.quantile(values, [.16, .5, .84])
        low = min(prior.ppf(.001), values.min())
        high = max(prior.ppf(.999), values.max())
        width = high-low
        low, high = low-.04*width, high+.04*width
        grid = np.linspace(low, high, 1200)
        x = np.exp(grid) if is_log else grid
        if np.ptp(values) == 0:
            # A short diagnostic run can have a fixed chain; do not invent a KDE width.
            posterior_density = np.zeros_like(grid)
        else:
            posterior_density = stats.gaussian_kde(values)(grid)
        # Both densities use the displayed coordinate: ln(parameter) on log axes.
        ax.fill_between(x, prior.pdf(grid), color=gray, alpha=.10, linewidth=0)
        ax.plot(x, prior.pdf(grid), color=gray, lw=1.1, ls=(0, (4, 2)))
        ax.fill_between(x, posterior_density, color=blue, alpha=.13, linewidth=0)
        ax.plot(x, posterior_density, color=blue, lw=1.5)
        peak = max(prior.pdf(grid).max(), posterior_density.max())
        if np.ptp(values) == 0:
            ax.vlines(np.exp(values[0]) if is_log else values[0], 0, peak, color=blue, lw=1.5)
        for quantile, color, marker in ((prior_q[1], gray, "o"), (post_q[1], blue, "D")):
            position = np.exp(quantile) if is_log else quantile
            ax.plot(position, -.065*peak, marker=marker, color=color,
                    ms=3.5, markeredgewidth=0, clip_on=False)
        if is_log:
            ax.set_xscale("log")
            candidates = np.array([v*10.**e for e in range(-5, 6) for v in (1, 3)])
            ticks = candidates[(candidates > np.exp(low)) & (candidates < np.exp(high))]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.minorticks_off()
        else:
            ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-.12*peak, 1.10*peak)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel(label, labelpad=5)
        ax.grid(False)
        back = np.exp if is_log else np.asarray
        summary.append({"parameter": spec.name, "coordinate": "ln" if is_log else "linear",
            "prior_q16_q50_q84": back(prior_q).tolist(), "posterior_q16_q50_q84": back(post_q).tolist(),
            "median_shift_prior_sd": float((post_q[1]-prior_q[1])/prior.std()),
            "posterior_to_prior_68_width": float((post_q[2]-post_q[0])/(prior_q[2]-prior_q[0]))})
    for ax in list(axes.flat)[len(selection):]:
        ax.set_axis_off()
    key = axes.flat[-1]
    key.legend(handles=[Line2D([], [], color=gray, lw=1.1, ls="--", label="Prior"),
                        Line2D([], [], color=blue, lw=1.5, label="Posterior")],
               loc="center", frameon=False, fontsize=10, handlelength=2.5)
    key.text(.5, .17, "Markers show medians", transform=key.transAxes,
             ha="center", va="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=.045, right=.985, bottom=.16, top=.975, hspace=.55, wspace=.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({
        "population": population, "draws": len(samples),
        "density_coordinate": "ln(parameter) for log axes; parameter otherwise",
        "parameters": summary}, indent=2) + "\n")
    return path


def write_potfile_plots(samples, specs, figures, prefix=None, *, posterior=None, log=print):
    """Create both figures for every population using the run's naming convention."""
    results = []
    grouped = None
    best_index = None
    if posterior is not None:
        from .chain_traces import trace_data
        traces = trace_data(posterior, specs)
        if traces is not None:
            grouped = traces["chains"]
        log_prob = np.asarray(posterior.log_prob)
        if np.isfinite(log_prob).any():
            best_index = int(np.nanargmax(log_prob))
    for population in potfile_populations(specs):
        for name, renderer in (("corner", plot_potfile_corner), ("prior_posterior", plot_potfile_prior_posterior)):
            basename = f"{population}_{name}.pdf"
            path = Path(figures) / (f"{prefix}_{basename}" if prefix else basename)
            kwargs = {"grouped_physical": grouped, "best_index": best_index} if name == "corner" else {}
            result = renderer(samples, specs, population, path, **kwargs)
            if result is None:
                log(f"[outputs] {population} {name} skipped: insufficient varying posterior draws")
            else:
                results.append(result)
    return results
