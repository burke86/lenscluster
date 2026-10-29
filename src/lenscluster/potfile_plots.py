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


def plot_potfile_corner(samples, specs, population, path, *, grouped_physical=None, best_index=None):
    from .corner_plots import plot_parameter_corner
    samples, selection = _selection(samples, specs, population)
    result = plot_parameter_corner(samples, selection, path, grouped_physical=grouped_physical, best_index=best_index)
    if result is not None:
        metadata = json.loads(path.with_suffix('.json').read_text())
        metadata['population'] = population
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return result


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
    """Create a prior-overlaid corner for every population using the run's naming convention."""
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
        for name, renderer in (("corner", plot_potfile_corner),):
            basename = f"{population}_{name}.pdf"
            path = Path(figures) / (f"{prefix}_{basename}" if prefix else basename)
            kwargs = {"grouped_physical": grouped, "best_index": best_index} if name == "corner" else {}
            result = renderer(samples, specs, population, path, **kwargs)
            if result is None:
                log(f"[outputs] {population} {name} skipped: insufficient varying posterior draws")
            else:
                results.append(result)
    return results
