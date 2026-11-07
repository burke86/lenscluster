"""Diagnostic figures of the init stage: did the search converge, and is the start a good lens model?"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluator import Evaluator
from .model import ModelState, member_log_deviations
from .plots import figure_path, plot_critical_curves

MODE_COLORS = plt.get_cmap("tab10")


def plot_convergence(search: dict[str, Any], path: Path, *, margin: float) -> Path:
    """Log posterior of every start against iteration; the best start and the margin are marked."""
    traces = -np.asarray(search["traces"], dtype=float)
    potential = np.asarray(search["potential"], dtype=float)
    assignment = np.asarray(search["assignment"])
    best_value = np.nanmax(-potential[np.isfinite(potential)])
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.4))
    order = np.argsort(potential)
    for rank, i in enumerate(order):
        trace = traces[i]
        valid = np.isfinite(trace)
        if not valid.any():
            continue
        color = MODE_COLORS(assignment[i] % 10) if assignment[i] >= 0 else "0.7"
        for axis in (ax, ax_zoom):
            axis.plot(np.flatnonzero(valid), trace[valid], color=color, lw=1.6 if rank == 0 else 0.7, alpha=1.0 if rank == 0 else 0.55)
    for axis in (ax, ax_zoom):
        axis.axhline(best_value - margin, color="0.3", ls="--", lw=1.0)
        axis.set_xlabel("iteration")
    ax.set_ylabel("log posterior")
    ax.set_yscale("symlog", linthresh=10.0)
    ax_zoom.set_ylim(best_value - 5 * margin, best_value + 0.2 * margin)
    ax_zoom.set_ylabel("log posterior (within 5 margins of the best)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_stage_convergence(search: dict[str, Any], path: Path) -> Path:
    """One panel per stage: log posterior of the carried starts against optimizer step."""
    traces = search["stage_traces"]
    fig, axes = plt.subplots(1, len(traces), figsize=(4.6 * len(traces), 4.0), squeeze=False)
    for ax, (name, values) in zip(axes[0], traces.items()):
        values = -np.asarray(values, dtype=float)
        best = np.nanmax(values[np.isfinite(values)]) if np.isfinite(values).any() else 0.0
        for i, trace in enumerate(values):
            ax.plot(np.clip(trace, best - 2000, None), lw=0.7, alpha=0.7)
        ax.set_ylim(best - 200, best + 5)
        ax.set_title(name.replace("_", " "), fontsize=11)
        ax.set_xlabel("step")
    axes[0, 0].set_ylabel("log posterior (clipped 200 below the best)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_modes(state: ModelState, search: dict[str, Any], path: Path, *, margin: float, max_parameters: int = 12) -> Path:
    """Sorted end-point log posteriors, and the global parameters of every start coloured by mode."""
    potential = np.asarray(search["potential"], dtype=float)
    assignment = np.asarray(search["assignment"])
    final_theta = np.asarray(search["final_theta"], dtype=float)
    finite = np.isfinite(potential)
    best_value = np.nanmax(-potential[finite])
    specs = state.parameter_specs
    global_specs = [(i, s) for i, s in enumerate(specs) if s.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"}]
    global_specs = [(i, s) for i, s in global_specs if np.nanstd(final_theta[finite, i]) > 0][:max_parameters]
    n_panels = 1 + len(global_specs)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows), squeeze=False)
    flat = list(axes.ravel())
    ax = flat[0]
    order = np.argsort(potential)
    values = -potential[order]
    colors = [MODE_COLORS(assignment[i] % 10) if assignment[i] >= 0 else "0.7" for i in order]
    ax.scatter(np.arange(len(order)), np.clip(values, best_value - 50 * margin, None), c=colors, s=22, zorder=3)
    ax.axhline(best_value - margin, color="0.3", ls="--", lw=1.0, label="mode margin")
    ax.set_xlabel("start (sorted)")
    ax.set_ylabel("log posterior at the end point")
    ax.set_ylim(best_value - 50 * margin, best_value + margin)
    ax.legend(loc="lower left", fontsize=9)
    from .model import convert_theta_to_physical
    physical = np.stack([convert_theta_to_physical(t, specs) if np.isfinite(t).all() else np.full(t.size, np.nan) for t in final_theta])
    for axis, (i, spec) in zip(flat[1:], global_specs):
        y = -potential
        axis.scatter(physical[finite, i], np.clip(y[finite], best_value - 50 * margin, None),
                     c=[MODE_COLORS(a % 10) if a >= 0 else "0.7" for a in assignment[finite]], s=18, zorder=3)
        axis.axhline(best_value - margin, color="0.3", ls="--", lw=0.8)
        axis.set_xlabel(spec.name, fontsize=9)
        axis.set_ylabel("log posterior", fontsize=9)
        axis.set_ylim(best_value - 50 * margin, best_value + margin)
        axis.tick_params(labelsize=8)
    for axis in flat[n_panels:]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_fit_at_start(state: ModelState, evaluator: Evaluator, theta: np.ndarray, path: Path) -> tuple[Path, dict[str, Any]]:
    """Whitened source-plane residual per image and the per-image log-likelihood, grouped by family."""
    stats = evaluator.statistics(jnp.asarray(theta))
    residual = np.asarray(stats["residual"])
    chol = np.asarray(stats["cholesky"])
    whitened = np.linalg.norm(np.linalg.solve(chol, residual[..., None])[..., 0], axis=-1)
    loglike = np.asarray(stats["loglike"])
    images = state.images
    family_index = np.asarray(images.family_index_per_image)
    families = list(images.family_ids)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    bins = np.linspace(0, max(3.0, float(np.nanmax(whitened)) * 1.05), 25)
    ax.hist(whitened, bins=bins, color="#3976a8", alpha=0.6)
    ax.axvline(np.sqrt(2.0), color="0.3", ls="--", lw=1.0, label=r"$\sqrt{2}$ (expected rms)")
    ax.set_xlabel(r"whitened source-plane residual $\|C^{-1/2}(\beta_i - \beta_f)\|$")
    ax.set_ylabel("images")
    ax.legend(fontsize=9)
    for axis, values, label in ((axes[1], whitened, "whitened residual"), (axes[2], loglike, "log-likelihood per image")):
        x = np.arange(len(values))
        axis.scatter(x, values, c=[MODE_COLORS(f % 10) for f in family_index], s=16)
        edges = np.flatnonzero(np.diff(family_index)) + 0.5
        for e in edges:
            axis.axvline(e, color="0.85", lw=0.6)
        centers = [np.mean(np.flatnonzero(family_index == f)) for f in range(len(families))]
        axis.set_xticks(centers)
        axis.set_xticklabels(families, rotation=90, fontsize=7)
        axis.set_ylabel(label)
    axes[1].axhline(np.sqrt(2.0), color="0.3", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    summary = {"whitened_residual_rms": float(np.sqrt(np.mean(whitened ** 2))), "whitened_residual_max": float(whitened.max()),
               "worst_image": str(images.image_labels[int(np.argmax(whitened))]), "loglike_min": float(loglike.min()),
               "worst_loglike_image": str(images.image_labels[int(np.argmin(loglike))]), "sigma_def": float(np.median(np.asarray(stats["sigma_def"]))),
               "sigma_extra": float(stats["sigma_extra"])}
    if "sigma_img" in stats:
        summary["sigma_img"] = float(stats["sigma_img"])
    return path, summary


def plot_scaling_at_start(state: ModelState, evaluator: Evaluator, theta: np.ndarray, path: Path) -> tuple[Path, dict[str, Any]]:
    """Member velocity dispersions and cut radii against magnitude at the start, with the fitted relation and the deviations."""
    from scipy.stats import norm
    physical = np.asarray(evaluator.physical(jnp.asarray(theta)))
    spec = state.packed_lens_spec
    specs = state.parameter_specs
    exact = set(state.members.exact_positions.tolist())
    dev_v, dev_r = (np.asarray(a) for a in member_log_deviations(spec, physical))
    rows = []
    for position, record in enumerate(state.scaling_component_records):
        c = int(record["component_index"])
        lum = float(spec.luminosity_ratio[c])
        def value(index, base):                                    # a fixed relation parameter has index -1
            return physical[int(index[c])] if int(index[c]) >= 0 else float(base[c])
        sigma_ref, cut_ref = value(spec.sigma_ref_param_index, spec.sigma_ref_base), value(spec.cut_ref_param_index, spec.cut_ref_base)
        alpha, beta = value(spec.alpha_sigma_param_index, spec.alpha_sigma_base), value(spec.beta_radius_param_index, spec.beta_radius_base)
        scatter = physical[int(spec.velocity_scatter_param_index[c])]
        offset = dev_v[c] / scatter if spec.deviation_mode != "mass" else dev_v[c] * (2 * alpha + beta) / alpha / scatter   # unit offset
        rows.append(dict(mag=float(record["catalog_mag"]), lum=lum, v=sigma_ref * lum ** alpha * np.exp(dev_v[c]),
                         v_relation=sigma_ref * lum ** alpha, cut=cut_ref * lum ** beta * np.exp(dev_r[c]), offset=offset, scatter=scatter,
                         exact=position in exact, member=record["catalog_id"], population=record["potfile_id"]))
    if not rows:
        return path, {}
    mag = np.asarray([r["mag"] for r in rows]); v = np.asarray([r["v"] for r in rows]); vr = np.asarray([r["v_relation"] for r in rows])
    cut = np.asarray([r["cut"] for r in rows]); offset = np.asarray([r["offset"] for r in rows]); is_exact = np.asarray([r["exact"] for r in rows])
    order = np.argsort(mag)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    for mask, marker, label in ((is_exact, "s", "exact"), (~is_exact, "o", "far-field")):
        if mask.any():
            ax.scatter(mag[mask], v[mask], marker=marker, s=16, alpha=0.75, label=label)
    ax.plot(mag[order], vr[order], color="black", lw=1.2, label="scaling relation")
    ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("catalog magnitude"); ax.set_ylabel("velocity dispersion (km/s)")
    ax.legend(fontsize=9)
    ax = axes[1]
    for mask, marker in ((is_exact, "s"), (~is_exact, "o")):
        if mask.any():
            ax.scatter(mag[mask], cut[mask], marker=marker, s=16, alpha=0.75)
    ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("catalog magnitude"); ax.set_ylabel("cut radius (kpc)")
    ax = axes[2]
    bins = np.linspace(min(-4, offset.min()), max(4, offset.max()), 33)
    ax.hist(offset, bins=bins, density=True, color="#3976a8", alpha=0.6, label="unit deviations at the start")
    grid = np.linspace(bins[0], bins[-1], 200)
    ax.plot(grid, norm.pdf(grid), color="black", lw=1.0, label="N(0, 1) prior")
    ax.set_xlabel("velocity offset (units of the scatter)"); ax.set_ylabel("density")
    ax.legend(fontsize=9, loc="upper left")
    lines = []
    for s_ in specs:
        if s_.component_family in {"scaling", "member_velocity_scatter", "error_scale"} or s_.field in {"velocity_log_scatter", "sigma_def"}:
            i = specs.index(s_)
            prior = state.scaling_priors.get(s_.name) if isinstance(state.scaling_priors, dict) else None
            lines.append(f"{s_.name} = {physical[i]:.4g}")
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9})
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    summary = {"offset_rms": float(np.sqrt(np.mean(offset ** 2))), "offset_max_abs": float(np.abs(offset).max()),
               "largest_deviation_member": str(rows[int(np.argmax(np.abs(offset)))]["member"]),
               "scatter": float(rows[0]["scatter"]), "v_disp_range": [float(v.min()), float(v.max())], "cut_kpc_range": [float(cut.min()), float(cut.max())]}
    return path, summary


def write_init_plots(state: ModelState, evaluator: Evaluator, search: dict[str, Any], theta: np.ndarray, figures_dir: Path, *,
                     margin: float, prefix: str | None, z_source: float, log=print) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(search, figure_path(figures_dir, "init_convergence.pdf", prefix), margin=margin)
    if search.get("stage_traces"):
        plot_stage_convergence(search, figure_path(figures_dir, "init_stage_convergence.pdf", prefix))
    plot_modes(state, search, figure_path(figures_dir, "init_modes.pdf", prefix), margin=margin)
    _, fit = plot_fit_at_start(state, evaluator, theta, figure_path(figures_dir, "init_fit.pdf", prefix))
    _, scaling = plot_scaling_at_start(state, evaluator, theta, figure_path(figures_dir, "init_scaling_relation.pdf", prefix))
    fit["scaling"] = scaling
    sources = np.asarray(evaluator.source_positions(jnp.asarray(theta)))
    plot_critical_curves(state, evaluator, theta, sources, z_source if z_source > state.z_lens else max(state.z_source_values),
                         figure_path(figures_dir, "init_critical_curves.pdf", prefix), grid_scale_arcsec=0.5)
    log(f"[init] fit at the start: whitened residual rms {fit['whitened_residual_rms']:.2f} (expected ~1.4), max {fit['whitened_residual_max']:.2f} "
        f"({fit['worst_image']}), " + (f"sigma_img {fit['sigma_img']:.3f}\" (median source-plane sigma_def {fit['sigma_def']:.4f}\": the floor under "
                                        f"the image-plane GP, the mass field's error under the knee kernel)" if "sigma_img" in fit
                                        else f"sigma_def {fit['sigma_def']:.3f}\", sigma_extra {fit['sigma_extra']:.3f}\""))
    if scaling:
        log(f"[init] members at the start: scatter {scaling['scatter']:.3f}, unit deviation rms {scaling['offset_rms']:.2f} (prior 1), "
            f"max |deviation| {scaling['offset_max_abs']:.2f} ({scaling['largest_deviation_member']}), "
            f"sigma {scaling['v_disp_range'][0]:.0f}-{scaling['v_disp_range'][1]:.0f} km/s, r_cut {scaling['cut_kpc_range'][0]:.1f}-{scaling['cut_kpc_range'][1]:.1f} kpc")
    return fit
