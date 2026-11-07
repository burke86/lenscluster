"""Figures for one run: image recovery, residual histogram, corner, scaling relation, critical curves, truth recovery."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from skimage.measure import find_contours

from .plot_style import PAPER_STYLE
from .evaluator import Evaluator
from .model import ModelState, ParameterSpec, PosteriorResults, member_log_deviations
from .truth_recovery import lens_quantities
from .utils import log10_dpie_mass_msun

STYLE = Path(__file__).with_name("style.mplstyle")
if STYLE.exists():
    plt.style.use(str(STYLE))

POINT_COLOR = "#3976a8"
MISSED_COLOR = "#c44e52"


def figure_path(figures_dir: Path, name: str, prefix: str | None) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / (f"{prefix}_{name}" if prefix else name)


def _finish(fig, path: Path, dpi: int = 220) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- image recovery
def posterior_residual_spread(draws: pd.DataFrame | None, best_count: int) -> dict[str, Any]:
    """RMS and median of the recovered residuals over the posterior draws that recovered exactly as many images as the
    best fit: their 16th to 84th percentiles, how many draws were used, and how many recovered more images than the best."""
    out = {"draws_used": 0, "draws_total": 0, "draws_more_than_best": 0, "rms_q16": np.nan, "rms_q84": np.nan, "median_q16": np.nan, "median_q84": np.nan}
    if draws is None or draws.empty or "draw_index" not in draws:
        return out
    residual = pd.to_numeric(draws["image_residual_arcsec"], errors="coerce")
    recovered = draws["recovered"].astype(bool) & np.isfinite(residual)
    per_draw = draws.assign(residual=residual, recovered=recovered).groupby("draw_index").agg(
        count=("recovered", "sum"), rms=("residual", lambda v: float(np.sqrt(np.nanmean(np.square(v[np.isfinite(v)])))) if np.isfinite(v).any() else np.nan),
        median=("residual", lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan))
    out["draws_total"] = int(len(per_draw))
    out["draws_more_than_best"] = int((per_draw["count"] > best_count).sum())
    same = per_draw[per_draw["count"] == best_count]
    out["draws_used"] = int(len(same))
    if len(same) >= 2:
        out["rms_q16"], out["rms_q84"] = (float(v) for v in np.nanpercentile(same["rms"], [16, 84]))
        out["median_q16"], out["median_q84"] = (float(v) for v in np.nanpercentile(same["median"], [16, 84]))
    return out


def posterior_residual_histogram_counts(draws: pd.DataFrame | None, bins: np.ndarray, *, min_recovered: int = 0) -> np.ndarray:
    """Raw counts for draws meeting the minimum recovered-image count."""
    if draws is None or draws.empty or "draw_index" not in draws:
        return np.empty((0, len(bins) - 1), dtype=int)
    counts = []
    for _, group in draws.groupby("draw_index", sort=True):
        values = pd.to_numeric(group["image_residual_arcsec"], errors="coerce").to_numpy(float)
        valid = group["recovered"].astype(bool).to_numpy() & np.isfinite(values) & (values >= 0)
        if valid.sum() < min_recovered:
            continue
        counts.append(np.histogram(values[valid], bins=bins)[0])
    return np.asarray(counts, dtype=int).reshape(-1, len(bins) - 1)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_image_residual_histogram(image_df: pd.DataFrame, path: Path, *, draws: pd.DataFrame | None = None, log=None) -> Path:
    """Best-fit residual histogram; posterior draws are used only for recovery warnings."""
    residual = pd.to_numeric(image_df["image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
    recovered = (image_df["image_recovery_status"].astype(str) == "recovered").to_numpy() & np.isfinite(residual)
    values = residual[recovered]
    spread = posterior_residual_spread(draws, int(values.size))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if values.size:
        bins = np.linspace(0.0, max(float(values.max()) * 1.04, 1.0e-6), min(48, max(14, int(np.sqrt(values.size) * 2.4))) + 1)
        counts, _ = np.histogram(values, bins=bins)
        ax.stairs(counts, bins, color="tab:blue", linewidth=1.5, label="Best fit", zorder=3)
        rms, median = float(np.sqrt(np.mean(values ** 2))), float(np.median(values))
        ax.axvline(rms, color="#ba5a2a", linestyle="--", linewidth=1.3, label="RMS", zorder=4)
        ax.axvline(median, color="#303030", linestyle=":", linewidth=1.3, label="Median", zorder=4)
        ax.set_xlim(0.0, bins[-1])
    text = f"Recovered: {values.size}/{len(image_df)}"
    if spread["draws_more_than_best"]:
        if log is not None:
            log(f"[outputs] warning: {spread['draws_more_than_best']} of {spread['draws_total']} posterior draws recovered more images "
                f"than the best fit ({values.size}); the best fit may not be the best image-plane solution")
    ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if values.size:
        ax.legend(loc="upper right", bbox_to_anchor=(1, .87), fontsize=8, frameon=False)
    ax.set_xlabel("Image-position residual (arcsec)")
    ax.set_ylabel("Number of images")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    return path


def plot_image_recovery(image_df: pd.DataFrame, extra_df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    x_obs, y_obs = image_df["x_obs_arcsec"].to_numpy(float), image_df["y_obs_arcsec"].to_numpy(float)
    x_model = image_df["x_model_q50"].fillna(image_df["x_model_arcsec"]).to_numpy(float)
    y_model = image_df["y_model_q50"].fillna(image_df["y_model_arcsec"]).to_numpy(float)
    recovered = (image_df["image_recovery_status"].astype(str) == "recovered").to_numpy()
    ax.scatter(x_obs[recovered], y_obs[recovered], marker="x", color=POINT_COLOR, s=18, linewidths=0.9, label="recovered")
    if (~recovered).any():
        ax.scatter(x_obs[~recovered], y_obs[~recovered], marker="x", color=MISSED_COLOR, s=18, linewidths=0.9, label="not recovered")
    finite = recovered & np.isfinite(x_model) & np.isfinite(y_model)
    if finite.any():
        xerr = [np.maximum(0, x_model - image_df["x_model_q16"].to_numpy(float))[finite], np.maximum(0, image_df["x_model_q84"].to_numpy(float) - x_model)[finite]]
        yerr = [np.maximum(0, y_model - image_df["y_model_q16"].to_numpy(float))[finite], np.maximum(0, image_df["y_model_q84"].to_numpy(float) - y_model)[finite]]
        ax.errorbar(x_model[finite], y_model[finite], xerr=np.nan_to_num(xerr), yerr=np.nan_to_num(yerr), fmt="o", color=POINT_COLOR,
                    alpha=0.75, markersize=3, elinewidth=0.8, capsize=1.5, label="model")
        for xo, yo, xm, ym in zip(x_obs[finite], y_obs[finite], x_model[finite], y_model[finite]):
            ax.plot([xo, xm], [yo, ym], color="0.6", lw=0.7)
    if extra_df is not None and not extra_df.empty:
        ax.scatter(extra_df["x_model_arcsec"], extra_df["y_model_arcsec"], marker="o", color="tab:blue", s=16, linewidths=0.0, label="extra")
    ax.invert_xaxis()
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.legend(loc="best", fontsize=8)
    residual = image_df["image_residual_q50"].fillna(image_df["image_residual_arcsec"]).to_numpy(float)
    index = np.arange(len(image_df))
    ok = np.isfinite(residual)
    if ok.any():
        yerr = [np.maximum(0, residual - image_df["image_residual_q16"].to_numpy(float))[ok], np.maximum(0, image_df["image_residual_q84"].to_numpy(float) - residual)[ok]]
        axes[1].errorbar(index[ok], residual[ok], yerr=np.nan_to_num(yerr), fmt="o", color="tab:blue", markersize=3)
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual (arcsec)")
    if len(image_df) <= 40:
        axes[1].set_xticks(index)
        axes[1].set_xticklabels(image_df["image_label"].astype(str), rotation=90, fontsize=7)
    fig.tight_layout()
    return _finish(fig, path, dpi=180)


# ----------------------------------------------------------------------------- posterior
def plot_corner(samples: np.ndarray, specs: list[ParameterSpec], path: Path, *, best: np.ndarray | None = None,
                grouped_physical=None, best_index=None) -> Path | None:
    from .corner_plots import plot_parameter_corner
    keep = [(i, spec, spec.name) for i, spec in enumerate(specs)
            if spec.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"}]
    if best_index is None and best is not None:
        matches = np.flatnonzero(np.all(np.asarray(samples) == best, axis=1))
        best_index = int(matches[0]) if len(matches) else None
    return plot_parameter_corner(samples, keep, path, grouped_physical=grouped_physical, best_index=best_index)


def member_scaling_table(state: ModelState, posterior: PosteriorResults) -> pd.DataFrame:
    """Posterior summary of every member's velocity dispersion, radii and mass, with its class."""
    physical = np.asarray(posterior.physical, dtype=float)
    spec_packed = state.packed_lens_spec
    exact = set(state.members.exact_positions.tolist())
    import jax
    dev_v, dev_r = (np.asarray(a) for a in jax.vmap(lambda row: member_log_deviations(spec_packed, row))(physical))   # (draws, components)
    rows = []
    for position, record in enumerate(state.scaling_component_records):
        c = int(record["component_index"])
        pop = record["potfile_id"]
        def value(index, base):                                    # a fixed relation parameter has index -1
            return physical[:, int(index[c])] if int(index[c]) >= 0 else np.full(len(physical), float(base[c]))
        sigma_ref = value(spec_packed.sigma_ref_param_index, spec_packed.sigma_ref_base)
        cut_ref = value(spec_packed.cut_ref_param_index, spec_packed.cut_ref_base)
        core_ref = value(spec_packed.core_ref_param_index, spec_packed.core_ref_base)
        alpha = value(spec_packed.alpha_sigma_param_index, spec_packed.alpha_sigma_base)
        beta = value(spec_packed.beta_radius_param_index, spec_packed.beta_radius_base)
        lum = float(spec_packed.luminosity_ratio[c])
        v_disp = sigma_ref * lum ** alpha * np.exp(dev_v[:, c])
        v_scaling = sigma_ref * lum ** alpha
        core, cut = core_ref * lum ** beta * np.exp(dev_r[:, c]), cut_ref * lum ** beta * np.exp(dev_r[:, c])
        mass = log10_dpie_mass_msun(v_disp, cut)
        row = {"potfile_id": pop, "catalog_id": record["catalog_id"], "component_index": c, "catalog_mag": record["catalog_mag"],
               "luminosity_ratio": lum, "member_class": "exact" if position in exact else "far-field",
               "centered": bool(state.members.centered[position]) if len(state.members.centered) else False,
               "x_centre": record["x_centre"], "y_centre": record["y_centre"]}
        for name, values in (("v_disp", v_disp), ("v_disp_scaling", v_scaling), ("core_radius_kpc", core), ("cut_radius_kpc", cut),
                             ("log10_mass_msun", mass), ("velocity_offset", dev_v[:, c]), ("radius_offset", dev_r[:, c])):
            q16, q50, q84 = np.percentile(values, [16, 50, 84])
            row.update({f"{name}_p16": q16, f"{name}_median": q50, f"{name}_p84": q84})
        rows.append(row)
    return pd.DataFrame(rows)


def population_scaling_draws(table: pd.DataFrame, posterior_physical: np.ndarray,
                             specs: list[ParameterSpec]) -> dict:
    """Joint population relations, without individual velocity/radius deviations."""
    names = {spec.name: i for i, spec in enumerate(specs)}
    result = {}
    for population, members in table.groupby("potfile_id", sort=True):
        # Recover the saved pivot, without depending on a possibly changed config.
        pivots = members["catalog_mag"].to_numpy() + 2.5 * np.log10(members["luminosity_ratio"].to_numpy())
        if not np.isfinite(pivots).all() or not np.allclose(pivots, pivots[0], rtol=0, atol=1e-8):
            raise ValueError(f"Inconsistent luminosity pivot for {population}")
        magnitude = np.linspace(members.catalog_mag.min(), members.catalog_mag.max(), 240)
        lum = 10. ** (-.4 * (magnitude - pivots[0]))
        def draws(field):
            return posterior_physical[:, names[f"{population}.{field}"]][:, None]
        velocity = draws("sigma") * lum ** draws("alpha_sigma")
        radius = draws("cutkpc") * lum ** draws("beta_radius")
        result[population] = (magnitude, {"v_disp": velocity, "cut_radius_kpc": radius,
                                        "log10_mass_msun": log10_dpie_mass_msun(velocity, radius)})
    return result


@plt.rc_context({"font.family": "sans-serif", "font.size": 11, "axes.labelsize": 12,
                 "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
                 "axes.linewidth": .9, "xtick.labelsize": 10, "ytick.labelsize": 10,
                 "xtick.major.width": .8, "ytick.major.width": .8,
                 "xtick.minor.width": .6, "ytick.minor.width": .6,
                 "xtick.major.size": 4, "ytick.major.size": 4,
                 "xtick.top": False, "ytick.right": False, "axes.grid": False,
                 "pdf.fonttype": 42})
def plot_scaling_relation(table: pd.DataFrame, posterior_physical: np.ndarray, specs: list[ParameterSpec], path: Path) -> Path:
    from matplotlib.patches import Patch
    fields = [("v_disp", r"$\sigma$ (km s$^{-1}$)", True),
              ("cut_radius_kpc", r"$r_{\rm cut}$ (kpc)", True),
              ("log10_mass_msun", r"$\log_{10}(M/M_\odot)$", False)]
    relations = population_scaling_draws(table, np.asarray(posterior_physical), specs)
    names = {spec.name: i for i, spec in enumerate(specs)}
    pivots = {}
    for population, members in table.groupby("potfile_id"):
        member = members.iloc[0]
        pivot = float(member.catalog_mag + 2.5 * np.log10(member.luminosity_ratio))
        velocity = posterior_physical[:, names[f"{population}.sigma"]]
        radius = posterior_physical[:, names[f"{population}.cutkpc"]]
        pivots[population] = (pivot, {"v_disp": np.median(velocity), "cut_radius_kpc": np.median(radius),
                                     "log10_mass_msun": np.median(log10_dpie_mass_msun(velocity, radius))})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True)
    handles = []
    relation_colors = ["black", "#77529B", "#237A66"]
    for ax, (field, label, logscale) in zip(axes, fields):
        for idx, (population, (mag, curves)) in enumerate(relations.items()):
            color = relation_colors[idx % len(relation_colors)]
            lo, med, hi = np.quantile(curves[field], [.16, .5, .84], axis=0)
            ax.fill_between(mag, lo, hi, color=color, alpha=.06, linewidth=0, zorder=1)
            line, = ax.plot(mag, med, color=color, lw=1.6, zorder=4,
                            label="Population relation" if len(relations) == 1 else str(population))
            if ax is axes[0]:
                handles.append(line)
            pivot, pivot_values = pivots[population]
            relative_luminosity = 10. ** (-.4 * (mag - pivot))
            if field == "v_disp":
                reference = pivot_values[field] * relative_luminosity ** .25
                reference_label = "Faber–Jackson"
            elif field == "cut_radius_kpc":
                reference = pivot_values[field] * relative_luminosity ** .5
                reference_label = "Faber–Jackson + constant M/L"
            else:
                reference = pivot_values[field] + np.log10(relative_luminosity)
                reference_label = "Constant M/L"
            reference_line, = ax.plot(mag, reference, color="tab:red", lw=1.3,
                                     ls=(0, (5, 3)), zorder=5, label=reference_label)
            if idx == 0:
                ax.legend(handles=[reference_line], loc="upper left", fontsize=8,
                          frameon=False, handlelength=2.3, handletextpad=.6)
            pivot_marker, = ax.plot(pivot, pivot_values[field], marker="o", ms=5.5, mfc="white",
                                   mec=color, mew=1.3, linestyle="none", zorder=10, label=r"$m_\star$")
            if ax is axes[0] and idx == 0:
                handles.append(pivot_marker)
        for kind, marker, color in (("far-field", "o", "tab:blue"), ("exact", "D", "tab:orange")):
            sub = table[table.member_class == kind]
            if sub.empty:
                continue
            yerr = np.array([sub[f"{field}_median"] - sub[f"{field}_p16"],
                             sub[f"{field}_p84"] - sub[f"{field}_median"]])
            container = ax.errorbar(sub.catalog_mag, sub[f"{field}_median"], yerr=yerr,
                fmt=marker, color=color, ecolor=matplotlib.colors.to_rgba(color, .23),
                markersize=3.1, markeredgewidth=0, elinewidth=.55, capsize=0,
                zorder=3 if kind == "exact" else 2,
                label=f"{kind.capitalize()} members")
            # Raise only the markers; retain the lighter error bars underneath the curves.
            container.lines[0].set_zorder(9 if kind == "exact" else 8)
            if ax is axes[0]:
                handles.append(container)
        if logscale:
            ax.set_yscale("log")
        ax.set_xlabel("Catalog magnitude")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.margins(y=.10)
    lo, hi = table.catalog_mag.min(), table.catalog_mag.max()
    axes[0].set_xlim(hi + .03 * (hi-lo), lo - .03 * (hi-lo))
    handles.append(Patch(facecolor=relation_colors[0], alpha=.06, edgecolor="none", label="16–84% relation interval"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=5,
               fontsize=9, frameon=False, handletextpad=.6, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.)
    return _finish(fig, path, dpi=250)


# ----------------------------------------------------------------------------- critical curves
def plot_critical_curves(state: ModelState, evaluator: Evaluator, theta: np.ndarray, sources: np.ndarray, z_source: float,
                         path: Path, *, half_fov_arcsec: float = 100.0, grid_scale_arcsec: float = 0.2) -> Path:
    n = max(2, int(round(2 * half_fov_arcsec / grid_scale_arcsec)) + 1)
    axis = np.linspace(-half_fov_arcsec, half_fov_arcsec, n)
    xx, yy = np.meshgrid(axis, axis)
    quantities = lens_quantities(evaluator, theta, z_source, xx.ravel(), yy.ravel())
    kappa = quantities["kappa"].reshape(xx.shape)
    shear = np.hypot(quantities["gamma1"], quantities["gamma2"]).reshape(xx.shape)
    fig, (image_ax, source_ax) = plt.subplots(1, 2, figsize=(12, 5))
    pixel = np.arange(n, dtype=float)
    physical = evaluator.physical(jnp.asarray(theta))
    packed = evaluator.packed_state_for_z(physical, z_source)
    for field, color in ((1.0 - kappa - shear, "black"), (1.0 - kappa + shear, "0.5")):
        for vertices in find_contours(field, 0.0):
            cx, cy = np.interp(vertices[:, 1], pixel, axis), np.interp(vertices[:, 0], pixel, axis)
            image_ax.plot(cx, cy, color=color, linewidth=1.0)
            xy = np.stack((cx, cy), -1)
            beta = xy - evaluator.total_fields_at(xy, packed)[0]
            source_ax.scatter(beta[:, 0], beta[:, 1], color=color, s=2, alpha=0.75, linewidths=0.0)
    cmap = plt.get_cmap("tab20", max(1, len(state.family_data)))
    for idx, family in enumerate(state.family_data):
        image_ax.scatter(family.x_obs, family.y_obs, color=cmap(idx), s=14, label=family.family_id)
    for idx, family_id in enumerate(state.images.family_ids):
        color_index = [f.family_id for f in state.family_data].index(family_id)
        source_ax.scatter(sources[idx, 0], sources[idx, 1], color=cmap(color_index), s=14)
    image_ax.invert_xaxis()
    image_ax.set_xlabel("x (arcsec)")
    image_ax.set_ylabel("y (arcsec)")
    source_ax.set_xlabel(r"$\beta_x$ (arcsec)")
    source_ax.set_ylabel(r"$\beta_y$ (arcsec)")
    fig.tight_layout()
    return _finish(fig, path, dpi=180)


# ----------------------------------------------------------------------------- truth recovery
def _shifted_diverging_colormap(name: str, zero_position: float, samples: int = 256) -> LinearSegmentedColormap:
    base = plt.get_cmap(name)
    positions = np.linspace(0.0, 1.0, samples)
    shifted = np.where(positions < zero_position, 0.5 * positions / zero_position, 0.5 + 0.5 * (positions - zero_position) / (1.0 - zero_position))
    return LinearSegmentedColormap.from_list(f"{name}_shifted", list(zip(positions, base(shifted))))


def plot_truth_recovery_residuals(recovery: dict[str, Any], image_df: pd.DataFrame | None, path: Path) -> Path | None:
    truth, model = recovery["truth"], recovery["model"]
    if "abs_mu" not in truth:
        return None
    x, y = recovery["x"], recovery["y"]

    def fractional(m, t):
        valid = np.isfinite(m) & np.isfinite(t) & (t > 0)
        out = np.full(t.shape, np.nan)
        np.divide(m - t, t, out=out, where=valid)
        return out

    residuals = (fractional(model["kappa"]["median"], truth["kappa"]), fractional(model["abs_mu"]["median"], truth["abs_mu"]))
    norm = Normalize(vmin=-1.0, vmax=2.0, clip=True)
    cmap = _shifted_diverging_colormap("RdBu_r", 1.0 / 3.0)
    cmap.set_bad("#e8e8e8")
    style = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.79, wspace=0.07)
        meshes = [ax.pcolormesh(x, y, np.ma.masked_invalid(r), cmap=cmap, norm=norm, shading="nearest", rasterized=True) for ax, r in zip(axes[:2], residuals)]
        handles = []
        for det, color, width, label in ((truth["detA"], "#252525", 0.85, "Truth"), (model["detA"]["median"], "#d55e00", 0.7, "Model")):
            finite = det[np.isfinite(det)]
            if finite.size and finite.min() <= 0.0 <= finite.max():
                axes[2].contour(x, y, np.ma.masked_invalid(det), levels=[0.0], colors=[color], linewidths=[width])
                handles.append(Line2D([], [], color=color, linewidth=width, label=label))
        if image_df is not None and not image_df.empty:
            positions = image_df[["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float)
            for ax in axes:
                ax.scatter(*positions.T, s=4, facecolors="none", edgecolors="white", linewidths=0.9, zorder=5)
                ax.scatter(*positions.T, s=4, facecolors="none", edgecolors="#333333", linewidths=0.35, zorder=6)
            handles.append(Line2D([], [], marker="o", markersize=3, markerfacecolor="none", markeredgecolor="#333333", linestyle="none", label="Images"))
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (arcsec)")
            ax.tick_params(which="both", direction="out", top=False, right=False)
        axes[0].set_ylabel("y (arcsec)")
        if handles:
            axes[2].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, handlelength=1.3, columnspacing=0.7)
        fig.canvas.draw()
        for ax, mesh, label in zip(axes[:2], meshes, (r"$(\kappa_{\rm model}-\kappa_{\rm true})/\kappa_{\rm true}$", r"$(|\mu_{\rm model}|-|\mu_{\rm true}|)/|\mu_{\rm true}|$")):
            box = ax.get_position()
            cax = fig.add_axes([box.x0, box.y1 + 0.045, box.width, 0.025])
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-1, 0, 1, 2], extend="both")
            cb.set_label(label, fontsize=8, labelpad=5)
            cb.ax.xaxis.set_label_position("top")
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.minorticks_off()
            cb.ax.tick_params(direction="out", length=2.5, width=0.6, labelsize=8, pad=2)
        return _finish(fig, path, dpi=600)


def plot_m2d_aperture_ratio(profile: pd.DataFrame, center: dict[str, Any], image_df: pd.DataFrame | None, mass_scale: float, path: Path) -> Path:
    fig, (ax, ax_resid) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    radius = profile["radius_arcsec"].to_numpy(float)
    truth = profile["kappa_true_sum"].to_numpy(float) * mass_scale
    model = profile["kappa_model_sum"].to_numpy(float) * mass_scale
    q16, q84 = profile["kappa_model_sum_q16"].to_numpy(float) * mass_scale, profile["kappa_model_sum_q84"].to_numpy(float) * mass_scale
    ratio = profile["m2d_ratio_median"].to_numpy(float)
    band = np.isfinite(q16) & np.isfinite(q84) & (q16 > 0)
    if band.any():
        ax.fill_between(radius[band], q16[band], q84[band], color="#2878A5", alpha=0.22, linewidth=0.0)
    ax.plot(radius, model, color="#2878A5", linewidth=1.8, label="Model")
    ax.plot(radius, truth, color="0.2", linestyle="--", linewidth=1.3, label="Truth")
    if image_df is not None and not image_df.empty and center is not None:
        radii = np.hypot(image_df["x_obs_arcsec"].to_numpy(float) - center["center_x_arcsec"], image_df["y_obs_arcsec"].to_numpy(float) - center["center_y_arcsec"])
        ax.vlines(radii[radii > 0], 0.0, 0.10, transform=ax.get_xaxis_transform(), color="0.20", linewidth=0.85, alpha=0.9, zorder=5)
    r16, r84 = profile["m2d_ratio_q16"].to_numpy(float), profile["m2d_ratio_q84"].to_numpy(float)
    rband = np.isfinite(r16) & np.isfinite(r84)
    if rband.any():
        ax_resid.fill_between(radius[rband], r16[rband], r84[rband], color="#2878A5", alpha=0.22, linewidth=0.0)
    ax_resid.plot(radius, ratio, color="#2878A5", linewidth=1.3)
    ax_resid.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
    ax_resid.set_ylim(0.9, 1.1)
    ax_resid.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax_resid.set_xlim(left=5.0)
    ax_resid.set_xlabel(r"$R$ (arcsec)", fontsize=10)
    ax_resid.set_ylabel(r"$M_{\rm model}/M_{\rm true}$", fontsize=10)
    ax.set_ylabel(r"$M_{\rm 2D}(<R)$ ($M_\odot$)", fontsize=10)
    shown = np.isfinite(truth) & np.isfinite(model) & (radius >= 4.9) & (truth > 0) & (model > 0)
    if shown.any():
        values = np.concatenate([truth[shown], model[shown]])
        ax.set_ylim(values.min() / 1.35, values.max() * 1.35)
        ax.set_xlim(5.0, radius[shown].max() * 1.03)
    for panel in (ax, ax_resid):
        panel.tick_params(which="major", labelsize=9, length=4, width=0.8, direction="in", top=True, right=True)
        panel.tick_params(which="minor", length=2, width=0.6, direction="in", top=True, right=True)
    ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=2.3)
    return _finish(fig, path, dpi=220)
