"""Aggregate figures of a mock-validation campaign, from each seed's recovery.json."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from matplotlib.ticker import MaxNLocator

from ..plot_style import PAPER_STYLE
from .aperture import COMPONENT_GROUPS
from .runner import CONVERGENCE_MIN_NEFF, CONVERGENCE_RHAT, convergence_verdict, RECOVERY_FILE

STYLE = Path(__file__).resolve().parent.parent / "style.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))

COLOR = "tab:blue"
TRUTH_COLOR = "0.2"
COMPONENT_COLORS = {"halo": "tab:blue", "bcg": "tab:orange", "members": "tab:green"}
COMPONENT_LABELS = {"halo": "Main halo", "bcg": "BCG", "members": "Members"}
PAPER_PARAMETERS = (
    "halo.log10_mass_msun", "halo.x_centre", "halo.y_centre", "halo.v_disp", "halo.core_radius_kpc", "halo.cut_radius_kpc",
    "bcg.log10_mass_msun", "bcg.x_centre", "bcg.y_centre", "bcg.v_disp", "bcg.core_radius_kpc", "bcg.cut_radius_kpc",
    "potfile.log10_mass_msun", "potfile.sigma", "potfile.cutkpc", "potfile.alpha_sigma", "potfile.beta_radius", "potfile.gamma_ml",
)
PARAMETER_LABELS = {
    "log10_mass_msun": r"$\log_{10} M$", "x_centre": r"$x$", "y_centre": r"$y$", "v_disp": r"$\sigma$", "core_radius_kpc": r"$r_{\rm core}$",
    "cut_radius_kpc": r"$r_{\rm cut}$", "sigma": r"$\sigma^\star$", "cutkpc": r"$r_{\rm cut}^\star$", "alpha_sigma": r"$\alpha$",
    "beta_radius": r"$\beta$", "gamma_ml": r"$\gamma_{M/L}$",
}
PARAMETER_UNITS = {"x_centre": "arcsec", "y_centre": "arcsec", "v_disp": "km/s", "core_radius_kpc": "kpc", "cut_radius_kpc": "kpc",
                   "sigma": "km/s", "cutkpc": "kpc", "log10_mass_msun": r"$M_\odot$"}
COMPONENT_NAMES = {"halo": "Halo", "bcg": "BCG", "potfile": "Members"}



def _polish(ax):
    ax.grid(False)


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    return path


@dataclass
class Campaign:
    root: Path
    seeds: list[int]
    parameters: pd.DataFrame
    magnification: pd.DataFrame
    residual_draws: pd.DataFrame
    subhalos: list[dict[str, Any]]
    m2d: pd.DataFrame
    summary: pd.DataFrame
    relation: list[dict[str, Any]]


def load_campaign(root: str | Path, *, converged_only: bool = True) -> Campaign:
    """The campaign's recovery tables.  With ``converged_only`` (the default) realizations whose sampler did not converge
    (max R-hat >= 1.01 or min n_eff < 50, recomputed from saved diagnostics) are left out of every table except the
    summary, which keeps all of them with their ``converged`` flag and reasons."""
    root = Path(root)
    files = sorted(root.glob("seed_*/recovery/" + RECOVERY_FILE))
    if not files:
        raise FileNotFoundError(f"No seed_*/recovery/{RECOVERY_FILE} under {root}")
    seeds, parameters, magnification, draws, subhalos, m2d, summary, relation = [], [], [], [], [], [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        seed = int(payload["seed"])
        converged, reasons = convergence_verdict({"max_r_hat": payload.get("max_r_hat"), "min_n_eff": payload.get("min_n_eff")})
        summary.append({"seed": seed, "mode": payload.get("mode"), "converged": converged, "convergence_reasons": "; ".join(reasons),
                        "max_r_hat": payload.get("max_r_hat"), "min_n_eff": payload.get("min_n_eff"),
                        "divergences": payload.get("divergences"), "audit": payload.get("audit"), "coverage_68": payload.get("coverage_68"),
                        "coverage_95": payload.get("coverage_95"), "scored_parameters": payload.get("scored_parameters"), **payload.get("fit", {})})
        if converged_only and not converged:
            continue
        seeds.append(seed)
        for key, store in (("parameters", parameters), ("magnification", magnification), ("image_residual_draws", draws), ("m2d", m2d)):
            store.append(pd.DataFrame(payload[key]).assign(seed=seed))
        subhalos.append({"seed": seed, **payload["subhalos"]})
        if payload.get("relation"):
            relation.append({"seed": seed, **payload["relation"]})
    if not seeds:
        raise ValueError(f"No converged realization under {root}")
    return Campaign(root, seeds, pd.concat(parameters, ignore_index=True), pd.concat(magnification, ignore_index=True),
                    pd.concat(draws, ignore_index=True), subhalos, pd.concat(m2d, ignore_index=True), pd.DataFrame(summary), relation)


def representative_seed(campaign: Campaign) -> int:
    """The seed with the smallest maximum R-hat (the first seed when R-hat is unavailable)."""
    summary = campaign.summary[campaign.summary["seed"].isin(campaign.seeds)].copy()
    summary["max_r_hat"] = pd.to_numeric(summary["max_r_hat"], errors="coerce")
    ordered = summary.sort_values(["max_r_hat", "seed"], na_position="last")
    return int(ordered["seed"].iloc[0])


# ----------------------------------------------------------------------------- parameter recovery
def _axis_label(parameter: str, role: str) -> str:
    component, field = parameter.split(".", 1)
    label = PARAMETER_LABELS.get(field, field)
    if field == "log10_mass_msun":
        label = r"$\log_{10}(M/M_\odot)$"
    unit = PARAMETER_UNITS.get(field)
    unit = None if field == "log10_mass_msun" else unit
    prefix = "True" if role == "truth" else "Fit"
    return f"{prefix} {label} ({unit})" if unit else f"{prefix} {label}"


def _draw_parameter_axis(ax, panel: pd.DataFrame, parameter: str) -> None:
    truth, median = panel["truth"].to_numpy(float), panel["median"].to_numpy(float)
    q16, q84 = panel["q16"].to_numpy(float), panel["q84"].to_numpy(float)
    ax.errorbar(truth, median, yerr=[np.maximum(0, median - q16), np.maximum(0, q84 - median)], fmt="o", color=COLOR, markersize=2.7,
                markeredgewidth=0, ecolor=matplotlib.colors.to_rgba(COLOR, .2), elinewidth=.55, capsize=0, alpha=.5, zorder=2)
    binned = parameter_recovery_bins(panel)
    if not binned.empty:
        ax.errorbar(binned.truth_mean, binned.recovered_mean, yerr=binned.recovered_sem,
                    fmt="o", color="tab:orange", markersize=3.4, elinewidth=.8,
                    capsize=1.5, zorder=4)
    values = np.concatenate([truth, median, q16, q84])
    values = values[np.isfinite(values)]
    if values.size:
        pad = 0.05 * (values.max() - values.min() or 1.0)
        limits = (values.min() - pad, values.max() + pad)
        ax.plot(limits, limits, color="0.25", linewidth=.7, linestyle="--", alpha=.7, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    ax.set_xlabel(_axis_label(parameter, "truth"), fontsize=8.5, labelpad=2)
    ax.set_ylabel(_axis_label(parameter, "fit"), fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.text(.04, .96, COMPONENT_NAMES.get(parameter.split('.')[0], 'Sources'), transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=".25")
    _polish(ax)


def parameter_recovery_bins(panel: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Mean recovery per truth bin, with complete mocks as the independent units."""
    data = panel[["seed", "truth", "median"]].copy()
    data = data[np.isfinite(data.truth) & np.isfinite(data["median"])]
    if data.empty:
        return pd.DataFrame()
    lo, hi = data.truth.min(), data.truth.max()
    if lo == hi:
        data["bin"] = 0
    else:
        edges = np.linspace(lo, hi, bins + 1)
        data["bin"] = np.clip(np.searchsorted(edges, data.truth, side="right") - 1, 0, bins - 1)
    rows = []
    for _, images in data.groupby("bin"):
        mocks = images.groupby("seed")[["truth", "median"]].mean()
        if len(mocks) < 2:
            continue
        rows.append({"truth_mean": mocks.truth.mean(), "recovered_mean": mocks["median"].mean(),
                     "recovered_sem": mocks["median"].std(ddof=1) / np.sqrt(len(mocks))})
    return pd.DataFrame(rows)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_parameter_recovery(campaign: Campaign, path: Path, *, ncols: int = 4) -> Path | None:
    df = campaign.parameters
    df = df[np.isfinite(df["truth"].to_numpy(float)) & np.isfinite(df["median"].to_numpy(float))]
    parameters = [p for p in PAPER_PARAMETERS if (df["parameter"] == p).any()]
    names = df["parameter"].astype(str)
    source = df[names.str.startswith("source.") & (names.str.endswith(".beta_x") | names.str.endswith(".beta_y"))]
    coordinates = [c for c in ("beta_x", "beta_y") if source["parameter"].str.endswith(c).any()]
    if not parameters and not coordinates:
        return None
    n_panels = len(parameters) + len(coordinates)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.1, 1.65 * nrows), squeeze=False)
    flat = list(axes.ravel())
    for ax, parameter in zip(flat, parameters):
        _draw_parameter_axis(ax, df[df["parameter"] == parameter], parameter)
    for ax, coordinate in zip(flat[len(parameters):], coordinates):
        rows = source[source["parameter"].str.endswith(coordinate)]
        _draw_parameter_axis(ax, rows, f"source.{coordinate}")
        symbol = r"$\beta_x$" if coordinate == "beta_x" else r"$\beta_y$"
        ax.set_xlabel(f"True {symbol} (arcsec)", fontsize=8.5, labelpad=2)
        ax.set_ylabel(f"Fit {symbol} (arcsec)", fontsize=8.5, labelpad=2)
    for ax in flat[n_panels:]:
        ax.axis("off")
    handles = flat[0].containers[:2]
    fig.legend(handles, ["Mocks", "Binned mocks"][:len(handles)], loc="upper center",
               ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(pad=.4, w_pad=.6, h_pad=1.1, rect=(0, 0, 1, .97))
    return _save(fig, path)


# ----------------------------------------------------------------------------- subhalo mass function
def _schechter_curve(selection: dict[str, Any], x: np.ndarray, normalization: float,
                     luminosity_range: tuple[float, float] | None = None) -> np.ndarray | None:
    """Input luminosity distribution mapped to dN/dlog10(M), without scatter."""
    try:
        alpha, anchor, exponent = (float(selection[k]) for k in ("schechter_alpha", "dpie_mass_anchor_msun", "mass_luminosity_exponent"))
        lower, upper = luminosity_range
        magnitude_lower = 10.0 ** (-0.4 * (float(selection["faint_magnitude_limit"]) - float(selection["pivot_magnitude"])))
        lower = max(float(lower), magnitude_lower)
        upper = float(upper)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (np.all(np.isfinite([alpha, anchor, exponent, normalization, lower, upper]))
            and anchor > 0 and exponent > 0 and normalization > 0 and 0 < lower < upper):
        return None
    luminosity = np.power(np.power(10.0, x) / anchor, 1.0 / exponent)
    # p(L) is normalized over the generator's selection, not the plot range.
    norm = quad(lambda ell: ell ** alpha * np.exp(-ell), lower, upper,
                epsabs=1e-12, epsrel=1e-12)[0]
    density = np.power(luminosity, alpha + 1.0) * np.exp(-luminosity)
    density *= normalization * np.log(10.0) / (exponent * norm)
    return np.where((luminosity >= lower) & (luminosity <= upper), density, 0.0)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_subhalo_mass_function(campaign: Campaign, path: Path) -> Path | None:
    truth_all = [np.asarray(s["truth_log10_mass"], float) for s in campaign.subhalos]
    recovered_all = [np.asarray(s["recovered_log10_mass"], float) for s in campaign.subhalos]
    values = np.concatenate([v for v in truth_all + recovered_all if v.size]) if any(v.size for v in truth_all + recovered_all) else np.asarray([])
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        lo, hi = lo - 0.25, hi + 0.25
    edges = np.linspace(lo, hi, int(np.clip(np.sqrt(values.size), 6, 18)) + 1)
    widths, centers = np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
    truth_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in truth_all])
    recovered_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in recovered_all])
    truth_mean, recovered_mean = truth_density.mean(axis=0), recovered_density.mean(axis=0)
    n_mocks = len(campaign.subhalos)
    # One independent unit is a complete mock, for both truth and recovery.
    truth_sem = truth_density.std(axis=0, ddof=1) / np.sqrt(n_mocks) if n_mocks > 1 else None
    recovered_sem = recovered_density.std(axis=0, ddof=1) / np.sqrt(n_mocks) if n_mocks > 1 else None
    fig, top = plt.subplots(figsize=(3.5, 2.8))
    truth_line = top.stairs(truth_mean, edges, color="0.25", linestyle="-",
                            linewidth=1.0, label="Truth", zorder=3)
    if truth_sem is not None:
        top.stairs(truth_mean + truth_sem, edges, baseline=truth_mean - truth_sem,
                   fill=True, color="0.25", alpha=.13, linewidth=0, zorder=1)
    recovered_bars = top.errorbar(centers, recovered_mean, yerr=recovered_sem,
                 fmt="o", color=COLOR, markersize=2.8, elinewidth=.7, capsize=1.5,
                 markeredgewidth=.6, label="Recovered", zorder=5)
    handles = [recovered_bars, truth_line]
    config_path = campaign.root / "campaign_config.json"
    luminosity_range = (json.loads(config_path.read_text()).get("mock", {}).get("luminosity_range")
                        if config_path.exists() else None)
    reference_x = np.linspace(edges[0], edges[-1], 2000)
    curves = [_schechter_curve(subhalo.get("selection", {}), reference_x, float(len(masses)), luminosity_range)
              for subhalo, masses in zip(campaign.subhalos, truth_all)]
    if all(curve is not None for curve in curves):
        reference, = top.plot(reference_x, np.mean(curves, axis=0), color="tab:orange", linestyle=":", linewidth=1.4,
                             label="Schechter")
        handles.insert(0, reference)
    truth_high = truth_mean + (truth_sem if truth_sem is not None else 0.0)
    recovered_high = recovered_mean + (recovered_sem if recovered_sem is not None else 0.0)
    ceiling = max(1.0, float(np.nanmax(np.concatenate([truth_high, recovered_high]))) * 1.18)
    top.set_ylim(-0.02 * ceiling, ceiling)
    top.legend(handles=handles, loc="upper left", frameon=False)
    top.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    top.set_ylabel(r"$\langle dN/d\log_{10} M\rangle$")
    _polish(top)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- magnification
@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_magnification_recovery(campaign: Campaign, path: Path) -> Path | None:
    df = campaign.magnification
    truth = np.abs(df["magnification_true_at_obs"].to_numpy(float))
    q16, q50, q84 = (np.abs(df[f"magnification_model_{q}"].to_numpy(float)) for q in ("q16", "q50", "q84"))
    lower, upper = np.minimum(np.minimum(q16, q50), q84), np.maximum(np.maximum(q16, q50), q84)
    valid = (truth > 0) & (q50 > 0) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return None
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    ax.errorbar(truth[valid], q50[valid], yerr=[np.maximum(0, q50 - lower)[valid], np.maximum(0, upper - q50)[valid]], fmt="o", color=COLOR,
                markersize=1.8, markeredgewidth=0, ecolor=matplotlib.colors.to_rgba(COLOR, .08),
                elinewidth=.4, capsize=0, alpha=.35, zorder=2, rasterized=True)
    ax.plot([], [], "o", color=COLOR, markersize=3, linestyle="none", label="Images")
    stats = binned_magnification_summary(df)
    if not stats.empty:
        stats.to_csv(path.with_name(path.stem + "_bins.csv"), index=False)
        shown = stats[(stats.n_images > 7) & (stats.n_mocks >= 2)]
        if not shown.empty:
            ax.errorbar(shown.truth_mean, shown.recovered_mean, yerr=shown.recovered_sem,
                        fmt="o", color="tab:orange", markersize=4, elinewidth=1,
                        capsize=2, label="Binned Images", zorder=5)
            ax.legend(loc="upper left", frameon=False, fontsize=8)
    log_values = np.log10(np.concatenate([truth[valid], lower[valid][lower[valid] > 0], upper[valid]]))
    pad = max(0.08 * np.ptp(log_values), 0.08)
    limits = (10 ** (log_values.min() - pad), 10 ** (log_values.max() + pad))
    ax.plot(limits, limits, color="black", linewidth=.85, linestyle="--", alpha=.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"True $|\mu|$")
    ax.set_ylabel(r"Recovered $|\mu|$")
    _polish(ax)
    fig.tight_layout()
    return _save(fig, path)


def binned_magnification_summary(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Equal-mock means and SEMs of saved median |mu|, binned by true |mu|.

    Summarize images within each mock first. Only occupied mocks contribute to
    a bin; these SEMs describe ensemble means, not individual posterior errors.
    """
    data = pd.DataFrame({"seed": df["seed"],
                         "truth": np.abs(df["magnification_true_at_obs"].to_numpy(float)),
                         "recovered": np.abs(df["magnification_model_q50"].to_numpy(float))})
    data = data[np.isfinite(data.truth) & np.isfinite(data.recovered) & (data.truth > 0) & (data.recovered > 0)].copy()
    if data.empty:
        return pd.DataFrame()
    lo, hi = data.truth.min(), data.truth.max()
    if hi == lo:
        lo, hi = lo / 1.1, hi * 1.1
    edges = np.geomspace(lo, hi, bins + 1)
    data["bin"] = np.clip(np.searchsorted(edges, data.truth, side="right") - 1, 0, bins - 1)
    rows = []
    for index, images in data.groupby("bin"):
        mocks = images.groupby("seed")[["truth", "recovered"]].mean()
        n = len(mocks)
        rows.append({"bin_lower": edges[index], "bin_upper": edges[index + 1],
                     "truth_mean": mocks.truth.mean(), "recovered_mean": mocks.recovered.mean(),
                     "recovered_sem": mocks.recovered.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                     "n_mocks": n, "n_images": len(images)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- image residual distribution
@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_image_residual_distribution(campaign: Campaign, path: Path, *, bins: int = 24) -> Path | None:
    df = campaign.residual_draws
    residual = pd.to_numeric(df["image_residual_arcsec"], errors="coerce").to_numpy(float)
    recovered = df["recovered"].astype(bool).to_numpy() & np.isfinite(residual) & (residual >= 0)
    if not recovered.any():
        return None
    x_max = float(np.ceil(np.quantile(residual[recovered], 0.99) * 1.10 / 0.05) * 0.05)
    edges = np.linspace(0.0, max(x_max, 1e-6), bins + 1)
    groups = [residual[recovered & (df["seed"] == seed).to_numpy() & (df["draw_index"] == draw).to_numpy()]
              for seed, draw in df.loc[recovered, ["seed", "draw_index"]].drop_duplicates().itertuples(index=False)]
    counts = np.stack([np.histogram(values, bins=edges)[0] for values in groups])
    h16, h50, h84 = np.percentile(counts, [16, 50, 84], axis=0)
    rms = float(np.median([np.sqrt(np.mean(values ** 2)) for values in groups]))
    median = float(np.median([np.median(values) for values in groups]))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.stairs(h84, edges, baseline=h16, fill=True, color=COLOR, alpha=0.08,
              linewidth=0.0, zorder=1)
    ax.stairs(h50, edges, color=COLOR, linewidth=1.5, label="Median", zorder=3)
    ax.axvline(rms, color="#ba5a2a", linestyle="--", linewidth=1.3,
               label="RMS", zorder=4)
    ax.axvline(median, color="#303030", linestyle=":", linewidth=1.3,
               label="Median", zorder=4)
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Image-position residual (arcsec)")
    ax.set_xlim(0.0, edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    _polish(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .87), fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------- aperture mass components
@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_m2d_components(campaign: Campaign, path: Path, *, seed: int | None = None) -> Path | None:
    """Top: enclosed mass per component for one representative realization.  Bottom: the campaign, median over
    realizations of each realization's posterior-median fractional mass error with the 16th to 84th percentile band across realizations,
    out to the median radius of the outermost image."""
    seed = representative_seed(campaign) if seed is None else int(seed)
    groups = (*COMPONENT_GROUPS, "total")
    colors = {**COMPONENT_COLORS, "total": "black"}
    labels = {**COMPONENT_LABELS, "total": "Total"}
    m2d = campaign.m2d[campaign.m2d["component"].isin(groups)]
    rows = m2d[m2d["seed"] == seed]
    if rows.empty:
        return None
    outer = float(np.nanmedian(pd.to_numeric(m2d["outer_image_radius_kpc"], errors="coerce"))) if "outer_image_radius_kpc" in m2d else np.inf
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.08})
    for group in groups:
        sel = rows[rows["component"] == group].sort_values("radius_kpc")
        if sel.empty:
            continue
        radius, color = sel["radius_kpc"].to_numpy(float), colors[group]
        line_order = 2 if group == "total" else 3
        top.plot(radius, sel["truth_m2d_msun"].to_numpy(float), color=color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=line_order)
        top.plot(radius, sel["model_q50"].to_numpy(float), color=color, linewidth=1.6, label=labels[group], zorder=line_order)
        stats = m2d[m2d["component"] == group].groupby("radius_kpc")["ratio_q50"].agg(
            q16=lambda v: np.nanpercentile(v, 16), q50="median", q84=lambda v: np.nanpercentile(v, 84)).sort_index()
        inside = stats.index.to_numpy(float) <= outer
        r_c = stats.index.to_numpy(float)[inside]
        q16, q50, q84 = (stats[q].to_numpy(float)[inside] - 1.0 for q in ("q16", "q50", "q84"))
        bottom.fill_between(r_c, q16, q84, color=color, alpha=0.12, linewidth=0.0,
                            zorder=0 if group == "total" else 1)
        bottom.plot(r_c, q50, color=color, linewidth=1.4, zorder=line_order)
    top.plot([], [], color="0.3", linestyle="--", linewidth=1.0, label="Truth")
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel(r"$M_{\rm 2D}(<R)$ ($M_\odot$)")
    top.legend(loc="lower right", frameon=False, fontsize=8)
    bottom.axhline(0.0, color="0.2", linestyle="--", linewidth=1.0)
    bottom.set_ylim(-0.3, 0.3)
    bottom.set_xlabel(r"$R$ (kpc)")
    bottom.set_ylabel(r"$M_{\rm fit}/M_{\rm true}-1$")
    for ax in (top, bottom):
        _polish(ax)
    bottom.yaxis.set_major_locator(MaxNLocator(3))
    fig.subplots_adjust(left=.20, right=.98, bottom=.13, top=.98)
    return _save(fig, path)


# ----------------------------------------------------------------------------- member scaling relation
@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_relation_recovery(campaign: Campaign, path: Path) -> Path | None:
    """Top: the recovered relation over the true one against L/L*, median and 16th to 84th band across realizations.
    Bottom: truth-centred posteriors of all realizations stacked in the (sigma*, r_cut*) and (alpha, beta) planes,
    68 and 95 per cent contours."""
    from scipy.ndimage import gaussian_filter
    rel = [r for r in campaign.relation if r.get("draws")]
    if len(rel) < 2:
        return None
    ell = np.geomspace(0.02, 20.0, 60)
    curves = {"sigma": [], "cutkpc": []}
    for r in rel:
        d, t = {k: np.asarray(v, float) for k, v in r["draws"].items()}, r["truth"]
        lp = ell / float(r["pivot_luminosity_ratio"])                     # L / L_pivot
        curves["sigma"].append(np.median(d["sigma"][:, None] * lp[None, :] ** d["alpha_sigma"][:, None], axis=0) / (t["sigma"] * lp ** t["alpha_sigma"]))
        curves["cutkpc"].append(np.median(d["cutkpc"][:, None] * lp[None, :] ** d["beta_radius"][:, None], axis=0) / (t["cutkpc"] * lp ** t["beta_radius"]))
    fig = plt.figure(figsize=(7.1, 5.4))
    gs = fig.add_gridspec(2, 2, left=.085, right=.985, bottom=.09, top=.985, hspace=.42, wspace=.28)
    top = [fig.add_subplot(gs[0, 0])]
    top.append(fig.add_subplot(gs[0, 1], sharex=top[0]))
    bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, key, label in zip(top, ("sigma", "cutkpc"), (r"$\sigma_{\rm fit}(L)\,/\,\sigma_{\rm true}(L)$", r"$r_{\rm cut,fit}(L)\,/\,r_{\rm cut,true}(L)$")):
        q16, q50, q84 = np.percentile(np.stack(curves[key]), [16, 50, 84], axis=0)
        ax.fill_between(ell, q16, q84, color=COLOR, alpha=0.18, linewidth=0)
        ax.plot(ell, q50, color=COLOR, linewidth=1.5)
        ax.axhline(1.0, color="0.2", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"$L/L_\star$")
        ax.set_ylabel(label)
        _polish(ax)
    planes = ((("sigma", "cutkpc"), (r"$\sigma_\star\,/\,\sigma_{\star,\rm true}-1$", r"$r_{\rm cut,\star}\,/\,r_{\rm cut,\star,true}-1$"), True),
              (("alpha_sigma", "beta_radius"), (r"$\alpha-\alpha_{\rm true}$", r"$\beta-\beta_{\rm true}$"), False))
    for ax, ((px, py), (lx, ly), fractional) in zip(bottom, planes):
        xs, ys = [], []
        for r in rel:
            x, y = np.asarray(r["draws"][px], float), np.asarray(r["draws"][py], float)
            tx, ty = float(r["truth"][px]), float(r["truth"][py])
            xs.append(x / tx - 1.0 if fractional else x - tx)
            ys.append(y / ty - 1.0 if fractional else y - ty)
        x, y = np.concatenate(xs), np.concatenate(ys)
        lim_x, lim_y = np.percentile(np.abs(x), 99.5), np.percentile(np.abs(y), 99.5)
        h, xe, ye = np.histogram2d(x, y, bins=50, range=[[-lim_x, lim_x], [-lim_y, lim_y]])
        hs = gaussian_filter(h, 1.2)
        hs /= hs.sum()
        order = np.sort(hs.ravel())[::-1]
        cum = np.cumsum(order)
        levels = [order[np.searchsorted(cum, q)] for q in (0.95, 0.68)]
        cx, cy = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
        inside = hs >= levels[0]                                         # axis limits: 1.7 times the reach of the 95% contour
        lim_x, lim_y = 1.7 * float(np.abs(cx[inside.any(axis=1)]).max()), 1.7 * float(np.abs(cy[inside.any(axis=0)]).max())
        ax.contourf(cx, cy, hs.T, levels=[levels[0], levels[1], hs.max()], colors=["#cfe0ec", "#80afcb"])
        ax.contour(cx, cy, hs.T, levels=levels, colors=["#1b4f6e"], linewidths=0.8)
        ax.plot(np.median([np.median(v) for v in xs]), np.median([np.median(v) for v in ys]), "x", color="k", markersize=6, markeredgewidth=1.2)
        ax.axhline(0.0, color="0.2", linestyle="--", linewidth=0.8)
        ax.axvline(0.0, color="0.2", linestyle="--", linewidth=0.8)
        ax.set_xlim(-lim_x, lim_x)
        ax.set_ylim(-lim_y, lim_y)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        _polish(ax)
    return _save(fig, path)


# ----------------------------------------------------------------------------- coverage and calibration
def coverage_table(campaign: Campaign) -> pd.DataFrame:
    """Per parameter across realizations: 68% and 95% coverage with binomial errors, median bias and median fractional
    bias.  Only scored rows count (finite truth inside the support); per-member offsets and source positions are pooled."""
    df = campaign.parameters.copy()
    if "scored" not in df:
        df["scored"] = np.isfinite(df["truth"].to_numpy(float))
    df = df[df["scored"].astype(bool)].copy()
    name = df["parameter"].astype(str)
    df["group"] = np.where(name.str.startswith("source.") & name.str.endswith("beta_x"), "source.beta_x",
                  np.where(name.str.startswith("source.") & name.str.endswith("beta_y"), "source.beta_y",
                  np.where(name.str.contains(r"\.member\d+\."), name.str.replace(r"\.member\d+\.", ".member.", regex=True), name)))
    rows = []
    for group, part in df.groupby("group", sort=False):
        n = len(part)
        for level in ("68", "95"):
            part[f"c{level}"] = part[f"covered_{level}"].astype(bool) if f"covered_{level}" in part else np.nan
        c68, c95 = float(part["c68"].mean()), float(part["c95"].mean()) if "covered_95" in part else np.nan
        rows.append({"parameter": group, "n": n, "coverage_68": c68, "coverage_68_err": math.sqrt(max(c68 * (1 - c68), 0.0) / n),
                     "coverage_95": c95, "coverage_95_err": math.sqrt(max(c95 * (1 - c95), 0.0) / n) if np.isfinite(c95) else np.nan,
                     "median_bias": float(np.nanmedian(part["bias"])), "median_fractional_bias": float(np.nanmedian(part["fractional_bias"])) if "fractional_bias" in part else np.nan,
                     "median_truth_percentile": float(np.nanmedian(part["truth_percentile"]))})
    return pd.DataFrame(rows)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_calibration(campaign: Campaign, path: Path) -> Path | None:
    """Probability-integral-transform check: the truth's percentile rank in the posterior, pooled over all scored
    parameters and realizations, should be uniform; the cumulative curve should follow the diagonal."""
    df = campaign.parameters
    scored = df["scored"].astype(bool) if "scored" in df else np.isfinite(df["truth"].to_numpy(float))
    ranks = pd.to_numeric(df.loc[scored, "truth_percentile"], errors="coerce").dropna().to_numpy(float)
    if ranks.size == 0:
        return None
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    edges = np.linspace(0.0, 1.0, 11)
    left.hist(ranks, bins=edges, color=COLOR, alpha=0.85)
    left.axhline(ranks.size / 10.0, color="0.25", linestyle="--", linewidth=1.0)
    left.set_xlabel("Posterior percentile of the truth")
    left.set_ylabel("Count")
    ordered = np.sort(ranks)
    right.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, color=COLOR, linewidth=1.6)
    right.plot([0, 1], [0, 1], color="0.25", linestyle="--", linewidth=1.0)
    band = 1.36 / math.sqrt(ordered.size)                      # 95% Kolmogorov band for a uniform sample
    right.fill_between([0, 1], [-band, 1 - band], [band, 1 + band], color="0.8", alpha=0.4, linewidth=0.0)
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_xlabel("Posterior percentile of the truth")
    right.set_ylabel("Cumulative fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _standardized_parameter_panels(campaign: Campaign, *, sources: bool = True):
    """Residuals in units of half the central 68% posterior width, not Gaussian z-scores."""
    df = campaign.parameters.copy()
    scale = (df["q84"] - df["q16"]) / 2
    valid = np.isfinite(df["truth"]) & np.isfinite(df["median"]) & np.isfinite(scale) & (scale > 0)
    df = df.loc[valid].copy()
    df["residual"] = (df["median"] - df["truth"]) / scale.loc[valid]
    df = df[np.isfinite(df["residual"])]
    panels = [(p, df[df["parameter"] == p].copy()) for p in PAPER_PARAMETERS if (df["parameter"] == p).any()]
    if sources:
        for coordinate in ("beta_x", "beta_y"):
            rows = df[df["parameter"].str.startswith("source.") & df["parameter"].str.endswith("." + coordinate)]
            if not rows.empty:
                panels.append(("source." + coordinate, rows.copy()))
    return panels


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_standardized_residuals(campaign: Campaign, path: Path, *, ncols: int = 4) -> Path | None:
    """Truth-dependent residuals; binned SEM uses mocks as the independent units, including for pooled sources."""
    panels = _standardized_parameter_panels(campaign)
    if not panels:
        return None
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.36 * nrows), squeeze=False)
    for ax, (parameter, data) in zip(axes.flat, panels):
        ax.axhline(0, color=".25", linestyle="--", linewidth=.8)
        ax.axhspan(-1, 1, color=".94", zorder=-3)
        raw = ax.scatter(data.truth, data.residual, s=5, color=COLOR, alpha=.22, linewidths=0,
                         rasterized=True, label="Individual estimates")
        data["bin"] = pd.cut(data.truth, bins=5, labels=False, include_lowest=True)
        mocks = data.groupby(["bin", "seed"])[["truth", "residual"]].mean()
        binned = mocks.groupby("bin").agg(x=("truth", "mean"), y=("residual", "mean"), sem=("residual", "sem")).dropna()
        trend = ax.errorbar(binned.x, binned.y, yerr=binned["sem"], fmt="o", markersize=4, color="tab:orange",
                            elinewidth=1, capsize=2, label="Mean residual and SEM across mocks")
        if parameter.startswith("source."):
            symbol = r"$\beta_x$" if parameter.endswith("beta_x") else r"$\beta_y$"
            label = f"True {symbol} (arcsec)"
        else:
            label = _axis_label(parameter, "truth")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(r"$(\mathrm{median}-\mathrm{truth})/s_{68}$", fontsize=9)
        ax.text(.04, .96, COMPONENT_NAMES.get(parameter.split(".")[0], "Sources"), transform=ax.transAxes,
                va="top", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        limit = max(3.5, data.residual.abs().max() * 1.1)
        ax.set_ylim(-limit, limit)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.legend([raw, trend], [raw.get_label(), trend.get_label()], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=1.3, w_pad=1.1)
    return _save(fig, path)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_bias_and_coverage(campaign: Campaign, path: Path) -> Path | None:
    """Global parameters only. Horizontal intervals describe residuals across mocks, not posterior intervals.

    Coverage uses the saved posterior-interval flags for scored rows with finite truth, independently of whether
    their central 68% width can be used to standardize the residual. Source coordinates are omitted.
    """
    from matplotlib.lines import Line2D

    panels = _standardized_parameter_panels(campaign, sources=False)
    if not panels:
        return None
    fig, (ax, coverage) = plt.subplots(1, 2, figsize=(9, max(3.5, .36 * len(panels) + 1)), sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1]})
    labels, fractions = [], []
    for i, (parameter, data) in enumerate(panels):
        q = np.quantile(data.residual, [.025, .16, .5, .84, .975])
        component, field = parameter.split(".", 1)
        color = COMPONENT_COLORS["members" if component == "potfile" else component]
        ax.plot(q[[0, 4]], [i, i], color=color, linewidth=1, alpha=.55)
        ax.plot(q[[1, 3]], [i, i], color=color, linewidth=4, solid_capstyle="butt")
        ax.plot(q[2], i, "o", color=color, markersize=5)
        rows = campaign.parameters
        rows = rows[(rows.parameter == parameter) & np.isfinite(rows.truth)]
        if "scored" in rows:
            rows = rows[rows.scored.astype(bool)]
        for column, marker, alpha in (("covered_68", "o", 1), ("covered_95", "s", .55)):
            if column in rows and rows[column].notna().any():
                fraction = rows[column].dropna().astype(float).mean()
                fractions.append(fraction)
                coverage.plot(fraction, i, marker, color=color, alpha=alpha, markersize=5)
        labels.append(COMPONENT_NAMES[component] + "  " + PARAMETER_LABELS.get(field, field))
    ax.axvspan(-1, 1, color=".94", zorder=-3)
    ax.axvline(0, color=".3", linestyle="--", linewidth=.8)
    coverage.axvline(.68, color=".3", linestyle="--", linewidth=.8)
    coverage.axvline(.95, color=".3", linestyle=":", linewidth=.8)
    ax.set_yticks(range(len(panels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$(\mathrm{median}-\mathrm{truth})/s_{68}$")
    coverage.set_xlabel("Empirical coverage")
    coverage.set_xlim(min(.35, min(fractions, default=.35) - .04), 1.03)
    coverage.tick_params(labelleft=False)
    fig.legend(handles=[Line2D([], [], color=".3", marker="o", linewidth=4, label="Median and central 68% across mocks"),
                        Line2D([], [], color=".3", linewidth=1, label="Central 95%"),
                        Line2D([], [], color=".3", marker="o", linestyle="", label="68% coverage"),
                        Line2D([], [], color=".3", marker="s", alpha=.55, linestyle="", label="95% coverage")],
               loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return _save(fig, path)


# ----------------------------------------------------------------------------- driver
def write_aggregate_plots(campaign_root: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    try:
        campaign, fallback = load_campaign(campaign_root), False
    except ValueError:                      # nothing converged (a smoke campaign): aggregate everything and say so
        campaign, fallback = load_campaign(campaign_root, converged_only=False), True
    output = Path(output_dir) if output_dir is not None else campaign.root / "aggregate_plots"
    output.mkdir(parents=True, exist_ok=True)
    campaign.parameters.to_csv(output / "parameter_recovery_table.csv", index=False)
    campaign.summary.to_csv(output / "seed_summary.csv", index=False)
    excluded = campaign.summary[~campaign.summary["converged"].astype(bool)]
    (output / "campaign_selection.json").write_text(json.dumps({
        "realizations": int(len(campaign.summary)), "converged": int(len(campaign.seeds)),
        "rule": f"max R-hat < {CONVERGENCE_RHAT} and min n_eff >= {CONVERGENCE_MIN_NEFF:.0f}", "all_realizations_used": fallback,
        "excluded": [{"seed": int(r.seed), "reasons": r.convergence_reasons} for r in excluded.itertuples()]}, indent=2) + "\n")
    coverage_table(campaign).to_csv(output / "mock_coverage_table.csv", index=False)
    from ..results_tex import write_mock_results
    write_mock_results(campaign, output, rhat_cut=CONVERGENCE_RHAT, min_ess=CONVERGENCE_MIN_NEFF)
    plots = [plot_parameter_recovery(campaign, output / "mock_parameter_recovery_paper.pdf"),
             plot_standardized_residuals(campaign, output / "mock_standardized_residuals.pdf"),
             plot_bias_and_coverage(campaign, output / "mock_bias_and_coverage.pdf"),
             plot_subhalo_mass_function(campaign, output / "mock_aggregate_subhalo_recovery_shmf.pdf"),
             plot_magnification_recovery(campaign, output / "mock_magnification_recovery.pdf"),
             plot_m2d_components(campaign, output / "mock_m2d_aperture_recovery_components.pdf"),
             plot_image_residual_distribution(campaign, output / "mock_aggregate_image_residual_distribution.pdf"),
             plot_calibration(campaign, output / "mock_calibration.pdf"),
             plot_relation_recovery(campaign, output / "mock_relation_recovery.pdf")]
    return [p for p in plots if p is not None]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create the paper figures of a mock-validation campaign.")
    parser.add_argument("campaign_root", help="Directory holding the seed_* folders (validation_runs/<campaign>/<run_name>).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    for path in write_aggregate_plots(args.campaign_root, args.output_dir):
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
