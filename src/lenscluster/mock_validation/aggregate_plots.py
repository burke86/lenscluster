from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from lenscluster.plot_style import apply_lenscluster_plot_style

DEFAULT_RUN_NAME = "single_bcg_production_source"
DEFAULT_STAGES = ("stage1_backprojected_centroid_fit", "stage2_free_source_forward_fit")
DEFAULT_VARIANTS = ("anisotropic", "isotropic")
DEFAULT_FORMATS = ("pdf",)
PAPER_PARAMETERS = (
    "halo.x_centre",
    "halo.y_centre",
    "halo.v_disp",
    "halo.core_radius_kpc",
    "halo.cut_radius_kpc",
    "bcg.x_centre",
    "bcg.y_centre",
    "bcg.v_disp",
    "bcg.core_radius_kpc",
    "bcg.cut_radius_kpc",
    "potfile.sigma",
    "potfile.cutkpc",
    "potfile.alpha_sigma",
    "potfile.gamma_ml",
    "cosmology_Om0",
    "cosmology_w0",
)
STAGE_ALIASES = {
    "stage1_backprojected_centroid_fit": "stage1",
    "stage2_free_source_forward_fit": "stage2",
}
STAGE_LABELS = {
    "stage1_backprojected_centroid_fit": "Stage 1",
    "stage2_free_source_forward_fit": "Stage 2",
}
VARIANT_LABELS = {
    "anisotropic": "Anisotropic",
    "isotropic": "Isotropic",
}
VARIANT_COLORS = {
    "anisotropic": "tab:blue",
    "isotropic": "tab:orange",
}
STAGE_MARKERS = {
    "stage1_backprojected_centroid_fit": "o",
    "stage2_free_source_forward_fit": "s",
}
METHOD_JITTER = {
    ("stage1_backprojected_centroid_fit", "anisotropic"): -0.018,
    ("stage1_backprojected_centroid_fit", "isotropic"): -0.006,
    ("stage2_free_source_forward_fit", "anisotropic"): 0.006,
    ("stage2_free_source_forward_fit", "isotropic"): 0.018,
}
SEED_VARIANT_JITTER = {
    "anisotropic": -0.015,
    "isotropic": 0.015,
}
FRACTIONAL_BIAS_SUFFIXES = (
    ".v_disp",
    ".core_radius_kpc",
    ".cut_radius_kpc",
)
FRACTIONAL_BIAS_PARAMETERS = {
    "potfile.sigma",
    "potfile.cutkpc",
    "source.sigma_int",
}
ABSOLUTE_BIAS_SUFFIXES = (
    ".x_centre",
    ".y_centre",
    ".beta_x",
    ".beta_y",
    ".e1",
    ".e2",
)
ABSOLUTE_BIAS_PARAMETERS = {
    "potfile.alpha_sigma",
    "potfile.gamma_ml",
    "cosmology_Om0",
    "cosmology_w0",
}


@dataclass(frozen=True)
class AggregatePlotOutputs:
    output_dir: Path
    parameter_table: Path
    aggregate_parameter_table: Path
    inventory_table: Path
    subhalo_shmf_seed_table: Path | None
    subhalo_shmf_summary_table: Path | None
    plot_paths: tuple[Path, ...]


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return items


def resolve_campaign_root(campaign_or_path: str | Path, validation_root: str | Path = "validation_runs") -> Path:
    campaign_path = Path(campaign_or_path)
    if campaign_path.exists():
        return campaign_path
    rooted = Path(validation_root) / campaign_path
    if rooted.exists():
        return rooted
    raise FileNotFoundError(f"Could not find campaign path {campaign_path} or {rooted}.")


def stage_alias(stage: str) -> str:
    return STAGE_ALIASES.get(str(stage), str(stage))


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(str(stage), str(stage).replace("_", " "))


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(str(variant), str(variant).replace("_", " ").title())


def method_label(stage: str, variant: str) -> str:
    return f"{stage_label(stage)} {variant_label(variant)}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_dir(seed_dir: Path) -> int | None:
    text = seed_dir.name.removeprefix("seed_")
    try:
        return int(text)
    except ValueError:
        return None


def _parameter_records_from_result(path: Path, stage: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    recovery = payload.get("validation", {}).get("recovery", {})
    stage_payload = recovery.get(stage_alias(stage), {})
    records = stage_payload.get("tables", {}).get("parameters", {}).get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} has no parameter recovery records for {stage_alias(stage)!r}.")
    return [record for record in records if isinstance(record, dict)]


def collect_parameter_recovery(
    campaign_root: str | Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(campaign_root)
    run_root = root / run_name
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_root}")
    campaign = root.name
    parameter_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    seed_dirs = sorted(path for path in run_root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {run_root}.")
    for seed_dir in seed_dirs:
        seed = _seed_from_dir(seed_dir)
        for variant in variants:
            for stage in stages:
                result_path = seed_dir / variant / "recovery" / stage / "results.json"
                inventory = {
                    "campaign": campaign,
                    "run_name": run_name,
                    "seed": seed,
                    "variant": variant,
                    "stage": stage,
                    "stage_label": stage_label(stage),
                    "method_label": method_label(stage, variant),
                    "result_path": str(result_path),
                    "exists": result_path.exists(),
                    "loaded": False,
                    "message": "",
                }
                if not result_path.exists():
                    message = f"Missing aggregate input: {result_path}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise FileNotFoundError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                try:
                    records = _parameter_records_from_result(result_path, stage)
                except Exception as exc:
                    message = f"Could not read {result_path}: {type(exc).__name__}: {exc}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                inventory["loaded"] = True
                inventory["message"] = f"loaded {len(records)} parameter rows"
                inventory_rows.append(inventory)
                for record in records:
                    row = {
                        "campaign": campaign,
                        "run_name": run_name,
                        "seed": seed,
                        "variant": variant,
                        "stage": stage,
                        "stage_label": stage_label(stage),
                        "method_label": method_label(stage, variant),
                    }
                    row.update(record)
                    parameter_rows.append(row)
    parameter_df = pd.DataFrame(parameter_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    if parameter_df.empty:
        raise ValueError(f"No parameter recovery records loaded from {run_root}.")
    for column in ("truth", "median", "q16", "q84", "bias", "truth_percentile"):
        if column in parameter_df:
            parameter_df[column] = pd.to_numeric(parameter_df[column], errors="coerce")
    if "covered_68" in parameter_df:
        parameter_df["covered_68"] = parameter_df["covered_68"].astype(bool)
    return parameter_df, inventory_df


def finite_recovery_rows(parameter_df: pd.DataFrame) -> pd.DataFrame:
    required = ["truth", "median", "q16", "q84"]
    missing = [column for column in required if column not in parameter_df]
    if missing:
        raise KeyError(f"Parameter table is missing required columns: {missing}")
    values = parameter_df[required].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    return parameter_df.loc[mask].copy()


def aggregate_parameter_recovery(
    parameter_df: pd.DataFrame,
    *,
    seed_aggregation: str = "median",
    aggregate_error: str = "seed-quantile",
    fractional_truth_floor: float = 1.0e-6,
) -> pd.DataFrame:
    if seed_aggregation != "median":
        raise ValueError("Only seed_aggregation='median' is currently supported.")
    if aggregate_error != "seed-quantile":
        raise ValueError("Only aggregate_error='seed-quantile' is currently supported.")
    df = finite_recovery_rows(parameter_df)
    if df.empty:
        raise ValueError("No finite parameter recovery rows are available to aggregate.")
    rows: list[dict[str, Any]] = []
    group_columns = ["campaign", "run_name", "parameter", "variant", "stage", "stage_label", "method_label"]
    for keys, group in df.groupby(group_columns, sort=False, dropna=False):
        group_dict = dict(zip(group_columns, keys))
        truth_values = group["truth"].to_numpy(dtype=float)
        recovered_values = group["median"].to_numpy(dtype=float)
        bias_values = recovered_values - truth_values
        denom = np.maximum(np.abs(truth_values), 1.0e-6)
        normalized_bias_values = bias_values / denom
        fractional_bias_values = np.full_like(bias_values, np.nan, dtype=float)
        parameter_name = str(group_dict["parameter"])
        fractional_eligible = is_fractional_bias_parameter(parameter_name) & (
            np.abs(truth_values) > float(fractional_truth_floor)
        )
        fractional_bias_values[fractional_eligible] = bias_values[fractional_eligible] / truth_values[fractional_eligible]
        truth_q16, truth_median, truth_q84 = np.quantile(truth_values, [0.16, 0.5, 0.84])
        rec_q16, rec_median, rec_q84 = np.quantile(recovered_values, [0.16, 0.5, 0.84])
        bias_q16, bias_median, bias_q84 = np.quantile(bias_values, [0.16, 0.5, 0.84])
        nb_q16, nb_median, nb_q84 = np.quantile(normalized_bias_values, [0.16, 0.5, 0.84])
        finite_fractional_bias = fractional_bias_values[np.isfinite(fractional_bias_values)]
        if finite_fractional_bias.size:
            fb_q16, fb_median, fb_q84 = np.quantile(finite_fractional_bias, [0.16, 0.5, 0.84])
        else:
            fb_q16 = fb_median = fb_q84 = np.nan
        row = {
            **group_dict,
            "truth": float(truth_median),
            "truth_q16": float(truth_q16),
            "truth_q84": float(truth_q84),
            "median": float(rec_median),
            "q16": float(rec_q16),
            "q84": float(rec_q84),
            "bias": float(rec_median - truth_median),
            "median_bias": float(np.median(bias_values)),
            "bias_q16": float(bias_q16),
            "bias_median": float(bias_median),
            "bias_q84": float(bias_q84),
            "normalized_bias_median": float(nb_median),
            "normalized_bias_q16": float(nb_q16),
            "normalized_bias_q84": float(nb_q84),
            "fractional_bias_q16": float(fb_q16),
            "fractional_bias_median": float(fb_median),
            "fractional_bias_q84": float(fb_q84),
            "coverage_68_fraction": float(group["covered_68"].astype(bool).mean()) if "covered_68" in group else np.nan,
            "truth_percentile_median": float(np.nanmedian(group["truth_percentile"].to_numpy(dtype=float)))
            if "truth_percentile" in group
            else np.nan,
            "n_seed_runs": int(group["seed"].nunique()),
            "seed_min": int(np.nanmin(group["seed"].to_numpy(dtype=float))),
            "seed_max": int(np.nanmax(group["seed"].to_numpy(dtype=float))),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _jittered_truth(values: np.ndarray, parameter: str, stage: str, variant: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    span = float(np.nanmax(finite) - np.nanmin(finite))
    scale = span if span > 0.0 else max(float(np.nanmax(np.abs(finite))), 1.0)
    return values + METHOD_JITTER.get((stage, variant), 0.0) * scale


def _jittered_seed_truth(values: np.ndarray, seeds: np.ndarray, variant: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    span = float(np.nanmax(finite) - np.nanmin(finite))
    scale = span if span > 0.0 else max(float(np.nanmax(np.abs(finite))), 1.0)
    seed_values = pd.Series(seeds).dropna().astype(int).drop_duplicates().sort_values().to_list()
    if len(seed_values) <= 1:
        seed_offsets = {seed: 0.0 for seed in seed_values}
    else:
        centered = np.linspace(-0.5, 0.5, len(seed_values))
        seed_offsets = {seed: 0.028 * float(offset) for seed, offset in zip(seed_values, centered)}
    jitter = np.full_like(values, SEED_VARIANT_JITTER.get(str(variant), 0.0) * scale, dtype=float)
    for index, seed in enumerate(seeds):
        try:
            jitter[index] += seed_offsets.get(int(seed), 0.0) * scale
        except (TypeError, ValueError):
            continue
    return values + jitter


def _axis_limits(*arrays: np.ndarray) -> tuple[float, float] | None:
    finite_parts = [np.asarray(array, dtype=float)[np.isfinite(array)] for array in arrays]
    finite = np.concatenate([part for part in finite_parts if part.size]) if any(part.size for part in finite_parts) else np.asarray([])
    if finite.size == 0:
        return None
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    if lo == hi:
        pad = max(abs(lo) * 0.08, 1.0e-3)
    else:
        pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


PARAMETER_DISPLAY_LABELS = {
    "bcg.core_radius_kpc": r"BCG $r_\mathrm{core}$",
    "bcg.cut_radius_kpc": r"BCG $r_\mathrm{cut}$",
    "bcg.v_disp": r"BCG $\sigma_v$",
    "bcg.x_centre": "BCG x",
    "bcg.y_centre": "BCG y",
    "halo.core_radius_kpc": r"Halo $r_\mathrm{core}$",
    "halo.cut_radius_kpc": r"Halo $r_\mathrm{cut}$",
    "halo.v_disp": r"Halo $\sigma_v$",
    "halo.x_centre": "Halo x",
    "halo.y_centre": "Halo y",
    "potfile.sigma": r"Potfile $\sigma$",
    "potfile.cutkpc": r"Potfile $r_\mathrm{cut}$",
    "potfile.alpha_sigma": r"Potfile $\alpha_\sigma$",
    "potfile.gamma_ml": r"Potfile $\gamma_\mathrm{ML}$",
    "cosmology_Om0": r"$\Omega_m$",
    "cosmology_w0": r"$w_0$",
    "source.sigma_int": r"Source $\sigma_\mathrm{int}$",
}


def _format_parameter_label(parameter: str) -> str:
    name = str(parameter)
    if name in PARAMETER_DISPLAY_LABELS:
        return PARAMETER_DISPLAY_LABELS[name]
    if name.startswith("source.") and name.endswith(".beta_x"):
        return r"Source $\beta_x$"
    if name.startswith("source.") and name.endswith(".beta_y"):
        return r"Source $\beta_y$"
    return name.replace("_", " ")


def _plot_recovery_page(
    page_df: pd.DataFrame,
    parameters: list[str],
    *,
    ncols: int = 3,
) -> plt.Figure:
    nrows = int(math.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    for ax, parameter in zip(axes.ravel(), parameters):
        panel = page_df[page_df["parameter"].astype(str) == parameter]
        for (stage, variant), group in panel.groupby(["stage", "variant"], sort=False):
            truth = group["truth"].to_numpy(dtype=float)
            median = group["median"].to_numpy(dtype=float)
            q16 = group["q16"].to_numpy(dtype=float)
            q84 = group["q84"].to_numpy(dtype=float)
            truth_q16 = group["truth_q16"].to_numpy(dtype=float) if "truth_q16" in group else truth
            truth_q84 = group["truth_q84"].to_numpy(dtype=float) if "truth_q84" in group else truth
            x = _jittered_truth(truth, parameter, str(stage), str(variant))
            yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
            xerr = [np.maximum(0.0, truth - truth_q16), np.maximum(0.0, truth_q84 - truth)]
            ax.errorbar(
                x,
                median,
                xerr=xerr,
                yerr=yerr,
                fmt=STAGE_MARKERS.get(str(stage), "o"),
                color=VARIANT_COLORS.get(str(variant), "0.35"),
                markerfacecolor="white" if str(stage).startswith("stage1") else VARIANT_COLORS.get(str(variant), "0.35"),
                markeredgewidth=1.2,
                linestyle="none",
                alpha=0.82,
                label=method_label(str(stage), str(variant)),
            )
        limits = _axis_limits(
            panel["truth"].to_numpy(dtype=float),
            panel["truth_q16"].to_numpy(dtype=float) if "truth_q16" in panel else panel["truth"].to_numpy(dtype=float),
            panel["truth_q84"].to_numpy(dtype=float) if "truth_q84" in panel else panel["truth"].to_numpy(dtype=float),
            panel["median"].to_numpy(dtype=float),
            panel["q16"].to_numpy(dtype=float),
            panel["q84"].to_numpy(dtype=float),
        )
        if limits is not None:
            ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.7)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        label = _format_parameter_label(parameter)
        ax.set_xlabel(f"{label} (truth)")
        ax.set_ylabel(f"{label} (fit)")
    for ax in axes.ravel()[len(parameters) :]:
        ax.axis("off")
    _add_method_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def stage2_seed_recovery_rows(
    parameter_df: pd.DataFrame,
    *,
    stage: str = "stage2_free_source_forward_fit",
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> pd.DataFrame:
    df = finite_recovery_rows(parameter_df)
    mask = df["stage"].astype(str).eq(str(stage)) & df["variant"].astype(str).isin(variants)
    return df.loc[mask].copy()


def source_position_recovery_rows(parameter_df: pd.DataFrame) -> pd.DataFrame:
    df = finite_recovery_rows(parameter_df)
    parameters = df["parameter"].astype(str)
    mask = parameters.str.startswith("source.") & (parameters.str.endswith(".beta_x") | parameters.str.endswith(".beta_y"))
    return df.loc[mask].copy()


def _plot_seed_recovery_page(
    page_df: pd.DataFrame,
    parameters: list[str],
    *,
    ncols: int = 3,
) -> plt.Figure:
    nrows = int(math.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    for ax, parameter in zip(axes.ravel(), parameters):
        _draw_seed_parameter_axis(ax, page_df, parameter)
    for ax in axes.ravel()[len(parameters) :]:
        ax.axis("off")
    _add_variant_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _draw_seed_parameter_axis(ax: plt.Axes, page_df: pd.DataFrame, parameter: str) -> None:
    panel = page_df[page_df["parameter"].astype(str) == parameter]
    for variant, group in panel.groupby("variant", sort=False):
        truth = group["truth"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        q16 = group["q16"].to_numpy(dtype=float)
        q84 = group["q84"].to_numpy(dtype=float)
        seeds = group["seed"].to_numpy() if "seed" in group else np.arange(len(group))
        x = _jittered_seed_truth(truth, seeds, str(variant))
        yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
        ax.errorbar(
            x,
            median,
            yerr=yerr,
            fmt=STAGE_MARKERS.get("stage2_free_source_forward_fit", "s"),
            color=VARIANT_COLORS.get(str(variant), "0.35"),
            markerfacecolor=VARIANT_COLORS.get(str(variant), "0.35"),
            markeredgewidth=1.0,
            linestyle="none",
            alpha=0.55,
            markersize=4.5,
            label=variant_label(str(variant)),
        )
    limits = _axis_limits(
        panel["truth"].to_numpy(dtype=float),
        panel["median"].to_numpy(dtype=float),
        panel["q16"].to_numpy(dtype=float),
        panel["q84"].to_numpy(dtype=float),
    )
    if limits is not None:
        ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.7)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    label = _format_parameter_label(parameter)
    ax.set_xlabel(f"{label} (truth)")
    ax.set_ylabel(f"{label} (fit)")


def _draw_source_position_recovery_axis(ax: plt.Axes, parameter_df: pd.DataFrame, *, include_legend: bool = True) -> bool:
    df = source_position_recovery_rows(parameter_df)
    if df.empty:
        ax.axis("off")
        return False
    plot_df = df.assign(coordinate=np.where(df["parameter"].astype(str).str.endswith(".beta_y"), "beta_y", "beta_x"))
    for (variant, coordinate), group in plot_df.groupby(["variant", "coordinate"], sort=False):
        truth = group["truth"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        q16 = group["q16"].to_numpy(dtype=float)
        q84 = group["q84"].to_numpy(dtype=float)
        seeds = group["seed"].to_numpy() if "seed" in group else np.arange(len(group))
        x = _jittered_seed_truth(truth, seeds, str(variant))
        yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
        ax.errorbar(
            x,
            median,
            yerr=yerr,
            fmt="^" if coordinate == "beta_y" else "o",
            color=VARIANT_COLORS.get(str(variant), "0.35"),
            markerfacecolor="white" if coordinate == "beta_y" else VARIANT_COLORS.get(str(variant), "0.35"),
            markeredgewidth=0.9,
            linestyle="none",
            alpha=0.42,
            markersize=4.0,
        )
    limits = _axis_limits(
        df["truth"].to_numpy(dtype=float),
        df["median"].to_numpy(dtype=float),
        df["q16"].to_numpy(dtype=float),
        df["q84"].to_numpy(dtype=float),
    )
    if limits is not None:
        ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.75)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    ax.set_xlabel("Source Position (truth)")
    ax.set_ylabel("Source Position (fit)")
    if include_legend:
        handles = [
            Line2D([0], [0], marker="o", color=color, linestyle="none", label=variant_label(variant))
            for variant, color in VARIANT_COLORS.items()
        ]
        handles.extend(
            [
                Line2D([0], [0], marker="o", color="0.3", linestyle="none", label=r"$\beta_x$"),
                Line2D([0], [0], marker="^", color="0.3", markerfacecolor="white", linestyle="none", label=r"$\beta_y$"),
            ]
        )
        ax.legend(handles=handles, loc="best", frameon=False)
    return True


def _plot_source_position_recovery_figure(parameter_df: pd.DataFrame) -> plt.Figure | None:
    df = source_position_recovery_rows(parameter_df)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    _draw_source_position_recovery_axis(ax, df)
    fig.tight_layout()
    return fig


def _plot_seed_recovery_single_page(
    page_df: pd.DataFrame,
    parameters: list[str],
    *,
    include_source_position_panel: bool = False,
    ncols: int = 4,
) -> plt.Figure:
    source_df = source_position_recovery_rows(page_df) if include_source_position_panel else pd.DataFrame()
    panel_count = len(parameters) + (0 if source_df.empty else 1)
    nrows = int(math.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.45 * nrows), squeeze=False)
    flat_axes = list(axes.ravel())
    for ax, parameter in zip(flat_axes, parameters):
        _draw_seed_parameter_axis(ax, page_df, parameter)
    next_axis = len(parameters)
    if not source_df.empty:
        _draw_source_position_recovery_axis(flat_axes[next_axis], source_df)
        next_axis += 1
    for ax in flat_axes[next_axis:]:
        ax.axis("off")
    _add_variant_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _add_variant_legend(fig: plt.Figure) -> None:
    variant_handles = [
        Line2D([0], [0], marker="s", color=color, linestyle="none", label=variant_label(variant))
        for variant, color in VARIANT_COLORS.items()
    ]
    fig.legend(handles=variant_handles, loc="upper center", ncol=len(variant_handles), frameon=False)


def _add_method_legend(fig: plt.Figure) -> None:
    variant_handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="none", label=variant_label(variant))
        for variant, color in VARIANT_COLORS.items()
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.2",
            markerfacecolor="white" if stage.startswith("stage1") else "0.2",
            linestyle="none",
            label=stage_label(stage),
        )
        for stage, marker in STAGE_MARKERS.items()
    ]
    fig.legend(
        handles=[*variant_handles, *stage_handles],
        loc="upper center",
        ncol=len(variant_handles) + len(stage_handles),
        frameon=False,
    )


def _save_multipage_recovery_plot(
    df: pd.DataFrame,
    parameters: list[str],
    output_base: Path,
    formats: tuple[str, ...],
    *,
    parameters_per_page: int = 9,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if not parameters:
        warnings.warn(f"No finite-truth parameters available for {output_base.name}.", RuntimeWarning, stacklevel=2)
        return tuple(paths)
    pages = [parameters[index : index + parameters_per_page] for index in range(0, len(parameters), parameters_per_page)]
    if "pdf" in formats:
        path = output_base.with_suffix(".pdf")
        with PdfPages(path) as pdf:
            for page in pages:
                fig = _plot_recovery_page(df, page)
                pdf.savefig(fig)
                plt.close(fig)
        paths.append(path)
    for fmt in formats:
        if fmt == "pdf":
            continue
        for page_index, page in enumerate(pages, start=1):
            suffix = f"_page{page_index:02d}" if len(pages) > 1 else ""
            path = output_base.with_name(f"{output_base.name}{suffix}").with_suffix(f".{fmt}")
            fig = _plot_recovery_page(df, page)
            fig.savefig(path, dpi=300)
            plt.close(fig)
            paths.append(path)
    return tuple(paths)


def _save_multipage_seed_recovery_plot(
    df: pd.DataFrame,
    parameters: list[str],
    output_base: Path,
    formats: tuple[str, ...],
    *,
    parameters_per_page: int = 9,
    include_source_position_panel: bool = False,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    if not parameters:
        warnings.warn(f"No finite stage-2 seed parameters available for {output_base.name}.", RuntimeWarning, stacklevel=2)
        return tuple(paths)
    pages = [parameters[index : index + parameters_per_page] for index in range(0, len(parameters), parameters_per_page)]
    if "pdf" in formats:
        path = output_base.with_suffix(".pdf")
        with PdfPages(path) as pdf:
            if include_source_position_panel:
                fig = _plot_seed_recovery_single_page(
                    df,
                    parameters,
                    include_source_position_panel=True,
                )
                pdf.savefig(fig)
                plt.close(fig)
            else:
                for page in pages:
                    fig = _plot_seed_recovery_page(df, page)
                    pdf.savefig(fig)
                    plt.close(fig)
        paths.append(path)
    for fmt in formats:
        if fmt == "pdf":
            continue
        if include_source_position_panel:
            path = output_base.with_suffix(f".{fmt}")
            fig = _plot_seed_recovery_single_page(df, parameters, include_source_position_panel=True)
            fig.savefig(path, dpi=300)
            plt.close(fig)
            paths.append(path)
        else:
            for page_index, page in enumerate(pages, start=1):
                suffix = f"_page{page_index:02d}" if len(pages) > 1 else ""
                path = output_base.with_name(f"{output_base.name}{suffix}").with_suffix(f".{fmt}")
                fig = _plot_seed_recovery_page(df, page)
                fig.savefig(path, dpi=300)
                plt.close(fig)
                paths.append(path)
    return tuple(paths)


def _normalized_bias(df: pd.DataFrame, floor: float = 1.0e-6) -> pd.Series:
    truth = df["truth"].to_numpy(dtype=float)
    median = df["median"].to_numpy(dtype=float)
    denom = np.maximum(np.abs(truth), float(floor))
    return pd.Series((median - truth) / denom, index=df.index)


def is_fractional_bias_parameter(parameter: str) -> bool:
    name = str(parameter)
    return name in FRACTIONAL_BIAS_PARAMETERS or name.endswith(FRACTIONAL_BIAS_SUFFIXES)


def is_absolute_bias_parameter(parameter: str) -> bool:
    name = str(parameter)
    if name in ABSOLUTE_BIAS_PARAMETERS or name.endswith(ABSOLUTE_BIAS_SUFFIXES):
        return True
    return not is_fractional_bias_parameter(name)


def _bias_metric_rows(
    df: pd.DataFrame,
    *,
    metric_column: str,
    max_parameters: int,
) -> tuple[pd.DataFrame, list[str]]:
    metric_values = pd.to_numeric(df[metric_column], errors="coerce")
    metric_df = df.assign(_bias_metric=metric_values)
    metric_df = metric_df[np.isfinite(metric_df["_bias_metric"].to_numpy(dtype=float))].copy()
    if metric_df.empty:
        return metric_df, []
    parameter_order = (
        metric_df.groupby("parameter")["_bias_metric"]
        .apply(lambda values: float(np.nanmedian(np.abs(values.to_numpy(dtype=float)))))
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()[:max_parameters]
    )
    return metric_df[metric_df["parameter"].astype(str).isin(parameter_order)].copy(), parameter_order


def _plot_bias_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    parameter_order: list[str],
    methods: list[str],
    *,
    ylabel: str,
    title: str,
    metric_column: str,
    metric_q16_column: str,
    metric_q84_column: str,
) -> None:
    if df.empty or not parameter_order:
        ax.text(0.5, 0.5, "No eligible parameters", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    positions: list[float] = []
    labels: list[str] = []
    for param_index, parameter in enumerate(parameter_order):
        base = param_index * (len(methods) + 1)
        for method_index, method in enumerate(methods):
            subset = df[(df["parameter"].astype(str) == parameter) & (df["method_label"].astype(str) == method)]
            x = float(base + method_index)
            labels.append(parameter if method_index == len(methods) // 2 else "")
            positions.append(x)
            if subset.empty:
                continue
            row = subset.iloc[0]
            y = float(row[metric_column])
            y_q16 = float(row[metric_q16_column])
            y_q84 = float(row[metric_q84_column])
            if not np.isfinite([y, y_q16, y_q84]).all():
                continue
            stage = str(row["stage"])
            variant = str(row["variant"])
            ax.errorbar(
                [x],
                [y],
                yerr=[[max(0.0, y - y_q16)], [max(0.0, y_q84 - y)]],
                fmt=STAGE_MARKERS.get(stage, "o"),
                color=VARIANT_COLORS.get(variant, "0.35"),
                markerfacecolor="white" if stage.startswith("stage1") else VARIANT_COLORS.get(variant, "0.35"),
                markeredgewidth=1.2,
                linestyle="none",
                alpha=0.9,
            )
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.set_xticks(positions, labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_parameter_bias_summary(
    parameter_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
    *,
    max_parameters: int = 24,
    fractional_truth_floor: float = 1.0e-6,
) -> tuple[Path, ...]:
    df = finite_recovery_rows(parameter_df)
    if df.empty:
        return tuple()
    df = df.copy()
    parameters = df["parameter"].astype(str)
    finite_truth = np.isfinite(df["truth"].to_numpy(dtype=float))
    away_from_zero = np.abs(df["truth"].to_numpy(dtype=float)) > float(fractional_truth_floor)
    fractional_mask = parameters.map(is_fractional_bias_parameter).to_numpy(dtype=bool) & finite_truth & away_from_zero
    absolute_mask = parameters.map(is_absolute_bias_parameter).to_numpy(dtype=bool) | ~fractional_mask
    fractional_df = df.loc[fractional_mask].copy()
    absolute_df = df.loc[absolute_mask].copy()
    if "fractional_bias_median" not in fractional_df:
        fractional_df["fractional_bias_median"] = (
            fractional_df["median"].to_numpy(dtype=float) - fractional_df["truth"].to_numpy(dtype=float)
        ) / fractional_df["truth"].to_numpy(dtype=float)
        fractional_df["fractional_bias_q16"] = (
            fractional_df["q16"].to_numpy(dtype=float) - fractional_df["truth"].to_numpy(dtype=float)
        ) / fractional_df["truth"].to_numpy(dtype=float)
        fractional_df["fractional_bias_q84"] = (
            fractional_df["q84"].to_numpy(dtype=float) - fractional_df["truth"].to_numpy(dtype=float)
        ) / fractional_df["truth"].to_numpy(dtype=float)
    if "bias_median" not in absolute_df:
        absolute_df["bias_median"] = absolute_df["bias"].to_numpy(dtype=float)
        absolute_df["bias_q16"] = absolute_df["q16"].to_numpy(dtype=float) - absolute_df["truth"].to_numpy(dtype=float)
        absolute_df["bias_q84"] = absolute_df["q84"].to_numpy(dtype=float) - absolute_df["truth"].to_numpy(dtype=float)
    fractional_df, fractional_order = _bias_metric_rows(
        fractional_df,
        metric_column="fractional_bias_median",
        max_parameters=max_parameters,
    )
    absolute_df, absolute_order = _bias_metric_rows(
        absolute_df,
        metric_column="bias_median",
        max_parameters=max_parameters,
    )
    methods = [method for method in (method_label(stage, variant) for stage in DEFAULT_STAGES for variant in DEFAULT_VARIANTS) if method in set(df["method_label"])]
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        width_parameters = max(len(fractional_order), len(absolute_order), 1)
        fig, axes = plt.subplots(2, 1, figsize=(max(7, 0.34 * width_parameters), 8.2), squeeze=False)
        _plot_bias_metric_panel(
            axes[0, 0],
            fractional_df,
            fractional_order,
            methods,
            ylabel="(recovered median - truth) / truth",
            title="Fractional bias for positive scale parameters",
            metric_column="fractional_bias_median",
            metric_q16_column="fractional_bias_q16",
            metric_q84_column="fractional_bias_q84",
        )
        _plot_bias_metric_panel(
            axes[1, 0],
            absolute_df,
            absolute_order,
            methods,
            ylabel="recovered median - truth",
            title="Absolute bias for signed or zero-centered parameters",
            metric_column="bias_median",
            metric_q16_column="bias_q16",
            metric_q84_column="bias_q84",
        )
        _add_method_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _positive_finite_values(values: list[Any] | np.ndarray) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _combined_recovery_path(seed_dir: Path, variant: str) -> Path:
    return seed_dir / variant / "recovery" / "results.json"


def _stage_manifest(payload: dict[str, Any], stage: str) -> dict[str, Any] | None:
    manifests = payload.get("solver", {}).get("stage_manifests", [])
    if not isinstance(manifests, list):
        return None
    for manifest in manifests:
        if isinstance(manifest, dict) and str(manifest.get("stage")) == str(stage):
            return manifest
    return None


def _table_records_from_manifest(manifest: dict[str, Any], table_name: str) -> list[dict[str, Any]]:
    artifact = manifest.get("table_artifacts", {}).get(table_name, {})
    if not isinstance(artifact, dict):
        return []
    data = artifact.get("data", {})
    if not isinstance(data, dict):
        return []
    records = data.get("records", [])
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _truth_subhalo_masses(payload: dict[str, Any]) -> np.ndarray:
    truth = payload.get("mock_cluster", {}).get("truth", {})
    subhalos = truth.get("subhalos", []) if isinstance(truth, dict) else []
    masses = [record.get("subhalo_mass_msun") for record in subhalos if isinstance(record, dict)]
    return _positive_finite_values(masses)


def _recovered_stage2_subhalo_log_masses(payload: dict[str, Any]) -> np.ndarray:
    manifest = _stage_manifest(payload, "stage2_free_source_forward_fit")
    if manifest is None:
        return np.empty((0,), dtype=float)
    records = _table_records_from_manifest(manifest, "scaling_relation_summary.csv")
    values: list[Any] = []
    for record in records:
        value = record.get("scaling_log10_mass_msun_best")
        if value is None:
            value = record.get("scaling_log10_mass_msun_median")
        values.append(value)
    log_mass = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    return log_mass[np.isfinite(log_mass)]


def _truth_subhalo_selection(payload: dict[str, Any]) -> dict[str, Any]:
    truth = payload.get("mock_cluster", {}).get("truth", {})
    selection = truth.get("subhalo_selection", {}) if isinstance(truth, dict) else {}
    return selection if isinstance(selection, dict) else {}


def _shmf_log_mass_bins(log_mass_values: list[np.ndarray]) -> np.ndarray | None:
    finite_arrays = [np.asarray(values, dtype=float) for values in log_mass_values if np.asarray(values).size]
    finite_arrays = [values[np.isfinite(values)] for values in finite_arrays]
    finite_arrays = [values for values in finite_arrays if values.size]
    if not finite_arrays:
        return None
    all_values = np.concatenate(finite_arrays)
    log_min = float(np.nanmin(all_values))
    log_max = float(np.nanmax(all_values))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if log_max <= log_min:
        log_min -= 0.25
        log_max += 0.25
    n_bins = int(np.clip(np.sqrt(max(int(all_values.size), 1)), 6, 18))
    return np.linspace(log_min, log_max, n_bins + 1)


def collect_subhalo_shmf_inputs(
    campaign_root: str | Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(campaign_root)
    run_root = root / run_name
    rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for seed_dir in sorted(path for path in run_root.glob("seed_*") if path.is_dir()):
        seed = _seed_from_dir(seed_dir)
        for variant in variants:
            path = _combined_recovery_path(seed_dir, variant)
            if not path.exists():
                message = f"Missing SHMF aggregate input: {path}"
                if strict:
                    raise FileNotFoundError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            try:
                payload = _load_json(path)
            except Exception as exc:
                message = f"Could not read {path}: {type(exc).__name__}: {exc}"
                if strict:
                    raise
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            truth_mass = _truth_subhalo_masses(payload)
            recovered_log_mass = _recovered_stage2_subhalo_log_masses(payload)
            if truth_mass.size == 0 or recovered_log_mass.size == 0:
                message = f"Missing truth or recovered SHMF masses in {path}"
                if strict:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            if not selection:
                selection = _truth_subhalo_selection(payload)
            rows.append(
                {
                    "campaign": root.name,
                    "run_name": run_name,
                    "seed": seed,
                    "variant": variant,
                    "truth_log_mass": np.log10(truth_mass),
                    "recovered_log_mass": recovered_log_mass,
                    "result_path": str(path),
                }
            )
    return rows, selection


def build_subhalo_shmf_tables(
    shmf_inputs: list[dict[str, Any]],
    *,
    log_mass_bins: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not shmf_inputs:
        return pd.DataFrame(), pd.DataFrame()
    if log_mass_bins is None:
        log_mass_bins = _shmf_log_mass_bins(
            [
                *(np.asarray(row["truth_log_mass"], dtype=float) for row in shmf_inputs),
                *(np.asarray(row["recovered_log_mass"], dtype=float) for row in shmf_inputs),
            ]
        )
    if log_mass_bins is None:
        return pd.DataFrame(), pd.DataFrame()
    bins = np.asarray(log_mass_bins, dtype=float)
    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    seed_rows: list[dict[str, Any]] = []
    for row in shmf_inputs:
        truth_counts, _ = np.histogram(np.asarray(row["truth_log_mass"], dtype=float), bins=bins)
        recovered_counts, _ = np.histogram(np.asarray(row["recovered_log_mass"], dtype=float), bins=bins)
        truth_density = truth_counts.astype(float) / bin_widths
        recovered_density = recovered_counts.astype(float) / bin_widths
        ratio = np.full_like(recovered_density, np.nan, dtype=float)
        valid_ratio = truth_density > 0.0
        ratio[valid_ratio] = recovered_density[valid_ratio] / truth_density[valid_ratio]
        for index, center in enumerate(bin_centers):
            seed_rows.append(
                {
                    "campaign": row["campaign"],
                    "run_name": row["run_name"],
                    "seed": row["seed"],
                    "variant": row["variant"],
                    "bin_index": index,
                    "log_mass_left": float(bins[index]),
                    "log_mass_right": float(bins[index + 1]),
                    "log_mass_center": float(center),
                    "bin_width": float(bin_widths[index]),
                    "truth_count": int(truth_counts[index]),
                    "recovered_count": int(recovered_counts[index]),
                    "truth_density": float(truth_density[index]),
                    "recovered_density": float(recovered_density[index]),
                    "recovered_to_truth": float(ratio[index]) if np.isfinite(ratio[index]) else np.nan,
                }
            )
    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []

    def add_summary_rows(frame: pd.DataFrame, *, series: str, variant: str, value_column: str) -> None:
        for bin_index, group in frame.groupby("bin_index", sort=True):
            values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            first = group.iloc[0]
            q16, median, q84 = np.quantile(values, [0.16, 0.5, 0.84])
            summary_rows.append(
                {
                    "series": series,
                    "variant": variant,
                    "bin_index": int(bin_index),
                    "log_mass_left": float(first["log_mass_left"]),
                    "log_mass_right": float(first["log_mass_right"]),
                    "log_mass_center": float(first["log_mass_center"]),
                    "value_q16": float(q16),
                    "value_median": float(median),
                    "value_q84": float(q84),
                    "n_seed_runs": int(group["seed"].nunique()),
                }
            )

    truth_frame = seed_df.drop_duplicates(["seed", "bin_index"])
    add_summary_rows(truth_frame, series="truth", variant="truth", value_column="truth_density")
    for variant, variant_frame in seed_df.groupby("variant", sort=False):
        add_summary_rows(variant_frame, series="recovered", variant=str(variant), value_column="recovered_density")
        add_summary_rows(variant_frame, series="ratio", variant=str(variant), value_column="recovered_to_truth")
    return seed_df, pd.DataFrame(summary_rows)


def _schechter_shmf_curve(selection: dict[str, Any], x_values: np.ndarray, normalization: float) -> np.ndarray | None:
    try:
        schechter_alpha = float(selection.get("schechter_alpha"))
        mass_ref = float(selection.get("mass_ref"))
        exponent = float(selection.get("mass_luminosity_exponent"))
    except (TypeError, ValueError):
        return None
    if not (
        np.isfinite(schechter_alpha)
        and np.isfinite(mass_ref)
        and mass_ref > 0.0
        and np.isfinite(exponent)
        and exponent > 0.0
        and np.isfinite(normalization)
        and normalization > 0.0
    ):
        return None
    mass_grid = np.power(10.0, x_values)
    luminosity_grid = np.power(mass_grid / mass_ref, 1.0 / exponent)
    density = np.power(luminosity_grid, schechter_alpha + 1.0) * np.exp(-luminosity_grid)
    density_norm = float(np.trapezoid(density, x_values))
    if not (np.isfinite(density_norm) and density_norm > 0.0):
        return None
    return density / density_norm * normalization


def _positive_shmf_plot_values(summary_df: pd.DataFrame) -> np.ndarray:
    empirical = summary_df[summary_df["series"].isin(["truth", "recovered"])]
    values: list[np.ndarray] = []
    for column in ("value_q16", "value_median", "value_q84"):
        array = pd.to_numeric(empirical[column], errors="coerce").to_numpy(dtype=float)
        array = array[np.isfinite(array) & (array > 0.0)]
        if array.size:
            values.append(array)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def plot_subhalo_shmf_summary(
    summary_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
    *,
    selection: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    if summary_df.empty:
        return tuple()
    paths: list[Path] = []
    selection = selection or {}
    empirical_y = _positive_shmf_plot_values(summary_df)
    y_floor = float(np.nanmin(empirical_y) * 0.5) if empirical_y.size else np.nan
    y_ceiling = float(np.nanmax(empirical_y) * 2.0) if empirical_y.size else np.nan
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
        top_ax, ratio_ax = axes
        truth = summary_df[(summary_df["series"] == "truth") & (summary_df["variant"] == "truth")].sort_values("bin_index")
        if not truth.empty:
            x = truth["log_mass_center"].to_numpy(dtype=float)
            y = truth["value_median"].to_numpy(dtype=float)
            y16 = truth["value_q16"].to_numpy(dtype=float)
            y84 = truth["value_q84"].to_numpy(dtype=float)
            mask = y > 0.0
            top_ax.fill_between(x[mask], y16[mask], y84[mask], color="0.7", alpha=0.35, step="mid")
            top_ax.step(x[mask], y[mask], where="mid", color="0.25", linewidth=2.0, label="Truth")
            bin_widths = (truth["log_mass_right"] - truth["log_mass_left"]).to_numpy(dtype=float)
            schechter_norm = float(np.nansum(np.where(mask, y * bin_widths, 0.0)))
            curve = _schechter_shmf_curve(selection, x, schechter_norm)
            if curve is not None:
                curve_mask = np.isfinite(curve)
                if np.isfinite(y_floor):
                    curve_mask &= curve >= y_floor
                if np.any(curve_mask):
                    top_ax.plot(x[curve_mask], curve[curve_mask], color="0.1", linestyle="--", linewidth=1.3, label="Schechter")
        for variant in DEFAULT_VARIANTS:
            recovered = summary_df[(summary_df["series"] == "recovered") & (summary_df["variant"] == variant)].sort_values("bin_index")
            if recovered.empty:
                continue
            x = recovered["log_mass_center"].to_numpy(dtype=float)
            y = recovered["value_median"].to_numpy(dtype=float)
            y16 = recovered["value_q16"].to_numpy(dtype=float)
            y84 = recovered["value_q84"].to_numpy(dtype=float)
            mask = y > 0.0
            color = VARIANT_COLORS.get(variant, "0.35")
            top_ax.fill_between(x[mask], y16[mask], y84[mask], color=color, alpha=0.18, step="mid")
            top_ax.step(x[mask], y[mask], where="mid", color=color, linewidth=2.0, label=variant_label(variant))
            ratio = summary_df[(summary_df["series"] == "ratio") & (summary_df["variant"] == variant)].sort_values("bin_index")
            if not ratio.empty:
                rx = ratio["log_mass_center"].to_numpy(dtype=float)
                ry = ratio["value_median"].to_numpy(dtype=float)
                ry16 = ratio["value_q16"].to_numpy(dtype=float)
                ry84 = ratio["value_q84"].to_numpy(dtype=float)
                finite = np.isfinite(ry)
                ratio_ax.fill_between(rx[finite], ry16[finite], ry84[finite], color=color, alpha=0.18, step="mid")
                ratio_ax.step(rx[finite], ry[finite], where="mid", color=color, linewidth=2.0)
        top_ax.set_yscale("log")
        if np.isfinite(y_floor) and np.isfinite(y_ceiling) and y_floor > 0.0 and y_ceiling > y_floor:
            top_ax.set_ylim(y_floor, y_ceiling)
        top_ax.set_ylabel(r"$dN/d\log M$")
        top_ax.legend(loc="best", frameon=False)
        ratio_ax.axhline(1.0, color="0.25", linewidth=1.0)
        ratio_ax.set_xlabel(r"$\log(M/M_\odot)$")
        ratio_ax.set_ylabel("Recovered / Truth")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def write_subhalo_shmf_outputs(
    campaign_root: str | Path,
    output_path: Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    strict: bool = False,
) -> tuple[Path | None, Path | None, tuple[Path, ...]]:
    shmf_inputs, selection = collect_subhalo_shmf_inputs(
        campaign_root,
        run_name=run_name,
        variants=variants,
        strict=strict,
    )
    seed_df, summary_df = build_subhalo_shmf_tables(shmf_inputs)
    if seed_df.empty or summary_df.empty:
        return None, None, tuple()
    seed_table = output_path / "subhalo_recovery_shmf_seed_table.csv"
    summary_table = output_path / "subhalo_recovery_shmf_summary_table.csv"
    seed_df.to_csv(seed_table, index=False)
    summary_df.to_csv(summary_table, index=False)
    plot_paths = plot_subhalo_shmf_summary(
        summary_df,
        output_path / "subhalo_recovery_shmf",
        formats,
        selection=selection,
    )
    return seed_table, summary_table, plot_paths


def write_aggregate_plots(
    campaign_or_path: str | Path,
    *,
    validation_root: str | Path = "validation_runs",
    run_name: str = DEFAULT_RUN_NAME,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    output_dir: str | Path | None = None,
    parameter_set: str = "both",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    strict: bool = False,
    seed_aggregation: str = "median",
    aggregate_error: str = "seed-quantile",
    fractional_truth_floor: float = 1.0e-6,
) -> AggregatePlotOutputs:
    if parameter_set not in {"paper", "all", "both"}:
        raise ValueError("parameter_set must be 'paper', 'all', or 'both'.")
    formats = tuple(fmt.lower().lstrip(".") for fmt in formats)
    if not formats:
        raise ValueError("formats must contain at least one format.")
    campaign_root = resolve_campaign_root(campaign_or_path, validation_root)
    output_path = Path(output_dir) if output_dir is not None else campaign_root / "aggregate_plots"
    output_path.mkdir(parents=True, exist_ok=True)
    parameter_df, inventory_df = collect_parameter_recovery(
        campaign_root,
        run_name=run_name,
        stages=stages,
        variants=variants,
        strict=strict,
    )
    parameter_table_path = output_path / "parameter_recovery_table.csv"
    aggregate_parameter_table_path = output_path / "aggregate_parameter_recovery_table.csv"
    inventory_path = output_path / "run_inventory.csv"
    aggregate_df = aggregate_parameter_recovery(
        parameter_df,
        seed_aggregation=seed_aggregation,
        aggregate_error=aggregate_error,
        fractional_truth_floor=fractional_truth_floor,
    )
    parameter_df.to_csv(parameter_table_path, index=False)
    aggregate_df.to_csv(aggregate_parameter_table_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)

    apply_lenscluster_plot_style()
    seed_df = stage2_seed_recovery_rows(parameter_df, variants=variants)
    seed_available_parameters = seed_df["parameter"].dropna().astype(str).drop_duplicates().tolist()
    plot_paths: list[Path] = []
    if parameter_set in {"paper", "both"}:
        paper_parameters = [parameter for parameter in PAPER_PARAMETERS if parameter in set(seed_available_parameters)]
        plot_paths.extend(
            _save_multipage_seed_recovery_plot(
                seed_df,
                paper_parameters,
                output_path / "parameter_recovery_paper",
                formats,
                include_source_position_panel=True,
            )
        )
    if parameter_set in {"all", "both"}:
        plot_paths.extend(
            _save_multipage_seed_recovery_plot(
                seed_df,
                seed_available_parameters,
                output_path / "parameter_recovery_all",
                formats,
            )
        )
    plot_paths.extend(
        plot_parameter_bias_summary(
            aggregate_df,
            output_path / "parameter_bias_summary",
            formats,
            fractional_truth_floor=fractional_truth_floor,
        )
    )
    shmf_seed_table, shmf_summary_table, shmf_plot_paths = write_subhalo_shmf_outputs(
        campaign_root,
        output_path,
        run_name=run_name,
        variants=variants,
        formats=formats,
        strict=strict,
    )
    plot_paths.extend(shmf_plot_paths)
    return AggregatePlotOutputs(
        output_dir=output_path,
        parameter_table=parameter_table_path,
        aggregate_parameter_table=aggregate_parameter_table_path,
        inventory_table=inventory_path,
        subhalo_shmf_seed_table=shmf_seed_table,
        subhalo_shmf_summary_table=shmf_summary_table,
        plot_paths=tuple(plot_paths),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create aggregate plots for mock-validation campaigns.")
    parser.add_argument("campaign_or_path", help="Campaign name under validation_runs/ or explicit campaign path.")
    parser.add_argument("--validation-root", default="validation_runs")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--stages", type=parse_csv_tuple, default=DEFAULT_STAGES)
    parser.add_argument("--variants", type=parse_csv_tuple, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parameter-set", choices=("paper", "all", "both"), default="both")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--formats", type=parse_csv_tuple, default=DEFAULT_FORMATS)
    parser.add_argument("--seed-aggregation", choices=("median",), default="median")
    parser.add_argument("--aggregate-error", choices=("seed-quantile",), default="seed-quantile")
    parser.add_argument("--fractional-truth-floor", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs = write_aggregate_plots(
        args.campaign_or_path,
        validation_root=args.validation_root,
        run_name=args.run_name,
        stages=args.stages,
        variants=args.variants,
        output_dir=args.output_dir,
        parameter_set=args.parameter_set,
        formats=args.formats,
        strict=bool(args.strict),
        seed_aggregation=args.seed_aggregation,
        aggregate_error=args.aggregate_error,
        fractional_truth_floor=float(args.fractional_truth_floor),
    )
    print(f"Wrote parameter table: {outputs.parameter_table}")
    print(f"Wrote aggregate parameter table: {outputs.aggregate_parameter_table}")
    print(f"Wrote inventory table: {outputs.inventory_table}")
    if outputs.subhalo_shmf_seed_table is not None:
        print(f"Wrote SHMF seed table: {outputs.subhalo_shmf_seed_table}")
    if outputs.subhalo_shmf_summary_table is not None:
        print(f"Wrote SHMF summary table: {outputs.subhalo_shmf_summary_table}")
    for path in outputs.plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
