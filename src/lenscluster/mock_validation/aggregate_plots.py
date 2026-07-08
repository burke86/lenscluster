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
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

import h5py

from lenscluster.plot_style import apply_lenscluster_plot_style
from lenscluster.utils import DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN, log10_dpie_mass_msun

DEFAULT_RUN_NAME = "single_bcg_production"
DEFAULT_STAGES = ("stage1_backprojected_centroid_fit", "stage2_free_source_forward_fit")
DEFAULT_VARIANTS = ("anisotropic", "anisotropic_maglikelihood")
DEFAULT_FORMATS = ("pdf",)
PAPER_PARAMETERS = (
    "halo.x_centre",
    "halo.y_centre",
    "halo.v_disp",
    "halo.core_radius_kpc",
    "halo.cut_radius_kpc",
    "halo.log10_mass_msun",
    "bcg.x_centre",
    "bcg.y_centre",
    "bcg.v_disp",
    "bcg.core_radius_kpc",
    "bcg.cut_radius_kpc",
    "bcg.log10_mass_msun",
    "potfile.sigma",
    "potfile.cutkpc",
    "potfile.log10_mass_msun",
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
    "anisotropic": "No flux",
    "anisotropic_maglikelihood": "Flux",
    "isotropic": "Isotropic",
}
VARIANT_COLORS = {
    "anisotropic": "tab:blue",
    "anisotropic_maglikelihood": "tab:orange",
    "isotropic": "tab:green",
}
STAGE_MARKERS = {
    "stage1_backprojected_centroid_fit": "o",
    "stage2_free_source_forward_fit": "s",
}
VARIANT_MARKERS = {
    "anisotropic": "o",
    "anisotropic_maglikelihood": "D",
    "isotropic": "s",
}
AGGREGATE_LEGEND_FONT_SIZE = 16
JOINT_SIGMA_RCUT_SPECS = {
    "potfile": {
        "sigma_parameter": "potfile.sigma",
        "rcut_parameter": "potfile.cutkpc",
        "filename": "sigma_rcut_recovery",
        "label": "Potfile",
    },
    "halo": {
        "sigma_parameter": "halo.v_disp",
        "rcut_parameter": "halo.cut_radius_kpc",
        "filename": "halo_sigma_rcut_recovery",
        "label": "Halo",
    },
    "bcg": {
        "sigma_parameter": "bcg.v_disp",
        "rcut_parameter": "bcg.cut_radius_kpc",
        "filename": "bcg_sigma_rcut_recovery",
        "label": "BCG",
    },
}
JOINT_SCALING_PARAMETERS = tuple(
    parameter
    for spec in JOINT_SIGMA_RCUT_SPECS.values()
    for parameter in (spec["sigma_parameter"], spec["rcut_parameter"])
)
JOINT_SCALING_LEVELS = (2.30, 5.99)
AGGREGATE_PARAMETER_POINT_ALPHA = 0.58
IMAGE_RESIDUAL_DISTRIBUTION_RANGE_QUANTILE = 0.99
IMAGE_RESIDUAL_DISTRIBUTION_RANGE_PADDING = 1.10
IMAGE_RESIDUAL_DISTRIBUTION_RANGE_STEP_ARCSEC = 0.05
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
    magnification_table: Path | None
    image_residual_table: Path | None
    image_residual_draws_table: Path | None
    subhalo_shmf_seed_table: Path | None
    subhalo_shmf_summary_table: Path | None
    m2d_aperture_table: Path | None
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


def _ordered_variants_from_df(df: pd.DataFrame, *, fallback: tuple[str, ...] = DEFAULT_VARIANTS) -> tuple[str, ...]:
    if df.empty or "variant" not in df:
        return fallback
    present = {str(value) for value in df["variant"].dropna().astype(str)}
    ordered = [variant for variant in fallback if variant in present]
    ordered.extend(sorted(present.difference(ordered)))
    return tuple(ordered) if ordered else fallback


def _darken_color(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * float(factor), 0.0, 1.0))


def _stage_marker_facecolor(stage: str, variant_color: str) -> str:
    return "white" if str(stage).startswith("stage1") else variant_color


def _stage_marker_edgecolor(stage: str, variant_color: str) -> str | tuple[float, float, float]:
    return variant_color if str(stage).startswith("stage1") else _darken_color(variant_color)


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


def _table_records_from_recovery_result(path: Path, stage: str, table_name: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    recovery = payload.get("validation", {}).get("recovery", {})
    stage_payload = recovery.get(stage_alias(stage), {})
    records = stage_payload.get("tables", {}).get(table_name, {}).get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} has no {table_name!r} recovery records for {stage_alias(stage)!r}.")
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


def collect_magnification_recovery(
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
    magnification_rows: list[dict[str, Any]] = []
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
                    message = f"Missing aggregate magnification input: {result_path}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise FileNotFoundError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                try:
                    records = _table_records_from_recovery_result(result_path, stage, "magnification")
                except Exception as exc:
                    message = f"Could not read magnification from {result_path}: {type(exc).__name__}: {exc}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                inventory["loaded"] = True
                inventory["message"] = f"loaded {len(records)} magnification rows"
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
                        "result_path": str(result_path),
                    }
                    row.update(record)
                    magnification_rows.append(row)
    magnification_df = pd.DataFrame(magnification_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    if not magnification_df.empty:
        for column in (
            "magnification_true",
            "magnification_recovered",
            "magnification_q16",
            "magnification_q50",
            "magnification_q84",
            "magnification_bias",
            "abs_magnification_fractional_error",
            "abs_magnification_fractional_error_q16",
            "abs_magnification_fractional_error_q50",
            "abs_magnification_fractional_error_q84",
        ):
            if column in magnification_df:
                magnification_df[column] = pd.to_numeric(magnification_df[column], errors="coerce")
        if "parity_match" in magnification_df:
            magnification_df["parity_match"] = magnification_df["parity_match"].astype(bool)
    return magnification_df, inventory_df


def collect_image_residual_recovery(
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
    residual_rows: list[dict[str, Any]] = []
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
                    message = f"Missing aggregate image residual input: {result_path}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise FileNotFoundError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                try:
                    records = _table_records_from_recovery_result(result_path, stage, "posterior_image_rms")
                except Exception as exc:
                    message = f"Could not read image residual RMS from {result_path}: {type(exc).__name__}: {exc}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                inventory["loaded"] = True
                inventory["message"] = f"loaded {len(records)} posterior image RMS rows"
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
                        "result_path": str(result_path),
                    }
                    row.update(record)
                    residual_rows.append(row)
    residual_df = pd.DataFrame(residual_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    if not residual_df.empty:
        for column in (
            "draw_index",
            "point_image_rms_arcsec",
            "point_recovered_image_count",
            "arc_aware_image_rms_arcsec",
            "arc_aware_recovered_image_count",
            "arc_supported_image_count",
            "total_image_count",
            "exact_failed_family_count",
        ):
            if column in residual_df:
                residual_df[column] = pd.to_numeric(residual_df[column], errors="coerce")
    return residual_df, inventory_df


def collect_image_residual_draws_recovery(
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
    residual_rows: list[dict[str, Any]] = []
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
                    message = f"Missing aggregate image residual-draw input: {result_path}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise FileNotFoundError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                try:
                    records = _table_records_from_recovery_result(result_path, stage, "posterior_image_residual_draws")
                except Exception as exc:
                    message = f"Could not read image residual draws from {result_path}: {type(exc).__name__}: {exc}"
                    inventory["message"] = message
                    inventory_rows.append(inventory)
                    if strict:
                        raise
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                inventory["loaded"] = True
                inventory["message"] = f"loaded {len(records)} posterior image residual rows"
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
                        "result_path": str(result_path),
                    }
                    row.update(record)
                    residual_rows.append(row)
    residual_df = pd.DataFrame(residual_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    if not residual_df.empty:
        for column in (
            "draw_index",
            "point_image_residual_arcsec",
            "arc_aware_image_residual_arcsec",
        ):
            if column in residual_df:
                residual_df[column] = pd.to_numeric(residual_df[column], errors="coerce")
        for column in ("point_recovered", "arc_aware_recovered", "arc_supported", "exact_failed"):
            if column in residual_df:
                residual_df[column] = residual_df[column].astype(bool)
    return residual_df, inventory_df


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


def _truth_x_positions(values: np.ndarray) -> np.ndarray:
    return values


def _seed_truth_x_positions(values: np.ndarray) -> np.ndarray:
    return values


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
    "bcg.log10_mass_msun": r"BCG $\log_{10} M$",
    "bcg.v_disp": r"BCG $\sigma_v$",
    "bcg.x_centre": "BCG x",
    "bcg.y_centre": "BCG y",
    "halo.core_radius_kpc": r"Halo $r_\mathrm{core}$",
    "halo.cut_radius_kpc": r"Halo $r_\mathrm{cut}$",
    "halo.log10_mass_msun": r"Halo $\log_{10} M$",
    "halo.v_disp": r"Halo $\sigma_v$",
    "halo.x_centre": "Halo x",
    "halo.y_centre": "Halo y",
    "potfile.sigma": r"Potfile $\sigma$",
    "potfile.cutkpc": r"Potfile $r_\mathrm{cut}$",
    "potfile.log10_mass_msun": r"Potfile $\log_{10} M$",
    "potfile.alpha_sigma": r"Potfile $\alpha_\sigma$",
    "potfile.gamma_ml": r"Potfile $\gamma_\mathrm{ML}$",
    "cosmology_Om0": r"$\Omega_m$",
    "cosmology_w0": r"$w_0$",
    "source.sigma_int": r"Source $\sigma_\mathrm{int}$",
}


PARAMETER_UNIT_LABELS = {
    "bcg.core_radius_kpc": "kpc",
    "bcg.cut_radius_kpc": "kpc",
    "bcg.log10_mass_msun": r"$M_\odot$",
    "bcg.v_disp": r"km s$^{-1}$",
    "bcg.x_centre": "arcsec",
    "bcg.y_centre": "arcsec",
    "halo.core_radius_kpc": "kpc",
    "halo.cut_radius_kpc": "kpc",
    "halo.log10_mass_msun": r"$M_\odot$",
    "halo.v_disp": r"km s$^{-1}$",
    "halo.x_centre": "arcsec",
    "halo.y_centre": "arcsec",
    "potfile.sigma": r"km s$^{-1}$",
    "potfile.cutkpc": "kpc",
    "potfile.log10_mass_msun": r"$M_\odot$",
    "source.sigma_int": "arcsec",
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


def _format_parameter_unit(parameter: str) -> str | None:
    name = str(parameter)
    if name in PARAMETER_UNIT_LABELS:
        return PARAMETER_UNIT_LABELS[name]
    if name.startswith("source.") and (name.endswith(".beta_x") or name.endswith(".beta_y")):
        return "arcsec"
    return None


def _format_parameter_axis_label(parameter: str, role: str) -> str:
    label = _format_parameter_label(parameter)
    unit = _format_parameter_unit(parameter)
    if unit is None:
        return f"{label} ({role})"
    return f"{label} ({role}; {unit})"


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
            stage = str(stage)
            variant = str(variant)
            variant_color = VARIANT_COLORS.get(variant, "0.35")
            truth = group["truth"].to_numpy(dtype=float)
            median = group["median"].to_numpy(dtype=float)
            q16 = group["q16"].to_numpy(dtype=float)
            q84 = group["q84"].to_numpy(dtype=float)
            truth_q16 = group["truth_q16"].to_numpy(dtype=float) if "truth_q16" in group else truth
            truth_q84 = group["truth_q84"].to_numpy(dtype=float) if "truth_q84" in group else truth
            x = _truth_x_positions(truth)
            yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
            xerr = [np.maximum(0.0, truth - truth_q16), np.maximum(0.0, truth_q84 - truth)]
            ax.errorbar(
                x,
                median,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=variant_color,
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                linestyle="none",
                alpha=0.30,
                zorder=1,
                label="_nolegend_",
            )
            ax.scatter(
                x,
                median,
                s=21.0,
                marker=VARIANT_MARKERS.get(variant, "o"),
                facecolors=_stage_marker_facecolor(stage, variant_color),
                edgecolors=_stage_marker_edgecolor(stage, variant_color),
                linewidths=0.75,
                alpha=AGGREGATE_PARAMETER_POINT_ALPHA,
                zorder=2,
                label=method_label(stage, variant),
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
        ax.set_xlabel(_format_parameter_axis_label(parameter, "truth"))
        ax.set_ylabel(_format_parameter_axis_label(parameter, "fit"))
    for ax in axes.ravel()[len(parameters) :]:
        ax.axis("off")
    _add_method_legend(fig, variants=_ordered_variants_from_df(page_df))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def seed_recovery_rows_for_stage(
    parameter_df: pd.DataFrame,
    *,
    stage: str = "stage2_free_source_forward_fit",
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> pd.DataFrame:
    df = finite_recovery_rows(parameter_df)
    mask = df["stage"].astype(str).eq(str(stage)) & df["variant"].astype(str).isin(variants)
    return df.loc[mask].copy()


PAPER_MASS_PARAMETER_SPECS = (
    ("halo.log10_mass_msun", "halo.v_disp", "halo.cut_radius_kpc"),
    ("bcg.log10_mass_msun", "bcg.v_disp", "bcg.cut_radius_kpc"),
    ("potfile.log10_mass_msun", "potfile.sigma", "potfile.cutkpc"),
)


def paper_mass_proxy_recovery_rows(parameter_df: pd.DataFrame) -> pd.DataFrame:
    if parameter_df.empty:
        return pd.DataFrame(columns=parameter_df.columns)
    key_columns = [
        column
        for column in ("campaign", "run_name", "seed", "variant", "stage", "stage_label", "method_label", "result_path")
        if column in parameter_df
    ]
    if not key_columns or "parameter" not in parameter_df:
        return pd.DataFrame(columns=parameter_df.columns)
    rows: list[dict[str, Any]] = []
    grouped = parameter_df.groupby(key_columns, sort=False, dropna=False)
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_key = dict(zip(key_columns, key_values))
        by_parameter = {str(record["parameter"]): record for record in group.to_dict("records")}
        for mass_parameter, sigma_parameter, rcut_parameter in PAPER_MASS_PARAMETER_SPECS:
            sigma_row = by_parameter.get(sigma_parameter)
            rcut_row = by_parameter.get(rcut_parameter)
            if sigma_row is None or rcut_row is None:
                continue
            row = dict(group_key)
            row["parameter"] = mass_parameter
            for output_column, input_column in (("truth", "truth"), ("median", "median"), ("q16", "q16"), ("q84", "q84")):
                row[output_column] = float(log10_dpie_mass_msun(sigma_row[input_column], rcut_row[input_column]))
            row["best_fit"] = row["median"]
            row["bias"] = float(row["median"] - row["truth"])
            row["covered_68"] = bool(row["q16"] <= row["truth"] <= row["q84"])
            if "truth_percentile" in parameter_df:
                row["truth_percentile"] = np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=parameter_df.columns)
    mass_df = pd.DataFrame(rows)
    for column in parameter_df.columns:
        if column not in mass_df:
            mass_df[column] = np.nan
    return mass_df.loc[:, list(parameter_df.columns)]


def paper_parameter_recovery_rows(parameter_df: pd.DataFrame) -> pd.DataFrame:
    mass_df = paper_mass_proxy_recovery_rows(parameter_df)
    if mass_df.empty:
        return parameter_df.copy()
    return pd.concat([parameter_df, mass_df], ignore_index=True, sort=False)


def stage2_seed_recovery_rows(
    parameter_df: pd.DataFrame,
    *,
    stage: str = "stage2_free_source_forward_fit",
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> pd.DataFrame:
    return seed_recovery_rows_for_stage(parameter_df, stage=stage, variants=variants)


def _paper_recovery_stage(stages: tuple[str, ...]) -> str:
    normalized = tuple(str(stage) for stage in stages)
    if "stage2_free_source_forward_fit" in normalized:
        return "stage2_free_source_forward_fit"
    if not normalized:
        raise ValueError("stages must contain at least one stage.")
    return normalized[0]


def _paper_plot_output_base(output_path: Path, stage: str, primary_stage: str) -> Path:
    if str(stage) == str(primary_stage):
        return output_path / "mock_parameter_recovery_paper"
    return output_path / f"mock_parameter_recovery_{stage_alias(stage)}_paper"


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
    _add_variant_legend(fig, variants=_ordered_variants_from_df(page_df))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _draw_seed_parameter_axis(ax: plt.Axes, page_df: pd.DataFrame, parameter: str) -> None:
    panel = page_df[page_df["parameter"].astype(str) == parameter]
    for variant, group in panel.groupby("variant", sort=False):
        variant = str(variant)
        stage = str(group["stage"].iloc[0]) if "stage" in group and not group.empty else "stage2_free_source_forward_fit"
        variant_color = VARIANT_COLORS.get(variant, "0.35")
        truth = group["truth"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        q16 = group["q16"].to_numpy(dtype=float)
        q84 = group["q84"].to_numpy(dtype=float)
        seeds = group["seed"].to_numpy() if "seed" in group else np.arange(len(group))
        x = _seed_truth_x_positions(truth)
        yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
        ax.errorbar(
            x,
            median,
            yerr=yerr,
            fmt="none",
            ecolor=variant_color,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
            linestyle="none",
            alpha=0.30,
            zorder=1,
            label="_nolegend_",
        )
        ax.scatter(
            x,
            median,
            s=21.0,
            marker=VARIANT_MARKERS.get(variant, "o"),
            facecolors=_stage_marker_facecolor(stage, variant_color),
            edgecolors=_stage_marker_edgecolor(stage, variant_color),
            linewidths=0.75,
            alpha=AGGREGATE_PARAMETER_POINT_ALPHA,
            zorder=2,
            label=variant_label(variant),
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
    ax.set_xlabel(_format_parameter_axis_label(parameter, "truth"))
    ax.set_ylabel(_format_parameter_axis_label(parameter, "fit"))


def _draw_source_position_recovery_axis(ax: plt.Axes, parameter_df: pd.DataFrame, *, include_legend: bool = True) -> bool:
    df = source_position_recovery_rows(parameter_df)
    if df.empty:
        ax.axis("off")
        return False
    plot_df = df.assign(coordinate=np.where(df["parameter"].astype(str).str.endswith(".beta_y"), "beta_y", "beta_x"))
    for (variant, coordinate), group in plot_df.groupby(["variant", "coordinate"], sort=False):
        variant = str(variant)
        stage = str(group["stage"].iloc[0]) if "stage" in group and not group.empty else "stage2_free_source_forward_fit"
        variant_color = VARIANT_COLORS.get(variant, "0.35")
        truth = group["truth"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        q16 = group["q16"].to_numpy(dtype=float)
        q84 = group["q84"].to_numpy(dtype=float)
        seeds = group["seed"].to_numpy() if "seed" in group else np.arange(len(group))
        x = _seed_truth_x_positions(truth)
        yerr = [np.maximum(0.0, median - q16), np.maximum(0.0, q84 - median)]
        ax.errorbar(
            x,
            median,
            yerr=yerr,
            fmt="none",
            ecolor=variant_color,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
            linestyle="none",
            alpha=0.30,
            zorder=1,
            label="_nolegend_",
        )
        ax.scatter(
            x,
            median,
            s=21.0,
            marker=VARIANT_MARKERS.get(variant, "o"),
            facecolors=_stage_marker_facecolor(stage, variant_color),
            edgecolors=_stage_marker_edgecolor(stage, variant_color),
            linewidths=0.75,
            alpha=AGGREGATE_PARAMETER_POINT_ALPHA,
            zorder=2,
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
    ax.set_xlabel("Source Position (truth; arcsec)")
    ax.set_ylabel("Source Position (fit; arcsec)")
    if include_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker=VARIANT_MARKERS.get(variant, "o"),
                color=color,
                markerfacecolor=color,
                markeredgecolor=_darken_color(color),
                linestyle="none",
                label=variant_label(variant),
            )
            for variant in _ordered_variants_from_df(df)
            for color in (VARIANT_COLORS.get(variant, "0.35"),)
        ]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=AGGREGATE_LEGEND_FONT_SIZE)
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
    _add_variant_legend(fig, variants=_ordered_variants_from_df(page_df))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _add_variant_legend(fig: plt.Figure, *, variants: tuple[str, ...] = DEFAULT_VARIANTS) -> None:
    variant_handles = [
        Line2D(
            [0],
            [0],
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=color,
            markerfacecolor=color,
            markeredgecolor=_darken_color(color),
            linestyle="none",
            label=variant_label(variant),
        )
        for variant in variants
        for color in (VARIANT_COLORS.get(variant, "0.35"),)
    ]
    fig.legend(
        handles=variant_handles,
        loc="upper center",
        ncol=len(variant_handles),
        frameon=False,
        fontsize=AGGREGATE_LEGEND_FONT_SIZE,
    )


def _add_method_legend(fig: plt.Figure, *, variants: tuple[str, ...] = DEFAULT_VARIANTS) -> None:
    variant_handles = [
        Line2D(
            [0],
            [0],
            marker=VARIANT_MARKERS.get(variant, "o"),
            color=color,
            markerfacecolor=color,
            markeredgecolor=_darken_color(color),
            linestyle="none",
            label=variant_label(variant),
        )
        for variant in variants
        for color in (VARIANT_COLORS.get(variant, "0.35"),)
    ]
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.2",
            markerfacecolor="white" if stage.startswith("stage1") else "0.2",
            markeredgecolor="0.2",
            linestyle="none",
            label=stage_label(stage),
        )
        for stage in STAGE_MARKERS
    ]
    fig.legend(
        handles=[*variant_handles, *stage_handles],
        loc="upper center",
        ncol=len(variant_handles) + len(stage_handles),
        frameon=False,
        fontsize=AGGREGATE_LEGEND_FONT_SIZE,
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
        warnings.warn(f"No finite seed parameters available for {output_base.name}.", RuntimeWarning, stacklevel=2)
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
            variant_color = VARIANT_COLORS.get(variant, "0.35")
            ax.errorbar(
                [x],
                [y],
                yerr=[[max(0.0, y - y_q16)], [max(0.0, y_q84 - y)]],
                fmt="none",
                ecolor=variant_color,
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                linestyle="none",
                alpha=0.30,
                zorder=1,
                label="_nolegend_",
            )
            ax.scatter(
                [x],
                [y],
                s=21.0,
                marker=VARIANT_MARKERS.get(variant, "o"),
                facecolors=_stage_marker_facecolor(stage, variant_color),
                edgecolors=_stage_marker_edgecolor(stage, variant_color),
                linewidths=0.75,
                alpha=0.86,
                zorder=2,
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
    variants = _ordered_variants_from_df(df)
    methods = [
        method
        for method in (method_label(stage, variant) for stage in DEFAULT_STAGES for variant in variants)
        if method in set(df["method_label"])
    ]
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
        _add_method_legend(fig, variants=variants)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _finite_magnification_rows(magnification_df: pd.DataFrame) -> pd.DataFrame:
    required = ["magnification_true", "magnification_q16", "magnification_q50", "magnification_q84"]
    if magnification_df.empty or any(column not in magnification_df for column in required):
        return pd.DataFrame()
    values = magnification_df[required].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    return magnification_df.loc[mask].copy()


def _magnification_recovery_output_base(output_path: Path, stage: str, primary_stage: str) -> Path:
    if str(stage) == str(primary_stage):
        return output_path / "mock_magnification_recovery"
    return output_path / f"mock_magnification_recovery_{stage_alias(stage)}"


def _image_residual_recovery_output_base(output_path: Path, stage: str, primary_stage: str) -> Path:
    if str(stage) == str(primary_stage):
        return output_path / "mock_image_residual_recovery"
    return output_path / f"mock_image_residual_recovery_{stage_alias(stage)}"


def _image_residual_distribution_output_base(output_path: Path, stage: str, primary_stage: str) -> Path:
    if str(stage) == str(primary_stage):
        return output_path / "mock_image_residual_distribution"
    return output_path / f"mock_image_residual_distribution_{stage_alias(stage)}"


def image_residual_seed_summary_rows(image_residual_df: pd.DataFrame) -> pd.DataFrame:
    if image_residual_df.empty:
        return pd.DataFrame()
    metric_columns = (
        "point_image_rms_arcsec",
        "arc_aware_image_rms_arcsec",
        "point_recovered_image_count",
        "arc_aware_recovered_image_count",
        "arc_supported_image_count",
        "total_image_count",
    )
    if not any(column in image_residual_df for column in metric_columns):
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("campaign", "run_name", "seed", "variant", "stage", "stage_label", "method_label", "result_path")
        if column in image_residual_df
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in image_residual_df.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        for column in metric_columns:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{column}_q16"] = np.nan
                row[f"{column}_q50"] = np.nan
                row[f"{column}_q84"] = np.nan
                row[f"{column}_std"] = np.nan
                continue
            row[f"{column}_q16"] = float(np.nanquantile(values, 0.16))
            row[f"{column}_q50"] = float(np.nanmedian(values))
            row[f"{column}_q84"] = float(np.nanquantile(values, 0.84))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _image_residual_distribution_range_max(residuals: np.ndarray) -> float:
    finite = np.asarray(residuals, dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    if finite.size == 0:
        return 1.0e-6
    robust_max = float(np.nanquantile(finite, IMAGE_RESIDUAL_DISTRIBUTION_RANGE_QUANTILE))
    padded = max(robust_max * IMAGE_RESIDUAL_DISTRIBUTION_RANGE_PADDING, 1.0e-6)
    step = IMAGE_RESIDUAL_DISTRIBUTION_RANGE_STEP_ARCSEC
    if not np.isfinite(step) or step <= 0.0:
        return padded
    return float(np.ceil(padded / step) * step)


def image_residual_distribution_summary_rows(
    image_residual_draws_df: pd.DataFrame,
    *,
    bin_count: int = 24,
) -> pd.DataFrame:
    required = {"variant", "draw_index", "arc_aware_image_residual_arcsec"}
    if image_residual_draws_df.empty or not required <= set(image_residual_draws_df.columns):
        return pd.DataFrame()
    df = image_residual_draws_df.copy()
    if "arc_aware_recovered" in df:
        recovered = df["arc_aware_recovered"].astype(bool).to_numpy(dtype=bool)
    else:
        recovered = np.ones(len(df), dtype=bool)
    if "exact_failed" in df:
        usable = ~df["exact_failed"].astype(bool).to_numpy(dtype=bool)
    else:
        usable = np.ones(len(df), dtype=bool)
    residual = pd.to_numeric(df["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
    finite_recovered = usable & recovered & np.isfinite(residual) & (residual >= 0.0)
    if not np.any(finite_recovered):
        return pd.DataFrame()
    max_residual = _image_residual_distribution_range_max(residual[finite_recovered])
    edges = np.linspace(0.0, max(max_residual, 1.0e-6), int(bin_count) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    group_columns = [
        column
        for column in ("campaign", "run_name", "seed", "variant", "stage", "stage_label", "method_label", "draw_index")
        if column in df
    ]
    sample_rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_dict = dict(zip(group_columns, key_values))
        group_residual = pd.to_numeric(group["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
        group_recovered = group["arc_aware_recovered"].astype(bool).to_numpy(dtype=bool) if "arc_aware_recovered" in group else np.ones(len(group), dtype=bool)
        group_usable = ~group["exact_failed"].astype(bool).to_numpy(dtype=bool) if "exact_failed" in group else np.ones(len(group), dtype=bool)
        denominator = int(np.count_nonzero(group_usable))
        if denominator <= 0:
            continue
        value_mask = group_usable & group_recovered & np.isfinite(group_residual) & (group_residual >= 0.0)
        counts, _ = np.histogram(group_residual[value_mask], bins=edges)
        cumulative_counts = np.cumsum(counts).astype(float)
        for index, center in enumerate(centers):
            sample_rows.append(
                {
                    **group_dict,
                    "bin_index": index,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "bin_center": float(center),
                    "hist_count": float(counts[index]),
                    "cumulative_fraction": float(cumulative_counts[index] / denominator),
                    "usable_image_count": denominator,
                }
            )
    sample_df = pd.DataFrame(sample_rows)
    if sample_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    summary_columns = [
        column
        for column in ("campaign", "run_name", "variant", "stage", "stage_label", "method_label", "bin_index")
        if column in sample_df
    ]
    for keys, group in sample_df.groupby(summary_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(summary_columns, key_values))
        row["bin_left"] = float(group["bin_left"].iloc[0])
        row["bin_right"] = float(group["bin_right"].iloc[0])
        row["bin_center"] = float(group["bin_center"].iloc[0])
        for column in ("hist_count", "cumulative_fraction"):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{column}_q16"] = np.nan
                row[f"{column}_q50"] = np.nan
                row[f"{column}_q84"] = np.nan
                continue
            q16, q50, q84 = np.nanquantile(values, [0.16, 0.50, 0.84])
            row[f"{column}_q16"] = float(q16)
            row[f"{column}_q50"] = float(q50)
            row[f"{column}_q84"] = float(q84)
        rows.append(row)
    return pd.DataFrame(rows)


def joint_sigma_rcut_recovery_rows(parameter_df: pd.DataFrame, *, component: str = "potfile") -> pd.DataFrame:
    if component not in JOINT_SIGMA_RCUT_SPECS:
        raise ValueError(f"component must be one of {sorted(JOINT_SIGMA_RCUT_SPECS)}.")
    spec = JOINT_SIGMA_RCUT_SPECS[component]
    sigma_parameter = str(spec["sigma_parameter"])
    rcut_parameter = str(spec["rcut_parameter"])
    required_parameters = (sigma_parameter, rcut_parameter)
    df = finite_recovery_rows(parameter_df)
    if df.empty:
        return pd.DataFrame()
    parameter_names = df["parameter"].astype(str)
    df = df.loc[parameter_names.isin(required_parameters)].copy()
    if df.empty:
        return pd.DataFrame()
    key_columns = [
        column
        for column in ("campaign", "run_name", "seed", "variant", "stage", "stage_label", "method_label", "result_path")
        if column in df
    ]
    duplicate_mask = df.duplicated([*key_columns, "parameter"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, [*key_columns, "parameter"]].to_dict("records")
        raise ValueError(f"Duplicate sigma/r_cut recovery rows found: {duplicates[:3]}")
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(key_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(key_columns, key_values))
        by_parameter = {str(record["parameter"]): record for record in group.to_dict("records")}
        missing = [name for name in required_parameters if name not in by_parameter]
        if missing:
            warnings.warn(
                f"Skipping incomplete {component} sigma/r_cut recovery group {row}: missing {missing}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        sigma = by_parameter[sigma_parameter]
        rcut = by_parameter[rcut_parameter]
        rows.append(
            {
                **row,
                "component": component,
                "component_label": str(spec["label"]),
                "sigma_parameter": sigma_parameter,
                "rcut_parameter": rcut_parameter,
                "sigma_truth": float(sigma["truth"]),
                "sigma_median": float(sigma["median"]),
                "sigma_q16": float(sigma["q16"]),
                "sigma_q84": float(sigma["q84"]),
                "rcut_truth": float(rcut["truth"]),
                "rcut_median": float(rcut["median"]),
                "rcut_q16": float(rcut["q16"]),
                "rcut_q84": float(rcut["q84"]),
            }
        )
    return pd.DataFrame(rows)


def _sigma_rcut_recovery_output_base(output_path: Path, stage: str, primary_stage: str, *, component: str = "potfile") -> Path:
    if component not in JOINT_SIGMA_RCUT_SPECS:
        raise ValueError(f"component must be one of {sorted(JOINT_SIGMA_RCUT_SPECS)}.")
    filename = str(JOINT_SIGMA_RCUT_SPECS[component]["filename"])
    if str(stage) == str(primary_stage):
        return output_path / f"mock_{filename}"
    return output_path / f"mock_{filename}_{stage_alias(stage)}"


def _add_covariance_ellipses(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    points = np.column_stack([x, y])
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 3:
        warnings.warn(
            f"Skipping sigma/r_cut covariance contours for {label}: need at least three finite points.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        warnings.warn(
            f"Skipping sigma/r_cut covariance contours for {label}: covariance is not finite.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0.0) or not np.isfinite(eigenvalues).all():
        warnings.warn(
            f"Skipping sigma/r_cut covariance contours for {label}: covariance is not full-rank.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(float(eigenvectors[1, 0]), float(eigenvectors[0, 0])))
    center = points.mean(axis=0)
    for level_index, level in enumerate(JOINT_SCALING_LEVELS):
        width, height = 2.0 * np.sqrt(eigenvalues * float(level))
        ellipse = Ellipse(
            xy=(float(center[0]), float(center[1])),
            width=float(width),
            height=float(height),
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=1.0 if level_index == 0 else 0.75,
            linestyle="-" if level_index == 0 else "--",
            alpha=0.55 if level_index == 0 else 0.35,
            zorder=1.2,
            label="_nolegend_",
        )
        ax.add_patch(ellipse)


def plot_joint_sigma_rcut_recovery(
    joint_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    if joint_df.empty:
        return tuple()
    required = ["sigma_truth", "sigma_median", "rcut_truth", "rcut_median"]
    values = joint_df[required].apply(pd.to_numeric, errors="coerce")
    plot_df = joint_df.loc[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)].copy()
    if plot_df.empty:
        return tuple()
    component_label = "Scaling"
    if "component_label" in plot_df:
        labels = plot_df["component_label"].dropna().astype(str).drop_duplicates().tolist()
        if len(labels) == 1:
            component_label = labels[0]
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, ax = plt.subplots(figsize=(6.2, 5.5))
        for (stage, variant), group in plot_df.groupby(["stage", "variant"], sort=False):
            stage = str(stage)
            variant = str(variant)
            variant_color = VARIANT_COLORS.get(variant, "0.35")
            x = group["sigma_median"].to_numpy(dtype=float)
            y = group["rcut_median"].to_numpy(dtype=float)
            _add_covariance_ellipses(
                ax,
                x,
                y,
                color=variant_color,
                label=method_label(stage, variant),
            )
            ax.scatter(
                x,
                y,
                s=28.0,
                marker=VARIANT_MARKERS.get(variant, "o"),
                facecolors=_stage_marker_facecolor(stage, variant_color),
                edgecolors=_stage_marker_edgecolor(stage, variant_color),
                linewidths=0.8,
                alpha=0.88,
                zorder=2,
                label=variant_label(variant),
            )
        truth_points = (
            plot_df[["sigma_truth", "rcut_truth"]]
            .drop_duplicates()
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if not truth_points.empty:
            ax.scatter(
                truth_points["sigma_truth"].to_numpy(dtype=float),
                truth_points["rcut_truth"].to_numpy(dtype=float),
                marker="*",
                s=95.0,
                facecolors="black",
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
                label="Truth",
            )
        limits = _axis_limits(
            plot_df["sigma_median"].to_numpy(dtype=float),
            plot_df["sigma_truth"].to_numpy(dtype=float),
        )
        if limits is not None:
            ax.set_xlim(limits)
        limits = _axis_limits(
            plot_df["rcut_median"].to_numpy(dtype=float),
            plot_df["rcut_truth"].to_numpy(dtype=float),
        )
        if limits is not None:
            ax.set_ylim(limits)
        ax.set_xlabel(rf"{component_label} $\sigma$ (fit)")
        ax.set_ylabel(rf"{component_label} $r_{{\rm cut}}$ (fit)")
        handles, labels = ax.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label == "_nolegend_" or label in unique:
                continue
            unique[label] = handle
        if unique:
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="best",
                frameon=False,
                fontsize=AGGREGATE_LEGEND_FONT_SIZE,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _absolute_magnification_interval(
    q16: np.ndarray,
    q50: np.ndarray,
    q84: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interval = np.vstack([np.abs(q16), np.abs(q50), np.abs(q84)])
    center = np.abs(q50)
    lower = np.nanmin(interval, axis=0)
    upper = np.nanmax(interval, axis=0)
    return lower, center, upper


def plot_aggregate_magnification_recovery(
    magnification_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    plot_df = _finite_magnification_rows(magnification_df)
    if plot_df.empty:
        return tuple()
    paths: list[Path] = []
    finite_values = np.concatenate(
        [
            np.abs(plot_df["magnification_true"].to_numpy(dtype=float)),
            np.abs(plot_df["magnification_q16"].to_numpy(dtype=float)),
            np.abs(plot_df["magnification_q50"].to_numpy(dtype=float)),
            np.abs(plot_df["magnification_q84"].to_numpy(dtype=float)),
        ]
    )
    finite_values = finite_values[np.isfinite(finite_values) & (finite_values > 0.0)]
    limits: tuple[float, float] | None = None
    if finite_values.size:
        log_values = np.log10(finite_values)
        lo = float(np.nanmin(log_values))
        hi = float(np.nanmax(log_values))
        pad = max(0.08 * (hi - lo), 0.08) if hi > lo else 0.08
        limits = (float(np.power(10.0, lo - pad)), float(np.power(10.0, hi + pad)))
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        labeled_variants: set[str] = set()
        for (stage, variant), group in plot_df.groupby(["stage", "variant"], sort=False):
            variant = str(variant)
            stage = str(stage)
            variant_color = VARIANT_COLORS.get(variant, "0.35")
            truth = np.abs(group["magnification_true"].to_numpy(dtype=float))
            q16 = group["magnification_q16"].to_numpy(dtype=float)
            q50 = group["magnification_q50"].to_numpy(dtype=float)
            q84 = group["magnification_q84"].to_numpy(dtype=float)
            abs_q16, abs_q50, abs_q84 = _absolute_magnification_interval(q16, q50, q84)
            valid = (truth > 0.0) & (abs_q16 > 0.0) & (abs_q50 > 0.0) & (abs_q84 > 0.0)
            if not np.any(valid):
                continue
            label = variant_label(variant) if variant not in labeled_variants else "_nolegend_"
            labeled_variants.add(variant)
            ax.errorbar(
                truth[valid],
                abs_q50[valid],
                yerr=[
                    np.maximum(0.0, abs_q50[valid] - abs_q16[valid]),
                    np.maximum(0.0, abs_q84[valid] - abs_q50[valid]),
                ],
                fmt="none",
                ecolor=variant_color,
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                linestyle="none",
                alpha=0.30,
                zorder=1,
                label="_nolegend_",
            )
            ax.scatter(
                truth[valid],
                abs_q50[valid],
                s=21.0,
                marker=VARIANT_MARKERS.get(variant, "o"),
                facecolors="white" if stage.startswith("stage1") else variant_color,
                edgecolors=variant_color if stage.startswith("stage1") else _darken_color(variant_color),
                linewidths=0.75,
                alpha=0.86,
                zorder=2,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if limits is not None:
            ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.75, zorder=0.5)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        ax.set_xlabel(r"Image $|\mu|$ (truth)")
        ax.set_ylabel(r"Image $|\mu|$ (fit)")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="best", frameon=False, fontsize=AGGREGATE_LEGEND_FONT_SIZE)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def plot_aggregate_image_residual_recovery(
    image_residual_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    summary_df = image_residual_seed_summary_rows(image_residual_df)
    if summary_df.empty:
        return tuple()
    metric = "arc_aware_image_rms_arcsec"
    q50_column = f"{metric}_q50"
    if q50_column not in summary_df:
        return tuple()
    variants = _ordered_variants_from_df(summary_df)
    x_positions = {variant: float(index) for index, variant in enumerate(variants)}
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        legend_handles: dict[str, Any] = {}
        for variant in variants:
            group = summary_df[summary_df["variant"].astype(str) == variant].copy()
            if group.empty:
                continue
            y = pd.to_numeric(group[q50_column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            y = y[finite]
            seed_offsets = np.linspace(-0.055, 0.055, y.size) if y.size > 1 else np.zeros(1)
            x = np.full(y.size, x_positions[variant], dtype=float) + seed_offsets
            color = VARIANT_COLORS.get(variant, "0.35")
            marker = VARIANT_MARKERS.get(variant, "o")
            ax.scatter(
                x,
                y,
                s=18.0,
                marker=marker,
                facecolors="white",
                edgecolors=color,
                linewidths=0.70,
                alpha=0.58,
                zorder=1,
                label="_nolegend_",
            )
            q16, q50, q84 = np.nanquantile(y, [0.16, 0.50, 0.84])
            center_x = np.asarray([x_positions[variant]], dtype=float)
            center_y = np.asarray([q50], dtype=float)
            ax.errorbar(
                center_x,
                center_y,
                yerr=[
                    np.asarray([max(0.0, q50 - q16)], dtype=float),
                    np.asarray([max(0.0, q84 - q50)], dtype=float),
                ],
                fmt="none",
                ecolor=color,
                elinewidth=1.15,
                capsize=3.0,
                capthick=1.15,
                alpha=0.88,
                zorder=2,
                label="_nolegend_",
            )
            handle = ax.scatter(
                center_x,
                center_y,
                s=56.0,
                marker=marker,
                facecolors=color,
                edgecolors=_darken_color(color),
                linewidths=0.85,
                alpha=0.95,
                zorder=3,
                label=variant_label(variant),
            )
            legend_handles.setdefault(variant_label(variant), handle)
        ax.set_xticks([x_positions[variant] for variant in variants])
        ax.set_xticklabels([variant_label(variant) for variant in variants], rotation=18, ha="right")
        ax.set_xlim(-0.5, max(0.5, len(variants) - 0.5))
        ax.set_xlabel("Fit variant")
        ax.set_ylabel("image RMS (arcsec)")
        if legend_handles:
            ax.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="best",
                frameon=False,
                fontsize=AGGREGATE_LEGEND_FONT_SIZE,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def plot_aggregate_image_residual_distribution(
    image_residual_draws_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    summary_df = image_residual_distribution_summary_rows(image_residual_draws_df)
    if summary_df.empty:
        return tuple()
    variants = _ordered_variants_from_df(summary_df)
    finite_right_edges = pd.to_numeric(summary_df["bin_right"], errors="coerce").to_numpy(dtype=float)
    finite_right_edges = finite_right_edges[np.isfinite(finite_right_edges)]
    x_max = float(np.nanmax(finite_right_edges)) if finite_right_edges.size else 1.0e-6
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        for variant in variants:
            group = summary_df[summary_df["variant"].astype(str) == variant].sort_values("bin_center")
            if group.empty:
                continue
            x = group["bin_center"].to_numpy(dtype=float)
            color = VARIANT_COLORS.get(variant, "0.35")
            hist_q16 = group["hist_count_q16"].to_numpy(dtype=float)
            hist_q50 = group["hist_count_q50"].to_numpy(dtype=float)
            hist_q84 = group["hist_count_q84"].to_numpy(dtype=float)
            finite_hist = np.isfinite(x) & np.isfinite(hist_q16) & np.isfinite(hist_q50) & np.isfinite(hist_q84)
            if np.any(finite_hist):
                ax.fill_between(
                    x[finite_hist],
                    hist_q16[finite_hist],
                    hist_q84[finite_hist],
                    color=color,
                    alpha=0.18,
                    step="mid",
                    linewidth=0.0,
                )
                ax.step(
                    x[finite_hist],
                    hist_q50[finite_hist],
                    where="mid",
                    color=color,
                    linewidth=2.0,
                    label=variant_label(variant),
                )
        ax.set_ylabel("N images")
        ax.set_xlabel("image residual (arcsec)")
        ax.set_xlim(0.0, x_max)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="best", frameon=False, fontsize=AGGREGATE_LEGEND_FONT_SIZE)
        fig.tight_layout()
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


def _dpie_truth_scaling_values(truth: dict[str, Any]) -> tuple[float, float, float, float] | None:
    parameter_truth = truth.get("parameter_truth", {})
    if not isinstance(parameter_truth, dict):
        return None
    try:
        sigma_ref = float(parameter_truth["potfile.sigma"])
        cut_ref_kpc = float(parameter_truth["potfile.cutkpc"])
        alpha_sigma = float(parameter_truth["potfile.alpha_sigma"])
        beta_radius = float(parameter_truth["potfile.beta_radius"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (sigma_ref, cut_ref_kpc, alpha_sigma, beta_radius)
    if not all(np.isfinite(values)) or sigma_ref <= 0.0 or cut_ref_kpc <= 0.0:
        return None
    return values


def dpie_truth_subhalo_masses_from_scaling(truth: dict[str, Any]) -> np.ndarray:
    """Recompute physical dPIE member masses for truth payloads that predate
    `subhalo_dpie_mass_msun` (mocks generated before the label/physical mass split)."""
    scaling = _dpie_truth_scaling_values(truth)
    if scaling is None:
        return np.empty((0,), dtype=float)
    sigma_ref, cut_ref_kpc, alpha_sigma, beta_radius = scaling
    subhalos = truth.get("subhalos", [])
    masses: list[float] = []
    for record in subhalos if isinstance(subhalos, list) else []:
        if not isinstance(record, dict):
            continue
        try:
            luminosity = float(record.get("luminosity_ratio"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(luminosity) and luminosity > 0.0):
            continue
        v_disp = sigma_ref * luminosity**alpha_sigma * math.exp(float(record.get("sigma_log_offset", 0.0)))
        cut_kpc = cut_ref_kpc * luminosity**beta_radius * math.exp(float(record.get("cut_log_offset", 0.0)))
        masses.append(float(10.0 ** log10_dpie_mass_msun(v_disp, cut_kpc)))
    return _positive_finite_values(masses)


def dpie_mass_anchor_from_truth(truth: dict[str, Any]) -> float | None:
    """Physical dPIE mass of an L=1, no-scatter member — the label->physical anchor."""
    scaling = _dpie_truth_scaling_values(truth)
    if scaling is None:
        return None
    sigma_ref, cut_ref_kpc, _alpha_sigma, _beta_radius = scaling
    return float(10.0 ** log10_dpie_mass_msun(sigma_ref, cut_ref_kpc))


def _truth_subhalo_masses(payload: dict[str, Any]) -> np.ndarray:
    truth = payload.get("mock_cluster", {}).get("truth", {})
    if not isinstance(truth, dict):
        return np.empty((0,), dtype=float)
    subhalos = truth.get("subhalos", [])
    masses = [
        record.get("subhalo_dpie_mass_msun")
        for record in (subhalos if isinstance(subhalos, list) else [])
        if isinstance(record, dict)
    ]
    values = _positive_finite_values(masses)
    if values.size:
        return values
    recomputed = dpie_truth_subhalo_masses_from_scaling(truth)
    if recomputed.size:
        warnings.warn(
            "truth.json lacks subhalo_dpie_mass_msun (mock generated before the "
            "label/physical mass split); recomputed physical dPIE masses from the "
            "scaling-relation truth. Do NOT compare against subhalo_mass_msun — that "
            "is the Schechter abundance label. NOTE: this campaign's stored per-run "
            "recovered-mass tables (scaling_relation_summary.csv) also predate the "
            "(3/2)*pi*sigma^2*rcut/G total-mass convention and are a factor 1.5 low; "
            "the SHMF ratio panel will read ~2/3 until the per-run plots are "
            "regenerated.",
            RuntimeWarning,
            stacklevel=2,
        )
    return recomputed


def _recovered_stage_subhalo_log_masses(payload: dict[str, Any], stage: str) -> np.ndarray:
    manifest = _stage_manifest(payload, stage)
    if manifest is None:
        return np.empty((0,), dtype=float)
    records = _table_records_from_manifest(manifest, "scaling_relation_summary.csv")
    values: list[Any] = []
    for record in records:
        # Promoted members carry individually fitted masses (their delta offsets model
        # exactly the per-member scatter the truth includes); fall back to the pure
        # scaling-relation mass for members without free columns.
        value = None
        for column in (
            "free_log10_mass_msun_best",
            "free_log10_mass_msun_median",
            "scaling_log10_mass_msun_best",
            "scaling_log10_mass_msun_median",
        ):
            candidate = pd.to_numeric(pd.Series([record.get(column)]), errors="coerce").iloc[0]
            if np.isfinite(candidate):
                value = float(candidate)
                break
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
    stage: str = "stage2_free_source_forward_fit",
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
            recovered_log_mass = _recovered_stage_subhalo_log_masses(payload, stage)
            if truth_mass.size == 0 or recovered_log_mass.size == 0:
                message = f"Missing truth or recovered SHMF masses for {stage} in {path}"
                if strict:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            if not selection:
                selection = _truth_subhalo_selection(payload)
                if selection and "dpie_mass_anchor_msun" not in selection:
                    truth = payload.get("mock_cluster", {}).get("truth", {})
                    anchor = dpie_mass_anchor_from_truth(truth if isinstance(truth, dict) else {})
                    if anchor is not None:
                        selection = {**selection, "dpie_mass_anchor_msun": anchor}
            rows.append(
                {
                    "campaign": root.name,
                    "run_name": run_name,
                    "seed": seed,
                    "variant": variant,
                    "stage": str(stage),
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
                    "stage": row.get("stage", "stage2_free_source_forward_fit"),
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
                    "stage": str(first.get("stage", "stage2_free_source_forward_fit")),
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

    for _stage, stage_frame in seed_df.groupby("stage", sort=False):
        truth_frame = stage_frame.drop_duplicates(["seed", "bin_index"])
        add_summary_rows(truth_frame, series="truth", variant="truth", value_column="truth_density")
        for variant, variant_frame in stage_frame.groupby("variant", sort=False):
            add_summary_rows(variant_frame, series="recovered", variant=str(variant), value_column="recovered_density")
            add_summary_rows(variant_frame, series="ratio", variant=str(variant), value_column="recovered_to_truth")
    return seed_df, pd.DataFrame(summary_rows)


def _schechter_shmf_curve(selection: dict[str, Any], x_values: np.ndarray, normalization: float) -> np.ndarray | None:
    # The empirical histograms are in physical dPIE mass, so the Schechter curve must
    # be anchored at the dPIE mass of an L=1 member, not the abundance-label mass_ref
    # (both scale as L^exponent, so only the anchor changes).
    try:
        schechter_alpha = float(selection.get("schechter_alpha"))
        mass_anchor = float(selection.get("dpie_mass_anchor_msun"))
        exponent = float(selection.get("mass_luminosity_exponent"))
    except (TypeError, ValueError):
        return None
    if not (
        np.isfinite(schechter_alpha)
        and np.isfinite(mass_anchor)
        and mass_anchor > 0.0
        and np.isfinite(exponent)
        and exponent > 0.0
        and np.isfinite(normalization)
        and normalization > 0.0
    ):
        return None
    mass_grid = np.power(10.0, x_values)
    luminosity_grid = np.power(mass_grid / mass_anchor, 1.0 / exponent)
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
    y_ceiling = float(np.nanmax(empirical_y) * 1.15) if empirical_y.size else np.nan
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
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y16) & np.isfinite(y84)
            top_ax.fill_between(x[mask], y16[mask], y84[mask], color="0.7", alpha=0.35, step="mid")
            top_ax.step(x[mask], y[mask], where="mid", color="0.25", linewidth=2.0, label="Truth")
            bin_widths = (truth["log_mass_right"] - truth["log_mass_left"]).to_numpy(dtype=float)
            schechter_norm = float(np.nansum(np.where(mask, y * bin_widths, 0.0)))
            curve = _schechter_shmf_curve(selection, x, schechter_norm)
            if curve is not None:
                curve_mask = np.isfinite(curve)
                if np.any(curve_mask):
                    top_ax.plot(x[curve_mask], curve[curve_mask], color="0.1", linestyle="--", linewidth=1.3, label="Schechter")
        for variant in _ordered_variants_from_df(
            summary_df[summary_df["series"].astype(str).isin(["recovered", "ratio"])]
        ):
            recovered = summary_df[(summary_df["series"] == "recovered") & (summary_df["variant"] == variant)].sort_values("bin_index")
            if recovered.empty:
                continue
            x = recovered["log_mass_center"].to_numpy(dtype=float)
            y = recovered["value_median"].to_numpy(dtype=float)
            y16 = recovered["value_q16"].to_numpy(dtype=float)
            y84 = recovered["value_q84"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y16) & np.isfinite(y84)
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
        if np.isfinite(y_ceiling) and y_ceiling > 0.0:
            top_ax.set_ylim(0.0, y_ceiling)
        top_ax.set_ylabel(r"$dN/d\log M$")
        top_ax.legend(loc="best", frameon=False, fontsize=AGGREGATE_LEGEND_FONT_SIZE)
        ratio_ax.axhline(1.0, color="0.25", linewidth=1.0)
        ratio_ax.set_xlabel(r"$\log(M/M_\odot)$")
        ratio_ax.set_ylabel("Recovered / Truth")
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _subhalo_shmf_output_base(output_path: Path, stage: str, primary_stage: str) -> Path:
    if str(stage) == str(primary_stage):
        return output_path / "mock_subhalo_recovery_shmf"
    return output_path / f"mock_subhalo_recovery_{stage_alias(stage)}_shmf"


def write_subhalo_shmf_outputs(
    campaign_root: str | Path,
    output_path: Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    strict: bool = False,
) -> tuple[Path | None, Path | None, tuple[Path, ...]]:
    primary_stage = _paper_recovery_stage(stages)
    selection: dict[str, Any] = {}
    seed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    plot_paths: list[Path] = []
    for stage in tuple(str(value) for value in stages):
        shmf_inputs, stage_selection = collect_subhalo_shmf_inputs(
            campaign_root,
            run_name=run_name,
            variants=variants,
            stage=stage,
            strict=strict,
        )
        if not selection and stage_selection:
            selection = stage_selection
        stage_seed_df, stage_summary_df = build_subhalo_shmf_tables(shmf_inputs)
        if stage_seed_df.empty or stage_summary_df.empty:
            continue
        seed_frames.append(stage_seed_df)
        summary_frames.append(stage_summary_df)
        plot_paths.extend(
            plot_subhalo_shmf_summary(
                stage_summary_df,
                _subhalo_shmf_output_base(output_path, stage, primary_stage),
                formats,
                selection=selection,
            )
        )
    seed_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if seed_df.empty or summary_df.empty:
        return None, None, tuple()
    seed_table = output_path / "subhalo_recovery_shmf_seed_table.csv"
    summary_table = output_path / "subhalo_recovery_shmf_summary_table.csv"
    seed_df.to_csv(seed_table, index=False)
    summary_df.to_csv(summary_table, index=False)
    return seed_table, summary_table, tuple(plot_paths)


# --------------------------------------------------------------------------------------
# Enclosed projected mass profile M2D(<R) recovery (total + component decomposition)
# --------------------------------------------------------------------------------------

M2D_APERTURE_RADII_KPC = tuple(float(v) for v in np.geomspace(3.0, 500.0, 36))
M2D_APERTURE_POSTERIOR_DRAWS = 24
M2D_APERTURE_AZIMUTH_SAMPLES = 64
M2D_APERTURE_INTEGRATION_RADII = 240
M2D_GHOST_CHAIN_LOGPROB_DROP = 20.0
M2D_COMPONENT_GROUPS = ("halo", "bcg", "members")
M2D_COMPONENT_COLORS = {"halo": "#4477AA", "bcg": "#CC6677", "members": "#228833"}
M2D_VARIANT_LINESTYLES = ("-", "--", "-.", ":")


def _dpie_sigma0_msun_per_kpc2(v_disp: np.ndarray, core_kpc: np.ndarray, cut_kpc: np.ndarray) -> np.ndarray:
    """Central surface density of the dPIE in the lenstool sigma convention
    (Eliasdottir+07: Sigma0 = (3/4) sigma^2 (s^2 - a^2) / (G a s^2))."""
    a = np.asarray(core_kpc, dtype=float)
    s = np.asarray(cut_kpc, dtype=float)
    v = np.asarray(v_disp, dtype=float)
    return 0.75 * np.square(v) * (np.square(s) - np.square(a)) / (
        DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN * a * np.square(s)
    )


def dpie_m2d_enclosed_msun(radii_kpc: np.ndarray, v_disp: float, core_kpc: float, cut_kpc: float) -> np.ndarray:
    """Analytic projected mass inside R for a centered circular dPIE [Msun].

    M2D(<R) = 2 pi Sigma0 (a s/(s-a)) [sqrt(a^2+R^2) - a - sqrt(s^2+R^2) + s];
    the R -> infinity limit equals the total mass (3/2) pi sigma^2 s (1-a^2/s^2)/G
    used by `lenscluster.utils.log10_dpie_mass_msun`.
    """
    radii = np.asarray(radii_kpc, dtype=float)
    a, s, v = float(core_kpc), float(cut_kpc), float(v_disp)
    sigma0 = float(_dpie_sigma0_msun_per_kpc2(v, a, s))
    return (
        2.0
        * np.pi
        * sigma0
        * (a * s / (s - a))
        * (np.sqrt(a * a + radii * radii) - a - np.sqrt(s * s + radii * radii) + s)
    )


def _dpie_surface_density_msun_per_kpc2(
    distance_kpc: np.ndarray,
    v_disp: np.ndarray,
    core_kpc: np.ndarray,
    cut_kpc: np.ndarray,
) -> np.ndarray:
    """Circularized dPIE surface density at `distance_kpc` from each component center.

    Broadcasts a trailing component axis: distance (..., C) with v/a/s of shape (C,).
    """
    a = np.asarray(core_kpc, dtype=float)
    s = np.asarray(cut_kpc, dtype=float)
    sigma0 = _dpie_sigma0_msun_per_kpc2(v_disp, a, s)
    d = np.asarray(distance_kpc, dtype=float)
    return sigma0 * (a * s / (s - a)) * (
        1.0 / np.sqrt(np.square(a) + np.square(d)) - 1.0 / np.sqrt(np.square(s) + np.square(d))
    )


def aperture_m2d_profiles_msun(
    radii_kpc: np.ndarray,
    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    integration_radii: int = M2D_APERTURE_INTEGRATION_RADII,
    azimuth_samples: int = M2D_APERTURE_AZIMUTH_SAMPLES,
) -> dict[str, np.ndarray]:
    """M2D(<R) about the cluster center (0,0) per component group, numerically.

    Each group maps to (v_disp, core_kpc, cut_kpc, x_kpc, y_kpc) component arrays.
    Off-center components are handled by azimuthally averaging the circularized
    surface density on a polar grid and integrating 2 pi r <Sigma>(r); ellipticity is
    neglected (aperture masses are azimuthal integrals and our mock ellipticities are
    mild). The 'total' key is the sum of the groups.
    """
    radii = np.asarray(radii_kpc, dtype=float)
    r_max = float(radii.max())
    r_grid = np.geomspace(max(1.0e-2, radii.min() / 20.0), r_max, int(integration_radii))
    phi = np.linspace(0.0, 2.0 * np.pi, int(azimuth_samples), endpoint=False)
    px = r_grid[:, None] * np.cos(phi)[None, :]
    py = r_grid[:, None] * np.sin(phi)[None, :]
    profiles: dict[str, np.ndarray] = {}
    total_mean_sigma = np.zeros_like(r_grid)
    for group, (v_disp, core_kpc, cut_kpc, x_kpc, y_kpc) in groups.items():
        v = np.asarray(v_disp, dtype=float)
        if v.size == 0:
            profiles[group] = np.zeros_like(radii)
            continue
        distance = np.sqrt(
            np.square(px[:, :, None] - np.asarray(x_kpc, dtype=float))
            + np.square(py[:, :, None] - np.asarray(y_kpc, dtype=float))
        )
        sigma = _dpie_surface_density_msun_per_kpc2(distance, v, core_kpc, cut_kpc).sum(axis=-1)
        mean_sigma = sigma.mean(axis=1)
        total_mean_sigma += mean_sigma
        profiles[group] = _cumulative_aperture_mass(radii, r_grid, mean_sigma)
    profiles["total"] = _cumulative_aperture_mass(radii, r_grid, total_mean_sigma)
    return profiles


def _cumulative_aperture_mass(radii_kpc: np.ndarray, r_grid: np.ndarray, mean_sigma: np.ndarray) -> np.ndarray:
    integrand = 2.0 * np.pi * r_grid * mean_sigma
    inner_disk = np.pi * r_grid[0] ** 2 * mean_sigma[0]
    cumulative = inner_disk + np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_grid))]
    )
    return np.interp(np.asarray(radii_kpc, dtype=float), r_grid, cumulative)


def _m2d_truth_groups(truth: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    kpc = float(truth["kpc_per_arcsec"])
    config = truth.get("config", {})

    def _large(entry: dict[str, Any]) -> tuple[np.ndarray, ...]:
        return (
            np.asarray([float(entry["v_disp"])]),
            np.asarray([float(entry["core_radius_arcsec"]) * kpc]),
            np.asarray([float(entry["cut_radius_arcsec"]) * kpc]),
            np.asarray([float(entry["x_centre"]) * kpc]),
            np.asarray([float(entry["y_centre"]) * kpc]),
        )

    members = truth.get("subhalo_components", [])
    return {
        "halo": _large(config["halo"]),
        "bcg": _large(config["bcg"]),
        "members": (
            np.asarray([float(m["v_disp"]) for m in members]),
            np.asarray([float(m["core_radius_arcsec"]) * kpc for m in members]),
            np.asarray([float(m["cut_radius_arcsec"]) * kpc for m in members]),
            np.asarray([float(m["x_centre"]) * kpc for m in members]),
            np.asarray([float(m["y_centre"]) * kpc for m in members]),
        ),
    }


def _m2d_model_groups(
    theta_by_name: dict[str, float],
    member_catalog: list[dict[str, Any]],
    kpc_per_arcsec: float,
    *,
    sigma_offset_width: float,
    mass_offset_width: float,
) -> dict[str, tuple[np.ndarray, ...]]:
    """Physical dPIE components implied by one posterior draw.

    Mirrors the packed-state mapping: members follow the scaling relation
    (beta = 1 + gamma_ml - 2 alpha_sigma) with promoted members applying their
    delta-unit offsets (radius delta = mass - 2 sigma, as in the sampler).
    """
    kpc = float(kpc_per_arcsec)

    def _large(prefix: str) -> tuple[np.ndarray, ...]:
        return (
            np.asarray([theta_by_name[f"{prefix}.v_disp"]]),
            np.asarray([theta_by_name[f"{prefix}.core_radius_kpc"]]),
            np.asarray([theta_by_name[f"{prefix}.cut_radius_kpc"]]),
            np.asarray([theta_by_name[f"{prefix}.x_centre"] * kpc]),
            np.asarray([theta_by_name[f"{prefix}.y_centre"] * kpc]),
        )

    sigma_ref = float(theta_by_name["potfile.sigma"])
    cut_ref = float(theta_by_name["potfile.cutkpc"])
    core_ref = float(theta_by_name["potfile.corekpc"])
    alpha = float(theta_by_name["potfile.alpha_sigma"])
    gamma = float(theta_by_name["potfile.gamma_ml"])
    beta = 1.0 + gamma - 2.0 * alpha
    v_disp, core, cut, xs, ys = [], [], [], [], []
    for member in member_catalog:
        member_id = str(member["id"])
        lum = float(member["luminosity_ratio"])
        sigma_delta = theta_by_name.get(f"potfile.{member_id}.independent_free_log_sigma_delta_unit")
        mass_delta = theta_by_name.get(f"potfile.{member_id}.independent_free_log_mass_delta_unit")
        log_sigma = float(sigma_offset_width) * float(sigma_delta) if sigma_delta is not None else 0.0
        log_mass = float(mass_offset_width) * float(mass_delta) if mass_delta is not None else 0.0
        radius_factor = math.exp(log_mass - 2.0 * log_sigma)
        v_disp.append(sigma_ref * lum**alpha * math.exp(log_sigma))
        core.append(core_ref * lum**beta * radius_factor)
        cut.append(cut_ref * lum**beta * radius_factor)
        xs.append(float(member["x_arcsec"]) * kpc)
        ys.append(float(member["y_arcsec"]) * kpc)
    return {
        "halo": _large("halo"),
        "bcg": _large("bcg"),
        "members": tuple(np.asarray(values, dtype=float) for values in (v_disp, core, cut, xs, ys)),
    }


def _m2d_posterior_draws(seed_dir: Path, variant: str, stage: str, max_draws: int) -> tuple[np.ndarray, list[str], dict[str, Any]] | None:
    bundle_path = seed_dir / variant / "solver" / stage / "artifacts" / "plot_bundle.h5"
    results_path = _combined_recovery_path(seed_dir, variant)
    if not bundle_path.exists() or not results_path.exists():
        return None
    payload = _load_json(results_path)
    stage_recovery = payload.get("validation", {}).get("recovery", {}).get(stage_alias(stage), {})
    names = stage_recovery.get("parameter_names")
    if not names:
        return None
    with h5py.File(bundle_path, "r") as bundle:
        grouped = bundle["posterior/grouped_samples"][...]
        grouped_log_prob = bundle["posterior/grouped_log_prob"][...]
        runtime_args = json.loads(bundle["runtime_args_json"][()])
    chain_log_prob = grouped_log_prob.mean(axis=1)
    keep = chain_log_prob >= chain_log_prob.max() - M2D_GHOST_CHAIN_LOGPROB_DROP
    samples = grouped[keep].reshape(-1, grouped.shape[-1])
    if samples.shape[0] == 0:
        return None
    indices = np.unique(np.linspace(0, samples.shape[0] - 1, int(max_draws)).astype(int))
    return samples[indices], list(names), runtime_args


def collect_m2d_aperture_inputs(
    campaign_root: str | Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    stage: str = "stage2_free_source_forward_fit",
    strict: bool = False,
    radii_kpc: tuple[float, ...] = M2D_APERTURE_RADII_KPC,
    max_draws: int = M2D_APERTURE_POSTERIOR_DRAWS,
) -> pd.DataFrame:
    """Per-run M2D(<R) posterior quantiles and truth per component group."""
    root = Path(campaign_root)
    radii = np.asarray(radii_kpc, dtype=float)
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(path for path in (root / run_name).glob("seed_*") if path.is_dir()):
        seed = _seed_from_dir(seed_dir)
        truth_path = seed_dir / "mock" / "truth.json"
        if not truth_path.exists():
            message = f"Missing truth.json for M2D aperture recovery: {truth_path}"
            if strict:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue
        truth = _load_json(truth_path)
        kpc = float(truth["kpc_per_arcsec"])
        truth_profiles = aperture_m2d_profiles_msun(radii, _m2d_truth_groups(truth))
        member_catalog = truth.get("subhalos", [])
        for variant in variants:
            loaded = _m2d_posterior_draws(seed_dir, variant, stage, max_draws)
            if loaded is None:
                message = f"Missing posterior bundle for M2D aperture recovery: {seed_dir.name}/{variant}/{stage}"
                if strict:
                    raise FileNotFoundError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            draws, names, runtime_args = loaded
            sigma_width = float(runtime_args.get("independent_scaling_free_log_sigma_offset_width", 0.0))
            mass_width = float(runtime_args.get("independent_scaling_free_log_mass_offset_width", 0.0))
            per_draw: dict[str, list[np.ndarray]] = {group: [] for group in (*M2D_COMPONENT_GROUPS, "total")}
            for draw in draws:
                theta_by_name = dict(zip(names, (float(value) for value in draw)))
                groups = _m2d_model_groups(
                    theta_by_name,
                    member_catalog,
                    kpc,
                    sigma_offset_width=sigma_width,
                    mass_offset_width=mass_width,
                )
                profiles = aperture_m2d_profiles_msun(radii, groups)
                for group, profile in profiles.items():
                    per_draw[group].append(profile)
            for group, profile_list in per_draw.items():
                stacked = np.stack(profile_list, axis=0)
                q16, q50, q84 = np.percentile(stacked, [16.0, 50.0, 84.0], axis=0)
                truth_profile = truth_profiles[group]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(truth_profile > 0.0, stacked / truth_profile, np.nan)
                r16, r50, r84 = np.nanpercentile(ratio, [16.0, 50.0, 84.0], axis=0)
                for k, radius in enumerate(radii):
                    rows.append(
                        {
                            "campaign": root.name,
                            "run_name": run_name,
                            "seed": seed,
                            "variant": variant,
                            "stage": str(stage),
                            "component": group,
                            "radius_kpc": float(radius),
                            "truth_m2d_msun": float(truth_profile[k]),
                            "model_q16": float(q16[k]),
                            "model_q50": float(q50[k]),
                            "model_q84": float(q84[k]),
                            "ratio_q16": float(r16[k]),
                            "ratio_q50": float(r50[k]),
                            "ratio_q84": float(r84[k]),
                        }
                    )
    return pd.DataFrame(rows)


def build_m2d_aperture_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run M2D quantiles across seeds.

    Bands: `ratio_seed_q16/q84` = seed-to-seed spread of per-run medians (accuracy);
    `ratio_post_q16/q84` = median per-run posterior interval (reported precision).
    """
    if seed_df.empty:
        return pd.DataFrame()
    grouped = seed_df.groupby(["stage", "variant", "component", "radius_kpc"])
    summary = grouped.agg(
        n_runs=("seed", "size"),
        truth_median=("truth_m2d_msun", "median"),
        model_median=("model_q50", "median"),
        model_q16=("model_q16", "median"),
        model_q84=("model_q84", "median"),
        ratio_median=("ratio_q50", "median"),
        ratio_seed_q16=("ratio_q50", lambda s: np.nanquantile(s, 0.16)),
        ratio_seed_q84=("ratio_q50", lambda s: np.nanquantile(s, 0.84)),
        ratio_post_q16=("ratio_q16", "median"),
        ratio_post_q84=("ratio_q84", "median"),
    ).reset_index()
    return summary


def _m2d_two_panel_axes() -> tuple[Any, Any, Any]:
    fig, (top_ax, resid_ax) = plt.subplots(
        2,
        1,
        figsize=(6.3, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    return fig, top_ax, resid_ax


def _m2d_finish_axes(top_ax: Any, resid_ax: Any, resid_values: np.ndarray) -> None:
    top_ax.set_xscale("log")
    top_ax.set_yscale("log")
    top_ax.set_ylabel(r"$M_{\rm 2D}(<R)$ ($M_\odot$)")
    top_ax.grid(True, alpha=0.25)
    top_ax.legend(loc="lower right", frameon=False, fontsize=AGGREGATE_LEGEND_FONT_SIZE)
    resid_ax.axhline(0.0, color="0.2", linestyle="--", linewidth=1.0)
    finite = resid_values[np.isfinite(resid_values)]
    limit = 0.1 if finite.size == 0 else float(min(0.5, max(0.1, 1.2 * np.max(np.abs(finite)))))
    resid_ax.set_ylim(-limit, limit)
    resid_ax.set_xlabel(r"$R$ (kpc)")
    resid_ax.set_ylabel(r"$M_{\rm 2D}/M_{\rm 2D}^{\rm true} - 1$")
    resid_ax.grid(True, alpha=0.25)


def plot_m2d_aperture_recovery(
    summary_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    """Total enclosed-mass profile: truth + per-variant posterior, ratio residual below."""
    totals = summary_df[summary_df["component"] == "total"]
    if totals.empty:
        return tuple()
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, top_ax, resid_ax = _m2d_two_panel_axes()
        resid_collect: list[np.ndarray] = []
        truth_drawn = False
        for variant in _ordered_variants_from_df(totals):
            selection = totals[totals["variant"] == variant].sort_values("radius_kpc")
            if selection.empty:
                continue
            radius = selection["radius_kpc"].to_numpy(dtype=float)
            color = VARIANT_COLORS.get(variant, "0.35")
            if not truth_drawn:
                top_ax.plot(
                    radius,
                    selection["truth_median"].to_numpy(dtype=float),
                    color="0.2",
                    linestyle="--",
                    linewidth=1.4,
                    label="truth",
                )
                truth_drawn = True
            top_ax.fill_between(
                radius,
                selection["model_q16"].to_numpy(dtype=float),
                selection["model_q84"].to_numpy(dtype=float),
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            top_ax.plot(
                radius,
                selection["model_median"].to_numpy(dtype=float),
                color=color,
                linewidth=1.8,
                label=variant_label(variant),
            )
            ratio_median = selection["ratio_median"].to_numpy(dtype=float)
            post_q16 = selection["ratio_post_q16"].to_numpy(dtype=float)
            post_q84 = selection["ratio_post_q84"].to_numpy(dtype=float)
            seed_q16 = selection["ratio_seed_q16"].to_numpy(dtype=float)
            seed_q84 = selection["ratio_seed_q84"].to_numpy(dtype=float)
            resid_ax.fill_between(radius, post_q16 - 1.0, post_q84 - 1.0, color=color, alpha=0.12, linewidth=0.0)
            resid_ax.fill_between(radius, seed_q16 - 1.0, seed_q84 - 1.0, color=color, alpha=0.28, linewidth=0.0)
            resid_ax.plot(radius, ratio_median - 1.0, color=color, linewidth=1.5)
            resid_collect.extend([post_q16 - 1.0, post_q84 - 1.0, seed_q16 - 1.0, seed_q84 - 1.0])
        _m2d_finish_axes(top_ax, resid_ax, np.concatenate(resid_collect) if resid_collect else np.asarray([]))
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def plot_m2d_aperture_components(
    summary_df: pd.DataFrame,
    output_base: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    """Component decomposition (halo/BCG/members): same two-panel layout, components
    as colors and variants as line styles."""
    components = summary_df[summary_df["component"].isin(M2D_COMPONENT_GROUPS)]
    if components.empty:
        return tuple()
    variants = _ordered_variants_from_df(components)
    paths: list[Path] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig, top_ax, resid_ax = _m2d_two_panel_axes()
        resid_collect: list[np.ndarray] = []
        for group in M2D_COMPONENT_GROUPS:
            color = M2D_COMPONENT_COLORS.get(group, "0.35")
            truth_drawn = False
            for variant_index, variant in enumerate(variants):
                selection = components[
                    (components["component"] == group) & (components["variant"] == variant)
                ].sort_values("radius_kpc")
                if selection.empty:
                    continue
                radius = selection["radius_kpc"].to_numpy(dtype=float)
                if not truth_drawn:
                    top_ax.plot(
                        radius,
                        selection["truth_median"].to_numpy(dtype=float),
                        color=color,
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.9,
                    )
                    truth_drawn = True
                linestyle = M2D_VARIANT_LINESTYLES[variant_index % len(M2D_VARIANT_LINESTYLES)]
                label = group if variant_index == 0 else None
                top_ax.plot(
                    radius,
                    selection["model_median"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.6,
                    label=label,
                )
                ratio_median = selection["ratio_median"].to_numpy(dtype=float)
                seed_q16 = selection["ratio_seed_q16"].to_numpy(dtype=float)
                seed_q84 = selection["ratio_seed_q84"].to_numpy(dtype=float)
                if variant_index == 0:
                    resid_ax.fill_between(radius, seed_q16 - 1.0, seed_q84 - 1.0, color=color, alpha=0.18, linewidth=0.0)
                resid_ax.plot(radius, ratio_median - 1.0, color=color, linestyle=linestyle, linewidth=1.4)
                resid_collect.extend([seed_q16 - 1.0, seed_q84 - 1.0, ratio_median - 1.0])
        for variant_index, variant in enumerate(variants):
            top_ax.plot(
                [],
                [],
                color="0.3",
                linestyle=M2D_VARIANT_LINESTYLES[variant_index % len(M2D_VARIANT_LINESTYLES)],
                linewidth=1.6,
                label=variant_label(variant),
            )
        top_ax.plot([], [], color="0.3", linestyle=":", linewidth=1.2, label="truth")
        _m2d_finish_axes(top_ax, resid_ax, np.concatenate(resid_collect) if resid_collect else np.asarray([]))
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return tuple(paths)


def _m2d_output_base(output_path: Path, stage: str, primary_stage: str, *, components: bool) -> Path:
    suffix = "_components" if components else ""
    if str(stage) == str(primary_stage):
        return output_path / f"mock_m2d_aperture_recovery{suffix}"
    return output_path / f"mock_m2d_aperture_recovery_{stage_alias(stage)}{suffix}"


def write_m2d_aperture_outputs(
    campaign_root: str | Path,
    output_path: Path,
    *,
    run_name: str = DEFAULT_RUN_NAME,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    strict: bool = False,
) -> tuple[Path | None, tuple[Path, ...]]:
    primary_stage = _paper_recovery_stage(stages)
    seed_frames: list[pd.DataFrame] = []
    plot_paths: list[Path] = []
    for stage in tuple(str(value) for value in stages):
        seed_df = collect_m2d_aperture_inputs(
            campaign_root,
            run_name=run_name,
            variants=variants,
            stage=stage,
            strict=strict,
        )
        if seed_df.empty:
            continue
        seed_frames.append(seed_df)
        summary_df = build_m2d_aperture_summary(seed_df)
        plot_paths.extend(
            plot_m2d_aperture_recovery(
                summary_df,
                _m2d_output_base(output_path, stage, primary_stage, components=False),
                formats,
            )
        )
        plot_paths.extend(
            plot_m2d_aperture_components(
                summary_df,
                _m2d_output_base(output_path, stage, primary_stage, components=True),
                formats,
            )
        )
    if not seed_frames:
        return None, tuple()
    seed_table = output_path / "m2d_aperture_recovery_table.csv"
    pd.concat(seed_frames, ignore_index=True).to_csv(seed_table, index=False)
    return seed_table, tuple(plot_paths)


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
    magnification_df, _magnification_inventory_df = collect_magnification_recovery(
        campaign_root,
        run_name=run_name,
        stages=stages,
        variants=variants,
        strict=strict,
    )
    image_residual_df, _image_residual_inventory_df = collect_image_residual_recovery(
        campaign_root,
        run_name=run_name,
        stages=stages,
        variants=variants,
        strict=strict,
    )
    image_residual_draws_df, _image_residual_draws_inventory_df = collect_image_residual_draws_recovery(
        campaign_root,
        run_name=run_name,
        stages=stages,
        variants=variants,
        strict=strict,
    )
    parameter_table_path = output_path / "parameter_recovery_table.csv"
    aggregate_parameter_table_path = output_path / "aggregate_parameter_recovery_table.csv"
    inventory_path = output_path / "run_inventory.csv"
    magnification_table_path = output_path / "magnification_recovery_table.csv"
    image_residual_table_path = output_path / "image_residual_recovery_table.csv"
    image_residual_draws_table_path = output_path / "image_residual_draws_table.csv"
    aggregate_df = aggregate_parameter_recovery(
        parameter_df,
        seed_aggregation=seed_aggregation,
        aggregate_error=aggregate_error,
        fractional_truth_floor=fractional_truth_floor,
    )
    parameter_df.to_csv(parameter_table_path, index=False)
    aggregate_df.to_csv(aggregate_parameter_table_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    if not magnification_df.empty:
        magnification_df.to_csv(magnification_table_path, index=False)
    if not image_residual_df.empty:
        image_residual_df.to_csv(image_residual_table_path, index=False)
    if not image_residual_draws_df.empty:
        image_residual_draws_df.to_csv(image_residual_draws_table_path, index=False)

    apply_lenscluster_plot_style()
    paper_stage = _paper_recovery_stage(stages)
    seed_df = seed_recovery_rows_for_stage(parameter_df, stage=paper_stage, variants=variants)
    seed_available_parameters = seed_df["parameter"].dropna().astype(str).drop_duplicates().tolist()
    plot_paths: list[Path] = []
    if parameter_set in {"paper", "both"}:
        for stage in tuple(str(value) for value in stages):
            stage_seed_df = seed_recovery_rows_for_stage(parameter_df, stage=stage, variants=variants)
            stage_paper_df = paper_parameter_recovery_rows(stage_seed_df)
            stage_available_parameters = stage_paper_df["parameter"].dropna().astype(str).drop_duplicates().tolist()
            paper_parameters = [parameter for parameter in PAPER_PARAMETERS if parameter in set(stage_available_parameters)]
            plot_paths.extend(
                _save_multipage_seed_recovery_plot(
                    stage_paper_df,
                    paper_parameters,
                    _paper_plot_output_base(output_path, stage, paper_stage),
                    formats,
                    include_source_position_panel=True,
                )
            )
    if parameter_set in {"all", "both"}:
        plot_paths.extend(
            _save_multipage_seed_recovery_plot(
                seed_df,
                seed_available_parameters,
                output_path / "mock_parameter_recovery_all",
                formats,
            )
        )
    plot_paths.extend(
        plot_parameter_bias_summary(
            aggregate_df,
            output_path / "mock_parameter_bias_summary",
            formats,
            fractional_truth_floor=fractional_truth_floor,
        )
    )
    for component in JOINT_SIGMA_RCUT_SPECS:
        joint_scaling_df = joint_sigma_rcut_recovery_rows(parameter_df, component=component)
        if joint_scaling_df.empty:
            continue
        for stage in tuple(str(value) for value in stages):
            stage_joint_scaling_df = joint_scaling_df[joint_scaling_df["stage"].astype(str) == stage].copy()
            plot_paths.extend(
                plot_joint_sigma_rcut_recovery(
                    stage_joint_scaling_df,
                    _sigma_rcut_recovery_output_base(output_path, stage, paper_stage, component=component),
                    formats,
                )
            )
    if not magnification_df.empty:
        for stage in tuple(str(value) for value in stages):
            stage_magnification_df = magnification_df[magnification_df["stage"].astype(str) == stage].copy()
            plot_paths.extend(
                plot_aggregate_magnification_recovery(
                    stage_magnification_df,
                    _magnification_recovery_output_base(output_path, stage, paper_stage),
                    formats,
                )
            )
    if not image_residual_df.empty:
        for stage in tuple(str(value) for value in stages):
            stage_image_residual_df = image_residual_df[image_residual_df["stage"].astype(str) == stage].copy()
            plot_paths.extend(
                plot_aggregate_image_residual_recovery(
                    stage_image_residual_df,
                    _image_residual_recovery_output_base(output_path, stage, paper_stage),
                    formats,
                )
            )
    if not image_residual_draws_df.empty:
        for stage in tuple(str(value) for value in stages):
            stage_image_residual_draws_df = image_residual_draws_df[
                image_residual_draws_df["stage"].astype(str) == stage
            ].copy()
            plot_paths.extend(
                plot_aggregate_image_residual_distribution(
                    stage_image_residual_draws_df,
                    _image_residual_distribution_output_base(output_path, stage, paper_stage),
                    formats,
                )
            )
    shmf_seed_table, shmf_summary_table, shmf_plot_paths = write_subhalo_shmf_outputs(
        campaign_root,
        output_path,
        run_name=run_name,
        stages=stages,
        variants=variants,
        formats=formats,
        strict=strict,
    )
    plot_paths.extend(shmf_plot_paths)
    m2d_table, m2d_plot_paths = write_m2d_aperture_outputs(
        campaign_root,
        output_path,
        run_name=run_name,
        stages=stages,
        variants=variants,
        formats=formats,
        strict=strict,
    )
    plot_paths.extend(m2d_plot_paths)
    return AggregatePlotOutputs(
        output_dir=output_path,
        parameter_table=parameter_table_path,
        aggregate_parameter_table=aggregate_parameter_table_path,
        inventory_table=inventory_path,
        magnification_table=magnification_table_path if not magnification_df.empty else None,
        image_residual_table=image_residual_table_path if not image_residual_df.empty else None,
        image_residual_draws_table=image_residual_draws_table_path if not image_residual_draws_df.empty else None,
        subhalo_shmf_seed_table=shmf_seed_table,
        subhalo_shmf_summary_table=shmf_summary_table,
        m2d_aperture_table=m2d_table,
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
    if outputs.magnification_table is not None:
        print(f"Wrote magnification table: {outputs.magnification_table}")
    if outputs.subhalo_shmf_seed_table is not None:
        print(f"Wrote SHMF seed table: {outputs.subhalo_shmf_seed_table}")
    if outputs.subhalo_shmf_summary_table is not None:
        print(f"Wrote SHMF summary table: {outputs.subhalo_shmf_summary_table}")
    for path in outputs.plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
