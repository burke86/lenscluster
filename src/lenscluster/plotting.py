from __future__ import annotations

import argparse
from collections import Counter
import importlib
import inspect
import json
import math
import re
import sys
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import norm
from skimage.measure import find_contours

from .plot_style import apply_lenscluster_plot_style

apply_lenscluster_plot_style()

try:
    import corner
except ImportError:  # pragma: no cover
    corner = None

from .image_diagnostics import (
    diagnostic_detail_array as _shared_diagnostic_detail_array,
    extra_image_rows as _shared_extra_image_rows,
    family_image_recovery_rows as _shared_family_image_recovery_rows,
    image_prediction_for_family_latent as _shared_image_prediction_for_family_latent,
    image_sigma_eff_arcsec as _shared_image_sigma_eff_arcsec,
    image_sigma_int_for_params as _shared_image_sigma_int_for_params,
    image_count_info_from_exact_details as _shared_image_count_info_from_exact_details,
    image_count_recovery_row as _shared_image_count_recovery_row,
    image_count_recovery_summary as _shared_image_count_recovery_summary,
    image_count_recovery_table as _shared_image_count_recovery_table,
    model_count_fields_from_count_info as _shared_model_count_fields_from_count_info,
    successful_image_count_info as _shared_successful_image_count_info,
    unavailable_image_count_info as _shared_unavailable_image_count_info,
)
from .jax_cosmology import critical_surface_density_angle_from_config
from .jax_cosmology import kpc_per_arcsec_from_config as _kpc_per_arcsec_from_config
from .model import BuildState, EvaluationResult, ParameterSpec, PosteriorResults
from .model import convert_theta_to_latent as _convert_theta_to_latent
from .model import display_lower as _display_lower
from .model import display_upper as _display_upper
from .utils import DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN
from .utils import Table as _RichTable
from .utils import log10_dpie_mass_msun as _log10_dpie_mass_msun
from .utils import jax_cpu_worker_count
from .utils import log_message as _log
from .utils import progress_context as _progress_context
from .utils import run_logged_phase as _run_logged_phase

DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02
DEFAULT_NUTS_INIT_JITTER_FRAC = 0.02
DEFAULT_SVI_STEPS = 2000
DEFAULT_SVI_LEARNING_RATE = 5.0e-3
CORNER_SIGMA_CONTOUR_LEVELS = tuple(float(1.0 - np.exp(-0.5 * sigma**2)) for sigma in (1.0, 2.0, 3.0))
CORNER_PLOT_KWARGS = {
    "show_titles": True,
    "title_fmt": ".3g",
    "quantiles": [0.16, 0.5, 0.84],
    "levels": CORNER_SIGMA_CONTOUR_LEVELS,
    "plot_datapoints": False,
    "fill_contours": True,
    "smooth": 1.0,
    "smooth1d": 1.0,
    "max_n_ticks": 4,
}
CORNER_PLOT_DPI = 300
CORNER_BEST_FIT_COLOR = "#d4a017"
CORNER_MAP_COLOR = "#d4a017"
CORNER_MAXIMUM_LIKELIHOOD_COLOR = "tab:orange"
CORNER_PREVIOUS_STAGE_COLOR = "tab:green"
CORNER_BAYES_OVERLAY_COLOR = "tab:red"
SMC_CORNER_MAX_PARAMS = 8
CAUSTIC_OVERLAY_FOV_ARCSEC = 200.0
CAUSTIC_PLOT_GRID_SCALE_ARCSEC = 0.2
ABSOLUTE_MAGNIFICATION_PLOT_CAP = 25.0
ABSOLUTE_MAGNIFICATION_RECOVERY_AXIS_MAX = 29.0
MAP_RECOVERY_HISTOGRAM_BINS = 80
MAP_RECOVERY_STAT_BINS = 16
RECOVERY_ONE_SIGMA_PERCENTILES = (16.0, 84.0)
RECOVERY_TWO_SIGMA_PERCENTILES = (2.5, 97.5)
MODEL_GRID_CHUNK_PIXELS = 1024 * 1024
FLUX_MAGNIFICATION_P_ARC_LOG10_FLOOR = 1.0e-6
DEFAULT_TRUTH_GRID_SIZE = 256
KAPPA_RECOVERY_LIMITS = (0.0, 5.0)
RECOVERY_IMAGE_POINT_COLUMNS = [
    "family_id",
    "image_label",
    "x_obs_arcsec",
    "y_obs_arcsec",
    "true_value",
    "model_value",
]
TRUTH_GRID_QUANTILE_PERCENTILES = (16.0, 50.0, 84.0)
TRUTH_GRID_QUANTILE_SUFFIXES = ("q16", "median", "q84")
TRUTH_GRID_MODE_MEDIAN = "median"
TRUTH_GRID_MODE_POSTERIOR = "posterior"
TRUTH_GRID_DRAW_SELECTION_ALL = "all_finite"
TRUTH_GRID_DRAW_SELECTION_GROUPED_RANDOM = "chain_stratified_random_without_replacement"
TRUTH_GRID_DRAW_SELECTION_FLAT_RANDOM = "flat_random_without_replacement"
TRUTH_RECOVERY_APERTURE_CENTER_SMOOTHING_SIGMA_PIX = 1.0
DEFAULT_RUNTIME_SEED = 12345
TRUTH_GRID_QUANTITY_OUTPUT_NAMES = {
    "kappa": "kappa",
    "gamma1": "gamma1",
    "gamma2": "gamma2",
    "detA": "detA",
    "mu": "mu",
    "abs_mu": "abs_mu",
}
CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE = "critical-arc-mixture-image-plane"
CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE_MODE = "critical-arc-anisotropic-image-plane"
CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD = 0.1
CRITICAL_ARC_BASE_PROB = 0.10
CRITICAL_ARC_MAX_PROB = 0.80
CRITICAL_ARC_SINGULAR_THRESHOLD = 0.20
CRITICAL_ARC_SINGULAR_SOFTNESS = 0.05
CRITICAL_ARC_SINGULAR_THRESHOLD_SAMPLE_NAME = "critical_arc_singular_threshold"
CRITICAL_ARC_SINGULAR_SOFTNESS_SAMPLE_NAME = "critical_arc_singular_softness"
CRITICAL_ARC_LOG_S_MIN_FLOOR = 1.0e-12
SUBHALO_TOTAL_MASS_RADIUS_FACTOR = 1.0e6
DPiE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN = DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN
SCALING_RESULTS_MASS_NOTE = (
    "M* = (3/2) * pi * vdisp*^2 * rcut* / G (Eliasdottir+07 eq. A11, "
    "lenstool sigma convention), with G = 4.30091e-6 kpc (km/s)^2 Msun^-1"
)
NUMPYRO_MODEL_PLOT_FILENAME = "numpyro_model.pdf"
NUMPYRO_MODEL_DEFAULT_LIKELIHOOD_MODE = "source"
NUMPYRO_MODEL_ROLE_PRIORITY = (
    "critical_arc_hyperparameter",
    "image_scatter",
    "source_scatter",
    "source_position",
    "cosmology",
    "active_scaling_gate",
    "independent_scaling_gate",
    "independent_scaling",
    "scaling_scatter",
    "scaling",
    "large",
    "other",
)
_SHOW_PLOTS = False


def _stage_scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if item is not None:
                return item
        return default
    return value


def _maybe_show_figure(fig: Any) -> None:
    if _SHOW_PLOTS:
        plt.figure(fig.number)
        plt.show()


def _finish_figure(fig: Any, path: Path, *, dpi: int = 180, bbox_inches: str | None = "tight", **savefig_kwargs: Any) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    _maybe_show_figure(fig)
    plt.close(fig)


def _finish_pdf_page(pdf: PdfPages, fig: Any, **savefig_kwargs: Any) -> None:
    pdf.savefig(fig, **savefig_kwargs)
    _maybe_show_figure(fig)
    plt.close(fig)


NUMPYRO_MODEL_ROLE_STYLE = {
    "critical_arc_hyperparameter": {
        "title": "Critical-arc gate",
        "detail": "singular-threshold controls",
        "fill": "#FDECEC",
        "line": "#C62828",
        "font": "#5C1616",
    },
    "image_scatter": {
        "title": "Intrinsic image scatter",
        "detail": "image-plane uncertainty",
        "fill": "#F3EAF7",
        "line": "#7B3F98",
        "font": "#3B2147",
    },
    "source_scatter": {
        "title": "Intrinsic source scatter",
        "detail": "source-plane uncertainty",
        "fill": "#EDE7F6",
        "line": "#5E35B1",
        "font": "#311B92",
    },
    "source_position": {
        "title": "Source positions",
        "detail": "sampled source coordinates",
        "fill": "#FFF3E0",
        "line": "#C77700",
        "font": "#4E342E",
    },
    "cosmology": {
        "title": "Sampled cosmology",
        "detail": "geometry parameters",
        "fill": "#EEF2FF",
        "line": "#3F51B5",
        "font": "#1A237E",
    },
    "independent_scaling": {
        "title": "Independent members",
        "detail": "free member-galaxy branches",
        "fill": "#E8F0F2",
        "line": "#546E7A",
        "font": "#263238",
    },
    "active_scaling_gate": {
        "title": "Active member gate",
        "detail": "SVI exact/cached split",
        "fill": "#E9F3FB",
        "line": "#1E6A9E",
        "font": "#123B55",
    },
    "independent_scaling_gate": {
        "title": "Member mixture gate",
        "detail": "magnitude population weights",
        "fill": "#EAF4EF",
        "line": "#2E7D59",
        "font": "#17412E",
    },
    "scaling_scatter": {
        "title": "Scaling-law scatter",
        "detail": "member-galaxy intrinsic scatter",
        "fill": "#E0F2F1",
        "line": "#00897B",
        "font": "#004D40",
    },
    "scaling": {
        "title": "Member scaling law",
        "detail": "galaxy-scale lens parameters",
        "fill": "#E8F6EF",
        "line": "#2E7D4F",
        "font": "#174B31",
    },
    "large": {
        "title": "Large-scale lens",
        "detail": "cluster-scale lens parameters",
        "fill": "#E8F1FA",
        "line": "#2B6CB0",
        "font": "#17324D",
    },
    "other": {
        "title": "Other sampled parameters",
        "detail": "uncategorized solver parameters",
        "fill": "#F7F9FA",
        "line": "#607D8B",
        "font": "#263238",
    },
}
SUBHALO_PROPERTIES_COLUMNS = [
    "component_index",
    "model_component_index",
    "potfile_id",
    "potfile_order",
    "catalog_id",
    "catalog_mag",
    "x_centre",
    "y_centre",
    "radius_arcsec",
    "sigma0",
    "Ra",
    "Rs",
    "mass_within_Rs_msun",
    "mass_within_1e6_Rs_msun",
]


def _caustic_plot_grid_axes(grid_scale_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    grid_scale = float(grid_scale_arcsec)
    if not np.isfinite(grid_scale) or grid_scale <= 0.0:
        raise ValueError("caustic plot grid scale must be positive.")
    half_fov = CAUSTIC_OVERLAY_FOV_ARCSEC / 2.0
    grid_num_pix = max(2, int(round(CAUSTIC_OVERLAY_FOV_ARCSEC / grid_scale)) + 1)
    axis = np.linspace(-half_fov, half_fov, grid_num_pix)
    return axis, axis.copy()


def _first_int_value(value: Any, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return int(default)
        return int(value[0])
    return int(value)


def plot_path(root: Path, name: str) -> Path:
    """Return an output plot path, creating the output directory first."""
    root.mkdir(parents=True, exist_ok=True)
    prefix = _cluster_plot_filename_prefix(root)
    output_name = _prefixed_plot_filename(name, prefix)
    path = root / output_name
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        path = path.with_suffix(".pdf")
    return path


def _plot_path(root: Path, name: str) -> Path:
    return plot_path(root, name)


def _cluster_plot_filename_prefix(root: Path) -> str | None:
    path = Path(root)
    candidates = [path.name]
    if path.parent != path:
        candidates.append(path.parent.name)
    for raw_name in candidates:
        name = str(raw_name).strip()
        match = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(?:S\d|PD)", name)
        if match:
            return match.group(1).lower()
    return None


def _prefixed_plot_filename(name: str, prefix: str | None) -> str:
    if prefix is None:
        return str(name)
    path = Path(str(name))
    filename = path.name
    if filename.startswith(f"{prefix}_"):
        return str(name)
    return str(path.with_name(f"{prefix}_{filename}"))


PlotTask = tuple[str, str, Callable[..., Any]]
PlotStage = tuple[str, list[PlotTask]]


def _active_sample_likelihood_mode(evaluator: Any, args: argparse.Namespace) -> str:
    mode = getattr(evaluator, "sample_likelihood_mode", None)
    if mode is None:
        mode = getattr(args, "sample_likelihood_mode", NUMPYRO_MODEL_DEFAULT_LIKELIHOOD_MODE)
    if mode is None:
        mode = NUMPYRO_MODEL_DEFAULT_LIKELIHOOD_MODE
    return str(mode)


def _uses_arc_aware_diagnostics(sample_likelihood_mode: Any) -> bool:
    return str(sample_likelihood_mode or "").strip() in {
        CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE,
        CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE_MODE,
    }


def _summary_uses_arc_aware_diagnostics(summary: dict[str, Any]) -> bool:
    return _uses_arc_aware_diagnostics(summary.get("sample_likelihood_mode"))


def _parameter_sample_sites_for_rendering(parameter_specs: list[ParameterSpec]) -> list[Any]:
    if not parameter_specs:
        return []
    from .cluster_solver import _parameter_sample_sites  # local import avoids a module import cycle at plotting import time

    return _parameter_sample_sites(parameter_specs)


def _numpyro_model_role_for_family(component_family: str) -> str:
    family = str(component_family or "").strip()
    return family if family in NUMPYRO_MODEL_ROLE_PRIORITY and family != "other" else "other"


def _numpyro_model_role_for_site(site: Any, parameter_specs: list[ParameterSpec]) -> str:
    roles: set[str] = set()
    for index in getattr(site, "indices", ()):
        idx = int(index)
        if 0 <= idx < len(parameter_specs):
            roles.add(_numpyro_model_role_for_family(str(getattr(parameter_specs[idx], "component_family", ""))))
    for role in NUMPYRO_MODEL_ROLE_PRIORITY:
        if role in roles:
            return role
    return "other"


def _numpyro_model_role_counts(
    parameter_specs: list[ParameterSpec],
    sample_sites: list[Any],
) -> tuple[Counter[str], Counter[str]]:
    parameter_counts: Counter[str] = Counter()
    for spec in parameter_specs:
        parameter_counts[_numpyro_model_role_for_family(str(getattr(spec, "component_family", "")))] += 1
    site_counts: Counter[str] = Counter()
    for site in sample_sites:
        site_counts[_numpyro_model_role_for_site(site, parameter_specs)] += 1
    return parameter_counts, site_counts


def _plural_count(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if int(count) == 1 else str(plural or f"{singular}s")
    return f"{int(count)} {word}"


def _state_family_and_image_counts(state: Any) -> tuple[int, int]:
    family_data = list(getattr(state, "family_data", []) or [])
    image_count = 0
    for family in family_data:
        value = family.get("n_images", 0) if isinstance(family, dict) else getattr(family, "n_images", 0)
        try:
            image_count += int(value)
        except (TypeError, ValueError):
            continue
    return len(family_data), image_count


def _numpyro_model_likelihood_label(sample_likelihood_mode: str) -> str:
    labels = {
        "source": "ln ℒ(η)\nsource-plane likelihood",
        "local-jacobian": "ln ℒ(η)\nlocal image-plane likelihood",
        "critical-arc-mixture-image-plane": "ln ℒ(η, {β_f})\ncritical-arc image-plane likelihood",
    }
    return labels.get(str(sample_likelihood_mode), f"ln ℒ(η)\n{str(sample_likelihood_mode).replace('-', ' ')} likelihood")


def _compact_numpyro_role_label(role: str, parameter_count: int, site_count: int) -> str:
    style = NUMPYRO_MODEL_ROLE_STYLE[role]
    return (
        f"{style['title']}\n"
        f"{_plural_count(parameter_count, 'parameter')} / {_plural_count(site_count, 'site')}"
    )


def _build_compact_numpyro_model_graph(
    *,
    state: BuildState,
    parameter_specs: list[ParameterSpec],
    sample_sites: list[Any],
    sample_likelihood_mode: str,
) -> Any:
    from graphviz import Digraph

    family_count, image_count = _state_family_and_image_counts(state)
    parameter_counts, site_counts = _numpyro_model_role_counts(parameter_specs, sample_sites)
    present_roles = [
        role
        for role in NUMPYRO_MODEL_ROLE_PRIORITY
        if int(parameter_counts.get(role, 0)) > 0 or int(site_counts.get(role, 0)) > 0
    ]
    graph = Digraph(name="numpyro_model")
    graph.attr(
        rankdir="LR",
        bgcolor="white",
        pad="0.18",
        nodesep="0.36",
        ranksep="0.72",
        splines="spline",
        concentrate="true",
        outputorder="edgesfirst",
        fontname="Helvetica",
    )
    graph.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="12", margin="0.10,0.07", penwidth="1.45")
    graph.attr("edge", color="#52656F", arrowsize="0.75", penwidth="1.1", fontname="Helvetica", fontsize="10")

    for role in present_roles:
        style = NUMPYRO_MODEL_ROLE_STYLE[role]
        graph.node(
            role,
            _compact_numpyro_role_label(role, int(parameter_counts.get(role, 0)), int(site_counts.get(role, 0))),
            fillcolor=style["fill"],
            color=style["line"],
            fontcolor=style["font"],
        )
    if not present_roles:
        graph.node(
            "fixed_model",
            "Fixed model\nno sampled parameters",
            fillcolor="#F7F9FA",
            color="#607D8B",
            fontcolor="#263238",
        )

    theta_label = "θ\nlatent parameter vector" if sample_sites else "fixed parameter vector\nno sample sites"
    graph.node(
        "theta",
        theta_label,
        shape="box",
        style="rounded,filled,dashed",
        fillcolor="#F7F9FA",
        color="#78909C",
        fontcolor="#263238",
    )
    graph.node(
        "likelihood",
        _numpyro_model_likelihood_label(sample_likelihood_mode),
        shape="box",
        style="rounded,filled,bold",
        fillcolor="#263238",
        color="#263238",
        fontcolor="white",
        penwidth="1.8",
    )
    for role in present_roles:
        graph.edge(role, "theta")
    if not present_roles:
        graph.edge("fixed_model", "theta")
    graph.edge("theta", "likelihood")
    if family_count > 0:
        graph.node(
            "observed_catalog",
            f"Observed image positions\n{_plural_count(image_count, 'position')}",
            shape="box",
            style="rounded,filled",
            fillcolor="#ECEFF1",
            color="#607D8B",
            fontcolor="#263238",
        )
        graph.edge("observed_catalog", "likelihood", style="dashed", color="#78909C")
    return graph


def _plot_numpyro_model(run_dir: Path, state: BuildState, evaluator: Any, args: argparse.Namespace) -> Path:
    parameter_specs = list(getattr(state, "parameter_specs", []) or [])
    sample_likelihood_mode = _active_sample_likelihood_mode(evaluator, args)
    sample_sites = _parameter_sample_sites_for_rendering(parameter_specs)
    compact_graph = _build_compact_numpyro_model_graph(
        state=state,
        parameter_specs=parameter_specs,
        sample_sites=sample_sites,
        sample_likelihood_mode=sample_likelihood_mode,
    )
    output_path = _plot_path(run_dir, NUMPYRO_MODEL_PLOT_FILENAME)
    rendered_path = Path(compact_graph.render(str(output_path.with_suffix("")), format="pdf", cleanup=True))
    if rendered_path != output_path and rendered_path.exists() and not output_path.exists():
        rendered_path.replace(output_path)
    return output_path


def _run_plot_tasks_with_progress(args: argparse.Namespace, plot_tasks: list[PlotTask]) -> None:
    if not plot_tasks:
        return
    if bool(getattr(args, "quiet", False)):
        for _display_name, phase_name, task in plot_tasks:
            _run_logged_phase(args, phase_name, task)
        return
    with _progress_context(
        args,
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task("plots", total=len(plot_tasks))
        for display_name, phase_name, task in plot_tasks:
            progress.update(task_id, description=f"plots: {display_name}")
            _run_logged_phase(args, phase_name, task)
            progress.advance(task_id)
        progress.update(task_id, description="plots: complete")


def _call_plot_task(task: Callable[..., Any], progress: Any | None = None) -> Any:
    signature = inspect.signature(task)
    if len(signature.parameters) == 0:
        return task()
    return task(progress)


def _run_plot_stages_with_progress(args: argparse.Namespace, stages: list[PlotStage]) -> None:
    active_stages = [(stage_name, tasks) for stage_name, tasks in stages if tasks]
    if not active_stages:
        return
    def run_stage(stage_name: str, tasks: list[PlotTask], progress: Any | None = None) -> None:
        subtask_id = None
        if progress is not None:
            subtask_id = progress.add_task(f"{stage_name}", total=len(tasks))
        for display_name, phase_name, task in tasks:
            if progress is not None and subtask_id is not None:
                progress.update(subtask_id, description=f"{stage_name}: {display_name}")
            _run_logged_phase(args, phase_name, lambda task=task, progress=progress: _call_plot_task(task, progress))
            if progress is not None and subtask_id is not None:
                progress.advance(subtask_id)
        if progress is not None and subtask_id is not None:
            progress.update(subtask_id, description=f"{stage_name}: complete")

    if bool(getattr(args, "quiet", False)):
        for stage_name, tasks in active_stages:
            _run_logged_phase(args, f"plots.{stage_name}", lambda stage_name=stage_name, tasks=tasks: run_stage(stage_name, tasks))
        return
    with _progress_context(
        args,
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        stage_task_id = progress.add_task("plot stages", total=len(active_stages))
        for stage_name, tasks in active_stages:
            progress.update(stage_task_id, description=f"plot stages: {stage_name}")
            _run_logged_phase(args, f"plots.{stage_name}", lambda stage_name=stage_name, tasks=tasks: run_stage(stage_name, tasks, progress))
            progress.advance(stage_task_id)
        progress.update(stage_task_id, description="plot stages: complete")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def _normalized_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0,), dtype=float)
    if weights is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if weight_array.size != n_samples or not np.all(np.isfinite(weight_array)):
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    total = float(np.sum(weight_array))
    if total <= 0.0:
        return np.full(n_samples, 1.0 / n_samples, dtype=float)
    return weight_array / total


def _valid_normalized_sample_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if weights is None or n_samples <= 0:
        return None
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if weight_array.size != n_samples:
        return None
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        return None
    total = float(np.sum(weight_array))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weight_array / total


def _effective_sample_size(weights: np.ndarray) -> float:
    normalized = _normalized_weights(weights, len(weights))
    if normalized.size == 0:
        return 0.0
    return float(1.0 / np.sum(np.square(normalized)))


def _family_color_map(family_ids: list[str] | np.ndarray | pd.Series) -> dict[str, tuple[float, float, float, float]]:
    ordered_family_ids = list(dict.fromkeys(str(family_id) for family_id in family_ids))
    cmap = plt.get_cmap("tab20", max(1, len(ordered_family_ids)))
    return {family_id: cmap(idx) for idx, family_id in enumerate(ordered_family_ids)}


def _color_with_alpha(color: Any, alpha: float) -> tuple[float, float, float, float]:
    try:
        rgba = np.asarray(to_rgba(color), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        rgba = np.asarray(color, dtype=float).reshape(-1)
    if rgba.size < 3:
        return (0.0, 0.0, 0.0, float(alpha))
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha))


def _metric_text(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "na"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text else "na"
    if not np.isfinite(numeric):
        return "na"
    if abs(numeric - round(numeric)) < 1.0e-10 and abs(numeric) < 1.0e9:
        return str(int(round(numeric)))
    return f"{numeric:.{precision}g}"


def _key_value_lines(items: list[tuple[str, Any]], *, key_width: int = 34) -> list[str]:
    return [f"{label:<{key_width}} {_metric_text(value)}" for label, value in items]


def _table_text(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    rendered = [
        {header: _metric_text(row.get(key)) for header, key in columns}
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rendered))
        for header, _key in columns
    }
    lines = [
        " ".join(header.ljust(widths[header]) for header, _key in columns),
        " ".join("-" * widths[header] for header, _key in columns),
    ]
    for row in rendered:
        lines.append(" ".join(row[header].ljust(widths[header]) for header, _key in columns))
    return lines


def _finite_sample_rows(samples: np.ndarray) -> np.ndarray:
    samples_array = np.asarray(samples, dtype=float)
    if samples_array.ndim != 2 or samples_array.size == 0:
        return np.empty((0, samples_array.shape[-1] if samples_array.ndim == 2 else 0), dtype=float)
    return samples_array[np.isfinite(samples_array).all(axis=1)]


def _uniform_normalized_plot_weights(samples: np.ndarray) -> np.ndarray | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] <= 0:
        return None
    return np.full(sample_array.shape[0], 1.0 / float(sample_array.shape[0]), dtype=float)


def _summary_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    if not parameter_specs:
        return pd.DataFrame(
            columns=[
                "potential_id",
                "profile_type",
                "component_family",
                "prior_kind",
                "parameter",
                "label",
                "map",
                "median",
                "mean",
                "std",
                "p16",
                "p84",
                "lower",
                "upper",
            ]
        )
    weights = _normalized_weights(sample_weights, samples.shape[0])
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(parameter_specs):
        values = samples[:, idx]
        q16, q50, q84 = _weighted_quantile(values, weights, [0.16, 0.50, 0.84])
        rows.append(
            {
                "potential_id": spec.potential_id,
                "profile_type": spec.profile_type,
                "component_family": spec.component_family,
                "prior_kind": spec.prior_kind,
                "parameter": spec.field,
                "label": spec.name,
                "map": float(best_fit[idx]),
                "median": float(q50),
                "mean": float(np.average(values, weights=weights)),
                "std": float(np.sqrt(np.average(np.square(values - np.average(values, weights=weights)), weights=weights))),
                "p16": float(q16),
                "p84": float(q84),
                "lower": _display_lower(spec),
                "upper": _display_upper(spec),
            }
        )
    return pd.DataFrame(rows)


def _write_potfile_summary_txt(tables_dir: Path, summary_df: pd.DataFrame) -> None:
    scaling_df = summary_df[summary_df["component_family"] == "scaling"].copy()
    if scaling_df.empty:
        return
    scaling_df = scaling_df.sort_values(["potential_id", "parameter", "label"]).reset_index(drop=True)
    lines: list[str] = []
    for potential_id, group_df in scaling_df.groupby("potential_id", sort=False):
        lines.append(str(potential_id))
        for row in group_df.itertuples(index=False):
            lines.append(f"  {row.parameter}: median={float(row.median):.6g}, std={float(row.std):.6g}")
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    (tables_dir / "potfile_summary.txt").write_text(content, encoding="utf-8")


def _scaling_results_summary_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    scaling_relation_mode: str,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = [
        "potfile_id",
        "scaling_relation_mode",
        "vdisp_star_median",
        "vdisp_star_p16",
        "vdisp_star_p84",
        "vdisp_star_map",
        "rcut_star_kpc_median",
        "rcut_star_kpc_p16",
        "rcut_star_kpc_p84",
        "rcut_star_kpc_map",
        "rcore_star_kpc_median",
        "rcore_star_kpc_p16",
        "rcore_star_kpc_p84",
        "rcore_star_kpc_map",
        "log10_m_star_msun_median",
        "log10_m_star_msun_p16",
        "log10_m_star_msun_p84",
        "log10_m_star_msun_map",
        "alpha_sigma_median",
        "alpha_sigma_p16",
        "alpha_sigma_p84",
        "alpha_sigma_map",
        "beta_radius_median",
        "beta_radius_p16",
        "beta_radius_p84",
        "beta_radius_map",
        "gamma_ml_median",
        "gamma_ml_p16",
        "gamma_ml_p84",
        "gamma_ml_map",
        "sigma_log_scatter_median",
        "sigma_log_scatter_p16",
        "sigma_log_scatter_p84",
        "mass_log_scatter_median",
        "mass_log_scatter_p16",
        "mass_log_scatter_p84",
        "m_star_definition",
    ]
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    if (
        not parameter_specs
        or sample_array.ndim != 2
        or sample_array.shape[0] == 0
        or sample_array.shape[1] == 0
    ):
        return pd.DataFrame(columns=columns)
    weights = _normalized_weights(sample_weights, sample_array.shape[0])
    specs_by_potfile: dict[str, dict[str, int]] = {}
    for idx, spec in enumerate(parameter_specs):
        if str(getattr(spec, "component_family", "")) not in {"scaling", "scaling_scatter", "independent_scaling"}:
            continue
        potfile_id = str(getattr(spec, "potential_id", ""))
        if not potfile_id:
            continue
        specs_by_potfile.setdefault(potfile_id, {})[str(getattr(spec, "field", ""))] = idx

    def _values(index: int | None) -> np.ndarray | None:
        if index is None or index < 0 or index >= sample_array.shape[1]:
            return None
        return np.asarray(sample_array[:, index], dtype=float)

    def _map(index: int | None) -> float:
        if index is None or index < 0 or index >= best_fit_array.size:
            return float("nan")
        return float(best_fit_array[index])

    def _add_summary(row: dict[str, Any], prefix: str, values: np.ndarray | None, map_value: float | None = None) -> None:
        if values is None:
            row[f"{prefix}_median"] = float("nan")
            row[f"{prefix}_p16"] = float("nan")
            row[f"{prefix}_p84"] = float("nan")
            if map_value is not None:
                row[f"{prefix}_map"] = float("nan")
            return
        summary = _finite_weighted_summary(np.asarray(values, dtype=float), weights)
        row[f"{prefix}_median"] = summary["median"]
        row[f"{prefix}_p16"] = summary["p16"]
        row[f"{prefix}_p84"] = summary["p84"]
        if map_value is not None:
            row[f"{prefix}_map"] = float(map_value) if np.isfinite(float(map_value)) else float("nan")

    rows: list[dict[str, Any]] = []
    mode = str(scaling_relation_mode or "direct-exponents")
    for potfile_id in sorted(specs_by_potfile):
        field_index = specs_by_potfile[potfile_id]
        sigma = _values(field_index.get("sigma"))
        cut = _values(field_index.get("cutkpc"))
        core = _values(field_index.get("corekpc"))
        if sigma is None and cut is None and core is None:
            continue
        row: dict[str, Any] = {
            "potfile_id": potfile_id,
            "scaling_relation_mode": mode,
            "m_star_definition": SCALING_RESULTS_MASS_NOTE,
        }
        _add_summary(row, "vdisp_star", sigma, _map(field_index.get("sigma")))
        _add_summary(row, "rcut_star_kpc", cut, _map(field_index.get("cutkpc")))
        _add_summary(row, "rcore_star_kpc", core, _map(field_index.get("corekpc")))
        if sigma is not None and cut is not None:
            log_mass = _log10_dpie_mass_msun(sigma, cut)
            sigma_map = _map(field_index.get("sigma"))
            cut_map = _map(field_index.get("cutkpc"))
            core_map = _map(field_index.get("corekpc"))
            log_mass_map = float(_log10_dpie_mass_msun(sigma_map, cut_map))
            _add_summary(row, "log10_m_star_msun", log_mass, log_mass_map)
        else:
            _add_summary(row, "log10_m_star_msun", None, float("nan"))

        alpha_sigma = _values(field_index.get("alpha_sigma"))
        gamma_ml = _values(field_index.get("gamma_ml"))
        if alpha_sigma is not None and gamma_ml is not None:
            alpha_map = _map(field_index.get("alpha_sigma"))
            gamma_map = _map(field_index.get("gamma_ml"))
            beta_radius = 1.0 + gamma_ml - 2.0 * alpha_sigma
            beta_map = 1.0 + gamma_map - 2.0 * alpha_map
            _add_summary(row, "alpha_sigma", alpha_sigma, alpha_map)
            _add_summary(row, "beta_radius", beta_radius, beta_map)
            _add_summary(row, "gamma_ml", gamma_ml, gamma_map)
        else:
            _add_summary(row, "alpha_sigma", None, float("nan"))
            _add_summary(row, "beta_radius", None, float("nan"))
            _add_summary(row, "gamma_ml", None, float("nan"))
        for prefix, field in (
            ("sigma_log_scatter", "sigma_log_scatter"),
            ("mass_log_scatter", "mass_log_scatter"),
        ):
            _add_summary(row, prefix, _values(field_index.get(field)))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def _interval_text(row: pd.Series, prefix: str, *, precision: int = 3) -> str:
    median = row.get(f"{prefix}_median", np.nan)
    p16 = row.get(f"{prefix}_p16", np.nan)
    p84 = row.get(f"{prefix}_p84", np.nan)
    if not (np.isfinite(median) and np.isfinite(p16) and np.isfinite(p84)):
        return "na"
    return f"{float(median):.{precision}g} -{float(median - p16):.{precision}g}/+{float(p84 - median):.{precision}g}"


def _build_scaling_results_rich_table(summary_df: pd.DataFrame) -> Any:
    table = _RichTable(
        title=f"Scaling Relation Results ({SCALING_RESULTS_MASS_NOTE})",
        show_lines=False,
    )
    columns = [
        ("potfile", "potfile_id"),
        ("mode", "scaling_relation_mode"),
        ("vdisp* [km/s]", "vdisp_star"),
        ("rcut* [kpc]", "rcut_star_kpc"),
        ("rcore* [kpc]", "rcore_star_kpc"),
        ("log10 M* [Msun]", "log10_m_star_msun"),
        ("alpha_sigma", "alpha_sigma"),
        ("beta_radius", "beta_radius"),
        ("gamma_ml", "gamma_ml"),
        ("sigma scatter", "sigma_log_scatter"),
        ("mass scatter", "mass_log_scatter"),
    ]
    for header, _prefix in columns:
        table.add_column(header)
    if summary_df.empty:
        table.add_row("no scaling relation parameters", *[""] * (len(columns) - 1))
        return table
    for row in summary_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        table.add_row(
            str(row_series.get("potfile_id", "")),
            str(row_series.get("scaling_relation_mode", "")),
            _interval_text(row_series, "vdisp_star"),
            _interval_text(row_series, "rcut_star_kpc"),
            _interval_text(row_series, "rcore_star_kpc"),
            _interval_text(row_series, "log10_m_star_msun"),
            _interval_text(row_series, "alpha_sigma"),
            _interval_text(row_series, "beta_radius"),
            _interval_text(row_series, "gamma_ml"),
            _interval_text(row_series, "sigma_log_scatter"),
            _interval_text(row_series, "mass_log_scatter"),
        )
    return table


def _potfile_constraint_diagnostics_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    scaling_rank_df: pd.DataFrame,
    sample_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    scaling_specs = [spec for spec in parameter_specs if spec.component_family == "scaling"]
    if not scaling_specs:
        return pd.DataFrame(
            columns=[
                "potfile_id",
                "parameter",
                "label",
                "prior_kind",
                "prior_mean",
                "prior_std",
                "posterior_mean",
                "posterior_median",
                "posterior_std",
                "map",
                "p16",
                "p84",
                "posterior_interval_width",
                "posterior_std_over_prior_std",
                "posterior_mean_minus_prior_mean_over_prior_std",
                "n_components",
                "n_active_components",
                "min_distance_arcsec_min",
                "min_distance_arcsec_median",
                "importance_max",
                "importance_median",
                "brightness_median",
            ]
        )
    weights = _normalized_weights(sample_weights, samples.shape[0])
    leverage_by_potfile: dict[str, dict[str, float]] = {}
    if not scaling_rank_df.empty:
        grouped = scaling_rank_df.groupby("potfile_id", sort=False)
        for potfile_id, group_df in grouped:
            leverage_by_potfile[str(potfile_id)] = {
                "n_components": int(len(group_df)),
                "n_active_components": int(np.sum(group_df["selected_active"].astype(bool))),
                "min_distance_arcsec_min": float(group_df["min_distance_arcsec"].min()),
                "min_distance_arcsec_median": float(group_df["min_distance_arcsec"].median()),
                "importance_max": float(group_df["importance"].max()),
                "importance_median": float(group_df["importance"].median()),
                "brightness_median": float(group_df["brightness"].median()),
            }
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(parameter_specs):
        if spec.component_family != "scaling":
            continue
        values = np.asarray(samples[:, idx], dtype=float)
        q16, q50, q84 = _weighted_quantile(values, weights, [0.16, 0.50, 0.84])
        posterior_mean = float(np.average(values, weights=weights))
        posterior_std = float(
            np.sqrt(np.average(np.square(values - posterior_mean), weights=weights))
        )
        prior_mean = spec.physical_mean if spec.physical_mean is not None else spec.mean
        prior_std = spec.physical_std if spec.physical_std is not None else spec.std
        prior_mean_value = float(prior_mean) if prior_mean is not None else float("nan")
        prior_std_value = float(prior_std) if prior_std is not None else float("nan")
        leverage = leverage_by_potfile.get(
            str(spec.potential_id),
            {
                "n_components": 0,
                "n_active_components": 0,
                "min_distance_arcsec_min": float("nan"),
                "min_distance_arcsec_median": float("nan"),
                "importance_max": float("nan"),
                "importance_median": float("nan"),
                "brightness_median": float("nan"),
            },
        )
        rows.append(
            {
                "potfile_id": str(spec.potential_id),
                "parameter": str(spec.field),
                "label": str(spec.name),
                "prior_kind": str(spec.prior_kind),
                "prior_mean": prior_mean_value,
                "prior_std": prior_std_value,
                "posterior_mean": posterior_mean,
                "posterior_median": float(q50),
                "posterior_std": posterior_std,
                "map": float(best_fit[idx]),
                "p16": float(q16),
                "p84": float(q84),
                "posterior_interval_width": float(q84 - q16),
                "posterior_std_over_prior_std": (
                    float(posterior_std / prior_std_value)
                    if np.isfinite(prior_std_value) and prior_std_value > 0.0
                    else float("nan")
                ),
                "posterior_mean_minus_prior_mean_over_prior_std": (
                    float((posterior_mean - prior_mean_value) / prior_std_value)
                    if np.isfinite(prior_mean_value) and np.isfinite(prior_std_value) and prior_std_value > 0.0
                    else float("nan")
                ),
                **leverage,
            }
        )
    return pd.DataFrame(rows).sort_values(["potfile_id", "parameter"]).reset_index(drop=True)


def _sigmoid_np(values: np.ndarray | float) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    clipped = np.clip(values_array, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


_INDEPENDENT_GATE_GH_NODES, _INDEPENDENT_GATE_GH_WEIGHTS = np.polynomial.hermite.hermgauss(15)
_INDEPENDENT_GATE_GH_NORMAL_WEIGHTS = _INDEPENDENT_GATE_GH_WEIGHTS / math.sqrt(math.pi)


def _logistic_normal_sigmoid_expectation_np(logits: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    logit_array = np.asarray(logits, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    shifted = logit_array[..., None] + math.sqrt(2.0) * sigma_array[..., None] * _INDEPENDENT_GATE_GH_NODES
    return np.sum(_sigmoid_np(shifted) * _INDEPENDENT_GATE_GH_NORMAL_WEIGHTS, axis=-1)


def _finite_weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    finite = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0.0)
    if value_array.size == 0 or weight_array.size != value_array.size or not np.any(finite):
        return {"mean": float("nan"), "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    finite_values = value_array[finite]
    finite_weights = weight_array[finite]
    total = float(np.sum(finite_weights))
    if total <= 0.0:
        finite_weights = np.full(finite_values.size, 1.0 / float(finite_values.size), dtype=float)
    else:
        finite_weights = finite_weights / total
    q16, q50, q84 = _weighted_quantile(finite_values, finite_weights, [0.16, 0.50, 0.84])
    return {
        "mean": float(np.average(finite_values, weights=finite_weights)),
        "median": float(q50),
        "p16": float(q16),
        "p84": float(q84),
    }


def _finite_weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    quantile_values = list(quantiles)
    finite = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0.0)
    if value_array.size == 0 or weight_array.size != value_array.size or not np.any(finite):
        return np.full(len(quantile_values), np.nan, dtype=float)
    finite_values = value_array[finite]
    finite_weights = weight_array[finite]
    total = float(np.sum(finite_weights))
    if total <= 0.0:
        finite_weights = np.full(finite_values.size, 1.0 / float(finite_values.size), dtype=float)
    else:
        finite_weights = finite_weights / total
    return _weighted_quantile(finite_values, finite_weights, quantile_values)


def _component_array_from_packed(
    packed_lens_spec: Any,
    field_name: str,
    n_components: int,
    *,
    dtype: Any,
    fill_value: float | int,
) -> np.ndarray:
    values = np.asarray(getattr(packed_lens_spec, field_name, []), dtype=dtype).reshape(-1)
    if values.size == n_components:
        return values
    if values.size == 0:
        return np.full(n_components, fill_value, dtype=dtype)
    raise ValueError(
        f"PackedLensSpec.{field_name} length {values.size} does not match component count {n_components}."
    )


def _active_scaling_diagnostics_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    scaling_rank_df: pd.DataFrame,
    packed_lens_spec: Any,
    *,
    freeze_threshold: float = 0.5,
    sample_weights: np.ndarray | None = None,
    active_population_diagnostics: dict[str, Any] | None = None,
    active_inference_likelihood: str = "blend",
) -> pd.DataFrame:
    columns = [
        "potfile_id",
        "potfile_order",
        "catalog_id",
        "catalog_row_index",
        "rank",
        "component_index",
        "catalog_mag",
        "active_magnitude_feature",
        "importance",
        "min_distance_arcsec",
        "x_centre",
        "y_centre",
        "active_gate_intercept_parameter_index",
        "active_gate_mag_slope_parameter_index",
        "active_gate_logit_offset_parameter_index",
        "active_gate_intercept_prior_kind",
        "active_gate_intercept_prior_lower",
        "active_gate_intercept_prior_upper",
        "active_gate_intercept_prior_mean",
        "active_gate_intercept_prior_std",
        "active_gate_mag_slope_prior_kind",
        "active_gate_mag_slope_prior_lower",
        "active_gate_mag_slope_prior_upper",
        "active_gate_mag_slope_prior_mean",
        "active_gate_mag_slope_prior_std",
        "p_active_gate_mean",
        "p_active_gate_median",
        "p_active_gate_p16",
        "p_active_gate_p84",
        "p_active_gate_map",
        "p_active_membership_mean",
        "p_active_membership_median",
        "p_active_membership_p16",
        "p_active_membership_p84",
        "p_active_membership_map",
        "active_loglike_delta_mean",
        "active_loglike_delta_median",
        "active_loglike_delta_p16",
        "active_loglike_delta_p84",
        "active_loglike_delta_map",
        "p_active_mean",
        "p_active_median",
        "p_active_p16",
        "p_active_p84",
        "p_active_map",
        "frozen_active",
        "active_inference_likelihood",
    ]
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    if (
        not parameter_specs
        or scaling_rank_df.empty
        or sample_array.ndim != 2
        or sample_array.shape[0] == 0
        or sample_array.shape[1] == 0
    ):
        return pd.DataFrame(columns=columns)
    n_components = int(np.asarray(getattr(packed_lens_spec, "profile_type", []), dtype=np.int32).size)
    if n_components <= 0:
        return pd.DataFrame(columns=columns)
    intercept_indices = _component_array_from_packed(
        packed_lens_spec,
        "active_gate_intercept_param_index",
        n_components,
        dtype=np.int32,
        fill_value=-1,
    )
    slope_indices = _component_array_from_packed(
        packed_lens_spec,
        "active_gate_mag_slope_param_index",
        n_components,
        dtype=np.int32,
        fill_value=-1,
    )
    offset_indices = _component_array_from_packed(
        packed_lens_spec,
        "active_gate_logit_offset_param_index",
        n_components,
        dtype=np.int32,
        fill_value=-1,
    )
    magnitude_features = _component_array_from_packed(
        packed_lens_spec,
        "active_magnitude_feature",
        n_components,
        dtype=float,
        fill_value=0.0,
    )
    weights = _normalized_weights(sample_weights, sample_array.shape[0])
    threshold = float(freeze_threshold)
    population_mode = str(active_inference_likelihood) == "population"
    population_by_component: dict[int, int] = {}
    population_membership_samples = None
    population_gate_samples = None
    population_delta_samples = None
    population_membership_map = None
    population_gate_map = None
    population_delta_map = None
    if isinstance(active_population_diagnostics, dict):
        component_values = np.asarray(active_population_diagnostics.get("component_indices", []), dtype=np.int32).reshape(-1)
        population_by_component = {int(component_index): idx for idx, component_index in enumerate(component_values.tolist())}
        population_membership_samples = np.asarray(
            active_population_diagnostics.get("membership_samples", []),
            dtype=float,
        )
        population_gate_samples = np.asarray(active_population_diagnostics.get("gate_samples", []), dtype=float)
        population_delta_samples = np.asarray(active_population_diagnostics.get("delta_samples", []), dtype=float)
        if "membership_map" in active_population_diagnostics:
            population_membership_map = np.asarray(active_population_diagnostics.get("membership_map"), dtype=float)
        if "gate_map" in active_population_diagnostics:
            population_gate_map = np.asarray(active_population_diagnostics.get("gate_map"), dtype=float)
        if "delta_map" in active_population_diagnostics:
            population_delta_map = np.asarray(active_population_diagnostics.get("delta_map"), dtype=float)

    def _prior_metadata(prefix: str, param_index: int) -> dict[str, Any]:
        if param_index < 0 or param_index >= len(parameter_specs):
            return {
                f"{prefix}_prior_kind": "",
                f"{prefix}_prior_lower": float("nan"),
                f"{prefix}_prior_upper": float("nan"),
                f"{prefix}_prior_mean": float("nan"),
                f"{prefix}_prior_std": float("nan"),
            }
        spec = parameter_specs[int(param_index)]
        mean = getattr(spec, "mean", None)
        std = getattr(spec, "std", None)
        return {
            f"{prefix}_prior_kind": str(getattr(spec, "prior_kind", "")),
            f"{prefix}_prior_lower": float(getattr(spec, "lower", float("nan"))),
            f"{prefix}_prior_upper": float(getattr(spec, "upper", float("nan"))),
            f"{prefix}_prior_mean": float(mean) if mean is not None else float("nan"),
            f"{prefix}_prior_std": float(std) if std is not None else float("nan"),
        }

    rows: list[dict[str, Any]] = []
    for row in scaling_rank_df.itertuples(index=False):
        component_index = int(getattr(row, "component_index", -1))
        if component_index < 0 or component_index >= n_components:
            continue
        intercept_idx = int(intercept_indices[component_index])
        slope_idx = int(slope_indices[component_index])
        offset_idx = int(offset_indices[component_index])
        if (
            intercept_idx < 0
            or slope_idx < 0
            or offset_idx < 0
            or intercept_idx >= sample_array.shape[1]
            or slope_idx >= sample_array.shape[1]
            or offset_idx >= sample_array.shape[1]
        ):
            continue
        feature = float(magnitude_features[component_index])
        logits = sample_array[:, intercept_idx] + sample_array[:, slope_idx] * feature + sample_array[:, offset_idx]
        gate_values = _sigmoid_np(logits)
        gate_summary = _finite_weighted_summary(gate_values, weights)
        gate_map = (
            float(
                _sigmoid_np(
                    best_fit_array[intercept_idx]
                    + best_fit_array[slope_idx] * feature
                    + best_fit_array[offset_idx]
                )
            )
            if max(intercept_idx, slope_idx, offset_idx) < best_fit_array.size
            else float("nan")
        )
        membership_values = gate_values
        membership_map = gate_map
        delta_values = np.full(sample_array.shape[0], np.nan, dtype=float)
        delta_map = float("nan")
        population_idx = population_by_component.get(int(component_index))
        if population_mode and population_idx is not None:
            if (
                population_membership_samples is not None
                and population_membership_samples.ndim == 2
                and population_membership_samples.shape[0] == sample_array.shape[0]
                and population_idx < population_membership_samples.shape[1]
            ):
                membership_values = population_membership_samples[:, population_idx]
            if (
                population_gate_samples is not None
                and population_gate_samples.ndim == 2
                and population_gate_samples.shape[0] == sample_array.shape[0]
                and population_idx < population_gate_samples.shape[1]
            ):
                gate_values = population_gate_samples[:, population_idx]
                gate_summary = _finite_weighted_summary(gate_values, weights)
            if (
                population_delta_samples is not None
                and population_delta_samples.ndim == 2
                and population_delta_samples.shape[0] == sample_array.shape[0]
                and population_idx < population_delta_samples.shape[1]
            ):
                delta_values = population_delta_samples[:, population_idx]
            if population_membership_map is not None and population_idx < population_membership_map.size:
                membership_map = float(population_membership_map[population_idx])
            if population_gate_map is not None and population_idx < population_gate_map.size:
                gate_map = float(population_gate_map[population_idx])
            if population_delta_map is not None and population_idx < population_delta_map.size:
                delta_map = float(population_delta_map[population_idx])
        membership_summary = _finite_weighted_summary(membership_values, weights)
        delta_summary = _finite_weighted_summary(delta_values, weights)
        p_median = float(membership_summary["median"])
        rows.append(
            {
                "potfile_id": str(getattr(row, "potfile_id", "")),
                "potfile_order": int(getattr(row, "potfile_order", -1)),
                "catalog_id": str(getattr(row, "catalog_id", "")),
                "catalog_row_index": int(getattr(row, "catalog_row_index", getattr(row, "row_index", -1))),
                "rank": int(getattr(row, "rank", -1)),
                "component_index": int(component_index),
                "catalog_mag": float(getattr(row, "catalog_mag", np.nan)),
                "active_magnitude_feature": feature,
                "importance": float(getattr(row, "importance", np.nan)),
                "min_distance_arcsec": float(getattr(row, "min_distance_arcsec", np.nan)),
                "x_centre": float(getattr(row, "x_centre", np.nan)),
                "y_centre": float(getattr(row, "y_centre", np.nan)),
                "active_gate_intercept_parameter_index": int(intercept_idx),
                "active_gate_mag_slope_parameter_index": int(slope_idx),
                "active_gate_logit_offset_parameter_index": int(offset_idx),
                **_prior_metadata("active_gate_intercept", int(intercept_idx)),
                **_prior_metadata("active_gate_mag_slope", int(slope_idx)),
                "p_active_gate_mean": gate_summary["mean"],
                "p_active_gate_median": gate_summary["median"],
                "p_active_gate_p16": gate_summary["p16"],
                "p_active_gate_p84": gate_summary["p84"],
                "p_active_gate_map": gate_map,
                "p_active_membership_mean": membership_summary["mean"],
                "p_active_membership_median": membership_summary["median"],
                "p_active_membership_p16": membership_summary["p16"],
                "p_active_membership_p84": membership_summary["p84"],
                "p_active_membership_map": membership_map,
                "active_loglike_delta_mean": delta_summary["mean"],
                "active_loglike_delta_median": delta_summary["median"],
                "active_loglike_delta_p16": delta_summary["p16"],
                "active_loglike_delta_p84": delta_summary["p84"],
                "active_loglike_delta_map": delta_map,
                "p_active_mean": membership_summary["mean"],
                "p_active_median": p_median,
                "p_active_p16": membership_summary["p16"],
                "p_active_p84": membership_summary["p84"],
                "p_active_map": membership_map,
                "frozen_active": bool(np.isfinite(p_median) and p_median >= threshold),
                "active_inference_likelihood": str(active_inference_likelihood),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["potfile_id", "rank"]).reset_index(drop=True)


def _prior_center_for_spec(spec: ParameterSpec) -> float | None:
    prior_kind = str(getattr(spec, "prior_kind", ""))
    if prior_kind in {"normal", "truncated_normal"}:
        mean = getattr(spec, "mean", None)
        if mean is None or not np.isfinite(float(mean)):
            return None
        return float(mean)
    if prior_kind == "uniform":
        lower = float(getattr(spec, "lower", np.nan))
        upper = float(getattr(spec, "upper", np.nan))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            return None
        return float(0.5 * (lower + upper))
    return None


def _prior_draws_for_spec(spec: ParameterSpec, rng: np.random.Generator, n_draws: int) -> np.ndarray | None:
    prior_kind = str(getattr(spec, "prior_kind", ""))
    if n_draws <= 0:
        return None
    if prior_kind == "normal":
        mean = getattr(spec, "mean", None)
        std = getattr(spec, "std", None)
        if mean is None or std is None or not np.isfinite(float(std)) or float(std) <= 0.0:
            return None
        return rng.normal(float(mean), float(std), size=int(n_draws))
    if prior_kind == "truncated_normal":
        mean = getattr(spec, "mean", None)
        std = getattr(spec, "std", None)
        if mean is None or std is None or not np.isfinite(float(std)) or float(std) <= 0.0:
            return None
        lower = float(getattr(spec, "lower", -np.inf))
        upper = float(getattr(spec, "upper", np.inf))
        low_cdf = 0.0 if not np.isfinite(lower) else float(norm.cdf((lower - float(mean)) / float(std)))
        high_cdf = 1.0 if not np.isfinite(upper) else float(norm.cdf((upper - float(mean)) / float(std)))
        if not np.isfinite(low_cdf) or not np.isfinite(high_cdf) or high_cdf <= low_cdf:
            return None
        u = rng.uniform(low_cdf, high_cdf, size=int(n_draws))
        u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        return float(mean) + float(std) * norm.ppf(u)
    if prior_kind == "uniform":
        lower = float(getattr(spec, "lower", np.nan))
        upper = float(getattr(spec, "upper", np.nan))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            return None
        return rng.uniform(lower, upper, size=int(n_draws))
    return None


def _gate_parameter_indices(
    gate_df: pd.DataFrame,
    *,
    intercept_column: str,
    slope_column: str,
) -> tuple[int, int] | None:
    if gate_df.empty or intercept_column not in gate_df or slope_column not in gate_df:
        return None
    intercept_values = pd.to_numeric(gate_df.get(intercept_column), errors="coerce").dropna().astype(int)
    slope_values = pd.to_numeric(gate_df.get(slope_column), errors="coerce").dropna().astype(int)
    intercept_values = intercept_values[intercept_values >= 0]
    slope_values = slope_values[slope_values >= 0]
    if intercept_values.empty or slope_values.empty:
        return None
    return int(intercept_values.iloc[0]), int(slope_values.iloc[0])


def _role_name_from_parameter_index_column(column_name: str) -> str:
    suffix = "_parameter_index"
    return str(column_name[: -len(suffix)] if column_name.endswith(suffix) else column_name)


def _first_string_value(frame: pd.DataFrame, column_name: str) -> str | None:
    if frame.empty or column_name not in frame:
        return None
    values = frame[column_name].dropna()
    if values.empty:
        return None
    value = str(values.iloc[0]).strip()
    return value or None


def _first_float_value(frame: pd.DataFrame, column_name: str) -> float | None:
    if frame.empty or column_name not in frame:
        return None
    values = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _prior_spec_from_metadata(frame: pd.DataFrame, role_name: str) -> Any | None:
    prior_kind = _first_string_value(frame, f"{role_name}_prior_kind")
    if prior_kind is None:
        return None
    lower = _first_float_value(frame, f"{role_name}_prior_lower")
    upper = _first_float_value(frame, f"{role_name}_prior_upper")
    mean = _first_float_value(frame, f"{role_name}_prior_mean")
    std = _first_float_value(frame, f"{role_name}_prior_std")
    return SimpleNamespace(
        prior_kind=prior_kind,
        lower=float("-inf") if lower is None else float(lower),
        upper=float("inf") if upper is None else float(upper),
        mean=mean,
        std=std,
    )


def _magnitude_feature_grid(
    feature_df: pd.DataFrame,
    *,
    feature_column: str,
    n_grid: int = 160,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feature_df.empty or "catalog_mag" not in feature_df or feature_column not in feature_df:
        return None
    mag_values = pd.to_numeric(feature_df.get("catalog_mag"), errors="coerce").to_numpy(dtype=float)
    feature_values = pd.to_numeric(feature_df.get(feature_column), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(mag_values) & np.isfinite(feature_values)
    if np.sum(finite) < 2:
        return None
    mag_min = float(np.nanmin(mag_values[finite]))
    mag_max = float(np.nanmax(mag_values[finite]))
    if not np.isfinite(mag_min) or not np.isfinite(mag_max) or mag_min == mag_max:
        return None
    coeff = np.polyfit(mag_values[finite], feature_values[finite], deg=1)
    mag_grid = np.linspace(mag_min, mag_max, int(n_grid))
    feature_grid = coeff[0] * mag_grid + coeff[1]
    return mag_grid, feature_grid


def _gate_prior_sigmoid_curve(
    feature_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    parameter_specs: list[ParameterSpec] | None,
    *,
    intercept_column: str,
    slope_column: str,
    feature_column: str,
    n_draws: int = 512,
) -> dict[str, np.ndarray] | None:
    indices = _gate_parameter_indices(gate_df, intercept_column=intercept_column, slope_column=slope_column)
    grid = _magnitude_feature_grid(feature_df, feature_column=feature_column)
    if grid is None:
        return None
    mag_grid, feature_grid = grid
    intercept_spec = None
    slope_spec = None
    intercept_seed = 0
    slope_seed = 0
    if indices is not None:
        intercept_idx, slope_idx = indices
        intercept_seed = int(intercept_idx)
        slope_seed = int(slope_idx)
        if parameter_specs and 0 <= intercept_idx < len(parameter_specs) and 0 <= slope_idx < len(parameter_specs):
            intercept_spec = parameter_specs[intercept_idx]
            slope_spec = parameter_specs[slope_idx]
    if intercept_spec is None:
        intercept_spec = _prior_spec_from_metadata(gate_df, _role_name_from_parameter_index_column(intercept_column))
    if slope_spec is None:
        slope_spec = _prior_spec_from_metadata(gate_df, _role_name_from_parameter_index_column(slope_column))
    if intercept_spec is None or slope_spec is None:
        return None
    intercept_center = _prior_center_for_spec(intercept_spec)
    slope_center = _prior_center_for_spec(slope_spec)
    if intercept_center is None or slope_center is None:
        return None
    center = _sigmoid_np(intercept_center + slope_center * feature_grid)
    rng = np.random.default_rng(71423 + 31 * int(intercept_seed) + 997 * int(slope_seed))
    intercept_draws = _prior_draws_for_spec(intercept_spec, rng, n_draws)
    slope_draws = _prior_draws_for_spec(slope_spec, rng, n_draws)
    if intercept_draws is None or slope_draws is None:
        return {"mag": mag_grid, "center": center}
    logits = intercept_draws[:, None] + slope_draws[:, None] * feature_grid[None, :]
    p_grid = _sigmoid_np(logits)
    return {
        "mag": mag_grid,
        "center": center,
        "p16": np.nanquantile(p_grid, 0.16, axis=0),
        "p84": np.nanquantile(p_grid, 0.84, axis=0),
    }


def _draw_gate_prior_sigmoid(
    ax: Any,
    feature_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    parameter_specs: list[ParameterSpec] | None,
    *,
    intercept_column: str,
    slope_column: str,
    feature_column: str,
    show_label: bool = True,
) -> bool:
    curve = _gate_prior_sigmoid_curve(
        feature_df,
        gate_df,
        parameter_specs,
        intercept_column=intercept_column,
        slope_column=slope_column,
        feature_column=feature_column,
    )
    if curve is None:
        return False
    order = np.argsort(curve["mag"])
    label_suffix = "" if show_label else "_nolegend_"
    if "p16" in curve and "p84" in curve:
        ax.fill_between(
            curve["mag"][order],
            curve["p16"][order],
            curve["p84"][order],
            color="tab:orange",
            alpha=0.14,
            linewidth=0.0,
            label="prior 16-84%" if show_label else label_suffix,
        )
    ax.plot(
        curve["mag"][order],
        curve["center"][order],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="prior center" if show_label else label_suffix,
    )
    return True


def _active_population_finite_frame(df: pd.DataFrame, numeric_columns: Sequence[str]) -> pd.DataFrame | None:
    if any(column not in df.columns for column in numeric_columns):
        return None
    frame = df.copy()
    mask = np.ones(len(frame), dtype=bool)
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        frame[column] = values
        mask &= np.isfinite(values)
    finite = frame.loc[mask].copy()
    if finite.empty:
        return None
    finite["frozen_active"] = finite.get("frozen_active", pd.Series(False, index=finite.index)).astype(bool)
    return finite


def _scatter_active_population_status(
    ax: Any,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    active_label: str = "frozen active",
    inactive_label: str = "frozen inactive",
    alpha: float = 0.82,
) -> None:
    active = frame[frame["frozen_active"]]
    inactive = frame[~frame["frozen_active"]]
    if not inactive.empty:
        ax.scatter(
            inactive[x_column],
            inactive[y_column],
            s=28,
            c="0.58",
            alpha=alpha,
            label=inactive_label,
        )
    if not active.empty:
        ax.scatter(
            active[x_column],
            active[y_column],
            s=34,
            c="tab:blue",
            edgecolors="black",
            linewidths=0.4,
            alpha=alpha,
            label=active_label,
        )


def _add_active_population_mixture_page2(pdf: PdfPages, df: pd.DataFrame, threshold: float) -> None:
    numeric_columns = [
        "p_active_gate_median",
        "p_active_membership_median",
        "active_loglike_delta_median",
    ]
    finite = _active_population_finite_frame(df, numeric_columns)
    if finite is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_gate, ax_delta_membership, ax_delta_rank, ax_hist = axes.reshape(-1)

    _scatter_active_population_status(
        ax_gate,
        finite,
        "p_active_gate_median",
        "p_active_membership_median",
    )
    ax_gate.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.0, label="y=x")
    ax_gate.axhline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_gate.axvline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_gate.set_xlim(-0.05, 1.05)
    ax_gate.set_ylim(-0.05, 1.05)
    ax_gate.set_title("Membership vs raw gate")
    ax_gate.set_xlabel("raw sigmoid gate median")
    ax_gate.set_ylabel("membership median")
    ax_gate.legend(loc="best", fontsize=8)

    _scatter_active_population_status(
        ax_delta_membership,
        finite,
        "active_loglike_delta_median",
        "p_active_membership_median",
    )
    ax_delta_membership.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_delta_membership.axhline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_delta_membership.set_ylim(-0.05, 1.05)
    ax_delta_membership.set_title("Membership vs inactive likelihood delta")
    ax_delta_membership.set_xlabel(r"$\Delta \log L = \log L_{\rm inactive} - \log L_{\rm exact}$")
    ax_delta_membership.set_ylabel("membership median")
    ax_delta_membership.text(
        0.02,
        0.02,
        "negative delta favors active",
        transform=ax_delta_membership.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax_delta_membership.legend(loc="best", fontsize=8)

    rank_frame = _active_population_finite_frame(
        finite,
        ["active_loglike_delta_median", "rank"] if "rank" in finite.columns else ["active_loglike_delta_median"],
    )
    mag_frame = _active_population_finite_frame(
        finite,
        ["active_loglike_delta_median", "catalog_mag"]
        if "catalog_mag" in finite.columns
        else ["active_loglike_delta_median"],
    )
    if rank_frame is not None and "rank" in rank_frame.columns:
        _scatter_active_population_status(ax_delta_rank, rank_frame, "rank", "active_loglike_delta_median")
        ax_delta_rank.set_xlabel("importance rank")
        ax_delta_rank.set_title("Inactive likelihood delta vs rank")
    elif mag_frame is not None and "catalog_mag" in mag_frame.columns:
        _scatter_active_population_status(ax_delta_rank, mag_frame, "catalog_mag", "active_loglike_delta_median")
        ax_delta_rank.invert_xaxis()
        ax_delta_rank.set_xlabel("catalog magnitude")
        ax_delta_rank.set_title("Inactive likelihood delta vs magnitude")
    else:
        ax_delta_rank.text(
            0.5,
            0.5,
            "rank/magnitude unavailable",
            transform=ax_delta_rank.transAxes,
            ha="center",
            va="center",
        )
        ax_delta_rank.set_title("Inactive likelihood delta")
    ax_delta_rank.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_delta_rank.set_ylabel(r"$\Delta \log L$")
    handles, labels = ax_delta_rank.get_legend_handles_labels()
    if handles and labels:
        ax_delta_rank.legend(loc="best", fontsize=8)

    active_delta = finite.loc[finite["frozen_active"], "active_loglike_delta_median"].to_numpy(dtype=float)
    inactive_delta = finite.loc[~finite["frozen_active"], "active_loglike_delta_median"].to_numpy(dtype=float)
    all_delta = finite["active_loglike_delta_median"].to_numpy(dtype=float)
    if all_delta.size <= 1 or np.nanmin(all_delta) == np.nanmax(all_delta):
        bins = 10
    else:
        bins = np.histogram_bin_edges(all_delta, bins="auto")
    ax_hist.hist(inactive_delta, bins=bins, color="0.65", alpha=0.75, label="frozen inactive")
    ax_hist.hist(active_delta, bins=bins, color="tab:blue", alpha=0.65, label="frozen active")
    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_hist.set_title("Inactive likelihood delta histogram")
    ax_hist.set_xlabel(r"$\Delta \log L$")
    ax_hist.set_ylabel("galaxies")
    ax_hist.legend(loc="best", fontsize=8)

    fig.suptitle("Population active mixture diagnostics", fontsize=13)
    _finish_pdf_page(pdf, fig, bbox_inches="tight")


def _add_active_population_mixture_page3(pdf: PdfPages, df: pd.DataFrame, threshold: float) -> None:
    numeric_columns = [
        "p_active_gate_median",
        "p_active_membership_median",
        "p_active_membership_p16",
        "p_active_membership_p84",
        "active_loglike_delta_median",
        "active_loglike_delta_p16",
        "active_loglike_delta_p84",
    ]
    finite = _active_population_finite_frame(df, numeric_columns)
    if finite is None:
        return
    finite["membership_width"] = np.maximum(
        0.0,
        finite["p_active_membership_p84"] - finite["p_active_membership_p16"],
    )
    finite["delta_width"] = np.maximum(
        0.0,
        finite["active_loglike_delta_p84"] - finite["active_loglike_delta_p16"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_membership_width, ax_delta_width, ax_gate_delta, ax_text = axes.reshape(-1)

    _scatter_active_population_status(
        ax_membership_width,
        finite,
        "p_active_membership_median",
        "membership_width",
    )
    ax_membership_width.axvline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_membership_width.set_xlim(-0.05, 1.05)
    ax_membership_width.set_title("Membership uncertainty")
    ax_membership_width.set_xlabel("membership median")
    ax_membership_width.set_ylabel("p84 - p16")
    ax_membership_width.legend(loc="best", fontsize=8)

    _scatter_active_population_status(
        ax_delta_width,
        finite,
        "active_loglike_delta_median",
        "delta_width",
    )
    ax_delta_width.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_delta_width.set_title("Likelihood-delta uncertainty")
    ax_delta_width.set_xlabel(r"$\Delta \log L$ median")
    ax_delta_width.set_ylabel("p84 - p16")
    ax_delta_width.legend(loc="best", fontsize=8)

    sc = ax_gate_delta.scatter(
        finite["active_loglike_delta_median"],
        finite["p_active_gate_median"],
        c=finite["p_active_membership_median"],
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        s=34,
        edgecolors=np.where(finite["frozen_active"].to_numpy(dtype=bool), "black", "none"),
        linewidths=np.where(finite["frozen_active"].to_numpy(dtype=bool), 0.45, 0.0),
        alpha=0.86,
    )
    ax_gate_delta.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_gate_delta.axhline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_gate_delta.set_ylim(-0.05, 1.05)
    ax_gate_delta.set_title("Gate vs inactive likelihood delta")
    ax_gate_delta.set_xlabel(r"$\Delta \log L$ median")
    ax_gate_delta.set_ylabel("raw sigmoid gate median")
    fig.colorbar(sc, ax=ax_gate_delta, label="membership median", fraction=0.045, pad=0.02)

    total = int(len(finite))
    frozen_active = int(finite["frozen_active"].sum())
    frozen_inactive = total - frozen_active
    gate_active = int((finite["p_active_gate_median"] >= threshold).sum())
    membership_active = int((finite["p_active_membership_median"] >= threshold).sum())
    summary_rows = [
        ("total candidates", f"{total:d}"),
        ("frozen active", f"{frozen_active:d}"),
        ("frozen inactive", f"{frozen_inactive:d}"),
        ("gate >= threshold", f"{gate_active:d}"),
        ("membership >= threshold", f"{membership_active:d}"),
        ("median gate", f"{np.nanmedian(finite['p_active_gate_median']):.3g}"),
        ("median membership", f"{np.nanmedian(finite['p_active_membership_median']):.3g}"),
        ("median delta", f"{np.nanmedian(finite['active_loglike_delta_median']):.3g}"),
    ]
    ax_text.axis("off")
    ax_text.set_title("Decision summary")
    row_text = "\n".join(f"{name:<24} {value:>10}" for name, value in summary_rows)
    ax_text.text(
        0.02,
        0.95,
        row_text,
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        linespacing=1.5,
    )
    ax_text.text(
        0.02,
        0.08,
        "Membership is the posterior active probability used for freeze decisions.",
        transform=ax_text.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
        wrap=True,
    )

    fig.suptitle("Population active mixture uncertainty and decisions", fontsize=13)
    _finish_pdf_page(pdf, fig, bbox_inches="tight")


def _plot_active_scaling_summary(
    plot_dir: Path,
    active_scaling_df: pd.DataFrame,
    *,
    parameter_specs: list[ParameterSpec] | None = None,
    freeze_threshold: float = 0.5,
) -> None:
    pdf_path = _plot_path(plot_dir, "active_scaling_summary.pdf")
    if active_scaling_df.empty:
        _write_placeholder_plot(pdf_path, "SVI active-scaling summary", "No SVI active-gate diagnostics are available.")
        return
    df = active_scaling_df.copy()
    df["potfile_id"] = df["potfile_id"].astype(str)
    df["frozen_active"] = df["frozen_active"].astype(bool)
    likelihood_values = (
        set(str(value) for value in df.get("active_inference_likelihood", pd.Series(dtype=object)).dropna().unique())
        if "active_inference_likelihood" in df
        else set()
    )
    population_mode = any(value.startswith("population") for value in likelihood_values)
    probability_label = "median active membership" if population_mode else "median p_active"
    posterior_label = "posterior active membership" if population_mode else "posterior p_active"
    threshold = float(freeze_threshold)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_map, ax_mag, ax_rank, ax_counts = axes.reshape(-1)

    norm = Normalize(vmin=0.0, vmax=1.0)
    finite_xy = np.isfinite(df["x_centre"].to_numpy(dtype=float)) & np.isfinite(df["y_centre"].to_numpy(dtype=float))
    finite_p = np.isfinite(df["p_active_median"].to_numpy(dtype=float))
    finite_map = df[finite_xy & finite_p]
    missing_map = df[finite_xy & ~finite_p]
    if finite_map.empty:
        ax_map.text(0.5, 0.5, "sky map unavailable", transform=ax_map.transAxes, ha="center", va="center")
    else:
        frozen = finite_map[finite_map["frozen_active"]]
        inactive = finite_map[~finite_map["frozen_active"]]
        mappable = ax_map.scatter(
            inactive["x_centre"],
            inactive["y_centre"],
            c=inactive["p_active_median"],
            cmap="viridis",
            norm=norm,
            s=34,
            marker="o",
            edgecolors="none",
            alpha=0.85,
            label="frozen inactive",
        )
        if not frozen.empty:
            mappable = ax_map.scatter(
                frozen["x_centre"],
                frozen["y_centre"],
                c=frozen["p_active_median"],
                cmap="viridis",
                norm=norm,
                s=52,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.95,
                label="frozen active",
            )
        fig.colorbar(mappable, ax=ax_map, label=probability_label, fraction=0.045, pad=0.02)
    if not missing_map.empty:
        ax_map.scatter(
            missing_map["x_centre"],
            missing_map["y_centre"],
            s=24,
            facecolors="white",
            edgecolors="0.35",
            linewidths=0.7,
            label="p_active unavailable",
        )
    ax_map.set_title("Sky map")
    ax_map.set_xlabel("x (arcsec)")
    ax_map.set_ylabel("y (arcsec)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.legend(loc="best", fontsize=8)

    finite_mag = df[
        np.isfinite(df["catalog_mag"].to_numpy(dtype=float))
        & np.isfinite(df["p_active_median"].to_numpy(dtype=float))
    ]
    if finite_mag.empty:
        ax_mag.text(0.5, 0.5, "p_active by magnitude unavailable", transform=ax_mag.transAxes, ha="center", va="center")
    else:
        prior_label_shown = False
        for _potfile_id, pot_df in finite_mag.groupby("potfile_id", sort=False):
            prior_label_shown = _draw_gate_prior_sigmoid(
                ax_mag,
                pot_df,
                pot_df,
                parameter_specs,
                intercept_column="active_gate_intercept_parameter_index",
                slope_column="active_gate_mag_slope_parameter_index",
                feature_column="active_magnitude_feature",
                show_label=not prior_label_shown,
            ) or prior_label_shown
        colors = np.where(finite_mag["frozen_active"].to_numpy(dtype=bool), "tab:blue", "0.55")
        ax_mag.scatter(finite_mag["catalog_mag"], finite_mag["p_active_median"], c=colors, s=30, alpha=0.85)
        ax_mag.errorbar(
            finite_mag["catalog_mag"],
            finite_mag["p_active_median"],
            yerr=np.vstack(
                [
                    np.maximum(0.0, finite_mag["p_active_median"] - finite_mag["p_active_p16"]),
                    np.maximum(0.0, finite_mag["p_active_p84"] - finite_mag["p_active_median"]),
                ]
            ),
            fmt="none",
            ecolor="0.35",
            elinewidth=0.8,
            alpha=0.45,
        )
    ax_mag.axhline(threshold, color="black", linestyle=":", linewidth=1.0, label=f"freeze threshold={threshold:.2f}")
    ax_mag.set_ylim(-0.05, 1.05)
    ax_mag.set_title("Probability vs magnitude")
    ax_mag.set_xlabel("catalog magnitude")
    ax_mag.set_ylabel(posterior_label)
    if population_mode:
        ax_mag.text(
            0.02,
            0.02,
            "raw sigmoid gate in CSV as p_active_gate_*",
            transform=ax_mag.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    ax_mag.invert_xaxis()
    ax_mag.legend(loc="best", fontsize=8)

    finite_rank = df[
        np.isfinite(df["rank"].to_numpy(dtype=float))
        & np.isfinite(df["p_active_median"].to_numpy(dtype=float))
    ]
    if finite_rank.empty:
        ax_rank.text(0.5, 0.5, "p_active by rank unavailable", transform=ax_rank.transAxes, ha="center", va="center")
    else:
        for potfile_id, pot_df in finite_rank.groupby("potfile_id", sort=False):
            ax_rank.plot(
                pot_df["rank"],
                pot_df["p_active_median"],
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                label=str(potfile_id),
            )
    ax_rank.axhline(threshold, color="black", linestyle=":", linewidth=1.0)
    ax_rank.set_ylim(-0.05, 1.05)
    ax_rank.set_title("Probability vs rank")
    ax_rank.set_xlabel("importance rank")
    ax_rank.set_ylabel(probability_label)
    ax_rank.legend(loc="best", fontsize=8)

    counts = (
        df.groupby(["potfile_id", "frozen_active"], observed=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "inactive", True: "active"})
    )
    for column in ("inactive", "active"):
        if column not in counts:
            counts[column] = 0
    counts = counts[["inactive", "active"]]
    x = np.arange(len(counts), dtype=float)
    ax_counts.bar(x, counts["inactive"], color="0.72", label="frozen inactive")
    ax_counts.bar(x, counts["active"], bottom=counts["inactive"], color="tab:blue", label="frozen active")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels([str(value) for value in counts.index], rotation=25, ha="right")
    ax_counts.set_title("Frozen counts")
    ax_counts.set_ylabel("galaxies")
    ax_counts.legend(loc="best", fontsize=8)

    with PdfPages(pdf_path) as pdf:
        _finish_pdf_page(pdf, fig, bbox_inches="tight")
        if population_mode:
            _add_active_population_mixture_page2(pdf, df, threshold)
            _add_active_population_mixture_page3(pdf, df, threshold)
        return


def _independent_scaling_diagnostics_table(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    scaling_rank_df: pd.DataFrame,
    packed_lens_spec: Any,
    sample_weights: np.ndarray | None = None,
    log_sigma_offset_width: float = 0.10,
    log_mass_offset_width: float = 0.20,
) -> pd.DataFrame:
    columns = [
        "potfile_id",
        "catalog_id",
        "rank",
        "component_index",
        "free_component_index",
        "catalog_mag",
        "independent_magnitude_feature",
        "scaling_v_disp_median",
        "scaling_v_disp_p16",
        "scaling_v_disp_p84",
        "scaling_v_disp_map",
        "scaling_core_radius_kpc_median",
        "scaling_core_radius_kpc_p16",
        "scaling_core_radius_kpc_p84",
        "scaling_core_radius_kpc_map",
        "scaling_cut_radius_kpc_median",
        "scaling_cut_radius_kpc_p16",
        "scaling_cut_radius_kpc_p84",
        "scaling_cut_radius_kpc_map",
        "free_v_disp_median",
        "free_v_disp_p16",
        "free_v_disp_p84",
        "free_v_disp_map",
        "free_core_radius_kpc_median",
        "free_core_radius_kpc_p16",
        "free_core_radius_kpc_p84",
        "free_core_radius_kpc_map",
        "free_cut_radius_kpc_median",
        "free_cut_radius_kpc_p16",
        "free_cut_radius_kpc_p84",
        "free_cut_radius_kpc_map",
        "delta_log_sigma_median",
        "delta_log_sigma_p16",
        "delta_log_sigma_p84",
        "delta_log_sigma_map",
        "delta_log_mass_median",
        "delta_log_mass_p16",
        "delta_log_mass_p84",
        "delta_log_mass_map",
        "log_sigma_offset_width",
        "log_mass_offset_width",
        "sigma_ratio_median",
        "sigma_ratio_p16",
        "sigma_ratio_p84",
        "sigma_ratio_map",
        "mass_ratio_median",
        "mass_ratio_p16",
        "mass_ratio_p84",
        "mass_ratio_map",
        "radius_ratio_median",
        "radius_ratio_p16",
        "radius_ratio_p84",
        "radius_ratio_map",
        "core_ratio_median",
        "core_ratio_p16",
        "core_ratio_p84",
        "core_ratio_map",
        "cut_ratio_median",
        "cut_ratio_p16",
        "cut_ratio_p84",
        "cut_ratio_map",
    ]
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    if (
        not parameter_specs
        or scaling_rank_df.empty
        or sample_array.ndim != 2
        or sample_array.shape[0] == 0
        or sample_array.shape[1] == 0
    ):
        return pd.DataFrame(columns=columns)
    selected_independent = scaling_rank_df.get(
        "selected_independent",
        pd.Series(False, index=scaling_rank_df.index),
    ).astype(bool)
    selected_df = scaling_rank_df.loc[selected_independent].copy()
    if selected_df.empty:
        return pd.DataFrame(columns=columns)
    n_components = int(np.asarray(getattr(packed_lens_spec, "profile_type", []), dtype=np.int32).size)
    if n_components <= 0:
        return pd.DataFrame(columns=columns)
    magnitude_features = _component_array_from_packed(
        packed_lens_spec,
        "independent_magnitude_feature",
        n_components,
        dtype=float,
        fill_value=0.0,
    )
    luminosity_ratio = _component_array_from_packed(
        packed_lens_spec,
        "luminosity_ratio",
        n_components,
        dtype=float,
        fill_value=1.0,
    )
    sigma_ref_base = _component_array_from_packed(packed_lens_spec, "sigma_ref_base", n_components, dtype=float, fill_value=0.0)
    cut_ref_base = _component_array_from_packed(packed_lens_spec, "cut_ref_base", n_components, dtype=float, fill_value=0.0)
    core_ref_base = _component_array_from_packed(packed_lens_spec, "core_ref_base", n_components, dtype=float, fill_value=0.0)
    alpha_sigma_base = _component_array_from_packed(
        packed_lens_spec, "alpha_sigma_base", n_components, dtype=float, fill_value=0.25
    )
    gamma_ml_base = _component_array_from_packed(
        packed_lens_spec, "gamma_ml_base", n_components, dtype=float, fill_value=0.2
    )
    sigma_ref_indices = _component_array_from_packed(
        packed_lens_spec, "sigma_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    cut_ref_indices = _component_array_from_packed(
        packed_lens_spec, "cut_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    core_ref_indices = _component_array_from_packed(
        packed_lens_spec, "core_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    alpha_sigma_indices = _component_array_from_packed(
        packed_lens_spec, "alpha_sigma_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    gamma_ml_indices = _component_array_from_packed(
        packed_lens_spec, "gamma_ml_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    v_disp_base = _component_array_from_packed(packed_lens_spec, "v_disp_base", n_components, dtype=float, fill_value=0.0)
    core_base = _component_array_from_packed(packed_lens_spec, "core_radius_kpc_base", n_components, dtype=float, fill_value=0.0)
    cut_base = _component_array_from_packed(packed_lens_spec, "cut_radius_kpc_base", n_components, dtype=float, fill_value=0.0)
    v_disp_indices = _component_array_from_packed(
        packed_lens_spec, "v_disp_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    core_indices = _component_array_from_packed(
        packed_lens_spec, "core_radius_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    cut_radius_indices = _component_array_from_packed(
        packed_lens_spec, "cut_radius_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    free_sigma_delta_indices = _component_array_from_packed(
        packed_lens_spec, "independent_free_log_sigma_delta_unit_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    free_mass_delta_indices = _component_array_from_packed(
        packed_lens_spec, "independent_free_log_mass_delta_unit_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    sigma_offset_width = float(log_sigma_offset_width)
    mass_offset_width = float(log_mass_offset_width)
    weights = _normalized_weights(sample_weights, sample_array.shape[0])

    def _values(base_array: np.ndarray, index_array: np.ndarray, component_index: int) -> np.ndarray:
        idx = int(index_array[component_index])
        if 0 <= idx < sample_array.shape[1]:
            return sample_array[:, idx]
        return np.full(sample_array.shape[0], float(base_array[component_index]), dtype=float)

    def _map_value(base_array: np.ndarray, index_array: np.ndarray, component_index: int) -> float:
        idx = int(index_array[component_index])
        if 0 <= idx < best_fit_array.size:
            return float(best_fit_array[idx])
        return float(base_array[component_index])

    def _add_summary(row_dict: dict[str, Any], prefix: str, values: np.ndarray, map_value: float) -> None:
        summary = _finite_weighted_summary(values, weights)
        row_dict[f"{prefix}_median"] = summary["median"]
        row_dict[f"{prefix}_p16"] = summary["p16"]
        row_dict[f"{prefix}_p84"] = summary["p84"]
        row_dict[f"{prefix}_map"] = float(map_value) if np.isfinite(map_value) else float("nan")

    def _effective_exponents(component_index: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        alpha_values = _values(alpha_sigma_base, alpha_sigma_indices, component_index)
        gamma_values = _values(gamma_ml_base, gamma_ml_indices, component_index)
        alpha_map = _map_value(alpha_sigma_base, alpha_sigma_indices, component_index)
        gamma_map = _map_value(gamma_ml_base, gamma_ml_indices, component_index)
        beta_values = 1.0 + gamma_values - 2.0 * alpha_values
        beta_map = 1.0 + gamma_map - 2.0 * alpha_map
        return alpha_values, beta_values, alpha_map, beta_map

    rows: list[dict[str, Any]] = []
    for row in selected_df.itertuples(index=False):
        component_index = int(getattr(row, "component_index"))
        if component_index < 0 or component_index >= n_components:
            continue
        free_component_index = int(getattr(row, "free_component_index", -1))
        if free_component_index < 0 or free_component_index >= n_components:
            continue
        feature = float(magnitude_features[component_index])

        sigma_ref = _values(sigma_ref_base, sigma_ref_indices, component_index)
        cut_ref = _values(cut_ref_base, cut_ref_indices, component_index)
        core_ref = _values(core_ref_base, core_ref_indices, component_index)
        alpha_sigma, beta_radius, alpha_sigma_map, beta_radius_map = _effective_exponents(component_index)
        lum = float(luminosity_ratio[component_index])
        size_luminosity_scale = np.power(lum, beta_radius)
        scaling_v_disp = sigma_ref * np.power(lum, alpha_sigma)
        scaling_core = core_ref * size_luminosity_scale
        scaling_cut = cut_ref * size_luminosity_scale

        sigma_ref_map = _map_value(sigma_ref_base, sigma_ref_indices, component_index)
        cut_ref_map = _map_value(cut_ref_base, cut_ref_indices, component_index)
        core_ref_map = _map_value(core_ref_base, core_ref_indices, component_index)
        size_luminosity_scale_map = float(np.power(lum, beta_radius_map))
        scaling_v_disp_map = sigma_ref_map * float(np.power(lum, alpha_sigma_map))
        scaling_core_map = core_ref_map * size_luminosity_scale_map
        scaling_cut_map = cut_ref_map * size_luminosity_scale_map

        log_displacement_free = int(free_sigma_delta_indices[free_component_index]) >= 0
        if log_displacement_free:
            free_sigma_delta_unit = _values(np.zeros(n_components, dtype=float), free_sigma_delta_indices, free_component_index)
            free_mass_delta_unit = _values(np.zeros(n_components, dtype=float), free_mass_delta_indices, free_component_index)
            delta_log_sigma = sigma_offset_width * free_sigma_delta_unit
            delta_log_mass = mass_offset_width * free_mass_delta_unit
            delta_log_radius = delta_log_mass - 2.0 * delta_log_sigma
            free_v_disp = scaling_v_disp * np.exp(delta_log_sigma)
            free_core = scaling_core * np.exp(delta_log_radius)
            free_cut = scaling_cut * np.exp(delta_log_radius)
            free_sigma_delta_unit_map = _map_value(np.zeros(n_components, dtype=float), free_sigma_delta_indices, free_component_index)
            free_mass_delta_unit_map = _map_value(np.zeros(n_components, dtype=float), free_mass_delta_indices, free_component_index)
            delta_log_sigma_map = sigma_offset_width * free_sigma_delta_unit_map
            delta_log_mass_map = mass_offset_width * free_mass_delta_unit_map
            delta_log_radius_map = delta_log_mass_map - 2.0 * delta_log_sigma_map
            free_v_disp_map = scaling_v_disp_map * float(np.exp(delta_log_sigma_map))
            free_core_map = scaling_core_map * float(np.exp(delta_log_radius_map))
            free_cut_map = scaling_cut_map * float(np.exp(delta_log_radius_map))
        else:
            free_v_disp = _values(v_disp_base, v_disp_indices, free_component_index)
            free_core = _values(core_base, core_indices, free_component_index)
            free_cut = _values(cut_base, cut_radius_indices, free_component_index)
            free_v_disp_map = _map_value(v_disp_base, v_disp_indices, free_component_index)
            free_core_map = _map_value(core_base, core_indices, free_component_index)
            free_cut_map = _map_value(cut_base, cut_radius_indices, free_component_index)
            delta_log_sigma = np.full(sample_array.shape[0], np.nan, dtype=float)
            delta_log_mass = np.full(sample_array.shape[0], np.nan, dtype=float)
            delta_log_radius = np.full(sample_array.shape[0], np.nan, dtype=float)
            delta_log_sigma_map = delta_log_mass_map = delta_log_radius_map = float("nan")

        row_dict: dict[str, Any] = {
            "potfile_id": str(getattr(row, "potfile_id")),
            "catalog_id": str(getattr(row, "catalog_id")),
            "rank": int(getattr(row, "rank")),
            "component_index": component_index,
            "free_component_index": free_component_index,
            "catalog_mag": float(getattr(row, "catalog_mag", np.nan)),
            "independent_magnitude_feature": feature,
            "free_v_disp_parameter_index": int(v_disp_indices[free_component_index]),
            "free_core_radius_kpc_parameter_index": int(core_indices[free_component_index]),
            "free_cut_radius_kpc_parameter_index": int(cut_radius_indices[free_component_index]),
        }
        _add_summary(row_dict, "scaling_v_disp", scaling_v_disp, scaling_v_disp_map)
        _add_summary(row_dict, "scaling_core_radius_kpc", scaling_core, scaling_core_map)
        _add_summary(row_dict, "scaling_cut_radius_kpc", scaling_cut, scaling_cut_map)
        _add_summary(row_dict, "free_v_disp", free_v_disp, free_v_disp_map)
        _add_summary(row_dict, "free_core_radius_kpc", free_core, free_core_map)
        _add_summary(row_dict, "free_cut_radius_kpc", free_cut, free_cut_map)
        _add_summary(row_dict, "delta_log_sigma", delta_log_sigma, delta_log_sigma_map)
        _add_summary(row_dict, "delta_log_mass", delta_log_mass, delta_log_mass_map)
        row_dict["log_sigma_offset_width"] = sigma_offset_width
        row_dict["log_mass_offset_width"] = mass_offset_width
        _add_summary(row_dict, "sigma_ratio", free_v_disp / scaling_v_disp, free_v_disp_map / scaling_v_disp_map)
        _add_summary(row_dict, "mass_ratio", np.exp(delta_log_mass), float(np.exp(delta_log_mass_map)))
        _add_summary(row_dict, "radius_ratio", np.exp(delta_log_radius), float(np.exp(delta_log_radius_map)))
        _add_summary(row_dict, "core_ratio", free_core / scaling_core, free_core_map / scaling_core_map)
        _add_summary(row_dict, "cut_ratio", free_cut / scaling_cut, free_cut_map / scaling_cut_map)
        rows.append(row_dict)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["potfile_id", "rank"]).reset_index(drop=True)


def _independent_scaling_plot_table(
    scaling_rank_df: pd.DataFrame,
    independent_scaling_df: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "potfile_id",
        "catalog_id",
        "rank",
        "component_index",
        "free_component_index",
        "catalog_mag",
        "independent_magnitude_feature",
        "x_centre",
        "y_centre",
        "selected_active",
        "selected_independent",
        "requested_active_count",
        "importance",
        "min_distance_arcsec",
        "independent_plot_class",
    ]
    posterior_columns = [
        "scaling_v_disp_median",
        "scaling_core_radius_kpc_median",
        "scaling_cut_radius_kpc_median",
        "free_v_disp_median",
        "free_core_radius_kpc_median",
        "free_cut_radius_kpc_median",
        "sigma_ratio_median",
        "sigma_ratio_p16",
        "sigma_ratio_p84",
        "mass_ratio_median",
        "mass_ratio_p16",
        "mass_ratio_p84",
        "radius_ratio_median",
        "radius_ratio_p16",
        "radius_ratio_p84",
        "core_ratio_median",
        "core_ratio_p16",
        "core_ratio_p84",
        "cut_ratio_median",
        "cut_ratio_p16",
        "cut_ratio_p84",
    ]
    if scaling_rank_df.empty:
        return pd.DataFrame(columns=base_columns + posterior_columns)
    plot_df = scaling_rank_df.copy()
    plot_df["potfile_id"] = plot_df["potfile_id"].astype(str)
    plot_df["catalog_id"] = plot_df["catalog_id"].astype(str)
    plot_df["component_index"] = plot_df["component_index"].astype(int)
    if "free_component_index" not in plot_df:
        plot_df["free_component_index"] = -1
    if "catalog_mag" not in plot_df:
        plot_df["catalog_mag"] = np.nan
    if "independent_magnitude_feature" not in plot_df:
        plot_df["independent_magnitude_feature"] = np.nan
    if "requested_active_count" not in plot_df:
        plot_df["requested_active_count"] = 0
    if "importance" not in plot_df:
        plot_df["importance"] = np.nan
    if "min_distance_arcsec" not in plot_df:
        plot_df["min_distance_arcsec"] = np.nan
    if "selected_active" not in plot_df:
        plot_df["selected_active"] = False
    if "selected_independent" not in plot_df:
        plot_df["selected_independent"] = False
    plot_df["selected_active"] = plot_df["selected_active"].astype(bool)
    plot_df["selected_independent"] = plot_df["selected_independent"].astype(bool)
    for column in posterior_columns:
        plot_df[column] = np.nan
    if not independent_scaling_df.empty:
        diag_df = independent_scaling_df.copy()
        diag_df["potfile_id"] = diag_df["potfile_id"].astype(str)
        diag_df["component_index"] = diag_df["component_index"].astype(int)
        merge_columns = ["potfile_id", "component_index"] + [
            column for column in posterior_columns if column in diag_df.columns
        ]
        plot_df = plot_df.drop(columns=[column for column in posterior_columns if column in plot_df.columns]).merge(
            diag_df[merge_columns],
            on=["potfile_id", "component_index"],
            how="left",
        )
        for column in posterior_columns:
            if column not in plot_df:
                plot_df[column] = np.nan
    plot_df["independent_plot_class"] = np.where(
        plot_df["selected_independent"],
        "independent_candidate",
        np.where(plot_df["selected_active"], "active_not_independent", "not_sampled"),
    )
    return plot_df.sort_values(["potfile_id", "rank"]).reset_index(drop=True)


FREE_GALAXY_SHAPE_COMPARISON_COLUMNS = [
    "member_type",
    "potfile_id",
    "catalog_id",
    "component_index",
    "catalog_component_index",
    "model_component_index",
    "catalog_ellipticite",
    "model_ellipticite",
    "delta_ellipticite",
    "catalog_angle_pos_deg",
    "model_angle_pos_deg",
    "delta_angle_pos_deg",
]


def _axis_ratio_to_lenstool_ellipticite_from_modulus(modulus: float) -> float:
    safe_modulus = min(max(float(modulus), 0.0), 1.0 - 1.0e-12)
    q = (1.0 - safe_modulus) / (1.0 + safe_modulus)
    q = min(max(float(q), 1.0e-3), 1.0)
    return float((1.0 - q * q) / (1.0 + q * q))


def _periodic_angle_deg(angle_deg: float) -> float:
    if not np.isfinite(float(angle_deg)):
        return float("nan")
    return float(((float(angle_deg) + 90.0) % 180.0) - 90.0)


def _wrapped_position_angle_delta_deg(model_angle_deg: float, catalog_angle_deg: float) -> float:
    if not np.isfinite(float(model_angle_deg)) or not np.isfinite(float(catalog_angle_deg)):
        return float("nan")
    return _periodic_angle_deg(float(model_angle_deg) - float(catalog_angle_deg))


def _e1e2_to_lenstool_shape(e1: float, e2: float) -> tuple[float, float]:
    e1_f = float(e1)
    e2_f = float(e2)
    if not np.isfinite(e1_f) or not np.isfinite(e2_f):
        return float("nan"), float("nan")
    modulus = float(np.hypot(e1_f, e2_f))
    ellipticite = _axis_ratio_to_lenstool_ellipticite_from_modulus(modulus)
    angle_pos = _periodic_angle_deg(0.5 * np.rad2deg(np.arctan2(e2_f, e1_f)))
    return ellipticite, angle_pos


def _packed_component_shape(
    packed_lens_spec: Any,
    best_fit: np.ndarray,
    component_index: int,
) -> tuple[float, float]:
    n_components = int(np.asarray(getattr(packed_lens_spec, "profile_type", []), dtype=np.int32).size)
    if component_index < 0 or component_index >= n_components:
        return float("nan"), float("nan")
    e1_base = _component_array_from_packed(packed_lens_spec, "e1_base", n_components, dtype=float, fill_value=0.0)
    e2_base = _component_array_from_packed(packed_lens_spec, "e2_base", n_components, dtype=float, fill_value=0.0)
    e1_indices = _component_array_from_packed(packed_lens_spec, "e1_param_index", n_components, dtype=np.int32, fill_value=-1)
    e2_indices = _component_array_from_packed(packed_lens_spec, "e2_param_index", n_components, dtype=np.int32, fill_value=-1)
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    e1 = float(e1_base[component_index])
    e2 = float(e2_base[component_index])
    e1_index = int(e1_indices[component_index])
    e2_index = int(e2_indices[component_index])
    if 0 <= e1_index < best_fit_array.size:
        e1 = float(best_fit_array[e1_index])
    if 0 <= e2_index < best_fit_array.size:
        e2 = float(best_fit_array[e2_index])
    return _e1e2_to_lenstool_shape(e1, e2)


def _free_galaxy_shape_row(
    *,
    member_type: str,
    potfile_id: str,
    catalog_id: str,
    component_index: int,
    catalog_component_index: int,
    model_component_index: int,
    packed_lens_spec: Any,
    best_fit: np.ndarray,
) -> dict[str, Any] | None:
    catalog_ellipticite, catalog_angle = _packed_component_shape(
        packed_lens_spec,
        np.empty((0,), dtype=float),
        int(catalog_component_index),
    )
    model_ellipticite, model_angle = _packed_component_shape(
        packed_lens_spec,
        best_fit,
        int(model_component_index),
    )
    if not (
        np.isfinite(catalog_ellipticite)
        and np.isfinite(catalog_angle)
        and np.isfinite(model_ellipticite)
        and np.isfinite(model_angle)
    ):
        return None
    return {
        "member_type": str(member_type),
        "potfile_id": str(potfile_id),
        "catalog_id": str(catalog_id),
        "component_index": int(component_index),
        "catalog_component_index": int(catalog_component_index),
        "model_component_index": int(model_component_index),
        "catalog_ellipticite": float(catalog_ellipticite),
        "model_ellipticite": float(model_ellipticite),
        "delta_ellipticite": float(model_ellipticite - catalog_ellipticite),
        "catalog_angle_pos_deg": float(catalog_angle),
        "model_angle_pos_deg": float(model_angle),
        "delta_angle_pos_deg": _wrapped_position_angle_delta_deg(model_angle, catalog_angle),
    }


def _free_galaxy_shape_comparison_table(
    state: BuildState,
    best_fit: np.ndarray,
) -> pd.DataFrame:
    packed_lens_spec = getattr(state, "packed_lens_spec", None)
    if packed_lens_spec is None:
        return pd.DataFrame(columns=FREE_GALAXY_SHAPE_COMPARISON_COLUMNS)
    rows: list[dict[str, Any]] = []
    for record in getattr(state, "scaling_component_records", []) or []:
        if not isinstance(record, dict):
            continue
        selected = bool(record.get("selected_independent", False))
        free_component_index = int(record.get("free_component_index", -1))
        if not selected and free_component_index < 0:
            continue
        component_index = int(record.get("component_index", -1))
        if free_component_index < 0:
            free_component_index = component_index
        row = _free_galaxy_shape_row(
            member_type="selected_scaling_free",
            potfile_id=str(record.get("potfile_id", "")),
            catalog_id=str(record.get("catalog_id", component_index)),
            component_index=component_index,
            catalog_component_index=component_index,
            model_component_index=free_component_index,
            packed_lens_spec=packed_lens_spec,
            best_fit=best_fit,
        )
        if row is not None:
            rows.append(row)

    for component_index, component in enumerate(getattr(state, "base_components", []) or []):
        if not isinstance(component, dict) or "independent_member_catalog_id" not in component:
            continue
        row = _free_galaxy_shape_row(
            member_type="independent_member_halo",
            potfile_id=str(component.get("independent_member_population_id", "")),
            catalog_id=str(component.get("independent_member_catalog_id", component.get("id", component_index))),
            component_index=int(component_index),
            catalog_component_index=int(component_index),
            model_component_index=int(component_index),
            packed_lens_spec=packed_lens_spec,
            best_fit=best_fit,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FREE_GALAXY_SHAPE_COMPARISON_COLUMNS)
    return pd.DataFrame(rows, columns=FREE_GALAXY_SHAPE_COMPARISON_COLUMNS).sort_values(
        ["member_type", "potfile_id", "catalog_id"],
    ).reset_index(drop=True)


def _plot_free_galaxy_shape_comparison(shape_df: pd.DataFrame, path: Path) -> None:
    if shape_df is None or shape_df.empty:
        _write_placeholder_plot(
            path,
            "Free Galaxy Shape Comparison",
            "No free member galaxy shape diagnostics are available.",
        )
        return
    df = shape_df.copy()
    for column in [
        "catalog_ellipticite",
        "model_ellipticite",
        "catalog_angle_pos_deg",
        "model_angle_pos_deg",
    ]:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    finite_shape = np.isfinite(df["catalog_ellipticite"]) & np.isfinite(df["model_ellipticite"])
    finite_angle = np.isfinite(df["catalog_angle_pos_deg"]) & np.isfinite(df["model_angle_pos_deg"])
    if not finite_shape.any() and not finite_angle.any():
        _write_placeholder_plot(
            path,
            "Free Galaxy Shape Comparison",
            "No finite catalog/model free-member shapes are available.",
        )
        return

    styles = {
        "selected_scaling_free": {"marker": "D", "color": "tab:blue", "label": "selected scaling free"},
        "independent_member_halo": {"marker": "s", "color": "tab:orange", "label": "independent member halo"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    for member_type, style in styles.items():
        mask = df["member_type"].astype(str) == member_type
        shape_mask = mask & finite_shape
        if shape_mask.any():
            axes[0].scatter(
                df.loc[shape_mask, "catalog_ellipticite"],
                df.loc[shape_mask, "model_ellipticite"],
                marker=style["marker"],
                color=style["color"],
                edgecolors="white",
                linewidths=0.45,
                s=48,
                alpha=0.85,
                label=style["label"],
            )
        angle_mask = mask & finite_angle
        if angle_mask.any():
            axes[1].scatter(
                df.loc[angle_mask, "catalog_angle_pos_deg"],
                df.loc[angle_mask, "model_angle_pos_deg"],
                marker=style["marker"],
                color=style["color"],
                edgecolors="white",
                linewidths=0.45,
                s=48,
                alpha=0.85,
                label=style["label"],
            )

    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("catalog ellipticite")
    axes[0].set_ylabel("model ellipticite")
    axes[0].set_title("Ellipticity")

    axes[1].plot([-90.0, 90.0], [-90.0, 90.0], color="0.25", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(-95.0, 95.0)
    axes[1].set_ylim(-95.0, 95.0)
    axes[1].set_xlabel("catalog angle_pos (deg)")
    axes[1].set_ylabel("model angle_pos (deg)")
    axes[1].set_title("Position Angle")

    if len(df) <= 30:
        for row in df.itertuples(index=False):
            label = str(getattr(row, "catalog_id", ""))
            if np.isfinite(float(getattr(row, "catalog_ellipticite"))) and np.isfinite(float(getattr(row, "model_ellipticite"))):
                axes[0].annotate(
                    label,
                    (float(getattr(row, "catalog_ellipticite")), float(getattr(row, "model_ellipticite"))),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
            if np.isfinite(float(getattr(row, "catalog_angle_pos_deg"))) and np.isfinite(float(getattr(row, "model_angle_pos_deg"))):
                axes[1].annotate(
                    label,
                    (float(getattr(row, "catalog_angle_pos_deg")), float(getattr(row, "model_angle_pos_deg"))),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )

    for ax in axes:
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _scaling_relation_summary_table(
    scaling_rank_df: pd.DataFrame,
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    packed_lens_spec: Any,
    *,
    sample_weights: np.ndarray | None = None,
    independent_scaling_df: pd.DataFrame | None = None,
    best_value: str | None = None,
    best_value_requested: str | None = None,
) -> pd.DataFrame:
    columns = [
        "potfile_id",
        "catalog_id",
        "best_value",
        "best_value_requested",
        "rank",
        "component_index",
        "free_component_index",
        "catalog_mag",
        "catalog_color",
        "luminosity_ratio",
        "anchor_mag",
        "alpha_sigma_median",
        "alpha_sigma_p16",
        "alpha_sigma_p84",
        "alpha_sigma_best",
        "beta_radius_median",
        "beta_radius_p16",
        "beta_radius_p84",
        "beta_radius_best",
        "gamma_ml_median",
        "gamma_ml_p16",
        "gamma_ml_p84",
        "gamma_ml_best",
        "log_softening_length_kpc_median",
        "log_softening_length_kpc_p16",
        "log_softening_length_kpc_p84",
        "log_softening_length_kpc_best",
        "softening_length_kpc_median",
        "softening_length_kpc_p16",
        "softening_length_kpc_p84",
        "softening_length_kpc_best",
        "selected_active",
        "selected_independent",
        "scaling_relation_class",
        "scaling_v_disp_median",
        "scaling_v_disp_p16",
        "scaling_v_disp_p84",
        "scaling_v_disp_best",
        "scaling_core_radius_kpc_median",
        "scaling_core_radius_kpc_p16",
        "scaling_core_radius_kpc_p84",
        "scaling_core_radius_kpc_best",
        "scaling_core_radius_effective_kpc_median",
        "scaling_core_radius_effective_kpc_p16",
        "scaling_core_radius_effective_kpc_p84",
        "scaling_core_radius_effective_kpc_best",
        "scaling_cut_radius_kpc_median",
        "scaling_cut_radius_kpc_p16",
        "scaling_cut_radius_kpc_p84",
        "scaling_cut_radius_kpc_best",
        "scaling_log10_mass_msun_median",
        "scaling_log10_mass_msun_p16",
        "scaling_log10_mass_msun_p84",
        "scaling_log10_mass_msun_best",
        "free_v_disp_median",
        "free_v_disp_p16",
        "free_v_disp_p84",
        "free_v_disp_best",
        "free_core_radius_kpc_median",
        "free_core_radius_kpc_p16",
        "free_core_radius_kpc_p84",
        "free_core_radius_kpc_best",
        "free_core_radius_effective_kpc_median",
        "free_core_radius_effective_kpc_p16",
        "free_core_radius_effective_kpc_p84",
        "free_core_radius_effective_kpc_best",
        "free_cut_radius_kpc_median",
        "free_cut_radius_kpc_p16",
        "free_cut_radius_kpc_p84",
        "free_cut_radius_kpc_best",
        "free_log10_mass_msun_median",
        "free_log10_mass_msun_p16",
        "free_log10_mass_msun_p84",
        "free_log10_mass_msun_best",
    ]
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    if (
        not parameter_specs
        or scaling_rank_df.empty
        or sample_array.ndim != 2
        or sample_array.shape[0] == 0
        or sample_array.shape[1] == 0
    ):
        return pd.DataFrame(columns=columns)
    n_components = int(np.asarray(getattr(packed_lens_spec, "profile_type", []), dtype=np.int32).size)
    if n_components <= 0:
        return pd.DataFrame(columns=columns)
    luminosity_ratio = _component_array_from_packed(
        packed_lens_spec,
        "luminosity_ratio",
        n_components,
        dtype=float,
        fill_value=1.0,
    )
    sigma_ref_base = _component_array_from_packed(packed_lens_spec, "sigma_ref_base", n_components, dtype=float, fill_value=0.0)
    cut_ref_base = _component_array_from_packed(packed_lens_spec, "cut_ref_base", n_components, dtype=float, fill_value=0.0)
    core_ref_base = _component_array_from_packed(packed_lens_spec, "core_ref_base", n_components, dtype=float, fill_value=0.0)
    alpha_sigma_base = _component_array_from_packed(
        packed_lens_spec, "alpha_sigma_base", n_components, dtype=float, fill_value=0.25
    )
    gamma_ml_base = _component_array_from_packed(
        packed_lens_spec, "gamma_ml_base", n_components, dtype=float, fill_value=0.2
    )
    sigma_ref_indices = _component_array_from_packed(
        packed_lens_spec, "sigma_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    cut_ref_indices = _component_array_from_packed(
        packed_lens_spec, "cut_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    core_ref_indices = _component_array_from_packed(
        packed_lens_spec, "core_ref_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    alpha_sigma_indices = _component_array_from_packed(
        packed_lens_spec, "alpha_sigma_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    gamma_ml_indices = _component_array_from_packed(
        packed_lens_spec, "gamma_ml_param_index", n_components, dtype=np.int32, fill_value=-1
    )
    weights = _normalized_weights(sample_weights, sample_array.shape[0])
    log_softening_index = next(
        (
            idx
            for idx, spec in enumerate(parameter_specs)
            if getattr(spec, "sample_name", "") == "log_softening_length_kpc"
        ),
        -1,
    )
    if 0 <= log_softening_index < sample_array.shape[1]:
        log_softening_values = sample_array[:, log_softening_index]
        log_softening_best = (
            float(best_fit_array[log_softening_index])
            if log_softening_index < best_fit_array.size
            else float("nan")
        )
        softening_values = np.exp(log_softening_values)
        softening_best = float(np.exp(log_softening_best)) if np.isfinite(log_softening_best) else float("nan")
    else:
        log_softening_values = np.full(sample_array.shape[0], np.nan, dtype=float)
        log_softening_best = float("nan")
        softening_values = np.zeros(sample_array.shape[0], dtype=float)
        softening_best = 0.0

    def _values(base_array: np.ndarray, index_array: np.ndarray, component_index: int) -> np.ndarray:
        idx = int(index_array[component_index])
        if 0 <= idx < sample_array.shape[1]:
            return sample_array[:, idx]
        return np.full(sample_array.shape[0], float(base_array[component_index]), dtype=float)

    def _best_value(base_array: np.ndarray, index_array: np.ndarray, component_index: int) -> float:
        idx = int(index_array[component_index])
        if 0 <= idx < best_fit_array.size:
            return float(best_fit_array[idx])
        return float(base_array[component_index])

    def _add_summary(row_dict: dict[str, Any], prefix: str, values: np.ndarray, selected_best_value: float) -> None:
        summary = _finite_weighted_summary(values, weights)
        row_dict[f"{prefix}_median"] = summary["median"]
        row_dict[f"{prefix}_p16"] = summary["p16"]
        row_dict[f"{prefix}_p84"] = summary["p84"]
        row_dict[f"{prefix}_best"] = (
            float(selected_best_value) if np.isfinite(selected_best_value) else float("nan")
        )

    def _effective_exponents(component_index: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        alpha_values = _values(alpha_sigma_base, alpha_sigma_indices, component_index)
        gamma_values = _values(gamma_ml_base, gamma_ml_indices, component_index)
        alpha_best = _best_value(alpha_sigma_base, alpha_sigma_indices, component_index)
        gamma_best = _best_value(gamma_ml_base, gamma_ml_indices, component_index)
        beta_values = 1.0 + gamma_values - 2.0 * alpha_values
        beta_best = 1.0 + gamma_best - 2.0 * alpha_best
        return alpha_values, beta_values, alpha_best, beta_best

    rows: list[dict[str, Any]] = []
    for row in scaling_rank_df.itertuples(index=False):
        component_index = int(getattr(row, "component_index", -1))
        if component_index < 0 or component_index >= n_components:
            continue
        selected_active = bool(getattr(row, "selected_active", False))
        selected_independent = bool(getattr(row, "selected_independent", False))
        relation_class = "free" if selected_independent else ("active" if selected_active else "inactive")
        sigma_ref = _values(sigma_ref_base, sigma_ref_indices, component_index)
        cut_ref = _values(cut_ref_base, cut_ref_indices, component_index)
        core_ref = _values(core_ref_base, core_ref_indices, component_index)
        alpha_sigma, beta_radius, alpha_sigma_best, beta_radius_best = _effective_exponents(component_index)
        gamma_values = _values(gamma_ml_base, gamma_ml_indices, component_index)
        gamma_best = _best_value(gamma_ml_base, gamma_ml_indices, component_index)
        lum = float(luminosity_ratio[component_index])
        size_luminosity_scale = np.power(lum, beta_radius)
        scaling_v_disp = sigma_ref * np.power(lum, alpha_sigma)
        scaling_core = core_ref * size_luminosity_scale
        scaling_cut = cut_ref * size_luminosity_scale
        scaling_log10_mass = _log10_dpie_mass_msun(scaling_v_disp, scaling_cut)
        sigma_ref_best = _best_value(sigma_ref_base, sigma_ref_indices, component_index)
        cut_ref_best = _best_value(cut_ref_base, cut_ref_indices, component_index)
        core_ref_best = _best_value(core_ref_base, core_ref_indices, component_index)
        catalog_mag = float(getattr(row, "catalog_mag", np.nan))
        anchor_mag = (
            catalog_mag + 2.5 * math.log10(lum)
            if np.isfinite(catalog_mag) and np.isfinite(lum) and lum > 0.0
            else float("nan")
        )
        row_dict: dict[str, Any] = {
            "potfile_id": str(getattr(row, "potfile_id", "")),
            "catalog_id": str(getattr(row, "catalog_id", "")),
            "best_value": "" if best_value is None else str(best_value),
            "best_value_requested": "" if best_value_requested is None else str(best_value_requested),
            "rank": int(getattr(row, "rank", -1)),
            "component_index": component_index,
            "free_component_index": int(getattr(row, "free_component_index", -1)),
            "catalog_mag": catalog_mag,
            "catalog_color": float(getattr(row, "catalog_color", np.nan)),
            "luminosity_ratio": lum,
            "anchor_mag": anchor_mag,
            "selected_active": selected_active,
            "selected_independent": selected_independent,
            "scaling_relation_class": relation_class,
        }
        _add_summary(row_dict, "alpha_sigma", alpha_sigma, alpha_sigma_best)
        _add_summary(row_dict, "beta_radius", beta_radius, beta_radius_best)
        _add_summary(row_dict, "gamma_ml", gamma_values, gamma_best)
        _add_summary(row_dict, "log_softening_length_kpc", log_softening_values, log_softening_best)
        _add_summary(row_dict, "softening_length_kpc", softening_values, softening_best)
        _add_summary(
            row_dict,
            "scaling_v_disp",
            scaling_v_disp,
            sigma_ref_best * float(np.power(lum, alpha_sigma_best)),
        )
        size_luminosity_scale_best = float(np.power(lum, beta_radius_best))
        _add_summary(
            row_dict,
            "scaling_core_radius_kpc",
            scaling_core,
            core_ref_best * size_luminosity_scale_best,
        )
        scaling_core_best = core_ref_best * size_luminosity_scale_best
        _add_summary(
            row_dict,
            "scaling_core_radius_effective_kpc",
            np.sqrt(np.square(scaling_core) + np.square(softening_values)),
            float(np.sqrt(scaling_core_best * scaling_core_best + softening_best * softening_best)),
        )
        _add_summary(
            row_dict,
            "scaling_cut_radius_kpc",
            scaling_cut,
            cut_ref_best * size_luminosity_scale_best,
        )
        _add_summary(
            row_dict,
            "scaling_log10_mass_msun",
            scaling_log10_mass,
            float(
                _log10_dpie_mass_msun(
                    sigma_ref_best * float(np.power(lum, alpha_sigma_best)),
                    cut_ref_best * size_luminosity_scale_best,
                )
            ),
        )
        rows.append(row_dict)
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows)
    free_columns = [
        "free_v_disp_median",
        "free_v_disp_p16",
        "free_v_disp_p84",
        "free_v_disp_best",
        "free_core_radius_kpc_median",
        "free_core_radius_kpc_p16",
        "free_core_radius_kpc_p84",
        "free_core_radius_kpc_best",
        "free_core_radius_effective_kpc_median",
        "free_core_radius_effective_kpc_p16",
        "free_core_radius_effective_kpc_p84",
        "free_core_radius_effective_kpc_best",
        "free_cut_radius_kpc_median",
        "free_cut_radius_kpc_p16",
        "free_cut_radius_kpc_p84",
        "free_cut_radius_kpc_best",
        "free_log10_mass_msun_median",
        "free_log10_mass_msun_p16",
        "free_log10_mass_msun_p84",
        "free_log10_mass_msun_best",
    ]
    for column in free_columns:
        result[column] = np.nan
    if independent_scaling_df is not None and not independent_scaling_df.empty:
        diag_df = independent_scaling_df.copy()
        if {"potfile_id", "component_index"}.issubset(diag_df.columns):
            diag_df["potfile_id"] = diag_df["potfile_id"].astype(str)
            diag_df["component_index"] = diag_df["component_index"].astype(int)
            rename_free_map_columns = {
                column: column.removesuffix("_map") + "_best"
                for column in diag_df.columns
                if column.startswith("free_") and column.endswith("_map")
            }
            if rename_free_map_columns:
                diag_df = diag_df.rename(columns=rename_free_map_columns)
            merge_columns = ["potfile_id", "component_index"] + [column for column in free_columns if column in diag_df.columns]
            result = result.drop(columns=[column for column in free_columns if column in result.columns]).merge(
                diag_df[merge_columns],
                on=["potfile_id", "component_index"],
                how="left",
            )
            for column in free_columns:
                if column not in result:
                    result[column] = np.nan
    for suffix in ("median", "p16", "p84", "best"):
        effective_column = f"free_core_radius_effective_kpc_{suffix}"
        if effective_column in result and np.isfinite(pd.to_numeric(result[effective_column], errors="coerce")).any():
            continue
        core_column = f"free_core_radius_kpc_{suffix}"
        softening_column = f"softening_length_kpc_{suffix}"
        if core_column in result and softening_column in result:
            free_core = pd.to_numeric(result[core_column], errors="coerce").to_numpy(dtype=float)
            softening = pd.to_numeric(result[softening_column], errors="coerce").to_numpy(dtype=float)
            result[effective_column] = np.sqrt(np.square(free_core) + np.square(softening))
    for suffix in ("median", "p16", "p84", "best"):
        mass_column = f"free_log10_mass_msun_{suffix}"
        if mass_column in result and np.isfinite(pd.to_numeric(result[mass_column], errors="coerce")).any():
            continue
        v_column = f"free_v_disp_{suffix}"
        cut_column = f"free_cut_radius_kpc_{suffix}"
        if v_column in result and cut_column in result:
            result[mass_column] = _log10_dpie_mass_msun(
                pd.to_numeric(result[v_column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(result[cut_column], errors="coerce").to_numpy(dtype=float),
            )
    return result.reindex(columns=columns).sort_values(["potfile_id", "rank"]).reset_index(drop=True)


def _constraint_strength_label(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 0.60:
        return "strongly constrained"
    if value <= 0.85:
        return "moderately constrained"
    return "weakly constrained"


def _prior_shift_label(value: float) -> str:
    if not np.isfinite(value):
        return "prior shift unknown"
    if abs(value) >= 0.75:
        return "material prior shift"
    if abs(value) >= 0.30:
        return "modest prior shift"
    return "close to prior mean"


def _potfile_leverage_label(
    max_importance: float,
    global_max_importance: float,
    min_distance_arcsec: float,
) -> str:
    importance_ratio = (
        float(max_importance / global_max_importance)
        if np.isfinite(max_importance) and np.isfinite(global_max_importance) and global_max_importance > 0.0
        else float("nan")
    )
    if (np.isfinite(importance_ratio) and importance_ratio >= 0.5) or (np.isfinite(min_distance_arcsec) and min_distance_arcsec <= 5.0):
        return "strong local leverage"
    if (np.isfinite(importance_ratio) and importance_ratio >= 0.1) or (np.isfinite(min_distance_arcsec) and min_distance_arcsec <= 20.0):
        return "moderate local leverage"
    return "weak local leverage"


def _write_potfile_constraint_summary_txt(tables_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    global_max_importance = float(potfile_diag_df["importance_max"].max()) if potfile_diag_df["importance_max"].notna().any() else float("nan")
    lines: list[str] = []
    grouped = potfile_diag_df.groupby("potfile_id", sort=False)
    for potfile_id, group_df in grouped:
        shrink = float(group_df["posterior_std_over_prior_std"].median())
        shift = float(group_df["posterior_mean_minus_prior_mean_over_prior_std"].abs().max())
        min_distance = float(group_df["min_distance_arcsec_min"].min())
        max_importance = float(group_df["importance_max"].max())
        n_components = int(group_df["n_components"].max())
        n_active = int(group_df["n_active_components"].max())
        lines.append(f"{potfile_id}")
        lines.append(
            f"  constraint: {_constraint_strength_label(shrink)} "
            f"(median posterior/prior std={shrink:.3g})"
        )
        lines.append(
            f"  prior shift: {_prior_shift_label(shift)} "
            f"(max |mean-prior|/prior_std={shift:.3g})"
        )
        lines.append(
            f"  leverage: {_potfile_leverage_label(max_importance, global_max_importance, min_distance)} "
            f"(min_distance={min_distance:.3g}\", max_importance={max_importance:.3g})"
        )
        lines.append(f"  active components: {n_active}/{n_components}")
        for row in group_df.itertuples(index=False):
            lines.append(
                "  "
                + f"{row.parameter}: prior={float(row.prior_mean):.6g}±{float(row.prior_std):.6g}, "
                + f"posterior={float(row.posterior_median):.6g}±{float(row.posterior_std):.6g}, "
                + f"best={float(row.map):.6g}"
            )
        lines.append("")
    content = "\n".join(lines).rstrip() + "\n"
    (tables_dir / "potfile_constraint_summary.txt").write_text(content, encoding="utf-8")


def _plot_potfile_prior_posterior(
    plot_dir: Path,
    potfile_diag_df: pd.DataFrame,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> None:
    if potfile_diag_df.empty or samples.size == 0:
        return
    scaling_indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == "scaling"]
    if not scaling_indices:
        return
    scaling_samples = np.asarray(samples[:, scaling_indices], dtype=float)
    nrows = len(scaling_indices)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 2.3 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, spec, values in zip(axes, [parameter_specs[idx] for idx in scaling_indices], scaling_samples.T):
        row = potfile_diag_df[(potfile_diag_df["potfile_id"] == str(spec.potential_id)) & (potfile_diag_df["parameter"] == str(spec.field))]
        if row.empty:
            continue
        record = row.iloc[0]
        finite_values = np.asarray(values[np.isfinite(values)], dtype=float)
        if finite_values.size == 0:
            continue
        ax.hist(
            finite_values,
            bins=min(40, max(10, int(np.sqrt(finite_values.size)))),
            density=True,
            color="tab:blue",
            alpha=0.45,
            label="posterior",
        )
        if np.isfinite(record["prior_mean"]) and np.isfinite(record["prior_std"]) and float(record["prior_std"]) > 0.0:
            x_min = min(float(np.min(finite_values)), float(record["prior_mean"] - 4.0 * record["prior_std"]))
            x_max = max(float(np.max(finite_values)), float(record["prior_mean"] + 4.0 * record["prior_std"]))
            x_grid = np.linspace(x_min, x_max, 400)
            ax.plot(
                x_grid,
                norm.pdf(x_grid, loc=float(record["prior_mean"]), scale=float(record["prior_std"])),
                color="tab:orange",
                linewidth=1.8,
                label="prior",
            )
        ax.axvline(float(record["posterior_median"]), color="tab:blue", linewidth=1.5, linestyle="-", label="median")
        ax.axvline(float(record["map"]), color="tab:red", linewidth=1.5, linestyle="--", label="best fit")
        ax.set_title(str(record["label"]))
        ax.set_ylabel("density")
    axes[-1].set_xlabel("parameter value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "potfile_prior_posterior.png"), dpi=180, bbox_inches="tight")


def _plot_potfile_leverage_summary(plot_dir: Path, potfile_diag_df: pd.DataFrame) -> None:
    if potfile_diag_df.empty:
        return
    pot_df = (
        potfile_diag_df.groupby("potfile_id", sort=False)
        .agg(
            n_components=("n_components", "max"),
            n_active_components=("n_active_components", "max"),
            min_distance_arcsec_min=("min_distance_arcsec_min", "min"),
            min_distance_arcsec_median=("min_distance_arcsec_median", "median"),
            importance_max=("importance_max", "max"),
            importance_median=("importance_median", "median"),
            brightness_median=("brightness_median", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    pot_labels = pot_df["potfile_id"].astype(str).tolist()
    axes[0, 0].bar(pot_labels, np.asarray(pot_df["n_active_components"], dtype=float), color="tab:orange")
    axes[0, 0].set_title("Active Components")
    axes[0, 0].set_ylabel("count")
    axes[0, 1].bar(pot_labels, np.asarray(pot_df["min_distance_arcsec_min"], dtype=float), color="tab:purple")
    axes[0, 1].set_title("Closest Distance To Images")
    axes[0, 1].set_ylabel("arcsec")
    importance_vals, floor = _logsafe_importance_values(pot_df["importance_max"])
    axes[1, 0].bar(pot_labels, importance_vals, color="tab:green")
    axes[1, 0].set_title("Max Importance")
    axes[1, 0].set_ylabel("importance")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(bottom=floor * 0.8)
    axes[1, 1].bar(pot_labels, np.asarray(pot_df["brightness_median"], dtype=float), color="tab:blue")
    axes[1, 1].set_title("Median Brightness")
    axes[1, 1].set_ylabel("brightness")
    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "potfile_leverage_summary.png"), dpi=180, bbox_inches="tight")


def _family_diagnostics_table(evaluator: ClusterJAXEvaluator, best_eval: EvaluationResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in evaluator.state.family_data:
        cache = evaluator.validation_cache[family.family_id]
        pred = best_eval.family_predictions[family.family_id]
        exact_rms = pred.get("exact_image_rms")
        if exact_rms is None or not np.isfinite(exact_rms):
            rms_residual = pred.get("approx_image_rms_arcsec", pred.get("source_plane_rms"))
        else:
            rms_residual = exact_rms
        rows.append(
            {
                "family_id": family.family_id,
                "z_source": family.z_source,
                "effective_z_source": family.effective_z_source,
                "n_images": family.n_images,
                "sigma_arcsec": family.sigma_arcsec,
                "source_sigma_int_arcsec": pred.get("source_sigma_int_arcsec"),
                "source_sigma_eff_arcsec": pred.get("source_sigma_eff_arcsec"),
                "source_plane_rms_arcsec": pred.get("source_plane_rms"),
                "approx_image_rms_arcsec": pred.get("approx_image_rms_arcsec"),
                "exact_image_rms_arcsec": pred.get("exact_image_rms"),
                "rms_residual_arcsec": rms_residual,
                "max_residual_arcsec": pred.get("residual_max"),
                "used_exact_refresh": pred.get("used_exact_refresh", False),
                "refresh_reason": pred.get("refresh_reason"),
                "exact_validation_count": cache.exact_validation_count,
                "multiplicity_mismatch_count": cache.multiplicity_mismatch_count,
                "match_failure_count": cache.match_failure_count,
            }
        )
    return pd.DataFrame(rows).sort_values(by="rms_residual_arcsec", ascending=False, na_position="last")


def _arc_aware_family_diagnostics_from_image_rows(image_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "family_id",
        "arc_aware_image_rms_arcsec",
        "arc_aware_recovered_image_count",
        "arc_aware_missing_image_count",
        "arc_supported_image_count",
        "arc_candidate_supported_image_count",
    ]
    if image_df is None or image_df.empty or "family_id" not in image_df:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for family_id, group_df in image_df.groupby("family_id", sort=False):
        if "arc_aware_image_residual_arcsec" in group_df:
            residuals = pd.to_numeric(group_df["arc_aware_image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(residuals)
            recovered_count = int(np.sum(finite))
            missing_count = int(max(0, len(group_df) - recovered_count))
            rms = float(np.sqrt(np.mean(np.square(residuals[finite])))) if np.any(finite) else np.nan
        else:
            recovered_count = 0
            missing_count = int(len(group_df))
            rms = np.nan
        if "arc_supported" in group_df:
            supported_count = int(np.sum(group_df["arc_supported"].astype(bool).to_numpy()))
        elif "arc_recovery_status" in group_df:
            supported_count = int(np.sum(group_df["arc_recovery_status"].astype(str).to_numpy() == "arc_supported"))
        else:
            supported_count = 0
        if "arc_candidate_supported" in group_df:
            candidate_supported_count = int(np.sum(group_df["arc_candidate_supported"].astype(bool).to_numpy()))
        else:
            candidate_supported_count = supported_count
        rows.append(
            {
                "family_id": str(family_id),
                "arc_aware_image_rms_arcsec": rms,
                "arc_aware_recovered_image_count": recovered_count,
                "arc_aware_missing_image_count": missing_count,
                "arc_supported_image_count": supported_count,
                "arc_candidate_supported_image_count": candidate_supported_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _red1_pos_sigma_arcsec(
    residual2: np.ndarray,
    image_sigma_int: np.ndarray | None,
    covariance_floor: np.ndarray | None,
    dof: int,
) -> float | None:
    if dof <= 0 or residual2.size == 0 or image_sigma_int is None or covariance_floor is None:
        return None
    residual2 = np.asarray(residual2, dtype=float).reshape(-1)
    image_sigma_int = np.asarray(image_sigma_int, dtype=float).reshape(-1)
    covariance_floor = np.asarray(covariance_floor, dtype=float).reshape(-1)
    if residual2.shape != image_sigma_int.shape or residual2.shape != covariance_floor.shape:
        return None
    floor_variance = np.square(image_sigma_int) + np.maximum(covariance_floor, 0.0)
    if not np.all(np.isfinite(residual2 + floor_variance)) or np.any(floor_variance < 0.0):
        return None
    if not np.any(residual2 > 0.0):
        return 0.0

    target = float(dof)

    def chi_for(pos_sigma: float) -> float:
        denom = np.square(float(pos_sigma)) + floor_variance
        if np.any(denom <= 0.0):
            return float("inf")
        return float(np.sum(residual2 / denom))

    chi_at_zero = chi_for(0.0)
    if not np.isfinite(chi_at_zero):
        floor_variance = np.maximum(floor_variance, 1.0e-18)
        chi_at_zero = chi_for(0.0)
    if not np.isfinite(chi_at_zero):
        return None
    if chi_at_zero <= target:
        return 0.0

    high = math.sqrt(float(np.sum(residual2)) / target)
    if not np.isfinite(high) or high <= 0.0:
        return None
    low = 0.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if chi_for(mid) > target:
            low = mid
        else:
            high = mid
    return float(high)


def _fit_quality_chi_square_summary(
    image_fit_quality_df: pd.DataFrame | None,
    state: BuildState,
    *,
    use_arc_aware_diagnostics: bool = False,
) -> dict[str, Any]:
    parameter_specs = list(getattr(state, "parameter_specs", []))
    n_families = int(len(getattr(state, "family_data", [])))
    n_observed_images = int(sum(int(getattr(family, "n_images", 0)) for family in getattr(state, "family_data", [])))
    sampled_non_source_parameters = int(
        sum(getattr(spec, "component_family", None) != "source_position" for spec in parameter_specs)
    )
    source_position_parameters = int(2 * n_families)
    k_effective = int(sampled_non_source_parameters + source_position_parameters)
    empty = {
        "observed_image_count": n_observed_images,
        "n_effective_parameters": k_effective,
        "sampled_non_source_position_parameters": sampled_non_source_parameters,
        "source_position_parameters": source_position_parameters,
        "chi_square_sigma_basis": "image_sigma_eff_arcsec",
        "chi_square_sigma_eff_median_arcsec": None,
        "chi_square_sigma_eff_min_arcsec": None,
        "chi_square_sigma_eff_max_arcsec": None,
        "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
        "headline_chi_square_red1_total_sigma_arcsec": None,
        "headline_chi_square_red1_pos_sigma_arcsec": None,
        "arc_aware_chi_square_red1_total_sigma_arcsec": None,
        "arc_aware_chi_square_red1_pos_sigma_arcsec": None,
        "headline_chi_square": None,
        "headline_n_data": 0,
        "headline_dof": int(-k_effective),
        "headline_reduced_chi_square": None,
        "headline_point_image_count": 0,
        "headline_missing_image_count": n_observed_images,
        "point_recovered_image_count": 0,
        "point_image_rms_arcsec": None,
        "point_image_median_residual_arcsec": None,
        "arc_aware_chi_square": None,
        "arc_aware_n_data": 0,
        "arc_aware_dof": int(-k_effective),
        "arc_aware_reduced_chi_square": None,
        "arc_aware_point_image_count": 0,
        "arc_aware_arc_supported_image_count": 0,
        "arc_aware_recovered_image_count": 0,
        "arc_aware_missing_image_count": n_observed_images,
        "image_residual_mean_arcsec": None,
        "image_residual_median_arcsec": None,
        "image_residual_max_arcsec": None,
        "arc_aware_valid_image_count": 0,
        "arc_aware_image_rms_arcsec": None,
        "arc_aware_image_residual_mean_arcsec": None,
        "arc_aware_image_residual_median_arcsec": None,
        "arc_aware_image_residual_max_arcsec": None,
        "covered_xy_1sigma_fraction": None,
    }
    if image_fit_quality_df is None or image_fit_quality_df.empty:
        return empty
    df = image_fit_quality_df.copy()
    observed_image_count = n_observed_images if n_observed_images > 0 else int(len(df))
    empty["observed_image_count"] = observed_image_count
    empty["headline_missing_image_count"] = observed_image_count
    empty["arc_aware_missing_image_count"] = observed_image_count
    required = [
        "x_model_arcsec",
        "y_model_arcsec",
        "x_obs_arcsec",
        "y_obs_arcsec",
        "sigma_arcsec",
        "image_sigma_eff_arcsec",
    ]
    if any(column not in df.columns for column in required):
        return empty
    failed = (
        df["exact_image_prediction_failed"].astype(bool).to_numpy()
        if "exact_image_prediction_failed" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    recovery_status = (
        df["image_recovery_status"].astype(str).to_numpy()
        if "image_recovery_status" in df.columns
        else np.full(len(df), "", dtype=object)
    )
    arc_recovery_status = (
        df["arc_recovery_status"].astype(str).to_numpy()
        if "arc_recovery_status" in df.columns
        else np.full(len(df), "", dtype=object)
    )
    if "arc_supported" in df.columns:
        arc_supported = df["arc_supported"].astype(bool).to_numpy()
    else:
        arc_supported = arc_recovery_status == "arc_supported"
    if "p_arc" in df.columns:
        p_arc = pd.to_numeric(df["p_arc"], errors="coerce").to_numpy(dtype=float)
        if "arc_recovery_p_arc_threshold" in df.columns:
            threshold = pd.to_numeric(df["arc_recovery_p_arc_threshold"], errors="coerce").to_numpy(dtype=float)
            threshold = np.where(
                np.isfinite(threshold) & (threshold >= 0.0) & (threshold <= 1.0),
                threshold,
                CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD,
            )
        else:
            threshold = np.full(len(df), CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD, dtype=float)
        arc_supported = arc_supported | (np.isfinite(p_arc) & (p_arc >= threshold))
    has_image_recovery_status = "image_recovery_status" in df.columns
    has_recovery_status = has_image_recovery_status or "arc_recovery_status" in df.columns
    if has_image_recovery_status:
        point_recovered = recovery_status == "recovered"
    elif "arc_recovery_status" in df.columns:
        point_recovered = arc_recovery_status == "point_recovered"
    else:
        point_recovered = ~failed
    x_model = pd.to_numeric(df["x_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_model = pd.to_numeric(df["y_model_arcsec"], errors="coerce").to_numpy(dtype=float)
    x_obs = pd.to_numeric(df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma_meas = pd.to_numeric(df["sigma_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma_eff = pd.to_numeric(df["image_sigma_eff_arcsec"], errors="coerce").to_numpy(dtype=float)
    image_sigma_int = (
        pd.to_numeric(df["image_sigma_int_arcsec"], errors="coerce").to_numpy(dtype=float)
        if "image_sigma_int_arcsec" in df.columns
        else None
    )
    covariance_floor = None
    if image_sigma_int is not None:
        covariance_floor = np.maximum(np.square(sigma_eff) - np.square(sigma_meas) - np.square(image_sigma_int), 0.0)
    point_valid = (
        point_recovered
        & (~failed)
        & np.isfinite(x_model + y_model + x_obs + y_obs + sigma_eff)
        & (sigma_eff > 0.0)
    )
    dx = x_model[point_valid] - x_obs[point_valid]
    dy = y_model[point_valid] - y_obs[point_valid]
    point_residual2 = np.square(dx) + np.square(dy)
    residuals = np.sqrt(np.square(dx) + np.square(dy))
    point_chi_square = (
        float(np.sum(point_residual2 / np.square(sigma_eff[point_valid])))
        if dx.size
        else 0.0
    )
    point_count = int(np.sum(point_valid))
    headline_n_data = int(2 * point_count)
    headline_dof = int(headline_n_data - k_effective)
    coverage_fraction = None
    if "covered_xy_1sigma" in df.columns:
        coverage_values = df.loc[point_valid, "covered_xy_1sigma"].astype(bool).to_numpy()
        coverage_fraction = float(np.mean(coverage_values)) if coverage_values.size else None
    point_sum_squares = float(np.sum(point_residual2))
    headline_red1_total_sigma = (
        float(math.sqrt(point_sum_squares / headline_dof)) if headline_dof > 0 and point_count else None
    )
    point_image_sigma_int = image_sigma_int[point_valid] if image_sigma_int is not None else None
    point_covariance_floor = covariance_floor[point_valid] if covariance_floor is not None else None

    def _final_summary(chi_sigma_values: np.ndarray, arc_fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "observed_image_count": observed_image_count,
            "n_effective_parameters": k_effective,
            "sampled_non_source_position_parameters": sampled_non_source_parameters,
            "source_position_parameters": source_position_parameters,
            "chi_square_sigma_basis": "image_sigma_eff_arcsec",
            "chi_square_sigma_eff_median_arcsec": float(np.median(chi_sigma_values)) if chi_sigma_values.size else None,
            "chi_square_sigma_eff_min_arcsec": float(np.min(chi_sigma_values)) if chi_sigma_values.size else None,
            "chi_square_sigma_eff_max_arcsec": float(np.max(chi_sigma_values)) if chi_sigma_values.size else None,
            "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
            "headline_chi_square_red1_total_sigma_arcsec": headline_red1_total_sigma,
            "headline_chi_square_red1_pos_sigma_arcsec": _red1_pos_sigma_arcsec(
                point_residual2,
                point_image_sigma_int,
                point_covariance_floor,
                headline_dof,
            ),
            "headline_chi_square": point_chi_square,
            "headline_n_data": headline_n_data,
            "headline_dof": headline_dof,
            "headline_reduced_chi_square": float(point_chi_square / headline_dof) if headline_dof > 0 else None,
            "headline_point_image_count": point_count,
            "headline_missing_image_count": int(max(0, observed_image_count - point_count)),
            "point_recovered_image_count": point_count,
            "point_image_rms_arcsec": float(np.sqrt(np.mean(point_residual2))) if point_residual2.size else None,
            "point_image_median_residual_arcsec": float(np.median(residuals)) if residuals.size else None,
            "image_residual_mean_arcsec": float(np.mean(residuals)) if residuals.size else None,
            "image_residual_median_arcsec": float(np.median(residuals)) if residuals.size else None,
            "image_residual_max_arcsec": float(np.max(residuals)) if residuals.size else None,
            "covered_xy_1sigma_fraction": coverage_fraction,
            **arc_fields,
        }

    if not use_arc_aware_diagnostics:
        return _final_summary(
            sigma_eff[point_valid],
            {
                "arc_aware_chi_square_red1_total_sigma_arcsec": None,
                "arc_aware_chi_square_red1_pos_sigma_arcsec": None,
                "arc_aware_chi_square": None,
                "arc_aware_n_data": 0,
                "arc_aware_dof": int(-k_effective),
                "arc_aware_reduced_chi_square": None,
                "arc_aware_point_image_count": 0,
                "arc_aware_arc_supported_image_count": 0,
                "arc_aware_recovered_image_count": 0,
                "arc_aware_missing_image_count": observed_image_count,
                "arc_aware_valid_image_count": 0,
                "arc_aware_image_rms_arcsec": None,
                "arc_aware_image_residual_mean_arcsec": None,
                "arc_aware_image_residual_median_arcsec": None,
                "arc_aware_image_residual_max_arcsec": None,
            },
        )

    arc_residual = _fit_quality_value(
        df,
        "arc_candidate_image_residual_arcsec",
        "arc_aware_image_residual_arcsec",
        "arc_curve_distance_arcsec",
    )
    arc_aware_point_valid = point_valid & (~arc_supported)
    arc_aware_dx = x_model[arc_aware_point_valid] - x_obs[arc_aware_point_valid]
    arc_aware_dy = y_model[arc_aware_point_valid] - y_obs[arc_aware_point_valid]
    arc_aware_point_residual2 = np.square(arc_aware_dx) + np.square(arc_aware_dy)
    arc_aware_point_residuals = np.sqrt(arc_aware_point_residual2)
    arc_aware_point_chi_square = (
        float(np.sum(arc_aware_point_residual2 / np.square(sigma_eff[arc_aware_point_valid])))
        if arc_aware_dx.size
        else 0.0
    )
    arc_aware_point_count = int(np.sum(arc_aware_point_valid))
    arc_valid = (
        arc_supported
        & np.isfinite(arc_residual + sigma_eff)
        & (sigma_eff > 0.0)
    )
    arc_supported_count = int(np.sum(arc_valid))
    arc_residual2 = np.square(arc_residual[arc_valid])
    arc_chi_square = (
        arc_aware_point_chi_square + float(np.sum(arc_residual2 / np.square(sigma_eff[arc_valid])))
        if arc_aware_point_count or arc_supported_count
        else 0.0
    )
    arc_aware_n_data = int(2 * arc_aware_point_count + arc_supported_count)
    arc_aware_dof = int(arc_aware_n_data - k_effective)
    arc_aware_residuals = (
        np.concatenate([arc_aware_point_residuals, arc_residual[arc_valid]])
        if arc_aware_point_count or arc_supported_count
        else np.asarray([])
    )
    arc_sum_squares = float(np.sum(arc_aware_point_residual2)) + float(np.sum(arc_residual2))
    arc_aware_red1_total_sigma = (
        float(math.sqrt(arc_sum_squares / arc_aware_dof))
        if arc_aware_dof > 0 and (arc_aware_point_count or arc_supported_count)
        else None
    )
    arc_aware_residual2 = (
        np.concatenate([arc_aware_point_residual2, arc_residual2])
        if arc_aware_point_count or arc_supported_count
        else np.asarray([], dtype=float)
    )
    arc_aware_image_sigma_int = (
        np.concatenate([image_sigma_int[arc_aware_point_valid], image_sigma_int[arc_valid]])
        if image_sigma_int is not None and (arc_aware_point_count or arc_supported_count)
        else None
    )
    arc_aware_covariance_floor = (
        np.concatenate([covariance_floor[arc_aware_point_valid], covariance_floor[arc_valid]])
        if covariance_floor is not None and (arc_aware_point_count or arc_supported_count)
        else None
    )
    return _final_summary(
        sigma_eff[point_valid | arc_valid],
        {
            "arc_aware_chi_square_red1_total_sigma_arcsec": arc_aware_red1_total_sigma,
            "arc_aware_chi_square_red1_pos_sigma_arcsec": _red1_pos_sigma_arcsec(
                arc_aware_residual2,
                arc_aware_image_sigma_int,
                arc_aware_covariance_floor,
                arc_aware_dof,
            ),
            "arc_aware_chi_square": arc_chi_square,
            "arc_aware_n_data": arc_aware_n_data,
            "arc_aware_dof": arc_aware_dof,
            "arc_aware_reduced_chi_square": float(arc_chi_square / arc_aware_dof) if arc_aware_dof > 0 else None,
            "arc_aware_point_image_count": arc_aware_point_count,
            "arc_aware_arc_supported_image_count": arc_supported_count,
            "arc_aware_recovered_image_count": int(arc_aware_point_count + arc_supported_count),
            "arc_aware_missing_image_count": int(max(0, observed_image_count - arc_aware_point_count - arc_supported_count)),
            "arc_aware_valid_image_count": int(arc_aware_point_count + arc_supported_count),
            "arc_aware_image_rms_arcsec": (
                float(np.sqrt(np.mean(np.square(arc_aware_residuals)))) if arc_aware_residuals.size else None
            ),
            "arc_aware_image_residual_mean_arcsec": float(np.mean(arc_aware_residuals)) if arc_aware_residuals.size else None,
            "arc_aware_image_residual_median_arcsec": float(np.median(arc_aware_residuals)) if arc_aware_residuals.size else None,
            "arc_aware_image_residual_max_arcsec": float(np.max(arc_aware_residuals)) if arc_aware_residuals.size else None,
        },
    )


def _chain_diagnostics_summary(results: PosteriorResults, parameter_specs: list[ParameterSpec]) -> dict[str, Any]:
    base = {
        "ess_min": None,
        "ess_median": None,
        "ess_worst_parameter": None,
        "rhat_max": None,
        "rhat_median": None,
        "rhat_worst_parameter": None,
    }
    if str(results.sampler) == "numpyro_jaxns":
        ns_ess = None
        if results.init_diagnostics:
            ns_ess = results.init_diagnostics.get("ns_ess")
        if ns_ess is None and results.sample_weights is not None and len(results.sample_weights) > 0:
            ns_ess = _effective_sample_size(results.sample_weights)
        base["ess_min"] = ns_ess
        base["ess_median"] = ns_ess
        return base
    grouped = results.grouped_samples
    if grouped is None:
        return base
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[1] < 2 or grouped_array.shape[2] == 0:
        return base
    try:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    except Exception:
        return base
    names = [
        parameter_specs[idx].name if idx < len(parameter_specs) else f"param_{idx}"
        for idx in range(grouped_array.shape[2])
    ]
    ess_values: list[tuple[str, float]] = []
    rhat_values: list[tuple[str, float]] = []
    for idx, name in enumerate(names):
        values = grouped_array[:, :, idx]
        if not np.isfinite(values).all():
            continue
        try:
            ess = float(np.asarray(effective_sample_size(values)).reshape(-1)[0])
        except Exception:
            ess = float("nan")
        if np.isfinite(ess):
            ess_values.append((name, ess))
        if grouped_array.shape[0] >= 2:
            try:
                rhat = float(np.asarray(split_gelman_rubin(values)).reshape(-1)[0])
            except Exception:
                rhat = float("nan")
            if np.isfinite(rhat):
                rhat_values.append((name, rhat))
    if ess_values:
        ess_array = np.asarray([value for _name, value in ess_values], dtype=float)
        worst_name, worst_value = min(ess_values, key=lambda item: item[1])
        base["ess_min"] = float(worst_value)
        base["ess_median"] = float(np.median(ess_array))
        base["ess_worst_parameter"] = worst_name
    if rhat_values:
        rhat_array = np.asarray([value for _name, value in rhat_values], dtype=float)
        worst_name, worst_value = max(rhat_values, key=lambda item: item[1])
        base["rhat_max"] = float(worst_value)
        base["rhat_median"] = float(np.median(rhat_array))
        base["rhat_worst_parameter"] = worst_name
    return base


def _grouped_chain_array(
    value: np.ndarray | None,
    n_chains: int,
    n_draws: int,
    *,
    dtype: Any = float,
) -> np.ndarray | None:
    if value is None or n_chains <= 0 or n_draws <= 0:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.shape == (n_chains, n_draws):
        return array
    flat = array.reshape(-1)
    expected = int(n_chains) * int(n_draws)
    if flat.size != expected:
        return None
    return flat.reshape((n_chains, n_draws))


def _finite_quantiles(values: np.ndarray, quantiles: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [float("nan") for _quantile in quantiles]
    return [float(value) for value in np.quantile(finite, quantiles)]


def _image_sigma_int_index(parameter_specs: list[ParameterSpec], n_params: int) -> int | None:
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "sample_name", "")) == "image_sigma_int":
            return idx
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "name", "")) == "image.sigma_int":
            return idx
    for idx, spec in enumerate(parameter_specs[:n_params]):
        if str(getattr(spec, "component_family", "")) == "image_scatter":
            return idx
    return None


def _chain_health_summary_table(
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
    *,
    max_tree_depth: int | None = None,
) -> pd.DataFrame:
    columns = [
        "chain",
        "chain_index",
        "chain_label",
        "n_draws",
        "log_prob_mean",
        "log_prob_median",
        "log_prob_max",
        "accept_prob_mean",
        "divergence_count",
        "max_tree_depth_saturation_fraction",
        "image_sigma_int_q16",
        "image_sigma_int_q50",
        "image_sigma_int_q84",
    ]
    grouped = results.grouped_samples
    if grouped is None:
        return pd.DataFrame(columns=columns)
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0:
        return pd.DataFrame(columns=columns)
    n_chains, n_draws, n_params = grouped_array.shape
    grouped_log_prob = (
        np.asarray(results.grouped_log_prob, dtype=float)
        if results.grouped_log_prob is not None
        else _grouped_chain_array(results.log_prob, n_chains, n_draws, dtype=float)
    )
    if grouped_log_prob is not None and grouped_log_prob.shape != (n_chains, n_draws):
        grouped_log_prob = None
    grouped_accept = _grouped_chain_array(results.accept_prob, n_chains, n_draws, dtype=float)
    grouped_diverging = _grouped_chain_array(results.diverging, n_chains, n_draws, dtype=bool)
    grouped_steps = _grouped_chain_array(results.num_steps, n_chains, n_draws, dtype=float)
    step_threshold = (2**int(max_tree_depth) - 1) if max_tree_depth is not None else None
    image_sigma_idx = _image_sigma_int_index(parameter_specs, n_params)
    labels = list((results.init_diagnostics or {}).get("chain_seed_labels", []))
    rows: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        log_prob = (
            np.asarray(grouped_log_prob[chain_idx], dtype=float)
            if grouped_log_prob is not None
            else np.asarray([], dtype=float)
        )
        finite_log_prob = log_prob[np.isfinite(log_prob)]
        accept = (
            np.asarray(grouped_accept[chain_idx], dtype=float)
            if grouped_accept is not None
            else np.asarray([], dtype=float)
        )
        finite_accept = accept[np.isfinite(accept)]
        diverging = (
            np.asarray(grouped_diverging[chain_idx], dtype=bool)
            if grouped_diverging is not None
            else np.asarray([], dtype=bool)
        )
        steps = (
            np.asarray(grouped_steps[chain_idx], dtype=float)
            if grouped_steps is not None
            else np.asarray([], dtype=float)
        )
        finite_steps = steps[np.isfinite(steps)]
        if image_sigma_idx is not None:
            sigma_q16, sigma_q50, sigma_q84 = _finite_quantiles(grouped_array[chain_idx, :, image_sigma_idx], [0.16, 0.50, 0.84])
        else:
            sigma_q16 = sigma_q50 = sigma_q84 = float("nan")
        saturation_fraction = float("nan")
        if step_threshold is not None and finite_steps.size:
            saturation_fraction = float(np.mean(finite_steps >= float(step_threshold)))
        rows.append(
            {
                "chain": int(chain_idx + 1),
                "chain_index": int(chain_idx),
                "chain_label": str(labels[chain_idx]) if chain_idx < len(labels) else f"chain {chain_idx + 1}",
                "n_draws": int(n_draws),
                "log_prob_mean": float(np.mean(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "log_prob_median": float(np.median(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "log_prob_max": float(np.max(finite_log_prob)) if finite_log_prob.size else float("nan"),
                "accept_prob_mean": float(np.mean(finite_accept)) if finite_accept.size else float("nan"),
                "divergence_count": int(np.sum(diverging)) if diverging.size else 0,
                "max_tree_depth_saturation_fraction": saturation_fraction,
                "image_sigma_int_q16": sigma_q16,
                "image_sigma_int_q50": sigma_q50,
                "image_sigma_int_q84": sigma_q84,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _chain_parameter_diagnostics_table(
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
) -> pd.DataFrame:
    base_columns = [
        "parameter_index",
        "parameter",
        "sample_name",
        "component_family",
        "ess",
        "rhat",
        "chain_median_spread",
        "chain_median_standardized_spread",
    ]
    grouped = results.grouped_samples
    if grouped is None:
        return pd.DataFrame(columns=base_columns)
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0 or grouped_array.shape[2] == 0:
        return pd.DataFrame(columns=base_columns)
    n_chains, _n_draws, n_params = grouped_array.shape
    try:
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    except Exception:
        effective_sample_size = None
        split_gelman_rubin = None
    rows: list[dict[str, Any]] = []
    for param_idx in range(n_params):
        spec = parameter_specs[param_idx] if param_idx < len(parameter_specs) else None
        values = grouped_array[:, :, param_idx]
        chain_quantiles = [_finite_quantiles(values[chain_idx], [0.16, 0.50, 0.84]) for chain_idx in range(n_chains)]
        chain_medians = np.asarray([item[1] for item in chain_quantiles], dtype=float)
        finite_medians = chain_medians[np.isfinite(chain_medians)]
        finite_values = values[np.isfinite(values)]
        spread = float(np.max(finite_medians) - np.min(finite_medians)) if finite_medians.size else float("nan")
        global_std = float(np.std(finite_values)) if finite_values.size else float("nan")
        standardized_spread = spread / global_std if np.isfinite(spread) and np.isfinite(global_std) and global_std > 0.0 else float("nan")
        ess = float("nan")
        rhat = float("nan")
        if np.isfinite(values).all():
            if effective_sample_size is not None:
                try:
                    ess = float(np.asarray(effective_sample_size(values)).reshape(-1)[0])
                except Exception:
                    ess = float("nan")
            if split_gelman_rubin is not None and n_chains >= 2:
                try:
                    rhat = float(np.asarray(split_gelman_rubin(values)).reshape(-1)[0])
                except Exception:
                    rhat = float("nan")
        row: dict[str, Any] = {
            "parameter_index": int(param_idx),
            "parameter": str(getattr(spec, "name", f"param_{param_idx}")),
            "sample_name": str(getattr(spec, "sample_name", f"param_{param_idx}")),
            "component_family": str(getattr(spec, "component_family", "")),
            "ess": ess,
            "rhat": rhat,
            "chain_median_spread": spread,
            "chain_median_standardized_spread": standardized_spread,
        }
        for chain_idx, (q16, q50, q84) in enumerate(chain_quantiles, start=1):
            row[f"chain_{chain_idx}_q16"] = q16
            row[f"chain_{chain_idx}_q50"] = q50
            row[f"chain_{chain_idx}_q84"] = q84
        rows.append(row)
    return pd.DataFrame(rows)


def _run_summary(
    args: argparse.Namespace,
    state: BuildState,
    runtime_sec: float,
    results: PosteriorResults,
    best_loglike: float,
    evaluator: ClusterJAXEvaluator,
    image_fit_quality_df: pd.DataFrame | None = None,
    image_count_recovery_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    init_diagnostics = dict(results.init_diagnostics or {})
    run_name = str(getattr(args, "run_name", None) or state.run_name)
    geometry_cache = getattr(state, "geometry_cache", None)
    sample_likelihood_mode = _active_sample_likelihood_mode(evaluator, args)
    use_arc_aware_diagnostics = _uses_arc_aware_diagnostics(sample_likelihood_mode)
    source_redshifts = np.asarray([float(family.z_source) for family in state.family_data], dtype=float)
    finite_source_redshifts = source_redshifts[np.isfinite(source_redshifts)]
    lens_redshift = getattr(state, "z_lens", None)
    chi_square_summary = _fit_quality_chi_square_summary(
        image_fit_quality_df,
        state,
        use_arc_aware_diagnostics=use_arc_aware_diagnostics,
    )
    image_count_summary = _image_count_recovery_summary(image_count_recovery_df)
    chain_summary = _chain_diagnostics_summary(results, state.parameter_specs)
    def _cosmology_summary_fields() -> dict[str, Any]:
        fields: dict[str, Any] = {
            "fit_cosmology_flat_wcdm": bool(getattr(state, "fit_cosmology_flat_wcdm", False)),
            "cosmology_H0_fixed": float(dict(getattr(state, "cosmo_config", {})).get("H0", np.nan)),
            "cosmology_redshift_binning": "fiducial_fixed",
        }
        sample_array = np.asarray(results.samples, dtype=float)
        log_prob = np.asarray(results.log_prob, dtype=float)
        best_index = None
        if sample_array.ndim == 2 and log_prob.shape[0] == sample_array.shape[0] and np.isfinite(log_prob).any():
            best_index = int(np.nanargmax(log_prob))
        for idx, spec in enumerate(state.parameter_specs):
            if spec.sample_name not in {"cosmology_Om0", "cosmology_w0"}:
                continue
            values = sample_array[:, idx] if sample_array.ndim == 2 and sample_array.shape[1] > idx else np.asarray([])
            finite = values[np.isfinite(values)]
            prefix = spec.sample_name
            fields[f"{prefix}_best"] = (
                float(sample_array[best_index, idx])
                if best_index is not None and sample_array.shape[1] > idx and np.isfinite(sample_array[best_index, idx])
                else None
            )
            if finite.size:
                q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
                fields[f"{prefix}_q16"] = float(q16)
                fields[f"{prefix}_median"] = float(q50)
                fields[f"{prefix}_q84"] = float(q84)
            else:
                fields[f"{prefix}_q16"] = None
                fields[f"{prefix}_median"] = None
                fields[f"{prefix}_q84"] = None
        return fields

    max_tree_depth = _first_int_value(getattr(args, "max_tree_depth", 10), 10)
    summary = {
        "run_name": run_name,
        "par_path": state.par_path,
        "fit_mode": state.fit_mode,
        "n_parameters": len(state.parameter_specs),
        "n_families": len(state.family_data),
        "n_images": int(sum(f.n_images for f in state.family_data)),
        "n_large_scale_parameters": int(sum(spec.component_family == "large" for spec in state.parameter_specs)),
        "n_scaling_parameters": int(sum(spec.component_family == "scaling" for spec in state.parameter_specs)),
        "n_independent_scaling_parameters": int(
            sum(spec.component_family == "independent_scaling" for spec in state.parameter_specs)
        ),
        "n_source_scatter_parameters": int(sum(spec.component_family == "source_scatter" for spec in state.parameter_specs)),
        "n_source_position_parameters": int(sum(spec.component_family == "source_position" for spec in state.parameter_specs)),
        "n_image_scatter_parameters": int(sum(spec.component_family == "image_scatter" for spec in state.parameter_specs)),
        "n_cosmology_parameters": int(sum(spec.component_family == "cosmology" for spec in state.parameter_specs)),
        "n_scaling_galaxy_components": int(np.sum(state.packed_lens_spec.component_family == 1)),
        "n_independent_scaling_galaxy_candidates": int(
            sum(bool(record.get("selected_independent", False)) for record in getattr(state, "scaling_component_records", []))
        ),
        "n_independent_scaling_free_branch_components": int(
            np.sum(state.packed_lens_spec.component_family == 2)
        ),
        "z_lens": float(lens_redshift) if lens_redshift is not None and np.isfinite(float(lens_redshift)) else None,
        "z_source_min": float(np.min(finite_source_redshifts)) if finite_source_redshifts.size else None,
        "z_source_max": float(np.max(finite_source_redshifts)) if finite_source_redshifts.size else None,
        "fit_method": str(_stage_scalar(getattr(args, "fit_method", None), "svi+nuts")),
        "sample_likelihood_mode": sample_likelihood_mode,
        "local_jacobian_metric": str(
            getattr(evaluator, "local_jacobian_metric_mode", init_diagnostics.get("local_jacobian_metric", "not_used"))
        ),
        "image_plane_mode": str(getattr(args, "image_plane_mode", "none")),
        "skip_stage3_image_plane_local_jacobian": bool(getattr(args, "skip_stage3_image_plane_local_jacobian", False)),
        "quick_diagnostics": bool(getattr(args, "quick_diagnostics", False)),
        "image_plane_newton_steps": int(getattr(args, "image_plane_newton_steps", 0)),
        "image_plane_scatter_floor_arcsec": float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0)),
        "arc_recovery_p_arc_threshold": float(
            getattr(
                args,
                "arc_recovery_p_arc_threshold",
                getattr(evaluator, "arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD),
            )
        ),
        "arc_aware_max_arclength_arcsec": float(
            getattr(
                args,
                "arc_aware_max_arclength_arcsec",
                getattr(evaluator, "arc_aware_max_arclength_arcsec", 5.0),
            )
        ),
        "arc_aware_curve_step_arcsec": float(
            getattr(
                args,
                "arc_aware_curve_step_arcsec",
                getattr(evaluator, "arc_aware_curve_step_arcsec", 0.1),
            )
        ),
        "image_plane_scatter_prior": str(getattr(args, "image_plane_scatter_prior", "log-uniform")),
        "image_plane_scatter_prior_median_arcsec": float(
            getattr(args, "image_plane_scatter_prior_median_arcsec", 0.3)
        ),
        "image_plane_scatter_prior_log_sigma": float(
            getattr(args, "image_plane_scatter_prior_log_sigma", 0.5)
        ),
        "likelihood_stabilizer_max_gain": float(getattr(args, "likelihood_stabilizer_max_gain", 0.0)),
        "likelihood_stabilizer_max_residual_arcsec": float(
            getattr(args, "likelihood_stabilizer_max_residual_arcsec", 0.0)
        ),
        "likelihood_stabilizer_residual_loss": str(getattr(args, "likelihood_stabilizer_residual_loss", "gaussian")),
        "likelihood_stabilizer_student_t_nu": float(getattr(args, "likelihood_stabilizer_student_t_nu", 4.0)),
        "source_position_parameterization": str(getattr(args, "source_position_parameterization", "prior-whitened")),
        "evidence_source_prior_sigma_arcsec": (
            None
            if getattr(args, "evidence_source_prior_sigma_arcsec", None) is None
            else float(getattr(args, "evidence_source_prior_sigma_arcsec"))
        ),
        "evidence_source_prior_mean_x_arcsec": float(getattr(args, "evidence_source_prior_mean_x_arcsec", 0.0)),
        "evidence_source_prior_mean_y_arcsec": float(getattr(args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        **_cosmology_summary_fields(),
        "sampling_engine": args.sampling_engine,
        "fit_sampling_engine": init_diagnostics.get(
            "fit_sampling_engine",
            str(getattr(args, "sampling_engine", getattr(evaluator, "sampling_engine", "full_flat"))),
        ),
        "final_validation_sampling_engine": init_diagnostics.get(
            "final_validation_sampling_engine",
            str(getattr(evaluator, "final_validation_sampling_engine", getattr(evaluator, "sampling_engine", "full_flat"))),
        ),
        "active_scaling_galaxies": list(evaluator.active_scaling_galaxies_by_potfile),
        "active_scaling_components": int(len(evaluator.active_scaling_component_indices)),
        "inactive_scaling_components": int(len(evaluator.inactive_scaling_component_indices)),
        "independent_scaling_model": "log_displacement",
        "scaling_relation_mode": str(getattr(state, "scaling_relation_mode", getattr(args, "scaling_relation_mode", "lenstool-denominator"))),
        "core_radius_scaling": (
            "shared_beta_radius"
            if str(getattr(state, "scaling_relation_mode", getattr(args, "scaling_relation_mode", "lenstool-denominator")))
            == "bergamini-ml"
            else "shared_cut_slope"
        ),
        "infer_active_scaling": bool(init_diagnostics.get("infer_active_scaling", getattr(state, "infer_active_scaling", False))),
        "active_scaling_frozen_from_svi": bool(
            init_diagnostics.get("infer_active_scaling", False)
            and getattr(state, "frozen_active_scaling_component_indices", None) is not None
        ),
        "active_scaling_frozen_from_previous_stage": bool(
            init_diagnostics.get(
                "active_scaling_frozen_from_previous_stage",
                getattr(state, "active_scaling_frozen_from_previous_stage", False),
            )
        ),
        "active_scaling_frozen_source_run_dir": init_diagnostics.get(
            "active_scaling_frozen_source_run_dir",
            getattr(state, "active_scaling_frozen_source_run_dir", None),
        ),
        "active_scaling_frozen_source_path": init_diagnostics.get(
            "active_scaling_frozen_source_path",
            getattr(state, "active_scaling_frozen_source_path", None),
        ),
        "active_scaling_freeze_threshold": init_diagnostics.get(
            "active_scaling_freeze_threshold",
            getattr(args, "active_scaling_freeze_threshold", None),
        ),
        "active_scaling_frozen_active_count": init_diagnostics.get(
            "active_scaling_frozen_active_count",
            (
                int(len(getattr(state, "frozen_active_scaling_component_indices", [])))
                if getattr(state, "frozen_active_scaling_component_indices", None) is not None
                else None
            ),
        ),
        "active_scaling_frozen_inactive_count": init_diagnostics.get("active_scaling_frozen_inactive_count"),
        "active_scaling_frozen_active_by_potfile": init_diagnostics.get("active_scaling_frozen_active_by_potfile", {}),
        "active_scaling_frozen_inactive_by_potfile": init_diagnostics.get("active_scaling_frozen_inactive_by_potfile", {}),
        "large_exact_components": int(len(getattr(evaluator, "large_component_indices", []))),
        "exact_scaling_components": int(
            len(getattr(evaluator, "exact_scaling_component_indices", evaluator.active_scaling_component_indices))
        ),
        "cached_scaling_components": int(
            len(getattr(evaluator, "cached_scaling_component_indices", evaluator.inactive_scaling_component_indices))
        ),
        "free_correction_scaling_components": int(
            len(getattr(evaluator, "free_correction_scaling_component_indices", []))
        ),
        "free_correction_free_branch_components": int(
            len(getattr(evaluator, "free_correction_free_component_indices", []))
        ),
        "excluded_scaling_components": int(len(getattr(evaluator, "excluded_scaling_component_indices", []))),
        "independent_scaling_components": int(len(getattr(evaluator, "independent_scaling_component_indices", []))),
        "independent_scaling_free_branch_components": int(
            len(getattr(evaluator, "independent_free_component_indices", []))
        ),
        "requested_active_scaling_by_potfile": evaluator.requested_active_scaling_by_potfile,
        "actual_active_scaling_by_potfile": evaluator.actual_active_scaling_by_potfile,
        "total_scaling_by_potfile": evaluator.total_scaling_by_potfile,
        "independent_scaling_settings": {
            "candidate_source": "active_scaling_galaxies",
            "free_branch": "bergamini_sigma_mass_log_displacement",
            "free_log_sigma_offset_width": float(
                getattr(
                    state,
                    "independent_scaling_free_log_sigma_offset_width",
                    getattr(args, "independent_scaling_free_log_sigma_offset_width", 0.10),
                )
            ),
            "free_log_mass_offset_width": float(
                getattr(
                    state,
                    "independent_scaling_free_log_mass_offset_width",
                    getattr(args, "independent_scaling_free_log_mass_offset_width", 0.20),
                )
            ),
            "solver_defined_scaling_law_priors": {
                "sigma_ref": {"mean": 100.0, "std": 40.0, "lower": 20.0, "upper": 500.0},
                "cut_ref_kpc": {"mean": 300.0, "std": 120.0, "lower": 20.0, "upper": 1000.0},
                "core_ref_kpc": {"median": 0.15, "log_sigma": 0.7, "lower": 0.01, "upper": 10.0},
                "vdslope": {"mean": 4.0, "std": 0.75, "lower": 1.5, "upper": 8.0},
                "slope": {"mean": 4.0, "std": 0.75, "lower": 1.5, "upper": 8.0},
            },
        },
        "independent_scaling_candidates_by_potfile": (
            {}
            if getattr(evaluator, "scaling_rank_df", pd.DataFrame()).empty
            else {
                str(potfile_id): int(group_df.get("selected_independent", pd.Series(False, index=group_df.index)).astype(bool).sum())
                for potfile_id, group_df in evaluator.scaling_rank_df.groupby("potfile_id", sort=False)
            }
        ),
        "fit_quality_reference_sample_kind": str(
            init_diagnostics.get("fit_quality_reference_sample_kind", "max_likelihood")
        ),
        "fit_quality_reference_sample_index": init_diagnostics.get("max_likelihood_sample_index"),
        "fit_quality_reference_source_loglike": init_diagnostics.get("max_likelihood_source_loglike"),
        "fit_quality_reference_log_prob": init_diagnostics.get("max_likelihood_sample_log_prob"),
        "sampler": str(results.sampler),
        "nuts_init_strategy_requested": init_diagnostics.get("strategy_requested", getattr(args, "nuts_init_strategy", "svi")),
        "nuts_init_strategy_used": init_diagnostics.get("strategy_used", getattr(args, "nuts_init_strategy", "svi")),
        "nuts_init_settings": {
            "boundary_frac": float(getattr(args, "nuts_init_boundary_frac", DEFAULT_NUTS_INIT_BOUNDARY_FRAC)),
            "jitter_frac": float(getattr(args, "nuts_init_jitter_frac", DEFAULT_NUTS_INIT_JITTER_FRAC)),
            "svi_steps": int(_stage_scalar(getattr(args, "svi_steps", None), DEFAULT_SVI_STEPS)),
            "svi_learning_rate": float(
                _stage_scalar(getattr(args, "svi_learning_rate", None), DEFAULT_SVI_LEARNING_RATE)
            ),
        },
        "nuts_init_diagnostics": {
            "distinct_chain_seeds": int(init_diagnostics.get("distinct_chain_seeds", 0)),
            "svi_used": bool(init_diagnostics.get("svi_used", False)),
            "svi_final_elbo_loss": float(init_diagnostics.get("svi_final_elbo_loss", float("nan"))),
            "chain_seed_labels": list(init_diagnostics.get("chain_seed_labels", [])),
            "svi_chain_start_labels": list(init_diagnostics.get("svi_chain_start_labels", [])),
            "requested_chains": int(init_diagnostics.get("requested_chains", results.num_chains)),
            "retained_finite_chains": int(init_diagnostics.get("retained_finite_chains", results.num_chains)),
            "dropped_nonfinite_chains": int(init_diagnostics.get("dropped_nonfinite_chains", 0)),
            "retained_chain_indices": list(init_diagnostics.get("retained_chain_indices", list(range(results.num_chains)))),
            "dropped_chain_indices": list(init_diagnostics.get("dropped_chain_indices", [])),
            "invalid_state_rejection_count": int(init_diagnostics.get("invalid_state_rejection_count", evaluator.invalid_state_rejection_count)),
            "invalid_state_reason_counts": {
                key: int(value)
                for key, value in dict(
                    init_diagnostics.get("invalid_state_reason_counts", evaluator.invalid_state_reason_counts)
                ).items()
            },
            "quality_metrics": dict(init_diagnostics.get("nuts_quality_metrics", {})),
            "quality_warnings": list(init_diagnostics.get("nuts_quality_warnings", [])),
            "svi_health_metrics": dict(init_diagnostics.get("svi_health_metrics") or {}),
            "svi_health_warnings": list(init_diagnostics.get("svi_health_warnings") or []),
        },
        "ns_settings": {
            "num_live_points": init_diagnostics.get("ns_num_live_points", getattr(args, "ns_num_live_points", None)),
            "max_samples": init_diagnostics.get("ns_max_samples", getattr(args, "ns_max_samples", None)),
            "dlogz": init_diagnostics.get("ns_dlogz", getattr(args, "ns_dlogz", None)),
        },
        "ns_log_z_mean": init_diagnostics.get("ns_log_z_mean"),
        "ns_log_z_uncert": init_diagnostics.get("ns_log_z_uncert"),
        "ns_ess": init_diagnostics.get("ns_ess"),
        "ns_total_num_samples": init_diagnostics.get("ns_total_num_samples"),
        "ns_posterior_samples": init_diagnostics.get("ns_posterior_samples"),
        "ns_posterior_resampling": init_diagnostics.get("ns_posterior_resampling"),
        "ns_total_num_likelihood_evaluations": init_diagnostics.get("ns_total_num_likelihood_evaluations"),
        "ns_termination_reason": init_diagnostics.get("ns_termination_reason"),
        "warmup": int(_stage_scalar(getattr(args, "warmup", None), results.warmup_steps)),
        "samples": int(_stage_scalar(getattr(args, "samples", None), results.sample_steps)),
        "chains": results.num_chains,
        "requested_chains": int(
            _stage_scalar(init_diagnostics.get("requested_chains", getattr(args, "chains", results.num_chains)), results.num_chains)
        ),
        "thin": int(_stage_scalar(getattr(args, "thin", None), 1)),
        "max_tree_depth": max_tree_depth,
        "target_accept": float(_stage_scalar(getattr(args, "target_accept", None), 0.85)),
        "runtime_sec": runtime_sec,
        "best_loglike": best_loglike,
        "seed": args.seed,
        "packed_fast_path": True,
        "uses_potfile_scaling": bool(state.potfiles and state.fit_mode in {"small-only", "joint"}),
        "surrogate_enabled": bool(evaluator.surrogate_enabled),
        "approximate_eval_count": int(evaluator.approximate_eval_count),
        "full_refresh_count": int(evaluator.full_refresh_count),
        "invalid_state_rejection_count": int(evaluator.invalid_state_rejection_count),
        "invalid_state_reason_counts": {key: int(value) for key, value in evaluator.invalid_state_reason_counts.items()},
        "stage2_large_scale_priors": {
            spec.sample_name: {"mean": spec.mean, "std": spec.std}
            for spec in state.parameter_specs
            if state.fit_mode == "small-only"
            and spec.component_family == "large"
            and spec.prior_kind in {"normal", "truncated_normal"}
        },
        "geometry_setup_timing_sec": {
            "family_redshift_binning": float(geometry_cache.family_redshift_binning_sec)
            if geometry_cache is not None
            else None,
            "geometry_cache_build": float(geometry_cache.geometry_cache_build_sec)
            if geometry_cache is not None
            else None,
        },
        "distinct_effective_source_planes": len({family.effective_z_source for family in state.family_data}),
        "mean_eval_wall_time_sec": float(np.mean(evaluator.eval_wall_times)) if evaluator.eval_wall_times else None,
        "median_eval_wall_time_sec": float(np.median(evaluator.eval_wall_times)) if evaluator.eval_wall_times else None,
        "timing_totals_sec": {key: float(value) for key, value in evaluator.timing_totals.items()},
        "accept_prob_mean": float(np.mean(results.accept_prob)) if results.accept_prob.size else None,
        "divergence_count": int(np.sum(results.diverging)) if results.diverging.size else 0,
        "mean_num_steps": float(np.mean(results.num_steps)) if results.num_steps.size else None,
        "max_tree_depth_saturation_fraction": (
            float(
                np.mean(
                    np.asarray(results.num_steps, dtype=float)
                    >= (2**max_tree_depth - 1)
                )
            )
            if results.num_steps.size
            else None
        ),
        "svi_health_metrics": dict(init_diagnostics.get("svi_health_metrics") or {}),
        "svi_health_warnings": list(init_diagnostics.get("svi_health_warnings") or []),
        "nuts_quality_warnings": list(init_diagnostics.get("nuts_quality_warnings", [])),
        "sample_weight_ess": float(_effective_sample_size(results.sample_weights))
        if results.sample_weights is not None and len(results.sample_weights) > 0
        else None,
        "temperature_schedule": results.temperature_schedule.tolist() if results.temperature_schedule is not None else None,
        "ess_history": results.ess_history.tolist() if results.ess_history is not None else None,
        "move_acceptance_history": results.move_acceptance_history.tolist() if results.move_acceptance_history is not None else None,
        **chi_square_summary,
        **image_count_summary,
        **chain_summary,
    }
    image_scatter_indices = [
        idx for idx, spec in enumerate(state.parameter_specs) if spec.component_family == "image_scatter"
    ]
    fixed_image_sigma_int = getattr(args, "fix_image_sigma_int_arcsec", None)
    summary["fixed_image_sigma_int_arcsec"] = (
        None if fixed_image_sigma_int is None else float(fixed_image_sigma_int)
    )
    summary["image_sigma_int_sampled"] = bool(image_scatter_indices) and fixed_image_sigma_int is None
    if fixed_image_sigma_int is not None:
        q50 = float(fixed_image_sigma_int)
        scatter_floor = float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0))
        summary["image_sigma_int_posterior"] = {
            "q16": q50,
            "median": q50,
            "q84": q50,
            "lower_arcsec": q50,
            "upper_arcsec": q50,
            "scatter_floor_arcsec": scatter_floor,
            "near_lower_bound": False,
            "floor_dominated": bool(q50 <= scatter_floor),
            "near_upper_bound": False,
            "fixed": True,
        }
        sigma_values = []
        for bin_item in getattr(state, "bin_data", []):
            if hasattr(bin_item, "sigma_per_image"):
                sigma_values.extend(np.asarray(bin_item.sigma_per_image, dtype=float).reshape(-1).tolist())
        sigma_array = np.asarray(sigma_values, dtype=float)
        finite_sigma = sigma_array[np.isfinite(sigma_array)]
        if finite_sigma.size:
            covariance_floor = float(getattr(args, "source_plane_covariance_floor", 0.0))
            sigma_eff2 = finite_sigma**2 + q50**2 + covariance_floor
            summary["image_sigma_eff_variance_arcsec2"] = {
                "min": float(np.min(sigma_eff2)),
                "median": float(np.median(sigma_eff2)),
                "max": float(np.max(sigma_eff2)),
            }
    elif image_scatter_indices:
        values = np.asarray(results.samples, dtype=float)[:, image_scatter_indices[0]]
        finite = values[np.isfinite(values)]
        if finite.size:
            spec = state.parameter_specs[image_scatter_indices[0]]
            upper_value = getattr(spec, "physical_upper", None)
            if upper_value is None:
                upper_value = getattr(spec, "upper", np.nan)
            upper = float(upper_value)
            q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
            lower_value = getattr(spec, "physical_lower", None)
            lower = float(lower_value) if lower_value is not None else float("nan")
            scatter_floor = float(getattr(args, "image_plane_scatter_floor_arcsec", 0.0))
            near_lower = bool(np.isfinite(lower) and q16 <= max(1.1 * lower, lower + 1.0e-9))
            floor_dominated = bool(scatter_floor > 0.0 and q50 <= scatter_floor)
            summary["image_sigma_int_posterior"] = {
                "q16": float(q16),
                "median": float(q50),
                "q84": float(q84),
                "lower_arcsec": lower,
                "upper_arcsec": upper,
                "scatter_floor_arcsec": scatter_floor,
                "near_lower_bound": near_lower,
                "floor_dominated": floor_dominated,
                "near_upper_bound": bool(np.isfinite(upper) and q84 >= 0.9 * upper),
            }
            sigma_values = []
            for bin_item in getattr(state, "bin_data", []):
                if hasattr(bin_item, "sigma_per_image"):
                    sigma_values.extend(np.asarray(bin_item.sigma_per_image, dtype=float).reshape(-1).tolist())
            sigma_array = np.asarray(sigma_values, dtype=float)
            finite_sigma = sigma_array[np.isfinite(sigma_array)]
            if finite_sigma.size:
                covariance_floor = float(getattr(args, "source_plane_covariance_floor", 0.0))
                sigma_eff2 = finite_sigma**2 + float(q50) ** 2 + covariance_floor
                summary["image_sigma_eff_variance_arcsec2"] = {
                    "min": float(np.min(sigma_eff2)),
                    "median": float(np.median(sigma_eff2)),
                    "max": float(np.max(sigma_eff2)),
                }
            summary["image_sigma_int_posterior"]["fixed"] = False
    critical_arc_threshold_indices = [
        idx
        for idx, spec in enumerate(state.parameter_specs)
        if getattr(spec, "sample_name", "") == CRITICAL_ARC_SINGULAR_THRESHOLD_SAMPLE_NAME
    ]
    summary["critical_arc_singular_threshold_sampled"] = bool(critical_arc_threshold_indices)
    if critical_arc_threshold_indices:
        idx = int(critical_arc_threshold_indices[0])
        values = np.asarray(results.samples, dtype=float)[:, idx]
        finite = values[np.isfinite(values)]
        if finite.size:
            spec = state.parameter_specs[idx]
            q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
            lower_value = getattr(spec, "physical_lower", None)
            upper_value = getattr(spec, "physical_upper", None)
            lower = float(lower_value) if lower_value is not None else float("nan")
            upper = float(upper_value) if upper_value is not None else float("nan")
            summary["critical_arc_singular_threshold_posterior"] = {
                "q16": float(q16),
                "median": float(q50),
                "q84": float(q84),
                "lower": lower,
                "upper": upper,
                "near_lower_bound": bool(np.isfinite(lower) and q16 <= max(1.1 * lower, lower + 1.0e-9)),
                "near_upper_bound": bool(np.isfinite(upper) and q84 >= 0.9 * upper),
            }
    critical_arc_softness_indices = [
        idx
        for idx, spec in enumerate(state.parameter_specs)
        if getattr(spec, "sample_name", "") == CRITICAL_ARC_SINGULAR_SOFTNESS_SAMPLE_NAME
    ]
    summary["critical_arc_singular_softness_sampled"] = bool(critical_arc_softness_indices)
    if critical_arc_softness_indices:
        idx = int(critical_arc_softness_indices[0])
        values = np.asarray(results.samples, dtype=float)[:, idx]
        finite = values[np.isfinite(values)]
        if finite.size:
            spec = state.parameter_specs[idx]
            q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
            lower_value = getattr(spec, "physical_lower", None)
            upper_value = getattr(spec, "physical_upper", None)
            lower = float(lower_value) if lower_value is not None else float("nan")
            upper = float(upper_value) if upper_value is not None else float("nan")
            summary["critical_arc_singular_softness_posterior"] = {
                "q16": float(q16),
                "median": float(q50),
                "q84": float(q84),
                "lower": lower,
                "upper": upper,
                "near_lower_bound": bool(np.isfinite(lower) and q16 <= max(1.1 * lower, lower + 1.0e-9)),
                "near_upper_bound": bool(np.isfinite(upper) and q84 >= 0.9 * upper),
            }
    return summary


def _format_run_summary_text(summary: dict[str, Any]) -> str:
    source_range = "na"
    if summary.get("z_source_min") is not None and summary.get("z_source_max") is not None:
        source_range = f"{_metric_text(summary.get('z_source_min'))}-{_metric_text(summary.get('z_source_max'))}"
    use_arc_aware_diagnostics = _summary_uses_arc_aware_diagnostics(summary)
    headline_chi_square = summary.get("headline_chi_square")
    headline_dof = summary.get("headline_dof")
    headline_reduced_chi_square = summary.get("headline_reduced_chi_square")
    arc_aware_chi_square = summary.get("arc_aware_chi_square")
    arc_aware_dof = summary.get("arc_aware_dof")
    arc_aware_reduced_chi_square = summary.get("arc_aware_reduced_chi_square")
    observed_image_count = summary.get("observed_image_count")
    point_recovered_count = summary.get("point_recovered_image_count")
    arc_aware_recovered_count = summary.get("arc_aware_recovered_image_count")
    arc_supported_count = summary.get("arc_aware_arc_supported_image_count")
    missing_count = summary.get("arc_aware_missing_image_count")
    p_arc_threshold = summary.get("arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
    recovery_gate_note = (
        "arc-aware recovery gate: "
        f"arc_recovery_p_arc_threshold={_metric_text(p_arc_threshold)}; "
        "support-curve distance is diagnostic only."
    )
    quality_items: list[tuple[str, Any]] = [
        ("headline_chi_square", headline_chi_square),
        ("headline dof", headline_dof),
        ("headline_reduced_chi_square", headline_reduced_chi_square),
        ("point image RMS arcsec", summary.get("point_image_rms_arcsec")),
        ("point median residual arcsec", summary.get("point_image_median_residual_arcsec")),
        ("point recovered images", f"{_metric_text(point_recovered_count)}/{_metric_text(observed_image_count)}"),
    ]
    if use_arc_aware_diagnostics:
        quality_items.extend(
            [
                ("arc_aware_chi_square", arc_aware_chi_square),
                ("arc-aware dof", arc_aware_dof),
                ("arc_aware_reduced_chi_square", arc_aware_reduced_chi_square),
                ("arc-aware image RMS arcsec", summary.get("arc_aware_image_rms_arcsec")),
                ("arc-aware median residual arcsec", summary.get("arc_aware_image_residual_median_arcsec")),
                ("arc-aware recovered images", f"{_metric_text(arc_aware_recovered_count)}/{_metric_text(observed_image_count)}"),
            ]
        )
    quality_items.extend(
        [
            ("chi-square sigma basis", summary.get("chi_square_sigma_basis")),
            ("chi-square median sigma arcsec", summary.get("chi_square_sigma_eff_median_arcsec")),
            ("chi-square min sigma arcsec", summary.get("chi_square_sigma_eff_min_arcsec")),
            ("chi-square max sigma arcsec", summary.get("chi_square_sigma_eff_max_arcsec")),
            (
                "headline red1 total sigma arcsec",
                summary.get("headline_chi_square_red1_total_sigma_arcsec"),
            ),
            (
                "headline red1 pos_sigma_arcsec",
                summary.get("headline_chi_square_red1_pos_sigma_arcsec"),
            ),
        ]
    )
    if use_arc_aware_diagnostics:
        quality_items.extend(
            [
                (
                    "arc-aware red1 total sigma arcsec",
                    summary.get("arc_aware_chi_square_red1_total_sigma_arcsec"),
                ),
                (
                    "arc-aware red1 pos_sigma_arcsec",
                    summary.get("arc_aware_chi_square_red1_pos_sigma_arcsec"),
                ),
            ]
        )
    quality_items.extend(
        [
            ("chi-square red1 calibration", summary.get("chi_square_red1_calibration_note")),
            ("effective parameters", summary.get("n_effective_parameters")),
            ("fit-quality reference", summary.get("fit_quality_reference_sample_kind")),
            ("fit-quality sample index", summary.get("fit_quality_reference_sample_index")),
            ("fit-quality source log likelihood", summary.get("fit_quality_reference_source_loglike")),
            ("fit-quality log probability", summary.get("fit_quality_reference_log_prob")),
        ]
    )
    if use_arc_aware_diagnostics:
        quality_items.extend(
            [
                ("N_arc_supported", arc_supported_count),
                ("N_missing", missing_count),
            ]
        )
    quality_footer = [recovery_gate_note] if use_arc_aware_diagnostics else []
    lines = [
        "Cluster Solver Run Summary",
        f"run_name={_metric_text(summary.get('run_name'))}",
        "",
        "Lensing Information",
        *(
            _key_value_lines(
                [
                    ("fit mode", summary.get("fit_mode")),
                    ("likelihood mode", summary.get("sample_likelihood_mode")),
                    ("fit sampling engine", summary.get("fit_sampling_engine", summary.get("sampling_engine"))),
                    ("final validation sampling engine", summary.get("final_validation_sampling_engine")),
                    ("sampler", summary.get("sampler")),
                    ("runtime seconds", summary.get("runtime_sec")),
                    ("families", summary.get("n_families")),
                    ("images", summary.get("n_images")),
                    ("parameters", summary.get("n_parameters")),
                    ("scaling components", summary.get("n_scaling_galaxy_components")),
                    ("active scaling components", summary.get("active_scaling_components")),
                    ("lens redshift", summary.get("z_lens")),
                    ("source redshift range", source_range),
                    ("effective source planes", summary.get("distinct_effective_source_planes")),
                    ("quick diagnostics", summary.get("quick_diagnostics")),
                    ("image scatter floor arcsec", summary.get("image_plane_scatter_floor_arcsec")),
                    ("image scatter prior", summary.get("image_plane_scatter_prior")),
                    ("image sigma int sampled", summary.get("image_sigma_int_sampled")),
                    ("fixed image sigma int arcsec", summary.get("fixed_image_sigma_int_arcsec")),
                    ("likelihood max gain", summary.get("likelihood_stabilizer_max_gain")),
                    ("likelihood max residual arcsec", summary.get("likelihood_stabilizer_max_residual_arcsec")),
                    ("likelihood residual loss", summary.get("likelihood_stabilizer_residual_loss")),
                    ("likelihood Student-t nu", summary.get("likelihood_stabilizer_student_t_nu")),
                    ("best log likelihood", summary.get("best_loglike")),
                ]
            )
        ),
        "",
        "Quality Of Fit",
        "chi-square sigma: total image-plane sigma (image_sigma_eff_arcsec)",
        *_key_value_lines(quality_items),
        *quality_footer,
    ]
    return "\n".join(lines).rstrip() + "\n"


def _format_sequential_run_summary_text(
    stage_summaries: list[dict[str, Any]],
    *,
    run_name: str,
    root_dir: str | Path,
) -> str:
    lines = [
        "Sequential Cluster Solver Run Summary",
        f"run_name={run_name}",
        f"root_dir={Path(root_dir)}",
        "",
        "Stage Quality Comparison",
    ]
    base_columns = [
        ("stage", "stage"),
        ("fit", "fit_method"),
        ("likelihood", "sample_likelihood_mode"),
        ("sampler", "sampler"),
        ("families", "n_families"),
        ("images", "n_images"),
        ("headline_chi2", "headline_chi_square"),
        ("headline_dof", "headline_dof"),
        ("headline_red", "headline_reduced_chi_square"),
        ("point_RMS", "point_image_rms_arcsec"),
        ("point_med", "point_image_median_residual_arcsec"),
        ("N_point", "point_recovered_image_count"),
    ]
    arc_columns = [
        ("arc_chi2", "arc_aware_chi_square"),
        ("arc_dof", "arc_aware_dof"),
        ("arc_red", "arc_aware_reduced_chi_square"),
        ("N_arc", "arc_aware_arc_supported_image_count"),
        ("N_arcaware", "arc_aware_recovered_image_count"),
        ("N_missing", "arc_aware_missing_image_count"),
        ("arc_RMS", "arc_aware_image_rms_arcsec"),
        ("arc_med", "arc_aware_image_residual_median_arcsec"),
    ]
    tail_columns = [
        ("ESS_min", "ess_min"),
        ("Rhat_max", "rhat_max"),
        ("runtime_s", "runtime_sec"),
    ]
    include_arc_columns = any(_summary_uses_arc_aware_diagnostics(summary) for summary in stage_summaries)
    if include_arc_columns:
        rows = []
        for summary in stage_summaries:
            row = dict(summary)
            if not _summary_uses_arc_aware_diagnostics(summary):
                for _header, key in arc_columns:
                    row[key] = None
            rows.append(row)
        columns = [*base_columns, *arc_columns, *tail_columns]
    else:
        rows = stage_summaries
        columns = [*base_columns, *tail_columns]
    lines.extend(_table_text(rows, columns))
    return "\n".join(lines).rstrip() + "\n"


def _corner_dynamic_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    plot_name: str,
) -> tuple[np.ndarray, list[ParameterSpec]] | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] < 2 or sample_array.shape[1] == 0:
        _log(None, f"[plot:corner] skipped {plot_name}: need at least two samples and one parameter")
        return None
    spans = np.nanmax(sample_array, axis=0) - np.nanmin(sample_array, axis=0)
    dynamic_mask = np.isfinite(spans) & (spans > 0.0)
    if int(np.sum(dynamic_mask)) < 2:
        _log(None, f"[plot:corner] skipped {plot_name}: fewer than two parameters have dynamic range")
        return None
    if not np.all(dynamic_mask):
        dropped = [spec.name for spec, keep in zip(parameter_specs, dynamic_mask) if not keep]
        _log(None, f"[plot:corner] {plot_name}: dropped constant parameters={', '.join(dropped)}")
    subset_samples = sample_array[:, dynamic_mask]
    subset_specs = [spec for spec, keep in zip(parameter_specs, dynamic_mask) if keep]
    if subset_samples.shape[1] > subset_samples.shape[0]:
        spans_subset = spans[dynamic_mask]
        keep_count = max(2, int(subset_samples.shape[0]))
        keep_order = np.argsort(-spans_subset)[:keep_count]
        keep_order = np.sort(keep_order)
        dropped = [spec.name for idx, spec in enumerate(subset_specs) if idx not in set(keep_order.tolist())]
        _log(
            None,
            (
                f"[plot:corner] {plot_name}: limited plotted parameters to {keep_count} "
                f"because samples={subset_samples.shape[0]} < parameters={subset_samples.shape[1]}; "
                f"dropped={', '.join(dropped)}"
            ),
        )
        subset_samples = subset_samples[:, keep_order]
        subset_specs = [subset_specs[idx] for idx in keep_order.tolist()]
    return subset_samples, subset_specs


def _corner_without_source_positions(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    plot_name: str = "corner.pdf",
) -> tuple[np.ndarray, list[ParameterSpec]]:
    sample_array = np.asarray(samples, dtype=float)
    n_columns = sample_array.shape[1] if sample_array.ndim == 2 else len(parameter_specs)
    excluded_families = {"source_position", "independent_scaling"}
    keep_indices = [
        idx
        for idx, spec in enumerate(parameter_specs)
        if getattr(spec, "component_family", None) not in excluded_families and idx < n_columns
    ]
    excluded = [
        getattr(spec, "name", str(spec))
        for idx, spec in enumerate(parameter_specs)
        if getattr(spec, "component_family", None) in excluded_families and idx < n_columns
    ]
    if excluded:
        _log(None, f"[plot:corner] {plot_name}: excluded default-hidden parameters={', '.join(excluded)}")
    if sample_array.ndim != 2:
        subset_specs = [
            spec
            for spec in parameter_specs
            if getattr(spec, "component_family", None) not in excluded_families
        ]
        return sample_array, subset_specs
    return sample_array[:, keep_indices], [parameter_specs[idx] for idx in keep_indices]


def _plot_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    map_values: dict[str, float] | None = None,
    maximum_likelihood_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
    *,
    output_name: str = "corner.pdf",
    plot_datapoints: bool = False,
) -> None:
    if corner is None or not parameter_specs:
        return
    corner_samples, corner_specs = _corner_without_source_positions(samples, parameter_specs, output_name)
    if not corner_specs:
        return
    finite_samples = _finite_sample_rows(corner_samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, corner_specs, output_name)
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, output_name)} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": bool(plot_datapoints)}
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **corner_kwargs,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, output_name)
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_map(fig, subset_specs, map_values)
    _overplot_corner_maximum_likelihood(fig, subset_specs, maximum_likelihood_values)
    _finish_figure(fig, _plot_path(plot_dir, output_name), dpi=CORNER_PLOT_DPI, bbox_inches="tight")


def _best_fit_values_for_specs(
    parameter_specs: list[ParameterSpec],
    best_fit: np.ndarray,
) -> dict[str, float]:
    best_fit_array = np.asarray(best_fit, dtype=float).reshape(-1)
    return {
        spec.name: float(best_fit_array[idx])
        for idx, spec in enumerate(parameter_specs)
        if idx < best_fit_array.size
    }


def _map_values_for_specs(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    log_prob: np.ndarray | None,
) -> dict[str, float]:
    if log_prob is None:
        return {}
    sample_array = np.asarray(samples, dtype=float)
    log_prob_array = np.asarray(log_prob, dtype=float).reshape(-1)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return {}
    if log_prob_array.size != sample_array.shape[0] or not np.isfinite(log_prob_array).any():
        return {}
    best_index = int(np.nanargmax(log_prob_array))
    map_row = sample_array[best_index]
    return {
        spec.name: float(map_row[idx])
        for idx, spec in enumerate(parameter_specs)
        if idx < map_row.size
    }


def _fit_vector_values_for_specs(
    parameter_specs: list[ParameterSpec],
    fit_vector: np.ndarray | None,
) -> dict[str, float]:
    if fit_vector is None:
        return {}
    fit_array = np.asarray(fit_vector, dtype=float).reshape(-1)
    if fit_array.size == 0:
        return {}
    return {
        spec.name: float(fit_array[idx])
        for idx, spec in enumerate(parameter_specs)
        if idx < fit_array.size and np.isfinite(fit_array[idx])
    }


def _sample_index_values_for_specs(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    sample_index: Any,
) -> dict[str, float]:
    try:
        index = int(sample_index)
    except (TypeError, ValueError):
        return {}
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or not (0 <= index < sample_array.shape[0]):
        return {}
    row = sample_array[index]
    return {
        spec.name: float(row[idx])
        for idx, spec in enumerate(parameter_specs)
        if idx < row.size and np.isfinite(row[idx])
    }


def _subset_values_for_specs(
    parameter_specs: list[ParameterSpec],
    values: dict[str, float],
) -> dict[str, float]:
    if not values:
        return {}
    return {
        spec.name: float(values[spec.name])
        for spec in parameter_specs
        if spec.name in values and np.isfinite(values[spec.name])
    }


def _corner_values_for_specs(
    parameter_specs: list[ParameterSpec],
    values_by_name: dict[str, float] | None,
) -> list[float]:
    if not values_by_name:
        return []
    values: list[float] = []
    for spec in parameter_specs:
        value = float("nan")
        for key in (getattr(spec, "name", None), getattr(spec, "sample_name", None)):
            if key is None or key not in values_by_name:
                continue
            try:
                candidate = float(values_by_name[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate):
                value = candidate
                break
        values.append(value)
    return values


BayesCornerOverlay = dict[str, np.ndarray]


def _bayes_dat_headers(path: Path) -> list[str]:
    headers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            header = stripped[1:].strip()
            if header:
                headers.append(header)
    return headers


def _bayes_kpc_per_arcsec(state: BuildState) -> float:
    try:
        z_lens = float(getattr(state, "z_lens"))
        cosmo_config = dict(getattr(state, "cosmo_config", {}) or {})
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(z_lens) or z_lens <= 0.0 or not cosmo_config:
        return 1.0
    try:
        return float(_kpc_per_arcsec_from_config(z_lens, cosmo_config))
    except Exception as exc:  # pragma: no cover - defensive fallback for malformed old artifacts
        _log(None, f"[plot:corner] bayes.dat radius conversion unavailable; using arcsec values: {exc}")
        return 1.0


def _bayes_potfile_id(state: BuildState, index: int) -> str | None:
    potfiles = list(getattr(state, "potfiles", []) or [])
    if index < 0 or index >= len(potfiles):
        return None
    potfile = potfiles[index]
    return str(potfile.get("id", f"potfile{index}")) if isinstance(potfile, dict) else f"potfile{index}"


def _bayes_dat_column_key(
    header: str,
    state: BuildState,
    *,
    kpc_per_arcsec: float,
) -> tuple[str, float] | None:
    base_header = re.sub(r"\s+", " ", str(header).split("(", 1)[0].strip())
    if not base_header:
        return None
    ignored = {"nsample", "ln(lhood)", "ln likelihood", "ln_lhood", "chi2"}
    if base_header.lower() in ignored:
        return None

    object_match = re.match(r"^O(\d+)\s*:\s*(.+)$", base_header, flags=re.IGNORECASE)
    if object_match:
        potential_id = str(int(object_match.group(1)))
        raw_field = object_match.group(2).strip().lower()
        field_map = {
            "x": ("x_centre", 1.0),
            "y": ("y_centre", 1.0),
            "emass": ("ellipticite", 1.0),
            "ellipticity": ("ellipticite", 1.0),
            "ellipticite": ("ellipticite", 1.0),
            "theta": ("angle_pos", 1.0),
            "angle": ("angle_pos", 1.0),
            "angle_pos": ("angle_pos", 1.0),
            "rc": ("core_radius_kpc", kpc_per_arcsec),
            "core": ("core_radius_kpc", kpc_per_arcsec),
            "core_radius": ("core_radius_kpc", kpc_per_arcsec),
            "rcut": ("cut_radius_kpc", kpc_per_arcsec),
            "cut": ("cut_radius_kpc", kpc_per_arcsec),
            "cut_radius": ("cut_radius_kpc", kpc_per_arcsec),
            "sigma": ("v_disp", 1.0),
            "v_disp": ("v_disp", 1.0),
        }
        mapped = field_map.get(raw_field)
        if mapped is None:
            return None
        field_name, scale = mapped
        return f"{potential_id}.{field_name}", float(scale)

    potfile_match = re.match(r"^Pot(\d+)\s+(.+)$", base_header, flags=re.IGNORECASE)
    if potfile_match:
        potfile_id = _bayes_potfile_id(state, int(potfile_match.group(1)))
        if potfile_id is None:
            return None
        raw_field = potfile_match.group(2).strip().lower()
        field_map = {
            "sigma": ("sigma", 1.0),
            "rcut": ("cutkpc", kpc_per_arcsec),
            "cut": ("cutkpc", kpc_per_arcsec),
            "cut_radius": ("cutkpc", kpc_per_arcsec),
            "rc": ("corekpc", kpc_per_arcsec),
            "core": ("corekpc", kpc_per_arcsec),
            "core_radius": ("corekpc", kpc_per_arcsec),
            "vdslope": ("vdslope", 1.0),
            "slope": ("slope", 1.0),
        }
        mapped = field_map.get(raw_field)
        if mapped is None:
            return None
        field_name, scale = mapped
        return f"{potfile_id}.{field_name}", float(scale)

    return None


def _load_bayes_corner_overlay(path: str | Path | None, state: BuildState) -> BayesCornerOverlay | None:
    if path is None:
        return None
    bayes_path = Path(path)
    if not bayes_path.exists():
        _log(None, f"[plot:corner] bayes.dat overlay skipped; missing file: {bayes_path}")
        return None
    try:
        headers = _bayes_dat_headers(bayes_path)
        data = np.loadtxt(bayes_path, comments="#", dtype=float)
    except Exception as exc:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; failed to read {bayes_path}: {exc}")
        return None
    if data.size == 0:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no samples in {bayes_path}")
        return None
    data_array = np.asarray(data, dtype=float)
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)
    if not headers:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no column headers in {bayes_path}")
        return None
    kpc_per_arcsec = _bayes_kpc_per_arcsec(state)
    overlay: BayesCornerOverlay = {}
    for column_idx, header in enumerate(headers[: data_array.shape[1]]):
        mapping = _bayes_dat_column_key(header, state, kpc_per_arcsec=kpc_per_arcsec)
        if mapping is None:
            continue
        key, scale = mapping
        if key in overlay:
            continue
        values = np.asarray(data_array[:, column_idx], dtype=float) * float(scale)
        if np.isfinite(values).any():
            overlay[key] = values
    if not overlay:
        _log(None, f"[plot:corner] bayes.dat overlay skipped; no recognized parameter columns in {bayes_path}")
        return None
    _log(None, f"[plot:corner] bayes.dat overlay loaded path={bayes_path} columns={len(overlay)} samples={data_array.shape[0]}")
    return overlay


def _bayes_overlay_column_for_spec(
    spec: ParameterSpec,
    overlay: BayesCornerOverlay,
) -> np.ndarray | None:
    keys = [
        str(getattr(spec, "name", "")),
        str(getattr(spec, "sample_name", "")),
        f"{getattr(spec, 'potential_id', '')}.{getattr(spec, 'field', '')}",
    ]
    for key in keys:
        if key and key in overlay:
            return np.asarray(overlay[key], dtype=float)
    return None


def _bayes_overlay_samples_for_specs(
    parameter_specs: list[ParameterSpec],
    overlay: BayesCornerOverlay | None,
    plot_name: str,
) -> np.ndarray | None:
    if not overlay or not parameter_specs:
        return None
    columns: list[np.ndarray] = []
    missing: list[str] = []
    expected_rows: int | None = None
    for spec in parameter_specs:
        values = _bayes_overlay_column_for_spec(spec, overlay)
        if values is None:
            missing.append(str(getattr(spec, "name", spec)))
            continue
        if expected_rows is None:
            expected_rows = int(values.shape[0])
        if values.shape[0] != expected_rows:
            missing.append(str(getattr(spec, "name", spec)))
            continue
        columns.append(values)
    if missing:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; unmatched parameters={', '.join(missing)}")
        return None
    if not columns:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; no matching columns")
        return None
    samples = _finite_sample_rows(np.column_stack(columns))
    if samples.shape[0] == 0:
        _log(None, f"[plot:corner] {plot_name}: bayes.dat overlay skipped; no finite matching samples")
        return None
    return samples


def _overplot_bayes_corner_contours(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    overlay: BayesCornerOverlay | None,
    plot_name: str,
) -> None:
    if corner is None or overlay is None:
        return
    bayes_samples = _bayes_overlay_samples_for_specs(parameter_specs, overlay, plot_name)
    if bayes_samples is None:
        return
    kwargs = {
        **CORNER_PLOT_KWARGS,
        "fig": fig,
        "color": CORNER_BAYES_OVERLAY_COLOR,
        "fill_contours": False,
        "no_fill_contours": True,
        "plot_datapoints": False,
        "plot_density": False,
        "show_titles": False,
        "quantiles": [],
    }
    corner.corner(
        bayes_samples,
        labels=[spec.name for spec in parameter_specs],
        weights=_uniform_normalized_plot_weights(bayes_samples),
        **kwargs,
    )


def _overplot_corner_map(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    map_values: dict[str, float] | None,
) -> None:
    if corner is None or not map_values:
        return
    xs = _corner_values_for_specs(parameter_specs, map_values)
    if not xs or not any(np.isfinite(xs)):
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_MAP_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


def _overplot_corner_maximum_likelihood(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    maximum_likelihood_values: dict[str, float] | None,
) -> None:
    if corner is None or not maximum_likelihood_values:
        return
    xs = _corner_values_for_specs(parameter_specs, maximum_likelihood_values)
    if not xs or not any(np.isfinite(xs)):
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_MAXIMUM_LIKELIHOOD_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


def _overplot_corner_previous_stage_best_fit(
    fig: Any,
    parameter_specs: list[ParameterSpec],
    previous_stage_best_values: dict[str, float] | None,
) -> None:
    if corner is None or not previous_stage_best_values:
        return
    xs = _corner_values_for_specs(parameter_specs, previous_stage_best_values)
    if not xs or not any(np.isfinite(xs)):
        return
    point_xs = [[float(value) if np.isfinite(value) else np.nan for value in xs]]
    corner.overplot_points(
        fig,
        point_xs,
        marker="x",
        color=CORNER_PREVIOUS_STAGE_COLOR,
        markersize=5,
        markeredgewidth=1.2,
    )


_POTFILE_CORNER_INDEPENDENT_FIELDS: set[str] = set()


def _potfile_corner_parameter_subset(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    potfile_indices = [
        idx
        for idx, spec in enumerate(parameter_specs)
        if getattr(spec, "component_family", None) == "scaling"
        or (
            getattr(spec, "component_family", None) == "independent_scaling"
            and getattr(spec, "field", None) in _POTFILE_CORNER_INDEPENDENT_FIELDS
        )
    ]
    if not potfile_indices:
        return [], np.empty((samples.shape[0], 0), dtype=float), np.empty((0,), dtype=float)
    subset_specs = [parameter_specs[idx] for idx in potfile_indices]
    subset_samples = np.asarray(samples[:, potfile_indices], dtype=float)
    subset_best_fit = np.asarray(best_fit[potfile_indices], dtype=float)
    return subset_specs, subset_samples, subset_best_fit


def _cosmology_parameter_subset(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    cosmology_indices = [
        idx for idx, spec in enumerate(parameter_specs) if getattr(spec, "component_family", None) == "cosmology"
    ]
    if not cosmology_indices:
        return [], np.empty((samples.shape[0], 0), dtype=float), np.empty((0,), dtype=float)
    subset_specs = [parameter_specs[idx] for idx in cosmology_indices]
    subset_samples = np.asarray(samples[:, cosmology_indices], dtype=float)
    subset_best_fit = np.asarray(best_fit[cosmology_indices], dtype=float)
    return subset_specs, subset_samples, subset_best_fit


def _scaling_grouped_subset(
    parameter_specs: list[ParameterSpec],
    grouped_samples: np.ndarray | None,
) -> tuple[list[ParameterSpec], np.ndarray | None]:
    scaling_indices = [idx for idx, spec in enumerate(parameter_specs) if spec.component_family == "scaling"]
    if grouped_samples is None or not scaling_indices:
        return [], None
    grouped_array = np.asarray(grouped_samples, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.size == 0:
        return [], None
    subset_specs = [parameter_specs[idx] for idx in scaling_indices]
    subset_samples = np.asarray(grouped_array[:, :, scaling_indices], dtype=float)
    return subset_specs, subset_samples


def _plot_potfile_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    map_values: dict[str, float] | None = None,
    maximum_likelihood_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "potfile_corner.pdf")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'potfile_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **CORNER_PLOT_KWARGS,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, "potfile_corner.pdf")
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_map(fig, subset_specs, map_values)
    _overplot_corner_maximum_likelihood(fig, subset_specs, maximum_likelihood_values)
    _finish_figure(fig, _plot_path(plot_dir, "potfile_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")


def _plot_cosmology_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    map_values: dict[str, float] | None = None,
    maximum_likelihood_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
    bayes_corner_overlay: BayesCornerOverlay | None = None,
    *,
    plot_datapoints: bool = False,
) -> None:
    if corner is None or samples.size == 0 or not parameter_specs:
        return
    finite_samples = _finite_sample_rows(samples)
    if finite_samples.shape[0] == 0:
        return
    subset = _corner_dynamic_subset(finite_samples, parameter_specs, "cosmology_corner.pdf")
    if subset is None:
        return
    finite_samples, subset_specs = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'cosmology_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(finite_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    truths = _corner_values_for_specs(subset_specs, truth_values) if truth_values else None
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": bool(plot_datapoints)}
    fig = corner.corner(
        finite_samples,
        labels=labels,
        truths=truths,
        weights=_uniform_normalized_plot_weights(finite_samples),
        **corner_kwargs,
    )
    _overplot_bayes_corner_contours(fig, subset_specs, bayes_corner_overlay, "cosmology_corner.pdf")
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_map(fig, subset_specs, map_values)
    _overplot_corner_maximum_likelihood(fig, subset_specs, maximum_likelihood_values)
    _finish_figure(fig, _plot_path(plot_dir, "cosmology_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")


def _plot_trace(
    plot_dir: Path,
    grouped_samples: np.ndarray | None,
    parameter_specs: list[ParameterSpec],
    *,
    output_name: str = "trace_plot.png",
) -> None:
    if grouped_samples is None or not parameter_specs:
        return
    grouped_array = np.asarray(grouped_samples, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.size == 0:
        return
    n_chains, n_draws, n_params = grouped_array.shape
    if n_chains == 0 or n_draws == 0 or n_params == 0:
        return
    finite_mask = np.isfinite(grouped_array).all(axis=(1, 2))
    grouped_array = grouped_array[finite_mask]
    if grouped_array.size == 0:
        return
    nrows = len(parameter_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(4, 2.2 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    draw_index = np.arange(grouped_array.shape[1], dtype=int)
    cmap = plt.get_cmap("tab10", grouped_array.shape[0])
    for param_index, (ax, spec) in enumerate(zip(axes, parameter_specs)):
        for chain_index in range(grouped_array.shape[0]):
            ax.plot(
                draw_index,
                grouped_array[chain_index, :, param_index],
                linewidth=1.0,
                alpha=0.85,
                color=cmap(chain_index),
                label=f"chain {chain_index + 1}" if param_index == 0 else None,
            )
        ax.set_ylabel(spec.name)
    axes[-1].set_xlabel("posterior draw")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, output_name), dpi=180, bbox_inches="tight")


def _ns_series(ns_diagnostics: dict[str, np.ndarray] | None, key: str) -> np.ndarray | None:
    if not ns_diagnostics or key not in ns_diagnostics:
        return None
    array = np.asarray(ns_diagnostics[key], dtype=float).reshape(-1)
    return array if array.size else None


def _ns_aligned_core(ns_diagnostics: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    log_l = _ns_series(ns_diagnostics, "log_L_samples")
    log_dp = _ns_series(ns_diagnostics, "log_dp_mean")
    log_x = _ns_series(ns_diagnostics, "log_X_mean")
    if log_l is None or log_dp is None or log_x is None:
        return None
    n = min(log_l.size, log_dp.size, log_x.size)
    if n == 0:
        return None
    log_l = log_l[:n]
    log_dp = log_dp[:n]
    log_x = log_x[:n]
    finite = np.isfinite(log_l) & np.isfinite(log_dp) & np.isfinite(log_x)
    if not np.any(finite):
        return None
    return -log_x[finite], log_l[finite], log_dp[finite], finite


def _ns_normalized_weights(log_dp: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(log_dp)
    if not np.any(finite):
        return None
    weights = np.zeros_like(log_dp, dtype=float)
    max_log = float(np.max(log_dp[finite]))
    weights[finite] = np.exp(log_dp[finite] - max_log)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _plot_ns_diagnostics(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None) -> None:
    aligned = _ns_aligned_core(ns_diagnostics)
    if aligned is None:
        return
    neg_log_x, log_l, log_dp, finite_mask = aligned
    weights = _ns_normalized_weights(log_dp)
    if weights is None:
        return
    n = neg_log_x.size
    live_points = _ns_series(ns_diagnostics, "num_live_points_per_sample")
    evals = _ns_series(ns_diagnostics, "num_likelihood_evaluations_per_sample")
    if live_points is not None:
        live_points = live_points[:finite_mask.size][finite_mask]
    if evals is not None:
        evals = evals[:finite_mask.size][finite_mask]
    log_efficiency = _ns_series(ns_diagnostics, "log_efficiency")
    mean_eff = float(np.exp(log_efficiency[0])) if log_efficiency is not None and np.isfinite(log_efficiency[0]) else None
    rel_l = np.exp(log_l - np.nanmax(log_l))
    cumulative_weight = np.cumsum(weights)
    log_x = -neg_log_x
    log_xl = log_x + log_l
    rel_xl = np.exp(log_xl - np.nanmax(log_xl[np.isfinite(log_xl)]))

    fig, axes = plt.subplots(6, 1, figsize=(9, 15), sharex=True)
    if live_points is not None and live_points.size == n:
        axes[0].plot(neg_log_x, live_points, color="black", linewidth=1.0)
    axes[0].set_ylabel(r"$n_{\rm live}$")
    axes[0].set_title("Nested Sampling Diagnostics")

    axes[1].plot(neg_log_x, rel_l, color="black", linewidth=1.0)
    axes[1].set_ylabel(r"$L / L_{\rm max}$")
    axes[1].set_yscale("log")

    axes[2].plot(neg_log_x, weights, color="black", linewidth=1.0)
    axes[2].set_ylabel("posterior mass")
    axes[2].set_yscale("log")

    axes[3].plot(neg_log_x, cumulative_weight, color="black", linewidth=1.0)
    axes[3].set_ylabel("cum. mass")
    axes[3].set_ylim(-0.02, 1.02)

    if evals is not None and evals.size == n:
        efficiency = np.where(evals > 0, 1.0 / evals, np.nan)
        axes[4].scatter(neg_log_x, efficiency, s=5, color="black", alpha=0.75)
    if mean_eff is not None:
        axes[4].axhline(mean_eff, color="tab:red", linestyle="--", linewidth=1.0, label=f"mean={mean_eff:.3g}")
        axes[4].legend(loc="best", fontsize=8)
    axes[4].set_ylabel("efficiency")

    axes[5].plot(neg_log_x, rel_xl, color="black", linewidth=1.0)
    axes[5].set_ylabel(r"$X L$ rel.")
    axes[5].set_xlabel(r"$-\log X$")

    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "ns_diagnostics.pdf"), dpi=220, bbox_inches="tight")


def _ns_trace_parameter_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    *,
    max_params: int = 8,
) -> tuple[np.ndarray, list[ParameterSpec]] | None:
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0 or not parameter_specs:
        return None
    n_params = min(samples.shape[1], len(parameter_specs))
    finite = np.isfinite(samples[:, :n_params])
    variances = np.asarray(
        [
            np.nanvar(np.where(finite[:, idx], samples[:, idx], np.nan))
            for idx in range(n_params)
        ],
        dtype=float,
    )
    candidate_indices = [idx for idx in np.argsort(-variances).tolist() if np.isfinite(variances[idx]) and variances[idx] > 0.0]
    if not candidate_indices:
        candidate_indices = list(range(n_params))
    selected = candidate_indices[:max_params]
    return np.asarray(samples[:, selected], dtype=float), [parameter_specs[idx] for idx in selected]


def _plot_ns_trace(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None, parameter_specs: list[ParameterSpec]) -> None:
    aligned = _ns_aligned_core(ns_diagnostics)
    if aligned is None or not ns_diagnostics or "samples" not in ns_diagnostics:
        return
    neg_log_x, _log_l, log_dp, finite_mask = aligned
    samples = np.asarray(ns_diagnostics["samples"], dtype=float)
    if samples.ndim != 2 or samples.shape[0] < finite_mask.size:
        return
    samples = samples[:finite_mask.size][finite_mask]
    subset = _ns_trace_parameter_subset(samples, parameter_specs)
    if subset is None:
        return
    subset_samples, subset_specs = subset
    weights = _ns_normalized_weights(log_dp)
    color_values = np.log10(np.maximum(weights, np.nextafter(0.0, 1.0))) if weights is not None else None
    nrows = len(subset_specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(4, 2.2 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    mappable = None
    for idx, (ax, spec) in enumerate(zip(axes, subset_specs)):
        values = subset_samples[:, idx]
        finite_values = np.isfinite(values)
        if color_values is None:
            ax.plot(neg_log_x[finite_values], values[finite_values], color="tab:blue", linewidth=0.7, alpha=0.75)
        else:
            mappable = ax.scatter(
                neg_log_x[finite_values],
                values[finite_values],
                c=color_values[finite_values],
                cmap="viridis",
                s=8,
                alpha=0.85,
                linewidths=0.0,
                rasterized=True,
            )
        ax.set_ylabel(spec.name)
    axes[-1].set_xlabel(r"$-\log X$")
    if mappable is not None:
        fig.colorbar(mappable, ax=axes, label=r"$\log_{10}$ posterior weight", shrink=0.9)
        fig.subplots_adjust(right=0.86, hspace=0.25)
    else:
        fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "ns_trace_plot.pdf"), dpi=220, bbox_inches="tight")


def _plot_ns_weight_diagnostics(plot_dir: Path, ns_diagnostics: dict[str, np.ndarray] | None) -> None:
    log_dp = _ns_series(ns_diagnostics, "log_dp_mean")
    if log_dp is None:
        return
    weights = _ns_normalized_weights(log_dp)
    if weights is None:
        return
    positive = weights[np.isfinite(weights) & (weights > 0.0)]
    if positive.size == 0:
        return
    sorted_weights = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_weights)
    ess = 1.0 / float(np.sum(np.square(positive)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(np.log10(positive), bins=min(60, max(12, int(np.sqrt(positive.size)))), color="tab:blue", alpha=0.8)
    axes[0].set_xlabel(r"$\log_{10}$ normalized weight")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Weight distribution; ESS={ess:.3g}")
    axes[1].plot(np.arange(1, sorted_weights.size + 1), cumulative, color="black", linewidth=1.2)
    axes[1].set_xlabel("samples sorted by descending weight")
    axes[1].set_ylabel("cumulative weight")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xscale("log")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "ns_weight_diagnostics.pdf"), dpi=220, bbox_inches="tight")


def _logsafe_importance_values(values: pd.Series | np.ndarray | list[float]) -> tuple[np.ndarray, float]:
    importance = np.asarray(values, dtype=float)
    positive = importance[np.isfinite(importance) & (importance > 0.0)]
    floor = float(max(np.min(positive) * 0.5, np.nextafter(0.0, 1.0))) if positive.size else 1.0
    clipped = np.where(np.isfinite(importance) & (importance > 0.0), importance, floor)
    return clipped, floor


def _apply_log_importance_axis(ax: plt.Axes, values: pd.Series | np.ndarray | list[float]) -> None:
    _, floor = _logsafe_importance_values(values)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor * 0.8)


def _plot_scaling_rank_scatter(plot_dir: Path, scaling_rank_df: pd.DataFrame) -> None:
    if scaling_rank_df.empty:
        return
    potfile_ids = scaling_rank_df["potfile_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(potfile_ids), 1, figsize=(10, max(4, 3.4 * len(potfile_ids))), sharex=False)
    if len(potfile_ids) == 1:
        axes = [axes]
    for ax, potfile_id in zip(axes, potfile_ids):
        pot_df = scaling_rank_df[scaling_rank_df["potfile_id"] == potfile_id].copy()
        active_df = pot_df[pot_df["selected_active"]]
        inactive_df = pot_df[~pot_df["selected_active"]]
        inactive_importance, _ = _logsafe_importance_values(inactive_df["importance"])
        active_importance, _ = _logsafe_importance_values(active_df["importance"])
        ax.scatter(
            inactive_df["min_distance_arcsec"],
            inactive_importance,
            color="tab:gray",
            alpha=0.55,
            s=20,
            label="inactive",
        )
        ax.scatter(
            active_df["min_distance_arcsec"],
            active_importance,
            color="tab:orange",
            alpha=0.9,
            s=28,
            label="active",
        )
        top_df = pot_df.sort_values("rank").head(min(5, len(pot_df)))
        top_importance, _ = _logsafe_importance_values(top_df["importance"])
        for row, y_plot in zip(top_df.itertuples(index=False), top_importance.tolist()):
            ax.annotate(str(row.catalog_id), (row.min_distance_arcsec, y_plot), fontsize=7, alpha=0.8)
        ax.set_title(f"{potfile_id}: importance vs distance")
        ax.set_xlabel("min distance to observed images (arcsec)")
        ax.set_ylabel("importance")
        ax.legend(loc="best", fontsize=8)
        _apply_log_importance_axis(ax, pot_df["importance"])
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "scaling_rank_scatter.png"), dpi=180, bbox_inches="tight")


def _load_perturbation_discovery_diagnostics_table(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "perturbation_discovery_diagnostics.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _plot_perturbation_discovery_diagnostics(plot_dir: Path, diagnostics_df: pd.DataFrame) -> None:
    required = {
        "component_index",
        "catalog_id",
        "image_index",
        "score",
        "alpha_norm",
        "jacobian_norm",
        "selected_pair",
        "selected_galaxy",
        "alpha_tol_arcsec",
        "jacobian_tol",
        "jacobian_weight",
        "threshold_score",
    }
    if diagnostics_df.empty or not required.issubset(diagnostics_df.columns):
        return
    df = diagnostics_df.copy()
    for column in (
        "score",
        "alpha_norm",
        "jacobian_norm",
        "alpha_tol_arcsec",
        "jacobian_tol",
        "jacobian_weight",
        "threshold_score",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["selected_pair"] = df["selected_pair"].astype(bool)
    df["selected_galaxy"] = df["selected_galaxy"].astype(bool)
    df = df[np.isfinite(df["score"].to_numpy(dtype=float))]
    if df.empty:
        return
    if "selection_mode" not in df:
        df["selection_mode"] = "threshold"
    if "top_k_requested" not in df:
        df["top_k_requested"] = np.nan
    if "rank_score" not in df:
        df["rank_score"] = df["score"]
    if "rank_position" not in df:
        df["rank_position"] = np.nan
    df["rank_score"] = pd.to_numeric(df["rank_score"], errors="coerce")
    df["rank_position"] = pd.to_numeric(df["rank_position"], errors="coerce")

    max_idx = df.groupby("component_index", sort=False)["score"].idxmax()
    galaxy_df = df.loc[max_idx].copy().sort_values("score", ascending=False).reset_index(drop=True)
    if galaxy_df["rank_position"].notna().any():
        galaxy_df = galaxy_df.sort_values(["rank_position", "score"], ascending=[True, False]).reset_index(drop=True)
    image_df = (
        df.groupby(["image_index", "family_id", "image_label"], dropna=False)
        .agg(selected_pairs=("selected_pair", "sum"), max_score=("score", "max"))
        .reset_index()
        .sort_values("image_index")
    )
    threshold = float(np.nanmedian(pd.to_numeric(df["threshold_score"], errors="coerce")))
    if not np.isfinite(threshold) or threshold <= 0.0:
        threshold = 1.0
    alpha_tol = float(np.nanmedian(df["alpha_tol_arcsec"]))
    jacobian_tol = float(np.nanmedian(df["jacobian_tol"]))
    jacobian_weight = float(np.nanmedian(df["jacobian_weight"]))
    selection_mode = str(df["selection_mode"].dropna().astype(str).iloc[0]) if "selection_mode" in df and df["selection_mode"].notna().any() else "threshold"
    top_k_values = pd.to_numeric(df["top_k_requested"], errors="coerce").dropna()
    top_k_requested = float(np.nanmedian(top_k_values)) if not top_k_values.empty else float("nan")
    selected_galaxies = int(galaxy_df["selected_galaxy"].sum())
    selected_pairs = int(df["selected_pair"].sum())
    total_candidates = int(galaxy_df["component_index"].nunique())

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.2), constrained_layout=True)
    axes_flat = axes.ravel()
    selected_color = "#d62728"
    unselected_color = "#6f7782"

    def draw_score_boundary(ax: Any, *, xmax: float | None = None) -> None:
        x_limit = float(xmax) if xmax is not None and np.isfinite(float(xmax)) and float(xmax) > 0.0 else threshold
        x_limit = max(x_limit, threshold)
        curve_alpha = np.linspace(0.0, x_limit, 300)
        if np.isfinite(jacobian_weight) and jacobian_weight > 0.0:
            valid = curve_alpha <= threshold
            if np.any(valid):
                curve_jac = np.sqrt(np.maximum(threshold**2 - curve_alpha[valid] ** 2, 0.0) / jacobian_weight)
                ax.plot(curve_alpha[valid], curve_jac, color="black", linestyle="--", linewidth=1.0, label="score = 1")
        else:
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0, label="score = 1")

    def finite_positive_floor(values: np.ndarray) -> float:
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size == 0:
            return 1.0e-3
        return max(float(np.nanmin(finite)) * 0.5, 1.0e-6)

    def fraction_spans_decade(values: np.ndarray) -> bool:
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size < 2:
            return False
        return float(np.nanmax(finite)) / max(float(np.nanmin(finite)), 1.0e-12) > 10.0

    x = np.arange(len(galaxy_df), dtype=int)
    colors = np.where(galaxy_df["selected_galaxy"].to_numpy(dtype=bool), selected_color, unselected_color)
    axes_flat[0].scatter(x, galaxy_df["score"], c=colors, s=26, linewidth=0.0)
    axes_flat[0].axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="score = 1")
    if selection_mode == "top_k" and selected_galaxies > 0 and selected_galaxies < len(galaxy_df):
        axes_flat[0].axvline(selected_galaxies - 0.5, color="black", linestyle=":", linewidth=1.0, label="top-k split")
    axes_flat[0].set_yscale("log")
    axes_flat[0].set_xlabel("galaxies sorted by max score")
    axes_flat[0].set_ylabel("max image-galaxy score")
    axes_flat[0].set_title("selection rank" if selection_mode == "top_k" else "selection threshold")
    axes_flat[0].legend(loc="best", fontsize=8)

    axes_flat[1].plot(x, galaxy_df["alpha_norm"], color="#4c78a8", linewidth=1.2, label="alpha fraction")
    axes_flat[1].plot(x, galaxy_df["jacobian_norm"], color="#f58518", linewidth=1.2, label="Jacobian fraction")
    axes_flat[1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, label="unit tolerance")
    axes_flat[1].set_yscale("log")
    axes_flat[1].set_xlabel("galaxies sorted by max score")
    axes_flat[1].set_ylabel("fraction of tolerance")
    axes_flat[1].set_title("worst-image fractions")
    axes_flat[1].legend(loc="best", fontsize=8)

    selected_mask = galaxy_df["selected_galaxy"].to_numpy(dtype=bool)
    axes_flat[2].scatter(
        galaxy_df.loc[~selected_mask, "alpha_norm"],
        galaxy_df.loc[~selected_mask, "jacobian_norm"],
        color=unselected_color,
        s=24,
        alpha=0.65,
        label="not selected",
    )
    axes_flat[2].scatter(
        galaxy_df.loc[selected_mask, "alpha_norm"],
        galaxy_df.loc[selected_mask, "jacobian_norm"],
        color=selected_color,
        edgecolor="black",
        linewidth=0.4,
        s=34,
        alpha=0.9,
        label="selected",
    )
    draw_score_boundary(axes_flat[2], xmax=float(np.nanmax(galaxy_df["alpha_norm"])) if len(galaxy_df) else threshold)
    axes_flat[2].set_xlabel("alpha fraction = |alpha| / alpha_tol")
    axes_flat[2].set_ylabel("Jacobian fraction = ||Delta A||_F / jacobian_tol")
    axes_flat[2].set_title("worst image per galaxy")
    axes_flat[2].legend(loc="best", fontsize=8)

    alpha_all = np.asarray(df["alpha_norm"], dtype=float)
    jac_all = np.asarray(df["jacobian_norm"], dtype=float)
    heatmap_mask = np.isfinite(alpha_all) & np.isfinite(jac_all)
    alpha_heat = np.maximum(alpha_all[heatmap_mask], finite_positive_floor(alpha_all))
    jac_heat = np.maximum(jac_all[heatmap_mask], finite_positive_floor(jac_all))
    if alpha_heat.size and jac_heat.size:
        use_log_x = fraction_spans_decade(alpha_heat)
        use_log_y = fraction_spans_decade(jac_heat)
        x_max = max(float(np.nanmax(alpha_heat)) * 1.05, threshold * 1.05)
        y_max = max(float(np.nanmax(jac_heat)) * 1.05, threshold * 1.05)
        x_min = finite_positive_floor(alpha_heat) if use_log_x else 0.0
        y_min = finite_positive_floor(jac_heat) if use_log_y else 0.0
        x_bins = np.geomspace(x_min, x_max, 48) if use_log_x else np.linspace(x_min, x_max, 48)
        y_bins = np.geomspace(y_min, y_max, 48) if use_log_y else np.linspace(y_min, y_max, 48)
        hist = axes_flat[3].hist2d(
            alpha_heat,
            jac_heat,
            bins=(x_bins, y_bins),
            cmap="magma",
            norm=LogNorm(),
        )
        fig.colorbar(hist[3], ax=axes_flat[3], label="image-galaxy pairs")
        if use_log_x:
            axes_flat[3].set_xscale("log")
        if use_log_y:
            axes_flat[3].set_yscale("log")
        draw_score_boundary(axes_flat[3], xmax=x_max)
    else:
        axes_flat[3].text(0.5, 0.5, "No finite alpha/Jacobian fractions", ha="center", va="center")
    axes_flat[3].set_xlabel("alpha fraction = |alpha| / alpha_tol")
    axes_flat[3].set_ylabel("Jacobian fraction = ||Delta A||_F / jacobian_tol")
    axes_flat[3].set_title("all image-galaxy pairs")

    image_x = np.arange(len(image_df), dtype=int)
    axes_flat[4].bar(image_x, image_df["selected_pairs"].to_numpy(dtype=float), color="#4c78a8", alpha=0.85)
    axes_flat[4].set_xlabel("image")
    axes_flat[4].set_ylabel("selected galaxy pairs")
    axes_flat[4].set_title("selected pairs by image")
    if len(image_df) <= 24:
        labels = [
            f"{row.family_id}:{row.image_label}" if str(row.family_id) else str(row.image_index)
            for row in image_df.itertuples(index=False)
        ]
        axes_flat[4].set_xticks(image_x)
        axes_flat[4].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

    annotation = (
        f"alpha_tol = {alpha_tol:.4g} arcsec\n"
        f"jacobian_tol = {jacobian_tol:.4g}\n"
        f"jacobian_weight = {jacobian_weight:.4g}\n"
        f"selection_mode = {selection_mode}\n"
        f"top_k_requested = {int(top_k_requested) if np.isfinite(top_k_requested) else 'none'}\n"
        f"candidates = {total_candidates}\n"
        f"selected galaxies = {selected_galaxies}\n"
        f"selected pairs = {selected_pairs}"
    )
    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.02,
        0.98,
        annotation,
        transform=axes_flat[5].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.92},
    )
    fig.suptitle("Perturbation discovery diagnostics", fontsize=14)
    _finish_figure(fig, _plot_path(plot_dir, "perturbation_discovery_diagnostics.pdf"), dpi=200, bbox_inches="tight")


def _plot_scaling_relation_summary(plot_dir: Path, relation_df: pd.DataFrame) -> None:
    output_path = _plot_path(plot_dir, "scaling_relation_summary.pdf")
    if relation_df.empty:
        _write_placeholder_plot(
            output_path,
            "Scaling relation summary",
            "No modeled scaling galaxies are available.",
        )
        return
    df = relation_df.copy()
    df["potfile_id"] = df["potfile_id"].astype(str)
    df["catalog_color"] = pd.to_numeric(df["catalog_color"], errors="coerce")
    finite_catalog_color = df["catalog_color"].to_numpy(dtype=float)
    finite_catalog_color = finite_catalog_color[np.isfinite(finite_catalog_color)]
    if finite_catalog_color.size:
        color_min, color_max = (
            float(value)
            for value in np.nanpercentile(finite_catalog_color, [5.0, 95.0])
        )
        if color_min == color_max:
            color_min -= 0.5
            color_max += 0.5
    else:
        color_min, color_max = 0.0, 1.0
    catalog_color_norm = Normalize(vmin=color_min, vmax=color_max, clip=True)
    catalog_color_cmap = plt.get_cmap("coolwarm")
    fields = [
        ("v_disp", "velocity dispersion (km/s)"),
        ("core_radius_kpc", "core radius (kpc)"),
        ("cut_radius_kpc", "cut radius (kpc)"),
        ("log10_mass_msun", "log10 mass (Msun)"),
    ]
    potfile_ids = df["potfile_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(potfile_ids),
        len(fields),
        figsize=(16.0, max(4.2, 3.8 * len(potfile_ids))),
        squeeze=False,
        constrained_layout=True,
    )

    def _finite_xy(frame: pd.DataFrame, y_column: str) -> pd.DataFrame:
        mag = pd.to_numeric(frame.get("catalog_mag"), errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame.get(y_column), errors="coerce").to_numpy(dtype=float)
        return frame[np.isfinite(mag) & np.isfinite(y)]

    def _errorbar_points(
        ax: Any,
        frame: pd.DataFrame,
        *,
        y_column: str,
        p16_column: str,
        p84_column: str,
        label: str,
        fmt: str,
        markerfacecolor: str | None = None,
        markeredgecolor: str | None = None,
        markersize: float = 5.0,
        alpha: float = 0.9,
    ) -> None:
        finite = _finite_xy(frame, y_column)
        if finite.empty:
            return
        x = finite["catalog_mag"].to_numpy(dtype=float)
        y = finite[y_column].to_numpy(dtype=float)
        y_low = pd.to_numeric(finite.get(p16_column), errors="coerce").to_numpy(dtype=float)
        y_high = pd.to_numeric(finite.get(p84_column), errors="coerce").to_numpy(dtype=float)
        finite_err = np.isfinite(y_low) & np.isfinite(y_high) & (y_low <= y_high)
        color_values = pd.to_numeric(finite["catalog_color"], errors="coerce").to_numpy(dtype=float)
        for point_index, (x_value, y_value, y_low_value, y_high_value, has_err, color_value) in enumerate(
            zip(x, y, y_low, y_high, finite_err, color_values, strict=False)
        ):
            point_color = (
                catalog_color_cmap(catalog_color_norm(float(color_value)))
                if np.isfinite(color_value)
                else to_rgba("0.55", alpha=alpha)
            )
            error_color = (point_color[0], point_color[1], point_color[2], min(alpha, 0.55))
            face_color = markerfacecolor if markerfacecolor is not None else point_color
            edge_color = markeredgecolor if markeredgecolor is not None else point_color
            if bool(has_err):
                ax.plot(
                    [x_value, x_value],
                    [float(y_low_value), float(y_high_value)],
                    color=error_color,
                    linewidth=0.8,
                    alpha=min(alpha, 0.55),
                    linestyle="-",
                    label="_nolegend_",
                )
            ax.errorbar(
                [x_value],
                [y_value],
                fmt=fmt,
                color=point_color,
                markersize=markersize,
                markerfacecolor=face_color,
                markeredgecolor=edge_color,
                alpha=alpha,
                linestyle="none",
                label=label if point_index == 0 else "_nolegend_",
            )

    for row_idx, potfile_id in enumerate(potfile_ids):
        pot_df = df[df["potfile_id"] == potfile_id].copy()
        class_values = pot_df.get("scaling_relation_class", pd.Series("", index=pot_df.index)).astype(str)
        inactive = pot_df[class_values == "inactive"]
        active = pot_df[class_values == "active"]
        free = pot_df[class_values == "free"]
        counts_text = (
            f"total: {len(pot_df):d}\n"
            f"inactive: {len(inactive):d}\n"
            f"active not free: {len(active):d}\n"
            f"free: {len(free):d}"
        )
        for col_idx, (field, ylabel) in enumerate(fields):
            ax = axes[row_idx, col_idx]
            scaling_prefix = f"scaling_{field}"
            free_prefix = f"free_{field}"
            y_column = f"{scaling_prefix}_best"
            _errorbar_points(
                ax,
                inactive,
                y_column=y_column,
                p16_column=f"{scaling_prefix}_p16",
                p84_column=f"{scaling_prefix}_p84",
                label="inactive/cached",
                fmt="o",
                markersize=4.2,
                alpha=0.70,
            )
            _errorbar_points(
                ax,
                active,
                y_column=y_column,
                p16_column=f"{scaling_prefix}_p16",
                p84_column=f"{scaling_prefix}_p84",
                label="active exact",
                fmt="o",
                markerfacecolor="none",
                markersize=5.2,
                alpha=0.95,
            )
            _errorbar_points(
                ax,
                free,
                y_column=f"{free_prefix}_best",
                p16_column=f"{free_prefix}_p16",
                p84_column=f"{free_prefix}_p84",
                label="free branch",
                fmt="*",
                markeredgecolor="0.10",
                markersize=7.0,
                alpha=0.95,
            )
            curve_df = _finite_xy(pot_df, y_column)
            curve_y = pd.to_numeric(curve_df.get(y_column), errors="coerce").to_numpy(dtype=float)
            curve_x = pd.to_numeric(curve_df.get("catalog_mag"), errors="coerce").to_numpy(dtype=float)
            positive = np.isfinite(curve_x) & np.isfinite(curve_y) & (curve_y > 0.0)
            if np.sum(positive) >= 2 and float(np.nanmin(curve_x[positive])) < float(np.nanmax(curve_x[positive])):
                grid = np.linspace(float(np.nanmin(curve_x[positive])), float(np.nanmax(curve_x[positive])), 160)
                if field == "log10_mass_msun":
                    coeff = np.polyfit(curve_x[positive], curve_y[positive], deg=1)
                    ax.plot(grid, coeff[0] * grid + coeff[1], color="black", linewidth=1.2, label="scaling relation")
                else:
                    coeff = np.polyfit(curve_x[positive], np.log(curve_y[positive]), deg=1)
                    ax.plot(grid, np.exp(coeff[0] * grid + coeff[1]), color="black", linewidth=1.2, label="scaling relation")
            if field == "log10_mass_msun":
                pot_mag = pd.to_numeric(pot_df.get("catalog_mag"), errors="coerce").to_numpy(dtype=float)
                anchor_values = pd.to_numeric(pot_df.get("anchor_mag"), errors="coerce").to_numpy(dtype=float)
                lum_values = pd.to_numeric(pot_df.get("luminosity_ratio"), errors="coerce").to_numpy(dtype=float)
                mass_values = pd.to_numeric(pot_df.get(y_column), errors="coerce").to_numpy(dtype=float)
                finite_constant = (
                    np.isfinite(pot_mag)
                    & np.isfinite(anchor_values)
                    & np.isfinite(lum_values)
                    & (lum_values > 0.0)
                    & np.isfinite(mass_values)
                )
                if np.sum(finite_constant) >= 2 and np.sum(positive) >= 2:
                    anchor_mag = float(np.nanmedian(anchor_values[finite_constant]))
                    mass_star = float(np.nanmedian(mass_values[finite_constant] - np.log10(lum_values[finite_constant])))
                    grid = np.linspace(float(np.nanmin(curve_x[positive])), float(np.nanmax(curve_x[positive])), 160)
                    constant_ml = mass_star - 0.4 * (grid - anchor_mag)
                    ax.plot(
                        grid,
                        constant_ml,
                        color="0.35",
                        linewidth=1.0,
                        linestyle="--",
                        label="constant M/L",
                    )
            finite_y = pd.to_numeric(pot_df.get(y_column), errors="coerce").to_numpy(dtype=float)
            if field != "log10_mass_msun" and np.any(np.isfinite(finite_y) & (finite_y > 0.0)):
                ax.set_yscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("catalog magnitude")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{potfile_id}: {ylabel}")
            ax.grid(True, color="0.90", linewidth=0.6)
            if col_idx == 0:
                slope_lines: list[str] = ["points: best fit; bars: 16-84% posterior"]
                for label, column in (
                    ("alpha_sigma", "alpha_sigma_best"),
                    ("beta_radius", "beta_radius_best"),
                    ("gamma_ml", "gamma_ml_best"),
                ):
                    values = pd.to_numeric(pot_df.get(column), errors="coerce").to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if values.size:
                        slope_lines.append(f"{label}: {float(np.nanmedian(values)):.3g}")
                values = pd.to_numeric(pot_df.get("gamma_ml_best"), errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    mass_slope = 1.0 + float(np.nanmedian(values))
                    slope_lines.append(f"dlogM/dlogL: {mass_slope:.3g}")
                slope_lines.append("constant M/L: gamma_ml = 0, dlogM/dlogL = 1")
                ax.text(
                    0.02,
                    0.96,
                    "\n".join(slope_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "0.65",
                        "alpha": 0.88,
                    },
                )
                ax.text(
                    0.98,
                    0.04,
                    counts_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "0.65",
                        "alpha": 0.88,
                    },
                )
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=7)
    mappable = ScalarMappable(norm=catalog_color_norm, cmap=catalog_color_cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("catalog color (F606W - F814W)")
    _finish_figure(fig, output_path, dpi=180, bbox_inches="tight")


def _plot_unavailable_axis(ax: Any, label: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, f"{label} unavailable", ha="center", va="center", fontsize=9)


def _plot_grouped_lines(
    ax: Any,
    values: np.ndarray | None,
    *,
    cmap: Any,
    ylabel: str,
    title: str,
    draw_index: np.ndarray,
) -> None:
    if values is None:
        _plot_unavailable_axis(ax, title)
        return
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        _plot_unavailable_axis(ax, title)
        return
    for chain_index in range(array.shape[0]):
        chain_values = array[chain_index]
        finite = np.isfinite(chain_values)
        if finite.any():
            ax.plot(
                draw_index[finite],
                chain_values[finite],
                linewidth=1.0,
                alpha=0.85,
                color=cmap(chain_index),
                label=f"chain {chain_index + 1}",
            )
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_chain_health(
    plot_dir: Path,
    results: PosteriorResults,
    parameter_specs: list[ParameterSpec],
    *,
    max_tree_depth: int | None = None,
) -> None:
    grouped = results.grouped_samples
    if grouped is None:
        return
    grouped_array = np.asarray(grouped, dtype=float)
    if grouped_array.ndim != 3 or grouped_array.shape[0] == 0 or grouped_array.shape[1] == 0:
        return
    n_chains, n_draws, n_params = grouped_array.shape
    grouped_log_prob = (
        np.asarray(results.grouped_log_prob, dtype=float)
        if results.grouped_log_prob is not None
        else _grouped_chain_array(results.log_prob, n_chains, n_draws, dtype=float)
    )
    if grouped_log_prob is not None and grouped_log_prob.shape != (n_chains, n_draws):
        grouped_log_prob = None
    grouped_accept = _grouped_chain_array(results.accept_prob, n_chains, n_draws, dtype=float)
    grouped_steps = _grouped_chain_array(results.num_steps, n_chains, n_draws, dtype=float)
    image_sigma_idx = _image_sigma_int_index(parameter_specs, n_params)
    image_sigma = grouped_array[:, :, image_sigma_idx] if image_sigma_idx is not None else None
    draw_index = np.arange(n_draws, dtype=int)
    cmap = plt.get_cmap("tab10", n_chains)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    _plot_grouped_lines(
        axes[0],
        grouped_log_prob,
        cmap=cmap,
        ylabel="log posterior",
        title="Chain Log Posterior",
        draw_index=draw_index,
    )
    _plot_grouped_lines(
        axes[1],
        grouped_accept,
        cmap=cmap,
        ylabel="accept prob",
        title="Chain Acceptance Probability",
        draw_index=draw_index,
    )
    _plot_grouped_lines(
        axes[2],
        grouped_steps,
        cmap=cmap,
        ylabel="NUTS steps",
        title="Chain NUTS Integrator Steps",
        draw_index=draw_index,
    )
    if max_tree_depth is not None and grouped_steps is not None:
        axes[2].axhline(2**int(max_tree_depth) - 1, color="black", linestyle="--", linewidth=0.8, alpha=0.75)
    _plot_grouped_lines(
        axes[3],
        image_sigma,
        cmap=cmap,
        ylabel="arcsec",
        title="image.sigma_int by Chain",
        draw_index=draw_index,
    )
    axes[-1].set_xlabel("posterior draw")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "chain_health.pdf"), dpi=180, bbox_inches="tight")


def _finite_1d_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    finite = np.isfinite(array)
    if not np.any(finite):
        return None
    return array[finite]


def _has_smc_plot_data(results: PosteriorResults) -> bool:
    return (
        str(getattr(results, "sampler", "")) == "blackjax_smc"
        or results.temperature_schedule is not None
        or results.ess_history is not None
        or results.move_acceptance_history is not None
        or results.sample_weights is not None
    )


def _plot_smc_diagnostics(plot_dir: Path, results: PosteriorResults) -> None:
    temperature = _finite_1d_array(results.temperature_schedule)
    ess = _finite_1d_array(results.ess_history)
    acceptance = _finite_1d_array(results.move_acceptance_history)
    if temperature is None and ess is None and acceptance is None:
        return

    init_diag = results.init_diagnostics or {}
    n_particles = int(init_diag.get("smc_particles", 0) or 0)
    if n_particles <= 0 and results.sample_weights is not None:
        n_particles = int(np.asarray(results.sample_weights).size)
    target_ess_frac = init_diag.get("smc_target_ess_frac")
    try:
        target_ess_frac_value = float(target_ess_frac)
    except (TypeError, ValueError):
        target_ess_frac_value = float("nan")
    mean_acceptance = init_diag.get("smc_mean_move_acceptance")
    try:
        mean_acceptance_value = float(mean_acceptance)
    except (TypeError, ValueError):
        mean_acceptance_value = float("nan")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    if temperature is not None:
        steps = np.arange(temperature.size, dtype=int)
        axes[0].plot(steps, temperature, marker="o", markersize=3, color="tab:blue", linewidth=1.1)
        axes[0].set_ylabel("temperature")
        axes[0].set_xlabel("SMC step")
        axes[0].set_ylim(min(-0.02, float(np.nanmin(temperature)) - 0.02), max(1.02, float(np.nanmax(temperature)) + 0.02))
        axes[0].set_title("SMC Adaptive Tempering")
        if temperature.size > 1:
            axes[1].plot(
                np.arange(1, temperature.size, dtype=int),
                np.diff(temperature),
                marker="o",
                markersize=3,
                color="tab:purple",
                linewidth=1.1,
            )
            axes[1].set_ylabel("delta temperature")
            axes[1].set_xlabel("SMC step")
        else:
            axes[1].axis("off")
    else:
        axes[0].axis("off")
        axes[1].axis("off")

    if ess is not None:
        ess_steps = np.arange(ess.size, dtype=int)
        if n_particles > 0:
            ess_values = ess / float(n_particles)
            axes[2].set_ylabel("ESS fraction")
            if np.isfinite(target_ess_frac_value) and target_ess_frac_value > 0.0:
                axes[2].axhline(
                    target_ess_frac_value,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"target={target_ess_frac_value:.3g}",
                )
                axes[2].legend(loc="best", fontsize=8)
            axes[2].set_ylim(0.0, max(1.02, float(np.nanmax(ess_values)) * 1.05))
        else:
            ess_values = ess
            axes[2].set_ylabel("ESS")
        axes[2].plot(ess_steps, ess_values, marker="o", markersize=3, color="tab:green", linewidth=1.1)
        axes[2].set_xlabel("SMC step")
    else:
        axes[2].axis("off")

    if acceptance is not None:
        acceptance_steps = np.arange(1, acceptance.size + 1, dtype=int)
        axes[3].plot(acceptance_steps, acceptance, marker="o", markersize=3, color="tab:orange", linewidth=1.1)
        if np.isfinite(mean_acceptance_value):
            axes[3].axhline(
                mean_acceptance_value,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"mean={mean_acceptance_value:.3g}",
            )
            axes[3].legend(loc="best", fontsize=8)
        axes[3].set_ylim(-0.02, max(1.02, float(np.nanmax(acceptance)) * 1.05))
        axes[3].set_ylabel("move acceptance")
        axes[3].set_xlabel("SMC mutation step")
    else:
        axes[3].axis("off")

    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "smc_diagnostics.pdf"), dpi=220, bbox_inches="tight")


def _plot_smc_weight_diagnostics(plot_dir: Path, results: PosteriorResults) -> None:
    n_samples = int(np.asarray(results.samples).shape[0]) if np.asarray(results.samples).ndim >= 1 else 0
    weights = _valid_normalized_sample_weights(results.sample_weights, n_samples)
    if weights is None:
        return
    positive = weights[np.isfinite(weights) & (weights > 0.0)]
    if positive.size == 0:
        return
    log_prob = np.asarray(results.log_prob, dtype=float).reshape(-1)
    log_prob = log_prob if log_prob.size == n_samples else None

    sorted_weights = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_weights)
    ess = 1.0 / float(np.sum(np.square(weights)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes[0, 0].hist(np.log10(positive), bins=min(60, max(12, int(np.sqrt(positive.size)))), color="tab:blue", alpha=0.8)
    axes[0, 0].set_xlabel(r"$\log_{10}$ normalized weight")
    axes[0, 0].set_ylabel("particle count")
    axes[0, 0].set_title(f"Final SMC weights; ESS={ess:.3g}")

    axes[0, 1].plot(np.arange(1, sorted_weights.size + 1), cumulative, color="black", linewidth=1.2)
    axes[0, 1].set_xlabel("particles sorted by descending weight")
    axes[0, 1].set_ylabel("cumulative weight")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_xscale("log")

    if log_prob is not None and np.isfinite(log_prob).any():
        finite_log_prob = np.isfinite(log_prob)
        ranked = np.sort(log_prob[finite_log_prob])[::-1]
        axes[1, 0].plot(np.arange(1, ranked.size + 1), ranked, color="tab:red", linewidth=1.0)
        axes[1, 0].set_xlabel("particles sorted by descending log posterior")
        axes[1, 0].set_ylabel("log posterior")
        positive_mask = (weights > 0.0) & finite_log_prob
        axes[1, 1].scatter(
            np.log10(weights[positive_mask]),
            log_prob[positive_mask],
            s=10,
            alpha=0.75,
            color="tab:purple",
            linewidths=0.0,
            rasterized=True,
        )
        axes[1, 1].set_xlabel(r"$\log_{10}$ normalized weight")
        axes[1, 1].set_ylabel("log posterior")
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "smc_weight_diagnostics.pdf"), dpi=220, bbox_inches="tight")


def _weighted_variance(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    means = np.sum(samples * weights[:, None], axis=0)
    return np.sum(np.square(samples - means[None, :]) * weights[:, None], axis=0)


def _smc_corner_subset(
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    sample_weights: np.ndarray | None,
    *,
    max_params: int = SMC_CORNER_MAX_PARAMS,
) -> tuple[np.ndarray, list[ParameterSpec], np.ndarray] | None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] < 2 or sample_array.shape[1] == 0 or not parameter_specs:
        return None
    weights = _valid_normalized_sample_weights(sample_weights, sample_array.shape[0])
    if weights is None:
        return None
    corner_samples, corner_specs = _corner_without_source_positions(sample_array, parameter_specs, "smc_corner.pdf")
    if corner_samples.ndim != 2 or not corner_specs:
        return None
    n_params = min(corner_samples.shape[1], len(corner_specs))
    corner_samples = np.asarray(corner_samples[:, :n_params], dtype=float)
    corner_specs = corner_specs[:n_params]
    finite_rows = np.isfinite(corner_samples).all(axis=1) & np.isfinite(weights)
    if int(np.sum(finite_rows)) < 2:
        return None
    corner_samples = corner_samples[finite_rows]
    weights = _valid_normalized_sample_weights(weights[finite_rows], int(np.sum(finite_rows)))
    if weights is None:
        return None

    spans = np.nanmax(corner_samples, axis=0) - np.nanmin(corner_samples, axis=0)
    variances = _weighted_variance(corner_samples, weights)
    dynamic = np.isfinite(spans) & (spans > 0.0) & np.isfinite(variances) & (variances > 0.0)
    if int(np.sum(dynamic)) < 2:
        return None
    cosmology_indices = [
        idx
        for idx, spec in enumerate(corner_specs)
        if dynamic[idx] and getattr(spec, "component_family", None) == "cosmology"
    ]
    other_indices = [
        idx
        for idx in np.argsort(-variances).tolist()
        if dynamic[idx] and idx not in set(cosmology_indices)
    ]
    selected = (cosmology_indices + other_indices)[: max(2, int(max_params))]
    if len(selected) < 2:
        return None
    return corner_samples[:, selected], [corner_specs[idx] for idx in selected], weights


def _plot_smc_corner(
    plot_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
    sample_weights: np.ndarray | None,
    best_fit_values: dict[str, float] | None = None,
    map_values: dict[str, float] | None = None,
    maximum_likelihood_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> None:
    if corner is None:
        return
    subset = _smc_corner_subset(samples, parameter_specs, sample_weights)
    if subset is None:
        return
    subset_samples, subset_specs, subset_weights = subset
    _log(
        None,
        f"[plot:corner] path={_plot_path(plot_dir, 'smc_corner.pdf')} ndim={len(subset_specs)} samples_shape={tuple(subset_samples.shape)}",
    )
    labels = [spec.name for spec in subset_specs]
    corner_kwargs = {**CORNER_PLOT_KWARGS, "plot_datapoints": True}
    fig = corner.corner(subset_samples, labels=labels, weights=subset_weights, **corner_kwargs)
    _overplot_corner_previous_stage_best_fit(fig, subset_specs, previous_stage_best_values)
    _overplot_corner_map(fig, subset_specs, map_values)
    _overplot_corner_maximum_likelihood(fig, subset_specs, maximum_likelihood_values)
    _finish_figure(fig, _plot_path(plot_dir, "smc_corner.pdf"), dpi=CORNER_PLOT_DPI, bbox_inches="tight")


def _source_plane_residual_components(
    state: BuildState,
    best_eval: EvaluationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx_values: list[np.ndarray] = []
    dy_values: list[np.ndarray] = []
    norm_x_values: list[np.ndarray] = []
    norm_y_values: list[np.ndarray] = []
    radial_values: list[np.ndarray] = []
    for family in state.family_data:
        if family.n_images < 2:
            continue
        pred = best_eval.family_predictions.get(family.family_id)
        if not pred or bool(pred.get("failed", False)):
            continue
        source_x = pred.get("source_x")
        source_y = pred.get("source_y")
        if source_x is None or source_y is None or not np.isfinite(source_x) or not np.isfinite(source_y):
            continue
        sigma = float(pred.get("source_sigma_eff_arcsec", family.sigma_arcsec))
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        if "source_beta_x" not in pred or "source_beta_y" not in pred:
            continue
        beta_x = np.asarray(pred["source_beta_x"], dtype=float)
        beta_y = np.asarray(pred["source_beta_y"], dtype=float)
        if beta_x.shape != family.x_obs.shape or beta_y.shape != family.y_obs.shape:
            continue
        dx = beta_x - float(source_x)
        dy = beta_y - float(source_y)
        finite = np.isfinite(dx) & np.isfinite(dy)
        if not finite.any():
            continue
        dx = dx[finite]
        dy = dy[finite]
        dx_values.append(dx)
        dy_values.append(dy)
        norm_x_values.append(dx / sigma)
        norm_y_values.append(dy / sigma)
        radial_values.append(np.sqrt(dx**2 + dy**2) / sigma)
    dx = np.concatenate(dx_values) if dx_values else np.asarray([], dtype=float)
    dy = np.concatenate(dy_values) if dy_values else np.asarray([], dtype=float)
    norm_x = np.concatenate(norm_x_values) if norm_x_values else np.asarray([], dtype=float)
    norm_y = np.concatenate(norm_y_values) if norm_y_values else np.asarray([], dtype=float)
    radial = np.concatenate(radial_values) if radial_values else np.asarray([], dtype=float)
    return dx, dy, norm_x, norm_y, radial


def _plot_gaussian_component_histogram(
    ax: Any,
    component: np.ndarray,
    *,
    xlabel: str,
    title: str,
    standard_normal: bool = False,
) -> None:
    bins = min(60, max(16, int(np.sqrt(component.size))))
    ax.hist(component, bins=bins, density=True, alpha=0.65, color="tab:blue", label="components")
    x_min = float(np.nanmin(component))
    x_max = float(np.nanmax(component))
    span = max(abs(x_min), abs(x_max), 1.0)
    x_grid = np.linspace(-1.05 * span, 1.05 * span, 400)
    mu = float(np.mean(component))
    std = float(np.std(component, ddof=1)) if component.size > 1 else float("nan")
    if standard_normal:
        ax.plot(x_grid, norm.pdf(x_grid, 0.0, 1.0), color="black", linestyle="--", label="N(0, 1)")
    if np.isfinite(std) and std > 0:
        ax.plot(x_grid, norm.pdf(x_grid, mu, std), color="tab:red", label=f"fit mu={mu:.2f}, sigma={std:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _plot_source_plane_residual_histogram(
    plot_dir: Path,
    state: BuildState,
    best_eval: EvaluationResult,
) -> None:
    dx, dy, norm_x, norm_y, radial = _source_plane_residual_components(state, best_eval)
    if radial.size == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    arcsec_component = np.concatenate([dx, dy])
    normalized_component = np.concatenate([norm_x, norm_y])
    _plot_gaussian_component_histogram(
        axes[0],
        arcsec_component,
        xlabel="Signed source-plane residual (arcsec)",
        title="Signed Components",
    )
    _plot_gaussian_component_histogram(
        axes[1],
        normalized_component,
        xlabel="Signed source-plane residual / sigma_eff",
        title="Normalized Components",
        standard_normal=True,
    )

    bins = min(60, max(16, int(np.sqrt(radial.size))))
    max_x = max(4.0, float(np.nanmax(radial)) * 1.05)
    axes[2].hist(radial, bins=bins, density=True, alpha=0.65, color="tab:purple", label="radial")
    x_grid = np.linspace(0.0, max_x, 400)
    rayleigh_pdf = x_grid * np.exp(-0.5 * x_grid**2)
    axes[2].plot(x_grid, rayleigh_pdf, color="black", linestyle="--", label="Rayleigh scale=1")
    median = float(np.median(radial))
    p68, p95 = np.quantile(radial, [0.68, 0.95])
    axes[2].axvline(median, color="tab:red", linewidth=1.5, label=f"median={median:.2f}")
    axes[2].text(
        0.98,
        0.95,
        f"N={radial.size}\n68%={p68:.2f}\n95%={p95:.2f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    axes[2].set_xlabel("Radial source-plane residual / sigma_eff")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Radial Residuals")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "source_plane_residual_histogram.png"), dpi=180, bbox_inches="tight")


def _fit_quality_quantiles(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def _fit_quality_median_std(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(finite))
    std = float(np.std(finite))
    return median - std, median, median + std


def _capped_fit_quality_samples(samples: np.ndarray, max_draws: int) -> np.ndarray:
    sample_array = _finite_sample_rows(samples)
    if sample_array.shape[0] == 0 or int(max_draws) <= 0:
        return np.empty((0, sample_array.shape[1]), dtype=float)
    if sample_array.shape[0] <= int(max_draws):
        return sample_array
    indices = np.linspace(0, sample_array.shape[0] - 1, int(max_draws), dtype=int)
    return sample_array[indices]


def _reported_physical_to_latent_vector(evaluator: Any, theta: np.ndarray) -> np.ndarray:
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        return np.asarray(evaluator.reported_physical_to_latent_parameter_vector(theta), dtype=float)
    state = getattr(evaluator, "state", None)
    parameter_specs = list(getattr(state, "parameter_specs", []))
    if not parameter_specs:
        return np.asarray(theta, dtype=float).copy()
    return _convert_theta_to_latent(np.asarray(theta, dtype=float), parameter_specs)


def _fit_quality_image_sigma_int(evaluator: Any, params_latent: np.ndarray) -> float:
    return _shared_image_sigma_int_for_params(evaluator, params_latent)


def _fit_quality_critical_arc_singular_threshold(evaluator: Any, params_latent: np.ndarray) -> float:
    if hasattr(evaluator, "_critical_arc_singular_threshold_numpy"):
        try:
            value = float(evaluator._critical_arc_singular_threshold_numpy(params_latent))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            return value
    try:
        value = float(getattr(evaluator, "critical_arc_singular_threshold"))
    except Exception:
        value = float("nan")
    return value if np.isfinite(value) else CRITICAL_ARC_SINGULAR_THRESHOLD


def _fit_quality_critical_arc_singular_softness(evaluator: Any, params_latent: np.ndarray) -> float:
    if hasattr(evaluator, "_critical_arc_singular_softness_numpy"):
        try:
            value = float(evaluator._critical_arc_singular_softness_numpy(params_latent))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            return value
    try:
        value = float(getattr(evaluator, "critical_arc_singular_softness"))
    except Exception:
        value = float("nan")
    return value if np.isfinite(value) else CRITICAL_ARC_SINGULAR_SOFTNESS


def _fit_quality_image_sigma_eff(
    measurement_sigma_arcsec: float,
    image_sigma_int_arcsec: float,
    covariance_floor: float,
) -> float:
    return _shared_image_sigma_eff_arcsec(measurement_sigma_arcsec, image_sigma_int_arcsec, covariance_floor)


def _finite_or(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _covered_by_inflated_interval(
    observed: float,
    lower: float,
    upper: float,
    center: float,
    sigma_eff: float,
) -> bool:
    if not np.isfinite(observed) or not np.isfinite(sigma_eff):
        return False
    lo = lower if np.isfinite(lower) else center
    hi = upper if np.isfinite(upper) else center
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    lo, hi = sorted((float(lo), float(hi)))
    return bool((float(observed) >= lo - float(sigma_eff)) and (float(observed) <= hi + float(sigma_eff)))


def _fit_quality_value(df: pd.DataFrame, column: str, *fallback_columns: str) -> np.ndarray:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    else:
        values = np.full(len(df), np.nan, dtype=float)
    for fallback_column in fallback_columns:
        if fallback_column not in df.columns:
            continue
        fallback = pd.to_numeric(df[fallback_column], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, fallback)
    return values


def _merge_fit_quality_with_magnification(image_df: pd.DataFrame, magnification_df: pd.DataFrame) -> pd.DataFrame:
    if image_df.empty or magnification_df.empty:
        return pd.DataFrame()
    mag_columns = [
        column
        for column in [
            "family_id",
            "image_label",
            "magnification_model",
            "magnification_model_q16",
            "magnification_model_q50",
            "magnification_model_q84",
        ]
        if column in magnification_df.columns
    ]
    if "image_label" not in mag_columns:
        return pd.DataFrame()
    merge_keys = ["image_label"]
    if "family_id" in image_df.columns and "family_id" in mag_columns:
        merge_keys.insert(0, "family_id")
    return image_df.merge(magnification_df[mag_columns], on=merge_keys, how="inner")


def _unavailable_image_count_info(family: Any, reason: str) -> dict[str, Any]:
    return _shared_unavailable_image_count_info(family, reason)


def _successful_image_count_info(family: Any) -> dict[str, Any]:
    return _shared_successful_image_count_info(family)


def _image_count_info_from_exact_details(family: Any, details: dict[str, Any]) -> dict[str, Any]:
    return _shared_image_count_info_from_exact_details(family, details)


def _model_count_fields_from_count_info(count_info: dict[str, Any]) -> dict[str, Any]:
    return _shared_model_count_fields_from_count_info(count_info)


def _image_count_recovery_row(family: Any, count_info: dict[str, Any]) -> dict[str, Any]:
    return _shared_image_count_recovery_row(family, count_info)


def _image_count_recovery_table(state: BuildState, image_df: pd.DataFrame) -> pd.DataFrame:
    return _shared_image_count_recovery_table(state, image_df)


def _image_count_recovery_summary(image_count_df: pd.DataFrame | None) -> dict[str, Any]:
    return _shared_image_count_recovery_summary(image_count_df)


def _clone_fit_quality_evaluator(evaluator: Any, args: argparse.Namespace) -> Any:
    from .cluster_solver import (  # local import avoids a module import cycle at plotting import time
        ClusterJAXEvaluator,
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD,
        DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC,
        DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC,
        DEFAULT_CRITICAL_ARC_BASE_PROB,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_MAX_PROB,
        DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC,
        DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS,
        DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_FINDER,
        DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_LM_MAX_ITER,
        DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_EXACT_IMAGE_NUM_ITER_MAX,
        DEFAULT_EXACT_IMAGE_PRECISION_LIMIT,
        DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_MAGNITUDE_MIN_RELIABILITY,
        DEFAULT_MAGNITUDE_MU_FLOOR,
        DEFAULT_MAGNITUDE_SIGMA_FLOOR,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
        DEFAULT_USE_MAGNITUDE_LIKELIHOOD,
        SAMPLE_LIKELIHOOD_SOURCE,
    )

    match_tolerance_arcsec = float(
        getattr(args, "match_tolerance_arcsec", getattr(evaluator, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
    )
    exact_image_min_distance_arcsec = float(
        getattr(
            args,
            "exact_image_min_distance_arcsec",
            getattr(evaluator, "exact_image_min_distance_arcsec", DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC),
        )
    )
    exact_image_precision_limit = float(
        getattr(
            args,
            "exact_image_precision_limit",
            getattr(evaluator, "exact_image_precision_limit", DEFAULT_EXACT_IMAGE_PRECISION_LIMIT),
        )
    )
    exact_image_num_iter_max = int(
        getattr(
            args,
            "exact_image_num_iter_max",
            getattr(evaluator, "exact_image_num_iter_max", DEFAULT_EXACT_IMAGE_NUM_ITER_MAX),
        )
    )
    exact_image_finder = str(
        getattr(
            args,
            "exact_image_finder",
            getattr(evaluator, "exact_image_finder", DEFAULT_EXACT_IMAGE_FINDER),
        )
    )
    exact_image_displacement_tol_arcsec = float(
        getattr(
            args,
            "exact_image_displacement_tol_arcsec",
            getattr(
                evaluator,
                "exact_image_displacement_tol_arcsec",
                DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
            ),
        )
    )
    exact_image_identification_tol_arcsec = float(
        getattr(
            args,
            "exact_image_identification_tol_arcsec",
            getattr(
                evaluator,
                "exact_image_identification_tol_arcsec",
                DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
            ),
        )
    )
    exact_image_lm_max_iter = int(
        getattr(
            args,
            "exact_image_lm_max_iter",
            getattr(evaluator, "exact_image_lm_max_iter", DEFAULT_EXACT_IMAGE_LM_MAX_ITER),
        )
    )
    exact_image_lm_trust_radius_arcsec = float(
        getattr(
            args,
            "exact_image_lm_trust_radius_arcsec",
            getattr(evaluator, "exact_image_lm_trust_radius_arcsec", DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC),
        )
    )
    exact_image_adaptive_max_levels = int(
        getattr(
            args,
            "exact_image_adaptive_max_levels",
            getattr(evaluator, "exact_image_adaptive_max_levels", DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS),
        )
    )

    return ClusterJAXEvaluator(
        state=evaluator.state,
        match_tolerance_arcsec=match_tolerance_arcsec,
        exact_image_min_distance_arcsec=exact_image_min_distance_arcsec,
        exact_image_precision_limit=exact_image_precision_limit,
        exact_image_num_iter_max=exact_image_num_iter_max,
        exact_image_finder=exact_image_finder,
        exact_image_displacement_tol_arcsec=exact_image_displacement_tol_arcsec,
        exact_image_identification_tol_arcsec=exact_image_identification_tol_arcsec,
        exact_image_lm_max_iter=exact_image_lm_max_iter,
        exact_image_lm_trust_radius_arcsec=exact_image_lm_trust_radius_arcsec,
        exact_image_adaptive_max_levels=exact_image_adaptive_max_levels,
        sampling_engine=str(getattr(args, "sampling_engine", getattr(evaluator, "sampling_engine", "full_flat"))),
        active_scaling_galaxies=getattr(args, "active_scaling_galaxies", None),
        active_scaling_selection=str(
            getattr(args, "active_scaling_selection", getattr(evaluator, "active_scaling_selection", "adaptive"))
        ),
        active_scaling_cumulative_fraction=float(
            getattr(
                args,
                "active_scaling_cumulative_fraction",
                getattr(evaluator, "active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION),
            )
        ),
        active_scaling_min=int(getattr(args, "active_scaling_min", getattr(evaluator, "active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN))),
        refresh_every=_stage_scalar(
            getattr(args, "refresh_every", None),
            getattr(evaluator, "refresh_every", DEFAULT_REFRESH_EVERY),
        ),
        refresh_param_drift_frac=float(
            getattr(args, "refresh_param_drift_frac", getattr(evaluator, "refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC))
        ),
        source_plane_covariance_floor=float(
            getattr(args, "source_plane_covariance_floor", getattr(evaluator, "source_plane_covariance_floor", 1.0e-6))
        ),
        source_plane_outlier_sigma_arcsec=float(
            getattr(
                args,
                "source_plane_outlier_sigma_arcsec",
                getattr(evaluator, "source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC),
            )
        ),
        sample_likelihood_mode=str(
            getattr(args, "sample_likelihood_mode", getattr(evaluator, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE))
        ),
        image_plane_newton_steps=int(getattr(args, "image_plane_newton_steps", getattr(evaluator, "image_plane_newton_steps", 0))),
        anchored_image_plane_solve_steps=int(
            getattr(
                args,
                "anchored_image_plane_solve_steps",
                getattr(evaluator, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS),
            )
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            getattr(
                args,
                "anchored_image_plane_trust_radius_arcsec",
                getattr(evaluator, "anchored_image_plane_trust_radius_arcsec", DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC),
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_relative",
                getattr(evaluator, "anchored_image_plane_lm_damping_relative", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE),
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            getattr(
                args,
                "anchored_image_plane_lm_damping_absolute",
                getattr(evaluator, "anchored_image_plane_lm_damping_absolute", DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE),
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            getattr(
                args,
                "critical_arc_critical_direction_sigma_arcsec",
                getattr(evaluator, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC),
            )
        ),
        critical_arc_base_prob=float(
            getattr(args, "critical_arc_base_prob", getattr(evaluator, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB))
        ),
        critical_arc_max_prob=float(
            getattr(args, "critical_arc_max_prob", getattr(evaluator, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB))
        ),
        critical_arc_singular_threshold=float(
            getattr(
                args,
                "critical_arc_singular_threshold",
                getattr(evaluator, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD),
            )
        ),
        critical_arc_singular_softness=float(
            getattr(
                args,
                "critical_arc_singular_softness",
                getattr(evaluator, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS),
            )
        ),
        critical_arc_lm_damping_relative=float(
            getattr(
                args,
                "critical_arc_lm_damping_relative",
                getattr(evaluator, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE),
            )
        ),
        critical_arc_lm_damping_absolute=float(
            getattr(
                args,
                "critical_arc_lm_damping_absolute",
                getattr(evaluator, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE),
            )
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            getattr(
                args,
                "critical_arc_lm_trust_radius_arcsec",
                getattr(evaluator, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC),
            )
        ),
        arc_recovery_p_arc_threshold=float(
            getattr(
                args,
                "arc_recovery_p_arc_threshold",
                getattr(evaluator, "arc_recovery_p_arc_threshold", DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD),
            )
        ),
        arc_aware_max_arclength_arcsec=float(
            getattr(
                args,
                "arc_aware_max_arclength_arcsec",
                getattr(evaluator, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC),
            )
        ),
        arc_aware_curve_step_arcsec=float(
            getattr(
                args,
                "arc_aware_curve_step_arcsec",
                getattr(evaluator, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC),
            )
        ),
        use_magnitude_likelihood=bool(
            getattr(
                args,
                "use_magnitude_likelihood",
                getattr(evaluator, "use_magnitude_likelihood", DEFAULT_USE_MAGNITUDE_LIKELIHOOD),
            )
        ),
        magnitude_sigma_floor=float(
            getattr(
                args,
                "magnitude_sigma_floor",
                getattr(args, "magnitude_sigma", getattr(evaluator, "magnitude_sigma_floor", DEFAULT_MAGNITUDE_SIGMA_FLOOR)),
            )
        ),
        magnitude_mu_floor=float(
            getattr(args, "magnitude_mu_floor", getattr(evaluator, "magnitude_mu_floor", DEFAULT_MAGNITUDE_MU_FLOOR))
        ),
        magnitude_min_reliability=float(
            getattr(
                args,
                "magnitude_min_reliability",
                getattr(evaluator, "magnitude_min_reliability", DEFAULT_MAGNITUDE_MIN_RELIABILITY),
            )
        ),
        image_plane_scatter_floor_arcsec=float(
            getattr(
                args,
                "image_plane_scatter_floor_arcsec",
                getattr(evaluator, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC),
            )
        ),
        fixed_image_sigma_int_arcsec=getattr(
            args,
            "fix_image_sigma_int_arcsec",
            getattr(evaluator, "fixed_image_sigma_int_arcsec", None),
        ),
        evidence_source_prior_sigma_arcsec=getattr(
            args,
            "evidence_source_prior_sigma_arcsec",
            getattr(evaluator, "evidence_source_prior_sigma_arcsec", None),
        ),
        evidence_source_prior_mean_x_arcsec=float(
            getattr(
                args,
                "evidence_source_prior_mean_x_arcsec",
                getattr(evaluator, "evidence_source_prior_mean_x_arcsec", 0.0),
            )
        ),
        evidence_source_prior_mean_y_arcsec=float(
            getattr(
                args,
                "evidence_source_prior_mean_y_arcsec",
                getattr(evaluator, "evidence_source_prior_mean_y_arcsec", 0.0),
            )
        ),
        quick_diagnostics=bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False))),
    )


def _fit_quality_detail_array(details: dict[str, Any], key: str, size: int, dtype: Any = float) -> np.ndarray | None:
    return _shared_diagnostic_detail_array(details, key, size, dtype)


def _fit_quality_extra_image_rows(
    family: Any,
    details: dict[str, Any],
    model_count_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    return _shared_extra_image_rows(family, details, model_count_fields)


def _fit_quality_prediction_for_family_latent(
    evaluator: Any,
    family: Any,
    params_latent: np.ndarray,
    image_sigma_int: float,
    covariance_floor: float,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    prediction = _shared_image_prediction_for_family_latent(
        evaluator,
        family,
        params_latent,
        image_sigma_int,
        covariance_floor,
        quick_diagnostics=quick_diagnostics,
        magnification_column="magnification_model",
    )
    return {
        "image_rows": prediction["image_rows"],
        "magnification_rows": prediction["magnification_rows"],
        "image_count_rows": prediction["image_count_rows"],
        "extra_image_rows": prediction["extra_image_rows"],
    }


def _fit_quality_prediction_for_latent(
    evaluator: Any,
    state: BuildState,
    params_latent: np.ndarray,
    quick_diagnostics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    params_latent = np.asarray(params_latent, dtype=float)
    image_sigma_int = _fit_quality_image_sigma_int(evaluator, params_latent)
    covariance_floor = _finite_or(getattr(evaluator, "source_plane_covariance_floor", 0.0), 0.0)
    prediction: dict[str, list[dict[str, Any]]] = {
        "image_rows": [],
        "magnification_rows": [],
        "image_count_rows": [],
        "extra_image_rows": [],
    }
    for family in state.family_data:
        family_prediction = _fit_quality_prediction_for_family_latent(
            evaluator,
            family,
            params_latent,
            image_sigma_int,
            covariance_floor,
            quick_diagnostics=quick_diagnostics,
        )
        prediction["image_rows"].extend(family_prediction["image_rows"])
        prediction["magnification_rows"].extend(family_prediction["magnification_rows"])
        prediction["image_count_rows"].extend(family_prediction.get("image_count_rows", []))
        prediction["extra_image_rows"].extend(family_prediction.get("extra_image_rows", []))
    return prediction


def _fit_quality_family_cost_metadata(
    family: Any,
    *,
    min_distance_arcsec: float = 0.2,
) -> dict[str, Any]:
    min_distance = _finite_or(min_distance_arcsec, 0.2)
    if min_distance <= 0.0:
        min_distance = 0.2
    search_window = _finite_or(getattr(family, "search_window", np.nan))
    if not np.isfinite(search_window):
        x_obs = np.asarray(getattr(family, "x_obs", []), dtype=float)
        y_obs = np.asarray(getattr(family, "y_obs", []), dtype=float)
        span_x = float(np.ptp(x_obs)) if x_obs.size > 1 else 0.0
        span_y = float(np.ptp(y_obs)) if y_obs.size > 1 else 0.0
        search_window = max(span_x, span_y) + 10.0
    num_pix = int(round(float(search_window) / float(min_distance)) + 0.5) if min_distance > 0.0 else 0
    grid_points = max(0, num_pix * num_pix)
    return {
        "min_distance": float(min_distance),
        "search_window": float(search_window),
        "num_pix": int(num_pix),
        "grid_points": int(grid_points),
    }


def _posterior_fit_quality_predictions(
    evaluator: Any,
    state: BuildState,
    sample_latents: list[np.ndarray],
    args: argparse.Namespace,
    progress: Any | None = None,
) -> list[dict[str, list[dict[str, Any]]]]:
    if not sample_latents:
        return []
    worker_count = max(1, int(jax_cpu_worker_count()))
    quick_diagnostics = bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False)))
    n_families = len(state.family_data)
    n_tasks = len(sample_latents) * n_families
    sample_latents = [np.asarray(sample, dtype=float) for sample in sample_latents]
    image_sigma_int_by_sample = [_fit_quality_image_sigma_int(evaluator, sample) for sample in sample_latents]
    covariance_floor = _finite_or(getattr(evaluator, "source_plane_covariance_floor", 0.0), 0.0)
    worker_evaluators: list[Any] = []
    worker_lock = threading.Lock()
    worker_local = threading.local()

    def worker_evaluator() -> Any:
        cached = getattr(worker_local, "evaluator", None)
        if cached is None:
            cached = _clone_fit_quality_evaluator(evaluator, args)
            worker_local.evaluator = cached
            with worker_lock:
                worker_evaluators.append(cached)
        return cached

    def evaluate_family(sample_index: int, family_index: int) -> tuple[int, int, dict[str, list[dict[str, Any]]]]:
        return (
            sample_index,
            family_index,
            _fit_quality_prediction_for_family_latent(
                worker_evaluator(),
                state.family_data[family_index],
                sample_latents[sample_index],
                image_sigma_int_by_sample[sample_index],
                covariance_floor,
                quick_diagnostics=quick_diagnostics,
            ),
        )

    family_predictions: list[list[dict[str, list[dict[str, Any]]] | None]] = [
        [None] * n_families for _ in sample_latents
    ]
    exact_image_min_distance_arcsec = _finite_or(
        getattr(evaluator, "exact_image_min_distance_arcsec", 0.2),
        0.2,
    )
    family_costs = [
        _fit_quality_family_cost_metadata(
            family,
            min_distance_arcsec=exact_image_min_distance_arcsec,
        )
        for family in state.family_data
    ]
    task_indices = [
        (sample_index, family_index)
        for sample_index in range(len(sample_latents))
        for family_index in range(n_families)
    ]
    task_indices.sort(key=lambda index: family_costs[index[1]]["grid_points"], reverse=True)

    completed_family_tasks = 0
    completed_draws = 0
    completed_families_by_draw = [0] * len(sample_latents)

    def family_progress_description(status: str, sample_index: int, family_index: int) -> str:
        family = state.family_data[family_index]
        cost = family_costs[family_index]
        return (
            f"fit quality exact: {completed_family_tasks}/{n_tasks} family diagnostics | "
            f"{status} draw {sample_index + 1}/{len(sample_latents)} "
            f"family={family.family_id} z={float(family.z_source):.4f} "
            f"window={float(cost['search_window']):.1f} grid={int(cost['num_pix'])}x{int(cost['num_pix'])}"
        )

    def draw_progress_description() -> str:
        return f"draw progress: {completed_draws}/{len(sample_latents)} complete"

    def update_task_progress(
        progress: Any | None,
        family_task_id: int | None,
        status: str,
        sample_index: int,
        family_index: int,
        *,
        advance: bool = False,
    ) -> None:
        if progress is None or family_task_id is None:
            return
        progress.update(family_task_id, description=family_progress_description(status, sample_index, family_index))
        if advance:
            progress.advance(family_task_id)

    def mark_family_completed(
        progress: Any | None,
        family_task_id: int | None,
        draw_task_id: int | None,
        sample_index: int,
        family_index: int,
    ) -> None:
        nonlocal completed_family_tasks, completed_draws
        completed_family_tasks += 1
        completed_families_by_draw[sample_index] += 1
        update_task_progress(progress, family_task_id, "completed", sample_index, family_index, advance=True)
        if completed_families_by_draw[sample_index] == n_families:
            completed_draws += 1
            if progress is not None and draw_task_id is not None:
                progress.update(draw_task_id, description=draw_progress_description())
                progress.advance(draw_task_id)
            if not quick_diagnostics:
                _log(
                    args,
                    (
                        f"[plot:fit_quality] draw {sample_index + 1}/{len(sample_latents)} complete "
                        f"families={completed_families_by_draw[sample_index]}/{n_families} "
                        f"completed_tasks={completed_family_tasks}/{n_tasks}"
                    ),
                )

    def combine_predictions() -> list[dict[str, list[dict[str, Any]]]]:
        predictions: list[dict[str, list[dict[str, Any]]]] = []
        for sample_predictions in family_predictions:
            combined = {
                "image_rows": [],
                "magnification_rows": [],
                "image_count_rows": [],
                "extra_image_rows": [],
            }
            for family_prediction in sample_predictions:
                if family_prediction is None:
                    continue
                combined["image_rows"].extend(family_prediction["image_rows"])
                combined["magnification_rows"].extend(family_prediction["magnification_rows"])
                combined["image_count_rows"].extend(family_prediction.get("image_count_rows", []))
                combined["extra_image_rows"].extend(family_prediction.get("extra_image_rows", []))
            predictions.append(combined)
        return predictions

    progress_enabled = n_tasks > 0 and not quick_diagnostics and not bool(getattr(args, "quiet", False))
    max_workers = min(worker_count, n_tasks)
    largest_num_pix = max((int(cost["num_pix"]) for cost in family_costs), default=0)
    total_grid_points = sum(int(cost["grid_points"]) for cost in family_costs) * len(sample_latents)
    _log(
        args,
        (
            f"[plot:fit_quality] family diagnostics tasks={n_tasks} workers={max_workers} "
            f"families={n_families} draws={len(sample_latents)} "
            f"largest_grid={largest_num_pix}x{largest_num_pix} total_grid_points={total_grid_points}"
        ),
    )
    own_progress = progress is None
    progress_cm = (
        _progress_context(
            args,
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )
        if progress_enabled and own_progress
        else None
    )
    family_task_id: int | None = None
    draw_task_id: int | None = None
    try:
        if progress_cm is not None:
            progress = progress_cm.__enter__()
        if progress is not None and progress_enabled:
            family_task_id = progress.add_task(f"fit quality exact: 0/{n_tasks} family diagnostics", total=n_tasks)
            draw_task_id = progress.add_task(draw_progress_description(), total=len(sample_latents))
        if worker_count <= 1 or n_tasks <= 1:
            for sample_index, family_index in task_indices:
                sample = sample_latents[sample_index]
                family = state.family_data[family_index]
                update_task_progress(progress, family_task_id, "running", sample_index, family_index)
                _sample_index, _family_index, prediction = (
                    sample_index,
                    family_index,
                    _fit_quality_prediction_for_family_latent(
                        evaluator,
                        family,
                        sample,
                        image_sigma_int_by_sample[sample_index],
                        covariance_floor,
                        quick_diagnostics=quick_diagnostics,
                    ),
                )
                family_predictions[_sample_index][_family_index] = prediction
                mark_family_completed(progress, family_task_id, draw_task_id, _sample_index, _family_index)
            return combine_predictions()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {}
                for sample_index, family_index in task_indices:
                    update_task_progress(progress, family_task_id, "queued", sample_index, family_index)
                    future_to_index[executor.submit(evaluate_family, sample_index, family_index)] = (
                        sample_index,
                        family_index,
                    )
                for future in as_completed(future_to_index):
                    sample_index, family_index, prediction = future.result()
                    family_predictions[sample_index][family_index] = prediction
                    mark_family_completed(progress, family_task_id, draw_task_id, sample_index, family_index)
        except Exception as exc:
            _log(None, f"[plot:fit_quality] parallel posterior diagnostics failed; retrying serially error={exc}")
            for cached_evaluator in worker_evaluators:
                if hasattr(cached_evaluator, "release_runtime_caches"):
                    cached_evaluator.release_runtime_caches()
            worker_evaluators.clear()
            for sample_index, family_index in task_indices:
                sample = sample_latents[sample_index]
                family = state.family_data[family_index]
                update_task_progress(progress, family_task_id, "running", sample_index, family_index)
                _sample_index, _family_index, prediction = (
                    sample_index,
                    family_index,
                    _fit_quality_prediction_for_family_latent(
                        evaluator,
                        family,
                        sample,
                        image_sigma_int_by_sample[sample_index],
                        covariance_floor,
                        quick_diagnostics=quick_diagnostics,
                    ),
                )
                family_predictions[_sample_index][_family_index] = prediction
                mark_family_completed(progress, family_task_id, draw_task_id, _sample_index, _family_index)
        return combine_predictions()
    finally:
        for cached_evaluator in worker_evaluators:
            if hasattr(cached_evaluator, "release_runtime_caches"):
                cached_evaluator.release_runtime_caches()
        if progress_cm is not None:
            progress_cm.__exit__(None, None, None)


def _fit_quality_tables(
    state: BuildState,
    evaluator: Any,
    best_fit: np.ndarray,
    results: PosteriorResults,
    args: argparse.Namespace,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    quick_diagnostics = bool(getattr(args, "quick_diagnostics", getattr(evaluator, "quick_diagnostics", False)))
    summary_fn = _fit_quality_median_std if quick_diagnostics else _fit_quality_quantiles

    max_draws = max(0, int(getattr(args, "posterior_image_diagnostic_draws", 0)))
    posterior_samples = _capped_fit_quality_samples(results.samples, max_draws)
    sample_latents = [
        _reported_physical_to_latent_vector(evaluator, np.asarray(sample, dtype=float))
        for sample in posterior_samples
    ]
    all_predictions = _posterior_fit_quality_predictions(
        evaluator,
        state,
        [best_fit_latent, *sample_latents],
        args,
        progress=progress,
    )
    best_prediction = (
        all_predictions[0]
        if all_predictions
        else {"image_rows": [], "magnification_rows": [], "image_count_rows": [], "extra_image_rows": []}
    )
    posterior_predictions = all_predictions[1:]

    image_draws_by_label: dict[str, list[dict[str, Any]]] = {}
    magnification_draws_by_label: dict[str, list[dict[str, Any]]] = {}
    for prediction in posterior_predictions:
        for row in prediction["image_rows"]:
            image_draws_by_label.setdefault(str(row["image_label"]), []).append(row)
        for row in prediction["magnification_rows"]:
            magnification_draws_by_label.setdefault(str(row["image_label"]), []).append(row)

    image_rows: list[dict[str, Any]] = []
    for row in best_prediction["image_rows"]:
        label = str(row["image_label"])
        draws = image_draws_by_label.get(label, [])
        x16, x50, x84 = summary_fn([draw["x_model_arcsec"] for draw in draws])
        y16, y50, y84 = summary_fn([draw["y_model_arcsec"] for draw in draws])
        r16, r50, r84 = summary_fn([draw["image_residual_arcsec"] for draw in draws])
        arc_r16, arc_r50, arc_r84 = summary_fn(
            [draw.get("arc_aware_image_residual_arcsec", np.nan) for draw in draws]
        )
        sigma_eff = _finite_or(row.get("image_sigma_eff_arcsec", np.nan))
        residual_norm = (
            float(row["image_residual_arcsec"]) / sigma_eff
            if np.isfinite(float(row["image_residual_arcsec"])) and np.isfinite(sigma_eff) and sigma_eff > 0.0
            else np.nan
        )
        residual_norm_q50 = r50 / sigma_eff if np.isfinite(r50) and np.isfinite(sigma_eff) and sigma_eff > 0.0 else np.nan
        covered_x = _covered_by_inflated_interval(
            float(row["x_obs_arcsec"]),
            x16,
            x84,
            float(row["x_model_arcsec"]),
            sigma_eff,
        )
        covered_y = _covered_by_inflated_interval(
            float(row["y_obs_arcsec"]),
            y16,
            y84,
            float(row["y_model_arcsec"]),
            sigma_eff,
        )
        valid_draws = sum(
            1
            for draw in draws
            if np.isfinite(float(draw["x_model_arcsec"]) + float(draw["y_model_arcsec"]))
        )
        image_rows.append(
            {
                **row,
                "x_model_q16": x16,
                "x_model_q50": x50,
                "x_model_q84": x84,
                "y_model_q16": y16,
                "y_model_q50": y50,
                "y_model_q84": y84,
                "image_residual_q16": r16,
                "image_residual_q50": r50,
                "image_residual_q84": r84,
                "arc_aware_image_residual_q16": arc_r16,
                "arc_aware_image_residual_q50": arc_r50,
                "arc_aware_image_residual_q84": arc_r84,
                "residual_norm": residual_norm,
                "residual_norm_q50": residual_norm_q50,
                "covered_x_1sigma": bool(covered_x),
                "covered_y_1sigma": bool(covered_y),
                "covered_xy_1sigma": bool(covered_x and covered_y),
                "posterior_valid_draws": int(valid_draws),
                "posterior_failed_draws": int(len(draws) - valid_draws),
            }
        )
    magnification_rows: list[dict[str, Any]] = []
    for row in best_prediction["magnification_rows"]:
        label = str(row["image_label"])
        draws = magnification_draws_by_label.get(label, [])
        q16, q50, q84 = summary_fn([draw["magnification_model"] for draw in draws])
        valid_draws = sum(1 for draw in draws if np.isfinite(float(draw["magnification_model"])))
        magnification_rows.append(
            {
                **row,
                "magnification_model_q16": q16,
                "magnification_model_q50": q50,
                "magnification_model_q84": q84,
                "posterior_valid_draws": int(valid_draws),
                "posterior_failed_draws": int(len(draws) - valid_draws),
            }
        )
    return pd.DataFrame(image_rows), pd.DataFrame(magnification_rows), pd.DataFrame(best_prediction.get("extra_image_rows", []))


def _cab_arc_diagnostics_table(evaluator: Any, best_fit: np.ndarray) -> pd.DataFrame:
    if not hasattr(evaluator, "_cab_morphology_details_for_arcs"):
        return pd.DataFrame()
    try:
        best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
        table = evaluator._cab_morphology_details_for_arcs(best_fit_latent)
    except Exception:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table
    return pd.DataFrame(table)


def _plot_image_recovery_fit_quality(
    image_df: pd.DataFrame,
    path: Path,
    extra_image_df: pd.DataFrame | None = None,
    *,
    use_arc_aware_diagnostics: bool = False,
) -> None:
    if image_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    x_obs = image_df["x_obs_arcsec"].to_numpy(dtype=float)
    y_obs = image_df["y_obs_arcsec"].to_numpy(dtype=float)
    x_best = _fit_quality_value(image_df, "x_model_arcsec")
    y_best = _fit_quality_value(image_df, "y_model_arcsec")
    x_model = _fit_quality_value(image_df, "x_model_q50", "x_model_arcsec")
    y_model = _fit_quality_value(image_df, "y_model_q50", "y_model_arcsec")
    x_model = np.where(np.isfinite(x_model), x_model, x_best)
    y_model = np.where(np.isfinite(y_model), y_model, y_best)

    if use_arc_aware_diagnostics:
        status = _image_catalog_effective_recovery_statuses(image_df)
    else:
        status = np.asarray(
            [
                "POINT_RECOVERED" if _image_catalog_point_recovered(row) else "MISSED"
                for _, row in image_df.iterrows()
            ],
            dtype=object,
        )
    point_recovered = status == "POINT_RECOVERED"
    arc_recovered = status == "ARC_RECOVERED"
    missed = status == "MISSED"
    finite_model = np.isfinite(x_model) & np.isfinite(y_model)
    finite_recovered_model = point_recovered & finite_model

    for status_name, mask, label in (
        ("POINT_RECOVERED", point_recovered, "point recovered"),
        ("ARC_RECOVERED", arc_recovered, "arc recovered"),
        ("MISSED", missed, "not recovered"),
    ):
        if mask.any():
            ax.scatter(
                x_obs[mask],
                y_obs[mask],
                marker="x",
                color=_image_catalog_status_color(status_name),
                s=18,
                linewidths=0.9,
                label=label,
            )
    if finite_recovered_model.any():
        x16 = _fit_quality_value(image_df, "x_model_q16")
        x84 = _fit_quality_value(image_df, "x_model_q84")
        y16 = _fit_quality_value(image_df, "y_model_q16")
        y84 = _fit_quality_value(image_df, "y_model_q84")
        xerr = [
            np.where(np.isfinite(x16), np.maximum(0.0, x_model - x16), 0.0)[finite_recovered_model],
            np.where(np.isfinite(x84), np.maximum(0.0, x84 - x_model), 0.0)[finite_recovered_model],
        ]
        yerr = [
            np.where(np.isfinite(y16), np.maximum(0.0, y_model - y16), 0.0)[finite_recovered_model],
            np.where(np.isfinite(y84), np.maximum(0.0, y84 - y_model), 0.0)[finite_recovered_model],
        ]
        ax.errorbar(
            x_model[finite_recovered_model],
            y_model[finite_recovered_model],
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=_color_with_alpha(_image_catalog_status_color("POINT_RECOVERED"), 0.75),
            ecolor=_color_with_alpha(_image_catalog_status_color("POINT_RECOVERED"), 0.35),
            markersize=3,
            elinewidth=0.8,
            capsize=1.5,
            label=None,
        )
    if extra_image_df is not None and not extra_image_df.empty:
        x_extra = _fit_quality_value(extra_image_df, "x_model_arcsec")
        y_extra = _fit_quality_value(extra_image_df, "y_model_arcsec")
        finite_extra = np.isfinite(x_extra) & np.isfinite(y_extra)
        if finite_extra.any():
            ax.scatter(
                x_extra[finite_extra],
                y_extra[finite_extra],
                marker="o",
                color="tab:blue",
                s=16,
                linewidths=0.0,
                label="extra",
            )
    for row, x_fit, y_fit, is_recovered in zip(image_df.itertuples(index=False), x_model, y_model, point_recovered):
        if is_recovered and np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Observed vs Model Image Positions")
    ax.legend(loc="best", fontsize=8)

    residual_best = image_df["image_residual_arcsec"].to_numpy(dtype=float)
    residual = image_df["image_residual_q50"].to_numpy(dtype=float)
    residual = np.where(np.isfinite(residual), residual, residual_best)
    x_index = np.arange(len(image_df))
    r16 = image_df["image_residual_q16"].to_numpy(dtype=float)
    r84 = image_df["image_residual_q84"].to_numpy(dtype=float)
    finite_residual = np.isfinite(residual)
    if finite_residual.any():
        yerr = [
            np.where(np.isfinite(r16), np.maximum(0.0, residual - r16), 0.0)[finite_residual],
            np.where(np.isfinite(r84), np.maximum(0.0, r84 - residual), 0.0)[finite_residual],
        ]
        axes[1].errorbar(
            x_index[finite_residual],
            residual[finite_residual],
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=3,
        )
    if use_arc_aware_diagnostics:
        arc_residual_best = _fit_quality_value(image_df, "arc_aware_image_residual_arcsec")
        arc_residual = _fit_quality_value(image_df, "arc_aware_image_residual_q50", "arc_aware_image_residual_arcsec")
        arc_residual = np.where(np.isfinite(arc_residual), arc_residual, arc_residual_best)
        finite_arc_residual = np.isfinite(arc_residual)
    else:
        arc_residual = np.full(len(image_df), np.nan, dtype=float)
        finite_arc_residual = np.zeros(len(image_df), dtype=bool)
    if finite_arc_residual.any():
        axes[1].scatter(
            x_index[finite_arc_residual],
            arc_residual[finite_arc_residual],
            color="tab:olive",
            marker="x",
            s=24,
            label="arc-aware",
        )
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual (arcsec)")
    axes[1].set_title("Image Residuals")
    if finite_arc_residual.any():
        axes[1].legend(loc="best", fontsize=8)
    if len(image_df) <= 40:
        axes[1].set_xticks(x_index)
        axes[1].set_xticklabels(image_df["image_label"].astype(str), rotation=90, fontsize=7)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_image_count_recovery(
    image_count_df: pd.DataFrame,
    path: Path,
    *,
    use_arc_aware_diagnostics: bool = False,
) -> None:
    if image_count_df.empty:
        return
    required = {"family_id", "observed_image_count", "recovered_image_count", "produced_image_count"}
    if not required.issubset(image_count_df.columns):
        return
    observed = pd.to_numeric(image_count_df["observed_image_count"], errors="coerce").to_numpy(dtype=float)
    recovered = pd.to_numeric(image_count_df["recovered_image_count"], errors="coerce").to_numpy(dtype=float)
    arc_aware_recovered = (
        pd.to_numeric(image_count_df["arc_aware_recovered_image_count"], errors="coerce").to_numpy(dtype=float)
        if use_arc_aware_diagnostics and "arc_aware_recovered_image_count" in image_count_df.columns
        else np.full(len(image_count_df), np.nan, dtype=float)
    )
    produced = pd.to_numeric(image_count_df["produced_image_count"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(recovered).any() or np.isfinite(arc_aware_recovered).any() or np.isfinite(produced).any()):
        return
    labels = image_count_df["family_id"].astype(str).to_numpy()
    y_index = np.arange(len(image_count_df))
    height = 0.18 if np.isfinite(arc_aware_recovered).any() else 0.24
    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.32 * len(image_count_df))))
    finite_observed = np.isfinite(observed)
    finite_recovered = np.isfinite(recovered)
    finite_arc_aware_recovered = np.isfinite(arc_aware_recovered)
    finite_produced = np.isfinite(produced)
    offsets = {
        "observed": -1.5 * height if finite_arc_aware_recovered.any() else -height,
        "recovered": -0.5 * height if finite_arc_aware_recovered.any() else 0.0,
        "arc_aware": 0.5 * height,
        "produced": 1.5 * height if finite_arc_aware_recovered.any() else height,
    }
    if finite_observed.any():
        ax.barh(
            y_index[finite_observed] + offsets["observed"],
            observed[finite_observed],
            height=height,
            color="0.65",
            label="observed",
        )
    if finite_recovered.any():
        ax.barh(
            y_index[finite_recovered] + offsets["recovered"],
            recovered[finite_recovered],
            height=height,
            color=_image_catalog_status_color("POINT_RECOVERED"),
            label="point recovered",
        )
    if finite_arc_aware_recovered.any():
        ax.barh(
            y_index[finite_arc_aware_recovered] + offsets["arc_aware"],
            arc_aware_recovered[finite_arc_aware_recovered],
            height=height,
            color=_image_catalog_status_color("ARC_RECOVERED"),
            label="point + arc recovered",
        )
    if finite_produced.any():
        ax.barh(
            y_index[finite_produced] + offsets["produced"],
            produced[finite_produced],
            height=height,
            color=_image_catalog_status_color("MODEL"),
            label="produced",
        )
    total_observed = int(np.nansum(observed))
    total_recovered = int(np.nansum(recovered)) if finite_recovered.any() else 0
    total_arc_aware_recovered = int(np.nansum(arc_aware_recovered)) if finite_arc_aware_recovered.any() else None
    total_produced = int(np.nansum(produced)) if finite_produced.any() else 0
    ax.set_yticks(y_index)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("image count")
    ax.set_ylabel("family")
    if total_arc_aware_recovered is None:
        title_counts = f"observed={total_observed} point={total_recovered} produced={total_produced}"
    else:
        title_counts = (
            f"observed={total_observed} point={total_recovered} "
            f"point+arc={total_arc_aware_recovered} produced={total_produced}"
        )
    ax.set_title(f"Image Count Recovery: {title_counts}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_model_magnification_fit_quality(magnification_df: pd.DataFrame, path: Path) -> None:
    if magnification_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(magnification_df)), 4.8))
    x_index = np.arange(len(magnification_df))
    best = magnification_df["magnification_model"].to_numpy(dtype=float)
    median = magnification_df["magnification_model_q50"].to_numpy(dtype=float)
    median = np.where(np.isfinite(median), median, best)
    q16 = magnification_df["magnification_model_q16"].to_numpy(dtype=float)
    q84 = magnification_df["magnification_model_q84"].to_numpy(dtype=float)
    finite = np.isfinite(median)
    if finite.any():
        yerr = [
            np.where(np.isfinite(q16), np.maximum(0.0, median - q16), 0.0)[finite],
            np.where(np.isfinite(q84), np.maximum(0.0, q84 - median), 0.0)[finite],
        ]
        ax.errorbar(
            x_index[finite],
            median[finite],
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=4,
            label="posterior interval",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("image")
    ax.set_ylabel("model signed magnification")
    ax.set_title("Model-Only Signed Magnification at Observed Images")
    if len(magnification_df) <= 80:
        ax.set_xticks(x_index)
        ax.set_xticklabels(magnification_df["image_label"].astype(str), rotation=90, fontsize=7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_normalized_image_residuals(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty:
        return
    residual_norm = _fit_quality_value(image_df, "residual_norm_q50", "residual_norm")
    finite = np.isfinite(residual_norm)
    if not finite.any():
        return
    labels = image_df["image_label"].astype(str).to_numpy()
    family_ids = image_df["family_id"].astype(str).to_numpy() if "family_id" in image_df.columns else np.full(len(image_df), "")
    unique_families = list(dict.fromkeys(family_ids.tolist()))
    color_map = {family_id: plt.get_cmap("tab20", max(1, len(unique_families)))(idx) for idx, family_id in enumerate(unique_families)}
    colors = [color_map[family_id] for family_id in family_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = min(60, max(12, int(np.sqrt(int(np.sum(finite))) * 2)))
    axes[0].hist(residual_norm[finite], bins=bins, color="tab:blue", alpha=0.75)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("image residual / sigma_eff")
    axes[0].set_ylabel("count")
    axes[0].set_title("Normalized Residual Distribution")

    x_index = np.arange(len(image_df))
    axes[1].scatter(x_index[finite], residual_norm[finite], c=np.asarray(colors, dtype=object)[finite].tolist(), s=28)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("image")
    axes[1].set_ylabel("image residual / sigma_eff")
    axes[1].set_title("Normalized Residual by Image")
    if len(image_df) <= 80:
        axes[1].set_xticks(x_index)
        axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _write_placeholder_plot(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_image_residual_histogram(
    image_df: pd.DataFrame,
    path: Path,
    *,
    use_arc_aware_diagnostics: bool = False,
) -> None:
    point_residual_all = _fit_quality_value(image_df, "point_image_residual_arcsec", "image_residual_arcsec")
    if any(column in image_df.columns for column in ("image_recovery_status", "arc_recovery_status", "exact_image_prediction_failed")):
        status = _image_catalog_effective_recovery_statuses(image_df)
        point_mask = np.asarray([_image_catalog_point_recovered(row) for _, row in image_df.iterrows()], dtype=bool)
        arc_mask = np.asarray([_image_catalog_arc_recovered(row) for _, row in image_df.iterrows()], dtype=bool)
    else:
        status = np.where(np.isfinite(point_residual_all), "POINT_RECOVERED", "MISSED")
        point_mask = status == "POINT_RECOVERED"
        arc_mask = np.zeros(len(image_df), dtype=bool)
    if not use_arc_aware_diagnostics:
        status = np.where(point_mask, "POINT_RECOVERED", "MISSED")
        arc_mask = np.zeros(len(image_df), dtype=bool)

    point_residual = point_residual_all[point_mask & np.isfinite(point_residual_all)]
    if use_arc_aware_diagnostics:
        arc_candidate_residual_all = _fit_quality_value(
            image_df,
            "arc_candidate_image_residual_arcsec",
            "arc_aware_image_residual_arcsec",
            "arc_curve_distance_arcsec",
        )
        arc_aware_mask = arc_mask | (point_mask & ~arc_mask)
        arc_residual_all = np.where(arc_mask, arc_candidate_residual_all, np.where(point_mask, point_residual_all, np.nan))
        arc_aware_residual = arc_residual_all[arc_aware_mask & np.isfinite(arc_residual_all)]
        finite_for_bins = np.concatenate([point_residual, arc_aware_residual])
    else:
        arc_aware_residual = np.asarray([], dtype=float)
        finite_for_bins = point_residual
    if finite_for_bins.size == 0:
        _write_placeholder_plot(
            path,
            "Image residual histogram",
            "No finite image residuals are available.",
        )
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    bin_count = min(48, max(14, int(np.sqrt(max(1, finite_for_bins.size)) * 2.4)))
    max_residual = float(np.nanmax(finite_for_bins))
    bins = np.linspace(0.0, max(max_residual * 1.04, 1.0e-6), bin_count + 1)
    point_color = "#4da3ff"
    arc_color = _image_catalog_status_color("ARC_RECOVERED")

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values.size else np.nan

    def median(values: np.ndarray) -> float:
        return float(np.median(values)) if values.size else np.nan

    point_rms = rms(point_residual)
    point_median = median(point_residual)
    arc_aware_rms = rms(arc_aware_residual)
    arc_aware_median = median(arc_aware_residual)
    total_count = int(len(image_df))

    if point_residual.size:
        ax.hist(
            point_residual,
            bins=bins,
            color=point_color,
            alpha=0.5,
            edgecolor="#1d4ed8",
            linewidth=0.85,
            label=(
                f"point recovery  {point_residual.size}/{total_count}  "
                f"RMS={point_rms:.3g}\"  median={point_median:.3g}\""
            ),
            zorder=2,
        )
        ax.axvline(point_rms, color=point_color, linestyle="-", linewidth=1.35, alpha=0.9)
    if use_arc_aware_diagnostics and arc_aware_residual.size:
        ax.hist(
            arc_aware_residual,
            bins=bins,
            histtype="step",
            color=arc_color,
            linewidth=2.0,
            label=(
                f"arc-aware  {arc_aware_residual.size}/{total_count}  "
                f"RMS={arc_aware_rms:.3g}\"  median={arc_aware_median:.3g}\""
            ),
            zorder=4,
        )
        ax.axvline(arc_aware_rms, color="#b7791f", linestyle="-", linewidth=1.45, alpha=0.95)
    missed_count = int(np.sum(status == "MISSED"))
    arc_supported_count = int(np.sum(arc_mask))
    annotation_lines = [
        "Image residuals",
        f"point: RMS {point_rms:.3g}\"  median {point_median:.3g}\"  ({point_residual.size}/{total_count})"
        if point_residual.size
        else f"point: na  (0/{total_count})",
    ]
    if use_arc_aware_diagnostics:
        annotation_lines.extend(
            [
                (
                    f"arc-aware: RMS {arc_aware_rms:.3g}\"  median {arc_aware_median:.3g}\"  "
                    f"({arc_aware_residual.size}/{total_count})"
                )
                if arc_aware_residual.size
                else f"arc-aware: na  (0/{total_count})",
                f"arc-supported: {arc_supported_count}/{total_count}",
            ]
        )
    annotation_lines.append(f"missed: {missed_count}/{total_count}")
    rms_annotation = "\n".join(annotation_lines)
    ax.text(
        0.98,
        0.95,
        rms_annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.88,
        },
    )
    ax.set_xlabel("image residual (arcsec)")
    ax.set_ylabel("N images")
    ax.set_title(
        "Image Residuals: Point and Arc-Aware Recovery"
        if use_arc_aware_diagnostics
        else "Image Residuals: Point Recovery"
    )
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(axis="x", alpha=0.10, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8.5, frameon=False)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _finite_plot_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _critical_arc_support_available(image_df: pd.DataFrame | None) -> bool:
    if image_df is None or image_df.empty:
        return False
    for column in (
        "arc_s_min",
        "arc_noncritical_direction_residual_arcsec",
        "arc_aware_image_residual_arcsec",
        "arc_aware_image_residual_q50",
        "p_arc",
        "arc_prior_probability",
    ):
        if column in image_df.columns and np.isfinite(pd.to_numeric(image_df[column], errors="coerce").to_numpy(dtype=float)).any():
            return True
    return False


def _critical_arc_status_counts(image_df: pd.DataFrame) -> dict[str, int]:
    status = _image_catalog_effective_recovery_statuses(image_df)
    return {
        "point_recovered": int(np.sum(status == "POINT_RECOVERED")),
        "arc_supported": int(np.sum(status == "ARC_RECOVERED")),
        "not_recovered": int(np.sum(status == "MISSED")),
    }


def _histogram_bins(values: np.ndarray) -> int:
    count = int(np.sum(np.isfinite(values)))
    return min(40, max(8, int(np.sqrt(max(count, 1)) * 2)))


def _mark_panel_unavailable(ax: Any, message: str = "No finite values") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color="0.4",
    )


def _log10_s_min_values(s_min: Any) -> np.ndarray:
    values = np.asarray(s_min, dtype=float)
    return np.log10(np.maximum(values, CRITICAL_ARC_LOG_S_MIN_FLOOR))


def _critical_arc_probability_curve(
    s_min: np.ndarray,
    *,
    base_prob: float,
    max_prob: float,
    singular_threshold: float,
    singular_softness: float,
    sample_likelihood_mode: str = CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE,
) -> np.ndarray:
    softness = max(float(singular_softness), 1.0e-12)
    argument = np.clip((float(singular_threshold) - np.asarray(s_min, dtype=float)) / softness, -700.0, 700.0)
    transition = 1.0 / (1.0 + np.exp(-argument))
    if str(sample_likelihood_mode) == CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE_MODE:
        return np.clip(transition, 1.0e-6, 1.0 - 1.0e-6)
    return np.clip(float(base_prob) + (float(max_prob) - float(base_prob)) * transition, 1.0e-6, 1.0 - 1.0e-6)


def _plot_critical_arc_support_histogram(
    image_df: pd.DataFrame,
    path: Path,
    *,
    arc_recovery_p_arc_threshold: float = CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD,
    critical_arc_base_prob: float = CRITICAL_ARC_BASE_PROB,
    critical_arc_max_prob: float = CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = CRITICAL_ARC_SINGULAR_SOFTNESS,
    sample_likelihood_mode: str = CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE,
) -> None:
    if not _critical_arc_support_available(image_df):
        _write_placeholder_plot(
            path,
            "Critical-arc support histogram",
            "No finite critical-arc support diagnostics are available.",
        )
        return

    strict_residual = _finite_plot_values(_fit_quality_value(image_df, "image_residual_q50", "image_residual_arcsec"))
    arc_residual = _finite_plot_values(
        _fit_quality_value(image_df, "arc_aware_image_residual_q50", "arc_aware_image_residual_arcsec")
    )
    curve_distance_raw = _fit_quality_value(image_df, "arc_curve_distance_arcsec", "arc_noncritical_direction_residual_arcsec")
    critical_direction_residual = _finite_plot_values(_fit_quality_value(image_df, "arc_critical_direction_residual_arcsec"))
    s_min_raw = _fit_quality_value(image_df, "arc_s_min")
    arc_prior_raw = _fit_quality_value(image_df, "arc_prior_probability")
    p_arc_raw = _fit_quality_value(image_df, "p_arc")
    curve_distance = _finite_plot_values(curve_distance_raw)
    s_min = _finite_plot_values(s_min_raw)
    arc_prior = _finite_plot_values(arc_prior_raw)
    p_arc_values = _finite_plot_values(p_arc_raw)
    finite_s_min = np.isfinite(s_min_raw)
    log_s_min_raw = np.full_like(s_min_raw, np.nan, dtype=float)
    log_s_min_raw[finite_s_min] = _log10_s_min_values(s_min_raw[finite_s_min])
    log_s_min = _finite_plot_values(log_s_min_raw)
    counts = _critical_arc_status_counts(image_df)
    strict_rms = float(np.sqrt(np.mean(np.square(strict_residual)))) if strict_residual.size else np.nan
    arc_rms = float(np.sqrt(np.mean(np.square(arc_residual)))) if arc_residual.size else np.nan
    status = _image_catalog_effective_recovery_statuses(image_df)
    status_colors = {
        "POINT_RECOVERED": _image_catalog_status_color("POINT_RECOVERED"),
        "ARC_RECOVERED": _image_catalog_status_color("ARC_RECOVERED"),
        "MISSED": _image_catalog_status_color("MISSED"),
    }

    threshold = float(singular_threshold)
    softness = float(singular_softness)
    threshold_log = np.log10(threshold) if np.isfinite(threshold) and threshold > 0.0 else np.nan
    transition_low = threshold - softness
    transition_high = threshold + softness
    transition_log_low = np.log10(transition_low) if np.isfinite(transition_low) and transition_low > 0.0 else np.nan
    transition_log_high = np.log10(transition_high) if np.isfinite(transition_high) and transition_high > 0.0 else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.4))
    residual_ax, noncritical_ax, critical_direction_ax, singular_ax, probability_ax, tuning_ax = axes.ravel()
    if strict_residual.size:
        residual_ax.hist(strict_residual, bins=_histogram_bins(strict_residual), color="tab:blue", alpha=0.52, label="strict")
    if arc_residual.size:
        residual_ax.hist(arc_residual, bins=_histogram_bins(arc_residual), color="tab:olive", alpha=0.48, label="arc-aware")
    if not strict_residual.size and not arc_residual.size:
        _mark_panel_unavailable(residual_ax)
    residual_ax.set_xlabel("image residual (arcsec)")
    residual_ax.set_ylabel("N images")
    residual_ax.set_title("Strict vs Arc-Aware Residual")
    if strict_residual.size or arc_residual.size:
        residual_ax.legend(loc="best", fontsize=8)
    residual_ax.text(
        0.98,
        0.95,
        (
            f"strict RMS={strict_rms:.3g}, N={strict_residual.size}\n"
            f"arc RMS={arc_rms:.3g}, N={arc_residual.size}\n"
            f"point={counts['point_recovered']} arc={counts['arc_supported']} missing={counts['not_recovered']}"
        ),
        transform=residual_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
    )

    if curve_distance.size:
        noncritical_ax.hist(curve_distance, bins=_histogram_bins(curve_distance), color="tab:green", alpha=0.75)
    else:
        _mark_panel_unavailable(noncritical_ax)
    noncritical_ax.set_xlabel("support-curve distance (arcsec)")
    noncritical_ax.set_ylabel("N images")
    noncritical_ax.set_title("Support-Curve Distance Diagnostic")

    if critical_direction_residual.size:
        critical_direction_log = np.log10(np.maximum(critical_direction_residual, 1.0e-6))
        critical_direction_ax.hist(critical_direction_log, bins=_histogram_bins(critical_direction_log), color="tab:purple", alpha=0.72)
    else:
        _mark_panel_unavailable(critical_direction_ax)
    critical_direction_ax.set_xlabel("log10 critical-direction residual (arcsec)")
    critical_direction_ax.set_ylabel("N images")
    critical_direction_ax.set_title("Critical-Direction Residual")

    if log_s_min.size:
        singular_ax.hist(log_s_min, bins=_histogram_bins(log_s_min), color="tab:orange", alpha=0.72)
        if np.isfinite(transition_log_low) and np.isfinite(transition_log_high) and transition_log_high > transition_log_low:
            singular_ax.axvspan(transition_log_low, transition_log_high, color="tab:orange", alpha=0.12, label="threshold +/- softness")
        if np.isfinite(threshold_log):
            singular_ax.axvline(threshold_log, color="black", linestyle="--", linewidth=1.1, label="singular threshold")
        singular_ax.set_xlabel("log10 smallest singular value")
    elif arc_prior.size:
        singular_ax.hist(arc_prior, bins=_histogram_bins(arc_prior), color="tab:orange", alpha=0.72)
        singular_ax.set_xlabel("arc prior probability")
    else:
        _mark_panel_unavailable(singular_ax)
    singular_ax.set_ylabel("N images")
    singular_ax.set_title("Local Criticality")
    if log_s_min.size and np.isfinite(threshold_log):
        singular_ax.legend(loc="best", fontsize=8)

    finite_probability = finite_s_min & np.isfinite(p_arc_raw)
    if finite_probability.any():
        for status_name, color in status_colors.items():
            mask = finite_probability & (status == status_name)
            if mask.any():
                probability_ax.scatter(
                    log_s_min_raw[mask],
                    p_arc_raw[mask],
                    s=28.0,
                    color=color,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.35,
                    label=_image_catalog_status_display_text(status_name),
                )
        other_mask = finite_probability & ~np.isin(status, list(status_colors))
        if other_mask.any():
            probability_ax.scatter(log_s_min_raw[other_mask], p_arc_raw[other_mask], s=28.0, color="0.4", alpha=0.65, label="other")
        finite_log_s = log_s_min_raw[finite_probability]
        curve_log_min = float(np.nanmin(finite_log_s))
        curve_log_max = float(np.nanmax(finite_log_s))
        if np.isfinite(threshold_log):
            curve_log_min = min(curve_log_min, threshold_log - 1.0)
            curve_log_max = max(curve_log_max, threshold_log + 1.0)
        if curve_log_max <= curve_log_min:
            curve_log_max = curve_log_min + 1.0
        curve_log_s = np.linspace(curve_log_min, curve_log_max, 256)
        curve_s = np.power(10.0, curve_log_s)
        if np.isfinite(arc_prior_raw).any():
            probability_ax.plot(
                curve_log_s,
                _critical_arc_probability_curve(
                    curve_s,
                    base_prob=critical_arc_base_prob,
                    max_prob=critical_arc_max_prob,
                    singular_threshold=threshold,
                    singular_softness=softness,
                    sample_likelihood_mode=sample_likelihood_mode,
                ),
                color="0.25",
                linestyle=":",
                linewidth=1.2,
                label="p_arc sigmoid",
            )
        if np.isfinite(threshold_log):
            probability_ax.axvline(threshold_log, color="black", linestyle="--", linewidth=1.0)
        probability_ax.axhline(
            float(arc_recovery_p_arc_threshold),
            color="tab:red",
            linestyle="-.",
            linewidth=1.2,
            label="p_arc threshold",
        )
        probability_ax.legend(loc="best", fontsize=7.5)
    elif p_arc_values.size:
        probability_ax.hist(p_arc_values, bins=_histogram_bins(p_arc_values), color="tab:cyan", alpha=0.72)
        probability_ax.axvline(
            float(arc_recovery_p_arc_threshold),
            color="tab:red",
            linestyle="-.",
            linewidth=1.2,
            label="p_arc threshold",
        )
        probability_ax.set_xlabel("p_arc")
        probability_ax.legend(loc="best", fontsize=7.5)
    else:
        _mark_panel_unavailable(probability_ax)
    if finite_probability.any():
        probability_ax.set_xlabel("log10 smallest singular value")
        probability_ax.set_ylabel("p_arc")
    elif p_arc_values.size:
        probability_ax.set_ylabel("N images")
    probability_ax.set_title("Singular-Value p_arc Gate")

    finite_tuning = finite_s_min & np.isfinite(curve_distance_raw) & np.isfinite(p_arc_raw)
    if finite_tuning.any():
        tuning_scatter = tuning_ax.scatter(
            log_s_min_raw[finite_tuning],
            curve_distance_raw[finite_tuning],
            c=p_arc_raw[finite_tuning],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=32.0,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        fig.colorbar(tuning_scatter, ax=tuning_ax, label="p_arc")
        if np.isfinite(threshold_log):
            tuning_ax.axvline(threshold_log, color="black", linestyle="--", linewidth=1.0, label="singular threshold")
        tuning_ax.legend(loc="best", fontsize=7.5)
    else:
        _mark_panel_unavailable(tuning_ax)
    tuning_ax.set_xlabel("log10 smallest singular value")
    tuning_ax.set_ylabel("support-curve distance (arcsec)")
    tuning_ax.set_title("p_arc vs Support Distance")

    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_critical_arc_support_phase_space(
    image_df: pd.DataFrame,
    path: Path,
    *,
    arc_recovery_p_arc_threshold: float = CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD,
    critical_arc_base_prob: float = CRITICAL_ARC_BASE_PROB,
    critical_arc_max_prob: float = CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = CRITICAL_ARC_SINGULAR_SOFTNESS,
    sample_likelihood_mode: str = CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE,
) -> None:
    if image_df.empty or "arc_s_min" not in image_df.columns or "p_arc" not in image_df.columns:
        _write_placeholder_plot(
            path,
            "Critical-arc support phase space",
            "No finite critical-arc phase-space diagnostics are available.",
        )
        return
    s_min = _fit_quality_value(image_df, "arc_s_min")
    p_arc = _fit_quality_value(image_df, "p_arc")
    finite = np.isfinite(s_min) & np.isfinite(p_arc)
    if not finite.any():
        _write_placeholder_plot(
            path,
            "Critical-arc support phase space",
            "No finite critical-arc phase-space diagnostics are available.",
        )
        return
    status = _image_catalog_effective_recovery_statuses(image_df)
    log_s_min = _log10_s_min_values(s_min)
    colors = {
        "POINT_RECOVERED": _image_catalog_status_color("POINT_RECOVERED"),
        "ARC_RECOVERED": _image_catalog_status_color("ARC_RECOVERED"),
        "MISSED": _image_catalog_status_color("MISSED"),
    }
    threshold = float(singular_threshold)
    softness = float(singular_softness)
    threshold_log = np.log10(threshold) if np.isfinite(threshold) and threshold > 0.0 else np.nan
    finite_log_s = log_s_min[finite]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    curve_log_min = float(np.nanmin(finite_log_s))
    curve_log_max = float(np.nanmax(finite_log_s))
    if np.isfinite(threshold_log):
        curve_log_min = min(curve_log_min, threshold_log - 1.0)
        curve_log_max = max(curve_log_max, threshold_log + 1.0)
    if curve_log_max <= curve_log_min:
        curve_log_max = curve_log_min + 1.0
    curve_log_s = np.linspace(curve_log_min, curve_log_max, 256)
    curve_s = np.power(10.0, curve_log_s)
    if (
        np.isfinite(threshold)
        and threshold > 0.0
        and np.isfinite(softness)
        and softness > 0.0
        and np.isfinite(float(critical_arc_base_prob))
        and np.isfinite(float(critical_arc_max_prob))
    ):
        ax.plot(
            curve_log_s,
            _critical_arc_probability_curve(
                curve_s,
                base_prob=float(critical_arc_base_prob),
                max_prob=float(critical_arc_max_prob),
                singular_threshold=threshold,
                singular_softness=softness,
                sample_likelihood_mode=sample_likelihood_mode,
            ),
            color="0.25",
            linestyle=":",
            linewidth=1.3,
            label=r"$p_{\rm arc}$ sigmoid",
            zorder=1,
        )
    for status_name, color in colors.items():
        mask = finite & (status == status_name)
        if mask.any():
            ax.scatter(
                log_s_min[mask],
                p_arc[mask],
                s=34.0,
                color=color,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                label=_image_catalog_status_display_text(status_name),
                zorder=3,
            )
    other_mask = finite & ~np.isin(status, list(colors))
    if other_mask.any():
        ax.scatter(
            log_s_min[other_mask],
            p_arc[other_mask],
            s=34.0,
            color="0.4",
            alpha=0.65,
            edgecolors="white",
            linewidths=0.35,
            label="other",
            zorder=3,
        )
    if np.isfinite(threshold_log):
        ax.axvline(threshold_log, color="black", linestyle="--", linewidth=1.0, label="singular threshold")
    ax.axhline(
        float(arc_recovery_p_arc_threshold),
        color="tab:red",
        linestyle="-.",
        linewidth=1.2,
        label=r"$p_{\rm arc}$ threshold",
    )
    ax.set_xlabel(r"$\log_{10} s_{\min}$")
    ax.set_ylabel(r"$p_{\rm arc}$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_critical_arc_recovery_by_family(image_count_df: pd.DataFrame, path: Path) -> None:
    required = {
        "family_id",
        "observed_image_count",
        "recovered_image_count",
        "arc_aware_recovered_image_count",
        "arc_supported_image_count",
    }
    if image_count_df is None or image_count_df.empty or not required.issubset(image_count_df.columns):
        _write_placeholder_plot(
            path,
            "Critical-arc recovery by family",
            "No family-level critical-arc recovery counts are available.",
        )
        return
    df = image_count_df.copy()
    for column in [
        "observed_image_count",
        "recovered_image_count",
        "missing_image_count",
        "arc_aware_recovered_image_count",
        "arc_aware_missing_image_count",
        "arc_supported_image_count",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if not np.isfinite(df["arc_aware_recovered_image_count"].to_numpy(dtype=float)).any():
        _write_placeholder_plot(
            path,
            "Critical-arc recovery by family",
            "No finite family-level critical-arc recovery counts are available.",
        )
        return
    df["_arc_missing_sort"] = np.nan_to_num(df.get("arc_aware_missing_image_count", np.nan), nan=-1.0)
    df["_strict_missing_sort"] = np.nan_to_num(df.get("missing_image_count", np.nan), nan=-1.0)
    df = df.sort_values(["_arc_missing_sort", "_strict_missing_sort", "family_id"], ascending=[False, False, True])
    labels = df["family_id"].astype(str).to_numpy()
    y = np.arange(len(df))
    height = 0.18
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.36 * len(df))))
    series = [
        ("observed", "observed_image_count", "0.65", -1.5 * height),
        ("point recovered", "recovered_image_count", _image_catalog_status_color("POINT_RECOVERED"), -0.5 * height),
        ("point + arc recovered", "arc_aware_recovered_image_count", _image_catalog_status_color("ARC_RECOVERED"), 0.5 * height),
        ("arc recovered only", "arc_supported_image_count", "#ffb300", 1.5 * height),
    ]
    for label, column, color, offset in series:
        values = df[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            ax.barh(y[finite] + offset, values[finite], height=height, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("image count")
    ax.set_ylabel("family")
    ax.set_title("Critical-Arc Recovery by Family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_residual_vs_magnification(image_df: pd.DataFrame, magnification_df: pd.DataFrame, path: Path) -> None:
    merged = _merge_fit_quality_with_magnification(image_df, magnification_df)
    if merged.empty:
        return
    abs_mu = np.abs(_fit_quality_value(merged, "magnification_model_q50", "magnification_model"))
    residual = _fit_quality_value(merged, "image_residual_q50", "image_residual_arcsec")
    residual_norm = _fit_quality_value(merged, "residual_norm_q50", "residual_norm")
    finite_any = np.isfinite(abs_mu) & (np.isfinite(residual) | np.isfinite(residual_norm))
    if not finite_any.any():
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    finite_residual = np.isfinite(abs_mu) & np.isfinite(residual)
    axes[0].scatter(abs_mu[finite_residual], residual[finite_residual], color="tab:blue", s=26)
    axes[0].set_ylabel("image residual (arcsec)")
    axes[0].set_title("Residual vs Magnification")

    finite_norm = np.isfinite(abs_mu) & np.isfinite(residual_norm)
    axes[1].scatter(abs_mu[finite_norm], residual_norm[finite_norm], color="tab:purple", s=26)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("image residual / sigma_eff")
    axes[1].set_title("Normalized Residual vs Magnification")
    for ax in axes:
        ax.set_xlabel("|model signed magnification|")
        positive = abs_mu[np.isfinite(abs_mu) & (abs_mu > 0.0)]
        if positive.size and float(np.nanmax(positive)) / max(float(np.nanmin(positive)), 1.0e-12) > 20.0:
            ax.set_xscale("log")
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _family_magnitude_matrix(family: Any) -> tuple[np.ndarray, np.ndarray]:
    n_images = int(getattr(family, "n_images", 0))
    magnitudes = np.asarray(getattr(family, "catalog_mag", np.empty((0,))), dtype=float)
    magnitude_errors = np.asarray(getattr(family, "catalog_mag_err", np.empty((0,))), dtype=float)
    if n_images <= 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    if magnitudes.ndim == 1:
        if magnitudes.shape != (n_images,):
            magnitudes = np.full((n_images,), np.nan, dtype=float)
        magnitudes = magnitudes[:, None]
    elif magnitudes.ndim == 2:
        if magnitudes.shape[0] != n_images:
            magnitudes = np.full((n_images, 0), np.nan, dtype=float)
    else:
        magnitudes = np.full((n_images, 0), np.nan, dtype=float)

    if magnitude_errors.ndim == 1:
        if magnitude_errors.shape == (n_images,):
            magnitude_errors = magnitude_errors[:, None]
        else:
            magnitude_errors = np.full(magnitudes.shape, np.nan, dtype=float)
    elif magnitude_errors.ndim == 2:
        if magnitude_errors.shape != magnitudes.shape:
            magnitude_errors = np.full(magnitudes.shape, np.nan, dtype=float)
    else:
        magnitude_errors = np.full(magnitudes.shape, np.nan, dtype=float)
    return magnitudes, magnitude_errors


def _family_magnitude_band_names_for_plot(family: Any, n_bands: int) -> list[str]:
    names = [str(name).removeprefix("mag_").lower() for name in getattr(family, "catalog_mag_band_names", []) or []]
    if len(names) == int(n_bands):
        return names
    if int(n_bands) == 1:
        return ["catalog"]
    if int(n_bands) == 7:
        return ["f105w", "f125w", "f140w", "f160w", "f435w", "f606w", "f814w"]
    return [f"band_{idx + 1}" for idx in range(int(n_bands))]


def _reference_magnitude_band_index(band_names: Sequence[str]) -> int:
    normalized = [str(name).removeprefix("mag_").lower() for name in band_names]
    for preferred in ("f160w", "hst_f160w"):
        if preferred in normalized:
            return int(normalized.index(preferred))
    for preferred in ("f140w", "hst_f140w", "f125w", "hst_f125w", "f105w", "hst_f105w"):
        if preferred in normalized:
            return int(normalized.index(preferred))
    return 0


def _flux_magnification_ratio_pair_table(
    state: BuildState,
    magnification_df: pd.DataFrame,
    image_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "family_id",
        "band",
        "band_index",
        "image_label_i",
        "image_label_j",
        "magnitude_i",
        "magnitude_j",
        "magnitude_error_i",
        "magnitude_error_j",
        "observed_mag_difference",
        "observed_mag_difference_error",
        "magnification_i",
        "magnification_j",
        "p_arc_i",
        "p_arc_j",
        "p_arc_pair",
        "model_mag_difference",
        "model_minus_observed",
    ]
    if magnification_df is None or magnification_df.empty:
        return pd.DataFrame(columns=columns)
    required = {"family_id", "image_label"}
    if not required.issubset(magnification_df.columns):
        return pd.DataFrame(columns=columns)
    mu_column = "magnification_model_q50" if "magnification_model_q50" in magnification_df.columns else "magnification_model"
    if mu_column not in magnification_df.columns:
        return pd.DataFrame(columns=columns)

    mu_lookup: dict[tuple[str, str], float] = {}
    for _, row in magnification_df.iterrows():
        family_id = str(row.get("family_id", ""))
        image_label = str(row.get("image_label", ""))
        mu_value = pd.to_numeric(pd.Series([row.get(mu_column, np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(mu_value) and "magnification_model" in magnification_df.columns:
            mu_value = pd.to_numeric(pd.Series([row.get("magnification_model", np.nan)]), errors="coerce").iloc[0]
        mu_lookup[(family_id, image_label)] = float(mu_value) if np.isfinite(mu_value) else np.nan

    p_arc_lookup: dict[tuple[str, str], float] = {}
    if image_df is not None and not image_df.empty and {"family_id", "image_label", "p_arc"}.issubset(image_df.columns):
        for _, row in image_df.iterrows():
            family_id = str(row.get("family_id", ""))
            image_label = str(row.get("image_label", ""))
            p_arc_value = pd.to_numeric(pd.Series([row.get("p_arc", np.nan)]), errors="coerce").iloc[0]
            p_arc_lookup[(family_id, image_label)] = float(p_arc_value) if np.isfinite(p_arc_value) else np.nan

    rows: list[dict[str, Any]] = []
    for family in getattr(state, "family_data", []) or []:
        family_id = str(getattr(family, "family_id", ""))
        labels = [str(label) for label in getattr(family, "image_labels", [])]
        n_images = len(labels)
        if n_images < 2:
            continue
        magnitudes, magnitude_errors = _family_magnitude_matrix(family)
        if magnitudes.shape[0] != n_images or magnitudes.shape[1] == 0:
            continue
        n_bands = int(magnitudes.shape[1])
        band_names = _family_magnitude_band_names_for_plot(family, n_bands)
        band_index = _reference_magnitude_band_index(band_names)
        band_name = band_names[band_index]
        for i in range(n_images - 1):
            mu_i = mu_lookup.get((family_id, labels[i]), np.nan)
            p_arc_i = p_arc_lookup.get((family_id, labels[i]), np.nan)
            for j in range(i + 1, n_images):
                mu_j = mu_lookup.get((family_id, labels[j]), np.nan)
                p_arc_j = p_arc_lookup.get((family_id, labels[j]), np.nan)
                if not (np.isfinite(mu_i) and np.isfinite(mu_j) and abs(mu_i) > 0.0 and abs(mu_j) > 0.0):
                    continue
                model_mag_difference = -2.5 * np.log10(abs(mu_i) / abs(mu_j))
                mag_i = float(magnitudes[i, band_index])
                mag_j = float(magnitudes[j, band_index])
                if not (np.isfinite(mag_i) and np.isfinite(mag_j)):
                    continue
                err_i = float(magnitude_errors[i, band_index]) if magnitude_errors.shape == magnitudes.shape else np.nan
                err_j = float(magnitude_errors[j, band_index]) if magnitude_errors.shape == magnitudes.shape else np.nan
                observed_mag_difference = mag_i - mag_j
                observed_mag_difference_error = (
                    float(np.sqrt(err_i**2 + err_j**2))
                    if np.isfinite(err_i) and np.isfinite(err_j) and err_i >= 0.0 and err_j >= 0.0
                    else np.nan
                )
                rows.append(
                    {
                        "family_id": family_id,
                        "band": band_name,
                        "band_index": int(band_index),
                        "image_label_i": labels[i],
                        "image_label_j": labels[j],
                        "magnitude_i": mag_i,
                        "magnitude_j": mag_j,
                        "magnitude_error_i": err_i,
                        "magnitude_error_j": err_j,
                        "observed_mag_difference": float(observed_mag_difference),
                        "observed_mag_difference_error": observed_mag_difference_error,
                        "magnification_i": float(mu_i),
                        "magnification_j": float(mu_j),
                        "p_arc_i": float(p_arc_i) if np.isfinite(p_arc_i) else np.nan,
                        "p_arc_j": float(p_arc_j) if np.isfinite(p_arc_j) else np.nan,
                        "p_arc_pair": float(np.nanmax([p_arc_i, p_arc_j])) if np.isfinite([p_arc_i, p_arc_j]).any() else np.nan,
                        "model_mag_difference": float(model_mag_difference),
                        "model_minus_observed": float(model_mag_difference - observed_mag_difference),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _plot_flux_magnification_ratio_consistency(pair_df: pd.DataFrame, path: Path) -> None:
    if pair_df is None or pair_df.empty:
        _write_placeholder_plot(
            path,
            "Flux-Magnification Ratio Consistency",
            "No same-family image pairs with finite magnitudes and model magnifications are available.",
        )
        return
    observed = pd.to_numeric(pair_df["observed_mag_difference"], errors="coerce").to_numpy(dtype=float)
    model = pd.to_numeric(pair_df["model_mag_difference"], errors="coerce").to_numpy(dtype=float)
    errors = pd.to_numeric(pair_df["observed_mag_difference_error"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(observed) & np.isfinite(model)
    if not np.any(finite):
        _write_placeholder_plot(
            path,
            "Flux-Magnification Ratio Consistency",
            "No finite same-family magnitude-ratio residuals are available.",
        )
        return

    observed = observed[finite]
    model = model[finite]
    errors = errors[finite]
    bands = pair_df.loc[finite, "band"].astype(str).to_numpy()
    p_arc_pair = (
        pd.to_numeric(pair_df.loc[finite, "p_arc_pair"], errors="coerce").to_numpy(dtype=float)
        if "p_arc_pair" in pair_df.columns
        else np.full(observed.shape, np.nan, dtype=float)
    )
    residual = model - observed
    bias = float(np.nanmedian(residual))
    nmad = float(1.4826 * np.nanmedian(np.abs(residual - bias)))
    rmse = float(np.sqrt(np.nanmean(np.square(residual))))

    combined = np.concatenate([observed, model])
    finite_combined = combined[np.isfinite(combined)]
    lower = float(np.nanmin(finite_combined))
    upper = float(np.nanmax(finite_combined))
    if lower == upper:
        span = max(abs(lower), 1.0) * 0.05
    else:
        span = upper - lower
    limits = (lower - 0.06 * span, upper + 0.06 * span)
    guide = np.linspace(limits[0], limits[1], 128)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(guide, guide, color="0.15", linewidth=1.0, label="1:1")
    finite_error = np.isfinite(errors) & (errors >= 0.0)
    if np.any(finite_error):
        ax.hlines(
            model[finite_error],
            observed[finite_error] - errors[finite_error],
            observed[finite_error] + errors[finite_error],
            color="0.6",
            linewidth=0.7,
            alpha=0.45,
            zorder=2,
        )
    finite_p_arc = np.isfinite(p_arc_pair)
    if np.any(finite_p_arc):
        log_p_arc_pair = np.full_like(p_arc_pair, np.nan, dtype=float)
        log_p_arc_pair[finite_p_arc] = np.log10(
            np.clip(p_arc_pair[finite_p_arc], FLUX_MAGNIFICATION_P_ARC_LOG10_FLOOR, 1.0)
        )
        scatter = ax.scatter(
            observed,
            model,
            c=log_p_arc_pair,
            cmap="viridis",
            vmin=float(np.log10(FLUX_MAGNIFICATION_P_ARC_LOG10_FLOOR)),
            vmax=0.0,
            marker="o",
            s=24,
            alpha=0.86,
            edgecolors="black",
            linewidths=0.35,
            label="image pairs",
            zorder=4,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(r"$\log_{10}$ pair $p_{\rm arc}$")
    else:
        band = bands[0] if len(set(bands.tolist())) == 1 else "image pairs"
        ax.scatter(
            observed,
            model,
            marker="o",
            s=24,
            color="tab:orange",
            alpha=0.82,
            edgecolors="black",
            linewidths=0.35,
            label=band,
            zorder=4,
        )
    metric_text = "\n".join(
        [
            f"bias: {_format_recovery_metric(bias)}",
            f"NMAD: {_format_recovery_metric(nmad)}",
            f"RMSE: {_format_recovery_metric(rmse)}",
            f"pairs: {int(np.count_nonzero(finite))}",
        ]
    )
    ax.text(
        0.98,
        0.04,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"observed $m_i - m_j$ (mag)")
    ax.set_ylabel(r"model $-2.5\log_{10}(|\mu_i|/|\mu_j|)$ (mag)")
    ax.set_title("Same-Family Flux-Magnification Ratio Consistency")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=220, bbox_inches="tight")


def _plot_residual_geometry_trends(image_df: pd.DataFrame, path: Path) -> None:
    if image_df.empty:
        return
    residual = _fit_quality_value(image_df, "image_residual_q50", "image_residual_arcsec")
    x_columns = [
        ("radius_arcsec", "radius (arcsec)", "Residual vs Radius"),
        ("angle_deg", "angle (deg)", "Residual vs Angle"),
        ("z_source", "source redshift", "Residual vs Redshift"),
    ]
    if not any(column in image_df.columns for column, _xlabel, _title in x_columns):
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    plotted = False
    for ax, (column, xlabel, title) in zip(axes, x_columns):
        x_values = _fit_quality_value(image_df, column)
        finite = np.isfinite(x_values) & np.isfinite(residual)
        if finite.any():
            ax.scatter(x_values[finite], residual[finite], color="tab:blue", s=24)
            plotted = True
        ax.set_xlabel(xlabel)
        ax.set_ylabel("image residual (arcsec)")
        ax.set_title(title)
    if not plotted:
        plt.close(fig)
        return
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_exact_vs_approx_prediction_error(family_df: pd.DataFrame, path: Path) -> None:
    if family_df.empty or not {"exact_image_rms_arcsec", "approx_image_rms_arcsec"}.issubset(family_df.columns):
        return
    exact = _fit_quality_value(family_df, "exact_image_rms_arcsec")
    approx = _fit_quality_value(family_df, "approx_image_rms_arcsec")
    finite = np.isfinite(exact) & np.isfinite(approx)
    if not finite.any():
        return
    labels = family_df["family_id"].astype(str).to_numpy() if "family_id" in family_df.columns else np.arange(len(family_df)).astype(str)
    exact = exact[finite]
    approx = approx[finite]
    labels = labels[finite]
    diff = exact - approx
    ratio = exact / np.where(np.abs(approx) > 1.0e-12, approx, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(approx, exact, color="tab:blue", s=30)
    finite_pair = np.concatenate([approx[np.isfinite(approx)], exact[np.isfinite(exact)]])
    if finite_pair.size:
        lo = float(np.nanmin(finite_pair))
        hi = float(np.nanmax(finite_pair))
        axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    axes[0].set_xlabel("approx image RMS (arcsec)")
    axes[0].set_ylabel("exact image RMS (arcsec)")
    axes[0].set_title("Exact vs Approx RMS")

    order = np.argsort(diff)
    axes[1].bar(np.arange(len(diff)), diff[order], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("exact - approx (arcsec)")
    axes[1].set_title("RMS Difference")

    axes[2].bar(np.arange(len(ratio)), ratio[order], color="tab:purple")
    axes[2].axhline(1.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("exact / approx")
    axes[2].set_title("RMS Ratio")
    if len(labels) <= 40:
        for ax in axes[1:]:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels[order], rotation=90, fontsize=7)
            ax.set_xlabel("family")
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_per_potential_summary(
    plot_dir: Path,
    summary_df: pd.DataFrame,
    previous_stage_best_values: dict[str, float] | None = None,
    parameter_specs: list[ParameterSpec] | None = None,
) -> None:
    if summary_df.empty:
        return
    previous_stage_by_label: dict[str, float] = {}
    if previous_stage_best_values and parameter_specs:
        for spec in parameter_specs:
            values = _corner_values_for_specs([spec], previous_stage_best_values)
            if values and np.isfinite(values[0]):
                previous_stage_by_label[str(spec.name)] = float(values[0])
    nrows = len(summary_df)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4, 1.4 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]
    for ax, row in zip(axes, summary_df.itertuples(index=False)):
        previous_stage_value = previous_stage_by_label.get(str(row.label))
        ax.hlines(1, row.p16, row.p84, linewidth=4, color="tab:blue")
        ax.scatter([row.median], [1], color="tab:blue", s=35, label="median")
        ax.scatter([row.map], [1], color=CORNER_BEST_FIT_COLOR, marker="x", s=30, label="best fit")
        if previous_stage_value is not None:
            ax.scatter(
                [previous_stage_value],
                [1],
                color=CORNER_PREVIOUS_STAGE_COLOR,
                marker="x",
                s=30,
                label="previous stage",
            )
        if np.isfinite(row.lower) and np.isfinite(row.upper):
            x_min = float(row.lower)
            x_max = float(row.upper)
        else:
            width = max(float(row.std), 0.5 * abs(float(row.p84) - float(row.p16)), 1.0e-3)
            x_min = float(min(row.p16, row.median, row.map) - 2.0 * width)
            x_max = float(max(row.p84, row.median, row.map) + 2.0 * width)
        for marker_value in (previous_stage_value,):
            if marker_value is None:
                continue
            span = max(abs(x_max - x_min), 1.0e-3)
            pad = 0.05 * span
            x_min = min(x_min, float(marker_value) - pad)
            x_max = max(x_max, float(marker_value) + pad)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])
        ax.set_title(row.label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "per_potential_summary.png"), dpi=180, bbox_inches="tight")


def _plot_timing_profile(plot_dir: Path, evaluator: ClusterJAXEvaluator) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    names = list(evaluator.timing_totals.keys())
    values = [evaluator.timing_totals[name] for name in names]
    ax.bar(names, values, color="tab:cyan")
    ax.set_ylabel("seconds")
    ax.set_title("Timing Totals")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "timing_profile.png"), dpi=180, bbox_inches="tight")


def _critical_curve_caustics(
    lens_model: Any,
    kwargs_lens: list[dict[str, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    include_tangential: bool = True,
    include_radial: bool = False,
) -> list[dict[str, np.ndarray]]:
    x_values = np.asarray(x_axis, dtype=float)
    y_values = np.asarray(y_axis, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size < 2 or y_values.size < 2:
        return []
    xx, yy = np.meshgrid(x_values, y_values)
    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(
        xx.ravel(),
        yy.ravel(),
        kwargs_lens,
    )
    shape = xx.shape
    f_xx = np.asarray(f_xx, dtype=float).reshape(shape)
    f_yy = np.asarray(f_yy, dtype=float).reshape(shape)
    f_xy = np.asarray(f_xy, dtype=float).reshape(shape)
    f_yx = np.asarray(f_yx, dtype=float).reshape(shape)
    kappa = 0.5 * (f_xx + f_yy)
    gamma1 = 0.5 * (f_xx - f_yy)
    gamma2 = 0.5 * (f_xy + f_yx)
    contours: list[dict[str, np.ndarray]] = []
    pixel_x = np.arange(x_values.size, dtype=float)
    pixel_y = np.arange(y_values.size, dtype=float)
    contour_fields: list[tuple[str, np.ndarray]] = []
    shear = np.hypot(gamma1, gamma2)
    if include_tangential:
        contour_fields.append(("tangential", 1.0 - kappa - shear))
    if include_radial:
        contour_fields.append(("radial", 1.0 - kappa + shear))
    for kind, field in contour_fields:
        for vertices in find_contours(field, 0.0):
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
                continue
            crit_x = np.interp(vertices[:, 1], pixel_x, x_values)
            crit_y = np.interp(vertices[:, 0], pixel_y, y_values)
            if not np.all(np.isfinite(crit_x)) or not np.all(np.isfinite(crit_y)):
                continue
            beta_x, beta_y = lens_model.ray_shooting(crit_x, crit_y, kwargs_lens)
            beta_x = np.asarray(beta_x, dtype=float)
            beta_y = np.asarray(beta_y, dtype=float)
            if beta_x.shape != crit_x.shape or beta_y.shape != crit_y.shape:
                continue
            if not np.all(np.isfinite(beta_x)) or not np.all(np.isfinite(beta_y)):
                continue
            contours.append(
                {
                    "kind": kind,
                    "critical_x": crit_x,
                    "critical_y": crit_y,
                    "caustic_x": beta_x,
                    "caustic_y": beta_y,
                }
            )
    return contours


def _tangential_critical_curve_caustics(
    lens_model: Any,
    kwargs_lens: list[dict[str, float]],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    return _critical_curve_caustics(
        lens_model,
        kwargs_lens,
        x_axis,
        y_axis,
        include_tangential=True,
        include_radial=False,
    )


def _plot_caustic_overlay(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_plot_grid_scale_arcsec: float,
    caustic_source_redshift: float,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:caustic_overlay] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    x_grid, y_grid = _caustic_plot_grid_axes(caustic_plot_grid_scale_arcsec)
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    contours = _tangential_critical_curve_caustics(model, kwargs_lens, x_grid, y_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image_ax, source_ax = axes
    for contour in contours:
        image_ax.plot(contour["critical_x"], contour["critical_y"], color="black", linewidth=1.0)
    cmap = plt.get_cmap("tab20", len(evaluator.state.family_data))
    for idx, fam in enumerate(evaluator.state.family_data):
        image_ax.scatter(fam.x_obs, fam.y_obs, color=cmap(idx), s=14, label=fam.family_id)
    image_ax.invert_xaxis()
    image_ax.set_xlabel("x (arcsec)")
    image_ax.set_ylabel("y (arcsec)")
    image_ax.set_title(f"Tangential Critical Lines + Images (z={z_source:g})")

    for contour in contours:
        source_ax.scatter(
            contour["caustic_x"],
            contour["caustic_y"],
            color="black",
            s=2,
            alpha=0.75,
            linewidths=0.0,
        )
    best_source_eval = evaluator.evaluate(best_fit_latent, validate_all_families=False)
    for idx, fam in enumerate(evaluator.state.family_data):
        pred = best_source_eval.family_predictions[fam.family_id]
        source_ax.scatter(pred["source_x"], pred["source_y"], color=cmap(idx), s=14)
    source_ax.set_xlabel(r"$\beta_x$ (arcsec)")
    source_ax.set_ylabel(r"$\beta_y$ (arcsec)")
    source_ax.set_title("Caustics + Source Centroids")
    fig.tight_layout()
    _finish_figure(fig, _plot_path(plot_dir, "caustic_overlay.png"), dpi=180, bbox_inches="tight")


def _load_kappa_true_fits(path: str | Path) -> tuple[np.ndarray, WCS]:
    fits_path = Path(path)
    with fits.open(fits_path, memmap=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            image = np.squeeze(np.asarray(data, dtype=float))
            if image.ndim != 2:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"(?s).*RADECSYS.*deprecated.*RADESYS.*",
                    category=FITSFixedWarning,
                )
                wcs = WCS(hdu.header).celestial
            if not wcs.has_celestial:
                continue
            return image, wcs
    raise ValueError(f"No 2D celestial WCS image found in {fits_path}")


def _validate_matching_truth_wcs(
    reference_wcs: WCS,
    candidate_wcs: WCS,
    image_shape: tuple[int, int],
    *,
    label: str,
) -> None:
    height, width = image_shape
    x_probe = np.asarray([0.0, float(width - 1), 0.0, float(width - 1), 0.5 * float(width - 1)], dtype=float)
    y_probe = np.asarray([0.0, 0.0, float(height - 1), float(height - 1), 0.5 * float(height - 1)], dtype=float)
    ref_ra, ref_dec = reference_wcs.pixel_to_world_values(x_probe, y_probe)
    candidate_ra, candidate_dec = candidate_wcs.pixel_to_world_values(x_probe, y_probe)
    if not (
        np.allclose(ref_ra, candidate_ra, rtol=0.0, atol=1.0e-8, equal_nan=True)
        and np.allclose(ref_dec, candidate_dec, rtol=0.0, atol=1.0e-8, equal_nan=True)
    ):
        raise ValueError(f"{label} truth FITS WCS does not match the kappa truth FITS WCS.")


def _truth_recovery_diagnostic_grid(
    truth_wcs: WCS,
    native_shape: tuple[int, int],
    truth_grid_size: int | None,
) -> tuple[WCS, tuple[int, int], dict[str, Any]]:
    native_height, native_width = (int(native_shape[0]), int(native_shape[1]))
    requested_size = int(truth_grid_size or 0)
    if requested_size <= 0 or (native_height <= requested_size and native_width <= requested_size):
        metadata = {
            "native_truth_height": native_height,
            "native_truth_width": native_width,
            "diagnostic_grid_height": native_height,
            "diagnostic_grid_width": native_width,
            "truth_grid_size": 0 if requested_size <= 0 else requested_size,
            "truth_grid_sampling": "native",
            "native_to_diagnostic_pixel_ratio_x": 1.0,
            "native_to_diagnostic_pixel_ratio_y": 1.0,
        }
        return truth_wcs.deepcopy(), (native_height, native_width), metadata

    diagnostic_shape = (requested_size, requested_size)
    scale_x = float(native_width) / float(requested_size)
    scale_y = float(native_height) / float(requested_size)
    diagnostic_wcs = truth_wcs.deepcopy()
    crpix = np.asarray(diagnostic_wcs.wcs.crpix, dtype=float).copy()
    if crpix.size >= 2:
        offset_x = 0.5 * scale_x - 0.5
        offset_y = 0.5 * scale_y - 0.5
        crpix[0] = 1.0 + (crpix[0] - offset_x - 1.0) / scale_x
        crpix[1] = 1.0 + (crpix[1] - offset_y - 1.0) / scale_y
        diagnostic_wcs.wcs.crpix = crpix
    pixel_scale_matrix = np.asarray(truth_wcs.pixel_scale_matrix, dtype=float)
    if pixel_scale_matrix.shape == (2, 2):
        diagnostic_wcs.wcs.cd = pixel_scale_matrix @ np.diag([scale_x, scale_y])
    else:
        cdelt = np.asarray(diagnostic_wcs.wcs.cdelt, dtype=float).copy()
        if cdelt.size >= 2:
            cdelt[0] *= scale_x
            cdelt[1] *= scale_y
            diagnostic_wcs.wcs.cdelt = cdelt
    metadata = {
        "native_truth_height": native_height,
        "native_truth_width": native_width,
        "diagnostic_grid_height": requested_size,
        "diagnostic_grid_width": requested_size,
        "truth_grid_size": requested_size,
        "truth_grid_sampling": "bilinear_reduced",
        "native_to_diagnostic_pixel_ratio_x": scale_x,
        "native_to_diagnostic_pixel_ratio_y": scale_y,
    }
    return diagnostic_wcs, diagnostic_shape, metadata


def _sample_wcs_image_on_wcs_grid(
    image: np.ndarray,
    image_wcs: WCS,
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> np.ndarray:
    y_pixels, x_pixels = np.indices(target_shape, dtype=float)
    ra_deg, dec_deg = target_wcs.pixel_to_world_values(x_pixels, y_pixels)
    source_x, source_y = image_wcs.world_to_pixel_values(ra_deg, dec_deg)
    sampled = map_coordinates(
        np.asarray(image, dtype=np.float64),
        [np.asarray(source_y, dtype=float), np.asarray(source_x, dtype=float)],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return np.asarray(sampled, dtype=np.float64).reshape(target_shape)


def _truth_recovery_sample_truth_image(
    image: np.ndarray,
    image_wcs: WCS,
    target_wcs: WCS,
    target_shape: tuple[int, int],
    *,
    native_shape: tuple[int, int],
) -> np.ndarray:
    if tuple(int(value) for value in target_shape) == tuple(int(value) for value in native_shape):
        return np.asarray(image, dtype=np.float64)
    return _sample_wcs_image_on_wcs_grid(image, image_wcs, target_wcs, target_shape)


def _signed_magnification_from_kappa_gamma(kappa: np.ndarray, gamma_x: np.ndarray, gamma_y: np.ndarray) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=float)
    gamma_x = np.asarray(gamma_x, dtype=float)
    gamma_y = np.asarray(gamma_y, dtype=float)
    denominator = (1.0 - kappa) ** 2 - gamma_x**2 - gamma_y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / denominator


def _critical_determinant_from_kappa_gamma(kappa: np.ndarray, gamma_x: np.ndarray, gamma_y: np.ndarray) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=float)
    gamma_x = np.asarray(gamma_x, dtype=float)
    gamma_y = np.asarray(gamma_y, dtype=float)
    return (1.0 - kappa) ** 2 - gamma_x**2 - gamma_y**2


def _radec_to_solver_arcsec_offsets(
    ra_deg: Any,
    dec_deg: Any,
    reference: tuple[int, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        _reference_type, ra0_deg, dec0_deg = reference
        ra0 = float(ra0_deg)
        dec0 = float(dec0_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("State reference must contain reference type, RA, and Dec.") from exc
    cos_dec0 = math.cos(math.radians(dec0))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    ra_values = np.asarray(ra_deg, dtype=float)
    dec_values = np.asarray(dec_deg, dtype=float)
    delta_ra = (ra0 - ra_values + 180.0) % 360.0 - 180.0
    x_arcsec = delta_ra * cos_dec0 * 3600.0
    y_arcsec = (dec_values - dec0) * 3600.0
    return x_arcsec, y_arcsec


def _solver_arcsec_offsets_to_radec(
    x_arcsec: Any,
    y_arcsec: Any,
    reference: tuple[int, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        _reference_type, ra0_deg, dec0_deg = reference
        ra0 = float(ra0_deg)
        dec0 = float(dec0_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("State reference must contain reference type, RA, and Dec.") from exc
    cos_dec0 = math.cos(math.radians(dec0))
    if cos_dec0 == 0.0:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    ra_values = (ra0 - x_values / (3600.0 * cos_dec0)) % 360.0
    dec_values = dec0 + y_values / 3600.0
    return ra_values, dec_values


def _sample_wcs_image_at_solver_arcsec(
    image: np.ndarray,
    image_wcs: WCS,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    reference: tuple[int, float, float],
) -> np.ndarray:
    x_values = np.asarray(x_arcsec, dtype=float).reshape(-1)
    y_values = np.asarray(y_arcsec, dtype=float).reshape(-1)
    sampled = np.full(x_values.shape, np.nan, dtype=float)
    finite_xy = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite_xy):
        return sampled
    ra_deg, dec_deg = _solver_arcsec_offsets_to_radec(x_values[finite_xy], y_values[finite_xy], reference)
    x_pixels, y_pixels = image_wcs.world_to_pixel_values(ra_deg, dec_deg)
    height, width = np.asarray(image).shape
    edge_tolerance = 1.0e-6
    finite_pixels = (
        np.isfinite(x_pixels)
        & np.isfinite(y_pixels)
        & (x_pixels >= -edge_tolerance)
        & (x_pixels <= float(width - 1) + edge_tolerance)
        & (y_pixels >= -edge_tolerance)
        & (y_pixels <= float(height - 1) + edge_tolerance)
    )
    if not np.any(finite_pixels):
        return sampled
    finite_indices = np.flatnonzero(finite_xy)[finite_pixels]
    sample_x = np.clip(x_pixels[finite_pixels], 0.0, float(width - 1))
    sample_y = np.clip(y_pixels[finite_pixels], 0.0, float(height - 1))
    sampled[finite_indices] = map_coordinates(
        np.asarray(image, dtype=float),
        [sample_y, sample_x],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return sampled


def _lens_model_quantity_on_arcsec_grid(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    caustic_source_redshift: float,
    quantity: str,
) -> np.ndarray:
    z_source = float(caustic_source_redshift)
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    if not hasattr(model, quantity):
        raise ValueError(f"Lens model does not provide quantity {quantity!r}.")
    model_quantity = getattr(model, quantity)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    flat_x = x_arcsec.reshape(-1)
    flat_y = y_arcsec.reshape(-1)
    flat_values = np.full(flat_x.shape, np.nan, dtype=float)
    chunk_size = MODEL_GRID_CHUNK_PIXELS
    for start in range(0, flat_x.size, chunk_size):
        stop = min(start + chunk_size, flat_x.size)
        finite = np.isfinite(flat_x[start:stop]) & np.isfinite(flat_y[start:stop])
        if not np.any(finite):
            continue
        chunk_values = np.full(stop - start, np.nan, dtype=float)
        chunk_values[finite] = np.asarray(
            model_quantity(
                flat_x[start:stop][finite],
                flat_y[start:stop][finite],
                kwargs_lens,
            ),
            dtype=float,
        ).reshape(-1)
        flat_values[start:stop] = chunk_values
    return flat_values.reshape(x_arcsec.shape)


def _empty_truth_grid_draw_indices() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "selection_order",
            "selection_mode",
            "truth_grid_draw_seed",
            "chain_index",
            "draw_index",
            "flat_index",
        ]
    )


def _truth_grid_draw_seed(seed: int | None) -> int:
    if seed is None:
        return DEFAULT_RUNTIME_SEED
    if isinstance(seed, bool):
        raise ValueError("truth-grid draw seed must be a nonnegative integer.")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("truth-grid draw seed must be a nonnegative integer.")
    return seed_int


def _truth_grid_flat_selection(
    samples: np.ndarray,
    max_draws: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    samples_array = np.asarray(samples, dtype=float)
    if samples_array.ndim != 2 or samples_array.shape[0] == 0 or samples_array.shape[1] == 0:
        raise ValueError("Truth-grid recovery requires non-empty finite posterior samples; no best-fit fallback is used.")
    finite_mask = np.isfinite(samples_array).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        raise ValueError("Truth-grid recovery requires finite posterior samples; no best-fit fallback is used.")
    selection_mode = TRUTH_GRID_DRAW_SELECTION_ALL
    selected_indices = finite_indices
    if max_draws is not None:
        max_draws_int = int(max_draws)
        if max_draws_int <= 0:
            raise ValueError("--posterior-truth-recovery-draws must be 'all' or a positive integer.")
        if finite_indices.size > max_draws_int:
            selection_mode = TRUTH_GRID_DRAW_SELECTION_FLAT_RANDOM
            rng = np.random.default_rng(seed)
            selected_indices = np.sort(rng.choice(finite_indices, size=max_draws_int, replace=False))
    rows = [
        {
            "selection_order": int(order),
            "selection_mode": selection_mode,
            "truth_grid_draw_seed": int(seed),
            "chain_index": -1,
            "draw_index": -1,
            "flat_index": int(flat_index),
        }
        for order, flat_index in enumerate(selected_indices)
    ]
    return samples_array[selected_indices], pd.DataFrame(rows, columns=_empty_truth_grid_draw_indices().columns), selection_mode


def _truth_grid_grouped_selection(
    grouped_samples: np.ndarray,
    max_draws: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    grouped = np.asarray(grouped_samples, dtype=float)
    if grouped.ndim != 3 or grouped.shape[0] == 0 or grouped.shape[1] == 0 or grouped.shape[2] == 0:
        raise ValueError("Truth-grid recovery requires non-empty finite grouped posterior samples.")
    finite_mask = np.isfinite(grouped).all(axis=2)
    available_by_chain = [np.flatnonzero(finite_mask[chain_index]) for chain_index in range(grouped.shape[0])]
    available_counts = np.asarray([draws.size for draws in available_by_chain], dtype=int)
    total_finite = int(available_counts.sum())
    if total_finite == 0:
        raise ValueError("Truth-grid recovery requires finite posterior samples; no best-fit fallback is used.")

    if max_draws is None or total_finite <= int(max_draws):
        selection_mode = TRUTH_GRID_DRAW_SELECTION_ALL
        quotas = available_counts
    else:
        max_draws_int = int(max_draws)
        if max_draws_int <= 0:
            raise ValueError("--posterior-truth-recovery-draws must be 'all' or a positive integer.")
        selection_mode = TRUTH_GRID_DRAW_SELECTION_GROUPED_RANDOM
        quotas = np.zeros_like(available_counts)
        for _ in range(max_draws_int):
            candidate_chains = np.flatnonzero(quotas < available_counts)
            if candidate_chains.size == 0:
                break
            min_quota = int(np.min(quotas[candidate_chains]))
            tied = candidate_chains[quotas[candidate_chains] == min_quota]
            quotas[int(tied[0])] += 1

    rng = np.random.default_rng(seed)
    selected_rows: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    selection_order = 0
    n_draws_per_chain = int(grouped.shape[1])
    for chain_index, quota in enumerate(quotas):
        quota_int = int(quota)
        if quota_int <= 0:
            continue
        available_draws = available_by_chain[chain_index]
        if selection_mode == TRUTH_GRID_DRAW_SELECTION_ALL:
            selected_draws = available_draws
        else:
            selected_draws = np.sort(rng.choice(available_draws, size=quota_int, replace=False))
        for draw_index in selected_draws:
            selected_rows.append(np.asarray(grouped[chain_index, int(draw_index)], dtype=float))
            audit_rows.append(
                {
                    "selection_order": int(selection_order),
                    "selection_mode": selection_mode,
                    "truth_grid_draw_seed": int(seed),
                    "chain_index": int(chain_index),
                    "draw_index": int(draw_index),
                    "flat_index": int(chain_index * n_draws_per_chain + int(draw_index)),
                }
            )
            selection_order += 1
    if not selected_rows:
        raise ValueError("Truth-grid recovery requires finite posterior samples; no best-fit fallback is used.")
    return (
        np.asarray(selected_rows, dtype=float),
        pd.DataFrame(audit_rows, columns=_empty_truth_grid_draw_indices().columns),
        selection_mode,
    )


def _select_truth_grid_posterior_samples(
    results: PosteriorResults,
    max_draws: int | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    seed_int = _truth_grid_draw_seed(seed)
    grouped_samples = getattr(results, "grouped_samples", None)
    if grouped_samples is not None:
        grouped = np.asarray(grouped_samples, dtype=float)
        if grouped.ndim == 3 and grouped.shape[0] > 0 and grouped.shape[1] > 0 and grouped.shape[2] > 0:
            return _truth_grid_grouped_selection(grouped, max_draws, seed_int)
    raw_samples = getattr(results, "samples", None)
    if raw_samples is None:
        raise ValueError("Truth-grid recovery requires non-empty finite posterior samples; no best-fit fallback is used.")
    return _truth_grid_flat_selection(np.asarray(raw_samples, dtype=float), max_draws, seed_int)


def _truth_grid_posterior_samples(
    results: PosteriorResults,
    max_draws: int | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    samples, _draw_indices, _selection_mode = _select_truth_grid_posterior_samples(
        results,
        max_draws=max_draws,
        seed=seed,
    )
    return samples


def _truth_grid_median_sample(results: PosteriorResults) -> np.ndarray:
    median_fit = getattr(results, "median_fit", None)
    if median_fit is not None:
        median_array = np.asarray(median_fit, dtype=float).reshape(-1)
        if median_array.size and np.isfinite(median_array).all():
            return median_array
    samples = _truth_grid_posterior_samples(results, max_draws=None)
    median_array = np.nanmedian(samples, axis=0)
    median_array = np.asarray(median_array, dtype=float).reshape(-1)
    if median_array.size == 0 or not np.isfinite(median_array).all():
        raise ValueError("Truth-grid median mode requires finite posterior median parameters; no best-fit fallback is used.")
    return median_array


def _truth_grid_arcsec_coordinates(
    evaluator: ClusterJAXEvaluator,
    truth_wcs: WCS,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    y_pixels, x_pixels = np.indices(image_shape, dtype=float)
    ra_deg, dec_deg = truth_wcs.pixel_to_world_values(x_pixels, y_pixels)
    return _radec_to_solver_arcsec_offsets(ra_deg, dec_deg, evaluator.state.reference)


def _truth_grid_latent_samples(
    evaluator: ClusterJAXEvaluator,
    samples_physical: np.ndarray,
) -> np.ndarray:
    rows = [
        _reported_physical_to_latent_vector(evaluator, np.asarray(sample, dtype=float))
        for sample in np.asarray(samples_physical, dtype=float)
    ]
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _truth_grid_jax_bulk_quantities_for_draw(
    evaluator: ClusterJAXEvaluator,
    latent_sample: np.ndarray,
    z_source: float,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> dict[str, np.ndarray]:
    if not hasattr(evaluator, "_flat_lensing_jacobian_for_components"):
        raise AttributeError("Evaluator does not expose a JAX bulk lensing Jacobian path.")
    if not hasattr(evaluator, "_build_truth_grid_packed_lens_state"):
        raise AttributeError("Evaluator does not expose trace-safe truth-grid packed lens-state construction.")

    latent_jax = jnp.asarray(latent_sample, dtype=jnp.float64)
    x_jax = jnp.asarray(x_arcsec, dtype=jnp.float64)
    y_jax = jnp.asarray(y_arcsec, dtype=jnp.float64)
    z_value = float(z_source)

    packed_state = evaluator._build_truth_grid_packed_lens_state(latent_jax, z_value)
    a00, a01, a10, a11 = evaluator._flat_lensing_jacobian_for_components(
        x_jax,
        y_jax,
        packed_state,
    )
    kappa = 1.0 - 0.5 * (a00 + a11)
    gamma1 = 0.5 * (a11 - a00)
    gamma2 = -0.5 * (a01 + a10)
    det_a = a00 * a11 - a01 * a10
    mu = 1.0 / det_a
    return {
        "kappa": np.asarray(kappa, dtype=np.float64),
        "gamma1": np.asarray(gamma1, dtype=np.float64),
        "gamma2": np.asarray(gamma2, dtype=np.float64),
        "detA": np.asarray(det_a, dtype=np.float64),
        "mu": np.asarray(mu, dtype=np.float64),
        "abs_mu": np.asarray(jnp.abs(mu), dtype=np.float64),
    }


def _write_truth_grid_quantile_fits(
    plot_dir: Path,
    truth_wcs: WCS,
    quantiles: dict[str, dict[str, np.ndarray]],
    *,
    suffixes: Sequence[str] = TRUTH_GRID_QUANTILE_SUFFIXES,
) -> None:
    fits_dir = plot_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    header = truth_wcs.to_header()
    suffix_set = set(str(suffix) for suffix in suffixes)
    for quantity, quantity_quantiles in quantiles.items():
        output_name = TRUTH_GRID_QUANTITY_OUTPUT_NAMES.get(quantity, quantity)
        for stale_suffix in set(TRUTH_GRID_QUANTILE_SUFFIXES) - suffix_set:
            stale_path = fits_dir / f"truth_recovery_{output_name}_model_{stale_suffix}.fits"
            try:
                stale_path.unlink(missing_ok=True)
            except TypeError:  # pragma: no cover - Python <3.8 compatibility guard
                if stale_path.exists():
                    stale_path.unlink()
        for suffix in suffixes:
            if suffix not in quantity_quantiles:
                continue
            image = np.asarray(quantity_quantiles[suffix], dtype=np.float64)
            fits.PrimaryHDU(image, header=header).writeto(
                fits_dir / f"truth_recovery_{output_name}_model_{suffix}.fits",
                overwrite=True,
            )


def _write_truth_grid_summary(
    plot_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    tables_dir = plot_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "truth_recovery_summary.csv"
    new_df = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty and "quantity" in existing:
            replace_quantities = set(new_df["quantity"].astype(str))
            existing = existing[~existing["quantity"].astype(str).isin(replace_quantities)]
            new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def _write_truth_grid_draw_indices(plot_dir: Path, draw_indices: pd.DataFrame) -> None:
    if draw_indices.empty:
        return
    tables_dir = plot_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    draw_indices.to_csv(tables_dir / "truth_recovery_draw_indices.csv", index=False)


def _truth_grid_cache_key(
    source_truth_fits: dict[str, str | Path],
    image_shape: tuple[int, int],
    z_source: float,
    max_draws: int | None,
    truth_grid_mode: str,
    draw_seed: int,
) -> tuple[Any, ...]:
    paths = tuple(sorted((str(key), str(value)) for key, value in source_truth_fits.items()))
    draw_key = None if max_draws is None else int(max_draws)
    return (
        str(truth_grid_mode),
        paths,
        tuple(int(value) for value in image_shape),
        float(z_source),
        draw_key,
        int(draw_seed),
    )


def _posterior_truth_grid_quantiles(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    results: PosteriorResults,
    truth_wcs: WCS,
    image_shape: tuple[int, int],
    z_source: float,
    *,
    source_truth_fits: dict[str, str | Path],
    quantities: Sequence[str],
    max_draws: int | None = None,
    truth_grid_mode: str = TRUTH_GRID_MODE_MEDIAN,
    cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    require_cache: bool = False,
    aperture_center: dict[str, Any] | None = None,
    aperture_kappa_true: np.ndarray | None = None,
    aperture_image_df: pd.DataFrame | None = None,
    aperture_n_radii: int = 40,
    progress: Any | None = None,
    truth_grid_metadata: dict[str, Any] | None = None,
    draw_seed: int | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    requested_quantities = tuple(dict.fromkeys(str(quantity) for quantity in quantities))
    truth_grid_mode = str(truth_grid_mode)
    if truth_grid_mode not in {TRUTH_GRID_MODE_MEDIAN, TRUTH_GRID_MODE_POSTERIOR}:
        raise ValueError(f"Unsupported truth-grid mode: {truth_grid_mode!r}")
    effective_max_draws = max_draws if truth_grid_mode == TRUTH_GRID_MODE_POSTERIOR else 1
    draw_seed_int = _truth_grid_draw_seed(draw_seed)
    cache_key = _truth_grid_cache_key(
        source_truth_fits,
        image_shape,
        z_source,
        effective_max_draws,
        truth_grid_mode,
        draw_seed_int,
    )
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        cached_quantiles = cached.get("quantiles", {})
        if set(requested_quantities).issubset(cached_quantiles):
            aperture_profile = cached.get("aperture_profile")
            cached_aperture_center = cached.get("aperture_center")
            if isinstance(aperture_profile, pd.DataFrame) and isinstance(cached_aperture_center, dict):
                _plot_truth_recovery_m2d_aperture_ratio(
                    plot_dir,
                    aperture_profile,
                    cached_aperture_center,
                    image_radii_arcsec=_truth_recovery_image_aperture_radii(
                        aperture_image_df,
                        cached_aperture_center,
                    ),
                )
            return (
                {quantity: cached_quantiles[quantity] for quantity in requested_quantities},
                np.asarray(cached["x_arcsec"], dtype=np.float64),
                np.asarray(cached["y_arcsec"], dtype=np.float64),
            )
    if require_cache:
        raise RuntimeError(
            "Truth-recovery plot requested posterior grids before the truth_recovery_grids stage populated them."
        )
    draw_indices = _empty_truth_grid_draw_indices()
    draw_selection_mode = "single_median_realization"
    if truth_grid_mode == TRUTH_GRID_MODE_POSTERIOR:
        samples, draw_indices, draw_selection_mode = _select_truth_grid_posterior_samples(
            results,
            max_draws=max_draws,
            seed=draw_seed_int,
        )
    else:
        samples = _truth_grid_median_sample(results)[None, :]
    if not hasattr(evaluator, "_build_truth_grid_packed_lens_state") or not hasattr(evaluator, "_flat_lensing_jacobian_for_components"):
        raise RuntimeError(
            "Truth-grid recovery requires the JAX bulk lensing Jacobian backend "
            "(_build_truth_grid_packed_lens_state and _flat_lensing_jacobian_for_components)."
        )
    latent_samples = _truth_grid_latent_samples(evaluator, samples)
    if latent_samples.shape[0] != samples.shape[0]:
        raise RuntimeError("Truth-grid recovery failed to convert posterior samples to latent JAX parameters.")
    backend_used = "jax_bulk_hessian"
    truth_grid_metadata = dict(truth_grid_metadata or {})
    x_arcsec, y_arcsec = _truth_grid_arcsec_coordinates(evaluator, truth_wcs, image_shape)
    flat_x = np.asarray(x_arcsec, dtype=np.float64).reshape(-1)
    flat_y = np.asarray(y_arcsec, dtype=np.float64).reshape(-1)
    total_pixels = int(flat_x.size)
    if aperture_center is None and aperture_kappa_true is not None and "kappa" in requested_quantities:
        aperture_center = _smoothed_truth_kappa_peak_aperture_center(
            aperture_kappa_true,
            x_arcsec,
            y_arcsec,
        )
        if aperture_center is None:
            _log(None, "[truth-recovery:m2d] skipped: no finite smoothed truth-kappa peak center")
    chunk_size = total_pixels
    chunk_count = 1
    estimated_grid_buffer_bytes = int(
        samples.shape[0]
        * total_pixels
        * len(requested_quantities)
        * np.dtype(np.float64).itemsize
    )
    _log(
        None,
        (
            "[truth-grid] posterior map grids "
            f"mode={truth_grid_mode} backend={backend_used} "
            f"draws={samples.shape[0]} quantities={len(requested_quantities)} "
            f"pixels={total_pixels} chunk_pixels={chunk_size} chunks={chunk_count} "
            f"dtype=float64 estimated_grid_buffer_bytes={estimated_grid_buffer_bytes}"
        ),
    )
    z_source = float(z_source)

    quantiles: dict[str, dict[str, np.ndarray]] = {
        quantity: {
            suffix: np.full(total_pixels, np.nan, dtype=np.float64)
            for suffix in (("median",) if truth_grid_mode == TRUTH_GRID_MODE_MEDIAN else TRUTH_GRID_QUANTILE_SUFFIXES)
        }
        for quantity in requested_quantities
    }
    aperture_profile_draw_sums: np.ndarray | None = None
    aperture_truth_sums: np.ndarray | None = None
    aperture_pixel_counts: np.ndarray | None = None
    aperture_radii: np.ndarray | None = None
    flat_aperture_radius: np.ndarray | None = None
    flat_aperture_valid: np.ndarray | None = None
    if aperture_center is not None and aperture_kappa_true is not None and "kappa" in requested_quantities:
        true_values = np.asarray(aperture_kappa_true, dtype=np.float64).reshape(-1)
        if true_values.size == total_pixels:
            center_x = float(aperture_center["center_x_arcsec"])
            center_y = float(aperture_center["center_y_arcsec"])
            flat_aperture_radius = np.hypot(flat_x - center_x, flat_y - center_y)
            flat_aperture_valid = np.isfinite(true_values) & np.isfinite(flat_aperture_radius)
            finite_radius = flat_aperture_radius[flat_aperture_valid]
            finite_radius = finite_radius[np.isfinite(finite_radius)]
            if finite_radius.size:
                pixel_scale = _truth_recovery_pixel_scale_arcsec(x_arcsec, y_arcsec)
                r_min = float(pixel_scale)
                r_max = float(np.nanmax(finite_radius))
                if np.isfinite(r_max) and r_max > 0.0:
                    if r_max < r_min:
                        r_min = r_max
                    aperture_radii = np.linspace(r_min, r_max, int(aperture_n_radii), dtype=np.float64)
                    aperture_profile_draw_sums = np.zeros((samples.shape[0], aperture_radii.size), dtype=np.float64)
                    aperture_truth_sums = np.full(aperture_radii.size, np.nan, dtype=np.float64)
                    aperture_pixel_counts = np.zeros(aperture_radii.size, dtype=int)
                    for radius_index, radius_arcsec in enumerate(aperture_radii):
                        in_aperture = flat_aperture_valid & (flat_aperture_radius <= float(radius_arcsec))
                        aperture_pixel_counts[radius_index] = int(np.count_nonzero(in_aperture))
                        if aperture_pixel_counts[radius_index] > 0:
                            aperture_truth_sums[radius_index] = float(np.nansum(true_values[in_aperture]))

    finite = np.isfinite(flat_x) & np.isfinite(flat_y)
    finite_indices = np.flatnonzero(finite)
    full_grid_values = {
        quantity: np.full((samples.shape[0], total_pixels), np.nan, dtype=np.float64)
        for quantity in requested_quantities
    }
    draw_task_id = None
    if progress is not None:
        draw_task_id = progress.add_task("truth_recovery_grids: posterior draws", total=int(samples.shape[0]))
    if finite_indices.size:
        finite_x = flat_x[finite]
        finite_y = flat_y[finite]
        for draw_index, latent_sample in enumerate(latent_samples):
            draw_quantities = _truth_grid_jax_bulk_quantities_for_draw(
                evaluator,
                latent_sample,
                z_source,
                finite_x,
                finite_y,
            )
            for quantity in requested_quantities:
                full_grid_values[quantity][draw_index, finite_indices] = np.asarray(
                    draw_quantities[quantity],
                    dtype=np.float64,
                )
            if draw_task_id is not None and progress is not None:
                progress.advance(draw_task_id)
    elif draw_task_id is not None and progress is not None:
        progress.advance(draw_task_id, advance=int(samples.shape[0]))
    if draw_task_id is not None and progress is not None:
        progress.update(draw_task_id, description="truth_recovery_grids: posterior draws complete")

    if (
        aperture_profile_draw_sums is not None
        and aperture_radii is not None
        and flat_aperture_radius is not None
        and flat_aperture_valid is not None
        and "kappa" in full_grid_values
    ):
        kappa_draws = full_grid_values["kappa"]
        for draw_index in range(samples.shape[0]):
            draw_kappa = np.asarray(kappa_draws[draw_index], dtype=np.float64)
            draw_valid = flat_aperture_valid & np.isfinite(draw_kappa)
            if not np.any(draw_valid):
                continue
            for radius_index, radius_arcsec in enumerate(aperture_radii):
                in_aperture = draw_valid & (flat_aperture_radius <= float(radius_arcsec))
                if np.any(in_aperture):
                    aperture_profile_draw_sums[draw_index, radius_index] = float(np.nansum(draw_kappa[in_aperture]))

    for quantity, values in full_grid_values.items():
        if truth_grid_mode == TRUTH_GRID_MODE_MEDIAN:
            quantiles[quantity]["median"] = np.asarray(values[0], dtype=np.float64)
        else:
            q16, q50, q84 = np.nanpercentile(
                values,
                TRUTH_GRID_QUANTILE_PERCENTILES,
                axis=0,
            )
            quantiles[quantity]["q16"] = np.asarray(q16, dtype=np.float64)
            quantiles[quantity]["median"] = np.asarray(q50, dtype=np.float64)
            quantiles[quantity]["q84"] = np.asarray(q84, dtype=np.float64)

    shaped_quantiles = {
        quantity: {
            suffix: values.reshape(image_shape)
            for suffix, values in quantity_quantiles.items()
        }
        for quantity, quantity_quantiles in quantiles.items()
    }
    output_suffixes = ("median",) if truth_grid_mode == TRUTH_GRID_MODE_MEDIAN else TRUTH_GRID_QUANTILE_SUFFIXES
    _write_truth_grid_quantile_fits(plot_dir, truth_wcs, shaped_quantiles, suffixes=output_suffixes)
    summary_rows: list[dict[str, Any]] = []
    for quantity, quantity_quantiles in shaped_quantiles.items():
        source_path = source_truth_fits.get(quantity) or source_truth_fits.get("kappa")
        finite_q16_count = (
            int(np.isfinite(quantity_quantiles["q16"]).sum())
            if "q16" in quantity_quantiles
            else 0
        )
        finite_q84_count = (
            int(np.isfinite(quantity_quantiles["q84"]).sum())
            if "q84" in quantity_quantiles
            else 0
        )
        summary_rows.append(
            {
                "quantity": quantity,
                "truth_grid_mode": truth_grid_mode,
                "truth_grid_backend": backend_used,
                "truth_grid_draw_seed": int(draw_seed_int),
                "truth_grid_draw_selection": draw_selection_mode,
                "spread_available": bool(truth_grid_mode == TRUTH_GRID_MODE_POSTERIOR),
                "draw_count_used": int(samples.shape[0]),
                "total_pixels": int(total_pixels),
                "finite_q16_pixel_count": finite_q16_count,
                "finite_median_pixel_count": int(np.isfinite(quantity_quantiles["median"]).sum()),
                "finite_q84_pixel_count": finite_q84_count,
                "chunk_pixels": int(chunk_size),
                "chunk_count": int(chunk_count),
                "dtype": "float64",
                "estimated_grid_buffer_memory_bytes": int(estimated_grid_buffer_bytes),
                "estimated_grid_buffer_memory_gb": float(estimated_grid_buffer_bytes) / float(1024**3),
                "native_truth_height": int(truth_grid_metadata.get("native_truth_height", image_shape[0])),
                "native_truth_width": int(truth_grid_metadata.get("native_truth_width", image_shape[1])),
                "diagnostic_grid_height": int(truth_grid_metadata.get("diagnostic_grid_height", image_shape[0])),
                "diagnostic_grid_width": int(truth_grid_metadata.get("diagnostic_grid_width", image_shape[1])),
                "truth_grid_size": int(truth_grid_metadata.get("truth_grid_size", image_shape[0])),
                "truth_grid_sampling": str(truth_grid_metadata.get("truth_grid_sampling", "native")),
                "native_to_diagnostic_pixel_ratio_x": float(
                    truth_grid_metadata.get("native_to_diagnostic_pixel_ratio_x", 1.0)
                ),
                "native_to_diagnostic_pixel_ratio_y": float(
                    truth_grid_metadata.get("native_to_diagnostic_pixel_ratio_y", 1.0)
                ),
                "source_truth_fits": "" if source_path is None else str(source_path),
                "source_redshift": float(z_source),
            }
        )
    _write_truth_grid_summary(plot_dir, summary_rows)
    if truth_grid_mode == TRUTH_GRID_MODE_POSTERIOR:
        _write_truth_grid_draw_indices(plot_dir, draw_indices)
    aperture_profile_df: pd.DataFrame | None = None
    if (
        aperture_profile_draw_sums is not None
        and aperture_truth_sums is not None
        and aperture_pixel_counts is not None
        and aperture_radii is not None
        and aperture_center is not None
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            draw_ratios = aperture_profile_draw_sums / aperture_truth_sums[None, :]
        if truth_grid_mode == TRUTH_GRID_MODE_POSTERIOR:
            ratio_q16, ratio_median, ratio_q84 = np.nanpercentile(draw_ratios, TRUTH_GRID_QUANTILE_PERCENTILES, axis=0)
            model_sum_q16, model_sum_median, model_sum_q84 = np.nanpercentile(
                aperture_profile_draw_sums,
                TRUTH_GRID_QUANTILE_PERCENTILES,
                axis=0,
            )
        else:
            ratio_median = np.asarray(draw_ratios[0], dtype=np.float64)
            ratio_q16 = np.full_like(ratio_median, np.nan, dtype=np.float64)
            ratio_q84 = np.full_like(ratio_median, np.nan, dtype=np.float64)
            model_sum_median = np.asarray(aperture_profile_draw_sums[0], dtype=np.float64)
            model_sum_q16 = np.full_like(model_sum_median, np.nan, dtype=np.float64)
            model_sum_q84 = np.full_like(model_sum_median, np.nan, dtype=np.float64)
        rows = []
        for radius_index, radius_arcsec in enumerate(aperture_radii):
            rows.append(
                {
                    "radius_arcsec": float(radius_arcsec),
                    "pixel_count": int(aperture_pixel_counts[radius_index]),
                    "kappa_true_sum": float(aperture_truth_sums[radius_index]),
                    "kappa_model_sum": float(model_sum_median[radius_index]),
                    "kappa_model_sum_q16": float(model_sum_q16[radius_index]),
                    "kappa_model_sum_q84": float(model_sum_q84[radius_index]),
                    "m2d_ratio": float(ratio_median[radius_index]),
                    "m2d_ratio_q16": float(ratio_q16[radius_index]),
                    "m2d_ratio_median": float(ratio_median[radius_index]),
                    "m2d_ratio_q84": float(ratio_q84[radius_index]),
                    "center_mode": str(aperture_center["center_mode"]),
                    "center_x_arcsec": float(aperture_center["center_x_arcsec"]),
                    "center_y_arcsec": float(aperture_center["center_y_arcsec"]),
                    "center_catalog_id": str(aperture_center["center_catalog_id"]),
                    "center_catalog_mag": float(aperture_center["center_catalog_mag"]),
                    "center_smoothing_sigma_pix": _finite_float_or_nan(
                        aperture_center.get("center_smoothing_sigma_pix")
                    ),
                    "center_smoothed_kappa_peak": _finite_float_or_nan(
                        aperture_center.get("center_smoothed_kappa_peak")
                    ),
                }
            )
        aperture_profile_df = pd.DataFrame(rows)
        tables_dir = plot_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        aperture_profile_df.to_csv(tables_dir / "truth_recovery_m2d_aperture_profile.csv", index=False)
        _plot_truth_recovery_m2d_aperture_ratio(
            plot_dir,
            aperture_profile_df,
            aperture_center,
            image_radii_arcsec=_truth_recovery_image_aperture_radii(aperture_image_df, aperture_center),
        )
    if cache is not None:
        cache[cache_key] = {
            "quantiles": shaped_quantiles,
            "x_arcsec": x_arcsec,
            "y_arcsec": y_arcsec,
            "aperture_profile": aperture_profile_df,
            "aperture_center": aperture_center,
        }
    return shaped_quantiles, x_arcsec, y_arcsec


def _empty_recovery_image_points() -> pd.DataFrame:
    return pd.DataFrame(columns=RECOVERY_IMAGE_POINT_COLUMNS)


def _observed_image_recovery_points(
    image_df: pd.DataFrame | None,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    truth_grid: np.ndarray,
    truth_wcs: WCS,
    caustic_source_redshift: float,
    model_quantity: str,
    *,
    model_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    if image_df is None or image_df.empty or "x_obs_arcsec" not in image_df or "y_obs_arcsec" not in image_df:
        return _empty_recovery_image_points()
    x_obs = pd.to_numeric(image_df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(image_df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    finite_xy = np.isfinite(x_obs) & np.isfinite(y_obs)
    true_values = np.full(x_obs.shape, np.nan, dtype=float)
    model_values = np.full(x_obs.shape, np.nan, dtype=float)
    if np.any(finite_xy):
        true_values[finite_xy] = _sample_wcs_image_at_solver_arcsec(
            truth_grid,
            truth_wcs,
            x_obs[finite_xy],
            y_obs[finite_xy],
            evaluator.state.reference,
        )
        model_sample = _lens_model_quantity_on_arcsec_grid(
            evaluator,
            best_fit,
            x_obs[finite_xy],
            y_obs[finite_xy],
            caustic_source_redshift,
            model_quantity,
        )
        model_values[finite_xy] = np.asarray(model_sample, dtype=float).reshape(-1)
    if model_transform is not None:
        model_values = np.asarray(model_transform(model_values), dtype=float)
    points = _empty_recovery_image_points()
    for column in ("family_id", "image_label"):
        if column in image_df:
            points[column] = image_df[column].astype(str).to_numpy()
    points["x_obs_arcsec"] = x_obs
    points["y_obs_arcsec"] = y_obs
    points["true_value"] = true_values
    points["model_value"] = model_values
    return points


def _observed_image_recovery_points_from_grids(
    image_df: pd.DataFrame | None,
    evaluator: ClusterJAXEvaluator,
    truth_grid: np.ndarray,
    model_grid: np.ndarray,
    truth_wcs: WCS,
) -> pd.DataFrame:
    if image_df is None or image_df.empty or "x_obs_arcsec" not in image_df or "y_obs_arcsec" not in image_df:
        return _empty_recovery_image_points()
    x_obs = pd.to_numeric(image_df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(image_df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    finite_xy = np.isfinite(x_obs) & np.isfinite(y_obs)
    true_values = np.full(x_obs.shape, np.nan, dtype=float)
    model_values = np.full(x_obs.shape, np.nan, dtype=float)
    if np.any(finite_xy):
        true_values[finite_xy] = _sample_wcs_image_at_solver_arcsec(
            truth_grid,
            truth_wcs,
            x_obs[finite_xy],
            y_obs[finite_xy],
            evaluator.state.reference,
        )
        model_values[finite_xy] = _sample_wcs_image_at_solver_arcsec(
            model_grid,
            truth_wcs,
            x_obs[finite_xy],
            y_obs[finite_xy],
            evaluator.state.reference,
        )
    points = _empty_recovery_image_points()
    for column in ("family_id", "image_label"):
        if column in image_df:
            points[column] = image_df[column].astype(str).to_numpy()
    points["x_obs_arcsec"] = x_obs
    points["y_obs_arcsec"] = y_obs
    points["true_value"] = true_values
    points["model_value"] = model_values
    return points


def _truth_recovery_member_overlay_table(evaluator: ClusterJAXEvaluator) -> pd.DataFrame:
    records = getattr(getattr(evaluator, "state", None), "scaling_component_records", []) or []
    rows: list[dict[str, Any]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        catalog_id = str(raw_record.get("catalog_id", "")).strip()
        x_centre = _finite_float_or_nan(raw_record.get("x_centre"))
        y_centre = _finite_float_or_nan(raw_record.get("y_centre"))
        if not catalog_id or not (np.isfinite(x_centre) and np.isfinite(y_centre)):
            continue
        free_component_index_value = _finite_float_or_nan(raw_record.get("free_component_index", -1))
        free_component_index = int(free_component_index_value) if np.isfinite(free_component_index_value) else -1
        rows.append(
            {
                "catalog_id": catalog_id,
                "x_arcsec": float(x_centre),
                "y_arcsec": float(y_centre),
                "free": bool(raw_record.get("selected_independent", False)) or free_component_index >= 0,
            }
        )
    return pd.DataFrame(rows, columns=["catalog_id", "x_arcsec", "y_arcsec", "free"])


def _truth_recovery_image_overlay_table(image_df: pd.DataFrame | None) -> pd.DataFrame:
    if image_df is None or image_df.empty or "x_obs_arcsec" not in image_df or "y_obs_arcsec" not in image_df:
        return pd.DataFrame(columns=["x_arcsec", "y_arcsec"])
    x_obs = pd.to_numeric(image_df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(image_df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_obs) & np.isfinite(y_obs)
    return pd.DataFrame(
        {
            "x_arcsec": x_obs[finite],
            "y_arcsec": y_obs[finite],
        },
        columns=["x_arcsec", "y_arcsec"],
    )


def _model_quantity_grid_for_wcs_pixels(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    image_wcs: WCS,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    caustic_source_redshift: float,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra_deg, dec_deg = image_wcs.pixel_to_world_values(x_pixels, y_pixels)
    x_arcsec, y_arcsec = _radec_to_solver_arcsec_offsets(ra_deg, dec_deg, evaluator.state.reference)
    values = _lens_model_quantity_on_arcsec_grid(
        evaluator,
        best_fit,
        x_arcsec,
        y_arcsec,
        caustic_source_redshift,
        quantity,
    )
    return values, x_arcsec, y_arcsec


def _kappa_model_grid_for_true_fits(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_shape: tuple[int, int],
    kappa_wcs: WCS,
    caustic_source_redshift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pixels, x_pixels = np.indices(kappa_true_shape, dtype=float)
    return _model_quantity_grid_for_wcs_pixels(
        evaluator,
        best_fit,
        kappa_wcs,
        x_pixels,
        y_pixels,
        caustic_source_redshift,
        "kappa",
    )


def _quantity_recovery_bin_table(
    true_values: np.ndarray,
    model_values: np.ndarray,
    quantity: str,
    n_bins: int = MAP_RECOVERY_STAT_BINS,
    limits: tuple[float, float] | None = None,
) -> pd.DataFrame:
    columns = [
        "bin_index",
        f"{quantity}_true_min",
        f"{quantity}_true_max",
        f"{quantity}_true_center",
        "sample_count",
        f"{quantity}_model_q16",
        f"{quantity}_model_median",
        f"{quantity}_model_q84",
    ]
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    model_values = np.asarray(model_values, dtype=float).reshape(-1)
    if true_values.size == 0 or model_values.size == 0:
        return pd.DataFrame(columns=columns)
    if limits is None:
        lower = float(np.nanmin(true_values))
        upper = float(np.nanmax(true_values))
    else:
        lower, upper = (float(value) for value in limits)
        in_limits = (true_values >= lower) & (true_values <= upper)
        true_values = true_values[in_limits]
        model_values = model_values[in_limits]
        if true_values.size == 0 or model_values.size == 0:
            return pd.DataFrame(columns=columns)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return pd.DataFrame(columns=columns)
    if lower == upper:
        span = max(abs(lower), 1.0) * 0.05
        lower -= span
        upper += span
    edges = np.linspace(lower, upper, int(n_bins) + 1)
    bin_indices = np.searchsorted(edges, true_values, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, int(n_bins) - 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(int(n_bins)):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        q16, q50, q84 = np.nanpercentile(
            model_values[in_bin],
            [RECOVERY_ONE_SIGMA_PERCENTILES[0], 50.0, RECOVERY_ONE_SIGMA_PERCENTILES[1]],
        )
        rows.append(
            {
                "bin_index": int(bin_index),
                f"{quantity}_true_min": float(edges[bin_index]),
                f"{quantity}_true_max": float(edges[bin_index + 1]),
                f"{quantity}_true_center": float(0.5 * (edges[bin_index] + edges[bin_index + 1])),
                "sample_count": int(np.sum(in_bin)),
                f"{quantity}_model_q16": float(q16),
                f"{quantity}_model_median": float(q50),
                f"{quantity}_model_q84": float(q84),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _nanpercentile_or_nan(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanpercentile(finite, percentile))


def _central_interval_half_width(lower: float, upper: float) -> float:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return float("nan")
    return 0.5 * abs(float(upper) - float(lower))


def _quantity_recovery_residual_statistics(true_values: np.ndarray, model_values: np.ndarray) -> dict[str, float]:
    true_array = np.asarray(true_values, dtype=float).reshape(-1)
    model_array = np.asarray(model_values, dtype=float).reshape(-1)
    finite = np.isfinite(true_array) & np.isfinite(model_array)
    if not np.any(finite):
        return {
            "bias_median": float("nan"),
            "spread_nmad": float("nan"),
            "rmse": float("nan"),
        }
    residual = model_array[finite] - true_array[finite]
    bias_median = float(np.nanmedian(residual))
    spread_nmad = float(1.4826 * np.nanmedian(np.abs(residual - bias_median)))
    rmse = float(np.sqrt(np.nanmean(np.square(residual))))
    return {
        "bias_median": bias_median,
        "spread_nmad": spread_nmad,
        "rmse": rmse,
    }


def _quantity_recovery_limits(true_values: np.ndarray, model_values: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([np.asarray(true_values, dtype=float).reshape(-1), np.asarray(model_values, dtype=float).reshape(-1)])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.nanmin(finite))
    upper = float(np.nanmax(finite))
    if lower == upper:
        span = max(abs(lower), 1.0) * 0.05
        return lower - span, upper + span
    span = upper - lower
    padded_lower = lower - 0.04 * span
    padded_upper = upper + 0.04 * span
    if lower >= 0.0:
        padded_lower = max(0.0, padded_lower)
    return float(padded_lower), float(padded_upper)


def _quantity_recovery_summary_table(
    quantity: str,
    true_values: np.ndarray,
    model_values: np.ndarray,
    total_pixel_count: int,
) -> pd.DataFrame:
    residual = model_values - true_values
    fractional_residual = np.full(true_values.shape, np.nan, dtype=float)
    nonzero = true_values != 0.0
    fractional_residual[nonzero] = residual[nonzero] / true_values[nonzero]
    residual_q2p5 = _nanpercentile_or_nan(residual, RECOVERY_TWO_SIGMA_PERCENTILES[0])
    residual_q16 = _nanpercentile_or_nan(residual, RECOVERY_ONE_SIGMA_PERCENTILES[0])
    residual_median = _nanpercentile_or_nan(residual, 50.0)
    residual_q84 = _nanpercentile_or_nan(residual, RECOVERY_ONE_SIGMA_PERCENTILES[1])
    residual_q97p5 = _nanpercentile_or_nan(residual, RECOVERY_TWO_SIGMA_PERCENTILES[1])
    fractional_residual_q2p5 = _nanpercentile_or_nan(
        fractional_residual,
        RECOVERY_TWO_SIGMA_PERCENTILES[0],
    )
    fractional_residual_q16 = _nanpercentile_or_nan(
        fractional_residual,
        RECOVERY_ONE_SIGMA_PERCENTILES[0],
    )
    fractional_residual_median = _nanpercentile_or_nan(fractional_residual, 50.0)
    fractional_residual_q84 = _nanpercentile_or_nan(
        fractional_residual,
        RECOVERY_ONE_SIGMA_PERCENTILES[1],
    )
    fractional_residual_q97p5 = _nanpercentile_or_nan(
        fractional_residual,
        RECOVERY_TWO_SIGMA_PERCENTILES[1],
    )
    residual_stats = _quantity_recovery_residual_statistics(true_values, model_values)
    row = {
        "quantity": quantity,
        "total_pixel_count": int(total_pixel_count),
        "finite_pixel_count": int(true_values.size),
        f"{quantity}_true_min": float(np.nanmin(true_values)) if true_values.size else float("nan"),
        f"{quantity}_true_max": float(np.nanmax(true_values)) if true_values.size else float("nan"),
        f"{quantity}_model_min": float(np.nanmin(model_values)) if model_values.size else float("nan"),
        f"{quantity}_model_max": float(np.nanmax(model_values)) if model_values.size else float("nan"),
        f"{quantity}_residual_q2p5": residual_q2p5,
        f"{quantity}_residual_q16": residual_q16,
        f"{quantity}_residual_median": residual_median,
        f"{quantity}_residual_q84": residual_q84,
        f"{quantity}_residual_q97p5": residual_q97p5,
        f"{quantity}_residual_sigma": _central_interval_half_width(residual_q16, residual_q84),
        f"{quantity}_bias_median": residual_stats["bias_median"],
        f"{quantity}_spread_nmad": residual_stats["spread_nmad"],
        f"{quantity}_rmse": residual_stats["rmse"],
        f"{quantity}_fractional_residual_q2p5": fractional_residual_q2p5,
        f"{quantity}_fractional_residual_q16": fractional_residual_q16,
        f"{quantity}_fractional_residual_median": fractional_residual_median,
        f"{quantity}_fractional_residual_q84": fractional_residual_q84,
        f"{quantity}_fractional_residual_q97p5": fractional_residual_q97p5,
        f"{quantity}_fractional_residual_sigma": _central_interval_half_width(
            fractional_residual_q16,
            fractional_residual_q84,
        ),
    }
    return pd.DataFrame([row])


def _format_recovery_metric(value: Any) -> str:
    try:
        float_value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(float_value):
        return "nan"
    return f"{float_value:.3g}"


def _quantity_recovery_reduced(
    true_grid: np.ndarray,
    model_grid: np.ndarray,
    quantity: str,
    *,
    histogram_bins: int = MAP_RECOVERY_HISTOGRAM_BINS,
    stat_bins: int = MAP_RECOVERY_STAT_BINS,
    limits: tuple[float, float] | None = None,
    stat_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    true_flat = np.asarray(true_grid, dtype=float).reshape(-1)
    model_flat = np.asarray(model_grid, dtype=float).reshape(-1)
    finite = np.isfinite(true_flat) & np.isfinite(model_flat)
    true_values = true_flat[finite]
    model_values = model_flat[finite]
    total_pixel_count = int(true_flat.size)
    if true_values.size == 0:
        resolved_limits = tuple(float(value) for value in limits) if limits is not None else (0.0, 1.0)
        hist_counts = np.zeros((int(histogram_bins), int(histogram_bins)), dtype=float)
        hist_x_edges = np.linspace(resolved_limits[0], resolved_limits[1], int(histogram_bins) + 1)
        hist_y_edges = hist_x_edges.copy()
    else:
        resolved_limits = tuple(float(value) for value in limits) if limits is not None else _quantity_recovery_limits(true_values, model_values)
        hist_counts, hist_x_edges, hist_y_edges = np.histogram2d(
            true_values,
            model_values,
            bins=int(histogram_bins),
            range=[resolved_limits, resolved_limits],
        )
    return {
        "quantity": quantity,
        "limits": resolved_limits,
        "hist_counts": hist_counts,
        "hist_x_edges": hist_x_edges,
        "hist_y_edges": hist_y_edges,
        "bin_table": _quantity_recovery_bin_table(true_values, model_values, quantity, n_bins=stat_bins, limits=stat_limits),
        "summary_table": _quantity_recovery_summary_table(quantity, true_values, model_values, total_pixel_count),
        "finite_pixel_count": int(true_values.size),
        "total_pixel_count": total_pixel_count,
    }


def _plot_quantity_recovery(
    recovery: dict[str, Any],
    output_path: Path,
    *,
    quantity: str,
    true_label: str,
    model_label: str,
    title: str | None = None,
    image_points: pd.DataFrame | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 5.7))
    if int(recovery.get("finite_pixel_count", 0)) == 0:
        ax.text(0.5, 0.5, "No finite recovery samples.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        limits = tuple(float(value) for value in recovery["limits"])
        hist_counts = np.asarray(recovery["hist_counts"], dtype=float)
        hist_x_edges = np.asarray(recovery["hist_x_edges"], dtype=float)
        hist_y_edges = np.asarray(recovery["hist_y_edges"], dtype=float)
        masked_counts = np.ma.masked_less_equal(hist_counts.T, 0.0)
        mesh = ax.pcolormesh(
            hist_x_edges,
            hist_y_edges,
            masked_counts,
            cmap="Blues",
            norm=LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(hist_counts)))),
            shading="auto",
        )
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("pixel count")
        guide_x = np.linspace(limits[0], limits[1], 256)
        ax.plot(guide_x, guide_x, color="0.15", linewidth=1.0, linestyle="-", label="1:1")
        summary_df = recovery["summary_table"]
        if not summary_df.empty:
            summary = summary_df.iloc[0]
            metric_text = "\n".join(
                [
                    f"bias: {_format_recovery_metric(summary.get(f'{quantity}_bias_median', np.nan))}",
                    f"NMAD: {_format_recovery_metric(summary.get(f'{quantity}_spread_nmad', np.nan))}",
                    f"RMSE: {_format_recovery_metric(summary.get(f'{quantity}_rmse', np.nan))}",
                    f"pixels: {int(summary.get('finite_pixel_count', 0))}",
                ]
            )
            ax.text(
                0.98,
                0.04,
                metric_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
            )
            one_sigma = float(summary.get(f"{quantity}_fractional_residual_sigma", np.nan))
            if np.isfinite(one_sigma):
                for multiple, linestyle, label in [
                    (1.0, "--", r"1$\sigma$ recovery"),
                    (2.0, ":", r"2$\sigma$ recovery"),
                ]:
                    band_width = multiple * one_sigma
                    upper_slope = 1.0 + band_width
                    lower_slope = 1.0 / upper_slope
                    ax.plot(
                        guide_x,
                        lower_slope * guide_x,
                        color="0.45",
                        linewidth=0.8,
                        linestyle=linestyle,
                    )
                    ax.plot(
                        guide_x,
                        upper_slope * guide_x,
                        color="0.45",
                        linewidth=0.8,
                        linestyle=linestyle,
                        label=label,
                    )
        bin_df = recovery["bin_table"]
        if not bin_df.empty:
            center = bin_df[f"{quantity}_true_center"].to_numpy(dtype=float)
            median = bin_df[f"{quantity}_model_median"].to_numpy(dtype=float)
            q16 = bin_df[f"{quantity}_model_q16"].to_numpy(dtype=float)
            q84 = bin_df[f"{quantity}_model_q84"].to_numpy(dtype=float)
            ax.plot(center, median, color="black", linewidth=1.8, label="median")
            ax.plot(center, q16, color="tab:blue", linewidth=1.2, label=r"1$\sigma$ (16th/84th)")
            ax.plot(center, q84, color="tab:blue", linewidth=1.2)
        if image_points is not None and not image_points.empty and {"true_value", "model_value"}.issubset(image_points.columns):
            point_true = pd.to_numeric(image_points["true_value"], errors="coerce").to_numpy(dtype=float)
            point_model = pd.to_numeric(image_points["model_value"], errors="coerce").to_numpy(dtype=float)
            finite_points = np.isfinite(point_true) & np.isfinite(point_model)
            in_limits = (
                finite_points
                & (point_true >= limits[0])
                & (point_true <= limits[1])
                & (point_model >= limits[0])
                & (point_model <= limits[1])
            )
            if np.any(in_limits):
                ax.scatter(
                    point_true[in_limits],
                    point_model[in_limits],
                    marker="o",
                    s=22,
                    facecolors="tab:orange",
                    edgecolors="black",
                    linewidths=0.45,
                    alpha=0.9,
                    label="observed images",
                    zorder=5,
                )
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(true_label)
        ax.set_ylabel(model_label)
        if title:
            ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    _finish_figure(fig, output_path, dpi=220, bbox_inches="tight")


def _write_kappa_recovery_from_grid(
    plot_dir: Path,
    kappa_true: np.ndarray,
    model_kappa: np.ndarray,
    z_source: float,
    *,
    image_points: pd.DataFrame | None = None,
) -> None:
    recovery = _quantity_recovery_reduced(kappa_true, model_kappa, "kappa", limits=KAPPA_RECOVERY_LIMITS)
    tables_dir = plot_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    recovery["bin_table"].to_csv(tables_dir / "truth_recovery_kappa_recovery_binned.csv", index=False)
    recovery["summary_table"].to_csv(tables_dir / "truth_recovery_kappa_recovery_summary.csv", index=False)
    _plot_quantity_recovery(
        recovery,
        _plot_path(plot_dir, "truth_recovery_kappa_recovery.pdf"),
        quantity="kappa",
        true_label=r"$\kappa_{\rm true}$",
        model_label=r"$\kappa_{\rm model}$",
        image_points=image_points,
    )


def _plot_truth_recovery_spatial_map(
    ax: Any,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    image_data: np.ndarray,
    *,
    cmap: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
) -> Any:
    mesh_kwargs: dict[str, Any] = {
        "cmap": cmap,
        "shading": "nearest",
        "edgecolors": "none",
        "linewidth": 0.0,
        "antialiased": False,
        "rasterized": True,
    }
    if norm is None:
        if vmin is not None:
            mesh_kwargs["vmin"] = float(vmin)
        if vmax is not None:
            mesh_kwargs["vmax"] = float(vmax)
    else:
        mesh_kwargs["norm"] = norm
    mesh = ax.pcolormesh(
        np.asarray(x_arcsec, dtype=float),
        np.asarray(y_arcsec, dtype=float),
        np.ma.masked_invalid(image_data),
        **mesh_kwargs,
    )
    ax.set_aspect("equal", adjustable="box")
    return mesh


def _shifted_diverging_colormap(cmap_name: str, zero_position: float, *, samples: int = 256) -> LinearSegmentedColormap:
    zero = float(zero_position)
    if not np.isfinite(zero) or zero <= 0.0 or zero >= 1.0:
        raise ValueError("zero_position must be finite and strictly between 0 and 1")
    sample_count = max(int(samples), 3)
    base_cmap = plt.get_cmap(cmap_name)
    left_count = max(2, int(round(sample_count * zero)) + 1)
    right_count = max(2, sample_count - left_count + 1)
    left_x = np.linspace(0.0, zero, left_count, dtype=float)
    right_x = np.linspace(zero, 1.0, right_count, dtype=float)[1:]
    left_colors = base_cmap(np.linspace(0.0, 0.5, left_count, dtype=float))
    right_colors = base_cmap(np.linspace(0.5, 1.0, right_count, dtype=float))[1:]
    color_points = [(float(x), color) for x, color in zip(left_x, left_colors, strict=True)]
    color_points.extend((float(x), color) for x, color in zip(right_x, right_colors, strict=True))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_zero_at_{zero:g}", color_points, N=sample_count)


def _plot_kappa_recovery(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_fits: str | Path,
    caustic_source_redshift: float,
    image_df: pd.DataFrame | None = None,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:kappa_recovery] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    kappa_true, kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    model_kappa, _x_arcsec, _y_arcsec = _kappa_model_grid_for_true_fits(
        evaluator,
        best_fit,
        kappa_true.shape,
        kappa_wcs,
        z_source,
    )
    image_points = _observed_image_recovery_points(
        image_df,
        evaluator,
        best_fit,
        kappa_true,
        kappa_wcs,
        z_source,
        "kappa",
    )
    _write_kappa_recovery_from_grid(plot_dir, kappa_true, model_kappa, z_source, image_points=image_points)


def _plot_kappa_true_comparison_from_grid(
    plot_dir: Path,
    kappa_true: np.ndarray,
    model_kappa: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    z_source: float,
    *,
    member_overlays: pd.DataFrame | None = None,
    image_overlays: pd.DataFrame | None = None,
) -> None:
    valid_residual = np.isfinite(model_kappa) & np.isfinite(kappa_true) & (kappa_true > 0.0)
    fractional_residual = np.full(kappa_true.shape, np.nan, dtype=float)
    fractional_residual[valid_residual] = (model_kappa[valid_residual] - kappa_true[valid_residual]) / kappa_true[valid_residual]

    def _plot_map_overlays(ax: Any) -> None:
        plotted_overlay = False
        if image_overlays is not None and not image_overlays.empty:
            image_x = pd.to_numeric(image_overlays.get("x_arcsec"), errors="coerce").to_numpy(dtype=float)
            image_y = pd.to_numeric(image_overlays.get("y_arcsec"), errors="coerce").to_numpy(dtype=float)
            finite_images = np.isfinite(image_x) & np.isfinite(image_y)
            if np.any(finite_images):
                plotted_overlay = True
                ax.scatter(
                    image_x[finite_images],
                    image_y[finite_images],
                    marker="x",
                    s=26,
                    c="black",
                    linewidths=0.85,
                    alpha=0.9,
                    label="observed images",
                    zorder=6,
                )
        if member_overlays is not None and not member_overlays.empty:
            member_x = pd.to_numeric(member_overlays.get("x_arcsec"), errors="coerce").to_numpy(dtype=float)
            member_y = pd.to_numeric(member_overlays.get("y_arcsec"), errors="coerce").to_numpy(dtype=float)
            member_free = member_overlays.get("free", pd.Series(False, index=member_overlays.index)).astype(bool).to_numpy()
            member_ids = member_overlays.get("catalog_id", pd.Series("", index=member_overlays.index)).astype(str).to_numpy()
            finite_members = np.isfinite(member_x) & np.isfinite(member_y)
            for is_free, marker, edgecolor, label in [
                (False, "s", "black", "not free"),
                (True, "D", "gold", "free"),
            ]:
                mask = finite_members & (member_free == is_free)
                if not np.any(mask):
                    continue
                plotted_overlay = True
                ax.scatter(
                    member_x[mask],
                    member_y[mask],
                    marker=marker,
                    s=38 if is_free else 32,
                    facecolors="none",
                    edgecolors=edgecolor,
                    linewidths=0.9,
                    alpha=0.95,
                    label=label,
                    zorder=7,
                )
            for catalog_id, x_value, y_value, finite in zip(member_ids, member_x, member_y, finite_members, strict=False):
                if not finite or not str(catalog_id).strip():
                    continue
                ax.text(
                    float(x_value),
                    float(y_value),
                    str(catalog_id),
                    fontsize=5.5,
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=8,
                    bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
                )
        if plotted_overlay and hasattr(ax, "get_legend_handles_labels"):
            _handles, labels = ax.get_legend_handles_labels()
        else:
            labels = []
        if labels and hasattr(ax, "legend"):
            ax.legend(loc="upper right", fontsize=7, frameon=True)

    def _plot_comparison_map(
        output_name: str,
        image_data: np.ndarray,
        *,
        cmap: str,
        colorbar_label: str,
        vmin: float | None = None,
        vmax: float | None = None,
        norm: Normalize | None = None,
        overlays: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        image = _plot_truth_recovery_spatial_map(
            ax,
            x_arcsec,
            y_arcsec,
            image_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)
        if overlays:
            _plot_map_overlays(ax)
        ax.set_xlabel("x (arcsec)")
        ax.set_ylabel("y (arcsec)")
        fig.tight_layout()
        _finish_figure(fig, _plot_path(plot_dir, output_name), dpi=180, bbox_inches="tight")

    _plot_comparison_map(
        "truth_recovery_kappa_fractional_residual.pdf",
        fractional_residual,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0),
        colorbar_label=r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$",
        overlays=True,
    )
    _plot_kappa_model_truth_fractional_residual_from_grid(
        plot_dir,
        kappa_true,
        model_kappa,
        x_arcsec,
        y_arcsec,
        z_source,
        fractional_residual=fractional_residual,
    )


def _plot_kappa_model_truth_fractional_residual_from_grid(
    plot_dir: Path,
    kappa_true: np.ndarray,
    model_kappa: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    z_source: float,
    *,
    fractional_residual: np.ndarray | None = None,
    output_name: str = "truth_recovery_kappa_model_truth_fractional_residual.pdf",
) -> None:
    if fractional_residual is None:
        valid_residual = np.isfinite(model_kappa) & np.isfinite(kappa_true) & (kappa_true > 0.0)
        fractional_residual = np.full(np.asarray(kappa_true).shape, np.nan, dtype=float)
        fractional_residual[valid_residual] = (
            np.asarray(model_kappa, dtype=float)[valid_residual] - np.asarray(kappa_true, dtype=float)[valid_residual]
        ) / np.asarray(kappa_true, dtype=float)[valid_residual]

    panels = [
        (
            np.asarray(model_kappa, dtype=float),
            r"$\kappa_{\rm model}$",
            "magma",
            {"vmin": 0.0, "vmax": 3.0},
        ),
        (
            np.asarray(kappa_true, dtype=float),
            r"$\kappa_{\rm true}$",
            "magma",
            {"vmin": 0.0, "vmax": 3.0},
        ),
        (
            np.asarray(fractional_residual, dtype=float),
            r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$",
            _shifted_diverging_colormap("RdBu_r", 1.0 / 3.0),
            {"norm": Normalize(vmin=-1.0, vmax=2.0, clip=True)},
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=True)
    if hasattr(fig, "subplots_adjust"):
        fig.subplots_adjust(top=0.82, wspace=0.08)
    panel_images: list[tuple[Any, Any, str]] = []
    for panel_index, (ax, (image_data, colorbar_label, cmap, kwargs)) in enumerate(
        zip(np.ravel(axes), panels, strict=True)
    ):
        image = _plot_truth_recovery_spatial_map(
            ax,
            x_arcsec,
            y_arcsec,
            image_data,
            cmap=cmap,
            **kwargs,
        )
        panel_images.append((image, ax, colorbar_label))
        ax.set_xlabel("x (arcsec)")
        if panel_index == 0:
            ax.set_ylabel("y (arcsec)")
    canvas = getattr(fig, "canvas", None)
    if canvas is not None and hasattr(canvas, "draw"):
        canvas.draw()
    for image, ax, colorbar_label in panel_images:
        bbox = ax.get_position()
        cax = fig.add_axes([float(bbox.x0), float(bbox.y1) + 0.018, float(bbox.width), 0.026])
        colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        colorbar.set_label(colorbar_label)
        colorbar_axis = getattr(colorbar, "ax", None)
        if colorbar_axis is not None:
            colorbar_axis.xaxis.set_label_position("top")
            colorbar_axis.xaxis.set_ticks_position("top")
    _finish_figure(fig, _plot_path(plot_dir, output_name), dpi=180, bbox_inches="tight")


def _finite_aware_gaussian_smooth(values: np.ndarray, sigma_pix: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    smoothed = np.full(array.shape, np.nan, dtype=float)
    if not np.any(finite):
        return smoothed
    weights = gaussian_filter(finite.astype(float), sigma=float(sigma_pix), mode="nearest")
    weighted_values = gaussian_filter(np.where(finite, array, 0.0), sigma=float(sigma_pix), mode="nearest")
    valid_weights = np.isfinite(weights) & (weights > 0.0)
    smoothed[valid_weights] = weighted_values[valid_weights] / weights[valid_weights]
    return smoothed


def _smoothed_truth_kappa_peak_aperture_center(
    kappa_true: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    *,
    sigma_pix: float = TRUTH_RECOVERY_APERTURE_CENTER_SMOOTHING_SIGMA_PIX,
) -> dict[str, Any] | None:
    true_values = np.asarray(kappa_true, dtype=float)
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    if true_values.shape != x_values.shape or true_values.shape != y_values.shape:
        raise ValueError("Truth kappa, x grid, and y grid must have matching shapes for aperture center selection.")
    smoothed = _finite_aware_gaussian_smooth(true_values, float(sigma_pix))
    valid = np.isfinite(true_values) & np.isfinite(smoothed) & np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid):
        return None
    score = np.where(valid, smoothed, -np.inf)
    row_index, col_index = np.unravel_index(int(np.argmax(score)), score.shape)
    peak_value = float(smoothed[row_index, col_index])
    if not np.isfinite(peak_value):
        return None
    return {
        "center_mode": "smoothed_truth_kappa_peak",
        "center_x_arcsec": float(x_values[row_index, col_index]),
        "center_y_arcsec": float(y_values[row_index, col_index]),
        "center_catalog_id": "",
        "center_catalog_mag": float("nan"),
        "center_smoothing_sigma_pix": float(sigma_pix),
        "center_smoothed_kappa_peak": peak_value,
    }


def _truth_recovery_pixel_scale_arcsec(x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> float:
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    steps: list[float] = []
    if x_values.shape[1] > 1:
        dx = np.diff(x_values, axis=1)
        dy = np.diff(y_values, axis=1)
        values = np.hypot(dx, dy)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size:
            steps.append(float(np.nanmedian(finite)))
    if y_values.shape[0] > 1:
        dx = np.diff(x_values, axis=0)
        dy = np.diff(y_values, axis=0)
        values = np.hypot(dx, dy)
        finite = values[np.isfinite(values) & (values > 0.0)]
        if finite.size:
            steps.append(float(np.nanmedian(finite)))
    finite_steps = [value for value in steps if np.isfinite(value) and value > 0.0]
    return float(min(finite_steps)) if finite_steps else 1.0


def _truth_recovery_aperture_profile(
    kappa_true: np.ndarray,
    model_kappa: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    center: dict[str, Any],
    *,
    n_radii: int = 40,
) -> pd.DataFrame:
    true_values = np.asarray(kappa_true, dtype=float)
    model_values = np.asarray(model_kappa, dtype=float)
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    valid = np.isfinite(true_values) & np.isfinite(model_values) & np.isfinite(x_values) & np.isfinite(y_values)
    columns = [
        "radius_arcsec",
        "pixel_count",
        "kappa_true_sum",
        "kappa_model_sum",
        "kappa_model_sum_q16",
        "kappa_model_sum_q84",
        "m2d_ratio",
        "m2d_ratio_q16",
        "m2d_ratio_median",
        "m2d_ratio_q84",
        "center_mode",
        "center_x_arcsec",
        "center_y_arcsec",
        "center_catalog_id",
        "center_catalog_mag",
        "center_smoothing_sigma_pix",
        "center_smoothed_kappa_peak",
    ]
    if not np.any(valid):
        return pd.DataFrame(columns=columns)
    radius = np.hypot(
        x_values - float(center["center_x_arcsec"]),
        y_values - float(center["center_y_arcsec"]),
    )
    valid_radius = radius[valid]
    finite_radius = valid_radius[np.isfinite(valid_radius)]
    if finite_radius.size == 0:
        return pd.DataFrame(columns=columns)
    pixel_scale = _truth_recovery_pixel_scale_arcsec(x_values, y_values)
    r_min = float(pixel_scale)
    r_max = float(np.nanmax(finite_radius))
    if not np.isfinite(r_max) or r_max <= 0.0:
        return pd.DataFrame(columns=columns)
    if r_max < r_min:
        r_min = r_max
    radii = np.linspace(r_min, r_max, int(n_radii), dtype=float)
    rows: list[dict[str, Any]] = []
    for radius_arcsec in radii:
        in_aperture = valid & (radius <= float(radius_arcsec))
        pixel_count = int(np.count_nonzero(in_aperture))
        if pixel_count == 0:
            true_sum = model_sum = ratio = float("nan")
        else:
            true_sum = float(np.nansum(true_values[in_aperture]))
            model_sum = float(np.nansum(model_values[in_aperture]))
            ratio = model_sum / true_sum if np.isfinite(true_sum) and true_sum != 0.0 else float("nan")
        rows.append(
            {
                "radius_arcsec": float(radius_arcsec),
                "pixel_count": pixel_count,
                "kappa_true_sum": true_sum,
                "kappa_model_sum": model_sum,
                "kappa_model_sum_q16": model_sum,
                "kappa_model_sum_q84": model_sum,
                "m2d_ratio": ratio,
                "m2d_ratio_q16": ratio,
                "m2d_ratio_median": ratio,
                "m2d_ratio_q84": ratio,
                "center_mode": str(center["center_mode"]),
                "center_x_arcsec": float(center["center_x_arcsec"]),
                "center_y_arcsec": float(center["center_y_arcsec"]),
                "center_catalog_id": str(center["center_catalog_id"]),
                "center_catalog_mag": float(center["center_catalog_mag"]),
                "center_smoothing_sigma_pix": _finite_float_or_nan(center.get("center_smoothing_sigma_pix")),
                "center_smoothed_kappa_peak": _finite_float_or_nan(center.get("center_smoothed_kappa_peak")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _truth_recovery_image_aperture_radii(
    image_df: pd.DataFrame | None,
    center: dict[str, Any] | None,
) -> np.ndarray:
    if image_df is None or image_df.empty or center is None:
        return np.asarray([], dtype=float)
    if "x_obs_arcsec" not in image_df or "y_obs_arcsec" not in image_df:
        return np.asarray([], dtype=float)
    center_x = _finite_float_or_nan(center.get("center_x_arcsec"))
    center_y = _finite_float_or_nan(center.get("center_y_arcsec"))
    if not (np.isfinite(center_x) and np.isfinite(center_y)):
        return np.asarray([], dtype=float)
    x_obs = pd.to_numeric(image_df["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(image_df["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
    radii = np.hypot(x_obs - center_x, y_obs - center_y)
    return radii[np.isfinite(radii) & (radii > 0.0)]


def _plot_truth_recovery_m2d_aperture_ratio(
    plot_dir: Path,
    profile_df: pd.DataFrame,
    center: dict[str, Any],
    *,
    image_radii_arcsec: np.ndarray | Sequence[float] | None = None,
) -> None:
    if profile_df.empty:
        return
    output_path = _plot_path(plot_dir, "truth_recovery_m2d_aperture_ratio.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(6.3, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    radius = pd.to_numeric(profile_df["radius_arcsec"], errors="coerce").to_numpy(dtype=float)
    truth_m2d = pd.to_numeric(profile_df["kappa_true_sum"], errors="coerce").to_numpy(dtype=float)
    model_m2d = pd.to_numeric(profile_df["kappa_model_sum"], errors="coerce").to_numpy(dtype=float)
    ratio_column = "m2d_ratio_median" if "m2d_ratio_median" in profile_df else "m2d_ratio"
    ratio = pd.to_numeric(profile_df[ratio_column], errors="coerce").to_numpy(dtype=float)
    positive_radius = np.isfinite(radius) & (radius > 0.0)
    finite_truth = positive_radius & np.isfinite(truth_m2d) & (truth_m2d > 0.0)
    finite_model = positive_radius & np.isfinite(model_m2d) & (model_m2d > 0.0)
    finite_residual = positive_radius & np.isfinite(ratio)
    if np.any(finite_truth) or np.any(finite_model):
        if np.any(finite_model):
            if {"kappa_model_sum_q16", "kappa_model_sum_q84"}.issubset(profile_df.columns):
                model_q16 = pd.to_numeric(profile_df["kappa_model_sum_q16"], errors="coerce").to_numpy(dtype=float)
                model_q84 = pd.to_numeric(profile_df["kappa_model_sum_q84"], errors="coerce").to_numpy(dtype=float)
                finite_model_band = finite_model & np.isfinite(model_q16) & (model_q16 > 0.0)
                finite_model_band &= np.isfinite(model_q84) & (model_q84 > 0.0)
                if np.any(finite_model_band):
                    ax.fill_between(
                        radius[finite_model_band],
                        model_q16[finite_model_band],
                        model_q84[finite_model_band],
                        color="tab:blue",
                        alpha=0.18,
                        linewidth=0.0,
                        label="16-84% posterior",
                    )
            ax.plot(
                radius[finite_model],
                model_m2d[finite_model],
                color="tab:blue",
                linewidth=1.8,
                label="model",
            )
        if np.any(finite_truth):
            ax.plot(
                radius[finite_truth],
                truth_m2d[finite_truth],
                color="0.2",
                linestyle="--",
                linewidth=1.3,
                label="truth",
            )
    else:
        ax.text(0.5, 0.5, "No finite aperture masses.", ha="center", va="center", transform=ax.transAxes)
    image_radii = np.asarray([] if image_radii_arcsec is None else image_radii_arcsec, dtype=float).reshape(-1)
    finite_image_radii = image_radii[np.isfinite(image_radii) & (image_radii > 0.0)]
    if finite_image_radii.size:
        ax.vlines(
            finite_image_radii,
            0.0,
            0.06,
            transform=ax.get_xaxis_transform(),
            color="0.45",
            linewidth=0.45,
            alpha=0.65,
            zorder=5,
        )
    if np.any(finite_residual):
        if {"m2d_ratio_q16", "m2d_ratio_q84"}.issubset(profile_df.columns):
            ratio_q16 = pd.to_numeric(profile_df["m2d_ratio_q16"], errors="coerce").to_numpy(dtype=float)
            ratio_q84 = pd.to_numeric(profile_df["m2d_ratio_q84"], errors="coerce").to_numpy(dtype=float)
            finite_residual_band = finite_residual & np.isfinite(ratio_q16) & np.isfinite(ratio_q84)
            if np.any(finite_residual_band):
                ax_resid.fill_between(
                    radius[finite_residual_band],
                    ratio_q16[finite_residual_band] - 1.0,
                    ratio_q84[finite_residual_band] - 1.0,
                    color="tab:blue",
                    alpha=0.18,
                    linewidth=0.0,
                )
        ax_resid.plot(
            radius[finite_residual],
            ratio[finite_residual] - 1.0,
            color="tab:blue",
            linewidth=1.3,
        )
    else:
        ax_resid.text(0.5, 0.5, "No finite residuals.", ha="center", va="center", transform=ax_resid.transAxes)
    ax_resid.axhline(0.0, color="0.2", linestyle="--", linewidth=1.0)
    ax_resid.set_ylim(-0.1, 0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax_resid.set_xlim(left=5.0)
    ax_resid.set_xlabel("R (arcsec)")
    ax_resid.set_ylabel("residual")
    ax.set_ylabel(r"enclosed $M_{\rm 2D}(<R)$")
    ax.set_title("Projected aperture mass recovery")
    ax.grid(True, alpha=0.25)
    ax_resid.grid(True, alpha=0.25)
    center_mode = str(center.get("center_mode", ""))
    if center_mode == "smoothed_truth_kappa_peak":
        center_label = "smoothed truth kappa peak"
    else:
        center_label = str(center.get("center_catalog_id", center_mode or "unknown"))
    annotation_lines = [
        f"center: {center_label}",
        f"x={float(center['center_x_arcsec']):.3g}\"  y={float(center['center_y_arcsec']):.3g}\"",
    ]
    smoothing_sigma = _finite_float_or_nan(center.get("center_smoothing_sigma_pix"))
    smoothed_peak = _finite_float_or_nan(center.get("center_smoothed_kappa_peak"))
    catalog_mag = _finite_float_or_nan(center.get("center_catalog_mag"))
    if np.isfinite(smoothing_sigma):
        annotation_lines.append(f"smoothing sigma={smoothing_sigma:g} pix")
    if np.isfinite(smoothed_peak):
        annotation_lines.append(f"smoothed kappa peak={smoothed_peak:.3g}")
    if center_mode != "smoothed_truth_kappa_peak" and np.isfinite(catalog_mag):
        annotation_lines.append(f"mag={catalog_mag:.3g}")
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    _finish_figure(fig, output_path, dpi=220, bbox_inches="tight")


def _write_truth_recovery_m2d_aperture_profile(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    kappa_true: np.ndarray,
    model_kappa: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> None:
    del evaluator
    center = _smoothed_truth_kappa_peak_aperture_center(kappa_true, x_arcsec, y_arcsec)
    if center is None:
        _log(None, "[truth-recovery:m2d] skipped: no finite smoothed truth-kappa peak center")
        return
    profile_df = _truth_recovery_aperture_profile(
        kappa_true,
        model_kappa,
        x_arcsec,
        y_arcsec,
        center,
    )
    if profile_df.empty:
        _log(None, "[truth-recovery:m2d] skipped: aperture profile has no finite pixels")
        return
    tables_dir = plot_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(tables_dir / "truth_recovery_m2d_aperture_profile.csv", index=False)
    _plot_truth_recovery_m2d_aperture_ratio(plot_dir, profile_df, center)


def _kappa_truth_grids(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_fits: str | Path,
    caustic_source_redshift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kappa_true, kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    model_kappa, x_arcsec, y_arcsec = _kappa_model_grid_for_true_fits(
        evaluator,
        best_fit,
        kappa_true.shape,
        kappa_wcs,
        caustic_source_redshift,
    )
    return kappa_true, model_kappa, x_arcsec, y_arcsec


def _plot_kappa_true_comparison(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_fits: str | Path,
    caustic_source_redshift: float,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:kappa_comparison] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    kappa_true, model_kappa, x_arcsec, y_arcsec = _kappa_truth_grids(
        evaluator,
        best_fit,
        kappa_true_fits,
        z_source,
    )
    _plot_kappa_true_comparison_from_grid(plot_dir, kappa_true, model_kappa, x_arcsec, y_arcsec, z_source)


def _plot_kappa_truth_diagnostics(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    results: PosteriorResults,
    kappa_true_fits: str | Path,
    caustic_source_redshift: float,
    image_df: pd.DataFrame | None = None,
    *,
    posterior_truth_recovery_draws: int | None = None,
    truth_grid_mode: str = TRUTH_GRID_MODE_MEDIAN,
    truth_grid_size: int = DEFAULT_TRUTH_GRID_SIZE,
    truth_grid_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    precompute_quantities: Sequence[str] | None = None,
    truth_grid_source_fits: dict[str, str | Path] | None = None,
    require_precomputed_truth_grid: bool = False,
    truth_grid_draw_seed: int | None = None,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:kappa_truth] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    native_kappa_true, native_kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    kappa_wcs, diagnostic_shape, grid_metadata = _truth_recovery_diagnostic_grid(
        native_kappa_wcs,
        native_kappa_true.shape,
        truth_grid_size,
    )
    kappa_true = _truth_recovery_sample_truth_image(
        native_kappa_true,
        native_kappa_wcs,
        kappa_wcs,
        diagnostic_shape,
        native_shape=native_kappa_true.shape,
    )
    quantities = tuple(precompute_quantities) if precompute_quantities is not None else ("kappa",)
    source_truth_fits = truth_grid_source_fits if truth_grid_source_fits is not None else {"kappa": kappa_true_fits}
    quantiles, x_arcsec, y_arcsec = _posterior_truth_grid_quantiles(
        plot_dir,
        evaluator,
        results,
        kappa_wcs,
        diagnostic_shape,
        z_source,
        source_truth_fits=source_truth_fits,
        quantities=quantities,
        max_draws=posterior_truth_recovery_draws,
        truth_grid_mode=truth_grid_mode,
        cache=truth_grid_cache,
        require_cache=require_precomputed_truth_grid,
        aperture_center=None,
        aperture_kappa_true=kappa_true,
        aperture_image_df=image_df,
        truth_grid_metadata=grid_metadata,
        draw_seed=truth_grid_draw_seed,
    )
    model_kappa = quantiles["kappa"]["median"]
    _plot_kappa_true_comparison_from_grid(
        plot_dir,
        kappa_true,
        model_kappa,
        x_arcsec,
        y_arcsec,
        z_source,
        member_overlays=_truth_recovery_member_overlay_table(evaluator),
        image_overlays=_truth_recovery_image_overlay_table(image_df),
    )
    image_points = _observed_image_recovery_points_from_grids(
        image_df,
        evaluator,
        kappa_true,
        model_kappa,
        kappa_wcs,
    )
    _write_kappa_recovery_from_grid(plot_dir, kappa_true, model_kappa, z_source, image_points=image_points)


def _absolute_mu_truth_grid(
    kappa_true_fits: str | Path,
    gammax_true_fits: str | Path,
    gammay_true_fits: str | Path,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
    truth_grid_size: int = 0,
) -> tuple[np.ndarray, WCS]:
    kappa_true, kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    gammax_true, gammax_wcs = _load_kappa_true_fits(gammax_true_fits)
    gammay_true, gammay_wcs = _load_kappa_true_fits(gammay_true_fits)
    if gammax_true.shape != kappa_true.shape:
        raise ValueError("gammax truth FITS shape does not match the kappa truth FITS shape.")
    if gammay_true.shape != kappa_true.shape:
        raise ValueError("gammay truth FITS shape does not match the kappa truth FITS shape.")
    _validate_matching_truth_wcs(kappa_wcs, gammax_wcs, kappa_true.shape, label="gammax")
    _validate_matching_truth_wcs(kappa_wcs, gammay_wcs, kappa_true.shape, label="gammay")
    truth_wcs, diagnostic_shape, _grid_metadata = _truth_recovery_diagnostic_grid(
        kappa_wcs,
        kappa_true.shape,
        truth_grid_size,
    )
    kappa_sampled = _truth_recovery_sample_truth_image(
        kappa_true,
        kappa_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=kappa_true.shape,
    )
    gammax_sampled = _truth_recovery_sample_truth_image(
        gammax_true,
        gammax_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=gammax_true.shape,
    )
    gammay_sampled = _truth_recovery_sample_truth_image(
        gammay_true,
        gammay_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=gammay_true.shape,
    )
    signed_mu_true = _signed_magnification_from_kappa_gamma(kappa_sampled, gammax_sampled, gammay_sampled)
    return np.abs(signed_mu_true), truth_wcs


def _absolute_mu_truth_grids(
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    kappa_true_fits: str | Path,
    gammax_true_fits: str | Path,
    gammay_true_fits: str | Path,
    caustic_source_redshift: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
    truth_grid_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    abs_mu_true, truth_wcs = _absolute_mu_truth_grid(
        kappa_true_fits,
        gammax_true_fits,
        gammay_true_fits,
        cap=cap,
        truth_grid_size=truth_grid_size,
    )
    y_pixels, x_pixels = np.indices(abs_mu_true.shape, dtype=float)
    model_mu, x_arcsec, y_arcsec = _model_quantity_grid_for_wcs_pixels(
        evaluator,
        best_fit,
        truth_wcs,
        x_pixels,
        y_pixels,
        caustic_source_redshift,
        "magnification",
    )
    abs_mu_model = np.abs(np.asarray(model_mu, dtype=float))
    return abs_mu_true, abs_mu_model, x_arcsec, y_arcsec


def _write_abs_mu_recovery_from_grid(
    plot_dir: Path,
    abs_mu_true: np.ndarray,
    abs_mu_model: np.ndarray,
    z_source: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
    image_points: pd.DataFrame | None = None,
) -> None:
    axis_limits = (0.0, float(ABSOLUTE_MAGNIFICATION_RECOVERY_AXIS_MAX))
    recovery = _quantity_recovery_reduced(
        abs_mu_true,
        abs_mu_model,
        "abs_mu",
        limits=axis_limits,
        stat_limits=axis_limits,
    )
    tables_dir = plot_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    recovery["bin_table"].to_csv(tables_dir / "truth_recovery_mu_recovery_binned.csv", index=False)
    recovery["summary_table"].to_csv(tables_dir / "truth_recovery_mu_recovery_summary.csv", index=False)
    _plot_quantity_recovery(
        recovery,
        _plot_path(plot_dir, "truth_recovery_mu_recovery.pdf"),
        quantity="abs_mu",
        true_label=r"$|\mu_{\rm true}|$",
        model_label=r"$|\mu_{\rm model}|$",
        image_points=image_points,
    )


def _plot_abs_mu_true_comparison_from_grid(
    plot_dir: Path,
    abs_mu_true: np.ndarray,
    abs_mu_model: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    z_source: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
) -> None:
    valid_residual = np.isfinite(abs_mu_model) & np.isfinite(abs_mu_true) & (abs_mu_true > 0.0)
    fractional_residual = np.full(abs_mu_true.shape, np.nan, dtype=float)
    fractional_residual[valid_residual] = (
        (abs_mu_model[valid_residual] - abs_mu_true[valid_residual]) / abs_mu_true[valid_residual]
    )
    _plot_abs_mu_model_truth_fractional_residual_from_grid(
        plot_dir,
        abs_mu_true,
        abs_mu_model,
        x_arcsec,
        y_arcsec,
        z_source,
        cap=cap,
        fractional_residual=fractional_residual,
    )


def _plot_abs_mu_model_truth_fractional_residual_from_grid(
    plot_dir: Path,
    abs_mu_true: np.ndarray,
    abs_mu_model: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    z_source: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
    fractional_residual: np.ndarray | None = None,
    output_name: str = "truth_recovery_mu_model_truth_fractional_residual.pdf",
) -> None:
    if fractional_residual is None:
        valid_residual = np.isfinite(abs_mu_model) & np.isfinite(abs_mu_true) & (abs_mu_true > 0.0)
        fractional_residual = np.full(np.asarray(abs_mu_true).shape, np.nan, dtype=float)
        fractional_residual[valid_residual] = (
            np.asarray(abs_mu_model, dtype=float)[valid_residual] - np.asarray(abs_mu_true, dtype=float)[valid_residual]
        ) / np.asarray(abs_mu_true, dtype=float)[valid_residual]

    panels = [
        (
            np.asarray(abs_mu_model, dtype=float),
            r"$|\mu_{\rm model}|$",
            "viridis",
            {"vmin": 0.0, "vmax": float(cap)},
        ),
        (
            np.asarray(abs_mu_true, dtype=float),
            r"$|\mu_{\rm true}|$",
            "viridis",
            {"vmin": 0.0, "vmax": float(cap)},
        ),
        (
            np.asarray(fractional_residual, dtype=float),
            r"$(|\mu_{\rm model}| - |\mu_{\rm true}|) / |\mu_{\rm true}|$",
            _shifted_diverging_colormap("RdBu_r", 1.0 / 3.0),
            {"norm": Normalize(vmin=-1.0, vmax=2.0, clip=True)},
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=True)
    if hasattr(fig, "subplots_adjust"):
        fig.subplots_adjust(top=0.82, wspace=0.08)
    panel_images: list[tuple[Any, Any, str]] = []
    for panel_index, (ax, (image_data, colorbar_label, cmap, kwargs)) in enumerate(
        zip(np.ravel(axes), panels, strict=True)
    ):
        image = _plot_truth_recovery_spatial_map(
            ax,
            x_arcsec,
            y_arcsec,
            image_data,
            cmap=cmap,
            **kwargs,
        )
        panel_images.append((image, ax, colorbar_label))
        ax.set_xlabel("x (arcsec)")
        if panel_index == 0:
            ax.set_ylabel("y (arcsec)")
    canvas = getattr(fig, "canvas", None)
    if canvas is not None and hasattr(canvas, "draw"):
        canvas.draw()
    for image, ax, colorbar_label in panel_images:
        bbox = ax.get_position()
        cax = fig.add_axes([float(bbox.x0), float(bbox.y1) + 0.018, float(bbox.width), 0.026])
        colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        colorbar.set_label(colorbar_label)
        colorbar_axis = getattr(colorbar, "ax", None)
        if colorbar_axis is not None:
            colorbar_axis.xaxis.set_label_position("top")
            colorbar_axis.xaxis.set_ticks_position("top")
    _finish_figure(fig, _plot_path(plot_dir, output_name), dpi=180, bbox_inches="tight")


def _has_zero_contour(field: np.ndarray) -> bool:
    finite = np.asarray(field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return float(np.nanmin(finite)) <= 0.0 <= float(np.nanmax(finite))


def _plot_critical_line_recovery_from_grid(
    plot_dir: Path,
    truth_determinant: np.ndarray,
    model_determinant: np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    z_source: float,
) -> None:
    output_path = _plot_path(plot_dir, "truth_recovery_critical_line_recovery.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    legend_handles: list[Line2D] = []
    contour_specs = [
        (
            np.asarray(truth_determinant, dtype=float),
            "black",
            "-",
            "truth critical line",
        ),
        (
            np.asarray(model_determinant, dtype=float),
            "tab:blue",
            "--",
            "model critical line",
        ),
    ]
    for determinant, color, linestyle, label in contour_specs:
        if not _has_zero_contour(determinant):
            continue
        ax.contour(
            x_arcsec,
            y_arcsec,
            determinant,
            levels=[0.0],
            colors=[color],
            linestyles=[linestyle],
            linewidths=[1.1],
        )
        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.1, label=label))
    if not legend_handles:
        ax.text(0.5, 0.5, "No zero-determinant critical line in grid.", ha="center", va="center", transform=ax.transAxes)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title(fr"Critical-line recovery ($z_s={float(z_source):g}$)")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    _finish_figure(fig, output_path, dpi=180, bbox_inches="tight")


def _plot_abs_mu_truth_diagnostics(
    plot_dir: Path,
    evaluator: ClusterJAXEvaluator,
    results: PosteriorResults,
    kappa_true_fits: str | Path,
    gammax_true_fits: str | Path,
    gammay_true_fits: str | Path,
    caustic_source_redshift: float,
    *,
    cap: float = ABSOLUTE_MAGNIFICATION_PLOT_CAP,
    image_df: pd.DataFrame | None = None,
    posterior_truth_recovery_draws: int | None = None,
    truth_grid_mode: str = TRUTH_GRID_MODE_MEDIAN,
    truth_grid_size: int = DEFAULT_TRUTH_GRID_SIZE,
    truth_grid_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    require_precomputed_truth_grid: bool = False,
    truth_grid_draw_seed: int | None = None,
) -> None:
    z_source = float(caustic_source_redshift)
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            None,
            f"[plot:mu_truth] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return
    native_kappa_true, native_kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    native_gammax_true, native_gammax_wcs = _load_kappa_true_fits(gammax_true_fits)
    native_gammay_true, native_gammay_wcs = _load_kappa_true_fits(gammay_true_fits)
    if native_gammax_true.shape != native_kappa_true.shape:
        raise ValueError("gammax truth FITS shape does not match the kappa truth FITS shape.")
    if native_gammay_true.shape != native_kappa_true.shape:
        raise ValueError("gammay truth FITS shape does not match the kappa truth FITS shape.")
    _validate_matching_truth_wcs(native_kappa_wcs, native_gammax_wcs, native_kappa_true.shape, label="gammax")
    _validate_matching_truth_wcs(native_kappa_wcs, native_gammay_wcs, native_kappa_true.shape, label="gammay")
    truth_wcs, diagnostic_shape, grid_metadata = _truth_recovery_diagnostic_grid(
        native_kappa_wcs,
        native_kappa_true.shape,
        truth_grid_size,
    )
    kappa_true = _truth_recovery_sample_truth_image(
        native_kappa_true,
        native_kappa_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=native_kappa_true.shape,
    )
    gammax_true = _truth_recovery_sample_truth_image(
        native_gammax_true,
        native_gammax_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=native_gammax_true.shape,
    )
    gammay_true = _truth_recovery_sample_truth_image(
        native_gammay_true,
        native_gammay_wcs,
        truth_wcs,
        diagnostic_shape,
        native_shape=native_gammay_true.shape,
    )
    abs_mu_true = np.abs(_signed_magnification_from_kappa_gamma(kappa_true, gammax_true, gammay_true))
    quantiles, x_arcsec, y_arcsec = _posterior_truth_grid_quantiles(
        plot_dir,
        evaluator,
        results,
        truth_wcs,
        diagnostic_shape,
        z_source,
        source_truth_fits={
            "kappa": kappa_true_fits,
            "gamma1": gammax_true_fits,
            "gamma2": gammay_true_fits,
            "detA": kappa_true_fits,
            "mu": kappa_true_fits,
            "abs_mu": kappa_true_fits,
        },
        quantities=("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu"),
        max_draws=posterior_truth_recovery_draws,
        truth_grid_mode=truth_grid_mode,
        cache=truth_grid_cache,
        require_cache=require_precomputed_truth_grid,
        aperture_center=None,
        aperture_kappa_true=kappa_true,
        truth_grid_metadata=grid_metadata,
        draw_seed=truth_grid_draw_seed,
    )
    abs_mu_model = quantiles["abs_mu"]["median"]
    truth_determinant = _critical_determinant_from_kappa_gamma(kappa_true, gammax_true, gammay_true)
    model_determinant = quantiles["detA"]["median"]
    _plot_critical_line_recovery_from_grid(
        plot_dir,
        truth_determinant,
        model_determinant,
        x_arcsec,
        y_arcsec,
        z_source,
    )
    _plot_abs_mu_true_comparison_from_grid(
        plot_dir,
        abs_mu_true,
        abs_mu_model,
        x_arcsec,
        y_arcsec,
        z_source,
        cap=cap,
    )
    image_points = _observed_image_recovery_points_from_grids(
        image_df,
        evaluator,
        abs_mu_true,
        abs_mu_model,
        truth_wcs,
    )
    _write_abs_mu_recovery_from_grid(plot_dir, abs_mu_true, abs_mu_model, z_source, cap=cap, image_points=image_points)


def _precompute_truth_recovery_grids(
    run_dir: Path,
    evaluator: ClusterJAXEvaluator,
    results: PosteriorResults,
    args: argparse.Namespace,
    truth_grid_cache: dict[tuple[Any, ...], dict[str, Any]],
    progress: Any | None = None,
) -> bool:
    if bool(getattr(args, "skip_grid_diagnostics", False)):
        return False
    kappa_true_fits = getattr(args, "kappa_true_fits", None)
    if kappa_true_fits is None or not str(kappa_true_fits).strip():
        return False
    z_source = float(getattr(args, "caustic_source_redshift", 9.0))
    z_lens = getattr(evaluator.state, "z_lens", None)
    if z_lens is not None and np.isfinite(float(z_lens)) and z_source <= float(z_lens):
        _log(
            args,
            f"[truth-recovery] skipped: caustic source redshift z={z_source:g} "
            f"is not behind lens redshift z={float(z_lens):g}",
        )
        return False

    gammax_true_fits = getattr(args, "gammax_true_fits", None)
    gammay_true_fits = getattr(args, "gammay_true_fits", None)
    truth_grid_size = int(getattr(args, "truth_grid_size", DEFAULT_TRUTH_GRID_SIZE))
    gamma_truth_available = (
        gammax_true_fits is not None
        and str(gammax_true_fits).strip()
        and gammay_true_fits is not None
        and str(gammay_true_fits).strip()
    )
    native_kappa_true, native_kappa_wcs = _load_kappa_true_fits(kappa_true_fits)
    kappa_wcs, diagnostic_shape, grid_metadata = _truth_recovery_diagnostic_grid(
        native_kappa_wcs,
        native_kappa_true.shape,
        truth_grid_size,
    )
    kappa_true = _truth_recovery_sample_truth_image(
        native_kappa_true,
        native_kappa_wcs,
        kappa_wcs,
        diagnostic_shape,
        native_shape=native_kappa_true.shape,
    )
    if gamma_truth_available:
        native_gammax_true, native_gammax_wcs = _load_kappa_true_fits(gammax_true_fits)
        native_gammay_true, native_gammay_wcs = _load_kappa_true_fits(gammay_true_fits)
        if native_gammax_true.shape != native_kappa_true.shape:
            raise ValueError("gammax truth FITS shape does not match the kappa truth FITS shape.")
        if native_gammay_true.shape != native_kappa_true.shape:
            raise ValueError("gammay truth FITS shape does not match the kappa truth FITS shape.")
        _validate_matching_truth_wcs(native_kappa_wcs, native_gammax_wcs, native_kappa_true.shape, label="gammax")
        _validate_matching_truth_wcs(native_kappa_wcs, native_gammay_wcs, native_kappa_true.shape, label="gammay")
        source_truth_fits = {
            "kappa": str(kappa_true_fits),
            "gamma1": str(gammax_true_fits),
            "gamma2": str(gammay_true_fits),
            "detA": str(kappa_true_fits),
            "mu": str(kappa_true_fits),
            "abs_mu": str(kappa_true_fits),
        }
        quantities = ("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
    else:
        source_truth_fits = {"kappa": str(kappa_true_fits)}
        quantities = ("kappa",)

    _posterior_truth_grid_quantiles(
        run_dir,
        evaluator,
        results,
        kappa_wcs,
        kappa_true.shape,
        z_source,
        source_truth_fits=source_truth_fits,
        quantities=quantities,
        max_draws=getattr(args, "posterior_truth_recovery_draws", None),
        truth_grid_mode=str(getattr(args, "truth_grid_mode", TRUTH_GRID_MODE_MEDIAN)),
        cache=truth_grid_cache,
        aperture_center=None,
        aperture_kappa_true=kappa_true,
        progress=progress,
        truth_grid_metadata=grid_metadata,
        draw_seed=getattr(args, "seed", DEFAULT_RUNTIME_SEED),
    )
    return True


def _empty_subhalo_properties_table() -> pd.DataFrame:
    return pd.DataFrame(columns=SUBHALO_PROPERTIES_COLUMNS)


def _finite_float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _subhalo_component_records(state: BuildState) -> list[dict[str, Any]]:
    from .cluster_solver import COMPONENT_FAMILY_INDEPENDENT_FREE, COMPONENT_FAMILY_SCALING

    packed = getattr(state, "packed_lens_spec", None)
    if packed is None:
        return []
    component_family = np.asarray(getattr(packed, "component_family", []), dtype=int)
    if component_family.size == 0:
        return []
    records: list[dict[str, Any]] = []
    for raw_record in getattr(state, "scaling_component_records", []) or []:
        if not isinstance(raw_record, dict) or "component_index" not in raw_record:
            continue
        try:
            component_index = int(raw_record["component_index"])
        except (TypeError, ValueError):
            continue
        if component_index < 0 or component_index >= component_family.size:
            continue
        if int(component_family[component_index]) != COMPONENT_FAMILY_SCALING:
            continue
        free_component_index = _finite_float_or_nan(raw_record.get("free_component_index", -1))
        model_component_index = int(free_component_index) if np.isfinite(free_component_index) else -1
        if model_component_index >= 0:
            if model_component_index >= component_family.size:
                continue
            if int(component_family[model_component_index]) != COMPONENT_FAMILY_INDEPENDENT_FREE:
                continue
        else:
            model_component_index = component_index
        record = dict(raw_record)
        record["component_index"] = component_index
        record["model_component_index"] = model_component_index
        records.append(record)
    return records


def _lens_mass_to_msun(mass_angle: Any, sigma_crit_angle: float) -> float:
    mass_value = _finite_float_or_nan(np.asarray(mass_angle).reshape(()))
    if not (np.isfinite(mass_value) and np.isfinite(float(sigma_crit_angle)) and float(sigma_crit_angle) > 0.0):
        return float("nan")
    return float(mass_value * float(sigma_crit_angle))


def _subhalo_properties_table(
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    caustic_source_redshift: float,
) -> pd.DataFrame:
    records = _subhalo_component_records(state)
    if not records:
        return _empty_subhalo_properties_table()

    z_source = float(caustic_source_redshift)
    try:
        sigma_crit_angle = critical_surface_density_angle_from_config(
            float(getattr(state, "z_lens")),
            z_source,
            getattr(state, "cosmo_config", None),
        )
    except Exception as exc:  # pragma: no cover - malformed artifacts should still write diagnostic rows
        _log(None, f"[plot:subhalo_properties] mass conversion unavailable: {exc}")
        sigma_crit_angle = float("nan")

    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
    model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
    if model is None:
        model, _ = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)

    packed = getattr(state, "packed_lens_spec", None)
    x_center_base = np.asarray(getattr(packed, "x_center_base", []), dtype=float)
    y_center_base = np.asarray(getattr(packed, "y_center_base", []), dtype=float)
    rows: list[dict[str, Any]] = []
    for record in records:
        component_index = int(record["component_index"])
        model_component_index = int(record.get("model_component_index", component_index))
        if model_component_index >= len(kwargs_lens):
            continue
        kwargs = dict(kwargs_lens[model_component_index])
        rs = _finite_float_or_nan(kwargs.get("Rs"))
        mass_within_rs = float("nan")
        mass_within_total = float("nan")
        if np.isfinite(rs) and rs > 0.0 and hasattr(model, "mass_3d"):
            bool_list = [idx == model_component_index for idx in range(len(kwargs_lens))]
            try:
                mass_within_rs = _lens_mass_to_msun(
                    model.mass_3d(rs, kwargs_lens, bool_list=bool_list),
                    sigma_crit_angle,
                )
                mass_within_total = _lens_mass_to_msun(
                    model.mass_3d(SUBHALO_TOTAL_MASS_RADIUS_FACTOR * rs, kwargs_lens, bool_list=bool_list),
                    sigma_crit_angle,
                )
            except Exception as exc:  # pragma: no cover - best-effort diagnostic for corrupt model rows
                _log(None, f"[plot:subhalo_properties] mass_3d failed component={component_index}: {exc}")
        x_centre = _finite_float_or_nan(record.get("x_centre"))
        y_centre = _finite_float_or_nan(record.get("y_centre"))
        if (not np.isfinite(x_centre)) and component_index < x_center_base.size:
            x_centre = _finite_float_or_nan(x_center_base[component_index])
        if (not np.isfinite(y_centre)) and component_index < y_center_base.size:
            y_centre = _finite_float_or_nan(y_center_base[component_index])
        radius = float(np.hypot(x_centre, y_centre)) if np.isfinite(x_centre) and np.isfinite(y_centre) else float("nan")
        rows.append(
            {
                "component_index": component_index,
                "model_component_index": model_component_index,
                "potfile_id": str(record.get("potfile_id", "")),
                "potfile_order": int(record.get("potfile_order", -1)),
                "catalog_id": str(record.get("catalog_id", f"component{component_index}")),
                "catalog_mag": _finite_float_or_nan(record.get("catalog_mag")),
                "x_centre": float(x_centre),
                "y_centre": float(y_centre),
                "radius_arcsec": radius,
                "sigma0": _finite_float_or_nan(kwargs.get("sigma0")),
                "Ra": _finite_float_or_nan(kwargs.get("Ra")),
                "Rs": rs,
                "mass_within_Rs_msun": mass_within_rs,
                "mass_within_1e6_Rs_msun": mass_within_total,
            }
        )
    if not rows:
        return _empty_subhalo_properties_table()
    return pd.DataFrame(rows, columns=SUBHALO_PROPERTIES_COLUMNS).sort_values(
        ["potfile_order", "component_index"],
    ).reset_index(drop=True)


def _finite_positive_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _finite_nonnegative_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values >= 0.0)]


def _subhalo_log_mass_bins(*mass_arrays: np.ndarray) -> np.ndarray | None:
    positive_arrays = [np.asarray(values, dtype=float) for values in mass_arrays if np.asarray(values).size]
    positive_arrays = [values[np.isfinite(values) & (values > 0.0)] for values in positive_arrays]
    positive_arrays = [values for values in positive_arrays if values.size]
    if not positive_arrays:
        return None
    log_mass = np.log10(np.concatenate(positive_arrays))
    log_min = float(np.nanmin(log_mass))
    log_max = float(np.nanmax(log_mass))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if log_max <= log_min:
        log_min -= 0.25
        log_max += 0.25
    n_bins = int(np.clip(np.sqrt(max(log_mass.size, 1)), 6, 18))
    return np.linspace(log_min, log_max, n_bins + 1)


def _subhalo_linear_bins(values: np.ndarray) -> np.ndarray | None:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values) & (finite_values >= 0.0)]
    if finite_values.size == 0:
        return None
    value_min = float(np.nanmin(finite_values))
    value_max = float(np.nanmax(finite_values))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return None
    if value_max <= value_min:
        padding = max(0.5, 0.1 * max(abs(value_min), 1.0))
        value_min = max(0.0, value_min - padding)
        value_max += padding
    n_bins = int(np.clip(np.sqrt(max(finite_values.size, 1)), 6, 18))
    return np.linspace(value_min, value_max, n_bins + 1)


def _plot_subhalo_mass_function(subhalo_df: pd.DataFrame, path: Path) -> None:
    mass_total = _finite_positive_column(subhalo_df, "mass_within_1e6_Rs_msun")
    bins = _subhalo_log_mass_bins(mass_total)
    if bins is None:
        _write_placeholder_plot(
            path,
            "Subhalo mass function",
            "No finite fitted subhalo masses are available.",
        )
        return
    bin_width = float(np.mean(np.diff(bins)))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if mass_total.size:
        log_mass_total = np.log10(mass_total)
        ax.hist(
            log_mass_total,
            bins=bins,
            weights=np.full(log_mass_total.size, 1.0 / bin_width),
            histtype="step",
            linewidth=2.0,
            color="tab:blue",
            label="Subhalo Mass",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_{\rm sub}/M_\odot)$")
    ax.set_ylabel(r"$dN/d\log_{10}M$")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _plot_subhalo_radial_distribution(subhalo_df: pd.DataFrame, path: Path) -> None:
    radius = _finite_nonnegative_column(subhalo_df, "radius_arcsec")
    bins = _subhalo_linear_bins(radius)
    if bins is None:
        _write_placeholder_plot(
            path,
            "Subhalo radial distribution",
            "No finite fitted subhalo radii are available.",
        )
        return
    bin_width = float(np.mean(np.diff(bins)))
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        _write_placeholder_plot(
            path,
            "Subhalo radial distribution",
            "Subhalo radial bins are invalid.",
        )
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        radius,
        bins=bins,
        weights=np.full(radius.size, 1.0 / bin_width),
        histtype="stepfilled",
        color="tab:blue",
        alpha=0.75,
    )
    ax.set_xlabel("cluster-centric radius (arcsec)")
    ax.set_ylabel(r"$dN/dR$ (arcsec$^{-1}$)")
    fig.tight_layout()
    _finish_figure(fig, path, dpi=180, bbox_inches="tight")


def _load_image_catalog_cutout_helpers() -> Any:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("plot_literature_family_cutouts")


def _image_catalog_family_cluster_enabled(args: argparse.Namespace, run_dir: Path) -> bool:
    image_dir = getattr(args, "image_catalog_family_cutout_image_dir", None)
    if image_dir is None or not str(image_dir).strip():
        return False
    stage_name = Path(run_dir).name
    if stage_name in {"stage1_backprojected_centroid_fit", "stage2_free_source_forward_fit"}:
        return True
    return not stage_name.startswith("stage")


def _image_catalog_family_cutout_enabled(args: argparse.Namespace, run_dir: Path) -> bool:
    if not _image_catalog_family_cluster_enabled(args, run_dir):
        return False
    if not bool(getattr(args, "image_catalog_family_cutouts", True)):
        return False
    stage_name = Path(run_dir).name
    if stage_name == "stage2_free_source_forward_fit":
        return True
    if stage_name == "stage1_backprojected_centroid_fit":
        return str(getattr(args, "stage2_forward_mode", "none")) == "none"
    return True


def _infer_image_catalog_cutout_cluster(state: BuildState) -> str:
    text = " ".join(
        [
            str(getattr(state, "run_name", "")),
            str(getattr(state, "par_path", "")),
        ]
    ).lower()
    candidates = (
        ("a2744", ("a2744", "abell2744")),
        ("ares", ("ares",)),
        ("m0416", ("m0416", "macs0416")),
        ("m1206", ("m1206", "macsj1206")),
        ("as1063", ("as1063", "abells1063", "rxcj2248", "a1063")),
        ("hera", ("hera",)),
        ("a370", ("a370", "abell370", "a307")),
        ("m0717", ("m0717", "macs0717")),
        ("m1149", ("m1149", "macs1149")),
    )
    for cluster, tokens in candidates:
        if any(token in text for token in tokens):
            return cluster
    return str(getattr(state, "run_name", "cluster")).split("_", 1)[0].lower() or "cluster"


def _arcsec_to_skycoord(x_arcsec: Any, y_arcsec: Any, reference: tuple[int, float, float]) -> SkyCoord | None:
    try:
        _reference_type, ra0_deg, dec0_deg = reference
        x_value = float(x_arcsec)
        y_value = float(y_arcsec)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x_value) or not np.isfinite(y_value):
        return None
    cos_dec0 = math.cos(math.radians(float(dec0_deg)))
    if cos_dec0 == 0.0:
        return None
    ra_deg = float(ra0_deg) - x_value / (3600.0 * cos_dec0)
    dec_deg = float(dec0_deg) + y_value / 3600.0
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")


def _cutout_npixels_for_image(image: Any, *, cutout_size_arcsec: float) -> int | None:
    if cutout_size_arcsec <= 0.0:
        return None
    pixel_scale = float(getattr(image, "pixel_scale_arcsec", np.nan))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        return None
    return max(1, int(math.ceil(float(cutout_size_arcsec) / pixel_scale)))


def _clamped_cutout_origin(raw_origin: int, npix: int, image_size: int) -> int:
    if image_size >= npix:
        return int(np.clip(raw_origin, 0, image_size - npix))
    return int(np.clip(raw_origin, image_size - npix, 0))


def _cutout_window_origin_xy(
    image: Any,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> tuple[int, int] | None:
    npix = _cutout_npixels_for_image(image, cutout_size_arcsec=cutout_size_arcsec)
    if npix is None:
        return None
    try:
        ny, nx = tuple(int(value) for value in image.shape[:2])
    except (AttributeError, TypeError, ValueError):
        return None
    center_x, center_y = image.wcs.world_to_pixel(center_coord)
    if not all(np.isfinite(value) for value in (center_x, center_y)):
        return None
    half = npix // 2
    raw_x0 = int(round(float(center_x))) - half
    raw_y0 = int(round(float(center_y))) - half
    if raw_x0 >= nx or raw_x0 + npix <= 0 or raw_y0 >= ny or raw_y0 + npix <= 0:
        return None
    return _clamped_cutout_origin(raw_x0, npix, nx), _clamped_cutout_origin(raw_y0, npix, ny)


def _cutout_pixel_xy(
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
) -> tuple[float, float] | None:
    origin = _cutout_window_origin_xy(image, center_coord, cutout_size_arcsec=cutout_size_arcsec)
    if origin is None:
        return None
    target_x, target_y = image.wcs.world_to_pixel(target_coord)
    if not all(np.isfinite(value) for value in (target_x, target_y)):
        return None
    x0, y0 = origin
    return float(target_x) - float(x0), float(target_y) - float(y0)


def _lock_cutout_axis_to_image(ax: plt.Axes, rendered_shape: tuple[int, ...]) -> None:
    if len(rendered_shape) < 2:
        return
    height = int(rendered_shape[0])
    width = int(rendered_shape[1])
    if height <= 0 or width <= 0:
        return
    ax.set_xlim(-0.5, float(width) - 0.5)
    ax.set_ylim(-0.5, float(height) - 0.5)
    ax.set_autoscale_on(False)


def _draw_cutout_circle(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    radius_arcsec: float,
    linestyle: str = "-",
    linewidth: float = 0.9,
    alpha: float = 0.85,
    zorder: float = 8,
) -> None:
    pixel = _cutout_pixel_xy(image, center_coord, target_coord, cutout_size_arcsec=cutout_size_arcsec)
    if pixel is None:
        return
    x, y = pixel
    height, width = rendered_shape
    radius = float(radius_arcsec) / float(image.pixel_scale_arcsec)
    if x < -radius or x > width - 1 + radius or y < -radius or y > height - 1 + radius:
        return
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            clip_on=True,
        )
    )


def _draw_cutout_segment(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    start_coord: SkyCoord,
    end_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    linewidth: float = 0.9,
    linestyle: str = "-",
    alpha: float = 0.85,
    zorder: float = 8,
) -> tuple[float, float] | None:
    start_pixel = _cutout_pixel_xy(image, center_coord, start_coord, cutout_size_arcsec=cutout_size_arcsec)
    end_pixel = _cutout_pixel_xy(image, center_coord, end_coord, cutout_size_arcsec=cutout_size_arcsec)
    if start_pixel is None or end_pixel is None:
        return None
    sx, sy = start_pixel
    ex, ey = end_pixel
    height, width = rendered_shape
    if max(sx, ex) < 0 or min(sx, ex) > width - 1 or max(sy, ey) < 0 or min(sy, ey) > height - 1:
        return None
    ax.plot(
        [sx, ex],
        [sy, ey],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        clip_on=True,
    )
    return ex, ey


def _arcsec_curve_values(value: Any) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return np.asarray([], dtype=float)
        try:
            return np.asarray(json.loads(text), dtype=float).reshape(-1)
        except (json.JSONDecodeError, TypeError, ValueError):
            return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _image_catalog_finite_polyline_points(x_values: Any, y_values: Any) -> np.ndarray:
    x_array = _arcsec_curve_values(x_values)
    y_array = _arcsec_curve_values(y_values)
    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.empty((0, 2), dtype=float)
    points = np.column_stack([x_array, y_array])
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] <= 1:
        return points
    keep = [0]
    for index in range(1, points.shape[0]):
        if float(np.linalg.norm(points[index] - points[keep[-1]])) > 0.0:
            keep.append(index)
    return points[np.asarray(keep, dtype=int)]


def _image_catalog_polyline_cumulative_arclength(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=float)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _image_catalog_project_point_to_polyline(
    point: np.ndarray,
    points: np.ndarray,
    *,
    preferred_s: float | None = None,
) -> dict[str, Any] | None:
    point_array = np.asarray(point, dtype=float).reshape(2)
    if points.shape[0] < 2 or not np.isfinite(point_array).all():
        return None
    segment = points[1:] - points[:-1]
    segment_len2 = np.sum(np.square(segment), axis=1)
    valid = segment_len2 > 0.0
    if not np.any(valid):
        return None
    start = points[:-1][valid]
    direction = segment[valid]
    valid_len2 = segment_len2[valid]
    t_values = np.clip(np.sum((point_array - start) * direction, axis=1) / valid_len2, 0.0, 1.0)
    projections = start + t_values[:, None] * direction
    distances = np.linalg.norm(projections - point_array, axis=1)
    cumulative = _image_catalog_polyline_cumulative_arclength(points)
    valid_indices = np.flatnonzero(valid)
    s_values = cumulative[valid_indices] + t_values * np.sqrt(valid_len2)
    min_distance = float(np.min(distances))
    tolerance = max(1.0e-9, 1.0e-6 * max(1.0, min_distance))
    candidates = np.flatnonzero(distances <= min_distance + tolerance)
    if preferred_s is not None and np.isfinite(preferred_s):
        best = int(candidates[np.argmin(np.abs(s_values[candidates] - float(preferred_s)))])
    else:
        best = int(candidates[0])
    return {
        "point": projections[best],
        "distance": float(distances[best]),
        "s": float(s_values[best]),
    }


def _image_catalog_point_at_polyline_s(points: np.ndarray, cumulative: np.ndarray, s_value: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.asarray([np.nan, np.nan], dtype=float)
    if points.shape[0] == 1:
        return points[0].copy()
    s_clamped = float(np.clip(s_value, cumulative[0], cumulative[-1]))
    index = int(np.searchsorted(cumulative, s_clamped, side="right") - 1)
    index = max(0, min(index, points.shape[0] - 2))
    segment_length = float(cumulative[index + 1] - cumulative[index])
    if segment_length <= 0.0:
        return points[index].copy()
    t_value = (s_clamped - float(cumulative[index])) / segment_length
    return points[index] + t_value * (points[index + 1] - points[index])


def _image_catalog_polyline_between_s(points: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    if points.shape[0] < 2 or not (np.isfinite(start_s) and np.isfinite(end_s)):
        return np.empty((0, 2), dtype=float)
    cumulative = _image_catalog_polyline_cumulative_arclength(points)
    start = _image_catalog_point_at_polyline_s(points, cumulative, start_s)
    end = _image_catalog_point_at_polyline_s(points, cumulative, end_s)
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return np.empty((0, 2), dtype=float)
    if abs(float(end_s) - float(start_s)) <= 1.0e-10:
        return np.vstack([start, end])
    low = min(float(start_s), float(end_s))
    high = max(float(start_s), float(end_s))
    interior_mask = (cumulative > low + 1.0e-10) & (cumulative < high - 1.0e-10)
    interior = points[interior_mask]
    if float(end_s) < float(start_s):
        interior = interior[::-1]
    return np.vstack([start, interior, end])


def _image_catalog_arc_support_geometry(row: pd.Series) -> dict[str, Any] | None:
    observed = _image_catalog_finite_arcsec_pair(row, "x_obs_arcsec", "y_obs_arcsec")
    anchor = _image_catalog_finite_arcsec_pair(row, "arc_support_anchor_x_arcsec", "arc_support_anchor_y_arcsec")
    if observed is None or anchor is None:
        return None
    points = _image_catalog_finite_polyline_points(
        row.get("arc_support_curve_x_arcsec", "[]"),
        row.get("arc_support_curve_y_arcsec", "[]"),
    )
    if points.shape[0] < 2:
        return None
    observed_point = np.asarray(observed, dtype=float)
    anchor_point = np.asarray(anchor, dtype=float)
    anchor_projection = _image_catalog_project_point_to_polyline(anchor_point, points)
    if anchor_projection is None:
        return None
    closest_projection = _image_catalog_project_point_to_polyline(
        observed_point,
        points,
        preferred_s=float(anchor_projection["s"]),
    )
    if closest_projection is None:
        return None
    tangential_curve = _image_catalog_polyline_between_s(
        points,
        float(closest_projection["s"]),
        float(anchor_projection["s"]),
    )
    if tangential_curve.shape[0] < 2:
        tangential_curve = np.vstack([closest_projection["point"], anchor_projection["point"]])
    return {
        "observed_arcsec": observed_point,
        "closest_arcsec": np.asarray(closest_projection["point"], dtype=float),
        "anchor_arcsec": anchor_point,
        "anchor_curve_arcsec": np.asarray(anchor_projection["point"], dtype=float),
        "residual_arcsec": float(closest_projection["distance"]),
        "closest_s_arcsec": float(closest_projection["s"]),
        "anchor_s_arcsec": float(anchor_projection["s"]),
        "tangential_curve_arcsec": np.asarray(tangential_curve, dtype=float),
    }


def _draw_cutout_polyline_arcsec(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    x_arcsec: Any,
    y_arcsec: Any,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    color: str,
    linewidth: float = 0.9,
    linestyle: str = "-",
    alpha: float = 0.85,
    zorder: float = 8,
) -> bool:
    x_values = _arcsec_curve_values(x_arcsec)
    y_values = _arcsec_curve_values(y_arcsec)
    if x_values.size < 2 or x_values.shape != y_values.shape:
        return False
    pixels: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        coord = _arcsec_to_skycoord(float(x_value), float(y_value), reference)
        if coord is None:
            pixels.append((np.nan, np.nan))
            continue
        pixel = _cutout_pixel_xy(image, center_coord, coord, cutout_size_arcsec=cutout_size_arcsec)
        pixels.append(pixel if pixel is not None else (np.nan, np.nan))
    pixel_array = np.asarray(pixels, dtype=float)
    finite = np.isfinite(pixel_array).all(axis=1)
    if np.sum(finite) < 2:
        return False
    height, width = rendered_shape
    visible = (
        finite
        & (pixel_array[:, 0] >= 0.0)
        & (pixel_array[:, 0] <= width - 1)
        & (pixel_array[:, 1] >= 0.0)
        & (pixel_array[:, 1] <= height - 1)
    )
    if not np.any(visible):
        return False
    finite_indices = np.flatnonzero(finite)
    runs = np.split(finite_indices, np.where(np.diff(finite_indices) != 1)[0] + 1)
    drawn = False
    for run in runs:
        if run.size < 2:
            continue
        ax.plot(
            pixel_array[run, 0],
            pixel_array[run, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            clip_on=True,
        )
        drawn = True
    return drawn


def _draw_image_catalog_arc_support_curve(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    alpha: float = 0.92,
    linewidth: float = 1.05,
    zorder: float = 10,
) -> bool:
    if not _image_catalog_draw_arc_anchor_overlays(row):
        return False
    return _draw_cutout_polyline_arcsec(
        ax,
        image,
        center_coord,
        row.get("arc_support_curve_x_arcsec", "[]"),
        row.get("arc_support_curve_y_arcsec", "[]"),
        reference,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("ARC_RECOVERED"),
        linewidth=linewidth,
        linestyle="--",
        alpha=alpha,
        zorder=zorder,
    )


def _draw_cutout_direction_frame(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    try:
        x0 = float(row["x_obs_arcsec"])
        y0 = float(row["y_obs_arcsec"])
        tx = float(row.get("arc_critical_direction_x", np.nan))
        ty = float(row.get("arc_critical_direction_y", np.nan))
        nx = float(row.get("arc_noncritical_direction_x", np.nan))
        ny = float(row.get("arc_noncritical_direction_y", np.nan))
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite([x0, y0, tx, ty, nx, ny])):
        return
    half_length = min(1.6, max(0.55, 0.16 * float(cutout_size_arcsec)))
    for label, vx, vy, color in (("T", tx, ty, "#00e5ff"), ("N", nx, ny, "#ff4da6")):
        start_coord = _arcsec_to_skycoord(x0 - half_length * vx, y0 - half_length * vy, reference)
        end_coord = _arcsec_to_skycoord(x0 + half_length * vx, y0 + half_length * vy, reference)
        if start_coord is None or end_coord is None:
            continue
        end_pixel = _draw_cutout_segment(
            ax,
            image,
            center_coord,
            start_coord,
            end_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            color=color,
            linewidth=1.05,
            alpha=0.9,
            zorder=11,
        )
        if end_pixel is not None:
            ax.text(
                end_pixel[0],
                end_pixel[1],
                label,
                color=color,
                fontsize=5.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=12,
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.35},
            )


def _format_cutout_float(value: Any, precision: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(numeric):
        return "na"
    return f"{numeric:.{precision}g}"


def _format_image_catalog_diagnostic_label(row: pd.Series) -> str:
    status = _image_catalog_compact_status_text(row, _image_catalog_observed_panel_status(row))
    point_recovered = _image_catalog_point_recovered(row)
    arc_recovered = _image_catalog_arc_recovered(row)
    point_status = "recovered" if point_recovered else ("arc recovered" if arc_recovered else "missed")
    arc_status = "recovered" if arc_recovered else ("supported" if _image_catalog_arc_candidate_supported(row) else "not supported")
    lines = [
        f"{row.get('image_label', '')} z={_format_cutout_float(row.get('z_source', row.get('catalog_z')), 3)}",
        (
            f"point={point_status} "
            f"r_point={_format_cutout_float(row.get('point_image_residual_arcsec', row.get('image_residual_arcsec')))} "
        ),
        (
            f"arc={arc_status} "
            f"r_arc={_format_cutout_float(row.get('arc_candidate_image_residual_arcsec', row.get('arc_curve_distance_arcsec')))}"
            f" p_arc={_format_cutout_float(row.get('p_arc'))}"
        ),
        (
            f"N={_format_cutout_float(row.get('arc_noncritical_direction_residual_arcsec'))} "
            f"T={_format_cutout_float(row.get('arc_critical_direction_residual_arcsec'))}"
        ),
        (
            f"curve={_format_cutout_float(row.get('arc_curve_distance_arcsec'))} "
            f"s={_format_cutout_float(row.get('arc_curve_arclength_arcsec'))}"
        ),
        (
            f"s={_format_cutout_float(row.get('arc_s_min'))}/"
            f"{_format_cutout_float(row.get('arc_s_max'))} "
            f"det={_format_cutout_float(row.get('arc_detA'))}"
        ),
    ]
    lines.append(status)
    return "\n".join(lines)


def _image_catalog_normalized_family_key(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        numeric = np.nan
    if np.isfinite(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    return text.lower()


def _image_catalog_normalized_label_key(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    family_text, separator, image_suffix = text.partition(".")
    if separator:
        return f"{_image_catalog_normalized_family_key(family_text)}.{image_suffix.strip().lower()}"
    return text.lower()


def _image_catalog_cutout_rows(state: BuildState, image_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in state.family_data:
        for label, x_obs, y_obs in zip(family.image_labels, family.x_obs, family.y_obs):
            coord = _arcsec_to_skycoord(float(x_obs), float(y_obs), state.reference)
            if coord is None:
                continue
            rows.append(
                {
                    "family_id": str(family.family_id),
                    "image_label": str(label),
                    "x_obs_arcsec": float(x_obs),
                    "y_obs_arcsec": float(y_obs),
                    "ra": float(coord.ra.deg),
                    "dec": float(coord.dec.deg),
                    "z_source": float(family.z_source),
                    "effective_z_source": float(family.effective_z_source),
                    "catalog_z": float(family.z_source),
                }
            )
    base = pd.DataFrame(rows)
    if base.empty or image_df is None or image_df.empty:
        return base
    diagnostics = image_df.copy()
    base["__family_key"] = base["family_id"].map(_image_catalog_normalized_family_key)
    base["__label_key"] = base["image_label"].map(_image_catalog_normalized_label_key)
    if "family_id" in diagnostics.columns:
        diagnostics["__family_key"] = diagnostics["family_id"].map(_image_catalog_normalized_family_key)
    else:
        diagnostics["__family_key"] = diagnostics.get("image_label", pd.Series("", index=diagnostics.index)).map(
            lambda value: _image_catalog_normalized_label_key(value).split(".", 1)[0]
        )
    if "image_label" in diagnostics.columns:
        diagnostics["__label_key"] = diagnostics["image_label"].map(_image_catalog_normalized_label_key)
    else:
        diagnostics["__label_key"] = ""
    diagnostics = diagnostics.drop_duplicates(["__family_key", "__label_key"], keep="first")
    diagnostic_payload = diagnostics.drop(
        columns=[
            column
            for column in ("family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec")
            if column in diagnostics.columns
        ]
    )
    merged = base.merge(
        diagnostic_payload,
        on=["__family_key", "__label_key"],
        how="left",
        suffixes=("", "_diagnostic"),
    )
    return merged.drop(columns=["__family_key", "__label_key"])


def _image_catalog_extra_cutout_rows(state: BuildState, extra_image_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "family_id",
        "extra_image_index",
        "image_label",
        "x_model_arcsec",
        "y_model_arcsec",
        "x_center_arcsec",
        "y_center_arcsec",
        "ra",
        "dec",
        "z_source",
        "effective_z_source",
        "catalog_z",
        "image_recovery_status",
    ]
    if extra_image_df is None or extra_image_df.empty:
        return pd.DataFrame(columns=columns)
    family_by_id = {str(family.family_id): family for family in getattr(state, "family_data", [])}
    family_by_normalized_id = {
        _image_catalog_normalized_family_key(family_id): family
        for family_id, family in family_by_id.items()
    }
    rows: list[dict[str, Any]] = []
    for _, row in extra_image_df.iterrows():
        family_id = str(row.get("family_id", ""))
        family = family_by_id.get(family_id) or family_by_normalized_id.get(_image_catalog_normalized_family_key(family_id))
        if family is not None:
            family_id = str(family.family_id)
        try:
            x_model = float(row.get("x_model_arcsec"))
            y_model = float(row.get("y_model_arcsec"))
        except (TypeError, ValueError):
            continue
        coord = _arcsec_to_skycoord(x_model, y_model, state.reference)
        if coord is None:
            continue

        def redshift_value(column: str, fallback: Any) -> float:
            try:
                value = float(row.get(column, fallback))
            except (TypeError, ValueError):
                value = float(fallback)
            return value if np.isfinite(value) else float(fallback)

        family_z = redshift_value("z_source", getattr(family, "z_source", np.nan))
        effective_z = redshift_value("effective_z_source", getattr(family, "effective_z_source", family_z))
        extra_index = row.get("extra_image_index", len(rows) + 1)
        extra_label = f"{family_id}.extra{extra_index}"
        extra_row = row.to_dict()
        extra_row.update(
            {
                "family_id": family_id,
                "extra_image_index": extra_index,
                "image_label": str(extra_label),
                "x_model_arcsec": x_model,
                "y_model_arcsec": y_model,
                "x_center_arcsec": x_model,
                "y_center_arcsec": y_model,
                "ra": float(coord.ra.deg),
                "dec": float(coord.dec.deg),
                "z_source": float(family_z),
                "effective_z_source": float(effective_z),
                "catalog_z": float(family_z),
                "image_recovery_status": "extra",
            }
        )
        rows.append(extra_row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _image_catalog_status_color(status: str) -> str:
    normalized = str(status).strip().upper()
    if normalized in {"POINT_RECOVERED", "POINT_SUPPORTED", "OBSERVED"}:
        return "#4da3ff"
    if normalized in {"ARC_RECOVERED", "ARC_SUPPORTED"}:
        return "#ffd54f"
    if normalized in {"MISSED", "NOT_RECOVERED"}:
        return "#ff4d5e"
    if normalized == "EXTRA":
        return "tab:purple"
    if normalized == "MODEL":
        return "#4caf50"
    if normalized == "FAMILY":
        return "#00e5ff"
    return "#f5f5f5"


def _image_catalog_status_display_text(status: str) -> str:
    normalized = str(status).strip().upper()
    labels = {
        "POINT_RECOVERED": "point recovered",
        "POINT_SUPPORTED": "point recovered",
        "OBSERVED": "point recovered",
        "ARC_RECOVERED": "arc recovered",
        "ARC_SUPPORTED": "arc recovered",
        "MISSED": "not recovered",
        "NOT_RECOVERED": "not recovered",
        "EXTRA": "extra",
    }
    return labels.get(normalized, str(status).strip().replace("_", " ").lower())


def _image_catalog_finite_arcsec_pair(row: pd.Series, x_column: str, y_column: str) -> tuple[float, float] | None:
    if x_column not in row or y_column not in row:
        return None
    try:
        x_value = float(row.get(x_column, np.nan))
        y_value = float(row.get(y_column, np.nan))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return None
    return x_value, y_value


def _image_catalog_has_finite_model_position(row: pd.Series) -> bool:
    return _image_catalog_finite_arcsec_pair(row, "x_model_arcsec", "y_model_arcsec") is not None


def _image_catalog_truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _image_catalog_point_recovered(row: pd.Series) -> bool:
    status = str(row.get("image_recovery_status", "")).strip().lower()
    if status == "recovered":
        return True
    if _image_catalog_truthy(row.get("exact_image_prediction_failed", False)):
        return False
    if status in {"not_recovered", "missing", "missed", "failed"}:
        return False
    return _image_catalog_has_finite_model_position(row)


def _image_catalog_preferred_point_recovered(row: pd.Series) -> bool:
    return _image_catalog_point_recovered(row)


def _image_catalog_arc_recovered(row: pd.Series) -> bool:
    return _image_catalog_truthy(row.get("arc_supported", False)) or _image_catalog_arc_candidate_supported(row)


def _image_catalog_arc_recovery_p_arc_threshold(row: pd.Series) -> float:
    try:
        threshold = float(row.get("arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD))
    except (TypeError, ValueError):
        threshold = float(CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        return float(CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
    return threshold


def _image_catalog_arc_candidate_supported(row: pd.Series) -> bool:
    try:
        p_arc = float(row.get("p_arc", np.nan))
    except (TypeError, ValueError):
        p_arc = np.nan
    return bool(np.isfinite(p_arc) and p_arc >= _image_catalog_arc_recovery_p_arc_threshold(row))


def _image_catalog_effective_recovery_status(row: pd.Series) -> str:
    if _image_catalog_arc_recovered(row):
        return "ARC_RECOVERED"
    if _image_catalog_point_recovered(row):
        return "POINT_RECOVERED"
    return "MISSED"


def _image_catalog_effective_recovery_statuses(data: pd.DataFrame) -> np.ndarray:
    if data is None or data.empty:
        return np.asarray([], dtype=object)
    return np.asarray([_image_catalog_effective_recovery_status(row) for _, row in data.iterrows()], dtype=object)


def _image_catalog_observed_panel_status(row: pd.Series) -> str:
    return _image_catalog_effective_recovery_status(row)


def _image_catalog_draw_arc_anchor_overlays(row: pd.Series) -> bool:
    return _image_catalog_arc_recovered(row)


def _image_catalog_display_model_arcsec(row: pd.Series) -> tuple[float, float] | None:
    return _image_catalog_finite_arcsec_pair(row, "x_model_arcsec", "y_model_arcsec")


def _image_catalog_display_model_coord(row: pd.Series, reference: tuple[int, float, float]) -> SkyCoord | None:
    pair = _image_catalog_display_model_arcsec(row)
    if pair is None:
        return None
    return _arcsec_to_skycoord(pair[0], pair[1], reference)


def _finite_image_catalog_points(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.empty((0, 2), dtype=float)
    points = [np.asarray(array, dtype=float).reshape(-1, 2) for array in arrays if np.asarray(array).size]
    if not points:
        return np.empty((0, 2), dtype=float)
    stacked = np.vstack(points)
    return stacked[np.isfinite(stacked).all(axis=1)]


def _image_catalog_xy_points(data: pd.DataFrame, x_column: str, y_column: str) -> np.ndarray:
    if data.empty or x_column not in data.columns or y_column not in data.columns:
        return np.empty((0, 2), dtype=float)
    x_values = pd.to_numeric(data[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(data[y_column], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([x_values, y_values])


def _image_catalog_arc_anchor_overlay_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    mask = np.asarray([_image_catalog_draw_arc_anchor_overlays(row) for _, row in data.iterrows()], dtype=bool)
    return data.loc[mask]


IMAGE_CATALOG_DETAIL_COLUMNS = 3
IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC = 40.0
IMAGE_CATALOG_OVERVIEW_LABEL_FONT_SIZE = 8.6
IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE = 8.0
IMAGE_CATALOG_STATUS_LABEL_FONT_SIZE = 8.0
IMAGE_CATALOG_MARKER_LABEL_FONT_SIZE = 6.4
IMAGE_CATALOG_LEGEND_FONT_SIZE = 12.0
IMAGE_CATALOG_TANGENTIAL_CRITICAL_COLOR = "#ffd54f"
IMAGE_CATALOG_RADIAL_CRITICAL_COLOR = "#ff4da6"


def _image_catalog_geometry_from_points(
    points: np.ndarray,
    *,
    minimum_side_arcsec: float = IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC,
) -> tuple[float, float, float]:
    finite = _finite_image_catalog_points(points)
    if finite.size == 0:
        return 0.0, 0.0, float(minimum_side_arcsec)
    x_min, y_min = np.min(finite, axis=0)
    x_max, y_max = np.max(finite, axis=0)
    span = float(max(x_max - x_min, y_max - y_min))
    padding = max(2.0, 0.15 * span)
    cutout_size = max(float(minimum_side_arcsec), span + 2.0 * padding)
    return float(0.5 * (x_min + x_max)), float(0.5 * (y_min + y_max)), float(cutout_size)


def _image_catalog_arc_support_curve_points(data: pd.DataFrame) -> np.ndarray:
    if data.empty:
        return np.empty((0, 2), dtype=float)
    curves: list[np.ndarray] = []
    for _, row in data.iterrows():
        if not _image_catalog_draw_arc_anchor_overlays(row):
            continue
        curve = _image_catalog_finite_polyline_points(
            row.get("arc_support_curve_x_arcsec", "[]"),
            row.get("arc_support_curve_y_arcsec", "[]"),
        )
        if curve.shape[0]:
            curves.append(curve)
    if not curves:
        return np.empty((0, 2), dtype=float)
    return np.vstack(curves)


def _image_catalog_cluster_overview_geometry(
    catalog_df: pd.DataFrame,
    extra_df: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    return _image_catalog_overview_geometry(
        catalog_df,
        extra_df if extra_df is not None else pd.DataFrame(),
        IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC,
    )


def _image_catalog_overview_geometry(
    observed: pd.DataFrame,
    extras: pd.DataFrame,
    default_cutout_size_arcsec: float,
) -> tuple[float, float, float]:
    arc_anchor_observed = _image_catalog_arc_anchor_overlay_rows(observed)
    point_model_observed = observed.loc[
        [
            (not _image_catalog_arc_recovered(row))
            and (_image_catalog_point_recovered(row) or not _image_catalog_draw_arc_anchor_overlays(row))
            for _, row in observed.iterrows()
        ]
    ] if not observed.empty else observed
    points = _finite_image_catalog_points(
        _image_catalog_xy_points(observed, "x_obs_arcsec", "y_obs_arcsec"),
        _image_catalog_xy_points(point_model_observed, "x_model_arcsec", "y_model_arcsec"),
        _image_catalog_xy_points(arc_anchor_observed, "arc_support_anchor_x_arcsec", "arc_support_anchor_y_arcsec"),
        _image_catalog_arc_support_curve_points(arc_anchor_observed),
        _image_catalog_xy_points(extras, "x_model_arcsec", "y_model_arcsec"),
    )
    return _image_catalog_geometry_from_points(
        points,
        minimum_side_arcsec=max(float(default_cutout_size_arcsec), IMAGE_CATALOG_MIN_OVERVIEW_SIZE_ARCSEC),
    )


def _image_catalog_family_block_layout(detail_count: int, detail_cols: int) -> dict[str, int]:
    detail_cols = max(1, int(detail_cols))
    detail_count = max(1, int(detail_count))
    overview_units = detail_cols
    detail_row_count = max(1, int(math.ceil(float(detail_count) / float(detail_cols))))
    return {
        "overview_units": int(overview_units),
        "detail_row_count": int(detail_row_count),
        "layout_rowspan": int(overview_units + detail_row_count),
    }


def _image_catalog_family_cutout_blocks(
    state: BuildState,
    catalog_df: pd.DataFrame,
    extra_df: pd.DataFrame | None,
    *,
    detail_cols: int,
    default_cutout_size_arcsec: float,
) -> list[dict[str, Any]]:
    detail_cols = max(1, int(detail_cols))
    if extra_df is None:
        extra_df = pd.DataFrame()
    catalog_family_ids = set(catalog_df.get("family_id", pd.Series(dtype=object)).astype(str))
    extra_family_ids = set(extra_df.get("family_id", pd.Series(dtype=object)).astype(str)) if not extra_df.empty else set()
    available_family_ids = catalog_family_ids | extra_family_ids
    family_ids = [str(family.family_id) for family in getattr(state, "family_data", []) if str(family.family_id) in available_family_ids]
    blocks: list[dict[str, Any]] = []
    family_by_id = {str(family.family_id): family for family in getattr(state, "family_data", [])}
    for family_id in family_ids:
        family = family_by_id.get(family_id)
        observed = (
            catalog_df.loc[catalog_df["family_id"].astype(str) == family_id].reset_index(drop=True)
            if not catalog_df.empty and "family_id" in catalog_df.columns
            else pd.DataFrame()
        )
        extras = (
            extra_df.loc[extra_df["family_id"].astype(str) == family_id].reset_index(drop=True)
            if not extra_df.empty and "family_id" in extra_df.columns
            else pd.DataFrame()
        )
        detail_panels: list[dict[str, Any]] = []
        for panel_index, (_, image_row) in enumerate(observed.iterrows()):
            row = image_row.to_dict()
            row.update(
                {
                    "panel_kind": "observed",
                    "panel_status": _image_catalog_observed_panel_status(image_row),
                    "panel_index": int(panel_index),
                    "x_center_arcsec": float(image_row.get("x_obs_arcsec", np.nan)),
                    "y_center_arcsec": float(image_row.get("y_obs_arcsec", np.nan)),
                    "cutout_size_arcsec": float(default_cutout_size_arcsec),
                }
            )
            detail_panels.append(row)
        for extra_offset, (_, extra_row) in enumerate(extras.iterrows(), start=len(detail_panels)):
            row = extra_row.to_dict()
            row.update(
                {
                    "panel_kind": "extra",
                    "panel_status": "EXTRA",
                    "panel_index": int(extra_offset),
                    "x_center_arcsec": float(extra_row.get("x_center_arcsec", extra_row.get("x_model_arcsec", np.nan))),
                    "y_center_arcsec": float(extra_row.get("y_center_arcsec", extra_row.get("y_model_arcsec", np.nan))),
                    "cutout_size_arcsec": float(default_cutout_size_arcsec),
                }
            )
            detail_panels.append(row)
        detail_count = max(1, len(detail_panels))
        layout = _image_catalog_family_block_layout(detail_count, detail_cols)
        overview_x, overview_y, overview_size = _image_catalog_overview_geometry(
            observed,
            extras,
            float(default_cutout_size_arcsec),
        )
        if family is not None and hasattr(family, "z_source"):
            z_source = float(getattr(family, "z_source"))
        elif not observed.empty and "z_source" in observed.columns:
            z_source = float(observed["z_source"].iloc[0])
        elif not extras.empty and "z_source" in extras.columns:
            z_source = float(extras["z_source"].iloc[0])
        else:
            z_source = np.nan
        if family is not None and hasattr(family, "effective_z_source"):
            effective_z = float(getattr(family, "effective_z_source"))
        elif not observed.empty and "effective_z_source" in observed.columns:
            effective_z = float(observed["effective_z_source"].iloc[0])
        elif not extras.empty and "effective_z_source" in extras.columns:
            effective_z = float(extras["effective_z_source"].iloc[0])
        else:
            effective_z = z_source
        blocks.append(
            {
                "family_id": family_id,
                "family": family,
                "z_source": z_source,
                "effective_z_source": effective_z,
                "observed": observed,
                "extras": extras,
                "detail_panels": detail_panels,
                "detail_cols": detail_cols,
                "overview_rowspan": int(layout["layout_rowspan"]),
                "overview_units": int(layout["overview_units"]),
                "detail_row_count": int(layout["detail_row_count"]),
                "layout_rowspan": int(layout["layout_rowspan"]),
                "overview_center_x_arcsec": float(overview_x),
                "overview_center_y_arcsec": float(overview_y),
                "overview_cutout_size_arcsec": float(overview_size),
            }
        )
    return blocks


def _image_catalog_family_block_pages(blocks: list[dict[str, Any]], row_slot_budget: int) -> list[list[dict[str, Any]]]:
    row_slot_budget = max(1, int(row_slot_budget))
    pages: list[list[dict[str, Any]]] = []
    current_page: list[dict[str, Any]] = []
    current_rows = 0
    for block in blocks:
        block_rows = max(1, int(block.get("layout_rowspan", block.get("overview_rowspan", 1))))
        if current_page and current_rows + block_rows > row_slot_budget:
            pages.append(current_page)
            current_page = []
            current_rows = 0
        current_page.append(block)
        current_rows += block_rows
    if current_page:
        pages.append(current_page)
    return pages


def _draw_image_catalog_critical_lines(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    center_x_arcsec: Any,
    center_y_arcsec: Any,
    reference: tuple[int, float, float],
    model: Any,
    kwargs_lens: list[dict[str, float]],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> None:
    try:
        x0 = float(center_x_arcsec)
        y0 = float(center_y_arcsec)
    except (TypeError, ValueError):
        return
    if not np.isfinite(x0) or not np.isfinite(y0):
        return
    half = 0.62 * float(cutout_size_arcsec)
    grid_scale = max(0.05, min(0.18, float(cutout_size_arcsec) / 80.0))
    n_grid = max(12, int(round(2.0 * half / grid_scale)) + 1)
    x_axis = np.linspace(x0 - half, x0 + half, n_grid)
    y_axis = np.linspace(y0 - half, y0 + half, n_grid)
    try:
        contours = _critical_curve_caustics(
            model,
            kwargs_lens,
            x_axis,
            y_axis,
            include_tangential=True,
            include_radial=True,
        )
    except Exception:
        return
    colors = {
        "tangential": IMAGE_CATALOG_TANGENTIAL_CRITICAL_COLOR,
        "radial": IMAGE_CATALOG_RADIAL_CRITICAL_COLOR,
    }
    linestyles = {"tangential": "-", "radial": "--"}
    for contour in contours:
        crit_x = np.asarray(contour.get("critical_x", []), dtype=float)
        crit_y = np.asarray(contour.get("critical_y", []), dtype=float)
        if crit_x.size < 2 or crit_x.shape != crit_y.shape:
            continue
        pixels: list[tuple[float, float]] = []
        for x_value, y_value in zip(crit_x, crit_y):
            coord = _arcsec_to_skycoord(x_value, y_value, reference)
            if coord is None:
                pixels.append((np.nan, np.nan))
                continue
            pixel = _cutout_pixel_xy(image, center_coord, coord, cutout_size_arcsec=cutout_size_arcsec)
            pixels.append(pixel if pixel is not None else (np.nan, np.nan))
        pixel_array = np.asarray(pixels, dtype=float)
        finite = np.isfinite(pixel_array).all(axis=1)
        if not finite.any():
            continue
        kind = str(contour.get("kind", "tangential"))
        ax.plot(
            pixel_array[finite, 0],
            pixel_array[finite, 1],
            color=colors.get(kind, "white"),
            linestyle=linestyles.get(kind, "-"),
            linewidth=0.65,
            alpha=0.78,
            zorder=6,
        )


def _draw_image_catalog_status_label(ax: plt.Axes, status: str) -> None:
    ax.text(
        0.965,
        0.965,
        str(status).upper(),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=IMAGE_CATALOG_STATUS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=_image_catalog_status_color(status),
        clip_on=True,
        zorder=22,
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 0.55},
    )


IMAGE_CATALOG_PANEL_TEXT_BBOX = {"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 0.55}


def _image_catalog_compact_status_text(row: pd.Series, fallback: str) -> str:
    if _image_catalog_arc_recovered(row):
        return "arc recovered"
    if _image_catalog_point_recovered(row):
        if _image_catalog_arc_candidate_supported(row):
            return "point recovered (arc supported)"
        return "point recovered"
    value = row.get("arc_recovery_status", fallback)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return _image_catalog_status_display_text(fallback)
    if text.lower() in {"not_recovered", "missing", "missed", "failed"} and str(fallback).strip().upper() in {
        "POINT_RECOVERED",
        "ARC_RECOVERED",
    }:
        return _image_catalog_status_display_text(fallback)
    return text.replace("_", " ").lower()


def _format_image_catalog_extra_label(row: pd.Series) -> str:
    label = str(row.get("image_label", "")).strip()
    if not label:
        label = f"Extra {_format_cutout_float(row.get('extra_image_index'), 3)}"
    return "\n".join(
        [
            f"{label}  extra",
            (
                f"model x={_format_cutout_float(row.get('x_model_arcsec'), 3)} "
                f"y={_format_cutout_float(row.get('y_model_arcsec'), 3)}"
            ),
        ]
    )


def _format_image_catalog_marker_label(row: pd.Series) -> str:
    for column in ("image_label", "family_id"):
        value = row.get(column, "")
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""


def _format_image_catalog_overview_label(block: dict[str, Any]) -> str:
    observed = block["observed"]
    extras = block["extras"]
    point_recovered = sum(1 for _, row in observed.iterrows() if _image_catalog_point_recovered(row))
    arc_recovered = sum(1 for _, row in observed.iterrows() if _image_catalog_arc_recovered(row))
    arc_supported = sum(1 for _, row in observed.iterrows() if _image_catalog_arc_candidate_supported(row))
    return "\n".join(
        [
            f"Family {block['family_id']}  z={_format_cutout_float(block.get('z_source'), 3)}",
            (
                f"Nobs={len(observed)}  Npoint_recovered={point_recovered}  "
                f"Narc_recovered={arc_recovered}  Narc_supported={arc_supported}  Nextra={len(extras)}"
            ),
        ]
    )


def _format_image_catalog_compact_detail_label(row: pd.Series) -> str:
    panel_status = str(row.get("panel_status", "OBSERVED"))
    lines = [
        f"{row.get('image_label', '')}  {_image_catalog_compact_status_text(row, panel_status)}",
    ]
    point_recovered = _image_catalog_point_recovered(row)
    arc_recovered = _image_catalog_arc_recovered(row)
    point_status = "recovered" if point_recovered else ("arc recovered" if arc_recovered else "missed")
    arc_status = "recovered" if arc_recovered else ("supported" if _image_catalog_arc_candidate_supported(row) else "not supported")

    def first_finite(*columns: str) -> float:
        value = np.nan
        for column in columns:
            try:
                candidate = float(row.get(column, np.nan))
            except (TypeError, ValueError):
                candidate = np.nan
            if np.isfinite(candidate):
                value = candidate
                break
        return value

    point_residual = first_finite("point_image_residual_arcsec", "image_residual_arcsec")
    arc_residual = first_finite("arc_candidate_image_residual_arcsec", "arc_curve_distance_arcsec")
    lines.append(f"point={point_status}  r_point={_format_cutout_float(point_residual)}")
    lines.append(
        f"arc={arc_status}  r_arc={_format_cutout_float(arc_residual)}  "
        f"p_arc={_format_cutout_float(row.get('p_arc'))}"
    )
    lines.append(
        f"d_curve={_format_cutout_float(row.get('arc_curve_distance_arcsec'))}  "
        f"N={_format_cutout_float(row.get('arc_noncritical_direction_residual_arcsec'))}  "
        f"T={_format_cutout_float(row.get('arc_critical_direction_residual_arcsec'))}"
    )
    lines.append(
        f"s={_format_cutout_float(row.get('arc_s_min'))}/{_format_cutout_float(row.get('arc_s_max'))}  "
        f"detA={_format_cutout_float(row.get('arc_detA'))}"
    )
    return "\n".join(lines)


def _draw_image_catalog_panel_text(ax: plt.Axes, label: str, *, fontsize: float = IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE) -> None:
    ax.text(
        0.035,
        0.965,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color="white",
        linespacing=0.95,
        clip_on=True,
        zorder=20,
        bbox=IMAGE_CATALOG_PANEL_TEXT_BBOX,
    )


def _image_catalog_legend_handles(*, include_critical_lines: bool = False) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("POINT_RECOVERED"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("POINT_RECOVERED"),
            label="point recovered image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("ARC_RECOVERED"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("ARC_RECOVERED"),
            label="arc recovered image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("MISSED"),
            markeredgewidth=1.25,
            color=_image_catalog_status_color("MISSED"),
            label="missed observed image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="--",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("MODEL"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("MODEL"),
            label="matched model image",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_image_catalog_status_color("EXTRA"),
            markeredgewidth=1.15,
            color=_image_catalog_status_color("EXTRA"),
            label="extra model image",
        ),
        Line2D(
            [0],
            [0],
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linestyle="--",
            linewidth=1.25,
            label="arc-support curve",
        ),
        Line2D([0], [0], color="#bdbdbd", linewidth=0.9, label="observed-to-model residual"),
        Line2D(
            [0],
            [0],
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linewidth=1.25,
            label="tangential arc displacement",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markeredgecolor=_image_catalog_status_color("MODEL"),
            color=_image_catalog_status_color("MODEL"),
            markersize=4.0,
            markeredgewidth=0.9,
            label="linearized arc anchor",
        ),
    ]
    if include_critical_lines:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=IMAGE_CATALOG_TANGENTIAL_CRITICAL_COLOR,
                    linestyle="-",
                    linewidth=1.05,
                    label="tangential critical line",
                ),
                Line2D(
                    [0],
                    [0],
                    color=IMAGE_CATALOG_RADIAL_CRITICAL_COLOR,
                    linestyle="--",
                    linewidth=1.05,
                    label="radial critical line",
                ),
            ]
        )
    return handles


def _add_image_catalog_axis_legend(ax: plt.Axes, *, include_critical_lines: bool = False) -> None:
    legend = ax.legend(
        handles=_image_catalog_legend_handles(include_critical_lines=include_critical_lines),
        loc="lower right",
        ncol=1,
        fontsize=IMAGE_CATALOG_LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.72,
        facecolor="black",
        edgecolor="0.45",
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=0.9,
        borderpad=0.45,
    )
    for text in legend.get_texts():
        text.set_color("white")


def _image_catalog_panel_center(row: pd.Series, reference: tuple[int, float, float]) -> tuple[SkyCoord | None, float, float]:
    try:
        x_center = float(row.get("x_center_arcsec"))
        y_center = float(row.get("y_center_arcsec"))
    except (TypeError, ValueError):
        x_center = y_center = np.nan
    coord = _arcsec_to_skycoord(x_center, y_center, reference)
    return coord, x_center, y_center


def _image_catalog_draw_rgb_cutout(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    center_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    render_config: Any | None = None,
) -> np.ndarray:
    cutouts = {
        str(band): helpers.extract_band_cutout(
            band_images[str(band)],
            center_coord,
            cutout_size_arcsec=cutout_size_arcsec,
        )
        for band in bands
    }
    rgb = helpers.make_rgb_cutout(cutouts, bands=bands, rgb_display=rgb_display)
    helpers.render_rgb_cutout_on_axis(ax, rgb, render_config=render_config)
    return rgb


def _draw_image_catalog_observed_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    target_coord: SkyCoord,
    *,
    status: str,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        target_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color(status),
        radius_arcsec=0.5,
        linewidth=1.6 if str(status).upper() in {"MISSED", "ARC_RECOVERED"} else 1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_marker_label(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    image_row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> bool:
    label = _format_image_catalog_marker_label(image_row)
    if not label:
        return False
    target_coord = _arcsec_to_skycoord(image_row.get("x_obs_arcsec"), image_row.get("y_obs_arcsec"), reference)
    if target_coord is None:
        return False
    pixel = _cutout_pixel_xy(image, center_coord, target_coord, cutout_size_arcsec=cutout_size_arcsec)
    if pixel is None:
        return False
    x_pixel, y_pixel = pixel
    height, width = rendered_shape
    if x_pixel < 0.0 or x_pixel > width - 1 or y_pixel < 0.0 or y_pixel > height - 1:
        return False
    ax.text(
        x_pixel + 4.0,
        y_pixel - 4.0,
        label,
        va="bottom",
        ha="left",
        fontsize=IMAGE_CATALOG_MARKER_LABEL_FONT_SIZE,
        color="white",
        clip_on=True,
        zorder=30,
        bbox=IMAGE_CATALOG_PANEL_TEXT_BBOX,
    )
    return True


def _draw_image_catalog_model_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    model_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        model_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("MODEL"),
        radius_arcsec=0.5,
        linestyle="--",
        linewidth=1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_arc_anchor_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    anchor_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 13,
) -> bool:
    pixel = _cutout_pixel_xy(image, center_coord, anchor_coord, cutout_size_arcsec=cutout_size_arcsec)
    if pixel is None:
        return False
    x_pixel, y_pixel = pixel
    height, width = rendered_shape
    if x_pixel < 0.0 or x_pixel > width - 1 or y_pixel < 0.0 or y_pixel > height - 1:
        return False
    ax.plot(
        [x_pixel],
        [y_pixel],
        marker="x",
        linestyle="None",
        color=_image_catalog_status_color("MODEL"),
        markersize=5.0,
        markeredgewidth=1.2,
        alpha=0.98,
        zorder=zorder,
        clip_on=True,
    )
    return True


def _draw_image_catalog_arc_supported_components(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
) -> bool:
    if not _image_catalog_draw_arc_anchor_overlays(row):
        return False
    geometry = _image_catalog_arc_support_geometry(row)
    if geometry is None:
        return False
    observed_coord = _arcsec_to_skycoord(*geometry["observed_arcsec"], reference)
    closest_coord = _arcsec_to_skycoord(*geometry["closest_arcsec"], reference)
    anchor_coord = _arcsec_to_skycoord(*geometry["anchor_arcsec"], reference)
    if observed_coord is None or closest_coord is None or anchor_coord is None:
        return False
    _draw_cutout_segment(
        ax,
        image,
        center_coord,
        observed_coord,
        closest_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
            color="#bdbdbd",
            linewidth=1.15,
        alpha=0.86,
        zorder=12,
    )
    tangential_curve = np.asarray(geometry["tangential_curve_arcsec"], dtype=float)
    if tangential_curve.shape[0] >= 2:
        _draw_cutout_polyline_arcsec(
            ax,
            image,
            center_coord,
            tangential_curve[:, 0],
            tangential_curve[:, 1],
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            color=_image_catalog_status_color("ARC_RECOVERED"),
            linewidth=1.55,
            alpha=0.96,
            zorder=11,
        )
    _draw_image_catalog_arc_anchor_marker(
        ax,
        image,
        center_coord,
        anchor_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
    )
    return True


def _draw_image_catalog_extra_marker(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    extra_coord: SkyCoord,
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    zorder: float = 10,
) -> None:
    _draw_cutout_circle(
        ax,
        image,
        center_coord,
        extra_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color=_image_catalog_status_color("EXTRA"),
        radius_arcsec=0.5,
        linewidth=1.45,
        alpha=0.95,
        zorder=zorder,
    )


def _draw_image_catalog_observed_row_overlays(
    ax: plt.Axes,
    image: Any,
    center_coord: SkyCoord,
    image_row: pd.Series,
    reference: tuple[int, float, float],
    *,
    cutout_size_arcsec: float,
    rendered_shape: tuple[int, int],
    arc_curve_alpha: float = 0.60,
    arc_curve_linewidth: float = 1.1,
    arc_curve_zorder: float = 7,
    residual_alpha: float = 0.72,
) -> None:
    target_coord = _arcsec_to_skycoord(image_row.get("x_obs_arcsec"), image_row.get("y_obs_arcsec"), reference)
    if target_coord is None:
        return
    status = _image_catalog_observed_panel_status(image_row)
    draw_arc_anchor_overlays = _image_catalog_draw_arc_anchor_overlays(image_row)
    if draw_arc_anchor_overlays:
        _draw_image_catalog_arc_support_curve(
            ax,
            image,
            center_coord,
            image_row,
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
            alpha=arc_curve_alpha,
            linewidth=arc_curve_linewidth,
            zorder=arc_curve_zorder,
        )
    _draw_image_catalog_observed_marker(
        ax,
        image,
        center_coord,
        target_coord,
        status=status,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
    )
    if draw_arc_anchor_overlays:
        _draw_image_catalog_arc_supported_components(
            ax,
            image,
            center_coord,
            image_row,
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rendered_shape,
        )
        if _image_catalog_arc_recovered(image_row):
            return
    model_coord = _image_catalog_display_model_coord(image_row, reference)
    if model_coord is None:
        return
    _draw_image_catalog_model_marker(
        ax,
        image,
        center_coord,
        model_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
    )
    _draw_cutout_segment(
        ax,
        image,
        center_coord,
        target_coord,
        model_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        rendered_shape=rendered_shape,
        color="#bdbdbd",
        linewidth=1.05,
        alpha=residual_alpha,
        zorder=9,
    )


def _draw_image_catalog_cluster_overview_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    catalog_df: pd.DataFrame,
    extra_df: pd.DataFrame | None,
    reference: tuple[int, float, float],
    render_config: Any | None = None,
) -> None:
    if extra_df is None:
        extra_df = pd.DataFrame()
    center_x, center_y, cutout_size_arcsec = _image_catalog_cluster_overview_geometry(catalog_df, extra_df)
    center_coord = _arcsec_to_skycoord(center_x, center_y, reference)
    if center_coord is None:
        ax.set_axis_off()
        return
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        render_config=render_config,
    )
    for _, image_row in catalog_df.iterrows():
        _draw_image_catalog_observed_row_overlays(
            ax,
            display_image,
            center_coord,
            image_row,
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
        _draw_image_catalog_marker_label(
            ax,
            display_image,
            center_coord,
            image_row,
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    for _, extra_row in extra_df.iterrows():
        extra_coord = _arcsec_to_skycoord(extra_row.get("x_model_arcsec"), extra_row.get("y_model_arcsec"), reference)
        if extra_coord is None:
            continue
        _draw_image_catalog_extra_marker(
            ax,
            display_image,
            center_coord,
            extra_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    _add_image_catalog_axis_legend(ax)


def _draw_image_catalog_overview_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    block: dict[str, Any],
    reference: tuple[int, float, float],
    model_pair: tuple[Any, list[dict[str, float]]] | None,
    render_config: Any | None = None,
    include_critical_lines: bool = True,
) -> None:
    center_coord = _arcsec_to_skycoord(
        block["overview_center_x_arcsec"],
        block["overview_center_y_arcsec"],
        reference,
    )
    if center_coord is None:
        ax.set_axis_off()
        return
    cutout_size_arcsec = float(block["overview_cutout_size_arcsec"])
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        render_config=render_config,
    )
    if model_pair is not None and include_critical_lines:
        model, kwargs_lens = model_pair
        _draw_image_catalog_critical_lines(
            ax,
            display_image,
            center_coord,
            block["overview_center_x_arcsec"],
            block["overview_center_y_arcsec"],
            reference,
            model,
            kwargs_lens,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    for _, image_row in block["observed"].iterrows():
        _draw_image_catalog_observed_row_overlays(
            ax,
            display_image,
            center_coord,
            image_row,
            reference,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    for _, extra_row in block["extras"].iterrows():
        extra_coord = _arcsec_to_skycoord(extra_row.get("x_model_arcsec"), extra_row.get("y_model_arcsec"), reference)
        if extra_coord is None:
            continue
        _draw_image_catalog_extra_marker(
            ax,
            display_image,
            center_coord,
            extra_coord,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    _draw_image_catalog_panel_text(ax, _format_image_catalog_overview_label(block), fontsize=IMAGE_CATALOG_OVERVIEW_LABEL_FONT_SIZE)
    _add_image_catalog_axis_legend(ax, include_critical_lines=model_pair is not None and include_critical_lines)


def _draw_image_catalog_detail_panel(
    ax: plt.Axes,
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    row: pd.Series,
    reference: tuple[int, float, float],
    model_pair: tuple[Any, list[dict[str, float]]] | None,
    render_config: Any | None = None,
    include_critical_lines: bool = True,
) -> None:
    center_coord, center_x, center_y = _image_catalog_panel_center(row, reference)
    if center_coord is None:
        ax.set_axis_off()
        return
    cutout_size_arcsec = float(row.get("cutout_size_arcsec", 10.0))
    rgb = _image_catalog_draw_rgb_cutout(
        ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        center_coord,
        cutout_size_arcsec=cutout_size_arcsec,
        render_config=render_config,
    )
    if model_pair is not None and include_critical_lines:
        model, kwargs_lens = model_pair
        _draw_image_catalog_critical_lines(
            ax,
            display_image,
            center_coord,
            center_x,
            center_y,
            reference,
            model,
            kwargs_lens,
            cutout_size_arcsec=cutout_size_arcsec,
            rendered_shape=rgb.shape[:2],
        )
    panel_kind = str(row.get("panel_kind", "observed"))
    panel_status = str(row.get("panel_status", "OBSERVED"))
    if panel_kind == "extra":
        extra_coord = _arcsec_to_skycoord(row.get("x_model_arcsec"), row.get("y_model_arcsec"), reference)
        if extra_coord is not None:
            _draw_image_catalog_extra_marker(
                ax,
                display_image,
                center_coord,
                extra_coord,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        label = _format_image_catalog_extra_label(row)
    else:
        observed_coord = _arcsec_to_skycoord(row.get("x_obs_arcsec"), row.get("y_obs_arcsec"), reference)
        draw_arc_anchor_overlays = _image_catalog_draw_arc_anchor_overlays(row)
        if observed_coord is not None:
            if draw_arc_anchor_overlays:
                _draw_image_catalog_arc_support_curve(
                    ax,
                    display_image,
                    center_coord,
                    row,
                    reference,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                    alpha=0.92,
                    linewidth=1.15,
                    zorder=10,
                )
            _draw_image_catalog_observed_marker(
                ax,
                display_image,
                center_coord,
                observed_coord,
                status=panel_status,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        if draw_arc_anchor_overlays:
            _draw_image_catalog_arc_supported_components(
                ax,
                display_image,
                center_coord,
                row,
                reference,
                cutout_size_arcsec=cutout_size_arcsec,
                rendered_shape=rgb.shape[:2],
            )
        if not _image_catalog_arc_recovered(row):
            model_coord = _image_catalog_display_model_coord(row, reference)
            if model_coord is not None:
                _draw_image_catalog_model_marker(
                    ax,
                    display_image,
                    center_coord,
                    model_coord,
                    cutout_size_arcsec=cutout_size_arcsec,
                    rendered_shape=rgb.shape[:2],
                )
                if observed_coord is not None:
                    _draw_cutout_segment(
                        ax,
                        display_image,
                        center_coord,
                        observed_coord,
                        model_coord,
                        cutout_size_arcsec=cutout_size_arcsec,
                        rendered_shape=rgb.shape[:2],
                        color="#bdbdbd",
                        linewidth=1.05,
                        alpha=0.75,
                        zorder=9,
                    )
        label = _format_image_catalog_compact_detail_label(row)
    _draw_image_catalog_panel_text(ax, label, fontsize=IMAGE_CATALOG_DETAIL_LABEL_FONT_SIZE)


def _image_catalog_critical_line_model_pair(
    evaluator: ClusterJAXEvaluator,
    best_fit_latent: np.ndarray,
    block: dict[str, Any],
    cache: dict[float, tuple[Any, list[dict[str, float]]]],
) -> tuple[Any, list[dict[str, float]]] | None:
    family_id = str(block.get("family_id", ""))
    try:
        z_source = float(block.get("z_source", np.nan))
    except (TypeError, ValueError):
        z_source = np.nan
    if not np.isfinite(z_source):
        _log(None, f"[plot:image_catalog_family_cutouts] skipped critical lines family={family_id}: invalid z_source")
        return None
    if z_source in cache:
        return cache[z_source]
    try:
        exact_models_by_z = getattr(evaluator, "exact_models_by_z", {})
        model = exact_models_by_z.get(z_source) if exact_models_by_z is not None else None
        if model is None:
            model, _ = evaluator._get_exact_model_solver(z_source)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
        kwargs_lens = evaluator._packed_to_kwargs_lens(packed_state)
    except Exception as exc:
        _log(
            None,
            f"[plot:image_catalog_family_cutouts] skipped critical lines family={family_id} z={z_source:g}: {exc}",
        )
        return None
    cache[z_source] = (model, kwargs_lens)
    return cache[z_source]


def _image_catalog_cluster_overview_figure(
    helpers: Any,
    band_images: dict[str, Any],
    bands: Sequence[str],
    rgb_display: Any,
    display_image: Any,
    catalog_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    reference: tuple[int, float, float],
    *,
    detail_cols: int,
    render_config: Any | None = None,
) -> plt.Figure:
    dpi = int(getattr(render_config, "dpi", getattr(helpers, "CUTOUT_FIGURE_DPI", 300)))
    cluster_units = detail_cols
    fig = plt.figure(
        figsize=helpers._figure_size(cluster_units, detail_cols),
        dpi=dpi,
    )
    helpers._style_cutout_figure(fig)
    grid = fig.add_gridspec(cluster_units, detail_cols)
    cluster_ax = fig.add_subplot(grid[:, :])
    helpers._style_cutout_axis(cluster_ax)
    _draw_image_catalog_cluster_overview_panel(
        cluster_ax,
        helpers,
        band_images,
        bands,
        rgb_display,
        display_image,
        catalog_df,
        extra_df,
        reference,
        render_config,
    )
    return fig


def _plot_image_catalog_family_cutouts(
    run_dir: Path,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    image_df: pd.DataFrame,
    extra_image_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    helpers = _load_image_catalog_cutout_helpers()
    image_dir = Path(str(getattr(args, "image_catalog_family_cutout_image_dir")))
    image_scale = str(getattr(args, "image_catalog_family_cutout_image_scale", "60mas"))
    requested_bands = getattr(args, "image_catalog_family_cutout_bands", None)
    bands = tuple(requested_bands) if requested_bands is not None else tuple(getattr(helpers, "DEFAULT_BANDS", ("F435W", "F606W", "F814W")))
    if len(bands) < 3:
        raise ValueError("--image-catalog-family-cutout-bands must contain at least three bands.")
    cutout_size_arcsec = float(getattr(helpers, "DEFAULT_CUTOUT_SIZE_ARCSEC", 10.0))
    cluster = _infer_image_catalog_cutout_cluster(state)
    band_paths = helpers.find_rgb_band_paths(image_dir, cluster=cluster, bands=bands, image_scale=image_scale)
    band_images = helpers.load_rgb_metadata(band_paths, bands=bands)
    render_config = helpers.build_family_cutout_render_config(
        mode=str(getattr(args, "image_catalog_family_cutout_mode", getattr(helpers, "FAMILY_CUTOUT_MODE_FULL", "full"))),
        dpi=getattr(args, "image_catalog_family_cutout_dpi", None),
        max_side_pixels=getattr(args, "image_catalog_family_cutout_max_side_pixels", None),
        critical_lines=str(
            getattr(
                args,
                "image_catalog_family_cutout_critical_lines",
                getattr(helpers, "FAMILY_CUTOUT_CRITICAL_LINES_AUTO", "auto"),
            )
        ),
    )
    render_dpi = int(getattr(render_config, "dpi", getattr(helpers, "CUTOUT_FIGURE_DPI", 300)))
    savefig_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "dpi": render_dpi}
    rgb_kwargs: dict[str, Any] = {}
    for arg_name, kwarg_name in (
        ("image_catalog_family_cutout_rgb_q", "q"),
        ("image_catalog_family_cutout_rgb_stretch", "stretch"),
        ("image_catalog_family_cutout_rgb_minimum", "minimum"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            rgb_kwargs[kwarg_name] = float(value)
    channel_gains = dict(getattr(helpers, "DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS", {"red": 1.0, "green": 1.0, "blue": 1.2}))
    channel_gain_supplied = False
    for arg_name, role in (
        ("image_catalog_family_cutout_rgb_red_gain", "red"),
        ("image_catalog_family_cutout_rgb_green_gain", "green"),
        ("image_catalog_family_cutout_rgb_blue_gain", "blue"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            channel_gains[role] = float(value)
            channel_gain_supplied = True
    if channel_gain_supplied:
        rgb_kwargs["channel_gains"] = channel_gains
    rgb_display = helpers.build_rgb_display(band_images, bands=bands, **rgb_kwargs)
    reference_band = str(getattr(helpers, "DEFAULT_HFF_RGB_REFERENCE_BAND", bands[-1]))
    display_image = band_images[reference_band] if reference_band in band_images else band_images[str(bands[-1])]
    arc_recovery_p_arc_threshold = float(
        getattr(
            evaluator,
            "arc_recovery_p_arc_threshold",
            getattr(args, "arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD),
        )
    )
    if (
        not np.isfinite(arc_recovery_p_arc_threshold)
        or arc_recovery_p_arc_threshold < 0.0
        or arc_recovery_p_arc_threshold > 1.0
    ):
        arc_recovery_p_arc_threshold = float(CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
    catalog_df = _image_catalog_cutout_rows(state, image_df)
    if not catalog_df.empty:
        catalog_df["arc_recovery_p_arc_threshold"] = arc_recovery_p_arc_threshold
    extra_df = _image_catalog_extra_cutout_rows(state, extra_image_df)
    detail_cutouts_enabled = _image_catalog_family_cutout_enabled(args, run_dir)
    if catalog_df.empty and extra_df.empty:
        _write_placeholder_plot(
            _plot_path(run_dir, "image_catalog_family_cluster.pdf"),
            "Image-catalog family cluster",
            "No image catalog rows are available.",
        )
        if detail_cutouts_enabled:
            _write_placeholder_plot(
                _plot_path(run_dir, "image_catalog_family_cutouts.pdf"),
                "Image-catalog family cutouts",
                "No image catalog rows are available.",
            )
        return

    output = _plot_path(run_dir, "image_catalog_family_cutouts.pdf")
    cluster_output = _plot_path(run_dir, "image_catalog_family_cluster.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_cols = IMAGE_CATALOG_DETAIL_COLUMNS
    draw_critical_lines = bool(getattr(render_config, "draw_critical_lines", True))
    if draw_critical_lines:
        try:
            best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
        except Exception as exc:
            _log(None, f"[plot:image_catalog_family_cutouts] skipped critical lines: best-fit conversion failed: {exc}")
            best_fit_latent = None
    else:
        best_fit_latent = None
    critical_model_pairs_by_z: dict[float, tuple[Any, list[dict[str, float]]]] = {}
    fig = _image_catalog_cluster_overview_figure(
        helpers,
        band_images,
        bands,
        rgb_display,
        display_image,
        catalog_df,
        extra_df,
        state.reference,
        detail_cols=detail_cols,
        render_config=render_config,
    )
    fig.savefig(cluster_output, facecolor=fig.get_facecolor(), **savefig_kwargs)
    if not detail_cutouts_enabled:
        _maybe_show_figure(fig)
        plt.close(fig)
        return
    blocks = _image_catalog_family_cutout_blocks(
        state,
        catalog_df,
        extra_df,
        detail_cols=detail_cols,
        default_cutout_size_arcsec=cutout_size_arcsec,
    )
    with PdfPages(output) as pdf:
        _finish_pdf_page(pdf, fig, facecolor=fig.get_facecolor(), **savefig_kwargs)

        for block in blocks:
            model_pair = (
                _image_catalog_critical_line_model_pair(
                    evaluator,
                    best_fit_latent,
                    block,
                    critical_model_pairs_by_z,
                )
                if best_fit_latent is not None
                else None
            )
            n_rows = max(1, int(block.get("layout_rowspan", block.get("overview_rowspan", 1))))
            n_cols = detail_cols
            fig = plt.figure(
                figsize=helpers._figure_size(n_rows, n_cols),
                dpi=render_dpi,
            )
            helpers._style_cutout_figure(fig)
            grid = fig.add_gridspec(n_rows, n_cols)
            overview_units = max(1, int(block.get("overview_units", detail_cols)))
            overview_ax = fig.add_subplot(grid[:overview_units, :])
            helpers._style_cutout_axis(overview_ax)
            _draw_image_catalog_overview_panel(
                overview_ax,
                helpers,
                band_images,
                bands,
                rgb_display,
                display_image,
                block,
                state.reference,
                model_pair,
                render_config=render_config,
                include_critical_lines=draw_critical_lines,
            )
            for panel_index, panel in enumerate(block["detail_panels"]):
                detail_row = overview_units + panel_index // detail_cols
                detail_col = panel_index % detail_cols
                ax = fig.add_subplot(grid[detail_row, detail_col])
                helpers._style_cutout_axis(ax)
                _draw_image_catalog_detail_panel(
                    ax,
                    helpers,
                    band_images,
                    bands,
                    rgb_display,
                    display_image,
                    pd.Series(panel),
                    state.reference,
                    model_pair,
                    render_config=render_config,
                    include_critical_lines=draw_critical_lines,
                )
            _finish_pdf_page(pdf, fig, facecolor=fig.get_facecolor(), **savefig_kwargs)


def _is_stage0_minimal_output(state: BuildState, args: argparse.Namespace) -> bool:
    return bool(getattr(state, "perturbation_discovery_stage0", False)) or bool(
        getattr(args, "perturbation_discovery_stage0", False)
    )


def _generate_plots_and_tables(
    run_dir: Path,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    best_eval: EvaluationResult,
    results: PosteriorResults,
    runtime_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    global _SHOW_PLOTS
    if _is_stage0_minimal_output(state, args):
        previous_show_setting = _SHOW_PLOTS
        _SHOW_PLOTS = bool(getattr(args, "show_plots", False))
        try:
            return _generate_stage0_minimal_plots_and_tables(
                run_dir=run_dir,
                tables_dir=tables_dir,
                state=state,
                evaluator=evaluator,
                best_fit=best_fit,
                best_eval=best_eval,
                results=results,
                runtime_sec=runtime_sec,
                args=args,
            )
        finally:
            _SHOW_PLOTS = previous_show_setting
    sample_likelihood_mode = _active_sample_likelihood_mode(evaluator, args)
    use_arc_aware_diagnostics = _uses_arc_aware_diagnostics(sample_likelihood_mode)
    max_tree_depth = _first_int_value(getattr(args, "max_tree_depth", 10), 10)
    best_fit_values = _best_fit_values_for_specs(state.parameter_specs, best_fit)
    map_values = _fit_vector_values_for_specs(state.parameter_specs, results.map_fit)
    if not map_values:
        map_values = _map_values_for_specs(state.parameter_specs, results.samples, results.log_prob)
    maximum_likelihood_values = _fit_vector_values_for_specs(
        state.parameter_specs,
        results.maximum_likelihood_fit,
    )
    if not maximum_likelihood_values:
        maximum_likelihood_values = _sample_index_values_for_specs(
            state.parameter_specs,
            results.samples,
            (results.init_diagnostics or {}).get("maximum_likelihood_sample_index"),
        )
    best_fit_latent = _reported_physical_to_latent_vector(evaluator, np.asarray(best_fit, dtype=float))
    critical_arc_singular_threshold_best_fit = _fit_quality_critical_arc_singular_threshold(
        evaluator,
        best_fit_latent,
    )
    critical_arc_singular_softness_best_fit = _fit_quality_critical_arc_singular_softness(
        evaluator,
        best_fit_latent,
    )
    previous_stage_best_values = getattr(state, "previous_stage_best_values", None)
    best_value_selected = str((results.init_diagnostics or {}).get("best_value_selected", "") or "")
    best_value_requested = str((results.init_diagnostics or {}).get("best_value_requested", "") or "")

    context: dict[str, Any] = {
        "summary_df": pd.DataFrame(),
        "family_df": pd.DataFrame(),
        "image_fit_quality_df": pd.DataFrame(),
        "model_magnification_df": pd.DataFrame(),
        "image_recovery_extra_df": pd.DataFrame(),
        "cab_arc_diagnostics_df": pd.DataFrame(),
        "image_count_recovery_df": pd.DataFrame(),
        "chain_health_df": pd.DataFrame(),
        "chain_parameter_diagnostics_df": pd.DataFrame(),
        "run_summary": {},
        "run_summary_text": "",
        "scaling_specs": [],
        "scaling_samples": np.empty((0, 0), dtype=float),
        "scaling_best_fit": np.empty((0,), dtype=float),
        "cosmology_specs": [],
        "cosmology_samples": np.empty((0, 0), dtype=float),
        "cosmology_best_fit": np.empty((0,), dtype=float),
        "bayes_corner_overlay": None,
        "potfile_constraint_df": pd.DataFrame(),
        "scaling_results_df": pd.DataFrame(),
        "independent_scaling_df": pd.DataFrame(),
        "independent_scaling_plot_df": pd.DataFrame(),
        "free_galaxy_shape_df": pd.DataFrame(),
        "scaling_relation_df": pd.DataFrame(),
        "trace_specs": [],
        "trace_grouped_samples": np.empty((0, 0, 0), dtype=float),
        "subhalo_df": pd.DataFrame(),
        "perturbation_discovery_df": pd.DataFrame(),
        "truth_grid_cache": {},
    }

    def _write_run_summary_files() -> None:
        (tables_dir / "run_summary.json").write_text(
            json.dumps(context["run_summary"], indent=2),
            encoding="utf-8",
        )
        (tables_dir / "run_summary.txt").write_text(context["run_summary_text"], encoding="utf-8")

    def _build_initial_run_summary() -> None:
        context["run_summary"] = _run_summary(
            args,
            state,
            runtime_sec,
            results,
            best_eval.loglike,
            evaluator,
            image_fit_quality_df=None,
            image_count_recovery_df=None,
        )
        context["run_summary"]["image_recovery_stage"] = "pending"
        context["run_summary_text"] = _format_run_summary_text(context["run_summary"])

    def _build_final_run_summary() -> None:
        context["run_summary"] = _run_summary(
            args,
            state,
            runtime_sec,
            results,
            best_eval.loglike,
            evaluator,
            image_fit_quality_df=context["image_fit_quality_df"],
            image_count_recovery_df=context["image_count_recovery_df"],
        )
        context["run_summary"]["image_recovery_stage"] = "complete"
        context["run_summary_text"] = _format_run_summary_text(context["run_summary"])

    def _scaling_corner_values() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        specs = context["scaling_specs"]
        best_values = _best_fit_values_for_specs(specs, context["scaling_best_fit"])
        return (
            best_values,
            _subset_values_for_specs(specs, map_values),
            _subset_values_for_specs(specs, maximum_likelihood_values),
        )

    def _cosmology_corner_values() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        specs = context["cosmology_specs"]
        best_values = _best_fit_values_for_specs(specs, context["cosmology_best_fit"])
        return (
            best_values,
            _subset_values_for_specs(specs, map_values),
            _subset_values_for_specs(specs, maximum_likelihood_values),
        )

    def _merge_arc_aware_family_diagnostics() -> None:
        if not use_arc_aware_diagnostics:
            return
        arc_aware_family_df = _arc_aware_family_diagnostics_from_image_rows(context["image_fit_quality_df"])
        if arc_aware_family_df.empty:
            return
        family_df = context["family_df"].copy()
        family_df["family_id"] = family_df["family_id"].astype(str)
        arc_aware_family_df["family_id"] = arc_aware_family_df["family_id"].astype(str)
        context["family_df"] = family_df.merge(arc_aware_family_df, on="family_id", how="left")

    def _critical_arc_support_plots_enabled() -> bool:
        return use_arc_aware_diagnostics

    def _critical_arc_support_plot_mode() -> str:
        if _uses_arc_aware_diagnostics(sample_likelihood_mode):
            return sample_likelihood_mode
        stage2_forward_mode = str(getattr(args, "stage2_forward_mode", "") or "").strip()
        if stage2_forward_mode == "critical-arc-anisotropic":
            return CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE_MODE
        if stage2_forward_mode == "critical-arc":
            return CRITICAL_ARC_MIXTURE_IMAGE_PLANE_MODE
        return sample_likelihood_mode

    run_diagnostics_tasks: list[PlotTask] = [
        (
            "summary_table",
            "plots.run_diagnostics.summary_table",
            lambda: context.__setitem__(
                "summary_df",
                _summary_table(
                    state.parameter_specs,
                    results.samples,
                    best_fit,
                    sample_weights=results.sample_weights,
                ),
            ),
        ),
        (
            "family_diagnostics_table",
            "plots.run_diagnostics.family_diagnostics_table",
            lambda: context.__setitem__("family_df", _family_diagnostics_table(evaluator, best_eval)),
        ),
        (
            "chain_health_table",
            "plots.run_diagnostics.chain_health_table",
            lambda: context.__setitem__(
                "chain_health_df",
                _chain_health_summary_table(
                    results,
                    state.parameter_specs,
                    max_tree_depth=max_tree_depth,
                ),
            ),
        ),
        (
            "chain_parameter_diagnostics_table",
            "plots.run_diagnostics.chain_parameter_diagnostics_table",
            lambda: context.__setitem__(
                "chain_parameter_diagnostics_df",
                _chain_parameter_diagnostics_table(results, state.parameter_specs),
            ),
        ),
        ("run_summary_initial", "plots.run_diagnostics.run_summary_initial", _build_initial_run_summary),
        (
            "potfile_corner_subset",
            "plots.run_diagnostics.potfile_corner_subset",
            lambda: (
                lambda subset: (
                    context.__setitem__("scaling_specs", subset[0]),
                    context.__setitem__("scaling_samples", subset[1]),
                    context.__setitem__("scaling_best_fit", subset[2]),
                )
            )(_potfile_corner_parameter_subset(state.parameter_specs, results.samples, best_fit)),
        ),
        (
            "cosmology_subset",
            "plots.run_diagnostics.cosmology_subset",
            lambda: (
                lambda subset: (
                    context.__setitem__("cosmology_specs", subset[0]),
                    context.__setitem__("cosmology_samples", subset[1]),
                    context.__setitem__("cosmology_best_fit", subset[2]),
                )
            )(_cosmology_parameter_subset(state.parameter_specs, results.samples, best_fit)),
        ),
        (
            "load_bayes_corner_overlay",
            "plots.run_diagnostics.load_bayes_corner_overlay",
            lambda: context.__setitem__(
                "bayes_corner_overlay",
                _load_bayes_corner_overlay(getattr(args, "corner_overlay_bayes_dat", None), state),
            ),
        ),
        (
            "potfile_constraint_table",
            "plots.run_diagnostics.potfile_constraint_table",
            lambda: context.__setitem__(
                "potfile_constraint_df",
                _potfile_constraint_diagnostics_table(
                    state.parameter_specs,
                    results.samples,
                    best_fit,
                    evaluator.scaling_rank_df,
                    sample_weights=results.sample_weights,
                ),
            ),
        ),
        (
            "scaling_results_summary_table",
            "plots.run_diagnostics.scaling_results_summary_table",
            lambda: context.__setitem__(
                "scaling_results_df",
                _scaling_results_summary_table(
                    state.parameter_specs,
                    results.samples,
                    best_fit,
                    str(getattr(state, "scaling_relation_mode", getattr(args, "scaling_relation_mode", "lenstool-denominator"))),
                    sample_weights=results.sample_weights,
                ),
            ),
        ),
        (
            "independent_scaling_table",
            "plots.run_diagnostics.independent_scaling_table",
            lambda: context.__setitem__(
                "independent_scaling_df",
                _independent_scaling_diagnostics_table(
                    state.parameter_specs,
                    results.samples,
                    best_fit,
                    evaluator.scaling_rank_df,
                    getattr(state, "packed_lens_spec", None),
                    sample_weights=results.sample_weights,
                    log_sigma_offset_width=float(getattr(state, "independent_scaling_free_log_sigma_offset_width", 0.10)),
                    log_mass_offset_width=float(getattr(state, "independent_scaling_free_log_mass_offset_width", 0.20)),
                ),
            ),
        ),
        (
            "independent_scaling_plot_table",
            "plots.run_diagnostics.independent_scaling_plot_table",
            lambda: context.__setitem__(
                "independent_scaling_plot_df",
                _independent_scaling_plot_table(evaluator.scaling_rank_df, context["independent_scaling_df"]),
            ),
        ),
        (
            "scaling_relation_table",
            "plots.run_diagnostics.scaling_relation_table",
            lambda: context.__setitem__(
                "scaling_relation_df",
                _scaling_relation_summary_table(
                    evaluator.scaling_rank_df,
                    state.parameter_specs,
                    results.samples,
                    best_fit,
                    getattr(state, "packed_lens_spec", None),
                    sample_weights=results.sample_weights,
                    independent_scaling_df=context["independent_scaling_df"],
                    best_value=best_value_selected or None,
                    best_value_requested=best_value_requested or None,
                ),
            ),
        ),
        (
            "scaling_grouped_subset",
            "plots.run_diagnostics.scaling_grouped_subset",
            lambda: (
                lambda subset: (
                    context.__setitem__("trace_specs", subset[0]),
                    context.__setitem__("trace_grouped_samples", subset[1]),
                )
            )(_scaling_grouped_subset(state.parameter_specs, results.grouped_samples)),
        ),
        (
            "subhalo_properties_table",
            "plots.run_diagnostics.subhalo_properties_table",
            lambda: context.__setitem__(
                "subhalo_df",
                _subhalo_properties_table(
                    state,
                    evaluator,
                    best_fit,
                    getattr(args, "caustic_source_redshift", 9.0),
                ),
            ),
        ),
        (
            "free_galaxy_shape_comparison_table",
            "plots.run_diagnostics.free_galaxy_shape_comparison_table",
            lambda: context.__setitem__(
                "free_galaxy_shape_df",
                _free_galaxy_shape_comparison_table(state, best_fit),
            ),
        ),
        (
            "perturbation_discovery_diagnostics_table",
            "plots.run_diagnostics.perturbation_discovery_diagnostics_table",
            lambda: context.__setitem__(
                "perturbation_discovery_df",
                _load_perturbation_discovery_diagnostics_table(tables_dir),
            ),
        ),
        (
            "write_potential_summary_csv",
            "plots.run_diagnostics.write_potential_summary_csv",
            lambda: context["summary_df"].to_csv(tables_dir / "potential_summary.csv", index=False),
        ),
        (
            "write_family_diagnostics_csv",
            "plots.run_diagnostics.write_family_diagnostics_csv",
            lambda: context["family_df"].to_csv(tables_dir / "family_diagnostics.csv", index=False),
        ),
        (
            "write_scaling_results_summary_csv",
            "plots.run_diagnostics.write_scaling_results_summary_csv",
            lambda: context["scaling_results_df"].to_csv(tables_dir / "scaling_results_summary.csv", index=False),
        ),
        (
            "scaling_results_summary_log",
            "plots.run_diagnostics.scaling_results_summary_log",
            lambda: (
                _log(args, "[scaling-results] no scaling relation parameters")
                if context["scaling_results_df"].empty
                else _log(
                    args,
                    f"[scaling-results] potfiles={len(context['scaling_results_df'])} mass_definition=\"{SCALING_RESULTS_MASS_NOTE}\"",
                    renderable=_build_scaling_results_rich_table(context["scaling_results_df"]),
                )
            ),
        ),
        (
            "write_subhalo_properties_csv",
            "plots.run_diagnostics.write_subhalo_properties_csv",
            lambda: context["subhalo_df"].to_csv(tables_dir / "subhalo_properties.csv", index=False),
        ),
        (
            "write_free_galaxy_shape_comparison_csv",
            "plots.run_diagnostics.write_free_galaxy_shape_comparison_csv",
            lambda: context["free_galaxy_shape_df"].to_csv(
                tables_dir / "free_galaxy_shape_comparison.csv",
                index=False,
            ),
        ),
        (
            "write_potfile_summary_txt",
            "plots.run_diagnostics.write_potfile_summary_txt",
            lambda: _write_potfile_summary_txt(tables_dir, context["summary_df"]),
        ),
        (
            "write_potfile_constraint_csv",
            "plots.run_diagnostics.write_potfile_constraint_csv",
            lambda: (
                context["potfile_constraint_df"].to_csv(tables_dir / "potfile_constraint_diagnostics.csv", index=False)
                if not context["potfile_constraint_df"].empty
                else None
            ),
        ),
        (
            "write_potfile_constraint_txt",
            "plots.run_diagnostics.write_potfile_constraint_txt",
            lambda: (
                _write_potfile_constraint_summary_txt(tables_dir, context["potfile_constraint_df"])
                if not context["potfile_constraint_df"].empty
                else None
            ),
        ),
        (
            "write_independent_scaling_csv",
            "plots.run_diagnostics.write_independent_scaling_csv",
            lambda: (
                context["independent_scaling_df"].to_csv(
                    tables_dir / "independent_scaling_diagnostics.csv",
                    index=False,
                )
                if not context["independent_scaling_df"].empty
                else None
            ),
        ),
        (
            "write_scaling_rank_csv",
            "plots.run_diagnostics.write_scaling_rank_csv",
            lambda: (
                evaluator.scaling_rank_df.to_csv(tables_dir / "scaling_rank_diagnostics.csv", index=False)
                if not evaluator.scaling_rank_df.empty
                else None
            ),
        ),
        (
            "write_scaling_relation_csv",
            "plots.run_diagnostics.write_scaling_relation_csv",
            lambda: (
                context["scaling_relation_df"].to_csv(tables_dir / "scaling_relation_summary.csv", index=False)
                if not context["scaling_relation_df"].empty
                else None
            ),
        ),
        (
            "write_chain_health_csv",
            "plots.run_diagnostics.write_chain_health_csv",
            lambda: (
                context["chain_health_df"].to_csv(tables_dir / "chain_health_summary.csv", index=False)
                if not context["chain_health_df"].empty
                else None
            ),
        ),
        (
            "write_chain_parameter_diagnostics_csv",
            "plots.run_diagnostics.write_chain_parameter_diagnostics_csv",
            lambda: (
                context["chain_parameter_diagnostics_df"].to_csv(tables_dir / "chain_parameter_diagnostics.csv", index=False)
                if not context["chain_parameter_diagnostics_df"].empty
                else None
            ),
        ),
        ("write_initial_run_summary", "plots.run_diagnostics.write_initial_run_summary", _write_run_summary_files),
        (
            "corner",
            "plots.corner",
            lambda: _plot_corner(
                run_dir,
                results.samples,
                state.parameter_specs,
                best_fit_values=best_fit_values,
                map_values=map_values,
                maximum_likelihood_values=maximum_likelihood_values,
                previous_stage_best_values=previous_stage_best_values,
                bayes_corner_overlay=context["bayes_corner_overlay"],
            ),
        ),
        (
            "cosmology_corner",
            "plots.cosmology_corner",
            lambda: (
                lambda values: _plot_cosmology_corner(
                    run_dir,
                    context["cosmology_samples"],
                    context["cosmology_specs"],
                    best_fit_values=values[0],
                    map_values=values[1],
                    maximum_likelihood_values=values[2],
                    previous_stage_best_values=previous_stage_best_values,
                    bayes_corner_overlay=context["bayes_corner_overlay"],
                )
            )(_cosmology_corner_values()),
        ),
        (
            "potfile_corner",
            "plots.potfile_corner",
            lambda: (
                lambda values: _plot_potfile_corner(
                    run_dir,
                    context["scaling_samples"],
                    context["scaling_specs"],
                    best_fit_values=values[0],
                    map_values=values[1],
                    maximum_likelihood_values=values[2],
                    previous_stage_best_values=previous_stage_best_values,
                    bayes_corner_overlay=context["bayes_corner_overlay"],
                )
            )(_scaling_corner_values()),
        ),
        (
            "potfile_prior_posterior",
            "plots.potfile_prior_posterior",
            lambda: _plot_potfile_prior_posterior(
                run_dir,
                context["potfile_constraint_df"],
                results.samples,
                state.parameter_specs,
            ),
        ),
        (
            "potfile_leverage_summary",
            "plots.potfile_leverage_summary",
            lambda: _plot_potfile_leverage_summary(run_dir, context["potfile_constraint_df"]),
        ),
        (
            "trace",
            "plots.trace",
            lambda: _plot_trace(run_dir, context["trace_grouped_samples"], context["trace_specs"]),
        ),
        (
            "scaling_rank_scatter",
            "plots.scaling_rank_scatter",
            lambda: _plot_scaling_rank_scatter(run_dir, evaluator.scaling_rank_df),
        ),
        (
            "scaling_relation_summary",
            "plots.scaling_relation_summary",
            lambda: _plot_scaling_relation_summary(run_dir, context["scaling_relation_df"]),
        ),
        (
            "perturbation_discovery_diagnostics",
            "plots.perturbation_discovery_diagnostics",
            lambda: _plot_perturbation_discovery_diagnostics(run_dir, context["perturbation_discovery_df"]),
        ),
        (
            "chain_health",
            "plots.chain_health",
            lambda: _plot_chain_health(
                run_dir,
                results,
                state.parameter_specs,
                max_tree_depth=max_tree_depth,
            ),
        ),
        (
            "source_plane_residual_histogram",
            "plots.source_plane_residual_histogram",
            lambda: _plot_source_plane_residual_histogram(run_dir, state, best_eval),
        ),
        (
            "subhalo_mass_function",
            "plots.subhalo_mass_function",
            lambda: _plot_subhalo_mass_function(context["subhalo_df"], _plot_path(run_dir, "subhalo_mass_function.pdf")),
        ),
        (
            "subhalo_radial_distribution",
            "plots.subhalo_radial_distribution",
            lambda: _plot_subhalo_radial_distribution(context["subhalo_df"], _plot_path(run_dir, "subhalo_radial_distribution.pdf")),
        ),
        (
            "free_galaxy_shape_comparison",
            "plots.free_galaxy_shape_comparison",
            lambda: _plot_free_galaxy_shape_comparison(
                context["free_galaxy_shape_df"],
                _plot_path(run_dir, "free_galaxy_shape_comparison.pdf"),
            ),
        ),
        (
            "per_potential_summary",
            "plots.per_potential_summary",
            lambda: _plot_per_potential_summary(
                run_dir,
                context["summary_df"],
                previous_stage_best_values=previous_stage_best_values,
                parameter_specs=state.parameter_specs,
            ),
        ),
        ("timing_profile", "plots.timing_profile", lambda: _plot_timing_profile(run_dir, evaluator)),
        *(
            [
                ("ns_diagnostics", "plots.ns_diagnostics", lambda: _plot_ns_diagnostics(run_dir, results.ns_diagnostics)),
                ("ns_trace", "plots.ns_trace", lambda: _plot_ns_trace(run_dir, results.ns_diagnostics, state.parameter_specs)),
                ("ns_weight_diagnostics", "plots.ns_weight_diagnostics", lambda: _plot_ns_weight_diagnostics(run_dir, results.ns_diagnostics)),
            ]
            if results.ns_diagnostics
            else []
        ),
        *(
            [
                ("smc_diagnostics", "plots.smc_diagnostics", lambda: _plot_smc_diagnostics(run_dir, results)),
                ("smc_weight_diagnostics", "plots.smc_weight_diagnostics", lambda: _plot_smc_weight_diagnostics(run_dir, results)),
                (
                    "smc_corner",
                    "plots.smc_corner",
                    lambda: _plot_smc_corner(
                        run_dir,
                        results.samples,
                        state.parameter_specs,
                        results.sample_weights,
                        best_fit_values=best_fit_values,
                        map_values=map_values,
                        maximum_likelihood_values=maximum_likelihood_values,
                        previous_stage_best_values=previous_stage_best_values,
                    ),
                ),
            ]
            if _has_smc_plot_data(results)
            else []
        ),
    ]
    if bool(getattr(args, "plot_numpyro_model", False)):
        run_diagnostics_tasks.append(
            (
                "numpyro_model",
                "plots.numpyro_model",
                lambda: _plot_numpyro_model(run_dir, state, evaluator, args),
            )
        )

    image_recovery_tasks: list[PlotTask] = [
        (
            "fit_quality_tables",
            "plots.image_recovery.fit_quality_tables",
            lambda progress=None: (
                lambda tables: (
                    context.__setitem__("image_fit_quality_df", tables[0]),
                    context.__setitem__("model_magnification_df", tables[1]),
                    context.__setitem__("image_recovery_extra_df", tables[2]),
                )
            )(_fit_quality_tables(state, evaluator, best_fit, results, args, progress=progress)),
        ),
        (
            "cab_arc_diagnostics_table",
            "plots.image_recovery.cab_arc_diagnostics_table",
            lambda: context.__setitem__("cab_arc_diagnostics_df", _cab_arc_diagnostics_table(evaluator, best_fit)),
        ),
        (
            "image_count_recovery_table",
            "plots.image_recovery.image_count_recovery_table",
            lambda: context.__setitem__("image_count_recovery_df", _image_count_recovery_table(state, context["image_fit_quality_df"])),
        ),
        (
            "arc_aware_family_diagnostics_table",
            "plots.image_recovery.arc_aware_family_diagnostics_table",
            _merge_arc_aware_family_diagnostics,
        ),
        ("run_summary_final", "plots.image_recovery.run_summary_final", _build_final_run_summary),
        (
            "write_family_diagnostics_csv",
            "plots.image_recovery.write_family_diagnostics_csv",
            lambda: context["family_df"].to_csv(tables_dir / "family_diagnostics.csv", index=False),
        ),
        (
            "write_image_fit_quality_csv",
            "plots.image_recovery.write_image_fit_quality_csv",
            lambda: context["image_fit_quality_df"].to_csv(tables_dir / "image_fit_quality.csv", index=False),
        ),
        (
            "write_image_count_recovery_csv",
            "plots.image_recovery.write_image_count_recovery_csv",
            lambda: context["image_count_recovery_df"].to_csv(tables_dir / "image_count_recovery.csv", index=False),
        ),
        (
            "write_image_recovery_extra_images_csv",
            "plots.image_recovery.write_image_recovery_extra_images_csv",
            lambda: context["image_recovery_extra_df"].to_csv(tables_dir / "image_recovery_extra_images.csv", index=False),
        ),
        (
            "write_cab_arc_diagnostics_csv",
            "plots.image_recovery.write_cab_arc_diagnostics_csv",
            lambda: (
                context["cab_arc_diagnostics_df"].to_csv(tables_dir / "cab_arc_diagnostics.csv", index=False)
                if not context["cab_arc_diagnostics_df"].empty or len(context["cab_arc_diagnostics_df"].columns) > 0
                else None
            ),
        ),
        (
            "write_model_magnification_csv",
            "plots.image_recovery.write_model_magnification_csv",
            lambda: context["model_magnification_df"].to_csv(tables_dir / "model_magnification.csv", index=False),
        ),
        (
            "flux_magnification_ratio_consistency_table",
            "plots.image_recovery.flux_magnification_ratio_consistency_table",
            lambda: context.__setitem__(
                "flux_magnification_ratio_pair_df",
                _flux_magnification_ratio_pair_table(
                    state,
                    context["model_magnification_df"],
                    context["image_fit_quality_df"],
                ),
            ),
        ),
        (
            "write_flux_magnification_ratio_consistency_csv",
            "plots.image_recovery.write_flux_magnification_ratio_consistency_csv",
            lambda: context["flux_magnification_ratio_pair_df"].to_csv(
                tables_dir / "flux_magnification_ratio_consistency.csv",
                index=False,
            ),
        ),
        ("write_final_run_summary", "plots.image_recovery.write_final_run_summary", _write_run_summary_files),
        (
            "image_recovery",
            "plots.image_recovery",
            lambda: _plot_image_recovery_fit_quality(
                context["image_fit_quality_df"],
                _plot_path(run_dir, "image_recovery.pdf"),
                context["image_recovery_extra_df"],
                use_arc_aware_diagnostics=use_arc_aware_diagnostics,
            ),
        ),
        *(
            [
                (
                    "image_catalog_family_cutouts",
                    "plots.image_catalog_family_cutouts",
                    lambda: _plot_image_catalog_family_cutouts(
                        run_dir,
                        state,
                        evaluator,
                        best_fit,
                        context["image_fit_quality_df"],
                        context["image_recovery_extra_df"],
                        args,
                    ),
                )
            ]
            if _image_catalog_family_cluster_enabled(args, run_dir)
            else []
        ),
        (
            "image_count_recovery",
            "plots.image_count_recovery",
            lambda: _plot_image_count_recovery(
                context["image_count_recovery_df"],
                _plot_path(run_dir, "image_count_recovery.pdf"),
                use_arc_aware_diagnostics=use_arc_aware_diagnostics,
            ),
        ),
        (
            "model_magnification",
            "plots.model_magnification",
            lambda: _plot_model_magnification_fit_quality(context["model_magnification_df"], _plot_path(run_dir, "model_magnification.pdf")),
        ),
        (
            "normalized_image_residuals",
            "plots.normalized_image_residuals",
            lambda: _plot_normalized_image_residuals(context["image_fit_quality_df"], _plot_path(run_dir, "normalized_image_residuals.pdf")),
        ),
        (
            "image_residual_histogram",
            "plots.image_residual_histogram",
            lambda: _plot_image_residual_histogram(
                context["image_fit_quality_df"],
                _plot_path(run_dir, "image_residual_histogram.pdf"),
                use_arc_aware_diagnostics=use_arc_aware_diagnostics,
            ),
        ),
        (
            "residual_vs_magnification",
            "plots.residual_vs_magnification",
            lambda: _plot_residual_vs_magnification(
                context["image_fit_quality_df"],
                context["model_magnification_df"],
                _plot_path(run_dir, "residual_vs_magnification.pdf"),
            ),
        ),
        (
            "flux_magnification_ratio_consistency",
            "plots.flux_magnification_ratio_consistency",
            lambda: _plot_flux_magnification_ratio_consistency(
                context["flux_magnification_ratio_pair_df"],
                _plot_path(run_dir, "flux_magnification_ratio_consistency.pdf"),
            ),
        ),
        (
            "residual_geometry_trends",
            "plots.residual_geometry_trends",
            lambda: _plot_residual_geometry_trends(context["image_fit_quality_df"], _plot_path(run_dir, "residual_geometry_trends.pdf")),
        ),
    ]
    if _critical_arc_support_plots_enabled():
        image_recovery_tasks.extend(
            [
                (
                    "critical_arc_support_histogram",
                    "plots.critical_arc_support_histogram",
                    lambda: _plot_critical_arc_support_histogram(
                        context["image_fit_quality_df"],
                        _plot_path(run_dir, "critical_arc_support_histogram.pdf"),
                        arc_recovery_p_arc_threshold=float(
                            getattr(evaluator, "arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
                        ),
                        critical_arc_base_prob=float(getattr(evaluator, "critical_arc_base_prob", CRITICAL_ARC_BASE_PROB)),
                        critical_arc_max_prob=float(getattr(evaluator, "critical_arc_max_prob", CRITICAL_ARC_MAX_PROB)),
                        singular_threshold=float(critical_arc_singular_threshold_best_fit),
                        singular_softness=float(critical_arc_singular_softness_best_fit),
                        sample_likelihood_mode=_critical_arc_support_plot_mode(),
                    ),
                ),
                (
                    "critical_arc_support_phase_space",
                    "plots.critical_arc_support_phase_space",
                    lambda: _plot_critical_arc_support_phase_space(
                        context["image_fit_quality_df"],
                        _plot_path(run_dir, "critical_arc_support_phase_space.pdf"),
                        arc_recovery_p_arc_threshold=float(
                            getattr(evaluator, "arc_recovery_p_arc_threshold", CRITICAL_ARC_RECOVERY_P_ARC_THRESHOLD)
                        ),
                        critical_arc_base_prob=float(getattr(evaluator, "critical_arc_base_prob", CRITICAL_ARC_BASE_PROB)),
                        critical_arc_max_prob=float(getattr(evaluator, "critical_arc_max_prob", CRITICAL_ARC_MAX_PROB)),
                        singular_threshold=float(critical_arc_singular_threshold_best_fit),
                        singular_softness=float(critical_arc_singular_softness_best_fit),
                        sample_likelihood_mode=_critical_arc_support_plot_mode(),
                    ),
                ),
                (
                    "critical_arc_recovery_by_family",
                    "plots.critical_arc_recovery_by_family",
                    lambda: _plot_critical_arc_recovery_by_family(
                        context["image_count_recovery_df"],
                        _plot_path(run_dir, "critical_arc_recovery_by_family.pdf"),
                    ),
                ),
            ]
        )

    truth_recovery_tasks: list[PlotTask] = []
    skip_grid_diagnostics = bool(getattr(args, "skip_grid_diagnostics", False))
    if skip_grid_diagnostics:
        _log(args, "[plots] grid diagnostics skipped by skip_grid_diagnostics=True")
    kappa_true_fits = getattr(args, "kappa_true_fits", None)
    gammax_true_fits = getattr(args, "gammax_true_fits", None)
    gammay_true_fits = getattr(args, "gammay_true_fits", None)
    posterior_truth_recovery_draws = getattr(args, "posterior_truth_recovery_draws", None)
    truth_grid_mode = str(getattr(args, "truth_grid_mode", TRUTH_GRID_MODE_MEDIAN))
    truth_grid_size = int(getattr(args, "truth_grid_size", DEFAULT_TRUTH_GRID_SIZE))
    truth_grid_draw_seed = _truth_grid_draw_seed(getattr(args, "seed", DEFAULT_RUNTIME_SEED))
    if not skip_grid_diagnostics and kappa_true_fits is not None and str(kappa_true_fits).strip():
        gamma_truth_available = (
            gammax_true_fits is not None
            and str(gammax_true_fits).strip()
            and gammay_true_fits is not None
            and str(gammay_true_fits).strip()
        )
        truth_grid_source_fits = {"kappa": str(kappa_true_fits)}
        if gamma_truth_available:
            truth_grid_source_fits = {
                "kappa": str(kappa_true_fits),
                "gamma1": str(gammax_true_fits),
                "gamma2": str(gammay_true_fits),
                "detA": str(kappa_true_fits),
                "mu": str(kappa_true_fits),
                "abs_mu": str(kappa_true_fits),
            }
            truth_recovery_tasks.append(
                (
                    "truth_recovery_grids",
                    "plots.truth_recovery.truth_recovery_grids",
                    lambda progress=None: _precompute_truth_recovery_grids(
                        run_dir,
                        evaluator,
                        results,
                        args,
                        context["truth_grid_cache"],
                        progress=progress,
                    ),
                )
            )
            truth_recovery_tasks.append(
                (
                    "mu_truth_diagnostics",
                    "plots.mu_truth_diagnostics",
                    lambda: _plot_abs_mu_truth_diagnostics(
                        run_dir,
                        evaluator,
                        results,
                        str(kappa_true_fits),
                        str(gammax_true_fits),
                        str(gammay_true_fits),
                        getattr(args, "caustic_source_redshift", 9.0),
                        image_df=context["image_fit_quality_df"],
                        posterior_truth_recovery_draws=posterior_truth_recovery_draws,
                        truth_grid_mode=truth_grid_mode,
                        truth_grid_size=truth_grid_size,
                        truth_grid_cache=context["truth_grid_cache"],
                        require_precomputed_truth_grid=True,
                        truth_grid_draw_seed=truth_grid_draw_seed,
                    ),
                )
            )
        if not gamma_truth_available:
            truth_recovery_tasks.append(
                (
                    "truth_recovery_grids",
                    "plots.truth_recovery.truth_recovery_grids",
                    lambda progress=None: _precompute_truth_recovery_grids(
                        run_dir,
                        evaluator,
                        results,
                        args,
                        context["truth_grid_cache"],
                        progress=progress,
                    ),
                )
            )
        truth_recovery_tasks.append(
            (
                "kappa_truth_diagnostics",
                "plots.kappa_truth_diagnostics",
                lambda: _plot_kappa_truth_diagnostics(
                    run_dir,
                    evaluator,
                    results,
                    str(kappa_true_fits),
                    getattr(args, "caustic_source_redshift", 9.0),
                    image_df=context["image_fit_quality_df"],
                    posterior_truth_recovery_draws=posterior_truth_recovery_draws,
                    truth_grid_mode=truth_grid_mode,
                    truth_grid_size=truth_grid_size,
                    truth_grid_cache=context["truth_grid_cache"],
                    require_precomputed_truth_grid=True,
                    precompute_quantities=(
                        ("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
                        if gamma_truth_available
                        else ("kappa",)
                    ),
                    truth_grid_source_fits=truth_grid_source_fits,
                    truth_grid_draw_seed=truth_grid_draw_seed,
                ),
            )
        )
    if not skip_grid_diagnostics:
        caustic_plot_grid_scale_arcsec = getattr(
            args,
            "caustic_plot_grid_scale_arcsec",
            CAUSTIC_PLOT_GRID_SCALE_ARCSEC,
        )
        truth_recovery_tasks.append(
            (
                "caustic_overlay",
                "plots.caustic_overlay",
                lambda: _plot_caustic_overlay(
                    run_dir,
                    evaluator,
                    best_fit,
                    caustic_plot_grid_scale_arcsec,
                    getattr(args, "caustic_source_redshift", 9.0),
                ),
            )
        )
    previous_show_setting = _SHOW_PLOTS
    _SHOW_PLOTS = bool(getattr(args, "show_plots", False))
    try:
        _run_plot_stages_with_progress(
            args,
            [
                ("run_diagnostics", run_diagnostics_tasks),
                ("image_recovery", image_recovery_tasks),
                ("truth_recovery", truth_recovery_tasks),
            ],
        )
    finally:
        _SHOW_PLOTS = previous_show_setting
    _log(args, "[done] run summary\n" + str(context["run_summary_text"]).rstrip())
    return context["run_summary"]


def _generate_stage0_minimal_plots_and_tables(
    *,
    run_dir: Path,
    tables_dir: Path,
    state: BuildState,
    evaluator: ClusterJAXEvaluator,
    best_fit: np.ndarray,
    best_eval: EvaluationResult,
    results: PosteriorResults,
    runtime_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    independent_scaling_df = _run_logged_phase(
        args,
        "plots.stage0.independent_scaling_table",
        lambda: _independent_scaling_diagnostics_table(
            state.parameter_specs,
            results.samples,
            best_fit,
            evaluator.scaling_rank_df,
            getattr(state, "packed_lens_spec", None),
            sample_weights=results.sample_weights,
            log_sigma_offset_width=float(getattr(state, "independent_scaling_free_log_sigma_offset_width", 0.10)),
            log_mass_offset_width=float(getattr(state, "independent_scaling_free_log_mass_offset_width", 0.20)),
        ),
    )
    best_value_selected = str((results.init_diagnostics or {}).get("best_value_selected", "") or "")
    best_value_requested = str((results.init_diagnostics or {}).get("best_value_requested", "") or "")
    scaling_relation_df = _run_logged_phase(
        args,
        "plots.stage0.scaling_relation_table",
        lambda: _scaling_relation_summary_table(
            evaluator.scaling_rank_df,
            state.parameter_specs,
            results.samples,
            best_fit,
            getattr(state, "packed_lens_spec", None),
            sample_weights=results.sample_weights,
            independent_scaling_df=independent_scaling_df,
            best_value=best_value_selected or None,
            best_value_requested=best_value_requested or None,
        ),
    )
    perturbation_discovery_df = _run_logged_phase(
        args,
        "plots.stage0.perturbation_discovery_diagnostics_table",
        lambda: _load_perturbation_discovery_diagnostics_table(tables_dir),
    )
    run_summary = _run_logged_phase(
        args,
        "plots.stage0.run_summary",
        lambda: _run_summary(
            args,
            state,
            runtime_sec,
            results,
            best_eval.loglike,
            evaluator,
            image_fit_quality_df=None,
            image_count_recovery_df=None,
        ),
    )
    run_summary["stage0_minimal_outputs"] = True
    run_summary["stage0_plot_outputs"] = [
        "scaling_relation_summary.pdf",
        *(
            ["perturbation_discovery_diagnostics.pdf"]
            if not perturbation_discovery_df.empty
            else []
        ),
    ]
    run_summary_text = _format_run_summary_text(run_summary)

    _run_logged_phase(
        args,
        "plots.stage0.write_scaling_relation_csv",
        lambda: scaling_relation_df.to_csv(tables_dir / "scaling_relation_summary.csv", index=False),
    )
    _run_logged_phase(
        args,
        "plots.stage0.write_run_summary_json",
        lambda: (tables_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8"),
    )
    _run_logged_phase(
        args,
        "plots.stage0.write_run_summary_txt",
        lambda: (tables_dir / "run_summary.txt").write_text(run_summary_text, encoding="utf-8"),
    )

    plot_tasks: list[PlotTask] = [
        (
            "scaling_relation_summary",
            "plots.stage0.scaling_relation_summary",
            lambda: _plot_scaling_relation_summary(run_dir, scaling_relation_df),
        ),
        *(
            [
                (
                    "perturbation_discovery_diagnostics",
                    "plots.stage0.perturbation_discovery_diagnostics",
                    lambda: _plot_perturbation_discovery_diagnostics(run_dir, perturbation_discovery_df),
                )
            ]
            if not perturbation_discovery_df.empty
            else []
        ),
    ]
    _run_plot_tasks_with_progress(args, plot_tasks)
    _log(args, "[done] stage0 minimal run summary\n" + run_summary_text.rstrip())
    return run_summary


def _infer_stage1_artifacts_dir(args: argparse.Namespace) -> Path:
    if args.stage1_run_dir:
        candidate = Path(args.stage1_run_dir)
        return candidate / "artifacts" if candidate.name != "artifacts" else candidate
    if args.run_name:
        candidate = Path(args.output_dir) / args.run_name / "stage1_large_only" / "artifacts"
        if candidate.exists():
            return candidate
    raise ValueError("Missing stage-1 artifacts for the requested internal stage.")
