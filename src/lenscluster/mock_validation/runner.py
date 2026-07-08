from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_lenscluster_validation")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba_cache_{os.getuid()}")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
try:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test environments
    class Progress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "Progress":
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            return 0

        def update(self, task_id: int, **kwargs: Any) -> None:
            return None

        def advance(self, task_id: int, advance: int = 1) -> None:
            return None

    class BarColumn:
        pass

    class MofNCompleteColumn:
        pass

    class TextColumn:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TimeElapsedColumn:
        pass

from ..jax_cosmology import (
    critical_surface_density_angle_from_config,
    flat_wcdm_config,
)
from .. import jax_lensing
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib import pyplot as plt

from ..plot_style import apply_lenscluster_plot_style

apply_lenscluster_plot_style()

from .generation import (
    DEFAULT_CAUSTIC_BOUNDARY_MARGIN_ARCSEC,
    DEFAULT_CAUSTIC_COMPUTE_WINDOW_ARCSEC,
    DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
    DEFAULT_CAUSTIC_MIN_AREA_ARCSEC2,
    ORIGINAL_DPIE_PROFILE_NAME,
    CausticContour,
    DPIETruth,
    MockClusterPaths,
    SingleBCGMockConfig,
    SourceTruth,
    _axis_ratio_to_lenstool_ellipticite,
    _caustic_config_from_truth,
    _caustic_contours_by_z_from_truth,
    _compute_tangential_caustic_contours,
    _dex_scatter_to_ln,
    _image_count_requirement_text,
    _lenstool_ellipticite_to_axis_ratio,
    _sample_point_in_caustic,
    _subhalo_mass_luminosity_exponent,
    generate_single_bcg_mock,
)
from ..image_diagnostics import (
    image_prediction_for_family_latent as _shared_image_prediction_for_family_latent,
    image_sigma_int_for_params as _shared_image_sigma_int_for_params,
    image_count_recovery_table as _image_count_recovery_table,
    posterior_image_rms_row as _shared_posterior_image_rms_row,
)
from ..config import LensClusterSolverConfig
from ..planning import compile_run_plan
from ..plotting import (
    _best_fit_values_for_specs,
    _cosmology_parameter_subset,
    _corner_without_source_positions,
    _plot_corner,
    _plot_cosmology_corner,
    _plot_critical_arc_recovery_by_family as _shared_plot_critical_arc_recovery_by_family,
    _plot_critical_arc_support_histogram as _shared_plot_critical_arc_support_histogram,
    _plot_critical_arc_support_phase_space as _shared_plot_critical_arc_support_phase_space,
    _image_catalog_arc_recovered,
    _image_catalog_effective_recovery_statuses,
    _image_catalog_point_recovered,
    _plot_potfile_corner,
    _potfile_corner_parameter_subset,
)
from ..utils import (
    close_debug_log as _close_debug_log,
    configure_debug_log as _configure_debug_log,
    fmt_seconds as _fmt_seconds,
    jax_cpu_worker_count,
    log10_dpie_mass_msun,
    log_exception as _log_exception,
    log_message as _log,
    log_stage_banner as _log_stage_banner,
    progress_context as _progress_context,
    run_logged_phase as _run_logged_phase,
)
from .subhalo_mass import dpie_mass_anchor_from_truth, dpie_truth_subhalo_masses_from_scaling

from ..runner import LensClusterRunner
from .config import MockValidationConfig, solver_config_for_single_bcg_mock


def _limit_recovery_axis_ticks(ax: Any, *, nbins: int = 8) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=int(nbins)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=int(nbins)))


FIT_METHOD_SVI = "svi"
FIT_METHOD_SVI_NUTS = "svi+nuts"
FIT_METHOD_NUTS = "nuts"
FIT_METHOD_NS = "ns"
FIT_METHOD_SMC = "smc"
FIT_METHOD_MCHMC = "mchmc"
FIT_METHOD_MCLMC = "mclmc"
MICROCANONICAL_FIT_METHODS = (FIT_METHOD_MCHMC, FIT_METHOD_MCLMC)
SOLVER_FIT_MODE_SEQUENTIAL = "sequential"
SOLVER_FIT_MODE_EVIDENCE_NS = "evidence-ns"
RESUME_MODE_ALL = "all"
RESUME_MODE_FAST = "fast"
RESUME_MODES = (RESUME_MODE_ALL, RESUME_MODE_FAST)
DEFAULT_MATCH_TOLERANCE = 1.5
DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC = 0.2
DEFAULT_EXACT_IMAGE_PRECISION_LIMIT = 1.0e-8
DEFAULT_EXACT_IMAGE_NUM_ITER_MAX = 200
EXACT_IMAGE_FINDER_LENSTRONOMY = "lenstronomy"
EXACT_IMAGE_FINDER_LOCAL_LM = "local-lm"
EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE = "local-lm-adaptive"
EXACT_IMAGE_FINDER_CHOICES = (
    EXACT_IMAGE_FINDER_LENSTRONOMY,
    EXACT_IMAGE_FINDER_LOCAL_LM,
    EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE,
)
DEFAULT_EXACT_IMAGE_FINDER = EXACT_IMAGE_FINDER_LENSTRONOMY
DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC = 1.0e-4
DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC = 1.0e-3
DEFAULT_EXACT_IMAGE_LM_MAX_ITER = 30
DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC = 1.0
DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS = 8
DEFAULT_REFRESH_EVERY = 250
JAX_DEVICE_AUTO = "auto"
JAX_DEVICE_CPU = "cpu"
JAX_DEVICE_GPU = "gpu"
JAX_DEVICE_CHOICES = (JAX_DEVICE_AUTO, JAX_DEVICE_CPU, JAX_DEVICE_GPU)
IMAGE_PLANE_MODE_NONE = "none"
IMAGE_PLANE_MODE_LOCAL_JACOBIAN = "local-jacobian"
IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA = "linearized-forward-beta-image-plane"
IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED = "linearized-forward-beta-blocked-image-plane"
IMAGE_PLANE_MODE_FORWARD_METRIC = "forward-metric-image-plane"
IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA = "anchored-solved-forward-beta-image-plane"
IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE = "critical-arc-mixture-image-plane"
IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA = "fold-regularized-forward-beta-image-plane"
IMAGE_PLANE_MODE_CATASTROPHE_NORMAL_FORM = "catastrophe-normal-form-image-plane"
EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE = "linearized-forward-beta-image-plane"
EVIDENCE_LIKELIHOOD_MODES = (
    EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
)
DEFAULT_EVIDENCE_LIKELIHOOD_MODE = EVIDENCE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
DEFAULT_SMC_PARTICLES = 4096
DEFAULT_SMC_MCMC_KERNEL = "rmh"
SMC_MCMC_KERNELS = ("rmh", "mala")
DEFAULT_SMC_MCMC_STEPS = 4
DEFAULT_SMC_TARGET_ESS_FRAC = 0.8
DEFAULT_SMC_MAX_TEMPERATURE_STEPS = 256
DEFAULT_SMC_RMH_SCALE = 1.0
DEFAULT_SMC_MALA_STEP_SIZE = 0.05
DEFAULT_MICROCANONICAL_TUNE_FRAC1 = 0.1
DEFAULT_MICROCANONICAL_TUNE_FRAC2 = 0.1
DEFAULT_MICROCANONICAL_TUNE_FRAC3 = 0.1
DEFAULT_MICROCANONICAL_DIAGONAL_PRECONDITIONING = True
DEFAULT_MCLMC_DESIRED_ENERGY_VAR = 5.0e-4
DEFAULT_MCLMC_TRUST_IN_ESTIMATE = 1.5
DEFAULT_MCLMC_NUM_EFFECTIVE_SAMPLES = 150
DEFAULT_MCLMC_LFACTOR = 0.4
DEFAULT_MCHMC_TARGET_ACCEPT = 0.9
DEFAULT_MCHMC_RANDOM_TRAJECTORY_LENGTH = True
DEFAULT_MCHMC_L_PROPOSAL_FACTOR = float("inf")
DEFAULT_MCHMC_DIVERGENCE_THRESHOLD = 1000.0
DEFAULT_MCHMC_NUM_WINDOWS = 1
DEFAULT_MCHMC_TUNING_FACTOR = 1.3
MCHMC_L_ESTIMATORS = ("avg", "max")
DEFAULT_MCHMC_L_ESTIMATOR = "avg"
DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC = 2.0
DEFAULT_IMAGE_SIGMA_INT_LOWER_ARCSEC = 1.0e-3
DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC = 2.0
DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC = 1.0e-3
IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM = "log-uniform"
IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL = "lognormal"
IMAGE_PLANE_SCATTER_PRIORS = (
    IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM,
    IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL,
)
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR = IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC = 0.3
DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA = 0.5
DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC = 0.30
DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC = 0.10
DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS = 0.05
DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN = 0.05


DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN = 0.0
DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC = 0.0
DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS = 3
DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC = 0.3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE = 1.0e-3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE = 1.0e-6
DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC = 5.0
DEFAULT_CRITICAL_ARC_BASE_PROB = 0.10
DEFAULT_CRITICAL_ARC_MAX_PROB = 0.80
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD = 0.20
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS = 0.05
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD_PRIOR_MEDIAN = 0.15
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD_PRIOR_LOG_SIGMA = 0.5
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD_LOWER = 0.03
DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD_UPPER = 0.40
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS_PRIOR_MEDIAN = 0.05
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS_PRIOR_LOG_SIGMA = 0.5
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS_LOWER = 0.005
DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS_UPPER = 0.20
DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE = 1.0e-3
DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE = 1.0e-6
DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC = 20.0
DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD = 0.1
DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC = 5.0
DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC = 0.1
DEFAULT_FOLD_CURVATURE_ARCSEC_INV = 1.0
CATASTROPHE_LIKELIHOOD_MOMENT = "moment"
CATASTROPHE_LIKELIHOOD_ENVELOPE = "envelope"
CATASTROPHE_LIKELIHOODS = (
    CATASTROPHE_LIKELIHOOD_MOMENT,
    CATASTROPHE_LIKELIHOOD_ENVELOPE,
)
DEFAULT_CATASTROPHE_LIKELIHOOD = CATASTROPHE_LIKELIHOOD_MOMENT
DEFAULT_CATASTROPHE_LAMBDA_ON = 0.03
DEFAULT_CATASTROPHE_LAMBDA_OFF = 0.08
DEFAULT_CATASTROPHE_GAP_ON = 1.0e-5
DEFAULT_CATASTROPHE_GAP_OFF = 1.0e-3
DEFAULT_CATASTROPHE_TANGENTIAL_VARIANCE_MIN = 0.0
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN = "gaussian"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T = "student-t"
LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES = (
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN,
    LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_STUDENT_T,
)
DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS = LIKELIHOOD_STABILIZER_RESIDUAL_LOSS_GAUSSIAN
DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU = 4.0
POSTERIOR_DIAGNOSTIC_MODE_EXACT = "exact"
POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE = "approximate"
POSTERIOR_DIAGNOSTIC_MODES = (
    POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
)


class _ValidationRecoveryProgress:
    def __init__(self, args: Any | None = None) -> None:
        self.args = args
        self.enabled = not bool(getattr(args, "quiet", False))
        self._progress_cm: Any | None = None
        self._progress: Any | None = None
        self._parent_task: int | None = None
        self._parent_total = 0

    def __enter__(self) -> "_ValidationRecoveryProgress":
        if not self.enabled:
            return self
        self._progress_cm = _progress_context(
            self.args,
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._progress = self._progress_cm.__enter__()
        self._parent_task = self._progress.add_task("recovery: starting", total=0)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._progress is not None and self._parent_task is not None and exc_type is None:
            self._progress.update(self._parent_task, description="recovery: complete")
        if self._progress_cm is not None:
            return bool(self._progress_cm.__exit__(exc_type, exc, traceback))
        return False

    def begin_phase(self, description: str) -> None:
        if self._progress is None or self._parent_task is None:
            return
        self._parent_total += 1
        self._progress.update(
            self._parent_task,
            total=self._parent_total,
            description=f"recovery: {description}",
        )

    def advance_phase(self) -> None:
        if self._progress is None or self._parent_task is None:
            return
        self._progress.advance(self._parent_task)

    def add_subtask(self, description: str, total: int | None) -> int | None:
        if self._progress is None:
            return None
        return self._progress.add_task(description, total=total)

    def update_subtask(self, task_id: int | None, description: str) -> None:
        if self._progress is None or task_id is None:
            return
        self._progress.update(task_id, description=description)

    def advance_subtask(self, task_id: int | None) -> None:
        if self._progress is None or task_id is None:
            return
        self._progress.advance(task_id)


class _ValidationMockProgress:
    def __init__(self, args: Any | None = None) -> None:
        self.args = args
        self.enabled = not bool(getattr(args, "quiet", False))
        self._progress_cm: Any | None = None
        self._progress: Any | None = None
        self._redshift_task: int | None = None
        self._family_task: int | None = None

    def __enter__(self) -> "_ValidationMockProgress":
        if not self.enabled:
            return self
        self._progress_cm = _progress_context(
            self.args,
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        self._progress = self._progress_cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._progress_cm is not None:
            return bool(self._progress_cm.__exit__(exc_type, exc, traceback))
        return False

    def callback(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if event == "subhalos_complete":
            _log(
                self.args,
                (
                    "[load] mock subhalos "
                    f"selected={int(payload.get('selected_subhalos', 0))}/{int(payload.get('requested_subhalos', 0))} "
                    f"parent_candidates={int(payload.get('parent_count', 0))} "
                    f"observable={int(payload.get('observable_count', 0))} "
                    f"retries={int(payload.get('retry_count', 0))} "
                    f"mag_cut<={float(payload.get('mag_faint_limit', np.nan)):.4g}"
                ),
            )
        elif event == "subhalo_lensing_warning":
            _log(
                self.args,
                (
                    "[warning] mock member lensing signal is sub-detectable "
                    f"max_theta_E={float(payload.get('max_member_theta_e_arcsec', np.nan)):.4g}\" "
                    f"pos_sigma={float(payload.get('pos_sigma_arcsec', np.nan)):.4g}\" "
                    f"ratio={float(payload.get('max_member_theta_e_over_pos_sigma', np.nan)):.4g}"
                ),
            )
        elif event == "redshift_start":
            self._ensure_redshift_task(int(payload.get("redshift_count", 0)))
            self._update_redshift(payload, prefix="mock caustics")
        elif event == "redshift_complete":
            self._ensure_redshift_task(int(payload.get("redshift_count", 0)))
            self._update_redshift(payload, prefix="mock caustics")
            if self._progress is not None and self._redshift_task is not None:
                self._progress.advance(self._redshift_task)
        elif event == "family_start":
            self._ensure_family_task(int(payload.get("family_count", 0)))
            self._update_family(payload, image_count=None)
        elif event == "family_queue_start":
            _log(
                self.args,
                (
                    "[load] mock image queue "
                    f"families={int(payload.get('family_count', 0))} "
                    f"workers={int(payload.get('image_solver_workers', 0))} "
                    f"max_attempts={int(payload.get('max_attempts', 0))} "
                    f"queued={int(payload.get('queued_families', 0))}"
                ),
            )
        elif event == "family_attempt":
            self._ensure_family_task(int(payload.get("family_count", 0)))
            self._update_family(payload, image_count=int(payload.get("image_count", 0)))
        elif event == "family_accept":
            self._ensure_family_task(int(payload.get("family_count", 0)))
            self._update_family(payload, image_count=int(payload.get("image_count", 0)), accepted=True)
            if self._progress is not None and self._family_task is not None:
                self._progress.advance(self._family_task)
        elif event == "outputs_start":
            _log(
                self.args,
                (
                    "[load] writing mock outputs "
                    f"families={int(payload.get('family_count', 0))} "
                    f"images={int(payload.get('image_count', 0))} "
                    f"subhalos={int(payload.get('subhalo_count', 0))}"
                ),
            )
        elif event == "outputs_complete":
            _log(
                self.args,
                (
                    "[load] mock outputs complete "
                    f"images={int(payload.get('image_count', 0))} "
                    f"par={payload.get('par_path')} "
                    f"catalog={payload.get('image_catalog_path')} "
                    f"truth={payload.get('truth_path')}"
                ),
            )

    def _ensure_redshift_task(self, total: int) -> None:
        if self._progress is None or self._redshift_task is not None:
            return
        self._redshift_task = self._progress.add_task("mock caustics: starting", total=max(0, int(total)))

    def _ensure_family_task(self, total: int) -> None:
        if self._progress is None or self._family_task is not None:
            return
        self._family_task = self._progress.add_task("mock families: starting", total=max(0, int(total)))

    def _update_redshift(self, payload: dict[str, Any], *, prefix: str) -> None:
        if self._progress is None or self._redshift_task is None:
            return
        grid = int(payload.get("caustic_grid_pixels", 0))
        self._progress.update(
            self._redshift_task,
            description=(
                f"{prefix}: z={float(payload.get('z_source', np.nan)):.4g} "
                f"{int(payload.get('redshift_index', 0))}/{int(payload.get('redshift_count', 0))} "
                f"components={int(payload.get('lens_component_count', 0))} grid={grid}x{grid} "
                f"contours={int(payload.get('caustic_count', 0))}"
            ),
        )

    def _update_family(
        self,
        payload: dict[str, Any],
        *,
        image_count: int | None,
        accepted: bool = False,
    ) -> None:
        if self._progress is None or self._family_task is None:
            return
        image_text = "images=na" if image_count is None else f"images={int(image_count)}"
        status = "accepted" if accepted else "search"
        attempt = int(payload.get("attempt", 0))
        max_attempts = int(payload.get("max_attempts", 0))
        attempt_text = "attempt=0/0" if attempt <= 0 and max_attempts <= 0 else f"attempt={attempt}/{max_attempts}"
        self._progress.update(
            self._family_task,
            description=(
                f"mock families: {status} "
                f"{int(payload.get('family_index', 0))}/{int(payload.get('family_count', 0))} "
                f"class={payload.get('caustic_class', 'unknown')} "
                f"z={float(payload.get('z_source', np.nan)):.4g} "
                f"{attempt_text} {image_text}"
            ),
        )


CHIRES_COLUMNS = (
    "index",
    "family_id",
    "z",
    "n_arcs",
    "chi_total",
    "chi_x",
    "chi_y",
    "chi_a",
    "source_rms_arcsec",
    "image_rms_arcsec",
    "dx_arcsec",
    "dy_arcsec",
    "n_warn",
)


def _validation_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _validation_jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return _validation_dataframe_payload(value)
    if isinstance(value, pd.Series):
        return _validation_jsonable(value.tolist())
    if isinstance(value, np.ndarray):
        return _validation_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _validation_jsonable(value.item())
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(key): _validation_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_validation_jsonable(item) for item in value]
    return str(value)


def _validation_dataframe_payload(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in df.columns],
        "records": _validation_jsonable(df.to_dict(orient="records")),
    }


def _write_strict_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_validation_jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _parse_chires_float(value: str) -> float | None:
    if value.upper() == "N/A":
        return None
    return float(value)


def load_chires_table(path: str | Path) -> pd.DataFrame:
    """Load a Lenstool ``chires.dat`` table.

    The file includes one row per image plus one summary row per family. Numeric
    ``N/A`` cells are returned as missing values by pandas.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("chi ") or line.startswith("N "):
                continue
            parts = line.split()
            if len(parts) != len(CHIRES_COLUMNS):
                continue
            row: dict[str, Any] = {
                "index": int(parts[0]),
                "family_id": parts[1],
                "z": float(parts[2]),
                "n_arcs": int(parts[3]),
                "n_warn": int(parts[12]),
            }
            for column, raw_value in zip(CHIRES_COLUMNS[4:12], parts[4:12]):
                row[column] = _parse_chires_float(raw_value)
            rows.append(row)
    return pd.DataFrame(rows, columns=CHIRES_COLUMNS)


def load_chires_family_summary(path: str | Path) -> pd.DataFrame:
    """Return only family-summary rows from a Lenstool ``chires.dat`` table."""
    table = load_chires_table(path)
    if table.empty:
        return table
    summary = table[table["n_arcs"] > 1].copy()
    return summary.sort_values(["index", "family_id"]).reset_index(drop=True)


def parameter_recovery_table(
    samples: np.ndarray,
    parameter_names: list[str],
    truth: dict[str, float],
    *,
    best_fit: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize posterior recovery against known truth values."""
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2:
        raise ValueError("samples must be a 2D array.")
    if sample_array.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample columns.")
    best_array = None if best_fit is None else np.asarray(best_fit, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(parameter_names):
        values = sample_array[:, index]
        finite = values[np.isfinite(values)]
        truth_value = truth.get(name)
        if finite.size:
            q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
        else:
            q16 = median = q84 = np.nan
        if truth_value is None or not np.isfinite(float(truth_value)) or finite.size == 0:
            truth_percentile = np.nan
            bias = np.nan
            covered_68 = False
        else:
            truth_f = float(truth_value)
            truth_percentile = float(np.mean(finite <= truth_f))
            bias = float(median - truth_f)
            covered_68 = bool(q16 <= truth_f <= q84)
        rows.append(
            {
                "parameter": name,
                "truth": np.nan if truth_value is None else float(truth_value),
                "best_fit": np.nan if best_array is None else float(best_array[index]),
                "q16": float(q16),
                "median": float(median),
                "q84": float(q84),
                "bias": float(bias),
                "truth_percentile": float(truth_percentile),
                "covered_68": covered_68,
            }
        )
    return pd.DataFrame(rows)


def magnification_recovery_table(
    truth_images: pd.DataFrame,
    recovered: pd.DataFrame,
    *,
    epsilon: float = 1.0e-8,
) -> pd.DataFrame:
    merged = truth_images.merge(recovered, on="image_label", how="left", suffixes=("_truth", "_recovered"))
    # The recovered magnification is evaluated at the OBSERVED (noise-perturbed) image
    # positions, so compare against the truth magnification at those same positions when
    # the mock provides it; older mocks only carry the truth at the true positions.
    truth_column = (
        "magnification_true_at_obs" if "magnification_true_at_obs" in merged else "magnification_true"
    )
    mu_true = pd.to_numeric(merged[truth_column], errors="coerce")
    mu_rec = pd.to_numeric(merged["magnification_recovered"], errors="coerce")
    merged["magnification_bias"] = mu_rec - mu_true
    denom = np.maximum(np.abs(mu_true.to_numpy(dtype=float)), float(epsilon))
    merged["abs_magnification_fractional_error"] = (
        np.abs(np.abs(mu_rec.to_numpy(dtype=float)) - np.abs(mu_true.to_numpy(dtype=float))) / denom
    )
    merged["parity_match"] = np.sign(mu_true.to_numpy(dtype=float)) == np.sign(mu_rec.to_numpy(dtype=float))
    for suffix in ("q16", "q50", "q84"):
        column = f"magnification_{suffix}"
        if column in merged:
            values = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=float)
            merged[f"abs_magnification_fractional_error_{suffix}"] = (
                np.abs(np.abs(values) - np.abs(mu_true.to_numpy(dtype=float))) / denom
            )
    return merged


def _load_truth(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parameter_truth_with_source_positions(truth: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in dict(truth.get("parameter_truth", {})).items():
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values[str(key)] = value_f
    for source in truth.get("sources", []):
        if not isinstance(source, dict):
            continue
        family_id = source.get("family_id")
        if family_id is None:
            continue
        for source_key, suffix in (("beta_x", "beta_x"), ("beta_y", "beta_y")):
            if source_key not in source:
                continue
            try:
                value_f = float(source[source_key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_f):
                values[f"source.{family_id}.{suffix}"] = value_f
    return values


def _load_plot_bundle(path: str | Path) -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from ..cluster_solver import _load_artifacts

    artifacts_dir = Path(path)
    if artifacts_dir.name != "artifacts":
        artifacts_dir = artifacts_dir / "artifacts"
    return _load_artifacts(artifacts_dir)


def _artifact_parameter_names(state: Any) -> list[str]:
    return [str(spec.name) for spec in state.parameter_specs]


def _artifact_arg(artifact_args: dict[str, Any] | None, name: str, default: Any) -> Any:
    if not artifact_args:
        return default
    value = artifact_args.get(name, default)
    return default if value is None else value


def _recovered_model_tables(
    state: Any,
    best_fit_physical: np.ndarray,
    images: pd.DataFrame,
    *,
    quick_diagnostics: bool = False,
    progress: _ValidationRecoveryProgress | None = None,
    artifact_args: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from ..cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_BASE_PROB,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_MAX_PROB,
        DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS,
        DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_FINDER,
        DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_LM_MAX_ITER,
        DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC,
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
        SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
        _parse_refresh_every_value,
    )

    match_tolerance_arcsec = float(_artifact_arg(artifact_args, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=match_tolerance_arcsec,
        exact_image_min_distance_arcsec=float(
            _artifact_arg(artifact_args, "exact_image_min_distance_arcsec", DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC)
        ),
        exact_image_precision_limit=float(
            _artifact_arg(artifact_args, "exact_image_precision_limit", DEFAULT_EXACT_IMAGE_PRECISION_LIMIT)
        ),
        exact_image_num_iter_max=int(
            _artifact_arg(artifact_args, "exact_image_num_iter_max", DEFAULT_EXACT_IMAGE_NUM_ITER_MAX)
        ),
        exact_image_finder=str(_artifact_arg(artifact_args, "exact_image_finder", DEFAULT_EXACT_IMAGE_FINDER)),
        exact_image_displacement_tol_arcsec=float(
            _artifact_arg(
                artifact_args,
                "exact_image_displacement_tol_arcsec",
                DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
            )
        ),
        exact_image_identification_tol_arcsec=float(
            _artifact_arg(
                artifact_args,
                "exact_image_identification_tol_arcsec",
                DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
            )
        ),
        exact_image_lm_max_iter=int(_artifact_arg(artifact_args, "exact_image_lm_max_iter", DEFAULT_EXACT_IMAGE_LM_MAX_ITER)),
        exact_image_lm_trust_radius_arcsec=float(
            _artifact_arg(
                artifact_args,
                "exact_image_lm_trust_radius_arcsec",
                DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC,
            )
        ),
        exact_image_adaptive_max_levels=int(
            _artifact_arg(
                artifact_args,
                "exact_image_adaptive_max_levels",
                DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS,
            )
        ),
        sampling_engine="full_flat",
        active_scaling_galaxies=_artifact_arg(artifact_args, "active_scaling_galaxies", DEFAULT_ACTIVE_SCALING_GALAXIES),
        active_scaling_selection=str(_artifact_arg(artifact_args, "active_scaling_selection", "adaptive")),
        active_scaling_cumulative_fraction=float(
            _artifact_arg(artifact_args, "active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
        ),
        active_scaling_min=int(_artifact_arg(artifact_args, "active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
        refresh_every=_parse_refresh_every_value(_artifact_arg(artifact_args, "refresh_every", DEFAULT_REFRESH_EVERY)),
        refresh_param_drift_frac=float(_artifact_arg(artifact_args, "refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
        source_plane_covariance_floor=float(_artifact_arg(artifact_args, "source_plane_covariance_floor", 1.0e-6)),
        source_plane_covariance_mode=str(
            _artifact_arg(
                artifact_args,
                "source_plane_covariance_mode",
                SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
            )
        ),
        source_plane_outlier_sigma_arcsec=float(
            _artifact_arg(artifact_args, "source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
        ),
        sample_likelihood_mode=str(_artifact_arg(artifact_args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE)),
        image_plane_newton_steps=int(_artifact_arg(artifact_args, "image_plane_newton_steps", 0)),
        anchored_image_plane_solve_steps=int(
            _artifact_arg(artifact_args, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
        ),
        anchored_image_plane_trust_radius_arcsec=float(
            _artifact_arg(
                artifact_args,
                "anchored_image_plane_trust_radius_arcsec",
                DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
            )
        ),
        anchored_image_plane_lm_damping_relative=float(
            _artifact_arg(
                artifact_args,
                "anchored_image_plane_lm_damping_relative",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            )
        ),
        anchored_image_plane_lm_damping_absolute=float(
            _artifact_arg(
                artifact_args,
                "anchored_image_plane_lm_damping_absolute",
                DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
            )
        ),
        critical_arc_critical_direction_sigma_arcsec=float(
            _artifact_arg(artifact_args, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
        ),
        critical_arc_base_prob=float(_artifact_arg(artifact_args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)),
        critical_arc_max_prob=float(_artifact_arg(artifact_args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)),
        critical_arc_singular_threshold=float(
            _artifact_arg(artifact_args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        critical_arc_singular_softness=float(
            _artifact_arg(artifact_args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        critical_arc_lm_damping_relative=float(
            _artifact_arg(artifact_args, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
        ),
        critical_arc_lm_damping_absolute=float(
            _artifact_arg(artifact_args, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
        ),
        critical_arc_lm_trust_radius_arcsec=float(
            _artifact_arg(artifact_args, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
        ),
        arc_recovery_p_arc_threshold=float(
            _artifact_arg(artifact_args, "arc_recovery_p_arc_threshold", DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD)
        ),
        arc_aware_max_arclength_arcsec=float(
            _artifact_arg(artifact_args, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
        ),
        arc_aware_curve_step_arcsec=float(
            _artifact_arg(artifact_args, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
        ),
        use_magnitude_likelihood=bool(
            _artifact_arg(artifact_args, "use_magnitude_likelihood", DEFAULT_USE_MAGNITUDE_LIKELIHOOD)
        ),
        magnitude_sigma_floor=float(
            _artifact_arg(
                artifact_args,
                "magnitude_sigma_floor",
                _artifact_arg(artifact_args, "magnitude_sigma", DEFAULT_MAGNITUDE_SIGMA_FLOOR),
            )
        ),
        magnitude_mu_floor=float(_artifact_arg(artifact_args, "magnitude_mu_floor", DEFAULT_MAGNITUDE_MU_FLOOR)),
        magnitude_min_reliability=float(
            _artifact_arg(artifact_args, "magnitude_min_reliability", DEFAULT_MAGNITUDE_MIN_RELIABILITY)
        ),
        fold_curvature_arcsec_inv=float(
            _artifact_arg(artifact_args, "fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
        ),
        catastrophe_likelihood=str(_artifact_arg(artifact_args, "catastrophe_likelihood", DEFAULT_CATASTROPHE_LIKELIHOOD)),
        catastrophe_lambda_on=float(_artifact_arg(artifact_args, "catastrophe_lambda_on", DEFAULT_CATASTROPHE_LAMBDA_ON)),
        catastrophe_lambda_off=float(_artifact_arg(artifact_args, "catastrophe_lambda_off", DEFAULT_CATASTROPHE_LAMBDA_OFF)),
        catastrophe_gap_on=float(_artifact_arg(artifact_args, "catastrophe_gap_on", DEFAULT_CATASTROPHE_GAP_ON)),
        catastrophe_gap_off=float(_artifact_arg(artifact_args, "catastrophe_gap_off", DEFAULT_CATASTROPHE_GAP_OFF)),
        catastrophe_tangential_variance_min=float(
            _artifact_arg(
                artifact_args,
                "catastrophe_tangential_variance_min",
                DEFAULT_CATASTROPHE_TANGENTIAL_VARIANCE_MIN,
            )
        ),
        image_plane_scatter_floor_arcsec=float(
            _artifact_arg(artifact_args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
        ),
        fixed_image_sigma_int_arcsec=_artifact_arg(artifact_args, "fix_image_sigma_int_arcsec", None),
        evidence_source_prior_sigma_arcsec=_artifact_arg(artifact_args, "evidence_source_prior_sigma_arcsec", None),
        evidence_source_prior_mean_x_arcsec=float(_artifact_arg(artifact_args, "evidence_source_prior_mean_x_arcsec", 0.0)),
        evidence_source_prior_mean_y_arcsec=float(_artifact_arg(artifact_args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        quick_diagnostics=bool(quick_diagnostics),
    )
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(np.asarray(best_fit_physical, dtype=float))
    else:
        best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    magnification_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    best_predictions = evaluator.evaluate(best_fit_latent).family_predictions
    image_sigma_int = _shared_image_sigma_int_for_params(evaluator, best_fit_latent)
    covariance_floor = max(float(getattr(evaluator, "source_plane_covariance_floor", 0.0)), 0.0)
    progress_task = progress.add_subtask("recovered models: families", total=len(state.family_data)) if progress else None
    for family in state.family_data:
        if progress:
            progress.update_subtask(
                progress_task,
                f"recovered models: family={family.family_id} z={float(family.z_source):.4f}",
            )
        prediction = best_predictions.get(str(family.family_id), {})
        family_prediction = _shared_image_prediction_for_family_latent(
            evaluator,
            family,
            best_fit_latent,
            image_sigma_int,
            covariance_floor,
            quick_diagnostics=quick_diagnostics,
            source_prediction=prediction,
            magnification_column="magnification_recovered",
        )
        family_result = family_prediction["family_result"]
        magnification_rows.extend(family_prediction["magnification_rows"])
        image_rows.extend(family_prediction["image_rows"])
        source_rows.append(
            {
                "family_id": str(family.family_id),
                "source_x_recovered": float(family_result["source_x"]),
                "source_y_recovered": float(family_result["source_y"]),
                "source_plane_rms_arcsec": float(family_result["source_plane_rms"]),
                "exact_image_rms_arcsec": float(family_result["exact_image_rms"]),
                "arc_aware_image_rms_arcsec": float(family_result["arc_aware_image_rms"]),
                "arc_aware_recovered_image_count": int(
                    np.sum(family_result["point_recovered_mask"]) + np.sum(family_result["arc_supported_mask"])
                ),
                "arc_aware_missing_image_count": int(
                    max(0, int(family.n_images) - int(np.sum(family_result["point_recovered_mask"]) + np.sum(family_result["arc_supported_mask"])))
                ),
                "arc_supported_image_count": int(np.sum(family_result["arc_supported_mask"])),
                "failed": bool(prediction.get("failed", False) or family_result["exact_failed"]),
            }
        )
        if progress:
            progress.advance_subtask(progress_task)
    return pd.DataFrame(magnification_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows)


def _quantile_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    q16, q50, q84 = np.quantile(array, [0.16, 0.5, 0.84])
    return float(q16), float(q50), float(q84)


def _median_std_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(array))
    std = float(np.std(array))
    return median - std, median, median + std


RECOVERY_PROFILE_POSTERIOR_DRAW_CAP = 128


def _capped_evenly_spaced_posterior_draws(
    samples: np.ndarray,
    *,
    max_draws: int = RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)
    if int(max_draws) <= 0:
        raise ValueError("max_draws must be positive.")
    if sample_array.shape[0] <= int(max_draws):
        return sample_array
    indices = np.linspace(0, sample_array.shape[0] - 1, int(max_draws), dtype=int)
    return sample_array[indices]


def _nanmedian_no_warning(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


def _finite_std_no_warning(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.std(array))


def _finite_quantile_no_warning(values: Any, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.quantile(array, float(q)))


def _posterior_image_rms_summary(posterior_image_rms_df: pd.DataFrame) -> dict[str, float]:
    if posterior_image_rms_df.empty:
        return {}
    summary: dict[str, float] = {}
    specs = {
        "point_image_rms_arcsec": "posterior_point_image_rms_arcsec",
        "arc_aware_image_rms_arcsec": "posterior_arc_aware_image_rms_arcsec",
    }
    for column, prefix in specs.items():
        if column not in posterior_image_rms_df:
            continue
        values = pd.to_numeric(posterior_image_rms_df[column], errors="coerce").to_numpy(dtype=float)
        summary[f"{prefix}_q16"] = _finite_quantile_no_warning(values, 0.16)
        summary[f"{prefix}_median"] = _nanmedian_no_warning(values)
        summary[f"{prefix}_q84"] = _finite_quantile_no_warning(values, 0.84)
        summary[f"{prefix}_std"] = _finite_std_no_warning(values)
    count_specs = {
        "point_recovered_image_count": "posterior_point_recovered_image_count",
        "arc_aware_recovered_image_count": "posterior_arc_aware_recovered_image_count",
        "arc_supported_image_count": "posterior_arc_supported_image_count",
        "total_image_count": "posterior_total_image_count",
        "exact_failed_family_count": "posterior_exact_failed_family_count",
    }
    for column, prefix in count_specs.items():
        if column not in posterior_image_rms_df:
            continue
        values = pd.to_numeric(posterior_image_rms_df[column], errors="coerce").to_numpy(dtype=float)
        summary[f"{prefix}_median"] = _nanmedian_no_warning(values)
        summary[f"{prefix}_std"] = _finite_std_no_warning(values)
    return summary


def _posterior_image_residual_draw_rows(draw_index: int, family_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in family_results:
        family_id = str(result.get("family_id", ""))
        image_labels = [str(label) for label in result.get("image_labels", [])]
        residuals = np.asarray(result.get("residuals", []), dtype=float).reshape(-1)
        arc_residuals = np.asarray(result.get("arc_aware_residuals", []), dtype=float).reshape(-1)
        point_mask = np.asarray(result.get("point_recovered_mask", []), dtype=bool).reshape(-1)
        arc_mask = np.asarray(result.get("arc_supported_mask", []), dtype=bool).reshape(-1)
        n = int(min(len(image_labels), residuals.size, arc_residuals.size, point_mask.size, arc_mask.size))
        exact_failed = bool(result.get("exact_failed", False))
        for index in range(n):
            point_recovered = bool(point_mask[index]) and np.isfinite(float(residuals[index]))
            arc_supported = bool(arc_mask[index]) and np.isfinite(float(arc_residuals[index]))
            rows.append(
                {
                    "draw_index": int(draw_index),
                    "image_label": image_labels[index],
                    "family_id": family_id,
                    "point_image_residual_arcsec": float(residuals[index]) if point_recovered else np.nan,
                    "arc_aware_image_residual_arcsec": (
                        float(arc_residuals[index])
                        if arc_supported
                        else (float(residuals[index]) if point_recovered else np.nan)
                    ),
                    "point_recovered": bool(point_recovered),
                    "arc_aware_recovered": bool(point_recovered or arc_supported),
                    "arc_supported": bool(arc_supported),
                    "exact_failed": bool(exact_failed),
                }
            )
    return rows


_VALIDATION_STAGE_ORDER = (
    "stage0_fast_initializer",
    "stage1_backprojected_centroid_fit",
    "stage2_free_source_forward_fit",
)


def _finite_mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def _finite_median(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.median(array))


def _metric_text(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "na"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text if text else "na"
    if not np.isfinite(value_f):
        return "na"
    return f"{value_f:.{precision}g}"


def _stage_root_from_run_dir(solver_run_dir: str | Path) -> Path:
    run_dir = Path(solver_run_dir)
    if run_dir.name in _VALIDATION_STAGE_ORDER:
        return run_dir.parent
    return run_dir


def _validation_stage_dirs(solver_run_dir: str | Path) -> list[tuple[str, Path]]:
    root = _stage_root_from_run_dir(solver_run_dir)
    if (root / "tables" / "run_summary.json").exists():
        return [(root.name, root)]
    stages: list[tuple[str, Path]] = []
    for stage_name in _VALIDATION_STAGE_ORDER:
        stage_dir = root / stage_name
        if (stage_dir / "tables" / "run_summary.json").exists():
            stages.append((stage_name, stage_dir))
    return stages


def _load_stage_run_summary(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "tables" / "run_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_family_recovery_metrics(stage_dir: Path) -> dict[str, Any]:
    path = stage_dir / "tables" / "family_diagnostics.csv"
    if not path.exists():
        return {
            "family_count": np.nan,
            "exact_family_count": np.nan,
            "failed_or_missing_exact": np.nan,
            "exact_image_rms_mean": np.nan,
            "exact_image_rms_median": np.nan,
            "source_rms_mean": np.nan,
            "approx_image_rms_mean": np.nan,
        }
    try:
        family_df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return {
            "family_count": np.nan,
            "exact_family_count": np.nan,
            "failed_or_missing_exact": np.nan,
            "exact_image_rms_mean": np.nan,
            "exact_image_rms_median": np.nan,
            "source_rms_mean": np.nan,
            "approx_image_rms_mean": np.nan,
        }
    family_count = int(len(family_df))
    exact_values = (
        pd.to_numeric(family_df.get("exact_image_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    exact_finite = exact_values[np.isfinite(exact_values)]
    source_values = (
        pd.to_numeric(family_df.get("source_plane_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    approx_values = (
        pd.to_numeric(family_df.get("approx_image_rms_arcsec", pd.Series(dtype=float)), errors="coerce")
        .to_numpy(dtype=float)
    )
    return {
        "family_count": family_count,
        "exact_family_count": int(exact_finite.size),
        "failed_or_missing_exact": int(family_count - exact_finite.size),
        "exact_image_rms_mean": _finite_mean(exact_values),
        "exact_image_rms_median": _finite_median(exact_values),
        "source_rms_mean": _finite_mean(source_values),
        "approx_image_rms_mean": _finite_mean(approx_values),
    }


def _stage_parameter_recovery_metrics(stage_dir: Path, truth: dict[str, Any]) -> dict[str, Any]:
    default = {
        "truth_parameter_count": np.nan,
        "parameter_median_abs_bias": np.nan,
        "parameter_mean_abs_bias": np.nan,
        "parameter_coverage_68_fraction": np.nan,
        "worst_parameter": "na",
        "worst_parameter_abs_bias": np.nan,
    }
    try:
        state, _saved_args, arrays, _init_diagnostics = _load_plot_bundle(stage_dir)
    except Exception:
        return default
    if "samples" not in arrays or "best_fit" not in arrays:
        return default
    try:
        table = parameter_recovery_table(
            np.asarray(arrays["samples"], dtype=float),
            _artifact_parameter_names(state),
            _parameter_truth_with_source_positions(truth),
            best_fit=np.asarray(arrays["best_fit"], dtype=float),
        )
    except Exception:
        return default
    if table.empty or "bias" not in table or "truth" not in table:
        return default
    truth_values = pd.to_numeric(table["truth"], errors="coerce").to_numpy(dtype=float)
    bias_values = pd.to_numeric(table["bias"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(truth_values) & np.isfinite(bias_values)
    if not np.any(finite_mask):
        return default
    finite_bias = bias_values[finite_mask]
    abs_bias = np.abs(finite_bias)
    finite_table = table.loc[finite_mask].reset_index(drop=True)
    worst_index = int(np.nanargmax(abs_bias))
    coverage_values = finite_table["covered_68"].astype(float).to_numpy(dtype=float) if "covered_68" in finite_table else np.asarray([], dtype=float)
    return {
        "truth_parameter_count": int(abs_bias.size),
        "parameter_median_abs_bias": float(np.median(abs_bias)),
        "parameter_mean_abs_bias": float(np.mean(abs_bias)),
        "parameter_coverage_68_fraction": _finite_mean(coverage_values),
        "worst_parameter": str(finite_table.loc[worst_index, "parameter"]),
        "worst_parameter_abs_bias": float(abs_bias[worst_index]),
    }


def _collect_validation_stage_recovery_metrics(
    solver_run_dir: str | Path,
    truth_path: str | Path,
) -> list[dict[str, Any]]:
    truth = _load_truth(truth_path)
    rows: list[dict[str, Any]] = []
    for stage_name, stage_dir in _validation_stage_dirs(solver_run_dir):
        run_summary = _load_stage_run_summary(stage_dir)
        row: dict[str, Any] = {
            "stage": stage_name,
            "stage_dir": str(stage_dir),
            "fit_method": run_summary.get("fit_method", "na"),
            "sample_likelihood_mode": run_summary.get("sample_likelihood_mode", "na"),
            "sampler": run_summary.get("sampler", "na"),
            "runtime_sec": run_summary.get("runtime_sec", np.nan),
            "best_loglike": run_summary.get("best_loglike", np.nan),
            "accept_prob_mean": run_summary.get("accept_prob_mean", np.nan),
            "divergence_count": run_summary.get("divergence_count", np.nan),
            "mean_num_steps": run_summary.get("mean_num_steps", np.nan),
            "n_families": run_summary.get("n_families", np.nan),
            "n_images": run_summary.get("n_images", np.nan),
            "fit_cosmology_flat_wcdm": run_summary.get("fit_cosmology_flat_wcdm", False),
            "cosmology_Om0_median": run_summary.get("cosmology_Om0_median", np.nan),
            "cosmology_w0_median": run_summary.get("cosmology_w0_median", np.nan),
        }
        row.update(_stage_family_recovery_metrics(stage_dir))
        row.update(_stage_parameter_recovery_metrics(stage_dir, truth))
        rows.append(row)
    return rows


def _format_validation_run_summary(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    seed: int,
    solver_run_dir: str | Path,
) -> str:
    solver_run_path = Path(solver_run_dir)
    solver_root = _stage_root_from_run_dir(solver_run_path)
    final_stage = solver_run_path.name if solver_run_path.name in _VALIDATION_STAGE_ORDER else (rows[-1]["stage"] if rows else "na")

    def first_finite_from_end(key: str) -> Any:
        for row in reversed(rows):
            value = row.get(key, np.nan)
            try:
                if np.isfinite(float(value)):
                    return value
            except (TypeError, ValueError):
                continue
        return np.nan

    family_count = first_finite_from_end("n_families")
    image_count = first_finite_from_end("n_images")
    lines = [
        "Validation recovery run summary",
        f"run_name={run_name}",
        f"seed={seed}",
        f"solver_root={solver_root}",
        f"final_stage={final_stage}",
        f"families={_metric_text(family_count)} images={_metric_text(image_count)}",
        "",
    ]
    if not rows:
        lines.append("No stage summaries were found.")
        return "\n".join(lines) + "\n"
    columns = [
        ("stage", "stage"),
        ("fit", "fit_method"),
        ("likelihood", "sample_likelihood_mode"),
        ("sampler", "sampler"),
        ("runtime_s", "runtime_sec"),
        ("best_loglike", "best_loglike"),
        ("accept", "accept_prob_mean"),
        ("div", "divergence_count"),
        ("steps", "mean_num_steps"),
        ("fit_cosmo", "fit_cosmology_flat_wcdm"),
        ("Om0_med", "cosmology_Om0_median"),
        ("w0_med", "cosmology_w0_median"),
        ("families", "family_count"),
        ("exact_fams", "exact_family_count"),
        ("failed_exact", "failed_or_missing_exact"),
        ("exact_image_rms_mean", "exact_image_rms_mean"),
        ("exact_image_rms_median", "exact_image_rms_median"),
        ("source_rms_mean", "source_rms_mean"),
        ("approx_image_rms_mean", "approx_image_rms_mean"),
        ("truth_params", "truth_parameter_count"),
        ("param_med_abs_bias", "parameter_median_abs_bias"),
        ("param_mean_abs_bias", "parameter_mean_abs_bias"),
        ("coverage68", "parameter_coverage_68_fraction"),
    ]
    rendered_rows = [
        {header: _metric_text(row.get(key)) for header, key in columns}
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(rendered[header]) for rendered in rendered_rows))
        for header, _key in columns
    }
    lines.append("Per-stage metrics:")
    lines.append(" ".join(header.ljust(widths[header]) for header, _key in columns))
    lines.append(" ".join("-" * widths[header] for header, _key in columns))
    for rendered in rendered_rows:
        lines.append(" ".join(rendered[header].ljust(widths[header]) for header, _key in columns))
    lines.extend(["", "Largest parameter bias by stage:"])
    for row in rows:
        lines.append(
            (
                f"{row.get('stage', 'na')}: "
                f"worst_parameter={_metric_text(row.get('worst_parameter'))} "
                f"worst_parameter_abs_bias={_metric_text(row.get('worst_parameter_abs_bias'))}"
            )
        )
    return "\n".join(lines) + "\n"


def write_validation_run_summary(
    solver_run_dir: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
    *,
    run_name: str,
    seed: int,
) -> Path:
    rows = _collect_validation_stage_recovery_metrics(solver_run_dir, truth_path)
    text = _format_validation_run_summary(rows, run_name=run_name, seed=int(seed), solver_run_dir=solver_run_dir)
    path = Path(output_dir) / "run_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _posterior_prediction_uncertainty_tables(
    state: Any,
    samples_physical: np.ndarray,
    images: pd.DataFrame,
    *,
    max_draws: int = 8,
    posterior_diagnostic_mode: str = POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    progress: _ValidationRecoveryProgress | None = None,
    artifact_args: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from ..cluster_solver import (
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
        DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
        DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS,
        DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_BASE_PROB,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE,
        DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE,
        DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_CRITICAL_ARC_MAX_PROB,
        DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
        DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
        DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC,
        DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS,
        DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_FINDER,
        DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
        DEFAULT_EXACT_IMAGE_LM_MAX_ITER,
        DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC,
        DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC,
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
        SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
        _parse_refresh_every_value,
    )

    diagnostic_mode = str(posterior_diagnostic_mode)
    if diagnostic_mode not in POSTERIOR_DIAGNOSTIC_MODES:
        raise ValueError(
            f"posterior_diagnostic_mode must be one of {POSTERIOR_DIAGNOSTIC_MODES}; got {diagnostic_mode!r}."
        )
    use_exact_predictions = diagnostic_mode == POSTERIOR_DIAGNOSTIC_MODE_EXACT
    summary_fn = _quantile_summary if use_exact_predictions else _median_std_summary

    sample_array = np.asarray(samples_physical, dtype=float)
    if sample_array.ndim != 2 or sample_array.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if sample_array.shape[0] > max_draws:
        indices = np.linspace(0, sample_array.shape[0] - 1, max_draws, dtype=int)
        sample_array = sample_array[indices]

    match_tolerance_arcsec = float(_artifact_arg(artifact_args, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))

    def make_evaluator() -> Any:
        return ClusterJAXEvaluator(
            state=state,
            match_tolerance_arcsec=match_tolerance_arcsec,
            exact_image_min_distance_arcsec=float(
                _artifact_arg(artifact_args, "exact_image_min_distance_arcsec", DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC)
            ),
            exact_image_precision_limit=float(
                _artifact_arg(artifact_args, "exact_image_precision_limit", DEFAULT_EXACT_IMAGE_PRECISION_LIMIT)
            ),
            exact_image_num_iter_max=int(
                _artifact_arg(artifact_args, "exact_image_num_iter_max", DEFAULT_EXACT_IMAGE_NUM_ITER_MAX)
            ),
            exact_image_finder=str(_artifact_arg(artifact_args, "exact_image_finder", DEFAULT_EXACT_IMAGE_FINDER)),
            exact_image_displacement_tol_arcsec=float(
                _artifact_arg(
                    artifact_args,
                    "exact_image_displacement_tol_arcsec",
                    DEFAULT_EXACT_IMAGE_DISPLACEMENT_TOL_ARCSEC,
                )
            ),
            exact_image_identification_tol_arcsec=float(
                _artifact_arg(
                    artifact_args,
                    "exact_image_identification_tol_arcsec",
                    DEFAULT_EXACT_IMAGE_IDENTIFICATION_TOL_ARCSEC,
                )
            ),
            exact_image_lm_max_iter=int(
                _artifact_arg(artifact_args, "exact_image_lm_max_iter", DEFAULT_EXACT_IMAGE_LM_MAX_ITER)
            ),
            exact_image_lm_trust_radius_arcsec=float(
                _artifact_arg(
                    artifact_args,
                    "exact_image_lm_trust_radius_arcsec",
                    DEFAULT_EXACT_IMAGE_LM_TRUST_RADIUS_ARCSEC,
                )
            ),
            exact_image_adaptive_max_levels=int(
                _artifact_arg(
                    artifact_args,
                    "exact_image_adaptive_max_levels",
                    DEFAULT_EXACT_IMAGE_ADAPTIVE_MAX_LEVELS,
                )
            ),
            sampling_engine="full_flat",
            active_scaling_galaxies=_artifact_arg(artifact_args, "active_scaling_galaxies", DEFAULT_ACTIVE_SCALING_GALAXIES),
            active_scaling_selection=str(_artifact_arg(artifact_args, "active_scaling_selection", "adaptive")),
            active_scaling_cumulative_fraction=float(
                _artifact_arg(artifact_args, "active_scaling_cumulative_fraction", DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION)
            ),
            active_scaling_min=int(_artifact_arg(artifact_args, "active_scaling_min", DEFAULT_ACTIVE_SCALING_MIN)),
            refresh_every=_parse_refresh_every_value(_artifact_arg(artifact_args, "refresh_every", DEFAULT_REFRESH_EVERY)),
            refresh_param_drift_frac=float(_artifact_arg(artifact_args, "refresh_param_drift_frac", DEFAULT_REFRESH_PARAM_DRIFT_FRAC)),
            source_plane_covariance_floor=float(_artifact_arg(artifact_args, "source_plane_covariance_floor", 1.0e-6)),
            source_plane_covariance_mode=str(
                _artifact_arg(
                    artifact_args,
                    "source_plane_covariance_mode",
                    SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
                )
            ),
            source_plane_outlier_sigma_arcsec=float(
                _artifact_arg(artifact_args, "source_plane_outlier_sigma_arcsec", DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC)
            ),
            sample_likelihood_mode=str(_artifact_arg(artifact_args, "sample_likelihood_mode", SAMPLE_LIKELIHOOD_SOURCE)),
            image_plane_newton_steps=int(_artifact_arg(artifact_args, "image_plane_newton_steps", 0)),
            anchored_image_plane_solve_steps=int(
                _artifact_arg(artifact_args, "anchored_image_plane_solve_steps", DEFAULT_ANCHORED_IMAGE_PLANE_SOLVE_STEPS)
            ),
            anchored_image_plane_trust_radius_arcsec=float(
                _artifact_arg(
                    artifact_args,
                    "anchored_image_plane_trust_radius_arcsec",
                    DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
                )
            ),
            anchored_image_plane_lm_damping_relative=float(
                _artifact_arg(
                    artifact_args,
                    "anchored_image_plane_lm_damping_relative",
                    DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
                )
            ),
            anchored_image_plane_lm_damping_absolute=float(
                _artifact_arg(
                    artifact_args,
                    "anchored_image_plane_lm_damping_absolute",
                    DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
                )
            ),
            critical_arc_critical_direction_sigma_arcsec=float(
                _artifact_arg(artifact_args, "critical_arc_critical_direction_sigma_arcsec", DEFAULT_CRITICAL_ARC_CRITICAL_DIRECTION_SIGMA_ARCSEC)
            ),
            critical_arc_base_prob=float(_artifact_arg(artifact_args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)),
            critical_arc_max_prob=float(_artifact_arg(artifact_args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)),
            critical_arc_singular_threshold=float(
                _artifact_arg(artifact_args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
            ),
            critical_arc_singular_softness=float(
                _artifact_arg(artifact_args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
            ),
            critical_arc_lm_damping_relative=float(
                _artifact_arg(artifact_args, "critical_arc_lm_damping_relative", DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE)
            ),
            critical_arc_lm_damping_absolute=float(
                _artifact_arg(artifact_args, "critical_arc_lm_damping_absolute", DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE)
            ),
            critical_arc_lm_trust_radius_arcsec=float(
                _artifact_arg(artifact_args, "critical_arc_lm_trust_radius_arcsec", DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC)
            ),
            arc_recovery_p_arc_threshold=float(
                _artifact_arg(artifact_args, "arc_recovery_p_arc_threshold", DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD)
            ),
            arc_aware_max_arclength_arcsec=float(
                _artifact_arg(artifact_args, "arc_aware_max_arclength_arcsec", DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC)
            ),
            arc_aware_curve_step_arcsec=float(
                _artifact_arg(artifact_args, "arc_aware_curve_step_arcsec", DEFAULT_ARC_AWARE_CURVE_STEP_ARCSEC)
            ),
            use_magnitude_likelihood=bool(
                _artifact_arg(artifact_args, "use_magnitude_likelihood", DEFAULT_USE_MAGNITUDE_LIKELIHOOD)
            ),
            magnitude_sigma_floor=float(
                _artifact_arg(
                    artifact_args,
                    "magnitude_sigma_floor",
                    _artifact_arg(artifact_args, "magnitude_sigma", DEFAULT_MAGNITUDE_SIGMA_FLOOR),
                )
            ),
            magnitude_mu_floor=float(_artifact_arg(artifact_args, "magnitude_mu_floor", DEFAULT_MAGNITUDE_MU_FLOOR)),
            magnitude_min_reliability=float(
                _artifact_arg(artifact_args, "magnitude_min_reliability", DEFAULT_MAGNITUDE_MIN_RELIABILITY)
            ),
            fold_curvature_arcsec_inv=float(
                _artifact_arg(artifact_args, "fold_curvature_arcsec_inv", DEFAULT_FOLD_CURVATURE_ARCSEC_INV)
            ),
            catastrophe_likelihood=str(_artifact_arg(artifact_args, "catastrophe_likelihood", DEFAULT_CATASTROPHE_LIKELIHOOD)),
            catastrophe_lambda_on=float(_artifact_arg(artifact_args, "catastrophe_lambda_on", DEFAULT_CATASTROPHE_LAMBDA_ON)),
            catastrophe_lambda_off=float(_artifact_arg(artifact_args, "catastrophe_lambda_off", DEFAULT_CATASTROPHE_LAMBDA_OFF)),
            catastrophe_gap_on=float(_artifact_arg(artifact_args, "catastrophe_gap_on", DEFAULT_CATASTROPHE_GAP_ON)),
            catastrophe_gap_off=float(_artifact_arg(artifact_args, "catastrophe_gap_off", DEFAULT_CATASTROPHE_GAP_OFF)),
            catastrophe_tangential_variance_min=float(
                _artifact_arg(
                    artifact_args,
                    "catastrophe_tangential_variance_min",
                    DEFAULT_CATASTROPHE_TANGENTIAL_VARIANCE_MIN,
                )
            ),
            image_plane_scatter_floor_arcsec=float(
                _artifact_arg(artifact_args, "image_plane_scatter_floor_arcsec", DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
            ),
            fixed_image_sigma_int_arcsec=_artifact_arg(artifact_args, "fix_image_sigma_int_arcsec", None),
            evidence_source_prior_sigma_arcsec=_artifact_arg(artifact_args, "evidence_source_prior_sigma_arcsec", None),
            evidence_source_prior_mean_x_arcsec=float(_artifact_arg(artifact_args, "evidence_source_prior_mean_x_arcsec", 0.0)),
            evidence_source_prior_mean_y_arcsec=float(_artifact_arg(artifact_args, "evidence_source_prior_mean_y_arcsec", 0.0)),
        )

    evaluator = make_evaluator()
    worker_count = min(max(1, int(jax_cpu_worker_count())), max(1, len(state.family_data)))
    thread_local = threading.local()
    worker_evaluators: list[Any] = []
    worker_lock = threading.Lock()

    def family_task_evaluator() -> Any:
        if worker_count <= 1:
            return evaluator
        local_evaluator = getattr(thread_local, "evaluator", None)
        if local_evaluator is None:
            local_evaluator = make_evaluator()
            thread_local.evaluator = local_evaluator
            with worker_lock:
                worker_evaluators.append(local_evaluator)
        return local_evaluator

    family_ids = [str(family.family_id) for family in state.family_data]
    mag_by_label: dict[str, list[float]] = {}
    x_by_label: dict[str, list[float]] = {}
    y_by_label: dict[str, list[float]] = {}
    residual_by_label: dict[str, list[float]] = {}
    arc_residual_by_label: dict[str, list[float]] = {}
    source_x_by_family: dict[str, list[float]] = {}
    source_y_by_family: dict[str, list[float]] = {}
    source_rms_by_family: dict[str, list[float]] = {}
    exact_rms_by_family: dict[str, list[float]] = {}
    arc_rms_by_family: dict[str, list[float]] = {}
    exact_failed_families: set[str] = set()
    posterior_rms_rows: list[dict[str, Any]] = []
    posterior_residual_rows: list[dict[str, Any]] = []

    n_draws = int(sample_array.shape[0])
    n_families = len(state.family_data)
    progress_task = (
        progress.add_subtask("posterior uncertainty: draws x families", total=n_draws * n_families)
        if progress
        else None
    )

    def process_family_prediction(
        sample_latent: np.ndarray,
        prediction: dict[str, Any],
        family: Any,
        *,
        skip_exact: bool,
    ) -> dict[str, Any]:
        task_evaluator = family_task_evaluator()
        if skip_exact and use_exact_predictions:
            family_prediction = _shared_image_prediction_for_family_latent(
                task_evaluator,
                family,
                sample_latent,
                _shared_image_sigma_int_for_params(task_evaluator, sample_latent),
                max(float(getattr(task_evaluator, "source_plane_covariance_floor", 0.0)), 0.0),
                quick_diagnostics=True,
                source_prediction=prediction,
                magnification_column="magnification",
            )
            result = dict(family_prediction["family_result"])
            result["exact_failed"] = True
            return result
        family_prediction = _shared_image_prediction_for_family_latent(
            task_evaluator,
            family,
            sample_latent,
            _shared_image_sigma_int_for_params(task_evaluator, sample_latent),
            max(float(getattr(task_evaluator, "source_plane_covariance_floor", 0.0)), 0.0),
            quick_diagnostics=not use_exact_predictions,
            source_prediction=prediction,
            magnification_column="magnification",
        )
        return dict(family_prediction["family_result"])

    def merge_family_result(result: dict[str, Any]) -> None:
        family_id = str(result["family_id"])
        for label, value in zip(result["magnification_labels"], result["magnification"]):
            mag_by_label.setdefault(str(label), []).append(float(value))
        for label, x_model, y_model, residual, arc_residual in zip(
            result["image_labels"],
            result["x_pred"],
            result["y_pred"],
            result["residuals"],
            result["arc_aware_residuals"],
        ):
            label = str(label)
            if not use_exact_predictions and not np.isfinite(float(x_model) + float(y_model) + float(residual)):
                continue
            x_by_label.setdefault(label, []).append(float(x_model))
            y_by_label.setdefault(label, []).append(float(y_model))
            residual_by_label.setdefault(label, []).append(float(residual))
            arc_residual_by_label.setdefault(label, []).append(float(arc_residual))
        source_x_by_family.setdefault(family_id, []).append(float(result["source_x"]))
        source_y_by_family.setdefault(family_id, []).append(float(result["source_y"]))
        source_rms_by_family.setdefault(family_id, []).append(float(result["source_plane_rms"]))
        exact_rms_by_family.setdefault(family_id, []).append(float(result["exact_image_rms"]))
        arc_rms_by_family.setdefault(family_id, []).append(float(result["arc_aware_image_rms"]))

    executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    try:
        for draw_index, sample in enumerate(sample_array, start=1):
            if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
                sample_latent = evaluator.reported_physical_to_latent_parameter_vector(sample)
            else:
                sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
            family_predictions = evaluator._family_source_summary(sample_latent)
            if worker_count <= 1:
                draw_results: list[dict[str, Any]] = []
                for family in state.family_data:
                    family_id = str(family.family_id)
                    if progress:
                        progress.update_subtask(
                            progress_task,
                            (
                                f"posterior uncertainty: draw={draw_index}/{n_draws} "
                                f"family={family_id} z={float(family.z_source):.4f} "
                                f"failed_exact={len(exact_failed_families)}"
                            ),
                        )
                    result = process_family_prediction(
                        sample_latent,
                        family_predictions.get(family_id, {}),
                        family,
                        skip_exact=family_id in exact_failed_families,
                    )
                    if result["exact_failed"]:
                        exact_failed_families.add(family_id)
                    merge_family_result(result)
                    draw_results.append(result)
                    if progress:
                        progress.advance_subtask(progress_task)
                if use_exact_predictions:
                    posterior_rms_rows.append(_shared_posterior_image_rms_row(draw_index, draw_results))
                    posterior_residual_rows.extend(_posterior_image_residual_draw_rows(draw_index, draw_results))
                continue

            if executor is None:  # pragma: no cover - defensive guard
                raise RuntimeError("posterior uncertainty worker executor was not initialized.")
            results_by_index: dict[int, dict[str, Any]] = {}
            future_by_index = {}
            failed_exact_count_by_index = {}
            for family_index, family in enumerate(state.family_data):
                family_id = str(family.family_id)
                failed_exact_count_by_index[family_index] = len(exact_failed_families)
                future = executor.submit(
                    process_family_prediction,
                    sample_latent,
                    family_predictions.get(family_id, {}),
                    family,
                    skip_exact=family_id in exact_failed_families,
                )
                future_by_index[future] = family_index
            for future in as_completed(future_by_index):
                family_index = future_by_index[future]
                family = state.family_data[family_index]
                family_id = str(family.family_id)
                result = future.result()
                results_by_index[family_index] = result
                if result["exact_failed"]:
                    exact_failed_families.add(family_id)
                if progress:
                    progress.update_subtask(
                        progress_task,
                        (
                            f"posterior uncertainty: draw={draw_index}/{n_draws} "
                            f"family={family_id} z={float(family.z_source):.4f} "
                            f"failed_exact={failed_exact_count_by_index[family_index]}"
                        ),
                    )
                    progress.advance_subtask(progress_task)
            for family_index in range(len(state.family_data)):
                merge_family_result(results_by_index[family_index])
            if use_exact_predictions:
                draw_results = [results_by_index[index] for index in range(len(state.family_data))]
                posterior_rms_rows.append(
                    _shared_posterior_image_rms_row(
                        draw_index,
                        draw_results,
                    )
                )
                posterior_residual_rows.extend(_posterior_image_residual_draw_rows(draw_index, draw_results))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for local_evaluator in worker_evaluators:
            if hasattr(local_evaluator, "release_runtime_caches"):
                local_evaluator.release_runtime_caches()
        if hasattr(evaluator, "release_runtime_caches"):
            evaluator.release_runtime_caches()

    mag_rows: list[dict[str, Any]] = []
    for label, values in mag_by_label.items():
        q16, q50, q84 = summary_fn(values)
        mag_rows.append(
            {
                "image_label": label,
                "magnification_q16": q16,
                "magnification_q50": q50,
                "magnification_q84": q84,
            }
        )

    image_rows: list[dict[str, Any]] = []
    for label in sorted(set(x_by_label) | set(y_by_label) | set(residual_by_label) | set(arc_residual_by_label)):
        x16, x50, x84 = summary_fn(x_by_label.get(label, []))
        y16, y50, y84 = summary_fn(y_by_label.get(label, []))
        r16, r50, r84 = summary_fn(residual_by_label.get(label, []))
        arc_r16, arc_r50, arc_r84 = summary_fn(arc_residual_by_label.get(label, []))
        image_rows.append(
            {
                "image_label": label,
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
            }
        )

    source_rows: list[dict[str, Any]] = []
    for family_id in sorted(set(source_x_by_family) | set(source_y_by_family)):
        sx16, sx50, sx84 = summary_fn(source_x_by_family.get(family_id, []))
        sy16, sy50, sy84 = summary_fn(source_y_by_family.get(family_id, []))
        sr16, sr50, sr84 = summary_fn(source_rms_by_family.get(family_id, []))
        er16, er50, er84 = summary_fn(exact_rms_by_family.get(family_id, []))
        ar16, ar50, ar84 = summary_fn(arc_rms_by_family.get(family_id, []))
        source_rows.append(
            {
                "family_id": family_id,
                "source_x_q16": sx16,
                "source_x_q50": sx50,
                "source_x_q84": sx84,
                "source_y_q16": sy16,
                "source_y_q50": sy50,
                "source_y_q84": sy84,
                "source_plane_rms_q16": sr16,
                "source_plane_rms_q50": sr50,
                "source_plane_rms_q84": sr84,
                "exact_image_rms_q16": er16,
                "exact_image_rms_q50": er50,
                "exact_image_rms_q84": er84,
                "arc_aware_image_rms_q16": ar16,
                "arc_aware_image_rms_q50": ar50,
                "arc_aware_image_rms_q84": ar84,
            }
        )
    posterior_rms_df = pd.DataFrame(posterior_rms_rows)
    posterior_residual_df = pd.DataFrame(posterior_residual_rows)
    return pd.DataFrame(mag_rows), pd.DataFrame(image_rows), pd.DataFrame(source_rows), posterior_rms_df, posterior_residual_df


def _magnifications_for_images(state: Any, best_fit_physical: np.ndarray, images: pd.DataFrame) -> pd.DataFrame:
    magnification, _image, _source = _recovered_model_tables(state, best_fit_physical, images)
    return magnification


def _effective_scaling_component_index(
    record: dict[str, Any],
    component_family: np.ndarray,
    *,
    context: str,
) -> int:
    from ..cluster_solver import COMPONENT_FAMILY_INDEPENDENT_FREE, COMPONENT_FAMILY_SCALING

    n_components = int(component_family.size)
    try:
        component_index = int(record["component_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context} scaling record is missing a valid component_index.") from exc
    if component_index < 0 or component_index >= n_components:
        raise ValueError(
            f"{context} scaling record component_index={component_index} is outside packed component count {n_components}."
        )
    if int(component_family[component_index]) != COMPONENT_FAMILY_SCALING:
        raise ValueError(f"{context} component_index={component_index} is not a scaling component.")

    raw_free_component_index = record.get("free_component_index", -1)
    try:
        free_component_index = int(raw_free_component_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} scaling record component_index={component_index} has invalid "
            f"free_component_index={raw_free_component_index!r}."
        ) from exc
    if free_component_index < 0:
        return component_index
    if free_component_index >= n_components:
        raise ValueError(
            f"{context} scaling record component_index={component_index} has free_component_index="
            f"{free_component_index} outside packed component count {n_components}."
        )
    if int(component_family[free_component_index]) != COMPONENT_FAMILY_INDEPENDENT_FREE:
        raise ValueError(
            f"{context} scaling record component_index={component_index} has free_component_index="
            f"{free_component_index}, but that component is not an independent-free component."
        )
    return free_component_index


def _effective_scaling_component_records(state: Any, *, context: str) -> list[dict[str, Any]]:
    component_family = np.asarray(state.packed_lens_spec.component_family, dtype=int)
    records: list[dict[str, Any]] = []
    for raw_record in getattr(state, "scaling_component_records", []) or []:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        record["model_component_index"] = _effective_scaling_component_index(
            record,
            component_family,
            context=context,
        )
        records.append(record)
    return records


def _mass_profile_component_groups(state: Any) -> tuple[dict[str, list[int]], dict[str, str]]:
    from ..cluster_solver import COMPONENT_FAMILY_LARGE

    component_family = np.asarray(state.packed_lens_spec.component_family, dtype=int)
    n_components = len(state.lens_model_list)
    large_indices = np.where(component_family == COMPONENT_FAMILY_LARGE)[0].astype(int).tolist()
    subhalo_indices = [
        int(record["model_component_index"])
        for record in _effective_scaling_component_records(state, context="mass/profile")
    ]
    physical_indices = large_indices + subhalo_indices
    group_indices: dict[str, list[int]] = {
        "total": physical_indices,
        "halo": [0] if n_components > 0 else [],
        "bcg": [1] if n_components > 1 else [],
        "subhalos": subhalo_indices,
    }
    group_indices["bcg_plus_subhalos"] = group_indices["bcg"] + group_indices["subhalos"]
    display_names = {
        "total": "total",
        "halo": "halo",
        "bcg": "BCG",
        "subhalos": "subhalos",
        "bcg_plus_subhalos": "BCG + subhalos",
    }
    return group_indices, display_names


def _surface_density_annulus_edges(radii_arcsec: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii_arcsec, dtype=float)
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_arcsec must be a one-dimensional array of positive finite radii.")
    radii = np.sort(radii)
    if radii.size == 1:
        return np.asarray([0.5 * radii[0], 1.5 * radii[0]], dtype=float)
    midpoints = 0.5 * (radii[:-1] + radii[1:])
    first_width = midpoints[0] - radii[0]
    last_width = radii[-1] - midpoints[-1]
    first_edge = max(0.0, radii[0] - first_width)
    last_edge = radii[-1] + last_width
    return np.concatenate(([first_edge], midpoints, [last_edge])).astype(float)


def _validate_profile_component_indices(state: Any, group: str, indices: list[int]) -> None:
    n_components = int(np.asarray(state.packed_lens_spec.profile_type).size)
    invalid = [int(index) for index in indices if int(index) < 0 or int(index) >= n_components]
    if invalid:
        raise ValueError(
            "Mass/profile component group "
            f"{group!r} references component indices {invalid}; packed component count is {n_components}."
        )


def _current_truth_component_records(truth: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = truth.get("lens_components")
    if not isinstance(records, list):
        if "kwargs_lens" in truth or "kwargs_lens_by_source_redshift" in truth:
            raise ValueError(
                "Current mock truth schema requires lens_components records; positional kwargs_lens truth is legacy. "
                "Regenerate the mock."
            )
        raise ValueError("Current mock truth schema requires lens_components records.")

    by_id: dict[str, dict[str, Any]] = {}
    subhalo_catalog_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"lens_components[{index}] must be a record.")
        record = dict(raw_record)
        role = str(record.get("component_role", "")).strip()
        if role in {"subhalo_placeholder", "placeholder"} or record.get("physical") is False:
            raise ValueError("Mock truth contains placeholder/nonphysical subhalo component records. Regenerate the mock.")
        component_id = str(record.get("component_id", "")).strip()
        if not component_id:
            raise ValueError(f"lens_components[{index}] is missing component_id.")
        if component_id in by_id:
            raise ValueError(f"Duplicate truth component_id {component_id!r}.")
        if role == "subhalo":
            catalog_id = str(record.get("catalog_id", component_id)).strip()
            if not catalog_id:
                raise ValueError(f"Subhalo truth component {component_id!r} is missing catalog_id.")
            if catalog_id in subhalo_catalog_ids:
                raise ValueError(f"Duplicate subhalo truth catalog_id {catalog_id!r}.")
            subhalo_catalog_ids.add(catalog_id)
            record["catalog_id"] = catalog_id
        by_id[component_id] = record
    return by_id


def _truth_record_kwargs_for_redshift(record: dict[str, Any], z_source: float, component_label: str) -> dict[str, Any]:
    kwargs_by_z = record.get("kwargs_by_source_redshift")
    if not isinstance(kwargs_by_z, dict):
        raise ValueError(f"Truth component {component_label!r} is missing kwargs_by_source_redshift.")
    exact_key = f"{float(z_source):.8f}"
    if exact_key in kwargs_by_z:
        kwargs = kwargs_by_z[exact_key]
    else:
        matches: list[Any] = []
        for key, value in kwargs_by_z.items():
            try:
                key_z = float(key)
            except (TypeError, ValueError):
                continue
            if abs(key_z - float(z_source)) <= 1.0e-8:
                matches.append(value)
        if len(matches) != 1:
            raise ValueError(f"Truth component {component_label!r} is missing source redshift {exact_key}.")
        kwargs = matches[0]
    if not isinstance(kwargs, dict):
        raise ValueError(f"Truth component {component_label!r} kwargs for z={exact_key} must be a record.")
    return dict(kwargs)


def _scaling_component_records_by_index(state: Any) -> dict[int, dict[str, Any]]:
    records = getattr(state, "scaling_component_records", None)
    if records is None:
        records = []
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            component_index = int(record["component_index"])
        except (KeyError, TypeError, ValueError):
            continue
        result[component_index] = dict(record)
    return result


def _packed_lens_state_from_current_truth_components(state: Any, truth: dict[str, Any], z_source: float) -> Any:
    from ..cluster_solver import (
        COMPONENT_FAMILY_INDEPENDENT_FREE,
        COMPONENT_FAMILY_LARGE,
        COMPONENT_FAMILY_SCALING,
        DP_IE_PROFILE,
        SHEAR_PROFILE,
        PackedLensState,
    )

    profile_type = np.asarray(state.packed_lens_spec.profile_type, dtype=np.int32)
    component_family = np.asarray(state.packed_lens_spec.component_family, dtype=int)
    n_components = int(profile_type.size)
    if int(component_family.size) != n_components:
        raise ValueError(
            "Packed component metadata mismatch: "
            f"component_family has {int(component_family.size)} entries for {n_components} profile entries."
        )
    truth_by_id = _current_truth_component_records(truth)
    scaling_records = _scaling_component_records_by_index(state)
    subhalo_truth_by_catalog = {
        str(record["catalog_id"]): record
        for record in truth_by_id.values()
        if str(record.get("component_role", "")) == "subhalo"
    }
    matched_truth_ids: set[str] = set()

    sigma0 = np.zeros(n_components, dtype=float)
    Ra = np.zeros(n_components, dtype=float)
    Rs = np.zeros(n_components, dtype=float)
    e1 = np.zeros(n_components, dtype=float)
    e2 = np.zeros(n_components, dtype=float)
    center_x = np.zeros(n_components, dtype=float)
    center_y = np.zeros(n_components, dtype=float)
    gamma1 = np.zeros(n_components, dtype=float)
    gamma2 = np.zeros(n_components, dtype=float)

    for index, profile in enumerate(profile_type.tolist()):
        family = int(component_family[index])
        target_index = int(index)
        if family == COMPONENT_FAMILY_INDEPENDENT_FREE:
            continue
        if family == COMPONENT_FAMILY_LARGE:
            if index == 0:
                truth_id = "halo"
            elif index == 1:
                truth_id = "bcg"
            else:
                raise ValueError(
                    "Packed large component "
                    f"index {index} has no current mock truth identity metadata."
                )
            record = truth_by_id.get(truth_id)
            if record is None:
                raise ValueError(f"Packed physical component {truth_id!r} has no matching truth component.")
            component_label = truth_id
            matched_truth_ids.add(truth_id)
        elif family == COMPONENT_FAMILY_SCALING:
            scaling_record = scaling_records.get(index)
            if scaling_record is None:
                raise ValueError(f"Packed scaling component index {index} is missing scaling_component_records metadata.")
            target_index = _effective_scaling_component_index(
                scaling_record,
                component_family,
                context="truth profile",
            )
            catalog_id = str(scaling_record.get("catalog_id", "")).strip()
            if not catalog_id:
                raise ValueError(f"Packed scaling component index {index} is missing catalog_id metadata.")
            record = subhalo_truth_by_catalog.get(catalog_id)
            if record is None:
                raise ValueError(f"Packed scaling component catalog_id {catalog_id!r} has no matching truth component.")
            component_label = f"subhalo catalog_id={catalog_id}"
            matched_truth_ids.add(str(record["component_id"]))
        else:
            raise ValueError(f"Packed component index {index} has unsupported component_family={family}.")

        kwargs = _truth_record_kwargs_for_redshift(record, z_source, component_label)
        target_profile = int(profile_type[target_index])
        if target_profile == DP_IE_PROFILE:
            missing = [field for field in ("sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y") if field not in kwargs]
            if missing:
                raise ValueError(f"Truth component {component_label!r} missing DPIE fields: {missing}.")
            sigma0[target_index] = float(kwargs["sigma0"])
            Ra[target_index] = float(kwargs["Ra"])
            Rs[target_index] = float(kwargs["Rs"])
            e1[target_index] = float(kwargs["e1"])
            e2[target_index] = float(kwargs["e2"])
            center_x[target_index] = float(kwargs["center_x"])
            center_y[target_index] = float(kwargs["center_y"])
        elif target_profile == SHEAR_PROFILE:
            missing = [field for field in ("gamma1", "gamma2") if field not in kwargs]
            if missing:
                raise ValueError(f"Truth component {component_label!r} missing SHEAR fields: {missing}.")
            gamma1[target_index] = float(kwargs["gamma1"])
            gamma2[target_index] = float(kwargs["gamma2"])
        else:
            raise ValueError(f"Unsupported profile_type={target_profile} in current mock truth component layout.")

    physical_truth_ids = {
        str(record["component_id"])
        for record in truth_by_id.values()
        if str(record.get("component_role", "")) in {"halo", "bcg", "subhalo"}
    }
    unmatched_truth_ids = sorted(physical_truth_ids - matched_truth_ids)
    if unmatched_truth_ids:
        raise ValueError(
            "Current mock truth contains physical components with no matching packed solver component: "
            f"{unmatched_truth_ids}."
        )

    import jax.numpy as jnp

    return PackedLensState(
        profile_type=jnp.asarray(profile_type, dtype=jnp.int32),
        sigma0=jnp.asarray(sigma0, dtype=jnp.float64),
        Ra=jnp.asarray(Ra, dtype=jnp.float64),
        Rs=jnp.asarray(Rs, dtype=jnp.float64),
        e1=jnp.asarray(e1, dtype=jnp.float64),
        e2=jnp.asarray(e2, dtype=jnp.float64),
        center_x=jnp.asarray(center_x, dtype=jnp.float64),
        center_y=jnp.asarray(center_y, dtype=jnp.float64),
        gamma1=jnp.asarray(gamma1, dtype=jnp.float64),
        gamma2=jnp.asarray(gamma2, dtype=jnp.float64),
    )


def _static_lens_state_from_packed(packed_state: Any) -> jax_lensing.StaticLensState:
    return jax_lensing.StaticLensState(
        profile_type=packed_state.profile_type,
        sigma0=packed_state.sigma0,
        Ra=packed_state.Ra,
        Rs=packed_state.Rs,
        e1=packed_state.e1,
        e2=packed_state.e2,
        center_x=packed_state.center_x,
        center_y=packed_state.center_y,
        gamma1=packed_state.gamma1,
        gamma2=packed_state.gamma2,
    )


def _grouped_deflection_magnitude_arcsec(
    evaluator: Any,
    packed_state: Any,
    radius_arcsec: float,
    indices: list[int],
) -> float:
    if not indices:
        return 0.0
    import jax.numpy as jnp

    x = jnp.asarray([float(radius_arcsec)], dtype=jnp.float64)
    y = jnp.asarray([0.0], dtype=jnp.float64)
    alpha_x, alpha_y, *_unused = evaluator._grouped_alpha_and_hessian_for_components(
        x,
        y,
        packed_state,
        np.asarray(indices, dtype=np.int32),
    )
    return float(np.hypot(float(np.asarray(alpha_x)[0]), float(np.asarray(alpha_y)[0])))


def _annular_surface_density_msun_per_arcsec2(
    evaluator: Any,
    packed_state: Any,
    indices: list[int],
    radii_arcsec: np.ndarray,
    sigma_crit_angle: float,
    *,
    n_radial: int = 80,
    n_azimuth: int = 96,
) -> np.ndarray:
    if not indices:
        return np.zeros_like(np.asarray(radii_arcsec, dtype=float), dtype=float)
    import jax.numpy as jnp

    radii = np.asarray(radii_arcsec, dtype=float)
    edges = _surface_density_annulus_edges(radii)
    theta = (np.arange(int(n_azimuth), dtype=float) + 0.5) * (2.0 * np.pi / float(n_azimuth))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    values: list[float] = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        if outer <= inner:
            values.append(np.nan)
            continue
        area_fraction = (np.arange(int(n_radial), dtype=float) + 0.5) / float(n_radial)
        radial = np.sqrt(inner * inner + area_fraction * (outer * outer - inner * inner))
        x = (radial[:, None] * cos_theta[None, :]).reshape(-1)
        y = (radial[:, None] * sin_theta[None, :]).reshape(-1)
        _alpha_x, _alpha_y, h_xx, _h_xy, _h_yx, h_yy = evaluator._grouped_alpha_and_hessian_for_components(
            jnp.asarray(x, dtype=jnp.float64),
            jnp.asarray(y, dtype=jnp.float64),
            packed_state,
            np.asarray(indices, dtype=np.int32),
        )
        kappa = 0.5 * (np.asarray(h_xx, dtype=float) + np.asarray(h_yy, dtype=float))
        mean_kappa = float(np.nanmean(kappa)) if kappa.size else np.nan
        values.append(mean_kappa * float(sigma_crit_angle))
    return np.asarray(values, dtype=float)


def _deflection_profile_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
) -> pd.DataFrame:
    mass_df, _surface_df = _mass_and_surface_density_profiles_for_samples(state, samples, truth, radii_arcsec)
    return mass_df


def _surface_density_profile_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
) -> pd.DataFrame:
    _mass_df, surface_df = _mass_and_surface_density_profiles_for_samples(state, samples, truth, radii_arcsec)
    return surface_df


def _mass_and_surface_density_profiles_for_samples(
    state: Any,
    samples: np.ndarray,
    truth: dict[str, Any],
    radii_arcsec: np.ndarray,
    *,
    progress: _ValidationRecoveryProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import jax.numpy as jnp

    from ..cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = truth["config"]
    cosmo_config = flat_wcdm_config(h0=70.0, om0=0.3)
    z_lens = float(config["z_lens"])
    z_source = float(config["source_redshift"])

    sigma_crit_angle = critical_surface_density_angle_from_config(z_lens, z_source, cosmo_config)

    def make_evaluator() -> Any:
        return ClusterJAXEvaluator(
            state=state,
            match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
            sampling_engine="full_flat",
            active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
            active_scaling_selection="adaptive",
            active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
            active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
            refresh_every=DEFAULT_REFRESH_EVERY,
            refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        )

    group_indices, display_names = _mass_profile_component_groups(state)
    for group, indices in group_indices.items():
        _validate_profile_component_indices(state, group, indices)
    truth_packed_state = _packed_lens_state_from_current_truth_components(state, truth, z_source)

    def empty_group_radius_values() -> dict[tuple[str, float], list[float]]:
        return {(group, float(radius)): [] for group in group_indices for radius in radii_arcsec}

    mass_values_by_group_radius = empty_group_radius_values()
    surface_values_by_group_radius = empty_group_radius_values()
    sample_array = np.asarray(samples, dtype=float)
    n_draws = int(sample_array.shape[0])
    worker_count = min(max(1, int(jax_cpu_worker_count())), max(1, n_draws))
    progress_task = (
        progress.add_subtask("profile bands: posterior draws", total=n_draws)
        if progress
        else None
    )
    worker_local = threading.local()
    worker_evaluators: list[Any] = []
    worker_lock = threading.Lock()
    serial_evaluator: Any | None = None
    truth_evaluator: Any | None = None

    def worker_context() -> Any:
        cached = getattr(worker_local, "profile_context", None)
        if cached is None:
            cached = make_evaluator()
            worker_local.profile_context = cached
            with worker_lock:
                worker_evaluators.append(cached)
        return cached

    def sample_profile_values(
        sample_index: int,
        sample: np.ndarray,
        evaluator: Any,
    ) -> tuple[int, dict[tuple[str, float], float], dict[tuple[str, float], float]]:
        sample_latent = _convert_theta_to_latent(sample, state.parameter_specs)
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(sample_latent, dtype=jnp.float64), z_source)
        mass_values: dict[tuple[str, float], float] = {}
        surface_values: dict[tuple[str, float], float] = {}
        for radius in radii_arcsec:
            radius_f = float(radius)
            for group, indices in group_indices.items():
                mass_values[(group, radius_f)] = _grouped_deflection_magnitude_arcsec(
                    evaluator,
                    packed_state,
                    radius_f,
                    indices,
                )
        for group, indices in group_indices.items():
            values = _annular_surface_density_msun_per_arcsec2(
                evaluator,
                packed_state,
                indices,
                radii_arcsec,
                sigma_crit_angle,
            )
            for radius, value in zip(radii_arcsec, values):
                surface_values[(group, float(radius))] = float(value)
        return sample_index, mass_values, surface_values

    def threaded_sample_profile_values(
        sample_index: int,
        sample: np.ndarray,
    ) -> tuple[int, dict[tuple[str, float], float], dict[tuple[str, float], float]]:
        evaluator = worker_context()
        return sample_profile_values(sample_index, sample, evaluator)

    def merge_sample_result(
        result: tuple[int, dict[tuple[str, float], float], dict[tuple[str, float], float]],
    ) -> None:
        _sample_index, mass_values, surface_values = result
        for key, value in mass_values.items():
            mass_values_by_group_radius[key].append(float(value))
        for key, value in surface_values.items():
            surface_values_by_group_radius[key].append(float(value))

    try:
        if worker_count <= 1 or n_draws <= 1:
            serial_evaluator = make_evaluator()
            for sample_index, sample in enumerate(sample_array, start=1):
                if progress:
                    progress.update_subtask(
                        progress_task,
                        f"profile bands: draw={sample_index}/{n_draws}",
                    )
                merge_sample_result(sample_profile_values(sample_index, sample, serial_evaluator))
                if progress:
                    progress.advance_subtask(progress_task)
        else:
            results_by_index: dict[int, tuple[int, dict[tuple[str, float], float], dict[tuple[str, float], float]]] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_by_index = {
                    executor.submit(threaded_sample_profile_values, sample_index, sample): sample_index
                    for sample_index, sample in enumerate(sample_array, start=1)
                }
                for future in as_completed(future_by_index):
                    sample_index = future_by_index[future]
                    if progress:
                        progress.update_subtask(
                            progress_task,
                            f"profile bands: draw={sample_index}/{n_draws} workers={worker_count}",
                        )
                    results_by_index[sample_index] = future.result()
                    if progress:
                        progress.advance_subtask(progress_task)
            for sample_index in range(1, n_draws + 1):
                merge_sample_result(results_by_index[sample_index])
    finally:
        if serial_evaluator is not None and hasattr(serial_evaluator, "release_runtime_caches"):
            serial_evaluator.release_runtime_caches()
        for local_evaluator in worker_evaluators:
            if hasattr(local_evaluator, "release_runtime_caches"):
                local_evaluator.release_runtime_caches()

    truth_evaluator = make_evaluator()
    try:
        truth_surface_values_by_group = {
            group: _annular_surface_density_msun_per_arcsec2(
                truth_evaluator,
                truth_packed_state,
                indices,
                radii_arcsec,
                sigma_crit_angle,
            )
            for group, indices in group_indices.items()
        }
        mass_rows: list[dict[str, Any]] = []
        surface_rows: list[dict[str, Any]] = []
        for group, indices in group_indices.items():
            if group in {"bcg", "subhalos"} and not indices:
                continue
            for radius_index, radius in enumerate(radii_arcsec):
                radius_f = float(radius)
                mass_finite = np.asarray(mass_values_by_group_radius[(group, radius_f)], dtype=float)
                mass_finite = mass_finite[np.isfinite(mass_finite)]
                mass_q16, mass_median, mass_q84 = (
                    np.quantile(mass_finite, [0.16, 0.5, 0.84]) if mass_finite.size else (np.nan, np.nan, np.nan)
                )
                mass_truth = _grouped_deflection_magnitude_arcsec(
                    truth_evaluator,
                    truth_packed_state,
                    radius_f,
                    indices,
                )
                mass_rows.append(
                    {
                        "radius_arcsec": radius_f,
                        "component": group,
                        "component_label": display_names[group],
                        "quantity": f"{group}_deflection_magnitude_arcsec",
                        "truth": mass_truth,
                        "q16": float(mass_q16),
                        "median": float(mass_median),
                        "q84": float(mass_q84),
                        "bias": float(mass_median - mass_truth),
                    }
                )

                surface_finite = np.asarray(surface_values_by_group_radius[(group, radius_f)], dtype=float)
                surface_finite = surface_finite[np.isfinite(surface_finite)]
                surface_q16, surface_median, surface_q84 = (
                    np.quantile(surface_finite, [0.16, 0.5, 0.84]) if surface_finite.size else (np.nan, np.nan, np.nan)
                )
                surface_truth = float(truth_surface_values_by_group[group][radius_index])
                surface_rows.append(
                    {
                        "radius_arcsec": radius_f,
                        "component": group,
                        "component_label": display_names[group],
                        "quantity": f"{group}_surface_density_msun_per_arcsec2",
                        "truth": surface_truth,
                        "q16": float(surface_q16),
                        "median": float(surface_median),
                        "q84": float(surface_q84),
                        "bias": float(surface_median - surface_truth),
                    }
                )
    finally:
        if truth_evaluator is not None and hasattr(truth_evaluator, "release_runtime_caches"):
            truth_evaluator.release_runtime_caches()
    return pd.DataFrame(mass_rows), pd.DataFrame(surface_rows)


def _recovered_caustic_contours_by_z(
    state: Any,
    best_fit_physical: np.ndarray,
    truth: dict[str, Any],
    z_keys: list[str],
    *,
    caustic_grid_scale_arcsec: float | None = None,
    progress: _ValidationRecoveryProgress | None = None,
) -> dict[str, list[CausticContour]]:
    import jax.numpy as jnp

    from ..cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    config = _plot_caustic_config_from_truth(
        truth,
        caustic_grid_scale_arcsec=caustic_grid_scale_arcsec,
    )
    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full_flat",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(np.asarray(best_fit_physical, dtype=float))
    else:
        best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    contours_by_z: dict[str, list[CausticContour]] = {}
    progress_task = progress.add_subtask("recovered caustics: redshifts", total=len(z_keys)) if progress else None
    for z_key in z_keys:
        if progress:
            progress.update_subtask(progress_task, f"recovered caustics: z={z_key}")
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            if progress:
                progress.advance_subtask(progress_task)
            continue
        packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
        contours = _compute_tangential_caustic_contours(_static_lens_state_from_packed(packed_state), config)
        if contours:
            contours_by_z[str(z_key)] = contours
        if progress:
            progress.advance_subtask(progress_task)
    return contours_by_z


def _plot_caustic_config_from_truth(
    truth: dict[str, Any],
    *,
    caustic_grid_scale_arcsec: float | None = None,
) -> SingleBCGMockConfig:
    config = _caustic_config_from_truth(truth)
    if caustic_grid_scale_arcsec is None:
        return config
    return replace(config, caustic_grid_scale_arcsec=float(caustic_grid_scale_arcsec))


def _truth_caustic_contours_by_z_for_plot(
    state: Any,
    truth: dict[str, Any],
    z_keys: list[str],
    *,
    caustic_grid_scale_arcsec: float | None = None,
    progress: _ValidationRecoveryProgress | None = None,
) -> dict[str, list[CausticContour]]:
    config = _plot_caustic_config_from_truth(
        truth,
        caustic_grid_scale_arcsec=caustic_grid_scale_arcsec,
    )
    contours_by_z: dict[str, list[CausticContour]] = {}
    progress_task = progress.add_subtask("truth plot caustics: redshifts", total=len(z_keys)) if progress else None
    for z_key in z_keys:
        if progress:
            progress.update_subtask(progress_task, f"truth plot caustics: z={z_key}")
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            if progress:
                progress.advance_subtask(progress_task)
            continue
        packed_state = _packed_lens_state_from_current_truth_components(state, truth, z_source)
        contours = _compute_tangential_caustic_contours(_static_lens_state_from_packed(packed_state), config)
        if contours:
            contours_by_z[str(z_key)] = contours
        if progress:
            progress.advance_subtask(progress_task)
    return contours_by_z


def _caustic_contour_payload(contours_by_z: dict[str, list[CausticContour]]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(z_source): [_validation_jsonable(contour) for contour in contours]
        for z_source, contours in sorted(contours_by_z.items(), key=lambda item: str(item[0]))
    }


def _recovery_payload_from_tables(
    *,
    run_dir: Path,
    output_dir: Path,
    posterior_image_diagnostic_draws: int,
    posterior_truth_recovery_draws: int,
    posterior_truth_recovery_draws_effective: int,
    posterior_truth_recovery_mode: str,
    diagnostic_worker_count: int,
    posterior_diagnostic_mode: str,
    quick_diagnostics: bool,
    samples: np.ndarray,
    best_fit_values: dict[str, float],
    previous_stage_best_values: Any,
    parameter_names: list[str],
    parameter_df: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    magnification_df: pd.DataFrame,
    posterior_image_rms_df: pd.DataFrame,
    posterior_image_residual_draws_df: pd.DataFrame,
    mass_profile_df: pd.DataFrame,
    surface_density_df: pd.DataFrame,
    summary: dict[str, float],
    summary_uncertainty: dict[str, tuple[float, float]],
    truth_caustics_by_z: dict[str, list[CausticContour]],
    recovered_caustics_by_z: dict[str, list[CausticContour]],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "posterior_diagnostics": {
            "posterior_image_diagnostic_draws": int(posterior_image_diagnostic_draws),
            "posterior_truth_recovery_draws": int(posterior_truth_recovery_draws),
            "posterior_truth_recovery_draws_effective": int(posterior_truth_recovery_draws_effective),
            "posterior_truth_recovery_mode": str(posterior_truth_recovery_mode),
            "workers": int(diagnostic_worker_count),
            "mode": str(posterior_diagnostic_mode),
            "quick_diagnostics": bool(quick_diagnostics),
        },
        "posterior_sample_count": int(np.asarray(samples).shape[0]),
        "parameter_names": list(parameter_names),
        "best_fit_values": best_fit_values,
        "previous_stage_best_values": previous_stage_best_values,
        "summary": summary,
        "summary_uncertainty": summary_uncertainty,
        "tables": {
            "parameters": _validation_dataframe_payload(parameter_df),
            "images": _validation_dataframe_payload(image_df),
            "sources": _validation_dataframe_payload(source_df),
            "magnification": _validation_dataframe_payload(magnification_df),
            "posterior_image_rms": _validation_dataframe_payload(posterior_image_rms_df),
            "posterior_image_residual_draws": _validation_dataframe_payload(posterior_image_residual_draws_df),
            "mass_profile": _validation_dataframe_payload(mass_profile_df),
            "surface_density": _validation_dataframe_payload(surface_density_df),
        },
        "caustics": {
            "truth_by_z": _caustic_contour_payload(truth_caustics_by_z),
            "recovered_by_z": _caustic_contour_payload(recovered_caustics_by_z),
        },
        "output_paths": output_paths,
    }


def _log_validation_approximation_items(args: Any | None, items: list[str]) -> None:
    if items:
        _log(args, "[validation] warning approximations active: " + "; ".join(items))


def write_recovery_outputs(
    run_dir: str | Path,
    truth_path: str | Path,
    mock_images_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    posterior_image_diagnostic_draws: int = 8,
    posterior_truth_recovery_draws: int = RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
    posterior_diagnostic_mode: str = POSTERIOR_DIAGNOSTIC_MODE_EXACT,
    critical_caustic_plot_grid_scale_arcsec: float = DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
    quick_diagnostics: bool = False,
    progress_args: Any | None = None,
    recovery_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_args: Any | None = None
    posterior_diagnostic_mode = str(posterior_diagnostic_mode)
    posterior_truth_recovery_draws = int(posterior_truth_recovery_draws)
    if quick_diagnostics:
        posterior_diagnostic_mode = POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
        _log_validation_approximation_items(
            progress_args,
            [
                "quick_diagnostics=active source-plane and median+/-std post-fit diagnostics; "
                "exact image-position validation skipped"
            ],
        )
    diagnostic_worker_count = max(1, int(jax_cpu_worker_count()))
    if posterior_diagnostic_mode == POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE:
        _log_validation_approximation_items(
            progress_args,
            [
                "posterior_diagnostic_mode=approximate median+/-std bars; "
                "exact per-draw image validation skipped; image-position posterior bars may be absent"
            ],
        )

    with _ValidationRecoveryProgress(progress_args) as recovery_progress:

        def run_recovery_phase(description: str, phase_name: str, fn):
            recovery_progress.begin_phase(description)
            result = _run_logged_phase(phase_args, phase_name, fn)
            recovery_progress.advance_phase()
            return result

        def load_inputs() -> tuple[dict[str, Any], pd.DataFrame]:
            truth_payload = _load_truth(truth_path)
            if mock_images_path is None:
                if "images" not in truth_payload:
                    raise ValueError(
                        "Truth file must contain an 'images' list when mock_images_path is not provided."
                    )
                image_table = pd.DataFrame(truth_payload["images"])
            else:
                image_table = pd.DataFrame(json.loads(Path(mock_images_path).read_text(encoding="utf-8")))
            return truth_payload, image_table

        truth, images = run_recovery_phase("load inputs", "validation.recovery.load_inputs", load_inputs)
        state, _saved_args, arrays, _init_diagnostics = run_recovery_phase(
            "load plot bundle",
            "validation.recovery.load_plot_bundle",
            lambda: _load_plot_bundle(run_dir),
        )
        samples = np.asarray(arrays["samples"], dtype=float)
        best_fit = np.asarray(arrays["best_fit"], dtype=float)
        if posterior_truth_recovery_draws <= 0:
            posterior_truth_recovery_mode = "best_fit"
            posterior_truth_recovery_draws_effective = 1
        else:
            posterior_truth_recovery_mode = "posterior"
            posterior_truth_recovery_draws_effective = min(int(samples.shape[0]), int(posterior_truth_recovery_draws))
        parameter_names = _artifact_parameter_names(state)
        truth_values = _parameter_truth_with_source_positions(truth)
        best_fit_values = _best_fit_values_for_specs(state.parameter_specs, best_fit)
        previous_stage_best_values = getattr(state, "previous_stage_best_values", None)
        parameter_df = run_recovery_phase(
            "parameter table",
            "validation.recovery.parameter_table",
            lambda: parameter_recovery_table(
                samples,
                parameter_names,
                truth_values,
                best_fit=best_fit,
            ),
        )
        recovered_mu, image_df, source_df = run_recovery_phase(
            "recovered model tables",
            "validation.recovery.recovered_model_tables",
            lambda: _recovered_model_tables(
                state,
                best_fit,
                images,
                quick_diagnostics=bool(quick_diagnostics),
                progress=recovery_progress,
                artifact_args=_saved_args,
            ),
        )
        (
            mag_uncertainty_df,
            image_uncertainty_df,
            source_uncertainty_df,
            posterior_image_rms_df,
            posterior_image_residual_draws_df,
        ) = run_recovery_phase(
            "posterior uncertainty",
            "validation.recovery.posterior_uncertainty_tables",
            lambda: _posterior_prediction_uncertainty_tables(
                state,
                samples,
                images,
                max_draws=int(posterior_image_diagnostic_draws),
                posterior_diagnostic_mode=posterior_diagnostic_mode,
                progress=recovery_progress,
                artifact_args=_saved_args,
            ),
        )

        def finalize_recovery_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            recovered_mu_local = recovered_mu
            image_df_local = image_df
            source_df_local = source_df
            if not mag_uncertainty_df.empty:
                recovered_mu_local = recovered_mu_local.merge(mag_uncertainty_df, on="image_label", how="left")
            if not image_uncertainty_df.empty:
                image_df_local = image_df_local.merge(image_uncertainty_df, on="image_label", how="left")
            if not source_uncertainty_df.empty:
                source_df_local = source_df_local.merge(source_uncertainty_df, on="family_id", how="left")
            magnification_df_local = magnification_recovery_table(images, recovered_mu_local)
            source_truth_df = pd.DataFrame(truth.get("sources", []))
            if not source_truth_df.empty:
                source_df_local = source_truth_df.merge(source_df_local, on="family_id", how="left")
                source_df_local["source_position_error_arcsec"] = np.hypot(
                    source_df_local["source_x_recovered"].to_numpy(dtype=float) - source_df_local["beta_x"].to_numpy(dtype=float),
                    source_df_local["source_y_recovered"].to_numpy(dtype=float) - source_df_local["beta_y"].to_numpy(dtype=float),
                )
                if {
                    "source_x_q16",
                    "source_x_q50",
                    "source_x_q84",
                    "source_y_q16",
                    "source_y_q50",
                    "source_y_q84",
                }.issubset(source_df_local.columns):
                    for suffix in ("q16", "q50", "q84"):
                        source_df_local[f"source_position_error_{suffix}"] = np.hypot(
                            source_df_local[f"source_x_{suffix}"].to_numpy(dtype=float) - source_df_local["beta_x"].to_numpy(dtype=float),
                            source_df_local[f"source_y_{suffix}"].to_numpy(dtype=float) - source_df_local["beta_y"].to_numpy(dtype=float),
                        )
            return recovered_mu_local, image_df_local, source_df_local, magnification_df_local

        recovered_mu, image_df, source_df, magnification_df = run_recovery_phase(
            "finalize tables",
            "validation.recovery.finalize_tables",
            finalize_recovery_tables,
        )
        mass_profile_df = pd.DataFrame()
        surface_density_df = pd.DataFrame()
        truth_caustics_by_z = run_recovery_phase(
            "truth caustics",
            "validation.recovery.truth_caustics",
            lambda: _caustic_contours_by_z_from_truth(truth),
        )
        truth_plot_caustics_by_z: dict[str, list[CausticContour]] = {}
        recovered_plot_caustics_by_z: dict[str, list[CausticContour]] = {}
        has_mass_profile_truth = "config" in truth and (
            "kwargs_lens" in truth or "kwargs_lens_by_source_redshift" in truth
        )
        if has_mass_profile_truth:
            profile_radii_arcsec = np.asarray([2.0, 5.0, 10.0, 20.0, 40.0], dtype=float)
            if posterior_truth_recovery_draws <= 0:
                profile_samples = best_fit.reshape(1, -1)
            else:
                profile_samples = _capped_evenly_spaced_posterior_draws(samples, max_draws=posterior_truth_recovery_draws)
            mass_profile_df, surface_density_df = run_recovery_phase(
                "mass/surface profile bands",
                "validation.recovery.mass_surface_density_profiles",
                lambda: _mass_and_surface_density_profiles_for_samples(
                    state,
                    profile_samples,
                    truth,
                    radii_arcsec=profile_radii_arcsec,
                    progress=recovery_progress,
                ),
            )
            truth_caustics_z9 = _select_critical_caustic_plot_contours(truth_caustics_by_z)
            if truth_caustics_z9:
                try:
                    plot_caustic_z_keys = sorted(truth_caustics_z9)
                    truth_plot_caustics_by_z = run_recovery_phase(
                        "truth plot caustics",
                        "validation.recovery.truth_plot_caustics",
                        lambda: _truth_caustic_contours_by_z_for_plot(
                            state,
                            truth,
                            plot_caustic_z_keys,
                            caustic_grid_scale_arcsec=float(critical_caustic_plot_grid_scale_arcsec),
                            progress=recovery_progress,
                        ),
                    )
                    recovered_plot_caustics_by_z = run_recovery_phase(
                        "recovered caustics",
                        "validation.recovery.recovered_caustics",
                        lambda: _recovered_caustic_contours_by_z(
                            state,
                            best_fit,
                            truth,
                            plot_caustic_z_keys,
                            caustic_grid_scale_arcsec=float(critical_caustic_plot_grid_scale_arcsec),
                            progress=recovery_progress,
                        ),
                    )
                except Exception as exc:  # pragma: no cover - defensive plotting fallback
                    print(f"[validation:critical-caustic] skipped recovered caustic computation: {exc}")
                    truth_plot_caustics_by_z = {}
                    recovered_plot_caustics_by_z = {}

        def build_summary() -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
            summary_payload = {
                "n_parameters": len(parameter_df),
                "median_abs_parameter_bias": float(np.nanmedian(np.abs(parameter_df["bias"]))),
                "parameter_coverage_68_fraction": float(np.mean(parameter_df["covered_68"])),
                "n_images": len(magnification_df),
                "median_image_residual_arcsec": _nanmedian_no_warning(image_df["image_residual_arcsec"]),
                "median_arc_aware_image_residual_arcsec": _nanmedian_no_warning(image_df["arc_aware_image_residual_arcsec"])
                if "arc_aware_image_residual_arcsec" in image_df
                else np.nan,
                "median_source_position_error_arcsec": _nanmedian_no_warning(source_df["source_position_error_arcsec"])
                if "source_position_error_arcsec" in source_df
                else np.nan,
                "median_abs_magnification_frac_error": _nanmedian_no_warning(
                    magnification_df["abs_magnification_fractional_error"]
                ),
                "parity_match_fraction": float(np.nanmean(magnification_df["parity_match"].astype(float))),
            }
            posterior_rms_summary = _posterior_image_rms_summary(posterior_image_rms_df)
            summary_payload.update(posterior_rms_summary)
            summary_uncertainty_payload = _summary_uncertainty(parameter_df, image_df, source_df, magnification_df)
            for prefix in ("posterior_point_image_rms_arcsec", "posterior_arc_aware_image_rms_arcsec"):
                q16 = posterior_rms_summary.get(f"{prefix}_q16", np.nan)
                q84 = posterior_rms_summary.get(f"{prefix}_q84", np.nan)
                summary_uncertainty_payload[f"{prefix}_median"] = (q16, q84)
            return summary_payload, summary_uncertainty_payload

        summary, summary_uncertainty = run_recovery_phase(
            "summary",
            "validation.recovery.summary",
            build_summary,
        )
        paths = {
            "corner_plot": output_dir / "corner.pdf",
            "potfile_corner_plot": output_dir / "potfile_corner.pdf",
            "parameter_recovery_log_plot": output_dir / "parameter_recovery_log.pdf",
            "parameter_recovery_linear_plot": output_dir / "parameter_recovery_linear.pdf",
            "mass_profile_plot": output_dir / "mass_profile_recovery.pdf",
            "surface_density_plot": output_dir / "surface_density_recovery.pdf",
            "critical_caustic_plot": output_dir / "critical_caustic_recovery.pdf",
            "magnification_plot": output_dir / "magnification_recovery.pdf",
            "absolute_magnification_plot": output_dir / "absolute_magnification_recovery.pdf",
            "image_recovery_plot": output_dir / "image_recovery.pdf",
            "image_residual_histogram_plot": output_dir / "image_residual_histogram.pdf",
            "source_recovery_plot": output_dir / "source_recovery.pdf",
            "subhalo_recovery_shmf_plot": output_dir / "subhalo_recovery_shmf.pdf",
            "subhalo_recovery_radial_plot": output_dir / "subhalo_recovery_radial.pdf",
            "summary_plot": output_dir / "validation_summary.pdf",
            "critical_arc_support_histogram_plot": output_dir / "critical_arc_support_histogram.pdf",
            "critical_arc_support_phase_space_plot": output_dir / "critical_arc_support_phase_space.pdf",
            "critical_arc_recovery_by_family_plot": output_dir / "critical_arc_recovery_by_family.pdf",
            "posterior_image_rms_table": output_dir / "posterior_image_rms.csv",
            "posterior_image_residual_draws_table": output_dir / "posterior_image_residual_draws.csv",
        }
        if not posterior_image_rms_df.empty:
            run_recovery_phase(
                "posterior image RMS table",
                "validation.recovery.write_posterior_image_rms_table",
                lambda: posterior_image_rms_df.to_csv(paths["posterior_image_rms_table"], index=False),
            )
        else:
            paths.pop("posterior_image_rms_table", None)
        if not posterior_image_residual_draws_df.empty:
            run_recovery_phase(
                "posterior image residual draws table",
                "validation.recovery.write_posterior_image_residual_draws_table",
                lambda: posterior_image_residual_draws_df.to_csv(paths["posterior_image_residual_draws_table"], index=False),
            )
        else:
            paths.pop("posterior_image_residual_draws_table", None)
        run_recovery_phase(
            "corner plot",
            "validation.recovery.plot_corner",
            lambda: _plot_corner_pdf(
                output_dir,
                samples,
                state.parameter_specs,
                "corner.pdf",
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            ),
        )
        scaling_specs, scaling_samples, scaling_best_fit = run_recovery_phase(
            "scaling subset",
            "validation.recovery.scaling_subset",
            lambda: _potfile_corner_parameter_subset(
                state.parameter_specs,
                samples,
                best_fit,
            ),
        )
        scaling_best_fit_values = _best_fit_values_for_specs(scaling_specs, scaling_best_fit)
        run_recovery_phase(
            "potfile corner plot",
            "validation.recovery.plot_potfile_corner",
            lambda: _plot_corner_pdf(
                output_dir,
                scaling_samples,
                scaling_specs,
                "potfile_corner.pdf",
                truth_values=truth_values,
                best_fit_values=scaling_best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            ),
        )
        if any(getattr(spec, "component_family", None) == "cosmology" for spec in state.parameter_specs):
            cosmology_specs, cosmology_samples, cosmology_best_fit = run_recovery_phase(
                "cosmology subset",
                "validation.recovery.cosmology_subset",
                lambda: _cosmology_parameter_subset(
                    state.parameter_specs,
                    samples,
                    best_fit,
                ),
            )
            if cosmology_specs:
                paths["cosmology_corner_plot"] = output_dir / "cosmology_corner.pdf"
                cosmology_best_fit_values = _best_fit_values_for_specs(cosmology_specs, cosmology_best_fit)
                run_recovery_phase(
                    "cosmology corner plot",
                    "validation.recovery.plot_cosmology_corner",
                    lambda: _plot_corner_pdf(
                        output_dir,
                        cosmology_samples,
                        cosmology_specs,
                        "cosmology_corner.pdf",
                        truth_values=truth_values,
                        best_fit_values=cosmology_best_fit_values,
                        previous_stage_best_values=previous_stage_best_values,
                    ),
                )
        run_recovery_phase(
            "parameter recovery log plot",
            "validation.recovery.plot_parameter_recovery_log",
            lambda: _plot_parameter_recovery(parameter_df, paths["parameter_recovery_log_plot"], scale="log_abs"),
        )
        run_recovery_phase(
            "parameter recovery linear plot",
            "validation.recovery.plot_parameter_recovery_linear",
            lambda: _plot_parameter_recovery(parameter_df, paths["parameter_recovery_linear_plot"], scale="linear"),
        )
        if not mass_profile_df.empty:
            run_recovery_phase(
                "mass profile plot",
                "validation.recovery.plot_mass_profile",
                lambda: _plot_mass_profile_recovery(mass_profile_df, paths["mass_profile_plot"]),
            )
        else:
            paths.pop("mass_profile_plot", None)
        if not surface_density_df.empty:
            run_recovery_phase(
                "surface density plot",
                "validation.recovery.plot_surface_density",
                lambda: _plot_surface_density_recovery(surface_density_df, paths["surface_density_plot"]),
            )
        else:
            paths.pop("surface_density_plot", None)
        truth_caustics_z9 = _select_critical_caustic_plot_contours(truth_plot_caustics_by_z)
        recovered_caustics_z9 = _select_critical_caustic_plot_contours(recovered_plot_caustics_by_z)
        if truth_caustics_z9 and recovered_caustics_z9:
            run_recovery_phase(
                "critical caustic plot",
                "validation.recovery.plot_critical_caustic",
                lambda: _plot_critical_caustic_recovery(
                    truth_caustics_z9,
                    recovered_caustics_z9,
                    images,
                    image_df,
                    source_df,
                    pd.DataFrame(truth.get("subhalos", [])),
                    paths["critical_caustic_plot"],
                ),
            )
        else:
            paths.pop("critical_caustic_plot", None)
        run_recovery_phase(
            "magnification plot",
            "validation.recovery.plot_magnification",
            lambda: _plot_magnification_recovery(magnification_df, paths["magnification_plot"]),
        )
        absolute_magnification_grid = run_recovery_phase(
            "absolute magnification grid",
            "validation.recovery.absolute_magnification_grid",
            lambda: _absolute_magnification_recovery_grid(
                state,
                best_fit,
                truth,
                grid_scale_arcsec=float(critical_caustic_plot_grid_scale_arcsec),
            ),
        )
        run_recovery_phase(
            "absolute magnification plot",
            "validation.recovery.plot_absolute_magnification",
            lambda: _plot_absolute_magnification_recovery(
                absolute_magnification_grid,
                paths["absolute_magnification_plot"],
            ),
        )
        run_recovery_phase(
            "image recovery plot",
            "validation.recovery.plot_image",
            lambda: _plot_image_recovery(image_df, paths["image_recovery_plot"]),
        )
        run_recovery_phase(
            "image residual histogram",
            "validation.recovery.plot_image_residual_histogram",
            lambda: _plot_image_residual_histogram(
                image_df,
                paths["image_residual_histogram_plot"],
                posterior_image_rms_df=posterior_image_rms_df,
                posterior_image_residual_draws_df=posterior_image_residual_draws_df,
            ),
        )
        critical_arc_image_count_df = _image_count_recovery_table(state, image_df)
        run_recovery_phase(
            "critical-arc support histogram",
            "validation.recovery.plot_critical_arc_support_histogram",
            lambda: _plot_critical_arc_support_histogram(
                image_df,
                paths["critical_arc_support_histogram_plot"],
                artifact_args=_saved_args,
            ),
        )
        run_recovery_phase(
            "critical-arc support phase space",
            "validation.recovery.plot_critical_arc_support_phase_space",
            lambda: _plot_critical_arc_support_phase_space(
                image_df,
                paths["critical_arc_support_phase_space_plot"],
                artifact_args=_saved_args,
            ),
        )
        run_recovery_phase(
            "critical-arc recovery by family",
            "validation.recovery.plot_critical_arc_recovery_by_family",
            lambda: _plot_critical_arc_recovery_by_family(
                critical_arc_image_count_df,
                paths["critical_arc_recovery_by_family_plot"],
            ),
        )
        run_recovery_phase(
            "source recovery plot",
            "validation.recovery.plot_source",
            lambda: _plot_source_recovery(source_df, paths["source_recovery_plot"]),
        )
        recovered_subhalo_df = run_recovery_phase(
            "recovered subhalo masses",
            "validation.recovery.recovered_subhalo_masses",
            lambda: _recovered_subhalo_mass_table(state, best_fit, truth),
        )
        run_recovery_phase(
            "subhalo SHMF recovery plot",
            "validation.recovery.plot_subhalo_recovery_shmf",
            lambda: _plot_subhalo_recovery_shmf(
                truth,
                recovered_subhalo_df,
                paths["subhalo_recovery_shmf_plot"],
            ),
        )
        run_recovery_phase(
            "subhalo radial recovery plot",
            "validation.recovery.plot_subhalo_recovery_radial",
            lambda: _plot_subhalo_recovery_radial(
                truth,
                recovered_subhalo_df,
                paths["subhalo_recovery_radial_plot"],
            ),
        )
        run_recovery_phase(
            "summary plot",
            "validation.recovery.plot_summary",
            lambda: _plot_validation_summary(summary, summary_uncertainty, paths["summary_plot"]),
        )
        if recovery_payload is not None:
            recovery_payload.clear()
            recovery_payload.update(
                _validation_jsonable(
                    _recovery_payload_from_tables(
                        run_dir=run_dir,
                        output_dir=output_dir,
                        posterior_image_diagnostic_draws=int(posterior_image_diagnostic_draws),
                        posterior_truth_recovery_draws=int(posterior_truth_recovery_draws),
                        posterior_truth_recovery_draws_effective=int(posterior_truth_recovery_draws_effective),
                        posterior_truth_recovery_mode=posterior_truth_recovery_mode,
                        diagnostic_worker_count=int(diagnostic_worker_count),
                        posterior_diagnostic_mode=posterior_diagnostic_mode,
                        quick_diagnostics=bool(quick_diagnostics),
                        samples=samples,
                        best_fit_values=best_fit_values,
                        previous_stage_best_values=previous_stage_best_values,
                        parameter_names=parameter_names,
                        parameter_df=parameter_df,
                        image_df=image_df,
                        source_df=source_df,
                        magnification_df=magnification_df,
                        posterior_image_rms_df=posterior_image_rms_df,
                        posterior_image_residual_draws_df=posterior_image_residual_draws_df,
                        mass_profile_df=mass_profile_df,
                        surface_density_df=surface_density_df,
                        summary=summary,
                        summary_uncertainty=summary_uncertainty,
                        truth_caustics_by_z=truth_plot_caustics_by_z,
                        recovered_caustics_by_z=recovered_plot_caustics_by_z,
                        output_paths=paths,
                    )
                )
            )
        return paths


PARAMETER_RECOVERY_LOG_ABS_FLOOR = 1.0e-4
CRITICAL_CAUSTIC_RECOVERY_SOURCE_REDSHIFT = 9.0
CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL = 1.0e-6
ABSOLUTE_MAGNIFICATION_RECOVERY_CAP = 25.0


@dataclass(frozen=True)
class _AbsoluteMagnificationRecoveryGrid:
    x_axis_arcsec: np.ndarray
    y_axis_arcsec: np.ndarray
    truth_abs_mu_raw: np.ndarray
    recovered_abs_mu_raw: np.ndarray
    truth_abs_mu: np.ndarray
    recovered_abs_mu: np.ndarray
    residual_abs_mu: np.ndarray
    z_source: float
    cap: float


def _log10_abs_parameter_values(values: np.ndarray, floor: float = PARAMETER_RECOVERY_LOG_ABS_FLOOR) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    transformed = np.full_like(array, np.nan, dtype=float)
    finite = np.isfinite(array)
    if np.any(finite):
        transformed[finite] = np.log10(np.maximum(np.abs(array[finite]), float(floor)))
    return transformed


def _plot_parameter_recovery(parameter_df: pd.DataFrame, path: Path, *, scale: str = "log_abs") -> None:
    if scale not in {"log_abs", "linear"}:
        raise ValueError("scale must be 'log_abs' or 'linear'.")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(parameter_df))))
    y = np.arange(len(parameter_df))
    median_raw = parameter_df["median"].to_numpy(dtype=float)
    q16_raw = parameter_df["q16"].to_numpy(dtype=float)
    q84_raw = parameter_df["q84"].to_numpy(dtype=float)
    truth_raw = parameter_df["truth"].to_numpy(dtype=float)
    if scale == "log_abs":
        median = _log10_abs_parameter_values(median_raw)
        q16 = _log10_abs_parameter_values(q16_raw)
        q84 = _log10_abs_parameter_values(q84_raw)
        truth = _log10_abs_parameter_values(truth_raw)
        xlabel = "log10(abs(parameter value))"
    else:
        median = median_raw
        q16 = q16_raw
        q84 = q84_raw
        truth = truth_raw
        xlabel = "parameter value"
    low = np.minimum(q16, q84)
    high = np.maximum(q16, q84)
    ax.errorbar(
        median,
        y,
        xerr=[np.maximum(0.0, median - low), np.maximum(0.0, high - median)],
        fmt="o",
        color="tab:blue",
        label="posterior 1 sigma",
    )
    ax.scatter(truth, y, marker="x", color="black", linewidths=1.6, label="truth", zorder=5)
    ax.set_yticks(y, parameter_df["parameter"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    finite_values = np.concatenate([median_raw, q16_raw, q84_raw, truth_raw])
    finite_values = finite_values[np.isfinite(finite_values)]
    if scale == "log_abs" and finite_values.size and np.any(np.abs(finite_values) < PARAMETER_RECOVERY_LOG_ABS_FLOOR):
        ax.text(
            0.98,
            0.02,
            f"abs(values) < {PARAMETER_RECOVERY_LOG_ABS_FLOOR:g} clipped",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
            transform=ax.transAxes,
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_corner_placeholder(samples: np.ndarray, parameter_names: list[str], path: Path, plot_name: str) -> None:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2 or sample_array.size == 0:
        n_samples = 0
        n_params = len(parameter_names)
        n_dynamic = 0
    else:
        finite_rows = sample_array[np.all(np.isfinite(sample_array), axis=1)]
        n_samples = int(finite_rows.shape[0])
        n_params = int(finite_rows.shape[1]) if finite_rows.ndim == 2 else len(parameter_names)
        if finite_rows.ndim == 2 and finite_rows.shape[0] > 0:
            spans = np.nanmax(finite_rows, axis=0) - np.nanmin(finite_rows, axis=0)
            n_dynamic = int(np.sum(np.isfinite(spans) & (spans > 0.0)))
        else:
            n_dynamic = 0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        f"{plot_name} was not generated",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        (
            "The saved posterior has fewer than two parameters with dynamic range.\n"
            f"finite samples: {n_samples}, parameters: {n_params}, dynamic parameters: {n_dynamic}.\n"
            "This usually means the sampler/guide posterior collapsed or all retained samples are identical."
        ),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_corner_pdf(
    output_dir: Path,
    samples: np.ndarray,
    parameter_specs: list[Any],
    filename: str = "corner.pdf",
    truth_values: dict[str, float] | None = None,
    best_fit_values: dict[str, float] | None = None,
    previous_stage_best_values: dict[str, float] | None = None,
) -> None:
    path = output_dir / filename
    if path.exists():
        path.unlink()
    try:
        if filename == "corner.pdf":
            _plot_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
        elif filename == "cosmology_corner.pdf":
            _plot_cosmology_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
        else:
            _plot_potfile_corner(
                output_dir,
                samples,
                parameter_specs,
                truth_values=truth_values,
                best_fit_values=best_fit_values,
                previous_stage_best_values=previous_stage_best_values,
            )
    except Exception as exc:  # pragma: no cover - defensive plotting fallback
        placeholder_samples, placeholder_specs = (
            _corner_without_source_positions(samples, parameter_specs, filename)
            if filename == "corner.pdf"
            else (samples, parameter_specs)
        )
        _write_corner_placeholder(
            placeholder_samples,
            [getattr(spec, "name", str(spec)) for spec in placeholder_specs],
            path,
            filename,
        )
        _log_message = f"[validation:corner] wrote placeholder {path}: {exc}"
        print(_log_message)
        return
    if not path.exists():
        placeholder_samples, placeholder_specs = (
            _corner_without_source_positions(samples, parameter_specs, filename)
            if filename == "corner.pdf"
            else (samples, parameter_specs)
        )
        _write_corner_placeholder(
            placeholder_samples,
            [getattr(spec, "name", str(spec)) for spec in placeholder_specs],
            path,
            filename,
        )


def _summary_uncertainty(
    parameter_df: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    magnification_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    def interval_from_columns(df: pd.DataFrame, q16_col: str, q84_col: str) -> tuple[float, float]:
        if q16_col not in df or q84_col not in df:
            return np.nan, np.nan
        low = _nanmedian_no_warning(df[q16_col])
        high = _nanmedian_no_warning(df[q84_col])
        return (min(low, high), max(low, high)) if np.isfinite(low + high) else (low, high)

    image_interval = interval_from_columns(image_df, "image_residual_q16", "image_residual_q84")
    arc_image_interval = interval_from_columns(
        image_df,
        "arc_aware_image_residual_q16",
        "arc_aware_image_residual_q84",
    )
    source_interval = interval_from_columns(source_df, "source_position_error_q16", "source_position_error_q84")
    mag_interval = interval_from_columns(
        magnification_df,
        "abs_magnification_fractional_error_q16",
        "abs_magnification_fractional_error_q84",
    )
    coverage_values = parameter_df["covered_68"].astype(float).to_numpy(dtype=float)
    coverage_se = (
        float(np.sqrt(np.nanmean(coverage_values) * (1.0 - np.nanmean(coverage_values)) / max(np.sum(np.isfinite(coverage_values)), 1)))
        if coverage_values.size
        else np.nan
    )
    coverage_mean = float(np.nanmean(coverage_values)) if coverage_values.size else np.nan
    parity_values = magnification_df["parity_match"].astype(float).to_numpy(dtype=float)
    parity_se = (
        float(np.sqrt(np.nanmean(parity_values) * (1.0 - np.nanmean(parity_values)) / max(np.sum(np.isfinite(parity_values)), 1)))
        if parity_values.size
        else np.nan
    )
    parity_mean = float(np.nanmean(parity_values)) if parity_values.size else np.nan
    return {
        "median_image_residual_arcsec": image_interval,
        "median_arc_aware_image_residual_arcsec": arc_image_interval,
        "median_source_position_error_arcsec": source_interval,
        "median_abs_magnification_frac_error": mag_interval,
        "parameter_coverage_68_fraction": (np.nan, np.nan) if not np.isfinite(coverage_se) else (coverage_mean - coverage_se, coverage_mean + coverage_se),
        "parity_match_fraction": (np.nan, np.nan) if not np.isfinite(parity_se) else (parity_mean - parity_se, parity_mean + parity_se),
    }


def _plot_mass_profile_recovery(profile_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax = axes[0]
    ratio_ax = axes[1]
    component_order = ["total", "halo", "bcg_plus_subhalos", "bcg", "subhalos"]
    colors = {
        "total": "tab:blue",
        "halo": "tab:orange",
        "bcg_plus_subhalos": "tab:green",
        "bcg": "tab:red",
        "subhalos": "tab:purple",
    }
    narrow_messages: list[str] = []
    for component in component_order:
        comp_df = profile_df[profile_df["component"] == component].sort_values("radius_arcsec")
        if comp_df.empty:
            continue
        label = str(comp_df["component_label"].iloc[0])
        color = colors.get(component, "0.4")
        radius = comp_df["radius_arcsec"].to_numpy(dtype=float)
        median = comp_df["median"].to_numpy(dtype=float)
        q16 = comp_df["q16"].to_numpy(dtype=float)
        q84 = comp_df["q84"].to_numpy(dtype=float)
        low = np.minimum(q16, q84)
        high = np.maximum(q16, q84)
        truth = comp_df["truth"].to_numpy(dtype=float)
        yerr = [np.maximum(0.0, median - low), np.maximum(0.0, high - median)]
        band_width = high - low
        finite_scale = np.nanmax(np.abs(median[np.isfinite(median)])) if np.isfinite(median).any() else np.nan
        narrow_band = bool(
            np.isfinite(finite_scale)
            and finite_scale > 0.0
            and np.isfinite(band_width).any()
            and np.nanmax(band_width[np.isfinite(band_width)]) < 0.003 * finite_scale
        )
        if narrow_band:
            narrow_messages.append(label)
        line_width = 2.0 if component == "total" else 1.4
        alpha = 0.22 if component == "total" else 0.14
        ax.fill_between(radius, low, high, color=color, alpha=alpha)
        ax.errorbar(
            radius,
            median,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
            linewidth=line_width,
            label=f"{label} posterior",
        )
        ax.plot(radius, median, color=color, linewidth=line_width)
        ax.plot(radius, truth, color=color, linestyle="--", linewidth=line_width, label=f"{label} truth")

        denom = np.maximum(np.abs(truth), 1.0e-12)
        ratio_median = (median - truth) / denom
        ratio_low = (low - truth) / denom
        ratio_high = (high - truth) / denom
        ratio_ax.fill_between(radius, ratio_low, ratio_high, color=color, alpha=alpha)
        ratio_ax.errorbar(
            radius,
            ratio_median,
            yerr=[
                np.maximum(0.0, ratio_median - np.minimum(ratio_low, ratio_high)),
                np.maximum(0.0, np.maximum(ratio_low, ratio_high) - ratio_median),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
        )
        ratio_ax.plot(radius, ratio_median, color=color, linewidth=line_width)

    ax.set_ylabel("deflection magnitude (arcsec)")
    ax.legend(loc="best", fontsize=8)
    if narrow_messages:
        ax.text(
            0.98,
            0.04,
            "narrow 1 sigma bands: " + ", ".join(narrow_messages),
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
        )
    ratio_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ratio_ax.set_xlabel("radius (arcsec)")
    ratio_ax.set_ylabel("(post. - truth) / truth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_surface_density_recovery(profile_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.08},
    )
    ax = axes[0]
    ratio_ax = axes[1]
    component_order = ["total", "halo", "bcg_plus_subhalos", "bcg", "subhalos"]
    colors = {
        "total": "tab:blue",
        "halo": "tab:orange",
        "bcg_plus_subhalos": "tab:green",
        "bcg": "tab:red",
        "subhalos": "tab:purple",
    }
    positive_values: list[float] = []
    for component in component_order:
        comp_df = profile_df[profile_df["component"] == component].sort_values("radius_arcsec")
        if comp_df.empty:
            continue
        label = str(comp_df["component_label"].iloc[0])
        color = colors.get(component, "0.4")
        radius = comp_df["radius_arcsec"].to_numpy(dtype=float)
        median = comp_df["median"].to_numpy(dtype=float)
        q16 = comp_df["q16"].to_numpy(dtype=float)
        q84 = comp_df["q84"].to_numpy(dtype=float)
        low = np.minimum(q16, q84)
        high = np.maximum(q16, q84)
        truth = comp_df["truth"].to_numpy(dtype=float)
        finite_positive = np.concatenate([median, low, high, truth])
        finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
        positive_values.extend(float(value) for value in finite_positive)
        yerr = [np.maximum(0.0, median - low), np.maximum(0.0, high - median)]
        line_width = 2.0 if component == "total" else 1.4
        alpha = 0.22 if component == "total" else 0.14
        ax.fill_between(radius, low, high, color=color, alpha=alpha)
        ax.errorbar(
            radius,
            median,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
            linewidth=line_width,
            label=f"{label} posterior",
        )
        ax.plot(radius, median, color=color, linewidth=line_width)
        ax.plot(radius, truth, color=color, linestyle="--", linewidth=line_width, label=f"{label} truth")

        denom = np.maximum(np.abs(truth), 1.0e-12)
        ratio_median = (median - truth) / denom
        ratio_low = (low - truth) / denom
        ratio_high = (high - truth) / denom
        ratio_ax.fill_between(radius, ratio_low, ratio_high, color=color, alpha=alpha)
        ratio_ax.errorbar(
            radius,
            ratio_median,
            yerr=[
                np.maximum(0.0, ratio_median - np.minimum(ratio_low, ratio_high)),
                np.maximum(0.0, np.maximum(ratio_low, ratio_high) - ratio_median),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4 if component == "total" else 3,
        )
        ratio_ax.plot(radius, ratio_median, color=color, linewidth=line_width)

    ax.set_ylabel(r"$\Sigma$ (M$_\odot$ arcsec$^{-2}$)")
    if positive_values:
        all_values = profile_df[["truth", "q16", "median", "q84"]].to_numpy(dtype=float).reshape(-1)
        finite_values = all_values[np.isfinite(all_values)]
        if finite_values.size and np.all(finite_values > 0.0):
            ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ratio_ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ratio_ax.set_xlabel("radius (arcsec)")
    ratio_ax.set_ylabel("(post. - truth) / truth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_magnification_recovery(magnification_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    truth_column = (
        "magnification_true_at_obs"
        if "magnification_true_at_obs" in magnification_df.columns
        else "magnification_true"
    )
    truth = magnification_df[truth_column].to_numpy(dtype=float)
    recovered = magnification_df["magnification_recovered"].to_numpy(dtype=float)
    plotted = recovered
    if {"magnification_q16", "magnification_q50", "magnification_q84"}.issubset(magnification_df.columns):
        q16 = magnification_df["magnification_q16"].to_numpy(dtype=float)
        q50 = magnification_df["magnification_q50"].to_numpy(dtype=float)
        q84 = magnification_df["magnification_q84"].to_numpy(dtype=float)
        plotted = q50
        ax.errorbar(
            truth,
            q50,
            yerr=[np.maximum(0.0, q50 - q16), np.maximum(0.0, q84 - q50)],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            alpha=0.8,
            label="posterior 1 sigma",
        )
    else:
        ax.scatter(truth, recovered, color="tab:blue", label="best fit")
    finite = np.concatenate([truth[np.isfinite(truth)], plotted[np.isfinite(plotted)]])
    if finite.size:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlabel("true signed magnification")
    ax.set_ylabel("recovered signed magnification")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _capped_absolute_magnification(values: np.ndarray, cap: float = ABSOLUTE_MAGNIFICATION_RECOVERY_CAP) -> np.ndarray:
    if float(cap) <= 0.0:
        raise ValueError("cap must be positive.")
    array = np.asarray(values, dtype=float)
    return np.minimum(np.abs(array), float(cap))


def _finite_source_redshifts_from_mapping(mapping: Any) -> list[float]:
    if not isinstance(mapping, dict):
        return []
    redshifts: list[float] = []
    for key, value in mapping.items():
        if value is None:
            continue
        try:
            z_source = float(key)
        except (TypeError, ValueError):
            continue
        if np.isfinite(z_source):
            redshifts.append(float(z_source))
    return redshifts


def _absolute_magnification_grid_source_redshift(truth: dict[str, Any]) -> float:
    redshifts = _finite_source_redshifts_from_mapping(truth.get("kwargs_lens_by_source_redshift", {}))
    if not redshifts:
        redshifts = _finite_source_redshifts_from_mapping(truth.get("caustics_by_source_redshift", {}))
    if redshifts:
        z_array = np.asarray(redshifts, dtype=float)
        z9_distance = np.abs(z_array - CRITICAL_CAUSTIC_RECOVERY_SOURCE_REDSHIFT)
        z9_matches = z9_distance <= CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL
        if np.any(z9_matches):
            return float(z_array[np.where(z9_matches)[0][np.argmin(z9_distance[z9_matches])]])
        return float(np.nanmax(z_array))

    raw_config = truth.get("config", {})
    truth_config = raw_config if isinstance(raw_config, dict) else {}
    try:
        z_source = float(truth_config["source_redshift"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot determine source redshift for absolute magnification recovery plot.") from exc
    if not np.isfinite(z_source):
        raise ValueError("Source redshift for absolute magnification recovery plot must be finite.")
    return float(z_source)


def _truth_kwargs_for_source_redshift(truth: dict[str, Any], z_source: float) -> list[dict[str, Any]]:
    truth_kwargs_by_z = truth.get("kwargs_lens_by_source_redshift", {})
    if isinstance(truth_kwargs_by_z, dict):
        for key in (f"{float(z_source):.8f}", str(float(z_source)), str(z_source)):
            if key in truth_kwargs_by_z:
                return list(truth_kwargs_by_z[key])
        for key, kwargs_lens in truth_kwargs_by_z.items():
            try:
                key_z = float(key)
            except (TypeError, ValueError):
                continue
            if abs(key_z - float(z_source)) <= CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL:
                return list(kwargs_lens)
    return list(truth.get("kwargs_lens", []))


def _absolute_magnification_grid_axes(
    truth: dict[str, Any],
    *,
    grid_scale_arcsec: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = _plot_caustic_config_from_truth(
        truth,
        caustic_grid_scale_arcsec=grid_scale_arcsec,
    )
    compute_window = float(config.caustic_compute_window_arcsec)
    grid_scale = float(config.caustic_grid_scale_arcsec)
    if compute_window <= 0.0 or grid_scale <= 0.0:
        raise ValueError("Magnification map window and grid scale must be positive.")
    num_pix = max(16, int(math.ceil(compute_window / grid_scale)) + 1)
    if num_pix % 2 == 0:
        num_pix += 1
    x_axis = np.linspace(-0.5 * compute_window, 0.5 * compute_window, num_pix)
    y_axis = np.linspace(-0.5 * compute_window, 0.5 * compute_window, num_pix)
    return x_axis.astype(float), y_axis.astype(float)


def _absolute_magnification_recovery_grid(
    state: Any,
    best_fit_physical: np.ndarray,
    truth: dict[str, Any],
    *,
    grid_scale_arcsec: float | None = None,
    cap: float = ABSOLUTE_MAGNIFICATION_RECOVERY_CAP,
) -> _AbsoluteMagnificationRecoveryGrid:
    import jax.numpy as jnp

    from ..cluster_solver import (
        DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        DEFAULT_ACTIVE_SCALING_GALAXIES,
        DEFAULT_ACTIVE_SCALING_MIN,
        DEFAULT_MATCH_TOLERANCE,
        DEFAULT_REFRESH_EVERY,
        DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
        ClusterJAXEvaluator,
        _convert_theta_to_latent,
    )

    z_source = _absolute_magnification_grid_source_redshift(truth)
    x_axis, y_axis = _absolute_magnification_grid_axes(
        truth,
        grid_scale_arcsec=grid_scale_arcsec,
    )
    xx, yy = np.meshgrid(x_axis, y_axis)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)

    config = _plot_caustic_config_from_truth(
        truth,
        caustic_grid_scale_arcsec=grid_scale_arcsec,
    )
    raw_config = truth.get("config", {})
    truth_config = raw_config if isinstance(raw_config, dict) else {}
    z_lens = float(truth_config.get("z_lens", config.z_lens))
    lens_model_list = list(truth.get("lens_model_list", getattr(state, "lens_model_list", [])))
    if not lens_model_list:
        raise ValueError("Cannot plot absolute magnification recovery without lens model components.")

    truth_model = LensModel(
        lens_model_list=lens_model_list,
        z_lens=z_lens,
        z_source=z_source,
        cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3),
    )
    truth_kwargs = _truth_kwargs_for_source_redshift(truth, z_source)
    truth_mu = np.asarray(
        truth_model.magnification(flat_x, flat_y, truth_kwargs),
        dtype=float,
    ).reshape(xx.shape)

    evaluator = ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=DEFAULT_MATCH_TOLERANCE,
        sampling_engine="full_flat",
        active_scaling_galaxies=DEFAULT_ACTIVE_SCALING_GALAXIES,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=DEFAULT_ACTIVE_SCALING_CUMULATIVE_FRACTION,
        active_scaling_min=DEFAULT_ACTIVE_SCALING_MIN,
        refresh_every=DEFAULT_REFRESH_EVERY,
        refresh_param_drift_frac=DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    if hasattr(evaluator, "reported_physical_to_latent_parameter_vector"):
        best_fit_latent = evaluator.reported_physical_to_latent_parameter_vector(np.asarray(best_fit_physical, dtype=float))
    else:
        best_fit_latent = _convert_theta_to_latent(np.asarray(best_fit_physical, dtype=float), state.parameter_specs)
    recovered_model, _solver = evaluator._get_exact_model_solver(z_source)
    packed_state = evaluator._build_packed_lens_state(jnp.asarray(best_fit_latent, dtype=jnp.float64), z_source)
    recovered_kwargs = evaluator._packed_to_kwargs_lens(packed_state)
    recovered_mu = np.asarray(
        recovered_model.magnification(flat_x, flat_y, recovered_kwargs),
        dtype=float,
    ).reshape(xx.shape)

    truth_abs_mu_raw = np.abs(np.asarray(truth_mu, dtype=float))
    recovered_abs_mu_raw = np.abs(np.asarray(recovered_mu, dtype=float))
    truth_abs_mu = np.minimum(truth_abs_mu_raw, float(cap))
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, float(cap))
    return _AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=x_axis,
        y_axis_arcsec=y_axis,
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=recovered_abs_mu_raw - truth_abs_mu_raw,
        z_source=float(z_source),
        cap=float(cap),
    )


def _plot_absolute_magnification_recovery(
    grid: _AbsoluteMagnificationRecoveryGrid,
    path: Path,
) -> None:
    extent = [
        float(grid.x_axis_arcsec[0]),
        float(grid.x_axis_arcsec[-1]),
        float(grid.y_axis_arcsec[0]),
        float(grid.y_axis_arcsec[-1]),
    ]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.2, 8.6))
    recovered_image = axes[0].imshow(
        np.ma.masked_invalid(grid.recovered_abs_mu),
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=float(grid.cap),
        aspect="equal",
    )
    recovered_colorbar = fig.colorbar(recovered_image, ax=axes[0], fraction=0.046, pad=0.04)
    recovered_colorbar.set_label(r"$|\mu_{\rm rec}|$")

    residual_values = np.asarray(grid.residual_abs_mu, dtype=float)
    residual_norm = TwoSlopeNorm(vmin=-25.0, vcenter=0.0, vmax=25.0)
    residual_image = axes[1].imshow(
        np.ma.masked_invalid(residual_values),
        origin="lower",
        extent=extent,
        cmap="RdBu",
        norm=residual_norm,
        aspect="equal",
    )
    residual_colorbar = fig.colorbar(residual_image, ax=axes[1], fraction=0.046, pad=0.04)
    residual_colorbar.set_label(r"$|\mu_{\rm rec}| - |\mu_{\rm truth}|$")

    for ax in axes:
        ax.invert_xaxis()
        ax.set_ylabel("y (arcsec)")
        _limit_recovery_axis_ticks(ax)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xlabel("x (arcsec)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_value_with_fallback(df: pd.DataFrame, column: str, *fallback_columns: str | None) -> np.ndarray:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    else:
        values = np.full(len(df), np.nan, dtype=float)
    for fallback_column in fallback_columns:
        if fallback_column is None or fallback_column not in df.columns:
            continue
        fallback = pd.to_numeric(df[fallback_column], errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, fallback)
    return values


def _plot_image_recovery(image_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if "image_recovery_status" in image_df.columns:
        status = image_df["image_recovery_status"].fillna("unknown").astype(str).to_numpy()
        x_obs = image_df["x_obs_arcsec"].to_numpy(dtype=float)
        y_obs = image_df["y_obs_arcsec"].to_numpy(dtype=float)
        recovered = status == "recovered"
        not_recovered = status == "not_recovered"
        unknown = ~(recovered | not_recovered)
        if recovered.any():
            ax.scatter(x_obs[recovered], y_obs[recovered], color="tab:green", marker="x", s=24, label="observed recovered")
        if not_recovered.any():
            ax.scatter(x_obs[not_recovered], y_obs[not_recovered], color="tab:red", marker="x", s=24, label="observed not recovered")
        if unknown.any():
            ax.scatter(x_obs[unknown], y_obs[unknown], color="black", s=22, label="observed")
    else:
        ax.scatter(image_df["x_obs_arcsec"], image_df["y_obs_arcsec"], color="black", s=22, label="observed")
    if {"x_model_q16", "x_model_q50", "x_model_q84", "y_model_q16", "y_model_q50", "y_model_q84"}.issubset(image_df.columns):
        x_model = _plot_value_with_fallback(image_df, "x_model_q50", "x_model_arcsec")
        y_model = _plot_value_with_fallback(image_df, "y_model_q50", "y_model_arcsec")
        finite_model = np.isfinite(x_model) & np.isfinite(y_model)
        x16 = _plot_value_with_fallback(image_df, "x_model_q16")
        x84 = _plot_value_with_fallback(image_df, "x_model_q84")
        y16 = _plot_value_with_fallback(image_df, "y_model_q16")
        y84 = _plot_value_with_fallback(image_df, "y_model_q84")
        ax.errorbar(
            x_model[finite_model],
            y_model[finite_model],
            xerr=[
                np.where(np.isfinite(x16), np.maximum(0.0, x_model - x16), 0.0)[finite_model],
                np.where(np.isfinite(x84), np.maximum(0.0, x84 - x_model), 0.0)[finite_model],
            ],
            yerr=[
                np.where(np.isfinite(y16), np.maximum(0.0, y_model - y16), 0.0)[finite_model],
                np.where(np.isfinite(y84), np.maximum(0.0, y84 - y_model), 0.0)[finite_model],
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=4,
            label="model 1 sigma",
        )
    else:
        x_model = image_df["x_model_arcsec"].to_numpy(dtype=float)
        y_model = image_df["y_model_arcsec"].to_numpy(dtype=float)
        finite_model = np.isfinite(x_model) & np.isfinite(y_model)
        ax.scatter(x_model[finite_model], y_model[finite_model], color="tab:blue", s=18, label="model")
    for row, x_fit, y_fit in zip(image_df.itertuples(index=False), x_model, y_model):
        if np.isfinite(x_fit) and np.isfinite(y_fit):
            ax.plot([row.x_obs_arcsec, x_fit], [row.y_obs_arcsec, y_fit], color="0.6", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Image positions")
    ax.legend(loc="best", fontsize=8)

    residual = _plot_value_with_fallback(image_df, "image_residual_q50", "image_residual_arcsec")
    x_index = np.arange(len(image_df))
    if {"image_residual_q16", "image_residual_q84"}.issubset(image_df.columns):
        finite_residual = np.isfinite(residual)
        r16 = _plot_value_with_fallback(image_df, "image_residual_q16")
        r84 = _plot_value_with_fallback(image_df, "image_residual_q84")
        axes[1].errorbar(
            x_index[finite_residual],
            residual[finite_residual],
            yerr=[
                np.where(np.isfinite(r16), np.maximum(0.0, residual - r16), 0.0)[finite_residual],
                np.where(np.isfinite(r84), np.maximum(0.0, r84 - residual), 0.0)[finite_residual],
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
        )
    else:
        finite_residual = np.isfinite(residual)
        axes[1].scatter(x_index[finite_residual], residual[finite_residual], color="tab:blue")
    arc_residual = _plot_value_with_fallback(
        image_df,
        "arc_aware_image_residual_q50",
        "arc_aware_image_residual_arcsec",
    )
    finite_arc_residual = np.isfinite(arc_residual)
    if np.any(finite_arc_residual):
        axes[1].scatter(
            x_index[finite_arc_residual],
            arc_residual[finite_arc_residual],
            color="tab:olive",
            marker="x",
            s=28,
            label="arc-aware",
        )
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual (arcsec)")
    axes[1].set_title("Image residuals with 1 sigma intervals")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _posterior_residual_histogram_summaries(
    posterior_image_residual_draws_df: pd.DataFrame,
    *,
    bin_count: int = 30,
) -> dict[str, dict[str, np.ndarray]]:
    if posterior_image_residual_draws_df.empty or "draw_index" not in posterior_image_residual_draws_df:
        return {}
    draw_index = pd.to_numeric(posterior_image_residual_draws_df["draw_index"], errors="coerce").to_numpy(dtype=float)
    finite_draws = np.asarray(sorted({int(value) for value in draw_index if np.isfinite(value)}), dtype=int)
    if finite_draws.size == 0:
        return {}

    series_specs = {
        "point": ("point_image_residual_arcsec", "point_recovered"),
        "arc_aware": ("arc_aware_image_residual_arcsec", "arc_aware_recovered"),
    }
    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    finite_values: list[np.ndarray] = []
    for name, (value_column, mask_column) in series_specs.items():
        if value_column not in posterior_image_residual_draws_df:
            arrays[name] = np.asarray([], dtype=float)
            masks[name] = np.zeros(len(posterior_image_residual_draws_df), dtype=bool)
            continue
        values = pd.to_numeric(posterior_image_residual_draws_df[value_column], errors="coerce").to_numpy(dtype=float)
        if mask_column in posterior_image_residual_draws_df:
            recovered_mask = posterior_image_residual_draws_df[mask_column].astype(bool).to_numpy(dtype=bool)
        else:
            recovered_mask = np.isfinite(values)
        finite_mask = recovered_mask & np.isfinite(values)
        arrays[name] = values
        masks[name] = finite_mask
        if np.any(finite_mask):
            finite_values.append(values[finite_mask])
    if not finite_values:
        return {}

    all_values = np.concatenate(finite_values)
    max_residual = float(np.nanmax(all_values))
    bins = np.linspace(0.0, max(max_residual * 1.04, 1.0e-6), int(bin_count) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    summaries: dict[str, dict[str, np.ndarray]] = {}
    for name in series_specs:
        values = arrays[name]
        finite_mask = masks[name]
        if not np.any(finite_mask):
            continue
        counts_by_draw = []
        for draw in finite_draws:
            draw_mask = (draw_index == float(draw)) & finite_mask
            counts, _edges = np.histogram(values[draw_mask], bins=bins)
            counts_by_draw.append(counts.astype(float))
        counts_array = np.asarray(counts_by_draw, dtype=float)
        if counts_array.size == 0:
            continue
        q16, median, q84 = np.quantile(counts_array, [0.16, 0.5, 0.84], axis=0)
        summaries[name] = {
            "bins": bins,
            "centers": centers,
            "widths": widths,
            "q16": q16,
            "median": median,
            "q84": q84,
        }
    return summaries


def _plot_image_residual_histogram(
    image_df: pd.DataFrame,
    path: Path,
    *,
    posterior_image_rms_df: pd.DataFrame | None = None,
    posterior_image_residual_draws_df: pd.DataFrame | None = None,
) -> None:
    point_residual_all = _plot_value_with_fallback(
        image_df,
        "image_residual_q50",
        "point_image_residual_arcsec",
        "image_residual_arcsec",
    )
    if any(column in image_df.columns for column in ("image_recovery_status", "arc_recovery_status", "exact_image_prediction_failed")):
        status = _image_catalog_effective_recovery_statuses(image_df)
        point_mask = np.asarray([_image_catalog_point_recovered(row) for _, row in image_df.iterrows()], dtype=bool)
        arc_mask = np.asarray([_image_catalog_arc_recovered(row) for _, row in image_df.iterrows()], dtype=bool)
    else:
        status = np.where(np.isfinite(point_residual_all), "POINT_RECOVERED", "MISSED")
        point_mask = status == "POINT_RECOVERED"
        arc_mask = np.zeros(len(image_df), dtype=bool)
    arc_candidate_residual_all = _plot_value_with_fallback(
        image_df,
        "arc_aware_image_residual_q50",
        "arc_candidate_image_residual_arcsec",
        "arc_aware_image_residual_arcsec",
    )
    if "arc_curve_distance_arcsec" in image_df.columns:
        curve_distance = _plot_value_with_fallback(image_df, "arc_curve_distance_arcsec")
        arc_candidate_residual_all = np.where(np.isfinite(arc_candidate_residual_all), arc_candidate_residual_all, curve_distance)
    arc_aware_mask = arc_mask | (point_mask & ~arc_mask)
    arc_residual_all = np.where(arc_mask, arc_candidate_residual_all, np.where(point_mask, point_residual_all, np.nan))
    residual = point_residual_all[point_mask & np.isfinite(point_residual_all)]
    arc_residual = arc_residual_all[arc_aware_mask & np.isfinite(arc_residual_all)]
    posterior_histograms = _posterior_residual_histogram_summaries(
        posterior_image_residual_draws_df if posterior_image_residual_draws_df is not None else pd.DataFrame(),
        bin_count=30,
    )
    has_posterior_histogram = bool(posterior_histograms)
    if residual.size == 0 and arc_residual.size == 0 and not has_posterior_histogram:
        _write_placeholder_plot(
            path,
            "Image residual histogram",
            "No finite image residuals are available.",
        )
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bin_count = 30
    total_count = int(len(image_df))
    total_rms = float(np.sqrt(np.mean(np.square(residual)))) if residual.size else np.nan
    arc_total_rms = float(np.sqrt(np.mean(np.square(arc_residual)))) if arc_residual.size else np.nan
    posterior_summary = _posterior_image_rms_summary(
        posterior_image_rms_df if posterior_image_rms_df is not None else pd.DataFrame()
    )
    point_rms_plot = posterior_summary.get("posterior_point_image_rms_arcsec_median", total_rms)
    point_rms_std = posterior_summary.get("posterior_point_image_rms_arcsec_std", np.nan)
    point_rms_q16 = posterior_summary.get("posterior_point_image_rms_arcsec_q16", np.nan)
    point_rms_q84 = posterior_summary.get("posterior_point_image_rms_arcsec_q84", np.nan)
    arc_rms_plot = posterior_summary.get("posterior_arc_aware_image_rms_arcsec_median", arc_total_rms)
    arc_rms_std = posterior_summary.get("posterior_arc_aware_image_rms_arcsec_std", np.nan)
    arc_rms_q16 = posterior_summary.get("posterior_arc_aware_image_rms_arcsec_q16", np.nan)
    arc_rms_q84 = posterior_summary.get("posterior_arc_aware_image_rms_arcsec_q84", np.nan)
    if not np.isfinite(point_rms_plot):
        point_rms_plot = total_rms
    if not np.isfinite(arc_rms_plot):
        arc_rms_plot = arc_total_rms
    if has_posterior_histogram:
        point_hist = posterior_histograms.get("point")
        if point_hist is not None:
            ax.bar(
                point_hist["centers"],
                point_hist["median"],
                width=point_hist["widths"],
                align="center",
                color="tab:blue",
                alpha=0.45,
                edgecolor="#1d4ed8",
                linewidth=0.85,
                label="point recovery posterior median",
            )
            ax.errorbar(
                point_hist["centers"],
                point_hist["median"],
                yerr=[
                    np.maximum(0.0, point_hist["median"] - point_hist["q16"]),
                    np.maximum(0.0, point_hist["q84"] - point_hist["median"]),
                ],
                fmt="none",
                ecolor="#1d4ed8",
                elinewidth=0.8,
                capsize=2.0,
                alpha=0.9,
            )
            ax.axvline(
                point_rms_plot,
                color="tab:red",
                linestyle="-.",
                linewidth=1.2,
                label="point RMS",
            )
            if np.isfinite(point_rms_q16 + point_rms_q84):
                ax.axvspan(point_rms_q16, point_rms_q84, color="tab:red", alpha=0.12, linewidth=0.0)
        arc_hist = posterior_histograms.get("arc_aware")
        if arc_hist is not None:
            ax.step(
                arc_hist["centers"],
                arc_hist["median"],
                where="mid",
                color="#ffd54f",
                linewidth=2.0,
                label="arc-aware posterior median",
            )
            ax.errorbar(
                arc_hist["centers"],
                arc_hist["median"],
                yerr=[
                    np.maximum(0.0, arc_hist["median"] - arc_hist["q16"]),
                    np.maximum(0.0, arc_hist["q84"] - arc_hist["median"]),
                ],
                fmt="none",
                ecolor="#b7791f",
                elinewidth=0.8,
                capsize=2.0,
                alpha=0.9,
            )
            ax.axvline(
                arc_rms_plot,
                color="tab:green",
                linestyle="-.",
                linewidth=1.2,
                label="arc-aware RMS",
            )
            if np.isfinite(arc_rms_q16 + arc_rms_q84):
                ax.axvspan(arc_rms_q16, arc_rms_q84, color="tab:green", alpha=0.12, linewidth=0.0)
    elif residual.size:
        ax.hist(
            residual,
            bins=bin_count,
            color="tab:blue",
            alpha=0.5,
            edgecolor="#1d4ed8",
            linewidth=0.85,
            label=f"point recovery {residual.size}/{total_count}",
        )
        ax.axvline(
            point_rms_plot,
            color="tab:red",
            linestyle="-.",
            linewidth=1.2,
            label="point RMS",
        )
        if np.isfinite(point_rms_q16 + point_rms_q84):
            ax.axvspan(point_rms_q16, point_rms_q84, color="tab:red", alpha=0.12, linewidth=0.0)
    if not has_posterior_histogram and arc_residual.size:
        ax.hist(
            arc_residual,
            bins=bin_count,
            histtype="step",
            color="#ffd54f",
            linewidth=2.0,
            label=f"arc-aware {arc_residual.size}/{total_count}",
        )
        ax.axvline(
            arc_rms_plot,
            color="tab:green",
            linestyle="-.",
            linewidth=1.2,
            label="arc-aware RMS",
        )
        if np.isfinite(arc_rms_q16 + arc_rms_q84):
            ax.axvspan(arc_rms_q16, arc_rms_q84, color="tab:green", alpha=0.12, linewidth=0.0)
    arc_supported_count = int(np.sum(arc_mask))
    missed_count = int(np.sum(status == "MISSED"))
    point_count_label = posterior_summary.get("posterior_point_recovered_image_count_median", float(residual.size))
    arc_count_label = posterior_summary.get("posterior_arc_aware_recovered_image_count_median", float(arc_residual.size))
    arc_supported_label = posterior_summary.get("posterior_arc_supported_image_count_median", float(arc_supported_count))
    point_rms_label = (
        f"{point_rms_plot:.3g} +/- {point_rms_std:.3g}"
        if np.isfinite(point_rms_plot) and np.isfinite(point_rms_std)
        else f"{point_rms_plot:.3g}"
    )
    arc_rms_label = (
        f"{arc_rms_plot:.3g} +/- {arc_rms_std:.3g}"
        if np.isfinite(arc_rms_plot) and np.isfinite(arc_rms_std)
        else f"{arc_rms_plot:.3g}"
    )
    rms_annotation = "\n".join(
        [
            f"Point RMS = {point_rms_label} arcsec ({point_count_label:.3g}/{total_count})",
            f"Arc-aware RMS = {arc_rms_label} arcsec ({arc_count_label:.3g}/{total_count})",
            f"arc-supported = {arc_supported_label:.3g}/{total_count}",
            f"missed = {missed_count}/{total_count}",
        ]
    )
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
            "alpha": 0.9,
        },
    )
    ax.set_xlabel("image residual (arcsec)")
    ax.set_ylabel("posterior median N images" if has_posterior_histogram else "N images")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_critical_arc_support_histogram(
    image_df: pd.DataFrame,
    path: Path,
    *,
    artifact_args: dict[str, Any] | None = None,
) -> None:
    _shared_plot_critical_arc_support_histogram(
        image_df,
        path,
        arc_recovery_p_arc_threshold=float(
            _artifact_arg(artifact_args, "arc_recovery_p_arc_threshold", DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD)
        ),
        critical_arc_base_prob=float(_artifact_arg(artifact_args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)),
        critical_arc_max_prob=float(_artifact_arg(artifact_args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)),
        singular_threshold=float(
            _artifact_arg(artifact_args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        singular_softness=float(
            _artifact_arg(artifact_args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        sample_likelihood_mode=str(_artifact_arg(artifact_args, "sample_likelihood_mode", "critical-arc-mixture-image-plane")),
    )


def _plot_critical_arc_support_phase_space(
    image_df: pd.DataFrame,
    path: Path,
    *,
    artifact_args: dict[str, Any] | None = None,
) -> None:
    _shared_plot_critical_arc_support_phase_space(
        image_df,
        path,
        arc_recovery_p_arc_threshold=float(
            _artifact_arg(artifact_args, "arc_recovery_p_arc_threshold", DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD)
        ),
        critical_arc_base_prob=float(_artifact_arg(artifact_args, "critical_arc_base_prob", DEFAULT_CRITICAL_ARC_BASE_PROB)),
        critical_arc_max_prob=float(_artifact_arg(artifact_args, "critical_arc_max_prob", DEFAULT_CRITICAL_ARC_MAX_PROB)),
        singular_threshold=float(
            _artifact_arg(artifact_args, "critical_arc_singular_threshold", DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD)
        ),
        singular_softness=float(
            _artifact_arg(artifact_args, "critical_arc_singular_softness", DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS)
        ),
        sample_likelihood_mode=str(_artifact_arg(artifact_args, "sample_likelihood_mode", "critical-arc-mixture-image-plane")),
    )


def _plot_critical_arc_recovery_by_family(image_count_df: pd.DataFrame, path: Path) -> None:
    _shared_plot_critical_arc_recovery_by_family(image_count_df, path)


def _plot_source_recovery(source_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        ax.scatter(source_df["beta_x"], source_df["beta_y"], color="black", s=28, label="truth")
    if {"source_x_q16", "source_x_q50", "source_x_q84", "source_y_q16", "source_y_q50", "source_y_q84"}.issubset(source_df.columns):
        sx = source_df["source_x_q50"].to_numpy(dtype=float)
        sy = source_df["source_y_q50"].to_numpy(dtype=float)
        ax.errorbar(
            sx,
            sy,
            xerr=[
                np.maximum(0.0, sx - source_df["source_x_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_x_q84"].to_numpy(dtype=float) - sx),
            ],
            yerr=[
                np.maximum(0.0, sy - source_df["source_y_q16"].to_numpy(dtype=float)),
                np.maximum(0.0, source_df["source_y_q84"].to_numpy(dtype=float) - sy),
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            markersize=5,
            label="recovered 1 sigma",
        )
    else:
        sx = source_df["source_x_recovered"].to_numpy(dtype=float)
        sy = source_df["source_y_recovered"].to_numpy(dtype=float)
        ax.scatter(source_df["source_x_recovered"], source_df["source_y_recovered"], color="tab:blue", s=24, label="recovered")
    if {"beta_x", "beta_y"}.issubset(source_df.columns):
        for row, sx_fit, sy_fit in zip(source_df.itertuples(index=False), sx, sy):
            if np.isfinite(sx_fit) and np.isfinite(sy_fit):
                ax.plot([row.beta_x, sx_fit], [row.beta_y, sy_fit], color="0.6", lw=0.8)
    ax.set_xlabel(r"$\beta_x$ (arcsec)")
    ax.set_ylabel(r"$\beta_y$ (arcsec)")
    ax.set_title("Source positions")
    ax.legend(loc="best", fontsize=8)

    if "source_position_error_arcsec" in source_df:
        values = (
            source_df["source_position_error_q50"].to_numpy(dtype=float)
            if "source_position_error_q50" in source_df
            else source_df["source_position_error_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_position_error_q16", "source_position_error_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_position_error_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_position_error_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source position error (arcsec)")
    else:
        values = (
            source_df["source_plane_rms_q50"].to_numpy(dtype=float)
            if "source_plane_rms_q50" in source_df
            else source_df["source_plane_rms_arcsec"].to_numpy(dtype=float)
        )
        x_index = np.arange(len(source_df))
        if {"source_plane_rms_q16", "source_plane_rms_q84"}.issubset(source_df.columns):
            axes[1].errorbar(
                x_index,
                values,
                yerr=[
                    np.maximum(0.0, values - source_df["source_plane_rms_q16"].to_numpy(dtype=float)),
                    np.maximum(0.0, source_df["source_plane_rms_q84"].to_numpy(dtype=float) - values),
                ],
                fmt="o",
                color="tab:blue",
                ecolor="tab:blue",
            )
        else:
            axes[1].scatter(x_index, values, color="tab:blue")
        axes[1].set_xlabel("family index")
        axes[1].set_ylabel("source-plane RMS (arcsec)")
    axes[1].set_title("Source recovery with 1 sigma intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_placeholder_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subhalo_selection(truth: dict[str, Any], path: Path) -> None:
    selection = truth.get("subhalo_selection", {}) if isinstance(truth, dict) else {}
    if not isinstance(selection, dict):
        selection = {}
    candidates = selection.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        _write_placeholder_plot(
            path,
            "Subhalo selection",
            "No parent-population data are available for this mock.",
        )
        return

    candidate_df = pd.DataFrame(candidates)
    required_columns = {"subhalo_mass_msun", "catalog_mag", "selected"}
    if not required_columns.issubset(candidate_df.columns):
        _write_placeholder_plot(
            path,
            "Subhalo selection",
            "Subhalo selection data are incomplete.",
        )
        return

    mass = pd.to_numeric(candidate_df["subhalo_mass_msun"], errors="coerce").to_numpy(dtype=float)
    magnitude = pd.to_numeric(candidate_df["catalog_mag"], errors="coerce").to_numpy(dtype=float)
    selected = candidate_df["selected"].fillna(False).astype(bool).to_numpy()
    passes_mag_cut = (
        candidate_df["passes_mag_cut"].fillna(False).astype(bool).to_numpy()
        if "passes_mag_cut" in candidate_df.columns
        else np.isfinite(magnitude)
    )
    finite = np.isfinite(mass) & (mass > 0.0) & np.isfinite(magnitude)
    if not np.any(finite):
        _write_placeholder_plot(
            path,
            "Subhalo selection",
            "No finite subhalo candidate masses are available.",
        )
        return

    mass = mass[finite]
    magnitude = magnitude[finite]
    selected = selected[finite]
    passes_mag_cut = passes_mag_cut[finite]
    log_mass = np.log10(mass)
    log_min = float(np.nanmin(log_mass))
    log_max = float(np.nanmax(log_mass))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        _write_placeholder_plot(path, "Subhalo selection", "No finite subhalo log masses are available.")
        return
    if log_max <= log_min:
        log_min -= 0.25
        log_max += 0.25
    n_bins = int(np.clip(np.sqrt(len(log_mass)), 6, 18))
    bins = np.linspace(log_min, log_max, n_bins + 1)
    parent_counts, _ = np.histogram(log_mass, bins=bins)
    selected_counts, _ = np.histogram(log_mass[selected], bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    schechter_alpha = float(selection.get("schechter_alpha", SingleBCGMockConfig().subhalo_schechter_alpha))
    mass_ref = float(selection.get("mass_ref", SingleBCGMockConfig().subhalo_mass_ref))
    exponent = float(
        selection.get("mass_luminosity_exponent", _subhalo_mass_luminosity_exponent(SingleBCGMockConfig()))
    )
    center_mass = np.power(10.0, centers)
    center_luminosity = np.power(center_mass / mass_ref, 1.0 / exponent)
    analytic = np.power(center_luminosity, schechter_alpha + 1.0) * np.exp(-center_luminosity)
    analytic_label = fr"Schechter $\alpha={schechter_alpha:.2g}$"
    analytic = analytic / np.sum(analytic) * float(len(log_mass)) if np.sum(analytic) > 0.0 else analytic

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    hist_ax, mag_ax = axes
    hist_ax.hist(log_mass, bins=bins, histtype="stepfilled", alpha=0.25, color="tab:blue", label="parent candidates")
    hist_ax.hist(log_mass[selected], bins=bins, histtype="step", color="tab:red", linewidth=2.0, label="selected")
    hist_ax.plot(centers, analytic, color="black", linestyle="--", linewidth=1.2, label=analytic_label)
    if np.any(selected):
        rug_y = max(float(np.nanmax(parent_counts)) if parent_counts.size else 1.0, 1.0) * 1.25
        hist_ax.scatter(log_mass[selected], np.full(np.count_nonzero(selected), rug_y), marker="v", s=22, color="tab:red")
    hist_ax.set_yscale("log")
    hist_ax.set_xlabel(r"$\log_{10}(M_{\rm sub}/M_\odot)$")
    hist_ax.set_ylabel(r"$dN/d\log_{10}M$")
    hist_ax.set_title("Parent Schechter LF draw")
    hist_ax.legend(loc="best", fontsize=8)

    unselected = ~selected
    mag_ax.scatter(
        mass[unselected & ~passes_mag_cut],
        magnitude[unselected & ~passes_mag_cut],
        s=16,
        color="0.75",
        alpha=0.7,
        label="rejected by mag cut",
    )
    mag_ax.scatter(
        mass[unselected & passes_mag_cut],
        magnitude[unselected & passes_mag_cut],
        s=18,
        color="tab:blue",
        alpha=0.65,
        label="observable parent",
    )
    if np.any(selected):
        mag_ax.scatter(
            mass[selected],
            magnitude[selected],
            s=44,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.3,
            label="selected",
        )
    faint_limit = float(selection.get("mag_faint_limit", np.nan))
    if np.isfinite(faint_limit):
        mag_ax.axhline(faint_limit, color="black", linestyle="--", linewidth=1.0, label="faint limit")
    mag_ax.set_xscale("log")
    mag_ax.invert_yaxis()
    mag_ax.set_xlabel(r"$M_{\rm sub}$ ($M_\odot$)")
    mag_ax.set_ylabel("catalog magnitude")
    mag_ax.set_title("Count-matched selection")
    mag_ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _positive_finite_series(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _updated_component_values(base: Any, param_index: Any, best_fit: np.ndarray) -> np.ndarray:
    values = np.asarray(base, dtype=float).copy()
    indices = np.asarray(param_index, dtype=int)
    best_fit_values = np.asarray(best_fit, dtype=float).reshape(-1)
    if values.shape != indices.shape:
        values = np.broadcast_to(values, indices.shape).astype(float, copy=True)
    valid = (indices >= 0) & (indices < best_fit_values.size)
    if np.any(valid):
        values[valid] = best_fit_values[indices[valid]]
    return values


def _truth_scaling_reference_value(
    truth: dict[str, Any],
    potfile_id: str,
    field: str,
    fallback: float,
) -> float:
    parameter_truth = truth.get("parameter_truth", {}) if isinstance(truth, dict) else {}
    if isinstance(parameter_truth, dict):
        for key in (f"{potfile_id}.{field}", f"potfile.{field}"):
            if key not in parameter_truth:
                continue
            try:
                value = float(parameter_truth[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return value
    return float(fallback)


def _finite_float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _recovered_subhalo_mass_table(state: Any, best_fit: np.ndarray, truth: dict[str, Any]) -> pd.DataFrame:
    packed = getattr(state, "packed_lens_spec", None)
    if packed is None:
        return pd.DataFrame()
    component_family = np.asarray(getattr(packed, "component_family", []), dtype=int)
    if component_family.size == 0:
        return pd.DataFrame()
    scaling_indices = np.where(component_family == 1)[0].astype(int)
    if scaling_indices.size == 0:
        return pd.DataFrame()

    x_center_base = np.asarray(getattr(packed, "x_center_base", np.full(component_family.size, np.nan)), dtype=float)
    y_center_base = np.asarray(getattr(packed, "y_center_base", np.full(component_family.size, np.nan)), dtype=float)
    record_by_component = {
        int(record["component_index"]): dict(record)
        for record in getattr(state, "scaling_component_records", []) or []
        if isinstance(record, dict) and "component_index" in record
    }
    best_fit_values = np.asarray(best_fit, dtype=float).reshape(-1)
    luminosity_ratio = np.asarray(getattr(packed, "luminosity_ratio", []), dtype=float)
    sigma_ref = _updated_component_values(
        getattr(packed, "sigma_ref_base", np.zeros(component_family.size, dtype=float)),
        getattr(packed, "sigma_ref_param_index", np.full(component_family.size, -1, dtype=int)),
        best_fit_values,
    )
    cut_ref = _updated_component_values(
        getattr(packed, "cut_ref_base", np.zeros(component_family.size, dtype=float)),
        getattr(packed, "cut_ref_param_index", np.full(component_family.size, -1, dtype=int)),
        best_fit_values,
    )
    alpha_sigma = _updated_component_values(
        getattr(packed, "alpha_sigma_base", np.full(component_family.size, 0.25, dtype=float)),
        getattr(packed, "alpha_sigma_param_index", np.full(component_family.size, -1, dtype=int)),
        best_fit_values,
    )
    gamma_ml = _updated_component_values(
        getattr(packed, "gamma_ml_base", np.full(component_family.size, 0.2, dtype=float)),
        getattr(packed, "gamma_ml_param_index", np.full(component_family.size, -1, dtype=int)),
        best_fit_values,
    )
    beta_radius = 1.0 + gamma_ml - 2.0 * alpha_sigma
    rows: list[dict[str, Any]] = []
    for component_index in scaling_indices:
        if component_index >= luminosity_ratio.size:
            continue
        luminosity = float(luminosity_ratio[component_index])
        sigma_value = float(sigma_ref[component_index])
        cut_value = float(cut_ref[component_index])
        alpha_sigma_value = float(alpha_sigma[component_index])
        beta_radius_value = float(beta_radius[component_index])
        gamma_ml_value = 2.0 * alpha_sigma_value + beta_radius_value - 1.0
        if not (
            np.isfinite(luminosity)
            and luminosity > 0.0
            and np.isfinite(sigma_value)
            and sigma_value > 0.0
            and np.isfinite(cut_value)
            and cut_value > 0.0
            and np.isfinite(alpha_sigma_value)
            and np.isfinite(beta_radius_value)
        ):
            continue
        record = record_by_component.get(int(component_index), {})
        potfile_id = str(record.get("potfile_id", "potfile"))
        x_arcsec = _finite_float_or_nan(record.get("x_centre"))
        y_arcsec = _finite_float_or_nan(record.get("y_centre"))
        if (not np.isfinite(x_arcsec)) and component_index < x_center_base.size:
            x_arcsec = _finite_float_or_nan(x_center_base[component_index])
        if (not np.isfinite(y_arcsec)) and component_index < y_center_base.size:
            y_arcsec = _finite_float_or_nan(y_center_base[component_index])
        recovered_radius = float(np.hypot(x_arcsec, y_arcsec)) if np.isfinite(x_arcsec) and np.isfinite(y_arcsec) else float("nan")
        truth_sigma_ref = _truth_scaling_reference_value(truth, potfile_id, "sigma", float(sigma_value))
        truth_cut_ref = _truth_scaling_reference_value(truth, potfile_id, "cutkpc", float(cut_value))
        normalization = 1.0
        if np.isfinite(truth_sigma_ref) and truth_sigma_ref > 0.0:
            normalization *= (sigma_value / truth_sigma_ref) ** 2
        if np.isfinite(truth_cut_ref) and truth_cut_ref > 0.0:
            normalization *= cut_value / truth_cut_ref
        # Physical dPIE mass of the fitted member profile — the same convention as the
        # truth `subhalo_dpie_mass_msun` and the aggregate SHMF.
        mass = float(
            10.0
            ** log10_dpie_mass_msun(
                sigma_value * luminosity**alpha_sigma_value,
                cut_value * luminosity**beta_radius_value,
            )
        )
        if not (np.isfinite(mass) and mass > 0.0):
            continue
        rows.append(
            {
                "component_index": int(component_index),
                "potfile_id": potfile_id,
                "catalog_id": str(record.get("catalog_id", f"component{component_index}")),
                "x_arcsec": float(x_arcsec),
                "y_arcsec": float(y_arcsec),
                "recovered_radius_arcsec": recovered_radius,
                "luminosity_ratio": luminosity,
                "sigma_ref": sigma_value,
                "cut_ref_kpc": cut_value,
                "alpha_sigma": alpha_sigma_value,
                "beta_radius": beta_radius_value,
                "gamma_ml": gamma_ml_value,
                "mass_normalization_ratio": float(normalization),
                "recovered_subhalo_mass_msun": float(mass),
            }
        )
    return pd.DataFrame(rows)


def _finite_nonnegative_series(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.empty((0,), dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values) & (values >= 0.0)]


def _finite_radius_from_xy(df: pd.DataFrame, x_column: str, y_column: str) -> np.ndarray:
    if not {x_column, y_column}.issubset(df.columns):
        return np.empty((0,), dtype=float)
    x_values = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return np.empty((0,), dtype=float)
    radii = np.hypot(x_values[finite], y_values[finite])
    return radii[np.isfinite(radii) & (radii >= 0.0)]


def _subhalo_log_mass_bins(*mass_arrays: np.ndarray) -> np.ndarray | None:
    positive_arrays = [np.asarray(values, dtype=float) for values in mass_arrays if np.asarray(values).size]
    positive_arrays = [values[np.isfinite(values) & (values > 0.0)] for values in positive_arrays]
    positive_arrays = [values for values in positive_arrays if values.size]
    if not positive_arrays:
        return None
    all_mass = np.concatenate(positive_arrays)
    log_mass = np.log10(all_mass)
    log_min = float(np.nanmin(log_mass))
    log_max = float(np.nanmax(log_mass))
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if log_max <= log_min:
        log_min -= 0.25
        log_max += 0.25
    n_bins = int(np.clip(np.sqrt(max(int(all_mass.size), 1)), 6, 18))
    return np.linspace(log_min, log_max, n_bins + 1)


def _subhalo_linear_bins(*value_arrays: np.ndarray) -> np.ndarray | None:
    finite_arrays = [np.asarray(values, dtype=float) for values in value_arrays if np.asarray(values).size]
    finite_arrays = [values[np.isfinite(values) & (values >= 0.0)] for values in finite_arrays]
    finite_arrays = [values for values in finite_arrays if values.size]
    if not finite_arrays:
        return None
    all_values = np.concatenate(finite_arrays)
    value_min = float(np.nanmin(all_values))
    value_max = float(np.nanmax(all_values))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return None
    if value_max <= value_min:
        padding = max(0.5, 0.1 * max(abs(value_min), 1.0))
        value_min = max(0.0, value_min - padding)
        value_max += padding
    n_bins = int(np.clip(np.sqrt(max(int(all_values.size), 1)), 6, 18))
    return np.linspace(value_min, value_max, n_bins + 1)


def _plot_subhalo_recovery_shmf(truth: dict[str, Any], recovered_subhalo_df: pd.DataFrame, path: Path) -> None:
    truth_dict = truth if isinstance(truth, dict) else {}
    truth_subhalo_df = pd.DataFrame(truth_dict.get("subhalos", []))
    truth_mass = _positive_finite_series(truth_subhalo_df, "subhalo_dpie_mass_msun")
    if truth_mass.size == 0:
        # Pre-fix mocks only recorded the Schechter abundance label; recompute the
        # physical dPIE masses so the comparison stays in one mass convention.
        truth_mass = dpie_truth_subhalo_masses_from_scaling(truth_dict)
    recovered_mass = _positive_finite_series(recovered_subhalo_df, "recovered_subhalo_mass_msun")
    selection = truth.get("subhalo_selection", {}) if isinstance(truth, dict) else {}
    if not isinstance(selection, dict):
        selection = {}
    if truth_mass.size == 0 or recovered_mass.size == 0:
        _write_placeholder_plot(
            path,
            "Recovered SHMF",
            "Selected truth and recovered subhalo masses are required for this comparison.",
        )
        return

    bins = _subhalo_log_mass_bins(truth_mass, recovered_mass)
    if bins is None:
        _write_placeholder_plot(path, "Recovered SHMF", "No finite subhalo log masses are available.")
        return
    truth_log_mass = np.log10(truth_mass)
    recovered_log_mass = np.log10(recovered_mass)
    bin_width = float(np.mean(np.diff(bins)))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        truth_log_mass,
        bins=bins,
        weights=np.full(truth_log_mass.size, 1.0 / bin_width),
        histtype="stepfilled",
        color="lightgray",
        alpha=0.8,
        label="Truth subhalos",
    )
    ax.hist(
        recovered_log_mass,
        bins=bins,
        weights=np.full(recovered_log_mass.size, 1.0 / bin_width),
        histtype="step",
        color="tab:blue",
        linewidth=2.0,
        label="recovered subhalos",
    )
    schechter_alpha = float(selection.get("schechter_alpha", SingleBCGMockConfig().subhalo_schechter_alpha))
    # Histograms are in physical dPIE mass, so anchor the Schechter curve at the dPIE
    # mass of an L=1 member instead of the abundance-label mass_ref.
    mass_anchor_value = selection.get("dpie_mass_anchor_msun")
    if mass_anchor_value is None:
        mass_anchor_value = dpie_mass_anchor_from_truth(truth_dict)
    mass_anchor = float(mass_anchor_value) if mass_anchor_value is not None else float("nan")
    exponent = float(
        selection.get("mass_luminosity_exponent", _subhalo_mass_luminosity_exponent(SingleBCGMockConfig()))
    )
    if (
        np.isfinite(schechter_alpha)
        and np.isfinite(mass_anchor)
        and mass_anchor > 0.0
        and np.isfinite(exponent)
        and exponent > 0.0
    ):
        log_mass_grid = np.linspace(float(bins[0]), float(bins[-1]), 256)
        mass_grid = np.power(10.0, log_mass_grid)
        luminosity_grid = np.power(mass_grid / mass_anchor, 1.0 / exponent)
        schechter_density = np.power(luminosity_grid, schechter_alpha + 1.0) * np.exp(-luminosity_grid)
        schechter_norm = float(np.trapezoid(schechter_density, log_mass_grid))
        if np.isfinite(schechter_norm) and schechter_norm > 0.0:
            schechter_density = schechter_density / schechter_norm * float(truth_log_mass.size)
            ax.plot(
                log_mass_grid,
                schechter_density,
                color="black",
                linestyle="--",
                linewidth=1.3,
                label=fr"Schechter $\alpha={schechter_alpha:.2g}$",
            )
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="x", which="minor", length=3.0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel(r"$\log_{10}(M_{\rm sub}/M_\odot)$", fontsize=14)
    ax.set_ylabel(r"$dN/d\log_{10}M$", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subhalo_recovery_radial(truth: dict[str, Any], recovered_subhalo_df: pd.DataFrame, path: Path) -> None:
    truth_subhalo_df = pd.DataFrame(truth.get("subhalos", [])) if isinstance(truth, dict) else pd.DataFrame()
    truth_radius = _finite_radius_from_xy(truth_subhalo_df, "x_arcsec", "y_arcsec")
    recovered_radius = _finite_nonnegative_series(recovered_subhalo_df, "recovered_radius_arcsec")
    if truth_radius.size == 0 or recovered_radius.size == 0:
        _write_placeholder_plot(
            path,
            "Recovered subhalo radial distribution",
            "Selected truth and recovered subhalo radii are required for this comparison.",
        )
        return

    bins = _subhalo_linear_bins(truth_radius, recovered_radius)
    if bins is None:
        _write_placeholder_plot(path, "Recovered subhalo radial distribution", "No finite subhalo radii are available.")
        return
    bin_width = float(np.mean(np.diff(bins)))
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        _write_placeholder_plot(path, "Recovered subhalo radial distribution", "Subhalo radial bins are invalid.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        truth_radius,
        bins=bins,
        weights=np.full(truth_radius.size, 1.0 / bin_width),
        histtype="stepfilled",
        color="lightgray",
        alpha=0.8,
        label="Truth subhalos",
    )
    ax.hist(
        recovered_radius,
        bins=bins,
        weights=np.full(recovered_radius.size, 1.0 / bin_width),
        histtype="step",
        color="tab:blue",
        linewidth=2.0,
        label="recovered subhalos",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="x", which="minor", length=3.0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel(r"$R_{\rm sub}$ (arcsec)", fontsize=14)
    ax.set_ylabel(r"$dN/dR$ (arcsec$^{-1}$)", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_prefit_subhalo_spatial_distribution(subhalo_df: pd.DataFrame, images: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    if {"x_obs_arcsec", "y_obs_arcsec"}.issubset(images.columns):
        image_x = pd.to_numeric(images["x_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
        image_y = pd.to_numeric(images["y_obs_arcsec"], errors="coerce").to_numpy(dtype=float)
        finite_images = np.isfinite(image_x) & np.isfinite(image_y)
        if np.any(finite_images):
            ax.scatter(image_x[finite_images], image_y[finite_images], color="black", marker="x", s=26, label="images")
            plotted = True

    if {"x_arcsec", "y_arcsec"}.issubset(subhalo_df.columns):
        subhalo_x = pd.to_numeric(subhalo_df["x_arcsec"], errors="coerce").to_numpy(dtype=float)
        subhalo_y = pd.to_numeric(subhalo_df["y_arcsec"], errors="coerce").to_numpy(dtype=float)
        finite_subhalos = np.isfinite(subhalo_x) & np.isfinite(subhalo_y)
        if np.any(finite_subhalos):
            sizes = np.full(np.count_nonzero(finite_subhalos), 34.0, dtype=float)
            if "luminosity_ratio" in subhalo_df.columns:
                luminosity = pd.to_numeric(subhalo_df["luminosity_ratio"], errors="coerce").to_numpy(dtype=float)
                luminosity = luminosity[finite_subhalos]
                finite_luminosity = np.isfinite(luminosity) & (luminosity > 0.0)
                if np.any(finite_luminosity):
                    sizes[finite_luminosity] = 12.0 + 80.0 * np.sqrt(luminosity[finite_luminosity])
            if "catalog_mag" in subhalo_df.columns:
                magnitude = pd.to_numeric(subhalo_df["catalog_mag"], errors="coerce").to_numpy(dtype=float)
                magnitude = magnitude[finite_subhalos]
                if np.any(np.isfinite(magnitude)):
                    scatter = ax.scatter(
                        subhalo_x[finite_subhalos],
                        subhalo_y[finite_subhalos],
                        s=sizes,
                        c=magnitude,
                        cmap="viridis_r",
                        alpha=0.75,
                        label="subhalos",
                    )
                    fig.colorbar(scatter, ax=ax, label="member magnitude")
                else:
                    ax.scatter(
                        subhalo_x[finite_subhalos],
                        subhalo_y[finite_subhalos],
                        s=sizes,
                        color="tab:purple",
                        alpha=0.75,
                        label="subhalos",
                    )
            else:
                ax.scatter(
                    subhalo_x[finite_subhalos],
                    subhalo_y[finite_subhalos],
                    s=sizes,
                    color="tab:purple",
                    alpha=0.75,
                    label="subhalos",
                )
            plotted = True

    if not plotted:
        plt.close(fig)
        _write_placeholder_plot(
            path,
            "Pre-fit subhalo spatial distribution",
            "No finite image or subhalo positions are available for this mock.",
        )
        return
    ax.scatter([0.0], [0.0], color="tab:red", marker="+", s=80, label="BCG")
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Pre-fit subhalo field")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_prefit_critical_line_contours(
    contours_by_z: dict[str, list[CausticContour]],
) -> dict[str, list[CausticContour]]:
    z9_contours = _select_critical_caustic_plot_contours(contours_by_z)
    if z9_contours:
        return z9_contours
    selected_key: str | None = None
    selected_z = -np.inf
    for z_key, contours in contours_by_z.items():
        if not contours:
            continue
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            continue
        if np.isfinite(z_source) and z_source > selected_z:
            selected_key = str(z_key)
            selected_z = z_source
    return {selected_key: contours_by_z[selected_key]} if selected_key is not None else {}


def _plot_prefit_critical_lines(truth: dict[str, Any], path: Path) -> None:
    contours_by_z = _select_prefit_critical_line_contours(_caustic_contours_by_z_from_truth(truth))
    if not contours_by_z:
        _write_placeholder_plot(
            path,
            "Pre-fit critical lines",
            "No truth critical-line contours are available for this mock.",
        )
        return

    z_label = next(iter(contours_by_z))
    try:
        z_display = f"{float(z_label):.4g}"
    except (TypeError, ValueError):
        z_display = str(z_label)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    image_ax, source_ax = axes
    class_colors = {"primary": "black", "subhalo": "tab:purple"}
    labeled_lines: set[str] = set()
    labeled_caustics: set[str] = set()
    for contours in contours_by_z.values():
        for contour in contours:
            caustic_class = str(contour.caustic_class)
            color = class_colors.get(caustic_class, "0.35")
            line_label = f"{caustic_class} critical line"
            image_ax.plot(
                contour.critical_x,
                contour.critical_y,
                color=color,
                lw=0.9,
                alpha=0.8,
                label=line_label if line_label not in labeled_lines else None,
            )
            labeled_lines.add(line_label)
            caustic_label = f"{caustic_class} caustic"
            source_ax.scatter(
                contour.beta_x,
                contour.beta_y,
                color=color,
                s=2.0,
                alpha=0.6,
                linewidths=0.0,
                label=caustic_label if caustic_label not in labeled_caustics else None,
            )
            labeled_caustics.add(caustic_label)

    image_ax.invert_xaxis()
    image_ax.set_aspect("equal", adjustable="box")
    image_ax.set_xlabel("x (arcsec)")
    image_ax.set_ylabel("y (arcsec)")
    image_ax.set_title(fr"Truth image plane, $z_s={z_display}$")
    image_ax.legend(loc="best", fontsize=7)
    source_ax.set_aspect("equal", adjustable="box")
    source_ax.set_xlabel(r"$\beta_x$ (arcsec)")
    source_ax.set_ylabel(r"$\beta_y$ (arcsec)")
    source_ax.set_title(fr"Truth source plane, $z_s={z_display}$")
    source_ax.legend(loc="best", fontsize=7)
    for ax in axes:
        _limit_recovery_axis_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_critical_caustic_plot_contours(
    contours_by_z: dict[str, list[CausticContour]],
) -> dict[str, list[CausticContour]]:
    selected: dict[str, list[CausticContour]] = {}
    for z_key, contours in contours_by_z.items():
        try:
            z_source = float(z_key)
        except (TypeError, ValueError):
            continue
        if abs(z_source - CRITICAL_CAUSTIC_RECOVERY_SOURCE_REDSHIFT) <= CRITICAL_CAUSTIC_RECOVERY_REDSHIFT_TOL and contours:
            selected[str(z_key)] = contours
    return selected


def _plot_critical_caustic_recovery(
    truth_contours_by_z: dict[str, list[CausticContour]],
    recovered_contours_by_z: dict[str, list[CausticContour]],
    images: pd.DataFrame,
    image_df: pd.DataFrame,
    source_df: pd.DataFrame,
    subhalo_df: pd.DataFrame,
    path: Path,
) -> None:
    truth_contours_by_z = _select_critical_caustic_plot_contours(truth_contours_by_z)
    recovered_contours_by_z = _select_critical_caustic_plot_contours(recovered_contours_by_z)
    if not truth_contours_by_z or not recovered_contours_by_z:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    image_ax, source_ax = axes

    truth_line_labeled = False
    recovered_line_labeled = False
    for contours in truth_contours_by_z.values():
        for contour in contours:
            image_ax.plot(
                contour.critical_x,
                contour.critical_y,
                color="black",
                lw=0.8,
                alpha=0.75,
                label="truth critical line" if not truth_line_labeled else None,
            )
            truth_line_labeled = True
    for contours in recovered_contours_by_z.values():
        for contour in contours:
            image_ax.plot(
                contour.critical_x,
                contour.critical_y,
                color="tab:blue",
                lw=0.8,
                linestyle="--",
                alpha=0.85,
                label="recovered critical line" if not recovered_line_labeled else None,
            )
            recovered_line_labeled = True

    image_ax.invert_xaxis()
    image_ax.set_aspect("equal", adjustable="box")
    image_ax.set_xlabel("x (arcsec)")
    image_ax.set_ylabel("y (arcsec)")
    image_ax.set_title(r"Image plane, $z_s=7$")
    image_ax.legend(loc="best", fontsize=7)

    truth_caustic_labeled = False
    recovered_caustic_labeled = False
    for contours in truth_contours_by_z.values():
        for contour in contours:
            source_ax.scatter(
                contour.beta_x,
                contour.beta_y,
                color="black",
                s=2.0,
                alpha=0.55,
                linewidths=0.0,
                label="truth caustic" if not truth_caustic_labeled else None,
            )
            truth_caustic_labeled = True
    for contours in recovered_contours_by_z.values():
        for contour in contours:
            source_ax.scatter(
                contour.beta_x,
                contour.beta_y,
                color="tab:blue",
                s=2.0,
                alpha=0.65,
                linewidths=0.0,
                label="recovered caustic" if not recovered_caustic_labeled else None,
            )
            recovered_caustic_labeled = True

    source_ax.set_aspect("equal", adjustable="box")
    source_ax.set_xlabel(r"$\beta_x$ (arcsec)")
    source_ax.set_ylabel(r"$\beta_y$ (arcsec)")
    source_ax.set_title(r"Source plane, $z_s=7$")
    source_ax.legend(loc="best", fontsize=7)
    for ax in axes:
        _limit_recovery_axis_ticks(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation_summary(summary: dict[str, float], uncertainty: dict[str, tuple[float, float]], path: Path) -> None:
    labels = [
        "median image residual",
        "median arc-aware residual",
        "median source error",
        "median |mu| frac. error",
        "parameter 1 sigma coverage",
        "parity match fraction",
    ]
    values = [
        summary["median_image_residual_arcsec"],
        summary.get("median_arc_aware_image_residual_arcsec", np.nan),
        summary["median_source_position_error_arcsec"],
        summary["median_abs_magnification_frac_error"],
        summary["parameter_coverage_68_fraction"],
        summary["parity_match_fraction"],
    ]
    keys = [
        "median_image_residual_arcsec",
        "median_arc_aware_image_residual_arcsec",
        "median_source_position_error_arcsec",
        "median_abs_magnification_frac_error",
        "parameter_coverage_68_fraction",
        "parity_match_fraction",
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(labels))
    ax.barh(y, values, color=["tab:blue", "tab:olive", "tab:cyan", "tab:purple", "tab:green", "tab:orange"], alpha=0.85)
    for idx, (key, value) in enumerate(zip(keys, values)):
        low, high = uncertainty.get(key, (np.nan, np.nan))
        if np.isfinite(value) and np.isfinite(low) and np.isfinite(high):
            ax.errorbar(
                value,
                idx,
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("metric value")
    ax.set_title("Mock recovery summary with 1 sigma intervals")
    for idx, value in enumerate(values):
        if np.isfinite(value):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _validation_mock_paths(mock_dir: str | Path) -> MockClusterPaths:
    root = Path(mock_dir)
    return MockClusterPaths(
        root=root,
        par_path=root / "single_bcg_mock.par",
        image_catalog_path=root / "obs_arcs.cat",
        truth_path=root / "truth.json",
        mock_images_path=root / "mock_images.json",
    )


def _validation_mock_complete(paths: MockClusterPaths) -> bool:
    return paths.par_path.exists() and paths.truth_path.exists() and paths.mock_images_path.exists()


def _load_existing_single_bcg_mock(mock_dir: str | Path) -> tuple[MockClusterPaths, pd.DataFrame, dict[str, Any]]:
    paths = _validation_mock_paths(mock_dir)
    if not _validation_mock_complete(paths):
        raise FileNotFoundError(f"Cannot resume; mock inputs are incomplete under {paths.root}")
    images = pd.DataFrame(json.loads(paths.mock_images_path.read_text(encoding="utf-8")))
    truth = _load_truth(paths.truth_path)
    return paths, images, truth


def _validation_recovery_output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "corner_plot": root / "corner.pdf",
        "potfile_corner_plot": root / "potfile_corner.pdf",
        "parameter_recovery_log_plot": root / "parameter_recovery_log.pdf",
        "parameter_recovery_linear_plot": root / "parameter_recovery_linear.pdf",
        "magnification_plot": root / "magnification_recovery.pdf",
        "absolute_magnification_plot": root / "absolute_magnification_recovery.pdf",
        "image_recovery_plot": root / "image_recovery.pdf",
        "image_residual_histogram_plot": root / "image_residual_histogram.pdf",
        "source_recovery_plot": root / "source_recovery.pdf",
        "subhalo_recovery_shmf_plot": root / "subhalo_recovery_shmf.pdf",
        "subhalo_recovery_radial_plot": root / "subhalo_recovery_radial.pdf",
        "summary_plot": root / "validation_summary.pdf",
    }


def _validation_prefit_output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "subhalo_shmf_plot": root / "subhalo_shmf.pdf",
        "prefit_subhalo_spatial_distribution_plot": root / "prefit_subhalo_spatial_distribution.pdf",
        "prefit_critical_lines_plot": root / "prefit_critical_lines.pdf",
    }


def write_prefit_validation_diagnostics(
    truth: dict[str, Any],
    images: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _validation_prefit_output_paths(root)
    subhalo_df = pd.DataFrame(truth.get("subhalos", [])) if isinstance(truth, dict) else pd.DataFrame()
    _plot_subhalo_selection(truth, paths["subhalo_shmf_plot"])
    _plot_prefit_subhalo_spatial_distribution(
        subhalo_df,
        pd.DataFrame(images),
        paths["prefit_subhalo_spatial_distribution_plot"],
    )
    _plot_prefit_critical_lines(truth, paths["prefit_critical_lines_plot"])
    return paths


def _validation_realization_complete(realization_dir: str | Path) -> bool:
    root = Path(realization_dir)
    if not (root / "run_summary.txt").exists():
        return False
    return all(path.exists() for path in _validation_recovery_output_paths(root).values())


def _validation_stage_has_recovery_artifacts(run_dir: str | Path) -> bool:
    artifacts_dir = Path(run_dir) / "artifacts"
    return (artifacts_dir / "plot_bundle.h5").exists() or (artifacts_dir / "posterior_arrays.npz").exists()


VALIDATION_RESULTS_SCHEMA_VERSION = 1


def _read_text_file_optional(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json_file_optional(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_table_artifact(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path,
        "name": path.name,
        "suffix": path.suffix.lower(),
    }
    try:
        if path.suffix.lower() == ".json":
            payload["data"] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            payload["data"] = _validation_dataframe_payload(pd.read_csv(path))
        elif path.suffix.lower() == ".txt":
            payload["text"] = path.read_text(encoding="utf-8")
        else:
            payload["text"] = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive bundle fallback
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _collect_stage_table_artifacts(stage_dir: str | Path) -> dict[str, dict[str, Any]]:
    tables_dir = Path(stage_dir) / "tables"
    if not tables_dir.exists():
        return {}
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(tables_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt"}:
            artifacts[path.name] = _read_table_artifact(path)
    return artifacts


def _validation_stage_manifest(solver_run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(solver_run_dir)
    root = _stage_root_from_run_dir(run_dir)
    candidates: list[tuple[str, Path]] = []
    if run_dir.name in _VALIDATION_STAGE_ORDER:
        for stage_name in _VALIDATION_STAGE_ORDER:
            stage_dir = root / stage_name
            if stage_dir.exists() or stage_dir == run_dir:
                candidates.append((stage_name, stage_dir))
    else:
        candidates.append((root.name, root))

    manifests: list[dict[str, Any]] = []
    for stage_name, stage_dir in candidates:
        artifacts_dir = stage_dir / "artifacts"
        tables_dir = stage_dir / "tables"
        manifests.append(
            {
                "stage": stage_name,
                "stage_dir": stage_dir,
                "exists": stage_dir.exists(),
                "artifacts_dir": artifacts_dir,
                "tables_dir": tables_dir,
                "has_artifacts_dir": artifacts_dir.exists(),
                "has_tables_dir": tables_dir.exists(),
                "has_plot_bundle": (artifacts_dir / "plot_bundle.h5").exists(),
                "has_posterior_arrays": (artifacts_dir / "posterior_arrays.npz").exists(),
                "run_summary_path": tables_dir / "run_summary.json",
                "run_summary": _load_stage_run_summary(stage_dir) if (tables_dir / "run_summary.json").exists() else None,
                "table_artifacts": _collect_stage_table_artifacts(stage_dir),
            }
        )
    return manifests


def _mock_input_payload(paths: MockClusterPaths, images: pd.DataFrame, truth_payload: dict[str, Any]) -> dict[str, Any]:
    member_catalog_path = paths.root / "members.cat"
    truth = _read_json_file_optional(paths.truth_path)
    if not isinstance(truth, dict):
        truth = truth_payload if isinstance(truth_payload, dict) else {}
    mock_images = _read_json_file_optional(paths.mock_images_path)
    if mock_images is None:
        mock_images = images.to_dict(orient="records")
    return {
        "paths": {
            "root": paths.root,
            "par_path": paths.par_path,
            "image_catalog_path": paths.image_catalog_path,
            "truth_path": paths.truth_path,
            "mock_images_path": paths.mock_images_path,
            "member_catalog_path": member_catalog_path if member_catalog_path.exists() else None,
        },
        "files": {
            "par_text": _read_text_file_optional(paths.par_path),
            "image_catalog_text": _read_text_file_optional(paths.image_catalog_path),
            "member_catalog_text": _read_text_file_optional(member_catalog_path) if member_catalog_path.exists() else None,
        },
        "truth": truth,
        "mock_images": mock_images,
        "images": images.to_dict(orient="records"),
        "sources": truth.get("sources", []),
        "subhalos": truth.get("subhalos", []),
        "parameter_truth": truth.get("parameter_truth", {}),
    }


def write_validation_results_json(
    *,
    config: MockValidationConfig,
    seed: int,
    realization_dir: str | Path,
    result_output_dir: str | Path | None = None,
    mock_config: SingleBCGMockConfig,
    paths: MockClusterPaths,
    images: pd.DataFrame,
    truth_payload: dict[str, Any],
    solver_run_dir: str | Path,
    summary_path: str | Path,
    output_paths: dict[str, Path],
    recovery_payload: dict[str, Any],
    recovery_stage: str = "stage2",
    recovery_payloads: dict[str, dict[str, Any]] | None = None,
) -> Path:
    root = _validation_root(config)
    output_root = Path(result_output_dir) if result_output_dir is not None else Path(realization_dir)
    output_path = output_root / "results.json"
    solver_run_path = Path(solver_run_dir)
    solver_root = _stage_root_from_run_dir(solver_run_path)
    debug_log_path = Path(realization_dir) / "run_debug.log"
    sequential_summary_path = solver_root / "sequential_summary.json"
    all_output_paths = dict(output_paths)
    all_output_paths["results_json"] = output_path
    recovery_by_stage = dict(recovery_payloads) if recovery_payloads is not None else {str(recovery_stage): recovery_payload}
    try:
        stage_recovery_metrics: Any = _collect_validation_stage_recovery_metrics(solver_run_path, paths.truth_path)
    except Exception as exc:  # pragma: no cover - defensive bundle fallback
        stage_recovery_metrics = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "schema_version": VALIDATION_RESULTS_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "run": {
            "run_name": str(config.paths.run_name),
            "mock": "single-bcg",
            "realization_dir": Path(realization_dir),
            "validation_root": root,
            "cwd": Path.cwd(),
            "executable": sys.executable,
            "argv": list(sys.argv),
            "config": config.to_json_dict(),
        },
        "mock_cluster": {
            "config": mock_config,
            **_mock_input_payload(paths, images, truth_payload),
        },
        "solver": {
            "run_name": str(config.solver.run_name),
            "root_dir": solver_root,
            "final_run_dir": solver_run_path,
            "final_stage": solver_run_path.name,
            "sequential_summary_path": sequential_summary_path,
            "sequential_summary": _read_json_file_optional(sequential_summary_path),
            "stage_manifests": _validation_stage_manifest(solver_run_path),
        },
        "validation": {
            "output_paths": all_output_paths,
            "recovery_stage": str(recovery_stage),
            "run_summary": {
                "path": Path(summary_path),
                "text": _read_text_file_optional(summary_path),
            },
            "stage_recovery_metrics": stage_recovery_metrics,
            "recovery": recovery_by_stage,
        },
        "debug_log": {
            "path": debug_log_path,
            "text": _read_text_file_optional(debug_log_path),
        },
    }
    return _write_strict_json(output_path, payload)


def _validation_root(config: MockValidationConfig) -> Path:
    root = Path(config.paths.output_dir)
    if config.paths.campaign_name is not None:
        root = root / str(config.paths.campaign_name)
    return root / str(config.paths.run_name)


def _validation_seed_dir(root: str | Path, seed: int) -> Path:
    return Path(root) / f"seed_{int(seed):08d}"


def _validation_run_dir(seed_dir: str | Path, variant_name: str | None) -> Path:
    root = Path(seed_dir)
    return root / str(variant_name) if variant_name is not None else root


def _resume_enabled(config: MockValidationConfig) -> bool:
    return bool(config.runtime.resume)


def _solver_final_stage_dir(plan: Any) -> Path:
    root = Path(plan.output.output_dir) / str(plan.output.run_name)
    if len(plan.stages) == 1 and plan.stages[0].name == "stage2":
        return root
    return root / plan.stages[-1].name


def _run_solver_plan(plan: Any, *, runner: LensClusterRunner | None = None) -> Path:
    active_runner = runner or LensClusterRunner()
    active_runner.run(plan)
    return _solver_final_stage_dir(plan)


def _run_solver_config(solver_config: LensClusterSolverConfig, *, runner: LensClusterRunner | None = None) -> Path:
    return _run_solver_plan(compile_run_plan(solver_config), runner=runner)


_RECOVERY_STAGE_ALIASES = {
    "stage1": "stage1_backprojected_centroid_fit",
    "stage2": "stage2_free_source_forward_fit",
}


def _solver_stage_dirs_by_recovery_alias(plan: Any) -> dict[str, Path]:
    root = Path(plan.output.output_dir) / str(plan.output.run_name)
    by_name = {stage.name: root / stage.name for stage in plan.stages}
    return {
        alias: by_name[stage_name]
        for alias, stage_name in _RECOVERY_STAGE_ALIASES.items()
        if stage_name in by_name
    }


def _recovery_stage_output_dir(realization_dir: str | Path, solver_stage_dir: str | Path) -> Path:
    return Path(realization_dir) / "recovery" / Path(solver_stage_dir).name


def _log_validation_runtime_summary(config: MockValidationConfig) -> None:
    _log(
        config,
        (
            f"[runtime] python={sys.executable} output_dir={config.paths.output_dir} "
            f"campaign={config.paths.campaign_name} run_name={config.paths.run_name} "
            f"variant={config.paths.variant_name} realizations={config.runtime.realizations} seed={config.runtime.seed}"
        ),
    )
    _log(
        config,
        (
            f"[validation] n_primary_families={config.mock.n_primary_families} "
            f"n_subhalo_families={config.mock.n_subhalo_families} n_subhalos={config.mock.n_subhalos} "
            f"primary_source_redshifts={','.join(f'{value:.4g}' for value in config.mock.primary_source_redshifts)} "
            f"subhalo_source_redshifts={','.join(f'{value:.4g}' for value in config.mock.subhalo_source_redshifts)} "
            f"pos_sigma={config.mock.pos_sigma_arcsec} "
            f"solver_run_name={config.solver.run_name} "
            f"posterior_image_diagnostic_mode={config.solver.template.image_diagnostics.posterior_image_diagnostic_mode}"
        ),
    )
    # Self-documenting run configuration: these settings changed campaign semantics in the
    # past without being visible in the run directory, so echo them into the debug log.
    _log(
        config,
        (
            f"[validation-config] subhalo_sigma_scatter_dex={config.mock.subhalo_sigma_scatter_dex:.6g} "
            f"subhalo_cut_scatter_dex={config.mock.subhalo_cut_scatter_dex:.6g} "
            f"scaling_scatter={config.solver.template.scaling.scaling_scatter} "
            f"perturbation_discovery_top_k={config.solver.template.perturbation.perturbation_discovery_top_k} "
            f"pos_sigma_likelihood={config.solver.template.likelihood.pos_sigma_arcsec} "
            f"critical_arc_anisotropic_covariance={config.solver.template.likelihood.critical_arc_anisotropic_covariance} "
            f"dense_mass={config.solver.template.runtime.dense_mass} "
            f"log_sigma_offset_width={config.solver.template.scaling.independent_scaling_free_log_sigma_offset_width} "
            f"log_mass_offset_width={config.solver.template.scaling.independent_scaling_free_log_mass_offset_width}"
        ),
    )


def run_single_bcg_validation(
    config: MockValidationConfig,
    *,
    runner: LensClusterRunner | None = None,
) -> list[dict[str, Path]]:
    config.validate()
    root = _validation_root(config)
    outputs: list[dict[str, Path]] = []
    total_start = time.time()
    for realization in range(int(config.runtime.realizations)):
        seed = int(config.runtime.seed) + realization
        mock_config = replace(config.mock, seed=seed)
        seed_dir = _validation_seed_dir(root, seed)
        realization_dir = _validation_run_dir(seed_dir, config.paths.variant_name)
        _configure_debug_log(config, str(config.paths.run_name), realization_dir)
        _log_validation_runtime_summary(config)
        realization_start = time.time()
        _log_stage_banner(
            config,
            f"VALIDATION REALIZATION {realization + 1}/{int(config.runtime.realizations)}",
            f"seed={seed} dir={realization_dir}",
        )
        _log(
            config,
            (
                f"[stage] realization start index={realization + 1}/{int(config.runtime.realizations)} "
                f"seed={seed} dir={realization_dir}"
            ),
        )
        _log(
            config,
            (
                f"[load] generating mock primary_families={mock_config.n_primary_families} "
                f"subhalo_families={mock_config.n_subhalo_families} subhalos={mock_config.n_subhalos} "
                f"image_count={_image_count_requirement_text(mock_config.min_images_per_family, mock_config.max_images_per_family)} "
                f"primary_seed_grid_pixel={mock_config.primary_seed_grid_pixel_arcsec:.4g} "
                f"subhalo_seed_grid_pixel={mock_config.subhalo_seed_grid_pixel_arcsec:.4g} "
                f"bcg_position_prior_half_width={mock_config.bcg_position_prior_half_width_arcsec:.4g}"
            ),
        )
        mock_dir = seed_dir / "mock"
        resume_mock_paths = _validation_mock_paths(mock_dir)
        if _resume_enabled(config) and _validation_mock_complete(resume_mock_paths):
            paths, images, truth_payload = _run_logged_phase(
                config,
                "validation.load_existing_single_bcg_mock",
                lambda: _load_existing_single_bcg_mock(mock_dir),
                detail=f"seed={seed}",
            )
            _log(config, f"[resume] reusing mock seed={seed} dir={mock_dir}")
        else:
            with _ValidationMockProgress(config) as mock_progress:
                progress_callback = mock_progress.callback if mock_progress.enabled else None
                paths, images, truth_payload = _run_logged_phase(
                    config,
                    "validation.generate_single_bcg_mock",
                    lambda: generate_single_bcg_mock(mock_dir, mock_config, progress_callback=progress_callback),
                    detail=f"seed={seed}",
                )
        _log(
            config,
            (
                f"[load] mock complete images={len(images)} par={paths.par_path} "
                f"catalog={paths.image_catalog_path} truth={paths.truth_path}"
            ),
        )
        realized_mock_config = _caustic_config_from_truth(truth_payload)
        _log(
            config,
            (
                f"[load] realized large-halo positions halo=({realized_mock_config.halo.x_centre:.4g},"
                f"{realized_mock_config.halo.y_centre:.4g}) bcg=({realized_mock_config.bcg.x_centre:.4g},"
                f"{realized_mock_config.bcg.y_centre:.4g})"
            ),
        )
        if _resume_enabled(config):
            _log(config, f"[resume] refreshing validation outputs seed={seed} dir={realization_dir}")
        prefit_dir = seed_dir / "prefit"
        _log(config, f"[output] writing pre-fit diagnostics to {prefit_dir}")
        prefit_output_paths = _run_logged_phase(
            config,
            "validation.write_prefit_diagnostics",
            lambda: write_prefit_validation_diagnostics(truth_payload, images, prefit_dir),
            detail=f"seed={seed}",
        )
        solver_output_dir = realization_dir
        solver_config = solver_config_for_single_bcg_mock(
            replace(config, mock=realized_mock_config),
            paths=paths,
            seed=seed,
            output_dir=solver_output_dir,
        )
        _log(config, f"[validation] launching solver run_name={solver_config.paths.run_name} output_dir={solver_output_dir}")
        solver_plan = compile_run_plan(solver_config)
        solver_run_dir = _run_logged_phase(
            config,
            "validation.cluster_solver",
            lambda: _run_solver_plan(solver_plan, runner=runner),
            detail=f"run_name={solver_config.paths.run_name}",
        )
        _log(config, f"[validation] solver complete final_run_dir={solver_run_dir}")
        solver_stage_dirs = _solver_stage_dirs_by_recovery_alias(solver_plan)
        missing_recovery_stages = [stage for stage in config.solver.recovery_stages if stage not in solver_stage_dirs]
        if missing_recovery_stages:
            available = ",".join(sorted(solver_stage_dirs)) or "none"
            requested = ",".join(missing_recovery_stages)
            raise ValueError(f"Requested recovery stage(s) unavailable: {requested}. Available recovery stages: {available}.")
        recovery_root = realization_dir / "recovery"
        summary_path = _run_logged_phase(
            config,
            "validation.write_run_summary_txt",
            lambda: write_validation_run_summary(
                solver_run_dir,
                paths.truth_path,
                recovery_root,
                run_name=str(config.paths.run_name),
                seed=seed,
            ),
            detail=f"seed={seed}",
        )
        _log(config, f"[output] validation run summary written to {summary_path}")
        recovery_payloads: dict[str, dict[str, Any]] = {}
        output_paths: dict[str, Path] = {}
        stage_results: dict[str, Path] = {}
        for recovery_stage in config.solver.recovery_stages:
            stage_solver_run_dir = solver_stage_dirs[recovery_stage]
            stage_recovery_dir = _recovery_stage_output_dir(realization_dir, stage_solver_run_dir)
            _log(
                config,
                f"[output] writing {recovery_stage} recovery outputs from {stage_solver_run_dir} to {stage_recovery_dir}",
            )
            recovery_payload: dict[str, Any] = {}
            stage_output_paths = _run_logged_phase(
                config,
                f"validation.write_recovery_outputs.{recovery_stage}",
                lambda stage_solver_run_dir=stage_solver_run_dir, stage_recovery_dir=stage_recovery_dir, recovery_payload=recovery_payload: write_recovery_outputs(
                    stage_solver_run_dir,
                    paths.truth_path,
                    paths.mock_images_path,
                    output_dir=stage_recovery_dir,
                    posterior_image_diagnostic_draws=int(
                        config.solver.template.image_diagnostics.posterior_image_diagnostic_draws
                    ),
                    posterior_diagnostic_mode=str(
                        config.solver.template.image_diagnostics.posterior_image_diagnostic_mode
                    ),
                    critical_caustic_plot_grid_scale_arcsec=float(
                        config.solver.template.truth.caustic_plot_grid_scale_arcsec
                        if config.solver.template.truth.caustic_plot_grid_scale_arcsec is not None
                        else DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC
                    ),
                    posterior_truth_recovery_draws=int(config.solver.template.truth.posterior_truth_recovery_draws or 0),
                    quick_diagnostics=bool(config.solver.template.runtime.quick_diagnostics),
                    progress_args=config,
                    recovery_payload=recovery_payload,
                ),
                detail=f"seed={seed} stage={recovery_stage}",
            )
            recovery_payloads[recovery_stage] = recovery_payload
            output_paths.update({f"{recovery_stage}_{name}": path for name, path in stage_output_paths.items()})
            stage_results_path = _run_logged_phase(
                config,
                f"validation.write_results_json.{recovery_stage}",
                lambda recovery_stage=recovery_stage, stage_solver_run_dir=stage_solver_run_dir, stage_recovery_dir=stage_recovery_dir, stage_output_paths=stage_output_paths, recovery_payload=recovery_payload: write_validation_results_json(
                    config=config,
                    seed=seed,
                    realization_dir=realization_dir,
                    result_output_dir=stage_recovery_dir,
                    mock_config=realized_mock_config,
                    paths=paths,
                    images=images,
                    truth_payload=truth_payload,
                    solver_run_dir=stage_solver_run_dir,
                    summary_path=summary_path,
                    output_paths=stage_output_paths,
                    recovery_payload=recovery_payload,
                    recovery_stage=recovery_stage,
                ),
                detail=f"seed={seed} stage={recovery_stage}",
            )
            stage_results[recovery_stage] = stage_results_path
            output_paths[f"{recovery_stage}_results_json"] = stage_results_path
        output_paths.update(prefit_output_paths)
        results_json_path = _run_logged_phase(
            config,
            "validation.write_recovery_index_json",
            lambda: write_validation_results_json(
                config=config,
                seed=seed,
                realization_dir=realization_dir,
                result_output_dir=recovery_root,
                mock_config=realized_mock_config,
                paths=paths,
                images=images,
                truth_payload=truth_payload,
                solver_run_dir=solver_run_dir,
                summary_path=summary_path,
                output_paths={**output_paths, **{f"{stage}_results_json": path for stage, path in stage_results.items()}},
                recovery_payload=recovery_payloads.get(config.solver.recovery_stages[-1], {}),
                recovery_stage=str(config.solver.recovery_stages[-1]),
                recovery_payloads=recovery_payloads,
            ),
            detail=f"seed={seed}",
        )
        output_paths["results_json"] = results_json_path
        _log(config, f"[output] validation results json written to {results_json_path}")
        _log(config, f"[output] recovery complete files={len(output_paths)} names={','.join(sorted(output_paths))}")
        outputs.append(output_paths)
        _log(
            config,
            (
                f"[stage] realization end index={realization + 1}/{int(config.runtime.realizations)} "
                f"elapsed={_fmt_seconds(time.time() - realization_start)}"
            ),
        )
    _log(config, f"[done] validation complete realizations={len(outputs)} elapsed={_fmt_seconds(time.time() - total_start)} root={root}")
    return outputs
