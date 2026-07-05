from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = f"/tmp/mpl_multi_cluster_solver_{os.getpid()}"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from . import cluster_solver as single
from .model import BuildState, ParameterSpec, PosteriorResults
from .plotting import (
    _best_fit_values_for_specs,
    _map_values_for_specs,
    _plot_corner,
    _plot_cosmology_corner,
    _plot_trace,
    _summary_table,
)


SHARED_COSMOLOGY_SAMPLE_NAMES = (
    single.COSMOLOGY_OM0_SAMPLE_NAME,
    single.COSMOLOGY_W0_SAMPLE_NAME,
)
DEFAULT_WARM_STAGE_PRIORITY = (
    single.STAGE2_FREE_SOURCE_FORWARD_FIT_DIR,
    single.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR,
)
SUPPORTED_WARM_STAGES = ("auto", *DEFAULT_WARM_STAGE_PRIORITY)
CLUSTER_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ClusterInput:
    key: str
    par_path: Path
    warm_run_dir: Path


@dataclass(frozen=True)
class WarmStageResolution:
    stage_name: str
    stage_artifacts_dir: Path
    stage1_artifacts_dir: Path


@dataclass
class MultiClusterContext:
    cluster: ClusterInput
    warm_stage: WarmStageResolution
    state: BuildState
    evaluator: Any
    local_to_global_indices: np.ndarray | None = None
    local_to_global_indices_jax: jnp.ndarray | None = None


@dataclass
class MultiClusterState:
    run_name: str
    parameter_specs: list[ParameterSpec]
    svi_init_values: dict[str, float] | None
    contexts: list[MultiClusterContext]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint flat-wCDM cosmology fit across multiple Lenstool par files."
    )
    parser.add_argument(
        "--cluster",
        action="append",
        nargs=3,
        metavar=("KEY", "PAR_PATH", "WARM_RUN_DIR"),
        required=True,
        help=(
            "Cluster key, Lenstool par file, and sequential warm-run directory. "
            "Repeat once per cluster."
        ),
    )
    parser.add_argument("--output-dir", default="results/multi_cluster_cosmology")
    parser.add_argument("--run-name", default="joint_cosmology")
    parser.add_argument(
        "--warm-stage",
        choices=SUPPORTED_WARM_STAGES,
        default="auto",
        help=(
            "Warm stage to initialize from. auto prefers "
            "stage2_free_source_forward_fit, then stage1_backprojected_centroid_fit."
        ),
    )
    parser.add_argument(
        "--image-plane-mode",
        choices=(
            single.IMAGE_PLANE_MODE_NONE,
            single.IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
            single.IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        ),
        default=single.IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        help="Likelihood used for the joint final stage.",
    )
    parser.add_argument(
        "--fit-method",
        choices=(single.FIT_METHOD_SVI, single.FIT_METHOD_SVI_NUTS),
        default=single.FIT_METHOD_SVI_NUTS,
    )
    parser.add_argument("--warmup", type=int, default=single.DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=single.DEFAULT_SAMPLES)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cosmology-init-om0", type=float, default=None)
    parser.add_argument("--cosmology-init-w0", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pos-sigma-arcsec", type=float, default=None)
    parser.add_argument(
        "--fov-limit-radius",
        type=float,
        default=None,
        metavar="ARCSEC",
        help="Inclusive circular FOV radius in arcsec for catalog image/member filtering.",
    )
    parser.add_argument(
        "--fov-limit-x",
        type=float,
        nargs=2,
        default=None,
        metavar=("X_LEFT", "X_RIGHT"),
        help="Inclusive x arcsec bounds for catalog image/member filtering; values are order-insensitive.",
    )
    parser.add_argument(
        "--fov-limit-y",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_BOTTOM", "Y_TOP"),
        help="Inclusive y arcsec bounds for catalog image/member filtering; values are order-insensitive.",
    )
    parser.add_argument(
        "--sampling-engine",
        choices=single.SAMPLING_ENGINES,
        default=single.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
    )
    parser.add_argument("--source-plane-covariance-floor", type=float, default=1.0e-6)
    parser.add_argument("--scaling-scatter", action="store_true")
    parser.add_argument("--refresh-every", type=single._parse_refresh_every_value, nargs="+", default=[single.DEFAULT_REFRESH_EVERY])
    parser.add_argument(
        "--refresh-param-drift-frac",
        type=float,
        default=single.DEFAULT_REFRESH_PARAM_DRIFT_FRAC,
    )
    parser.add_argument("--z-bin-efficiency-tol", type=float, default=single.DEFAULT_Z_BIN_EFFICIENCY_TOL)
    parser.add_argument(
        "--image-plane-newton-steps",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
    )
    parser.add_argument(
        "--linearized-beta-prior-sigma-arcsec",
        type=float,
        default=single.DEFAULT_LINEARIZED_BETA_PRIOR_SIGMA_ARCSEC,
    )
    parser.add_argument(
        "--source-position-parameterization",
        choices=single.SOURCE_POSITION_PARAMETERIZATIONS,
        default=single.SOURCE_POSITION_PARAMETERIZATION_CONDITIONAL_WHITENED,
    )
    parser.add_argument(
        "--image-plane-scatter-upper-arcsec",
        type=float,
        default=single.DEFAULT_IMAGE_SIGMA_INT_UPPER_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-floor-arcsec",
        type=float,
        default=single.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-prior",
        choices=single.IMAGE_PLANE_SCATTER_PRIORS,
        default=single.DEFAULT_IMAGE_PLANE_SCATTER_PRIOR,
    )
    parser.add_argument(
        "--image-plane-scatter-prior-median-arcsec",
        type=float,
        default=single.DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_MEDIAN_ARCSEC,
    )
    parser.add_argument(
        "--image-plane-scatter-prior-log-sigma",
        type=float,
        default=single.DEFAULT_IMAGE_PLANE_SCATTER_PRIOR_LOG_SIGMA,
    )
    parser.add_argument(
        "--fix-image-sigma-int-arcsec",
        type=float,
        default=None,
        help="Use deterministic intrinsic image-plane scatter instead of sampling image.sigma_int.",
    )
    parser.add_argument("--image-presence-penalty-weight", type=float, default=None)
    parser.add_argument(
        "--image-presence-match-radius-arcsec",
        type=float,
        default=single.DEFAULT_IMAGE_PRESENCE_MATCH_RADIUS_ARCSEC,
    )
    parser.add_argument(
        "--image-presence-temperature-arcsec",
        type=float,
        default=single.DEFAULT_IMAGE_PRESENCE_TEMPERATURE_ARCSEC,
    )
    parser.add_argument(
        "--image-presence-count-softness",
        type=float,
        default=single.DEFAULT_IMAGE_PRESENCE_COUNT_SOFTNESS,
    )
    parser.add_argument(
        "--image-presence-count-margin",
        type=float,
        default=single.DEFAULT_IMAGE_PRESENCE_COUNT_MARGIN,
    )
    parser.add_argument(
        "--likelihood-stabilizer-max-gain",
        type=float,
        default=single.DEFAULT_LIKELIHOOD_STABILIZER_MAX_GAIN,
    )
    parser.add_argument(
        "--likelihood-stabilizer-max-residual-arcsec",
        type=float,
        default=single.DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
    )
    parser.add_argument(
        "--likelihood-stabilizer-residual-loss",
        choices=single.LIKELIHOOD_STABILIZER_RESIDUAL_LOSSES,
        default=single.DEFAULT_LIKELIHOOD_STABILIZER_RESIDUAL_LOSS,
    )
    parser.add_argument(
        "--likelihood-stabilizer-student-t-nu",
        type=float,
        default=single.DEFAULT_LIKELIHOOD_STABILIZER_STUDENT_T_NU,
    )
    parser.add_argument(
        "--source-plane-outlier-sigma-arcsec",
        type=float,
        default=single.DEFAULT_SOURCE_PLANE_OUTLIER_SIGMA_ARCSEC,
    )
    parser.add_argument("--match-tolerance-arcsec", type=float, default=single.DEFAULT_MATCH_TOLERANCE)
    parser.add_argument("--max-tree-depth", type=int, default=single.DEFAULT_MAX_TREE_DEPTH)
    parser.add_argument("--target-accept", type=float, default=single.DEFAULT_TARGET_ACCEPT)
    parser.add_argument("--initial-step-size", type=float, default=single.DEFAULT_INITIAL_STEP_SIZE)
    parser.add_argument(
        "--dense-mass",
        choices=(single.NUTS_DENSE_MASS_FULL, single.NUTS_DENSE_MASS_DIAGONAL),
        default=single.DEFAULT_NUTS_DENSE_MASS,
        help="NumPyro NUTS mass-matrix adaptation for blocked samplers: full dense matrix or diagonal mass.",
    )
    parser.add_argument("--blocked-nuts-cycles", type=int, default=None)
    parser.add_argument("--blocked-nuts-pilot-warmup", type=int, default=None)
    parser.add_argument("--nuts-init-boundary-frac", type=float, default=single.DEFAULT_NUTS_INIT_BOUNDARY_FRAC)
    parser.add_argument("--nuts-init-jitter-frac", type=float, default=single.DEFAULT_NUTS_INIT_JITTER_FRAC)
    parser.add_argument("--svi-steps", type=int, nargs="+", default=[single.DEFAULT_SVI_STEPS])
    parser.add_argument("--svi-learning-rate", type=float, default=single.DEFAULT_SVI_LEARNING_RATE)
    parser.add_argument("--quick-diagnostics", action="store_true")
    parser.set_defaults(
        fit_mode=single.FIT_MODE_JOINT,
        fit_cosmology_flat_wcdm=True,
        sample_likelihood_mode=single.SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        evidence_source_prior_sigma_arcsec=None,
        evidence_source_prior_mean_x_arcsec=0.0,
        evidence_source_prior_mean_y_arcsec=0.0,
        ns_num_live_points=None,
        ns_max_samples=None,
        ns_dlogz=single.DEFAULT_NS_DLOGZ,
        sampler="numpyro_nuts",
        stage1_run_dir=None,
        truth=None,
    )
    args = parser.parse_args(argv)
    if not np.isfinite(float(args.likelihood_stabilizer_max_gain)) or float(args.likelihood_stabilizer_max_gain) < 0.0:
        raise SystemExit("--likelihood-stabilizer-max-gain must be non-negative.")
    if (
        not np.isfinite(float(args.image_plane_scatter_floor_arcsec))
        or float(args.image_plane_scatter_floor_arcsec) <= 0.0
    ):
        raise SystemExit("--image-plane-scatter-floor-arcsec must be positive.")
    image_scatter_floor = float(args.image_plane_scatter_floor_arcsec)
    image_scatter_upper = float(args.image_plane_scatter_upper_arcsec)
    if not np.isfinite(image_scatter_upper) or image_scatter_upper <= image_scatter_floor:
        raise SystemExit(
            "--image-plane-scatter-upper-arcsec must be greater than "
            "--image-plane-scatter-floor-arcsec."
        )
    if (
        not np.isfinite(float(args.image_plane_scatter_prior_median_arcsec))
        or float(args.image_plane_scatter_prior_median_arcsec) <= 0.0
    ):
        raise SystemExit("--image-plane-scatter-prior-median-arcsec must be positive.")
    if (
        str(args.image_plane_scatter_prior) == single.IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL
        and not (image_scatter_floor < float(args.image_plane_scatter_prior_median_arcsec) < image_scatter_upper)
    ):
        raise SystemExit(
            "--image-plane-scatter-prior-median-arcsec must be between "
            "--image-plane-scatter-floor-arcsec and --image-plane-scatter-upper-arcsec for lognormal scatter priors."
        )
    if (
        not np.isfinite(float(args.image_plane_scatter_prior_log_sigma))
        or float(args.image_plane_scatter_prior_log_sigma) <= 0.0
    ):
        raise SystemExit("--image-plane-scatter-prior-log-sigma must be positive.")
    if args.fix_image_sigma_int_arcsec is not None and (
        not np.isfinite(float(args.fix_image_sigma_int_arcsec)) or float(args.fix_image_sigma_int_arcsec) < 0.0
    ):
        raise SystemExit("--fix-image-sigma-int-arcsec must be finite and nonnegative.")
    if (
        not np.isfinite(float(args.likelihood_stabilizer_max_residual_arcsec))
        or float(args.likelihood_stabilizer_max_residual_arcsec) < 0.0
    ):
        raise SystemExit("--likelihood-stabilizer-max-residual-arcsec must be non-negative.")
    if not np.isfinite(float(args.likelihood_stabilizer_student_t_nu)) or float(args.likelihood_stabilizer_student_t_nu) <= 0.0:
        raise SystemExit("--likelihood-stabilizer-student-t-nu must be positive.")
    if args.blocked_nuts_cycles is not None and int(args.blocked_nuts_cycles) <= 0:
        raise SystemExit("--blocked-nuts-cycles must be positive when provided.")
    if args.blocked_nuts_pilot_warmup is not None and int(args.blocked_nuts_pilot_warmup) < 0:
        raise SystemExit("--blocked-nuts-pilot-warmup must be non-negative when provided.")
    try:
        single._cosmology_init_overrides_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.cluster_inputs = _parse_cluster_inputs(args.cluster)
    args.sample_likelihood_mode = _sample_likelihood_mode_from_image_plane_mode(args.image_plane_mode)
    return args


def _parse_cluster_inputs(cluster_values: list[list[str]]) -> list[ClusterInput]:
    inputs: list[ClusterInput] = []
    seen: set[str] = set()
    for raw_key, raw_par_path, raw_warm_run_dir in cluster_values:
        key = str(raw_key)
        if not CLUSTER_KEY_RE.match(key):
            raise SystemExit(
                f"Invalid cluster key {key!r}; use letters, digits, and underscores, starting with a letter."
            )
        if key in seen:
            raise SystemExit(f"Duplicate cluster key {key!r}.")
        seen.add(key)
        inputs.append(
            ClusterInput(
                key=key,
                par_path=Path(raw_par_path),
                warm_run_dir=Path(raw_warm_run_dir),
            )
        )
    if len(inputs) < 2:
        raise SystemExit("At least two --cluster entries are required for a joint fit.")
    return inputs


def _sample_likelihood_mode_from_image_plane_mode(image_plane_mode: str) -> str:
    if str(image_plane_mode) == single.IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA:
        return single.SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    if str(image_plane_mode) == single.IMAGE_PLANE_MODE_LOCAL_JACOBIAN:
        return single.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN
    return single.SAMPLE_LIKELIHOOD_SOURCE


def _has_warm_artifacts(artifacts_dir: Path) -> bool:
    return single._has_plot_artifacts(Path(artifacts_dir))


def _resolve_warm_stage(cluster: ClusterInput, warm_stage: str) -> WarmStageResolution:
    if not cluster.par_path.is_file():
        raise FileNotFoundError(f"Cluster {cluster.key}: par file does not exist: {cluster.par_path}")
    stage1_artifacts_dir = cluster.warm_run_dir / single.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR / "artifacts"
    if not _has_warm_artifacts(stage1_artifacts_dir):
        raise FileNotFoundError(
            f"Cluster {cluster.key}: missing stage1 warm artifacts: {stage1_artifacts_dir}"
        )
    stages = DEFAULT_WARM_STAGE_PRIORITY if warm_stage == "auto" else (warm_stage,)
    for stage_name in stages:
        stage_artifacts_dir = cluster.warm_run_dir / stage_name / "artifacts"
        if _has_warm_artifacts(stage_artifacts_dir):
            return WarmStageResolution(
                stage_name=stage_name,
                stage_artifacts_dir=stage_artifacts_dir,
                stage1_artifacts_dir=stage1_artifacts_dir,
            )
    searched = ", ".join(str(cluster.warm_run_dir / stage / "artifacts") for stage in stages)
    raise FileNotFoundError(f"Cluster {cluster.key}: missing requested warm artifacts; searched {searched}")


def _is_shared_cosmology_spec(spec: ParameterSpec) -> bool:
    return spec.sample_name in SHARED_COSMOLOGY_SAMPLE_NAMES or spec.component_family == "cosmology"


def _prefixed_sample_name(cluster_key: str, sample_name: str) -> str:
    if sample_name in SHARED_COSMOLOGY_SAMPLE_NAMES:
        return sample_name
    return f"{cluster_key}__{sample_name}"


def _prefix_parameter_spec(cluster_key: str, spec: ParameterSpec) -> ParameterSpec:
    if _is_shared_cosmology_spec(spec):
        return spec
    parent_sample_name = (
        None
        if spec.parent_sample_name is None
        else _prefixed_sample_name(cluster_key, spec.parent_sample_name)
    )
    return replace(
        spec,
        name=f"{cluster_key}.{spec.name}",
        sample_name=_prefixed_sample_name(cluster_key, spec.sample_name),
        potential_id=f"{cluster_key}__{spec.potential_id}",
        parent_sample_name=parent_sample_name,
    )


def _validate_compatible_cosmologies(contexts: list[MultiClusterContext]) -> None:
    if not contexts:
        raise ValueError("No cluster contexts were provided.")
    h0_values: list[tuple[str, float]] = []
    for context in contexts:
        config = dict(getattr(context.state, "cosmo_config", {}) or {})
        class_name = str(config.get("class", "FlatLambdaCDM"))
        if class_name not in {"FlatLambdaCDM", "FlatwCDM"}:
            raise ValueError(
                f"Cluster {context.cluster.key}: unsupported cosmology class {class_name!r}; "
                "joint cosmology currently requires flat LambdaCDM/flat wCDM inputs."
            )
        h0_values.append((context.cluster.key, float(config.get("H0", 70.0))))
    reference_key, reference_h0 = h0_values[0]
    for key, h0 in h0_values[1:]:
        if not np.isclose(h0, reference_h0, rtol=0.0, atol=1.0e-10):
            raise ValueError(
                f"Cluster {key}: fixed H0={h0} does not match {reference_key} H0={reference_h0}."
            )


def _build_global_parameter_layout(contexts: list[MultiClusterContext]) -> list[ParameterSpec]:
    global_specs: list[ParameterSpec] = []
    global_index_by_sample: dict[str, int] = {}
    shared_cosmology_reference: dict[str, ParameterSpec] = {}
    for context in contexts:
        local_to_global: list[int] = []
        for spec in context.state.parameter_specs:
            if _is_shared_cosmology_spec(spec):
                if spec.sample_name not in SHARED_COSMOLOGY_SAMPLE_NAMES:
                    raise ValueError(
                        f"Cluster {context.cluster.key}: unsupported cosmology sample {spec.sample_name!r}."
                    )
                existing = shared_cosmology_reference.get(spec.sample_name)
                if existing is None:
                    shared_cosmology_reference[spec.sample_name] = spec
                    global_index_by_sample[spec.sample_name] = len(global_specs)
                    global_specs.append(spec)
                else:
                    _validate_matching_shared_spec(existing, spec, context.cluster.key)
                local_to_global.append(global_index_by_sample[spec.sample_name])
                continue
            prefixed = _prefix_parameter_spec(context.cluster.key, spec)
            if prefixed.sample_name in global_index_by_sample:
                raise ValueError(f"Duplicate global sample name {prefixed.sample_name!r}.")
            global_index_by_sample[prefixed.sample_name] = len(global_specs)
            local_to_global.append(len(global_specs))
            global_specs.append(prefixed)
        context.local_to_global_indices = np.asarray(local_to_global, dtype=np.int32)
        context.local_to_global_indices_jax = jnp.asarray(context.local_to_global_indices, dtype=jnp.int32)
    missing_cosmology = sorted(set(SHARED_COSMOLOGY_SAMPLE_NAMES) - set(shared_cosmology_reference))
    if missing_cosmology:
        raise ValueError(f"Missing shared cosmology parameter(s): {', '.join(missing_cosmology)}")
    return global_specs


def _validate_matching_shared_spec(reference: ParameterSpec, candidate: ParameterSpec, cluster_key: str) -> None:
    fields = ("lower", "upper", "prior_kind", "transform_kind", "component_family")
    for field_name in fields:
        if getattr(reference, field_name) != getattr(candidate, field_name):
            raise ValueError(
                f"Cluster {cluster_key}: shared cosmology spec {candidate.sample_name!r} has "
                f"different {field_name} from the first cluster."
            )


def _global_init_values(
    global_specs: list[ParameterSpec],
    contexts: list[MultiClusterContext],
) -> dict[str, float]:
    init_values: dict[str, float] = {}
    for context in contexts:
        local_init = dict(context.state.svi_init_values or {})
        for spec in context.state.parameter_specs:
            if spec.sample_name not in local_init:
                continue
            global_sample_name = _prefixed_sample_name(context.cluster.key, spec.sample_name)
            init_values.setdefault(global_sample_name, float(local_init[spec.sample_name]))
    for spec in global_specs:
        if spec.sample_name in init_values:
            continue
        if spec.mean is not None:
            init_values[spec.sample_name] = float(spec.mean)
        elif spec.prior_kind == "uniform":
            init_values[spec.sample_name] = 0.5 * (float(spec.lower) + float(spec.upper))
    return init_values


class MultiClusterJAXEvaluator:
    def __init__(self, contexts: list[MultiClusterContext]):
        self.contexts = contexts
        self.surrogate_enabled = any(bool(getattr(context.evaluator, "surrogate_enabled", False)) for context in contexts)
        self.timing_totals: dict[str, float] = {
            "initial_jit_compile": 0.0,
            "svi_runtime": 0.0,
            "nuts_runtime": 0.0,
        }
        self.invalid_state_rejection_count = 0
        self.invalid_state_reason_counts = {name: 0 for name in single.INVALID_STATE_REASON_NAMES}
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _local_theta(self, params: jnp.ndarray, context: MultiClusterContext) -> jnp.ndarray:
        if context.local_to_global_indices_jax is None:
            raise ValueError(f"Cluster {context.cluster.key} is missing global parameter mapping.")
        return jnp.take(jnp.asarray(params, dtype=jnp.float64), context.local_to_global_indices_jax)

    def _source_loglike_impl(self, params: jnp.ndarray) -> jnp.ndarray:
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for context in self.contexts:
            local_theta = self._local_theta(params, context)
            total = total + context.evaluator._source_loglike_fn(local_theta)
        return jnp.nan_to_num(total, nan=single.BAD_LOG_LIKE, posinf=single.BAD_LOG_LIKE, neginf=single.BAD_LOG_LIKE)

    def source_loglike(self, params: np.ndarray | jnp.ndarray) -> float:
        value = float(self._source_loglike_fn(jnp.asarray(params, dtype=jnp.float64)))
        self._sync_invalid_state_counts()
        return value

    def refresh_surrogate(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        for context in self.contexts:
            local_theta = self.local_theta_numpy(reference_params, context)
            if hasattr(context.evaluator, "refresh_surrogate"):
                context.evaluator.refresh_surrogate(local_theta, reason=reason)
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)
        self._sync_invalid_state_counts()

    def refresh_scaling_scatter_cache(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        for context in self.contexts:
            if hasattr(context.evaluator, "refresh_scaling_scatter_cache"):
                context.evaluator.refresh_scaling_scatter_cache(
                    self.local_theta_numpy(reference_params, context),
                    reason=reason,
                )
        self._sync_invalid_state_counts()

    def refresh_source_metric_cache(self, reference_params: np.ndarray, reason: str = "manual") -> None:
        for context in self.contexts:
            if hasattr(context.evaluator, "refresh_source_metric_cache"):
                context.evaluator.refresh_source_metric_cache(
                    self.local_theta_numpy(reference_params, context),
                    reason=reason,
                )
        self._sync_invalid_state_counts()

    def local_theta_numpy(self, params: np.ndarray, context: MultiClusterContext) -> np.ndarray:
        if context.local_to_global_indices is None:
            raise ValueError(f"Cluster {context.cluster.key} is missing global parameter mapping.")
        return np.asarray(params, dtype=float)[context.local_to_global_indices]

    def cluster_loglikes(self, params: np.ndarray) -> dict[str, float]:
        return {
            context.cluster.key: float(context.evaluator.source_loglike(self.local_theta_numpy(params, context)))
            for context in self.contexts
        }

    def release_runtime_caches(self) -> None:
        for context in self.contexts:
            if hasattr(context.evaluator, "release_runtime_caches"):
                context.evaluator.release_runtime_caches()

    def _sync_invalid_state_counts(self) -> None:
        self.invalid_state_rejection_count = int(
            sum(int(getattr(context.evaluator, "invalid_state_rejection_count", 0)) for context in self.contexts)
        )
        counts = {name: 0 for name in single.INVALID_STATE_REASON_NAMES}
        for context in self.contexts:
            for name, value in dict(getattr(context.evaluator, "invalid_state_reason_counts", {})).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        self.invalid_state_reason_counts = counts


def _build_local_context(cluster: ClusterInput, warm_stage: WarmStageResolution, args: argparse.Namespace) -> MultiClusterContext:
    stage1_summary = single._load_stage1_summary(warm_stage.stage1_artifacts_dir)
    warm_best_values = single._physical_best_fit_values_from_artifacts(warm_stage.stage_artifacts_dir)
    source_position_priors = (
        single._source_position_prior_values_from_artifacts(warm_stage.stage_artifacts_dir)
        if single._sample_likelihood_uses_explicit_beta(args.sample_likelihood_mode)
        else None
    )
    local_args = single._clone_args(
        args,
        par_path=str(cluster.par_path),
        run_name=f"{args.run_name}/{cluster.key}",
        fit_mode=single.FIT_MODE_JOINT,
        fit_method=args.fit_method,
        fit_cosmology_flat_wcdm=True,
        sample_likelihood_mode=args.sample_likelihood_mode,
    )
    state = single._build_state_from_inputs(
        local_args,
        fit_mode_override=single.FIT_MODE_JOINT,
        stage1_prior_summary=stage1_summary,
        svi_init_physical_values=warm_best_values,
        source_position_prior_values=source_position_priors,
        previous_stage_best_values=warm_best_values,
    )
    evaluator, _midpoint = single._prepare_direct_evaluator(local_args, state)
    return MultiClusterContext(
        cluster=cluster,
        warm_stage=warm_stage,
        state=state,
        evaluator=evaluator,
    )


def _build_multi_cluster_state(args: argparse.Namespace) -> tuple[MultiClusterState, MultiClusterJAXEvaluator]:
    contexts: list[MultiClusterContext] = []
    for cluster in args.cluster_inputs:
        warm_stage = _resolve_warm_stage(cluster, args.warm_stage)
        contexts.append(_build_local_context(cluster, warm_stage, args))
    _validate_compatible_cosmologies(contexts)
    global_specs = _build_global_parameter_layout(contexts)
    state = MultiClusterState(
        run_name=str(args.run_name),
        parameter_specs=global_specs,
        svi_init_values=_global_init_values(global_specs, contexts),
        contexts=contexts,
    )
    evaluator = MultiClusterJAXEvaluator(contexts)
    return state, evaluator


def _initialize_multi_evaluator(
    args: argparse.Namespace,
    state: MultiClusterState,
    evaluator: MultiClusterJAXEvaluator,
) -> np.ndarray:
    midpoint = single._reference_theta_from_init_values(
        state.parameter_specs,
        state.svi_init_values,
        single._default_theta(state.parameter_specs),
    )
    evaluator.refresh_surrogate(midpoint, reason="multi_initial")
    evaluator.refresh_scaling_scatter_cache(midpoint, reason="multi_initial")
    evaluator.refresh_source_metric_cache(midpoint, reason="multi_initial")
    if not bool(getattr(args, "quiet", False)):
        print("[compile] tracing joint likelihood")
    start = time.time()
    compile_loglike = evaluator.source_loglike(midpoint)
    evaluator.timing_totals["initial_jit_compile"] += time.time() - start
    if not bool(getattr(args, "quiet", False)):
        print(f"[compile] joint likelihood ready loglike={compile_loglike:.3f}")
    return midpoint


def _run_joint_inference(
    args: argparse.Namespace,
    state: MultiClusterState,
    evaluator: MultiClusterJAXEvaluator,
    midpoint: np.ndarray,
) -> tuple[np.ndarray, PosteriorResults]:
    sample_model = single._posterior_model(state.parameter_specs, evaluator)
    if args.fit_method == single.FIT_METHOD_SVI:
        best_fit, posterior, _diagnostics = single._run_svi_fit(args, state, evaluator, sample_model)
    else:
        best_fit, svi_posterior, svi_diagnostics = single._run_svi_fit(args, state, evaluator, sample_model)
        svi_chain_seeds = single._prepare_svi_health_for_nuts(
            args,
            state.parameter_specs,
            evaluator,
            best_fit,
            svi_posterior,
            svi_diagnostics,
        )
        nuts_init = single._nuts_initialization_from_svi_center(
            args,
            state.parameter_specs,
            best_fit,
            svi_diagnostics,
            chain_seeds=svi_chain_seeds,
        )
        posterior = single._run_numpyro_nuts_sampler(args, state, evaluator, sample_model, nuts_init)
        usable, reason = single._nuts_posterior_is_usable(posterior)
        log_prob = np.asarray(posterior.log_prob, dtype=float)
        if usable and posterior.samples.size and log_prob.size and np.isfinite(log_prob).any():
            best_fit = np.asarray(posterior.samples[int(np.nanargmax(log_prob))], dtype=float)
        else:
            if not bool(getattr(args, "quiet", False)):
                print(f"[nuts] posterior rejected; using SVI guide posterior reason={reason}")
            posterior = svi_posterior
            posterior.sampler = "svi_fallback_after_failed_nuts"
    best_fit = single._max_likelihood_best_fit_from_posterior(args, evaluator, posterior, best_fit)
    evaluator.refresh_surrogate(best_fit, reason="post_joint_max_likelihood")
    evaluator.refresh_scaling_scatter_cache(best_fit, reason="post_joint_max_likelihood")
    evaluator.refresh_source_metric_cache(best_fit, reason="post_joint_max_likelihood")
    return np.asarray(best_fit if best_fit.size else midpoint, dtype=float), posterior


def _posterior_to_physical(
    posterior: PosteriorResults,
    parameter_specs: list[ParameterSpec],
) -> PosteriorResults:
    return single._posterior_results_to_physical(posterior, parameter_specs)


def _theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    return single._convert_theta_to_physical(theta, parameter_specs)


def _write_json_dataset(group: h5py.Group, name: str, payload: Any) -> None:
    group.create_dataset(name, data=np.bytes_(json.dumps(payload, default=_json_default)))


def _read_json_dataset(group: h5py.Group, name: str) -> Any:
    if name not in group:
        raise ValueError(f"Saved multi-cluster plot bundle is missing {name!r}.")
    raw = group[name][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = np.bytes_(raw).decode("utf-8")
    return json.loads(text)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _write_plot_bundle_h5(
    path: Path,
    args: argparse.Namespace,
    state: MultiClusterState,
    best_fit: np.ndarray,
    posterior: PosteriorResults,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["kind"] = "multi_cluster_joint_cosmology"
        posterior_group = handle.create_group("posterior")
        posterior_group.create_dataset("samples", data=np.asarray(posterior.samples, dtype=float))
        posterior_group.create_dataset("log_prob", data=np.asarray(posterior.log_prob, dtype=float))
        posterior_group.create_dataset("accept_prob", data=np.asarray(posterior.accept_prob, dtype=float))
        posterior_group.create_dataset("diverging", data=np.asarray(posterior.diverging, dtype=bool))
        posterior_group.create_dataset("num_steps", data=np.asarray(posterior.num_steps, dtype=float))
        posterior_group.create_dataset("best_fit", data=np.asarray(best_fit, dtype=float))
        posterior_group.attrs["sampler"] = str(posterior.sampler)
        if posterior.grouped_samples is not None:
            posterior_group.create_dataset("grouped_samples", data=np.asarray(posterior.grouped_samples, dtype=float))
        if posterior.grouped_log_prob is not None:
            posterior_group.create_dataset("grouped_log_prob", data=np.asarray(posterior.grouped_log_prob, dtype=float))
        if posterior.sample_weights is not None:
            posterior_group.create_dataset("sample_weights", data=np.asarray(posterior.sample_weights, dtype=float))
        _write_json_dataset(
            handle,
            "cli_args_json",
            {
                key: value
                for key, value in vars(args).items()
                if key not in {"cluster", "cluster_inputs"}
            }
            | {
                "clusters": [
                    {
                        "key": item.key,
                        "par_path": str(item.par_path),
                        "warm_run_dir": str(item.warm_run_dir),
                    }
                    for item in args.cluster_inputs
                ],
            },
        )
        _write_json_dataset(handle, "init_diagnostics_json", posterior.init_diagnostics or {})
        state_group = handle.create_group("state")
        _write_json_dataset(
            state_group,
            "multi_state_meta_json",
            {
                "run_name": state.run_name,
                "parameter_specs": [spec.__dict__ for spec in state.parameter_specs],
                "clusters": [
                    {
                        "key": context.cluster.key,
                        "par_path": str(context.cluster.par_path),
                        "warm_run_dir": str(context.cluster.warm_run_dir),
                        "warm_stage": context.warm_stage.stage_name,
                        "warm_stage_artifacts_dir": str(context.warm_stage.stage_artifacts_dir),
                        "stage1_artifacts_dir": str(context.warm_stage.stage1_artifacts_dir),
                        "local_parameter_count": len(context.state.parameter_specs),
                        "families": len(context.state.family_data),
                        "images": int(sum(family.n_images for family in context.state.family_data)),
                        "z_lens": float(context.state.z_lens),
                        "cosmo_config": context.state.cosmo_config,
                        "local_to_global_indices": (
                            []
                            if context.local_to_global_indices is None
                            else context.local_to_global_indices.tolist()
                        ),
                    }
                    for context in state.contexts
                ],
            },
        )


def _parameter_spec_from_payload(payload: dict[str, Any]) -> ParameterSpec:
    required = {
        "name",
        "sample_name",
        "potential_id",
        "profile_type",
        "field",
        "prior_kind",
        "lower",
        "upper",
        "step",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Saved multi-cluster plot bundle is missing ParameterSpec field(s): {', '.join(missing)}")
    return ParameterSpec(**payload)


def _load_plot_bundle_for_plots(
    path: Path,
) -> tuple[MultiClusterState, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Saved multi-cluster plot bundle does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("kind", "")) != "multi_cluster_joint_cosmology":
            raise ValueError(f"Unsupported multi-cluster plot bundle kind: {handle.attrs.get('kind')!r}")
        if "posterior" not in handle:
            raise ValueError("Saved multi-cluster plot bundle is missing 'posterior'.")
        posterior_group = handle["posterior"]
        for dataset_name in ("samples", "best_fit"):
            if dataset_name not in posterior_group:
                raise ValueError(f"Saved multi-cluster plot bundle is missing posterior/{dataset_name}.")
        samples = np.asarray(posterior_group["samples"], dtype=float)
        best_fit = np.asarray(posterior_group["best_fit"], dtype=float)
        log_prob = (
            np.asarray(posterior_group["log_prob"], dtype=float)
            if "log_prob" in posterior_group
            else None
        )
        grouped_samples = (
            np.asarray(posterior_group["grouped_samples"], dtype=float)
            if "grouped_samples" in posterior_group
            else None
        )
        if samples.ndim != 2:
            raise ValueError(f"Saved posterior samples must be 2-D; got shape={samples.shape}.")
        if best_fit.ndim != 1:
            raise ValueError(f"Saved best fit must be 1-D; got shape={best_fit.shape}.")
        if grouped_samples is not None and grouped_samples.ndim != 3:
            raise ValueError(f"Saved grouped posterior samples must be 3-D; got shape={grouped_samples.shape}.")
        if "state" not in handle:
            raise ValueError("Saved multi-cluster plot bundle is missing 'state'.")
        meta = _read_json_dataset(handle["state"], "multi_state_meta_json")

    spec_payloads = meta.get("parameter_specs")
    if not isinstance(spec_payloads, list) or not spec_payloads:
        raise ValueError("Saved multi-cluster plot bundle is missing parameter_specs metadata.")
    parameter_specs = [_parameter_spec_from_payload(dict(payload)) for payload in spec_payloads]
    if samples.shape[1] != len(parameter_specs):
        raise ValueError(
            f"Saved posterior samples have {samples.shape[1]} columns but metadata has {len(parameter_specs)} parameters."
        )
    if best_fit.shape[0] != len(parameter_specs):
        raise ValueError(
            f"Saved best fit has {best_fit.shape[0]} values but metadata has {len(parameter_specs)} parameters."
        )
    if grouped_samples is not None and grouped_samples.shape[2] != len(parameter_specs):
        raise ValueError(
            f"Saved grouped posterior samples have {grouped_samples.shape[2]} columns but metadata has "
            f"{len(parameter_specs)} parameters."
        )
    cluster_payloads = meta.get("clusters")
    if not isinstance(cluster_payloads, list) or not cluster_payloads:
        raise ValueError("Saved multi-cluster plot bundle is missing clusters metadata.")
    contexts: list[MultiClusterContext] = []
    for payload in cluster_payloads:
        cluster_key = str(payload.get("key", ""))
        if not cluster_key:
            raise ValueError("Saved multi-cluster plot bundle has a cluster without a key.")
        raw_indices = payload.get("local_to_global_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(
                f"Saved multi-cluster plot bundle is missing local_to_global_indices for cluster {cluster_key!r}."
            )
        indices = np.asarray(raw_indices, dtype=np.int32)
        if np.any(indices < 0) or np.any(indices >= len(parameter_specs)):
            raise ValueError(f"Saved local_to_global_indices for cluster {cluster_key!r} are out of range.")
        contexts.append(
            MultiClusterContext(
                cluster=ClusterInput(
                    key=cluster_key,
                    par_path=Path(str(payload.get("par_path", ""))),
                    warm_run_dir=Path(str(payload.get("warm_run_dir", ""))),
                ),
                warm_stage=WarmStageResolution(
                    stage_name=str(payload.get("warm_stage", "")),
                    stage_artifacts_dir=Path(str(payload.get("warm_stage_artifacts_dir", ""))),
                    stage1_artifacts_dir=Path(str(payload.get("stage1_artifacts_dir", ""))),
                ),
                state=SimpleNamespace(parameter_specs=[parameter_specs[int(idx)] for idx in indices]),
                evaluator=SimpleNamespace(),
                local_to_global_indices=indices,
                local_to_global_indices_jax=jnp.asarray(indices, dtype=jnp.int32),
            )
        )
    state = MultiClusterState(
        run_name=str(meta.get("run_name", "")),
        parameter_specs=parameter_specs,
        svi_init_values=None,
        contexts=contexts,
    )
    return state, samples, best_fit, grouped_samples, log_prob


def _cosmology_parameter_subset(
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    indices = [
        idx
        for idx, spec in enumerate(parameter_specs)
        if spec.sample_name in SHARED_COSMOLOGY_SAMPLE_NAMES
    ]
    return (
        [parameter_specs[idx] for idx in indices],
        np.asarray(samples[:, indices], dtype=float) if indices else np.empty((np.asarray(samples).shape[0], 0)),
        np.asarray(best_fit[indices], dtype=float) if indices else np.empty((0,), dtype=float),
    )


def _cosmology_grouped_subset(
    parameter_specs: list[ParameterSpec],
    grouped_samples: np.ndarray | None,
) -> tuple[list[ParameterSpec], np.ndarray | None]:
    if grouped_samples is None:
        return [], None
    grouped_array = np.asarray(grouped_samples, dtype=float)
    indices = [
        idx
        for idx, spec in enumerate(parameter_specs)
        if spec.sample_name in SHARED_COSMOLOGY_SAMPLE_NAMES
    ]
    if not indices or grouped_array.ndim != 3 or grouped_array.size == 0:
        return [], None
    return [parameter_specs[idx] for idx in indices], np.asarray(grouped_array[:, :, indices], dtype=float)


def _cluster_corner_parameter_subset(
    context: MultiClusterContext,
    parameter_specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
) -> tuple[list[ParameterSpec], np.ndarray, np.ndarray]:
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float)
    selected_indices: list[int] = []
    if context.local_to_global_indices is not None:
        for raw_idx in np.asarray(context.local_to_global_indices, dtype=np.int32).tolist():
            idx = int(raw_idx)
            if idx < 0 or idx >= len(parameter_specs):
                continue
            if getattr(parameter_specs[idx], "component_family", None) == "source_position":
                continue
            selected_indices.append(idx)
    else:
        global_index_by_sample = {spec.sample_name: idx for idx, spec in enumerate(parameter_specs)}
        for local_spec in context.state.parameter_specs:
            if getattr(local_spec, "component_family", None) == "source_position":
                continue
            global_sample_name = _prefixed_sample_name(context.cluster.key, local_spec.sample_name)
            global_index = global_index_by_sample.get(global_sample_name)
            if global_index is None:
                continue
            selected_indices.append(global_index)
    deduped_indices: list[int] = []
    seen_indices: set[int] = set()
    for idx in selected_indices:
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        deduped_indices.append(idx)
    selected_indices = deduped_indices
    subset_specs = [parameter_specs[idx] for idx in selected_indices]
    subset_samples = (
        np.asarray(sample_array[:, selected_indices], dtype=float)
        if selected_indices and sample_array.ndim == 2
        else np.empty((sample_array.shape[0] if sample_array.ndim else 0, 0), dtype=float)
    )
    subset_best = (
        np.asarray(best_fit_array[selected_indices], dtype=float)
        if selected_indices
        else np.empty((0,), dtype=float)
    )
    return subset_specs, subset_samples, subset_best


def _plot_cosmology_prior_histograms(
    plots_dir: Path,
    cosmo_specs: list[ParameterSpec],
    cosmo_samples: np.ndarray,
    cosmo_best: np.ndarray,
) -> None:
    sample_array = np.asarray(cosmo_samples, dtype=float)
    if not cosmo_specs or sample_array.ndim != 2 or sample_array.size == 0:
        return
    import matplotlib.pyplot as plt

    n_params = min(len(cosmo_specs), sample_array.shape[1])
    if n_params == 0:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_params, figsize=(4.6 * n_params, 3.6), squeeze=False)
    for idx, spec in enumerate(cosmo_specs[:n_params]):
        ax = axes[0, idx]
        values = sample_array[:, idx]
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(
                values,
                bins=min(40, max(10, int(np.sqrt(values.size)))),
                density=False,
                color="tab:blue",
                alpha=0.68,
                label="posterior samples",
            )
            ax.axvline(float(np.median(values)), color="tab:orange", linewidth=1.5, label="median")
        lower = float(spec.physical_lower if spec.physical_lower is not None else spec.lower)
        upper = float(spec.physical_upper if spec.physical_upper is not None else spec.upper)
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            prior_density = 1.0 / (upper - lower)
            prior_ax = ax.twinx()
            prior_ax.hlines(
                prior_density,
                lower,
                upper,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="uniform prior density",
            )
            prior_ax.set_ylim(0.0, prior_density * 1.25)
            prior_ax.set_ylabel("prior density")
            if values.size:
                ax.set_xlim(min(lower, float(np.nanmin(values))), max(upper, float(np.nanmax(values))))
            else:
                ax.set_xlim(lower, upper)
        if idx < len(cosmo_best) and np.isfinite(float(cosmo_best[idx])):
            ax.axvline(float(cosmo_best[idx]), color="tab:red", linewidth=1.5, linestyle="--", label="best fit")
        ax.set_xlabel(spec.name)
        ax.set_ylabel("posterior samples")
    legend_by_label: dict[str, object] = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)
    handles = list(legend_by_label.values())
    labels = list(legend_by_label.keys())
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "cosmology_prior_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_multi_cluster_corners(
    plots_dir: Path,
    state: MultiClusterState,
    samples: np.ndarray,
    best_fit: np.ndarray,
    log_prob: np.ndarray | None = None,
) -> None:
    sample_array = np.asarray(samples, dtype=float)
    best_fit_array = np.asarray(best_fit, dtype=float)
    _plot_corner(
        plots_dir,
        sample_array,
        state.parameter_specs,
        best_fit_values=_best_fit_values_for_specs(state.parameter_specs, best_fit_array),
        map_values=_map_values_for_specs(state.parameter_specs, sample_array, log_prob),
        output_name="corner.pdf",
    )
    for context in state.contexts:
        subset_specs, subset_samples, subset_best = _cluster_corner_parameter_subset(
            context,
            state.parameter_specs,
            sample_array,
            best_fit_array,
        )
        if not subset_specs:
            continue
        _plot_corner(
            plots_dir,
            subset_samples,
            subset_specs,
            best_fit_values=_best_fit_values_for_specs(subset_specs, subset_best),
            map_values=_map_values_for_specs(subset_specs, subset_samples, log_prob),
            output_name=f"{context.cluster.key}_cluster_cosmology_corner.pdf",
        )


def _plot_cosmology_trace(
    plots_dir: Path,
    parameter_specs: list[ParameterSpec],
    grouped_samples: np.ndarray | None,
) -> None:
    trace_specs, trace_grouped_samples = _cosmology_grouped_subset(parameter_specs, grouped_samples)
    _plot_trace(
        plots_dir,
        trace_grouped_samples,
        trace_specs,
        output_name="cosmology_trace_plot.pdf",
    )


def _rerender_plots_from_bundle(args: argparse.Namespace, run_dir: Path) -> None:
    state, samples, best_fit, grouped_samples, log_prob = _load_plot_bundle_for_plots(
        run_dir / "artifacts" / "plot_bundle.h5"
    )
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cosmo_specs, cosmo_samples, cosmo_best = _cosmology_parameter_subset(state.parameter_specs, samples, best_fit)
    _plot_cosmology_corner(
        plots_dir,
        cosmo_samples,
        cosmo_specs,
        best_fit_values=_best_fit_values_for_specs(cosmo_specs, cosmo_best),
        map_values=_map_values_for_specs(cosmo_specs, cosmo_samples, log_prob),
    )
    _plot_cosmology_prior_histograms(plots_dir, cosmo_specs, cosmo_samples, cosmo_best)
    _plot_multi_cluster_corners(plots_dir, state, samples, best_fit, log_prob=log_prob)
    _plot_cosmology_trace(plots_dir, state.parameter_specs, grouped_samples)
    _write_fallback_cosmology_plot_if_needed(plots_dir, cosmo_specs, cosmo_samples, cosmo_best)
    if not bool(getattr(args, "quiet", False)):
        print(f"[resume] refreshed joint plots: {run_dir}")


def _write_outputs(
    args: argparse.Namespace,
    state: MultiClusterState,
    evaluator: MultiClusterJAXEvaluator,
    best_fit_physical: np.ndarray,
    posterior_physical: PosteriorResults,
    runtime_sec: float,
) -> Path:
    run_dir = Path(args.output_dir) / str(args.run_name)
    artifacts_dir = run_dir / "artifacts"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_plot_bundle_h5(artifacts_dir / "plot_bundle.h5", args, state, best_fit_physical, posterior_physical)
    parameter_summary = _summary_table(
        state.parameter_specs,
        np.asarray(posterior_physical.samples, dtype=float),
        np.asarray(best_fit_physical, dtype=float),
        posterior_physical.sample_weights,
    )
    parameter_summary.to_csv(tables_dir / "parameter_summary.csv", index=False)
    cosmo_specs, cosmo_samples, cosmo_best = _cosmology_parameter_subset(
        state.parameter_specs,
        np.asarray(posterior_physical.samples, dtype=float),
        np.asarray(best_fit_physical, dtype=float),
    )
    cosmology_summary = _summary_table(cosmo_specs, cosmo_samples, cosmo_best, posterior_physical.sample_weights)
    cosmology_summary.to_csv(tables_dir / "cosmology_summary.csv", index=False)
    cluster_loglikes = evaluator.cluster_loglikes(
        single._convert_theta_to_latent(best_fit_physical, state.parameter_specs)
    )
    cluster_loglike_df = pd.DataFrame(
        [{"cluster_key": key, "source_loglike": value} for key, value in sorted(cluster_loglikes.items())]
    )
    cluster_loglike_df.to_csv(tables_dir / "cluster_loglikes.csv", index=False)
    run_summary = {
        "run_name": args.run_name,
        "n_clusters": len(state.contexts),
        "clusters": [context.cluster.key for context in state.contexts],
        "n_parameters": len(state.parameter_specs),
        "n_cosmology_parameters": len(cosmo_specs),
        "fit_method": args.fit_method,
        "sampler": posterior_physical.sampler,
        "warmup": int(args.warmup),
        "samples": int(args.samples),
        "chains": int(args.chains),
        "runtime_sec": float(runtime_sec),
        "best_source_loglike": float(sum(cluster_loglikes.values())),
        "warm_stages": {context.cluster.key: context.warm_stage.stage_name for context in state.contexts},
        "fixed_H0": float(dict(state.contexts[0].state.cosmo_config).get("H0", 70.0)),
    }
    (tables_dir / "joint_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    if not args.skip_plots:
        _plot_cosmology_corner(
            plots_dir,
            cosmo_samples,
            cosmo_specs,
            best_fit_values=_best_fit_values_for_specs(cosmo_specs, cosmo_best),
            map_values=_map_values_for_specs(cosmo_specs, cosmo_samples, posterior_physical.log_prob),
        )
        _plot_cosmology_prior_histograms(plots_dir, cosmo_specs, cosmo_samples, cosmo_best)
        _plot_multi_cluster_corners(
            plots_dir,
            state,
            np.asarray(posterior_physical.samples, dtype=float),
            np.asarray(best_fit_physical, dtype=float),
            log_prob=posterior_physical.log_prob,
        )
        _plot_cosmology_trace(
            plots_dir,
            state.parameter_specs,
            posterior_physical.grouped_samples,
        )
        _write_fallback_cosmology_plot_if_needed(plots_dir, cosmo_specs, cosmo_samples, cosmo_best)
    return run_dir


def _write_fallback_cosmology_plot_if_needed(
    plots_dir: Path,
    cosmo_specs: list[ParameterSpec],
    cosmo_samples: np.ndarray,
    cosmo_best: np.ndarray,
) -> None:
    output_path = plots_dir / "cosmology_corner.pdf"
    if output_path.exists() or len(cosmo_specs) == 0 or cosmo_samples.size == 0:
        return
    import matplotlib.pyplot as plt

    n_params = len(cosmo_specs)
    fig, axes = plt.subplots(1, n_params, figsize=(4.0 * n_params, 3.2), squeeze=False)
    for idx, spec in enumerate(cosmo_specs):
        ax = axes[0, idx]
        values = np.asarray(cosmo_samples[:, idx], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=min(40, max(10, int(np.sqrt(values.size)))), color="tab:blue", alpha=0.75)
        if idx < len(cosmo_best):
            ax.axvline(float(cosmo_best[idx]), color="tab:red", linestyle="--", linewidth=1.5)
        ax.set_xlabel(spec.name)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.output_dir) / str(args.run_name)
    complete = (run_dir / "artifacts" / "plot_bundle.h5").exists() and (
        run_dir / "tables" / "joint_run_summary.json"
    ).exists()
    if args.resume and complete:
        if args.skip_plots:
            if not args.quiet:
                print(f"[resume] joint run already complete: {run_dir} skip_plots=True")
        else:
            if not args.quiet:
                print(f"[resume] refreshing joint plots: {run_dir}")
            _rerender_plots_from_bundle(args, run_dir)
        return run_dir
    start = time.time()
    state, evaluator = _build_multi_cluster_state(args)
    midpoint = _initialize_multi_evaluator(args, state, evaluator)
    best_fit, posterior = _run_joint_inference(args, state, evaluator, midpoint)
    best_fit_physical = _theta_to_physical(best_fit, state.parameter_specs)
    posterior_physical = _posterior_to_physical(posterior, state.parameter_specs)
    run_dir = _write_outputs(
        args,
        state,
        evaluator,
        best_fit_physical,
        posterior_physical,
        runtime_sec=time.time() - start,
    )
    evaluator.release_runtime_caches()
    if not args.quiet:
        print(f"[done] joint run written to {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()
