import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
import jax
import jax.numpy as jnp
from astropy.cosmology import FlatwCDM

import lenscluster.cluster_solver as cluster_solver
import lenscluster.image_diagnostics as image_diagnostics
import lenscluster.mock_validation.generation as mock_generation
import lenscluster.multi_cluster_solver as multi_cluster_solver
import lenscluster.planning as planning
import lenscluster.plotting as plotting
import lenscluster.mock_validation as mock_validation_api
import lenscluster.mock_validation.runner as validation
from lenscluster.config import ImageDiagnosticsConfig, LensClusterSolverConfig, LikelihoodConfig, PriorConfig, RuntimeConfig, StageScheduleConfig, TruthRecoveryConfig, WorkflowConfig
from lenscluster.utils import log10_dpie_mass_msun
from lenscluster.planning import SolverRuntime
from lenscluster.jax_cosmology import (
    ARCSEC_TO_RAD,
    C_LIGHT_KM_S,
    dpie_sigma0_factor,
    dpie_sigma0_factor_from_lensing_efficiency,
    dpie_sigma0_from_vel_disp,
    flat_wcdm_comoving_distance_mpc,
    flat_wcdm_kpc_per_arcsec,
    flat_wcdm_lens_geometry_factors,
    flat_wcdm_lensing_efficiency,
)
from lenscluster.cluster_solver import (
    IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
    IMAGE_PLANE_MODE_CATASTROPHE_NORMAL_FORM,
    IMAGE_PLANE_MODE_CRITICAL_ARC_MIXTURE,
    IMAGE_PLANE_MODE_FOLD_REGULARIZED_FORWARD_BETA,
    IMAGE_PLANE_MODE_FORWARD_METRIC,
    IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA_BLOCKED,
    IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    IMAGE_PLANE_MODE_NONE,
    SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_CATASTROPHE_NORMAL_FORM_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
    SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    SAMPLE_LIKELIHOOD_SOURCE,
    _adaptive_active_scaling_count,
    _anchored_solved_image_plane_step_from_jacobian,
    _build_cosmology,
    _build_parameter_specs,
    _catastrophe_normal_form_image_plane_bin_loglike,
    _critical_arc_branch_probability,
    _critical_arc_geometry_from_jacobian,
    _critical_arc_lm_step_from_jacobian,
    _critical_arc_lm_geometry_from_jacobian,
    _critical_arc_mixture_image_plane_bin_loglike,
    _critical_arc_projected_quadratics,
    _effective_image_presence_penalty_weight,
    _fold_regularized_image_plane_bin_loglike,
    _forward_metric_image_plane_bin_loglike,
    _forward_metric_image_presence_residual2,
    _jittered_2x2_covariance_det,
    _linearized_image_plane_bin_loglike,
    _linearized_image_plane_residual_from_jacobian,
    _smooth_residual_cap,
    _source_plane_bin_loglike,
    _soft_observed_image_presence_loglike_from_residual2,
    _soft_observed_image_presence_loglike,
    _local_jacobian_bin_loglike,
    _normalize_stage_fit_controls,
    _validation_metrics_summary,
)

from lenscluster.lenstool_parser import _load_arc_constraints_catalog, _split_image_label, load_best_par
from lenscluster.model import (
    ArcConstraintData,
    BuildState,
    ChainSeed,
    GeometryCache,
    EvaluationResult,
    FamilyData,
    FamilyValidationCache,
    PackedLensDetails,
    PackedLensSpec,
    PackedLensState,
    ParameterSpec,
    NUTSInitialization,
    PosteriorResults,
    Stage1PriorSummary,
    SurrogateBinCache,
)
from lenscluster.plotting import (
    _active_scaling_diagnostics_table,
    _corner_without_source_positions,
    _gate_prior_sigmoid_curve,
    _independent_scaling_diagnostics_table,
    _independent_scaling_plot_table,
    _plot_active_scaling_summary,
    _plot_path,
    _run_summary,
)
from lenscluster.mock_validation.runner import (
    PARAMETER_RECOVERY_LOG_ABS_FLOOR,
    SingleBCGMockConfig,
    generate_single_bcg_mock,
    load_chires_family_summary,
    load_chires_table,
    magnification_recovery_table,
    _log10_abs_parameter_values,
    _parameter_truth_with_source_positions,
    parameter_recovery_table,
)
from lenscluster.utils import _rich_log_text, _should_log_to_console, format_stage_banner


def _fixed_potfile_scaling_config(
    *,
    sigma_ref: float = 245.0,
    core_radius_arcsec: float = 0.03,
    cut_radius_arcsec: float = 35.0,
    z_lens: float = 0.396,
) -> dict[str, float]:
    kpc_per_arcsec = float(
        mock_generation.kpc_per_arcsec_from_config(
            float(z_lens),
            mock_generation.flat_wcdm_config(h0=70.0, om0=0.3),
        )
    )
    cut_radius_kpc = float(cut_radius_arcsec) * kpc_per_arcsec
    mass_ref = 10.0 ** float(log10_dpie_mass_msun(float(sigma_ref), cut_radius_kpc))
    return {
        "subhalo_dpie_mass_ref_min_msun": mass_ref,
        "subhalo_dpie_mass_ref_max_msun": mass_ref * (1.0 + 1.0e-8),
        "subhalo_dpie_mass_pivot_msun": mass_ref,
        "subhalo_core_radius_pivot_kpc": float(core_radius_arcsec) * kpc_per_arcsec,
        "subhalo_core_radius_mass_slope": 0.0,
        "subhalo_core_radius_scatter_dex": 0.0,
        "subhalo_cut_radius_pivot_kpc": cut_radius_kpc,
        "subhalo_cut_radius_mass_slope": 0.0,
        "subhalo_cut_radius_scatter_dex": 0.0,
        "subhalo_alpha_sigma_half_width": 0.0,
        "subhalo_gamma_ml_half_width": 0.0,
    }


def _fixed_large_component_mass_config() -> dict[str, float]:
    total_mass = 7.6035e14
    bcg_mass = 1.6639e13
    return {
        **_fixed_potfile_scaling_config(),
        "total_large_component_mass_min_msun": total_mass,
        "total_large_component_mass_max_msun": total_mass * (1.0 + 1.0e-8),
        "large_component_mass_pivot_msun": total_mass - bcg_mass,
        "halo_core_radius_pivot_kpc": 27.0,
        "halo_core_radius_mass_slope": 0.0,
        "halo_core_radius_scatter_dex": 0.0,
        "halo_cut_radius_pivot_kpc": 1175.0,
        "halo_cut_radius_mass_slope": 0.0,
        "halo_cut_radius_scatter_dex": 0.0,
        "bcg_mass_fraction_median": bcg_mass / total_mass,
        "bcg_mass_fraction_scatter_dex": 0.0,
        "bcg_mass_fraction_min": 0.02,
        "bcg_mass_fraction_max": 0.024,
        "bcg_core_radius_pivot_kpc": 0.8,
        "bcg_core_radius_mass_slope": 0.0,
        "bcg_core_radius_scatter_dex": 0.0,
        "bcg_cut_radius_pivot_kpc": 187.0,
        "bcg_mass_pivot_msun": bcg_mass,
        "bcg_cut_radius_mass_slope": 0.0,
        "bcg_cut_radius_scatter_dex": 0.0,
    }


def _locator_max_tick_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "Locator attempting to generate" in record.getMessage()
    ]


def _fresh_process_solver_runtime_echo_worker(
    result_queue: Any,
    args: SolverRuntime,
    fit_mode: str,
    run_name: str,
    *_worker_args: Any,
) -> None:
    try:
        if args.seed != 12345:
            raise AssertionError(f"unexpected seed={args.seed!r}")
        if fit_mode != cluster_solver.STAGE2_FREE_SOURCE_FORWARD_FIT_DIR:
            raise AssertionError(f"unexpected fit_mode={fit_mode!r}")
        result_queue.put({"status": "ok", "run_dir": str(Path(args.output_dir) / run_name)})
    except BaseException:
        result_queue.put({"status": "error", "traceback": "worker failed to read SolverRuntime"})


def _install_recording_progress(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    instances: list[Any] = []

    class RecordingProgress:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.events: list[tuple[Any, ...]] = []
            self._next_task_id = 1
            instances.append(self)

        def __enter__(self):
            self.events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.events.append(("exit", exc_type))
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            task_id = self._next_task_id
            self._next_task_id += 1
            self.events.append(("add_task", task_id, description, total))
            return task_id

        def update(self, task_id: int, **kwargs: object) -> None:
            self.events.append(("update", task_id, kwargs))

        def advance(self, task_id: int, advance: int = 1) -> None:
            self.events.append(("advance", task_id, advance))

    def recording_progress_context(*args: object, **kwargs: object) -> RecordingProgress:
        return RecordingProgress(*args, **kwargs)

    monkeypatch.setattr(validation, "_progress_context", recording_progress_context)
    return instances


def test_load_chires_table_parses_image_and_summary_rows(tmp_path: Path) -> None:
    path = tmp_path / "chires.dat"
    path.write_text(
        "\n".join(
            [
                "chi multiples",
                " N    ID    z   Narcs    chip    chix    chiy    chia   rmss     rmsi    dx      dy    nwarn",
                " 6    13c 1.005   1     21.67    0.00    0.00    0.00   0.343    0.00    0.12   -0.32  1",
                " 6     13 1.005   3     43.20    0.00    0.00    0.00   0.279    0.00    N/A     N/A   3",
            ]
        ),
        encoding="utf-8",
    )

    table = load_chires_table(path)
    summary = load_chires_family_summary(path)

    assert table.shape[0] == 2
    assert summary.shape[0] == 1
    assert summary.loc[0, "family_id"] == "13"
    assert summary.loc[0, "n_arcs"] == 3
    assert summary.loc[0, "source_rms_arcsec"] == 0.279
    assert pd.isna(summary.loc[0, "dx_arcsec"])


def test_stage_banner_formatter_renders_delimiter_and_details() -> None:
    lines = format_stage_banner(
        "STAGE 2: stage2_free_source_forward_fit",
        "fit_method=svi run_name=fit/stage2_free_source_forward_fit",
    )

    assert lines[0].startswith("[stage] ====")
    assert "STAGE 2: stage2_free_source_forward_fit" in lines[0]
    assert lines[0].endswith("====")
    assert lines[1] == "[stage] fit_method=svi run_name=fit/stage2_free_source_forward_fit"


def test_stage_banner_rich_rendering_styles_delimiter_and_title() -> None:
    banner = format_stage_banner("STAGE 2: stage2_free_source_forward_fit")[0]
    rendered = _rich_log_text("2026-05-15T12:00:00", banner)
    styles = [str(span.style) for span in rendered.spans]

    assert rendered.plain == f"2026-05-15T12:00:00 {banner}"
    assert "bold white on magenta" in styles
    assert styles.count("bold magenta") >= 3


def test_stage_start_rich_rendering_keeps_normal_stage_style() -> None:
    rendered = _rich_log_text("2026-05-15T12:00:00", "[stage] start run_name=fit/stage2_free_source_forward_fit")
    styles = [str(span.style) for span in rendered.spans]

    assert rendered.plain == "2026-05-15T12:00:00 [stage] start run_name=fit/stage2_free_source_forward_fit"
    assert "bold white on magenta" not in styles
    assert styles.count("bold magenta") == 1


def test_smc_progress_logs_are_console_visible() -> None:
    rendered = _rich_log_text("2026-05-15T12:00:00", "[smc] step=1 temperature=0.1")
    styles = [str(span.style) for span in rendered.spans]

    assert _should_log_to_console("[smc] step=1 temperature=0.1")
    assert rendered.plain == "2026-05-15T12:00:00 [smc] step=1 temperature=0.1"
    assert "bold yellow" in styles


def test_stage_banner_title_falls_back_to_run_name() -> None:
    assert cluster_solver._stage_banner_title_from_run_name("custom_run") == "custom_run"


def test_generate_single_bcg_mock_parses_and_has_finite_magnifications(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        **_fixed_large_component_mass_config(),
        seed=7,
        n_primary_families=2,
        pos_sigma_arcsec=0.0,
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
        caustic_compute_window_arcsec=40.0,
        caustic_grid_scale_arcsec=2.0,
        mock_image_candidate_batch_size=8,
    )

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    _parsed, _potentials_df, images_df, _arcs_df, potentials_with_priors = load_best_par(paths.par_path)

    assert paths.par_path.exists()
    assert paths.image_catalog_path.exists()
    arc_rows = [
        line for line in paths.image_catalog_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    arc_labels = [row.split()[0] for row in arc_rows]
    assert arc_rows
    assert all(not row.startswith((" ", "	")) for row in arc_rows)
    assert all(
        label.split(".", 1)[0].isdigit()
        and label.split(".", 1)[1].isalpha()
        and label.split(".", 1)[1].islower()
        for label in arc_labels
    )
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 2
    assert sorted(images_df.groupby("family_id")["catalog_z"].first().round(3).tolist()) == [1.5, 2.0]
    assert np.isfinite(images_df["catalog_mag"].to_numpy(dtype=float)).all()
    assert np.unique(np.round(images_df["catalog_mag"].to_numpy(dtype=float), 8)).size > 1
    assert np.isfinite(images_df["catalog_mag_err"].to_numpy(dtype=float)).all()
    np.testing.assert_allclose(
        images_df["catalog_mag_err"].to_numpy(dtype=float),
        np.full(len(images_df), config.image_magnitude_sigma_mag),
    )
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert all(int(source["n_images"]) >= 3 for source in truth["sources"])
    assert all(
        config.source_magnitude_min <= float(source["source_magnitude_true"]) <= config.source_magnitude_max
        for source in truth["sources"]
    )
    assert truth["config"]["primary_source_redshifts"] == [1.5, 2.0, 3.0]
    assert truth["config"]["subhalo_source_redshifts"] == [1.5, 2.0, 3.0]
    assert truth["config"]["source_magnitude_min"] == pytest.approx(config.source_magnitude_min)
    assert truth["config"]["source_magnitude_max"] == pytest.approx(config.source_magnitude_max)
    assert truth["config"]["image_magnitude_sigma_mag"] == pytest.approx(config.image_magnitude_sigma_mag)
    assert truth["config"]["image_magnitude_mu_floor"] == pytest.approx(config.image_magnitude_mu_floor)
    assert "source_redshifts" not in truth["config"]
    for component_id in ("halo", "bcg"):
        component_config = truth["config"][component_id]
        assert truth["parameter_truth"][f"{component_id}.x_centre"] == pytest.approx(component_config["x_centre"])
        assert truth["parameter_truth"][f"{component_id}.y_centre"] == pytest.approx(component_config["y_centre"])
        assert truth["parameter_truth"][f"{component_id}.v_disp"] == pytest.approx(component_config["v_disp"])
        assert truth["parameter_truth"][f"{component_id}.core_radius_kpc"] == pytest.approx(
            component_config["core_radius_kpc"]
        )
        assert truth["parameter_truth"][f"{component_id}.cut_radius_kpc"] == pytest.approx(
            component_config["cut_radius_kpc"]
        )
        component_record = next(
            component for component in truth["lens_components"] if component["component_id"] == component_id
        )
        kwargs_record = component_record["kwargs_by_source_redshift"][f"{config.primary_source_redshifts[0]:.8f}"]
        assert kwargs_record["center_x"] == pytest.approx(component_config["x_centre"])
        assert kwargs_record["center_y"] == pytest.approx(component_config["y_centre"])
        assert kwargs_record["Ra"] == pytest.approx(component_config["core_radius_arcsec"])
        assert kwargs_record["Rs"] == pytest.approx(component_config["cut_radius_arcsec"])
    assert 5.0e14 <= truth["config"]["sampled_total_large_component_mass_msun"] <= 2.0e15
    assert truth["parameter_truth"]["large_components.total_mass_msun"] == pytest.approx(
        truth["config"]["sampled_total_large_component_mass_msun"]
    )
    assert truth["parameter_truth"]["large_components.halo_mass_msun"] == pytest.approx(
        truth["config"]["sampled_halo_mass_msun"]
    )
    assert truth["parameter_truth"]["large_components.bcg_mass_msun"] == pytest.approx(
        truth["config"]["sampled_bcg_mass_msun"]
    )
    assert truth["parameter_truth"]["large_components.bcg_mass_fraction"] == pytest.approx(
        truth["config"]["sampled_bcg_mass_fraction"]
    )
    assert truth["config"]["sampled_halo_mass_msun"] + truth["config"]["sampled_bcg_mass_msun"] == pytest.approx(
        truth["config"]["sampled_total_large_component_mass_msun"]
    )
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()
    for column in (
        "source_magnitude_true",
        "image_magnitude_true",
        "image_magnitude_obs",
        "image_magnitude_err",
    ):
        assert column in images
        assert np.isfinite(images[column].to_numpy(dtype=float)).all()
    expected_image_magnitude_true = images["source_magnitude_true"].to_numpy(dtype=float) - 2.5 * np.log10(
        np.abs(images["magnification_true_at_obs"].to_numpy(dtype=float)) + config.image_magnitude_mu_floor
    )
    np.testing.assert_allclose(
        images["image_magnitude_true"].to_numpy(dtype=float),
        expected_image_magnitude_true,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    magnitude_residual = images["image_magnitude_obs"].to_numpy(dtype=float) - images[
        "image_magnitude_true"
    ].to_numpy(dtype=float)
    assert np.isfinite(magnitude_residual).all()
    assert np.max(np.abs(magnitude_residual)) < 6.0 * config.image_magnitude_sigma_mag
    np.testing.assert_allclose(
        images["image_magnitude_err"].to_numpy(dtype=float),
        np.full(len(images), config.image_magnitude_sigma_mag),
    )
    image_truth_records = truth["images"]
    assert all("source_magnitude_true" in row for row in image_truth_records)
    assert all("image_magnitude_true" in row for row in image_truth_records)
    assert all("image_magnitude_obs" in row for row in image_truth_records)
    assert all("image_magnitude_err" in row for row in image_truth_records)
    assert set(truth["parameter_truth"]) >= {"halo.v_disp", "bcg.v_disp"}
    # source.sigma_int has no truth entry: the mock never injects intrinsic source-plane
    # scatter, so recording one produced a spurious -1 fractional bias in recovery tables.
    assert "source.sigma_int" not in truth["parameter_truth"]
    first_source = truth["sources"][0]
    assert truth["parameter_truth"]["source.1.beta_x"] == pytest.approx(first_source["beta_x"])
    assert truth["parameter_truth"]["source.1.beta_y"] == pytest.approx(first_source["beta_y"])
    first_z_key = sorted(truth["caustics_by_source_redshift"])[0]
    first_caustic = truth["caustics_by_source_redshift"][first_z_key][0]
    for key in ("critical_x", "critical_y", "caustic_beta_x", "caustic_beta_y"):
        values = np.asarray(first_caustic[key], dtype=float)
        assert values.ndim == 1
        assert values.size >= 3
        assert np.isfinite(values).all()


def test_write_single_bcg_mock_par_can_tighten_bcg_position_prior(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(bcg_position_prior_half_width_arcsec=2.0)
    path = tmp_path / "single_bcg_mock.par"
    images = pd.DataFrame({"x_obs_arcsec": [0.0], "y_obs_arcsec": [0.0]})

    mock_generation._write_single_bcg_par(path, config, images)
    par_lines = path.read_text(encoding="utf-8").splitlines()

    def position_bounds(component_id: str, field_name: str) -> tuple[float, float]:
        in_limit_block = False
        for line in par_lines:
            stripped = line.strip()
            if stripped == f"limit {component_id}":
                in_limit_block = True
                continue
            if in_limit_block and stripped == "end":
                break
            if in_limit_block and stripped.startswith(field_name):
                parts = stripped.split()
                return float(parts[2]), float(parts[3])
        raise AssertionError(f"missing {field_name} limit for {component_id}")

    assert position_bounds("halo", "x_centre") == pytest.approx((-8.0, 8.0))
    assert position_bounds("halo", "y_centre") == pytest.approx((-8.0, 8.0))
    assert position_bounds("bcg", "x_centre") == pytest.approx((-1.65, 2.35))
    assert position_bounds("bcg", "y_centre") == pytest.approx((-2.22, 1.78))


def test_generate_single_bcg_mock_respects_max_images_per_family(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        **_fixed_large_component_mass_config(),
        seed=7,
        n_primary_families=1,
        min_images_per_family=2,
        max_images_per_family=3,
        pos_sigma_arcsec=0.0,
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
        caustic_compute_window_arcsec=40.0,
        caustic_grid_scale_arcsec=2.0,
        mock_image_candidate_batch_size=8,
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    family_sizes = images.groupby("family_id").size()
    assert (family_sizes >= config.min_images_per_family).all()
    assert (family_sizes <= config.max_images_per_family).all()
    assert all(
        config.min_images_per_family <= int(source["n_images"]) <= int(config.max_images_per_family)
        for source in truth["sources"]
    )
    assert truth["config"]["max_images_per_family"] == config.max_images_per_family
    restored_config = validation._caustic_config_from_truth(truth)
    assert restored_config.max_images_per_family == config.max_images_per_family


def test_caustic_config_from_truth_restores_mock_image_grid_settings() -> None:
    restored_config = validation._caustic_config_from_truth(
        {
            "config": {
                "primary_seed_grid_pixel_arcsec": 2.5,
                "subhalo_seed_grid_pixel_arcsec": 0.75,
                "mock_image_search_window_arcsec": 123.0,
            }
        }
    )

    assert restored_config.primary_seed_grid_pixel_arcsec == pytest.approx(2.5)
    assert restored_config.subhalo_seed_grid_pixel_arcsec == pytest.approx(0.75)
    assert restored_config.mock_image_search_window_arcsec == pytest.approx(123.0)


def test_caustic_config_from_truth_reads_legacy_min_distance_keys() -> None:
    # truth.json files written before the seed-grid rename carry the old key names.
    restored_config = validation._caustic_config_from_truth(
        {
            "config": {
                "primary_image_min_distance_arcsec": 2.0,
                "subhalo_image_min_distance_arcsec": 1.0,
            }
        }
    )

    assert restored_config.primary_seed_grid_pixel_arcsec == pytest.approx(2.0)
    assert restored_config.subhalo_seed_grid_pixel_arcsec == pytest.approx(1.0)



def _jax_mock_test_config(**updates: Any) -> SingleBCGMockConfig:
    values = dict(
        **_fixed_large_component_mass_config(),
        seed=7,
        n_primary_families=1,
        n_subhalo_families=0,
        primary_source_redshifts=(2.0,),
        subhalo_source_redshifts=(2.0,),
        n_subhalos=0,
        pos_sigma_arcsec=0.0,
        max_sources_to_try=60,
        mock_image_candidate_batch_size=8,
        mock_image_seed_cap=16,
        caustic_compute_window_arcsec=40.0,
        caustic_grid_scale_arcsec=2.0,
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
    )
    values.update(updates)
    return SingleBCGMockConfig(**values)


def test_single_bcg_mock_realizes_large_halo_positions_from_seed() -> None:
    config = SingleBCGMockConfig(
        seed=31,
        n_primary_families=1,
        halo_position_half_width_arcsec=10.0,
        bcg_position_offset_radius_arcsec=2.0,
    )
    rng = np.random.default_rng(config.seed)

    realized = mock_generation._realize_large_halo_positions(config, rng)
    repeated = mock_generation._realize_large_halo_positions(config, np.random.default_rng(config.seed))
    changed = mock_generation._realize_large_halo_positions(
        replace(config, seed=config.seed + 1),
        np.random.default_rng(config.seed + 1),
    )

    assert -10.0 <= realized.halo.x_centre <= 10.0
    assert -10.0 <= realized.halo.y_centre <= 10.0
    assert math.hypot(
        realized.bcg.x_centre - realized.halo.x_centre,
        realized.bcg.y_centre - realized.halo.y_centre,
    ) <= 2.0
    assert repeated.halo == realized.halo
    assert repeated.bcg == realized.bcg
    assert (changed.halo.x_centre, changed.halo.y_centre, changed.bcg.x_centre, changed.bcg.y_centre) != (
        realized.halo.x_centre,
        realized.halo.y_centre,
        realized.bcg.x_centre,
        realized.bcg.y_centre,
    )


def test_single_bcg_mock_realized_positions_can_be_fixed() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
    )

    realized = mock_generation._realize_large_halo_positions(config, np.random.default_rng(4))

    assert realized.halo.x_centre == pytest.approx(config.halo.x_centre)
    assert realized.halo.y_centre == pytest.approx(config.halo.y_centre)
    assert realized.bcg.x_centre == pytest.approx(config.halo.x_centre)
    assert realized.bcg.y_centre == pytest.approx(config.halo.y_centre)


def test_single_bcg_mock_realizes_large_halo_masses_from_seed() -> None:
    config = SingleBCGMockConfig(
        seed=37,
        n_primary_families=1,
        total_large_component_mass_min_msun=5.0e14,
        total_large_component_mass_max_msun=2.0e15,
        halo_core_radius_scatter_dex=0.0,
        bcg_core_radius_scatter_dex=0.0,
    )
    kpc_per_arcsec = 5.34

    realized = mock_generation._realize_large_halo_masses(config, np.random.default_rng(config.seed), kpc_per_arcsec)
    repeated = mock_generation._realize_large_halo_masses(
        config,
        np.random.default_rng(config.seed),
        kpc_per_arcsec,
    )
    changed = mock_generation._realize_large_halo_masses(
        replace(config, seed=config.seed + 1),
        np.random.default_rng(config.seed + 1),
        kpc_per_arcsec,
    )

    assert 5.0e14 <= float(realized.sampled_total_large_component_mass_msun) <= 2.0e15
    assert float(realized.sampled_halo_mass_msun) + float(realized.sampled_bcg_mass_msun) == pytest.approx(
        float(realized.sampled_total_large_component_mass_msun)
    )
    assert config.bcg_mass_fraction_min <= float(realized.sampled_bcg_mass_fraction) <= config.bcg_mass_fraction_max
    halo_core_kpc = float(realized.halo.core_radius_arcsec) * kpc_per_arcsec
    halo_cut_kpc = float(realized.halo.cut_radius_arcsec) * kpc_per_arcsec
    bcg_core_kpc = float(realized.bcg.core_radius_arcsec) * kpc_per_arcsec
    bcg_cut_kpc = float(realized.bcg.cut_radius_arcsec) * kpc_per_arcsec
    assert halo_core_kpc == pytest.approx(
        config.halo_core_radius_pivot_kpc
        * (float(realized.sampled_halo_mass_msun) / config.large_component_mass_pivot_msun)
        ** config.halo_core_radius_mass_slope
    )
    assert bcg_core_kpc == pytest.approx(
        config.bcg_core_radius_pivot_kpc
        * (float(realized.sampled_bcg_mass_msun) / config.bcg_mass_pivot_msun)
        ** config.bcg_core_radius_mass_slope
    )
    assert 10.0 ** float(log10_dpie_mass_msun(realized.halo.v_disp, halo_cut_kpc)) == pytest.approx(
        float(realized.sampled_halo_mass_msun)
    )
    assert 10.0 ** float(log10_dpie_mass_msun(realized.bcg.v_disp, bcg_cut_kpc)) == pytest.approx(
        float(realized.sampled_bcg_mass_msun)
    )
    assert repeated.halo == realized.halo
    assert repeated.bcg == realized.bcg
    assert repeated.sampled_total_large_component_mass_msun == pytest.approx(
        float(realized.sampled_total_large_component_mass_msun)
    )
    assert (
        changed.halo.v_disp,
        changed.halo.core_radius_arcsec,
        changed.halo.cut_radius_arcsec,
        changed.bcg.v_disp,
        changed.bcg.core_radius_arcsec,
        changed.bcg.cut_radius_arcsec,
    ) != (
        realized.halo.v_disp,
        realized.halo.core_radius_arcsec,
        realized.halo.cut_radius_arcsec,
        realized.bcg.v_disp,
        realized.bcg.core_radius_arcsec,
        realized.bcg.cut_radius_arcsec,
    )


def test_single_bcg_mock_realizes_potfile_scaling_from_seed() -> None:
    assert SingleBCGMockConfig().subhalo_dpie_mass_pivot_msun == pytest.approx(1.0e12)
    assert SingleBCGMockConfig().subhalo_cut_radius_pivot_kpc == pytest.approx(15.0)
    assert SingleBCGMockConfig().subhalo_cut_radius_mass_slope == pytest.approx(0.50)
    assert SingleBCGMockConfig().subhalo_cut_radius_prior_fractional_half_width == pytest.approx(0.25)
    config = SingleBCGMockConfig(
        seed=41,
        n_primary_families=1,
        n_subhalos=4,
        subhalo_dpie_mass_ref_min_msun=1.0e11,
        subhalo_dpie_mass_ref_max_msun=3.0e12,
        subhalo_core_radius_scatter_dex=0.0,
        subhalo_cut_radius_scatter_dex=0.0,
    )
    kpc_per_arcsec = 5.34

    realized = mock_generation._realize_potfile_scaling(config, np.random.default_rng(config.seed), kpc_per_arcsec)
    repeated = mock_generation._realize_potfile_scaling(
        config,
        np.random.default_rng(config.seed),
        kpc_per_arcsec,
    )
    changed = mock_generation._realize_potfile_scaling(
        replace(config, seed=config.seed + 1),
        np.random.default_rng(config.seed + 1),
        kpc_per_arcsec,
    )

    assert 1.0e11 <= float(realized.sampled_subhalo_dpie_mass_ref_msun) <= 3.0e12
    core_kpc = float(realized.subhalo_core_radius_arcsec) * kpc_per_arcsec
    cut_kpc = float(realized.subhalo_cut_radius_arcsec) * kpc_per_arcsec
    assert mock_generation.MOCK_POTFILE_CORE_PRIOR_LOWER_KPC <= core_kpc <= mock_generation.MOCK_POTFILE_CORE_PRIOR_UPPER_KPC
    assert core_kpc == pytest.approx(
        config.subhalo_core_radius_pivot_kpc
        * (float(realized.sampled_subhalo_dpie_mass_ref_msun) / config.subhalo_dpie_mass_pivot_msun)
        ** config.subhalo_core_radius_mass_slope
    )
    assert cut_kpc == pytest.approx(
        config.subhalo_cut_radius_pivot_kpc
        * (float(realized.sampled_subhalo_dpie_mass_ref_msun) / config.subhalo_dpie_mass_pivot_msun)
        ** config.subhalo_cut_radius_mass_slope
    )
    assert 10.0 ** float(log10_dpie_mass_msun(realized.subhalo_sigma_ref, cut_kpc)) == pytest.approx(
        float(realized.sampled_subhalo_dpie_mass_ref_msun)
    )
    assert realized.subhalo_mass_ref == pytest.approx(float(realized.sampled_subhalo_dpie_mass_ref_msun))
    assert repeated.subhalo_sigma_ref == pytest.approx(realized.subhalo_sigma_ref)
    assert repeated.subhalo_core_radius_arcsec == pytest.approx(realized.subhalo_core_radius_arcsec)
    assert repeated.subhalo_cut_radius_arcsec == pytest.approx(realized.subhalo_cut_radius_arcsec)
    assert changed.subhalo_core_radius_arcsec != pytest.approx(realized.subhalo_core_radius_arcsec)
    assert changed.sampled_subhalo_dpie_mass_ref_msun != pytest.approx(
        float(realized.sampled_subhalo_dpie_mass_ref_msun)
    )


def test_single_bcg_mock_potfile_scaling_pivot_matches_constant_ml_anchor() -> None:
    config = SingleBCGMockConfig(
        seed=42,
        n_primary_families=1,
        n_subhalos=4,
        subhalo_dpie_mass_ref_min_msun=1.0e12,
        subhalo_dpie_mass_ref_max_msun=1.0e12 * (1.0 + 1.0e-8),
        subhalo_core_radius_scatter_dex=0.0,
        subhalo_cut_radius_scatter_dex=0.0,
    )
    kpc_per_arcsec = 5.34

    realized = mock_generation._realize_potfile_scaling(config, np.random.default_rng(config.seed), kpc_per_arcsec)
    cut_kpc = float(realized.subhalo_cut_radius_arcsec) * kpc_per_arcsec

    assert cut_kpc == pytest.approx(15.0)
    assert realized.subhalo_sigma_ref == pytest.approx(245.0, rel=0.02)
    assert 10.0 ** float(log10_dpie_mass_msun(realized.subhalo_sigma_ref, cut_kpc)) == pytest.approx(
        float(realized.sampled_subhalo_dpie_mass_ref_msun)
    )


def test_single_bcg_mock_potfile_scaling_endpoint_sanity() -> None:
    kpc_per_arcsec = 5.34
    for mass_ref in (1.0e11, 3.0e12):
        config = SingleBCGMockConfig(
            seed=42,
            n_primary_families=1,
            n_subhalos=4,
            subhalo_dpie_mass_ref_min_msun=mass_ref,
            subhalo_dpie_mass_ref_max_msun=mass_ref * (1.0 + 1.0e-8),
            subhalo_core_radius_scatter_dex=0.0,
            subhalo_cut_radius_scatter_dex=0.0,
        )
        realized = mock_generation._realize_potfile_scaling(config, np.random.default_rng(config.seed), kpc_per_arcsec)
        cut_kpc = float(realized.subhalo_cut_radius_arcsec) * kpc_per_arcsec

        assert 4.7 <= cut_kpc <= 26.0
        assert 138.0 <= float(realized.subhalo_sigma_ref) <= 325.0


def test_realized_potfile_truth_support_guard_rejects_invalid_core() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        n_subhalos=1,
        subhalo_core_radius_arcsec=0.001,
        subhalo_cut_radius_arcsec=3.0,
        subhalo_sigma_ref=245.0,
    )

    with pytest.raises(ValueError, match="potfile.corekpc truth"):
        mock_generation._validate_realized_potfile_truth_support(config, kpc_per_arcsec=5.0)


def test_member_einstein_radius_diagnostic_flags_subdetectable_population() -> None:
    config = SingleBCGMockConfig(n_primary_families=1, n_subhalos=1, pos_sigma_arcsec=10.0)
    component = mock_generation.DPIETruth(
        potential_id="member001",
        x_centre=0.0,
        y_centre=0.0,
        ellipticite=0.0,
        angle_pos=0.0,
        core_radius_arcsec=0.1,
        cut_radius_arcsec=3.0,
        v_disp=10.0,
    )

    diagnostic = mock_generation._member_einstein_radius_diagnostic(
        config,
        [component],
        (2.0,),
        mock_generation.flat_wcdm_config(h0=70.0, om0=0.3),
    )

    assert diagnostic["max_member_theta_e_arcsec"] < diagnostic["pos_sigma_arcsec"]
    assert diagnostic["max_member_theta_e_over_pos_sigma"] < 1.0


def test_single_bcg_mock_realizes_potfile_exponents_from_seed() -> None:
    config = SingleBCGMockConfig(
        seed=43,
        n_primary_families=1,
        n_subhalos=4,
        subhalo_alpha_sigma=0.25,
        subhalo_alpha_sigma_half_width=0.05,
        subhalo_gamma_ml=0.0,
        subhalo_gamma_ml_half_width=0.10,
    )

    realized = mock_generation._realize_potfile_exponents(config, np.random.default_rng(config.seed))
    repeated = mock_generation._realize_potfile_exponents(config, np.random.default_rng(config.seed))
    changed = mock_generation._realize_potfile_exponents(
        replace(config, seed=config.seed + 1),
        np.random.default_rng(config.seed + 1),
    )

    assert 0.20 <= realized.subhalo_alpha_sigma <= 0.30
    assert -0.10 <= realized.subhalo_gamma_ml <= 0.10
    assert realized.sampled_subhalo_alpha_sigma == pytest.approx(realized.subhalo_alpha_sigma)
    assert realized.sampled_subhalo_gamma_ml == pytest.approx(realized.subhalo_gamma_ml)
    assert mock_generation._subhalo_beta_radius_from_config(realized) == pytest.approx(
        1.0 + realized.subhalo_gamma_ml - 2.0 * realized.subhalo_alpha_sigma
    )
    assert mock_generation._subhalo_mass_luminosity_exponent(realized) == pytest.approx(1.0 + realized.subhalo_gamma_ml)
    assert repeated.subhalo_alpha_sigma == pytest.approx(realized.subhalo_alpha_sigma)
    assert repeated.subhalo_gamma_ml == pytest.approx(realized.subhalo_gamma_ml)
    assert (
        changed.subhalo_alpha_sigma,
        changed.subhalo_gamma_ml,
    ) != pytest.approx((realized.subhalo_alpha_sigma, realized.subhalo_gamma_ml))


def test_single_bcg_mock_fixed_potfile_exponents_do_not_advance_rng() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        n_subhalos=4,
        subhalo_alpha_sigma=0.24,
        subhalo_alpha_sigma_half_width=0.0,
        subhalo_gamma_ml=0.02,
        subhalo_gamma_ml_half_width=0.0,
    )
    rng = np.random.default_rng(51)
    expected_next = np.random.default_rng(51).uniform()

    realized = mock_generation._realize_potfile_exponents(config, rng)

    assert realized.subhalo_alpha_sigma == pytest.approx(0.24)
    assert realized.subhalo_gamma_ml == pytest.approx(0.02)
    assert realized.sampled_subhalo_alpha_sigma == pytest.approx(0.24)
    assert realized.sampled_subhalo_gamma_ml == pytest.approx(0.02)
    assert rng.uniform() == pytest.approx(expected_next)


def test_generate_single_bcg_mock_emits_progress_events(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    config = _jax_mock_test_config()

    _paths, images, truth = generate_single_bcg_mock(
        tmp_path,
        config,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    event_names = [event for event, _payload in events]
    assert event_names[:2] == ["subhalos_start", "subhalos_complete"]
    assert "redshift_start" in event_names
    assert "redshift_complete" in event_names
    assert "family_attempt" in event_names
    assert "family_accept" in event_names
    assert event_names[-1] == "outputs_complete"
    accepted_payload = next(payload for event, payload in events if event == "family_accept")
    assert accepted_payload["family_index"] == 1
    assert accepted_payload["image_count"] >= config.min_images_per_family
    assert accepted_payload["image_candidate_batch_size"] == config.mock_image_candidate_batch_size
    assert accepted_payload["mock_image_seed_cap"] == config.mock_image_seed_cap
    assert accepted_payload["mock_image_search_window_arcsec"] == pytest.approx(config.mock_image_search_window_arcsec)
    queue_payload = next(payload for event, payload in events if event == "family_queue_start")
    assert queue_payload["image_solver_workers"] == min(config.mock_generation_workers, config.n_families)
    mapped_grid_payload = next(payload for event, payload in events if event == "mapped_grid_complete")
    assert mapped_grid_payload["mock_image_search_window_arcsec"] == pytest.approx(config.mock_image_search_window_arcsec)
    redshift_payload = next(payload for event, payload in events if event == "redshift_start")
    assert redshift_payload["caustic_grid_chunk_memory_gb"] == pytest.approx(config.mock_caustic_grid_chunk_memory_gb)
    assert redshift_payload["caustic_grid_chunk_points"] > 0
    assert redshift_payload["caustic_grid_chunk_count"] > 0
    assert images["family_id"].nunique() == 1
    assert truth["sources"][0]["n_images"] == len(images)


def test_mock_family_source_redshifts_excludes_unused_default_source_redshift() -> None:
    config = _jax_mock_test_config(
        source_redshift=7.0,
        n_primary_families=2,
        n_subhalo_families=0,
        primary_source_redshifts=(2.0, 3.0),
    )

    _primary_family_redshifts, _subhalo_family_redshifts, needed = mock_generation._mock_family_source_redshifts(config)

    assert needed == (2.0, 3.0)


def test_generate_single_bcg_mock_worker_count_is_deterministic(tmp_path: Path) -> None:
    base = _jax_mock_test_config(
        seed=19,
        n_primary_families=2,
        primary_source_redshifts=(2.0, 3.0),
        max_sources_to_try=80,
    )

    _paths_one, images_one, truth_one = generate_single_bcg_mock(
        tmp_path / "workers1",
        replace(base, mock_generation_workers=1),
    )
    _paths_two, images_two, truth_two = generate_single_bcg_mock(
        tmp_path / "workers2",
        replace(base, mock_generation_workers=2),
    )

    pd.testing.assert_frame_equal(images_one.reset_index(drop=True), images_two.reset_index(drop=True))
    assert truth_one["sources"] == truth_two["sources"]
    assert truth_one["caustics_by_source_redshift"] == truth_two["caustics_by_source_redshift"]


def test_jax_lensing_tiled_alpha_hessian_matches_canonical_helper() -> None:
    lens_state = mock_generation.jax_lensing.static_lens_state_from_kwargs(
        [mock_generation.ORIGINAL_DPIE_PROFILE_NAME, mock_generation.ORIGINAL_DPIE_PROFILE_NAME],
        [
            {
                "sigma0": 2.3,
                "Ra": 0.2,
                "Rs": 8.0,
                "e1": 0.05,
                "e2": -0.02,
                "center_x": 0.1,
                "center_y": -0.2,
            },
            {
                "sigma0": 0.8,
                "Ra": 0.1,
                "Rs": 4.0,
                "e1": -0.03,
                "e2": 0.04,
                "center_x": -0.5,
                "center_y": 0.3,
            },
        ],
    )
    x = jnp.asarray(np.linspace(-2.0, 2.0, 11), dtype=jnp.float64)
    y = jnp.asarray(np.linspace(1.5, -1.5, 11), dtype=jnp.float64)

    expected = mock_generation.jax_lensing.alpha_and_hessian(x, y, lens_state)
    actual = mock_generation.jax_lensing.alpha_and_hessian_tiled(x, y, lens_state, chunk_size=4)

    for expected_values, actual_values in zip(expected, actual):
        np.testing.assert_allclose(np.asarray(actual_values), np.asarray(expected_values), rtol=1.0e-10, atol=1.0e-10)


def test_compute_tangential_caustics_stable_across_memory_budgets() -> None:
    base = _jax_mock_test_config(
        caustic_compute_window_arcsec=40.0,
        caustic_grid_scale_arcsec=2.0,
    )
    cosmo_config = mock_generation.flat_wcdm_config(h0=70.0, om0=0.3)
    lens_state = mock_generation._mock_static_lens_state(base, [], 2.0, cosmo_config)
    small_chunks = replace(base, mock_caustic_grid_chunk_memory_gb=1.0e-5)
    large_chunks = replace(base, mock_caustic_grid_chunk_memory_gb=0.01)

    small_contours = mock_generation._compute_tangential_caustic_contours(lens_state, small_chunks)
    large_contours = mock_generation._compute_tangential_caustic_contours(lens_state, large_chunks)

    assert len(small_contours) == len(large_contours)
    np.testing.assert_allclose(
        [contour.caustic_area_arcsec2 for contour in small_contours],
        [contour.caustic_area_arcsec2 for contour in large_contours],
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_generate_single_bcg_mock_uses_family_class_seed_grid_pixels(tmp_path: Path) -> None:
    config = _jax_mock_test_config(
        seed=11,
        n_primary_families=1,
        n_subhalo_families=1,
        n_subhalos=8,
        **_fixed_potfile_scaling_config(sigma_ref=300.0, cut_radius_arcsec=45.0),
        primary_source_redshifts=(2.0,),
        subhalo_source_redshifts=(2.0,),
        primary_seed_grid_pixel_arcsec=2.5,
        subhalo_seed_grid_pixel_arcsec=0.75,
        mock_image_search_window_arcsec=90.0,
        max_sources_to_try=100,
        caustic_compute_window_arcsec=80.0,
        caustic_grid_scale_arcsec=1.0,
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    family_sizes = images.groupby("family_id").size().to_dict()
    assert family_sizes == {source["family_id"]: source["n_images"] for source in truth["sources"]}
    assert truth["config"]["primary_seed_grid_pixel_arcsec"] == pytest.approx(2.5)
    assert truth["config"]["subhalo_seed_grid_pixel_arcsec"] == pytest.approx(0.75)
    assert truth["config"]["mock_image_search_window_arcsec"] == pytest.approx(90.0)
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary", "subhalo"}


def test_generate_single_bcg_mock_batch_size_does_not_change_seeded_output(tmp_path: Path) -> None:
    base = _jax_mock_test_config(seed=17, n_primary_families=2, primary_source_redshifts=(2.0, 3.0), max_sources_to_try=80)
    small_batch = replace(base, mock_image_candidate_batch_size=4)
    large_batch = replace(base, mock_image_candidate_batch_size=16)

    _paths_a, images_a, truth_a = generate_single_bcg_mock(tmp_path / "a", small_batch)
    _paths_b, images_b, truth_b = generate_single_bcg_mock(tmp_path / "b", large_batch)

    assert images_a.to_dict("records") == images_b.to_dict("records")
    assert truth_a["sources"] == truth_b["sources"]
    assert truth_a["images"] == truth_b["images"]


def test_generate_single_bcg_mock_refined_images_ray_shoot_to_source(tmp_path: Path) -> None:
    config = _jax_mock_test_config(mock_image_precision_limit=1.0e-7)
    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    realized_config = mock_generation._caustic_config_from_truth(truth)
    cosmo_config = mock_generation.flat_wcdm_config(h0=70.0, om0=0.3)
    lens_state = mock_generation._mock_static_lens_state(realized_config, [], 2.0, cosmo_config)

    beta_x, beta_y = mock_generation.jax_lensing.ray_shooting(
        mock_generation.jnp.asarray(images["x_true_arcsec"].to_numpy(dtype=float), dtype=mock_generation.jnp.float64),
        mock_generation.jnp.asarray(images["y_true_arcsec"].to_numpy(dtype=float), dtype=mock_generation.jnp.float64),
        lens_state,
    )
    source = truth["sources"][0]
    residual = np.hypot(np.asarray(beta_x) - source["beta_x"], np.asarray(beta_y) - source["beta_y"])
    assert float(np.max(residual)) <= 5.0 * realized_config.mock_image_precision_limit


def test_generate_single_bcg_mock_with_subhalos_current_truth_schema(tmp_path: Path) -> None:
    config = _jax_mock_test_config(
        n_subhalos=6,
        n_primary_families=1,
        max_sources_to_try=80,
        caustic_compute_window_arcsec=80.0,
        caustic_grid_scale_arcsec=2.0,
    )

    _paths, _images, truth = generate_single_bcg_mock(tmp_path, config)

    components = truth["lens_components"]
    assert [component["component_role"] for component in components[:2]] == ["halo", "bcg"]
    assert sum(component["component_role"] == "subhalo" for component in components) == len(truth["subhalos"])
    assert all(component["profile_name"] == mock_generation.ORIGINAL_DPIE_PROFILE_NAME for component in components)


def test_generate_single_bcg_mock_raises_when_family_exhausts_attempts(tmp_path: Path) -> None:
    config = _jax_mock_test_config(max_sources_to_try=1, min_images_per_family=100, mock_generation_workers=2)

    with pytest.raises(RuntimeError, match="Failed to generate a primary source"):
        generate_single_bcg_mock(tmp_path, config)


def test_generate_single_bcg_mock_drops_exhausted_subhalo_family(tmp_path: Path) -> None:
    # A subhalo image-separation requirement no subhalo caustic can satisfy exhausts the
    # subhalo family; the seed must survive with the family dropped and recorded.
    config = _jax_mock_test_config(
        n_subhalos=6,
        n_primary_families=1,
        n_subhalo_families=1,
        subhalo_source_redshifts=(2.0,),
        subhalo_seed_grid_pixel_arcsec=60.0,
        max_sources_to_try=20,
        caustic_compute_window_arcsec=80.0,
        caustic_grid_scale_arcsec=2.0,
    )

    _paths, _images, truth = generate_single_bcg_mock(tmp_path, config)

    dropped = truth.get("dropped_families", [])
    assert len(dropped) == 1
    assert dropped[0]["caustic_class"] == "subhalo"
    assert dropped[0]["z_source"] == pytest.approx(2.0)
    assert dropped[0]["reason"] in {"attempts_exhausted", "no_usable_caustics"}
    accepted_classes = [str(source["caustic_class"]) for source in truth["sources"]]
    assert accepted_classes == ["primary"]
    stats = truth["generation_stats"]
    assert stats["dropped_family_count"] == 1
    by_class = {row["caustic_class"]: row for row in stats["families"]}
    assert by_class["primary"]["accepted"] is True
    assert by_class["primary"]["accepted_attempt"] == by_class["primary"]["attempts_used"] or by_class["primary"]["attempts_used"] >= 1
    assert by_class["subhalo"]["accepted"] is False
    assert by_class["subhalo"]["drop_reason"] in {"attempts_exhausted", "no_usable_caustics"}
    # Every image row carries the truth magnification at the observed position.
    images_payload = truth["images"]
    assert all("magnification_true_at_obs" in row for row in images_payload)
    assert all(np.isfinite(float(row["magnification_true_at_obs"])) for row in images_payload)


def test_mock_generation_has_no_lenstronomy_image_solver_surface() -> None:
    source_text = Path(mock_generation.__file__).read_text(encoding="utf-8")
    assert not hasattr(mock_generation, "LensEquationSolver")
    assert not hasattr(mock_generation, "LensModel")
    assert not hasattr(SingleBCGMockConfig, "image_generation_backend")
    assert "image_position_from_source" not in source_text
    assert "DPIENIE" not in source_text


def test_load_best_par_defaults_missing_potfile_slopes_to_four(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0 nan\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "missing_slopes.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potfile
    filein 3 members.cat
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5
    sigma 1 10. 200.
    cutkpc 1 1. 40.
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, _arcs_df, _potentials_with_priors = load_best_par(par_path)

    potfile = parsed["potfiles"][0]
    assert "vdslope_nominal" not in potfile
    assert "slope_nominal" not in potfile
    assert potfile["catalog_df"].loc[0, "catalog_color"] != potfile["catalog_df"].loc[0, "catalog_color"]


def test_load_best_par_mode9_potfile_nominal_values_use_mean(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0 nan\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "mode9_potfile.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potfile
    filein 3 members.cat
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5
    sigma 9 245.0 55.0 190.0 300.0
    cutkpc 9 20.0 5.0 1.0 40.0
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, _arcs_df, _potentials_with_priors = load_best_par(par_path)

    potfile = parsed["potfiles"][0]
    assert potfile["sigma_nominal"] == pytest.approx(245.0)
    assert potfile["cutkpc_nominal"] == pytest.approx(20.0)


def test_load_best_par_strips_inline_comments_in_named_blocks(tmp_path: Path) -> None:
    catalog_path = tmp_path / "members.cat"
    catalog_path.write_text(
        "#REFERENCE 0\n"
        "1 39.970000 -1.580000 1.0 1.0 0.0 19.5000 1.0 nan\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "inline_comments.par"
    par_path.write_text(
        """
runmode
    reference 3 39.971340 -1.582260
    end

potentiel 1 # main halo
    profil 81
    x_centre 0 #-0.853
    y_centre 0 # -1.999
    ellipticite 0.1
    angle_pos 12.0
    core_radius 1.0
    cut_radius 20.0
    v_disp 700.0
    z_lens 0.375000
    # full-line comment inside a block
    end

potfile # member scaling
    filein 3 members.cat # cluster members
    zlens 0.375000
    type 81
    corekpc 0.150000
    mag0 19.5 # BCG mag
    sigma 1 10. 200. # real prior list
    cutkpc 1 1. 40. # real prior list
    vdslope 0 3.5 0
    slope 0 3.0 0
    end
fini
""",
        encoding="utf-8",
    )

    parsed, potentials_df, _images_df, _arcs_df, potentials_with_priors = load_best_par(par_path)

    assert potentials_df.loc[0, "id"] == "1"
    assert potentials_df.loc[0, "x_centre"] == pytest.approx(0.0)
    assert potentials_df.loc[0, "y_centre"] == pytest.approx(0.0)
    assert potentials_with_priors[0]["id"] == "1"
    potfile = parsed["potfiles"][0]
    assert potfile["mag0"] == pytest.approx(19.5)
    assert potfile["sigma"] == [1, 10.0, 200.0]
    assert potfile["sigma_nominal"] == pytest.approx(105.0)
    assert potfile["cutkpc_nominal"] == pytest.approx(20.5)


def test_load_best_par_preserves_repeated_limit_blocks_after_pluralization(tmp_path: Path) -> None:
    par_path = tmp_path / "repeated_limits.par"
    par_path.write_text(
        """
runmode
    reference 3 181.55062 -8.8009361
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

potentiel 1
    profil 81
    x_centre 0.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 1
    v_disp 1 450.0 1200.0 0.1
    end

potentiel 2
    profil 81
    x_centre 10.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 2
    x_centre 1 0.0 30.0 0.1
    end

potentiel 3
    profil 81
    x_centre -35.0
    y_centre -12.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 0.0
    cut_radius 200.0
    v_disp 650.0
    z_lens 0.439
    end

limit 3
    core_radius 1 0.0 40.0 0.1
    end

potentiel 4
    profil 14
    gamma 0.1
    angle_pos 0.0
    z_lens 0.439
    end

limit 4
    gamma 1 0.0 1.0 0.1
    end
fini
""",
        encoding="utf-8",
    )

    parsed, _potentials_df, _images_df, _arcs_df, potentials_with_priors = load_best_par(par_path)

    assert "limit" not in parsed
    assert [item["id"] for item in parsed["limits"]] == ["1", "2", "3", "4"]
    priors_by_id = {item["id"]: item["priors"] for item in potentials_with_priors}
    assert "v_disp" in priors_by_id["1"]
    assert "x_centre" in priors_by_id["2"]
    assert "core_radius_kpc" in priors_by_id["3"]
    assert "gamma" in priors_by_id["4"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2.a", ("2", "a")),
        ("A.b", ("A", "b")),
        ("A200.ab", ("A200", "ab")),
        ("AB12.c", ("AB12", "c")),
    ],
)
def test_split_image_label_requires_strict_family_dot_image(label: str, expected: tuple[str, str]) -> None:
    assert _split_image_label(label) == expected


@pytest.mark.parametrize("label", ["1a", "A200a", "2.1a", "2.A", "fam.a", "candidate", "A-1.a"])
def test_split_image_label_rejects_invalid_labels(label: str) -> None:
    with pytest.raises(ValueError, match=r"expected FAMILY\.image"):
        _split_image_label(label)


def test_load_best_par_groups_strict_image_labels(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.c 10.0004 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "strict_labels.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, _arcs_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["image_label"].astype(str).tolist() == ["1.a", "1.b", "1.c"]
    assert images_df["family_id"].astype(str).tolist() == ["1", "1", "1"]
    assert images_df["image_id"].astype(str).tolist() == ["a", "b", "c"]


def test_load_best_par_arcfile_reference3_loads_independent_arcs(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.0 0.0 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 1.0 0.0 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 3\n"
        "arc_A 0.25 -0.50 -1 0.82 0.09 0.05 0.02 0.7\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "with_arcs.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, arcs_df, _potentials_with_priors = load_best_par(par_path)

    assert not any(column.startswith("arc_") for column in images_df.columns)
    assert arcs_df["arc_id"].astype(str).tolist() == ["arc_A"]
    first = arcs_df.iloc[0]
    assert float(first["z_arc"]) == pytest.approx(-1.0)
    x_anchor, y_anchor = cluster_solver._radec_to_offsets_arcsec(
        np.asarray([first["arc_anchor_ra"]], dtype=float),
        np.asarray([first["arc_anchor_dec"]], dtype=float),
        10.0,
        20.0,
    )
    assert float(x_anchor[0]) == pytest.approx(0.25)
    assert float(y_anchor[0]) == pytest.approx(-0.50)
    assert float(first["arc_tangent_angle_rad"]) == pytest.approx(0.82)
    assert float(first["arc_curvature_arcsec_inv"]) == pytest.approx(0.09)
    assert float(first["arc_sigma_tangent_angle_rad"]) == pytest.approx(0.05)
    assert float(first["arc_sigma_curvature_arcsec_inv"]) == pytest.approx(0.02)
    assert float(first["arc_reliability"]) == pytest.approx(0.7)


def test_load_best_par_arcfile_accepts_arc_ids_unlinked_to_images(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 10.0002 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 0\n"
        "independent_arc 10.0000 20.0000 2.0 0.82 0.09 0.05 0.02 1.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "unknown_arc.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, arcs_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["image_label"].astype(str).tolist() == ["1.a", "1.b"]
    assert arcs_df["arc_id"].astype(str).tolist() == ["independent_arc"]
    assert float(arcs_df.iloc[0]["z_arc"]) == pytest.approx(2.0)


def test_load_arc_constraints_catalog_accepts_allowed_redshift_values(tmp_path: Path) -> None:
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 0\n"
        "arc_z 10.0000 20.0000 2.0 0.82 0.09 0.05 0.02 1.0\n"
        "arc_zero 10.0001 20.0000 0 0.72 0.08 0.05 0.02 1.0\n"
        "arc_unknown 10.0002 20.0000 -1 0.62 0.07 0.05 0.02 1.0\n",
        encoding="utf-8",
    )

    arcs_df = _load_arc_constraints_catalog(arc_catalog_path, None)

    assert arcs_df["arc_id"].astype(str).tolist() == ["arc_z", "arc_zero", "arc_unknown"]
    np.testing.assert_allclose(arcs_df["z_arc"].to_numpy(dtype=float), [2.0, 0.0, -1.0])


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("arc_A 10.0000 20.0000 0.82 0.09 0.05 0.02", "fewer than 8"),
        ("arc_A 10.0000 20.0000 -0.5 0.82 0.09 0.05 0.02", "invalid z_arc"),
        ("arc_A 10.0000 20.0000 2.0 0.82 -0.09 0.05 0.02", "negative curvature"),
        ("arc_A 10.0000 20.0000 2.0 0.82 0.09 0.00 0.02", "non-positive tangent sigma"),
        ("arc_A 10.0000 20.0000 2.0 0.82 0.09 0.05 0.00", "non-positive curvature sigma"),
    ],
)
def test_load_arc_constraints_catalog_rejects_invalid_rows(
    tmp_path: Path,
    row: str,
    match: str,
) -> None:
    arc_catalog_path = tmp_path / "invalid_arc_constraints.dat"
    arc_catalog_path.write_text(f"#REFERENCE 0\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_arc_constraints_catalog(arc_catalog_path, None)


def test_load_arc_constraints_catalog_rejects_duplicate_arc_ids(tmp_path: Path) -> None:
    arc_catalog_path = tmp_path / "duplicate_arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 0\n"
        "arc_A 10.0000 20.0000 2.0 0.82 0.09 0.05 0.02\n"
        "arc_A 10.0001 20.0000 2.0 0.72 0.08 0.05 0.02\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate arc_id"):
        _load_arc_constraints_catalog(arc_catalog_path, None)


def test_cab_tangent_angle_residual_wraps_modulo_pi() -> None:
    residual = cluster_solver._cab_tangent_angle_residual(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(np.pi - 0.05, dtype=jnp.float64),
    )

    assert float(residual) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "entries",
    [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, -0.1, -0.1, 0.0),
        (1.0, -1.0e-6, -1.0e-6, 1.0),
    ],
)
def test_cab_tangent_frame_has_finite_weighted_gradient(entries: tuple[float, float, float, float]) -> None:
    def objective(values: jnp.ndarray) -> jnp.ndarray:
        frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(
            values[0],
            values[1],
            values[2],
            values[3],
        )
        axial_signal = jnp.sum(frame.branch_weight * jnp.sin(frame.tangent_angle_rad), axis=-1)
        return frame.frame_weight * axial_signal

    values = jnp.asarray(entries, dtype=jnp.float64)
    frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(values[0], values[1], values[2], values[3])
    gradient = jax.grad(objective)(values)

    assert bool(frame.finite)
    assert np.all(np.isfinite(np.asarray(frame.tangent_angle_rad)))
    assert np.all(np.isfinite(np.asarray(frame.branch_weight)))
    assert np.isfinite(float(frame.frame_weight))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_cab_tangent_frame_weights_fade_at_ambiguous_limits() -> None:
    zero_shear = cluster_solver._cab_tangent_frame_from_jacobian_entries(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    kappa_one_shear = cluster_solver._cab_tangent_frame_from_jacobian_entries(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(-0.1, dtype=jnp.float64),
        jnp.asarray(-0.1, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    axis_ratio_1p1 = cluster_solver._cab_tangent_frame_from_jacobian_entries(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.1, dtype=jnp.float64),
    )
    clear_frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )

    assert float(zero_shear.frame_weight) == pytest.approx(0.0, abs=1.0e-12)
    assert float(kappa_one_shear.frame_weight) == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(np.asarray(kappa_one_shear.branch_weight), [0.5, 0.5], rtol=1.0e-6, atol=1.0e-8)
    assert float(axis_ratio_1p1.frame_weight) < 0.25
    assert float(clear_frame.frame_weight) > 0.99
    assert float(clear_frame.branch_weight[0]) > 0.99


@pytest.mark.parametrize(
    "entries",
    [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, -0.1, -0.1, 0.0),
    ],
)
def test_cab_morphology_loglike_has_finite_jacobian_gradient(
    entries: tuple[float, float, float, float],
) -> None:
    def loglike(values: jnp.ndarray) -> jnp.ndarray:
        frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(
            values[0:1],
            values[1:2],
            values[2:3],
            values[3:4],
        )
        return cluster_solver._cab_morphology_arc_catalog_loglike(
            predicted_tangent_angle_rad=frame.tangent_angle_rad,
            predicted_curvature_arcsec_inv=jnp.zeros_like(frame.tangent_angle_rad),
            prediction_finite=jnp.repeat(frame.finite[..., jnp.newaxis], 2, axis=-1),
            observed_tangent_angle_rad=jnp.asarray([0.2], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([0.0], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([0.05], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([0.02], dtype=jnp.float64),
            reliability=jnp.asarray([1.0], dtype=jnp.float64),
            active_arcs=jnp.asarray([True]),
            branch_weight=frame.branch_weight,
            frame_weight=jnp.repeat(frame.frame_weight[..., jnp.newaxis], 2, axis=-1),
        )

    values = jnp.asarray(entries, dtype=jnp.float64)
    gradient = jax.grad(loglike)(values)

    assert np.isfinite(float(loglike(values)))
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize("gamma1", [0.0, 1.0e-6])
def test_cab_morphology_physical_shear_gradient_is_bounded(gamma1: float) -> None:
    def loglike_from_gamma2(gamma2: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(1.0, dtype=jnp.float64)
        gamma1_value = jnp.asarray(gamma1, dtype=jnp.float64)
        frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(
            (p - gamma1_value)[jnp.newaxis],
            (-gamma2)[jnp.newaxis],
            (-gamma2)[jnp.newaxis],
            (p + gamma1_value)[jnp.newaxis],
        )
        return cluster_solver._cab_morphology_arc_catalog_loglike(
            predicted_tangent_angle_rad=frame.tangent_angle_rad,
            predicted_curvature_arcsec_inv=jnp.zeros_like(frame.tangent_angle_rad),
            prediction_finite=jnp.repeat(frame.finite[..., jnp.newaxis], 2, axis=-1),
            observed_tangent_angle_rad=jnp.asarray([0.0], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([0.0], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([0.05], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([0.02], dtype=jnp.float64),
            reliability=jnp.asarray([1.0], dtype=jnp.float64),
            active_arcs=jnp.asarray([True]),
            branch_weight=frame.branch_weight,
            frame_weight=jnp.repeat(frame.frame_weight[..., jnp.newaxis], 2, axis=-1),
        )

    derivative = jax.grad(loglike_from_gamma2)(jnp.asarray(0.0, dtype=jnp.float64))

    assert np.isfinite(float(derivative))
    assert abs(float(derivative)) < 1.0e3


def test_cab_morphology_kappa_crossing_gradient_is_physically_gated() -> None:
    gamma = jnp.asarray(0.3, dtype=jnp.float64)
    p_values = jnp.asarray(
        [3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 0.0, -3.0e-4, -1.0e-3, -3.0e-3, -1.0e-2, -3.0e-2],
        dtype=jnp.float64,
    )

    def row_loglike(p: jnp.ndarray) -> jnp.ndarray:
        frame = cluster_solver._cab_tangent_frame_from_jacobian_entries(
            p[jnp.newaxis],
            (-gamma)[jnp.newaxis],
            (-gamma)[jnp.newaxis],
            p[jnp.newaxis],
        )
        return cluster_solver._cab_morphology_arc_catalog_loglike(
            predicted_tangent_angle_rad=frame.tangent_angle_rad,
            predicted_curvature_arcsec_inv=jnp.zeros_like(frame.tangent_angle_rad),
            prediction_finite=jnp.repeat(frame.finite[..., jnp.newaxis], 2, axis=-1),
            observed_tangent_angle_rad=jnp.asarray([0.25 * jnp.pi], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([0.0], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([0.1], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([0.02], dtype=jnp.float64),
            reliability=jnp.asarray([1.0], dtype=jnp.float64),
            active_arcs=jnp.asarray([True]),
            branch_weight=frame.branch_weight,
            frame_weight=jnp.repeat(frame.frame_weight[..., jnp.newaxis], 2, axis=-1),
        )

    values = jax.vmap(row_loglike)(p_values)
    gradients = jax.vmap(jax.grad(row_loglike))(p_values)

    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert float(jnp.max(jnp.abs(gradients))) < 500.0


def test_cab_morphology_loglike_rewards_matching_constraints_and_masks_rows() -> None:
    observed_angle = jnp.asarray([0.25, 1.0], dtype=jnp.float64)
    observed_curvature = jnp.asarray([0.08, 0.9], dtype=jnp.float64)
    sigma_angle = jnp.asarray([0.05, 0.05], dtype=jnp.float64)
    sigma_curvature = jnp.asarray([0.02, 0.02], dtype=jnp.float64)
    reliability = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    mask = jnp.asarray([True, False])
    exact = cluster_solver._cab_morphology_arc_catalog_loglike(
        predicted_tangent_angle_rad=observed_angle,
        predicted_curvature_arcsec_inv=observed_curvature,
        prediction_finite=jnp.asarray([True, True]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        active_arcs=mask,
    )
    perturbed = cluster_solver._cab_morphology_arc_catalog_loglike(
        predicted_tangent_angle_rad=observed_angle + jnp.asarray([0.10, 10.0], dtype=jnp.float64),
        predicted_curvature_arcsec_inv=observed_curvature + jnp.asarray([0.04, 10.0], dtype=jnp.float64),
        prediction_finite=jnp.asarray([True, True]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        active_arcs=mask,
    )
    no_constraints = cluster_solver._cab_morphology_arc_catalog_loglike(
        predicted_tangent_angle_rad=observed_angle + 10.0,
        predicted_curvature_arcsec_inv=observed_curvature + 10.0,
        prediction_finite=jnp.asarray([False, False]),
        observed_tangent_angle_rad=observed_angle,
        observed_curvature_arcsec_inv=observed_curvature,
        sigma_tangent_angle_rad=sigma_angle,
        sigma_curvature_arcsec_inv=sigma_curvature,
        reliability=reliability,
        active_arcs=jnp.asarray([False, False]),
    )

    assert float(exact) > float(perturbed)
    assert float(no_constraints) == pytest.approx(0.0)


def test_cab_morphology_outlier_branch_limits_bad_constraint_penalty() -> None:
    common = dict(
        predicted_tangent_angle_rad=jnp.asarray([np.pi / 2.0], dtype=jnp.float64),
        predicted_curvature_arcsec_inv=jnp.asarray([10.0], dtype=jnp.float64),
        prediction_finite=jnp.asarray([True]),
        observed_tangent_angle_rad=jnp.asarray([0.0], dtype=jnp.float64),
        observed_curvature_arcsec_inv=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_tangent_angle_rad=jnp.asarray([0.01], dtype=jnp.float64),
        sigma_curvature_arcsec_inv=jnp.asarray([0.01], dtype=jnp.float64),
        reliability=jnp.asarray([1.0], dtype=jnp.float64),
        active_arcs=jnp.asarray([True]),
    )

    trusted = cluster_solver._cab_morphology_arc_catalog_loglike(**common, frame_weight=jnp.asarray([1.0], dtype=jnp.float64))
    ambiguous = cluster_solver._cab_morphology_arc_catalog_loglike(**common, frame_weight=jnp.asarray([0.0], dtype=jnp.float64))

    assert np.isfinite(float(trusted))
    assert np.isfinite(float(ambiguous))
    assert float(ambiguous) > float(trusted)
    sigma_curvature_outlier = max(
        cluster_solver.CAB_OUTLIER_CURVATURE_SIGMA_ARCSEC_INV,
        3.0 * 0.01,
    )
    expected_outlier = (
        -math.log(math.pi)
        + 0.5 * math.log(2.0 / (math.pi * sigma_curvature_outlier**2))
        - 0.5 * (0.0 / sigma_curvature_outlier) ** 2
    )
    assert float(ambiguous) == pytest.approx(expected_outlier, abs=1.0e-10)


def test_cab_outlier_is_two_dimensional_density() -> None:
    def outlier_for(observed_curvature: float, sigma_curvature: float) -> float:
        terms = cluster_solver._cab_morphology_terms(
            predicted_tangent_angle_rad=jnp.asarray([0.3], dtype=jnp.float64),
            predicted_curvature_arcsec_inv=jnp.asarray([0.05], dtype=jnp.float64),
            prediction_finite=jnp.asarray([True]),
            observed_tangent_angle_rad=jnp.asarray([0.3], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([observed_curvature], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([0.05], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([sigma_curvature], dtype=jnp.float64),
            reliability=jnp.asarray([0.9], dtype=jnp.float64),
            active_arcs=jnp.asarray([True]),
        )
        return float(terms.outlier_ll[0])

    sigma_curvature_outlier = max(
        cluster_solver.CAB_OUTLIER_CURVATURE_SIGMA_ARCSEC_INV,
        3.0 * 0.02,
    )
    expected_quad_difference = -0.5 * (0.2**2 - 0.0**2) / sigma_curvature_outlier**2
    assert outlier_for(0.2, 0.02) - outlier_for(0.0, 0.02) == pytest.approx(
        expected_quad_difference, abs=1.0e-12
    )

    broad_sigma_curvature = 0.5
    expected_broad = (
        -math.log(math.pi)
        + 0.5 * math.log(2.0 / (math.pi * (3.0 * broad_sigma_curvature) ** 2))
        - 0.5 * (0.1 / (3.0 * broad_sigma_curvature)) ** 2
    )
    assert outlier_for(0.1, broad_sigma_curvature) == pytest.approx(expected_broad, abs=1.0e-12)


def test_cab_inlier_density_normalized_on_axial_halfline_support() -> None:
    # The inlier density must integrate to 1 over theta in [0, pi) x kappa in [0, inf).
    predicted_angle = 0.7
    predicted_curvature = 0.05
    sigma_tangent = 0.2
    sigma_curvature = 0.03
    n_theta, n_kappa = 400, 1000
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    kappas = np.linspace(0.0, 0.5, n_kappa)  # well past predicted + many sigma
    dtheta = float(np.pi / n_theta)
    dkappa = float(kappas[1] - kappas[0])
    th_grid, ka_grid = np.meshgrid(thetas, kappas, indexing="ij")
    obs_angle = jnp.asarray(th_grid.ravel(), dtype=jnp.float64)
    obs_curv = jnp.asarray(ka_grid.ravel(), dtype=jnp.float64)
    n = int(obs_angle.shape[0])
    terms = cluster_solver._cab_morphology_terms(
        predicted_tangent_angle_rad=jnp.full((n,), predicted_angle, dtype=jnp.float64),
        predicted_curvature_arcsec_inv=jnp.full((n,), predicted_curvature, dtype=jnp.float64),
        prediction_finite=jnp.ones((n,), dtype=bool),
        observed_tangent_angle_rad=obs_angle,
        observed_curvature_arcsec_inv=obs_curv,
        sigma_tangent_angle_rad=jnp.full((n,), sigma_tangent, dtype=jnp.float64),
        sigma_curvature_arcsec_inv=jnp.full((n,), sigma_curvature, dtype=jnp.float64),
        reliability=jnp.ones((n,), dtype=jnp.float64),
        active_arcs=jnp.ones((n,), dtype=bool),
        tangent_sigma_floor_rad=1.0e-6,
        curvature_sigma_floor_arcsec_inv=1.0e-6,
    )
    density = np.asarray(jnp.exp(terms.inlier_ll))
    integral = float(density.sum()) * dtheta * dkappa
    assert integral == pytest.approx(1.0, abs=0.02)


def test_cab_inlier_tangent_matches_gaussian_only_for_small_sigma() -> None:
    # inlier_ll difference between a residual delta and 0 isolates the tangent term
    # (curvature is identical and cancels). It must match the Gaussian -0.5 (d/s)^2
    # for small sigma and deviate for broad sigma near the arctrace cap.
    def tangent_logdiff(sigma: float, delta: float) -> float:
        terms = cluster_solver._cab_morphology_terms(
            predicted_tangent_angle_rad=jnp.asarray([0.3 + delta, 0.3], dtype=jnp.float64),
            predicted_curvature_arcsec_inv=jnp.asarray([0.05, 0.05], dtype=jnp.float64),
            prediction_finite=jnp.asarray([True, True]),
            observed_tangent_angle_rad=jnp.asarray([0.3, 0.3], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([0.05, 0.05], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([sigma, sigma], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([0.02, 0.02], dtype=jnp.float64),
            reliability=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
            active_arcs=jnp.asarray([True, True]),
            tangent_sigma_floor_rad=1.0e-6,
        )
        return float(terms.inlier_ll[0] - terms.inlier_ll[1])

    gauss_small = -0.5 * (0.01 / 0.02) ** 2
    assert tangent_logdiff(0.02, 0.01) == pytest.approx(gauss_small, rel=1.0e-2)
    gauss_broad = -0.5 * (0.5 / 0.9) ** 2
    assert abs(tangent_logdiff(0.9, 0.5) - gauss_broad) > 1.0e-3


def test_cab_inlier_curvature_gradient_bounded_near_zero() -> None:
    # The lower-truncation normalizer -log Phi(predicted/sigma) keeps the curvature
    # gradient finite and bounded by the Mills ratio (predicted >= 0).
    sigma_c = 0.05

    def inlier_ll(pred_curv: jnp.ndarray) -> jnp.ndarray:
        terms = cluster_solver._cab_morphology_terms(
            predicted_tangent_angle_rad=jnp.asarray([0.3], dtype=jnp.float64),
            predicted_curvature_arcsec_inv=jnp.reshape(pred_curv, (1,)),
            prediction_finite=jnp.asarray([True]),
            observed_tangent_angle_rad=jnp.asarray([0.3], dtype=jnp.float64),
            observed_curvature_arcsec_inv=jnp.asarray([0.0], dtype=jnp.float64),
            sigma_tangent_angle_rad=jnp.asarray([0.1], dtype=jnp.float64),
            sigma_curvature_arcsec_inv=jnp.asarray([sigma_c], dtype=jnp.float64),
            reliability=jnp.asarray([1.0], dtype=jnp.float64),
            active_arcs=jnp.asarray([True]),
            curvature_sigma_floor_arcsec_inv=1.0e-6,
        )
        return terms.inlier_ll[0]

    grad = jax.grad(inlier_ll)
    # At predicted == observed == 0 the only gradient is the truncation term:
    # -phi(0)/Phi(0)/sigma = -0.79788.../sigma.
    g0 = float(grad(jnp.asarray(0.0, dtype=jnp.float64)))
    assert g0 == pytest.approx(-0.7978845608 / sigma_c, rel=1.0e-4)
    for p in (0.0, 0.001, 0.01, 0.05, 0.2):
        gp = float(grad(jnp.asarray(p, dtype=jnp.float64)))
        assert np.isfinite(gp)
        assert abs(gp) <= 0.8 / sigma_c + p / sigma_c**2 + 1.0e-6


def test_cab_likelihood_weight_defaults_to_one_when_arc_data_exists() -> None:
    state_with_arcs = SimpleNamespace(arc_data=SimpleNamespace(n_arcs=1), family_data=[], bin_data=[])
    state_without_arcs = SimpleNamespace(arc_data=None, family_data=[], bin_data=[])

    assert cluster_solver._effective_cab_likelihood_weight(None, state_with_arcs) == pytest.approx(1.0)
    assert cluster_solver._effective_cab_likelihood_weight(None, state_without_arcs) == pytest.approx(0.0)
    assert cluster_solver._effective_cab_likelihood_weight(0.25, state_with_arcs) == pytest.approx(0.25)


def test_cab_arcs_survive_parser_prep_and_traced_conversion(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 3\n"
        "1.a 0.0 0.0 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 1.0 0.0 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    arc_catalog_path = tmp_path / "arc_constraints.dat"
    arc_catalog_path.write_text(
        "#REFERENCE 3\n"
        "arc_A 0.25 -0.50 0 0.82 0.09 0.05 0.02 0.7\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "with_arcs.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    arcfile 1 arc_constraints.dat
    end
fini
""",
        encoding="utf-8",
    )
    parsed, _potentials_df, images_df, arcs_df, _potentials_with_priors = load_best_par(par_path)
    reference = cluster_solver._extract_reference(parsed)
    arc_data = cluster_solver._prepare_arc_constraint_data(arcs_df, reference)
    families, _elapsed = cluster_solver._prepare_family_data(
        images_df,
        0.2,
        reference,
        z_lens=0.4,
        cosmo_config=cluster_solver._build_cosmology(parsed),
        z_bin_efficiency_tol=0.0,
    )
    bins = cluster_solver._build_bin_data(families)
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.cosmology_effective_z_to_index = {}

    traced_bin = cluster_solver.ClusterJAXEvaluator._prepare_traced_bin_data(fake, bins[0])
    traced_arcs = cluster_solver.ClusterJAXEvaluator._prepare_traced_arc_constraint_data(fake, arc_data)

    assert arc_data is not None
    assert arc_data.arc_ids == ["arc_A"]
    np.testing.assert_allclose(arc_data.z_arc, [0.0])
    np.testing.assert_allclose(arc_data.anchor_x, [0.25])
    np.testing.assert_allclose(arc_data.anchor_y, [-0.5])
    np.testing.assert_allclose(arc_data.tangent_angle_rad, [0.82])
    np.testing.assert_allclose(arc_data.curvature_arcsec_inv, [0.09])
    assert traced_arcs.arc_ids == ("arc_A",)
    np.testing.assert_allclose(np.asarray(traced_arcs.tangent_angle_rad), [0.82])
    np.testing.assert_allclose(np.asarray(traced_arcs.curvature_arcsec_inv), [0.09])
    assert not hasattr(families[0], "arc_has_constraint")
    assert not hasattr(bins[0], "arc_has_constraint")
    assert not hasattr(traced_bin, "arc_constraint_count")


def test_prepare_family_data_rejects_legacy_cab_image_columns() -> None:
    images = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.a"],
            "catalog_z": [2.0],
            "catalog_source": ["manual"],
            "ra": [10.0],
            "dec": [20.0],
            "arc_has_constraint": [True],
            "arc_anchor_ra": [10.0],
            "arc_anchor_dec": [20.0],
        }
    )

    with pytest.raises(ValueError, match="legacy CAB arc columns"):
        cluster_solver._prepare_family_data(
            images,
            sigma_arcsec=0.5,
            reference=(3, 10.0, 20.0),
            z_lens=0.3,
            cosmo_config={},
            z_bin_efficiency_tol=0.01,
        )


def test_prepare_arc_constraint_data_rejects_missing_measurements() -> None:
    arcs = pd.DataFrame(
        {
            "arc_id": ["arc_A"],
            "arc_anchor_ra": [10.0],
            "arc_anchor_dec": [20.0],
            "z_arc": [2.0],
            "arc_tangent_angle_rad": [0.5],
            "arc_curvature_arcsec_inv": [0.1],
            "arc_sigma_tangent_angle_rad": [0.05],
            "arc_reliability": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        cluster_solver._prepare_arc_constraint_data(arcs, reference=(3, 10.0, 20.0))


def test_cab_diagnostic_loglikes_match_arc_catalog_loglike_terms() -> None:
    arc_data = ArcConstraintData(
        arc_ids=["arc_A", "arc_B"],
        z_arc=np.asarray([2.0, -1.0], dtype=float),
        anchor_x=np.asarray([0.0, 1.0], dtype=float),
        anchor_y=np.asarray([0.0, 0.0], dtype=float),
        tangent_angle_rad=np.asarray([0.20, 0.0], dtype=float),
        curvature_arcsec_inv=np.asarray([0.08, 0.0], dtype=float),
        sigma_tangent_angle_rad=np.asarray([0.05, 0.05], dtype=float),
        sigma_curvature_arcsec_inv=np.asarray([0.02, 0.02], dtype=float),
        reliability=np.asarray([0.8, 1.0], dtype=float),
    )
    traced_arcs = cluster_solver.TracedArcConstraintData(
        arc_ids=("arc_A", "arc_B"),
        z_arc=jnp.asarray(arc_data.z_arc, dtype=jnp.float64),
        anchor_x=jnp.asarray(arc_data.anchor_x, dtype=jnp.float64),
        anchor_y=jnp.asarray(arc_data.anchor_y, dtype=jnp.float64),
        tangent_angle_rad=jnp.asarray(arc_data.tangent_angle_rad, dtype=jnp.float64),
        curvature_arcsec_inv=jnp.asarray(arc_data.curvature_arcsec_inv, dtype=jnp.float64),
        sigma_tangent_angle_rad=jnp.asarray(arc_data.sigma_tangent_angle_rad, dtype=jnp.float64),
        sigma_curvature_arcsec_inv=jnp.asarray(arc_data.sigma_curvature_arcsec_inv, dtype=jnp.float64),
        reliability=jnp.asarray(arc_data.reliability, dtype=jnp.float64),
        n_arcs=2,
    )
    prediction = cluster_solver._CabMorphologyPrediction(
        tangent_angle_rad=jnp.asarray([[0.18, 1.25], [0.0, 1.0]], dtype=jnp.float64),
        curvature_arcsec_inv=jnp.asarray([[0.10, 0.30], [0.0, 0.0]], dtype=jnp.float64),
        branch_weight=jnp.asarray([[0.90, 0.10], [0.50, 0.50]], dtype=jnp.float64),
        frame_weight=jnp.asarray([[0.95, 0.95], [0.0, 0.0]], dtype=jnp.float64),
        finite=jnp.asarray([[True, True], [True, True]]),
    )

    class FakeCabEvaluator:
        cab_likelihood_weight = 0.4
        cab_tangent_sigma_floor_rad = cluster_solver.DEFAULT_CAB_TANGENT_SIGMA_FLOOR_RAD
        cab_curvature_sigma_floor_arcsec_inv = cluster_solver.DEFAULT_CAB_CURVATURE_SIGMA_FLOOR_ARCSEC_INV
        cab_finite_difference_step_arcsec = cluster_solver.DEFAULT_CAB_FINITE_DIFFERENCE_STEP_ARCSEC
        state = SimpleNamespace(arc_data=arc_data)

        def _physical_parameter_vector(self, params):
            return params

        def _build_cab_packed_lens_state_with_validity_from_physical(self, *_args, **_kwargs):
            return {}, SimpleNamespace(is_valid=jnp.asarray(True))

        def _fit_component_indices(self):
            return None

        def _prepare_traced_arc_constraint_data(self, _arc_data):
            return traced_arcs

        def _cab_morphology_predictions_for_anchors(self, *_args, **_kwargs):
            return prediction

    evaluator = FakeCabEvaluator()
    details = cluster_solver.ClusterJAXEvaluator._cab_morphology_details_for_arcs(
        evaluator,
        np.asarray([0.0], dtype=float),
    )
    direct = cluster_solver.ClusterJAXEvaluator._cab_morphology_loglike_for_arcs(
        evaluator,
        traced_arcs,
        {},
    )

    assert details["cab_finite"].tolist() == [True, True]
    assert float(np.sum(details["cab_loglike"])) == pytest.approx(float(direct))
    assert details["arc_id"].astype(str).tolist() == ["arc_A", "arc_B"]


def test_load_best_par_rejects_invalid_image_label(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.1a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "invalid_label.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Invalid image label '2\.1a'.*expected FAMILY\.image"):
        load_best_par(par_path)


def test_load_best_par_allows_family_redshifts_equal_after_rounding(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.000000001 25.0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.000000002 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "consistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    _parsed, _potentials_df, images_df, _arcs_df, _potentials_with_priors = load_best_par(par_path)

    assert images_df["family_id"].astype(str).tolist() == ["2", "2"]
    assert images_df["image_id"].astype(str).tolist() == ["a", "b"]


def test_load_best_par_rejects_inconsistent_family_redshifts(tmp_path: Path) -> None:
    image_catalog_path = tmp_path / "obs_arcs.dat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.1 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "inconsistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    end

image
    multfile 1 obs_arcs.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        load_best_par(par_path)


def test_load_best_par_rejects_cross_catalog_family_redshift_mismatch(tmp_path: Path) -> None:
    first_catalog_path = tmp_path / "obs_a.dat"
    first_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.a 10.0000 20.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    second_catalog_path = tmp_path / "obs_b.dat"
    second_catalog_path.write_text(
        "#REFERENCE 0\n"
        "2.b 10.0002 20.0000 1.0 1.0 0.0 2.1 25.0\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "cross_catalog_inconsistent_redshift.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 20.0
    image 1 obs_a.dat
    end

image
    multfile 1 obs_b.dat
    end
fini
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        load_best_par(par_path)


def test_prepare_family_data_rejects_inconsistent_family_redshift_backstop() -> None:
    images = pd.DataFrame(
        {
            "family_id": ["2", "2"],
            "image_label": ["2.a", "2.b"],
            "catalog_z": [2.0, 2.1],
            "catalog_source": ["manual", "manual"],
            "ra": [10.0, 10.0002],
            "dec": [20.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match=r"Family 2 has inconsistent catalog_z values \[2.0, 2.1\]"):
        cluster_solver._prepare_family_data(
            images,
            sigma_arcsec=0.5,
            reference=(3, 10.0, 20.0),
            z_lens=0.3,
            cosmo_config={},
            z_bin_efficiency_tol=0.01,
        )


def test_filter_non_positive_redshift_families_keeps_singletons_separate() -> None:
    images = pd.DataFrame(
        {
            "family_id": ["valid", "valid", "zero", "zero", "negative", "negative", "single"],
            "image_label": ["v.1", "v.2", "z.1", "z.2", "n.1", "n.2", "s.1"],
            "catalog_z": [2.0, 2.0, 0.0, 0.0, -1.0, -1.0, 3.0],
        }
    )

    filtered, n_images, n_families, family_ids = cluster_solver._filter_non_positive_redshift_families(images)

    assert n_images == 4
    assert n_families == 2
    assert family_ids == ["negative", "zero"]
    assert filtered["family_id"].astype(str).tolist() == ["valid", "valid", "single"]

    singleton_filtered, n_singleton_images, n_singleton_families = cluster_solver._filter_singleton_families(filtered)
    assert n_singleton_images == 1
    assert n_singleton_families == 1
    assert singleton_filtered["family_id"].astype(str).tolist() == ["valid", "valid"]


def test_fov_limit_mask_combines_radius_and_order_insensitive_bounds() -> None:
    limit = cluster_solver._fov_limit_from_args(
        argparse.Namespace(
            fov_limit_radius=5.0,
            fov_limit_x=[2.0, -2.0],
            fov_limit_y=[1.0, -1.0],
        )
    )

    assert limit is not None
    mask = cluster_solver._fov_mask_from_offsets(
        np.asarray([0.0, 2.0, -2.0, 0.0, 3.0, 5.0]),
        np.asarray([0.0, 0.0, 0.0, 2.0, 4.0, 0.0]),
        limit,
    )

    assert mask.tolist() == [True, True, True, False, False, False]


def test_input_archive_copies_raw_text_inputs_and_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    par_path = source_dir / "input.par"
    member_path = source_dir / "members.cat"
    image_path = source_dir / "obs_arcs.cat"
    arc_path = source_dir / "arc_constraints.cat"
    par_path.write_text("par file\n", encoding="utf-8")
    member_path.write_text("members\n", encoding="utf-8")
    image_path.write_text("images\n", encoding="utf-8")
    arc_path.write_text("arcs\n", encoding="utf-8")
    state = _minimal_stage4_surrogate_state()
    state.run_name = "archive_run"
    state.par_path = str(par_path)
    state.parsed = {
        "image": {
            "multfile": [1, "obs_arcs.cat"],
            "arcfile": [1, "arc_constraints.cat"],
        }
    }
    state.potfiles = [{"id": "potfile", "catalog_path": str(member_path)}]
    truth_path = tmp_path / "truth.fits"
    truth_path.write_bytes(b"fits")
    args = argparse.Namespace(
        kappa_true_fits=str(truth_path),
        gammax_true_fits=None,
        gammay_true_fits=None,
        image_catalog_family_cutout_image_dir=None,
        image_catalog_family_cutout_bands=None,
        quiet=True,
    )
    run_dir = tmp_path / "results" / "archive_run"

    manifest = cluster_solver._archive_run_inputs(args, state, run_dir)

    archive_dir = run_dir / "inputs"
    manifest_path = archive_dir / "input_manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == manifest
    copied = {item["kind"]: item for item in loaded["copied_files"]}
    assert set(copied) == {"par", "potfile", "image_catalog", "arc_catalog"}
    assert (archive_dir / copied["par"]["archived_path"]).read_text(encoding="utf-8") == "par file\n"
    assert (archive_dir / copied["potfile"]["archived_path"]).read_text(encoding="utf-8") == "members\n"
    assert (archive_dir / copied["image_catalog"]["archived_path"]).read_text(encoding="utf-8") == "images\n"
    assert (archive_dir / copied["arc_catalog"]["archived_path"]).read_text(encoding="utf-8") == "arcs\n"
    assert copied["par"]["sha256"] == hashlib.sha256(par_path.read_bytes()).hexdigest()
    assert copied["par"]["size_bytes"] == par_path.stat().st_size
    assert loaded["skipped_large_files"] == [
        {
            "arg": "kappa_true_fits",
            "exists": True,
            "is_file": True,
            "kind": "kappa_true_fits",
            "path": str(truth_path.resolve()),
            "size_bytes": 4,
        }
    ]


def test_input_archive_skips_config_native_par_sentinel(tmp_path: Path) -> None:
    image_path = tmp_path / "obs_arcs.cat"
    image_path.write_text("images\n", encoding="utf-8")
    state = _minimal_stage4_surrogate_state()
    state.run_name = "archive_config_run"
    state.par_path = cluster_solver.CONFIG_NATIVE_PAR_SENTINEL
    state.parsed = {"image": {"multfile": [1, str(image_path)]}}
    args = argparse.Namespace(
        kappa_true_fits=None,
        gammax_true_fits=None,
        gammay_true_fits=None,
        image_catalog_family_cutout_image_dir=None,
        image_catalog_family_cutout_bands=None,
        quiet=True,
    )

    manifest = cluster_solver._archive_run_inputs(args, state, tmp_path / "results" / "archive_config_run")

    copied = {item["kind"]: item for item in manifest["copied_files"]}
    assert set(copied) == {"image_catalog"}


def test_input_archive_handles_duplicate_basenames_and_missing_optional_arc(tmp_path: Path) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    par_path = source_a / "input.par"
    first_member_path = source_a / "members.cat"
    second_member_path = source_b / "members.cat"
    image_path = source_a / "obs.cat"
    par_path.write_text("par\n", encoding="utf-8")
    first_member_path.write_text("first\n", encoding="utf-8")
    second_member_path.write_text("second\n", encoding="utf-8")
    image_path.write_text("image\n", encoding="utf-8")
    state = _minimal_stage4_surrogate_state()
    state.run_name = "archive_duplicates"
    state.par_path = str(par_path)
    state.parsed = {"image": {"multfile": [1, "obs.cat"]}}
    state.potfiles = [
        {"id": "potfile", "catalog_path": str(first_member_path)},
        {"id": "potfile2", "catalog_path": str(second_member_path)},
        {"id": "duplicate_same_path", "catalog_path": str(first_member_path)},
    ]
    args = argparse.Namespace(
        kappa_true_fits=None,
        gammax_true_fits=None,
        gammay_true_fits=None,
        image_catalog_family_cutout_image_dir=None,
        image_catalog_family_cutout_bands=None,
        quiet=True,
    )

    manifest = cluster_solver._archive_run_inputs(args, state, tmp_path / "results" / "archive_duplicates")

    potfile_entries = [item for item in manifest["copied_files"] if item["kind"] == "potfile"]
    assert [item["archived_path"] for item in potfile_entries] == [
        "potfiles/members.cat",
        "potfiles/members__2.cat",
    ]
    assert not any(item["kind"] == "arc_catalog" for item in manifest["copied_files"])


def test_potfile_member_brightest_filter_sorts_by_mag_and_broadcasts() -> None:
    potfiles = [
        {
            "id": "potfile_a",
            "catalog_df": pd.DataFrame(
                {
                    "id": ["faint", "bright_b", "bright_a"],
                    "catalog_mag": [21.0, 18.0, 18.0],
                }
            ),
        },
        {
            "id": "potfile_b",
            "catalog_df": pd.DataFrame(
                {
                    "id": ["only"],
                    "catalog_mag": [22.0],
                }
            ),
        },
    ]

    filtered, summary = cluster_solver._filter_potfiles_by_brightest_members(potfiles, [2])

    assert filtered[0]["catalog_df"]["id"].astype(str).tolist() == ["bright_a", "bright_b"]
    assert filtered[1]["catalog_df"]["id"].astype(str).tolist() == ["only"]
    assert summary == {
        "potfile_a": {"total": 3, "kept": 2, "dropped": 1, "n": 2},
        "potfile_b": {"total": 1, "kept": 1, "dropped": 0, "n": 2},
    }


def test_potfile_member_brightest_filter_validates_counts_and_magnitudes() -> None:
    potfiles = [
        {"id": "potfile_a", "catalog_df": pd.DataFrame({"id": ["a"], "catalog_mag": [20.0]})},
        {"id": "potfile_b", "catalog_df": pd.DataFrame({"id": ["b"], "catalog_mag": [21.0]})},
    ]

    assert cluster_solver._normalize_potfile_member_brightest_counts([3, 4], potfiles) == [3, 4]
    with pytest.raises(ValueError, match="exactly one value per potfile"):
        cluster_solver._normalize_potfile_member_brightest_counts([1, 2, 3], potfiles)
    with pytest.raises(ValueError, match="positive integers"):
        cluster_solver._normalize_potfile_member_brightest_counts([0], potfiles)
    with pytest.raises(ValueError, match="catalog_mag contains non-finite"):
        cluster_solver._filter_potfiles_by_brightest_members(
            [{"id": "bad", "catalog_df": pd.DataFrame({"id": ["a"], "catalog_mag": [np.nan]})}],
            [1],
        )


def test_potfile_member_mag_max_filter_is_inclusive_and_broadcasts() -> None:
    potfiles = [
        {
            "id": "potfile_a",
            "catalog_df": pd.DataFrame(
                {
                    "id": ["bright", "edge", "faint"],
                    "catalog_mag": [19.0, 22.0, 22.1],
                }
            ),
        },
        {
            "id": "potfile_b",
            "catalog_df": pd.DataFrame(
                {
                    "id": ["kept", "dropped"],
                    "catalog_mag": [21.5, 23.0],
                }
            ),
        },
    ]

    filtered, summary = cluster_solver._filter_potfiles_by_member_mag_max(potfiles, [22.0])

    assert filtered[0]["catalog_df"]["id"].astype(str).tolist() == ["bright", "edge"]
    assert filtered[1]["catalog_df"]["id"].astype(str).tolist() == ["kept"]
    assert summary == {
        "potfile_a": {"total": 3, "kept": 2, "dropped": 1, "mag_max": 22.0},
        "potfile_b": {"total": 2, "kept": 1, "dropped": 1, "mag_max": 22.0},
    }


def test_potfile_member_mag_max_filter_validates_values_and_magnitudes() -> None:
    potfiles = [
        {"id": "potfile_a", "catalog_df": pd.DataFrame({"id": ["a"], "catalog_mag": [20.0]})},
        {"id": "potfile_b", "catalog_df": pd.DataFrame({"id": ["b"], "catalog_mag": [21.0]})},
    ]

    assert cluster_solver._normalize_potfile_member_mag_max_values([20.0, 21.0], potfiles) == [20.0, 21.0]
    with pytest.raises(ValueError, match="exactly one value per potfile"):
        cluster_solver._normalize_potfile_member_mag_max_values([20.0, 21.0, 22.0], potfiles)
    with pytest.raises(ValueError, match="must be finite"):
        cluster_solver._normalize_potfile_member_mag_max_values([float("nan")], potfiles)
    with pytest.raises(ValueError, match="catalog_mag contains non-finite"):
        cluster_solver._filter_potfiles_by_member_mag_max(
            [{"id": "bad", "catalog_df": pd.DataFrame({"id": ["a"], "catalog_mag": [np.nan]})}],
            [22.0],
        )


def test_multi_cluster_solver_parses_fov_limit_args() -> None:
    args = multi_cluster_solver._parse_args(
        [
            "--cluster",
            "a",
            "a.par",
            "warm/a",
            "--cluster",
            "b",
            "b.par",
            "warm/b",
            "--fov-limit-radius",
            "30",
            "--fov-limit-x",
            "12",
            "-12",
            "--fov-limit-y",
            "-8",
            "8",
        ]
    )

    assert args.fov_limit_radius == pytest.approx(30.0)
    assert args.fov_limit_x == [12.0, -12.0]
    assert args.fov_limit_y == [-8.0, 8.0]


def test_generate_single_bcg_mock_with_subhalos_uses_potfile(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        seed=11,
        n_primary_families=1,
        n_subhalos=8,
        pos_sigma_arcsec=0.0,
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
        caustic_compute_window_arcsec=80.0,
        caustic_grid_scale_arcsec=2.0,
        mock_image_candidate_batch_size=8,
    )

    paths, images, truth = generate_single_bcg_mock(tmp_path, config)
    parsed, _potentials_df, images_df, _arcs_df, potentials_with_priors = load_best_par(paths.par_path)

    assert (tmp_path / "members.cat").exists()
    assert len(parsed["potfiles"]) == 1
    potfile = parsed["potfiles"][0]
    realized_config = mock_generation._caustic_config_from_truth(truth)
    assert len(potfile["catalog_df"]) == config.n_subhalos
    assert potfile["core_arcsec"] == pytest.approx(float(realized_config.subhalo_core_radius_arcsec), abs=1.0e-8)
    assert potfile["sigma_nominal"] == pytest.approx(realized_config.subhalo_sigma_ref)
    assert potfile["cut_arcsec_nominal"] == pytest.approx(float(realized_config.subhalo_cut_radius_arcsec), abs=1.0e-8)
    assert parsed["grille"]["nlens"] == 2 + config.n_subhalos
    assert parsed["grille"]["nlens_opt"] == 2
    image_radii = np.hypot(images["x_obs_arcsec"].to_numpy(dtype=float), images["y_obs_arcsec"].to_numpy(dtype=float))
    subhalo_radii = np.asarray(
        [np.hypot(row["x_arcsec"], row["y_arcsec"]) for row in truth["subhalos"]],
        dtype=float,
    )
    assert parsed["champ"]["dmax"] == int(np.ceil(max(image_radii.max(), subhalo_radii.max()) + 10.0))
    assert len(potentials_with_priors) == 2
    assert images_df["family_id"].nunique() == 1
    assert len(truth["subhalos"]) == config.n_subhalos
    assert len(truth["subhalo_components"]) == config.n_subhalos
    assert len(truth["lens_model_list"]) == 2 + config.n_subhalos
    truth_kwargs = truth["kwargs_lens_by_source_redshift"]["2.00000000"]
    assert len(truth_kwargs) == len(truth["lens_model_list"])
    assert all(float(row["sigma0"]) > 0.0 for row in truth_kwargs[2:])
    truth_components = truth["lens_components"]
    assert len(truth_components) == 2 + config.n_subhalos
    assert [row["component_id"] for row in truth_components[:2]] == ["halo", "bcg"]
    subhalo_truth_components = truth_components[2:]
    assert [row["component_role"] for row in subhalo_truth_components] == ["subhalo"] * config.n_subhalos
    assert {row["catalog_id"] for row in subhalo_truth_components} == {row["id"] for row in truth["subhalos"]}
    assert all(row.get("component_role") != "subhalo_placeholder" for row in truth_components)
    assert all(float(row["catalog_mag"]) <= config.subhalo_mag_faint_limit for row in truth["subhalos"])
    assert all("subhalo_mass_msun" in row for row in truth["subhalos"])
    assert all("subhalo_parent_rank" in row for row in truth["subhalos"])
    # Physical dPIE mass of each realized member profile (scaling relation + scatter),
    # distinct from the Schechter abundance label above.
    truth_kpc_per_arcsec = float(truth["kpc_per_arcsec"])
    components_by_id = {row["potential_id"]: row for row in truth["subhalo_components"]}
    for row in truth["subhalos"]:
        component = components_by_id[row["id"]]
        expected_mass = 10.0 ** float(
            log10_dpie_mass_msun(
                component["v_disp"],
                float(component["cut_radius_arcsec"]) * truth_kpc_per_arcsec,
            )
        )
        assert row["subhalo_dpie_mass_msun"] == pytest.approx(expected_mass)
    assert all(row["selected_by_mag_cut"] is True for row in truth["subhalos"])
    selection = truth["subhalo_selection"]
    assert selection["schechter_alpha"] == pytest.approx(config.subhalo_schechter_alpha)
    assert selection["luminosity_star"] == pytest.approx(1.0)
    assert selection["luminosity_peak"] == pytest.approx(config.subhalo_schechter_alpha + 1.0)
    assert selection["mass_luminosity_exponent"] == pytest.approx(1.0 + realized_config.subhalo_gamma_ml)
    assert selection["max_member_theta_e_arcsec"] > 0.0
    assert selection["pos_sigma_arcsec"] == pytest.approx(config.pos_sigma_arcsec)
    assert math.isinf(selection["max_member_theta_e_over_pos_sigma"])
    assert truth["parameter_truth"]["potfile.sigma"] == pytest.approx(realized_config.subhalo_sigma_ref)
    assert truth["parameter_truth"]["potfile.corekpc"] == pytest.approx(
        float(realized_config.subhalo_core_radius_arcsec) * truth_kpc_per_arcsec
    )
    assert truth["parameter_truth"]["potfile.cutkpc"] == pytest.approx(
        float(realized_config.subhalo_cut_radius_arcsec) * truth_kpc_per_arcsec
    )
    assert truth["parameter_truth"]["potfile.dpie_mass_ref_msun"] == pytest.approx(
        float(realized_config.sampled_subhalo_dpie_mass_ref_msun)
    )
    assert truth["parameter_truth"]["potfile.alpha_sigma"] == pytest.approx(realized_config.subhalo_alpha_sigma)
    assert truth["parameter_truth"]["potfile.gamma_ml"] == pytest.approx(realized_config.subhalo_gamma_ml)
    assert truth["parameter_truth"]["potfile.beta_radius"] == pytest.approx(
        mock_generation._subhalo_beta_radius_from_config(realized_config)
    )
    assert truth["parameter_truth"]["potfile.sampled_alpha_sigma"] == pytest.approx(
        realized_config.sampled_subhalo_alpha_sigma
    )
    assert truth["parameter_truth"]["potfile.sampled_gamma_ml"] == pytest.approx(
        realized_config.sampled_subhalo_gamma_ml
    )
    assert "potfile.sigma_log_scatter" not in truth["parameter_truth"]
    assert "potfile.mass_log_scatter" not in truth["parameter_truth"]
    assert selection["mass_peak"] == pytest.approx(
        realized_config.subhalo_mass_ref
        * (realized_config.subhalo_schechter_alpha + 1.0) ** selection["mass_luminosity_exponent"]
    )
    assert selection["dpie_mass_anchor_msun"] == pytest.approx(
        float(realized_config.sampled_subhalo_dpie_mass_ref_msun)
    )
    for row in truth["subhalos"]:
        luminosity = float(row["luminosity_ratio"])
        assert row["subhalo_dpie_mass_msun"] / selection["dpie_mass_anchor_msun"] == pytest.approx(
            luminosity ** (1.0 + realized_config.subhalo_gamma_ml)
        )
    assert selection["parent_count"] >= config.n_subhalos
    assert selection["selected_count"] == config.n_subhalos
    candidates = selection["candidates"]
    selected_candidates = [row for row in candidates if row["selected"]]
    assert len(selected_candidates) == config.n_subhalos
    assert all(row["passes_mag_cut"] for row in selected_candidates)
    candidate_indices = {row["subhalo_candidate_index"] for row in candidates}
    selected_indices = {row["subhalo_candidate_index"] for row in selected_candidates}
    assert {row["subhalo_candidate_index"] for row in truth["subhalos"]} == selected_indices
    assert selected_indices <= candidate_indices
    assert {source["caustic_class"] for source in truth["sources"]} == {"primary"}
    assert np.isfinite(images["magnification_true"].to_numpy(dtype=float)).all()


def test_subhalo_schechter_sampler_has_log_space_peak_and_high_mass_tail() -> None:
    config = SingleBCGMockConfig()
    rng = np.random.default_rng(123)
    luminosity_min, luminosity_max = mock_generation._subhalo_schechter_luminosity_bounds(config)

    luminosities = mock_generation._sample_truncated_schechter_luminosities(
        rng,
        180_000,
        luminosity_min=luminosity_min,
        luminosity_max=luminosity_max,
        alpha=config.subhalo_schechter_alpha,
    )
    masses = mock_generation._subhalo_mass_from_luminosity_ratio(luminosities, config)

    assert np.isfinite(luminosities).all()
    assert luminosities.min() >= luminosity_min
    assert luminosities.max() <= luminosity_max
    assert np.isfinite(masses).all()
    assert masses.min() >= config.subhalo_mass_min
    assert masses.max() <= config.subhalo_mass_max
    log_edges = np.linspace(np.log10(luminosity_min), np.log10(luminosity_max), 40)
    counts, edges = np.histogram(np.log10(luminosities), bins=log_edges)
    peak_center = 10.0 ** (0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]))

    assert peak_center == pytest.approx(config.subhalo_schechter_alpha + 1.0, rel=0.25)
    assert np.count_nonzero(masses > config.subhalo_mass_ref) > 0


def test_subhalo_schechter_selection_draws_from_observable_distribution() -> None:
    config = SingleBCGMockConfig(n_subhalos=8, subhalo_parent_factor=5)

    subhalos, selection = mock_generation._generate_schechter_subhalo_candidates(config, np.random.default_rng(0))

    assert len(subhalos) == config.n_subhalos
    assert selection is not None
    candidates = selection["candidates"]
    selected = [row for row in candidates if row["selected"]]
    unselected_observable = [row for row in candidates if not row["selected"] and row["passes_mag_cut"]]
    assert len(selected) == config.n_subhalos
    assert all(row["catalog_mag"] <= config.subhalo_mag_faint_limit for row in selected)
    assert all(row["passes_mag_cut"] for row in selected)
    assert unselected_observable

    faintest_selected_mag = max(float(row["catalog_mag"]) for row in selected)
    brightest_unselected_mag = min(float(row["catalog_mag"]) for row in unselected_observable)
    assert brightest_unselected_mag < faintest_selected_mag


def test_subhalo_schechter_luminosity_to_mass_conversion_uses_dpie_scaling() -> None:
    config = SingleBCGMockConfig()
    exponent = mock_generation._subhalo_mass_luminosity_exponent(config)
    luminosity_peak = config.subhalo_schechter_alpha + 1.0
    one_mag_fainter_luminosity = 10.0**-0.4
    luminosities = np.asarray([1.0, luminosity_peak, 10.0, one_mag_fainter_luminosity])

    masses = mock_generation._subhalo_mass_from_luminosity_ratio(luminosities, config)
    mags = mock_generation._subhalo_magnitude_from_luminosity_ratio(luminosities, config)

    assert config.subhalo_alpha_sigma == pytest.approx(0.25)
    assert config.subhalo_gamma_ml == pytest.approx(0.0)
    assert mock_generation._subhalo_beta_radius_from_config(config) == pytest.approx(0.50)
    assert exponent == pytest.approx(1.0)
    assert masses[0] == pytest.approx(config.subhalo_mass_ref)
    assert mags[0] == pytest.approx(config.subhalo_mag0)
    assert masses[1] == pytest.approx(config.subhalo_mass_ref * luminosity_peak**exponent)
    assert masses[2] > masses[0]
    assert mags[2] < mags[0]
    assert masses[3] / masses[0] == pytest.approx(10.0**-0.4)
    assert mags[3] == pytest.approx(config.subhalo_mag0 + 1.0)


def test_subhalo_schechter_selection_raises_when_mag_cut_rejects_candidates() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        n_subhalos=3,
        subhalo_mag_faint_limit=-100.0,
    )

    with pytest.raises(RuntimeError, match="Failed to select 3 Schechter subhalos"):
        mock_generation._generate_subhalo_catalog(config, np.random.default_rng(5))


def test_subhalo_spatial_default_compact_gamma_preserves_forced_core() -> None:
    config = SingleBCGMockConfig(n_primary_families=1, n_subhalos=8, subhalo_parent_factor=5)

    subhalos, selection = mock_generation._generate_subhalo_catalog_payload(config, np.random.default_rng(4))

    radii = np.asarray([math.hypot(float(row["x_arcsec"]), float(row["y_arcsec"])) for row in subhalos])
    assert len(subhalos) == config.n_subhalos
    assert np.all(radii[: config.subhalo_force_core_count] >= config.subhalo_force_core_min_arcsec)
    assert np.all(radii[: config.subhalo_force_core_count] <= config.subhalo_force_core_max_arcsec)
    assert selection is not None
    assert selection["spatial_distribution"] == "compact_gamma"
    assert selection["force_core_count"] == config.subhalo_force_core_count


def test_subhalo_spatial_dpie_positions_are_finite_and_within_field() -> None:
    config = SingleBCGMockConfig(
        n_primary_families=1,
        n_subhalos=32,
        subhalo_parent_factor=5,
        subhalo_spatial_distribution="dpie",
        subhalo_field_radius_arcsec=250.0,
        subhalo_spatial_core_radius_arcsec=25.0,
        subhalo_spatial_cut_radius_arcsec=180.0,
        subhalo_force_core_count=0,
    )

    subhalos, selection = mock_generation._generate_subhalo_catalog_payload(config, np.random.default_rng(8))

    radii = np.asarray([math.hypot(float(row["x_arcsec"]), float(row["y_arcsec"])) for row in subhalos])
    assert len(subhalos) == config.n_subhalos
    assert np.isfinite(radii).all()
    assert np.all(radii >= 0.0)
    assert np.all(radii <= config.subhalo_field_radius_arcsec)
    assert selection is not None
    assert selection["spatial_distribution"] == "dpie"
    assert selection["spatial_core_radius_arcsec"] == pytest.approx(config.subhalo_spatial_core_radius_arcsec)
    assert selection["spatial_cut_radius_arcsec"] == pytest.approx(config.subhalo_spatial_cut_radius_arcsec)


def test_subhalo_spatial_dpie_is_broader_than_compact_gamma() -> None:
    common = dict(n_primary_families=1, n_subhalos=300, subhalo_parent_factor=3)
    compact = SingleBCGMockConfig(seed=3, **common)
    dpie = SingleBCGMockConfig(
        seed=3,
        subhalo_spatial_distribution="dpie",
        subhalo_field_radius_arcsec=250.0,
        subhalo_spatial_core_radius_arcsec=25.0,
        subhalo_spatial_cut_radius_arcsec=180.0,
        subhalo_force_core_count=0,
        **common,
    )

    compact_radii = mock_generation._sample_subhalo_radii(compact, np.random.default_rng(12), compact.n_subhalos)
    dpie_radii = mock_generation._sample_subhalo_radii(dpie, np.random.default_rng(12), dpie.n_subhalos)

    assert np.median(dpie_radii) > np.median(compact_radii)
    assert np.count_nonzero(dpie_radii > compact.subhalo_field_radius_arcsec) > 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"subhalo_spatial_distribution": "unknown"}, "subhalo_spatial_distribution"),
        ({"subhalo_field_radius_arcsec": 0.0}, "subhalo_field_radius_arcsec"),
        ({"subhalo_spatial_core_radius_arcsec": 30.0, "subhalo_spatial_cut_radius_arcsec": 30.0}, "must exceed"),
        ({"subhalo_force_core_count": -1}, "subhalo_force_core_count"),
        ({"subhalo_force_core_min_arcsec": 20.0, "subhalo_force_core_max_arcsec": 10.0}, "must exceed"),
        ({"subhalo_field_radius_arcsec": 10.0, "subhalo_force_core_max_arcsec": 18.0}, "must not exceed"),
        ({"subhalo_dpie_mass_ref_min_msun": 0.0}, "subhalo_dpie_mass_ref"),
        (
            {"subhalo_dpie_mass_ref_min_msun": 1.0e12, "subhalo_dpie_mass_ref_max_msun": 1.0e12},
            "subhalo_dpie_mass_ref",
        ),
        ({"subhalo_dpie_mass_pivot_msun": 0.0}, "subhalo_dpie_mass_pivot_msun"),
        ({"subhalo_core_radius_pivot_kpc": float("nan")}, "subhalo_core_radius_pivot_kpc"),
        ({"subhalo_core_radius_pivot_kpc": 0.001}, "potfile.corekpc truth inside solver prior support"),
        ({"subhalo_core_radius_mass_slope": -0.1}, "subhalo_core_radius_mass_slope"),
        ({"subhalo_core_radius_scatter_dex": -0.1}, "subhalo_core_radius_scatter_dex"),
        ({"subhalo_cut_radius_pivot_kpc": 0.0}, "subhalo_cut_radius_pivot_kpc"),
        ({"subhalo_cut_radius_mass_slope": -0.1}, "subhalo_cut_radius_mass_slope"),
        ({"subhalo_cut_radius_scatter_dex": -0.1}, "subhalo_cut_radius_scatter_dex"),
        ({"sampled_subhalo_dpie_mass_ref_msun": -1.0}, "sampled_subhalo_dpie_mass_ref_msun"),
        ({"mock_image_candidate_batch_size": 0}, "mock_image_candidate_batch_size"),
        ({"mock_image_seed_cap": 0}, "mock_image_seed_cap"),
        ({"mock_image_lm_max_iter": 0}, "mock_image_lm_max_iter"),
        ({"mock_generation_workers": 0}, "mock_generation_workers"),
        ({"mock_image_search_window_arcsec": 0.0}, "mock_image_search_window_arcsec"),
        ({"mock_image_precision_limit": 0.0}, "mock_image_precision_limit"),
        ({"mock_caustic_grid_chunk_memory_gb": 0.0}, "mock_caustic_grid_chunk_memory_gb"),
        ({"mock_caustic_grid_chunk_memory_gb": -1.0}, "mock_caustic_grid_chunk_memory_gb"),
        ({"mock_caustic_grid_chunk_memory_gb": float("nan")}, "mock_caustic_grid_chunk_memory_gb"),
        ({"mock_caustic_grid_chunk_memory_gb": float("inf")}, "mock_caustic_grid_chunk_memory_gb"),
        ({"subhalo_alpha_sigma": float("nan")}, "subhalo_alpha_sigma"),
        ({"subhalo_gamma_ml": float("nan")}, "subhalo_gamma_ml"),
        ({"subhalo_alpha_sigma_half_width": -0.1}, "subhalo_alpha_sigma_half_width"),
        ({"subhalo_gamma_ml_half_width": float("nan")}, "subhalo_gamma_ml_half_width"),
        (
            {"subhalo_alpha_sigma": 0.49, "subhalo_alpha_sigma_half_width": 0.02, "subhalo_gamma_ml": 0.0},
            "alpha/gamma windows",
        ),
        ({"sampled_subhalo_alpha_sigma": float("nan")}, "sampled_subhalo_alpha_sigma"),
        ({"sampled_subhalo_gamma_ml": float("nan")}, "sampled_subhalo_gamma_ml"),
        ({"subhalo_alpha_sigma": 0.6, "subhalo_gamma_ml": 0.0}, "nonpositive beta_radius"),
        ({"source_magnitude_min": float("nan")}, "source_magnitude_min/source_magnitude_max"),
        ({"source_magnitude_max": float("inf")}, "source_magnitude_min/source_magnitude_max"),
        ({"source_magnitude_min": 27.0, "source_magnitude_max": 27.0}, "source_magnitude_min/source_magnitude_max"),
        ({"image_magnitude_sigma_mag": 0.0}, "image_magnitude_sigma_mag"),
        ({"image_magnitude_sigma_mag": float("nan")}, "image_magnitude_sigma_mag"),
        ({"image_magnitude_mu_floor": 0.0}, "image_magnitude_mu_floor"),
        ({"image_magnitude_mu_floor": float("inf")}, "image_magnitude_mu_floor"),
        ({"halo_position_half_width_arcsec": -1.0}, "halo_position_half_width_arcsec"),
        ({"halo_position_half_width_arcsec": float("nan")}, "halo_position_half_width_arcsec"),
        ({"bcg_position_offset_radius_arcsec": -1.0}, "bcg_position_offset_radius_arcsec"),
        ({"bcg_position_offset_radius_arcsec": float("inf")}, "bcg_position_offset_radius_arcsec"),
        ({"total_large_component_mass_min_msun": 0.0}, "total_large_component_mass"),
        (
            {"total_large_component_mass_min_msun": 5.0e14, "total_large_component_mass_max_msun": 5.0e14},
            "total_large_component_mass",
        ),
        ({"large_component_mass_pivot_msun": 0.0}, "large_component_mass_pivot_msun"),
        ({"halo_core_radius_pivot_kpc": 0.0}, "halo_core_radius_pivot_kpc"),
        ({"halo_core_radius_mass_slope": -0.1}, "halo_core_radius_mass_slope"),
        ({"halo_core_radius_scatter_dex": -0.1}, "halo_core_radius_scatter_dex"),
        ({"halo_cut_radius_pivot_kpc": float("nan")}, "halo_cut_radius_pivot_kpc"),
        ({"halo_cut_radius_mass_slope": -0.1}, "halo_cut_radius_mass_slope"),
        ({"halo_cut_radius_scatter_dex": -0.1}, "halo_cut_radius_scatter_dex"),
        ({"halo_cut_radius_prior_fractional_half_width": 0.0}, "halo_cut_radius_prior_fractional_half_width"),
        ({"halo_cut_radius_prior_fractional_half_width": -0.1}, "halo_cut_radius_prior_fractional_half_width"),
        ({"halo_cut_radius_prior_fractional_half_width": 1.0}, "halo_cut_radius_prior_fractional_half_width"),
        ({"halo_cut_radius_prior_fractional_half_width": float("nan")}, "halo_cut_radius_prior_fractional_half_width"),
        ({"bcg_mass_fraction_median": 0.0}, "bcg_mass_fraction"),
        ({"bcg_mass_fraction_min": 0.1, "bcg_mass_fraction_max": 0.05}, "bcg_mass_fraction"),
        ({"bcg_mass_fraction_min": 0.005, "bcg_mass_fraction_median": 0.2, "bcg_mass_fraction_max": 0.08}, "bcg_mass_fraction"),
        ({"bcg_mass_fraction_max": 1.0}, "bcg_mass_fraction"),
        ({"bcg_mass_fraction_scatter_dex": -0.1}, "bcg_mass_fraction_scatter_dex"),
        ({"bcg_core_radius_pivot_kpc": float("nan")}, "bcg_core_radius_pivot_kpc"),
        ({"bcg_core_radius_mass_slope": -0.1}, "bcg_core_radius_mass_slope"),
        ({"bcg_core_radius_scatter_dex": -0.1}, "bcg_core_radius_scatter_dex"),
        ({"bcg_cut_radius_pivot_kpc": 0.0}, "bcg_cut_radius_pivot_kpc"),
        ({"bcg_mass_pivot_msun": float("inf")}, "bcg_mass_pivot_msun"),
        ({"bcg_cut_radius_mass_slope": -0.1}, "bcg_cut_radius_mass_slope"),
        ({"bcg_cut_radius_scatter_dex": -0.1}, "bcg_cut_radius_scatter_dex"),
        ({"bcg_cut_radius_prior_fractional_half_width": 0.0}, "bcg_cut_radius_prior_fractional_half_width"),
        ({"bcg_cut_radius_prior_fractional_half_width": -0.1}, "bcg_cut_radius_prior_fractional_half_width"),
        ({"bcg_cut_radius_prior_fractional_half_width": 1.0}, "bcg_cut_radius_prior_fractional_half_width"),
        ({"bcg_cut_radius_prior_fractional_half_width": float("inf")}, "bcg_cut_radius_prior_fractional_half_width"),
        ({"sampled_total_large_component_mass_msun": -1.0}, "sampled_total_large_component_mass_msun"),
        ({"sampled_bcg_mass_fraction": 1.0}, "sampled_bcg_mass_fraction"),
        (
            {"subhalo_cut_radius_prior_fractional_half_width": 0.0},
            "subhalo_cut_radius_prior_fractional_half_width",
        ),
        (
            {"subhalo_cut_radius_prior_fractional_half_width": -0.1},
            "subhalo_cut_radius_prior_fractional_half_width",
        ),
        (
            {"subhalo_cut_radius_prior_fractional_half_width": 1.0},
            "subhalo_cut_radius_prior_fractional_half_width",
        ),
        (
            {"subhalo_cut_radius_prior_fractional_half_width": float("nan")},
            "subhalo_cut_radius_prior_fractional_half_width",
        ),
    ],
)
def test_subhalo_spatial_invalid_config_raises(updates: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SingleBCGMockConfig(n_primary_families=1, **updates).validate()


def test_caustic_config_from_truth_rejects_legacy_subhalo_beta_radius() -> None:
    with pytest.raises(ValueError, match="subhalo_beta_radius"):
        mock_generation._caustic_config_from_truth({"config": {"subhalo_beta_radius": 0.5}})


def test_subhalo_spatial_config_round_trips_from_truth() -> None:
    truth = {
        "config": SingleBCGMockConfig(
            n_primary_families=1,
            halo=mock_generation.DPIETruth(
                potential_id="halo",
                x_centre=4.2,
                y_centre=-3.1,
                ellipticite=0.28,
                angle_pos=35.0,
                core_radius_arcsec=5.0,
                cut_radius_arcsec=220.0,
                v_disp=760.0,
            ),
            bcg=mock_generation.DPIETruth(
                potential_id="bcg",
                x_centre=4.8,
                y_centre=-2.6,
                ellipticite=0.12,
                angle_pos=10.0,
                core_radius_arcsec=0.15,
                cut_radius_arcsec=35.0,
                v_disp=285.0,
            ),
            halo_position_half_width_arcsec=10.0,
            bcg_position_offset_radius_arcsec=2.0,
            total_large_component_mass_min_msun=5.0e14,
            total_large_component_mass_max_msun=2.0e15,
            large_component_mass_pivot_msun=8.0e14,
            halo_core_radius_pivot_kpc=28.0,
            halo_core_radius_mass_slope=0.2,
            halo_core_radius_scatter_dex=0.02,
            halo_cut_radius_pivot_kpc=1200.0,
            halo_cut_radius_mass_slope=0.4,
            halo_cut_radius_scatter_dex=0.03,
            halo_cut_radius_prior_fractional_half_width=0.07,
            bcg_mass_fraction_median=0.025,
            bcg_mass_fraction_scatter_dex=0.12,
            bcg_mass_fraction_min=0.01,
            bcg_mass_fraction_max=0.07,
            bcg_core_radius_pivot_kpc=0.9,
            bcg_core_radius_mass_slope=0.15,
            bcg_core_radius_scatter_dex=0.05,
            bcg_cut_radius_pivot_kpc=190.0,
            bcg_mass_pivot_msun=1.8e13,
            bcg_cut_radius_mass_slope=0.25,
            bcg_cut_radius_scatter_dex=0.04,
            bcg_cut_radius_prior_fractional_half_width=0.08,
            sampled_total_large_component_mass_msun=7.5e14,
            sampled_halo_mass_msun=7.3e14,
            sampled_bcg_mass_msun=2.0e13,
            sampled_bcg_mass_fraction=2.0e13 / 7.5e14,
            subhalo_spatial_distribution="dpie",
            subhalo_field_radius_arcsec=250.0,
            subhalo_spatial_core_radius_arcsec=25.0,
            subhalo_spatial_cut_radius_arcsec=180.0,
            subhalo_force_core_count=0,
            subhalo_dpie_mass_ref_min_msun=2.0e11,
            subhalo_dpie_mass_ref_max_msun=2.0e12,
            subhalo_dpie_mass_pivot_msun=8.0e11,
            subhalo_core_radius_pivot_kpc=0.002,
            subhalo_core_radius_mass_slope=0.2,
            subhalo_core_radius_scatter_dex=0.03,
            subhalo_cut_radius_pivot_kpc=120.0,
            subhalo_cut_radius_mass_slope=0.4,
            subhalo_cut_radius_scatter_dex=0.05,
            sampled_subhalo_dpie_mass_ref_msun=7.0e11,
            subhalo_mag0=18.0,
            subhalo_sigma_ref=110.0,
            subhalo_sigma_ref_std=12.0,
            subhalo_core_radius_arcsec=0.0004,
            subhalo_cut_radius_arcsec=24.0,
            subhalo_cut_lower_arcsec=6.0,
            subhalo_cut_upper_arcsec=90.0,
            subhalo_cut_radius_prior_fractional_half_width=0.09,
            subhalo_alpha_sigma=0.22,
            subhalo_alpha_sigma_half_width=0.03,
            subhalo_gamma_ml=0.04,
            subhalo_gamma_ml_half_width=0.06,
            sampled_subhalo_alpha_sigma=0.23,
            sampled_subhalo_gamma_ml=0.02,
            source_magnitude_min=23.5,
            source_magnitude_max=26.5,
            image_magnitude_sigma_mag=0.12,
            image_magnitude_mu_floor=2.0e-3,
        ).to_json_dict()
    }

    config = mock_generation._caustic_config_from_truth(truth)

    assert config.subhalo_spatial_distribution == "dpie"
    assert config.subhalo_field_radius_arcsec == pytest.approx(250.0)
    assert config.subhalo_spatial_core_radius_arcsec == pytest.approx(25.0)
    assert config.subhalo_spatial_cut_radius_arcsec == pytest.approx(180.0)
    assert config.subhalo_force_core_count == 0
    assert config.subhalo_dpie_mass_ref_min_msun == pytest.approx(2.0e11)
    assert config.subhalo_dpie_mass_ref_max_msun == pytest.approx(2.0e12)
    assert config.subhalo_dpie_mass_pivot_msun == pytest.approx(8.0e11)
    assert config.subhalo_core_radius_pivot_kpc == pytest.approx(0.002)
    assert config.subhalo_core_radius_mass_slope == pytest.approx(0.2)
    assert config.subhalo_core_radius_scatter_dex == pytest.approx(0.03)
    assert config.subhalo_cut_radius_pivot_kpc == pytest.approx(120.0)
    assert config.subhalo_cut_radius_mass_slope == pytest.approx(0.4)
    assert config.subhalo_cut_radius_scatter_dex == pytest.approx(0.05)
    assert config.sampled_subhalo_dpie_mass_ref_msun == pytest.approx(7.0e11)
    assert config.subhalo_mag0 == pytest.approx(18.0)
    assert config.subhalo_sigma_ref == pytest.approx(110.0)
    assert config.subhalo_sigma_ref_std == pytest.approx(12.0)
    assert config.subhalo_core_radius_arcsec == pytest.approx(0.0004)
    assert config.subhalo_cut_radius_arcsec == pytest.approx(24.0)
    assert config.subhalo_cut_lower_arcsec == pytest.approx(6.0)
    assert config.subhalo_cut_upper_arcsec == pytest.approx(90.0)
    assert config.subhalo_cut_radius_prior_fractional_half_width == pytest.approx(0.09)
    assert config.subhalo_alpha_sigma == pytest.approx(0.22)
    assert config.subhalo_alpha_sigma_half_width == pytest.approx(0.03)
    assert config.subhalo_gamma_ml == pytest.approx(0.04)
    assert config.subhalo_gamma_ml_half_width == pytest.approx(0.06)
    assert config.sampled_subhalo_alpha_sigma == pytest.approx(0.23)
    assert config.sampled_subhalo_gamma_ml == pytest.approx(0.02)
    assert config.halo.x_centre == pytest.approx(4.2)
    assert config.halo.y_centre == pytest.approx(-3.1)
    assert config.bcg.x_centre == pytest.approx(4.8)
    assert config.bcg.y_centre == pytest.approx(-2.6)
    assert config.halo_position_half_width_arcsec == pytest.approx(10.0)
    assert config.bcg_position_offset_radius_arcsec == pytest.approx(2.0)
    assert config.total_large_component_mass_min_msun == pytest.approx(5.0e14)
    assert config.total_large_component_mass_max_msun == pytest.approx(2.0e15)
    assert config.large_component_mass_pivot_msun == pytest.approx(8.0e14)
    assert config.halo_core_radius_pivot_kpc == pytest.approx(28.0)
    assert config.halo_core_radius_mass_slope == pytest.approx(0.2)
    assert config.halo_core_radius_scatter_dex == pytest.approx(0.02)
    assert config.halo_cut_radius_pivot_kpc == pytest.approx(1200.0)
    assert config.halo_cut_radius_mass_slope == pytest.approx(0.4)
    assert config.halo_cut_radius_scatter_dex == pytest.approx(0.03)
    assert config.halo_cut_radius_prior_fractional_half_width == pytest.approx(0.07)
    assert config.bcg_mass_fraction_median == pytest.approx(0.025)
    assert config.bcg_mass_fraction_scatter_dex == pytest.approx(0.12)
    assert config.bcg_mass_fraction_min == pytest.approx(0.01)
    assert config.bcg_mass_fraction_max == pytest.approx(0.07)
    assert config.bcg_core_radius_pivot_kpc == pytest.approx(0.9)
    assert config.bcg_core_radius_mass_slope == pytest.approx(0.15)
    assert config.bcg_core_radius_scatter_dex == pytest.approx(0.05)
    assert config.bcg_cut_radius_pivot_kpc == pytest.approx(190.0)
    assert config.bcg_mass_pivot_msun == pytest.approx(1.8e13)
    assert config.bcg_cut_radius_mass_slope == pytest.approx(0.25)
    assert config.bcg_cut_radius_scatter_dex == pytest.approx(0.04)
    assert config.bcg_cut_radius_prior_fractional_half_width == pytest.approx(0.08)
    assert config.sampled_total_large_component_mass_msun == pytest.approx(7.5e14)
    assert config.sampled_halo_mass_msun == pytest.approx(7.3e14)
    assert config.sampled_bcg_mass_msun == pytest.approx(2.0e13)
    assert config.sampled_bcg_mass_fraction == pytest.approx(2.0e13 / 7.5e14)
    assert config.source_magnitude_min == pytest.approx(23.5)
    assert config.source_magnitude_max == pytest.approx(26.5)
    assert config.image_magnitude_sigma_mag == pytest.approx(0.12)
    assert config.image_magnitude_mu_floor == pytest.approx(2.0e-3)


def test_plot_subhalo_shmf_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalo_selection": {
            "schechter_alpha": -0.7,
            "mass_ref": 1.0e12,
            "mass_luminosity_exponent": 1.2,
            "mag_faint_limit": 24.0,
            "candidates": [
                {
                    "subhalo_candidate_index": 0,
                    "subhalo_mass_msun": 1.0e12,
                    "catalog_mag": 17.0,
                    "luminosity_ratio": 1.0,
                    "subhalo_parent_rank": 1,
                    "passes_mag_cut": True,
                    "selected": True,
                    "selected_member_id": "member001",
                },
                {
                    "subhalo_candidate_index": 1,
                    "subhalo_mass_msun": 2.4e11,
                    "catalog_mag": 18.3,
                    "luminosity_ratio": 0.3,
                    "subhalo_parent_rank": 2,
                    "passes_mag_cut": True,
                    "selected": False,
                },
                {
                    "subhalo_candidate_index": 2,
                    "subhalo_mass_msun": 1.0e9,
                    "catalog_mag": 25.4,
                    "luminosity_ratio": 0.003,
                    "subhalo_parent_rank": 3,
                    "passes_mag_cut": False,
                    "selected": False,
                },
            ],
        }
    }
    path = tmp_path / "subhalo_shmf.pdf"

    validation._plot_subhalo_selection(truth, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_shmf_writes_placeholder_without_data(tmp_path: Path) -> None:
    path = tmp_path / "subhalo_shmf.pdf"

    validation._plot_subhalo_selection({}, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_recovered_subhalo_mass_table_uses_recovered_scaling_parameters() -> None:
    packed = SimpleNamespace(
        component_family=np.asarray([0, 1, 1], dtype=int),
        x_center_base=np.asarray([0.0, -3.0, 6.0], dtype=float),
        y_center_base=np.asarray([0.0, 4.0, 8.0], dtype=float),
        luminosity_ratio=np.asarray([1.0, 1.0, 0.25], dtype=float),
        sigma_ref_base=np.asarray([0.0, 245.0, 245.0], dtype=float),
        cut_ref_base=np.asarray([0.0, 40.0, 40.0], dtype=float),
        alpha_sigma_base=np.asarray([0.0, 0.25, 0.25], dtype=float),
        gamma_ml_base=np.asarray([0.0, 0.0, 0.0], dtype=float),
        sigma_ref_param_index=np.asarray([-1, 0, 0], dtype=int),
        cut_ref_param_index=np.asarray([-1, 1, 1], dtype=int),
        alpha_sigma_param_index=np.asarray([-1, 2, 2], dtype=int),
        gamma_ml_param_index=np.asarray([-1, 3, 3], dtype=int),
    )
    state = SimpleNamespace(
        packed_lens_spec=packed,
        scaling_component_records=[
            {
                "component_index": 1,
                "potfile_id": "potfile",
                "catalog_id": "member001",
                "x_centre": 3.0,
                "y_centre": 4.0,
            },
            {"component_index": 2, "potfile_id": "potfile", "catalog_id": "member002"},
        ],
    )
    best_fit = np.asarray([300.0, 60.0, 0.30, 0.60], dtype=float)
    truth = {
        "parameter_truth": {"potfile.sigma": 245.0, "potfile.cutkpc": 40.0},
        "subhalo_selection": {"mass_ref": 1.0e12},
    }

    table = validation._recovered_subhalo_mass_table(state, best_fit, truth)

    # Recovered masses are physical dPIE masses of the fitted member profiles:
    # sigma_i = sigma_ref_fit * L^alpha_fit, cut_i = cut_ref_fit * L^beta_fit,
    # with beta = 1 + gamma_ml - 2*alpha.
    beta = 1.0 + 0.60 - 2.0 * 0.30
    assert table["catalog_id"].tolist() == ["member001", "member002"]
    assert table["recovered_subhalo_mass_msun"].tolist() == pytest.approx(
        [
            float(10.0 ** log10_dpie_mass_msun(300.0, 60.0)),
            float(10.0 ** log10_dpie_mass_msun(300.0 * 0.25**0.30, 60.0 * 0.25**beta)),
        ]
    )
    assert table["x_arcsec"].tolist() == pytest.approx([3.0, 6.0])
    assert table["y_arcsec"].tolist() == pytest.approx([4.0, 8.0])
    assert table["recovered_radius_arcsec"].tolist() == pytest.approx([5.0, 10.0])


def test_plot_absolute_magnification_recovery_writes_pdf(tmp_path: Path) -> None:
    truth_abs_mu_raw = np.abs(
        np.asarray(
            [
                [1.0, -8.0, 120.0],
                [3.0, 12.0, 25.0],
                [np.nan, 45.0, 70.0],
            ]
        )
    )
    recovered_abs_mu_raw = np.abs(
        np.asarray(
            [
                [2.0, -10.0, 100.0],
                [5.0, 8.0, 20.0],
                [4.0, np.inf, 60.0],
            ]
        )
    )
    truth_abs_mu = np.minimum(truth_abs_mu_raw, 25.0)
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, 25.0)
    grid = validation._AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=np.asarray([-1.0, 0.0, 1.0]),
        y_axis_arcsec=np.asarray([-1.0, 0.0, 1.0]),
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=recovered_abs_mu_raw - truth_abs_mu_raw,
        z_source=7.0,
        cap=25.0,
    )
    path = tmp_path / "absolute_magnification_recovery.pdf"

    validation._plot_absolute_magnification_recovery(grid, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_limit_recovery_axis_ticks_replaces_dense_locators(caplog: pytest.LogCaptureFixture) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-13.0, 274.0)
        ax.set_ylim(-13.0, 274.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        validation._limit_recovery_axis_ticks(ax)

        assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)
        assert isinstance(ax.yaxis.get_major_locator(), MaxNLocator)
        with caplog.at_level(logging.WARNING, logger="matplotlib.ticker"):
            fig.canvas.draw()
    finally:
        plt.close(fig)

    assert _locator_max_tick_messages(caplog) == []


def test_plot_absolute_magnification_recovery_limits_ticks_for_large_ranges(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    truth_abs_mu_raw = np.abs(np.asarray([[1.0, 8.0, 120.0], [3.0, 12.0, 25.0], [2.0, 45.0, 70.0]]))
    recovered_abs_mu_raw = np.abs(np.asarray([[2.0, 10.0, 100.0], [5.0, 8.0, 20.0], [4.0, 30.0, 60.0]]))
    truth_abs_mu = np.minimum(truth_abs_mu_raw, 25.0)
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, 25.0)
    grid = validation._AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=np.asarray([-13.0, 130.5, 274.0]),
        y_axis_arcsec=np.asarray([-13.0, 130.5, 274.0]),
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=recovered_abs_mu_raw - truth_abs_mu_raw,
        z_source=7.0,
        cap=25.0,
    )
    path = tmp_path / "absolute_magnification_recovery_large_range.pdf"

    with caplog.at_level(logging.WARNING, logger="matplotlib.ticker"):
        validation._plot_absolute_magnification_recovery(grid, path)

    assert path.exists()
    assert path.stat().st_size > 0
    assert _locator_max_tick_messages(caplog) == []


def test_plot_absolute_magnification_recovery_uses_capped_viridis_and_centered_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import TwoSlopeNorm

    truth_abs_mu_raw = np.abs(np.asarray([[5.0, -80.0], [20.0, np.nan]]))
    recovered_abs_mu_raw = np.abs(np.asarray([[10.0, 100.0], [-5.0, np.inf]]))
    truth_abs_mu = np.minimum(truth_abs_mu_raw, 25.0)
    recovered_abs_mu = np.minimum(recovered_abs_mu_raw, 25.0)
    residual_abs_mu = recovered_abs_mu_raw - truth_abs_mu_raw
    grid = validation._AbsoluteMagnificationRecoveryGrid(
        x_axis_arcsec=np.asarray([-1.0, 1.0]),
        y_axis_arcsec=np.asarray([-1.0, 1.0]),
        truth_abs_mu_raw=truth_abs_mu_raw,
        recovered_abs_mu_raw=recovered_abs_mu_raw,
        truth_abs_mu=truth_abs_mu,
        recovered_abs_mu=recovered_abs_mu,
        residual_abs_mu=residual_abs_mu,
        z_source=7.0,
        cap=25.0,
    )
    imshow_calls: list[tuple[np.ndarray, dict[str, Any]]] = []
    colorbar_labels: list[str] = []
    subplot_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    layout_axes: list[Axes] = []
    xlabels: list[tuple[Axes, str]] = []
    ylabels: list[tuple[Axes, str]] = []
    tick_params_calls: list[tuple[Axes, dict[str, Any]]] = []
    original_subplots = validation.plt.subplots
    original_imshow = Axes.imshow
    original_colorbar_set_label = Colorbar.set_label
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel
    original_tick_params = Axes.tick_params

    def record_subplots(*args: Any, **kwargs: Any) -> Any:
        subplot_calls.append((args, dict(kwargs)))
        fig, axes = original_subplots(*args, **kwargs)
        layout_axes.extend(np.ravel(axes).tolist())
        return fig, axes

    def record_imshow(self: Axes, data: Any, *args: Any, **kwargs: Any) -> Any:
        imshow_calls.append((np.ma.asarray(data).filled(np.nan), dict(kwargs)))
        return original_imshow(self, data, *args, **kwargs)

    def fail_set_title(self: Axes, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("absolute magnification recovery plot should not draw panel titles")

    def record_colorbar_label(self: Colorbar, label: str, *args: Any, **kwargs: Any) -> Any:
        colorbar_labels.append(str(label))
        return original_colorbar_set_label(self, label, *args, **kwargs)

    def record_set_xlabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        xlabels.append((self, str(label)))
        return original_set_xlabel(self, label, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append((self, str(label)))
        return original_set_ylabel(self, label, *args, **kwargs)

    def record_tick_params(self: Axes, *args: Any, **kwargs: Any) -> Any:
        tick_params_calls.append((self, dict(kwargs)))
        return original_tick_params(self, *args, **kwargs)

    monkeypatch.setattr(validation.plt, "subplots", record_subplots)
    monkeypatch.setattr(Axes, "imshow", record_imshow)
    monkeypatch.setattr(Axes, "set_title", fail_set_title)
    monkeypatch.setattr(Colorbar, "set_label", record_colorbar_label)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "tick_params", record_tick_params)

    validation._plot_absolute_magnification_recovery(grid, tmp_path / "absolute_magnification_recovery.pdf")

    assert subplot_calls == [((2, 1), {"sharex": True, "figsize": (6.2, 8.6)})]
    assert len(layout_axes) == 2
    assert len(imshow_calls) == 2
    left_data, left_kwargs = imshow_calls[0]
    right_data, right_kwargs = imshow_calls[1]
    assert np.nanmax(left_data) <= 25.0
    assert left_kwargs["cmap"] == "viridis"
    assert left_kwargs["vmin"] == 0.0
    assert left_kwargs["vmax"] == 25.0
    np.testing.assert_allclose(right_data, residual_abs_mu, equal_nan=True)
    assert right_kwargs["cmap"] == "RdBu"
    assert isinstance(right_kwargs["norm"], TwoSlopeNorm)
    assert right_kwargs["norm"].vmin == pytest.approx(-25.0)
    assert right_kwargs["norm"].vcenter == pytest.approx(0.0)
    assert right_kwargs["norm"].vmax == pytest.approx(25.0)
    nonempty_colorbar_labels = [label for label in colorbar_labels if label]
    assert nonempty_colorbar_labels == [
        r"$|\mu_{\rm rec}|$",
        r"$|\mu_{\rm rec}| - |\mu_{\rm truth}|$",
    ]
    assert not any("capped at" in label for label in nonempty_colorbar_labels)
    assert not any("min(" in label for label in nonempty_colorbar_labels)
    panel_xlabels = [(ax, label) for ax, label in xlabels if ax in layout_axes]
    panel_ylabels = [(ax, label) for ax, label in ylabels if ax in layout_axes]
    assert layout_axes[0].get_xlabel() == ""
    assert layout_axes[1].get_xlabel() == "x (arcsec)"
    assert not any(ax is layout_axes[0] and label == "x (arcsec)" for ax, label in panel_xlabels)
    assert any(ax is layout_axes[1] and label == "x (arcsec)" for ax, label in panel_xlabels)
    assert panel_ylabels == [(layout_axes[0], "y (arcsec)"), (layout_axes[1], "y (arcsec)")]
    assert any(
        ax is layout_axes[0]
        and kwargs.get("axis") == "x"
        and kwargs.get("labelbottom") is False
        for ax, kwargs in tick_params_calls
    )


def test_absolute_magnification_grid_source_redshift_prefers_z9_then_highest() -> None:
    assert validation._absolute_magnification_grid_source_redshift(
        {"kwargs_lens_by_source_redshift": {"2.00000000": [], "9.00000000": []}}
    ) == pytest.approx(9.0)
    assert validation._absolute_magnification_grid_source_redshift(
        {"kwargs_lens_by_source_redshift": {"2.00000000": [], "5.00000000": []}}
    ) == pytest.approx(5.0)
    assert validation._absolute_magnification_grid_source_redshift(
        {"config": {"source_redshift": 3.0}}
    ) == pytest.approx(3.0)


def test_plot_image_residual_histogram_writes_pdf(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [0.02, 0.05, 0.08, 0.13]})
    path = tmp_path / "image_residual_histogram.pdf"

    validation._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_image_residual_histogram_reports_point_and_arc_aware_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "not_recovered", "not_recovered", "recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "not_recovered", "point_recovered"],
            "arc_supported": [False, True, False, False],
            "image_residual_arcsec": [9.0, 9.0, 0.30, 9.0],
            "image_residual_q50": [0.04, np.nan, 0.08, np.inf],
            "point_image_residual_arcsec": [0.04, np.nan, np.nan, 9.0],
            "arc_aware_image_residual_arcsec": [0.04, np.nan, np.nan, 9.0],
            "arc_curve_distance_arcsec": [np.nan, 0.20, 0.07, np.nan],
        }
    )
    hist_values: list[np.ndarray] = []
    hist_bins: list[Any] = []
    axvline_values: list[float] = []
    texts: list[str] = []
    xlabels: list[str] = []
    ylabels: list[str] = []
    original_hist = Axes.hist
    original_axvline = Axes.axvline
    original_text = Axes.text
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_values.append(np.asarray(values, dtype=float))
        hist_bins.append(kwargs.get("bins"))
        return original_hist(self, values, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_text(self: Axes, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 3:
            texts.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    def record_set_xlabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        xlabels.append(str(label))
        return original_set_xlabel(self, label, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append(str(label))
        return original_set_ylabel(self, label, *args, **kwargs)

    def fail_set_title(self: Axes, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("image residual histogram should not draw a title")

    monkeypatch.setattr(Axes, "hist", record_hist)
    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "text", record_text)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "set_title", fail_set_title)

    validation._plot_image_residual_histogram(image_df, tmp_path / "image_residual_histogram.pdf")

    assert len(hist_values) == 2
    np.testing.assert_allclose(hist_values[0], [0.04, 9.0])
    np.testing.assert_allclose(hist_values[1], [0.04, 0.20, 9.0])
    assert hist_bins == [30, 30]
    assert axvline_values[0] == pytest.approx(float(np.sqrt(np.mean(np.square(hist_values[0])))))
    assert axvline_values[1] == pytest.approx(float(np.sqrt(np.mean(np.square(hist_values[1])))))
    assert any("Point RMS" in text and "(2/4)" in text and "Arc-aware RMS" in text and "(3/4)" in text for text in texts)
    assert any("arc-supported = 1/4" in text and "missed = 1/4" in text for text in texts)
    assert xlabels == ["image residual (arcsec)"]
    assert ylabels == ["N images"]


def test_plot_image_residual_histogram_uses_posterior_rms_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered"],
            "image_residual_arcsec": [8.0, 8.0],
            "image_residual_q50": [0.1, 0.2],
        }
    )
    posterior_rms_df = pd.DataFrame(
        {
            "draw_index": [1, 2, 3],
            "point_image_rms_arcsec": [0.3, 0.5, 0.7],
            "point_recovered_image_count": [2, 2, 2],
            "arc_aware_image_rms_arcsec": [0.25, 0.45, 0.65],
            "arc_aware_recovered_image_count": [2, 2, 2],
            "arc_supported_image_count": [0, 0, 0],
            "total_image_count": [2, 2, 2],
            "exact_failed_family_count": [0, 0, 0],
        }
    )
    axvline_values: list[float] = []
    axvspan_values: list[tuple[float, float]] = []
    texts: list[str] = []
    original_axvline = Axes.axvline
    original_axvspan = Axes.axvspan
    original_text = Axes.text

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_axvspan(self: Axes, xmin: Any, xmax: Any, *args: Any, **kwargs: Any) -> Any:
        axvspan_values.append((float(xmin), float(xmax)))
        return original_axvspan(self, xmin, xmax, *args, **kwargs)

    def record_text(self: Axes, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 3:
            texts.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "axvspan", record_axvspan)
    monkeypatch.setattr(Axes, "text", record_text)

    validation._plot_image_residual_histogram(
        image_df,
        tmp_path / "image_residual_histogram.pdf",
        posterior_image_rms_df=posterior_rms_df,
    )

    assert axvline_values[0] == pytest.approx(0.5)
    assert axvline_values[1] == pytest.approx(0.45)
    assert axvspan_values[0] == pytest.approx((0.364, 0.636))
    assert axvspan_values[1] == pytest.approx((0.314, 0.586))
    assert any("0.5 +/-" in text and "0.45 +/-" in text for text in texts)


def test_posterior_residual_histogram_summaries_compute_bin_count_quantiles() -> None:
    posterior_residual_draws_df = pd.DataFrame(
        {
            "draw_index": [1, 1, 2, 2, 3, 3],
            "image_label": ["1.1", "1.2", "1.1", "1.2", "1.1", "1.2"],
            "family_id": ["1", "1", "1", "1", "1", "1"],
            "point_image_residual_arcsec": [0.1, np.nan, 0.1, 0.6, np.nan, 0.6],
            "arc_aware_image_residual_arcsec": [0.1, 0.6, 0.1, 0.6, np.nan, 0.6],
            "point_recovered": [True, False, True, True, False, True],
            "arc_aware_recovered": [True, True, True, True, False, True],
            "arc_supported": [False, True, False, False, False, False],
            "exact_failed": [False, False, False, False, False, False],
        }
    )

    summaries = validation._posterior_residual_histogram_summaries(posterior_residual_draws_df, bin_count=2)

    np.testing.assert_allclose(summaries["point"]["median"], [1.0, 1.0])
    np.testing.assert_allclose(summaries["point"]["q16"], [0.32, 0.32])
    np.testing.assert_allclose(summaries["point"]["q84"], [1.0, 1.0])
    np.testing.assert_allclose(summaries["arc_aware"]["median"], [1.0, 1.0])
    np.testing.assert_allclose(summaries["arc_aware"]["q16"], [0.32, 1.0])
    np.testing.assert_allclose(summaries["arc_aware"]["q84"], [1.0, 1.0])


def test_plot_image_residual_histogram_uses_posterior_bin_count_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported"],
            "image_residual_arcsec": [0.1, np.nan],
            "arc_aware_image_residual_arcsec": [0.1, 0.6],
        }
    )
    posterior_residual_draws_df = pd.DataFrame(
        {
            "draw_index": [1, 1, 2, 2, 3, 3],
            "image_label": ["1.1", "1.2", "1.1", "1.2", "1.1", "1.2"],
            "family_id": ["1", "1", "1", "1", "1", "1"],
            "point_image_residual_arcsec": [0.1, np.nan, 0.1, 0.6, np.nan, 0.6],
            "arc_aware_image_residual_arcsec": [0.1, 0.6, 0.1, 0.6, np.nan, 0.6],
            "point_recovered": [True, False, True, True, False, True],
            "arc_aware_recovered": [True, True, True, True, False, True],
            "arc_supported": [False, True, False, False, False, False],
            "exact_failed": [False, False, False, False, False, False],
        }
    )
    bars: list[np.ndarray] = []
    errorbars: list[tuple[np.ndarray, Any]] = []
    ylabels: list[str] = []
    hist_calls: list[Any] = []
    original_bar = Axes.bar
    original_errorbar = Axes.errorbar
    original_set_ylabel = Axes.set_ylabel
    original_hist = Axes.hist

    def record_bar(self: Axes, x: Any, height: Any, *args: Any, **kwargs: Any) -> Any:
        bars.append(np.asarray(height, dtype=float))
        return original_bar(self, x, height, *args, **kwargs)

    def record_errorbar(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        errorbars.append((np.asarray(y, dtype=float), kwargs.get("yerr")))
        return original_errorbar(self, x, y, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append(str(label))
        return original_set_ylabel(self, label, *args, **kwargs)

    def record_hist(self: Axes, *args: Any, **kwargs: Any) -> Any:
        hist_calls.append(args)
        return original_hist(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", record_bar)
    monkeypatch.setattr(Axes, "errorbar", record_errorbar)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "hist", record_hist)

    validation._plot_image_residual_histogram(
        image_df,
        tmp_path / "image_residual_histogram.pdf",
        posterior_image_residual_draws_df=posterior_residual_draws_df,
    )

    assert not hist_calls
    assert bars
    np.testing.assert_allclose(bars[0], validation._posterior_residual_histogram_summaries(posterior_residual_draws_df)["point"]["median"])
    assert len(errorbars) == 2
    assert ylabels == ["posterior median N images"]


def test_plot_image_recovery_value_fallback_uses_best_fit_for_nan_q50() -> None:
    image_df = pd.DataFrame(
        {
            "x_model_arcsec": [1.0, 2.0, 3.0],
            "x_model_q50": [np.nan, 20.0, np.inf],
        }
    )

    values = validation._plot_value_with_fallback(image_df, "x_model_q50", "x_model_arcsec")

    np.testing.assert_allclose(values, [1.0, 20.0, 3.0])


def test_plot_image_residual_histogram_writes_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [np.nan, np.inf, -np.inf]})
    path = tmp_path / "image_residual_histogram.pdf"

    validation._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_critical_arc_support_histogram_uses_q50_and_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [9.0, 9.0],
            "image_residual_q50": [0.4, 0.8],
            "arc_aware_image_residual_arcsec": [0.3, 0.2],
            "arc_aware_image_residual_q50": [0.02, np.nan],
            "arc_curve_distance_arcsec": [0.04, 0.45],
            "arc_noncritical_direction_residual_arcsec": [0.01, 0.25],
            "arc_critical_direction_residual_arcsec": [4.0, 0.5],
            "arc_s_min": [0.04, 0.5],
            "arc_prior_probability": [0.75, 0.12],
            "p_arc": [0.82, 0.31],
            "arc_recovery_status": ["arc_supported", "not_recovered"],
            "arc_supported": [True, False],
        }
    )
    hist_values: list[np.ndarray] = []
    scatter_xy: list[tuple[np.ndarray, np.ndarray]] = []
    axvline_values: list[float] = []
    axhline_values: list[float] = []
    original_hist = Axes.hist
    original_scatter = Axes.scatter
    original_axvline = Axes.axvline
    original_axhline = Axes.axhline

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_values.append(np.asarray(values, dtype=float))
        return original_hist(self, values, *args, **kwargs)

    def record_scatter(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        scatter_xy.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_scatter(self, x, y, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_axhline(self: Axes, y: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axhline_values.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", record_hist)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "axhline", record_axhline)

    validation._plot_critical_arc_support_histogram(
        image_df,
        tmp_path / "critical_arc_support_histogram.pdf",
    )

    assert any(np.allclose(values, [0.02, 0.2]) for values in hist_values)
    assert any(np.allclose(values, [0.04, 0.45]) for values in hist_values)
    expected_log_s_min = np.log10(np.asarray([0.04, 0.5], dtype=float))
    assert any(np.allclose(np.sort(values), np.sort(expected_log_s_min)) for values in hist_values)
    probability_points = [
        (x_values, y_values)
        for x_values, y_values in scatter_xy
        if y_values.size and np.all(np.isin(np.round(y_values, 8), np.round([0.82, 0.31], 8)))
    ]
    all_probability_x = np.concatenate([item[0] for item in probability_points])
    all_probability_y = np.concatenate([item[1] for item in probability_points])
    np.testing.assert_allclose(np.sort(all_probability_x), np.sort(expected_log_s_min))
    np.testing.assert_allclose(np.sort(all_probability_y), [0.31, 0.82])
    assert any(value == pytest.approx(float(np.log10(0.2))) for value in axvline_values)
    assert any(value == pytest.approx(0.1) for value in axhline_values)


@pytest.mark.parametrize(
    "sample_likelihood_mode",
    [
        SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    ],
)
def test_plotting_arc_aware_diagnostics_enabled_for_critical_arc_modes(sample_likelihood_mode: str) -> None:
    assert plotting._uses_arc_aware_diagnostics(sample_likelihood_mode)


def test_image_catalog_arc_candidate_support_uses_point_one_default_threshold() -> None:
    assert not plotting._image_catalog_arc_candidate_supported(pd.Series({"p_arc": 0.09}))
    assert plotting._image_catalog_arc_candidate_supported(pd.Series({"p_arc": 0.10}))
    assert not plotting._image_catalog_arc_candidate_supported(
        pd.Series({"p_arc": 0.10, "arc_recovery_p_arc_threshold": 0.5})
    )


def test_image_catalog_marker_label_prefers_image_label() -> None:
    assert plotting._format_image_catalog_marker_label(
        pd.Series({"family_id": "10", "image_label": "10.b"})
    ) == "10.b"
    assert plotting._format_image_catalog_marker_label(pd.Series({"family_id": "10", "image_label": ""})) == "10"


def test_image_catalog_cluster_overview_draws_observed_image_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    original_text = plotting.plt.Axes.text

    def record_text(self: Any, x: Any, y: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        labels.append(str(text))
        return original_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(plotting.plt.Axes, "text", record_text)
    monkeypatch.setattr(plotting, "_image_catalog_cluster_overview_geometry", lambda *_args, **_kwargs: (0.0, 0.0, 10.0))
    monkeypatch.setattr(plotting, "_arcsec_to_skycoord", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(plotting, "_image_catalog_draw_rgb_cutout", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=float))
    monkeypatch.setattr(plotting, "_draw_image_catalog_observed_row_overlays", lambda *args, **kwargs: None)
    monkeypatch.setattr(plotting, "_add_image_catalog_axis_legend", lambda *args, **kwargs: None)

    pixel_positions = iter([(20.0, 30.0), (55.0, 70.0)])

    def fake_cutout_pixel_xy(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
        return next(pixel_positions)

    monkeypatch.setattr(plotting, "_cutout_pixel_xy", fake_cutout_pixel_xy)

    fig, ax = plotting.plt.subplots()
    try:
        plotting._draw_image_catalog_cluster_overview_panel(
            ax,
            helpers=SimpleNamespace(),
            band_images={},
            bands=("F435W", "F606W", "F814W"),
            rgb_display=object(),
            display_image=object(),
            catalog_df=pd.DataFrame(
                {
                    "family_id": ["1", "10"],
                    "image_label": ["1.a", "10.b"],
                    "x_obs_arcsec": [0.0, 1.0],
                    "y_obs_arcsec": [0.0, 1.0],
                }
            ),
            extra_df=pd.DataFrame(),
            reference=(0, 0.0, 0.0),
        )
    finally:
        plotting.plt.close(fig)

    assert "1.a" in labels
    assert "10.b" in labels


def test_plot_critical_arc_support_histogram_overlays_configured_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.02, 0.08, 0.3],
            "arc_prior_probability": [0.65, 0.45, 0.22],
            "p_arc": [0.72, 0.47, 0.18],
            "arc_curve_distance_arcsec": [0.03, 0.2, 0.8],
            "image_recovery_status": ["not_recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["arc_supported", "point_recovered", "not_recovered"],
            "arc_supported": [True, False, False],
        }
    )
    sigmoid_curves: list[tuple[np.ndarray, np.ndarray]] = []
    original_plot = Axes.plot

    def record_plot(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("label") == "p_arc sigmoid":
            sigmoid_curves.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    validation._shared_plot_critical_arc_support_histogram(
        image_df,
        tmp_path / "critical_arc_support_histogram.pdf",
        arc_recovery_p_arc_threshold=0.5,
        critical_arc_base_prob=0.20,
        critical_arc_max_prob=0.70,
        singular_threshold=0.08,
        singular_softness=0.03,
    )

    assert len(sigmoid_curves) == 1
    x_values, y_values = sigmoid_curves[0]
    s_values = np.power(10.0, x_values)
    expected = 0.20 + (0.70 - 0.20) / (1.0 + np.exp(-((0.08 - s_values) / 0.03)))
    np.testing.assert_allclose(y_values, expected, rtol=1.0e-12, atol=1.0e-12)


def test_plot_critical_arc_support_histogram_overlays_anisotropic_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.02, 0.08, 0.3],
            "arc_prior_probability": [0.73, 0.5, 0.001],
            "p_arc": [0.73, 0.5, 0.001],
            "arc_curve_distance_arcsec": [0.03, 0.2, 0.8],
            "image_recovery_status": ["not_recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["arc_supported", "point_recovered", "not_recovered"],
            "arc_supported": [True, False, False],
        }
    )
    sigmoid_curves: list[tuple[np.ndarray, np.ndarray]] = []
    original_plot = Axes.plot

    def record_plot(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("label") == "p_arc sigmoid":
            sigmoid_curves.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    validation._shared_plot_critical_arc_support_histogram(
        image_df,
        tmp_path / "critical_arc_support_histogram.pdf",
        arc_recovery_p_arc_threshold=0.5,
        critical_arc_base_prob=0.20,
        critical_arc_max_prob=0.70,
        singular_threshold=0.08,
        singular_softness=0.03,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    )

    assert len(sigmoid_curves) == 1
    x_values, y_values = sigmoid_curves[0]
    s_values = np.power(10.0, x_values)
    expected = np.clip(1.0 / (1.0 + np.exp(-((0.08 - s_values) / 0.03))), 1.0e-6, 1.0 - 1.0e-6)
    mixture_expected = 0.20 + (0.70 - 0.20) * expected
    np.testing.assert_allclose(y_values, expected, rtol=1.0e-12, atol=1.0e-12)
    assert not np.allclose(y_values, mixture_expected)


def test_plot_critical_arc_support_phase_space_draws_support_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.04, 0.5, np.nan],
            "arc_curve_distance_arcsec": [0.04, 0.45, 0.1],
            "arc_noncritical_direction_residual_arcsec": [0.03, 0.6, 0.1],
            "arc_critical_direction_residual_arcsec": [5.0, 0.1, 1.0],
            "p_arc": [0.82, 0.31, 0.9],
            "image_recovery_status": ["not_recovered", "not_recovered", "recovered"],
            "arc_recovery_status": ["arc_supported", "not_recovered", "point_recovered"],
            "arc_supported": [True, False, False],
        }
    )
    scatter_xy: list[tuple[np.ndarray, np.ndarray]] = []
    axvline_values: list[float] = []
    axhline_values: list[float] = []
    xlabels: list[str] = []
    ylabels: list[str] = []
    original_scatter = Axes.scatter
    original_axvline = Axes.axvline
    original_axhline = Axes.axhline
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel

    def record_scatter(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        scatter_xy.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_scatter(self, x, y, *args, **kwargs)

    def record_axvline(self: Axes, x: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axvline_values.append(float(x))
        return original_axvline(self, x, *args, **kwargs)

    def record_axhline(self: Axes, y: Any = 0, *args: Any, **kwargs: Any) -> Any:
        axhline_values.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    def record_set_xlabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        xlabels.append(str(label))
        return original_set_xlabel(self, label, *args, **kwargs)

    def record_set_ylabel(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        ylabels.append(str(label))
        return original_set_ylabel(self, label, *args, **kwargs)

    def fail_set_title(self: Axes, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("critical-arc phase-space plot should not draw a title")

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Axes, "axvline", record_axvline)
    monkeypatch.setattr(Axes, "axhline", record_axhline)
    monkeypatch.setattr(Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(Axes, "set_title", fail_set_title)

    validation._plot_critical_arc_support_phase_space(
        image_df,
        tmp_path / "critical_arc_support_phase_space.pdf",
    )

    all_x = np.concatenate([item[0] for item in scatter_xy])
    all_y = np.concatenate([item[1] for item in scatter_xy])
    np.testing.assert_allclose(np.sort(all_x), np.sort(np.log10(np.asarray([0.04, 0.5], dtype=float))))
    np.testing.assert_allclose(np.sort(all_y), [0.31, 0.82])
    assert axvline_values == [pytest.approx(float(np.log10(0.2)))]
    assert axhline_values == [pytest.approx(0.1)]
    assert xlabels == [r"$\log_{10} s_{\min}$"]
    assert ylabels == [r"$p_{\rm arc}$"]


def test_plot_critical_arc_support_phase_space_overlays_configured_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.02, 0.08, 0.3],
            "p_arc": [0.72, 0.47, 0.18],
            "image_recovery_status": ["not_recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["arc_supported", "point_recovered", "not_recovered"],
            "arc_supported": [True, False, False],
        }
    )
    sigmoid_curves: list[tuple[np.ndarray, np.ndarray]] = []
    original_plot = Axes.plot

    def record_plot(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("label") == r"$p_{\rm arc}$ sigmoid":
            sigmoid_curves.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    validation._shared_plot_critical_arc_support_phase_space(
        image_df,
        tmp_path / "critical_arc_support_phase_space.pdf",
        arc_recovery_p_arc_threshold=0.5,
        critical_arc_base_prob=0.20,
        critical_arc_max_prob=0.70,
        singular_threshold=0.08,
        singular_softness=0.03,
    )

    assert len(sigmoid_curves) == 1
    x_values, y_values = sigmoid_curves[0]
    s_values = np.power(10.0, x_values)
    expected = 0.20 + (0.70 - 0.20) / (1.0 + np.exp(-((0.08 - s_values) / 0.03)))
    np.testing.assert_allclose(y_values, expected, rtol=1.0e-12, atol=1.0e-12)


def test_plot_critical_arc_support_phase_space_overlays_anisotropic_sigmoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    image_df = pd.DataFrame(
        {
            "arc_s_min": [0.02, 0.08, 0.3],
            "p_arc": [0.73, 0.5, 0.001],
            "image_recovery_status": ["not_recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["arc_supported", "point_recovered", "not_recovered"],
            "arc_supported": [True, False, False],
        }
    )
    sigmoid_curves: list[tuple[np.ndarray, np.ndarray]] = []
    original_plot = Axes.plot

    def record_plot(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("label") == r"$p_{\rm arc}$ sigmoid":
            sigmoid_curves.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    validation._shared_plot_critical_arc_support_phase_space(
        image_df,
        tmp_path / "critical_arc_support_phase_space.pdf",
        arc_recovery_p_arc_threshold=0.5,
        critical_arc_base_prob=0.20,
        critical_arc_max_prob=0.70,
        singular_threshold=0.08,
        singular_softness=0.03,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    )

    assert len(sigmoid_curves) == 1
    x_values, y_values = sigmoid_curves[0]
    s_values = np.power(10.0, x_values)
    expected = np.clip(1.0 / (1.0 + np.exp(-((0.08 - s_values) / 0.03))), 1.0e-6, 1.0 - 1.0e-6)
    mixture_expected = 0.20 + (0.70 - 0.20) * expected
    np.testing.assert_allclose(y_values, expected, rtol=1.0e-12, atol=1.0e-12)
    assert not np.allclose(y_values, mixture_expected)


def test_validation_critical_arc_support_plots_forward_sample_likelihood_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_df = pd.DataFrame({"arc_s_min": [0.08], "p_arc": [0.5]})
    captured: dict[str, str] = {}

    def fake_histogram(_image_df: pd.DataFrame, _path: Path, **kwargs: Any) -> None:
        captured["histogram_mode"] = str(kwargs["sample_likelihood_mode"])
        Path(_path).write_bytes(b"%PDF-1.4\n")

    def fake_phase_space(_image_df: pd.DataFrame, _path: Path, **kwargs: Any) -> None:
        captured["phase_space_mode"] = str(kwargs["sample_likelihood_mode"])
        Path(_path).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(validation, "_shared_plot_critical_arc_support_histogram", fake_histogram)
    monkeypatch.setattr(validation, "_shared_plot_critical_arc_support_phase_space", fake_phase_space)

    artifact_args = {"sample_likelihood_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE}
    validation._plot_critical_arc_support_histogram(
        image_df,
        tmp_path / "critical_arc_support_histogram.pdf",
        artifact_args=artifact_args,
    )
    validation._plot_critical_arc_support_phase_space(
        image_df,
        tmp_path / "critical_arc_support_phase_space.pdf",
        artifact_args=artifact_args,
    )

    assert captured == {
        "histogram_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
        "phase_space_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    }


def test_plot_critical_arc_support_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame({"image_residual_arcsec": [0.1], "arc_s_min": [np.nan]})
    path = tmp_path / "critical_arc_support_histogram.pdf"

    validation._plot_critical_arc_support_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_shmf_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalos": [
            {"subhalo_dpie_mass_msun": 1.0e11},
            {"subhalo_dpie_mass_msun": 3.0e11},
            {"subhalo_dpie_mass_msun": 1.2e12},
        ]
    }
    recovered = pd.DataFrame(
        {
            "recovered_subhalo_mass_msun": [1.2e11, 2.5e11, 9.0e11],
        }
    )
    path = tmp_path / "subhalo_recovery_shmf.pdf"

    validation._plot_subhalo_recovery_shmf(truth, recovered, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_shmf_uses_truth_schechter_without_parent_or_rugs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = {
        "subhalos": [
            {"subhalo_dpie_mass_msun": 1.0e11},
            {"subhalo_dpie_mass_msun": 3.0e11},
            {"subhalo_dpie_mass_msun": 1.2e12},
        ],
        "subhalo_selection": {
            "schechter_alpha": -0.7,
            "mass_ref": 1.0e12,
            "dpie_mass_anchor_msun": 1.0e12,
            "mass_luminosity_exponent": 1.0,
            "candidates": [
                {"subhalo_mass_msun": 5.0e10},
                {"subhalo_mass_msun": 1.0e11},
                {"subhalo_mass_msun": 3.0e11},
                {"subhalo_mass_msun": 1.2e12},
                {"subhalo_mass_msun": 2.0e12},
            ],
        },
    }
    recovered = pd.DataFrame(
        {
            "recovered_subhalo_mass_msun": [1.2e11, 2.5e11, 9.0e11],
        }
    )
    hist_kwargs: list[dict[str, Any]] = []
    plot_labels: list[str | None] = []
    schechter_integrals: list[float] = []
    xlabel_fontsizes: list[float | None] = []
    ylabel_fontsizes: list[float | None] = []
    legend_fontsizes: list[float | None] = []
    legend_locs: list[str | None] = []
    minor_locator_args: list[tuple[Any, ...]] = []
    original_hist = validation.plt.Axes.hist
    original_plot = validation.plt.Axes.plot
    original_set_xlabel = validation.plt.Axes.set_xlabel
    original_set_ylabel = validation.plt.Axes.set_ylabel
    original_legend = validation.plt.Axes.legend
    original_minor_locator = validation.AutoMinorLocator

    def record_hist(self: Any, *args: Any, **kwargs: Any) -> Any:
        hist_kwargs.append(dict(kwargs))
        return original_hist(self, *args, **kwargs)

    def record_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        plot_labels.append(kwargs.get("label"))
        if len(args) >= 2:
            schechter_integrals.append(float(np.trapezoid(np.asarray(args[1], dtype=float), np.asarray(args[0], dtype=float))))
        return original_plot(self, *args, **kwargs)

    def fail_scatter(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo recovery SHMF should not draw scatter rug markers")

    def fail_title(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo recovery SHMF should not draw a title")

    def fail_yscale(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo recovery SHMF should keep a linear y axis")

    def record_set_xlabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        xlabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_xlabel(self, *args, **kwargs)

    def record_set_ylabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        ylabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_ylabel(self, *args, **kwargs)

    def record_legend(self: Any, *args: Any, **kwargs: Any) -> Any:
        legend_fontsizes.append(kwargs.get("fontsize"))
        legend_locs.append(kwargs.get("loc"))
        return original_legend(self, *args, **kwargs)

    def record_minor_locator(*args: Any, **kwargs: Any) -> Any:
        minor_locator_args.append(args)
        return original_minor_locator(*args, **kwargs)

    monkeypatch.setattr(validation.plt.Axes, "hist", record_hist)
    monkeypatch.setattr(validation.plt.Axes, "plot", record_plot)
    monkeypatch.setattr(validation.plt.Axes, "scatter", fail_scatter)
    monkeypatch.setattr(validation.plt.Axes, "set_title", fail_title)
    monkeypatch.setattr(validation.plt.Axes, "set_yscale", fail_yscale)
    monkeypatch.setattr(validation.plt.Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(validation.plt.Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(validation.plt.Axes, "legend", record_legend)
    monkeypatch.setattr(validation, "AutoMinorLocator", record_minor_locator)

    path = tmp_path / "subhalo_recovery_shmf.pdf"

    validation._plot_subhalo_recovery_shmf(truth, recovered, path)

    hist_labels = [kwargs.get("label") for kwargs in hist_kwargs]
    assert "parent candidates" not in hist_labels
    assert "Truth subhalos" in hist_labels
    truth_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "Truth subhalos")
    assert truth_hist["color"] == "lightgray"
    assert truth_hist["alpha"] == pytest.approx(0.8)
    assert any(label is not None and label.startswith("Schechter") for label in plot_labels)
    assert schechter_integrals == [pytest.approx(3.0)]
    assert minor_locator_args == [(5,)]
    assert xlabel_fontsizes == [14]
    assert ylabel_fontsizes == [14]
    assert legend_fontsizes == [11]
    assert legend_locs == ["upper left"]
    assert path.exists()
    assert path.stat().st_size > 0


def _dpie_mass_truth_payload() -> dict[str, Any]:
    return {
        "parameter_truth": {
            "potfile.sigma": 245.0,
            "potfile.cutkpc": 187.0,
            "potfile.alpha_sigma": 0.25,
            "potfile.beta_radius": 0.5,
        },
        "subhalos": [
            {"luminosity_ratio": 1.0, "sigma_log_offset": 0.0, "cut_log_offset": 0.0},
            {"luminosity_ratio": 0.1, "sigma_log_offset": 0.05, "cut_log_offset": 0.0},
        ],
    }


def test_aggregate_plots_module_help_has_no_runpy_warning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "lenscluster.mock_validation.aggregate_plots",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
    assert "Create aggregate plots for mock-validation campaigns." in result.stdout


def test_subhalo_mass_helpers_recompute_dpie_truth_masses() -> None:
    from lenscluster.mock_validation.subhalo_mass import (
        dpie_mass_anchor_from_truth,
        dpie_truth_subhalo_masses_from_scaling,
    )

    truth = _dpie_mass_truth_payload()
    expected = [
        10.0 ** float(log10_dpie_mass_msun(245.0, 187.0)),
        10.0
        ** float(
            log10_dpie_mass_msun(
                245.0 * 0.1**0.25 * math.exp(0.05),
                187.0 * 0.1**0.5,
            )
        ),
    ]
    assert dpie_mass_anchor_from_truth(truth) == pytest.approx(expected[0])
    np.testing.assert_allclose(dpie_truth_subhalo_masses_from_scaling(truth), expected)


def test_aggregate_truth_subhalo_masses_prefer_recorded_dpie_masses() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    payload = {
        "mock_cluster": {
            "truth": {
                "subhalos": [
                    {"subhalo_dpie_mass_msun": 2.0e12, "subhalo_mass_msun": 1.0e11},
                    {"subhalo_dpie_mass_msun": 5.0e11, "subhalo_mass_msun": 3.0e10},
                ]
            }
        }
    }
    np.testing.assert_allclose(aggregate_plots._truth_subhalo_masses(payload), [2.0e12, 5.0e11])


def test_aggregate_truth_subhalo_masses_recompute_for_pre_fix_mocks_and_warn() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    truth = _dpie_mass_truth_payload()
    # Pre-fix mocks recorded only the abundance label.
    for row in truth["subhalos"]:
        row["subhalo_mass_msun"] = 1.0e12 * row["luminosity_ratio"]
    payload = {"mock_cluster": {"truth": truth}}
    with pytest.warns(RuntimeWarning, match="subhalo_dpie_mass_msun"):
        masses = aggregate_plots._truth_subhalo_masses(payload)
    expected = [
        10.0 ** float(log10_dpie_mass_msun(245.0, 187.0)),
        10.0
        ** float(
            log10_dpie_mass_msun(
                245.0 * 0.1**0.25 * math.exp(0.05),
                187.0 * 0.1**0.5,
            )
        ),
    ]
    np.testing.assert_allclose(masses, expected)


def test_aggregate_schechter_curve_requires_dpie_anchor() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    x = np.linspace(10.0, 13.0, 64)
    selection = {"schechter_alpha": -0.7, "mass_ref": 1.0e12, "mass_luminosity_exponent": 1.0}
    assert aggregate_plots._schechter_shmf_curve(selection, x, 10.0) is None
    curve = aggregate_plots._schechter_shmf_curve(
        {**selection, "dpie_mass_anchor_msun": 8.2e12}, x, 10.0
    )
    assert curve is not None
    # Curve integrates to the requested normalization over log-mass.
    assert float(np.trapezoid(curve, x)) == pytest.approx(10.0)
    # Anchoring at the dPIE L=1 mass peaks the curve near anchor * (alpha+1).
    assert x[int(np.argmax(curve))] == pytest.approx(np.log10(8.2e12 * 0.3), abs=0.1)


def test_aggregate_recovered_masses_prefer_free_columns_for_promoted_members() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    payload = {
        "solver": {
            "stage_manifests": [
                {
                    "stage": "stage2_free_source_forward_fit",
                    "table_artifacts": {
                        "scaling_relation_summary.csv": {
                            "data": {
                                "records": [
                                    # Promoted member: free mass wins.
                                    {
                                        "free_log10_mass_msun_best": 11.5,
                                        "scaling_log10_mass_msun_best": 11.0,
                                    },
                                    # Unpromoted member: free columns are NaN.
                                    {
                                        "free_log10_mass_msun_best": float("nan"),
                                        "scaling_log10_mass_msun_best": 10.5,
                                    },
                                    # Best missing entirely: median fallback.
                                    {"scaling_log10_mass_msun_median": 10.0},
                                ]
                            }
                        }
                    },
                }
            ]
        }
    }
    values = aggregate_plots._recovered_stage_subhalo_log_masses(payload, "stage2_free_source_forward_fit")
    np.testing.assert_allclose(values, [11.5, 10.5, 10.0])


def test_m2d_enclosed_analytic_matches_total_mass_convention() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    v_disp, core, cut = 245.0, 0.8, 187.0
    total = 10.0 ** float(log10_dpie_mass_msun(v_disp, cut))
    asymptotic = float(aggregate_plots.dpie_m2d_enclosed_msun(np.asarray([1.0e9]), v_disp, core, cut)[0])
    # The analytic profile converges to the utils total mass times the exact
    # (1 - a^2/s^2) core term that the helper documents as neglected.
    assert asymptotic == pytest.approx(total * (1.0 - (core / cut) ** 2), rel=1.0e-6)
    profile = aggregate_plots.dpie_m2d_enclosed_msun(np.asarray([5.0, 50.0, 500.0]), v_disp, core, cut)
    assert np.all(np.diff(profile) > 0.0)


def test_m2d_numeric_aperture_matches_analytic_for_centered_component() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    radii = np.geomspace(5.0, 400.0, 12)
    v_disp, core, cut = 300.0, 1.5, 150.0
    groups = {
        "halo": (
            np.asarray([v_disp]),
            np.asarray([core]),
            np.asarray([cut]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        )
    }
    profiles = aggregate_plots.aperture_m2d_profiles_msun(radii, groups)
    analytic = aggregate_plots.dpie_m2d_enclosed_msun(radii, v_disp, core, cut)
    np.testing.assert_allclose(profiles["halo"], analytic, rtol=6.0e-3)
    np.testing.assert_allclose(profiles["total"], analytic, rtol=6.0e-3)
    # A far-away component contributes ~nothing inside a small aperture.
    groups["members"] = (
        np.asarray([200.0]),
        np.asarray([0.5]),
        np.asarray([40.0]),
        np.asarray([2000.0]),
        np.asarray([0.0]),
    )
    profiles = aggregate_plots.aperture_m2d_profiles_msun(radii, groups)
    assert profiles["members"][0] < 1.0e-3 * profiles["halo"][0]


def test_m2d_model_groups_apply_scaling_and_promoted_offsets() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    theta = {
        "halo.v_disp": 760.0, "halo.core_radius_kpc": 27.0, "halo.cut_radius_kpc": 1150.0,
        "halo.x_centre": 0.0, "halo.y_centre": 0.0,
        "bcg.v_disp": 285.0, "bcg.core_radius_kpc": 0.8, "bcg.cut_radius_kpc": 187.0,
        "bcg.x_centre": 0.35, "bcg.y_centre": -0.22,
        "potfile.sigma": 245.0, "potfile.cutkpc": 187.0, "potfile.corekpc": 0.8,
        "potfile.alpha_sigma": 0.25, "potfile.gamma_ml": 0.1,
        "potfile.member002.independent_free_log_sigma_delta_unit": 1.0,
        "potfile.member002.independent_free_log_mass_delta_unit": -0.5,
    }
    members = [
        {"id": "member001", "luminosity_ratio": 0.25, "x_arcsec": 10.0, "y_arcsec": -4.0},
        {"id": "member002", "luminosity_ratio": 1.0, "x_arcsec": -6.0, "y_arcsec": 2.0},
    ]
    groups = aggregate_plots._m2d_model_groups(
        theta, members, 5.0, sigma_offset_width=0.1, mass_offset_width=0.2
    )
    v, core, cut, xs, ys = groups["members"]
    beta = 1.0 + 0.1 - 2.0 * 0.25
    assert v[0] == pytest.approx(245.0 * 0.25**0.25)
    assert cut[0] == pytest.approx(187.0 * 0.25**beta)
    # Promoted member: sigma gets exp(width * delta), radius gets exp(mass - 2 sigma).
    assert v[1] == pytest.approx(245.0 * math.exp(0.1 * 1.0))
    assert cut[1] == pytest.approx(187.0 * math.exp(0.2 * -0.5 - 2.0 * 0.1 * 1.0))
    assert xs[1] == pytest.approx(-30.0)
    assert groups["halo"][0][0] == pytest.approx(760.0)
    assert groups["bcg"][3][0] == pytest.approx(0.35 * 5.0)


def test_m2d_aggregate_legend_uses_shared_font_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    legend_calls: list[dict[str, object]] = []
    original_legend = Axes.legend

    def record_legend(self: Axes, *args: object, **kwargs: object) -> object:
        legend_calls.append(dict(kwargs))
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", record_legend)
    fig, top_ax, resid_ax = aggregate_plots._m2d_two_panel_axes()
    try:
        top_ax.plot([1.0, 2.0], [1.0, 3.0], label="model")
        aggregate_plots._m2d_finish_axes(top_ax, resid_ax, np.asarray([0.0, 0.05]))
        assert legend_calls
        assert legend_calls[-1]["fontsize"] == aggregate_plots.AGGREGATE_LEGEND_FONT_SIZE
        assert legend_calls[-1]["loc"] == "lower right"
    finally:
        plt.close(fig)


def test_dpie_mass_helper_matches_integrated_profile_total() -> None:
    # Convention lock: the helper must equal the Eliasdottir A11 / lenstronomy
    # mass_tot (2*pi*sigma0*Ra*Rs*Sigma_cr) of the exact profile the pipeline builds
    # from v_disp, for the lenstool sigma_dPIE convention of our amplitude packing.
    from lenscluster.jax_cosmology import (
        critical_surface_density_angle_msun_per_arcsec2,
        dpie_sigma0_from_vel_disp,
        flat_wcdm_config,
        kpc_per_arcsec_from_config,
    )

    cosmo = flat_wcdm_config(h0=70.0, om0=0.3)
    z_lens, v_disp, ra_arcsec, rs_arcsec = 0.375, 245.0, 0.03, 35.0
    kpc_per_arcsec = float(kpc_per_arcsec_from_config(z_lens, cosmo))
    helper_mass = 10.0 ** float(log10_dpie_mass_msun(v_disp, rs_arcsec * kpc_per_arcsec))
    profile_masses = []
    for z_source in (2.0, 9.0):
        sigma0 = dpie_sigma0_from_vel_disp(v_disp, ra_arcsec, rs_arcsec, z_lens, z_source, cosmo)
        sigma_crit = float(
            np.asarray(critical_surface_density_angle_msun_per_arcsec2(z_lens, z_source, 70.0, 0.3, -1.0))
        )
        profile_masses.append(2.0 * np.pi * sigma0 * ra_arcsec * rs_arcsec * sigma_crit)
    # Independent of source redshift, and equal to the helper up to the neglected
    # (1 - Ra^2/Rs^2) core term (<1e-3 relative here).
    assert profile_masses[0] == pytest.approx(profile_masses[1], rel=1.0e-8)
    assert helper_mass == pytest.approx(profile_masses[0], rel=1.0e-3)
    # Hard value check of the formula itself.
    assert float(log10_dpie_mass_msun(245.0, 187.0)) == pytest.approx(
        math.log10(1.5 * math.pi * 245.0**2 * 187.0 / 4.30091e-6)
    )


def test_plot_subhalo_recovery_radial_writes_pdf(tmp_path: Path) -> None:
    truth = {
        "subhalos": [
            {"x_arcsec": 3.0, "y_arcsec": 4.0},
            {"x_arcsec": 8.0, "y_arcsec": 6.0},
            {"x_arcsec": 12.0, "y_arcsec": 16.0},
        ]
    }
    recovered = pd.DataFrame(
        {
            "recovered_radius_arcsec": [6.0, 12.0, 18.0],
        }
    )
    path = tmp_path / "subhalo_recovery_radial.pdf"

    validation._plot_subhalo_recovery_radial(truth, recovered, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_subhalo_recovery_radial_uses_truth_style_without_parent_or_rugs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = {
        "subhalos": [
            {"x_arcsec": 3.0, "y_arcsec": 4.0},
            {"x_arcsec": 8.0, "y_arcsec": 6.0},
            {"x_arcsec": 12.0, "y_arcsec": 16.0},
        ],
        "subhalo_selection": {
            "candidates": [
                {"subhalo_mass_msun": 5.0e10},
                {"subhalo_mass_msun": 1.0e11},
            ],
        },
    }
    recovered = pd.DataFrame(
        {
            "recovered_radius_arcsec": [6.0, 12.0, 18.0],
        }
    )
    hist_kwargs: list[dict[str, Any]] = []
    xlabel_fontsizes: list[float | None] = []
    ylabel_fontsizes: list[float | None] = []
    legend_fontsizes: list[float | None] = []
    legend_locs: list[str | None] = []
    minor_locator_args: list[tuple[Any, ...]] = []
    original_hist = validation.plt.Axes.hist
    original_set_xlabel = validation.plt.Axes.set_xlabel
    original_set_ylabel = validation.plt.Axes.set_ylabel
    original_legend = validation.plt.Axes.legend
    original_minor_locator = validation.AutoMinorLocator

    def record_hist(self: Any, *args: Any, **kwargs: Any) -> Any:
        hist_kwargs.append(dict(kwargs))
        return original_hist(self, *args, **kwargs)

    def fail_scatter(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw scatter rug markers")

    def fail_title(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw a title")

    def fail_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subhalo radial recovery should not draw line plots")

    def record_set_xlabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        xlabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_xlabel(self, *args, **kwargs)

    def record_set_ylabel(self: Any, *args: Any, **kwargs: Any) -> Any:
        ylabel_fontsizes.append(kwargs.get("fontsize"))
        return original_set_ylabel(self, *args, **kwargs)

    def record_legend(self: Any, *args: Any, **kwargs: Any) -> Any:
        legend_fontsizes.append(kwargs.get("fontsize"))
        legend_locs.append(kwargs.get("loc"))
        return original_legend(self, *args, **kwargs)

    def record_minor_locator(*args: Any, **kwargs: Any) -> Any:
        minor_locator_args.append(args)
        return original_minor_locator(*args, **kwargs)

    monkeypatch.setattr(validation.plt.Axes, "hist", record_hist)
    monkeypatch.setattr(validation.plt.Axes, "plot", fail_plot)
    monkeypatch.setattr(validation.plt.Axes, "scatter", fail_scatter)
    monkeypatch.setattr(validation.plt.Axes, "set_title", fail_title)
    monkeypatch.setattr(validation.plt.Axes, "set_xlabel", record_set_xlabel)
    monkeypatch.setattr(validation.plt.Axes, "set_ylabel", record_set_ylabel)
    monkeypatch.setattr(validation.plt.Axes, "legend", record_legend)
    monkeypatch.setattr(validation, "AutoMinorLocator", record_minor_locator)

    path = tmp_path / "subhalo_recovery_radial.pdf"

    validation._plot_subhalo_recovery_radial(truth, recovered, path)

    hist_labels = [kwargs.get("label") for kwargs in hist_kwargs]
    assert "parent candidates" not in hist_labels
    assert "Truth subhalos" in hist_labels
    assert "recovered subhalos" in hist_labels
    truth_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "Truth subhalos")
    recovered_hist = next(kwargs for kwargs in hist_kwargs if kwargs.get("label") == "recovered subhalos")
    assert truth_hist["color"] == "lightgray"
    assert truth_hist["alpha"] == pytest.approx(0.8)
    assert recovered_hist["histtype"] == "step"
    assert recovered_hist["color"] == "tab:blue"
    assert minor_locator_args == [(5,)]
    assert xlabel_fontsizes == [14]
    assert ylabel_fontsizes == [14]
    assert legend_fontsizes == [11]
    assert legend_locs == ["upper left"]
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prefit_subhalo_spatial_distribution_writes_partial_pdf(tmp_path: Path) -> None:
    subhalo_df = pd.DataFrame(
        {
            "x_arcsec": [2.5, -1.0],
            "y_arcsec": [-0.5, 1.5],
            "luminosity_ratio": [1.0, 0.25],
            "catalog_mag": [17.0, 19.0],
        }
    )
    images = pd.DataFrame({"x_obs_arcsec": [-1.0, 1.0], "y_obs_arcsec": [0.0, 0.5]})
    path = tmp_path / "prefit_subhalo_spatial_distribution.pdf"

    validation._plot_prefit_subhalo_spatial_distribution(subhalo_df, images, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_select_prefit_critical_line_contours_prefers_z9_then_highest() -> None:
    z2_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.1, 0.0, 0.1]),
        beta_y=np.asarray([0.0, 0.1, 0.0]),
        critical_x=np.asarray([-1.0, 0.0, 1.0]),
        critical_y=np.asarray([0.0, 1.0, 0.0]),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )
    z9_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.2, 0.0, 0.2]),
        beta_y=np.asarray([0.0, 0.2, 0.0]),
        critical_x=np.asarray([-2.0, 0.0, 2.0]),
        critical_y=np.asarray([0.0, 2.0, 0.0]),
        caustic_area_arcsec2=0.04,
        critical_area_arcsec2=4.0,
    )

    selected = validation._select_prefit_critical_line_contours(
        {"2.00000000": [z2_contour], "9.00000000": [z9_contour]}
    )
    fallback = validation._select_prefit_critical_line_contours(
        {"2.00000000": [z2_contour], "5.00000000": [z9_contour]}
    )

    assert list(selected) == ["9.00000000"]
    assert selected["9.00000000"][0] is z9_contour
    assert list(fallback) == ["5.00000000"]
    assert fallback["5.00000000"][0] is z9_contour


def test_plot_prefit_critical_lines_writes_pdf(tmp_path: Path) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-2.0, 0.0, 2.0, -2.0],
        "critical_y": [0.0, 2.0, 0.0, 0.0],
        "caustic_beta_x": [-0.2, 0.0, 0.2, -0.2],
        "caustic_beta_y": [0.0, 0.2, 0.0, 0.0],
        "caustic_area_arcsec2": 0.04,
        "critical_area_arcsec2": 4.0,
    }
    path = tmp_path / "prefit_critical_lines.pdf"

    validation._plot_prefit_critical_lines({"caustics_by_source_redshift": {"9.00000000": [contour_payload]}}, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_prefit_critical_lines_limits_ticks_for_large_ranges(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-13.0, 130.5, 274.0, -13.0],
        "critical_y": [-13.0, 274.0, -13.0, -13.0],
        "caustic_beta_x": [-13.0, 130.5, 274.0, -13.0],
        "caustic_beta_y": [-13.0, 274.0, -13.0, -13.0],
        "caustic_area_arcsec2": 0.04,
        "critical_area_arcsec2": 4.0,
    }
    path = tmp_path / "prefit_critical_lines_large_range.pdf"

    with caplog.at_level(logging.WARNING, logger="matplotlib.ticker"):
        validation._plot_prefit_critical_lines({"caustics_by_source_redshift": {"9.00000000": [contour_payload]}}, path)

    assert path.exists()
    assert path.stat().st_size > 0
    assert _locator_max_tick_messages(caplog) == []


def test_write_prefit_validation_diagnostics_writes_all_outputs(tmp_path: Path) -> None:
    contour_payload = {
        "caustic_index": 0,
        "caustic_class": "primary",
        "critical_x": [-2.0, 0.0, 2.0, -2.0],
        "critical_y": [0.0, 2.0, 0.0, 0.0],
        "caustic_beta_x": [-0.2, 0.0, 0.2, -0.2],
        "caustic_beta_y": [0.0, 0.2, 0.0, 0.0],
    }
    truth = {
        "subhalos": [
            {
                "subhalo_mass_msun": 1.0e11,
                "x_arcsec": 2.0,
                "y_arcsec": -1.0,
                "luminosity_ratio": 1.0,
                "catalog_mag": 17.0,
            }
        ],
        "subhalo_selection": {"candidates": [{"subhalo_mass_msun": 1.0e11, "selected": True}]},
        "caustics_by_source_redshift": {"9.00000000": [contour_payload]},
    }
    images = pd.DataFrame({"x_obs_arcsec": [0.0], "y_obs_arcsec": [1.0]})

    outputs = validation.write_prefit_validation_diagnostics(truth, images, tmp_path)

    assert outputs == {
        "subhalo_shmf_plot": tmp_path / "subhalo_shmf.pdf",
        "prefit_subhalo_spatial_distribution_plot": tmp_path / "prefit_subhalo_spatial_distribution.pdf",
        "prefit_critical_lines_plot": tmp_path / "prefit_critical_lines.pdf",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_single_bcg_mock_can_request_subhalo_caustic_family(tmp_path: Path) -> None:
    config = SingleBCGMockConfig(
        seed=11,
        n_primary_families=1,
        n_subhalo_families=1,
        n_subhalos=8,
        **_fixed_potfile_scaling_config(sigma_ref=300.0, cut_radius_arcsec=45.0),
        pos_sigma_arcsec=0.0,
        primary_source_redshifts=(2.0,),
        subhalo_source_redshifts=(2.0,),
        halo_position_half_width_arcsec=0.0,
        bcg_position_offset_radius_arcsec=0.0,
        caustic_compute_window_arcsec=80.0,
        caustic_grid_scale_arcsec=1.0,
        mock_image_candidate_batch_size=8,
    )

    _paths, images, truth = generate_single_bcg_mock(tmp_path, config)

    assert [source["caustic_class"] for source in truth["sources"]] == ["primary", "subhalo"]
    assert images["family_id"].nunique() == 2
    assert (images.groupby("family_id").size() >= config.min_images_per_family).all()


def test_caustic_classifier_marks_largest_critical_curve_primary() -> None:
    small = validation.CausticContour(
        caustic_index=0,
        caustic_class="subhalo",
        beta_x=np.asarray([-0.1, 0.1, 0.1, -0.1, -0.1]),
        beta_y=np.asarray([-0.1, -0.1, 0.1, 0.1, -0.1]),
        critical_x=np.asarray([-1.0, 1.0, 1.0, -1.0, -1.0]),
        critical_y=np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0]),
        caustic_area_arcsec2=0.04,
        critical_area_arcsec2=4.0,
    )
    large = validation.CausticContour(
        caustic_index=1,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.3, 0.3, -0.3, -0.3]),
        beta_y=np.asarray([-0.3, -0.3, 0.3, 0.3, -0.3]),
        critical_x=np.asarray([-3.0, 3.0, 3.0, -3.0, -3.0]),
        critical_y=np.asarray([-3.0, -3.0, 3.0, 3.0, -3.0]),
        caustic_area_arcsec2=0.36,
        critical_area_arcsec2=36.0,
    )

    assert max([small, large], key=lambda item: item.critical_area_arcsec2).caustic_class == "primary"


def test_primary_contour_position_uses_critical_area_and_keeps_paired_caustic() -> None:
    larger_caustic_smaller_critical = (
        np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.asarray([10.0, 11.0, 11.0, 10.0, 10.0]),
        np.asarray([20.0, 20.0, 21.0, 21.0, 20.0]),
        1.0,
        100.0,
    )
    smaller_caustic_larger_critical = (
        np.asarray([0.0, 3.0, 3.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 3.0, 3.0, 0.0]),
        np.asarray([-2.0, -1.5, -1.5, -2.0, -2.0]),
        np.asarray([4.0, 4.0, 4.5, 4.5, 4.0]),
        9.0,
        0.25,
    )

    contours = [larger_caustic_smaller_critical, smaller_caustic_larger_critical]
    primary_position = mock_generation._primary_contour_position(contours)
    results = []
    for index, (crit_x, crit_y, beta_x, beta_y, crit_area, caustic_area) in enumerate(contours):
        results.append(
            validation.CausticContour(
                caustic_index=index,
                caustic_class="primary" if index == primary_position else "subhalo",
                beta_x=beta_x,
                beta_y=beta_y,
                critical_x=crit_x,
                critical_y=crit_y,
                caustic_area_arcsec2=caustic_area,
                critical_area_arcsec2=crit_area,
            )
        )

    primary = next(contour for contour in results if contour.caustic_class == "primary")
    assert primary.caustic_index == 1
    assert primary.critical_area_arcsec2 == pytest.approx(9.0)
    assert primary.caustic_area_arcsec2 == pytest.approx(0.25)
    np.testing.assert_allclose(primary.beta_x, smaller_caustic_larger_critical[2])
    np.testing.assert_allclose(primary.beta_y, smaller_caustic_larger_critical[3])


def test_sample_point_in_caustic_returns_inside_point() -> None:
    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        beta_y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
        critical_x=np.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        critical_y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
        caustic_area_arcsec2=1.0,
        critical_area_arcsec2=1.0,
    )

    x, y = validation._sample_point_in_caustic(contour, np.random.default_rng(1))

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert validation.MplPath(np.column_stack([contour.beta_x, contour.beta_y])).contains_point((x, y))


def test_cluster_solver_cosmology_accepts_lenstool_cosmology_keys() -> None:
    cosmo = _build_cosmology({"cosmology": {"H0": 67.74, "omega": 0.3089, "lambda": 0.6911}})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 67.74)
    np.testing.assert_allclose(cosmo["Om0"], 0.3089)


def test_cluster_solver_cosmology_accepts_cosmologie_modern_keys() -> None:
    cosmo = _build_cosmology({"cosmologie": {"H0": 67.74, "omegaM": 0.3089, "omegaX": 0.6911}})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 67.74)
    np.testing.assert_allclose(cosmo["Om0"], 0.3089)


def test_cluster_solver_cosmology_defaults_without_block() -> None:
    cosmo = _build_cosmology({})

    assert cosmo["class"] == "FlatLambdaCDM"
    np.testing.assert_allclose(cosmo["H0"], 70.0)
    np.testing.assert_allclose(cosmo["Om0"], 0.3)


def test_flat_wcdm_jax_distances_match_astropy() -> None:
    z_lens = 0.3734
    source_redshifts = np.asarray([1.5, 3.0, 7.0], dtype=float)
    h0 = 70.0
    om0 = 0.31
    w0 = -0.8
    cosmo = FlatwCDM(H0=h0, Om0=om0, w0=w0)

    chi_source = flat_wcdm_comoving_distance_mpc(source_redshifts, h0, om0, w0)
    efficiency = flat_wcdm_lensing_efficiency(z_lens, source_redshifts, h0, om0, w0)
    kpc_per_arcsec = flat_wcdm_kpc_per_arcsec(z_lens, h0, om0, w0)

    np.testing.assert_allclose(
        np.asarray(chi_source),
        cosmo.comoving_distance(source_redshifts).value,
        rtol=1.0e-3,
        atol=1.0e-3,
    )
    np.testing.assert_allclose(
        float(kpc_per_arcsec),
        cosmo.kpc_proper_per_arcmin(z_lens).to("kpc/arcsec").value,
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(efficiency),
        (cosmo.angular_diameter_distance_z1z2(z_lens, source_redshifts) / cosmo.angular_diameter_distance(source_redshifts)).value,
        rtol=1.0e-3,
    )
    assert np.all(np.asarray(dpie_sigma0_factor_from_lensing_efficiency(efficiency)) > 0.0)


def test_flat_wcdm_vectorized_lens_geometry_matches_scalar_helpers() -> None:
    z_lens = 0.3734
    source_redshifts = np.asarray([1.5, 3.0, 7.0], dtype=float)
    h0 = 70.0
    om0 = 0.31
    w0 = -0.8

    kpc_per_arcsec, efficiency, sigma0_factors = flat_wcdm_lens_geometry_factors(
        z_lens,
        source_redshifts,
        h0,
        om0,
        w0,
    )

    np.testing.assert_allclose(
        float(kpc_per_arcsec),
        float(flat_wcdm_kpc_per_arcsec(z_lens, h0, om0, w0)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(efficiency),
        np.asarray(flat_wcdm_lensing_efficiency(z_lens, source_redshifts, h0, om0, w0)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(sigma0_factors),
        np.asarray(dpie_sigma0_factor(z_lens, source_redshifts, h0, om0, w0)),
        rtol=1.0e-10,
    )


def test_dpie_sigma0_from_vel_disp_uses_lenstool_profile81_normalization() -> None:
    z_lens = 0.396
    z_source = 2.0
    h0 = 70.0
    om0 = 0.3
    w0 = -1.0
    vel_disp = 760.0
    ra_arcsec = 5.0
    rs_arcsec = 220.0
    cosmo_config = {"class": "FlatLambdaCDM", "H0": h0, "Om0": om0}

    efficiency = float(flat_wcdm_lensing_efficiency(z_lens, z_source, h0, om0, w0))
    expected_factor = (1.0 / C_LIGHT_KM_S) ** 2 * 3.0 * np.pi * efficiency / ARCSEC_TO_RAD
    expected_sigma0 = vel_disp**2 * expected_factor / ra_arcsec

    assert float(dpie_sigma0_factor(z_lens, z_source, h0, om0, w0)) == pytest.approx(expected_factor)
    assert dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        rs_arcsec,
        z_lens,
        z_source,
        cosmo_config,
    ) == pytest.approx(expected_sigma0)
    assert dpie_sigma0_from_vel_disp(
        vel_disp,
        ra_arcsec,
        400.0,
        z_lens,
        z_source,
        cosmo_config,
    ) == pytest.approx(expected_sigma0)
    assert math.isnan(dpie_sigma0_from_vel_disp(vel_disp, ra_arcsec, ra_arcsec, z_lens, z_source, cosmo_config))


def test_catalog_shape_to_ellipticity_round_trips_lenstool_axis_ratio() -> None:
    ellipticite, theta = cluster_solver._catalog_shape_to_ellipticity(2.0, 1.0, 12.0)

    assert ellipticite == pytest.approx(0.6)
    assert theta == pytest.approx(12.0)
    q = math.sqrt((1.0 - ellipticite) / (1.0 + ellipticite))
    assert q == pytest.approx(0.5)


def test_cosmology_parameter_specs_use_broad_uniform_priors() -> None:
    specs = cluster_solver._build_cosmology_parameter_specs(0, FlatwCDM(H0=70.0, Om0=0.31, w0=-0.8))

    assert [spec.sample_name for spec in specs] == ["cosmology_Om0", "cosmology_w0"]
    assert [spec.component_family for spec in specs] == ["cosmology", "cosmology"]
    assert specs[0].prior_kind == "uniform"
    assert specs[0].lower == pytest.approx(0.05)
    assert specs[0].upper == pytest.approx(0.6)
    assert specs[0].physical_mean == pytest.approx(0.31)
    assert specs[1].lower == pytest.approx(-2.0)
    assert specs[1].upper == pytest.approx(-0.3)
    assert specs[1].physical_mean == pytest.approx(-0.8)


def test_image_plane_scatter_default_log_uniform_prior_uses_default_floor() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=2.0,
    )

    assert spec.sample_name == "image_sigma_int"
    assert spec.prior_kind == "uniform"
    assert spec.lower == pytest.approx(np.log(cluster_solver.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC))
    assert spec.upper == pytest.approx(np.log(2.0))
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(cluster_solver.DEFAULT_IMAGE_PLANE_SCATTER_FLOOR_ARCSEC)
    assert spec.physical_upper == pytest.approx(2.0)


def test_image_plane_scatter_log_uniform_prior_uses_floor_as_lower_support() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=0.5,
        floor_arcsec=0.05,
    )

    assert spec.prior_kind == "uniform"
    assert spec.lower == pytest.approx(np.log(0.05))
    assert spec.upper == pytest.approx(np.log(0.5))
    assert spec.physical_lower == pytest.approx(0.05)
    assert spec.physical_upper == pytest.approx(0.5)


def test_image_plane_scatter_lognormal_prior_uses_truncated_normal_latent_log_sigma() -> None:
    spec = cluster_solver._build_image_scatter_parameter_spec(
        start_index=0,
        upper_arcsec=0.5,
        floor_arcsec=0.05,
        prior=cluster_solver.IMAGE_PLANE_SCATTER_PRIOR_LOGNORMAL,
        prior_median_arcsec=0.25,
        prior_log_sigma=0.4,
    )

    physical = cluster_solver._convert_theta_to_physical(np.asarray([np.log(0.25)]), [spec])
    log_prob = cluster_solver._prior_log_prob([spec], jnp.asarray([np.log(0.25)], dtype=jnp.float64))

    assert spec.prior_kind == "truncated_normal"
    assert spec.lower == pytest.approx(np.log(0.05))
    assert spec.upper == pytest.approx(np.log(0.5))
    assert spec.mean == pytest.approx(np.log(0.25))
    assert spec.std == pytest.approx(0.4)
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(0.05)
    assert spec.physical_upper == pytest.approx(0.5)
    assert spec.physical_mean == pytest.approx(0.25)
    assert physical.tolist() == pytest.approx([0.25])
    assert np.isfinite(float(log_prob))


def test_critical_arc_singular_threshold_prior_uses_truncated_lognormal() -> None:
    spec = cluster_solver._build_critical_arc_singular_threshold_parameter_spec(
        start_index=0,
        lower=0.03,
        upper=0.40,
        prior_median=0.15,
        prior_log_sigma=0.5,
    )

    physical = cluster_solver._convert_theta_to_physical(np.asarray([np.log(0.15)]), [spec])
    log_prob = cluster_solver._prior_log_prob([spec], jnp.asarray([np.log(0.15)], dtype=jnp.float64))

    assert spec.name == "critical_arc.singular_threshold"
    assert spec.sample_name == cluster_solver.CRITICAL_ARC_SINGULAR_THRESHOLD_SAMPLE_NAME
    assert spec.component_family == cluster_solver.CRITICAL_ARC_HYPERPARAMETER_COMPONENT_FAMILY
    assert spec.prior_kind == "truncated_normal"
    assert spec.lower == pytest.approx(np.log(0.03))
    assert spec.upper == pytest.approx(np.log(0.40))
    assert spec.mean == pytest.approx(np.log(0.15))
    assert spec.std == pytest.approx(0.5)
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(0.03)
    assert spec.physical_upper == pytest.approx(0.40)
    assert spec.physical_mean == pytest.approx(0.15)
    assert physical.tolist() == pytest.approx([0.15])
    assert np.isfinite(float(log_prob))


def test_critical_arc_singular_softness_prior_uses_truncated_lognormal() -> None:
    spec = cluster_solver._build_critical_arc_singular_softness_parameter_spec(
        start_index=0,
        lower=0.005,
        upper=0.20,
        prior_median=0.05,
        prior_log_sigma=0.5,
    )

    physical = cluster_solver._convert_theta_to_physical(np.asarray([np.log(0.05)]), [spec])
    log_prob = cluster_solver._prior_log_prob([spec], jnp.asarray([np.log(0.05)], dtype=jnp.float64))

    assert spec.name == "critical_arc.singular_softness"
    assert spec.sample_name == cluster_solver.CRITICAL_ARC_SINGULAR_SOFTNESS_SAMPLE_NAME
    assert spec.component_family == cluster_solver.CRITICAL_ARC_HYPERPARAMETER_COMPONENT_FAMILY
    assert spec.prior_kind == "truncated_normal"
    assert spec.lower == pytest.approx(np.log(0.005))
    assert spec.upper == pytest.approx(np.log(0.20))
    assert spec.mean == pytest.approx(np.log(0.05))
    assert spec.std == pytest.approx(0.5)
    assert spec.transform_kind == "log_positive"
    assert spec.physical_lower == pytest.approx(0.005)
    assert spec.physical_upper == pytest.approx(0.20)
    assert spec.physical_mean == pytest.approx(0.05)
    assert physical.tolist() == pytest.approx([0.05])
    assert np.isfinite(float(log_prob))


@pytest.mark.parametrize(
    ("floor_arcsec", "upper_arcsec", "match"),
    [
        (0.0, 2.0, "floor"),
        (float("nan"), 2.0, "floor"),
        (0.1, 0.1, "upper"),
        (0.1, float("nan"), "upper"),
    ],
)
def test_image_plane_scatter_parameter_spec_rejects_invalid_direct_bounds(
    floor_arcsec: float,
    upper_arcsec: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        cluster_solver._build_image_scatter_parameter_spec(
            start_index=0,
            upper_arcsec=upper_arcsec,
            floor_arcsec=floor_arcsec,
        )


def test_family_data_search_center_uses_observed_bounding_box_midpoint() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b", "1.c", "1.d", "1.e"],
        x_obs=np.asarray([7.9, -19.9, -22.2, 9.5, 0.6], dtype=float),
        y_obs=np.asarray([-26.4, 13.1, -0.3, 12.6, 0.9], dtype=float),
        reliability=np.ones(5, dtype=float),
    )

    assert family.x_center == pytest.approx(0.5 * (np.min(family.x_obs) + np.max(family.x_obs)))
    assert family.y_center == pytest.approx(0.5 * (np.min(family.y_obs) + np.max(family.y_obs)))
    assert family.x_center != pytest.approx(float(np.mean(family.x_obs)))
    assert family.y_center != pytest.approx(float(np.mean(family.y_obs)))
    half_window = 0.5 * family.search_window
    assert np.all(np.abs(family.x_obs - family.x_center) <= half_window)
    assert np.all(np.abs(family.y_obs - family.y_center) <= half_window)
    assert np.any(np.abs(family.y_obs - float(np.mean(family.y_obs))) > half_window)


def test_prior_whitened_source_position_specs_map_unit_offsets_to_physical_beta() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )

    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="prior-whitened",
    )

    assert [spec.sample_name for spec in specs] == ["source_1_beta_x", "source_1_beta_y"]
    assert [spec.mean for spec in specs] == [0.0, 0.0]
    assert [spec.std for spec in specs] == [1.0, 1.0]
    assert [spec.transform_kind for spec in specs] == ["affine", "affine"]
    assert specs[0].physical_mean == pytest.approx(0.25)
    assert specs[1].physical_mean == pytest.approx(-0.5)
    assert specs[0].transform_offset == pytest.approx(0.25)
    assert specs[1].transform_offset == pytest.approx(-0.5)
    assert specs[0].transform_scale == pytest.approx(0.3)
    assert specs[1].transform_scale == pytest.approx(0.3)
    assert cluster_solver._convert_theta_to_physical(np.asarray([0.0, 0.0]), specs).tolist() == pytest.approx([0.25, -0.5])
    assert cluster_solver._convert_theta_to_physical(np.asarray([1.0, 1.0]), specs).tolist() == pytest.approx([0.55, -0.2])


def test_direct_source_position_specs_keep_physical_beta_sampling() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )

    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="direct",
    )

    assert [spec.mean for spec in specs] == pytest.approx([0.25, -0.5])
    assert [spec.std for spec in specs] == pytest.approx([0.3, 0.3])
    assert [spec.transform_kind for spec in specs] == ["identity", "identity"]


def test_source_position_specs_share_one_vector_sample_site() -> None:
    families = [
        FamilyData(
            family_id="1",
            z_source=2.0,
            effective_z_source=2.0,
            sigma_arcsec=0.15,
            image_labels=["1.1", "1.2"],
            x_obs=np.asarray([0.0, 1.0]),
            y_obs=np.asarray([0.0, 1.0]),
            reliability=np.ones(2),
        ),
        FamilyData(
            family_id="2",
            z_source=3.0,
            effective_z_source=3.0,
            sigma_arcsec=0.15,
            image_labels=["2.1", "2.2"],
            x_obs=np.asarray([2.0, 3.0]),
            y_obs=np.asarray([2.0, 3.0]),
            reliability=np.ones(2),
        ),
    ]

    specs = cluster_solver._build_source_position_parameter_specs(
        families,
        {"1": (0.25, -0.5), "2": (1.25, -1.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="prior-whitened",
    )

    assert [spec.sample_name for spec in specs] == [
        "source_1_beta_x",
        "source_1_beta_y",
        "source_2_beta_x",
        "source_2_beta_y",
    ]
    assert {spec.sample_site_name for spec in specs} == {cluster_solver.SOURCE_POSITION_VECTOR_SAMPLE_SITE_NAME}
    assert [spec.sample_site_index for spec in specs] == [0, 1, 2, 3]
    assert [spec.potential_id for spec in specs] == ["1", "1", "2", "2"]
    assert [spec.field for spec in specs] == ["beta_x", "beta_y", "beta_x", "beta_y"]
    assert [spec.transform_offset for spec in specs] == pytest.approx([0.25, -0.5, 1.25, -1.5])

    sites = cluster_solver._parameter_sample_sites(specs)
    distribution = cluster_solver._distribution_for_sample_site(sites[0], specs)

    assert len(sites) == 1
    assert sites[0].name == cluster_solver.SOURCE_POSITION_VECTOR_SAMPLE_SITE_NAME
    assert sites[0].indices == (0, 1, 2, 3)
    assert tuple(distribution.event_shape) == (4,)


def test_explicit_source_position_parameterization_uses_state_metadata_not_spec_shape() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.0, 0.0)},
        start_index=0,
        beta_prior_sigma_arcsec=1.0,
        parameterization="direct",
    )

    state = SimpleNamespace(parameter_specs=specs, source_position_parameterization="direct")

    assert cluster_solver._explicit_source_position_parameterization_for_state(state) == "direct"


def test_explicit_source_position_parameterization_requires_metadata() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.0, 0.0)},
        start_index=0,
        beta_prior_sigma_arcsec=1.0,
        parameterization="conditional-whitened",
    )

    with pytest.raises(ValueError, match="missing explicit source_position_parameterization"):
        cluster_solver._explicit_source_position_parameterization_for_state(SimpleNamespace(parameter_specs=specs))


def test_prior_whitened_source_position_prior_matches_direct_prior_up_to_jacobian() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    centers = {"1": (0.25, -0.5)}
    sigma = 0.3
    direct_specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        centers,
        start_index=0,
        beta_prior_sigma_arcsec=sigma,
        parameterization="direct",
    )
    whitened_specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        centers,
        start_index=0,
        beta_prior_sigma_arcsec=sigma,
        parameterization="prior-whitened",
    )
    eta = np.asarray([0.7, -1.2])
    beta = cluster_solver._convert_theta_to_physical(eta, whitened_specs)

    direct_logp = float(cluster_solver._prior_log_prob(direct_specs, jnp.asarray(beta)))
    whitened_logp = float(cluster_solver._prior_log_prob(whitened_specs, jnp.asarray(eta)))

    assert whitened_logp - direct_logp == pytest.approx(2.0 * np.log(sigma))


def test_conditional_whitened_source_transport_correction_matches_change_of_variables() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.15,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 1.0]),
        reliability=np.ones(2),
    )
    specs = cluster_solver._build_source_position_parameter_specs(
        [family],
        {"1": (0.25, -0.5)},
        start_index=0,
        beta_prior_sigma_arcsec=0.3,
        parameterization="conditional-whitened",
    )
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.4], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0, 0.8], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True], dtype=bool),
    )
    fake_evaluator = SimpleNamespace(
        state=SimpleNamespace(parameter_specs=specs),
        source_position_param_indices_by_family={"1": (0, 1)},
        source_plane_covariance_floor=0.01,
        image_plane_scatter_floor_arcsec=0.3,
        likelihood_stabilizer_max_gain=0.0,
        likelihood_stabilizer_max_residual_arcsec=0.0,
    )
    eta = jnp.asarray([0.7, -1.2], dtype=jnp.float64)
    beta_x = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    beta_y = jnp.asarray([-0.2, -0.1], dtype=jnp.float64)
    image_sigma_int = jnp.asarray(0.05, dtype=jnp.float64)
    jacobian_entries = (
        jnp.asarray([1.2, 0.8], dtype=jnp.float64),
        jnp.asarray([0.15, -0.05], dtype=jnp.float64),
        jnp.asarray([0.2, 0.1], dtype=jnp.float64),
        jnp.asarray([0.9, 1.1], dtype=jnp.float64),
    )

    source_x, source_y, finite, correction = (
        cluster_solver.ClusterJAXEvaluator._conditional_source_position_transport_for_bin(
            fake_evaluator,
            eta,
            traced_bin,
            beta_x,
            beta_y,
            image_sigma_int,
            jacobian_entries,
        )
    )

    sigma2 = np.asarray(traced_bin.sigma_per_image) ** 2 + float(image_sigma_int) ** 2 + 0.01
    weights = np.asarray(traced_bin.reliability_per_image) / sigma2
    prior_precision = 1.0 / 0.3**2
    precision_matrix = prior_precision * np.eye(2)
    rhs = prior_precision * np.asarray([0.25, -0.5])
    jac_arrays = [np.asarray(item, dtype=float) for item in jacobian_entries]
    for image_index, weight in enumerate(weights):
        a_matrix = np.asarray(
            [
                [jac_arrays[0][image_index], jac_arrays[1][image_index]],
                [jac_arrays[2][image_index], jac_arrays[3][image_index]],
            ],
            dtype=float,
        )
        inv_a = np.linalg.inv(a_matrix)
        lambda_i = float(weight) * inv_a.T @ inv_a
        beta_i = np.asarray([float(beta_x[image_index]), float(beta_y[image_index])])
        precision_matrix += lambda_i
        rhs += lambda_i @ beta_i
    covariance = np.linalg.inv(precision_matrix)
    mean = covariance @ rhs
    chol = np.linalg.cholesky(covariance)
    expected_source = mean + chol @ np.asarray(eta, dtype=float)

    assert bool(finite)
    assert np.asarray(source_x).tolist() == pytest.approx([expected_source[0], expected_source[0]])
    assert np.asarray(source_y).tolist() == pytest.approx([expected_source[1], expected_source[1]])

    fake_evaluator.source_metric_cache_by_z = {2.0: {"jac_a00": np.full(2, 99.0)}}
    source_x_cached, source_y_cached, finite_cached, correction_cached = (
        cluster_solver.ClusterJAXEvaluator._conditional_source_position_transport_for_bin(
            fake_evaluator,
            eta,
            traced_bin,
            beta_x,
            beta_y,
            image_sigma_int,
            jacobian_entries,
        )
    )
    assert bool(finite_cached)
    np.testing.assert_allclose(np.asarray(source_x_cached), np.asarray(source_x))
    np.testing.assert_allclose(np.asarray(source_y_cached), np.asarray(source_y))
    assert float(correction_cached) == pytest.approx(float(correction))

    beta_prior_logp = -0.5 * (
        np.sum(((expected_source - np.asarray([0.25, -0.5])) / 0.3) ** 2)
        + 2.0 * np.log(2.0 * np.pi * 0.3**2)
    )
    eta_prior_logp = -0.5 * (float(np.sum(np.asarray(eta) ** 2)) + 2.0 * np.log(2.0 * np.pi))
    log_det = float(np.log(np.linalg.det(chol)))

    assert eta_prior_logp + float(correction) == pytest.approx(beta_prior_logp + log_det)


def _conditional_inverse_cache_fake_evaluator() -> Any:
    specs = [
        ParameterSpec(
            name="scale",
            sample_name="scale",
            potential_id="scale",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_ref",
            prior_kind="uniform",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="scaling",
        ),
        ParameterSpec(
            name="1.beta_x",
            sample_name="source_1_beta_x",
            potential_id="1",
            profile_type=0,
            field="beta_x",
            prior_kind="normal",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            name="1.beta_y",
            sample_name="source_1_beta_y",
            potential_id="1",
            profile_type=0,
            field="beta_y",
            prior_kind="normal",
            lower=-100.0,
            upper=100.0,
            step=0.1,
            component_family="source_position",
        ),
    ]
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.state = SimpleNamespace(parameter_specs=specs)
    evaluator.source_position_conditional = True
    evaluator.source_position_param_indices_by_family = {"1": (1, 2)}
    evaluator._conditional_source_inverse_basis_cache = {}
    evaluator.reported_calls = 0
    transform = np.asarray([[2.0, 0.5], [-0.25, 1.5]], dtype=float)

    def reported_physical_parameter_vector(params: np.ndarray) -> np.ndarray:
        evaluator.reported_calls += 1
        params = np.asarray(params, dtype=float)
        physical = params.copy()
        mean = np.asarray([10.0 + 2.0 * params[0], -5.0 + 3.0 * params[0]], dtype=float)
        physical[1:3] = mean + transform @ params[1:3]
        return physical

    evaluator.reported_physical_parameter_vector = reported_physical_parameter_vector
    return evaluator


def test_conditional_reported_physical_to_latent_reuses_cached_source_basis() -> None:
    evaluator = _conditional_inverse_cache_fake_evaluator()
    target_latent = np.asarray([4.0, 0.6, -0.2], dtype=float)
    reported_physical = evaluator.reported_physical_parameter_vector(target_latent)
    evaluator.reported_calls = 0

    converted_once = evaluator.reported_physical_to_latent_parameter_vector(reported_physical)
    converted_twice = evaluator.reported_physical_to_latent_parameter_vector(reported_physical)

    np.testing.assert_allclose(converted_once, target_latent)
    np.testing.assert_allclose(converted_twice, target_latent)
    assert len(evaluator._conditional_source_inverse_basis_cache) == 1
    assert evaluator.reported_calls == 3


def test_conditional_reported_physical_to_latent_cache_keys_non_source_state() -> None:
    evaluator = _conditional_inverse_cache_fake_evaluator()
    first_latent = np.asarray([4.0, 0.6, -0.2], dtype=float)
    second_latent = np.asarray([5.0, 0.6, -0.2], dtype=float)
    first_reported = evaluator.reported_physical_parameter_vector(first_latent)
    second_reported = evaluator.reported_physical_parameter_vector(second_latent)
    evaluator.reported_calls = 0

    first_converted = evaluator.reported_physical_to_latent_parameter_vector(first_reported)
    second_converted = evaluator.reported_physical_to_latent_parameter_vector(second_reported)

    np.testing.assert_allclose(first_converted, first_latent)
    np.testing.assert_allclose(second_converted, second_latent)
    assert len(evaluator._conditional_source_inverse_basis_cache) == 2
    assert evaluator.reported_calls == 6


def _minimal_stage4_surrogate_state(
    *,
    fit_cosmology_flat_wcdm: bool = False,
    include_source_positions: bool = False,
) -> BuildState:
    n_components = 2
    int_minus_one = np.full(n_components, -1, dtype=np.int32)
    float_zero = np.zeros(n_components, dtype=float)
    packed = PackedLensSpec(
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        component_family=np.asarray([0, 1], dtype=np.int32),
        x_center_base=np.asarray([0.0, 1.0], dtype=float),
        y_center_base=np.asarray([0.0, 1.0], dtype=float),
        e1_base=float_zero.copy(),
        e2_base=float_zero.copy(),
        core_radius_kpc_base=np.asarray([5.0, 1.0], dtype=float),
        cut_radius_kpc_base=np.asarray([100.0, 20.0], dtype=float),
        v_disp_base=np.asarray([900.0, 120.0], dtype=float),
        gamma1_base=float_zero.copy(),
        gamma2_base=float_zero.copy(),
        x_center_param_index=int_minus_one.copy(),
        y_center_param_index=int_minus_one.copy(),
        e1_param_index=int_minus_one.copy(),
        e2_param_index=int_minus_one.copy(),
        core_radius_param_index=int_minus_one.copy(),
        cut_radius_param_index=int_minus_one.copy(),
        v_disp_param_index=int_minus_one.copy(),
        gamma1_param_index=int_minus_one.copy(),
        gamma2_param_index=int_minus_one.copy(),
        luminosity_ratio=np.ones(n_components, dtype=float),
        sigma_ref_base=np.asarray([0.0, 120.0], dtype=float),
        cut_ref_base=np.asarray([0.0, 20.0], dtype=float),
        core_ref_base=np.asarray([0.0, 1.0], dtype=float),
        sigma_ref_param_index=int_minus_one.copy(),
        cut_ref_param_index=int_minus_one.copy(),
        core_ref_param_index=int_minus_one.copy(),
        sigma_log_scatter_param_index=int_minus_one.copy(),
        mass_log_scatter_param_index=int_minus_one.copy(),
    )
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
        reliability=np.ones(2, dtype=float),
    )
    bin_data = cluster_solver.BinData(
        effective_z_source=2.0,
        family_ids=["1"],
        family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        x_obs=family.x_obs,
        y_obs=family.y_obs,
        sigma_per_image=np.full(2, 0.1, dtype=float),
        reliability_per_image=np.ones(2, dtype=float),
    )
    geometry_cache = GeometryCache(
        effective_z_source_values=[2.0],
        exact_z_source_values=[2.0],
        family_z_source_map={"1": 2.0},
        family_effective_z_source_map={"1": 2.0},
        dpie_sigma0_factor_by_effective_z={2.0: 1.0},
        dpie_sigma0_factor_by_exact_z={2.0: 1.0},
        lens_quadrature_z=[0.4],
        lens_quadrature_weights=[1.0],
        effective_z_quadrature_z=[[2.0]],
        effective_z_quadrature_weights=[[1.0]],
        exact_z_quadrature_z=[[2.0]],
        exact_z_quadrature_weights=[[1.0]],
    )
    parameter_specs = [
        ParameterSpec(
            name="sub_scale",
            sample_name="sub_scale",
            potential_id="sub",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_ref",
            prior_kind="uniform",
            lower=0.0,
            upper=2.0,
            step=0.1,
            component_family="scaling",
        )
    ]
    if include_source_positions:
        parameter_specs.extend(
            [
                ParameterSpec(
                    name="source.1.beta_x",
                    sample_name="source_1_beta_x",
                    potential_id="1",
                    profile_type=0,
                    field="beta_x",
                    prior_kind="normal",
                    lower=-np.inf,
                    upper=np.inf,
                    step=0.1,
                    mean=0.0,
                    std=1.0,
                    physical_mean=0.0,
                    physical_std=1.0,
                    component_family="source_position",
                ),
                ParameterSpec(
                    name="source.1.beta_y",
                    sample_name="source_1_beta_y",
                    potential_id="1",
                    profile_type=0,
                    field="beta_y",
                    prior_kind="normal",
                    lower=-np.inf,
                    upper=np.inf,
                    step=0.1,
                    mean=0.0,
                    std=1.0,
                    physical_mean=0.0,
                    physical_std=1.0,
                    component_family="source_position",
                ),
            ]
        )
    if fit_cosmology_flat_wcdm:
        parameter_specs.extend(
            [
                ParameterSpec(
                    name="cosmology.Om0",
                    sample_name=cluster_solver.COSMOLOGY_OM0_SAMPLE_NAME,
                    potential_id="cosmology",
                    profile_type=0,
                    field="Om0",
                    prior_kind="uniform",
                    lower=0.05,
                    upper=0.6,
                    step=0.01,
                    component_family="cosmology",
                ),
                ParameterSpec(
                    name="cosmology.w0",
                    sample_name=cluster_solver.COSMOLOGY_W0_SAMPLE_NAME,
                    potential_id="cosmology",
                    profile_type=0,
                    field="w0",
                    prior_kind="uniform",
                    lower=-2.0,
                    upper=-0.3,
                    step=0.05,
                    component_family="cosmology",
                ),
            ]
        )

    return BuildState(
        run_name="surrogate",
        par_path="mock.par",
        cosmo_config={"class": "FlatLambdaCDM", "H0": 70.0, "Om0": 0.3},
        z_lens=0.4,
        sigma_arcsec=0.1,
        parsed={},
        parameter_specs=parameter_specs,
        base_components=[],
        packed_lens_spec=packed,
        family_data=[family],
        bin_data=[bin_data],
        arc_data=None,
        lens_model_list=["DPIE_NIE", "DPIE_NIE"],
        reference=(0, 0.0, 0.0),
        fit_mode="joint",
        potfiles=[],
        scaling_component_records=[],
        geometry_cache=geometry_cache,
        fit_cosmology_flat_wcdm=fit_cosmology_flat_wcdm,
        source_position_parameterization="direct" if include_source_positions else "prior-whitened",
    )


def test_active_scaling_candidate_selection_drives_independent_mixture() -> None:
    potfiles = [
        {
            "id": "members",
            "mag0": 20.0,
            "catalog_df": pd.DataFrame(
                {
                    "id": ["faint", "bright", "mid"],
                    "ra": [10.0, 10.0, 10.0],
                    "dec": [0.0, 0.0, 0.0],
                    "catalog_mag": [21.0, 18.0, 19.5],
                }
            ),
        }
    ]
    reference = (0, 10.0, 0.0)
    images_df = pd.DataFrame(columns=["ra", "dec"])

    selected, rank_info, requested_counts, counts = cluster_solver._select_active_scaling_candidates(
        potfiles,
        reference,
        images_df,
        [1],
        active_scaling_selection="fixed",
        active_scaling_cumulative_fraction=0.995,
        active_scaling_min=1,
    )
    selected_adaptive, _rank_info_adaptive, _requested_adaptive, counts_adaptive = (
        cluster_solver._select_active_scaling_candidates(
            potfiles,
            reference,
            images_df,
            [2],
            active_scaling_selection="adaptive",
            active_scaling_cumulative_fraction=0.995,
            active_scaling_min=1,
        )
    )
    selected_all, _rank_info_all, requested_counts_all, counts_all = cluster_solver._select_active_scaling_candidates(
        potfiles,
        reference,
        images_df,
        [-1],
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=0.995,
        active_scaling_min=1,
    )

    assert selected == [{1}]
    assert requested_counts == {"members": 1}
    assert counts == {"members": 1}
    assert rank_info[(0, 1)]["rank"] == 1
    assert selected_adaptive == [{1, 2}]
    assert counts_adaptive == {"members": 2}
    assert selected_all == [{0, 1, 2}]
    assert requested_counts_all == {"members": 3}
    assert counts_all == {"members": 3}


def _write_active_scaling_potfile_fixture(tmp_path: Path) -> Path:
    image_catalog_path = tmp_path / "obs_arcs.cat"
    image_catalog_path.write_text(
        "#REFERENCE 0\n"
        "1.a 10.0000 0.0000 1.0 1.0 0.0 2.0 25.0\n"
        "1.b 9.9998 0.0000 1.0 1.0 0.0 2.0 25.0\n",
        encoding="utf-8",
    )
    member_catalog_path = tmp_path / "members.cat"
    member_catalog_path.write_text(
        "#REFERENCE 0\n"
        "faint 10.0000 0.0000 1.0 1.0 0.0 21.0 1.0 nan\n"
        "bright 10.0001 0.0000 1.0 1.0 0.0 18.0 1.0 nan\n"
        "mid 10.0002 0.0000 1.0 1.0 0.0 19.5 1.0 nan\n",
        encoding="utf-8",
    )
    par_path = tmp_path / "input.par"
    par_path.write_text(
        """
runmode
    reference 3 10.0 0.0
    end

image
    multfile 1 obs_arcs.cat
    end

cosmology
    H0 70.0
    omega 0.3
    lambda 0.7
    end

potentiel 1
    profil 81
    x_centre 0.0
    y_centre 0.0
    ellipticite 0.0
    angle_pos 0.0
    core_radius 1.0
    cut_radius 100.0
    v_disp 700.0
    z_lens 0.3
    end

potfile
    filein 3 members.cat
    zlens 0.3
    type 81
    corekpc 0.15
    mag0 20.0
    sigma 1 10.0 200.0
    cutkpc 1 1.0 40.0
    vdslope 0 4.0 0.0
    slope 0 4.0 0.0
    end
fini
""",
        encoding="utf-8",
    )
    return par_path


def test_perturbation_discovery_score_table_matches_threshold_rule() -> None:
    state = SimpleNamespace(
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g1",
                "catalog_row_index": 0,
                "component_index": 10,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g2",
                "catalog_row_index": 1,
                "component_index": 11,
            },
        ],
        family_data=[
            SimpleNamespace(family_id="1", image_labels=["a", "b"]),
        ],
    )
    flat_data = SimpleNamespace(
        family_ids=("1",),
        global_family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        local_image_index_per_image=np.asarray([0, 1], dtype=np.int32),
    )
    table, score, selected_pair, selected_galaxy = cluster_solver._perturbation_discovery_score_table(
        state,
        flat_data,
        np.asarray([10, 11], dtype=np.int32),
        np.asarray([20, 11], dtype=np.int32),
        np.asarray([[0.2, 0.0], [0.05, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.6], [0.0, 0.0]], dtype=float),
        alpha_tol=0.1,
        jacobian_tol=0.5,
        jacobian_weight=1.0,
    )

    np.testing.assert_allclose(score, [[2.0, 1.2], [0.5, 0.0]])
    np.testing.assert_array_equal(selected_pair, [[True, True], [False, False]])
    np.testing.assert_array_equal(selected_galaxy, [True, False])
    assert len(table) == 4
    assert table.loc[0, "catalog_id"] == "g1"
    assert table.loc[0, "component_index"] == 10
    assert table.loc[0, "evaluation_component_index"] == 20
    assert table.loc[0, "family_id"] == "1"
    assert table.loc[0, "image_label"] == "a"
    assert bool(table.loc[0, "selected_pair"]) is True
    assert bool(table.loc[2, "selected_galaxy"]) is False


def test_perturbation_discovery_scores_free_branch_but_selects_scaling_identity() -> None:
    state = SimpleNamespace(
        sigma_arcsec=0.02,
        independent_scaling_free_log_sigma_offset_width=0.10,
        potfiles=[{"id": "members"}],
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g1",
                "catalog_row_index": 0,
                "component_index": 10,
                "free_component_index": 20,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g2",
                "catalog_row_index": 1,
                "component_index": 11,
                "free_component_index": -1,
            },
        ],
        family_data=[
            SimpleNamespace(family_id="1", image_labels=["a", "b"]),
        ],
    )
    flat_data = SimpleNamespace(
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
        family_ids=("1",),
        global_family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        local_image_index_per_image=np.asarray([0, 1], dtype=np.int32),
    )

    class FakeEvaluator:
        scaling_component_indices = np.asarray([10, 11], dtype=np.int32)
        flat_critical_arc_data = flat_data
        perturbation_discovery_jacobian_weight = 1.0

        def _physical_parameter_vector(self, reference):
            return reference

        def _build_flat_packed_lens_state_with_validity_from_physical(
            self,
            _physical_params,
            _flat_data,
            *,
            stop_gradient,
        ):
            del stop_gradient
            return SimpleNamespace(), SimpleNamespace(is_valid=np.asarray(True), reason_flags=np.asarray([], dtype=bool))

        def _record_invalid_state_host(self, _reason_flags):
            raise AssertionError("state should be valid")

        def _flat_component_alpha_and_jacobian_delta_rows_for_components(
            self,
            _x_obs,
            _y_obs,
            _packed_state,
            components,
        ):
            np.testing.assert_array_equal(np.asarray(components, dtype=np.int32), np.asarray([20, 11], dtype=np.int32))
            return (
                np.asarray([[0.2, 0.0], [0.0, 0.0]], dtype=float),
                np.zeros((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
                np.zeros((2, 2), dtype=float),
            )

    selected, diagnostics, table = cluster_solver._perturbation_discovery_union_from_evaluator(
        state,
        FakeEvaluator(),
        np.zeros(1, dtype=float),
    )

    assert selected == [{0}]
    assert diagnostics["count"] == 1
    assert diagnostics["evaluated_free_branch_galaxies"] == 1
    assert table.loc[0, "component_index"] == 10
    assert table.loc[0, "evaluation_component_index"] == 20
    assert bool(table.loc[0, "selected_galaxy"]) is True


def test_perturbation_discovery_top_k_caps_threshold_passing_members() -> None:
    state = SimpleNamespace(
        sigma_arcsec=0.2,
        independent_scaling_free_log_sigma_offset_width=0.10,
        potfiles=[{"id": "members"}],
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g1",
                "catalog_row_index": 0,
                "component_index": 10,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g2",
                "catalog_row_index": 1,
                "component_index": 11,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "catalog_id": "g3",
                "catalog_row_index": 2,
                "component_index": 12,
            },
        ],
        family_data=[SimpleNamespace(family_id="1", image_labels=["a", "b"])],
    )
    flat_data = SimpleNamespace(
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
        family_ids=("1",),
        global_family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        local_image_index_per_image=np.asarray([0, 1], dtype=np.int32),
    )

    class FakeEvaluator:
        scaling_component_indices = np.asarray([10, 11, 12], dtype=np.int32)
        flat_critical_arc_data = flat_data
        perturbation_discovery_jacobian_weight = 1.0
        perturbation_discovery_top_k = 1

        def _physical_parameter_vector(self, reference):
            return reference

        def _build_flat_packed_lens_state_with_validity_from_physical(
            self,
            _physical_params,
            _flat_data,
            *,
            stop_gradient,
        ):
            del stop_gradient
            return SimpleNamespace(), SimpleNamespace(is_valid=np.asarray(True), reason_flags=np.asarray([], dtype=bool))

        def _record_invalid_state_host(self, _reason_flags):
            raise AssertionError("state should be valid")

        def _flat_component_alpha_and_jacobian_delta_rows_for_components(
            self,
            _x_obs,
            _y_obs,
            _packed_state,
            components,
        ):
            np.testing.assert_array_equal(np.asarray(components, dtype=np.int32), np.asarray([10, 11, 12], dtype=np.int32))
            return (
                np.asarray([[3.0, 0.0], [2.0, 4.0], [1.0, 3.0]], dtype=float),
                np.zeros((3, 2), dtype=float),
                np.zeros((3, 2), dtype=float),
                np.zeros((3, 2), dtype=float),
                np.zeros((3, 2), dtype=float),
                np.zeros((3, 2), dtype=float),
            )

    selected, diagnostics, table = cluster_solver._perturbation_discovery_union_from_evaluator(
        state,
        FakeEvaluator(),
        np.zeros(1, dtype=float),
    )

    assert selected == [{0, 1}]
    assert diagnostics["selection_mode"] == "top_k"
    assert diagnostics["top_k_requested"] == 1
    assert diagnostics["count"] == 2
    assert diagnostics["pairs"] == 2
    assert diagnostics["threshold_selected_galaxies"] == 3
    assert diagnostics["top_k_selected_galaxies"] == 2
    assert set(table.columns) >= {
        "selection_mode",
        "top_k_requested",
        "rank_score",
        "rank_position",
        "image_rank_position",
    }
    selected_by_id = table.groupby("catalog_id")["selected_galaxy"].first().to_dict()
    assert selected_by_id == {"g1": True, "g2": True, "g3": False}
    rank_by_id = table.groupby("catalog_id")["rank_position"].first().to_dict()
    assert rank_by_id == {"g1": 2, "g2": 1, "g3": 3}
    selected_pairs = table.pivot(index="catalog_id", columns="image_index", values="selected_pair")
    assert selected_pairs.to_dict() == {
        0: {"g1": True, "g2": False, "g3": False},
        1: {"g1": False, "g2": True, "g3": False},
    }
    image_rank = table.pivot(index="catalog_id", columns="image_index", values="image_rank_position")
    assert image_rank.to_dict() == {
        0: {"g1": 1, "g2": 2, "g3": 3},
        1: {"g1": 3, "g2": 1, "g3": 2},
    }


def test_perturbation_discovery_top_k_larger_than_candidates_selects_all_threshold_passers() -> None:
    score = np.asarray([[1.2, 1.1], [1.4, 1.3]], dtype=float)
    selected, selected_pair, rank_score, rank_position, image_rank_position = (
        cluster_solver._perturbation_discovery_top_k_selected(score, 10)
    )

    np.testing.assert_array_equal(selected, np.asarray([True, True]))
    np.testing.assert_array_equal(selected_pair, np.asarray([[True, True], [True, True]]))
    np.testing.assert_allclose(rank_score, np.asarray([1.2, 1.4]))
    np.testing.assert_array_equal(rank_position, np.asarray([2, 1]))
    np.testing.assert_array_equal(image_rank_position, np.asarray([[2, 2], [1, 1]]))


def test_perturbation_discovery_top_k_never_admits_below_threshold_members() -> None:
    # top_k is a pure cap: members whose plausible offset is undetectable
    # (score <= 1) are never promoted, no matter how large top_k is.
    score = np.asarray([[0.2, 0.1], [0.4, 0.3]], dtype=float)
    selected, selected_pair, _rank_score, _rank_position, _image_rank_position = (
        cluster_solver._perturbation_discovery_top_k_selected(score, 10)
    )

    np.testing.assert_array_equal(selected, np.asarray([False, False]))
    assert not selected_pair.any()


def test_perturbation_discovery_top_k_breaks_ties_by_candidate_order() -> None:
    score = np.asarray([[1.5, 0.0], [1.5, 2.0], [0.5, 2.0]], dtype=float)
    selected, selected_pair, rank_score, rank_position, image_rank_position = (
        cluster_solver._perturbation_discovery_top_k_selected(score, 1)
    )

    np.testing.assert_array_equal(selected_pair, np.asarray([[True, False], [False, True], [False, False]]))
    np.testing.assert_array_equal(selected, np.asarray([True, True, False]))
    np.testing.assert_allclose(rank_score, np.asarray([1.5, 2.0, 2.0]))
    np.testing.assert_array_equal(rank_position, np.asarray([3, 1, 2]))
    np.testing.assert_array_equal(image_rank_position, np.asarray([[1, 3], [2, 1], [3, 2]]))


def _minimal_active_gate_state() -> BuildState:
    state = _minimal_stage4_surrogate_state()
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "mag0": 20.0,
            "catalog_df": pd.DataFrame(
                {
                    "id": ["galaxy-a"],
                    "ra": [0.0],
                    "dec": [0.0],
                    "catalog_mag": [19.0],
                    "catalog_color": [0.8],
                }
            ),
        }
    ]
    records = [
        {
            "potfile_id": "members",
            "potfile_order": 0,
            "catalog_row_index": 0,
            "component_index": 1,
            "catalog_id": "galaxy-a",
            "catalog_mag": 19.0,
            "catalog_color": 0.8,
            "independent_magnitude_feature": 0.5,
            "x_centre": 1.0,
            "y_centre": 1.0,
            "rank": 1,
            "importance": 2.0,
            "min_distance_arcsec": 0.75,
            "selected_independent": False,
            "free_component_index": -1,
        }
    ]
    active_specs, active_indices = cluster_solver._build_active_scaling_parameter_specs(
        potfiles,
        records,
        start_index=len(state.parameter_specs),
        prior_prob=0.5,
        logit_prior_sigma=1.5,
        mag_slope_prior_sigma=1.0,
        local_logit_prior_sigma=0.75,
        active_selected_counts={"members": 1},
        freeze_threshold=0.5,
    )
    packed, updated_records = cluster_solver._packed_lens_spec_with_active_scaling_gates(
        state.packed_lens_spec,
        records,
        active_indices,
    )
    return replace(
        state,
        parameter_specs=[*state.parameter_specs, *active_specs],
        packed_lens_spec=packed,
        potfiles=potfiles,
        scaling_component_records=updated_records,
        infer_active_scaling=True,
    )


def test_active_scaling_gate_specs_append_after_ordinary_specs() -> None:
    state = _minimal_stage4_surrogate_state()
    records = [
        {
            "potfile_id": "members",
            "potfile_order": 0,
            "catalog_row_index": 0,
            "component_index": 1,
            "catalog_id": "galaxy-a",
            "independent_magnitude_feature": 0.5,
        }
    ]
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame({"id": ["galaxy-a"]}),
        }
    ]

    specs, indices = cluster_solver._build_active_scaling_parameter_specs(
        potfiles,
        records,
        start_index=len(state.parameter_specs),
        prior_prob=0.25,
        logit_prior_sigma=1.5,
        mag_slope_prior_sigma=1.0,
        local_logit_prior_sigma=0.75,
    )

    assert [spec.component_family for spec in specs] == [cluster_solver.ACTIVE_SCALING_GATE_COMPONENT_FAMILY] * 3
    assert specs[0].field == "active_gate_mag_slope"
    assert specs[2].field == "active_gate_logit_offset"
    assert specs[2].sample_site_name == "members_active_gate_logit_offset"
    assert specs[2].sample_site_index == 0
    assert indices[1]["active_gate_mag_slope"] == len(state.parameter_specs)
    assert indices[1]["active_gate_intercept"] == len(state.parameter_specs) + 1
    assert indices[1]["active_gate_logit_offset"] == len(state.parameter_specs) + 2


def test_active_scaling_probability_responds_to_gate_parameters() -> None:
    state = _minimal_active_gate_state()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        active_scaling_galaxies=[0],
        active_scaling_selection="fixed",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    theta = cluster_solver._default_theta(state.parameter_specs)
    theta[1] = 2.0
    theta[2] = 0.0
    theta[3] = -1.0
    physical = evaluator._physical_parameter_vector(jnp.asarray(theta, dtype=jnp.float64))

    p_active = np.asarray(evaluator._active_scaling_probability_from_physical(physical), dtype=float)
    assert p_active[0] == pytest.approx(1.0)
    assert p_active[1] == pytest.approx(0.5, abs=1.0e-6)

    theta[3] = 1.0
    physical = evaluator._physical_parameter_vector(jnp.asarray(theta, dtype=jnp.float64))
    p_active = np.asarray(evaluator._active_scaling_probability_from_physical(physical), dtype=float)
    assert p_active[1] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)), abs=1.0e-6)


def test_active_population_mixture_terms_are_stable_and_directional() -> None:
    q = jnp.asarray([0.25, 0.25, 0.25, 1.0e-15, 1.0 - 1.0e-15], dtype=jnp.float64)
    delta = jnp.asarray([0.0, -30.0, 30.0, 0.0, 0.0], dtype=jnp.float64)

    terms, responsibility = cluster_solver._active_population_mixture_terms(q, delta)

    assert np.all(np.isfinite(np.asarray(terms)))
    assert np.all(np.isfinite(np.asarray(responsibility)))
    assert float(responsibility[0]) == pytest.approx(0.25, abs=1.0e-8)
    assert float(responsibility[1]) > 0.999999
    assert float(responsibility[2]) < 1.0e-10
    assert 0.0 <= float(responsibility[3]) <= 1.0
    assert 0.0 <= float(responsibility[4]) <= 1.0


def test_linearized_source_population_delta_centers_family_translation() -> None:
    delta = cluster_solver._linearized_source_plane_population_delta(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        delta_beta_x=jnp.asarray([[1.0, 1.0]], dtype=jnp.float64),
        delta_beta_y=jnp.asarray([[0.5, 0.5]], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        inv_abs_mu=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
    )

    assert float(delta[0]) == pytest.approx(0.0, abs=1.0e-10)


def test_linearized_source_population_delta_is_directional_and_matches_small_shift() -> None:
    beta_x = jnp.asarray([0.0, 2.0], dtype=jnp.float64)
    beta_y = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    family_idx = jnp.asarray([0, 0], dtype=jnp.int32)
    reliability = jnp.asarray([0.999999, 0.999999], dtype=jnp.float64)
    constraints = jnp.asarray([True, True])
    sigma = jnp.ones(2, dtype=jnp.float64)
    zeros = jnp.zeros(2, dtype=jnp.float64)
    toward = jnp.asarray([[0.0, -0.1]], dtype=jnp.float64)
    away = jnp.asarray([[0.0, 0.1]], dtype=jnp.float64)

    toward_delta = cluster_solver._linearized_source_plane_population_delta(
        beta_x,
        beta_y,
        toward,
        jnp.zeros_like(toward),
        family_idx,
        1,
        sigma,
        reliability,
        constraints,
        jnp.asarray(0.0, dtype=jnp.float64),
        zeros,
        zeros,
        jnp.ones(2, dtype=jnp.float64),
        1.0e-9,
    )
    away_delta = cluster_solver._linearized_source_plane_population_delta(
        beta_x,
        beta_y,
        away,
        jnp.zeros_like(away),
        family_idx,
        1,
        sigma,
        reliability,
        constraints,
        jnp.asarray(0.0, dtype=jnp.float64),
        zeros,
        zeros,
        jnp.ones(2, dtype=jnp.float64),
        1.0e-9,
    )

    assert float(toward_delta[0]) > 0.0
    assert float(away_delta[0]) < 0.0

    small_shift = jnp.asarray([[0.0, -1.0e-3]], dtype=jnp.float64)
    linearized = cluster_solver._linearized_source_plane_population_delta(
        beta_x,
        beta_y,
        small_shift,
        jnp.zeros_like(small_shift),
        family_idx,
        1,
        sigma,
        reliability,
        constraints,
        jnp.asarray(0.0, dtype=jnp.float64),
        zeros,
        zeros,
        jnp.ones(2, dtype=jnp.float64),
        1.0e-9,
    )
    base = _source_plane_bin_loglike(
        beta_x=beta_x,
        beta_y=beta_y,
        family_idx=family_idx,
        n_families=1,
        sigma_per_image=sigma,
        reliability_per_image=reliability,
        image_has_constraint=constraints,
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=zeros,
        scatter_var_y=zeros,
        inv_abs_mu=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )
    shifted = _source_plane_bin_loglike(
        beta_x=beta_x + small_shift[0],
        beta_y=beta_y,
        family_idx=family_idx,
        n_families=1,
        sigma_per_image=sigma,
        reliability_per_image=reliability,
        image_has_constraint=constraints,
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=zeros,
        scatter_var_y=zeros,
        inv_abs_mu=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )

    assert float(linearized[0]) == pytest.approx(float(shifted - base), rel=1.0e-4, abs=1.0e-8)


def test_linearized_local_jacobian_source_only_delta_matches_source_for_identity_jacobian() -> None:
    beta_x = jnp.asarray([0.0, 2.0], dtype=jnp.float64)
    beta_y = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    delta_x = jnp.asarray([[0.0, -0.25]], dtype=jnp.float64)
    delta_y = jnp.asarray([[0.0, 0.0]], dtype=jnp.float64)
    family_idx = jnp.asarray([0, 0], dtype=jnp.int32)
    reliability = jnp.asarray([0.999999, 0.999999], dtype=jnp.float64)
    constraints = jnp.asarray([True, True])
    sigma = jnp.ones(2, dtype=jnp.float64)
    zeros = jnp.zeros(2, dtype=jnp.float64)

    source_delta = cluster_solver._linearized_source_plane_population_delta(
        beta_x,
        beta_y,
        delta_x,
        delta_y,
        family_idx,
        1,
        sigma,
        reliability,
        constraints,
        jnp.asarray(0.0, dtype=jnp.float64),
        zeros,
        zeros,
        jnp.ones(2, dtype=jnp.float64),
        1.0e-9,
    )
    local_delta = cluster_solver._linearized_local_jacobian_population_delta_source_only(
        beta_x,
        beta_y,
        delta_x,
        delta_y,
        family_idx,
        1,
        sigma,
        reliability,
        constraints,
        jnp.asarray(0.0, dtype=jnp.float64),
        zeros,
        zeros,
        jnp.ones(2, dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        jnp.zeros(2, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        1.0e-9,
    )

    np.testing.assert_allclose(np.asarray(local_delta), np.asarray(source_delta), rtol=1.0e-8, atol=1.0e-8)


def test_active_scaling_diagnostics_and_summary_write_outputs(tmp_path: Path) -> None:
    state = _minimal_active_gate_state()
    samples = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    best_fit = samples[-1]
    scaling_rank_df = pd.DataFrame(state.scaling_component_records)
    scaling_rank_df["selected_active"] = False

    table = _active_scaling_diagnostics_table(
        state.parameter_specs,
        samples,
        best_fit,
        scaling_rank_df,
        state.packed_lens_spec,
        freeze_threshold=0.5,
    )
    _plot_active_scaling_summary(tmp_path, table, parameter_specs=state.parameter_specs, freeze_threshold=0.5)

    assert table.shape[0] == 1
    assert table.loc[0, "p_active_median"] == pytest.approx(0.5, abs=1.0e-6)
    assert bool(table.loc[0, "frozen_active"])
    assert _plot_path(tmp_path, "active_scaling_summary.pdf").is_file()
    assert not (tmp_path / "active_scaling_summary.png").exists()
    assert table.loc[0, "active_gate_intercept_prior_kind"] == "normal"
    assert table.loc[0, "active_gate_mag_slope_prior_kind"] == "truncated_normal"
    assert float(table.loc[0, "active_gate_intercept_prior_mean"]) == pytest.approx(0.0)
    assert float(table.loc[0, "active_gate_mag_slope_prior_lower"]) == pytest.approx(0.0)


def test_active_scaling_population_diagnostics_freeze_from_membership() -> None:
    state = _minimal_active_gate_state()
    samples = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    scaling_rank_df = pd.DataFrame(state.scaling_component_records)
    scaling_rank_df["selected_active"] = False
    diagnostics = {
        "component_indices": np.asarray([1], dtype=np.int32),
        "gate_samples": np.asarray([[0.2], [0.2], [0.2]], dtype=float),
        "membership_samples": np.asarray([[0.9], [0.9], [0.9]], dtype=float),
        "delta_samples": np.asarray([[-4.0], [-4.0], [-4.0]], dtype=float),
        "gate_map": np.asarray([0.2], dtype=float),
        "membership_map": np.asarray([0.9], dtype=float),
        "delta_map": np.asarray([-4.0], dtype=float),
    }

    table = _active_scaling_diagnostics_table(
        state.parameter_specs,
        samples,
        samples[0],
        scaling_rank_df,
        state.packed_lens_spec,
        freeze_threshold=0.5,
        active_population_diagnostics=diagnostics,
        active_inference_likelihood="population",
    )

    assert table.loc[0, "p_active_gate_median"] == pytest.approx(0.2)
    assert table.loc[0, "p_active_membership_median"] == pytest.approx(0.9)
    assert table.loc[0, "p_active_median"] == pytest.approx(0.9)
    assert table.loc[0, "active_loglike_delta_median"] == pytest.approx(-4.0)
    assert bool(table.loc[0, "frozen_active"])
    assert table.loc[0, "active_inference_likelihood"] == "population"


def _active_scaling_resume_diagnostics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "potfile_id": ["members", "members"],
            "potfile_order": [0, 0],
            "catalog_id": ["galaxy-a", "galaxy-b"],
            "catalog_row_index": [0, 1],
            "rank": [1, 2],
            "component_index": [1, 2],
            "catalog_mag": [18.5, 21.0],
            "active_magnitude_feature": [1.0, -1.0],
            "importance": [10.0, 2.0],
            "min_distance_arcsec": [0.5, 4.0],
            "x_centre": [0.0, 1.0],
            "y_centre": [0.0, -1.0],
            "active_gate_intercept_parameter_index": [10, 10],
            "active_gate_mag_slope_parameter_index": [11, 11],
            "active_gate_logit_offset_parameter_index": [12, 13],
            "active_gate_intercept_prior_kind": ["normal", "normal"],
            "active_gate_intercept_prior_lower": [-np.inf, -np.inf],
            "active_gate_intercept_prior_upper": [np.inf, np.inf],
            "active_gate_intercept_prior_mean": [0.0, 0.0],
            "active_gate_intercept_prior_std": [1.5, 1.5],
            "active_gate_mag_slope_prior_kind": ["truncated_normal", "truncated_normal"],
            "active_gate_mag_slope_prior_lower": [0.0, 0.0],
            "active_gate_mag_slope_prior_upper": [np.inf, np.inf],
            "active_gate_mag_slope_prior_mean": [0.0, 0.0],
            "active_gate_mag_slope_prior_std": [1.0, 1.0],
            "p_active_mean": [0.8, 0.2],
            "p_active_median": [0.85, 0.15],
            "p_active_p16": [0.70, 0.05],
            "p_active_p84": [0.95, 0.35],
            "p_active_map": [0.9, 0.1],
            "frozen_active": [True, False],
        }
    )


def test_active_scaling_prior_curve_reconstructs_from_resume_metadata() -> None:
    diagnostics_df = _active_scaling_resume_diagnostics_df()

    curve = _gate_prior_sigmoid_curve(
        diagnostics_df,
        diagnostics_df,
        [],
        intercept_column="active_gate_intercept_parameter_index",
        slope_column="active_gate_mag_slope_parameter_index",
        feature_column="active_magnitude_feature",
    )

    assert curve is not None
    assert {"mag", "center", "p16", "p84"}.issubset(curve)
    assert curve["mag"].shape == curve["center"].shape == curve["p16"].shape == curve["p84"].shape
    assert np.all(np.isfinite(curve["center"]))
    assert np.all((curve["center"] >= 0.0) & (curve["center"] <= 1.0))


def test_resume_regenerates_active_scaling_summary_from_diagnostics(tmp_path: Path) -> None:
    state = replace(_minimal_active_gate_state(), parameter_specs=[])
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _active_scaling_resume_diagnostics_df().to_csv(tables_dir / "active_scaling_diagnostics.csv", index=False)

    regenerated = cluster_solver._regenerate_active_scaling_summary_from_diagnostics(
        argparse.Namespace(quiet=True),
        tmp_path,
        state,
        {"active_scaling_freeze_threshold": 0.4},
        {},
    )

    assert regenerated is True
    assert _plot_path(tmp_path, "active_scaling_summary.pdf").is_file()
    assert not (tmp_path / "active_scaling_summary.png").exists()


def test_resume_active_scaling_summary_skips_missing_or_empty_diagnostics(tmp_path: Path) -> None:
    state = _minimal_active_gate_state()
    args = argparse.Namespace(quiet=True)

    assert (
        cluster_solver._regenerate_active_scaling_summary_from_diagnostics(args, tmp_path, state, {}, {})
        is False
    )
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _active_scaling_resume_diagnostics_df().iloc[0:0].to_csv(
        tables_dir / "active_scaling_diagnostics.csv",
        index=False,
    )

    assert (
        cluster_solver._regenerate_active_scaling_summary_from_diagnostics(args, tmp_path, state, {}, {})
        is False
    )
    assert not _plot_path(tmp_path, "active_scaling_summary.pdf").exists()


def test_load_frozen_active_scaling_from_previous_stage_maps_by_catalog_identity(tmp_path: Path) -> None:
    state = _stage4_state_with_independent_and_inactive_scaling()
    stage3_dir = tmp_path / "fit" / "stage3_image_plane"
    tables_dir = stage3_dir / "tables"
    tables_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "potfile_id": ["members", "members"],
            "catalog_id": ["galaxy-a", "galaxy-b"],
            "component_index": [101, 102],
            "frozen_active": [False, True],
        }
    ).to_csv(tables_dir / "active_scaling_diagnostics.csv", index=False)

    frozen_indices, diagnostics = cluster_solver._load_frozen_active_scaling_from_previous_stage(stage3_dir, state)

    assert frozen_indices.tolist() == [3]
    assert diagnostics["source"] == "diagnostics_csv"
    assert diagnostics["frozen_active_count"] == 1
    assert diagnostics["frozen_inactive_count"] == 1


def test_previous_stage_frozen_active_scaling_missing_split_has_actionable_error(tmp_path: Path) -> None:
    state = _stage4_state_with_independent_and_inactive_scaling()

    with pytest.raises(ValueError, match="Stage 4 must reuse a frozen active-scaling split"):
        cluster_solver._apply_previous_stage_frozen_active_scaling(
            argparse.Namespace(quiet=True),
            state,
            tmp_path / "fit" / "stage3_image_plane",
            require_frozen_active_scaling=True,
        )


def test_independent_scaling_candidates_are_active_and_not_inactive_surrogate() -> None:
    state = _minimal_stage4_surrogate_state()
    state = replace(
        state,
        potfiles=[{"id": "members", "mag0": 20.0}],
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 1,
                "catalog_id": "galaxy-a",
                "catalog_mag": 20.0,
                "catalog_color": 1.0,
                "x_centre": 1.0,
                "y_centre": 1.0,
                "selected_independent": True,
            }
        ],
    )

    evaluator = cluster_solver.ClusterJAXEvaluator(
        state,
        match_tolerance_arcsec=cluster_solver.DEFAULT_MATCH_TOLERANCE,
        active_scaling_galaxies=[0],
        active_scaling_selection="fixed",
    )

    assert evaluator.independent_scaling_component_indices.tolist() == [1]
    assert evaluator.active_scaling_component_indices.tolist() == [1]
    assert evaluator.inactive_scaling_component_indices.tolist() == []
    assert evaluator.exact_scaling_component_indices.tolist() == [1]
    assert evaluator.cached_scaling_component_indices.tolist() == []
    assert evaluator.free_correction_scaling_component_indices.tolist() == [1]
    assert evaluator.free_correction_free_component_indices.tolist() == []
    assert evaluator.excluded_scaling_component_indices.tolist() == []
    assert bool(evaluator.scaling_rank_df.loc[0, "selected_independent"])
    assert bool(evaluator.scaling_rank_df.loc[0, "selected_active"])


def test_independent_scaling_vector_sample_site_uses_full_uniform_vector() -> None:
    specs = [
        ParameterSpec(
            name=f"members.galaxy{i}.independent_free_v_disp",
            sample_name=f"members_galaxy{i}_independent_free_v_disp",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_v_disp",
            prior_kind="uniform",
            lower=np.log(30.0),
            upper=np.log(500.0),
            step=0.1,
            component_family="independent_scaling",
            transform_kind="log_positive",
            sample_site_name="members_independent_free_v_disp",
            sample_site_index=i,
        )
        for i in range(3)
    ]

    sites = cluster_solver._parameter_sample_sites(specs)
    distribution = cluster_solver._distribution_for_sample_site(sites[0], specs)

    assert len(sites) == 1
    assert sites[0].indices == (0, 1, 2)
    assert tuple(distribution.event_shape) == (3,)
    assert np.isfinite(float(distribution.log_prob(jnp.asarray([np.log(100.0), np.log(120.0), np.log(140.0)]))))


def _single_independent_branch_evaluator() -> cluster_solver.ClusterJAXEvaluator:
    int_missing = np.full(2, -1, dtype=np.int32)
    float_zero = np.zeros(2, dtype=float)
    packed = PackedLensSpec(
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        component_family=np.asarray(
            [cluster_solver.COMPONENT_FAMILY_SCALING, cluster_solver.COMPONENT_FAMILY_INDEPENDENT_FREE],
            dtype=np.int32,
        ),
        x_center_base=float_zero.copy(),
        y_center_base=float_zero.copy(),
        e1_base=float_zero.copy(),
        e2_base=float_zero.copy(),
        core_radius_kpc_base=np.asarray([1.0, 2.0], dtype=float),
        cut_radius_kpc_base=np.asarray([20.0, 50.0], dtype=float),
        v_disp_base=np.asarray([100.0, 200.0], dtype=float),
        gamma1_base=float_zero.copy(),
        gamma2_base=float_zero.copy(),
        x_center_param_index=int_missing.copy(),
        y_center_param_index=int_missing.copy(),
        e1_param_index=int_missing.copy(),
        e2_param_index=int_missing.copy(),
        core_radius_param_index=np.asarray([-1, 3], dtype=np.int32),
        cut_radius_param_index=np.asarray([-1, 4], dtype=np.int32),
        v_disp_param_index=np.asarray([-1, 2], dtype=np.int32),
        gamma1_param_index=int_missing.copy(),
        gamma2_param_index=int_missing.copy(),
        luminosity_ratio=np.ones(2, dtype=float),
        sigma_ref_base=np.asarray([100.0, 0.0], dtype=float),
        cut_ref_base=np.asarray([20.0, 0.0], dtype=float),
        core_ref_base=np.asarray([1.0, 0.0], dtype=float),
        alpha_sigma_base=np.asarray([0.25, 0.0], dtype=float),
        gamma_ml_base=np.asarray([0.0, 0.0], dtype=float),
        sigma_ref_param_index=int_missing.copy(),
        cut_ref_param_index=int_missing.copy(),
        core_ref_param_index=int_missing.copy(),
        alpha_sigma_param_index=int_missing.copy(),
        gamma_ml_param_index=int_missing.copy(),
        sigma_log_scatter_param_index=int_missing.copy(),
        mass_log_scatter_param_index=int_missing.copy(),
        independent_branch_role=np.asarray(
            [cluster_solver.INDEPENDENT_BRANCH_SCALING, cluster_solver.INDEPENDENT_BRANCH_FREE],
            dtype=np.int32,
        ),
        independent_magnitude_feature=np.asarray([0.0, 0.0], dtype=float),
    )
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.state = SimpleNamespace(packed_lens_spec=packed)
    evaluator.packed_spec_jax = cluster_solver.ClusterJAXEvaluator._prepare_packed_spec_arrays(evaluator)
    return evaluator


def test_packed_lens_state_independent_candidates_are_deterministically_free() -> None:
    evaluator = _single_independent_branch_evaluator()

    low_params = jnp.asarray([0.0, -60.0, 200.0, 2.0, 50.0, 1.0e-8], dtype=jnp.float64)
    low_state, low_details = evaluator._build_packed_lens_state_details_from_physical(
        low_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )
    high_params = jnp.asarray([0.0, 60.0, 200.0, 2.0, 50.0, 1.0e-8], dtype=jnp.float64)
    high_state, high_details = evaluator._build_packed_lens_state_details_from_physical(
        high_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )
    mid_params = jnp.asarray([0.0, 0.0, 200.0, 2.0, 50.0, 1.0e-8], dtype=jnp.float64)
    mid_state, mid_details = evaluator._build_packed_lens_state_details_from_physical(
        mid_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )

    assert tuple(low_state.sigma0.shape) == (2,)
    assert tuple(high_state.sigma0.shape) == (2,)
    np.testing.assert_allclose(np.asarray(low_details.independent_branch_weight), [0.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(high_details.independent_branch_weight), [0.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(mid_details.independent_branch_weight), [0.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(mid_state.sigma0), [0.0, 200.0**2 / 2.0], rtol=1.0e-10)


def test_packed_lens_state_scales_core_radius_with_beta_radius() -> None:
    evaluator = _single_independent_branch_evaluator()
    evaluator.state.packed_lens_spec.luminosity_ratio[0] = 4.0
    evaluator.packed_spec_jax = cluster_solver.ClusterJAXEvaluator._prepare_packed_spec_arrays(evaluator)

    physical_params = jnp.asarray([0.0, 0.0, 200.0, 2.0, 50.0, 1.0e-8], dtype=jnp.float64)
    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        physical_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )

    np.testing.assert_allclose(np.asarray(details.ra_raw)[0], 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.rs_raw)[0], 40.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Ra)[0], 2.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Rs)[0], 40.0, rtol=1.0e-12)


def test_packed_lens_state_applies_sampled_log_softening_length_in_quadrature() -> None:
    evaluator = _single_independent_branch_evaluator()
    evaluator.log_softening_length_param_index = 0

    physical_params = jnp.asarray([np.log(4.0), 0.0, 200.0, 3.0, 50.0], dtype=jnp.float64)
    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        physical_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )

    np.testing.assert_allclose(np.asarray(details.core_radius_kpc)[1], 3.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.softening_length_kpc), 4.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.log_softening_length_kpc), np.log(4.0), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.core_radius_effective_kpc)[1], 5.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Ra)[1], 5.0, rtol=1.0e-12)


def test_packed_lens_state_non_scaling_cut_is_excess_above_effective_core() -> None:
    evaluator = _single_independent_branch_evaluator()
    evaluator.state.packed_lens_spec.cut_radius_excess_above_effective_core = np.asarray(
        [False, True],
        dtype=bool,
    )
    evaluator.packed_spec_jax = cluster_solver.ClusterJAXEvaluator._prepare_packed_spec_arrays(evaluator)
    evaluator.log_softening_length_param_index = 0

    physical_params = jnp.asarray([np.log(4.0), 0.0, 200.0, 3.0, 7.0], dtype=jnp.float64)
    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        physical_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )

    np.testing.assert_allclose(np.asarray(details.core_radius_effective_kpc)[1], 5.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.rs_raw)[1], 12.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Rs)[1], 12.0, rtol=1.0e-12)
    assert np.asarray(details.rs_raw)[1] > np.asarray(details.ra_raw)[1]


def test_packed_lens_state_gamma_ml_controls_size_exponent() -> None:
    evaluator = _single_independent_branch_evaluator()
    packed_spec = evaluator.state.packed_lens_spec
    packed_spec.luminosity_ratio[0] = 4.0
    packed_spec.alpha_sigma_base = np.full(2, 0.25, dtype=float)
    packed_spec.gamma_ml_base = np.full(2, 0.0, dtype=float)
    packed_spec.alpha_sigma_param_index = np.asarray([0, -1], dtype=np.int32)
    packed_spec.gamma_ml_param_index = np.asarray([1, -1], dtype=np.int32)
    evaluator.packed_spec_jax = cluster_solver.ClusterJAXEvaluator._prepare_packed_spec_arrays(evaluator)

    physical_params = jnp.asarray([0.25, 0.2, 200.0, 2.0, 50.0, 1.0e-8], dtype=jnp.float64)
    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        physical_params,
        2.0,
        kpc_per_arcsec=jnp.asarray(1.0),
        dpie_sigma0_factor=jnp.asarray(1.0),
    )

    expected_beta = 0.7
    expected_size = float(4.0**expected_beta)
    np.testing.assert_allclose(np.asarray(details.alpha_sigma)[0], 0.25, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.beta_radius)[0], expected_beta, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(details.gamma_ml)[0], 2.0 * 0.25 + expected_beta - 1.0, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Ra)[0], expected_size, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(packed_state.Rs)[0], 20.0 * expected_size, rtol=1.0e-12)


def test_independent_scaling_parameter_specs_log_displacement_replaces_direct_free_parameters() -> None:
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame(
                {
                    "id": ["g1", "g2"],
                    "catalog_a": [2.0, 20.0],
                    "catalog_b": [1.0, 1.0],
                    "catalog_theta": [0.0, 45.0],
                }
            ),
        }
    ]

    specs, indices = cluster_solver._build_independent_scaling_parameter_specs(
        potfiles,
        [{0, 1}],
        start_index=20,
        log_sigma_offset_width=0.22,
        log_mass_offset_width=0.33,
    )

    fields = [spec.field for spec in specs]
    assert "independent_free_v_disp" not in fields
    assert "independent_free_core_radius_kpc" not in fields
    assert "independent_free_cut_radius_kpc" not in fields
    for field_name in (
        "independent_free_log_sigma_delta_unit",
        "independent_free_log_mass_delta_unit",
        "independent_free_e1",
        "independent_free_e2",
    ):
        field_specs = [spec for spec in specs if spec.field == field_name]
        assert len(field_specs) == 2
        assert all(spec.transform_kind == "identity" for spec in field_specs)
    for field_name in (
        "independent_free_log_sigma_delta_unit",
        "independent_free_log_mass_delta_unit",
    ):
        field_specs = [spec for spec in specs if spec.field == field_name]
        assert {spec.sample_site_index for spec in field_specs} == {0, 1}
        assert all(spec.prior_kind == "normal" for spec in field_specs)
    shape_specs = [spec for spec in specs if spec.field in {"independent_free_e1", "independent_free_e2"}]
    assert shape_specs
    assert all(spec.prior_kind == "truncated_normal" for spec in shape_specs)
    assert all(spec.std == pytest.approx(0.25) for spec in shape_specs)
    assert all(spec.lower == pytest.approx(-0.35) for spec in shape_specs)
    assert all(spec.upper == pytest.approx(0.35) for spec in shape_specs)
    assert all(abs(float(spec.mean)) <= 0.30 for spec in shape_specs)
    assert {spec.sample_site_name for spec in shape_specs} == {"members_independent_free_e_shape"}
    assert [spec.sample_site_index for spec in shape_specs] == [0, 1, 2, 3]
    sites = cluster_solver._parameter_sample_sites(specs)
    shape_sites = [
        site
        for site in sites
        if any(specs[index].field in {"independent_free_e1", "independent_free_e2"} for index in site.indices)
    ]
    assert len(shape_sites) == 1
    assert len(shape_sites[0].indices) == 4
    assert len(sites) == 3
    assert not any("tau" in field for field in fields)
    assert not any("offset" in field for field in fields)
    assert not any("scatter" in field for field in fields)
    assert not any("residual" in field for field in fields)
    assert indices[0][0]["independent_free_log_sigma_delta_unit"] >= 0
    assert indices[0][1]["independent_free_log_sigma_delta_unit"] >= 0
    assert indices[0][0]["independent_free_e1"] >= 0
    assert indices[0][0]["independent_free_e2"] >= 0


def _independent_scaling_plot_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    scaling_rank_df = pd.DataFrame(
        {
            "potfile_id": ["members", "members", "members"],
            "catalog_id": ["galaxy-a", "galaxy-b", "galaxy-c"],
            "rank": [1, 2, 3],
            "component_index": [1, 2, 3],
            "free_component_index": [4, -1, -1],
            "catalog_mag": [19.0, 20.0, 21.0],
            "independent_magnitude_feature": [-1.0, 0.0, 1.0],
            "x_centre": [0.0, 1.0, 2.0],
            "y_centre": [0.5, -0.5, 1.5],
            "selected_active": [True, True, False],
            "selected_independent": [True, False, False],
            "requested_active_count": [2, 2, 2],
            "importance": [0.7, 0.2, 0.1],
            "min_distance_arcsec": [0.3, 0.8, 1.5],
        }
    )
    independent_df = pd.DataFrame(
        {
            "potfile_id": ["members"],
            "component_index": [1],
            "scaling_v_disp_median": [100.0],
            "scaling_core_radius_kpc_median": [0.15],
            "scaling_cut_radius_kpc_median": [300.0],
            "free_v_disp_median": [120.0],
            "free_core_radius_kpc_median": [0.1425],
            "free_cut_radius_kpc_median": [255.0],
            "sigma_ratio_median": [1.20],
            "sigma_ratio_p16": [1.10],
            "sigma_ratio_p84": [1.30],
            "mass_ratio_median": [1.10],
            "mass_ratio_p16": [1.00],
            "mass_ratio_p84": [1.20],
            "radius_ratio_median": [0.85],
            "radius_ratio_p16": [0.80],
            "radius_ratio_p84": [0.90],
            "core_ratio_median": [0.95],
            "core_ratio_p16": [0.90],
            "core_ratio_p84": [1.00],
            "cut_ratio_median": [0.85],
            "cut_ratio_p16": [0.80],
            "cut_ratio_p84": [0.90],
        }
    )
    return scaling_rank_df, independent_df


def _independent_scaling_gate_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="members.galaxy-a.independent_free_log_sigma_delta_unit",
            sample_name="members_galaxy_a_independent_free_log_sigma_delta_unit",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_log_sigma_delta_unit",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="members.galaxy-a.independent_free_log_mass_delta_unit",
            sample_name="members_galaxy_a_independent_free_log_mass_delta_unit",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_log_mass_delta_unit",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            component_family="independent_scaling",
        ),
    ]


def test_independent_scaling_plot_table_preserves_all_modeled_galaxies() -> None:
    scaling_rank_df, independent_df = _independent_scaling_plot_fixture()

    plot_df = _independent_scaling_plot_table(scaling_rank_df, independent_df)

    assert plot_df["catalog_id"].tolist() == ["galaxy-a", "galaxy-b", "galaxy-c"]
    independent_row = plot_df[plot_df["catalog_id"] == "galaxy-a"].iloc[0]
    assert independent_row["independent_plot_class"] == "independent_candidate"
    assert bool(independent_row["selected_active"])
    assert bool(independent_row["selected_independent"])
    assert int(independent_row["component_index"]) == 1
    assert int(independent_row["rank"]) == 1
    assert int(independent_row["free_component_index"]) == 4
    assert float(independent_row["sigma_ratio_median"]) == pytest.approx(1.20)
    assert float(independent_row["mass_ratio_median"]) == pytest.approx(1.10)
    assert float(independent_row["radius_ratio_median"]) == pytest.approx(0.85)
    active_row = plot_df[plot_df["catalog_id"] == "galaxy-b"].iloc[0]
    inactive_row = plot_df[plot_df["catalog_id"] == "galaxy-c"].iloc[0]
    assert active_row["independent_plot_class"] == "active_not_independent"
    assert inactive_row["independent_plot_class"] == "not_sampled"
    assert np.isnan(float(active_row["sigma_ratio_median"]))
    assert np.isnan(float(inactive_row["sigma_ratio_median"]))


def test_default_corner_excludes_independent_scaling_internals() -> None:
    specs = [
        ParameterSpec(
            name="members.sigma_ref",
            sample_name="members_sigma_ref",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_ref",
            prior_kind="uniform",
            lower=0.0,
            upper=1.0,
            step=0.1,
            component_family="scaling",
        ),
        ParameterSpec(
            name="members.galaxy-a.independent_free_log_sigma_delta_unit",
            sample_name="members_galaxy_a_independent_free_log_sigma_delta_unit",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_log_sigma_delta_unit",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            mean=0.0,
            std=1.0,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="source.1.beta_x",
            sample_name="source_1_beta_x",
            potential_id="1",
            profile_type=0,
            field="beta_x",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            mean=0.0,
            std=1.0,
            component_family="source_position",
        ),
    ]

    corner_samples, corner_specs = _corner_without_source_positions(np.ones((2, 3), dtype=float), specs)

    assert corner_samples.shape == (2, 1)
    assert [spec.component_family for spec in corner_specs] == ["scaling"]


def test_independent_branch_weights_follow_deterministic_branch_roles() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.packed_spec_jax = {
        "luminosity_ratio": jnp.ones(3, dtype=jnp.float64),
        "independent_branch_role": jnp.asarray(
            [
                cluster_solver.INDEPENDENT_BRANCH_NONE,
                cluster_solver.INDEPENDENT_BRANCH_SCALING,
                cluster_solver.INDEPENDENT_BRANCH_FREE,
            ],
            dtype=jnp.int32,
        ),
        "independent_magnitude_feature": jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
    }

    weights = evaluator._independent_branch_weight_from_physical(
        jnp.asarray([0.0, 2.0, 1.0e-8], dtype=jnp.float64)
    )

    np.testing.assert_allclose(
        np.asarray(weights),
        [1.0, 0.0, 1.0],
        rtol=1.0e-10,
    )


def _minimal_two_bin_local_jacobian_state() -> BuildState:
    state = _minimal_stage4_surrogate_state()
    family1 = state.family_data[0]
    bin1 = state.bin_data[0]
    family2 = FamilyData(
        family_id="2",
        z_source=3.0,
        effective_z_source=3.0,
        sigma_arcsec=0.1,
        image_labels=["2.1", "2.2"],
        x_obs=np.asarray([0.2, 0.9], dtype=float),
        y_obs=np.asarray([0.1, -0.2], dtype=float),
        reliability=np.asarray([0.95, 0.9], dtype=float),
    )
    bin2 = cluster_solver.BinData(
        effective_z_source=3.0,
        family_ids=["2"],
        family_index_per_image=np.asarray([0, 0], dtype=np.int32),
        x_obs=family2.x_obs,
        y_obs=family2.y_obs,
        sigma_per_image=np.full(2, 0.12, dtype=float),
        reliability_per_image=family2.reliability,
    )
    state.family_data = [family1, family2]
    state.bin_data = [bin1, bin2]
    state.geometry_cache = GeometryCache(
        effective_z_source_values=[2.0, 3.0],
        exact_z_source_values=[2.0, 3.0],
        family_z_source_map={"1": 2.0, "2": 3.0},
        family_effective_z_source_map={"1": 2.0, "2": 3.0},
        dpie_sigma0_factor_by_effective_z={2.0: 1.0, 3.0: 1.35},
        dpie_sigma0_factor_by_exact_z={2.0: 1.0, 3.0: 1.35},
        lens_quadrature_z=[0.4],
        lens_quadrature_weights=[1.0],
        effective_z_quadrature_z=[[2.0], [3.0]],
        effective_z_quadrature_weights=[[1.0], [1.0]],
        exact_z_quadrature_z=[[2.0], [3.0]],
        exact_z_quadrature_weights=[[1.0], [1.0]],
    )
    return state


def _minimal_two_bin_local_jacobian_state_with_scaling_scatter() -> BuildState:
    state = _minimal_two_bin_local_jacobian_state()
    scatter_index = len(state.parameter_specs)
    state.parameter_specs = [
        *state.parameter_specs,
        ParameterSpec(
            name="potfile.sigma_log_scatter",
            sample_name="potfile_sigma_log_scatter",
            potential_id="potfile",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="sigma_log_scatter",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=float(np.log(cluster_solver.DEFAULT_SCALING_SCATTER_PRIOR_MEDIAN)),
            std=cluster_solver.DEFAULT_SCALING_SCATTER_PRIOR_LOG_SIGMA,
            component_family="scaling_scatter",
            transform_kind="log_positive",
            physical_lower=0.0,
            physical_upper=None,
            physical_mean=cluster_solver.DEFAULT_SCALING_SCATTER_PRIOR_MEDIAN,
        ),
    ]
    sigma_scatter_indices = state.packed_lens_spec.sigma_log_scatter_param_index.copy()
    sigma_scatter_indices[1] = scatter_index
    state.packed_lens_spec = replace(
        state.packed_lens_spec,
        sigma_log_scatter_param_index=sigma_scatter_indices,
    )
    return state


def _exact_per_bin_local_jacobian_loglike(
    evaluator: cluster_solver.ClusterJAXEvaluator,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    physical_params = evaluator._physical_parameter_vector(theta)
    source_sigma_int = evaluator._source_sigma_int_from_physical(physical_params)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    invalid_seen = jnp.asarray(False)
    for bin_data in evaluator.traced_bin_data:
        packed_state, validity = evaluator._build_packed_lens_state_with_validity_from_physical(
            physical_params,
            bin_data.effective_z_source,
            stop_gradient=True,
        )
        invalid = ~validity.is_valid
        beta_x, beta_y = jax.lax.cond(
            invalid,
            lambda _: (bin_data.x_obs, bin_data.y_obs),
            lambda current_state: evaluator._ray_shooting_for_components(
                bin_data.effective_z_source,
                bin_data.x_obs,
                bin_data.y_obs,
                current_state,
            ),
            packed_state,
        )
        jac_a00, jac_a01, jac_a10, jac_a11 = evaluator._lensing_jacobian_for_components(
            bin_data.effective_z_source,
            bin_data.x_obs,
            bin_data.y_obs,
            packed_state,
        )
        scatter_var_x, scatter_var_y = evaluator._scaling_scatter_extra_variance_from_physical(
            physical_params,
            bin_data,
            beta_x,
            beta_y,
        )
        bin_loglike = _local_jacobian_bin_loglike(
            beta_x=beta_x,
            beta_y=beta_y,
            family_idx=bin_data.family_index_per_image,
            n_families=bin_data.n_families,
            sigma_per_image=bin_data.sigma_per_image,
            reliability_per_image=jnp.clip(bin_data.reliability_per_image, 1.0e-6, 1.0 - 1.0e-6),
            image_has_constraint=bin_data.image_has_constraint,
            source_sigma_int=source_sigma_int,
            scatter_var_x=scatter_var_x,
            scatter_var_y=scatter_var_y,
            jac_a00=jac_a00,
            jac_a01=jac_a01,
            jac_a10=jac_a10,
            jac_a11=jac_a11,
            covariance_floor=evaluator.source_plane_covariance_floor,
            outlier_sigma_arcsec=evaluator.source_plane_outlier_sigma_arcsec,
            max_gain=evaluator.likelihood_stabilizer_max_gain,
            max_residual_arcsec=evaluator.likelihood_stabilizer_max_residual_arcsec,
            residual_loss=evaluator.likelihood_stabilizer_residual_loss,
            student_t_nu=evaluator.likelihood_stabilizer_student_t_nu,
        )
        total = jnp.where(invalid, total, total + bin_loglike)
        invalid_seen = jnp.logical_or(invalid_seen, invalid)
    return jnp.where(invalid_seen, jnp.asarray(cluster_solver.BAD_LOG_LIKE, dtype=jnp.float64), total)


def test_critical_arc_stage4_refreshing_surrogate_enables() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_plane_newton_steps=0,
    )

    assert evaluator.surrogate_enabled is True
    assert evaluator.image_presence_penalty_weight == pytest.approx(0.0)


def test_refreshing_surrogate_flat_builds_flat_cache() -> None:
    state = _minimal_stage4_surrogate_state()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    evaluator.refresh_surrogate(cluster_solver._default_theta(state.parameter_specs), reason="test")

    assert evaluator.surrogate_enabled is True
    assert evaluator.surrogate_cache_by_z
    assert evaluator.flat_surrogate_cache is not None
    np.testing.assert_array_equal(
        np.asarray(evaluator.flat_surrogate_cache.inactive_alpha_x).shape,
        np.asarray(evaluator.flat_critical_arc_data.x_obs).shape,
    )


def _stage4_state_with_independent_and_inactive_scaling() -> BuildState:
    state = _minimal_stage4_surrogate_state()
    packed = state.packed_lens_spec
    append_values = {
        "profile_type": [cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE],
        "component_family": [cluster_solver.COMPONENT_FAMILY_INDEPENDENT_FREE, cluster_solver.COMPONENT_FAMILY_SCALING],
        "x_center_base": [1.0, -1.5],
        "y_center_base": [1.0, 0.5],
        "core_radius_kpc_base": [2.0, 1.0],
        "cut_radius_kpc_base": [30.0, 15.0],
        "v_disp_base": [150.0, 80.0],
        "luminosity_ratio": [1.0, 0.75],
        "sigma_ref_base": [0.0, 80.0],
        "cut_ref_base": [0.0, 15.0],
        "core_ref_base": [0.0, 1.0],
        "vdslope_base": [0.0, 4.0],
        "slope_base": [0.0, 4.0],
        "sigma_ref_base": [120.0, 80.0],
        "cut_ref_base": [20.0, 15.0],
        "core_ref_base": [1.0, 1.0],
        "vdslope_base": [1.0, 4.0],
        "slope_base": [1.0, 4.0],
    }
    updates: dict[str, np.ndarray] = {}
    for field_name in PackedLensSpec.__dataclass_fields__:
        values = np.asarray(getattr(packed, field_name))
        if values.size == 0:
            continue
        fill_value = append_values.get(field_name)
        if fill_value is None:
            fill_value = [-1, -1] if field_name.endswith("_param_index") else [0.0, 0.0]
        updates[field_name] = np.concatenate([values, np.asarray(fill_value, dtype=values.dtype)])
    updates.update(
        {
            "independent_branch_role": np.asarray(
                [
                    cluster_solver.INDEPENDENT_BRANCH_NONE,
                    cluster_solver.INDEPENDENT_BRANCH_SCALING,
                    cluster_solver.INDEPENDENT_BRANCH_FREE,
                    cluster_solver.INDEPENDENT_BRANCH_NONE,
                ],
                dtype=np.int32,
            ),
            "independent_free_log_sigma_delta_unit_param_index": np.asarray([-1, -1, 1, -1], dtype=np.int32),
            "independent_free_log_mass_delta_unit_param_index": np.asarray([-1, -1, 2, -1], dtype=np.int32),
            "independent_magnitude_feature": np.asarray([0.0, 0.5, 0.5, 0.0], dtype=float),
        }
    )
    independent_specs = [
        ParameterSpec(
            name="members.galaxy-a.independent_free_log_sigma_delta_unit",
            sample_name="members_galaxy_a_independent_free_log_sigma_delta_unit",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_log_sigma_delta_unit",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            mean=0.0,
            std=1.0,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="members.galaxy-a.independent_free_log_mass_delta_unit",
            sample_name="members_galaxy_a_independent_free_log_mass_delta_unit",
            potential_id="members",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field="independent_free_log_mass_delta_unit",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            mean=0.0,
            std=1.0,
            component_family="independent_scaling",
        ),
    ]
    return replace(
        state,
        parameter_specs=[*state.parameter_specs, *independent_specs],
        packed_lens_spec=replace(packed, **updates),
        lens_model_list=[*state.lens_model_list, "DPIE_NIE", "DPIE_NIE"],
        potfiles=[{"id": "members", "mag0": 20.0}],
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 1,
                "catalog_id": "galaxy-a",
                "catalog_mag": 19.0,
                "catalog_color": 1.1,
                "independent_magnitude_feature": 0.5,
                "x_centre": 1.0,
                "y_centre": 1.0,
                "selected_independent": True,
                "free_component_index": 2,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 3,
                "catalog_id": "galaxy-b",
                "catalog_mag": 20.0,
                "catalog_color": 0.9,
                "independent_magnitude_feature": 0.0,
                "x_centre": -1.5,
                "y_centre": 0.5,
                "selected_independent": False,
                "free_component_index": -1,
            },
        ],
    )


def _stage4_state_with_all_scaling_selected_independent() -> BuildState:
    state = _stage4_state_with_independent_and_inactive_scaling()
    records = [dict(record) for record in state.scaling_component_records]
    for record in records:
        record["selected_independent"] = True
    return replace(state, scaling_component_records=records)


def test_active_blocked_nuts_block_library_is_seeded_and_reusable() -> None:
    blocks = tuple(
        cluster_solver.BlockedNUTSParameterBlock(f"galaxy_{idx}", (idx,))
        for idx in range(6)
    )

    first = cluster_solver._active_blocked_nuts_block_library(
        blocks,
        block_size=2,
        library_size=4,
        seed=17,
    )
    second = cluster_solver._active_blocked_nuts_block_library(
        blocks,
        block_size=2,
        library_size=4,
        seed=17,
    )
    third = cluster_solver._active_blocked_nuts_block_library(
        blocks,
        block_size=2,
        library_size=4,
        seed=18,
    )

    assert [block.indices for block in first] == [block.indices for block in second]
    assert [block.indices for block in first] != [block.indices for block in third]
    assert all(len(block.indices) == 2 for block in first)


def test_scalar_index_block_nuts_once_updates_only_selected_indices() -> None:
    specs = [
        ParameterSpec(
            name=f"p{idx}",
            sample_name=f"p{idx}",
            potential_id="toy",
            profile_type=cluster_solver.DP_IE_PROFILE,
            field=f"p{idx}",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            step=0.1,
            mean=0.0,
            std=1.0,
        )
        for idx in range(3)
    ]

    class ToyEvaluator:
        def _source_loglike_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(jnp.square(theta))

    args = SimpleNamespace(
        target_accept=0.8,
        max_tree_depth=2,
        initial_step_size=0.05,
        dense_mass=cluster_solver.NUTS_DENSE_MASS_DIAGONAL,
        debug_timings=False,
    )
    theta = np.asarray([0.1, 0.2, 0.3], dtype=float)
    block = cluster_solver.BlockedNUTSParameterBlock("active", (1,))
    block_model = cluster_solver._scalar_index_block_posterior_model(
        specs,
        ToyEvaluator(),
        len(block.indices),
    )

    updated, metrics, adapted = cluster_solver._run_scalar_index_block_nuts_once(
        args,
        specs,
        block_model,
        theta,
        block,
        jax.random.PRNGKey(123),
        num_warmup=0,
        adapt=False,
    )

    assert np.isfinite(updated).all()
    assert np.isfinite(metrics["accept_prob"])
    assert np.isfinite(adapted["step_size"])
    np.testing.assert_allclose(updated[[0, 2]], theta[[0, 2]])


def test_refreshing_surrogate_flat_keeps_independent_scaling_exact() -> None:
    state = _stage4_state_with_independent_and_inactive_scaling()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        active_scaling_galaxies=[0],
        active_scaling_selection="fixed",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    evaluator.refresh_surrogate(reference, reason="test")

    assert evaluator.independent_scaling_component_indices.tolist() == [1]
    assert evaluator.independent_free_component_indices.tolist() == [2]
    assert evaluator.active_scaling_component_indices.tolist() == [1]
    assert evaluator.inactive_scaling_component_indices.tolist() == [3]
    assert evaluator.exact_scaling_component_indices.tolist() == [1]
    assert evaluator.cached_scaling_component_indices.tolist() == [3]
    assert evaluator.free_correction_scaling_component_indices.tolist() == [1]
    assert evaluator.free_correction_free_component_indices.tolist() == [2]
    assert evaluator.excluded_scaling_component_indices.tolist() == []
    assert evaluator.active_component_indices.tolist() == [0, 1, 2]
    assert evaluator.surrogate_enabled is True
    assert evaluator.flat_surrogate_cache is not None

    flat_data = evaluator.flat_critical_arc_data
    physical = evaluator._physical_parameter_vector(jnp.asarray(reference, dtype=jnp.float64))
    packed_state, validity = evaluator._build_flat_packed_lens_state_with_validity_from_physical(
        physical,
        flat_data,
        stop_gradient=False,
    )
    assert bool(np.asarray(validity.is_valid, dtype=bool))
    inactive_beta_x, inactive_beta_y = evaluator._flat_ray_shooting_for_components(
        flat_data.x_obs,
        flat_data.y_obs,
        packed_state,
        np.asarray([3], dtype=np.int32),
    )
    both_beta_x, both_beta_y = evaluator._flat_ray_shooting_for_components(
        flat_data.x_obs,
        flat_data.y_obs,
        packed_state,
        np.asarray([1, 2, 3], dtype=np.int32),
    )
    expected_inactive_alpha_x = np.asarray(flat_data.x_obs - inactive_beta_x, dtype=float)
    expected_inactive_alpha_y = np.asarray(flat_data.y_obs - inactive_beta_y, dtype=float)
    both_alpha_x = np.asarray(flat_data.x_obs - both_beta_x, dtype=float)
    both_alpha_y = np.asarray(flat_data.y_obs - both_beta_y, dtype=float)

    np.testing.assert_allclose(
        np.asarray(evaluator.flat_surrogate_cache.inactive_alpha_x),
        expected_inactive_alpha_x,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.flat_surrogate_cache.inactive_alpha_y),
        expected_inactive_alpha_y,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert np.max(np.abs(both_alpha_x - expected_inactive_alpha_x) + np.abs(both_alpha_y - expected_inactive_alpha_y)) > 1.0e-8


def test_truth_grid_jax_bulk_uses_trace_safe_real_evaluator_packed_state(tmp_path: Path) -> None:
    from astropy.wcs import WCS

    state = _stage4_state_with_independent_and_inactive_scaling()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    theta = cluster_solver._default_theta(state.parameter_specs)
    posterior = PosteriorResults(
        samples=np.vstack([theta, theta]),
        log_prob=np.zeros(2, dtype=float),
        accept_prob=np.zeros(2, dtype=float),
        diverging=np.zeros(2, dtype=bool),
        num_steps=np.zeros(2, dtype=int),
        warmup_steps=0,
        sample_steps=2,
        num_chains=1,
    )
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    quantiles, _x_arcsec, _y_arcsec = plotting._posterior_truth_grid_quantiles(
        tmp_path,
        evaluator,
        posterior,
        wcs,
        (2, 2),
        float(evaluator.traced_bin_data[0].effective_z_source),
        source_truth_fits={"kappa": "kappa.fits"},
        quantities=("kappa", "detA", "mu"),
        truth_grid_mode=plotting.TRUTH_GRID_MODE_POSTERIOR,
    )

    assert np.isfinite(quantiles["kappa"]["median"]).all()
    assert np.isfinite(quantiles["detA"]["median"]).all()
    assert np.isfinite(quantiles["mu"]["median"]).all()
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_summary.csv")
    assert set(summary["truth_grid_backend"]) == {"jax_bulk_hessian"}


def test_stage0_full_flat_initial_state_has_no_free_candidates() -> None:
    state = replace(_minimal_stage4_surrogate_state(), perturbation_discovery_stage0=True)
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )

    assert evaluator.active_scaling_component_indices.tolist() == evaluator.scaling_component_indices.tolist()
    assert evaluator.exact_scaling_component_indices.tolist() == evaluator.scaling_component_indices.tolist()
    assert evaluator.cached_scaling_component_indices.tolist() == []
    assert evaluator.inactive_scaling_component_indices.tolist() == []
    assert evaluator.free_correction_scaling_component_indices.tolist() == []
    assert evaluator.free_correction_free_component_indices.tolist() == []
    assert evaluator.excluded_scaling_component_indices.tolist() == []
    assert evaluator.surrogate_enabled is False
    selected_active = evaluator.scaling_rank_df.loc[
        evaluator.scaling_rank_df["selected_active"].astype(bool),
        "component_index",
    ].astype(int).tolist()
    assert selected_active == []
    assert evaluator.scaling_rank_df["selected_independent"].astype(bool).sum() == 0
    assert cluster_solver._active_scaling_rank_summary(evaluator) == {}


def test_transfer_theta_and_posterior_by_sample_name_initializes_new_parameters() -> None:
    old_specs = [
        ParameterSpec("a", "a", "p", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec("b", "b", "p", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]
    new_specs = [
        ParameterSpec("b", "b", "p", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec("c", "c", "p", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=2.0, std=1.0),
        ParameterSpec("a", "a", "p", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]
    old_theta = np.asarray([1.5, -0.5], dtype=float)
    posterior = PosteriorResults(
        samples=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        log_prob=np.asarray([0.0, 1.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=2,
        num_chains=0,
    )

    transferred_theta, diagnostics = cluster_solver._transfer_theta_by_sample_name(old_specs, old_theta, new_specs)
    transferred_posterior = cluster_solver._transfer_posterior_samples_by_sample_name(posterior, old_specs, new_specs)

    np.testing.assert_allclose(transferred_theta, [-0.5, 2.0, 1.5])
    assert diagnostics["copied"] == 2
    assert diagnostics["initialized"] == 1
    np.testing.assert_allclose(transferred_posterior.samples, [[2.0, 2.0, 1.0], [4.0, 2.0, 3.0]])


@pytest.mark.parametrize("sample_likelihood_mode", [SAMPLE_LIKELIHOOD_SOURCE, SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN])
def test_stage0_full_flat_matches_full_flat_value_and_gradient_for_all_independent(sample_likelihood_mode: str) -> None:
    state = _stage4_state_with_all_scaling_selected_independent()
    full = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=sample_likelihood_mode,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    discovery = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=sample_likelihood_mode,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    theta = np.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=float)
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(discovery._source_loglike_fn(theta_jax)),
        np.asarray(full._source_loglike_fn(theta_jax)),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(discovery._source_loglike_fn)(theta_jax)),
        np.asarray(jax.grad(full._source_loglike_fn)(theta_jax)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_perturbation_discovery_source_matches_full_flat_value_and_gradient() -> None:
    state = _minimal_stage4_surrogate_state()
    full = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    discovery = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    theta = np.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=float)
    theta[0] = 1.15
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(discovery._source_loglike_fn(theta_jax)),
        np.asarray(full._source_loglike_fn(theta_jax)),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(discovery._source_loglike_fn)(theta_jax)),
        np.asarray(jax.grad(full._source_loglike_fn)(theta_jax)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_stage0_full_flat_source_magnification_covariance_does_not_require_metric_cache() -> None:
    state = _minimal_stage4_surrogate_state()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
    )
    evaluator.source_metric_cache_by_z = {}
    evaluator.source_metric_reference_params = None
    evaluator.flat_source_metric_cache = None
    theta = np.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=float)
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)

    value = evaluator._source_loglike_fn(theta_jax)
    gradient = jax.grad(evaluator._source_loglike_fn)(theta_jax)

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()


def test_perturbation_discovery_local_jacobian_matches_full_flat_value_and_gradient() -> None:
    state = _minimal_stage4_surrogate_state()
    full = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    discovery = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    theta = np.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=float)
    theta[0] = 1.15
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(discovery._source_loglike_fn(theta_jax)),
        np.asarray(full._source_loglike_fn(theta_jax)),
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(discovery._source_loglike_fn)(theta_jax)),
        np.asarray(jax.grad(full._source_loglike_fn)(theta_jax)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_refreshing_surrogate_flat_backfills_per_bin_surrogate_cache_from_flat_refresh() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    legacy = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    legacy.refresh_surrogate(reference, reason="test")
    flat.refresh_surrogate(reference, reason="test")

    assert flat.flat_surrogate_cache is not None
    assert set(flat.surrogate_cache_by_z) == set(legacy.surrogate_cache_by_z)
    for z_value, legacy_cache in legacy.surrogate_cache_by_z.items():
        flat_cache = flat.surrogate_cache_by_z[z_value]
        for key in (
            "inactive_alpha_x",
            "inactive_alpha_y",
            "inactive_alpha_dx_dparams",
            "inactive_alpha_dy_dparams",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(flat_cache, key), dtype=float),
                np.asarray(getattr(legacy_cache, key), dtype=float),
                rtol=1.0e-8,
                atol=1.0e-8,
            )


def _assert_cache_dicts_allclose(
    left: dict[float, dict[str, Any]],
    right: dict[float, dict[str, Any]],
    keys: tuple[str, ...],
) -> None:
    assert set(left) == set(right)
    for z_source in left:
        for key in keys:
            np.testing.assert_allclose(
                np.asarray(left[z_source][key], dtype=float),
                np.asarray(right[z_source][key], dtype=float),
                rtol=1.0e-8,
                atol=1.0e-8,
            )


def test_refreshing_surrogate_flat_source_metric_cache_matches_per_bin_refresh() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    legacy = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    legacy.refresh_source_metric_cache(reference, reason="test")
    flat.refresh_source_metric_cache(reference, reason="test")

    _assert_cache_dicts_allclose(
        flat.source_metric_cache_by_z,
        legacy.source_metric_cache_by_z,
        ("inv_abs_mu", "jac_a00", "jac_a01", "jac_a10", "jac_a11"),
    )
    np.testing.assert_allclose(
        np.asarray(flat.flat_source_metric_cache.jac_a00),
        np.concatenate([legacy.source_metric_cache_by_z[2.0]["jac_a00"], legacy.source_metric_cache_by_z[3.0]["jac_a00"]]),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_refreshing_surrogate_flat_scaling_scatter_cache_matches_per_bin_refresh() -> None:
    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    legacy = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    legacy.refresh_scaling_scatter_cache(reference, reason="test")
    flat.refresh_scaling_scatter_cache(reference, reason="test")

    _assert_cache_dicts_allclose(
        flat.scaling_scatter_cache_by_z,
        legacy.scaling_scatter_cache_by_z,
        cluster_solver.SCALING_SCATTER_CACHE_KEYS,
    )
    np.testing.assert_allclose(
        np.asarray(flat.flat_critical_arc_data.sigma_scatter_cov00),
        np.concatenate(
            [legacy.scaling_scatter_cache_by_z[2.0]["sigma_cov00"], legacy.scaling_scatter_cache_by_z[3.0]["sigma_cov00"]]
        ),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_scatter_covariance_terms_are_sum_of_outer_products_not_coherent() -> None:
    # Two anti-aligned members: independent scatter still induces variance, while the
    # coherent construction (sum first, then square) is exactly zero.
    v = np.asarray([[0.3, -0.1, 0.2]], dtype=float)
    deriv_x = np.concatenate([v, -v], axis=0)
    deriv_y = np.concatenate([2.0 * v, -2.0 * v], axis=0)
    cov00, cov01, cov11 = cluster_solver._scatter_covariance_terms_from_member_derivatives(deriv_x, deriv_y)
    np.testing.assert_allclose(cov00, 2.0 * np.square(v[0]))
    np.testing.assert_allclose(cov01, 2.0 * v[0] * (2.0 * v[0]))
    np.testing.assert_allclose(cov11, 2.0 * np.square(2.0 * v[0]))
    coherent00 = np.square(deriv_x.sum(axis=0))
    np.testing.assert_allclose(coherent00, 0.0)


def test_scatter_covariance_coherent_exceeds_per_member_for_aligned_members() -> None:
    # G identical (aligned) members: the coherent construction over-counts by exactly G.
    n_members = 5
    v_x = np.tile(np.asarray([[0.4, 0.1]]), (n_members, 1))
    v_y = np.tile(np.asarray([[-0.2, 0.3]]), (n_members, 1))
    cov00, cov01, cov11 = cluster_solver._scatter_covariance_terms_from_member_derivatives(v_x, v_y)
    coherent00 = np.square(v_x.sum(axis=0))
    coherent01 = v_x.sum(axis=0) * v_y.sum(axis=0)
    coherent11 = np.square(v_y.sum(axis=0))
    np.testing.assert_allclose(coherent00, n_members * cov00)
    np.testing.assert_allclose(coherent01, n_members * cov01)
    np.testing.assert_allclose(coherent11, n_members * cov11)
    assert np.all(coherent00 > cov00)
    with pytest.raises(ValueError):
        cluster_solver._scatter_covariance_terms_from_member_derivatives(v_x, v_y[:, :1])


def test_scatter_image_covariance_congruence_identity_inverse() -> None:
    # With an identity inverse operator, the image covariance equals the (scaled)
    # source covariance; the diagonal fallback returns its variance terms.
    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    evaluator.refresh_scaling_scatter_cache(reference, reason="test")
    assert evaluator.scaling_scatter_cache_by_z, "scatter cache must populate for this state"
    flat_data = evaluator.flat_critical_arc_data
    physical = evaluator._physical_parameter_vector(jnp.asarray(reference, dtype=jnp.float64))
    sigma_scale, mass_scale = evaluator._scaling_scatter_field_scales_from_physical(physical)
    ones = jnp.ones_like(flat_data.x_obs)
    zeros = jnp.zeros_like(flat_data.x_obs)
    cov00, cov01, cov11, finite = evaluator._flat_scaling_scatter_image_covariance_from_inverse(
        physical, flat_data, ones, zeros, zeros, ones, jnp.ones_like(flat_data.x_obs, dtype=bool)
    )
    expected00 = (
        jnp.square(sigma_scale) * flat_data.sigma_scatter_cov00
        + jnp.square(mass_scale) * flat_data.mass_scatter_cov00
    )
    expected01 = (
        jnp.square(sigma_scale) * flat_data.sigma_scatter_cov01
        + jnp.square(mass_scale) * flat_data.mass_scatter_cov01
    )
    expected11 = (
        jnp.square(sigma_scale) * flat_data.sigma_scatter_cov11
        + jnp.square(mass_scale) * flat_data.mass_scatter_cov11
    )
    np.testing.assert_allclose(np.asarray(cov00), np.asarray(expected00), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(cov01), np.asarray(expected01), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(cov11), np.asarray(expected11), rtol=1e-10)
    assert bool(np.asarray(finite).all())
    var_x, var_y = evaluator._flat_scaling_scatter_extra_variance_from_physical(physical, flat_data)
    np.testing.assert_allclose(np.asarray(var_x), np.asarray(expected00), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(var_y), np.asarray(expected11), rtol=1e-10)


def test_scaling_scatter_disabled_when_no_cached_scaling_components() -> None:
    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    evaluator.active_scaling_component_indices = evaluator.scaling_component_indices.copy()
    evaluator.exact_scaling_component_indices = evaluator.scaling_component_indices.copy()
    evaluator.cached_scaling_component_indices = np.asarray([], dtype=np.int32)
    evaluator.inactive_scaling_component_indices = np.asarray([], dtype=np.int32)
    reference = cluster_solver._default_theta(state.parameter_specs)

    evaluator.refresh_scaling_scatter_cache(reference, reason="test")

    assert evaluator._scaling_scatter_component_indices().size == 0
    assert evaluator.scaling_scatter_cache_by_z == {}
    assert evaluator.scaling_scatter_reference_params is None
    np.testing.assert_allclose(np.asarray(evaluator.flat_critical_arc_data.sigma_scatter_cov00), 0.0)
    physical = evaluator._physical_parameter_vector(jnp.asarray(reference, dtype=jnp.float64))
    var_x, var_y = evaluator._flat_scaling_scatter_extra_variance_from_physical(
        physical,
        evaluator.flat_critical_arc_data,
    )
    np.testing.assert_allclose(np.asarray(var_x), 0.0)
    np.testing.assert_allclose(np.asarray(var_y), 0.0)


def test_refreshing_surrogate_flat_accepts_critical_arc_and_builds_jacobian_cache() -> None:
    state = _minimal_stage4_surrogate_state(include_source_positions=True)
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
    )
    evaluator.refresh_surrogate(cluster_solver._default_theta(state.parameter_specs), reason="test")

    assert evaluator.flat_surrogate_cache is not None
    assert evaluator.flat_surrogate_cache.inactive_jacobian_delta_a00 is not None
    assert evaluator.flat_surrogate_cache.inactive_jacobian_delta_da00_dparams is not None


def test_refreshing_surrogate_flat_rejects_still_unsupported_likelihood_mode() -> None:
    with pytest.raises(ValueError, match="refreshing_surrogate_flat.*does not yet support"):
        cluster_solver.ClusterJAXEvaluator(
            state=_minimal_stage4_surrogate_state(include_source_positions=True),
            match_tolerance_arcsec=0.1,
            sampling_engine="refreshing_surrogate_flat",
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        )


def test_full_flat_critical_arc_flat_data_packs_images_and_source_indices() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(include_source_positions=True),
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
    )

    flat = evaluator.flat_critical_arc_data

    assert flat.n_families == 1
    assert flat.family_ids == ("1",)
    np.testing.assert_array_equal(np.asarray(flat.global_family_index_per_image), np.asarray([0, 0]))
    np.testing.assert_array_equal(np.asarray(flat.effective_z_index_per_image), np.asarray([0, 0]))
    np.testing.assert_array_equal(np.asarray(flat.bin_index_per_image), np.asarray([0, 0]))
    np.testing.assert_array_equal(np.asarray(flat.local_image_index_per_image), np.asarray([0, 1]))
    assert np.asarray(flat.global_family_source_x_param_index)[0] == 1
    assert np.asarray(flat.global_family_source_y_param_index)[0] == 2


def test_full_flat_rejects_unsupported_likelihood_mode() -> None:
    with pytest.raises(ValueError, match="full_flat"):
        cluster_solver.ClusterJAXEvaluator(
            state=_minimal_stage4_surrogate_state(include_source_positions=True),
            match_tolerance_arcsec=0.1,
            sampling_engine="full_flat",
            sample_likelihood_mode=SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE,
        )


def test_full_flat_source_matches_full_loglike_and_gradient() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    full = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)
    for evaluator in (full, flat):
        evaluator.refresh_source_metric_cache(reference, reason="test")
    theta = jnp.asarray([1.1], dtype=jnp.float64)

    full_value = full._source_loglike_fn(theta)
    flat_value = flat._source_loglike_fn(theta)
    full_grad = jax.grad(full._source_loglike_fn)(theta)
    flat_grad = jax.grad(flat._source_loglike_fn)(theta)

    np.testing.assert_allclose(np.asarray(flat_value), np.asarray(full_value), rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(flat_grad), np.asarray(full_grad), rtol=1.0e-6, atol=1.0e-6)


def test_full_flat_source_unit_covariance_uses_identity_metric_cache() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    unit = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)

    unit.refresh_source_metric_cache(reference, reason="test")

    assert unit.source_metric_cache_by_z == {}
    assert unit.source_metric_reference_params is not None
    np.testing.assert_allclose(np.asarray(unit.flat_source_metric_cache.inv_abs_mu), np.ones(4), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(unit.flat_source_metric_cache.jac_a00), np.ones(4), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(unit.flat_source_metric_cache.jac_a01), np.zeros(4), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(unit.flat_source_metric_cache.jac_a10), np.zeros(4), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(unit.flat_source_metric_cache.jac_a11), np.ones(4), rtol=0.0, atol=0.0)

    theta = jnp.asarray([1.1], dtype=jnp.float64)
    value = unit._source_loglike_fn(theta)
    grad = jax.grad(unit._source_loglike_fn)(theta)

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_full_flat_critical_arc_matches_full_direct_source_positions() -> None:
    state = _minimal_stage4_surrogate_state(include_source_positions=True)
    full = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_presence_penalty_weight=0.4,
        fixed_image_sigma_int_arcsec=0.02,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_presence_penalty_weight=0.4,
        fixed_image_sigma_int_arcsec=0.02,
    )
    theta = jnp.asarray([1.0, 0.05, -0.03], dtype=jnp.float64)

    full_value = full._source_loglike_fn(theta)
    flat_value = flat._source_loglike_fn(theta)
    full_grad = jax.grad(full._source_loglike_fn)(theta)
    flat_grad = jax.grad(flat._source_loglike_fn)(theta)

    np.testing.assert_allclose(np.asarray(flat_value), np.asarray(full_value), rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(flat_grad), np.asarray(full_grad), rtol=1.0e-6, atol=1.0e-6)


def test_full_flat_combined_exact_helper_matches_separate_bulk_calls() -> None:
    state = _minimal_stage4_surrogate_state()
    packed = state.packed_lens_spec
    state.packed_lens_spec = replace(
        packed,
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE], dtype=np.int32),
        component_family=np.asarray([0, 0], dtype=np.int32),
        gamma1_base=np.asarray([0.0, 0.08], dtype=float),
        gamma2_base=np.asarray([0.0, -0.03], dtype=float),
    )
    state.lens_model_list = ["DPIE_NIE", "SHEAR"]
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    theta = jnp.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=jnp.float64)
    physical_params = flat._physical_parameter_vector(theta)
    packed_state, validity = flat._build_flat_packed_lens_state_with_validity_from_physical(
        physical_params,
        flat.flat_critical_arc_data,
        stop_gradient=False,
    )

    assert bool(np.asarray(validity.is_valid, dtype=bool))
    beta_x, beta_y = flat._flat_ray_shooting_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
    )
    jacobian_entries = flat._flat_lensing_jacobian_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
    )
    combined = flat._flat_ray_shooting_and_lensing_jacobian_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
    )

    for actual, expected in zip(combined, (beta_x, beta_y, *jacobian_entries), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-8, atol=1.0e-8)


def test_full_flat_sparse_pair_alpha_jacobian_matches_rectangular_rows() -> None:
    state = _minimal_stage4_surrogate_state()
    packed = state.packed_lens_spec
    state.packed_lens_spec = replace(
        packed,
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE], dtype=np.int32),
        component_family=np.asarray([0, 0], dtype=np.int32),
        gamma1_base=np.asarray([0.0, 0.08], dtype=float),
        gamma2_base=np.asarray([0.0, -0.03], dtype=float),
    )
    state.lens_model_list = ["DPIE_NIE", "SHEAR"]
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    theta = jnp.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=jnp.float64)
    physical_params = flat._physical_parameter_vector(theta)
    packed_state, validity = flat._build_flat_packed_lens_state_with_validity_from_physical(
        physical_params,
        flat.flat_critical_arc_data,
        stop_gradient=False,
    )
    assert bool(np.asarray(validity.is_valid, dtype=bool))

    pair_image_indices = np.asarray([0, 1, 0, 1], dtype=np.int32)
    pair_component_indices = np.asarray([0, 0, 1, 1], dtype=np.int32)
    pair_x = jnp.take(flat.flat_critical_arc_data.x_obs, jnp.asarray(pair_image_indices, dtype=jnp.int32))
    pair_y = jnp.take(flat.flat_critical_arc_data.y_obs, jnp.asarray(pair_image_indices, dtype=jnp.int32))
    sparse = flat._flat_component_alpha_and_jacobian_delta_for_pairs(
        pair_x,
        pair_y,
        packed_state,
        pair_component_indices,
        pair_image_indices,
    )
    rectangular = flat._flat_component_alpha_and_jacobian_delta_rows_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
        np.asarray([0, 1], dtype=np.int32),
    )

    component_row = {0: 0, 1: 1}
    for actual, rows in zip(sparse, rectangular, strict=True):
        expected = np.asarray(
            [
                np.asarray(rows)[component_row[int(component)], int(image_index)]
                for image_index, component in zip(pair_image_indices, pair_component_indices, strict=True)
            ],
            dtype=float,
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-8, atol=1.0e-8)


def test_full_flat_alpha_only_component_rows_match_alpha_jacobian_rows() -> None:
    state = _minimal_stage4_surrogate_state()
    packed = state.packed_lens_spec
    state.packed_lens_spec = replace(
        packed,
        profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE], dtype=np.int32),
        component_family=np.asarray([0, 0], dtype=np.int32),
        gamma1_base=np.asarray([0.0, 0.08], dtype=float),
        gamma2_base=np.asarray([0.0, -0.03], dtype=float),
    )
    state.lens_model_list = ["DPIE_NIE", "SHEAR"]
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
    )
    theta = jnp.asarray(cluster_solver._default_theta(state.parameter_specs), dtype=jnp.float64)
    physical_params = flat._physical_parameter_vector(theta)
    packed_state, validity = flat._build_flat_packed_lens_state_with_validity_from_physical(
        physical_params,
        flat.flat_critical_arc_data,
        stop_gradient=False,
    )
    component_indices = np.asarray([0, 1], dtype=np.int32)

    assert bool(np.asarray(validity.is_valid, dtype=bool))
    alpha_x, alpha_y = flat._flat_component_alpha_rows_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
        component_indices,
    )
    alpha_jac_rows = flat._flat_component_alpha_and_jacobian_delta_rows_for_components(
        flat.flat_critical_arc_data.x_obs,
        flat.flat_critical_arc_data.y_obs,
        packed_state,
        component_indices,
    )

    np.testing.assert_allclose(np.asarray(alpha_x), np.asarray(alpha_jac_rows[0]), rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(alpha_y), np.asarray(alpha_jac_rows[1]), rtol=1.0e-8, atol=1.0e-8)


def test_full_flat_local_jacobian_uses_current_exact_flat_jacobian() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    theta = jnp.asarray([1.1], dtype=jnp.float64)

    flat_value = flat._source_loglike_fn(theta)
    reference_value = _exact_per_bin_local_jacobian_loglike(flat, theta)
    flat_grad = jax.grad(flat._source_loglike_fn)(theta)
    reference_grad = jax.grad(lambda value: _exact_per_bin_local_jacobian_loglike(flat, value))(theta)

    np.testing.assert_allclose(np.asarray(flat_value), np.asarray(reference_value), rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(flat_grad), np.asarray(reference_grad), rtol=1.0e-6, atol=1.0e-6)


def test_full_flat_local_jacobian_ignores_source_metric_cache() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    theta = jnp.asarray([1.1], dtype=jnp.float64)
    reference_value = flat._source_loglike_fn(theta)
    flat.source_metric_cache_by_z = {
        float(bin_data.effective_z_source): {
            "inv_abs_mu": np.full(np.asarray(bin_data.x_obs).shape, 1.0e6, dtype=float),
            "jac_a00": np.full(np.asarray(bin_data.x_obs).shape, 100.0, dtype=float),
            "jac_a01": np.full(np.asarray(bin_data.x_obs).shape, -50.0, dtype=float),
            "jac_a10": np.full(np.asarray(bin_data.x_obs).shape, 25.0, dtype=float),
            "jac_a11": np.full(np.asarray(bin_data.x_obs).shape, -75.0, dtype=float),
        }
        for bin_data in flat.traced_bin_data
    }
    flat._source_loglike_fn = jax.jit(flat._source_loglike_impl)

    poisoned_value = flat._source_loglike_fn(theta)

    np.testing.assert_allclose(np.asarray(poisoned_value), np.asarray(reference_value), rtol=1.0e-10, atol=1.0e-10)


def test_full_flat_source_metric_cache_matches_per_bin_refresh() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    legacy = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)

    legacy.refresh_source_metric_cache(reference, reason="test")
    flat.refresh_source_metric_cache(reference, reason="test")

    _assert_cache_dicts_allclose(
        flat.source_metric_cache_by_z,
        legacy.source_metric_cache_by_z,
        ("inv_abs_mu", "jac_a00", "jac_a01", "jac_a10", "jac_a11"),
    )
    np.testing.assert_allclose(
        np.asarray(flat.flat_source_metric_cache.jac_a00),
        np.concatenate([legacy.source_metric_cache_by_z[2.0]["jac_a00"], legacy.source_metric_cache_by_z[3.0]["jac_a00"]]),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_full_flat_local_jacobian_refreshes_metric_cache_with_unit_flag() -> None:
    state = _minimal_two_bin_local_jacobian_state()
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_UNIT,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)

    flat.refresh_source_metric_cache(reference, reason="test")

    assert sorted(flat.source_metric_cache_by_z) == [2.0, 3.0]
    assert np.asarray(flat.flat_source_metric_cache.inv_abs_mu).shape == (4,)
    assert np.asarray(flat.flat_source_metric_cache.jac_a00).shape == (4,)


def test_full_flat_scaling_scatter_cache_matches_per_bin_refresh() -> None:
    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    legacy = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    flat = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    )
    reference = cluster_solver._default_theta(state.parameter_specs)

    legacy.refresh_scaling_scatter_cache(reference, reason="test")
    flat.refresh_scaling_scatter_cache(reference, reason="test")

    _assert_cache_dicts_allclose(
        flat.scaling_scatter_cache_by_z,
        legacy.scaling_scatter_cache_by_z,
        cluster_solver.SCALING_SCATTER_CACHE_KEYS,
    )
    np.testing.assert_allclose(
        np.asarray(flat.flat_critical_arc_data.sigma_scatter_cov00),
        np.concatenate(
            [legacy.scaling_scatter_cache_by_z[2.0]["sigma_cov00"], legacy.scaling_scatter_cache_by_z[3.0]["sigma_cov00"]]
        ),
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def test_lens_model_bulk_accepts_per_image_sigma0_matrix() -> None:
    model = cluster_solver.LensModelBulk(unique_lens_model_list=["DPIE_NIE", "SHEAR"], multi_plane=False)
    x = jnp.asarray([0.2, 1.0, 3.0], dtype=jnp.float64)
    y = jnp.asarray([0.4, 2.0, -1.0], dtype=jnp.float64)
    index_list = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    base_kwargs = {
        "Ra": jnp.asarray([0.15, 0.0, 0.08], dtype=jnp.float64),
        "Rs": jnp.asarray([3.0, 0.0, 1.7], dtype=jnp.float64),
        "e1": jnp.asarray([0.05, 0.0, -0.03], dtype=jnp.float64),
        "e2": jnp.asarray([-0.02, 0.0, 0.04], dtype=jnp.float64),
        "center_x": jnp.asarray([0.1, 0.0, -0.2], dtype=jnp.float64),
        "center_y": jnp.asarray([-0.1, 0.0, 0.3], dtype=jnp.float64),
        "gamma1": jnp.asarray([0.0, 0.08, 0.0], dtype=jnp.float64),
        "gamma2": jnp.asarray([0.0, -0.03, 0.0], dtype=jnp.float64),
        "ra_0": jnp.zeros(3, dtype=jnp.float64),
        "dec_0": jnp.zeros(3, dtype=jnp.float64),
    }
    sigma0 = jnp.asarray(
        [
            [1.2, 1.3, 1.4],
            [0.0, 0.0, 0.0],
            [0.4, 0.5, 0.6],
        ],
        dtype=jnp.float64,
    )
    flat_kwargs = {"all_kwargs": {**base_kwargs, "sigma0": sigma0}, "index_list": index_list}

    flat_ray = model.ray_shooting(x, y, flat_kwargs)
    flat_hessian = model.hessian(x, y, flat_kwargs)

    separate_ray = []
    separate_hessian = []
    for image_index in range(3):
        point_kwargs = {"all_kwargs": {**base_kwargs, "sigma0": sigma0[:, image_index]}, "index_list": index_list}
        ray = model.ray_shooting(x[image_index : image_index + 1], y[image_index : image_index + 1], point_kwargs)
        hessian = model.hessian(x[image_index : image_index + 1], y[image_index : image_index + 1], point_kwargs)
        separate_ray.append([float(ray[0][0]), float(ray[1][0])])
        separate_hessian.append([float(value[0]) for value in hessian])

    np.testing.assert_allclose(np.asarray(flat_ray), np.asarray(separate_ray).T, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(flat_hessian), np.asarray(separate_hessian).T, rtol=1.0e-5, atol=1.0e-5)


def test_critical_arc_stage4_explicit_image_presence_penalty_is_honored() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=_minimal_stage4_surrogate_state(),
        match_tolerance_arcsec=0.1,
        sampling_engine="refreshing_surrogate_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        image_plane_newton_steps=0,
        image_presence_penalty_weight=2.0,
    )

    assert evaluator.image_presence_penalty_weight == pytest.approx(2.0)


def _fold_regularized_source_loglike_fake(
    surrogate_cache_by_z: dict[float, object],
    *,
    image_presence_penalty_weight: float,
    signed_curvature_calls: list[int],
) -> object:
    traced_bin = cluster_solver.TracedBinData(
        effective_z_source=2.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        x_obs=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
        y_obs=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True], dtype=bool),
        constrained_image_indices=jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )

    class FakeEvaluator:
        source_position_conditional = False
        source_position_param_indices_by_family = {"1": (0, 1)}
        surrogate_enabled = True
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_FOLD_REGULARIZED_FORWARD_BETA_IMAGE_PLANE
        fold_curvature_arcsec_inv = 1.0
        image_plane_scatter_floor_arcsec = 1.0e-3
        source_plane_covariance_floor = 1.0e-6
        source_plane_outlier_sigma_arcsec = 10.0
        image_presence_match_radius_arcsec = 0.30
        image_presence_temperature_arcsec = 0.10
        image_presence_count_softness = 0.05
        image_presence_count_margin = 0.05
        likelihood_stabilizer_max_gain = 0.0
        likelihood_stabilizer_max_residual_arcsec = 0.0
        likelihood_stabilizer_residual_loss = "gaussian"
        likelihood_stabilizer_student_t_nu = 4.0
        fit_cosmology_flat_wcdm = False

        def __init__(self) -> None:
            self.surrogate_cache_by_z = surrogate_cache_by_z
            self.image_presence_penalty_weight = image_presence_penalty_weight
            self.traced_bin_data = [traced_bin]

        def _physical_parameter_vector(self, params):
            return params

        def _source_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.0, dtype=jnp.float64)

        def _image_sigma_int_from_physical(self, _physical_params):
            return jnp.asarray(0.05, dtype=jnp.float64)

        def _surrogate_beta(self, _params, _physical_params, bin_data):
            return bin_data.x_obs + 0.4, bin_data.y_obs - 0.1, jnp.asarray(False), {}

        def _surrogate_jacobian_entries(self, _params, bin_data, _packed_state, _invalid):
            return (
                jnp.asarray([0.05, 1.0, 0.05], dtype=jnp.float64),
                jnp.zeros_like(bin_data.x_obs),
                jnp.zeros_like(bin_data.y_obs),
                jnp.ones_like(bin_data.y_obs),
            )

        def _scaling_scatter_extra_variance_from_physical(self, _physical_params, bin_data, _beta_x, _beta_y):
            return jnp.zeros_like(bin_data.x_obs), jnp.zeros_like(bin_data.y_obs)

        def _fold_signed_curvature_from_observed_jacobian(
            self,
            _z_source,
            x_obs,
            _y_obs,
            _packed_state,
            _observed_jacobian_entries,
            _component_indices=None,
            *,
            fold_frame=None,
            image_indices=None,
            fill_value=1.0,
        ):
            del fold_frame, fill_value
            signed_curvature_calls.append(
                int(image_indices.shape[0]) if image_indices is not None else int(x_obs.shape[0])
            )
            return jnp.full_like(x_obs, 2.0)

        def _cab_morphology_loglike_for_arcs(self, *_args, **_kwargs):
            return jnp.asarray(0.0, dtype=jnp.float64)

    fake = FakeEvaluator()
    fake._source_position_vectors_for_bin = cluster_solver.ClusterJAXEvaluator._source_position_vectors_for_bin.__get__(
        fake,
        FakeEvaluator,
    )
    fake._explicit_source_position_vectors_for_bin = (
        cluster_solver.ClusterJAXEvaluator._explicit_source_position_vectors_for_bin.__get__(fake, FakeEvaluator)
    )
    return fake


def test_packed_lens_spec_converts_fixed_lenstool_shape_to_e1e2_base() -> None:
    packed = cluster_solver._build_packed_lens_spec(
        [
            {
                "profil": cluster_solver.DP_IE_PROFILE,
                "ellipticite": 0.28,
                "angle_pos": 12.0,
                "core_radius_kpc": 5.0,
                "cut_radius_kpc": 100.0,
                "v_disp": 900.0,
            },
        ],
        [[]],
    )

    expected_e1, expected_e2 = cluster_solver._lenstool_shape_to_e1e2(0.28, 12.0)

    np.testing.assert_allclose(packed.e1_base, [expected_e1])
    np.testing.assert_allclose(packed.e2_base, [expected_e2])
    np.testing.assert_array_equal(packed.e1_param_index, [-1])
    np.testing.assert_array_equal(packed.e2_param_index, [-1])


def test_packed_lens_spec_converts_fixed_shear_to_gamma1gamma2_base() -> None:
    packed = cluster_solver._build_packed_lens_spec(
        [
            {
                "profil": cluster_solver.SHEAR_PROFILE,
                "gamma": 0.04,
                "angle_pos": 40.0,
            },
        ],
        [[]],
    )

    expected_gamma1, expected_gamma2 = cluster_solver._shear_polar_to_gamma1gamma2(0.04, 40.0)

    np.testing.assert_allclose(packed.gamma1_base, [expected_gamma1])
    np.testing.assert_allclose(packed.gamma2_base, [expected_gamma2])
    np.testing.assert_array_equal(packed.gamma1_param_index, [-1])
    np.testing.assert_array_equal(packed.gamma2_param_index, [-1])


def test_packed_lens_state_uses_direct_gamma1gamma2_for_shear() -> None:
    state = _minimal_stage4_surrogate_state()
    state.packed_lens_spec.profile_type[:] = np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE])
    state.packed_lens_spec.component_family[:] = np.asarray([0, 0])
    state.packed_lens_spec.gamma1_base[:] = np.asarray([0.0, 0.01], dtype=float)
    state.packed_lens_spec.gamma2_base[:] = np.asarray([0.0, -0.02], dtype=float)
    state.packed_lens_spec.gamma1_param_index[1] = 0
    state.packed_lens_spec.gamma2_param_index[1] = 1
    evaluator = cluster_solver.ClusterJAXEvaluator(
        state=state,
        match_tolerance_arcsec=0.1,
        sampling_engine="full_flat",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_newton_steps=0,
    )

    packed_state, details = evaluator._build_packed_lens_state_details_from_physical(
        jnp.asarray([0.12, -0.08], dtype=jnp.float64),
        2.0,
        kpc_per_arcsec=jnp.asarray(10.0, dtype=jnp.float64),
        dpie_sigma0_factor=jnp.asarray(1.0, dtype=jnp.float64),
    )

    np.testing.assert_allclose(np.asarray(details.gamma1), [0.0, 0.12])
    np.testing.assert_allclose(np.asarray(details.gamma2), [0.0, -0.08])
    np.testing.assert_allclose(np.asarray(packed_state.gamma1), [0.0, 0.12])
    np.testing.assert_allclose(np.asarray(packed_state.gamma2), [0.0, -0.08])


def test_scaling_scatter_specs_use_plain_lognormal_prior() -> None:
    specs, indices = cluster_solver._build_scaling_scatter_parameter_specs(
        [{"id": "potfile", "type": 81}],
        start_index=7,
    )

    assert indices == [{"sigma": 7, "mass": 8}]
    assert [spec.field for spec in specs] == ["sigma_log_scatter", "mass_log_scatter"]
    for spec in specs:
        assert spec.prior_kind == "normal"
        assert spec.lower == float("-inf")
        assert spec.upper == float("inf")
        assert spec.mean == pytest.approx(np.log(0.02))
        assert spec.std == pytest.approx(1.0)
        assert spec.transform_kind == "log_positive"
        assert spec.physical_lower == 0.0
        assert spec.physical_upper is None
        assert spec.physical_mean == pytest.approx(0.02)


def test_solver_active_approximation_warning_reports_active_features(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_logs: list[str] = []
    state = SimpleNamespace(
        family_data=[object(), object()],
        bin_data=[SimpleNamespace(effective_z_source=1.0)],
        parameter_specs=[SimpleNamespace(component_family="source_position")],
        potfiles=[{"id": "members"}],
    )
    evaluator = SimpleNamespace(
        state=state,
        surrogate_enabled=True,
        inactive_scaling_component_indices=np.asarray([2, 3, 4], dtype=int),
        active_scaling_component_indices=np.asarray([0, 1], dtype=int),
        scaling_component_indices=np.asarray([0, 1, 2, 3, 4], dtype=int),
        _scaling_scatter_component_indices=lambda: np.asarray([2, 3, 4], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        source_position_parameterization="prior-whitened",
        scaling_scatter_param_indices=np.asarray([5], dtype=int),
        source_metric_cache_by_z={2.0: {}},
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        active_scaling_selection="adaptive",
        active_scaling_cumulative_fraction=0.995,
        quick_diagnostics=False,
    )

    monkeypatch.setattr(cluster_solver, "_log", lambda _args, message, **_kwargs: captured_logs.append(str(message)))

    cluster_solver._log_solver_active_approximation_warning(argparse.Namespace(), evaluator)

    assert len(captured_logs) == 1
    warning = captured_logs[0]
    assert "refreshing_surrogate_flat=active" in warning
    assert "cached_scaling=3" in warning
    assert "free_correction=0" in warning
    assert "z_bins=active grouped_families=2 bins=1" in warning
    assert "sample_likelihood=linearized-forward-beta-image-plane" in warning
    assert "source_position_parameterization=prior-whitened" in warning
    assert "active_scaling_subset=active 2/5" in warning
    assert "scaling_scatter_cache=linearized" in warning
    assert "source_metric_cache=refreshed" in warning


def test_stage1_model_summary_rows_include_partition_and_cache_counts() -> None:
    evaluator = SimpleNamespace(
        state=SimpleNamespace(perturbation_discovery_stage0=False),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        sample_likelihood_mode=cluster_solver.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        large_component_indices=np.asarray([0, 1], dtype=int),
        scaling_component_indices=np.asarray([2, 3, 4, 5], dtype=int),
        exact_scaling_component_indices=np.asarray([2], dtype=int),
        active_scaling_component_indices=np.asarray([2], dtype=int),
        free_correction_scaling_component_indices=np.asarray([2], dtype=int),
        cached_scaling_component_indices=np.asarray([3, 4, 5], dtype=int),
        inactive_scaling_component_indices=np.asarray([3, 4, 5], dtype=int),
        _scaling_scatter_component_indices=lambda: np.asarray([3, 4, 5], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        scaling_scatter_param_indices=np.asarray([8], dtype=int),
        source_metric_cache_by_z={2.0: object()},
    )

    rows = cluster_solver._stage1_model_summary_rows(evaluator)
    row_values = {name: value for name, value, *_rest in rows}

    assert row_values["sampling_engine"] == "refreshing_surrogate_flat"
    assert row_values["sample_likelihood"] == "local-jacobian"
    assert row_values["large_exact"] == "2"
    assert row_values["total_scaling"] == "4"
    assert row_values["selected_exact_scaling"] == "1/4"
    assert row_values["free_scaling"] == "1"
    assert row_values["cached_inactive_scaling"] == "3"
    assert row_values["excluded_scaling"] == "0"
    assert row_values["local_jacobian_metric"] == "frozen_refreshed"
    assert row_values["scaling_scatter_cache"] == "linearized"
    assert row_values["source_metric_cache"] == "refreshed"


def test_stage1_model_summary_reports_scatter_disabled_when_no_cached_scaling() -> None:
    evaluator = SimpleNamespace(
        state=SimpleNamespace(perturbation_discovery_stage0=False),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        sample_likelihood_mode=cluster_solver.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        large_component_indices=np.asarray([0, 1], dtype=int),
        scaling_component_indices=np.asarray([2, 3], dtype=int),
        exact_scaling_component_indices=np.asarray([2, 3], dtype=int),
        active_scaling_component_indices=np.asarray([2, 3], dtype=int),
        free_correction_scaling_component_indices=np.asarray([2, 3], dtype=int),
        cached_scaling_component_indices=np.asarray([], dtype=int),
        inactive_scaling_component_indices=np.asarray([], dtype=int),
        _scaling_scatter_component_indices=lambda: np.asarray([], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        scaling_scatter_param_indices=np.asarray([8, 9], dtype=int),
        source_metric_cache_by_z={2.0: object()},
    )

    rows = cluster_solver._stage1_model_summary_rows(evaluator)
    row_values = {name: value for name, value, *_rest in rows}

    assert row_values["selected_exact_scaling"] == "2/2"
    assert row_values["free_scaling"] == "2"
    assert row_values["cached_inactive_scaling"] == "0"
    assert row_values["scaling_scatter_cache"] == "disabled_no_cached_scaling"


def test_stage1_model_summary_logs_and_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logs: list[str] = []
    renderables: list[Any] = []
    evaluator = SimpleNamespace(
        state=SimpleNamespace(perturbation_discovery_stage0=False),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        sample_likelihood_mode=cluster_solver.SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        large_component_indices=np.asarray([0], dtype=int),
        scaling_component_indices=np.asarray([1, 2], dtype=int),
        exact_scaling_component_indices=np.asarray([1], dtype=int),
        active_scaling_component_indices=np.asarray([1], dtype=int),
        free_correction_scaling_component_indices=np.asarray([1], dtype=int),
        cached_scaling_component_indices=np.asarray([2], dtype=int),
        inactive_scaling_component_indices=np.asarray([2], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    monkeypatch.setattr(
        cluster_solver,
        "_log",
        lambda _args, message, **kwargs: (logs.append(str(message)), renderables.append(kwargs.get("renderable"))),
    )
    run_dir = tmp_path / "fit" / cluster_solver.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR

    cluster_solver._log_stage1_model_summary_table(argparse.Namespace(), evaluator, run_dir)

    assert logs and logs[0].startswith("[stage1-model-summary]")
    assert "| selected_exact_scaling | 1/2 |" in logs[0]
    assert getattr(renderables[0], "title", None) == "Stage 1 model summary"
    csv_path = run_dir / "tables" / "stage_model_summary.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "cached_inactive_scaling,1,active" in csv_text


def test_stage_model_summary_logs_stage2_and_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logs: list[str] = []
    renderables: list[Any] = []
    evaluator = SimpleNamespace(
        state=SimpleNamespace(perturbation_discovery_stage0=False),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        sample_likelihood_mode=cluster_solver.SAMPLE_LIKELIHOOD_SOURCE,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        large_component_indices=np.asarray([0], dtype=int),
        scaling_component_indices=np.asarray([1, 2, 3], dtype=int),
        exact_scaling_component_indices=np.asarray([1, 2], dtype=int),
        active_scaling_component_indices=np.asarray([1, 2], dtype=int),
        free_correction_scaling_component_indices=np.asarray([1, 2], dtype=int),
        cached_scaling_component_indices=np.asarray([3], dtype=int),
        inactive_scaling_component_indices=np.asarray([3], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )
    monkeypatch.setattr(
        cluster_solver,
        "_log",
        lambda _args, message, **kwargs: (logs.append(str(message)), renderables.append(kwargs.get("renderable"))),
    )
    run_dir = tmp_path / "fit" / cluster_solver.STAGE2_FREE_SOURCE_FORWARD_FIT_DIR

    cluster_solver._log_stage_model_summary_table(argparse.Namespace(), evaluator, run_dir)

    assert logs and logs[0].startswith("[stage2-model-summary]")
    assert "| selected_exact_scaling | 2/3 |" in logs[0]
    assert getattr(renderables[0], "title", None) == "Stage 2 model summary"
    csv_path = run_dir / "tables" / "stage_model_summary.csv"
    assert csv_path.exists()
    assert "cached_inactive_scaling,1,active" in csv_path.read_text(encoding="utf-8")


def test_truth_recovery_stage_summary_logs_stage1_and_stage2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logs: list[str] = []
    renderables: list[Any] = []
    monkeypatch.setattr(
        cluster_solver,
        "_log",
        lambda _args, message, **kwargs: (logs.append(str(message)), renderables.append(kwargs.get("renderable"))),
    )

    for stage_name, expected_prefix, expected_title in [
        (cluster_solver.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR, "[stage1-truth-recovery-summary]", "Stage 1 truth recovery summary"),
        (cluster_solver.STAGE2_FREE_SOURCE_FORWARD_FIT_DIR, "[stage2-truth-recovery-summary]", "Stage 2 truth recovery summary"),
    ]:
        run_dir = tmp_path / "fit" / stage_name
        tables_dir = run_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "quantity": "kappa",
                    "finite_pixel_count": 10,
                    "kappa_bias_median": 0.1,
                    "kappa_spread_nmad": 0.2,
                    "kappa_rmse": 0.3,
                }
            ]
        ).to_csv(tables_dir / "truth_recovery_kappa_recovery_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "quantity": "abs_mu",
                    "finite_pixel_count": 8,
                    "abs_mu_bias_median": -0.4,
                    "abs_mu_spread_nmad": 0.5,
                    "abs_mu_rmse": 0.6,
                }
            ]
        ).to_csv(tables_dir / "truth_recovery_mu_recovery_summary.csv", index=False)

        before = len(logs)
        cluster_solver._log_truth_recovery_stage_summary_table(argparse.Namespace(), run_dir)

        assert logs[before].startswith(expected_prefix)
        assert "| kappa | 10 | 0.1 | 0.2 | 0.3 |" in logs[before]
        assert "| abs_mu | 8 | -0.4 | 0.5 | 0.6 |" in logs[before]
        assert getattr(renderables[before], "title", None) == expected_title
        csv_path = tables_dir / "truth_recovery_stage_summary.csv"
        assert csv_path.exists()
        summary = pd.read_csv(csv_path)
        assert summary["quantity"].tolist() == ["kappa", "abs_mu"]


def test_truth_recovery_stage_summary_skips_when_no_recovery_csvs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logs: list[str] = []
    monkeypatch.setattr(cluster_solver, "_log", lambda _args, message, **_kwargs: logs.append(str(message)))

    cluster_solver._log_truth_recovery_stage_summary_table(
        argparse.Namespace(),
        tmp_path / "fit" / cluster_solver.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR,
    )

    assert logs == []
    assert not (tmp_path / "fit" / cluster_solver.STAGE1_BACKPROJECTED_CENTROID_FIT_DIR / "tables" / "truth_recovery_stage_summary.csv").exists()


def test_active_approximation_table_reports_frozen_refreshed_local_jacobian_metric() -> None:
    evaluator = SimpleNamespace(
        state=SimpleNamespace(family_data=[], bin_data=[], parameter_specs=[]),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        surrogate_enabled=True,
        active_scaling_component_indices=np.asarray([0, 1], dtype=int),
        inactive_scaling_component_indices=np.asarray([2, 3], dtype=int),
        cached_scaling_component_indices=np.asarray([2, 3], dtype=int),
        exact_scaling_component_indices=np.asarray([0, 1], dtype=int),
        scaling_component_indices=np.asarray([0, 1, 2, 3], dtype=int),
        free_correction_scaling_component_indices=np.asarray([], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        active_scaling_selection="fixed",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        quick_diagnostics=False,
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={0: object()},
    )

    rows = {row[0]: row[1] for row in cluster_solver._active_approximation_rows(evaluator)}

    assert rows["local_jacobian_metric"] == cluster_solver.LOCAL_JACOBIAN_METRIC_FROZEN_REFRESHED
    assert rows["source_metric_cache"] == "refreshed"
    assert cluster_solver._local_jacobian_metric_mode(evaluator) == cluster_solver.LOCAL_JACOBIAN_METRIC_FROZEN_REFRESHED


def test_active_approximation_table_reports_current_exact_local_jacobian_metric() -> None:
    evaluator = SimpleNamespace(
        state=SimpleNamespace(family_data=[], bin_data=[], parameter_specs=[]),
        sampling_engine=cluster_solver.SAMPLING_ENGINE_FULL_FLAT,
        surrogate_enabled=False,
        active_scaling_component_indices=np.asarray([0, 1, 2], dtype=int),
        inactive_scaling_component_indices=np.asarray([], dtype=int),
        cached_scaling_component_indices=np.asarray([], dtype=int),
        exact_scaling_component_indices=np.asarray([0, 1, 2], dtype=int),
        scaling_component_indices=np.asarray([0, 1, 2], dtype=int),
        free_correction_scaling_component_indices=np.asarray([], dtype=int),
        excluded_scaling_component_indices=np.asarray([], dtype=int),
        active_scaling_selection="fixed",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        source_plane_covariance_mode=cluster_solver.SOURCE_PLANE_COVARIANCE_MODE_MAGNIFICATION,
        quick_diagnostics=False,
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    rows = {row[0]: row[1] for row in cluster_solver._active_approximation_rows(evaluator)}

    assert rows["local_jacobian_metric"] == cluster_solver.LOCAL_JACOBIAN_METRIC_CURRENT_EXACT
    assert cluster_solver._local_jacobian_metric_mode(evaluator) == cluster_solver.LOCAL_JACOBIAN_METRIC_CURRENT_EXACT


def test_refreshing_surrogate_flat_local_jacobian_does_not_request_inactive_jacobian_cache() -> None:
    assert not cluster_solver._flat_surrogate_refresh_needs_inactive_jacobian(
        SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
        sampling_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
    )


def test_run_summary_stage_fit_quality_table_formats_point_arc_and_missing_sampler_metrics() -> None:
    source_summary = {
        "run_name": "mock/stage3_image_plane",
        "sampler": "numpyro_nuts",
        "sample_likelihood_mode": SAMPLE_LIKELIHOOD_SOURCE,
        "observed_image_count": 4,
        "point_recovered_image_count": 3,
        "point_image_rms_arcsec": 0.42,
        "point_image_median_residual_arcsec": 0.31,
        "headline_chi_square": 5.0,
        "headline_dof": 2,
        "headline_reduced_chi_square": 2.5,
        "accept_prob_mean": None,
        "divergence_count": 0,
        "max_tree_depth_saturation_fraction": None,
        "ess_min": None,
        "rhat_max": None,
        "runtime_sec": 12.5,
    }
    source_text = cluster_solver._format_stage_fit_quality_table(source_summary)

    assert source_text.startswith("[stage-fit-quality] stage3_image_plane")
    assert "| point recovered images | 3/4 |" in source_text
    assert "| point RMS residual arcsec | 0.42 |" in source_text
    assert "| point median residual arcsec | 0.31 |" in source_text
    assert "| accept probability mean | na |" in source_text
    assert "arc-aware" not in source_text

    arc_summary = {
        **source_summary,
        "run_name": "mock/stage4_critical_arc_mixture_image_plane",
        "sample_likelihood_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        "arc_aware_recovered_image_count": 4,
        "arc_aware_arc_supported_image_count": 1,
        "arc_aware_missing_image_count": 0,
        "arc_aware_image_rms_arcsec": 0.33,
        "arc_aware_image_residual_median_arcsec": 0.22,
        "arc_aware_reduced_chi_square": 1.7,
    }
    arc_text = cluster_solver._format_stage_fit_quality_table(arc_summary)
    rich_table = cluster_solver._stage_fit_quality_table(arc_summary)

    assert "| arc-aware recovered images | 4/4 |" in arc_text
    assert "| arc-supported images | 1 |" in arc_text
    assert "| missed images | 0 |" in arc_text
    assert "| arc-aware RMS residual arcsec | 0.33 |" in arc_text
    assert "| arc-aware median residual arcsec | 0.22 |" in arc_text
    assert rich_table is not None


def test_active_scaling_fraction_status_thresholds() -> None:
    assert cluster_solver._active_scaling_fraction_status(0.995) == ("excellent", "bold green")
    assert cluster_solver._active_scaling_fraction_status(0.98) == ("good", "green")
    assert cluster_solver._active_scaling_fraction_status(0.95) == ("watch", "bold yellow")
    assert cluster_solver._active_scaling_fraction_status(0.9397) == ("risky", "bold red")
    assert cluster_solver._active_scaling_fraction_status(float("nan")) == ("unknown", "dim")
    assert cluster_solver._format_active_scaling_realized_fraction(0.9397) == "0.9397 (risky)"


def test_active_approximation_rich_table_has_two_columns() -> None:
    rows = [
        ("sampling_engine", "refreshing_surrogate_flat", "engine"),
        ("active_scaling", "50/330", "active"),
        ("active_scaling_realized_fraction", "0.9397 (risky)", "selection", "bold red"),
    ]

    table = cluster_solver._build_active_approximation_rich_table(rows)

    assert getattr(table, "title", None) == "Active approximations"
    assert [column.header for column in table.columns] == ["name", "value"]
    assert table.columns[0]._cells == [
        "[bold yellow]sampling_engine[/]",
        "[bold cyan]active_scaling[/]",
        "[cyan]active_scaling_realized_fraction[/]",
    ]
    assert table.columns[1]._cells == [
        "[bold yellow]refreshing_surrogate_flat[/]",
        "[bold cyan]50/330[/]",
        "[bold red]0.9397 (risky)[/]",
    ]


def test_prepare_direct_evaluator_logs_approximation_table_before_refresh_and_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_events: list[str] = []

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = SimpleNamespace(
                family_data=[object()],
                bin_data=[SimpleNamespace(effective_z_source=2.0)],
                parameter_specs=[],
            )
            self.sampling_engine = cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT
            self.surrogate_enabled = True
            self.active_scaling_component_indices = np.asarray([0], dtype=int)
            self.inactive_scaling_component_indices = np.asarray([1], dtype=int)
            self.scaling_component_indices = np.asarray([0, 1], dtype=int)
            self.active_scaling_selection = "adaptive"
            self.active_scaling_cumulative_fraction = 0.995
            self.sample_likelihood_mode = SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN
            self.quick_diagnostics = False
            self.scaling_scatter_param_indices = np.asarray([], dtype=int)
            self.source_metric_cache_by_z = {}
            self.timing_totals = {"initial_jit_compile": 0.0}

        def refresh_surrogate(self, _params: np.ndarray, reason: str) -> None:
            captured_events.append(f"refresh_surrogate:{reason}")

        def refresh_scaling_scatter_cache(self, _params: np.ndarray, reason: str) -> None:
            captured_events.append(f"refresh_scaling:{reason}")

        def refresh_source_metric_cache(self, _params: np.ndarray, reason: str) -> None:
            captured_events.append(f"refresh_source_metric:{reason}")

        def source_loglike(self, _params: np.ndarray) -> float:
            captured_events.append("source_loglike")
            return -1.0

    state = SimpleNamespace(parameter_specs=[])
    args = argparse.Namespace()

    monkeypatch.setattr(cluster_solver, "_build_cluster_evaluator_from_args", lambda _args, _state: FakeEvaluator())
    monkeypatch.setattr(
        cluster_solver,
        "_log",
        lambda _args, message, **_kwargs: captured_events.append(str(message).splitlines()[0]),
    )
    monkeypatch.setattr(
        cluster_solver,
        "_log_evaluator_summary",
        lambda _args, _evaluator: captured_events.append("evaluator_summary"),
    )
    monkeypatch.setattr(cluster_solver, "_log_evaluator_memory_shape_diagnostics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_clear_jax_caches", lambda *_args, **_kwargs: None)

    cluster_solver._prepare_direct_evaluator(args, state)  # type: ignore[arg-type]

    approximation_index = captured_events.index("[approximations]")
    assert approximation_index < captured_events.index("evaluator_summary")
    assert approximation_index < captured_events.index("[surrogate] initializing exact_scaling=1 cached_scaling=1 free_correction=0")
    assert approximation_index < captured_events.index("refresh_surrogate:svi_nuts_initial")
    assert approximation_index < captured_events.index("[compile] tracing first JAX likelihood evaluation")
    assert approximation_index < captured_events.index("source_loglike")


def test_solver_active_approximation_items_warn_when_image_presence_curvature_dominates() -> None:
    state = SimpleNamespace(
        family_data=[object()],
        bin_data=[
            SimpleNamespace(
                sigma_per_image=np.asarray([0.2, 0.2], dtype=float),
            )
        ],
        parameter_specs=[SimpleNamespace(component_family="source_position")],
        potfiles=[],
    )
    evaluator = SimpleNamespace(
        state=state,
        surrogate_enabled=False,
        inactive_scaling_component_indices=np.asarray([], dtype=int),
        active_scaling_component_indices=np.asarray([], dtype=int),
        scaling_component_indices=np.asarray([], dtype=int),
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        source_position_parameterization="conditional-whitened",
        image_presence_penalty_weight=3.0,
        image_presence_temperature_arcsec=0.05,
        image_plane_scatter_floor_arcsec=0.1,
        source_plane_covariance_floor=0.0,
        scaling_scatter_param_indices=np.asarray([], dtype=int),
        source_metric_cache_by_z={},
    )

    items = cluster_solver._solver_active_approximation_items(evaluator)

    assert any("image_presence_penalty=non-Gaussian conditional transport is approximate" in item for item in items)
    assert any("image_presence_penalty=curvature may dominate Gaussian beta transport" in item for item in items)


def _critical_arc_loglike_for_test(
    *,
    residual_x: float,
    residual_y: float,
    jac_a00: float = 0.0,
    jac_a01: float = 0.0,
    jac_a10: float = 0.0,
    jac_a11: float = 1.0,
    reliability: float = 0.999,
    outlier_sigma_arcsec: float = 50.0,
) -> float:
    return float(
        _critical_arc_loglike_jax_for_test(
            residual_x=residual_x,
            residual_y=residual_y,
            jac_a00=jac_a00,
            jac_a01=jac_a01,
            jac_a10=jac_a10,
            jac_a11=jac_a11,
            reliability=reliability,
            outlier_sigma_arcsec=outlier_sigma_arcsec,
        )
    )


def _critical_arc_loglike_jax_for_test(
    *,
    residual_x: Any,
    residual_y: Any,
    jac_a00: Any = 0.0,
    jac_a01: Any = 0.0,
    jac_a10: Any = 0.0,
    jac_a11: Any = 1.0,
    reliability: Any = 0.999,
    outlier_sigma_arcsec: float = 50.0,
) -> jnp.ndarray:
    def scalar_array(value: Any) -> jnp.ndarray:
        return jnp.reshape(jnp.asarray(value, dtype=jnp.float64), (1,))

    return _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=scalar_array(residual_x),
        residual_y=scalar_array(residual_y),
        jac_a00=scalar_array(jac_a00),
        jac_a01=scalar_array(jac_a01),
        jac_a10=scalar_array(jac_a10),
        jac_a11=scalar_array(jac_a11),
        family_idx=None,
        n_families=None,
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=scalar_array(reliability),
        image_has_constraint=jnp.asarray([True], dtype=bool),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=outlier_sigma_arcsec,
        image_presence_penalty_weight=0.0,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )


def test_critical_arc_branch_probability_increases_near_singular() -> None:
    probabilities = np.asarray(
        _critical_arc_branch_probability(
            jnp.asarray([1.0, 0.01], dtype=jnp.float64),
            base_prob=0.10,
            max_prob=0.80,
            singular_threshold=0.20,
            singular_softness=0.05,
        )
    )

    assert probabilities[1] > probabilities[0]
    assert probabilities[0] >= 0.10
    assert probabilities[1] <= 0.80


def test_critical_arc_branch_probability_accepts_sampled_jax_threshold() -> None:
    @jax.jit
    def probability_for_threshold(threshold: jnp.ndarray) -> jnp.ndarray:
        return _critical_arc_branch_probability(
            jnp.asarray([0.12], dtype=jnp.float64),
            base_prob=0.10,
            max_prob=0.80,
            singular_threshold=threshold,
            singular_softness=jnp.asarray(0.02, dtype=jnp.float64),
        )[0]

    low_threshold = float(probability_for_threshold(jnp.asarray(0.05, dtype=jnp.float64)))
    high_threshold = float(probability_for_threshold(jnp.asarray(0.20, dtype=jnp.float64)))

    assert np.isfinite(low_threshold)
    assert np.isfinite(high_threshold)
    assert high_threshold > low_threshold


def test_critical_arc_lm_step_rank_deficient_jacobian_is_finite_and_bounded() -> None:
    dx, dy, finite = _critical_arc_lm_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([-50.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([0.0], dtype=jnp.float64),
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    radius = np.hypot(np.asarray(dx), np.asarray(dy))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(radius).all()
    assert float(radius[0]) <= 20.0


def test_critical_arc_fused_lm_geometry_matches_anchored_step() -> None:
    f_x = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    f_y = jnp.asarray([0.1, 0.4], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 0.4], dtype=jnp.float64)
    jac_a01 = jnp.asarray([0.1, -0.3], dtype=jnp.float64)
    jac_a10 = jnp.asarray([0.0, 0.2], dtype=jnp.float64)
    jac_a11 = jnp.asarray([1.0, 0.8], dtype=jnp.float64)

    fused_dx, fused_dy, *_geometry, fused_finite = _critical_arc_lm_geometry_from_jacobian(
        f_x,
        f_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )
    anchored_dx, anchored_dy, anchored_finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x,
        f_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        trust_radius_arcsec=20.0,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    np.testing.assert_allclose(np.asarray(fused_dx), np.asarray(anchored_dx), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(fused_dy), np.asarray(anchored_dy), rtol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(fused_finite), np.asarray(anchored_finite))


def test_critical_arc_projector_quadratics_match_frame_projection() -> None:
    residual_x = jnp.asarray([0.7, -0.4], dtype=jnp.float64)
    residual_y = jnp.asarray([0.2, 0.5], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 0.4], dtype=jnp.float64)
    jac_a01 = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    jac_a10 = jnp.asarray([0.1, 0.1], dtype=jnp.float64)
    jac_a11 = jnp.asarray([1.0, 0.8], dtype=jnp.float64)

    *_singulars, critical_p00, critical_p01, critical_p11, finite = _critical_arc_geometry_from_jacobian(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    critical_direction_quad, noncritical_direction_quad = _critical_arc_projected_quadratics(
        residual_x,
        residual_y,
        critical_p00,
        critical_p01,
        critical_p11,
    )
    critical_direction_x, critical_direction_y, noncritical_direction_x, noncritical_direction_y, *_frame = cluster_solver._critical_arc_jacobian_frame(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )
    expected_critical_direction_quad = jnp.square(residual_x * critical_direction_x + residual_y * critical_direction_y)
    expected_noncritical_direction_quad = jnp.square(residual_x * noncritical_direction_x + residual_y * noncritical_direction_y)

    assert bool(np.asarray(jnp.all(finite)))
    np.testing.assert_allclose(
        np.asarray(critical_direction_quad),
        np.asarray(expected_critical_direction_quad),
        rtol=5.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(noncritical_direction_quad),
        np.asarray(expected_noncritical_direction_quad),
        rtol=5.0e-6,
        atol=1.0e-6,
    )


def test_critical_arc_projector_degenerate_normal_matrix_is_isotropic() -> None:
    (
        critical_p00,
        critical_p01,
        critical_p11,
    ) = cluster_solver._critical_arc_critical_direction_projector_from_normal_entries(
        normal00=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        normal01=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        normal11=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        trace=jnp.asarray([0.0, 4.0], dtype=jnp.float64),
    )

    np.testing.assert_allclose(np.asarray(critical_p00), np.asarray([0.5, 0.5]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(critical_p01), np.asarray([0.0, 0.0]), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(critical_p11), np.asarray([0.5, 0.5]), rtol=0.0, atol=1.0e-12)


def test_critical_arc_mixture_is_finite_for_singular_jacobian() -> None:
    loglike = _critical_arc_loglike_for_test(
        residual_x=2.0,
        residual_y=-1.0,
        jac_a00=0.0,
        jac_a01=0.0,
        jac_a10=0.0,
        jac_a11=0.0,
    )

    assert np.isfinite(loglike)


def test_critical_arc_mixture_terms_sum_to_bin_loglike_without_presence_penalty() -> None:
    residual_x = jnp.asarray([0.2, -0.5], dtype=jnp.float64)
    residual_y = jnp.asarray([0.1, 0.4], dtype=jnp.float64)
    jac_a00 = jnp.asarray([1.0, 0.4], dtype=jnp.float64)
    jac_a01 = jnp.asarray([0.0, -0.1], dtype=jnp.float64)
    jac_a10 = jnp.asarray([0.0, -0.1], dtype=jnp.float64)
    jac_a11 = jnp.asarray([1.0, 0.8], dtype=jnp.float64)
    image_has_constraint = jnp.asarray([True, False], dtype=bool)
    (
        singular_min,
        singular_max,
        critical_p00,
        critical_p01,
        critical_p11,
        finite,
    ) = _critical_arc_geometry_from_jacobian(
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
    )

    terms = cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma_per_image=jnp.asarray([0.05, 0.08], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.9], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        singular_min=singular_min,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )
    direct = _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=residual_x,
        residual_y=residual_y,
        jac_a00=jac_a00,
        jac_a01=jac_a01,
        jac_a10=jac_a10,
        jac_a11=jac_a11,
        family_idx=None,
        n_families=None,
        sigma_per_image=jnp.asarray([0.05, 0.08], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.9], dtype=jnp.float64),
        image_has_constraint=image_has_constraint,
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        image_presence_penalty_weight=0.0,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
        singular_min_precomputed=singular_min,
        singular_max_precomputed=singular_max,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
    )
    expected = jnp.sum(jnp.where(image_has_constraint, terms.mixture_ll, 0.0))

    assert bool(np.asarray(jnp.all(finite)))
    assert float(direct) == pytest.approx(float(expected), abs=1.0e-12)


def _expected_fold_extra_var(
    sigma_per_image: float,
    singular_min: float,
    singular_threshold: float,
    singular_softness: float,
) -> float:
    """Independent re-derivation of the always-on magnification-fold extra variance.

    Mirrors the formula in ``_critical_arc_mixture_image_plane_terms``: near a critical curve
    the base covariance is inflated along the critical direction by ``(sigma_per_image /
    singular_min)^2``, gated by the same sigmoid as ``arc_prob`` and capped.
    """
    gate = 1.0 / (1.0 + math.exp(-(singular_threshold - singular_min) / singular_softness))
    floor = float(cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_FLOOR)
    cap = float(cluster_solver.CRITICAL_ARC_FOLD_MAX_ARC_SIGMA_ARCSEC)
    return min((sigma_per_image / max(singular_min, floor)) ** 2 * gate, cap**2)


def _expected_critical_arc_gaussian_ll_full_covariance(
    *,
    residual_x: float,
    residual_y: float,
    sigma2: float,
    extra_var: float,
    projector_entries: tuple[float, float, float],
    scatter_cov_entries: tuple[float, float, float],
) -> float:
    p00, p01, p11 = projector_entries
    cov00, cov01, cov11 = scatter_cov_entries
    covariance = np.asarray(
        [
            [sigma2 + cov00 + extra_var * p00, cov01 + extra_var * p01],
            [cov01 + extra_var * p01, sigma2 + cov11 + extra_var * p11],
        ],
        dtype=float,
    )
    residual = np.asarray([residual_x, residual_y], dtype=float)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return -0.5 * (
        float(residual @ np.linalg.solve(covariance, residual))
        + 2.0 * math.log(2.0 * math.pi)
        + float(logdet)
    )


def _critical_arc_terms_for_projector_test(
    projector_entries: tuple[float, float, float],
    residual_x: float,
    residual_y: float,
    critical_direction_sigma_arcsec: float,
):
    return cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
        residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        singular_min=jnp.asarray([0.05], dtype=jnp.float64),
        critical_direction_projector_entries=(
            jnp.asarray([projector_entries[0]], dtype=jnp.float64),
            jnp.asarray([projector_entries[1]], dtype=jnp.float64),
            jnp.asarray([projector_entries[2]], dtype=jnp.float64),
        ),
        residual_loss="gaussian",
        critical_direction_sigma_arcsec=critical_direction_sigma_arcsec,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )


def test_critical_arc_arc_branch_degenerate_limit_is_isotropic() -> None:
    residual_x, residual_y = 0.7, -0.4
    sigma_arc = 5.0
    terms = _critical_arc_terms_for_projector_test((0.5, 0.0, 0.5), residual_x, residual_y, sigma_arc)
    sigma2 = 0.05**2 + 1.0e-12
    # Fold gate is decoupled from the arc-mixture singular_threshold/softness: it uses its own
    # baked-in constants, so the expected arc-branch fold must use them too.
    fold = _expected_fold_extra_var(
        0.05,
        0.05,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_THRESHOLD,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_SOFTNESS,
    )
    isotropic_var = sigma2 + 0.5 * (sigma_arc**2 + fold)
    expected_arc_ll = (
        -math.log(2.0 * math.pi * isotropic_var)
        - (residual_x**2 + residual_y**2) / (2.0 * isotropic_var)
    )

    assert float(terms.arc_ll[0]) == pytest.approx(expected_arc_ll, rel=1.0e-12)


@pytest.mark.parametrize("projector_entries", [(1.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
def test_critical_arc_arc_branch_reduces_to_two_axis_form_for_exact_projector(
    projector_entries: tuple[float, float, float],
) -> None:
    residual_x, residual_y = 0.7, -0.4
    sigma_arc = 5.0
    terms = _critical_arc_terms_for_projector_test(projector_entries, residual_x, residual_y, sigma_arc)
    p00, p01, p11 = projector_entries
    sigma2 = 0.05**2 + 1.0e-12
    # Fold gate is decoupled from the arc-mixture singular_threshold/softness: it uses its own
    # baked-in constants, so the expected arc-branch fold must use them too.
    fold = _expected_fold_extra_var(
        0.05,
        0.05,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_THRESHOLD,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_SOFTNESS,
    )
    sigma_parallel2 = sigma2 + sigma_arc**2 + fold
    critical_quad = residual_x**2 * p00 + 2.0 * residual_x * residual_y * p01 + residual_y**2 * p11
    noncritical_quad = residual_x**2 + residual_y**2 - critical_quad
    expected_arc_ll = -0.5 * (
        noncritical_quad / sigma2
        + critical_quad / sigma_parallel2
        + 2.0 * math.log(2.0 * math.pi)
        + math.log(sigma2)
        + math.log(sigma_parallel2)
    )

    assert float(terms.arc_ll[0]) == pytest.approx(expected_arc_ll, rel=1.0e-12)


def test_critical_arc_arc_branch_is_normalized_density_for_regularized_projector() -> None:
    residual = np.asarray([0.7, 0.2])
    sigma_arc = 5.0
    (
        singular_min,
        _singular_max,
        critical_p00,
        critical_p01,
        critical_p11,
        finite,
    ) = _critical_arc_geometry_from_jacobian(
        jnp.asarray([1.1], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.3], dtype=jnp.float64),
    )
    terms = cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=jnp.asarray([residual[0]], dtype=jnp.float64),
        residual_y=jnp.asarray([residual[1]], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        singular_min=singular_min,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        residual_loss="gaussian",
        critical_direction_sigma_arcsec=sigma_arc,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )
    projector = np.asarray(
        [
            [float(critical_p00[0]), float(critical_p01[0])],
            [float(critical_p01[0]), float(critical_p11[0])],
        ]
    )
    sigma2 = 0.05**2 + 1.0e-12
    fold = _expected_fold_extra_var(
        0.05,
        float(np.asarray(singular_min)[0]),
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_THRESHOLD,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_SOFTNESS,
    )
    covariance = sigma2 * np.eye(2) + (sigma_arc**2 + fold) * projector
    expected_arc_ll = -0.5 * (
        float(residual @ np.linalg.solve(covariance, residual))
        + 2.0 * math.log(2.0 * math.pi)
        + float(np.linalg.slogdet(covariance)[1])
    )

    assert bool(np.asarray(jnp.all(finite)))
    assert float(terms.arc_ll[0]) == pytest.approx(expected_arc_ll, rel=1.0e-12)


def test_critical_arc_point_branch_uses_full_scatter_covariance() -> None:
    residual_x, residual_y = 0.3, -0.2
    sigma_per_image = 0.2
    image_sigma_int = 0.03
    singular_min = 0.08
    projector_entries = (0.7, 0.2, 0.3)
    scatter_cov_entries = (0.04, -0.01, 0.02)
    terms = cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
        residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma_per_image], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(image_sigma_int, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        singular_min=jnp.asarray([singular_min], dtype=jnp.float64),
        critical_direction_projector_entries=(
            jnp.asarray([projector_entries[0]], dtype=jnp.float64),
            jnp.asarray([projector_entries[1]], dtype=jnp.float64),
            jnp.asarray([projector_entries[2]], dtype=jnp.float64),
        ),
        residual_loss="gaussian",
        critical_direction_sigma_arcsec=0.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
        scatter_cov00=jnp.asarray([scatter_cov_entries[0]], dtype=jnp.float64),
        scatter_cov01=jnp.asarray([scatter_cov_entries[1]], dtype=jnp.float64),
        scatter_cov11=jnp.asarray([scatter_cov_entries[2]], dtype=jnp.float64),
    )
    sigma2 = sigma_per_image**2 + image_sigma_int**2 + 1.0e-12
    fold = _expected_fold_extra_var(
        sigma_per_image,
        singular_min,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_THRESHOLD,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_SOFTNESS,
    )
    expected_point_ll = _expected_critical_arc_gaussian_ll_full_covariance(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma2=sigma2,
        extra_var=fold,
        projector_entries=projector_entries,
        scatter_cov_entries=scatter_cov_entries,
    )

    assert float(terms.point_ll[0]) == pytest.approx(expected_point_ll, rel=1.0e-12)


def test_critical_arc_arc_branch_uses_full_scatter_covariance() -> None:
    residual_x, residual_y = -0.35, 0.45
    sigma_per_image = 0.18
    image_sigma_int = 0.02
    singular_min = 0.04
    sigma_arc = 1.7
    projector_entries = (0.6, -0.25, 0.4)
    scatter_cov_entries = (0.03, 0.012, 0.05)
    terms = cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
        residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma_per_image], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(image_sigma_int, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        singular_min=jnp.asarray([singular_min], dtype=jnp.float64),
        critical_direction_projector_entries=(
            jnp.asarray([projector_entries[0]], dtype=jnp.float64),
            jnp.asarray([projector_entries[1]], dtype=jnp.float64),
            jnp.asarray([projector_entries[2]], dtype=jnp.float64),
        ),
        residual_loss="gaussian",
        critical_direction_sigma_arcsec=sigma_arc,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
        scatter_cov00=jnp.asarray([scatter_cov_entries[0]], dtype=jnp.float64),
        scatter_cov01=jnp.asarray([scatter_cov_entries[1]], dtype=jnp.float64),
        scatter_cov11=jnp.asarray([scatter_cov_entries[2]], dtype=jnp.float64),
    )
    sigma2 = sigma_per_image**2 + image_sigma_int**2 + 1.0e-12
    fold = _expected_fold_extra_var(
        sigma_per_image,
        singular_min,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_THRESHOLD,
        cluster_solver.CRITICAL_ARC_FOLD_SINGULAR_SOFTNESS,
    )
    expected_arc_ll = _expected_critical_arc_gaussian_ll_full_covariance(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma2=sigma2,
        extra_var=fold + sigma_arc**2,
        projector_entries=projector_entries,
        scatter_cov_entries=scatter_cov_entries,
    )

    assert float(terms.arc_ll[0]) == pytest.approx(expected_arc_ll, rel=1.0e-12)


def _critical_arc_point_ll_for_fold_test(singular_min: float, residual_x: float):
    # Exact projector P = diag(1, 0): the critical (tangential) direction is axis 0.
    return cluster_solver._critical_arc_mixture_image_plane_terms(
        residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.25], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=50.0,
        singular_min=jnp.asarray([singular_min], dtype=jnp.float64),
        critical_direction_projector_entries=(
            jnp.asarray([1.0], dtype=jnp.float64),
            jnp.asarray([0.0], dtype=jnp.float64),
            jnp.asarray([0.0], dtype=jnp.float64),
        ),
        residual_loss="gaussian",
        critical_direction_sigma_arcsec=0.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.05,
        singular_softness=0.02,
    ).point_ll[0]


def test_critical_arc_magnification_fold_off_far_from_critical() -> None:
    # singular_min >> threshold: the gate is closed, so the point branch is the original
    # isotropic sigma^2 I form -- point images away from caustics are untouched.
    sigma2 = 0.25**2
    point_ll = float(_critical_arc_point_ll_for_fold_test(1.0, 0.4))
    expected_iso = -0.5 * (
        0.4**2 / sigma2 + 2.0 * math.log(2.0 * math.pi) + 2.0 * math.log(sigma2)
    )
    assert point_ll == pytest.approx(expected_iso, rel=1.0e-9)


def test_critical_arc_magnification_fold_bounds_tangential_gradient_on_critical() -> None:
    # The whole point of the fold: on a critical curve the tangential-residual gradient of the
    # point branch collapses (instead of blowing up with magnification), so NUTS can move.
    g_far = float(jax.grad(lambda r: _critical_arc_point_ll_for_fold_test(1.0, r))(jnp.asarray(0.4)))
    g_near = float(jax.grad(lambda r: _critical_arc_point_ll_for_fold_test(1.0e-6, r))(jnp.asarray(0.4)))
    assert abs(g_near) < 1.0e-3 * abs(g_far)


@pytest.mark.parametrize("lambda_min", [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-12])
def test_critical_arc_gating_gradient_does_not_explode_through_singular_min(lambda_min: float) -> None:
    # Regression: stage4 SVI init failed ("Cannot find valid initial parameters") because the
    # geometric gating (fold + arc_prob) is differentiated through singular_min = sqrt(lambda_min),
    # whose 1/singular_min slope in the transition zone made d(loglike)/d(lambda_min) spike to
    # ~1e6 per image and overflow numpyro's init check. The gating is detached (stop_gradient),
    # so its lambda_min-channel gradient must be ~0 (it can only flow through residuals/sigma2).
    def loglike_from_lambda(lam: jnp.ndarray) -> jnp.ndarray:
        singular_min = jnp.sqrt(jnp.maximum(lam, 0.0) + 1.0e-24).reshape((1,))
        terms = cluster_solver._critical_arc_mixture_image_plane_terms(
            residual_x=jnp.asarray([0.5], dtype=jnp.float64),
            residual_y=jnp.asarray([0.4], dtype=jnp.float64),
            sigma_per_image=jnp.asarray([0.25], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            covariance_floor=1.0e-12,
            outlier_sigma_arcsec=50.0,
            singular_min=singular_min,
            critical_direction_projector_entries=(
                jnp.asarray([1.0], dtype=jnp.float64),
                jnp.asarray([0.0], dtype=jnp.float64),
                jnp.asarray([0.0], dtype=jnp.float64),
            ),
            residual_loss="gaussian",
            critical_direction_sigma_arcsec=10.0,
            base_prob=0.10,
            max_prob=0.85,
            singular_threshold=0.05,
            singular_softness=0.02,
        )
        return jnp.sum(terms.mixture_ll)

    grad = float(jax.grad(loglike_from_lambda)(jnp.asarray(lambda_min)))
    assert np.isfinite(grad)
    assert abs(grad) < 1.0e-6  # detached gating -> no lambda_min-channel gradient


@pytest.mark.parametrize(
    "jacobian_entries",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, -0.1, -0.1, 0.0),
        (1.0, 0.0, 0.0, 1.0),
    ],
)
def test_critical_arc_mixture_has_finite_jacobian_gradient(
    jacobian_entries: tuple[float, float, float, float],
) -> None:
    def loglike_from_jacobian(entries: jnp.ndarray) -> jnp.ndarray:
        return _critical_arc_loglike_jax_for_test(
            residual_x=2.0,
            residual_y=-1.0,
            jac_a00=entries[0],
            jac_a01=entries[1],
            jac_a10=entries[2],
            jac_a11=entries[3],
        )

    entries = jnp.asarray(jacobian_entries, dtype=jnp.float64)
    value = loglike_from_jacobian(entries)
    gradient = jax.grad(loglike_from_jacobian)(entries)

    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize("gamma1", [0.0, 1.0e-6])
def test_critical_arc_mixture_physical_shear_gradient_is_bounded(gamma1: float) -> None:
    def loglike_from_gamma2(gamma2: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(1.0, dtype=jnp.float64)
        gamma1_value = jnp.asarray(gamma1, dtype=jnp.float64)
        return _critical_arc_loglike_jax_for_test(
            residual_x=2.0,
            residual_y=-1.0,
            jac_a00=p - gamma1_value,
            jac_a01=-gamma2,
            jac_a10=-gamma2,
            jac_a11=p + gamma1_value,
        )

    gradient = jax.grad(loglike_from_gamma2)(jnp.asarray(0.0, dtype=jnp.float64))

    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) < 1.0e3


def test_critical_arc_large_critical_direction_displacement_is_cheaper_than_noncritical() -> None:
    critical_direction_loglike = _critical_arc_loglike_for_test(residual_x=2.0, residual_y=0.0)
    noncritical_direction_loglike = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=2.0)

    assert critical_direction_loglike > noncritical_direction_loglike


def test_critical_arc_image_presence_counts_along_arc_displacement_as_recovered() -> None:
    kwargs = dict(
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        image_presence_match_radius_arcsec=0.5,
        image_presence_temperature_arcsec=0.1,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.05,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.85,
        singular_threshold=0.20,
        singular_softness=0.05,
    )

    def penalty_delta(residual_x: float, residual_y: float) -> float:
        base = _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
            residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
            image_presence_penalty_weight=0.0,
            **kwargs,
        )
        with_presence = _critical_arc_mixture_image_plane_bin_loglike(
            residual_x=jnp.asarray([residual_x], dtype=jnp.float64),
            residual_y=jnp.asarray([residual_y], dtype=jnp.float64),
            image_presence_penalty_weight=2.0,
            **kwargs,
        )
        return float(with_presence - base)

    along_arc_penalty = penalty_delta(2.0, 0.0)
    noncritical_penalty = penalty_delta(0.0, 2.0)

    assert along_arc_penalty > -0.05
    assert noncritical_penalty < -0.5
    assert along_arc_penalty > noncritical_penalty + 0.5


def test_critical_arc_student_t_keeps_critical_direction_cheaper_than_noncritical() -> None:
    kwargs = dict(
        jac_a00=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        family_idx=None,
        n_families=None,
        sigma_per_image=jnp.asarray([0.05], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True], dtype=bool),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=50.0,
        image_presence_penalty_weight=0.0,
        residual_loss="student-t",
        student_t_nu=4.0,
        critical_direction_sigma_arcsec=5.0,
        base_prob=0.10,
        max_prob=0.80,
        singular_threshold=0.20,
        singular_softness=0.05,
    )

    critical_direction_loglike = _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=jnp.asarray([2.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0], dtype=jnp.float64),
        **kwargs,
    )
    noncritical_direction_loglike = _critical_arc_mixture_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([2.0], dtype=jnp.float64),
        **kwargs,
    )

    assert np.isfinite(float(critical_direction_loglike))
    assert np.isfinite(float(noncritical_direction_loglike))
    assert float(critical_direction_loglike) > float(noncritical_direction_loglike)


def test_critical_arc_det_sign_crossing_stays_finite_and_continuous() -> None:
    positive = _critical_arc_loglike_for_test(
        residual_x=0.4,
        residual_y=0.2,
        jac_a00=1.0,
        jac_a11=0.02,
    )
    negative = _critical_arc_loglike_for_test(
        residual_x=0.4,
        residual_y=0.2,
        jac_a00=1.0,
        jac_a11=-0.02,
    )

    assert np.isfinite(positive)
    assert np.isfinite(negative)
    assert positive == pytest.approx(negative, abs=1.0e-12)


def test_critical_arc_reliability_outlier_mixture_remains_active() -> None:
    high_reliability = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=8.0, reliability=0.999)
    low_reliability = _critical_arc_loglike_for_test(residual_x=0.0, residual_y=8.0, reliability=0.10)

    assert low_reliability > high_reliability


def _constant_arc_jacobian(eps: float = 0.05):
    def jacobian_at(x_values, _y_values):
        x_array = np.asarray(x_values, dtype=float).reshape(-1)
        return (
            np.full_like(x_array, eps, dtype=float),
            np.zeros_like(x_array, dtype=float),
            np.zeros_like(x_array, dtype=float),
            np.ones_like(x_array, dtype=float),
        )

    return jacobian_at


def _circle_tangent_arc_jacobian(eps: float = 0.05):
    def jacobian_at(x_values, y_values):
        x_array = np.asarray(x_values, dtype=float).reshape(-1)
        y_array = np.asarray(y_values, dtype=float).reshape(-1)
        radius = np.maximum(np.hypot(x_array, y_array), 1.0e-12)
        tangent_x = -y_array / radius
        tangent_y = x_array / radius
        normal_x = x_array / radius
        normal_y = y_array / radius
        a00 = eps * tangent_x * tangent_x + normal_x * normal_x
        a01 = eps * tangent_x * tangent_y + normal_x * normal_y
        a10 = eps * tangent_y * tangent_x + normal_y * normal_x
        a11 = eps * tangent_y * tangent_y + normal_y * normal_y
        return a00, a01, a10, a11

    return jacobian_at


def _critical_arc_gate_value(
    s_min: float,
    *,
    base_prob: float = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB,
    max_prob: float = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB,
    singular_threshold: float = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
) -> float:
    transition = 1.0 / (1.0 + math.exp(-((float(singular_threshold) - float(s_min)) / float(singular_softness))))
    return float(base_prob) + (float(max_prob) - float(base_prob)) * transition


def _critical_arc_anisotropic_gate_value(
    s_min: float,
    *,
    singular_threshold: float = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD,
    singular_softness: float = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS,
) -> float:
    return 1.0 / (1.0 + math.exp(-((float(singular_threshold) - float(s_min)) / float(singular_softness))))


def test_critical_arc_anisotropic_covariance_false_matches_isotropic_terms() -> None:
    residual_x = jnp.asarray([0.08, -0.05], dtype=jnp.float64)
    residual_y = jnp.asarray([0.03, 0.04], dtype=jnp.float64)
    sigma_per_image = jnp.asarray([0.04, 0.05], dtype=jnp.float64)
    reliability = jnp.asarray([0.99, 0.98], dtype=jnp.float64)
    image_sigma_int = jnp.asarray(0.0, dtype=jnp.float64)
    singular_min = jnp.asarray([1.0e-3, 0.02], dtype=jnp.float64)
    critical_p00 = jnp.asarray([0.7, 0.2], dtype=jnp.float64)
    critical_p01 = jnp.asarray([0.1, -0.05], dtype=jnp.float64)
    critical_p11 = jnp.asarray([0.3, 0.8], dtype=jnp.float64)

    enabled = cluster_solver._critical_arc_anisotropic_image_plane_terms(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma_per_image=sigma_per_image,
        reliability_per_image=reliability,
        image_sigma_int=image_sigma_int,
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        singular_min=singular_min,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        critical_arc_anisotropic_covariance=True,
        critical_direction_sigma_arcsec=5.0,
    )
    disabled = cluster_solver._critical_arc_anisotropic_image_plane_terms(
        residual_x=residual_x,
        residual_y=residual_y,
        sigma_per_image=sigma_per_image,
        reliability_per_image=reliability,
        image_sigma_int=image_sigma_int,
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        singular_min=singular_min,
        critical_direction_projector_entries=(critical_p00, critical_p01, critical_p11),
        critical_arc_anisotropic_covariance=False,
        critical_direction_sigma_arcsec=5.0,
    )
    sigma2 = cluster_solver._image_plane_effective_sigma2(sigma_per_image, image_sigma_int, 0.0)
    projector_det = jnp.maximum(critical_p00 * critical_p11 - jnp.square(critical_p01), 0.0)
    isotropic_quad, isotropic_logdet = cluster_solver._critical_arc_aniso_quad_logdet(
        residual_x,
        residual_y,
        sigma2,
        jnp.zeros_like(sigma2),
        critical_p00,
        critical_p01,
        critical_p11,
        projector_det,
    )
    expected_inlier_ll = -0.5 * (isotropic_quad + 2.0 * jnp.log(2.0 * jnp.pi) + isotropic_logdet)
    outlier_sigma2 = jnp.square(jnp.asarray(10.0, dtype=jnp.float64))
    expected_outlier_ll = -0.5 * (
        (jnp.square(residual_x) + jnp.square(residual_y)) / outlier_sigma2
        + 2.0 * jnp.log(2.0 * jnp.pi * outlier_sigma2)
    )
    expected_mixture_ll = jnp.logaddexp(
        jnp.log(reliability) + expected_inlier_ll,
        jnp.log1p(-reliability) + expected_outlier_ll,
    )

    np.testing.assert_allclose(np.asarray(disabled.inlier_ll), np.asarray(expected_inlier_ll), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(disabled.mixture_ll), np.asarray(expected_mixture_ll), rtol=1.0e-12)
    assert not np.allclose(np.asarray(enabled.mixture_ll), np.asarray(disabled.mixture_ll))


def test_critical_arc_aware_support_saves_arc_candidate_for_point_recovery() -> None:
    eps = 0.03
    critical_direction_delta = 4.0
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * critical_direction_delta]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        point_recovered_mask=np.asarray([True]),
        point_residual_arcsec=np.asarray([0.07]),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["point_image_residual_arcsec"][0] == pytest.approx(0.07)
    assert "p_arc" in details
    assert "arc_branch_responsibility" not in details
    assert "arc_log_odds" in details
    assert "arc_inlier_responsibility" not in details
    assert "arc_margin_log_weight_minus_point" not in details
    assert details["arc_candidate_supported"].tolist() == [True]
    assert details["p_arc"][0] == pytest.approx(_critical_arc_gate_value(details["arc_s_min"][0]))
    assert details["arc_prior_probability"][0] == pytest.approx(details["p_arc"][0])
    assert details["arc_log_odds"][0] == pytest.approx(math.log(details["p_arc"][0] / (1.0 - details["p_arc"][0])))
    assert details["arc_candidate_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["preferred_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["preferred_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_aware_image_rms_arcsec"] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_finite"].tolist() == [True]
    assert details["arc_support_finite_mask"].tolist() == [True]
    assert json.loads(details["arc_support_curve_x_arcsec"][0])
    assert json.loads(details["arc_support_curve_y_arcsec"][0])


def test_critical_arc_aware_support_uses_anisotropic_p_arc_for_anisotropic_mode() -> None:
    eps = 0.03
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
        sample_likelihood_mode=cluster_solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
    )

    anisotropic_gate = _critical_arc_anisotropic_gate_value(details["arc_s_min"][0])
    mixture_gate = _critical_arc_gate_value(details["arc_s_min"][0])
    assert details["p_arc"][0] == pytest.approx(anisotropic_gate)
    assert details["arc_prior_probability"][0] == pytest.approx(anisotropic_gate)
    assert details["p_arc"][0] != pytest.approx(mixture_gate)
    assert details["arc_candidate_supported"].tolist() == [True]


def test_critical_arc_aware_support_keeps_point_on_exact_tie() -> None:
    eps = 0.03
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        point_recovered_mask=np.asarray([True]),
        point_residual_arcsec=np.asarray([0.0]),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_candidate_supported"].tolist() == [True]
    assert details["p_arc"][0] >= 0.5
    assert details["arc_candidate_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["preferred_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)


def test_critical_arc_aware_support_tolerates_large_critical_direction_residual() -> None:
    eps = 0.03
    critical_direction_delta = 4.0
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * critical_direction_delta]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["p_arc"][0] >= 0.5
    assert details["arc_noncritical_direction_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_critical_direction_residual_arcsec"][0] > 1.0
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_distance_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_arclength_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_support_anchor_x_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_support_anchor_y_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_finite"].tolist() == [True]
    assert json.loads(details["arc_support_curve_x_arcsec"][0])
    assert json.loads(details["arc_support_curve_y_arcsec"][0])
    assert math.hypot(details["arc_critical_direction_x"][0], details["arc_critical_direction_y"][0]) == pytest.approx(1.0)
    assert math.hypot(details["arc_noncritical_direction_x"][0], details["arc_noncritical_direction_y"][0]) == pytest.approx(1.0)
    assert details["arc_s_min"][0] == pytest.approx(eps, abs=1.0e-5)
    assert details["arc_s_max"][0] == pytest.approx(1.0, abs=1.0e-5)
    assert details["arc_detA"][0] == pytest.approx(eps)
    assert details["arc_support_finite_mask"].tolist() == [True]


def test_critical_arc_aware_support_rejects_below_configured_p_arc_threshold() -> None:
    eps = 0.03
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        arc_recovery_p_arc_threshold=0.7,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_prior_probability"][0] > 0.6
    assert details["p_arc"][0] < 0.7
    assert details["arc_candidate_supported"].tolist() == [False]
    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert np.isnan(details["arc_aware_image_residual_arcsec"][0])


def test_critical_arc_aware_support_extends_curve_for_large_critical_residual() -> None:
    eps = 0.03
    critical_direction_delta = 6.0
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * critical_direction_delta]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_candidate_supported"].tolist() == [True]
    assert details["p_arc"][0] >= 0.5
    assert details["arc_curve_distance_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_arclength_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_support_anchor_x_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_support_anchor_y_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(details["arc_curve_distance_arcsec"][0])


def test_critical_arc_aware_support_anchor_keeps_only_noncritical_component() -> None:
    eps = 0.03
    noncritical_delta = 0.97
    critical_delta = 13.3
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([eps * critical_delta]),
        beta_residual_y=np.asarray([-noncritical_delta]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=_constant_arc_jacobian(eps),
        trust_radius_arcsec=100.0,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_candidate_supported"].tolist() == [True]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_noncritical_direction_residual_arcsec"][0] < 1.0
    assert details["arc_critical_direction_residual_arcsec"][0] > 10.0
    anchor_x = details["arc_support_anchor_x_arcsec"][0]
    anchor_y = details["arc_support_anchor_y_arcsec"][0]
    critical_offset = (
        anchor_x * details["arc_critical_direction_x"][0]
        + anchor_y * details["arc_critical_direction_y"][0]
    )
    noncritical_offset = (
        anchor_x * details["arc_noncritical_direction_x"][0]
        + anchor_y * details["arc_noncritical_direction_y"][0]
    )
    assert critical_offset == pytest.approx(0.0, abs=1.0e-6)
    assert abs(noncritical_offset) == pytest.approx(
        details["arc_noncritical_direction_residual_arcsec"][0],
        abs=1.0e-6,
    )
    assert details["arc_curve_distance_arcsec"][0] == pytest.approx(
        details["arc_noncritical_direction_residual_arcsec"][0],
        abs=1.0e-6,
    )
    assert details["arc_curve_arclength_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)


def test_critical_arc_aware_support_keeps_partial_curve_when_far_branch_fails() -> None:
    eps = 0.03

    def partial_arc_jacobian(x_values, y_values):
        x_array = np.asarray(x_values, dtype=float)
        if np.any(x_array > 6.2):
            nan = np.full_like(x_array, np.nan, dtype=float)
            return nan, nan, nan, nan
        return _constant_arc_jacobian(eps)(x_values, y_values)

    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * 6.0]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=partial_arc_jacobian,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_curve_finite"].tolist() == [True]
    assert details["arc_curve_distance_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)
    assert details["arc_curve_arclength_arcsec"][0] == pytest.approx(0.0, abs=1.0e-6)


def test_critical_arc_aware_support_accepts_curved_support_curve() -> None:
    eps = 0.03
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([1.0]),
        jacobian_at=_circle_tangent_arc_jacobian(eps),
        curve_step_arcsec=0.05,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_curve_distance_arcsec"][0] < 0.08
    assert details["arc_curve_arclength_arcsec"][0] >= 0.0
    assert details["arc_curve_arclength_arcsec"][0] <= cluster_solver.DEFAULT_ARC_AWARE_MAX_ARCLENGTH_ARCSEC
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(details["arc_curve_distance_arcsec"][0])


def test_critical_arc_aware_support_curve_trace_failure_is_diagnostic_only() -> None:
    def broken_jacobian_at(_x_values, _y_values):
        raise RuntimeError("no curve")

    eps = 0.03
    details = cluster_solver._arc_aware_image_support_from_local_linearization(
        beta_residual_x=np.asarray([-eps * 4.0]),
        beta_residual_y=np.asarray([0.0]),
        jac_a00=np.asarray([eps]),
        jac_a01=np.asarray([0.0]),
        jac_a10=np.asarray([0.0]),
        jac_a11=np.asarray([1.0]),
        theta_obs_x=np.asarray([0.0]),
        theta_obs_y=np.asarray([0.0]),
        jacobian_at=broken_jacobian_at,
        lm_damping_relative=0.0,
        lm_damping_absolute=0.0,
    )

    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert details["arc_candidate_supported"].tolist() == [True]
    assert details["p_arc"][0] >= 0.5
    assert details["arc_curve_finite"].tolist() == [False]
    assert details["arc_aware_image_residual_arcsec"][0] == pytest.approx(
        details["arc_noncritical_direction_residual_arcsec"][0]
    )


def test_critical_arc_aware_evaluator_uses_configured_validation_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=1,
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.models_by_effective_z = {2.0: object()}
    fake.arc_recovery_p_arc_threshold = 0.65
    fake.arc_aware_max_arclength_arcsec = 12.0
    fake.arc_aware_curve_step_arcsec = 0.2
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake.sample_likelihood_mode = cluster_solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE
    fake._build_packed_lens_state = lambda _params, _z_source: {}

    def fake_ray_shooting(_z_source, _x, _y, _packed_state):
        return jnp.asarray([0.0], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64)

    fake._ray_shooting_for_components = fake_ray_shooting
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.03], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    captured: dict[str, Any] = {}

    def fake_arc_support(*_args, **kwargs):
        captured["arc_recovery_p_arc_threshold"] = float(kwargs["arc_recovery_p_arc_threshold"])
        captured["max_arclength_arcsec"] = float(kwargs["max_arclength_arcsec"])
        captured["curve_step_arcsec"] = float(kwargs["curve_step_arcsec"])
        captured["sample_likelihood_mode"] = str(kwargs["sample_likelihood_mode"])
        return {"sentinel": True}

    monkeypatch.setattr(cluster_solver, "_arc_aware_image_support_from_local_linearization", fake_arc_support)

    details = cluster_solver.ClusterJAXEvaluator._arc_aware_image_support_details(
        fake,
        np.asarray([], dtype=float),
        family,
        0.0,
        0.0,
    )

    assert details == {"sentinel": True}
    assert captured["arc_recovery_p_arc_threshold"] == pytest.approx(0.65)
    assert captured["max_arclength_arcsec"] == pytest.approx(12.0)
    assert captured["curve_step_arcsec"] == pytest.approx(0.2)
    assert captured["sample_likelihood_mode"] == cluster_solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE


@pytest.mark.parametrize(
    "sample_likelihood_mode",
    [
        cluster_solver.SAMPLE_LIKELIHOOD_SOURCE,
        cluster_solver.SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN,
    ],
)
def test_arc_aware_evaluator_uses_point_only_recovery_for_noncritical_modes(
    monkeypatch: pytest.MonkeyPatch,
    sample_likelihood_mode: str,
) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.sample_likelihood_mode = sample_likelihood_mode
    match_details = {
        "recovered_image_mask": np.asarray([True, False], dtype=bool),
        "matched_model_x_arcsec": np.asarray([0.2, np.nan], dtype=float),
        "matched_model_y_arcsec": np.asarray([0.0, np.nan], dtype=float),
    }

    def fail_arc_support(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("critical-arc support should not be computed for non-critical likelihoods")

    monkeypatch.setattr(cluster_solver, "_arc_aware_image_support_from_local_linearization", fail_arc_support)

    details = cluster_solver.ClusterJAXEvaluator._arc_aware_image_support_details(
        fake,
        np.asarray([], dtype=float),
        family,
        0.0,
        0.0,
        match_details,
    )

    assert details["arc_recovery_status"].tolist() == ["point_recovered", "not_recovered"]
    assert details["preferred_recovery_status"].tolist() == ["point_recovered", "not_recovered"]
    assert details["point_image_residual_arcsec"][0] == pytest.approx(0.2)
    assert np.isnan(details["point_image_residual_arcsec"][1])
    assert np.isnan(details["p_arc"]).all()
    assert details["arc_supported"].tolist() == [False, False]
    assert details["arc_candidate_supported"].tolist() == [False, False]
    assert int(details["arc_supported_image_count"]) == 0
    assert int(details["arc_candidate_supported_image_count"]) == 0


def test_critical_arc_aware_evaluator_uses_sampled_singular_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=1,
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    threshold_spec = cluster_solver._build_critical_arc_singular_threshold_parameter_spec(
        start_index=0,
        lower=0.03,
        upper=0.40,
        prior_median=0.15,
        prior_log_sigma=0.5,
    )
    softness_spec = cluster_solver._build_critical_arc_singular_softness_parameter_spec(
        start_index=1,
        lower=0.005,
        upper=0.20,
        prior_median=0.05,
        prior_log_sigma=0.5,
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.state = SimpleNamespace(parameter_specs=[threshold_spec, softness_spec])
    fake.models_by_effective_z = {2.0: object()}
    fake.arc_recovery_p_arc_threshold = 0.65
    fake.arc_aware_max_arclength_arcsec = 12.0
    fake.arc_aware_curve_step_arcsec = 0.2
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = 0.05
    fake.critical_arc_singular_threshold_param_index = 0
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake.critical_arc_singular_softness_param_index = 1
    fake.sample_likelihood_mode = cluster_solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._ray_shooting_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
    )
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.03], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    captured: dict[str, float] = {}

    def fake_arc_support(*_args, **kwargs):
        captured["singular_threshold"] = float(kwargs["singular_threshold"])
        captured["singular_softness"] = float(kwargs["singular_softness"])
        return {"sentinel": True}

    monkeypatch.setattr(cluster_solver, "_arc_aware_image_support_from_local_linearization", fake_arc_support)

    details = cluster_solver.ClusterJAXEvaluator._arc_aware_image_support_details(
        fake,
        np.asarray([np.log(0.20), np.log(0.08)], dtype=float),
        family,
        0.0,
        0.0,
    )

    assert details == {"sentinel": True}
    assert captured["singular_threshold"] == pytest.approx(0.20)
    assert captured["singular_softness"] == pytest.approx(0.08)


def test_critical_arc_aware_recovery_rows_and_counts_propagate() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        n_images=3,
        sigma_arcsec=0.2,
        image_labels=["1.1", "1.2", "1.3"],
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )
    exact_details = {
        "produced_image_count": 1,
        "recovered_image_count": 1,
        "missing_image_count": 2,
        "extra_image_count": 0,
        "multiplicity_failed": True,
        "multiplicity_failure_reason": "missing_model_images",
        "matched_model_x_arcsec": np.asarray([0.05, np.nan, np.nan], dtype=float),
        "matched_model_y_arcsec": np.asarray([0.0, np.nan, np.nan], dtype=float),
        "recovered_image_mask": np.asarray([True, False, False], dtype=bool),
        "point_image_residual_arcsec": np.asarray([0.05, np.nan, np.nan], dtype=float),
        "arc_candidate_supported": np.asarray([True, True, False], dtype=bool),
        "arc_candidate_image_residual_arcsec": np.asarray([0.03, 0.08, np.nan], dtype=float),
        "preferred_recovery_status": np.asarray(["point_recovered", "arc_supported", "not_recovered"], dtype=object),
        "preferred_image_residual_arcsec": np.asarray([0.05, 0.08, np.nan], dtype=float),
        "arc_recovery_status": np.asarray(["point_recovered", "arc_supported", "not_recovered"], dtype=object),
        "arc_aware_image_residual_arcsec": np.asarray([0.05, 0.08, np.nan], dtype=float),
        "arc_noncritical_direction_residual_arcsec": np.asarray([0.02, 0.08, 0.6], dtype=float),
        "arc_critical_direction_residual_arcsec": np.asarray([0.04, 5.0, 1.0], dtype=float),
        "arc_critical_direction_x": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "arc_critical_direction_y": np.asarray([0.0, 1.0, 1.0], dtype=float),
        "arc_noncritical_direction_x": np.asarray([0.0, 1.0, 1.0], dtype=float),
        "arc_noncritical_direction_y": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "arc_s_min": np.asarray([0.1, 0.04, 0.04], dtype=float),
        "arc_s_max": np.asarray([1.0, 1.2, 1.3], dtype=float),
        "arc_detA": np.asarray([0.1, 0.048, 0.052], dtype=float),
        "arc_prior_probability": np.asarray([0.6, 0.75, 0.75], dtype=float),
        "p_arc": np.asarray([0.55, 0.82, 0.21], dtype=float),
        "arc_log_odds": np.asarray([0.2, 1.5, -1.3], dtype=float),
        "arc_curve_distance_arcsec": np.asarray([0.03, 0.08, np.nan], dtype=float),
        "arc_curve_arclength_arcsec": np.asarray([2.1, 3.2, np.nan], dtype=float),
        "arc_curve_finite": np.asarray([True, True, False], dtype=bool),
        "arc_support_anchor_x_arcsec": np.asarray([0.03, 1.0, np.nan], dtype=float),
        "arc_support_anchor_y_arcsec": np.asarray([0.0, 3.2, np.nan], dtype=float),
        "arc_support_curve_x_arcsec": np.asarray(["[0,0.03]", "[1,1,1]", "[]"], dtype=object),
        "arc_support_curve_y_arcsec": np.asarray(["[0,0]", "[-1,0,1]", "[]"], dtype=object),
        "arc_supported_mask": np.asarray([False, True, False], dtype=bool),
        "arc_support_finite_mask": np.asarray([True, True, True], dtype=bool),
        "failed": True,
    }

    image_rows, _extra_rows, _count_info = image_diagnostics.family_image_recovery_rows(family, exact_details)
    row_by_label = {row["image_label"]: row for row in image_rows}

    assert row_by_label["1.1"]["image_recovery_status"] == "recovered"
    assert row_by_label["1.1"]["point_image_residual_arcsec"] == pytest.approx(0.05)
    assert row_by_label["1.1"]["arc_candidate_supported"] is True
    assert row_by_label["1.1"]["arc_candidate_image_residual_arcsec"] == pytest.approx(0.03)
    assert row_by_label["1.1"]["preferred_recovery_status"] == "point_recovered"
    assert row_by_label["1.1"]["preferred_image_residual_arcsec"] == pytest.approx(0.05)
    assert row_by_label["1.1"]["arc_recovery_status"] == "point_recovered"
    assert row_by_label["1.1"]["arc_supported"] is False
    assert row_by_label["1.1"]["arc_aware_image_residual_arcsec"] == pytest.approx(0.05)
    assert row_by_label["1.2"]["image_recovery_status"] == "not_recovered"
    assert row_by_label["1.2"]["arc_recovery_status"] == "arc_supported"
    assert row_by_label["1.2"]["arc_supported"] is True
    assert row_by_label["1.2"]["arc_support_finite"] is True
    assert row_by_label["1.2"]["arc_critical_direction_y"] == pytest.approx(1.0)
    assert row_by_label["1.2"]["arc_s_max"] == pytest.approx(1.2)
    assert row_by_label["1.2"]["arc_detA"] == pytest.approx(0.048)
    assert "arc_inlier_responsibility" not in row_by_label["1.2"]
    assert "arc_margin_log_weight_minus_point" not in row_by_label["1.2"]
    assert "arc_branch_responsibility" not in row_by_label["1.2"]
    assert row_by_label["1.2"]["p_arc"] == pytest.approx(0.82)
    assert row_by_label["1.2"]["arc_log_odds"] == pytest.approx(1.5)
    assert row_by_label["1.2"]["arc_aware_image_residual_arcsec"] == pytest.approx(0.08)
    assert row_by_label["1.2"]["arc_curve_distance_arcsec"] == pytest.approx(0.08)
    assert row_by_label["1.2"]["arc_curve_arclength_arcsec"] == pytest.approx(3.2)
    assert row_by_label["1.2"]["arc_curve_finite"] is True
    assert row_by_label["1.2"]["arc_support_curve_x_arcsec"] == "[1,1,1]"
    assert row_by_label["1.2"]["arc_support_curve_y_arcsec"] == "[-1,0,1]"
    assert not any(key.startswith("cab_") for key in row_by_label["1.2"])
    assert row_by_label["1.3"]["arc_recovery_status"] == "not_recovered"
    assert np.isnan(row_by_label["1.3"]["arc_aware_image_residual_arcsec"])

    table = image_diagnostics.image_count_recovery_table(SimpleNamespace(family_data=[family]), pd.DataFrame(image_rows))
    assert int(table.loc[0, "arc_aware_recovered_image_count"]) == 2
    assert int(table.loc[0, "arc_aware_missing_image_count"]) == 1
    assert int(table.loc[0, "arc_supported_image_count"]) == 1
    assert int(table.loc[0, "arc_candidate_supported_image_count"]) == 2
    assert table.loc[0, "arc_aware_image_rms_arcsec"] == pytest.approx(math.sqrt((0.05**2 + 0.08**2) / 2.0))
    assert not image_diagnostics.exact_details_hard_failed(
        {
            "multiplicity_failure_reason": "exact_image_prediction_failed",
            "arc_aware_image_residual_arcsec": np.asarray([np.nan, 0.08], dtype=float),
            "failed": True,
        }
    )
    assert image_diagnostics.exact_details_hard_failed(
        {
            "multiplicity_failure_reason": "exact_image_prediction_failed",
            "arc_aware_image_residual_arcsec": np.asarray([np.nan, np.nan], dtype=float),
            "failed": True,
        }
    )


def test_critical_arc_aware_support_survives_exact_solver_failure() -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
        reliability=np.ones(1, dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.validation_cache = {"1": FamilyValidationCache()}
    fake.source_plane_covariance_floor = 0.0
    fake.sample_likelihood_mode = SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    fake.critical_arc_source_position_policy = cluster_solver.CRITICAL_ARC_SOURCE_POSITION_POLICY_SAMPLED
    fake.models_by_effective_z = {2.0: object()}
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._source_sigma_int_numpy = lambda _params: 0.0
    fake._source_position_for_family_numpy = lambda _params, _family_id: (0.0, 0.0)
    fake._exact_source_ray_shooting = (
        lambda _family, _packed_state: (np.asarray([-0.2], dtype=float), np.asarray([0.0], dtype=float))
    )
    fake._ray_shooting_for_components = (
        lambda _z_source, _x, _y, _packed_state: (jnp.asarray([-0.2], dtype=jnp.float64), jnp.asarray([0.0], dtype=jnp.float64))
    )
    fake._lensing_jacobian_for_components = lambda _z_source, _x, _y, _packed_state: (
        jnp.asarray([0.03], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )

    def fail_exact_solver(_family, _packed_state, _source_x, _source_y):
        raise RuntimeError("solver failed")

    fake._solve_exact_images_lenstronomy = fail_exact_solver

    details = cluster_solver.ClusterJAXEvaluator._exact_family_prediction_details(
        fake,
        np.asarray([], dtype=float),
        family,
    )

    assert details["failed"] is True
    assert details["multiplicity_failure_reason"] == "exact_image_prediction_failed"
    assert details["arc_recovery_status"].tolist() == ["arc_supported"]
    assert details["arc_supported_mask"].tolist() == [True]
    assert np.isfinite(details["arc_aware_image_residual_arcsec"][0])
    assert not image_diagnostics.exact_details_hard_failed(details)


def test_exact_recovery_arc_support_falls_back_when_jacobian_unavailable() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.2,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
        reliability=np.ones(1, dtype=float),
    )
    fake = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    fake.validation_cache = {"1": FamilyValidationCache()}
    fake.source_plane_covariance_floor = 0.0
    fake.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    fake.models_by_effective_z = {2.0: object()}
    fake.critical_arc_lm_trust_radius_arcsec = cluster_solver.DEFAULT_CRITICAL_ARC_LM_TRUST_RADIUS_ARCSEC
    fake.critical_arc_lm_damping_relative = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_RELATIVE
    fake.critical_arc_lm_damping_absolute = cluster_solver.DEFAULT_CRITICAL_ARC_LM_DAMPING_ABSOLUTE
    fake.critical_arc_base_prob = cluster_solver.DEFAULT_CRITICAL_ARC_BASE_PROB
    fake.critical_arc_max_prob = cluster_solver.DEFAULT_CRITICAL_ARC_MAX_PROB
    fake.critical_arc_singular_threshold = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_THRESHOLD
    fake.critical_arc_singular_softness = cluster_solver.DEFAULT_CRITICAL_ARC_SINGULAR_SOFTNESS
    fake._build_packed_lens_state = lambda _params, _z_source: {}
    fake._source_sigma_int_numpy = lambda _params: 0.0
    fake._source_position_for_family_numpy = lambda _params, _family_id: None
    fake._exact_source_ray_shooting = (
        lambda _family, _packed_state: (np.asarray([-0.2], dtype=float), np.asarray([0.0], dtype=float))
    )
    fake._ray_shooting_for_components = (
        lambda _z_source, _x, _y, _packed_state: (
            jnp.asarray([-0.2], dtype=jnp.float64),
            jnp.asarray([0.0], dtype=jnp.float64),
        )
    )

    def fail_jacobian(_z_source, _x, _y, _packed_state):
        raise RuntimeError("jacobian unavailable")

    def fail_exact_solver(_family, _packed_state, _source_x, _source_y):
        raise RuntimeError("solver failed")

    fake._lensing_jacobian_for_components = fail_jacobian
    fake._solve_exact_images_lenstronomy = fail_exact_solver

    details = cluster_solver.ClusterJAXEvaluator._exact_family_prediction_details(
        fake,
        np.asarray([], dtype=float),
        family,
    )

    assert details["failed"] is True
    assert details["arc_recovery_status"].tolist() == ["not_recovered"]
    assert details["arc_supported_mask"].tolist() == [False]
    assert np.isnan(details["arc_s_min"][0])
    assert np.isnan(details["arc_noncritical_direction_residual_arcsec"][0])
    assert image_diagnostics.exact_details_hard_failed(details)


def test_flat_wcdm_jax_factor_is_differentiable() -> None:
    z_lens = 0.3734
    z_source = 3.0
    h0 = 70.0

    def factor(om0: jnp.ndarray) -> jnp.ndarray:
        return dpie_sigma0_factor(z_lens, z_source, h0, om0, -1.0)

    grad_value = jax.grad(factor)(jnp.asarray(0.3, dtype=jnp.float64))

    assert np.isfinite(float(grad_value))


def test_cluster_solver_original_profile_maps_dpie_to_dpie_nie() -> None:
    _specs, _assignments, lens_model_list = _build_parameter_specs(
        [
            {"id": "halo", "profil": 81, "priors": {}},
            {"id": "shear", "profil": 14, "priors": {}},
        ],
    )

    assert lens_model_list == ["DPIE_NIE", "SHEAR"]


def test_decode_mode9_prior_accepts_finite_and_lower_only_truncation() -> None:
    finite = cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0, 1500.0], "halo.v_disp")
    lower_only = cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0], "halo.v_disp")

    assert finite == {
        "prior_kind": "truncated_normal",
        "lower": 300.0,
        "upper": 1500.0,
        "step": 100.0,
        "mean": 900.0,
        "std": 100.0,
    }
    assert lower_only["prior_kind"] == "truncated_normal"
    assert lower_only["lower"] == pytest.approx(300.0)
    assert np.isinf(lower_only["upper"])
    assert lower_only["mean"] == pytest.approx(900.0)
    assert lower_only["std"] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="std must be positive"):
        cluster_solver._decode_parameter_prior([9, 900.0, 0.0, 300.0], "halo.v_disp")
    with pytest.raises(ValueError, match="lower bound must be less"):
        cluster_solver._decode_parameter_prior([9, 900.0, 100.0, 300.0, 300.0], "halo.v_disp")


def test_dpie_elliptic_shape_priors_emit_direct_e1e2_specs() -> None:
    specs, assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {
                    "ellipticite": [1, 0.0, 0.8, 0.02],
                    "angle_pos": [1, -180.0, 180.0, 0.5],
                },
            },
        ],
    )

    fields = [spec.field for spec in specs]
    sample_names = [spec.sample_name for spec in specs]

    assert fields == ["e1", "e2"]
    assert sample_names == ["halo_e1", "halo_e2"]
    assert assignments == [[("e1", 0), ("e2", 1)]]
    assert all(spec.prior_kind == "uniform" for spec in specs)
    assert all(spec.transform_kind == "identity" for spec in specs)
    np.testing.assert_allclose([specs[0].lower, specs[0].upper], [-0.5, 0.5], atol=1.0e-12)
    np.testing.assert_allclose([specs[1].lower, specs[1].upper], [-0.5, 0.5], atol=1.0e-12)


def test_dpie_elliptic_shape_partial_prior_raises() -> None:
    with pytest.raises(ValueError, match="partial shape priors"):
        _build_parameter_specs(
            [
                {
                    "id": "halo",
                    "profil": cluster_solver.DP_IE_PROFILE,
                    "priors": {"ellipticite": [1, 0.0, 0.8, 0.02]},
                },
            ],
        )


def test_real_inputs_emit_direct_dpie_shape_specs_without_ff_sims_shear() -> None:
    a2744_path = Path("data/Bergamini/A2744_Bergamini23/Bergamini23_A2744_Normal.par")
    hera_path = Path("data/ff_sims/hera/hera_lenscluster.par")
    _parsed, _potentials_df, _images_df, _arcs_df, a2744_potentials = load_best_par(a2744_path)
    _parsed, _potentials_df, _images_df, _arcs_df, hera_potentials = load_best_par(hera_path)

    a2744_specs, _a2744_assignments, _a2744_lens = _build_parameter_specs(a2744_potentials)
    hera_specs, _hera_assignments, _hera_lens = _build_parameter_specs(hera_potentials)

    a2744_shape_fields = [spec.field for spec in a2744_specs if spec.potential_id in {str(i) for i in range(1, 9)}]
    hera_shape_specs = [
        spec
        for spec in hera_specs
        if spec.potential_id in {"O1", "O2"} and spec.field in {"e1", "e2", "ellipticite", "angle_pos"}
    ]
    hera_shear_specs = [spec for spec in hera_specs if spec.potential_id == "S1"]

    assert a2744_shape_fields.count("e1") == 8
    assert a2744_shape_fields.count("e2") == 8
    assert "ellipticite" not in a2744_shape_fields
    assert "angle_pos" not in a2744_shape_fields
    assert [spec.sample_name for spec in hera_shape_specs] == ["O1_e1", "O1_e2", "O2_e1", "O2_e2"]
    assert hera_shear_specs == []


def test_shear_gamma_angle_priors_emit_direct_gamma1gamma2_specs() -> None:
    specs, assignments, lens_model_list = _build_parameter_specs(
        [
            {
                "id": "S1",
                "profil": cluster_solver.SHEAR_PROFILE,
                "priors": {
                    "gamma": [1, 0.0, 0.3, 0.005],
                    "angle_pos": [1, -180.0, 180.0, 0.5],
                },
            },
        ],
    )

    assert lens_model_list == ["SHEAR"]
    assert [spec.field for spec in specs] == ["gamma1", "gamma2"]
    assert [spec.sample_name for spec in specs] == ["S1_gamma1", "S1_gamma2"]
    assert assignments == [[("gamma1", 0), ("gamma2", 1)]]
    assert all(spec.prior_kind == "uniform" for spec in specs)
    assert all(spec.transform_kind == "identity" for spec in specs)
    np.testing.assert_allclose([specs[0].lower, specs[0].upper], [-0.3, 0.3], atol=1.0e-12)
    np.testing.assert_allclose([specs[1].lower, specs[1].upper], [-0.3, 0.3], atol=1.0e-12)


def test_shear_partial_or_nonuniform_polar_prior_raises() -> None:
    with pytest.raises(ValueError, match="partial shear priors"):
        _build_parameter_specs(
            [
                {
                    "id": "S1",
                    "profil": cluster_solver.SHEAR_PROFILE,
                    "priors": {"gamma": [1, 0.0, 0.3, 0.005]},
                },
            ],
        )

    with pytest.raises(ValueError, match="requires uniform gamma and angle_pos priors"):
        _build_parameter_specs(
            [
                {
                    "id": "S1",
                    "profil": cluster_solver.SHEAR_PROFILE,
                    "priors": {
                        "gamma": [3, 0.04, 0.01],
                        "angle_pos": [1, -180.0, 180.0, 0.5],
                    },
                },
            ],
        )


def test_non_scaling_dpie_cut_prior_samples_excess_above_reference_effective_core() -> None:
    specs, assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "core_radius_kpc": 3.0,
                "priors": {
                    "cut_radius_kpc": [9, 20.0, 5.0, 1.0, 60.0],
                },
            },
        ],
        softening_length_kpc=4.0,
    )

    spec = specs[0]
    expected_reference_core = 5.0
    expected_mean, expected_std = cluster_solver._positive_lognormal_parameters(
        15.0,
        5.0,
        floor=cluster_solver.SAFE_RADIUS_MARGIN_KPC,
    )

    assert assignments == [[("cut_radius_kpc", 0)]]
    assert spec.transform_kind == "log_positive"
    assert spec.transform_offset == pytest.approx(expected_reference_core)
    assert spec.physical_mean == pytest.approx(20.0)
    assert spec.physical_upper == pytest.approx(60.0)
    assert spec.mean == pytest.approx(expected_mean)
    assert spec.std == pytest.approx(expected_std)
    assert spec.lower == pytest.approx(np.log(cluster_solver.SAFE_RADIUS_MARGIN_KPC))
    assert spec.upper == pytest.approx(np.log(60.0 - expected_reference_core))


def test_non_scaling_dpie_cut_prior_rejects_upper_below_reference_effective_core() -> None:
    with pytest.raises(ValueError, match="upper bound .* must exceed reference effective core"):
        _build_parameter_specs(
            [
                {
                    "id": "halo",
                    "profil": cluster_solver.DP_IE_PROFILE,
                    "core_radius_kpc": 3.0,
                    "priors": {
                        "cut_radius_kpc": [9, 4.0, 1.0, 1.0, 4.5],
                    },
                },
            ],
            softening_length_kpc=4.0,
        )


def test_dynamic_cut_initialization_converts_reported_total_cut_to_excess() -> None:
    spec = ParameterSpec(
        name="halo.cut_radius_kpc",
        sample_name="halo_cut_radius_kpc",
        potential_id="halo",
        profile_type=cluster_solver.DP_IE_PROFILE,
        field="cut_radius_kpc",
        prior_kind="truncated_normal",
        lower=np.log(cluster_solver.SAFE_RADIUS_MARGIN_KPC),
        upper=np.log(100.0),
        step=1.0,
        mean=np.log(10.0),
        std=0.5,
        transform_kind="log_positive",
        transform_offset=5.0,
    )

    latent = cluster_solver._initial_latent_value_from_physical(17.0, spec)

    assert latent == pytest.approx(np.log(12.0))


def test_dpie_v_disp_normal_prior_uses_log_positive_transform() -> None:
    specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [3, 900.0, 600.0, 0.1]},
            },
        ],
    )

    spec = specs[0]
    expected_mean, expected_std = cluster_solver._positive_lognormal_parameters(
        900.0,
        600.0,
        floor=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
    )
    latent = np.asarray(
        [[spec.lower], [np.log(1.0)], [float(spec.mean)]],
        dtype=float,
    )
    physical = cluster_solver._convert_sample_matrix_to_physical(
        latent,
        specs,
    )
    prior = cluster_solver._distribution_for_spec(spec)

    assert spec.name == "halo.v_disp"
    assert spec.prior_kind == "truncated_normal"
    assert spec.transform_kind == "log_positive"
    assert spec.lower == pytest.approx(np.log(cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S))
    assert np.isinf(spec.upper)
    assert spec.mean == pytest.approx(expected_mean)
    assert spec.std == pytest.approx(expected_std)
    assert spec.physical_lower == pytest.approx(cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S)
    assert spec.physical_mean == pytest.approx(900.0)
    assert spec.physical_std == pytest.approx(600.0)
    np.testing.assert_allclose(
        physical[:, 0],
        [cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S, 1.0, math.exp(expected_mean)],
    )
    assert np.isfinite(float(prior.log_prob(jnp.asarray(np.log(1.0)))))
    with pytest.warns(UserWarning, match="Out-of-support values"):
        below_floor_log_prob = prior.log_prob(jnp.asarray(spec.lower - 1.0))
    assert float(below_floor_log_prob) == -np.inf


def test_dpie_v_disp_mode9_prior_uses_log_positive_truncation_bounds() -> None:
    specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [9, 900.0, 100.0, 300.0, 1500.0]},
            },
        ],
    )

    spec = specs[0]
    expected_mean, expected_std = cluster_solver._positive_lognormal_parameters(
        900.0,
        100.0,
        floor=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
    )

    assert spec.name == "halo.v_disp"
    assert spec.prior_kind == "truncated_normal"
    assert spec.transform_kind == "log_positive"
    assert spec.lower == pytest.approx(np.log(300.0))
    assert spec.upper == pytest.approx(np.log(1500.0))
    assert spec.mean == pytest.approx(expected_mean)
    assert spec.std == pytest.approx(expected_std)
    assert spec.physical_lower == pytest.approx(300.0)
    assert spec.physical_upper == pytest.approx(1500.0)
    assert spec.physical_mean == pytest.approx(900.0)
    assert spec.physical_std == pytest.approx(100.0)


def test_dpie_v_disp_uniform_prior_uses_log_positive_bounds() -> None:
    specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [1, 450.0, 1200.0, 10.0]},
            },
        ],
    )

    spec = specs[0]
    physical_values = np.asarray([450.0, 900.0, 1200.0], dtype=float)
    latent = cluster_solver._convert_theta_to_latent(physical_values, specs * 3).reshape(-1)

    assert spec.name == "halo.v_disp"
    assert spec.prior_kind == "uniform"
    assert spec.transform_kind == "log_positive"
    assert spec.lower == pytest.approx(np.log(450.0))
    assert spec.upper == pytest.approx(np.log(1200.0))
    assert spec.physical_lower == pytest.approx(450.0)
    assert spec.physical_upper == pytest.approx(1200.0)
    assert np.log(450.0) <= float(latent[0]) <= np.log(1200.0)
    assert np.log(450.0) <= float(latent[1]) <= np.log(1200.0)
    assert np.log(450.0) <= float(latent[2]) <= np.log(1200.0)
    np.testing.assert_allclose(latent, np.log(physical_values))
    np.testing.assert_allclose(cluster_solver._convert_theta_to_physical(latent, specs * 3), physical_values)


def test_potfile_mode9_scaling_priors_transform_bounds_to_latent_log_space() -> None:
    specs, param_indices, _lens_model_list = cluster_solver._build_scaling_parameter_specs(
        [
            {
                "id": "members",
                "type": cluster_solver.DP_IE_PROFILE,
                "catalog_df": pd.DataFrame({"id": ["member"]}),
                "core_arcsec": 0.1,
                "sigma": [9, 245.0, 55.0, 190.0, 300.0],
                "cut": [9, 5.25, 4.75, 0.5, 10.0],
            }
        ],
        kpc_per_arcsec=10.0,
    )
    sigma_spec = specs[param_indices[0]["sigma"]]
    cut_spec = specs[param_indices[0]["cutkpc"]]
    core_spec = specs[param_indices[0]["corekpc"]]
    # Configured mode-9 entries are honored: sigma in km/s, cut in arcsec * kpc_per_arcsec.
    expected_sigma_mean, expected_sigma_std = cluster_solver._positive_lognormal_parameters(
        245.0, 55.0, floor=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S
    )
    expected_cut_mean, expected_cut_std = cluster_solver._positive_lognormal_parameters(
        52.5, 47.5, floor=cluster_solver.SAFE_RADIUS_MARGIN_KPC
    )

    assert cluster_solver.SOLVER_POTFILE_POSITIVE_PRIOR_COORDINATE == "log_positive_latent"
    assert sigma_spec.prior_kind == "truncated_normal"
    assert sigma_spec.transform_kind == "log_positive"
    assert sigma_spec.lower == pytest.approx(np.log(190.0))
    assert sigma_spec.upper == pytest.approx(np.log(300.0))
    assert sigma_spec.mean == pytest.approx(expected_sigma_mean)
    assert sigma_spec.std == pytest.approx(expected_sigma_std)
    assert sigma_spec.physical_lower == pytest.approx(190.0)
    assert sigma_spec.physical_upper == pytest.approx(300.0)
    assert sigma_spec.physical_mean == pytest.approx(245.0)
    assert sigma_spec.physical_std == pytest.approx(55.0)

    assert cut_spec.prior_kind == "truncated_normal"
    assert cut_spec.transform_kind == "log_positive"
    assert cut_spec.transform_offset == pytest.approx(0.0)
    assert cut_spec.lower == pytest.approx(np.log(5.0))
    assert cut_spec.upper == pytest.approx(np.log(100.0))
    assert cut_spec.mean == pytest.approx(expected_cut_mean)
    assert cut_spec.std == pytest.approx(expected_cut_std)
    assert cut_spec.physical_lower == pytest.approx(5.0)
    assert cut_spec.physical_upper == pytest.approx(100.0)
    assert cut_spec.physical_mean == pytest.approx(52.5)
    assert cut_spec.physical_std == pytest.approx(47.5)

    assert core_spec.prior_kind == "truncated_normal"
    assert core_spec.transform_kind == "log_positive"
    assert core_spec.lower == pytest.approx(np.log(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_LOWER_KPC))
    assert core_spec.upper == pytest.approx(np.log(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_UPPER_KPC))
    assert core_spec.mean == pytest.approx(np.log(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_MEDIAN_KPC))
    assert core_spec.std == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_LOG_SIGMA)
    assert core_spec.physical_lower == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_LOWER_KPC)
    assert core_spec.physical_upper == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_UPPER_KPC)
    assert core_spec.physical_mean == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CORE_REF_MEDIAN_KPC)
    assert core_spec.physical_std is None


def _scaling_prior_potfile(**entries: object) -> dict[str, object]:
    return {
        "id": "members",
        "type": cluster_solver.DP_IE_PROFILE,
        "catalog_df": pd.DataFrame({"id": ["member"]}),
        "core_arcsec": 0.1,
        **entries,
    }


def test_scaling_specs_honor_configured_normal_sigma_and_uniform_cutkpc_priors() -> None:
    # The mock-validation shapes: sigma ~ Normal(245, 35) km/s, cutkpc ~ uniform in kpc.
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [_scaling_prior_potfile(sigma=[3, 245.0, 35.0], cutkpc=[1, 41.3, 413.0, 37.2])],
        kpc_per_arcsec=5.0,
    )
    sigma_spec = specs[param_indices[0]["sigma"]]
    cut_spec = specs[param_indices[0]["cutkpc"]]

    expected_mean, expected_std = cluster_solver._positive_lognormal_parameters(
        245.0, 35.0, floor=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S
    )
    # Configured normal priors are truncated at the baked default physical bounds so
    # early stage-0 SVI cannot collapse the members far below the prior (jul6_0045).
    assert sigma_spec.prior_kind == "truncated_normal"
    assert sigma_spec.transform_kind == "log_positive"
    assert sigma_spec.lower == pytest.approx(np.log(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_LOWER))
    assert sigma_spec.upper == pytest.approx(np.log(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_UPPER))
    assert sigma_spec.mean == pytest.approx(expected_mean)
    assert sigma_spec.std == pytest.approx(expected_std)
    assert sigma_spec.physical_mean == pytest.approx(245.0)
    assert sigma_spec.physical_std == pytest.approx(35.0)
    assert sigma_spec.physical_lower == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_LOWER)
    assert sigma_spec.physical_upper == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_UPPER)

    # A uniform entry on a log-positive field samples log-uniform in the physical value.
    assert cut_spec.prior_kind == "uniform"
    assert cut_spec.transform_kind == "log_positive"
    assert cut_spec.lower == pytest.approx(np.log(41.3))
    assert cut_spec.upper == pytest.approx(np.log(413.0))
    assert cut_spec.mean is None and cut_spec.std is None
    assert cut_spec.physical_lower == pytest.approx(41.3)
    assert cut_spec.physical_upper == pytest.approx(413.0)
    assert cut_spec.step == pytest.approx(37.2)


def test_scaling_specs_kind3_cutkpc_prior_gets_default_bounds() -> None:
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [_scaling_prior_potfile(cutkpc=[3, 187.0, 50.0])],
        kpc_per_arcsec=5.0,
    )
    cut_spec = specs[param_indices[0]["cutkpc"]]
    assert cut_spec.prior_kind == "truncated_normal"
    assert cut_spec.physical_mean == pytest.approx(187.0)
    assert cut_spec.physical_std == pytest.approx(50.0)
    assert cut_spec.physical_lower == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CUT_REF_LOWER_KPC)
    assert cut_spec.physical_upper == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CUT_REF_UPPER_KPC)


def test_bounded_normal_prior_serializes_to_truncated_normal_limit_entry() -> None:
    entry = cluster_solver._prior_config_to_limit_entry(
        PriorConfig("normal", mean=15.0, std=3.75, lower=3.75, upper=60.0)
    )

    assert entry == pytest.approx([9.0, 15.0, 3.75, 3.75, 60.0])


def test_scaling_specs_scale_cut_arcsec_prior_to_kpc() -> None:
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [_scaling_prior_potfile(cut=[1, 8.0, 80.0, 1.0])],
        kpc_per_arcsec=5.0,
    )
    cut_spec = specs[param_indices[0]["cutkpc"]]
    assert cut_spec.prior_kind == "uniform"
    assert cut_spec.physical_lower == pytest.approx(40.0)
    assert cut_spec.physical_upper == pytest.approx(400.0)


def _simple_spec(name: str, family: str, *, prior_kind: str = "normal", mean: float = 0.0,
                 std: float = 1.0, lower: float = -10.0, upper: float = 10.0) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        sample_name=name.replace(".", "_"),
        potential_id="pot",
        profile_type=cluster_solver.DP_IE_PROFILE,
        field=name.split(".")[-1],
        prior_kind=prior_kind,
        lower=lower,
        upper=upper,
        step=0.1,
        mean=mean,
        std=std,
        component_family=family,
    )


def test_stage0_member_frozen_site_values_freezes_member_families() -> None:
    specs = [
        _simple_spec("halo.v_disp", "large"),
        _simple_spec("potfile.sigma", "scaling"),
        _simple_spec("potfile.sigma_log_scatter", "scaling_scatter"),
        _simple_spec("potfile.member001.delta", "independent_scaling"),
        _simple_spec("source.1.beta_x", "source_position"),
    ]
    theta = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    frozen = cluster_solver._stage0_member_frozen_site_values(specs, theta)
    assert set(frozen) == {"potfile_sigma", "potfile_sigma_log_scatter", "potfile_member001_delta"}
    assert float(frozen["potfile_sigma"]) == pytest.approx(2.0)
    assert float(frozen["potfile_member001_delta"]) == pytest.approx(4.0)
    # No member-family specs -> nothing frozen (staging is skipped by the caller).
    assert cluster_solver._stage0_member_frozen_site_values(specs[:1], theta[:1]) == {}


def test_multi_start_best_index_picks_lowest_tail_mean() -> None:
    losses = [
        np.linspace(100.0, 50.0, 100),   # tail mean ~50
        np.linspace(100.0, 10.0, 100),   # tail mean ~10 -> winner
        np.full(100, 30.0),
    ]
    best, scores = cluster_solver._multi_start_best_index(losses, tail_fraction=0.1)
    assert best == 1
    assert scores[2] == pytest.approx(30.0)
    # A start that diverged to NaN cannot win.
    losses[1] = np.full(100, np.nan)
    best, _ = cluster_solver._multi_start_best_index(losses, tail_fraction=0.1)
    assert best == 2
    with pytest.raises(ValueError):
        cluster_solver._multi_start_best_index([np.full(10, np.nan)], tail_fraction=0.1)


def test_stage0_collapse_sentinel_warns_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [
        _simple_spec("potfile.sigma", "scaling", prior_kind="truncated_normal", mean=5.5, std=0.14, lower=3.0, upper=6.2),
        _simple_spec("potfile.cutkpc", "scaling", prior_kind="uniform", lower=3.7, upper=6.0),
        _simple_spec("halo.v_disp", "large", prior_kind="truncated_normal", mean=6.6, std=0.1, lower=6.0, upper=7.1),
    ]
    state = SimpleNamespace(parameter_specs=specs)
    messages: list[str] = []
    monkeypatch.setattr(cluster_solver, "_log", lambda _args, message, **_kw: messages.append(str(message)))
    args = argparse.Namespace(quiet=False)

    # Collapsed: sigma ~20 prior-sigma low and cutkpc pinned to the lower prior edge.
    payload = cluster_solver._warn_if_stage0_scaling_state_collapsed(
        args, state, np.asarray([2.7, 3.70001, 6.6])
    )
    assert payload["stage0_scaling_state_suspect"] is True
    assert payload["stage0_scaling_state_prior_z"]["potfile.sigma"] == pytest.approx((2.7 - 5.5) / 0.14)
    assert any("STAGE-0 SCALING STATE SUSPECT" in message for message in messages)
    assert any("potfile.sigma" in message for message in messages)

    # Healthy state: silent, flag off; the non-scaling halo spec is never checked.
    messages.clear()
    payload = cluster_solver._warn_if_stage0_scaling_state_collapsed(
        args, state, np.asarray([5.4, 4.9, 0.0])
    )
    assert payload["stage0_scaling_state_suspect"] is False
    assert messages == []


def test_scaling_specs_fall_back_to_defaults_for_fixed_or_absent_priors() -> None:
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        [_scaling_prior_potfile(sigma=[0, 245.0])],
        kpc_per_arcsec=5.0,
    )
    sigma_spec = specs[param_indices[0]["sigma"]]
    cut_spec = specs[param_indices[0]["cutkpc"]]
    assert sigma_spec.prior_kind == "truncated_normal"
    assert sigma_spec.physical_mean == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_MEAN)
    assert sigma_spec.physical_std == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_SIGMA_REF_STD)
    assert cut_spec.prior_kind == "truncated_normal"
    assert cut_spec.physical_mean == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CUT_REF_MEAN_KPC)
    assert cut_spec.physical_std == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_CUT_REF_STD_KPC)


@pytest.mark.parametrize(
    "entries",
    [
        {"sigma": [1, 50.0]},  # uniform without upper
        {"sigma": [1, 300.0, 200.0]},  # inverted bounds
        {"sigma": [3, -245.0, 35.0]},  # nonpositive mean on a positive scale field
        {"cutkpc": [9, 187.0, 50.0, 400.0, 40.0]},  # truncated normal with upper < lower
        {"sigma": [7, 1.0, 2.0]},  # unknown limit kind
    ],
)
def test_scaling_specs_reject_malformed_prior_entries(entries: dict) -> None:
    with pytest.raises(ValueError):
        cluster_solver._build_scaling_parameter_specs(
            [_scaling_prior_potfile(**entries)],
            kpc_per_arcsec=5.0,
        )


def test_log_softening_length_parameter_spec_is_disabled_at_zero() -> None:
    spec = cluster_solver._build_log_softening_length_parameter_spec(
        start_index=0,
        softening_length_kpc=0.0,
        prior_log_sigma=0.15,
    )

    assert spec is None


def test_log_softening_length_parameter_spec_samples_log_coordinate() -> None:
    spec = cluster_solver._build_log_softening_length_parameter_spec(
        start_index=0,
        softening_length_kpc=4.0,
        prior_log_sigma=0.2,
    )

    assert spec is not None
    assert spec.sample_name == cluster_solver.LOG_SOFTENING_LENGTH_SAMPLE_NAME
    assert spec.field == cluster_solver.LOG_SOFTENING_LENGTH_SAMPLE_NAME
    assert spec.component_family == cluster_solver.SOFTENING_LENGTH_COMPONENT_FAMILY
    assert spec.prior_kind == "normal"
    assert spec.transform_kind == "identity"
    assert spec.mean == pytest.approx(np.log(4.0))
    assert spec.std == pytest.approx(0.2)
    assert spec.physical_mean == pytest.approx(np.log(4.0))
    assert spec.physical_std == pytest.approx(0.2)


def test_scaling_parameter_specs_use_direct_exponent_mode_only() -> None:
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame({"id": ["member"]}),
            "core_arcsec": 0.1,
            "sigma": [9, 245.0, 55.0, 190.0, 300.0],
            "cut": [9, 5.25, 4.75, 0.5, 10.0],
        }
    ]
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(potfiles, kpc_per_arcsec=10.0)

    assert {"alpha_sigma", "gamma_ml"} <= {spec.field for spec in specs}
    assert {"vdslope", "slope", "beta_radius"}.isdisjoint({spec.field for spec in specs})

    alpha_spec = specs[param_indices[0]["alpha_sigma"]]
    gamma_spec = specs[param_indices[0]["gamma_ml"]]
    assert alpha_spec.mean == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_ALPHA_SIGMA_MEAN)
    assert alpha_spec.std == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_ALPHA_SIGMA_STD)
    assert alpha_spec.lower == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_ALPHA_SIGMA_LOWER)
    assert alpha_spec.upper == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_ALPHA_SIGMA_UPPER)
    assert gamma_spec.mean == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_GAMMA_ML_MEAN)
    assert gamma_spec.std == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_GAMMA_ML_STD)
    assert gamma_spec.lower == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_GAMMA_ML_LOWER)
    assert gamma_spec.upper == pytest.approx(cluster_solver.DEFAULT_SOLVER_POTFILE_GAMMA_ML_UPPER)


def test_scaling_parameter_specs_use_configured_alpha_gamma_priors() -> None:
    potfiles = [
        {
            "id": "members",
            "type": cluster_solver.DP_IE_PROFILE,
            "catalog_df": pd.DataFrame({"id": ["member"]}),
            "core_arcsec": 0.1,
            "sigma": [9, 245.0, 55.0, 190.0, 300.0],
            "cut": [9, 5.25, 4.75, 0.5, 10.0],
        }
    ]
    specs, param_indices, _ = cluster_solver._build_scaling_parameter_specs(
        potfiles,
        kpc_per_arcsec=10.0,
        alpha_sigma_prior_mean=0.24,
        alpha_sigma_prior_std=0.08,
        alpha_sigma_prior_lower=0.05,
        alpha_sigma_prior_upper=0.50,
        gamma_ml_prior_mean=-0.05,
        gamma_ml_prior_std=0.25,
        gamma_ml_prior_lower=-0.80,
        gamma_ml_prior_upper=0.80,
    )

    alpha_spec = specs[param_indices[0]["alpha_sigma"]]
    gamma_spec = specs[param_indices[0]["gamma_ml"]]
    assert alpha_spec.mean == pytest.approx(0.24)
    assert alpha_spec.std == pytest.approx(0.08)
    assert alpha_spec.lower == pytest.approx(0.05)
    assert alpha_spec.upper == pytest.approx(0.50)
    assert gamma_spec.mean == pytest.approx(-0.05)
    assert gamma_spec.std == pytest.approx(0.25)
    assert gamma_spec.lower == pytest.approx(-0.80)
    assert gamma_spec.upper == pytest.approx(0.80)


def _log_uniform_image_plane_scatter_spec(floor_arcsec: float = 0.1, upper_arcsec: float = 0.5) -> ParameterSpec:
    return cluster_solver._build_image_scatter_parameter_spec(
        0,
        floor_arcsec=floor_arcsec,
        upper_arcsec=upper_arcsec,
        prior=cluster_solver.IMAGE_PLANE_SCATTER_PRIOR_LOG_UNIFORM,
    )


def test_initial_latent_value_from_physical_clips_log_uniform_image_plane_scatter_floor() -> None:
    spec = _log_uniform_image_plane_scatter_spec()

    clipped_init = cluster_solver._initial_latent_value_from_physical(0.1, spec)

    assert spec.lower < clipped_init < spec.upper
    assert clipped_init == pytest.approx(
        spec.lower + cluster_solver.DEFAULT_NUTS_INIT_BOUNDARY_FRAC * (spec.upper - spec.lower)
    )
    assert math.exp(clipped_init) == pytest.approx(0.10327, rel=1.0e-4)


def test_svi_initial_value_dict_clips_exact_bound_image_plane_scatter_init() -> None:
    spec = _log_uniform_image_plane_scatter_spec()

    payload = cluster_solver._svi_initial_value_dict([spec], {"image_sigma_int": spec.lower})
    assert payload is not None

    unconstrained = cluster_solver.unconstrain_fn(
        cluster_solver._sample_site_model([spec]),
        model_args=(),
        model_kwargs={},
        params=payload,
    )

    clipped_value = float(np.asarray(payload["image_sigma_int"], dtype=float))
    assert spec.lower < clipped_value < spec.upper
    assert np.all(np.isfinite(np.asarray(unconstrained["image_sigma_int"], dtype=float)))


def test_stage2_large_priors_convert_log_vdisp_summary_values_to_latent_moments() -> None:
    large_specs, _assignments, _lens_model_list = _build_parameter_specs(
        [
            {
                "id": "halo",
                "profil": cluster_solver.DP_IE_PROFILE,
                "priors": {"v_disp": [3, 900.0, 600.0, 0.1]},
            },
        ],
    )
    summary = Stage1PriorSummary(
        map_values={"halo_v_disp": 920.0},
        means={"halo_v_disp": 900.0},
        stds={"halo_v_disp": 90.0},
    )

    stage2_specs = cluster_solver._build_stage2_large_parameter_specs(large_specs, summary)
    stage2_spec = stage2_specs[0]
    expected_mean, expected_std = cluster_solver._positive_lognormal_parameters(
        900.0,
        90.0,
        floor=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
    )
    physical_center = cluster_solver._convert_theta_to_physical(np.asarray([stage2_spec.mean]), stage2_specs)[0]

    assert stage2_spec.prior_kind == "truncated_normal"
    assert stage2_spec.transform_kind == "log_positive"
    assert stage2_spec.lower == pytest.approx(np.log(cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S))
    assert stage2_spec.mean == pytest.approx(expected_mean)
    assert stage2_spec.std == pytest.approx(expected_std)
    assert stage2_spec.physical_mean == pytest.approx(900.0)
    assert stage2_spec.physical_std == pytest.approx(90.0)
    assert physical_center == pytest.approx(math.exp(expected_mean))


def test_truncated_vdisp_default_theta_and_init_are_clipped_to_support() -> None:
    spec = ParameterSpec(
        "halo.v_disp",
        "halo_v_disp",
        "halo",
        cluster_solver.DP_IE_PROFILE,
        "v_disp",
        "truncated_normal",
        cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
        np.inf,
        0.1,
        mean=-10.0,
        std=1.0,
        transform_kind="identity",
        physical_lower=cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S,
        physical_mean=-10.0,
        physical_std=1.0,
    )

    default_theta = cluster_solver._default_theta([spec])
    clipped_init = cluster_solver._initial_latent_value_from_physical(-5.0, spec)

    assert default_theta[0] > cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S
    assert clipped_init > cluster_solver.VDISP_TRUNCATION_FLOOR_KM_S


def test_packed_lens_validity_allows_negative_finite_vdisp_and_rejects_nonfinite() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    details = PackedLensDetails(
        is_dpie=jnp.asarray([True, True]),
        is_shear=jnp.asarray([False, False]),
        is_scaling=jnp.asarray([False, False]),
        sigma0=jnp.asarray([1.0, 1.0]),
        ra_raw=jnp.asarray([1.0, 1.0]),
        rs_raw=jnp.asarray([10.0, 10.0]),
        core_radius_kpc=jnp.asarray([1.0, 1.0]),
        core_radius_effective_kpc=jnp.asarray([1.0, 1.0]),
        softening_length_kpc=jnp.asarray(0.0),
        log_softening_length_kpc=jnp.asarray(float("-inf")),
        v_disp=jnp.asarray([500.0, -10.0]),
        alpha_sigma=jnp.asarray([0.25, 0.25]),
        beta_radius=jnp.asarray([0.5, 0.5]),
        gamma_ml=jnp.asarray([0.0, 0.0]),
        x_center=jnp.asarray([0.0, 0.0]),
        y_center=jnp.asarray([0.0, 0.0]),
        gamma1=jnp.asarray([0.0, 0.0]),
        gamma2=jnp.asarray([0.0, 0.0]),
        e1=jnp.asarray([0.0, 0.0]),
        e2=jnp.asarray([0.0, 0.0]),
        factor_array=jnp.asarray(1.0),
        independent_branch_weight=jnp.asarray([1.0, 1.0]),
    )

    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)
    reason_index = cluster_solver.INVALID_STATE_REASON_NAMES.index("nonpositive_vdisp")

    assert bool(validity.is_valid) is True
    assert bool(np.asarray(validity.reason_flags)[reason_index]) is False

    details = details._replace(v_disp=jnp.asarray([500.0, jnp.nan]))
    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)

    assert bool(validity.is_valid) is False
    assert bool(np.asarray(validity.reason_flags)[reason_index]) is True

    details = details._replace(
        v_disp=jnp.asarray([500.0, -10.0]),
        e1=jnp.asarray([0.8, 0.0]),
        e2=jnp.asarray([0.7, 0.0]),
    )
    validity = cluster_solver.ClusterJAXEvaluator._packed_lens_validity(evaluator, details)
    shape_reason_index = cluster_solver.INVALID_STATE_REASON_NAMES.index("nonfinite_shape")

    assert bool(validity.is_valid) is False
    assert bool(np.asarray(validity.reason_flags)[shape_reason_index]) is True


def test_invalid_state_audit_draw_indices_are_deterministic_and_capped() -> None:
    first = cluster_solver._invalid_state_audit_draw_indices(1000, max_draws=7)
    second = cluster_solver._invalid_state_audit_draw_indices(1000, max_draws=7)

    np.testing.assert_array_equal(first, second)
    assert first.size == 7
    assert first[0] == 0
    assert first[-1] == 999


def test_invalid_state_audit_counts_reason_flags_from_retained_draws() -> None:
    reason_flags = np.zeros(len(cluster_solver.INVALID_STATE_REASON_NAMES), dtype=bool)
    reason_flags[cluster_solver.INVALID_STATE_REASON_NAMES.index("rs_not_greater_than_ra")] = True

    class FakeAuditEvaluator:
        flat_critical_arc_data = object()
        traced_arc_data = None
        cab_likelihood_weight = 0.0

        def _physical_parameter_vector(self, params):
            return params

        def _build_flat_packed_lens_state_with_validity_from_physical(self, physical_params, _flat_data, *, stop_gradient):
            del stop_gradient
            invalid = bool(np.asarray(physical_params)[0] > 0.5)
            flags = reason_flags if invalid else np.zeros_like(reason_flags)
            return SimpleNamespace(), SimpleNamespace(is_valid=jnp.asarray(not invalid), reason_flags=jnp.asarray(flags))

    audit = cluster_solver._audit_invalid_state_draws(
        FakeAuditEvaluator(),
        np.asarray([[0.0], [1.0], [2.0]], dtype=float),
    )

    assert audit["invalid_state_audit_draw_count"] == 3
    assert audit["invalid_state_audit_invalid_draw_count"] == 2
    assert audit["invalid_state_audit_reason_counts"]["rs_not_greater_than_ra"] == 2


def test_invalid_state_handling_does_not_use_jax_debug_callback() -> None:
    assert not hasattr(cluster_solver.ClusterJAXEvaluator, "_maybe_record_invalid_state")
    assert "_emit_invalid_state_callback" not in inspect.getsource(cluster_solver.ClusterJAXEvaluator)
    assert "debug.callback(self._record_invalid_state" not in inspect.getsource(cluster_solver)


def test_invalid_state_debug_callback_uses_are_only_progress_reporting() -> None:
    source = inspect.getsource(cluster_solver)
    callback_lines = [
        line.strip()
        for line in source.splitlines()
        if "jax.debug.callback" in line or "debug.callback" in line
    ]
    assert callback_lines
    assert all(
        "report_warmup_transition" in line or "report_progress" in line
        for line in callback_lines
    )


def test_invalid_state_audit_runs_when_debug_callback_raises(monkeypatch) -> None:
    def fail_callback(*_args, **_kwargs):
        raise AssertionError("invalid-state audit must not call jax.debug.callback")

    monkeypatch.setattr(jax.debug, "callback", fail_callback)
    audit = cluster_solver._audit_invalid_state_draws(
        SimpleNamespace(
            flat_critical_arc_data=None,
            traced_arc_data=None,
            cab_likelihood_weight=0.0,
            traced_bin_data=(),
            _physical_parameter_vector=lambda params: params,
        ),
        np.asarray([[0.0]], dtype=float),
    )
    assert audit["invalid_state_audit_draw_count"] == 1
    assert audit["invalid_state_audit_invalid_draw_count"] == 0


def test_nuts_quality_diagnostics_flag_stuck_tree_depth_and_rhat() -> None:
    grouped = np.stack(
        [np.full((10, 1), float(chain_index), dtype=float) for chain_index in range(4)],
        axis=0,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.zeros(40, dtype=float),
        accept_prob=np.ones(40, dtype=float),
        diverging=np.zeros(40, dtype=bool),
        num_steps=np.full(40, 255.0, dtype=float),
        warmup_steps=10,
        sample_steps=10,
        num_chains=4,
        grouped_samples=grouped,
        sampler="numpyro_nuts",
    )
    spec = ParameterSpec("halo.v_disp", "halo_v_disp", "halo", 81, "v_disp", "normal", -np.inf, np.inf, 0.1)

    metrics, warnings = cluster_solver._nuts_quality_diagnostics(
        argparse.Namespace(max_tree_depth=8),
        posterior,
        [spec],
    )

    assert metrics["max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert any("max-tree-depth saturation" in warning for warning in warnings)
    assert any("extreme Rhat" in warning for warning in warnings)


def test_sampler_debug_extra_fields_are_flag_gated() -> None:
    default_fields = cluster_solver._sampler_debug_extra_field_names(
        argparse.Namespace(debug_sampler_diagnostics=False)
    )
    debug_fields = cluster_solver._sampler_debug_extra_field_names(
        argparse.Namespace(debug_sampler_diagnostics=True)
    )

    assert default_fields == ("accept_prob", "diverging", "num_steps", "potential_energy")
    assert "energy" in debug_fields
    assert "adapt_state.step_size" in debug_fields
    assert "adapt_state.inverse_mass_matrix" in debug_fields


def test_sanitize_grouped_posterior_filters_nested_debug_extra_fields() -> None:
    spec = ParameterSpec(
        "theta",
        "theta",
        "theta",
        0,
        "theta",
        "normal",
        -np.inf,
        np.inf,
        0.1,
        mean=0.0,
        std=1.0,
    )
    samples = {"theta": np.asarray([[0.0, 0.1], [np.nan, 0.2]], dtype=float)}
    extra = {
        "potential_energy": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((2, 2, 1, 1), dtype=float)},
    }

    filtered_samples, filtered_extra, diagnostics = cluster_solver._sanitize_grouped_posterior(
        samples,
        extra,
        [spec],
    )

    assert diagnostics["retained_chain_indices"] == [0]
    assert filtered_samples["theta"].shape == (1, 2)
    assert filtered_extra["potential_energy"].shape == (1, 2)
    assert filtered_extra["adapt_state.inverse_mass_matrix"][("theta",)].shape == (1, 2, 1, 1)


def test_nuts_integrator_debug_table_flags_constant_stuck_chain(tmp_path: Path) -> None:
    grouped = np.stack(
        [
            np.linspace(0.0, 1.0, 5, dtype=float).reshape(5, 1),
            np.full((5, 1), 2.0, dtype=float),
        ],
        axis=0,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.zeros(10, dtype=float),
        accept_prob=np.ones(10, dtype=float),
        diverging=np.zeros(10, dtype=bool),
        num_steps=np.full(10, 255.0, dtype=float),
        warmup_steps=5,
        sample_steps=5,
        num_chains=2,
        init_diagnostics={"chain_seed_labels": ["a", "b"]},
        grouped_samples=grouped,
        grouped_log_prob=np.zeros((2, 5), dtype=float),
        sampler="numpyro_nuts",
    )
    extra = {
        "accept_prob": np.ones((2, 5), dtype=float),
        "num_steps": np.full((2, 5), 255.0, dtype=float),
        "potential_energy": np.zeros((2, 5), dtype=float),
        "energy": np.asarray([[1.0, 1.1, 0.9, 1.0, 1.2], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=float),
        "adapt_state.step_size": np.asarray([[0.01] * 5, [1.0e-8] * 5], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((2, 5, 1, 1), dtype=float)},
    }

    df = cluster_solver._write_nuts_integrator_debug_table(
        tmp_path / "nuts_integrator_diagnostics.csv",
        argparse.Namespace(max_tree_depth=8),
        posterior,
        extra,
        [0, 1],
    )

    assert (tmp_path / "nuts_integrator_diagnostics.csv").exists()
    assert df.loc[1, "stuck_parameter_count_range_lt_1e_9"] == 1
    assert df.loc[1, "parameter_range_min"] == pytest.approx(0.0)
    assert df.loc[1, "max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert df.loc[1, "step_size_last"] == pytest.approx(1.0e-8)


def test_microcanonical_transition_debug_table_flags_chain_health(tmp_path: Path) -> None:
    grouped = np.stack(
        [
            np.linspace(0.0, 1.0, 5, dtype=float).reshape(5, 1),
            np.full((5, 1), 2.0, dtype=float),
        ],
        axis=0,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.zeros(10, dtype=float),
        accept_prob=np.ones(10, dtype=float),
        diverging=np.asarray([False] * 5 + [True, False, False, False, False], dtype=bool),
        num_steps=np.ones(10, dtype=float),
        warmup_steps=5,
        sample_steps=5,
        num_chains=2,
        init_diagnostics={
            "requested_chains": 2,
            "retained_finite_chains": 2,
            "dropped_nonfinite_chains": 0,
            "chain_seed_labels": ["a", "b"],
        },
        grouped_samples=grouped,
        grouped_log_prob=np.zeros((2, 5), dtype=float),
        sampler=cluster_solver.FIT_METHOD_MCLMC,
    )
    extra = {
        "accept_prob": np.ones((2, 5), dtype=float),
        "num_steps": np.ones((2, 5), dtype=float),
        "potential_energy": np.zeros((2, 5), dtype=float),
        "energy": np.asarray([[1.0, 1.1, 0.9, 1.0, 1.2], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=float),
        "diverging": np.asarray([[False] * 5, [True, False, False, False, False]], dtype=bool),
        "is_accepted": np.asarray([[True] * 5, [False, True, True, True, True]], dtype=bool),
        "nonans": np.asarray([[True] * 5, [False, True, True, True, True]], dtype=bool),
        "state_logdensity": np.asarray([[0.0, -0.1, -0.2, -0.1, 0.0], [-1.0] * 5], dtype=float),
        "microcanonical_L": np.asarray([1.5, 2.5], dtype=float),
        "microcanonical_step_size": np.asarray([0.1, 0.2], dtype=float),
        "microcanonical_inverse_mass_matrix": np.ones((2, 1, 1), dtype=float),
    }

    df = cluster_solver._write_microcanonical_transition_debug_table(
        tmp_path / "microcanonical_transition_diagnostics.csv",
        posterior,
        extra,
        [0, 1],
    )

    assert (tmp_path / "microcanonical_transition_diagnostics.csv").exists()
    assert df.loc[1, "stuck_parameter_count_range_lt_1e_9"] == 1
    assert df.loc[1, "parameter_range_min"] == pytest.approx(0.0)
    assert df.loc[0, "microcanonical_L"] == pytest.approx(1.5)
    assert df.loc[1, "microcanonical_step_size"] == pytest.approx(0.2)
    assert df.loc[1, "nonans_fraction"] == pytest.approx(0.8)
    assert df.loc[1, "divergence_count"] == 1
    assert "energy_mean" in df
    assert "energy_diff_mean_square" in df


def test_sampler_debug_writer_creates_core_csvs(tmp_path: Path) -> None:
    spec = ParameterSpec(
        "theta",
        "theta",
        "theta",
        0,
        "theta",
        "normal",
        -np.inf,
        np.inf,
        0.1,
        mean=0.0,
        std=1.0,
    )
    grouped = np.asarray([[[0.0], [0.1], [0.2]]], dtype=float)
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.asarray([0.0, -0.1, -0.2], dtype=float),
        accept_prob=np.ones(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3, dtype=float),
        warmup_steps=2,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        grouped_samples=grouped,
        grouped_log_prob=np.asarray([[0.0, -0.1, -0.2]], dtype=float),
        sampler="numpyro_nuts",
    )
    nuts_init = NUTSInitialization(
        init_params={},
        chain_seeds=[ChainSeed(values=np.asarray([0.0], dtype=float), source_label="seed")],
        diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        reference_theta=np.asarray([0.0], dtype=float),
    )

    class FakeEvaluator:
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE

        def _source_loglike_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(jnp.square(theta))

    extra = {
        "accept_prob": np.ones((1, 3), dtype=float),
        "num_steps": np.ones((1, 3), dtype=float),
        "potential_energy": -posterior.grouped_log_prob,
        "energy": np.asarray([[1.0, 1.1, 1.2]], dtype=float),
        "adapt_state.step_size": np.asarray([[0.01, 0.01, 0.01]], dtype=float),
        "adapt_state.inverse_mass_matrix": {("theta",): np.ones((1, 3, 1, 1), dtype=float)},
    }

    diagnostics = cluster_solver._write_sampler_debug_diagnostics(
        argparse.Namespace(output_dir=str(tmp_path), debug_sampler_diagnostics=True, max_tree_depth=8),
        SimpleNamespace(run_name="run", parameter_specs=[spec]),
        FakeEvaluator(),
        nuts_init,
        posterior,
        extra,
    )

    tables = tmp_path / "run" / "tables"
    assert diagnostics["nuts_integrator_diagnostics_rows"] == 1
    integrator_df = pd.read_csv(tables / "nuts_integrator_diagnostics.csv")
    state_df = pd.read_csv(tables / "sampler_state_diagnostics.csv")
    scan_df = pd.read_csv(tables / "sampler_direction_scan.csv")
    assert {"chain", "step_size_last", "energy_bfmi_like", "inverse_mass_diag_median"} <= set(
        integrator_df.columns
    )
    assert {"state_label", "prior_log_prob", "likelihood_log_prob", "gradient_norm"} <= set(
        state_df.columns
    )
    assert {"state_label", "parameter", "delta", "posterior_log_prob"} <= set(scan_df.columns)


def test_sampler_debug_writer_creates_microcanonical_csvs(tmp_path: Path) -> None:
    spec = ParameterSpec(
        "theta",
        "theta",
        "theta",
        0,
        "theta",
        "normal",
        -np.inf,
        np.inf,
        0.1,
        mean=0.0,
        std=1.0,
    )
    grouped = np.asarray([[[0.0], [0.1], [0.2]]], dtype=float)
    posterior = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        log_prob=np.asarray([0.0, -0.1, -0.2], dtype=float),
        accept_prob=np.ones(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3, dtype=float),
        warmup_steps=2,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        grouped_samples=grouped,
        grouped_log_prob=np.asarray([[0.0, -0.1, -0.2]], dtype=float),
        sampler=cluster_solver.FIT_METHOD_MCLMC,
    )
    sampler_init = NUTSInitialization(
        init_params={},
        chain_seeds=[ChainSeed(values=np.asarray([0.0], dtype=float), source_label="seed")],
        diagnostics={"retained_chain_indices": [0], "chain_seed_labels": ["seed"]},
        reference_theta=np.asarray([0.0], dtype=float),
    )

    class FakeEvaluator:
        sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE

        def _source_loglike_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
            return -0.5 * jnp.sum(jnp.square(theta))

    extra = {
        "accept_prob": np.ones((1, 3), dtype=float),
        "num_steps": np.ones((1, 3), dtype=float),
        "potential_energy": -posterior.grouped_log_prob,
        "energy": np.asarray([[1.0, 1.1, 1.2]], dtype=float),
        "diverging": np.zeros((1, 3), dtype=bool),
        "is_accepted": np.ones((1, 3), dtype=bool),
        "nonans": np.ones((1, 3), dtype=bool),
        "state_logdensity": np.asarray([[0.0, -0.1, -0.2]], dtype=float),
        "microcanonical_L": np.asarray([1.5], dtype=float),
        "microcanonical_step_size": np.asarray([0.1], dtype=float),
        "microcanonical_inverse_mass_matrix": np.ones((1, 1, 1), dtype=float),
    }

    diagnostics = cluster_solver._write_sampler_debug_diagnostics(
        argparse.Namespace(output_dir=str(tmp_path), debug_sampler_diagnostics=True, max_tree_depth=8),
        SimpleNamespace(run_name="run", parameter_specs=[spec]),
        FakeEvaluator(),
        sampler_init,
        posterior,
        extra,
    )

    tables = tmp_path / "run" / "tables"
    assert diagnostics["microcanonical_transition_diagnostics_rows"] == 1
    assert (tables / "microcanonical_transition_diagnostics.csv").exists()
    assert (tables / "sampler_state_diagnostics.csv").exists()
    assert (tables / "sampler_direction_scan.csv").exists()
    assert not (tables / "nuts_integrator_diagnostics.csv").exists()


def _svi_health_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            "halo.x_center",
            "halo_x_center",
            "halo",
            81,
            "x_center",
            "normal",
            -np.inf,
            np.inf,
            0.1,
            mean=0.0,
            std=1.0,
        ),
        ParameterSpec(
            "halo.y_center",
            "halo_y_center",
            "halo",
            81,
            "y_center",
            "uniform",
            -2.0,
            2.0,
            0.1,
        ),
    ]


def _svi_health_samples() -> np.ndarray:
    grid = np.linspace(-0.2, 0.2, 100)
    return np.column_stack([grid, grid[::-1]])


def _svi_health_chain_seeds() -> list[ChainSeed]:
    return [
        ChainSeed(values=np.asarray([0.01, -0.01], dtype=float), source_label="chain_1"),
        ChainSeed(values=np.asarray([-0.01, 0.01], dtype=float), source_label="chain_2"),
    ]


def test_svi_health_diagnostics_healthy_guide_has_no_warnings() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert warnings == []
    assert metrics["guide_finite_draw_fraction"] == pytest.approx(1.0)
    assert metrics["center_log_prob_percentile"] == pytest.approx(50.0)
    assert metrics["chain_start_distinct_count"] == 2


def test_svi_health_diagnostics_warns_on_nonfinite_guide_draws() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])
    log_prob[5] = np.nan

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert metrics["guide_finite_draw_fraction"] < 1.0
    assert any("non-finite SVI guide draws" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_large_logprob_spread() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-5000.0, 0.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    assert metrics["guide_log_prob_q95_q05_width"] > cluster_solver.SVI_HEALTH_LOGPROB_SPREAD_WARNING
    assert any("wide SVI guide log-prob spread" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_poor_chain_start_logprob() -> None:
    samples = _svi_health_samples()
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-100.0, -9.9], dtype=float),
    )

    assert metrics["chain_start_log_prob_delta_min_from_guide_q05"] < -50.0
    assert any("poor SVI chain-start log-prob" in warning for warning in warnings)


def test_svi_health_diagnostics_warns_on_near_zero_parameter_spread() -> None:
    samples = _svi_health_samples()
    samples[:, 0] = 0.0
    log_prob = np.linspace(-12.0, -8.0, samples.shape[0])

    metrics, warnings = cluster_solver._svi_health_diagnostics(
        _svi_health_specs(),
        samples,
        log_prob,
        np.asarray([0.0, 0.0], dtype=float),
        -10.0,
        _svi_health_chain_seeds(),
        np.asarray([-10.2, -9.9], dtype=float),
    )

    worst = metrics["guide_worst_std_over_prior_scale"][0]
    assert worst["sample_name"] == "halo_x_center"
    assert worst["std_over_prior_scale"] == pytest.approx(0.0)
    assert any("near-zero SVI guide spread" in warning for warning in warnings)


def test_svi_health_diagnostics_are_recorded_in_init_diagnostics() -> None:
    posterior = PosteriorResults(
        samples=_svi_health_samples(),
        log_prob=np.linspace(-12.0, -8.0, 100),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=100,
        num_chains=0,
        init_diagnostics={},
        sampler="svi",
    )
    svi_diagnostics: dict[str, Any] = {}
    metrics = {"guide_finite_draw_fraction": 1.0}
    warnings = ["synthetic SVI health warning"]

    cluster_solver._record_svi_health_diagnostics(posterior, svi_diagnostics, metrics, warnings)

    assert posterior.init_diagnostics["svi_health_metrics"] == metrics
    assert posterior.init_diagnostics["svi_health_warnings"] == warnings
    assert svi_diagnostics["svi_health_metrics"] == metrics
    assert svi_diagnostics["svi_health_warnings"] == warnings


def test_svi_refresh_cache_probe_delta_is_bounded_and_tracks_metadata() -> None:
    before_cache = {
        1.0: {
            "a": np.arange(1000, dtype=float),
            "b": np.ones((2, 2), dtype=float),
            "remove": np.ones(5, dtype=float),
        }
    }
    after_cache = {
        1.0: {
            "a": np.arange(1000, dtype=float) + 5.0,
            "added": np.ones(5, dtype=float),
            "b": np.ones((3, 2), dtype=float),
        }
    }

    before = cluster_solver._svi_refresh_cache_snapshot(
        before_cache,
        np.asarray([0.0, 0.0], dtype=float),
        max_arrays=2,
        max_values_per_array=4,
    )
    after = cluster_solver._svi_refresh_cache_snapshot(
        after_cache,
        np.asarray([1.0, 0.0], dtype=float),
        max_arrays=2,
        max_values_per_array=4,
    )

    assert before.shapes == {
        "z=1.a": (1000,),
        "z=1.b": (2, 2),
        "z=1.remove": (5,),
    }
    assert before.total_nbytes == (1000 + 4 + 5) * np.dtype(float).itemsize
    assert set(before.probes) == {"z=1.a", "z=1.b"}
    assert before.probes["z=1.a"].values.tolist() == pytest.approx([0.0, 333.0, 666.0, 999.0])

    delta = cluster_solver._svi_refresh_cache_delta(before, after)

    assert delta.compared_arrays == 1
    assert delta.compared_values == 4
    assert delta.max_abs == pytest.approx(5.0)
    assert delta.rms == pytest.approx(5.0)
    assert delta.added_arrays == 1
    assert delta.removed_arrays == 1
    assert delta.shape_changed_arrays == 1
    assert delta.before_nbytes == before.total_nbytes
    assert delta.after_nbytes == after.total_nbytes


def test_svi_refresh_cache_probe_delta_reports_unchanged_disabled_and_empty() -> None:
    cache = {1.0: {"a": np.asarray([1.0, 2.0, 3.0], dtype=float)}}
    before = cluster_solver._svi_refresh_cache_snapshot(cache, np.asarray([0.0], dtype=float))
    after = cluster_solver._svi_refresh_cache_snapshot(cache, np.asarray([0.0], dtype=float))

    unchanged = cluster_solver._svi_refresh_cache_delta(before, after)
    assert unchanged.compared_values == 3
    assert unchanged.max_abs == pytest.approx(0.0)
    assert unchanged.rms == pytest.approx(0.0)

    disabled = cluster_solver._svi_refresh_cache_snapshot({}, None, enabled=False)
    assert cluster_solver._format_svi_refresh_cache_delta(
        "surrogate",
        cluster_solver._svi_refresh_cache_delta(disabled, disabled),
    ) == "surrogate=disabled"

    empty = cluster_solver._svi_refresh_cache_snapshot({}, None)
    assert cluster_solver._format_svi_refresh_cache_delta(
        "scaling",
        cluster_solver._svi_refresh_cache_delta(empty, empty),
    ) == "scaling=empty arrays=0 mb=0"


def test_svi_refresh_cache_status_classifies_thresholds() -> None:
    def delta(rms: float, max_abs: float) -> cluster_solver._SviRefreshCacheDelta:
        return cluster_solver._SviRefreshCacheDelta(
            enabled=True,
            before_arrays=1,
            after_arrays=1,
            before_nbytes=8,
            after_nbytes=8,
            compared_arrays=1,
            compared_values=4,
            added_arrays=0,
            removed_arrays=0,
            shape_changed_arrays=0,
            max_abs=max_abs,
            rms=rms,
        )

    assert cluster_solver._classify_svi_refresh_cache_delta("surrogate", delta(0.049, 0.19)).status == "good"
    assert cluster_solver._classify_svi_refresh_cache_delta("surrogate", delta(0.05, 0.19)).status == "watch"
    assert cluster_solver._classify_svi_refresh_cache_delta("surrogate", delta(0.12, 0.19)).status == "stale"
    assert cluster_solver._classify_svi_refresh_cache_delta("scaling", delta(0.29, 0.99)).status == "good"
    assert cluster_solver._classify_svi_refresh_cache_delta("scaling", delta(0.30, 0.99)).status == "watch"
    assert cluster_solver._classify_svi_refresh_cache_delta("source_metric", delta(0.05, 0.19)).status == "watch"

    disabled = cluster_solver._SviRefreshCacheDelta(
        enabled=False,
        before_arrays=0,
        after_arrays=0,
        before_nbytes=0,
        after_nbytes=0,
        compared_arrays=0,
        compared_values=0,
        added_arrays=0,
        removed_arrays=0,
        shape_changed_arrays=0,
        max_abs=float("nan"),
        rms=float("nan"),
    )
    assert cluster_solver._classify_svi_refresh_cache_delta("surrogate", disabled).status == "disabled"


def test_svi_refresh_status_rich_table_has_status_columns() -> None:
    rows = [
        cluster_solver._SviRefreshCacheStatus(
            name="surrogate",
            rms="0.018",
            max_abs="0.044",
            status="good",
            meaning="local approximation stable",
            style="bold green",
        ),
        cluster_solver._SviRefreshCacheStatus(
            name="source_metric",
            rms="0.05",
            max_abs="0.138",
            status="watch",
            meaning="lensing metric changed mildly",
            style="bold yellow",
        ),
    ]
    parameter_delta = cluster_solver._SviRefreshParameterDelta(
        theta_l2=3.0,
        theta_linf=2.36,
        top_changes="source_sigma_int:2.36,O1_y_centre:1.17",
    )

    table = cluster_solver._build_svi_refresh_status_rich_table(
        block_index=2,
        remaining_steps=4000,
        center_shift=3.77,
        parameter_delta=parameter_delta,
        rows=rows,
    )

    assert getattr(table, "title", None) == "SVI refresh: block 2, remaining 4000"
    assert [column.header for column in table.columns] == ["cache", "rms", "max", "status", "meaning"]
    assert table.columns[0]._cells == ["surrogate", "source_metric"]
    assert table.columns[3]._cells == ["good", "watch"]


def test_pre_nuts_refresh_uses_plain_fresh_cache_message() -> None:
    class FakeEvaluator:
        surrogate_enabled = True
        active_inference_enabled = True

        def __init__(self) -> None:
            self.refresh_reasons: list[str] = []

        def refresh_active_inference_cache(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"active:{reason}")

        def refresh_surrogate(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"surrogate:{reason}")

        def refresh_scaling_scatter_cache(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"scaling:{reason}")

        def refresh_source_metric_cache(self, _params, reason: str) -> None:
            self.refresh_reasons.append(f"source_metric:{reason}")

    evaluator = FakeEvaluator()

    message = cluster_solver._refresh_pre_nuts_caches(
        evaluator,
        np.asarray([0.0], dtype=float),
        "pre_nuts",
    )

    assert evaluator.refresh_reasons == [
        "active:pre_nuts",
        "surrogate:pre_nuts",
        "scaling:pre_nuts",
        "source_metric:pre_nuts",
    ]
    assert message.startswith("[svi] pre_nuts using freshly refreshed caches")
    assert "reason=pre_nuts" in message
    assert "surrogate=refreshed" in message
    assert "scaling=refreshed" in message
    assert "source_metric=refreshed" in message
    assert "surrogate_rms=" not in message
    assert "scaling_rms=" not in message
    assert "source_metric_rms=" not in message
    assert "surrogate_max=" not in message
    assert "scaling_max=" not in message
    assert "source_metric_max=" not in message


def test_nuts_svi_deviation_rows_classify_good_watch_large_unknown() -> None:
    specs = [
        ParameterSpec(f"p{i}", f"p{i}", "lens", i, "value", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0)
        for i in range(3)
    ]
    reference = np.asarray([0.0, 0.0, 0.0], dtype=float)

    good = PosteriorResults(
        samples=np.asarray([[0.10, 0.02, 0.00], [0.10, -0.02, 0.00], [0.10, 0.00, 0.02]], dtype=float),
        log_prob=np.zeros(3),
        accept_prob=np.ones(3),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    good_rows, good_top = cluster_solver._nuts_svi_deviation_rows(good, reference, specs)
    assert {row.metric: row.status for row in good_rows}["median_shift_linf"] == "good"
    assert {row.metric: row.status for row in good_rows}["median_shift_rms"] == "good"
    assert good_top.startswith("p0:")

    watch = PosteriorResults(
        samples=np.asarray([[0.40, 0.20, -0.10], [0.50, 0.10, -0.10], [0.30, 0.30, -0.10]], dtype=float),
        log_prob=np.zeros(3),
        accept_prob=np.ones(3),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    watch_rows, _watch_top = cluster_solver._nuts_svi_deviation_rows(watch, reference, specs)
    assert {row.metric: row.status for row in watch_rows}["median_shift_linf"] == "watch"

    large = PosteriorResults(
        samples=np.asarray([[1.20, 0.00, 0.00], [1.30, 0.00, 0.00], [1.10, 0.00, 0.00]], dtype=float),
        log_prob=np.zeros(3),
        accept_prob=np.ones(3),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    large_rows, _large_top = cluster_solver._nuts_svi_deviation_rows(large, reference, specs)
    assert {row.metric: row.status for row in large_rows}["median_shift_linf"] == "large"

    unknown = PosteriorResults(
        samples=np.empty((0, 3), dtype=float),
        log_prob=np.empty((0,), dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=0,
        num_chains=0,
    )
    unknown_rows, unknown_top = cluster_solver._nuts_svi_deviation_rows(unknown, reference, specs)
    assert {row.metric: row.status for row in unknown_rows}["median_shift_linf"] == "unknown"
    assert unknown_top == "none"


def test_nuts_svi_deviation_rich_table_has_status_columns() -> None:
    rows = [
        cluster_solver._NutsSviDeviationStatus(
            metric="median_shift_linf",
            value="0.42",
            status="watch",
            meaning="largest parameter moved mildly",
            style="bold yellow",
        ),
        cluster_solver._NutsSviDeviationStatus(
            metric="largest_moves",
            value="O1_y_centre:0.42",
            status="watch",
            meaning="top posterior median shifts from SVI",
            style="bold yellow",
        ),
    ]

    table = cluster_solver._build_nuts_svi_deviation_rich_table(rows)

    assert getattr(table, "title", None) == "NUTS deviation from SVI"
    assert [column.header for column in table.columns] == ["metric", "value", "status", "meaning"]
    assert table.columns[0]._cells == ["median_shift_linf", "largest_moves"]
    assert table.columns[2]._cells == ["watch", "watch"]


def test_log_nuts_svi_deviation_table_emits_plain_message_and_renderable(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    renderables: list[Any] = []

    def fake_log(_args: argparse.Namespace, message: str, *, renderable: Any | None = None) -> None:
        messages.append(message)
        renderables.append(renderable)

    monkeypatch.setattr(cluster_solver, "_log", fake_log)
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0)
    ]
    posterior = PosteriorResults(
        samples=np.asarray([[0.1], [0.2], [0.3]], dtype=float),
        log_prob=np.zeros(3),
        accept_prob=np.ones(3),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.ones(3),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    nuts_init = NUTSInitialization(
        init_params={"halo_x": jnp.asarray([0.0], dtype=jnp.float64)},
        chain_seeds=[ChainSeed(values=np.asarray([0.0], dtype=float), source_label="seed")],
        diagnostics={"svi_used": True},
        reference_theta=np.asarray([0.0], dtype=float),
    )

    cluster_solver._log_nuts_svi_deviation_table(
        argparse.Namespace(quiet=True),
        SimpleNamespace(parameter_specs=specs),
        posterior,
        nuts_init,
    )

    assert len(messages) == 1
    assert messages[0].startswith("[nuts] svi_deviation ")
    assert "median_shift_linf=" in messages[0]
    assert "median_shift_rms=" in messages[0]
    assert "max_shift_over_posterior_sigma=" in messages[0]
    assert "top=halo_x:" in messages[0]
    assert getattr(renderables[0], "title", None) == "NUTS deviation from SVI"


def test_svi_refresh_array_nbytes_handles_numpy_and_jax_arrays() -> None:
    cache = {
        1.0: {
            "numpy": np.ones((2, 3), dtype=np.float32),
            "jax": jnp.ones((4,), dtype=jnp.float64),
            "ignored": "not-an-array",
        }
    }

    snapshot = cluster_solver._svi_refresh_cache_snapshot(cache, np.asarray([0.0], dtype=float))

    assert snapshot.shapes == {
        "z=1.jax": (4,),
        "z=1.numpy": (2, 3),
    }
    assert snapshot.total_nbytes == 6 * np.dtype(np.float32).itemsize + 4 * np.dtype(np.float64).itemsize


def test_perturbation_discovery_final_diagnostic_messages_include_counts() -> None:
    union_diag = {
        "candidate_galaxies": 200,
        "n_images": 42,
        "pairs": 210,
        "exact_unique": 31,
        "count": 29,
        "score_fraction": 0.87,
        "alpha_tol_arcsec": 0.02,
        "jacobian_tol": 0.03,
        "jacobian_weight": 0.5,
        "max_selected_score": 7.0,
        "max_unselected_score": 0.9,
    }
    svi_final_message = cluster_solver._perturbation_discovery_svi_final_log_message(
        union_diag,
        stage1_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
    )

    assert "[perturbation-discovery:svi-final]" in svi_final_message
    assert "candidate_galaxies=200" in svi_final_message
    assert "images=42" in svi_final_message
    assert "pairs=210" in svi_final_message
    assert "unique_selected=31" in svi_final_message
    assert "independent_candidates=29" in svi_final_message
    assert "alpha_tol_arcsec=0.02" in svi_final_message
    assert "jacobian_tol=0.03" in svi_final_message
    assert "jacobian_weight=0.5" in svi_final_message
    assert "score_fraction=0.87" in svi_final_message
    assert "stage0_final_rebuild=full_flat" in svi_final_message
    assert "stage1_engine=refreshing_surrogate_flat" in svi_final_message
    assert "final_svi_polish_steps" not in svi_final_message

    counts = cluster_solver._perturbation_discovery_final_model_counts(
        SimpleNamespace(
            active_scaling_component_indices=np.arange(29, dtype=np.int32),
            exact_scaling_component_indices=np.arange(29, dtype=np.int32),
            cached_scaling_component_indices=np.arange(29, 200, dtype=np.int32),
            independent_scaling_component_indices=np.arange(29, dtype=np.int32),
            free_correction_scaling_component_indices=np.arange(29, dtype=np.int32),
            large_component_indices=np.arange(2, dtype=np.int32),
            scaling_component_indices=np.arange(200, dtype=np.int32),
        )
    )
    final_model_message = cluster_solver._perturbation_discovery_final_model_log_message(
        counts,
        old_parameter_count=8,
        new_parameter_count=95,
    )

    assert "[perturbation-discovery:final-model]" in final_model_message
    assert "active_scaling=29" in final_model_message
    assert "exact_scaling=29" in final_model_message
    assert "cached_scaling=171" in final_model_message
    assert "independent_scaling=29" in final_model_message
    assert "free_correction_candidates=29" in final_model_message
    assert "large_exact=2" in final_model_message
    assert "total_scaling=200" in final_model_message
    assert "old_parameters=8" in final_model_message
    assert "new_parameters=95" in final_model_message
    diagnostics = cluster_solver._perturbation_discovery_svi_final_diagnostics(
        union_diag,
        {"count": 29},
        counts,
        stage1_engine=cluster_solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
    )
    assert diagnostics["perturbation_discovery_svi_final_independent_candidates"] == 29
    assert diagnostics["perturbation_discovery_svi_final_pairs"] == 210
    assert diagnostics["perturbation_discovery_svi_final_candidate_galaxies"] == 200
    assert diagnostics["perturbation_discovery_svi_final_images"] == 42
    assert diagnostics["perturbation_discovery_svi_final_rebuild_engine"] == "full_flat"
    assert diagnostics["perturbation_discovery_svi_final_stage1_engine"] == "refreshing_surrogate_flat"
    assert diagnostics["perturbation_discovery_final_strict_active"] is True


def test_rejects_legacy_pjaffe_artifacts() -> None:
    with pytest.raises(ValueError, match="unsupported compact/PJAFFE"):
        cluster_solver._validate_supported_lens_model_list(["DPIE_NIE", "PJAFFE_ELLIPSE_POTENTIAL"], "legacy")


def test_bulk_lensing_jacobian_matches_manual_dpie_finite_difference() -> None:
    class ForbiddenBulkModel:
        def ray_shooting(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("old LensModelBulk.ray_shooting path should not be used")

        def hessian(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("old LensModelBulk.hessian path should not be used")

    fake = SimpleNamespace(
        use_bulk_ray_shooting=True,
        bulk_index_list=np.asarray([0], dtype=np.int32),
        models_by_effective_z={2.0: ForbiddenBulkModel()},
        state=SimpleNamespace(
            lens_model_list=["DPIE_NIE"],
            packed_lens_spec=SimpleNamespace(profile_type=np.asarray([cluster_solver.DP_IE_PROFILE], dtype=np.int32)),
        ),
    )
    fake._component_indices_np = cluster_solver.ClusterJAXEvaluator._component_indices_np.__get__(
        fake,
        type(fake),
    )
    fake._split_grouped_component_indices = cluster_solver.ClusterJAXEvaluator._split_grouped_component_indices.__get__(
        fake,
        type(fake),
    )
    fake._take_packed_components = cluster_solver.ClusterJAXEvaluator._take_packed_components.__get__(
        fake,
        type(fake),
    )
    fake._grouped_dpie_params = cluster_solver.ClusterJAXEvaluator._grouped_dpie_params.__get__(
        fake,
        type(fake),
    )
    fake._grouped_shear_alpha_and_hessian = cluster_solver.ClusterJAXEvaluator._grouped_shear_alpha_and_hessian.__get__(
        fake,
        type(fake),
    )
    fake._grouped_alpha_and_hessian_for_components = cluster_solver.ClusterJAXEvaluator._grouped_alpha_and_hessian_for_components.__get__(
        fake,
        type(fake),
    )
    packed_state = PackedLensState(
        profile_type=jnp.asarray([cluster_solver.DP_IE_PROFILE], dtype=jnp.int32),
        sigma0=jnp.asarray([1.2], dtype=jnp.float64),
        Ra=jnp.asarray([0.15], dtype=jnp.float64),
        Rs=jnp.asarray([3.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05], dtype=jnp.float64),
        e2=jnp.asarray([-0.02], dtype=jnp.float64),
        center_x=jnp.asarray([0.1], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0], dtype=jnp.float64),
    )
    x = jnp.asarray([0.2, 1.0, 3.0], dtype=jnp.float64)
    y = jnp.asarray([0.4, 2.0, -1.0], dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)

    jacobian = cluster_solver.ClusterJAXEvaluator._lensing_jacobian_for_components(fake, 2.0, x, y, packed_state)
    beta_x_plus, beta_y_plus = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(
        fake,
        2.0,
        x + eps,
        y,
        packed_state,
    )
    beta_x_minus, beta_y_minus = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(
        fake,
        2.0,
        x - eps,
        y,
        packed_state,
    )
    beta_x_y_plus, beta_y_y_plus = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(
        fake,
        2.0,
        x,
        y + eps,
        packed_state,
    )
    beta_x_y_minus, beta_y_y_minus = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(
        fake,
        2.0,
        x,
        y - eps,
        packed_state,
    )
    expected = (
        (beta_x_plus - beta_x_minus) / (2.0 * eps),
        (beta_x_y_plus - beta_x_y_minus) / (2.0 * eps),
        (beta_y_plus - beta_y_minus) / (2.0 * eps),
        (beta_y_y_plus - beta_y_y_minus) / (2.0 * eps),
    )

    for value, reference in zip(jacobian, expected):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference), atol=1.0e-5, rtol=1.0e-5)


def test_grouped_lensing_backend_rejects_unsupported_profile_code() -> None:
    fake = SimpleNamespace(
        state=SimpleNamespace(
            lens_model_list=["DPIE_NIE", "UNSUPPORTED"],
            packed_lens_spec=SimpleNamespace(
                profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, 999], dtype=np.int32)
            ),
        ),
    )
    fake._component_indices_np = cluster_solver.ClusterJAXEvaluator._component_indices_np.__get__(
        fake,
        type(fake),
    )

    with pytest.raises(RuntimeError, match="Mandatory grouped lensing backend"):
        cluster_solver.ClusterJAXEvaluator._split_grouped_component_indices(fake)


def test_packed_lens_state_is_array_only_pytree_for_grouped_lensing() -> None:
    packed_state = PackedLensState(
        profile_type=jnp.asarray([cluster_solver.DP_IE_PROFILE], dtype=jnp.int32),
        sigma0=jnp.asarray([1.2], dtype=jnp.float64),
        Ra=jnp.asarray([0.15], dtype=jnp.float64),
        Rs=jnp.asarray([3.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05], dtype=jnp.float64),
        e2=jnp.asarray([-0.02], dtype=jnp.float64),
        center_x=jnp.asarray([0.1], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0], dtype=jnp.float64),
    )
    leaves, _treedef = jax.tree_util.tree_flatten(packed_state)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    fake = SimpleNamespace(
        state=SimpleNamespace(
            lens_model_list=["DPIE_NIE"],
            packed_lens_spec=SimpleNamespace(
                profile_type=np.asarray([cluster_solver.DP_IE_PROFILE], dtype=np.int32)
            ),
        ),
    )
    for name in (
        "_component_indices_np",
        "_split_grouped_component_indices",
        "_take_packed_components",
        "_grouped_dpie_params",
        "_grouped_shear_alpha_and_hessian",
        "_grouped_alpha_and_hessian_for_components",
    ):
        setattr(fake, name, getattr(cluster_solver.ClusterJAXEvaluator, name).__get__(fake, type(fake)))

    x = jnp.asarray([0.2, 1.0], dtype=jnp.float64)
    y = jnp.asarray([0.4, -1.0], dtype=jnp.float64)

    @jax.jit
    def evaluate(current_state: PackedLensState) -> tuple[jnp.ndarray, ...]:
        beta_x, beta_y = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(
            fake,
            2.0,
            x,
            y,
            current_state,
        )
        jacobian = cluster_solver.ClusterJAXEvaluator._lensing_jacobian_for_components(
            fake,
            2.0,
            x,
            y,
            current_state,
        )
        return beta_x, beta_y, *jacobian

    outputs = evaluate(packed_state)
    assert len(outputs) == 6
    assert all(np.asarray(value).shape == (2,) for value in outputs)


def test_mock_recovery_static_lens_state_from_packed_preserves_arrays() -> None:
    packed_state = PackedLensState(
        profile_type=jnp.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE], dtype=jnp.int32),
        sigma0=jnp.asarray([1.2, 0.0], dtype=jnp.float64),
        Ra=jnp.asarray([0.15, 0.0], dtype=jnp.float64),
        Rs=jnp.asarray([3.0, 0.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05, 0.0], dtype=jnp.float64),
        e2=jnp.asarray([-0.02, 0.0], dtype=jnp.float64),
        center_x=jnp.asarray([0.1, 0.0], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1, 0.0], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0, 0.01], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0, -0.02], dtype=jnp.float64),
    )

    static_state = validation._static_lens_state_from_packed(packed_state)

    assert isinstance(static_state, validation.jax_lensing.StaticLensState)
    for field_name in packed_state._fields:
        np.testing.assert_allclose(np.asarray(getattr(static_state, field_name)), np.asarray(getattr(packed_state, field_name)))


def test_truth_caustic_plot_uses_static_jax_lens_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SimpleNamespace()
    truth = {"config": SingleBCGMockConfig().to_json_dict()}
    packed_state = PackedLensState(
        profile_type=jnp.asarray([cluster_solver.DP_IE_PROFILE], dtype=jnp.int32),
        sigma0=jnp.asarray([1.2], dtype=jnp.float64),
        Ra=jnp.asarray([0.15], dtype=jnp.float64),
        Rs=jnp.asarray([3.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05], dtype=jnp.float64),
        e2=jnp.asarray([-0.02], dtype=jnp.float64),
        center_x=jnp.asarray([0.1], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0], dtype=jnp.float64),
    )
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setattr(validation, "_packed_lens_state_from_current_truth_components", lambda *_args: packed_state)

    def fake_compute(*args: Any) -> list[validation.CausticContour]:
        calls.append(args)
        assert len(args) == 2
        assert isinstance(args[0], validation.jax_lensing.StaticLensState)
        assert isinstance(args[1], SingleBCGMockConfig)
        return []

    monkeypatch.setattr(validation, "_compute_tangential_caustic_contours", fake_compute)

    result = validation._truth_caustic_contours_by_z_for_plot(state, truth, ["2.0"])

    assert result == {}
    assert len(calls) == 1


def test_recovered_caustic_plot_uses_static_jax_lens_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SimpleNamespace(parameter_specs=[], lens_model_list=["DPIE_NIE"])
    best_fit = np.empty((0,), dtype=float)
    truth = {"config": SingleBCGMockConfig().to_json_dict()}
    packed_state = PackedLensState(
        profile_type=jnp.asarray([cluster_solver.DP_IE_PROFILE], dtype=jnp.int32),
        sigma0=jnp.asarray([1.2], dtype=jnp.float64),
        Ra=jnp.asarray([0.15], dtype=jnp.float64),
        Rs=jnp.asarray([3.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05], dtype=jnp.float64),
        e2=jnp.asarray([-0.02], dtype=jnp.float64),
        center_x=jnp.asarray([0.1], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0], dtype=jnp.float64),
    )
    calls: list[tuple[Any, ...]] = []

    class FakeEvaluator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return values

        def _build_packed_lens_state(self, _values: Any, _z_source: float) -> PackedLensState:
            return packed_state

    def fake_compute(*args: Any) -> list[validation.CausticContour]:
        calls.append(args)
        assert len(args) == 2
        assert isinstance(args[0], validation.jax_lensing.StaticLensState)
        assert isinstance(args[1], SingleBCGMockConfig)
        return []

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(validation, "_compute_tangential_caustic_contours", fake_compute)

    result = validation._recovered_caustic_contours_by_z(state, best_fit, truth, ["2.0"])

    assert result == {}
    assert len(calls) == 1


def test_grouped_lensing_hot_paths_do_not_string_index_packed_state() -> None:
    for method in (
        cluster_solver.ClusterJAXEvaluator._grouped_dpie_params,
        cluster_solver.ClusterJAXEvaluator._grouped_shear_alpha_and_hessian,
        cluster_solver.ClusterJAXEvaluator._grouped_alpha_and_hessian_for_components,
        cluster_solver.ClusterJAXEvaluator._grouped_component_rows_for_components,
        cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components,
        cluster_solver.ClusterJAXEvaluator._lensing_jacobian_for_components,
        cluster_solver.ClusterJAXEvaluator._flat_ray_shooting_and_lensing_jacobian_for_components,
    ):
        assert "packed_state[" not in inspect.getsource(method)


def test_grouped_dpie_shear_backend_matches_lensmodelbulk_reference() -> None:
    lens_model_list = ["DPIE_NIE", "DPIE_NIE", "SHEAR"]
    packed_state = PackedLensState(
        profile_type=jnp.asarray(
            [cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE],
            dtype=jnp.int32,
        ),
        sigma0=jnp.asarray([1.2, 0.7, 0.0], dtype=jnp.float64),
        Ra=jnp.asarray([0.15, 0.08, 0.0], dtype=jnp.float64),
        Rs=jnp.asarray([3.0, 1.8, 0.0], dtype=jnp.float64),
        e1=jnp.asarray([0.05, -0.04, 0.0], dtype=jnp.float64),
        e2=jnp.asarray([-0.02, 0.03, 0.0], dtype=jnp.float64),
        center_x=jnp.asarray([0.1, -0.3, 0.0], dtype=jnp.float64),
        center_y=jnp.asarray([-0.1, 0.2, 0.0], dtype=jnp.float64),
        gamma1=jnp.asarray([0.0, 0.0, 0.04], dtype=jnp.float64),
        gamma2=jnp.asarray([0.0, 0.0, -0.015], dtype=jnp.float64),
    )
    fake = SimpleNamespace(
        state=SimpleNamespace(
            lens_model_list=lens_model_list,
            packed_lens_spec=SimpleNamespace(
                profile_type=np.asarray(
                    [cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE, cluster_solver.SHEAR_PROFILE],
                    dtype=np.int32,
                )
            ),
        ),
    )
    for name in (
        "_component_indices_np",
        "_split_grouped_component_indices",
        "_take_packed_components",
        "_grouped_dpie_params",
        "_grouped_shear_alpha_and_hessian",
        "_grouped_alpha_and_hessian_for_components",
    ):
        setattr(fake, name, getattr(cluster_solver.ClusterJAXEvaluator, name).__get__(fake, type(fake)))

    x = jnp.asarray([-1.2, 0.2, 1.0, 2.5], dtype=jnp.float64)
    y = jnp.asarray([0.7, -0.4, 1.5, -2.0], dtype=jnp.float64)
    grouped_beta = cluster_solver.ClusterJAXEvaluator._ray_shooting_for_components(fake, 2.0, x, y, packed_state)
    grouped_jacobian = cluster_solver.ClusterJAXEvaluator._lensing_jacobian_for_components(fake, 2.0, x, y, packed_state)

    model = cluster_solver.LensModelBulk(unique_lens_model_list=["DPIE_NIE", "SHEAR"], multi_plane=False)
    kwargs = model.prepare_ray_shooting_kwargs(
        lens_model_list,
        [
            {
                "sigma0": 1.2,
                "Ra": 0.15,
                "Rs": 3.0,
                "e1": 0.05,
                "e2": -0.02,
                "center_x": 0.1,
                "center_y": -0.1,
            },
            {
                "sigma0": 0.7,
                "Ra": 0.08,
                "Rs": 1.8,
                "e1": -0.04,
                "e2": 0.03,
                "center_x": -0.3,
                "center_y": 0.2,
            },
            {"gamma1": 0.04, "gamma2": -0.015, "ra_0": 0.0, "dec_0": 0.0},
        ],
    )
    reference_beta = model.ray_shooting(x, y, kwargs)
    h_xx, h_xy, h_yx, h_yy = model.hessian(x, y, kwargs)
    reference_jacobian = (1.0 - h_xx, -h_xy, -h_yx, 1.0 - h_yy)

    for value, reference in zip(grouped_beta, reference_beta):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference), atol=1.0e-10, rtol=1.0e-10)
    for value, reference in zip(grouped_jacobian, reference_jacobian):
        np.testing.assert_allclose(np.asarray(value), np.asarray(reference), atol=1.0e-10, rtol=1.0e-10)


def test_cluster_solver_arc_recovery_p_arc_threshold_defaults_to_point_one(
) -> None:
    assert cluster_solver.DEFAULT_ARC_RECOVERY_P_ARC_THRESHOLD == pytest.approx(0.1)
    assert planning.SOLVER_RUNTIME_DEFAULTS["arc_recovery_p_arc_threshold"] == pytest.approx(0.1)
    assert ImageDiagnosticsConfig().arc_recovery_p_arc_threshold == pytest.approx(0.1)
    assert ImageDiagnosticsConfig().critical_arc_singular_threshold == pytest.approx(0.05)


def test_solver_config_image_diagnostics_sets_critical_arc_thresholds(
) -> None:
    config = LensClusterSolverConfig(
        image_diagnostics=ImageDiagnosticsConfig(
            arc_recovery_p_arc_threshold=0.35,
            critical_arc_singular_threshold=0.12,
        )
    )

    payload = planning._runtime_payload(config)

    assert payload["arc_recovery_p_arc_threshold"] == pytest.approx(0.35)
    assert payload["critical_arc_singular_threshold"] == pytest.approx(0.12)


def test_direct_nuts_initialization_uses_reference_and_clipped_chain_seeds() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "uniform", -1.0, 1.0, 0.1),
        ParameterSpec(
            "halo.y",
            "halo_y",
            "halo",
            0,
            "y",
            "truncated_normal",
            -2.0,
            2.0,
            0.1,
            mean=0.0,
            std=1.0,
        ),
    ]
    args = argparse.Namespace(
        chains=3,
        seed=12,
        nuts_init_jitter_frac=0.2,
        nuts_init_boundary_frac=0.1,
    )

    init = cluster_solver._nuts_initialization_from_reference(
        args,
        specs,
        np.asarray([0.5, 1.0], dtype=float),
    )

    assert init.diagnostics["strategy_requested"] == cluster_solver.FIT_METHOD_NUTS
    assert init.diagnostics["strategy_used"] == "previous_stage"
    assert init.diagnostics["svi_used"] is False
    assert init.diagnostics["chain_seed_labels"] == [
        "direct_nuts_chain_1",
        "direct_nuts_chain_2",
        "direct_nuts_chain_3",
    ]
    assert np.asarray(init.init_params["halo_x"]).shape == (3,)
    assert np.asarray(init.init_params["halo_y"]).shape == (3,)
    for seed in init.chain_seeds:
        assert -1.0 < seed.values[0] < 1.0
        assert -2.0 < seed.values[1] < 2.0
    np.testing.assert_allclose(init.reference_theta, np.asarray([0.5, 1.0], dtype=float))


@pytest.mark.parametrize("explicit_method", [False, True])
def test_numpyro_nuts_sampler_parallel_requires_device_per_chain(
    monkeypatch: pytest.MonkeyPatch,
    explicit_method: bool,
) -> None:
    monkeypatch.setattr(cluster_solver.jax, "device_count", lambda: 1)
    monkeypatch.setattr(cluster_solver, "NUTS", lambda *_args, **_kwargs: pytest.fail("NUTS should not be built"))
    monkeypatch.setattr(cluster_solver, "MCMC", lambda *_args, **_kwargs: pytest.fail("MCMC should not be built"))
    args_payload: dict[str, Any] = {"chains": 2}
    if explicit_method:
        args_payload["nuts_chain_method"] = "parallel"

    with pytest.raises(RuntimeError, match="chains=2 jax_device_count=1"):
        cluster_solver._run_numpyro_nuts_sampler(
            argparse.Namespace(**args_payload),
            SimpleNamespace(parameter_specs=[]),
            SimpleNamespace(),
            object(),
            NUTSInitialization(
                init_params={},
                chain_seeds=[],
                diagnostics={},
                reference_theta=np.asarray([], dtype=float),
            ),
        )


@pytest.mark.parametrize(
    ("dense_mass", "expected_numpyro_dense_mass"),
    [
        (cluster_solver.NUTS_DENSE_MASS_FULL, True),
        (cluster_solver.NUTS_DENSE_MASS_DIAGONAL, False),
    ],
)
def test_block_nuts_once_passes_dense_mass_to_numpyro(
    monkeypatch: pytest.MonkeyPatch,
    dense_mass: str,
    expected_numpyro_dense_mass: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_nuts(_model, **kwargs):
        captured["nuts_kwargs"] = dict(kwargs)
        return object()

    class FakeMCMC:
        def __init__(self, *_args, **_kwargs) -> None:
            self.last_state = SimpleNamespace(
                adapt_state=SimpleNamespace(
                    step_size=jnp.asarray(0.1),
                    inverse_mass_matrix=jnp.asarray([1.0]),
                )
            )

        def run(self, *_args, **_kwargs) -> None:
            return None

        def get_samples(self, group_by_chain: bool = False):
            assert group_by_chain is False
            return {"halo_x": jnp.asarray([0.25], dtype=jnp.float64)}

        def get_extra_fields(self, group_by_chain: bool = False):
            assert group_by_chain is False
            return {
                "accept_prob": jnp.asarray([0.9], dtype=jnp.float64),
                "diverging": jnp.asarray([False], dtype=bool),
                "num_steps": jnp.asarray([3.0], dtype=jnp.float64),
                "potential_energy": jnp.asarray([1.0], dtype=jnp.float64),
            }

    monkeypatch.setattr(cluster_solver, "NUTS", fake_nuts)
    monkeypatch.setattr(cluster_solver, "MCMC", FakeMCMC)
    monkeypatch.setattr(cluster_solver, "_conditioned_posterior_model_for_block", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cluster_solver, "_block_init_params", lambda *_args, **_kwargs: {})

    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0)
    ]
    args = argparse.Namespace(
        target_accept=0.8,
        max_tree_depth=4,
        initial_step_size=0.1,
        dense_mass=dense_mass,
        quiet=True,
    )
    evaluator = SimpleNamespace()
    block = cluster_solver.BlockedNUTSParameterBlock("non_source", (0,))

    cluster_solver._run_block_nuts_once(
        args,
        specs,
        evaluator,
        np.asarray([0.0], dtype=float),
        block,
        jax.random.PRNGKey(0),
        num_warmup=0,
    )

    assert captured["nuts_kwargs"]["dense_mass"] is expected_numpyro_dense_mass


def test_blocked_nuts_parameter_blocks_partition_source_positions() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "halo", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec(
            "source.1.beta_x",
            "source_1_beta_x",
            "1",
            0,
            "beta_x",
            "normal",
            -np.inf,
            np.inf,
            0.1,
            mean=0.0,
            std=1.0,
            component_family="source_position",
        ),
        ParameterSpec("scale.sigma", "scale_sigma", "scale", 0, "sigma", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]

    blocks = cluster_solver._blocked_nuts_parameter_blocks(specs)

    assert [(block.name, block.indices) for block in blocks] == [
        ("non_source", (0, 2)),
        ("source_position", (1,)),
    ]


def test_cluster_solver_rejects_unavailable_explicit_gpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster_solver.jax, "devices", lambda kind=None: [] if kind == "gpu" else [object()])
    with pytest.raises(ValueError, match="--smc-device=gpu"):
        cluster_solver._resolve_jax_device("gpu", flag_name="--smc-device")


def test_cluster_solver_resolves_cpu_and_auto_devices() -> None:
    assert cluster_solver._resolve_jax_device("auto", flag_name="--jax-default-device") is None
    cpu_device = cluster_solver._resolve_jax_device("cpu", flag_name="--jax-default-device")
    assert cpu_device is not None
    assert cluster_solver._jax_device_backend(cpu_device) == "cpu"


def test_image_presence_effective_default_weight_only_stage4() -> None:
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_FORWARD_METRIC_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_FORWARD_METRIC,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_ANCHORED_SOLVED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_ANCHORED_SOLVED_FORWARD_BETA,
    ) == pytest.approx(2.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(0.0)
    assert _effective_image_presence_penalty_weight(
        None,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="evidence-ns",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    ) == pytest.approx(0.0)
    assert _effective_image_presence_penalty_weight(
        0.0,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    ) == pytest.approx(0.0)


def test_stage_fit_controls_reject_four_values() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi", "svi+nuts"],
        warmup=[2000, 1000, 0, 500],
        samples=[250, 100, 20, 80],
        max_tree_depth=[10, 8, 6, 5],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=False,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_refreshing_surrogate_stage4_newton_steps() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "svi"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 20],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        sampling_engine="refreshing_surrogate_flat",
        image_plane_newton_steps=1,
    )

    with pytest.raises(SystemExit, match="refreshing_surrogate_flat"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_three_values_without_stage4_mode() -> None:
    args = argparse.Namespace(
        fit_method=["svi", "svi", "svi"],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_two_values_without_image_plane_stage() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_two_values_for_non_sequential_runs() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500],
        warmup=1,
        samples=2,
        fit_mode="joint",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_four_svi_step_values() -> None:
    args = argparse.Namespace(
        fit_method="svi+nuts",
        svi_steps=[1000, 500, 100, 50],
        warmup=1,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="at most three"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_stage2_without_image_plane() -> None:
    args = argparse.Namespace(
        fit_method="ns",
        warmup=0,
        samples=50,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_final_local_jacobian_stage3() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "ns"],
        warmup=[2000, 0],
        samples=[250, 100],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_final_stage4() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "svi+nuts", "ns"],
        warmup=[2000, 1000, 0],
        samples=[250, 100, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_for_stage4_when_stage3_skipped() -> None:
    args = argparse.Namespace(
        fit_method=["svi+nuts", "ns"],
        warmup=[2000, 0],
        samples=[250, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LINEARIZED_FORWARD_BETA,
        skip_stage3_image_plane_local_jacobian=True,
    )

    with pytest.raises(SystemExit, match="evidence-ns"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_ns_before_final_image_plane_stage() -> None:
    args = argparse.Namespace(
        fit_method=["ns", "svi"],
        warmup=[0, 100],
        samples=[50, 50],
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_scalar_ns_when_image_plane_stage_enabled() -> None:
    args = argparse.Namespace(
        fit_method="ns",
        warmup=0,
        samples=50,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_skip_stage3_without_stage4_mode() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=2,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_LOCAL_JACOBIAN,
        skip_stage3_image_plane_local_jacobian=True,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_non_positive_samples() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=0,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_negative_max_tree_depth() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=2,
        max_tree_depth=-1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="max-tree-depth"):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_non_positive_ns_max_samples() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        ns_max_samples=0,
    )

    with pytest.raises(SystemExit):
        _normalize_stage_fit_controls(args)


def test_stage_fit_controls_reject_invalid_ns_max_samples_string() -> None:
    args = argparse.Namespace(
        fit_method="svi",
        warmup=0,
        samples=1,
        fit_mode="sequential",
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        ns_max_samples="forever",
    )

    with pytest.raises(SystemExit, match="positive integer"):
        _normalize_stage_fit_controls(args)


def test_cluster_solver_rejects_resume_fast_outside_sequential() -> None:
    with pytest.raises(SystemExit, match="--resume fast"):
        cluster_solver._normalize_stage_fit_controls(
            argparse.Namespace(fit_mode=cluster_solver.FIT_MODE_EVIDENCE_NS, resume="fast")
        )


def _fast_solver_template(
    *,
    stage2_forward_mode: str = "none",
    quick_diagnostics: bool = False,
) -> LensClusterSolverConfig:
    if stage2_forward_mode == "none":
        schedule = StageScheduleConfig(
            fit_method=("svi",),
            svi_steps=(10, 10),
            refresh_every=(None, None),
            warmup=(1,),
            samples=(1,),
            sampling_refresh_runs=(1,),
            max_tree_depth=(1,),
        )
    else:
        schedule = StageScheduleConfig(
            fit_method=("svi", "svi"),
            svi_steps=(10, 10, 10),
            refresh_every=(None, None, None),
            warmup=(1, 1),
            samples=(1, 1),
            sampling_refresh_runs=(1, 1),
            max_tree_depth=(1, 1),
        )
    return LensClusterSolverConfig(
        runtime=RuntimeConfig(skip_plots=True, quick_diagnostics=quick_diagnostics),
        workflow=WorkflowConfig(stage2_forward_mode=stage2_forward_mode),
        schedule=schedule,
        likelihood=LikelihoodConfig(pos_sigma_arcsec=0.05),
        truth=TruthRecoveryConfig(
            posterior_truth_recovery_draws=4,
            caustic_plot_grid_scale_arcsec=0.2,
        ),
    )


def _small_mock_config(**updates: Any) -> SingleBCGMockConfig:
    return SingleBCGMockConfig(
        seed=12345,
        n_primary_families=1,
        n_subhalo_families=0,
        primary_source_redshifts=(1.5,),
        subhalo_source_redshifts=(1.5,),
        max_sources_to_try=20,
        **updates,
    )


def _mock_validation_config(
    tmp_path: Path,
    *,
    mock: SingleBCGMockConfig | None = None,
    runtime: mock_validation_api.MockValidationRuntimeConfig | None = None,
    solver_template: LensClusterSolverConfig | None = None,
) -> mock_validation_api.MockValidationConfig:
    return mock_validation_api.MockValidationConfig(
        mock=mock or _small_mock_config(),
        paths=mock_validation_api.MockValidationPathsConfig(output_dir=tmp_path, run_name="validation_log"),
        runtime=runtime or mock_validation_api.MockValidationRuntimeConfig(realizations=1, seed=12345),
        solver=mock_validation_api.MockValidationSolverConfig(
            template=solver_template or _fast_solver_template(),
            run_name="solver",
        ),
    )


def _write_minimal_mock_files(root: Path, *, with_members: bool = False) -> tuple[validation.MockClusterPaths, pd.DataFrame, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    paths = validation.MockClusterPaths(
        root=root,
        par_path=root / "single_bcg_mock.par",
        image_catalog_path=root / "obs_arcs.cat",
        truth_path=root / "truth.json",
        mock_images_path=root / "mock_images.json",
    )
    paths.par_path.write_text("mock par\n", encoding="utf-8")
    paths.image_catalog_path.write_text("#REFERENCE 3\n1.1 0.0 0.0 0.3 0.3 0.0 1.5 25.0\n", encoding="utf-8")
    truth = {
        "parameter_truth": {"halo.v_disp": 760.0, "bcg.v_disp": 285.0},
        "sources": [{"family_id": "1", "beta_x": 0.1, "beta_y": -0.2, "z_source": 1.5}],
        "subhalos": [],
    }
    paths.truth_path.write_text(json.dumps(truth), encoding="utf-8")
    paths.mock_images_path.write_text(json.dumps([{"family_id": "1", "image_label": "1.1"}]), encoding="utf-8")
    if with_members:
        (root / "members.cat").write_text("1 0.0 0.0 22.0\n", encoding="utf-8")
    images = pd.DataFrame(
        [
            {
                "family_id": "1",
                "image_label": "1.1",
                "x_obs_arcsec": 0.0,
                "y_obs_arcsec": 0.0,
            }
        ]
    )
    return paths, images, truth


def test_mock_validation_config_defaults_validate(tmp_path: Path) -> None:
    config = _mock_validation_config(tmp_path)

    assert config.validate() is config
    payload = config.to_json_dict()
    assert payload["paths"]["run_name"] == "validation_log"
    assert payload["paths"]["campaign_name"] is None
    assert payload["paths"]["variant_name"] is None
    assert "recovery" not in payload
    assert payload["solver"]["run_name"] == "solver"
    assert payload["solver"]["recovery_stages"] == ["stage2"]
    assert payload["solver"]["template"]["workflow"]["sampling_engine"] == "refreshing_surrogate_flat"


@pytest.mark.parametrize(
    ("make_config", "message"),
    [
        (lambda base: replace(base, runtime=replace(base.runtime, realizations=0)), "realizations"),
        (lambda base: replace(base, runtime=replace(base.runtime, seed=-1)), "runtime.seed"),
        (lambda base: replace(base, mock=replace(base.mock, seed=-1)), "seed"),
        (
            lambda base: replace(base, mock=replace(base.mock, n_primary_families=0, n_subhalo_families=0)),
            "at least one source family",
        ),
        (lambda base: replace(base, mock=replace(base.mock, min_images_per_family=1)), "min_images_per_family"),
        (
            lambda base: replace(base, mock=replace(base.mock, min_images_per_family=4, max_images_per_family=3)),
            "max_images_per_family",
        ),
        (
            lambda base: replace(
                base,
                solver=replace(
                    base.solver,
                    template=replace(base.solver.template, schedule=replace(base.solver.template.schedule, svi_steps=(10,))),
                ),
            ),
            "svi_steps",
        ),
        (
            lambda base: replace(
                base,
                paths=replace(base.paths, campaign_name=""),
            ),
            "paths.campaign_name",
        ),
        (
            lambda base: replace(
                base,
                paths=replace(base.paths, campaign_name="../bad"),
            ),
            "paths.campaign_name",
        ),
        (
            lambda base: replace(
                base,
                paths=replace(base.paths, variant_name="bad/name"),
            ),
            "paths.variant_name",
        ),
        (
            lambda base: replace(
                base,
                solver=replace(
                    base.solver,
                    template=replace(
                        base.solver.template,
                        image_diagnostics=replace(
                            base.solver.template.image_diagnostics,
                            posterior_image_diagnostic_mode="fast",
                        ),
                    ),
                ),
            ),
            "posterior_image_diagnostic_mode",
        ),
        (
            lambda base: replace(
                base,
                solver=replace(
                    base.solver,
                    template=replace(
                        base.solver.template,
                        truth=replace(base.solver.template.truth, posterior_truth_recovery_draws=0),
                    ),
                ),
            ),
            "posterior_truth_recovery_draws",
        ),
        (
            lambda base: replace(
                base,
                solver=replace(
                    base.solver,
                    template=replace(
                        base.solver.template,
                        truth=replace(base.solver.template.truth, caustic_plot_grid_scale_arcsec=0.0),
                    ),
                ),
            ),
            "caustic_plot_grid_scale_arcsec",
        ),
        (
            lambda base: replace(base, solver=replace(base.solver, recovery_stages=())),
            "solver.recovery_stages",
        ),
        (
            lambda base: replace(base, solver=replace(base.solver, recovery_stages=("stage1", "stage1"))),
            "solver.recovery_stages",
        ),
        (
            lambda base: replace(base, solver=replace(base.solver, recovery_stages=("final",))),
            "solver.recovery_stages",
        ),
    ],
)
def test_mock_validation_config_rejects_invalid_values(
    tmp_path: Path,
    make_config: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_config(_mock_validation_config(tmp_path)).validate()


def test_single_bcg_mock_lens_model_config_without_subhalos(tmp_path: Path) -> None:
    mock = _small_mock_config(n_subhalos=0, pos_sigma_arcsec=0.0)
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock")

    model = mock_validation_api.single_bcg_mock_lens_model_config(
        mock,
        paths,
        image_constraints_sigma_arcsec=0.05,
    )

    assert [halo.id for halo in model.large_halos] == ["halo", "bcg"]
    assert model.member_populations == ()
    assert model.image_constraints is not None
    assert Path(model.image_constraints.catalog_path) == paths.image_catalog_path
    assert model.image_constraints.sigma_arcsec == pytest.approx(0.05)
    assert model.large_halos[0].priors["v_disp"].kind == "uniform"
    for halo in model.large_halos:
        prior = halo.priors["cut_radius_kpc"]
        assert prior.kind == "uniform"
        assert prior.lower == pytest.approx(0.9 * halo.cut_radius_kpc)
        assert prior.upper == pytest.approx(1.1 * halo.cut_radius_kpc)


def test_single_bcg_mock_lens_model_config_with_subhalos(tmp_path: Path) -> None:
    mock = _small_mock_config(n_subhalos=2)
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock", with_members=True)

    model = mock_validation_api.single_bcg_mock_lens_model_config(
        mock,
        paths,
        image_constraints_sigma_arcsec=0.05,
    )

    assert len(model.member_populations) == 1
    population = model.member_populations[0]
    assert population.id == "potfile"
    assert Path(population.catalog_path) == paths.root / "members.cat"
    assert population.sigma_prior.kind == "normal"
    assert population.cutkpc_prior.kind == "normal"
    assert population.cutkpc_prior.mean == pytest.approx(population.cutkpc)
    assert population.cutkpc_prior.std == pytest.approx(0.25 * population.cutkpc)
    assert population.cutkpc_prior.lower == pytest.approx(0.25 * population.cutkpc)
    assert population.cutkpc_prior.upper == pytest.approx(4.0 * population.cutkpc)


def test_single_bcg_mock_lens_model_config_honors_nondefault_cut_radius_prior_width(
    tmp_path: Path,
) -> None:
    mock = _small_mock_config(
        n_subhalos=2,
        halo_cut_radius_prior_fractional_half_width=0.05,
        bcg_cut_radius_prior_fractional_half_width=0.05,
        subhalo_cut_radius_prior_fractional_half_width=0.05,
    )
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock", with_members=True)

    model = mock_validation_api.single_bcg_mock_lens_model_config(
        mock,
        paths,
        image_constraints_sigma_arcsec=0.05,
    )

    for halo in model.large_halos:
        prior = halo.priors["cut_radius_kpc"]
        assert prior.lower == pytest.approx(0.95 * halo.cut_radius_kpc)
        assert prior.upper == pytest.approx(1.05 * halo.cut_radius_kpc)
    population = model.member_populations[0]
    assert population.cutkpc_prior.mean == pytest.approx(population.cutkpc)
    assert population.cutkpc_prior.std == pytest.approx(0.05 * population.cutkpc)
    assert population.cutkpc_prior.lower == pytest.approx(0.05 * population.cutkpc)
    assert population.cutkpc_prior.upper == pytest.approx(20.0 * population.cutkpc)


def test_single_bcg_mock_lens_model_config_requires_positive_image_constraint_sigma(tmp_path: Path) -> None:
    mock = _small_mock_config(pos_sigma_arcsec=0.0)
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock")

    with pytest.raises(ValueError, match="image_constraints_sigma_arcsec"):
        mock_validation_api.single_bcg_mock_lens_model_config(
            mock,
            paths,
            image_constraints_sigma_arcsec=0.0,
        )


def test_solver_config_for_single_bcg_mock_sets_paths_seed_model_and_recovery_inputs(tmp_path: Path) -> None:
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock")
    config = _mock_validation_config(
        tmp_path,
        runtime=mock_validation_api.MockValidationRuntimeConfig(realizations=1, seed=11, resume="fast", quiet=True),
    )

    solver_config = mock_validation_api.solver_config_for_single_bcg_mock(
        config,
        paths=paths,
        seed=77,
        output_dir=tmp_path / "solver",
    )

    assert Path(solver_config.paths.output_dir) == tmp_path / "solver"
    assert solver_config.paths.run_name == "solver"
    assert solver_config.runtime.seed == 77
    assert solver_config.runtime.resume == "fast"
    assert solver_config.runtime.quiet is True
    assert solver_config.model is not None
    assert solver_config.model.image_constraints is not None
    assert Path(solver_config.model.image_constraints.catalog_path) == paths.image_catalog_path
    assert solver_config.model.image_constraints.sigma_arcsec == pytest.approx(0.05)
    assert planning.compile_run_plan(solver_config).stages[-1].name == "stage1_backprojected_centroid_fit"


def test_solver_config_for_single_bcg_mock_requires_solver_image_constraint_sigma(tmp_path: Path) -> None:
    paths, _images, _truth = _write_minimal_mock_files(tmp_path / "mock")
    solver_template = replace(_fast_solver_template(), likelihood=LikelihoodConfig(pos_sigma_arcsec=None))
    config = _mock_validation_config(tmp_path, solver_template=solver_template)

    with pytest.raises(ValueError, match="pos_sigma_arcsec is required"):
        mock_validation_api.solver_config_for_single_bcg_mock(
            config,
            paths=paths,
            seed=77,
            output_dir=tmp_path / "solver",
        )


def test_mock_validation_has_no_cli_or_legacy_runner_surface() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("lenscluster.mock_validation.cli")
    assert not hasattr(mock_validation_api, "ValidationStageFitControls")
    assert not hasattr(validation, "_build_parser")
    assert not hasattr(validation, "_run_cluster_solver")
    assert "subprocess" not in validation.__dict__


def test_validation_jsonable_sanitizes_structured_values(tmp_path: Path) -> None:
    converted = validation._validation_jsonable(
        {
            "path": tmp_path / "x",
            "array": np.asarray([1.0, np.nan, np.inf]),
            "scalar": np.float64(np.nan),
            "df": pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]}),
            "component": validation.DPIETruth(
                potential_id="halo",
                x_centre=0.0,
                y_centre=0.0,
                ellipticite=0.2,
                angle_pos=35.0,
                core_radius_arcsec=5.0,
                cut_radius_arcsec=220.0,
                v_disp=760.0,
            ),
        }
    )

    assert converted["path"] == str(tmp_path / "x")
    assert converted["array"] == [1.0, None, None]
    assert converted["scalar"] is None
    assert converted["df"]["records"][1]["a"] is None
    assert converted["component"]["potential_id"] == "halo"
    json.dumps(converted, allow_nan=False)


def test_write_validation_results_json_embeds_stage_table_artifacts(tmp_path: Path) -> None:
    config = replace(
        _mock_validation_config(tmp_path),
        paths=mock_validation_api.MockValidationPathsConfig(
            output_dir=tmp_path,
            campaign_name="campaign_a",
            run_name="validation_log",
            variant_name="anisotropic",
        ),
    )
    root = tmp_path / "campaign_a" / "validation_log"
    seed_dir = root / "seed_00012345"
    realization_dir = seed_dir / "anisotropic"
    paths, images, truth = _write_minimal_mock_files(seed_dir / "mock", with_members=True)
    solver_run_dir = realization_dir / "solver" / "stage1_backprojected_centroid_fit"
    tables_dir = solver_run_dir / "tables"
    tables_dir.mkdir(parents=True)
    (tables_dir / "run_summary.json").write_text(json.dumps({"fit_method": "svi", "n_images": 1}), encoding="utf-8")
    (tables_dir / "family_diagnostics.csv").write_text(
        "family_id,exact_image_rms_arcsec\n1,0.03\n",
        encoding="utf-8",
    )
    (tables_dir / "notes.txt").write_text("table notes", encoding="utf-8")
    summary_path = realization_dir / "run_summary.txt"
    summary_path.write_text("summary text", encoding="utf-8")
    (realization_dir / "run_debug.log").write_text("debug text", encoding="utf-8")

    path = validation.write_validation_results_json(
        config=config,
        seed=12345,
        realization_dir=realization_dir,
        mock_config=config.mock,
        paths=paths,
        images=images,
        truth_payload=truth,
        solver_run_dir=solver_run_dir,
        summary_path=summary_path,
        output_paths={"summary_plot": realization_dir / "validation_summary.pdf"},
        recovery_payload={
            "summary": {"median_abs_parameter_bias": np.nan},
            "tables": {"parameters": {"records": [{"parameter": "halo.v_disp", "bias": np.inf}]}},
        },
        recovery_stage="stage1",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload["solver"]["stage_manifests"][0]
    assert path == realization_dir / "results.json"
    assert payload["run"]["config"]["paths"]["run_name"] == "validation_log"
    assert payload["run"]["config"]["paths"]["campaign_name"] == "campaign_a"
    assert payload["run"]["config"]["paths"]["variant_name"] == "anisotropic"
    assert payload["mock_cluster"]["files"]["member_catalog_text"] == "1 0.0 0.0 22.0\n"
    assert payload["mock_cluster"]["truth"]["sources"][0]["family_id"] == "1"
    assert payload["validation"]["run_summary"]["text"] == "summary text"
    assert payload["validation"]["output_paths"]["results_json"] == str(path)
    assert payload["validation"]["recovery"]["stage1"]["summary"]["median_abs_parameter_bias"] is None
    assert payload["validation"]["recovery"]["stage1"]["tables"]["parameters"]["records"][0]["bias"] is None
    assert payload["debug_log"]["text"] == "debug text"
    assert stage["stage"] == "stage1_backprojected_centroid_fit"
    assert stage["table_artifacts"]["run_summary.json"]["data"]["fit_method"] == "svi"
    assert stage["table_artifacts"]["family_diagnostics.csv"]["data"]["records"][0]["family_id"] == 1
    assert stage["table_artifacts"]["notes.txt"]["text"] == "table notes"


class _FakeLensClusterRunner:
    def __init__(self) -> None:
        self.plans: list[Any] = []

    def run(self, plan: Any) -> None:
        self.plans.append(plan)
        root = Path(plan.output.output_dir) / str(plan.output.run_name)
        for stage in plan.stages:
            stage_dir = root / stage.name
            tables_dir = stage_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            (tables_dir / "run_summary.json").write_text(
                json.dumps({"fit_method": stage.fit_method, "n_images": 1}),
                encoding="utf-8",
            )
            (tables_dir / "family_diagnostics.csv").write_text(
                "family_id,exact_image_rms_arcsec\n1,0.03\n",
                encoding="utf-8",
            )


def test_run_single_bcg_validation_uses_config_runner_and_final_compiled_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recovery_calls: list[dict[str, Any]] = []

    def fake_generate(root: Path, mock_config: SingleBCGMockConfig, progress_callback=None):
        del progress_callback
        paths, images, truth = _write_minimal_mock_files(Path(root))
        assert mock_config.seed == 12345
        return paths, images, truth

    def fake_recovery(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir,
        posterior_image_diagnostic_draws,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        critical_caustic_plot_grid_scale_arcsec=validation.DEFAULT_CAUSTIC_GRID_SCALE_ARCSEC,
        posterior_truth_recovery_draws=validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        quick_diagnostics=False,
        progress_args=None,
        recovery_payload=None,
    ):
        recovery_calls.append(
            {
                "run_dir": Path(run_dir),
                "truth_path": Path(truth_path),
                "mock_images_path": Path(mock_images_path),
                "output_dir": Path(output_dir),
                "posterior_image_diagnostic_draws": posterior_image_diagnostic_draws,
                "posterior_diagnostic_mode": posterior_diagnostic_mode,
                "critical_caustic_plot_grid_scale_arcsec": critical_caustic_plot_grid_scale_arcsec,
                "posterior_truth_recovery_draws": posterior_truth_recovery_draws,
                "quick_diagnostics": quick_diagnostics,
                "progress_args": progress_args,
            }
        )
        if recovery_payload is not None:
            recovery_payload.update({"summary": {"output_dir": str(Path(output_dir))}, "tables": {}})
        return {"summary_plot": Path(output_dir) / "validation_summary.pdf"}

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fake_generate)
    monkeypatch.setattr(
        validation,
        "write_prefit_validation_diagnostics",
        lambda _truth, _images, output_dir: _expected_prefit_output_paths(Path(output_dir)),
    )
    monkeypatch.setattr(validation, "write_recovery_outputs", fake_recovery)
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    solver_template = replace(
        _fast_solver_template(stage2_forward_mode="linearized", quick_diagnostics=True),
        image_diagnostics=ImageDiagnosticsConfig(
            posterior_image_diagnostic_draws=3,
            posterior_image_diagnostic_mode="approximate",
        ),
        truth=TruthRecoveryConfig(
            posterior_truth_recovery_draws=5,
            caustic_plot_grid_scale_arcsec=0.4,
        ),
    )
    config = _mock_validation_config(
        tmp_path,
        solver_template=solver_template,
    )
    config = replace(
        config,
        solver=replace(config.solver, recovery_stages=("stage1", "stage2")),
    )
    fake_runner = _FakeLensClusterRunner()

    outputs = validation.run_single_bcg_validation(config, runner=fake_runner)
    validation._close_debug_log()

    root = tmp_path / "validation_log"
    seed_dir = root / "seed_00012345"
    realization_dir = seed_dir
    stage1_dir = realization_dir / "solver" / "stage1_backprojected_centroid_fit"
    stage2_dir = realization_dir / "solver" / "stage2_free_source_forward_fit"
    assert [stage.name for stage in fake_runner.plans[0].stages] == [
        "stage0_fast_initializer",
        "stage1_backprojected_centroid_fit",
        "stage2_free_source_forward_fit",
    ]
    assert [call["run_dir"] for call in recovery_calls] == [stage1_dir, stage2_dir]
    assert [call["output_dir"] for call in recovery_calls] == [
        realization_dir / "recovery" / "stage1_backprojected_centroid_fit",
        realization_dir / "recovery" / "stage2_free_source_forward_fit",
    ]
    assert recovery_calls[0]["posterior_image_diagnostic_draws"] == 3
    assert recovery_calls[0]["posterior_diagnostic_mode"] == "approximate"
    assert recovery_calls[0]["critical_caustic_plot_grid_scale_arcsec"] == pytest.approx(0.4)
    assert recovery_calls[0]["posterior_truth_recovery_draws"] == 5
    assert recovery_calls[0]["quick_diagnostics"] is True
    assert recovery_calls[0]["progress_args"] is config
    assert outputs == [
        {
            **_expected_prefit_output_paths(seed_dir / "prefit"),
            "stage1_summary_plot": realization_dir / "recovery" / "stage1_backprojected_centroid_fit" / "validation_summary.pdf",
            "stage1_results_json": realization_dir / "recovery" / "stage1_backprojected_centroid_fit" / "results.json",
            "stage2_summary_plot": realization_dir / "recovery" / "stage2_free_source_forward_fit" / "validation_summary.pdf",
            "stage2_results_json": realization_dir / "recovery" / "stage2_free_source_forward_fit" / "results.json",
            "results_json": realization_dir / "recovery" / "results.json",
        }
    ]
    assert not any("/fit/" in path.as_posix() for output in outputs for path in output.values())


def test_run_single_bcg_validation_resume_reuses_existing_mock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        _mock_validation_config(tmp_path),
        paths=mock_validation_api.MockValidationPathsConfig(
            output_dir=tmp_path,
            campaign_name="campaign_a",
            run_name="validation_log",
            variant_name="isotropic",
        ),
        runtime=mock_validation_api.MockValidationRuntimeConfig(realizations=1, seed=12345, resume="all"),
        solver=replace(_mock_validation_config(tmp_path).solver, recovery_stages=("stage1",)),
    )
    root = tmp_path / "campaign_a" / "validation_log"
    seed_dir = root / "seed_00012345"
    realization_dir = seed_dir / "isotropic"
    mock_dir = seed_dir / "mock"
    _write_minimal_mock_files(mock_dir)
    calls: list[str] = []

    def fail_generate(*_args, **_kwargs):
        raise AssertionError("generate_single_bcg_mock should not run in resume mode")

    monkeypatch.setattr(validation, "generate_single_bcg_mock", fail_generate)
    monkeypatch.setattr(
        validation,
        "write_prefit_validation_diagnostics",
        lambda _truth, _images, output_dir: _expected_prefit_output_paths(Path(output_dir)),
    )
    monkeypatch.setattr(
        validation,
        "write_recovery_outputs",
        lambda _run_dir, _truth_path, _mock_images_path, output_dir, **_kwargs: calls.append("recovery")
        or {"summary_plot": Path(output_dir) / "validation_summary.pdf"},
    )
    monkeypatch.setattr(
        validation,
        "write_validation_run_summary",
        lambda _solver_run_dir, _truth_path, output_dir, run_name, seed: Path(output_dir) / "run_summary.txt",
    )
    outputs = validation.run_single_bcg_validation(config, runner=_FakeLensClusterRunner())
    validation._close_debug_log()

    assert calls == ["recovery"]
    assert outputs[0]["results_json"] == realization_dir / "recovery" / "results.json"


def _touch_complete_stage(stage_dir: Path) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"")
    (stage_dir / "tables").mkdir(parents=True, exist_ok=True)
    (stage_dir / "tables" / "run_summary.json").write_text("{}", encoding="utf-8")


def _touch_stage1_summary(stage_dir: Path, map_values: dict[str, float] | None = None) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    payload = {"map_values": map_values or {}, "means": {}, "stds": {}}
    (stage_dir / "artifacts" / "stage1_prior_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _touch_artifact_checkpoint(stage_dir: Path) -> None:
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"")


def _write_fake_stage_tables(
    stage_dir: Path,
    *,
    fit_method: str,
    likelihood: str,
    sampler: str,
    runtime_sec: float,
    best_loglike: float,
    exact_values: list[float],
    source_values: list[float],
    approx_values: list[float],
) -> None:
    tables_dir = stage_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "fit_method": fit_method,
                "sample_likelihood_mode": likelihood,
                "sampler": sampler,
                "runtime_sec": runtime_sec,
                "best_loglike": best_loglike,
                "accept_prob_mean": 0.91,
                "divergence_count": 2,
                "mean_num_steps": 17.5,
                "n_families": len(exact_values),
                "n_images": 12,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "family_id": [str(index + 1) for index in range(len(exact_values))],
            "exact_image_rms_arcsec": exact_values,
            "source_plane_rms_arcsec": source_values,
            "approx_image_rms_arcsec": approx_values,
        }
    ).to_csv(tables_dir / "family_diagnostics.csv", index=False)


def test_save_artifacts_replaces_plot_bundle_atomically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    final_path = artifacts_dir / "plot_bundle.h5"
    final_path.write_bytes(b"old")

    def fake_write(path, *_args):
        Path(path).write_bytes(b"new")

    monkeypatch.setattr(cluster_solver, "_save_plot_bundle_h5", fake_write)

    cluster_solver._save_artifacts(
        artifacts_dir,
        SimpleNamespace(),
        argparse.Namespace(),
        np.asarray([], dtype=float),
        SimpleNamespace(),
    )

    assert final_path.read_bytes() == b"new"
    assert not (artifacts_dir / ".plot_bundle.h5.tmp").exists()


def test_save_artifacts_keeps_existing_plot_bundle_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    final_path = artifacts_dir / "plot_bundle.h5"
    final_path.write_bytes(b"old")

    def fake_write(path, *_args):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(cluster_solver, "_save_plot_bundle_h5", fake_write)

    with pytest.raises(RuntimeError, match="write failed"):
        cluster_solver._save_artifacts(
            artifacts_dir,
            SimpleNamespace(),
            argparse.Namespace(),
            np.asarray([], dtype=float),
            SimpleNamespace(),
        )

    assert final_path.read_bytes() == b"old"
    assert not (artifacts_dir / ".plot_bundle.h5.tmp").exists()


def test_validation_stage_recovery_metrics_collects_ordered_stage_summaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_root = tmp_path / "solver" / "fit"
    stage2_dir = solver_root / "stage2_free_source_forward_fit"
    stage1_dir = solver_root / "stage1_backprojected_centroid_fit"
    _write_fake_stage_tables(
        stage2_dir,
        fit_method="svi+nuts",
        likelihood="source",
        sampler="numpyro_nuts",
        runtime_sec=20.0,
        best_loglike=-2.0,
        exact_values=[0.4, np.nan],
        source_values=[0.2, 0.3],
        approx_values=[0.5, 0.7],
    )
    _write_fake_stage_tables(
        stage1_dir,
        fit_method="svi",
        likelihood="source",
        sampler="svi",
        runtime_sec=10.0,
        best_loglike=-3.0,
        exact_values=[0.8, 1.0],
        source_values=[0.6, 0.4],
        approx_values=[0.9, 1.1],
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"parameter_truth": {"p1": 1.0, "p2": 2.0}}), encoding="utf-8")

    def fake_load_plot_bundle(stage_dir: Path):
        if Path(stage_dir).name == "stage1_backprojected_centroid_fit":
            samples = np.asarray([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], dtype=float)
        else:
            samples = np.asarray([[1.5, 2.0], [1.5, 2.0], [1.5, 2.0]], dtype=float)
        state = SimpleNamespace(parameter_specs=[SimpleNamespace(name="p1"), SimpleNamespace(name="p2")])
        arrays = {"samples": samples, "best_fit": np.median(samples, axis=0)}
        return state, {}, arrays, {}

    monkeypatch.setattr(validation, "_load_plot_bundle", fake_load_plot_bundle)

    rows = validation._collect_validation_stage_recovery_metrics(stage2_dir, truth_path)

    assert [row["stage"] for row in rows] == [
        "stage1_backprojected_centroid_fit",
        "stage2_free_source_forward_fit",
    ]
    assert rows[0]["exact_image_rms_mean"] == pytest.approx(0.9)
    assert rows[1]["failed_or_missing_exact"] == 1
    assert rows[0]["truth_parameter_count"] == 2
    assert rows[0]["parameter_mean_abs_bias"] == pytest.approx(0.5)
    assert rows[0]["worst_parameter"] == "p2"


def test_validation_stage_recovery_metrics_accepts_single_direct_solver_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_root = tmp_path / "solver" / "fit"
    _write_fake_stage_tables(
        solver_root,
        fit_method="ns",
        likelihood=SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE,
        sampler="numpyro_jaxns",
        runtime_sec=30.0,
        best_loglike=-5.0,
        exact_values=[0.2],
        source_values=[0.1],
        approx_values=[0.3],
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"parameter_truth": {}}), encoding="utf-8")
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _stage_dir: (SimpleNamespace(parameter_specs=[]), {}, {"samples": np.empty((1, 0)), "best_fit": np.empty((0,))}, {}),
    )

    rows = validation._collect_validation_stage_recovery_metrics(solver_root, truth_path)

    assert [row["stage"] for row in rows] == ["fit"]
    assert rows[0]["fit_method"] == "ns"
    assert rows[0]["sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE


def test_validation_run_summary_text_formats_metrics_and_na_values(tmp_path: Path) -> None:
    rows = [
        {
            "stage": "stage1_backprojected_centroid_fit",
            "fit_method": "svi",
            "sample_likelihood_mode": "source",
            "sampler": "svi",
            "runtime_sec": 10.0,
            "best_loglike": -3.0,
            "accept_prob_mean": np.nan,
            "divergence_count": np.nan,
            "mean_num_steps": np.nan,
            "n_families": 2,
            "n_images": 12,
            "family_count": 2,
            "exact_family_count": 2,
            "failed_or_missing_exact": 0,
            "exact_image_rms_mean": 0.9,
            "exact_image_rms_median": 0.9,
            "source_rms_mean": 0.5,
            "approx_image_rms_mean": 1.0,
            "truth_parameter_count": 2,
            "parameter_median_abs_bias": 0.5,
            "parameter_mean_abs_bias": 0.5,
            "parameter_coverage_68_fraction": 0.5,
            "worst_parameter": "p2",
            "worst_parameter_abs_bias": 1.0,
        },
        {
            "stage": "stage2_free_source_forward_fit",
            "fit_method": "svi+nuts",
            "sample_likelihood_mode": "source",
            "sampler": "numpyro_nuts",
            "runtime_sec": 20.0,
            "best_loglike": -2.0,
            "accept_prob_mean": 0.91,
            "divergence_count": 2,
            "mean_num_steps": 17.5,
            "n_families": 2,
            "n_images": 12,
            "family_count": 2,
            "exact_family_count": 1,
            "failed_or_missing_exact": 1,
            "exact_image_rms_mean": 0.4,
            "exact_image_rms_median": 0.4,
            "source_rms_mean": 0.25,
            "approx_image_rms_mean": 0.6,
            "truth_parameter_count": np.nan,
            "parameter_median_abs_bias": np.nan,
            "parameter_mean_abs_bias": np.nan,
            "parameter_coverage_68_fraction": np.nan,
            "worst_parameter": "na",
            "worst_parameter_abs_bias": np.nan,
        },
    ]

    text = validation._format_validation_run_summary(
        rows,
        run_name="validation_log",
        seed=12345,
        solver_run_dir=tmp_path / "solver" / "fit" / "stage2_free_source_forward_fit",
    )

    assert "run_name=validation_log" in text
    assert "final_stage=stage2_free_source_forward_fit" in text
    assert "exact_image_rms_mean" in text
    assert "source_rms_mean" in text
    assert "param_med_abs_bias" in text
    assert "stage1_backprojected_centroid_fit" in text
    assert "stage2_free_source_forward_fit" in text
    assert " na " in text
    assert "worst_parameter=p2" in text


def test_solver_sequential_run_summary_txt_aggregates_existing_stages(tmp_path: Path) -> None:
    root = tmp_path / "mock_run"
    stage1 = root / "stage1_backprojected_centroid_fit"
    stage2 = root / "stage2_free_source_forward_fit"
    missing_stage = root / "stage0_fast_initializer"
    for stage, headline_chi_square, arc_chi_square in [(stage1, 10.0, 8.0), (stage2, 4.0, 3.0)]:
        tables_dir = stage / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "run_summary.json").write_text(
            json.dumps(
                    {
                        "fit_method": "svi+nuts",
                        "sample_likelihood_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
                        "sampler": "numpyro_nuts",
                    "n_families": 2,
                    "n_images": 6,
                    "headline_chi_square": headline_chi_square,
                    "headline_dof": 3,
                    "headline_reduced_chi_square": headline_chi_square / 3.0,
                    "arc_aware_chi_square": arc_chi_square,
                    "arc_aware_dof": 4,
                    "arc_aware_reduced_chi_square": arc_chi_square / 4.0,
                    "arc_aware_arc_supported_image_count": 1,
                    "arc_aware_missing_image_count": 2,
                    "ess_min": 12.0,
                    "rhat_max": 1.01,
                    "runtime_sec": 20.0,
                }
            ),
            encoding="utf-8",
        )

    path, text = cluster_solver._write_sequential_run_summary_txt(
        root,
        "mock_run",
        [stage1, missing_stage, stage2],
    )

    assert path == root / "run_summary.txt"
    assert path.exists()
    assert "Sequential Cluster Solver Run Summary" in text
    assert "Stage Quality Comparison" in text
    assert "stage1_backprojected_centroid_fit" in text
    assert "stage2_free_source_forward_fit" in text
    assert "stage0_fast_initializer" not in text
    assert "headline_chi2" in text
    assert "arc_chi2" in text
    assert "AIC" not in text
    assert "BIC" not in text


def test_plot_run_summary_includes_nested_sampling_evidence() -> None:
    args = argparse.Namespace(
        run_name="ns_run",
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        skip_stage3_image_plane_local_jacobian=False,
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        sampling_engine="full_flat",
        nuts_init_boundary_frac=0.02,
        nuts_init_jitter_frac=0.02,
        image_plane_scatter_floor_arcsec=0.05,
        image_plane_scatter_prior="lognormal",
        image_plane_scatter_prior_median_arcsec=0.25,
        image_plane_scatter_prior_log_sigma=0.4,
        svi_steps=10,
        svi_learning_rate=0.005,
        ns_num_live_points=200,
        ns_max_samples=None,
        ns_dlogz=0.01,
        warmup=0,
        samples=25,
        chains=4,
        thin=1,
        max_tree_depth=8,
        target_accept=0.85,
        seed=123,
    )
    threshold_spec = cluster_solver._build_critical_arc_singular_threshold_parameter_spec(
        start_index=0,
        lower=0.03,
        upper=0.40,
        prior_median=0.15,
        prior_log_sigma=0.5,
    )
    softness_spec = cluster_solver._build_critical_arc_singular_softness_parameter_spec(
        start_index=1,
        lower=0.005,
        upper=0.20,
        prior_median=0.05,
        prior_log_sigma=0.5,
    )
    state = SimpleNamespace(
        run_name="ns_run",
        par_path="mock.par",
        fit_mode="joint",
        parameter_specs=[threshold_spec, softness_spec],
        family_data=[],
        packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
        potfiles=[],
        geometry_cache=None,
        cosmo_config={"H0": 70.0},
        fit_cosmology_flat_wcdm=False,
    )
    evaluator = SimpleNamespace(
        active_scaling_galaxies_by_potfile=[],
        active_scaling_component_indices=[],
        inactive_scaling_component_indices=[],
        requested_active_scaling_by_potfile={},
        actual_active_scaling_by_potfile={},
        total_scaling_by_potfile={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        eval_wall_times=[],
        timing_totals={},
        surrogate_enabled=False,
        approximate_eval_count=0,
        full_refresh_count=0,
    )
    posterior = PosteriorResults(
        samples=np.asarray([[0.05, 0.02], [0.10, 0.05], [0.20, 0.10]], dtype=float),
        log_prob=np.asarray([-3.0, -2.0, -1.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=25,
        num_chains=0,
        sampler="numpyro_jaxns",
        init_diagnostics={
            "strategy_requested": "ns",
            "strategy_used": "ns",
            "svi_used": False,
            "ns_num_live_points": 200,
            "ns_max_samples": None,
            "ns_dlogz": 0.01,
            "ns_log_z_mean": -12.5,
            "ns_log_z_uncert": 0.2,
            "ns_ess": 42.0,
            "ns_total_num_samples": 321,
            "ns_posterior_samples": 4096,
            "ns_posterior_resampling": "jaxns.get_samples",
            "ns_total_num_likelihood_evaluations": 1234,
            "ns_termination_reason": 1,
        },
    )

    summary = _run_summary(args, state, 1.5, posterior, -3.0, evaluator)

    assert summary["sampler"] == "numpyro_jaxns"
    assert summary["ns_settings"] == {"num_live_points": 200, "max_samples": None, "dlogz": 0.01}
    assert summary["ns_log_z_mean"] == pytest.approx(-12.5)
    assert summary["ns_log_z_uncert"] == pytest.approx(0.2)
    assert summary["ns_posterior_samples"] == 4096
    assert summary["ns_posterior_resampling"] == "jaxns.get_samples"
    assert summary["ns_total_num_likelihood_evaluations"] == 1234
    assert summary["likelihood_stabilizer_max_gain"] == pytest.approx(0.0)
    assert summary["likelihood_stabilizer_max_residual_arcsec"] == pytest.approx(0.0)
    assert summary["likelihood_stabilizer_residual_loss"] == "gaussian"
    assert summary["likelihood_stabilizer_student_t_nu"] == pytest.approx(4.0)
    assert summary["image_plane_scatter_floor_arcsec"] == pytest.approx(0.05)
    assert summary["image_plane_scatter_prior"] == "lognormal"
    assert summary["image_plane_scatter_prior_median_arcsec"] == pytest.approx(0.25)
    assert summary["image_plane_scatter_prior_log_sigma"] == pytest.approx(0.4)
    assert summary["critical_arc_singular_threshold_sampled"] is True
    assert summary["critical_arc_singular_threshold_posterior"]["median"] == pytest.approx(0.10)
    assert summary["critical_arc_singular_threshold_posterior"]["lower"] == pytest.approx(0.03)
    assert summary["critical_arc_singular_threshold_posterior"]["upper"] == pytest.approx(0.40)
    assert summary["critical_arc_singular_softness_sampled"] is True
    assert summary["critical_arc_singular_softness_posterior"]["median"] == pytest.approx(0.05)
    assert summary["critical_arc_singular_softness_posterior"]["lower"] == pytest.approx(0.005)
    assert summary["critical_arc_singular_softness_posterior"]["upper"] == pytest.approx(0.20)
    assert "linearized_image_plane_max_gain" not in summary
    assert "linearized_image_plane_max_residual_arcsec" not in summary
    assert "linearized_image_plane_residual_loss" not in summary
    assert "linearized_image_plane_student_t_nu" not in summary


def test_posterior_logprob_matrix_batches_without_changing_values() -> None:
    spec = ParameterSpec(
        name="x",
        sample_name="x",
        potential_id="mock",
        profile_type=81,
        field="x",
        prior_kind="uniform",
        lower=-2.0,
        upper=2.0,
        step=0.1,
    )
    evaluator = SimpleNamespace(_source_loglike_fn=lambda theta: -0.5 * jnp.square(theta[0]))
    samples = np.linspace(-1.0, 1.0, 5, dtype=float).reshape(-1, 1)

    batched = cluster_solver._posterior_logprob_matrix([spec], evaluator, samples, batch_size=2)
    unbatched = cluster_solver._posterior_logprob_matrix([spec], evaluator, samples, batch_size=10)

    np.testing.assert_allclose(batched, unbatched)


def test_smc_normalized_coordinate_round_trip_and_prior_particles() -> None:
    specs = [
        ParameterSpec(
            name="u",
            sample_name="u",
            potential_id="mock",
            profile_type=81,
            field="u",
            prior_kind="uniform",
            lower=-2.0,
            upper=3.0,
            step=0.1,
        ),
        ParameterSpec(
            name="n",
            sample_name="n",
            potential_id="mock",
            profile_type=81,
            field="n",
            prior_kind="normal",
            lower=float("-inf"),
            upper=float("inf"),
            step=0.1,
            mean=10.0,
            std=2.0,
        ),
        ParameterSpec(
            name="t",
            sample_name="t",
            potential_id="mock",
            profile_type=81,
            field="t",
            prior_kind="truncated_normal",
            lower=0.0,
            upper=2.0,
            step=0.1,
            mean=1.0,
            std=0.3,
        ),
    ]
    normalization = cluster_solver._smc_normalization_arrays(specs)
    theta = jnp.asarray([0.25, 11.0, 1.2], dtype=jnp.float64)

    normalized = cluster_solver._smc_theta_to_normalized(theta, normalization)
    round_tripped = cluster_solver._smc_normalized_to_theta(normalized, normalization)
    particles = cluster_solver._smc_prior_particles(jax.random.PRNGKey(11), specs, 7)

    np.testing.assert_allclose(np.asarray(round_tripped), np.asarray(theta), rtol=1.0e-9, atol=1.0e-9)
    assert particles.shape == (7, 3)
    assert np.isfinite(np.asarray(particles)).all()


def _synthetic_microcanonical_runner_inputs(
    *,
    chains: int = 3,
    quiet: bool = True,
    warmup: int = 0,
    samples: int = 2,
    output_dir: str = ".",
    debug_sampler_diagnostics: bool = False,
) -> tuple[argparse.Namespace, SimpleNamespace, SimpleNamespace]:
    specs = [
        ParameterSpec("x", "x", "mock", 0, "x", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
        ParameterSpec("y", "y", "mock", 0, "y", "normal", -np.inf, np.inf, 0.1, mean=0.0, std=1.0),
    ]
    args = argparse.Namespace(
        seed=17,
        chains=chains,
        warmup=warmup,
        samples=samples,
        thin=1,
        quiet=quiet,
        output_dir=output_dir,
        debug_sampler_diagnostics=debug_sampler_diagnostics,
        jax_default_device="cpu",
        nuts_init_jitter_frac=0.0,
        nuts_init_boundary_frac=0.02,
        microcanonical_diagonal_preconditioning=True,
        microcanonical_tune_frac1=0.1,
        microcanonical_tune_frac2=0.1,
        microcanonical_tune_frac3=0.1,
        mclmc_desired_energy_var=5.0e-4,
        mclmc_trust_in_estimate=1.5,
        mclmc_num_effective_samples=20,
        mclmc_lfactor=0.4,
        mchmc_target_accept=0.9,
        mchmc_random_trajectory_length=False,
        mchmc_l_proposal_factor=float("inf"),
        mchmc_divergence_threshold=1000.0,
        mchmc_num_windows=1,
        mchmc_tuning_factor=1.3,
        mchmc_l_estimator="avg",
    )
    state = SimpleNamespace(run_name="run", parameter_specs=specs, svi_init_values={"x": 0.0, "y": 0.0})
    evaluator = SimpleNamespace(
        timing_totals={},
        invalid_state_rejection_count=0,
        invalid_state_reason_counts={},
        surrogate_enabled=False,
        _source_loglike_fn=lambda theta: -0.5 * jnp.sum(jnp.square(theta)),
    )
    return args, state, evaluator


def _install_synthetic_microcanonical_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster_solver, "_blackjax_microcanonical_components", lambda: (object(), object(), object(), object()))

    def fake_run_chain(
        _args,
        _method,
        chain_index,
        _chain_seed,
        _logdensity_fn,
        *,
        blackjax,
        mclmc_module,
        adjusted_dynamic,
        integrators,
        progress_reporter=None,
    ):
        if progress_reporter is not None:
            progress_reporter.advance_warmup(chain_index)
            progress_reporter.advance_production_to(chain_index, int(_args.samples))
        samples = np.asarray(
            [
                [10.0 * chain_index + 1.0, 10.0 * chain_index + 2.0],
                [10.0 * chain_index + 3.0, 10.0 * chain_index + 4.0],
            ],
            dtype=float,
        )
        info = {
            "state_logdensity": np.asarray([-float(chain_index), -float(chain_index) - 0.1], dtype=float),
            "accept_prob": np.full(2, 0.8 + 0.01 * chain_index, dtype=float),
            "diverging": np.zeros(2, dtype=bool),
            "num_steps": np.full(2, chain_index + 1.0, dtype=float),
            "energy": np.full(2, chain_index + 0.5, dtype=float),
            "is_accepted": np.ones(2, dtype=bool),
            "nonans": np.ones(2, dtype=bool),
        }
        diagnostics = {
            "chain_index": int(chain_index),
            "chain_seed_label": f"fake_chain_{chain_index}",
            "microcanonical_L": float(chain_index + 1.0),
            "microcanonical_step_size": float(chain_index + 0.1),
            "microcanonical_inverse_mass_matrix": np.ones(2, dtype=float),
            "microcanonical_tuning_steps": 0,
            "microcanonical_tuning_skipped": True,
            "microcanonical_tuning_runtime_sec": float(chain_index + 0.25),
            "microcanonical_production_runtime_sec": float(chain_index + 0.5),
        }
        return samples, info, diagnostics

    monkeypatch.setattr(cluster_solver, "_run_microcanonical_chain", fake_run_chain)


def test_microcanonical_progress_reporter_clamps_delta_updates() -> None:
    events: list[tuple[int, int]] = []

    class FakeProgress:
        def advance(self, task_id: int, advance: int = 1) -> None:
            events.append((task_id, advance))

    reporter = cluster_solver._MicrocanonicalProgressReporter(
        FakeProgress(),
        overall_task=11,
        chain_task=12,
        chains=1,
        warmup=5,
        samples=3,
    )

    reporter.advance_warmup_to(0, 2)
    reporter.advance_warmup_to(0, 2)
    reporter.advance_warmup_to(0, 1)
    reporter.advance_warmup_to(0, 99)
    reporter.advance_production_to(0, 1)
    reporter.advance_production_to(0, 1)
    reporter.advance_production_to(0, 0)
    reporter.advance_production_to(0, 99)
    reporter.advance_chain_complete()

    assert events == [
        (11, 2),
        (11, 3),
        (11, 1),
        (11, 2),
        (12, 1),
    ]


def test_blackjax_microcanonical_production_progress_callback_chunks_exactly() -> None:
    args = argparse.Namespace(
        seed=17,
        warmup=0,
        samples=100,
        quiet=False,
        microcanonical_diagonal_preconditioning=True,
        microcanonical_tune_frac1=0.1,
        microcanonical_tune_frac2=0.1,
        microcanonical_tune_frac3=0.1,
        mclmc_desired_energy_var=5.0e-4,
        mclmc_trust_in_estimate=1.5,
        mclmc_num_effective_samples=20,
        mclmc_lfactor=0.4,
        mchmc_target_accept=0.9,
        mchmc_random_trajectory_length=False,
        mchmc_l_proposal_factor=float("inf"),
        mchmc_divergence_threshold=1000.0,
        mchmc_num_windows=1,
        mchmc_tuning_factor=1.3,
        mchmc_l_estimator="avg",
    )
    events: list[tuple[int, int]] = []

    class FakeProgress:
        def advance(self, task_id: int, advance: int = 1) -> None:
            events.append((task_id, advance))

    reporter = cluster_solver._MicrocanonicalProgressReporter(
        FakeProgress(),
        overall_task=1,
        chain_task=2,
        chains=1,
        warmup=0,
        samples=100,
    )
    blackjax, mclmc_module, adjusted_dynamic, integrators = cluster_solver._blackjax_microcanonical_components()

    def logdensity(theta):
        return -0.5 * jnp.sum(jnp.square(theta))

    samples, _info, _diagnostics = cluster_solver._run_microcanonical_chain(
        args,
        cluster_solver.FIT_METHOD_MCLMC,
        0,
        ChainSeed(values=np.asarray([0.1, -0.2], dtype=float), source_label="seed"),
        logdensity,
        blackjax=blackjax,
        mclmc_module=mclmc_module,
        adjusted_dynamic=adjusted_dynamic,
        integrators=integrators,
        progress_reporter=reporter,
    )

    assert samples.shape == (100, 2)
    assert sum(advance for task_id, advance in events if task_id == 1) == 100
    assert all(advance == 2 for task_id, advance in events if task_id == 1)
    assert not any(task_id == 2 for task_id, _advance in events)


@pytest.mark.parametrize("fit_method", [cluster_solver.FIT_METHOD_MCLMC, cluster_solver.FIT_METHOD_MCHMC])
def test_blackjax_microcanonical_warmup_progress_callbacks_fill_warmup(
    fit_method: str,
) -> None:
    args = argparse.Namespace(
        seed=17,
        warmup=20,
        samples=2,
        quiet=False,
        microcanonical_diagonal_preconditioning=True,
        microcanonical_tune_frac1=0.1,
        microcanonical_tune_frac2=0.1,
        microcanonical_tune_frac3=0.1,
        mclmc_desired_energy_var=5.0e-4,
        mclmc_trust_in_estimate=1.5,
        mclmc_num_effective_samples=20,
        mclmc_lfactor=0.4,
        mchmc_target_accept=0.9,
        mchmc_random_trajectory_length=False,
        mchmc_l_proposal_factor=float("inf"),
        mchmc_divergence_threshold=1000.0,
        mchmc_num_windows=1,
        mchmc_tuning_factor=1.3,
        mchmc_l_estimator="avg",
    )
    events: list[tuple[int, int]] = []

    class FakeProgress:
        def advance(self, task_id: int, advance: int = 1) -> None:
            events.append((task_id, advance))

    reporter = cluster_solver._MicrocanonicalProgressReporter(
        FakeProgress(),
        overall_task=1,
        chain_task=2,
        chains=1,
        warmup=20,
        samples=2,
    )
    blackjax, mclmc_module, adjusted_dynamic, integrators = cluster_solver._blackjax_microcanonical_components()

    def logdensity(theta):
        return -0.5 * jnp.sum(jnp.square(theta))

    samples, _info, _diagnostics = cluster_solver._run_microcanonical_chain(
        args,
        fit_method,
        0,
        ChainSeed(values=np.asarray([0.1, -0.2], dtype=float), source_label="seed"),
        logdensity,
        blackjax=blackjax,
        mclmc_module=mclmc_module,
        adjusted_dynamic=adjusted_dynamic,
        integrators=integrators,
        progress_reporter=reporter,
    )

    assert samples.shape == (2, 2)
    overall_advances = [advance for task_id, advance in events if task_id == 1]
    assert sum(overall_advances) == 22
    assert max(overall_advances) < 20
    assert not any(task_id == 2 for task_id, _advance in events)


def test_select_best_fit_from_posterior_defaults_to_map_and_records_max_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = PosteriorResults(
        samples=np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        log_prob=np.asarray([-3.0, -1.0, -2.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={},
    )
    monkeypatch.setattr(cluster_solver, "_source_loglike_matrix", lambda _evaluator, _samples: np.asarray([0.0, 1.0, 4.0]))

    selected = cluster_solver._select_best_fit_from_posterior(
        argparse.Namespace(best_value=cluster_solver.BEST_VALUE_MAP, quiet=True),
        object(),
        posterior,
        np.asarray([-99.0], dtype=float),
    )

    np.testing.assert_allclose(selected, [1.0])
    diagnostics = posterior.init_diagnostics
    assert diagnostics["best_value_selected"] == cluster_solver.BEST_VALUE_MAP
    assert diagnostics["best_value_selected_sample_index"] == 1
    assert diagnostics["map_sample_index"] == 1
    assert diagnostics["maximum_likelihood_sample_index"] == 2
    assert diagnostics["maximum_likelihood_source_loglike"] == pytest.approx(4.0)


def test_select_best_fit_from_posterior_can_choose_maximum_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = PosteriorResults(
        samples=np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        log_prob=np.asarray([-3.0, -1.0, -2.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={},
    )
    monkeypatch.setattr(cluster_solver, "_source_loglike_matrix", lambda _evaluator, _samples: np.asarray([0.0, 1.0, 4.0]))

    selected = cluster_solver._select_best_fit_from_posterior(
        argparse.Namespace(best_value=cluster_solver.BEST_VALUE_MAXIMUM_LIKELIHOOD, quiet=True),
        object(),
        posterior,
        np.asarray([-99.0], dtype=float),
    )

    np.testing.assert_allclose(selected, [2.0])
    diagnostics = posterior.init_diagnostics
    assert diagnostics["best_value_selected"] == cluster_solver.BEST_VALUE_MAXIMUM_LIKELIHOOD
    assert diagnostics["best_value_selected_sample_index"] == 2
    assert diagnostics["best_value_selected_source_loglike"] == pytest.approx(4.0)


def test_select_best_fit_from_posterior_can_choose_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = PosteriorResults(
        samples=np.asarray([[0.0, 10.0], [2.0, 20.0], [100.0, 30.0]], dtype=float),
        log_prob=np.asarray([-3.0, -1.0, -2.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
        init_diagnostics={},
    )
    monkeypatch.setattr(
        cluster_solver,
        "_source_loglike_matrix",
        lambda _evaluator, _samples: np.asarray([0.0, 1.0, 4.0]),
    )

    class Evaluator:
        def source_loglike(self, theta: np.ndarray) -> float:
            return -float(np.sum((np.asarray(theta, dtype=float) - np.asarray([2.0, 20.0])) ** 2))

    selected = cluster_solver._select_best_fit_from_posterior(
        argparse.Namespace(best_value=cluster_solver.BEST_VALUE_MEDIAN, quiet=True),
        Evaluator(),
        posterior,
        np.asarray([-99.0, -99.0], dtype=float),
    )

    np.testing.assert_allclose(selected, [2.0, 20.0])
    np.testing.assert_allclose(posterior.median_fit, [2.0, 20.0])
    diagnostics = posterior.init_diagnostics
    assert diagnostics["best_value_selected"] == cluster_solver.BEST_VALUE_MEDIAN
    assert diagnostics["best_value_selected_sample_index"] is None
    assert diagnostics["best_value_selected_log_prob"] is None
    assert diagnostics["best_value_selected_source_loglike"] == pytest.approx(0.0)
    assert diagnostics["median_sample_index"] is None
    assert diagnostics["median_sample_log_prob"] is None


def test_select_best_fit_from_posterior_falls_back_when_requested_candidate_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = PosteriorResults(
        samples=np.asarray([[0.0], [1.0]], dtype=float),
        log_prob=np.asarray([-2.0, -1.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=2,
        num_chains=1,
        init_diagnostics={},
    )
    monkeypatch.setattr(cluster_solver, "_source_loglike_matrix", lambda _evaluator, _samples: np.asarray([np.nan, np.nan]))

    selected = cluster_solver._select_best_fit_from_posterior(
        argparse.Namespace(best_value=cluster_solver.BEST_VALUE_MAXIMUM_LIKELIHOOD, quiet=True),
        object(),
        posterior,
        np.asarray([-99.0], dtype=float),
    )

    np.testing.assert_allclose(selected, [1.0])
    diagnostics = posterior.init_diagnostics
    assert diagnostics["best_value_requested"] == cluster_solver.BEST_VALUE_MAXIMUM_LIKELIHOOD
    assert diagnostics["best_value_selected"] == cluster_solver.BEST_VALUE_MAP
    assert diagnostics["best_value_fallback_reason"] == "requested_maximum-likelihood_unavailable"


def test_select_best_fit_from_posterior_falls_back_when_median_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = PosteriorResults(
        samples=np.asarray([[0.0, np.nan], [1.0, np.nan]], dtype=float),
        log_prob=np.asarray([-2.0, -1.0], dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=2,
        num_chains=1,
        init_diagnostics={},
    )
    monkeypatch.setattr(cluster_solver, "_source_loglike_matrix", lambda _evaluator, _samples: np.asarray([np.nan, np.nan]))

    selected = cluster_solver._select_best_fit_from_posterior(
        argparse.Namespace(best_value=cluster_solver.BEST_VALUE_MEDIAN, quiet=True),
        object(),
        posterior,
        np.asarray([-99.0, -99.0], dtype=float),
    )

    np.testing.assert_allclose(selected, [1.0, np.nan], equal_nan=True)
    diagnostics = posterior.init_diagnostics
    assert diagnostics["best_value_requested"] == cluster_solver.BEST_VALUE_MEDIAN
    assert diagnostics["best_value_selected"] == cluster_solver.BEST_VALUE_MAP
    assert diagnostics["best_value_fallback_reason"] == "requested_median_unavailable"
    assert posterior.median_fit is None


def _expected_prefit_output_paths(realization_dir: Path) -> dict[str, Path]:
    return {
        "subhalo_shmf_plot": realization_dir / "subhalo_shmf.pdf",
        "prefit_subhalo_spatial_distribution_plot": realization_dir / "prefit_subhalo_spatial_distribution.pdf",
        "prefit_critical_lines_plot": realization_dir / "prefit_critical_lines.pdf",
    }


def _source_plane_loglike_common(**updates) -> dict[str, Any]:
    payload = dict(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        inv_abs_mu=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )
    payload.update(updates)
    return payload


def test_source_plane_loglike_matches_default_gaussian_weighted_centroid() -> None:
    value = _source_plane_bin_loglike(**_source_plane_loglike_common())

    expected = 2.0 * (-0.5 * (1.0 + np.log((2.0 * np.pi) ** 2)))
    np.testing.assert_allclose(float(value), expected, rtol=0.0, atol=5.0e-6)


def test_source_plane_max_gain_floors_near_critical_magnification_weight() -> None:
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        inv_abs_mu=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _source_plane_bin_loglike(**common)
    stabilized = _source_plane_bin_loglike(max_gain=50.0, **common)

    assert np.isfinite(float(stabilized))
    assert float(stabilized) > float(raw)


def test_source_plane_smooth_residual_cap_bounds_large_residual() -> None:
    capped_x, capped_y = _smooth_residual_cap(
        jnp.asarray([100.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        max_residual_arcsec=3.0,
    )
    radius = float(np.hypot(float(np.asarray(capped_x)[0]), float(np.asarray(capped_y)[0])))
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 200.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _source_plane_bin_loglike(**common)
    capped = _source_plane_bin_loglike(max_residual_arcsec=3.0, **common)

    assert radius <= 3.0 + 1.0e-10
    assert radius > 2.9
    assert np.isfinite(float(capped))
    assert float(capped) > float(raw)


def test_source_plane_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = _source_plane_loglike_common(
        beta_x=jnp.asarray([0.0, 10.0], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        outlier_sigma_arcsec=1.0e-3,
    )

    gaussian = _source_plane_bin_loglike(residual_loss="gaussian", **common)
    student_t = _source_plane_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_local_jacobian_loglike_matches_diagonal_weighted_centroid() -> None:
    value = _local_jacobian_bin_loglike(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )

    expected = 2.0 * (-0.5 * (1.0 + np.log((2.0 * np.pi) ** 2)))
    np.testing.assert_allclose(float(value), expected, rtol=0.0, atol=5.0e-6)


def test_local_jacobian_loglike_changes_with_off_diagonal_covariance() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=100.0,
    )
    diagonal = _local_jacobian_bin_loglike(jac_a01=jnp.zeros(2, dtype=jnp.float64), **common)
    sheared = _local_jacobian_bin_loglike(jac_a01=jnp.asarray([0.5, 0.5], dtype=jnp.float64), **common)

    assert np.isfinite(float(sheared))
    assert not np.isclose(float(diagonal), float(sheared))


def test_jittered_2x2_covariance_det_stabilizes_rank_one_covariance() -> None:
    c00, c11, det = _jittered_2x2_covariance_det(
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
    )
    inv00 = c11 / det
    inv11 = c00 / det
    inv01 = -jnp.asarray([1.0], dtype=jnp.float64) / det

    assert float(c00[0]) > 1.0
    assert float(c11[0]) > 1.0
    assert float(det[0]) > cluster_solver.MIN_COVARIANCE_DETERMINANT_GUARD
    assert np.isfinite(float(jnp.log(det[0])))
    assert np.all(np.isfinite(np.asarray([inv00[0], inv01[0], inv11[0]])))


class _TinyCriticalDetEvaluator:
    def __init__(self, det_values: list[float]) -> None:
        self.det_values = np.asarray(det_values, dtype=float)
        self.fit_cosmology_flat_wcdm = False
        self.traced_bin_data = (
            SimpleNamespace(
                effective_z_source=2.0,
                family_ids=("1", "2"),
                family_index_per_image=jnp.asarray([0, 0, 1], dtype=jnp.int32),
                x_obs=jnp.asarray([10.0, 11.0, -5.0], dtype=jnp.float64),
                y_obs=jnp.asarray([20.0, 21.0, 7.0], dtype=jnp.float64),
                effective_z_index=0,
            ),
        )

    def reported_physical_to_latent_parameter_vector(self, best_fit_physical):
        return np.asarray(best_fit_physical, dtype=float)

    def _physical_parameter_vector(self, params):
        return jnp.asarray(params, dtype=jnp.float64)

    def _build_packed_lens_state_with_validity_from_physical(self, _physical_params, _z_source, **_kwargs):
        return {}, {"is_valid": jnp.asarray(True)}

    def _lensing_jacobian_for_components(self, _z_source, _x_obs, _y_obs, _packed_state):
        return (
            jnp.asarray(self.det_values, dtype=jnp.float64),
            jnp.zeros_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
            jnp.zeros_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
            jnp.ones_like(jnp.asarray(self.det_values, dtype=jnp.float64)),
        )


def _tiny_critical_det_state():
    return SimpleNamespace(
        parameter_specs=[],
        family_data=[
            FamilyData(
                family_id="1",
                z_source=2.0,
                effective_z_source=2.0,
                sigma_arcsec=0.1,
                image_labels=["1.1", "1.2"],
                x_obs=np.asarray([10.0, 11.0], dtype=float),
                y_obs=np.asarray([20.0, 21.0], dtype=float),
                reliability=np.ones(2, dtype=float),
            ),
            FamilyData(
                family_id="2",
                z_source=2.3,
                effective_z_source=2.0,
                sigma_arcsec=0.1,
                image_labels=["2.1"],
                x_obs=np.asarray([-5.0], dtype=float),
                y_obs=np.asarray([7.0], dtype=float),
                reliability=np.ones(1, dtype=float),
            ),
        ],
    )


def test_critical_det_diagnostic_flags_only_images_below_threshold() -> None:
    result = cluster_solver._stage3_critical_det_diagnostic_from_evaluator(
        _tiny_critical_det_state(),
        _TinyCriticalDetEvaluator([5.0e-3, 2.0e-2, -9.0e-3]),
        np.asarray([], dtype=float),
        threshold=1.0e-2,
    )

    assert result.total_images == 3
    assert result.min_abs_detA == pytest.approx(5.0e-3)
    assert result.flagged["family_id"].tolist() == ["1", "2"]
    assert result.flagged["image_label"].tolist() == ["1.1", "2.1"]
    np.testing.assert_allclose(result.flagged["detA"].to_numpy(dtype=float), [5.0e-3, -9.0e-3])
    np.testing.assert_allclose(result.flagged["abs_detA"].to_numpy(dtype=float), [5.0e-3, 9.0e-3])


def test_critical_det_diagnostic_empty_table_has_headers(tmp_path: Path) -> None:
    result = cluster_solver._stage3_critical_det_diagnostic_from_evaluator(
        _tiny_critical_det_state(),
        _TinyCriticalDetEvaluator([5.0e-3, 2.0e-2, -9.0e-3]),
        np.asarray([], dtype=float),
        threshold=1.0e-4,
    )
    table_path = cluster_solver._write_critical_det_diagnostic_table(
        result,
        tmp_path / "critical_det_images.csv",
    )

    assert result.flagged.empty
    assert list(result.flagged.columns) == list(cluster_solver.CRITICAL_DET_DIAGNOSTIC_COLUMNS)
    loaded = pd.read_csv(table_path)
    assert loaded.empty
    assert list(loaded.columns) == list(cluster_solver.CRITICAL_DET_DIAGNOSTIC_COLUMNS)


def test_local_jacobian_loglike_stabilizes_near_singular_covariance() -> None:
    value = _local_jacobian_bin_loglike(
        beta_x=jnp.asarray([0.0, 0.01], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.ones(2, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=100.0,
    )

    assert np.isfinite(float(value))


def test_local_jacobian_max_gain_makes_near_critical_covariance_less_punitive() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.ones(2, dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.zeros(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-12,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _local_jacobian_bin_loglike(**common)
    stabilized = _local_jacobian_bin_loglike(max_gain=50.0, **common)

    assert np.isfinite(float(stabilized))
    assert float(stabilized) > float(raw)


def test_local_jacobian_smooth_residual_cap_is_less_punitive_for_large_residual() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 200.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=1.0e-3,
    )

    raw = _local_jacobian_bin_loglike(**common)
    capped = _local_jacobian_bin_loglike(max_residual_arcsec=3.0, **common)

    assert np.isfinite(float(capped))
    assert float(capped) > float(raw)


def test_local_jacobian_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = dict(
        beta_x=jnp.asarray([0.0, 10.0], dtype=jnp.float64),
        beta_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999999, 0.999999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        source_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=1.0e-3,
    )

    gaussian = _local_jacobian_bin_loglike(residual_loss="gaussian", **common)
    student_t = _local_jacobian_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_forward_metric_image_plane_matches_linearized_identity_metric() -> None:
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        residual_loss="gaussian",
    )
    forward = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.03, -0.02], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.01, 0.04], dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        **common,
    )
    linearized = _linearized_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.03, -0.02], dtype=jnp.float64),
        residual_y=jnp.asarray([0.01, 0.04], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        image_presence_penalty_weight=0.0,
        **common,
    )

    np.testing.assert_allclose(float(forward), float(linearized), rtol=1.0e-7, atol=1.0e-10)


def test_fold_regularized_image_plane_matches_linearized_identity_with_zero_curvature() -> None:
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.95, 0.80], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        residual_loss="gaussian",
    )
    residual_x = jnp.asarray([0.03, -0.02], dtype=jnp.float64)
    residual_y = jnp.asarray([0.01, 0.04], dtype=jnp.float64)
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        fold_curvature_arcsec_inv=0.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        **common,
    )
    linearized = _linearized_image_plane_bin_loglike(
        residual_x=residual_x,
        residual_y=residual_y,
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        image_presence_penalty_weight=0.0,
        **common,
    )

    np.testing.assert_allclose(float(fold_regularized), float(linearized), rtol=1.0e-7, atol=1.0e-10)


def test_catastrophe_normal_form_matches_forward_metric_when_gate_off() -> None:
    residual_x = jnp.asarray([0.03, -0.02], dtype=jnp.float64)
    residual_y = jnp.asarray([0.01, 0.04], dtype=jnp.float64)
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.95, 0.80], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.asarray([1.0e-4, 2.0e-4], dtype=jnp.float64),
        scatter_var_y=jnp.asarray([2.0e-4, 1.0e-4], dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    jacobian = dict(
        jac_a00=jnp.asarray([1.0, 1.2], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.2, -0.1], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.5, 1.4], dtype=jnp.float64),
    )
    catastrophe = _catastrophe_normal_form_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        catastrophe_kappa=jnp.asarray([100.0, -50.0], dtype=jnp.float64),
        catastrophe_rho=jnp.asarray([-20.0, 30.0], dtype=jnp.float64),
        catastrophe_lambda_on=0.03,
        catastrophe_lambda_off=0.08,
        catastrophe_tangential_variance_min=0.0,
        **jacobian,
        **common,
    )
    forward = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        **jacobian,
        **common,
    )

    np.testing.assert_allclose(float(catastrophe), float(forward), rtol=1.0e-10, atol=1.0e-10)


def test_catastrophe_normal_form_zero_jacobian_has_finite_value_and_gradient() -> None:
    def loglike(rx):
        return _catastrophe_normal_form_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([rx], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.2], dtype=jnp.float64),
            jac_a00=jnp.zeros(1, dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.zeros(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            catastrophe_kappa=jnp.asarray([2.0], dtype=jnp.float64),
            catastrophe_rho=jnp.asarray([0.0], dtype=jnp.float64),
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.1, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_catastrophe_normal_form_near_critical_has_finite_jacobian_gradient() -> None:
    def loglike(a00):
        return _catastrophe_normal_form_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([0.08], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.02], dtype=jnp.float64),
            jac_a00=jnp.asarray([a00], dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.ones(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            catastrophe_kappa=jnp.asarray([2.0], dtype=jnp.float64),
            catastrophe_rho=jnp.asarray([0.0], dtype=jnp.float64),
            catastrophe_lambda_on=0.03,
            catastrophe_lambda_off=0.08,
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.0, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_catastrophe_normal_form_negative_correction_stays_finite() -> None:
    value = _catastrophe_normal_form_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.02], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.01], dtype=jnp.float64),
        jac_a00=jnp.asarray([0.02], dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=1.0e-3,
        catastrophe_kappa=jnp.asarray([0.0], dtype=jnp.float64),
        catastrophe_rho=jnp.asarray([-4.0], dtype=jnp.float64),
        catastrophe_lambda_on=0.03,
        catastrophe_lambda_off=0.08,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    assert np.isfinite(float(value))


def _synthetic_fold_jacobian_entries(x: Any, y: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x_array = jnp.asarray(x, dtype=jnp.float64)
    y_array = jnp.asarray(y, dtype=jnp.float64)
    return (
        0.08 + 0.30 * x_array + 0.05 * y_array,
        0.02 + 0.07 * x_array - 0.03 * y_array,
        -0.04 + 0.11 * y_array,
        1.20 + 0.02 * x_array + 0.09 * y_array,
    )


def _reference_fold_signed_curvature(
    x_obs: jnp.ndarray,
    y_obs: jnp.ndarray,
    observed_jacobian_entries: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fold_frame: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    (
        source_critical_x,
        source_critical_y,
        _source_noncritical_x,
        _source_noncritical_y,
        image_critical_x,
        image_critical_y,
        _image_noncritical_x,
        _image_noncritical_y,
        _singular_min,
        _singular_max,
        frame_finite,
    ) = fold_frame
    del observed_jacobian_entries
    step = jnp.asarray(cluster_solver.DEFAULT_FOLD_CURVATURE_FINITE_DIFFERENCE_STEP_ARCSEC, dtype=jnp.float64)
    plus_entries = _synthetic_fold_jacobian_entries(
        x_obs + step * image_critical_x,
        y_obs + step * image_critical_y,
    )
    minus_entries = _synthetic_fold_jacobian_entries(
        x_obs - step * image_critical_x,
        y_obs - step * image_critical_y,
    )
    da00_dcrit = (plus_entries[0] - minus_entries[0]) / (2.0 * step)
    da01_dcrit = (plus_entries[1] - minus_entries[1]) / (2.0 * step)
    da10_dcrit = (plus_entries[2] - minus_entries[2]) / (2.0 * step)
    da11_dcrit = (plus_entries[3] - minus_entries[3]) / (2.0 * step)
    d_a_v_x = da00_dcrit * image_critical_x + da01_dcrit * image_critical_y
    d_a_v_y = da10_dcrit * image_critical_x + da11_dcrit * image_critical_y
    kappa_eff = source_critical_x * d_a_v_x + source_critical_y * d_a_v_y
    return jnp.where(frame_finite & jnp.isfinite(kappa_eff), kappa_eff, jnp.full_like(kappa_eff, jnp.nan))


class _FoldCurvatureProbeEvaluator:
    use_bulk_ray_shooting = True

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _lensing_jacobian_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: dict[str, Any],
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del z_source, packed_state, component_indices
        self.calls.append(int(jnp.asarray(x).shape[0]))
        return _synthetic_fold_jacobian_entries(x, y)


def test_fold_signed_curvature_packs_centered_difference_probes() -> None:
    x_obs = jnp.asarray([-0.4, 0.1, 0.9], dtype=jnp.float64)
    y_obs = jnp.asarray([0.2, -0.3, 0.7], dtype=jnp.float64)
    observed_jacobian_entries = _synthetic_fold_jacobian_entries(x_obs, y_obs)
    fold_frame = cluster_solver._fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
    expected = _reference_fold_signed_curvature(x_obs, y_obs, observed_jacobian_entries, fold_frame)
    fake = _FoldCurvatureProbeEvaluator()

    actual = cluster_solver.ClusterJAXEvaluator._fold_signed_curvature_from_observed_jacobian(
        fake,
        2.0,
        x_obs,
        y_obs,
        {},
        observed_jacobian_entries,
        fold_frame=fold_frame,
        fill_value=1.0,
    )

    assert fake.calls == [2 * x_obs.shape[0]]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1.0e-10, atol=1.0e-10)


def test_fold_signed_curvature_skips_unconstrained_rows_without_changing_loglike() -> None:
    x_obs = jnp.asarray([-0.5, 0.0, 0.6], dtype=jnp.float64)
    y_obs = jnp.asarray([0.4, -0.2, 0.3], dtype=jnp.float64)
    observed_jacobian_entries = _synthetic_fold_jacobian_entries(x_obs, y_obs)
    fold_frame = cluster_solver._fold_regularized_singular_frame_from_jacobian(*observed_jacobian_entries)
    expected_all = _reference_fold_signed_curvature(x_obs, y_obs, observed_jacobian_entries, fold_frame)
    constrained_indices = jnp.asarray([0, 2], dtype=jnp.int32)
    fake = _FoldCurvatureProbeEvaluator()

    skipped = cluster_solver.ClusterJAXEvaluator._fold_signed_curvature_from_observed_jacobian(
        fake,
        2.0,
        x_obs,
        y_obs,
        {},
        observed_jacobian_entries,
        fold_frame=fold_frame,
        image_indices=constrained_indices,
        fill_value=7.0,
    )

    assert fake.calls == [2 * constrained_indices.shape[0]]
    np.testing.assert_allclose(np.asarray(skipped)[[0, 2]], np.asarray(expected_all)[[0, 2]], rtol=1.0e-10, atol=1.0e-10)
    assert float(np.asarray(skipped)[1]) == pytest.approx(7.0)

    common = dict(
        residual_beta_x=jnp.asarray([0.05, 10.0, -0.04], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([-0.02, -10.0, 0.03], dtype=jnp.float64),
        jac_a00=observed_jacobian_entries[0],
        jac_a01=observed_jacobian_entries[1],
        jac_a10=observed_jacobian_entries[2],
        jac_a11=observed_jacobian_entries[3],
        sigma_per_image=jnp.asarray([0.2, 0.2, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, False, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(3, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(3, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        fold_curvature_arcsec_inv=7.0,
        fold_frame=fold_frame,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    reference_loglike = _fold_regularized_image_plane_bin_loglike(
        fold_kappa_eff=expected_all,
        **common,
    )
    skipped_loglike = _fold_regularized_image_plane_bin_loglike(
        fold_kappa_eff=skipped,
        **common,
    )

    np.testing.assert_allclose(float(skipped_loglike), float(reference_loglike), rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_exact_fold_uses_root_distance_exponent() -> None:
    residual = 0.25
    sigma = 1.0
    curvature = 2.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=curvature,
        fold_kappa_eff=jnp.asarray([curvature], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = 2.0 * abs(residual) / (curvature * sigma**2)
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-8, atol=1.0e-8)


def test_fold_regularized_image_plane_does_not_include_source_plane_logdet() -> None:
    residual = 0.25
    sigma = 1.0
    curvature = 2.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=curvature,
        fold_kappa_eff=jnp.asarray([curvature], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = 2.0 * abs(residual) / (curvature * sigma**2)
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    pseudo_covariance_logdet_extra = -0.5 * math.log(0.5 * curvature * sigma**2 * abs(residual))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-8, atol=1.0e-8)
    assert not math.isclose(float(value), expected + pseudo_covariance_logdet_extra, rel_tol=1.0e-8, abs_tol=1.0e-8)


def test_fold_regularized_image_plane_noncritical_residual_stays_quadratic() -> None:
    residual = 0.20
    sigma = 1.0
    value = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.0], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([residual], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([sigma], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        fold_curvature_arcsec_inv=100.0,
        fold_kappa_eff=jnp.asarray([100.0], dtype=jnp.float64),
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )

    expected_quad = residual**2 / sigma**2
    expected = math.log(1.0 - 1.0e-6) - 0.5 * (expected_quad + 2.0 * math.log(2.0 * math.pi * sigma**2))
    np.testing.assert_allclose(float(value), expected, rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_away_from_criticality_matches_forward_metric() -> None:
    residual_x = jnp.asarray([0.03, -0.02], dtype=jnp.float64)
    residual_y = jnp.asarray([0.01, 0.04], dtype=jnp.float64)
    common = dict(
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.95, 0.80], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        fold_curvature_arcsec_inv=2.0,
        fold_kappa_eff=jnp.asarray([2.0, -3.0], dtype=jnp.float64),
        **common,
    )
    forward = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=residual_x,
        residual_beta_y=residual_y,
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.ones(2, dtype=jnp.float64),
        **common,
    )

    np.testing.assert_allclose(float(fold_regularized), float(forward), rtol=1.0e-6, atol=1.0e-7)


def test_fold_regularized_image_plane_wrong_side_root_falls_back_to_forward_metric() -> None:
    residual = -0.25
    common = dict(
        residual_beta_x=jnp.asarray([residual], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=1.0e-3,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    fold_regularized = _fold_regularized_image_plane_bin_loglike(
        fold_curvature_arcsec_inv=2.0,
        fold_kappa_eff=jnp.asarray([2.0], dtype=jnp.float64),
        **common,
    )
    forward = _forward_metric_image_plane_bin_loglike(**common)

    np.testing.assert_allclose(float(fold_regularized), float(forward), rtol=1.0e-10, atol=1.0e-10)


def test_fold_regularized_image_plane_zero_jacobian_has_finite_value_and_gradient() -> None:
    def loglike(rx):
        return _fold_regularized_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([rx], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.2], dtype=jnp.float64),
            jac_a00=jnp.zeros(1, dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.zeros(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            fold_curvature_arcsec_inv=1.0,
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.1, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_fold_regularized_image_plane_near_caustic_has_finite_value_and_gradient() -> None:
    def loglike(rx):
        return _fold_regularized_image_plane_bin_loglike(
            residual_beta_x=jnp.asarray([rx], dtype=jnp.float64),
            residual_beta_y=jnp.asarray([0.05], dtype=jnp.float64),
            jac_a00=jnp.asarray([1.0e-3], dtype=jnp.float64),
            jac_a01=jnp.zeros(1, dtype=jnp.float64),
            jac_a10=jnp.zeros(1, dtype=jnp.float64),
            jac_a11=jnp.ones(1, dtype=jnp.float64),
            sigma_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
            image_has_constraint=jnp.asarray([True]),
            image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
            scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
            scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
            covariance_floor=1.0e-6,
            outlier_sigma_arcsec=1.0e-3,
            fold_curvature_arcsec_inv=2.0,
            fold_kappa_eff=jnp.asarray([2.0], dtype=jnp.float64),
            max_gain=0.0,
            max_residual_arcsec=0.0,
            residual_loss="gaussian",
        )

    value = loglike(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(loglike)(jnp.asarray(0.1, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


def test_forward_metric_image_plane_adds_image_presence_penalty_from_forward_metric() -> None:
    residual_beta_x = jnp.asarray([0.0, 0.03, 0.8], dtype=jnp.float64)
    residual_beta_y = jnp.asarray([0.0, -0.02, 0.0], dtype=jnp.float64)
    jac_a00 = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float64)
    jac_a01 = jnp.zeros(3, dtype=jnp.float64)
    jac_a10 = jnp.zeros(3, dtype=jnp.float64)
    jac_a11 = jnp.ones(3, dtype=jnp.float64)
    family_idx = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    reliability = jnp.asarray([0.9, 0.9, 0.9], dtype=jnp.float64)
    common = dict(
        residual_beta_x=residual_beta_x,
        residual_beta_y=residual_beta_y,
        jac_a00=jac_a00,
        jac_a01=jac_a01,
        jac_a10=jac_a10,
        jac_a11=jac_a11,
        family_idx=family_idx,
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=reliability,
        image_has_constraint=jnp.asarray([True, True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(3, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(3, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
    )
    zero_weight = _forward_metric_image_plane_bin_loglike(
        image_presence_penalty_weight=0.0,
        **common,
    )
    with_penalty = _forward_metric_image_plane_bin_loglike(
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
        **common,
    )
    presence_residual2, presence_finite = _forward_metric_image_presence_residual2(
        residual_beta_x,
        residual_beta_y,
        jac_a00,
        jac_a01,
        jac_a10,
        jac_a11,
        common["sigma_per_image"],
        common["image_sigma_int"],
        common["covariance_floor"],
        max_gain=0.0,
    )
    expected_penalty = _soft_observed_image_presence_loglike_from_residual2(
        residual2=presence_residual2,
        family_idx=family_idx,
        n_families=1,
        reliability_per_image=reliability,
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=3.0,
        match_radius_arcsec=0.1,
        temperature_arcsec=0.05,
        count_softness=0.05,
        count_margin=0.0,
    )

    assert bool(np.all(np.asarray(presence_finite)))
    assert float(expected_penalty) < 0.0
    np.testing.assert_allclose(
        float(with_penalty),
        float(zero_weight + expected_penalty),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_forward_metric_image_presence_residual_matches_identity_image_plane() -> None:
    residual_x = jnp.asarray([0.02, -0.04, 0.10], dtype=jnp.float64)
    residual_y = jnp.asarray([0.03, 0.05, -0.20], dtype=jnp.float64)
    residual2, finite = _forward_metric_image_presence_residual2(
        residual_x,
        residual_y,
        jac_a00=jnp.ones(3, dtype=jnp.float64),
        jac_a01=jnp.zeros(3, dtype=jnp.float64),
        jac_a10=jnp.zeros(3, dtype=jnp.float64),
        jac_a11=jnp.ones(3, dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.10, 0.20, 0.30], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        max_gain=0.0,
    )

    assert bool(np.all(np.asarray(finite)))
    np.testing.assert_allclose(
        np.asarray(residual2),
        np.asarray(jnp.square(residual_x) + jnp.square(residual_y)),
        rtol=1.0e-7,
        atol=1.0e-12,
    )


def test_forward_metric_image_presence_is_finite_for_singular_jacobian() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.05, 0.3], dtype=jnp.float64),
        jac_a00=jnp.zeros(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.zeros(2, dtype=jnp.float64),
        jac_a11=jnp.zeros(2, dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > -1.0e20


def test_forward_metric_image_presence_is_finite_for_near_rank_one_covariance() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.1, -0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.05, 0.3], dtype=jnp.float64),
        jac_a00=jnp.ones(2, dtype=jnp.float64),
        jac_a01=jnp.zeros(2, dtype=jnp.float64),
        jac_a10=jnp.ones(2, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12, 1.0e-12], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9, 0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(2, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(2, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
        max_gain=0.0,
        max_residual_arcsec=0.0,
        residual_loss="gaussian",
        image_presence_penalty_weight=3.0,
        image_presence_match_radius_arcsec=0.1,
        image_presence_temperature_arcsec=0.05,
        image_presence_count_softness=0.05,
        image_presence_count_margin=0.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > -1.0e20


def test_forward_metric_image_plane_uses_current_jacobian_covariance() -> None:
    common = dict(
        residual_beta_x=jnp.asarray([0.2], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.1], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=1.0e-9,
        outlier_sigma_arcsec=10.0,
    )
    identity = _forward_metric_image_plane_bin_loglike(
        jac_a00=jnp.ones(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.ones(1, dtype=jnp.float64),
        **common,
    )
    magnified = _forward_metric_image_plane_bin_loglike(
        jac_a00=jnp.asarray([2.0], dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.asarray([2.0], dtype=jnp.float64),
        **common,
    )

    assert np.isfinite(float(magnified))
    assert not np.isclose(float(identity), float(magnified))


def test_forward_metric_image_plane_stabilizes_near_singular_covariance() -> None:
    value = _forward_metric_image_plane_bin_loglike(
        residual_beta_x=jnp.asarray([0.01], dtype=jnp.float64),
        residual_beta_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.ones(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.ones(1, dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0e-12], dtype=jnp.float64),
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.9], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        scatter_var_x=jnp.zeros(1, dtype=jnp.float64),
        scatter_var_y=jnp.zeros(1, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=10.0,
    )

    assert np.isfinite(float(value))


def test_linearized_image_plane_residual_identity_jacobian() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([-0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert np.allclose(np.asarray(dx), [-0.2])
    assert np.allclose(np.asarray(dy), [0.1])


def test_linearized_image_plane_default_damped_inverse_is_finite_for_singular_jacobian() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([-2.0], dtype=jnp.float64),
        jac_a00=jnp.zeros(1, dtype=jnp.float64),
        jac_a01=jnp.zeros(1, dtype=jnp.float64),
        jac_a10=jnp.zeros(1, dtype=jnp.float64),
        jac_a11=jnp.zeros(1, dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert np.isfinite(float(np.asarray(dx)[0]))
    assert np.isfinite(float(np.asarray(dy)[0]))
    assert float(np.asarray(dx)[0]) == pytest.approx(0.0)
    assert float(np.asarray(dy)[0]) == pytest.approx(0.0)


def test_linearized_image_plane_damped_inverse_limits_near_critical_gain() -> None:
    default_dx, _, default_finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0e-6], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([1.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0e-6], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        max_gain=50.0,
    )

    assert bool(np.asarray(default_finite)[0])
    assert np.isfinite(float(np.asarray(default_dx)[0]))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(float(np.asarray(dx)[0]))
    assert abs(float(np.asarray(dx)[0])) < 1.0
    assert abs(float(np.asarray(dx)[0])) < 1.0e-3 * abs(float(np.asarray(default_dx)[0]))
    assert float(np.asarray(dy)[0]) == pytest.approx(0.0)


def test_linearized_image_plane_smooth_residual_cap_bounds_large_step() -> None:
    dx, dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        max_residual_arcsec=3.0,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert radius <= 3.0 + 1.0e-10
    assert radius > 2.9


def test_anchored_solved_image_plane_zero_residual_for_matching_source() -> None:
    class FakeEvaluator:
        anchored_image_plane_solve_steps = 3
        anchored_image_plane_trust_radius_arcsec = 0.3
        anchored_image_plane_lm_damping_relative = 1.0e-3
        anchored_image_plane_lm_damping_absolute = 1.0e-6

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x, y

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(y), jnp.ones_like(y)

    fake = FakeEvaluator()
    residual_method = cluster_solver.ClusterJAXEvaluator._anchored_solved_image_plane_residuals_for_components.__get__(
        fake,
        FakeEvaluator,
    )
    x_obs = jnp.asarray([0.2, -0.4], dtype=jnp.float64)
    y_obs = jnp.asarray([0.1, 0.3], dtype=jnp.float64)
    dx, dy, finite = residual_method(2.0, x_obs, y_obs, x_obs, y_obs, {})

    assert bool(np.all(np.asarray(finite)))
    np.testing.assert_allclose(np.asarray(dx), np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(dy), np.zeros(2), atol=1.0e-12)


def test_anchored_solved_image_plane_improves_nonlinear_map_over_one_linear_step() -> None:
    class FakeEvaluator:
        anchored_image_plane_solve_steps = 3
        anchored_image_plane_trust_radius_arcsec = 10.0
        anchored_image_plane_lm_damping_relative = 1.0e-8
        anchored_image_plane_lm_damping_absolute = 1.0e-12

        def _ray_shooting_for_components(self, _z_source, x, y, _packed_state):
            return x + x**2, y

        def _lensing_jacobian_for_components(self, _z_source, x, y, _packed_state):
            return 1.0 + 2.0 * x, jnp.zeros_like(x), jnp.zeros_like(y), jnp.ones_like(y)

    fake = FakeEvaluator()
    residual_method = cluster_solver.ClusterJAXEvaluator._anchored_solved_image_plane_residuals_for_components.__get__(
        fake,
        FakeEvaluator,
    )
    x_obs = jnp.asarray([0.5], dtype=jnp.float64)
    y_obs = jnp.asarray([0.0], dtype=jnp.float64)
    target_x = jnp.asarray([0.0], dtype=jnp.float64)
    target_y = jnp.asarray([0.0], dtype=jnp.float64)
    anchored_dx, anchored_dy, anchored_finite = residual_method(2.0, x_obs, y_obs, target_x, target_y, {})

    beta_obs_x, beta_obs_y = fake._ray_shooting_for_components(2.0, x_obs, y_obs, {})
    jacobian = fake._lensing_jacobian_for_components(2.0, x_obs, y_obs, {})
    linear_dx, linear_dy, linear_finite = _linearized_image_plane_residual_from_jacobian(
        beta_obs_x - target_x,
        beta_obs_y - target_y,
        *jacobian,
        max_residual_arcsec=0.0,
    )
    linear_x = x_obs + linear_dx
    linear_y = y_obs + linear_dy
    anchored_x = x_obs + anchored_dx
    anchored_y = y_obs + anchored_dy
    linear_beta_x, linear_beta_y = fake._ray_shooting_for_components(2.0, linear_x, linear_y, {})
    anchored_beta_x, anchored_beta_y = fake._ray_shooting_for_components(2.0, anchored_x, anchored_y, {})
    linear_residual = float(np.hypot(np.asarray(linear_beta_x - target_x)[0], np.asarray(linear_beta_y - target_y)[0]))
    anchored_residual = float(
        np.hypot(np.asarray(anchored_beta_x - target_x)[0], np.asarray(anchored_beta_y - target_y)[0])
    )

    assert bool(np.asarray(linear_finite)[0])
    assert bool(np.asarray(anchored_finite)[0])
    assert anchored_residual < 0.2 * linear_residual


def test_anchored_solved_image_plane_rank_deficient_jacobian_is_finite_and_bounded() -> None:
    dx, dy, finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([100.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([0.0], dtype=jnp.float64),
        trust_radius_arcsec=0.3,
        lm_damping_relative=1.0e-3,
        lm_damping_absolute=1.0e-6,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert np.isfinite(radius)
    assert radius <= 0.3 + 1.0e-10


def test_anchored_solved_image_plane_trust_radius_smoothly_bounds_step() -> None:
    dx, dy, finite = _anchored_solved_image_plane_step_from_jacobian(
        f_x=jnp.asarray([100.0], dtype=jnp.float64),
        f_y=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
        trust_radius_arcsec=0.3,
        lm_damping_relative=1.0e-8,
        lm_damping_absolute=1.0e-12,
    )

    radius = float(np.hypot(float(np.asarray(dx)[0]), float(np.asarray(dy)[0])))
    assert bool(np.asarray(finite)[0])
    assert radius <= 0.3 + 1.0e-10
    assert radius > 0.29


def test_anchored_solved_image_plane_student_t_and_presence_scoring_is_finite() -> None:
    common = dict(
        residual_x=jnp.asarray([2.0, 0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        image_presence_penalty_weight=2.0,
        image_presence_match_radius_arcsec=0.3,
        image_presence_temperature_arcsec=0.1,
    )
    gaussian = _linearized_image_plane_bin_loglike(residual_loss="gaussian", **common)
    student_t = _linearized_image_plane_bin_loglike(residual_loss="student-t", student_t_nu=4.0, **common)

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_linearized_image_plane_loglike_scores_zero_residual() -> None:
    value = _linearized_image_plane_bin_loglike(
        residual_x=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        image_sigma_int=jnp.asarray(0.01, dtype=jnp.float64),
        covariance_floor=1.0e-6,
        outlier_sigma_arcsec=10.0,
        image_presence_penalty_weight=2.0,
    )

    assert np.isfinite(float(value))
    assert float(value) > 0.0


def test_image_plane_scatter_effective_sigma2_uses_sigma_int_and_covariance_floor() -> None:
    sigma2 = cluster_solver._image_plane_effective_sigma2(
        sigma_per_image=jnp.asarray([0.2], dtype=jnp.float64),
        image_sigma_int=jnp.asarray(0.1, dtype=jnp.float64),
        covariance_floor=0.01,
    )

    assert np.asarray(sigma2).tolist() == pytest.approx([0.2**2 + 0.1**2 + 0.01])


def test_linearized_image_plane_loglike_uses_effective_sigma_components() -> None:
    common = dict(
        residual_x=jnp.asarray([0.12, -0.05], dtype=jnp.float64),
        residual_y=jnp.asarray([0.02, 0.09], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([0.999, 0.999], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True]),
        covariance_floor=0.01,
        outlier_sigma_arcsec=10.0,
        image_presence_penalty_weight=0.0,
    )

    zero_scatter_value = _linearized_image_plane_bin_loglike(
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        **common,
    )
    sampled_scatter_value = _linearized_image_plane_bin_loglike(
        image_sigma_int=jnp.asarray(0.1, dtype=jnp.float64),
        **common,
    )

    assert np.isfinite(float(zero_scatter_value))
    assert np.isfinite(float(sampled_scatter_value))
    assert float(sampled_scatter_value) != pytest.approx(float(zero_scatter_value), abs=1.0e-12)


def test_linearized_image_plane_student_t_loss_is_less_punitive_for_large_residual() -> None:
    common = dict(
        residual_x=jnp.asarray([5.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0], dtype=jnp.int32),
        n_families=1,
        sigma_per_image=jnp.asarray([0.1], dtype=jnp.float64),
        reliability_per_image=jnp.asarray([1.0], dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True]),
        image_sigma_int=jnp.asarray(0.0, dtype=jnp.float64),
        covariance_floor=0.0,
        outlier_sigma_arcsec=1.0e-3,
        image_presence_penalty_weight=0.0,
    )
    gaussian = _linearized_image_plane_bin_loglike(
        residual_loss="gaussian",
        **common,
    )
    student_t = _linearized_image_plane_bin_loglike(
        residual_loss="student-t",
        student_t_nu=4.0,
        **common,
    )

    assert np.isfinite(float(student_t))
    assert float(student_t) > float(gaussian)


def test_soft_observed_image_presence_penalty_is_near_zero_for_present_images() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.01, -0.01], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, -0.01, 0.01], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert np.isfinite(float(value))
    assert -1.0e-3 < float(value) <= 0.0


def test_soft_observed_image_presence_penalty_detects_missing_observed_anchor() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert float(value) < -0.5


def test_soft_observed_image_presence_penalty_respects_reliability() -> None:
    common = dict(
        residual_x=jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=2.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )
    high_reliability = _soft_observed_image_presence_loglike(
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        **common,
    )
    low_reliability = _soft_observed_image_presence_loglike(
        reliability_per_image=jnp.asarray([1.0, 1.0, 0.1], dtype=jnp.float64),
        **common,
    )

    assert float(low_reliability) > float(high_reliability)


def test_soft_observed_image_presence_penalty_zero_weight_is_neutral() -> None:
    value = _soft_observed_image_presence_loglike(
        residual_x=jnp.asarray([0.0, 0.0, 10.0], dtype=jnp.float64),
        residual_y=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        family_idx=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        n_families=1,
        reliability_per_image=jnp.ones(3, dtype=jnp.float64),
        image_has_constraint=jnp.asarray([True, True, True]),
        penalty_weight=0.0,
        match_radius_arcsec=0.30,
        temperature_arcsec=0.10,
        count_softness=0.05,
        count_margin=0.05,
    )

    assert float(value) == 0.0


def test_linearized_image_plane_off_diagonal_jacobian_changes_residual() -> None:
    diagonal_dx, diagonal_dy, _ = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )
    sheared_dx, sheared_dy, finite = _linearized_image_plane_residual_from_jacobian(
        f_x=jnp.asarray([0.2], dtype=jnp.float64),
        f_y=jnp.asarray([0.1], dtype=jnp.float64),
        jac_a00=jnp.asarray([1.0], dtype=jnp.float64),
        jac_a01=jnp.asarray([0.5], dtype=jnp.float64),
        jac_a10=jnp.asarray([0.0], dtype=jnp.float64),
        jac_a11=jnp.asarray([1.0], dtype=jnp.float64),
    )

    assert bool(np.asarray(finite)[0])
    assert not np.allclose(
        np.asarray([diagonal_dx, diagonal_dy]),
        np.asarray([sheared_dx, sheared_dy]),
    )


def test_validation_metrics_summary_reports_finite_aggregates() -> None:
    result = EvaluationResult(
        loglike=-1.0,
        family_predictions={
            "exact_good_a": {
                "exact_image_rms": 0.2,
                "approx_image_rms_arcsec": 0.3,
                "source_plane_rms": 0.02,
            },
            "exact_good_b": {
                "exact_image_rms": 0.4,
                "approx_image_rms_arcsec": 0.5,
                "source_plane_rms": 0.04,
            },
            "failed_exact": {
                "failed": True,
                "exact_image_rms": 9.0,
                "approx_image_rms_arcsec": 0.7,
            },
            "approx_only": {
                "approx_image_rms_arcsec": 0.9,
                "source_plane_rms": 0.08,
            },
            "invalid_metrics": {
                "exact_image_rms": float("nan"),
                "approx_image_rms_arcsec": None,
                "source_plane_rms": "bad",
            },
        },
    )

    summary = _validation_metrics_summary(result)

    assert "validated_families=5" in summary
    assert "exact_families=2" in summary
    assert "exact_image_rms_mean=0.3" in summary
    assert "exact_image_rms_median=0.3" in summary
    assert "approx_image_rms_mean=0.6" in summary
    assert "source_rms_mean=0.04667" in summary
    assert "nan" not in summary.lower()


def test_evaluate_uses_source_summary_without_exact_image_prediction() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=3)
    cache = FamilyValidationCache()
    evaluator.state = SimpleNamespace(family_data=[family])
    evaluator.surrogate_enabled = False
    evaluator.quick_diagnostics = False
    evaluator.validation_cache = {"1": cache}
    evaluator.source_loglike = lambda _params: -123.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.2, "failed": False}}

    def fail_exact_prediction(_params, _exact_family):
        raise AssertionError("source-summary evaluation should not solve exact image positions")

    evaluator._exact_family_prediction = fail_exact_prediction

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=False)

    assert result.loglike == pytest.approx(-123.0)
    assert result.family_predictions["1"]["failed"] is False
    assert result.family_predictions["1"]["approx_image_rms_arcsec"] == pytest.approx(0.2)
    assert result.family_predictions["1"]["used_exact_refresh"] is False
    assert result.family_predictions["1"]["refresh_reason"] == "source_summary"
    assert cache.exact_validation_count == 0
    assert cache.multiplicity_mismatch_count == 0
    assert cache.match_failure_count == 0


def test_image_match_diagnostics_counts_extra_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 1.02, 5.0], dtype=float),
        np.asarray([0.01, 0.02, 5.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 3
    assert diagnostics["recovered_image_count"] == 2
    assert diagnostics["missing_image_count"] == 0
    assert diagnostics["extra_image_count"] == 1
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "extra_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, 1.02], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, 0.02], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([5.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([5.0], dtype=float))


def test_image_match_diagnostics_recovers_all_observed_with_extra_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=3,
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 1.02, 2.03, 8.0, -9.0], dtype=float),
        np.asarray([0.01, 0.02, 0.03, 1.0, -1.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 5
    assert diagnostics["recovered_image_count"] == 3
    assert diagnostics["missing_image_count"] == 0
    assert diagnostics["extra_image_count"] == 2
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "extra_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, True, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, 1.02, 2.03], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, 0.02, 0.03], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([8.0, -9.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([1.0, -1.0], dtype=float))


def test_image_match_diagnostics_counts_missing_model_images() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=3,
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 2.02], dtype=float),
        np.asarray([0.01, 0.02], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 2
    assert diagnostics["recovered_image_count"] == 2
    assert diagnostics["missing_image_count"] == 1
    assert diagnostics["extra_image_count"] == 0
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "missing_model_images"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, False, True], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, np.nan, 2.02], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, np.nan, 0.02], dtype=float))
    assert diagnostics["extra_model_x_arcsec"].size == 0
    assert diagnostics["extra_model_y_arcsec"].size == 0


def test_image_match_diagnostics_counts_partial_same_multiplicity_match_failure() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.match_tolerance_arcsec = 0.2
    family = SimpleNamespace(
        family_id="1",
        n_images=2,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0], dtype=float),
    )

    diagnostics = evaluator._image_match_diagnostics(
        np.asarray([0.01, 4.0], dtype=float),
        np.asarray([0.01, 0.0], dtype=float),
        family,
    )

    assert diagnostics["produced_image_count"] == 2
    assert diagnostics["recovered_image_count"] == 1
    assert diagnostics["missing_image_count"] == 1
    assert diagnostics["extra_image_count"] == 1
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "match_tolerance_exceeded"
    np.testing.assert_array_equal(diagnostics["recovered_image_mask"], np.asarray([True, False], dtype=bool))
    np.testing.assert_allclose(diagnostics["matched_model_x_arcsec"], np.asarray([0.01, np.nan], dtype=float))
    np.testing.assert_allclose(diagnostics["matched_model_y_arcsec"], np.asarray([0.01, np.nan], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_x_arcsec"], np.asarray([4.0], dtype=float))
    np.testing.assert_allclose(diagnostics["extra_model_y_arcsec"], np.asarray([0.0], dtype=float))


def test_exact_image_solver_uses_family_bounding_box_search_center() -> None:
    captured_kwargs: list[dict[str, Any]] = []

    class FakeSolver:
        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.timing_totals = {}
    evaluator._get_exact_model_solver = lambda _z_source: (object(), FakeSolver())
    evaluator._packed_to_kwargs_lens = lambda _packed_state: []
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b", "1.c", "1.d", "1.e"],
        x_obs=np.asarray([7.9, -19.9, -22.2, 9.5, 0.6], dtype=float),
        y_obs=np.asarray([-26.4, 13.1, -0.3, 12.6, 0.9], dtype=float),
        reliability=np.ones(5, dtype=float),
    )

    x_pred, y_pred = evaluator._solve_exact_images_lenstronomy(
        family,
        packed_state={},
        source_x=1.2,
        source_y=-3.4,
    )

    assert x_pred.tolist() == [0.0]
    assert y_pred.tolist() == [0.0]
    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["solver"] == "lenstronomy"
    assert kwargs["min_distance"] == pytest.approx(cluster_solver.DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC)
    assert kwargs["search_window"] == pytest.approx(family.search_window)
    assert kwargs["x_center"] == pytest.approx(0.5 * (np.min(family.x_obs) + np.max(family.x_obs)))
    assert kwargs["y_center"] == pytest.approx(0.5 * (np.min(family.y_obs) + np.max(family.y_obs)))


def test_exact_image_solver_uses_configured_search_controls() -> None:
    captured_kwargs: list[dict[str, Any]] = []

    class FakeSolver:
        def image_position_from_source(self, _source_x, _source_y, _kwargs_lens, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)

    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.timing_totals = {}
    evaluator.exact_image_min_distance_arcsec = 0.5
    evaluator.exact_image_precision_limit = 1.0e-5
    evaluator.exact_image_num_iter_max = 80
    evaluator._get_exact_model_solver = lambda _z_source: (object(), FakeSolver())
    evaluator._packed_to_kwargs_lens = lambda _packed_state: []
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a"],
        x_obs=np.asarray([7.9], dtype=float),
        y_obs=np.asarray([-26.4], dtype=float),
        reliability=np.ones(1, dtype=float),
    )

    evaluator._solve_exact_images_lenstronomy(
        family,
        packed_state={},
        source_x=1.2,
        source_y=-3.4,
    )

    assert len(captured_kwargs) == 1
    kwargs = captured_kwargs[0]
    assert kwargs["min_distance"] == pytest.approx(0.5)
    assert kwargs["precision_limit"] == pytest.approx(1.0e-5)
    assert kwargs["num_iter_max"] == 80


def test_exact_image_local_lm_converges_for_identity_lens() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.exact_image_lm_max_iter = 20
    evaluator.exact_image_lm_trust_radius_arcsec = 1.0
    evaluator.exact_image_displacement_tol_arcsec = 1.0e-4
    evaluator.exact_image_identification_tol_arcsec = 1.0e-3
    evaluator.exact_image_precision_limit = 1.0e-8
    evaluator.anchored_image_plane_lm_damping_relative = 1.0e-6
    evaluator.anchored_image_plane_lm_damping_absolute = 1.0e-12

    def identity_ray_shooting(_family, _packed_state, x, y):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ones = np.ones_like(x_arr)
        zeros = np.zeros_like(x_arr)
        return x_arr, y_arr, ones, zeros, zeros, ones

    evaluator._exact_ray_shooting_and_jacobian_numpy = identity_ray_shooting
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b"],
        x_obs=np.asarray([0.35, -0.2], dtype=float),
        y_obs=np.asarray([-0.25, 0.15], dtype=float),
        reliability=np.ones(2, dtype=float),
    )

    result = evaluator._refine_exact_images_local_lm(
        family,
        packed_state={},
        source_x=0.0,
        source_y=0.0,
        start_x=family.x_obs,
        start_y=family.y_obs,
    )

    np.testing.assert_allclose(result["x"], np.zeros(2), atol=1.0e-4)
    np.testing.assert_allclose(result["y"], np.zeros(2), atol=1.0e-4)
    assert np.all(result["identification_pass"])
    assert float(np.nanmax(result["final_step_arcsec"])) <= 1.0e-3


def test_exact_image_dispatcher_compares_local_lm_to_lenstronomy_seam() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.timing_totals = {}
    evaluator.exact_image_lm_max_iter = 20
    evaluator.exact_image_lm_trust_radius_arcsec = 1.0
    evaluator.exact_image_displacement_tol_arcsec = 1.0e-4
    evaluator.exact_image_identification_tol_arcsec = 1.0e-3
    evaluator.exact_image_precision_limit = 1.0e-8
    evaluator.anchored_image_plane_lm_damping_relative = 1.0e-6
    evaluator.anchored_image_plane_lm_damping_absolute = 1.0e-12
    evaluator.exact_image_finder = cluster_solver.EXACT_IMAGE_FINDER_LENSTRONOMY
    evaluator._solve_exact_images_lenstronomy = lambda *_args, **_kwargs: (
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
    )

    def identity_ray_shooting(_family, _packed_state, x, y):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ones = np.ones_like(x_arr)
        zeros = np.zeros_like(x_arr)
        return x_arr, y_arr, ones, zeros, zeros, ones

    evaluator._exact_ray_shooting_and_jacobian_numpy = identity_ray_shooting
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a", "1.b"],
        x_obs=np.asarray([0.4, -0.3], dtype=float),
        y_obs=np.asarray([0.2, -0.1], dtype=float),
        reliability=np.ones(2, dtype=float),
    )

    len_x, len_y = evaluator._solve_exact_images(family, {}, 0.0, 0.0)
    evaluator.exact_image_finder = cluster_solver.EXACT_IMAGE_FINDER_LOCAL_LM
    lm_x, lm_y = evaluator._solve_exact_images(family, {}, 0.0, 0.0)

    np.testing.assert_allclose(lm_x, len_x, atol=1.0e-3)
    np.testing.assert_allclose(lm_y, len_y, atol=1.0e-3)
    diagnostics = evaluator._last_exact_image_solver_diagnostics
    assert diagnostics["exact_image_finder"] == cluster_solver.EXACT_IMAGE_FINDER_LOCAL_LM
    assert np.all(diagnostics["exact_image_identification_pass"])


def test_exact_image_local_lm_adaptive_refines_failed_anchor() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.timing_totals = {}
    evaluator.exact_image_finder = cluster_solver.EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE
    evaluator.exact_image_lm_trust_radius_arcsec = 0.2
    evaluator.exact_image_min_distance_arcsec = 0.2
    evaluator.exact_image_adaptive_max_levels = 2
    calls = {"count": 0}

    def fake_refine(_family, _packed_state, _source_x, _source_y, start_x, start_y):
        calls["count"] += 1
        n = np.asarray(start_x, dtype=float).size
        if calls["count"] == 1:
            return {
                "x": np.asarray(start_x, dtype=float),
                "y": np.asarray(start_y, dtype=float),
                "converged": np.zeros(n, dtype=bool),
                "finite": np.ones(n, dtype=bool),
                "identification_pass": np.zeros(n, dtype=bool),
                "iterations": np.ones(n, dtype=int),
                "final_step_arcsec": np.full(n, 1.0, dtype=float),
                "final_source_residual_arcsec": np.full(n, 1.0, dtype=float),
            }
        x = np.asarray(start_x, dtype=float)
        y = np.asarray(start_y, dtype=float)
        best = int(np.argmin(np.square(x) + np.square(y)))
        passed = np.zeros(n, dtype=bool)
        passed[best] = True
        step = np.full(n, 1.0, dtype=float)
        step[best] = 1.0e-5
        return {
            "x": x * 0.0,
            "y": y * 0.0,
            "converged": passed.copy(),
            "finite": np.ones(n, dtype=bool),
            "identification_pass": passed,
            "iterations": np.ones(n, dtype=int),
            "final_step_arcsec": step,
            "final_source_residual_arcsec": step,
        }

    evaluator._refine_exact_images_local_lm = fake_refine
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.1,
        image_labels=["1.a"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
        reliability=np.ones(1, dtype=float),
    )

    x_pred, y_pred = evaluator._solve_exact_images(family, {}, 0.0, 0.0)

    np.testing.assert_allclose(x_pred, np.asarray([0.0]))
    np.testing.assert_allclose(y_pred, np.asarray([0.0]))
    assert calls["count"] > 1
    assert evaluator._last_exact_image_solver_diagnostics["exact_image_adaptive_used"].tolist() == [True]


def test_exact_family_prediction_details_reports_solver_failure_counts() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=2)
    cache = FamilyValidationCache()
    evaluator.validation_cache = {"1": cache}
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}

    def fail_ray_shooting(_family, _packed_state):
        raise RuntimeError("ray shooting failed")

    evaluator._exact_source_ray_shooting = fail_ray_shooting

    diagnostics = evaluator._exact_family_prediction_details(np.asarray([], dtype=float), family)

    assert diagnostics["failed"] is True
    assert np.isnan(diagnostics["produced_image_count"])
    assert np.isnan(diagnostics["recovered_image_count"])
    assert diagnostics["multiplicity_failed"] is True
    assert diagnostics["multiplicity_failure_reason"] == "source_ray_shooting_failed"
    assert cache.multiplicity_mismatch_count == 1


def test_quick_diagnostics_evaluate_skips_exact_image_prediction() -> None:
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    family = SimpleNamespace(family_id="1", z_source=2.0, n_images=2)
    evaluator.state = SimpleNamespace(family_data=[family])
    evaluator.surrogate_enabled = False
    evaluator.quick_diagnostics = True
    evaluator.source_loglike = lambda _params: -12.0
    evaluator._family_source_summary = lambda _params: {"1": {"source_plane_rms": 0.25}}

    def fail_exact(_params, _family):
        raise AssertionError("quick diagnostics should not call exact image prediction")

    evaluator._exact_family_prediction = fail_exact

    result = evaluator.evaluate(np.asarray([0.0], dtype=float), validate_all_families=True)

    assert result.loglike == pytest.approx(-12.0)
    assert result.family_predictions["1"]["approx_image_rms_arcsec"] == pytest.approx(0.25)
    assert result.family_predictions["1"]["used_exact_refresh"] is False
    assert result.family_predictions["1"]["refresh_reason"] == "quick_diagnostics"
    assert result.family_predictions["1"]["x_pred"].shape == (2,)
    assert np.isnan(result.family_predictions["1"]["x_pred"]).all()
    assert result.family_predictions["1"]["y_pred"].shape == (2,)
    assert np.isnan(result.family_predictions["1"]["y_pred"]).all()


def test_exact_family_prediction_handles_zero_measurement_sigma() -> None:
    family = FamilyData(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=0.0,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 1.0]),
        y_obs=np.asarray([0.0, 0.0]),
        reliability=np.ones(2),
    )

    class FakeModel:
        def ray_shooting(self, _x, _y, _kwargs_lens):
            return (
                jnp.asarray([0.0, 2.0], dtype=jnp.float64),
                jnp.asarray([0.0, 0.0], dtype=jnp.float64),
            )

    class FakeSolver:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        def image_position_from_source(self, source_x, source_y, _kwargs_lens, **_kwargs):
            self.calls.append((float(source_x), float(source_y)))
            return np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 0.0], dtype=float)

    solver = FakeSolver()
    cache = FamilyValidationCache()
    evaluator = cluster_solver.ClusterJAXEvaluator.__new__(cluster_solver.ClusterJAXEvaluator)
    evaluator.validation_cache = {"1": cache}
    evaluator.source_plane_covariance_floor = 1.0e-6
    evaluator.sample_likelihood_mode = SAMPLE_LIKELIHOOD_SOURCE
    evaluator.timing_totals = {"exact_solver": 0.0}
    evaluator._get_exact_model_solver = lambda _z_source: (FakeModel(), solver)
    evaluator._build_packed_lens_state = lambda _params, _z_source: {}
    evaluator._packed_to_kwargs_lens = lambda _packed_state: []
    evaluator._source_sigma_int_numpy = lambda _params: 0.0
    evaluator._source_position_for_family_numpy = lambda _params, _family_id: None
    evaluator._match_images = lambda x_pred, y_pred, _family: (np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float))

    prediction = cluster_solver.ClusterJAXEvaluator._exact_family_prediction(
        evaluator,
        np.asarray([], dtype=float),
        family,
    )

    assert prediction is not None
    assert solver.calls == [(pytest.approx(1.0), pytest.approx(0.0))]
    assert cache.last_source_x == pytest.approx(1.0)
    assert cache.last_source_y == pytest.approx(0.0)
    assert np.isfinite(cache.source_plane_rms)


def test_sampling_refresh_runs_rejects_non_positive_values() -> None:
    args = argparse.Namespace(
        fit_mode="sequential",
        fit_method="svi+nuts",
        warmup=[10],
        samples=[5],
        sampling_refresh_runs=[0],
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="--sampling-refresh-runs values must be positive"):
        cluster_solver._normalize_stage_fit_controls(args)


def test_sampling_refresh_runs_rejects_unsupported_sampler() -> None:
    args = argparse.Namespace(
        fit_mode="sequential",
        fit_method="svi",
        warmup=[0],
        samples=[5],
        sampling_refresh_runs=[2],
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
    )

    with pytest.raises(SystemExit, match="supported only for --fit-method nuts or svi\\+nuts"):
        cluster_solver._normalize_stage_fit_controls(args)


def test_rerender_plots_banner_includes_stage_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    banners: list[tuple[str, str | None]] = []

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda _args, title, details=None: banners.append((title, details)))
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (_ for _ in ()).throw(RuntimeError("stop after banner")))
    args = argparse.Namespace()
    run_dir = tmp_path / "fit" / "stage3_image_plane"

    with pytest.raises(RuntimeError, match="stop after banner"):
        cluster_solver._rerender_plots(args, run_dir)

    assert banners == [("PLOTS ONLY: STAGE 3: stage3_image_plane", f"run_dir={run_dir}")]


def test_rerender_plots_preserves_saved_stage_likelihood_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeEvaluator:
        def __init__(self, *args: Any, sample_likelihood_mode: str = SAMPLE_LIKELIHOOD_SOURCE, **_kwargs: Any) -> None:
            self.sample_likelihood_mode = str(sample_likelihood_mode)
            self.surrogate_enabled = False
            self.timing_totals = {"plot_runtime": 0.0}
            captured["evaluator_sample_likelihood_mode"] = self.sample_likelihood_mode

        def reported_physical_to_latent_parameter_vector(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=float)

        def refresh_scaling_scatter_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def refresh_source_metric_cache(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def evaluate(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(loglike=0.0)

    state = SimpleNamespace(parameter_specs=[], fit_mode="joint")
    arrays = {
        "best_fit": np.empty((0,), dtype=float),
        "samples": np.empty((0, 0), dtype=float),
        "log_prob": np.empty((0,), dtype=float),
        "accept_prob": np.empty((0,), dtype=float),
        "diverging": np.empty((0,), dtype=bool),
        "num_steps": np.empty((0,), dtype=float),
    }
    saved_args = {
        "model_config": {"workflow": {"stage2_forward_mode": "critical-arc-anisotropic"}},
        "run_name": "fit/stage2_free_source_forward_fit",
        "fit_mode": "joint",
        "sample_likelihood_mode": SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE,
        "stage2_forward_mode": "critical-arc-anisotropic",
        "quick_diagnostics": False,
        "warmup": 0,
        "samples": 0,
        "chains": 1,
    }
    current_args = SolverRuntime(
        {
            "model_config": {"workflow": {"stage2_forward_mode": "none"}},
            "run_name": "fit",
            "sample_likelihood_mode": SAMPLE_LIKELIHOOD_SOURCE,
            "stage2_forward_mode": "none",
            "quick_diagnostics": False,
        }
    )

    monkeypatch.setattr(cluster_solver, "_configure_debug_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_run_logged_phase", lambda _args, _phase_name, fn, **_kwargs: fn())
    monkeypatch.setattr(cluster_solver, "_load_artifacts", lambda _artifacts_dir: (state, saved_args, arrays, {}))
    monkeypatch.setattr(cluster_solver, "_infer_previous_stage_best_values_for_plots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_state_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_maybe_convert_loaded_posterior_arrays_to_physical", lambda arrays, *_args: (arrays, False))
    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_log_active_approximation_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_solver_active_approximation_warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_log_posterior_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cluster_solver,
        "_generate_plots_and_tables",
        lambda **kwargs: captured.update(
            {
                "plot_sample_likelihood_mode": kwargs["args"].sample_likelihood_mode,
                "plot_stage2_forward_mode": kwargs["args"].stage2_forward_mode,
            }
        ),
    )
    monkeypatch.setattr(cluster_solver, "_regenerate_active_scaling_summary_from_diagnostics", lambda *_args, **_kwargs: False)

    cluster_solver._rerender_plots(
        current_args,
        tmp_path / "fit" / "stage2_free_source_forward_fit",
        exact_diagnostics_stage="stage2_free_source_forward_fit",
    )

    assert captured["evaluator_sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE
    assert captured["plot_sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE
    assert captured["plot_stage2_forward_mode"] == "critical-arc-anisotropic"


def test_sequential_stage2_linearized_runs_as_free_source_forward_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, bool, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                sample_likelihood_mode,
                bool(getattr(args, "use_magnitude_likelihood", False)),
                svi_init_physical_values,
                source_position_prior_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1200.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    monkeypatch.setattr(cluster_solver, "_member_shape_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_selected_independent_by_potfile_from_artifacts", lambda _artifacts_dir: [{2, 5}])
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "svi+nuts"],
        svi_steps=[1000, 400, 200],
        refresh_every=[100, 50, 25],
        warmup=[2000, 1000, 10],
        samples=[250, 100, 20],
        stage0_likelihood=cluster_solver.STAGE1_LIKELIHOOD_SOURCE,
        stage1_likelihood=cluster_solver.STAGE1_LIKELIHOOD_LOCAL_JACOBIAN,
        stage2_forward_mode=cluster_solver.STAGE2_FORWARD_MODE_LINEARIZED,
        stage2_sampling_engine=cluster_solver.STAGE2_SAMPLING_ENGINE_INHERIT,
        stage2_fresh_process=False,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        model_config=object(),
        use_magnitude_likelihood=True,
    )
    args = SolverRuntime(vars(args))
    stage_fit_controls = {
        "stage0": cluster_solver.StageFitControls("svi", 1000, 100, 2000, 250, 1, 8),
        "stage1": cluster_solver.StageFitControls("svi+nuts", 400, 50, 2000, 250, 1, 8),
        "stage2": cluster_solver.StageFitControls("svi+nuts", 200, 25, 1000, 100, 1, 8),
    }

    cluster_solver._run_sequential_v2(args, stage_fit_controls)

    assert [item[1] for item in calls] == [
        "fit/stage0_fast_initializer",
        "fit/stage1_backprojected_centroid_fit",
        "fit/stage2_free_source_forward_fit",
    ]
    assert calls[0][2] == SAMPLE_LIKELIHOOD_SOURCE
    assert calls[0][3] is False
    assert calls[1][2] == SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN
    assert calls[1][3] is False
    assert calls[1][4] == {"halo_v_disp": 1200.0}
    assert calls[2][2] == SAMPLE_LIKELIHOOD_LINEARIZED_FORWARD_BETA_IMAGE_PLANE
    assert calls[2][3] is True
    assert calls[2][4] == {"halo_v_disp": 1200.0}
    assert calls[2][5] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["configured_use_magnitude_likelihood"] is True
    assert summary["stage0_use_magnitude_likelihood"] is False
    assert summary["stage1_use_magnitude_likelihood"] is False
    assert summary["stage2_use_magnitude_likelihood"] is True
    assert summary["stage0_run_dir"].endswith("stage0_fast_initializer")
    assert summary["stage1_run_dir"].endswith("stage1_backprojected_centroid_fit")
    assert summary["stage2_run_dir"].endswith("stage2_free_source_forward_fit")
    assert summary["stage0_selected_free_scaling"] == 2
    assert summary["stage0_likelihood"] == cluster_solver.STAGE1_LIKELIHOOD_SOURCE
    assert summary["stage0_sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_SOURCE
    assert summary["stage0_source_position_policy"] == cluster_solver.CRITICAL_ARC_SOURCE_POSITION_POLICY_SAMPLED
    assert summary["stage1_sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_LOCAL_JACOBIAN
    assert summary["stage2_forward_mode"] == cluster_solver.STAGE2_FORWARD_MODE_LINEARIZED


def test_fresh_process_stage_accepts_pickled_solver_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = SolverRuntime(
        {
            "output_dir": str(tmp_path),
            "seed": 12345,
            "sampling_engine": "full_flat",
        }
    )
    monkeypatch.setattr(cluster_solver, "_run_single_stage_spawn_worker", _fresh_process_solver_runtime_echo_worker)

    run_dir = cluster_solver._run_single_stage_in_fresh_process(
        runtime,
        cluster_solver.STAGE2_FREE_SOURCE_FORWARD_FIT_DIR,
        "fit/stage2_free_source_forward_fit",
    )

    assert run_dir == tmp_path / "fit" / "stage2_free_source_forward_fit"


def test_sequential_stage2_critical_arc_runs_as_free_source_forward_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, str, bool, dict[str, float] | None, dict[str, tuple[float, float]] | None]] = []

    def fake_run_single_stage(
        args,
        fit_mode,
        run_name,
        sample_likelihood_mode=SAMPLE_LIKELIHOOD_SOURCE,
        svi_init_physical_values=None,
        source_position_prior_values=None,
        **_kwargs,
    ):
        calls.append(
            (
                fit_mode,
                run_name,
                sample_likelihood_mode,
                args.fit_method,
                bool(getattr(args, "use_magnitude_likelihood", False)),
                svi_init_physical_values,
                source_position_prior_values,
            )
        )
        return tmp_path / run_name

    monkeypatch.setattr(cluster_solver, "_run_single_stage", fake_run_single_stage)
    monkeypatch.setattr(cluster_solver, "_log_stage_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cluster_solver, "_physical_best_fit_values_from_artifacts", lambda _artifacts_dir: {"halo_v_disp": 1200.0})
    monkeypatch.setattr(cluster_solver, "_source_position_prior_values_from_artifacts", lambda _artifacts_dir: {"1": (0.1, -0.2)})
    monkeypatch.setattr(cluster_solver, "_member_shape_values_from_artifacts", lambda _artifacts_dir: {})
    monkeypatch.setattr(cluster_solver, "_selected_independent_by_potfile_from_artifacts", lambda _artifacts_dir: [{2, 5}])
    args = argparse.Namespace(
        run_name="fit",
        par_path="data/clustersim/input.par",
        output_dir=str(tmp_path),
        fit_method=["svi+nuts", "svi+nuts", "smc"],
        svi_steps=[1000, 400, 200],
        refresh_every=[100, 50, 25],
        warmup=[2000, 1000, 10],
        samples=[250, 100, 20],
        stage0_likelihood=cluster_solver.STAGE1_LIKELIHOOD_LOCAL_JACOBIAN,
        stage1_likelihood=cluster_solver.STAGE1_LIKELIHOOD_LOCAL_JACOBIAN,
        stage2_forward_mode=cluster_solver.STAGE2_FORWARD_MODE_CRITICAL_ARC,
        stage2_sampling_engine=cluster_solver.STAGE2_SAMPLING_ENGINE_INHERIT,
        stage2_fresh_process=False,
        image_plane_mode=IMAGE_PLANE_MODE_NONE,
        fit_mode="sequential",
        sampling_engine="refreshing_surrogate_flat",
        image_plane_newton_steps=0,
        source_position_parameterization="prior-whitened",
        use_magnitude_likelihood=False,
        critical_arc_critical_direction_sigma_arcsec=4.5,
        critical_arc_base_prob=0.2,
        critical_arc_max_prob=0.7,
        critical_arc_singular_threshold=0.15,
        sample_critical_arc_singular_threshold=True,
        critical_arc_singular_softness=0.04,
        critical_arc_lm_damping_relative=0.002,
        critical_arc_lm_damping_absolute=1.0e-5,
        critical_arc_lm_trust_radius_arcsec=18.0,
        model_config=object(),
    )
    args = SolverRuntime(vars(args))
    stage_fit_controls = {
        "stage0": cluster_solver.StageFitControls("svi", 1000, 100, 2000, 250, 1, 8),
        "stage1": cluster_solver.StageFitControls("svi+nuts", 400, 50, 2000, 250, 1, 8),
        "stage2": cluster_solver.StageFitControls("smc", 200, 25, 1000, 100, 1, 8),
    }

    cluster_solver._run_sequential_v2(args, stage_fit_controls)

    assert [item[1] for item in calls] == [
        "fit/stage0_fast_initializer",
        "fit/stage1_backprojected_centroid_fit",
        "fit/stage2_free_source_forward_fit",
    ]
    assert calls[2][2] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
    assert calls[2][3] == "smc"
    assert calls[0][4] is False
    assert calls[1][4] is False
    assert calls[2][4] is False
    assert calls[2][5] == {"halo_v_disp": 1200.0}
    assert calls[2][6] == {"1": (0.1, -0.2)}
    summary = json.loads((tmp_path / "fit" / "sequential_summary.json").read_text(encoding="utf-8"))
    assert summary["configured_use_magnitude_likelihood"] is False
    assert summary["stage0_use_magnitude_likelihood"] is False
    assert summary["stage1_use_magnitude_likelihood"] is False
    assert summary["stage2_use_magnitude_likelihood"] is False
    assert summary["stage0_run_dir"].endswith("stage0_fast_initializer")
    assert summary["stage2_run_dir"].endswith("stage2_free_source_forward_fit")
    assert summary["stage0_selected_free_scaling"] == 2
    assert summary["stage2_forward_mode"] == cluster_solver.STAGE2_FORWARD_MODE_CRITICAL_ARC
    assert summary["stage2_sample_likelihood_mode"] == SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE


def test_adaptive_active_scaling_count_uses_importance_curve() -> None:
    importance = np.asarray([10.0, 4.0, 1.0, 0.2, 0.05, 0.01])

    selected, cumulative_count, knee_count = _adaptive_active_scaling_count(
        importance,
        cumulative_fraction=0.95,
        min_count=2,
        max_count=5,
    )

    assert cumulative_count == 3
    assert 1 <= knee_count <= len(importance)
    assert selected == 3


def test_adaptive_active_scaling_count_respects_cap() -> None:
    importance = np.ones(10)

    selected, _cumulative_count, _knee_count = _adaptive_active_scaling_count(
        importance,
        cumulative_fraction=0.99,
        min_count=2,
        max_count=4,
    )

    assert selected == 4


def test_parameter_recovery_table_exact_truth_has_zero_bias() -> None:
    samples = np.asarray(
        [
            [1.0, 2.0, 0.15, 0.1, -0.2],
            [1.0, 2.0, 0.15, 0.1, -0.2],
            [1.0, 2.0, 0.15, 0.1, -0.2],
        ]
    )
    table = parameter_recovery_table(
        samples,
        ["halo.v_disp", "bcg.v_disp", "potfile.corekpc", "source.1.beta_x", "source.1.beta_y"],
        {
            "halo.v_disp": 1.0,
            "bcg.v_disp": 2.0,
            "potfile.corekpc": 0.15,
            "source.1.beta_x": 0.1,
            "source.1.beta_y": -0.2,
        },
        best_fit=np.asarray([1.0, 2.0, 0.15, 0.1, -0.2]),
    )

    np.testing.assert_allclose(table["bias"], 0.0)
    assert table["covered_68"].tolist() == [True, True, True, True, True]
    np.testing.assert_allclose(table["truth_percentile"], 1.0)


def test_parameter_truth_with_source_positions_derives_legacy_source_truth() -> None:
    truth = {
        "parameter_truth": {"halo.v_disp": 1000.0},
        "sources": [
            {"family_id": "1", "beta_x": 0.25, "beta_y": -0.15},
            {"family_id": "2", "beta_x": "-0.1", "beta_y": "0.3"},
        ],
    }

    values = _parameter_truth_with_source_positions(truth)

    assert values["halo.v_disp"] == 1000.0
    assert values["source.1.beta_x"] == pytest.approx(0.25)
    assert values["source.1.beta_y"] == pytest.approx(-0.15)
    assert values["source.2.beta_x"] == pytest.approx(-0.1)
    assert values["source.2.beta_y"] == pytest.approx(0.3)


def test_log10_abs_parameter_values_handles_signs_and_zero() -> None:
    values = _log10_abs_parameter_values(np.asarray([10.0, 0.1, -0.1, 0.0, 1.0e-6, -1.0e-6], dtype=float))

    np.testing.assert_allclose(
        values,
        np.asarray(
            [
                1.0,
                -1.0,
                -1.0,
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
                np.log10(PARAMETER_RECOVERY_LOG_ABS_FLOOR),
            ],
            dtype=float,
        ),
    )


def test_plot_parameter_recovery_writes_log_and_linear_pdfs(tmp_path: Path) -> None:
    parameter_df = pd.DataFrame(
        {
            "parameter": ["halo.v_disp", "source.1.beta_x", "source.1.beta_y", "source.2.beta_x"],
            "truth": [760.0, 0.25, -0.08, 0.0],
            "q16": [700.0, 0.1, -0.2, -0.05],
            "median": [780.0, 0.2, -0.1, 1.0e-6],
            "q84": [820.0, 0.4, -0.02, 0.05],
        }
    )
    log_path = tmp_path / "parameter_recovery_log.pdf"
    linear_path = tmp_path / "parameter_recovery_linear.pdf"

    validation._plot_parameter_recovery(parameter_df, log_path, scale="log_abs")
    validation._plot_parameter_recovery(parameter_df, linear_path, scale="linear")

    assert log_path.exists()
    assert log_path.stat().st_size > 0
    assert linear_path.exists()
    assert linear_path.stat().st_size > 0


def test_plot_parameter_recovery_linear_handles_negative_zero_positive(tmp_path: Path) -> None:
    parameter_df = pd.DataFrame(
        {
            "parameter": ["negative", "zero", "positive"],
            "truth": [-2.0, 0.0, 3.0],
            "q16": [-2.5, -0.1, 2.5],
            "median": [-2.1, 0.0, 3.1],
            "q84": [-1.8, 0.1, 3.6],
        }
    )
    path = tmp_path / "parameter_recovery_linear.pdf"

    validation._plot_parameter_recovery(parameter_df, path, scale="linear")

    assert path.exists()
    assert path.stat().st_size > 0


def test_magnification_recovery_table_handles_sign_and_small_truth() -> None:
    truth = pd.DataFrame(
        {
            "image_label": ["1.1", "1.2", "1.3"],
            "magnification_true": [2.0, -4.0, 0.0],
        }
    )
    recovered = pd.DataFrame(
        {
            "image_label": ["1.1", "1.2", "1.3"],
            "magnification_recovered": [2.5, 4.0, 0.1],
        }
    )

    table = magnification_recovery_table(truth, recovered, epsilon=0.5)

    assert table["parity_match"].tolist() == [True, False, False]
    np.testing.assert_allclose(table["abs_magnification_fractional_error"], [0.25, 0.0, 0.2])


def test_annular_surface_density_helper_returns_finite_positive_values() -> None:
    class FakeEvaluator:
        def _grouped_alpha_and_hessian_for_components(self, x, y, packed_state, component_indices):
            del component_indices
            radius = jnp.maximum(jnp.sqrt(jnp.square(x) + jnp.square(y)), 1.0e-6)
            scale = jnp.sum(jnp.asarray(packed_state.sigma0, dtype=jnp.float64))
            zeros = jnp.zeros_like(x, dtype=jnp.float64)
            return zeros, zeros, scale / radius, zeros, zeros, zeros

    packed_state = SimpleNamespace(sigma0=jnp.asarray([1.0], dtype=jnp.float64))
    values = validation._annular_surface_density_msun_per_arcsec2(
        FakeEvaluator(),
        packed_state,
        [0],
        np.asarray([2.0, 5.0], dtype=float),
        sigma_crit_angle=10.0,
        n_radial=8,
        n_azimuth=12,
    )

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    assert values[0] > values[1]


def test_capped_evenly_spaced_posterior_draws_preserves_small_inputs_and_spans_large_inputs() -> None:
    fewer = np.arange(20, dtype=float).reshape(10, 2)
    exact = np.arange(256, dtype=float).reshape(128, 2)
    larger = np.arange(400, dtype=float).reshape(200, 2)

    np.testing.assert_array_equal(
        validation._capped_evenly_spaced_posterior_draws(fewer),
        fewer,
    )
    np.testing.assert_array_equal(
        validation._capped_evenly_spaced_posterior_draws(exact),
        exact,
    )

    capped = validation._capped_evenly_spaced_posterior_draws(larger)
    expected_indices = np.linspace(
        0,
        larger.shape[0] - 1,
        validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP,
        dtype=int,
    )
    np.testing.assert_array_equal(capped, larger[expected_indices])
    assert capped.shape == (validation.RECOVERY_PROFILE_POSTERIOR_DRAW_CAP, 2)
    np.testing.assert_array_equal(capped[0], larger[0])
    np.testing.assert_array_equal(capped[-1], larger[-1])


def test_validation_recovery_progress_tracks_parent_and_subtasks(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)

    with validation._ValidationRecoveryProgress(argparse.Namespace(quiet=False)) as progress:
        progress.begin_phase("posterior uncertainty")
        subtask = progress.add_subtask("posterior uncertainty: draws x families", total=6)
        progress.update_subtask(subtask, "posterior uncertainty: draw=1/3 family=1 z=2.0000 failed_exact=0")
        progress.advance_subtask(subtask)
        progress.advance_phase()

    assert len(progress_instances) == 1
    recorder = progress_instances[0]
    assert recorder.kwargs["transient"] is True
    assert ("add_task", 1, "recovery: starting", 0) in recorder.events
    assert (
        "update",
        1,
        {"total": 1, "description": "recovery: posterior uncertainty"},
    ) in recorder.events
    assert ("add_task", 2, "posterior uncertainty: draws x families", 6) in recorder.events
    assert ("advance", 2, 1) in recorder.events
    assert ("advance", 1, 1) in recorder.events
    assert ("update", 1, {"description": "recovery: complete"}) in recorder.events

    quiet_instances = _install_recording_progress(monkeypatch)
    with validation._ValidationRecoveryProgress(argparse.Namespace(quiet=True)) as quiet_progress:
        quiet_progress.begin_phase("load inputs")
        quiet_subtask = quiet_progress.add_subtask("unused", total=1)
        quiet_progress.advance_subtask(quiet_subtask)
        quiet_progress.advance_phase()

    assert quiet_instances == []


def test_validation_mock_progress_tracks_redshifts_families_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(validation, "_log", lambda _args, message: messages.append(message))

    with validation._ValidationMockProgress(argparse.Namespace(quiet=False)) as progress:
        progress.callback(
            "subhalos_complete",
            {
                "requested_subhalos": 100,
                "selected_subhalos": 100,
                "parent_count": 500,
                "observable_count": 420,
                "retry_count": 0,
                "mag_faint_limit": 24.0,
            },
        )
        progress.callback(
            "redshift_start",
            {
                "z_source": 2.0,
                "redshift_index": 1,
                "redshift_count": 2,
                "lens_component_count": 102,
                "caustic_grid_pixels": 801,
            },
        )
        progress.callback(
            "redshift_complete",
            {
                "z_source": 2.0,
                "redshift_index": 1,
                "redshift_count": 2,
                "lens_component_count": 102,
                "caustic_grid_pixels": 801,
                "caustic_compute_window_arcsec": 160.0,
                "caustic_grid_scale_arcsec": 0.2,
                "caustic_count": 3,
            },
        )
        progress.callback(
            "family_start",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "max_attempts": 100,
            },
        )
        progress.callback(
            "family_attempt",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 25,
                "max_attempts": 100,
                "image_count": 2,
            },
        )
        progress.callback(
            "family_accept",
            {
                "family_index": 1,
                "family_count": 3,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 26,
                "max_attempts": 100,
                "image_count": 4,
                "caustic_index": 0,
            },
        )
        progress.callback(
            "outputs_complete",
            {
                "image_count": 12,
                "par_path": "mock/single_bcg_mock.par",
                "image_catalog_path": "mock/obs_arcs.cat",
                "truth_path": "mock/truth.json",
            },
        )

    assert len(progress_instances) == 1
    recorder = progress_instances[0]
    assert recorder.kwargs["transient"] is True
    assert ("add_task", 1, "mock caustics: starting", 2) in recorder.events
    assert ("add_task", 2, "mock families: starting", 3) in recorder.events
    assert ("advance", 1, 1) in recorder.events
    assert ("advance", 2, 1) in recorder.events
    assert any(
        event[0] == "update" and "attempt=25/100" in event[2].get("description", "")
        for event in recorder.events
    )
    assert any("mock subhalos selected=100/100" in message for message in messages)
    assert not any("mock caustics" in message for message in messages)
    assert not any("mock family" in message for message in messages)
    assert any("mock outputs complete images=12" in message for message in messages)


def test_validation_mock_progress_quiet_suppresses_rich_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    progress_instances = _install_recording_progress(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(validation, "_log", lambda _args, message: messages.append(message))

    with validation._ValidationMockProgress(argparse.Namespace(quiet=True)) as progress:
        progress.callback(
            "family_attempt",
            {
                "family_index": 1,
                "family_count": 1,
                "caustic_class": "primary",
                "z_source": 2.0,
                "attempt": 25,
                "max_attempts": 100,
                "image_count": 2,
            },
        )

    assert progress_instances == []
    assert messages == []


def test_recovered_model_tables_reports_partial_image_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_sampling_engines: list[str] = []
    families = [
        SimpleNamespace(
            family_id="1",
            z_source=2.0,
            effective_z_source=2.0,
            sigma_arcsec=0.1,
            n_images=2,
            image_labels=["1.1", "1.2"],
            x_obs=np.asarray([0.0, 1.0], dtype=float),
            y_obs=np.asarray([0.0, 0.0], dtype=float),
        ),
        SimpleNamespace(
            family_id="2",
            z_source=3.0,
            effective_z_source=3.0,
            sigma_arcsec=0.1,
            n_images=3,
            image_labels=["2.1", "2.2", "2.3"],
            x_obs=np.asarray([2.0, 3.0, 4.0], dtype=float),
            y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
        ),
    ]
    state = SimpleNamespace(parameter_specs=[], family_data=families)
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "2", "2", "2"],
            "image_label": ["1.1", "1.2", "2.1", "2.2", "2.3"],
            "x_obs_arcsec": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    class FakeModel:
        def magnification(self, x, y, kwargs_lens):
            del y, kwargs_lens
            return np.asarray(x, dtype=float) + 10.0

    class FakeEvaluator:
        source_plane_covariance_floor = 0.0

        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]
            constructed_sampling_engines.append(str(kwargs["sampling_engine"]))
            if str(kwargs["sampling_engine"]) != "full_flat":
                raise AssertionError("recovery diagnostics must not reuse surrogate sampling engines")

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent):
            return 0.2

        def evaluate(self, _sample_latent):
            return SimpleNamespace(
                family_predictions={
                    "1": {"source_x": 0.1, "source_y": 0.2, "source_plane_rms": 0.3, "failed": False},
                    "2": {"source_x": 1.1, "source_y": 1.2, "source_plane_rms": 1.3, "failed": False},
                }
            )

        def _get_exact_model_solver(self, _z_source):
            return FakeModel(), None

        def _build_packed_lens_state(self, _sample_latent, _z_source):
            return {}

        def _packed_to_kwargs_lens(self, _packed_state):
            return []

        def _exact_family_prediction_details(self, _sample_latent, family):
            if str(family.family_id) == "1":
                return {
                    "produced_image_count": 2,
                    "recovered_image_count": 2,
                    "missing_image_count": 0,
                    "extra_image_count": 0,
                    "multiplicity_failed": False,
                    "multiplicity_failure_reason": "",
                    "matched_model_x_arcsec": np.asarray([0.05, 1.05], dtype=float),
                    "matched_model_y_arcsec": np.asarray([0.0, 0.0], dtype=float),
                    "recovered_image_mask": np.asarray([True, True], dtype=bool),
                    "failed": False,
                    "exact_image_rms": 0.05,
                }
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([2.1, np.nan, 4.2], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0], dtype=float),
                "extra_model_y_arcsec": np.asarray([-2.0], dtype=float),
                "failed": True,
            }

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)

    magnification_df, image_df, source_df = validation._recovered_model_tables(
        state,
        np.asarray([0.0], dtype=float),
        images,
        artifact_args={"sampling_engine": "refreshing_surrogate_flat"},
    )

    assert constructed_sampling_engines == ["full_flat"]
    assert len(image_df) == 5
    indexed = image_df.set_index("image_label")
    assert indexed["image_recovery_status"].tolist() == ["recovered", "recovered", "recovered", "not_recovered", "recovered"]
    assert indexed["exact_image_prediction_failed"].tolist() == [False, False, False, True, False]
    assert indexed.loc["2.1", "x_model_arcsec"] == pytest.approx(2.1)
    assert np.isnan(indexed.loc["2.2", "x_model_arcsec"])
    assert indexed.loc["2.3", "image_residual_arcsec"] == pytest.approx(0.2)
    assert int(indexed.loc["2.1", "model_recovered_image_count"]) == 2
    assert int(indexed.loc["2.1", "model_missing_image_count"]) == 1
    assert int(indexed.loc["2.1", "model_extra_image_count"]) == 1
    assert magnification_df.set_index("image_label").loc["2.3", "magnification_recovered"] == pytest.approx(14.0)
    assert source_df.set_index("family_id").loc["1", "exact_image_rms_arcsec"] == pytest.approx(0.05)


def test_posterior_prediction_uncertainty_tables_advances_progress_per_draw_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_sampling_engines: list[str] = []

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]
            constructed_sampling_engines.append(str(kwargs["sampling_engine"]))
            if str(kwargs["sampling_engine"]) != "full_flat":
                raise AssertionError("recovery diagnostics must not reuse surrogate sampling engines")

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + 0.1,
                    "y_pred": family.y_obs + 0.2,
                    "source_x": 0.01 * float(family.family_id),
                    "source_y": 0.02 * float(family.family_id),
                    "source_plane_rms": 0.03,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {}

        def _packed_to_kwargs_lens(self, packed_state):
            return []

        def _exact_family_prediction(self, sample_latent, family):
            return family.x_obs + 0.01, family.y_obs + 0.02, 0.03

    class FakeProgress:
        def __init__(self) -> None:
            self.added: list[tuple[str, int | None]] = []
            self.updated: list[str] = []
            self.advanced: list[int | None] = []

        def add_subtask(self, description: str, total: int | None) -> int:
            self.added.append((description, total))
            return 17

        def update_subtask(self, task_id: int | None, description: str) -> None:
            self.updated.append(description)

        def advance_subtask(self, task_id: int | None) -> None:
            self.advanced.append(task_id)

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([0.0], dtype=float),
                y_obs=np.asarray([0.0], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([1.0], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "2"],
            "image_label": ["1.1", "2.1"],
            "x_obs_arcsec": [0.0, 1.0],
            "y_obs_arcsec": [0.0, 1.0],
        }
    )
    progress = FakeProgress()

    validation._posterior_prediction_uncertainty_tables(
        state,
        np.zeros((5, 0), dtype=float),
        images,
        max_draws=3,
        progress=progress,
        artifact_args={"sampling_engine": "refreshing_surrogate_flat"},
    )

    assert constructed_sampling_engines == ["full_flat"]
    assert progress.added == [("posterior uncertainty: draws x families", 6)]
    assert progress.advanced == [17] * 6
    assert len(progress.updated) == 6
    assert progress.updated[0] == "posterior uncertainty: draw=1/3 family=1 z=2.0000 failed_exact=0"
    assert progress.updated[-1] == "posterior uncertainty: draw=3/3 family=2 z=3.0000 failed_exact=0"


def test_posterior_prediction_uncertainty_tables_threaded_matches_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            sample_offset = float(kwargs_lens[0]["sample_offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + sample_offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + sample_offset,
                    "y_pred": family.y_obs - sample_offset,
                    "source_x": sample_offset + 0.1 * float(family.family_id),
                    "source_y": -sample_offset + 0.2 * float(family.family_id),
                    "source_plane_rms": 0.01 + sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {"sample_offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"sample_offset": float(packed_state["sample_offset"])}]

        def _exact_family_prediction(self, sample_latent, family):
            sample_offset = float(np.sum(sample_latent))
            return (
                family.x_obs + sample_offset + 0.01 * float(family.family_id),
                family.y_obs - sample_offset - 0.02 * float(family.family_id),
                0.03 + sample_offset + 0.001 * float(family.family_id),
            )

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=2,
                image_labels=["1.1", "1.2"],
                x_obs=np.asarray([0.0, 1.0], dtype=float),
                y_obs=np.asarray([0.5, 1.5], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([2.0], dtype=float),
                y_obs=np.asarray([2.5], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "2"],
            "image_label": ["1.1", "1.2", "2.1"],
            "x_obs_arcsec": [0.0, 1.0, 2.0],
            "y_obs_arcsec": [0.5, 1.5, 2.5],
        }
    )
    samples = np.asarray([[0.0], [0.5], [1.0], [1.5]], dtype=float)

    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    serial = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
    )
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    threaded = validation._posterior_prediction_uncertainty_tables(
        state,
        samples,
        images,
        max_draws=3,
    )

    for serial_df, threaded_df, key in zip(serial, threaded, ["image_label", "image_label", "family_id", "draw_index", "draw_index"]):
        pd.testing.assert_frame_equal(
            serial_df.sort_values(key).reset_index(drop=True),
            threaded_df.sort_values(key).reset_index(drop=True),
        )


def test_posterior_prediction_uncertainty_tables_threaded_skips_failed_exact_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[str] = []
    exact_call_lock = threading.Lock()

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            return np.ones_like(np.asarray(x, dtype=float))

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": family.x_obs + sample_offset,
                    "y_pred": family.y_obs + sample_offset,
                    "source_x": sample_offset,
                    "source_y": sample_offset,
                    "source_plane_rms": sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {}

        def _packed_to_kwargs_lens(self, packed_state):
            return []

        def _exact_family_prediction(self, sample_latent, family):
            with exact_call_lock:
                exact_calls.append(str(family.family_id))
            if str(family.family_id) == "2":
                return None
            return family.x_obs + 0.01, family.y_obs + 0.02, 0.03

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([0.0], dtype=float),
                y_obs=np.asarray([0.0], dtype=float),
            ),
            SimpleNamespace(
                family_id="2",
                z_source=3.0,
                n_images=1,
                image_labels=["2.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([1.0], dtype=float),
            ),
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "2"],
            "image_label": ["1.1", "2.1"],
            "x_obs_arcsec": [0.0, 1.0],
            "y_obs_arcsec": [0.0, 1.0],
        }
    )

    _mag_df, _image_df, source_df, posterior_rms_df, posterior_residual_draws_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        images,
        max_draws=3,
    )

    assert exact_calls.count("1") == 3
    assert exact_calls.count("2") == 1
    family2 = source_df[source_df["family_id"] == "2"].iloc[0]
    assert np.isnan(family2["exact_image_rms_q50"])
    assert posterior_rms_df["exact_failed_family_count"].tolist() == [1, 1, 1]
    assert posterior_rms_df["point_recovered_image_count"].tolist() == [1, 1, 1]
    assert posterior_residual_draws_df["draw_index"].nunique() == 3
    assert posterior_residual_draws_df["point_recovered"].sum() == 3


def test_posterior_prediction_uncertainty_tables_keeps_partial_exact_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_offsets: list[float] = []

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            del y
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            offset = float(np.sum(sample_latent))
            return {
                "1": {
                    "source_x": offset,
                    "source_y": -offset,
                    "source_plane_rms": offset + 0.5,
                }
            }

        def _get_exact_model_solver(self, _z_source):
            return FakeExactModel(), None

        def _build_packed_lens_state(self, sample_latent, _z_source):
            return {"offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"offset": float(packed_state["offset"])}]

        def _exact_family_prediction_details(self, sample_latent, _family):
            offset = float(np.sum(sample_latent))
            exact_offsets.append(offset)
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([0.1 + offset, np.nan, 2.1 + offset], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0 + offset], dtype=float),
                "extra_model_y_arcsec": np.asarray([-2.0], dtype=float),
                "arc_recovery_status": np.asarray(["point_recovered", "arc_supported", "point_recovered"], dtype=object),
                "arc_aware_image_residual_arcsec": np.asarray([0.1 + offset, 0.05 + offset, 0.1 + offset], dtype=float),
                "arc_noncritical_direction_residual_arcsec": np.asarray([0.02, 0.05 + offset, 0.03], dtype=float),
                "arc_critical_direction_residual_arcsec": np.asarray([0.0, 5.0 + offset, 0.0], dtype=float),
                "arc_s_min": np.asarray([0.2, 0.04, 0.2], dtype=float),
                "arc_prior_probability": np.asarray([0.4, 0.75, 0.4], dtype=float),
                "p_arc": np.asarray([0.3, 0.85, 0.3], dtype=float),
                "arc_log_odds": np.asarray([-0.8, 1.7, -0.8], dtype=float),
                "arc_supported_mask": np.asarray([False, True, False], dtype=bool),
                "arc_aware_image_rms_arcsec": 0.25 + offset,
                "failed": True,
            }

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=3,
                image_labels=["1.1", "1.2", "1.3"],
                x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
                y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
            )
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1", "1", "1"],
            "image_label": ["1.1", "1.2", "1.3"],
            "x_obs_arcsec": [0.0, 1.0, 2.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
        }
    )

    _mag_df, image_df, source_df, posterior_rms_df, posterior_residual_draws_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        images,
        max_draws=3,
    )

    assert exact_offsets == [0.0, 1.0, 2.0]
    indexed = image_df.set_index("image_label")
    assert indexed.loc["1.1", "x_model_q50"] == pytest.approx(1.1)
    assert np.isnan(indexed.loc["1.2", "x_model_q50"])
    assert indexed.loc["1.3", "x_model_q50"] == pytest.approx(3.1)
    assert indexed.loc["1.3", "image_residual_q50"] == pytest.approx(1.1)
    assert indexed.loc["1.2", "arc_aware_image_residual_q50"] == pytest.approx(1.05)
    assert source_df.set_index("family_id").loc["1", "arc_aware_image_rms_q50"] == pytest.approx(1.25)
    expected_point_rms = [
        math.sqrt((0.1**2 + 0.1**2) / 2.0),
        math.sqrt((1.1**2 + 1.1**2) / 2.0),
        math.sqrt((2.1**2 + 2.1**2) / 2.0),
    ]
    expected_arc_rms = [
        math.sqrt((0.1**2 + 0.1**2 + 0.05**2) / 3.0),
        math.sqrt((1.1**2 + 1.1**2 + 1.05**2) / 3.0),
        math.sqrt((2.1**2 + 2.1**2 + 2.05**2) / 3.0),
    ]
    np.testing.assert_allclose(posterior_rms_df["point_image_rms_arcsec"].to_numpy(dtype=float), expected_point_rms)
    np.testing.assert_allclose(posterior_rms_df["arc_aware_image_rms_arcsec"].to_numpy(dtype=float), expected_arc_rms)
    assert posterior_rms_df["point_recovered_image_count"].tolist() == [2, 2, 2]
    assert posterior_rms_df["arc_aware_recovered_image_count"].tolist() == [3, 3, 3]
    assert posterior_rms_df["arc_supported_image_count"].tolist() == [1, 1, 1]
    assert set(posterior_residual_draws_df.columns) >= {
        "draw_index",
        "image_label",
        "family_id",
        "point_image_residual_arcsec",
        "arc_aware_image_residual_arcsec",
        "point_recovered",
        "arc_aware_recovered",
        "arc_supported",
        "exact_failed",
    }
    assert len(posterior_residual_draws_df) == 9
    assert posterior_residual_draws_df["point_recovered"].sum() == 6
    assert posterior_residual_draws_df["arc_aware_recovered"].sum() == 9
    assert posterior_residual_draws_df["arc_supported"].sum() == 3


def test_posterior_prediction_uncertainty_tables_approximate_uses_median_std_without_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_calls: list[str] = []

    class FakeExactModel:
        def magnification(self, x, y, kwargs_lens):
            sample_offset = float(kwargs_lens[0]["sample_offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + sample_offset

    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            self.state = kwargs["state"]

        def reported_physical_to_latent_parameter_vector(self, sample):
            return np.asarray(sample, dtype=float)

        def _family_source_summary(self, sample_latent):
            sample_offset = float(np.sum(sample_latent))
            return {
                str(family.family_id): {
                    "x_pred": np.full(family.n_images, np.nan),
                    "y_pred": np.full(family.n_images, np.nan),
                    "source_x": sample_offset,
                    "source_y": 2.0 * sample_offset,
                    "source_plane_rms": 10.0 + sample_offset,
                }
                for family in self.state.family_data
            }

        def _get_exact_model_solver(self, z_source):
            return FakeExactModel(), object()

        def _build_packed_lens_state(self, sample_latent, z_source):
            return {"sample_offset": float(np.sum(np.asarray(sample_latent, dtype=float)))}

        def _packed_to_kwargs_lens(self, packed_state):
            return [{"sample_offset": float(packed_state["sample_offset"])}]

        def _exact_family_prediction(self, sample_latent, family):
            exact_calls.append(str(family.family_id))
            raise AssertionError("approximate mode should not call exact image validation")

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 2)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                n_images=1,
                image_labels=["1.1"],
                x_obs=np.asarray([1.0], dtype=float),
                y_obs=np.asarray([2.0], dtype=float),
            )
        ],
    )
    images = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [1.0],
            "y_obs_arcsec": [2.0],
        }
    )

    mag_df, image_df, source_df, posterior_rms_df, posterior_residual_draws_df = validation._posterior_prediction_uncertainty_tables(
        state,
        np.asarray([[0.0], [1.0], [3.0]], dtype=float),
        images,
        max_draws=3,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert exact_calls == []
    assert image_df.empty
    assert posterior_rms_df.empty
    assert posterior_residual_draws_df.empty
    mag_row = mag_df.set_index("image_label").loc["1.1"]
    source_row = source_df.set_index("family_id").loc["1"]

    mag_values = np.asarray([3.0, 4.0, 6.0], dtype=float)
    assert mag_row["magnification_q16"] == pytest.approx(float(np.median(mag_values) - np.std(mag_values)))
    assert mag_row["magnification_q50"] == pytest.approx(float(np.median(mag_values)))
    assert mag_row["magnification_q84"] == pytest.approx(float(np.median(mag_values) + np.std(mag_values)))

    source_x_values = np.asarray([0.0, 1.0, 3.0], dtype=float)
    assert source_row["source_x_q16"] == pytest.approx(float(np.median(source_x_values) - np.std(source_x_values)))
    assert source_row["source_x_q50"] == pytest.approx(float(np.median(source_x_values)))
    assert source_row["source_x_q84"] == pytest.approx(float(np.median(source_x_values) + np.std(source_x_values)))

    source_y_values = np.asarray([0.0, 2.0, 6.0], dtype=float)
    assert source_row["source_y_q16"] == pytest.approx(float(np.median(source_y_values) - np.std(source_y_values)))
    assert source_row["source_y_q50"] == pytest.approx(float(np.median(source_y_values)))
    assert source_row["source_y_q84"] == pytest.approx(float(np.median(source_y_values) + np.std(source_y_values)))

    source_rms_values = np.asarray([10.0, 11.0, 13.0], dtype=float)
    assert source_row["source_plane_rms_q16"] == pytest.approx(
        float(np.median(source_rms_values) - np.std(source_rms_values))
    )
    assert source_row["source_plane_rms_q50"] == pytest.approx(float(np.median(source_rms_values)))
    assert source_row["source_plane_rms_q84"] == pytest.approx(
        float(np.median(source_rms_values) + np.std(source_rms_values))
    )


def test_write_recovery_outputs_caps_mass_profile_posterior_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "config": {"z_lens": 0.4, "source_redshift": 2.0},
                "kwargs_lens": [],
                "images": [{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}],
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "out"
    all_samples = np.arange(200, dtype=float).reshape(200, 1)
    best_fit = np.asarray([123.0], dtype=float)
    posterior_truth_recovery_draws = 32
    captured_profile_samples: list[np.ndarray] = []
    captured_posterior_modes: list[str | None] = []
    captured_recovered_quick: list[bool] = []
    captured_logs: list[str] = []
    approximate_recovery_payload: dict[str, Any] = {}

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")

    def fake_mass_profiles(_state, profile_samples, _truth, radii_arcsec, **_kwargs):
        captured_profile_samples.append(np.asarray(profile_samples, dtype=float))
        return pd.DataFrame(), pd.DataFrame()

    def fake_posterior_uncertainty(*_args, **kwargs):
        mode = kwargs.get("posterior_diagnostic_mode")
        captured_posterior_modes.append(mode)
        posterior_rms = (
            pd.DataFrame()
            if mode == validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
            else pd.DataFrame(
                {
                    "draw_index": [1, 2, 3],
                    "point_image_rms_arcsec": [0.1, 0.2, 0.4],
                    "point_recovered_image_count": [1, 1, 1],
                    "arc_aware_image_rms_arcsec": [0.08, 0.15, 0.3],
                    "arc_aware_recovered_image_count": [1, 2, 2],
                    "arc_supported_image_count": [0, 1, 1],
                    "total_image_count": [1, 1, 1],
                    "exact_failed_family_count": [0, 0, 0],
                }
            )
        )
        posterior_residual_draws = (
            pd.DataFrame()
            if mode == validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE
            else pd.DataFrame(
                {
                    "draw_index": [1, 1, 2, 2],
                    "image_label": ["1.1", "1.2", "1.1", "1.2"],
                    "family_id": ["1", "1", "1", "1"],
                    "point_image_residual_arcsec": [0.1, np.nan, 0.2, np.nan],
                    "arc_aware_image_residual_arcsec": [0.1, 0.4, 0.2, 0.5],
                    "point_recovered": [True, False, True, False],
                    "arc_aware_recovered": [True, True, True, True],
                    "arc_supported": [False, True, False, True],
                    "exact_failed": [False, False, False, False],
                }
            )
        )
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            posterior_rms,
            posterior_residual_draws,
        )

    def fake_recovered_model_tables(*_args, **kwargs):
        captured_recovered_quick.append(bool(kwargs.get("quick_diagnostics", False)))
        return (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            pd.DataFrame({"image_label": ["1.1"], "image_residual_arcsec": [0.05]}),
            pd.DataFrame({"family_id": ["1"]}),
        )

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(validation, "_log", lambda _args, message: captured_logs.append(str(message)))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 5)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            SimpleNamespace(
                parameter_specs=[],
                lens_model_list=[],
                packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
            ),
            {},
            {"samples": all_samples, "best_fit": best_fit},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: ["p"])
    monkeypatch.setattr(validation, "_parameter_truth_with_source_positions", lambda _truth: {"p": 0.0})
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["p"],
                "truth": [0.0],
                "q16": [-1.0],
                "median": [0.0],
                "q84": [1.0],
                "bias": [0.0],
                "covered_68": [True],
            }
        ),
    )
    monkeypatch.setattr(validation, "_recovered_model_tables", fake_recovered_model_tables)
    monkeypatch.setattr(
        validation,
        "_posterior_prediction_uncertainty_tables",
        fake_posterior_uncertainty,
    )
    monkeypatch.setattr(validation, "_mass_and_surface_density_profiles_for_samples", fake_mass_profiles)
    monkeypatch.setattr(
        validation,
        "_potfile_corner_parameter_subset",
        lambda *_args, **_kwargs: ([], np.zeros((200, 0), dtype=float), np.zeros((0,), dtype=float)),
    )
    monkeypatch.setattr(
        validation,
        "_plot_corner_pdf",
        lambda output_dir, _samples, _specs, filename="corner.pdf", truth_values=None, best_fit_values=None, **_kwargs: touch(
            Path(output_dir) / filename
        ),
    )
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path, **_kwargs: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
        posterior_truth_recovery_draws=posterior_truth_recovery_draws,
        recovery_payload=approximate_recovery_payload,
    )

    approximate_warnings = [
        message
        for message in captured_logs
        if "warning approximations active" in message
        and "posterior_diagnostic_mode=approximate" in message
    ]
    assert len(approximate_warnings) == 1
    assert "median+/-std" in approximate_warnings[0]
    assert "exact per-draw image validation skipped" in approximate_warnings[0]
    assert approximate_recovery_payload["posterior_diagnostics"]["workers"] == 5
    assert approximate_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws"] == posterior_truth_recovery_draws
    assert approximate_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws_effective"] == posterior_truth_recovery_draws
    assert approximate_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_mode"] == "posterior"
    assert not (output_dir / "posterior_image_rms.csv").exists()
    assert not (output_dir / "posterior_image_residual_draws.csv").exists()
    assert "posterior_image_rms" in approximate_recovery_payload["tables"]
    assert "posterior_image_residual_draws" in approximate_recovery_payload["tables"]

    captured_logs.clear()
    exact_recovery_payload: dict[str, Any] = {}
    exact_output_dir = tmp_path / "out_exact"
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=exact_output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        posterior_truth_recovery_draws=posterior_truth_recovery_draws,
        recovery_payload=exact_recovery_payload,
    )

    assert not any("posterior_diagnostic_mode=approximate" in message for message in captured_logs)
    assert (exact_output_dir / "posterior_image_rms.csv").exists()
    assert (exact_output_dir / "posterior_image_residual_draws.csv").exists()
    assert "posterior_image_rms" in exact_recovery_payload["tables"]
    assert "posterior_image_residual_draws" in exact_recovery_payload["tables"]
    assert exact_recovery_payload["summary"]["posterior_point_image_rms_arcsec_median"] == pytest.approx(0.2)
    assert exact_recovery_payload["summary"]["posterior_arc_aware_image_rms_arcsec_median"] == pytest.approx(0.15)

    captured_logs.clear()
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_quick",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        posterior_truth_recovery_draws=posterior_truth_recovery_draws,
        quick_diagnostics=True,
    )

    assert any("quick_diagnostics=active" in message for message in captured_logs)
    assert captured_posterior_modes == [
        validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
        validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    ]
    assert captured_recovered_quick == [False, False, True]
    assert len(captured_profile_samples) == 3
    expected_indices = np.linspace(
        0,
        all_samples.shape[0] - 1,
        posterior_truth_recovery_draws,
        dtype=int,
    )
    for profile_samples in captured_profile_samples:
        assert profile_samples.shape == (posterior_truth_recovery_draws, 1)
        np.testing.assert_array_equal(profile_samples, all_samples[expected_indices])

    best_fit_only_recovery_payload: dict[str, Any] = {}
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_best_fit_only",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        posterior_truth_recovery_draws=0,
        recovery_payload=best_fit_only_recovery_payload,
    )

    assert captured_profile_samples[-1].shape == (1, 1)
    np.testing.assert_array_equal(captured_profile_samples[-1], best_fit.reshape(1, -1))
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws"] == 0
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws_effective"] == 1
    assert best_fit_only_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_mode"] == "best_fit"

    negative_draws_recovery_payload: dict[str, Any] = {}
    validation.write_recovery_outputs(
        run_dir,
        truth_path,
        mock_images_path,
        output_dir=tmp_path / "out_negative_draws",
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_EXACT,
        posterior_truth_recovery_draws=-4,
        recovery_payload=negative_draws_recovery_payload,
    )

    assert captured_profile_samples[-1].shape == (1, 1)
    np.testing.assert_array_equal(captured_profile_samples[-1], best_fit.reshape(1, -1))
    assert negative_draws_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws"] == -4
    assert negative_draws_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_draws_effective"] == 1
    assert negative_draws_recovery_payload["posterior_diagnostics"]["posterior_truth_recovery_mode"] == "best_fit"


def test_write_recovery_outputs_always_includes_critical_arc_support_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "config": {"z_lens": 0.4, "source_redshift": 2.0},
                "kwargs_lens": [],
                "images": [{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}],
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    touched: list[Path] = []

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")
        touched.append(path)

    state = SimpleNamespace(
        parameter_specs=[],
        lens_model_list=[],
        packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
        family_data=[
            SimpleNamespace(
                family_id="1",
                z_source=2.0,
                effective_z_source=2.0,
                n_images=1,
            )
        ],
    )
    image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "z_source": [2.0],
            "effective_z_source": [2.0],
            "image_residual_arcsec": [0.4],
            "arc_aware_image_residual_arcsec": [0.03],
            "arc_noncritical_direction_residual_arcsec": [0.03],
            "arc_critical_direction_residual_arcsec": [4.0],
            "arc_s_min": [0.04],
            "arc_prior_probability": [0.75],
            "p_arc": [0.82],
            "arc_recovery_status": ["arc_supported"],
            "arc_supported": [True],
            "model_produced_image_count": [1],
            "model_recovered_image_count": [0],
            "model_missing_image_count": [1],
            "model_extra_image_count": [0],
            "model_multiplicity_failed": [True],
            "model_multiplicity_failure_reason": ["missing_model_images"],
        }
    )

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            state,
            {"sample_likelihood_mode": SAMPLE_LIKELIHOOD_SOURCE},
            {"samples": np.zeros((3, 0), dtype=float), "best_fit": np.zeros((0,), dtype=float)},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: [])
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["p"],
                "truth": [0.0],
                "q16": [-1.0],
                "median": [0.0],
                "q84": [1.0],
                "bias": [0.0],
                "covered_68": [True],
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "_recovered_model_tables",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            image_df.copy(),
            pd.DataFrame({"family_id": ["1"]}),
        ),
    )
    monkeypatch.setattr(validation, "_posterior_prediction_uncertainty_tables", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_mass_and_surface_density_profiles_for_samples", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_potfile_corner_parameter_subset", lambda *_args, **_kwargs: ([], np.zeros((3, 0), dtype=float), np.zeros((0,), dtype=float)))
    monkeypatch.setattr(validation, "_plot_corner_pdf", lambda output_dir, _samples, _specs, filename="corner.pdf", **_kwargs: touch(Path(output_dir) / filename))
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path, **_kwargs: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_support_histogram", lambda _df, path, **_kwargs: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_support_phase_space", lambda _df, path, **_kwargs: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_critical_arc_recovery_by_family", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        tmp_path / "run",
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert outputs["critical_arc_support_histogram_plot"] == output_dir / "critical_arc_support_histogram.pdf"
    assert outputs["critical_arc_support_phase_space_plot"] == output_dir / "critical_arc_support_phase_space.pdf"
    assert outputs["critical_arc_recovery_by_family_plot"] == output_dir / "critical_arc_recovery_by_family.pdf"
    assert outputs["critical_arc_support_histogram_plot"] in touched
    assert outputs["critical_arc_support_phase_space_plot"] in touched
    assert outputs["critical_arc_recovery_by_family_plot"] in touched


def test_write_recovery_outputs_includes_cosmology_corner_for_cosmology_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "parameter_truth": {"cosmology_Om0": 0.3, "cosmology_w0": -1.0},
                "subhalos": [],
            }
        ),
        encoding="utf-8",
    )
    mock_images_path = tmp_path / "mock_images.json"
    mock_images_path.write_text(
        json.dumps([{"family_id": "1", "image_label": "1.1", "magnification_true": 2.0}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    samples = np.asarray([[0.28, -1.1], [0.30, -1.0], [0.32, -0.9]], dtype=float)
    best_fit = np.asarray([0.31, -0.95], dtype=float)
    specs = [
        ParameterSpec(
            name="cosmology.Om0",
            sample_name="cosmology_Om0",
            potential_id="cosmology",
            profile_type=0,
            field="Om0",
            prior_kind="uniform",
            lower=0.05,
            upper=0.6,
            step=0.01,
            component_family="cosmology",
        ),
        ParameterSpec(
            name="cosmology.w0",
            sample_name="cosmology_w0",
            potential_id="cosmology",
            profile_type=0,
            field="w0",
            prior_kind="uniform",
            lower=-2.0,
            upper=-0.3,
            step=0.05,
            component_family="cosmology",
        ),
    ]
    plot_calls: list[tuple[str, list[str], np.ndarray]] = []

    def fake_logged_phase(args, phase_name, fn, **kwargs):
        return fn()

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pdf", encoding="utf-8")

    def fake_plot_corner_pdf(
        output_dir_arg,
        plot_samples,
        plot_specs,
        filename="corner.pdf",
        truth_values=None,
        best_fit_values=None,
        previous_stage_best_values=None,
    ):
        del truth_values, best_fit_values, previous_stage_best_values
        plot_calls.append((str(filename), [spec.name for spec in plot_specs], np.asarray(plot_samples, dtype=float)))
        touch(Path(output_dir_arg) / str(filename))

    monkeypatch.setattr(validation, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(
        validation,
        "_load_plot_bundle",
        lambda _run_dir: (
            SimpleNamespace(
                parameter_specs=specs,
                lens_model_list=[],
                packed_lens_spec=SimpleNamespace(component_family=np.asarray([], dtype=int)),
            ),
            {},
            {"samples": samples, "best_fit": best_fit},
            {},
        ),
    )
    monkeypatch.setattr(validation, "_artifact_parameter_names", lambda _state: ["cosmology.Om0", "cosmology.w0"])
    monkeypatch.setattr(
        validation,
        "parameter_recovery_table",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "parameter": ["cosmology.Om0", "cosmology.w0"],
                "truth": [0.3, -1.0],
                "q16": [0.28, -1.1],
                "median": [0.30, -1.0],
                "q84": [0.32, -0.9],
                "bias": [0.0, 0.0],
                "covered_68": [True, True],
            }
        ),
    )
    monkeypatch.setattr(
        validation,
        "_recovered_model_tables",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"image_label": ["1.1"], "magnification_recovered": [2.1]}),
            pd.DataFrame({"image_label": ["1.1"], "image_residual_arcsec": [0.05]}),
            pd.DataFrame({"family_id": ["1"]}),
        ),
    )
    monkeypatch.setattr(validation, "_posterior_prediction_uncertainty_tables", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(validation, "_plot_corner_pdf", fake_plot_corner_pdf)
    monkeypatch.setattr(validation, "_plot_parameter_recovery", lambda _df, path, scale="log_abs": touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_magnification_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_absolute_magnification_recovery_grid", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(validation, "_plot_absolute_magnification_recovery", lambda _grid, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_image_residual_histogram", lambda _df, path, **_kwargs: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_source_recovery", lambda _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_shmf", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_subhalo_recovery_radial", lambda _truth, _df, path: touch(Path(path)))
    monkeypatch.setattr(validation, "_plot_validation_summary", lambda _summary, _uncertainty, path: touch(Path(path)))

    outputs = validation.write_recovery_outputs(
        tmp_path / "run",
        truth_path,
        mock_images_path,
        output_dir=output_dir,
        posterior_diagnostic_mode=validation.POSTERIOR_DIAGNOSTIC_MODE_APPROXIMATE,
    )

    assert outputs["cosmology_corner_plot"] == output_dir / "cosmology_corner.pdf"
    cosmology_calls = [call for call in plot_calls if call[0] == "cosmology_corner.pdf"]
    assert len(cosmology_calls) == 1
    assert cosmology_calls[0][1] == ["cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(cosmology_calls[0][2], samples)


def _truth_component_record(
    component_id: str,
    role: str,
    sigma0: float,
    *,
    catalog_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "component_id": component_id,
        "component_role": role,
        "profile_name": "DPIE_NIE",
        "kwargs_by_source_redshift": {
            "2.00000000": {
                "sigma0": sigma0,
                "Ra": 0.1,
                "Rs": 1.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        },
    }
    if catalog_id is not None:
        record["catalog_id"] = catalog_id
    return record


def test_combined_mass_surface_density_profiles_match_separate_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _build_packed_lens_state(self, sample_latent, z_source):
            scale = float(np.asarray(sample_latent, dtype=float).reshape(-1)[0])
            return SimpleNamespace(sigma0=jnp.asarray([scale, 2.0 * scale, 3.0 * scale], dtype=jnp.float64))

        def _grouped_alpha_and_hessian_for_components(self, x, y, packed_state, component_indices):
            indices = np.asarray(component_indices, dtype=int).reshape(-1)
            scale = jnp.sum(jnp.take(jnp.asarray(packed_state.sigma0, dtype=jnp.float64), jnp.asarray(indices, dtype=jnp.int32)))
            zeros = jnp.zeros_like(x, dtype=jnp.float64)
            return x * scale, zeros, scale * jnp.ones_like(x), zeros, zeros, zeros

        def release_runtime_caches(self) -> None:
            return None

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)

    state = SimpleNamespace(
        lens_model_list=["fake_halo", "fake_bcg", "fake_subhalo"],
        parameter_specs=[],
        scaling_component_records=[
            {"component_index": 2, "catalog_id": "member001"},
        ],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 0, 1], dtype=int),
            profile_type=np.asarray([cluster_solver.DP_IE_PROFILE] * 3, dtype=np.int32),
        ),
    )
    samples = np.asarray([[1.0], [3.0]], dtype=float)
    truth = {
        "config": {"z_lens": 0.4, "source_redshift": 2.0},
        "lens_components": [
            _truth_component_record("halo", "halo", 10.0),
            _truth_component_record("bcg", "bcg", 20.0),
            _truth_component_record("member001", "subhalo", 30.0, catalog_id="member001"),
        ],
    }
    radii = np.asarray([2.0, 5.0], dtype=float)

    mass_df, surface_df = validation._mass_and_surface_density_profiles_for_samples(state, samples, truth, radii)
    expected_mass_df = validation._deflection_profile_for_samples(state, samples, truth, radii)
    expected_surface_df = validation._surface_density_profile_for_samples(state, samples, truth, radii)

    pd.testing.assert_frame_equal(mass_df.reset_index(drop=True), expected_mass_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(surface_df.reset_index(drop=True), expected_surface_df.reset_index(drop=True))

    executor_worker_counts: list[int] = []
    submitted_samples: list[int] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            executor_worker_counts.append(int(max_workers))

        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

        def submit(self, fn: Any, sample_index: int, sample: np.ndarray) -> Future[Any]:
            submitted_samples.append(int(sample_index))
            future: Future[Any] = Future()
            try:
                future.set_result(fn(sample_index, sample))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 4)
    monkeypatch.setattr(validation, "ThreadPoolExecutor", InlineExecutor)
    threaded_mass_df, threaded_surface_df = validation._mass_and_surface_density_profiles_for_samples(
        state,
        samples,
        truth,
        radii,
    )

    assert executor_worker_counts == [2]
    assert submitted_samples == [1, 2]
    pd.testing.assert_frame_equal(threaded_mass_df.reset_index(drop=True), mass_df.reset_index(drop=True))
    pd.testing.assert_frame_equal(threaded_surface_df.reset_index(drop=True), surface_df.reset_index(drop=True))


def test_mass_surface_density_profiles_handle_subhalo_component_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _build_packed_lens_state(self, sample_latent, z_source):
            del z_source
            scale = float(np.asarray(sample_latent, dtype=float).reshape(-1)[0])
            return SimpleNamespace(
                sigma0=jnp.asarray([scale, 2.0 * scale, 3.0 * scale, 100.0 * scale, 4.0 * scale], dtype=jnp.float64)
            )

        def _grouped_alpha_and_hessian_for_components(self, x, y, packed_state, component_indices):
            indices = np.asarray(component_indices, dtype=int).reshape(-1)
            scale = jnp.sum(jnp.take(jnp.asarray(packed_state.sigma0, dtype=jnp.float64), jnp.asarray(indices, dtype=jnp.int32)))
            zeros = jnp.zeros_like(x, dtype=jnp.float64)
            return x * scale, zeros, scale * jnp.ones_like(x), zeros, zeros, zeros

        def release_runtime_caches(self) -> None:
            return None

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    monkeypatch.setattr(cluster_solver, "_convert_theta_to_latent", lambda sample, _specs: np.asarray(sample, dtype=float))
    monkeypatch.setattr(validation, "jax_cpu_worker_count", lambda: 1)

    state = SimpleNamespace(
        lens_model_list=["DPIE_NIE", "DPIE_NIE", "DPIE_NIE", "DPIE_NIE", "DPIE_NIE"],
        parameter_specs=[],
        scaling_component_records=[
            {"component_index": 2, "catalog_id": "member001", "free_component_index": 3},
            {"component_index": 4, "catalog_id": "member002", "free_component_index": -1},
        ],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 0, 1, 2, 1], dtype=int),
            profile_type=np.asarray([cluster_solver.DP_IE_PROFILE] * 5, dtype=np.int32),
        ),
    )
    truth = {
        "config": {"z_lens": 0.4, "source_redshift": 2.0},
        "lens_components": [
            _truth_component_record("halo", "halo", 10.0),
            _truth_component_record("bcg", "bcg", 20.0),
            _truth_component_record("member001", "subhalo", 30.0, catalog_id="member001"),
            _truth_component_record("member002", "subhalo", 40.0, catalog_id="member002"),
        ],
    }

    mass_df, surface_df = validation._mass_and_surface_density_profiles_for_samples(
        state,
        np.asarray([[1.0], [2.0]], dtype=float),
        truth,
        np.asarray([2.0, 5.0], dtype=float),
    )

    assert {"total", "halo", "bcg", "subhalos", "bcg_plus_subhalos"}.issubset(set(mass_df["component"]))
    assert {"total", "halo", "bcg", "subhalos", "bcg_plus_subhalos"}.issubset(set(surface_df["component"]))
    assert "auxiliary" not in set(mass_df["component"])
    assert "auxiliary" not in set(surface_df["component"])
    assert np.all(np.isfinite(mass_df["median"]))
    assert np.all(np.isfinite(surface_df["median"]))
    mass_by_component_radius = {
        (str(row["component"]), float(row["radius_arcsec"])): row
        for row in mass_df.to_dict("records")
    }
    assert mass_by_component_radius[("subhalos", 2.0)]["median"] == pytest.approx(312.0)
    assert mass_by_component_radius[("subhalos", 2.0)]["truth"] == pytest.approx(140.0)
    assert mass_by_component_radius[("bcg_plus_subhalos", 2.0)]["median"] == pytest.approx(318.0)
    assert mass_by_component_radius[("bcg_plus_subhalos", 2.0)]["truth"] == pytest.approx(180.0)
    assert mass_by_component_radius[("total", 2.0)]["median"] == pytest.approx(321.0)
    assert mass_by_component_radius[("total", 2.0)]["truth"] == pytest.approx(200.0)


def test_mass_surface_density_profiles_reject_legacy_positional_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    state = SimpleNamespace(
        lens_model_list=["DPIE_NIE", "DPIE_NIE"],
        parameter_specs=[],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 0], dtype=int),
            profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        ),
    )
    truth = {
        "config": {"z_lens": 0.4, "source_redshift": 2.0},
        "kwargs_lens_by_source_redshift": {
            "2.00000000": [
                {"sigma0": 10.0, "Ra": 0.1, "Rs": 1.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
            ]
        },
    }

    with pytest.raises(ValueError, match="lens_components.*legacy.*Regenerate"):
        validation._mass_and_surface_density_profiles_for_samples(
            state,
            np.asarray([[1.0]], dtype=float),
            truth,
            np.asarray([2.0], dtype=float),
        )


@pytest.mark.parametrize(
    ("lens_components", "message"),
    [
        (
            [
                _truth_component_record("halo", "halo", 10.0),
                _truth_component_record("bcg", "bcg", 20.0),
                _truth_component_record("member001", "subhalo_placeholder", 0.0, catalog_id="member001"),
            ],
            "placeholder.*Regenerate",
        ),
        (
            [
                _truth_component_record("halo", "halo", 10.0),
                _truth_component_record("halo", "halo", 11.0),
            ],
            "Duplicate truth component_id",
        ),
        (
            [
                _truth_component_record("halo", "halo", 10.0),
                _truth_component_record("bcg", "bcg", 20.0),
                _truth_component_record("member001", "subhalo", 30.0, catalog_id="member001"),
                _truth_component_record("member002", "subhalo", 40.0, catalog_id="member001"),
            ],
            "Duplicate subhalo truth catalog_id",
        ),
    ],
)
def test_mass_surface_density_profiles_reject_invalid_current_truth_components(
    monkeypatch: pytest.MonkeyPatch,
    lens_components: list[dict[str, Any]],
    message: str,
) -> None:
    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    state = SimpleNamespace(
        lens_model_list=["DPIE_NIE", "DPIE_NIE"],
        parameter_specs=[],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 0], dtype=int),
            profile_type=np.asarray([cluster_solver.DP_IE_PROFILE, cluster_solver.DP_IE_PROFILE], dtype=np.int32),
        ),
    )
    truth = {"config": {"z_lens": 0.4, "source_redshift": 2.0}, "lens_components": lens_components}

    with pytest.raises(ValueError, match=message):
        validation._mass_and_surface_density_profiles_for_samples(
            state,
            np.asarray([[1.0]], dtype=float),
            truth,
            np.asarray([2.0], dtype=float),
        )


def test_mass_surface_density_profiles_reject_unmatched_packed_physical_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(cluster_solver, "ClusterJAXEvaluator", FakeEvaluator)
    state = SimpleNamespace(
        lens_model_list=["DPIE_NIE", "DPIE_NIE", "DPIE_NIE"],
        parameter_specs=[],
        scaling_component_records=[{"component_index": 2, "catalog_id": "missing-member"}],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 0, 1], dtype=int),
            profile_type=np.asarray([cluster_solver.DP_IE_PROFILE] * 3, dtype=np.int32),
        ),
    )
    truth = {
        "config": {"z_lens": 0.4, "source_redshift": 2.0},
        "lens_components": [
            _truth_component_record("halo", "halo", 10.0),
            _truth_component_record("bcg", "bcg", 20.0),
            _truth_component_record("member001", "subhalo", 30.0, catalog_id="member001"),
        ],
    }

    with pytest.raises(ValueError, match="missing-member.*no matching truth component"):
        validation._mass_and_surface_density_profiles_for_samples(
            state,
            np.asarray([[1.0]], dtype=float),
            truth,
            np.asarray([2.0], dtype=float),
        )


def test_plot_surface_density_recovery_writes_pdf(tmp_path: Path) -> None:
    rows = []
    for component, label, scale in [("total", "total", 1.0), ("halo", "halo", 0.7)]:
        for radius in [2.0, 5.0, 10.0]:
            truth = 1.0e12 * scale / radius
            rows.append(
                {
                    "radius_arcsec": radius,
                    "component": component,
                    "component_label": label,
                    "truth": truth,
                    "q16": 0.9 * truth,
                    "median": 1.05 * truth,
                    "q84": 1.2 * truth,
                }
            )
    path = tmp_path / "surface_density_recovery.pdf"

    validation._plot_surface_density_recovery(pd.DataFrame(rows), path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_critical_caustic_recovery_writes_pdf(tmp_path: Path) -> None:
    truth_contour_z2 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.2, 0.0, 0.2, 0.0, -0.2]),
        beta_y=np.asarray([0.0, 0.2, 0.0, -0.2, 0.0]),
        critical_x=np.asarray([-2.0, 0.0, 2.0, 0.0, -2.0]),
        critical_y=np.asarray([0.0, 2.0, 0.0, -2.0, 0.0]),
        caustic_area_arcsec2=0.08,
        critical_area_arcsec2=8.0,
    )
    truth_contour_z9 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.0, 0.3, 0.0, -0.3]),
        beta_y=np.asarray([0.0, 0.3, 0.0, -0.3, 0.0]),
        critical_x=np.asarray([-3.0, 0.0, 3.0, 0.0, -3.0]),
        critical_y=np.asarray([0.0, 3.0, 0.0, -3.0, 0.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    recovered_contour_z2 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.18, 0.02, 0.22, 0.02, -0.18]),
        beta_y=np.asarray([0.01, 0.21, 0.01, -0.19, 0.01]),
        critical_x=np.asarray([-1.9, 0.1, 2.1, 0.1, -1.9]),
        critical_y=np.asarray([0.1, 2.1, 0.1, -1.9, 0.1]),
        caustic_area_arcsec2=0.08,
        critical_area_arcsec2=8.0,
    )
    recovered_contour_z9 = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.28, 0.02, 0.32, 0.02, -0.28]),
        beta_y=np.asarray([0.01, 0.31, 0.01, -0.29, 0.01]),
        critical_x=np.asarray([-2.9, 0.1, 3.1, 0.1, -2.9]),
        critical_y=np.asarray([0.1, 3.1, 0.1, -2.9, 0.1]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    images = pd.DataFrame(
        {
            "x_obs_arcsec": [-1.0, 1.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 1.0],
        }
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [-1.0, 1.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 1.0],
            "x_model_arcsec": [-0.9, 1.1, 0.1],
            "y_model_arcsec": [0.0, 0.1, 1.1],
        }
    )
    source_df = pd.DataFrame(
        {
            "beta_x": [0.0],
            "beta_y": [0.0],
            "source_x_recovered": [0.02],
            "source_y_recovered": [0.01],
        }
    )
    subhalo_df = pd.DataFrame(
        {
            "x_arcsec": [2.5],
            "y_arcsec": [-1.0],
            "luminosity_ratio": [1.0],
        }
    )
    path = tmp_path / "critical_caustic_recovery.pdf"

    validation._plot_critical_caustic_recovery(
        {"2.00000000": [truth_contour_z2], "9.00000000": [truth_contour_z9]},
        {"2.00000000": [recovered_contour_z2], "9.00000000": [recovered_contour_z9]},
        images,
        image_df,
        source_df,
        subhalo_df,
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_critical_caustic_recovery_limits_ticks_for_large_ranges(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    truth_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-13.0, 130.5, 274.0, -13.0]),
        beta_y=np.asarray([-13.0, 274.0, -13.0, -13.0]),
        critical_x=np.asarray([-13.0, 130.5, 274.0, -13.0]),
        critical_y=np.asarray([-13.0, 274.0, -13.0, -13.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    recovered_contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-12.0, 131.0, 273.0, -12.0]),
        beta_y=np.asarray([-12.0, 273.0, -12.0, -12.0]),
        critical_x=np.asarray([-12.0, 131.0, 273.0, -12.0]),
        critical_y=np.asarray([-12.0, 273.0, -12.0, -12.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    path = tmp_path / "critical_caustic_recovery_large_range.pdf"

    with caplog.at_level(logging.WARNING, logger="matplotlib.ticker"):
        validation._plot_critical_caustic_recovery(
            {"9.00000000": [truth_contour]},
            {"9.00000000": [recovered_contour]},
            pd.DataFrame({"x_obs_arcsec": [-1.0], "y_obs_arcsec": [0.0]}),
            pd.DataFrame(
                {
                    "x_obs_arcsec": [-1.0],
                    "y_obs_arcsec": [0.0],
                    "x_model_arcsec": [-0.9],
                    "y_model_arcsec": [0.1],
                }
            ),
            pd.DataFrame(
                {
                    "beta_x": [0.0],
                    "beta_y": [0.0],
                    "source_x_recovered": [0.02],
                    "source_y_recovered": [0.01],
                }
            ),
            pd.DataFrame({"x_arcsec": [2.5], "y_arcsec": [-1.0], "luminosity_ratio": [1.0]}),
            path,
        )

    assert path.exists()
    assert path.stat().st_size > 0
    assert _locator_max_tick_messages(caplog) == []


def test_select_critical_caustic_plot_contours_keeps_z9_only() -> None:
    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([0.0, 0.1, 0.0]),
        beta_y=np.asarray([0.0, 0.1, 0.0]),
        critical_x=np.asarray([0.0, 1.0, 0.0]),
        critical_y=np.asarray([0.0, 1.0, 0.0]),
        caustic_area_arcsec2=0.01,
        critical_area_arcsec2=1.0,
    )

    selected = validation._select_critical_caustic_plot_contours(
        {
            "2.00000000": [contour],
            "9.00000000": [contour],
            "bad-key": [contour],
            "9.00000200": [contour],
            "9.00000050": [],
        }
    )

    assert selected == {"9.00000000": [contour]}


def test_plot_critical_caustic_recovery_ignores_marker_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    contour = validation.CausticContour(
        caustic_index=0,
        caustic_class="primary",
        beta_x=np.asarray([-0.3, 0.0, 0.3, 0.0, -0.3]),
        beta_y=np.asarray([0.0, 0.3, 0.0, -0.3, 0.0]),
        critical_x=np.asarray([-3.0, 0.0, 3.0, 0.0, -3.0]),
        critical_y=np.asarray([0.0, 3.0, 0.0, -3.0, 0.0]),
        caustic_area_arcsec2=0.18,
        critical_area_arcsec2=18.0,
    )
    scatter_labels: list[str | None] = []
    original_scatter = Axes.scatter

    def record_scatter(self: Axes, *args: object, **kwargs: object) -> object:
        scatter_labels.append(kwargs.get("label"))  # type: ignore[arg-type]
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)

    validation._plot_critical_caustic_recovery(
        {"9.00000000": [contour]},
        {"9.00000000": [contour]},
        pd.DataFrame({"x_obs_arcsec": [-1.0], "y_obs_arcsec": [0.0]}),
        pd.DataFrame(
            {
                "x_obs_arcsec": [-1.0],
                "y_obs_arcsec": [0.0],
                "x_model_arcsec": [-0.9],
                "y_model_arcsec": [0.1],
            }
        ),
        pd.DataFrame(
            {
                "beta_x": [0.0],
                "beta_y": [0.0],
                "source_x_recovered": [0.02],
                "source_y_recovered": [0.01],
            }
        ),
        pd.DataFrame({"x_arcsec": [2.5], "y_arcsec": [-1.0], "luminosity_ratio": [1.0]}),
        tmp_path / "critical_caustic_recovery.pdf",
    )

    assert scatter_labels == ["truth caustic", "recovered caustic"]


def _write_aggregate_result(
    root: Path,
    *,
    seed: int,
    variant: str,
    stage: str,
    records: list[dict[str, object]] | None = None,
    magnification_records: list[dict[str, object]] | None = None,
    posterior_image_rms_records: list[dict[str, object]] | None = None,
    posterior_image_residual_draws_records: list[dict[str, object]] | None = None,
) -> Path:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    stage_alias = aggregate_plots.stage_alias(stage)
    path = (
        root
        / aggregate_plots.DEFAULT_RUN_NAME
        / f"seed_{seed:08d}"
        / variant
        / "recovery"
        / stage
        / "results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "validation": {
            "recovery": {
                stage_alias: {
                    "tables": {
                        "parameters": {
                            "columns": [
                                "parameter",
                                "truth",
                                "best_fit",
                                "q16",
                                "median",
                                "q84",
                                "bias",
                                "truth_percentile",
                                "covered_68",
                            ],
                            "records": records
                            or [
                                {
                                    "parameter": "halo.x_centre",
                                    "truth": 1.0,
                                    "best_fit": 1.05,
                                    "q16": 0.9,
                                    "median": 1.04,
                                    "q84": 1.2,
                                    "bias": 0.04,
                                    "truth_percentile": 0.4,
                                    "covered_68": True,
                                },
                                {
                                    "parameter": "halo.v_disp",
                                    "truth": 760.0,
                                    "best_fit": 770.0,
                                    "q16": 720.0,
                                    "median": 765.0,
                                    "q84": 805.0,
                                    "bias": 5.0,
                                    "truth_percentile": 0.45,
                                    "covered_68": True,
                                },
                            ],
                        },
                        "magnification": {
                            "columns": [
                                "image_label",
                                "magnification_true",
                                "magnification_recovered",
                                "magnification_q16",
                                "magnification_q50",
                                "magnification_q84",
                                "magnification_bias",
                                "abs_magnification_fractional_error",
                                "parity_match",
                            ],
                            "records": magnification_records
                            or [
                                {
                                    "image_label": "1.1",
                                    "magnification_true": 2.0,
                                    "magnification_recovered": 2.1,
                                    "magnification_q16": 1.8,
                                    "magnification_q50": 2.05,
                                    "magnification_q84": 2.3,
                                    "magnification_bias": 0.1,
                                    "abs_magnification_fractional_error": 0.05,
                                    "parity_match": True,
                                },
                                {
                                    "image_label": "1.2",
                                    "magnification_true": -4.0,
                                    "magnification_recovered": -3.7,
                                    "magnification_q16": -4.2,
                                    "magnification_q50": -3.8,
                                    "magnification_q84": -3.4,
                                    "magnification_bias": 0.3,
                                    "abs_magnification_fractional_error": 0.075,
                                    "parity_match": True,
                                },
                            ],
                        },
                        "posterior_image_rms": {
                            "columns": [
                                "draw_index",
                                "point_image_rms_arcsec",
                                "point_recovered_image_count",
                                "arc_aware_image_rms_arcsec",
                                "arc_aware_recovered_image_count",
                                "arc_supported_image_count",
                                "total_image_count",
                                "exact_failed_family_count",
                            ],
                            "records": posterior_image_rms_records
                            or [
                                {
                                    "draw_index": 0,
                                    "point_image_rms_arcsec": 0.18,
                                    "point_recovered_image_count": 42,
                                    "arc_aware_image_rms_arcsec": 0.14,
                                    "arc_aware_recovered_image_count": 45,
                                    "arc_supported_image_count": 3,
                                    "total_image_count": 48,
                                    "exact_failed_family_count": 0,
                                },
                                {
                                    "draw_index": 1,
                                    "point_image_rms_arcsec": 0.22,
                                    "point_recovered_image_count": 41,
                                    "arc_aware_image_rms_arcsec": 0.16,
                                    "arc_aware_recovered_image_count": 45,
                                    "arc_supported_image_count": 4,
                                    "total_image_count": 48,
                                    "exact_failed_family_count": 0,
                                },
                            ],
                        },
                        "posterior_image_residual_draws": {
                            "columns": [
                                "draw_index",
                                "image_label",
                                "family_id",
                                "point_image_residual_arcsec",
                                "arc_aware_image_residual_arcsec",
                                "point_recovered",
                                "arc_aware_recovered",
                                "arc_supported",
                                "exact_failed",
                            ],
                            "records": posterior_image_residual_draws_records
                            or [
                                {
                                    "draw_index": 0,
                                    "image_label": "1.1",
                                    "family_id": "1",
                                    "point_image_residual_arcsec": 0.12,
                                    "arc_aware_image_residual_arcsec": 0.10,
                                    "point_recovered": True,
                                    "arc_aware_recovered": True,
                                    "arc_supported": False,
                                    "exact_failed": False,
                                },
                                {
                                    "draw_index": 0,
                                    "image_label": "1.2",
                                    "family_id": "1",
                                    "point_image_residual_arcsec": np.nan,
                                    "arc_aware_image_residual_arcsec": 0.55,
                                    "point_recovered": False,
                                    "arc_aware_recovered": True,
                                    "arc_supported": True,
                                    "exact_failed": False,
                                },
                                {
                                    "draw_index": 1,
                                    "image_label": "1.1",
                                    "family_id": "1",
                                    "point_image_residual_arcsec": 0.18,
                                    "arc_aware_image_residual_arcsec": 0.16,
                                    "point_recovered": True,
                                    "arc_aware_recovered": True,
                                    "arc_supported": False,
                                    "exact_failed": False,
                                },
                                {
                                    "draw_index": 1,
                                    "image_label": "1.2",
                                    "family_id": "1",
                                    "point_image_residual_arcsec": np.nan,
                                    "arc_aware_image_residual_arcsec": np.nan,
                                    "point_recovered": False,
                                    "arc_aware_recovered": False,
                                    "arc_supported": False,
                                    "exact_failed": False,
                                },
                            ],
                        },
                    }
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_aggregate_combined_recovery(
    root: Path,
    *,
    seed: int,
    variant: str,
    truth_masses: list[float],
    recovered_log_masses: list[float],
    stages: tuple[str, ...] = ("stage2_free_source_forward_fit",),
) -> Path:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    path = root / aggregate_plots.DEFAULT_RUN_NAME / f"seed_{seed:08d}" / variant / "recovery" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mock_cluster": {
            "truth": {
                "subhalos": [
                    {"id": f"member{index:03d}", "subhalo_dpie_mass_msun": mass}
                    for index, mass in enumerate(truth_masses, start=1)
                ],
                "subhalo_selection": {
                    "schechter_alpha": -1.3,
                    "mass_ref": 1.0e12,
                    "dpie_mass_anchor_msun": 8.2e12,
                    "mass_luminosity_exponent": 1.0,
                },
            }
        },
        "solver": {
            "stage_manifests": [
                {
                    "stage": stage,
                    "table_artifacts": {
                        "scaling_relation_summary.csv": {
                            "data": {
                                "columns": ["catalog_id", "scaling_log10_mass_msun_best"],
                                "records": [
                                    {
                                        "catalog_id": f"member{index:03d}",
                                        "scaling_log10_mass_msun_best": log_mass,
                                    }
                                    for index, log_mass in enumerate(recovered_log_masses, start=1)
                                ],
                            }
                        }
                    },
                }
                for stage in stages
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_parameter_recovery_loads_stage_and_variant_methods(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for stage in aggregate_plots.DEFAULT_STAGES:
        for variant in aggregate_plots.DEFAULT_VARIANTS:
            _write_aggregate_result(campaign, seed=12345, variant=variant, stage=stage)

    parameter_df, inventory_df = aggregate_plots.collect_parameter_recovery(campaign)

    assert len(parameter_df) == 8
    assert inventory_df["loaded"].all()
    assert set(parameter_df["method_label"]) == {
        "Stage 1 No flux",
        "Stage 1 Flux",
        "Stage 2 No flux",
        "Stage 2 Flux",
    }
    assert set(parameter_df["stage_label"]) == {"Stage 1", "Stage 2"}
    assert {"truth", "median", "q16", "q84", "covered_68"} <= set(parameter_df.columns)


def test_aggregate_magnification_recovery_loads_stage_and_variant_methods(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed in (12345, 12346):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                _write_aggregate_result(campaign, seed=seed, variant=variant, stage=stage)

    magnification_df, inventory_df = aggregate_plots.collect_magnification_recovery(campaign)

    assert len(magnification_df) == 16
    assert inventory_df["loaded"].all()
    assert set(magnification_df["stage_label"]) == {"Stage 1", "Stage 2"}
    assert set(magnification_df["variant"]) == {"anisotropic", "anisotropic_maglikelihood"}
    assert set(magnification_df["seed"]) == {12345, 12346}
    assert {
        "magnification_true",
        "magnification_q16",
        "magnification_q50",
        "magnification_q84",
    } <= set(magnification_df.columns)
    assert magnification_df["magnification_q50"].dtype.kind in {"f", "i"}


def test_aggregate_image_residual_recovery_loads_posterior_rms(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed in (12345, 12346):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                _write_aggregate_result(campaign, seed=seed, variant=variant, stage=stage)

    image_residual_df, inventory_df = aggregate_plots.collect_image_residual_recovery(campaign)
    seed_summary_df = aggregate_plots.image_residual_seed_summary_rows(image_residual_df)

    assert len(image_residual_df) == 16
    assert inventory_df["loaded"].all()
    assert set(image_residual_df["stage_label"]) == {"Stage 1", "Stage 2"}
    assert set(image_residual_df["variant"]) == {"anisotropic", "anisotropic_maglikelihood"}
    assert set(image_residual_df["seed"]) == {12345, 12346}
    assert {
        "point_image_rms_arcsec",
        "arc_aware_image_rms_arcsec",
        "point_recovered_image_count",
        "arc_supported_image_count",
    } <= set(image_residual_df.columns)
    assert image_residual_df["point_image_rms_arcsec"].dtype.kind in {"f", "i"}
    assert len(seed_summary_df) == 8
    assert "point_image_rms_arcsec_q50" in seed_summary_df
    assert "arc_aware_image_rms_arcsec_q50" in seed_summary_df
    assert seed_summary_df["point_image_rms_arcsec_q50"].iloc[0] == pytest.approx(0.20)
    assert seed_summary_df["arc_aware_image_rms_arcsec_q50"].iloc[0] == pytest.approx(0.15)


def test_aggregate_image_residual_draws_load_and_summarize_distribution(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    residuals_with_tail = np.concatenate([np.full(100, 0.8229), np.array([2.999])])
    assert aggregate_plots._image_residual_distribution_range_max(residuals_with_tail) == pytest.approx(0.95)

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed in (12345, 12346):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                _write_aggregate_result(campaign, seed=seed, variant=variant, stage=stage)

    residual_draws_df, inventory_df = aggregate_plots.collect_image_residual_draws_recovery(campaign)
    summary_df = aggregate_plots.image_residual_distribution_summary_rows(residual_draws_df, bin_count=4)

    assert len(residual_draws_df) == 32
    assert inventory_df["loaded"].all()
    assert set(residual_draws_df["variant"]) == {"anisotropic", "anisotropic_maglikelihood"}
    assert set(residual_draws_df["stage_label"]) == {"Stage 1", "Stage 2"}
    assert {
        "arc_aware_image_residual_arcsec",
        "arc_aware_recovered",
        "arc_supported",
        "exact_failed",
    } <= set(residual_draws_df.columns)
    assert not summary_df.empty
    assert {
        "bin_center",
        "hist_count_q16",
        "hist_count_q50",
        "hist_count_q84",
        "cumulative_fraction_q16",
        "cumulative_fraction_q50",
        "cumulative_fraction_q84",
    } <= set(summary_df.columns)
    assert set(summary_df["variant"]) == {"anisotropic", "anisotropic_maglikelihood"}


def test_aggregate_magnification_recovery_plot_uses_variant_legend_and_paper_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    magnification_df = pd.DataFrame(
        [
            {
                "stage": "stage1_backprojected_centroid_fit",
                "variant": "anisotropic",
                "magnification_true": 2.0,
                "magnification_q16": 1.7,
                "magnification_q50": 2.1,
                "magnification_q84": 2.4,
            },
            {
                "stage": "stage1_backprojected_centroid_fit",
                "variant": "anisotropic_maglikelihood",
                "magnification_true": -3.0,
                "magnification_q16": -3.5,
                "magnification_q50": -3.1,
                "magnification_q84": -2.7,
            },
            {
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic",
                "magnification_true": 4.0,
                "magnification_q16": 3.7,
                "magnification_q50": 4.2,
                "magnification_q84": 4.5,
            },
        ]
    )
    saved_figures: list[Figure] = []
    errorbar_calls: list[dict[str, object]] = []
    scatter_calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar
    original_scatter = Axes.scatter

    def record_savefig(self: Figure, *_args: object, **_kwargs: object) -> None:
        saved_figures.append(self)

    def record_errorbar(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        errorbar_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "yerr": kwargs.get("yerr"),
                "kwargs": dict(kwargs),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    def record_scatter(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        scatter_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "kwargs": dict(kwargs),
            }
        )
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    try:
        aggregate_plots.plot_aggregate_magnification_recovery(
            magnification_df,
            tmp_path / "magnification_recovery",
            ("pdf",),
        )
        assert len(saved_figures) == 1
        ax = saved_figures[0].axes[0]
        assert ax.get_xlabel() == r"Image $|\mu|$ (truth)"
        assert ax.get_ylabel() == r"Image $|\mu|$ (fit)"
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        legend = ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["No flux", "Flux"]
        assert "Stage 1" not in labels
        assert "Stage 2" not in labels
        assert "Stage 2 No flux" not in labels
        mag_call = next(call for call in errorbar_calls if np.asarray(call["x"]).shape == (1,) and call["x"][0] == pytest.approx(3.0))
        np.testing.assert_allclose(mag_call["y"], [3.1])
        errorbar_kwargs = mag_call["kwargs"]
        assert errorbar_kwargs["fmt"] == "none"
        assert errorbar_kwargs["elinewidth"] == pytest.approx(0.65)
        assert errorbar_kwargs["capsize"] == pytest.approx(1.8)
        assert errorbar_kwargs["capthick"] == pytest.approx(0.65)
        assert errorbar_kwargs["alpha"] == pytest.approx(0.30)
        assert errorbar_kwargs["zorder"] == 1
        assert errorbar_kwargs["label"] == "_nolegend_"
        yerr = mag_call["yerr"]
        assert isinstance(yerr, list)
        np.testing.assert_allclose(yerr[0], [0.4])
        np.testing.assert_allclose(yerr[1], [0.4])
        mag_scatter = next(call for call in scatter_calls if np.asarray(call["x"]).shape == (1,) and call["x"][0] == pytest.approx(3.0))
        np.testing.assert_allclose(mag_scatter["y"], [3.1])
        scatter_kwargs = mag_scatter["kwargs"]
        assert scatter_kwargs["marker"] == "D"
        assert scatter_kwargs["facecolors"] == "white"
        assert scatter_kwargs["edgecolors"] == "tab:orange"
        assert scatter_kwargs["s"] == pytest.approx(21.0)
        assert scatter_kwargs["linewidths"] == pytest.approx(0.75)
        assert scatter_kwargs["zorder"] == 2
        stage2_scatter = next(call for call in scatter_calls if np.asarray(call["x"]).shape == (1,) and call["x"][0] == pytest.approx(4.0))
        assert stage2_scatter["kwargs"]["marker"] == "o"
        assert stage2_scatter["kwargs"]["facecolors"] == "tab:blue"
    finally:
        plt.close("all")


def test_aggregate_image_residual_recovery_plot_summarizes_seed_posterior_rms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    image_residual_df = pd.DataFrame(
        [
            {
                "seed": 12345,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic",
                "draw_index": 0,
                "point_image_rms_arcsec": 0.18,
                "arc_aware_image_rms_arcsec": 0.12,
                "point_recovered_image_count": 40,
                "arc_aware_recovered_image_count": 43,
                "arc_supported_image_count": 3,
                "total_image_count": 45,
            },
            {
                "seed": 12345,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic",
                "draw_index": 1,
                "point_image_rms_arcsec": 0.22,
                "arc_aware_image_rms_arcsec": 0.14,
                "point_recovered_image_count": 41,
                "arc_aware_recovered_image_count": 43,
                "arc_supported_image_count": 2,
                "total_image_count": 45,
            },
            {
                "seed": 12346,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "draw_index": 0,
                "point_image_rms_arcsec": 0.11,
                "arc_aware_image_rms_arcsec": 0.09,
                "point_recovered_image_count": 42,
                "arc_aware_recovered_image_count": 44,
                "arc_supported_image_count": 2,
                "total_image_count": 45,
            },
            {
                "seed": 12346,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "draw_index": 1,
                "point_image_rms_arcsec": 0.13,
                "arc_aware_image_rms_arcsec": 0.10,
                "point_recovered_image_count": 42,
                "arc_aware_recovered_image_count": 44,
                "arc_supported_image_count": 2,
                "total_image_count": 45,
            },
        ]
    )
    saved_figures: list[Figure] = []
    errorbar_calls: list[dict[str, object]] = []
    scatter_calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar
    original_scatter = Axes.scatter

    def record_savefig(self: Figure, *_args: object, **_kwargs: object) -> None:
        saved_figures.append(self)

    def record_errorbar(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        errorbar_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "yerr": kwargs.get("yerr"),
                "kwargs": dict(kwargs),
            }
        )
        return original_errorbar(self, x, y, *args, **kwargs)

    def record_scatter(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        scatter_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "kwargs": dict(kwargs),
            }
        )
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    try:
        aggregate_plots.plot_aggregate_image_residual_recovery(
            image_residual_df,
            tmp_path / "image_residual_recovery",
            ("pdf",),
        )
        assert len(saved_figures) == 1
        axes = saved_figures[0].axes
        assert len(axes) == 1
        assert axes[0].get_ylabel() == "image RMS (arcsec)"
        assert axes[0].get_title() == ""
        assert axes[0].get_xlabel() == "Fit variant"
        legend = axes[0].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "No flux",
            "Flux",
        ]
        assert all(call["kwargs"]["fmt"] == "none" for call in errorbar_calls)
        assert all(call["kwargs"]["elinewidth"] == pytest.approx(1.15) for call in errorbar_calls)
        assert all(call["kwargs"]["capsize"] == pytest.approx(3.0) for call in errorbar_calls)
        summary_calls = [call for call in scatter_calls if call["kwargs"].get("s") == pytest.approx(56.0)]
        assert len(summary_calls) == 2
        mag_summary = next(call for call in summary_calls if call["kwargs"]["marker"] == "D")
        np.testing.assert_allclose(mag_summary["y"], [0.095], rtol=0.0, atol=0.03)
        assert mag_summary["kwargs"]["facecolors"] == "tab:orange"
        assert not any(line.get_visible() for line in axes[0].get_xgridlines())
        assert not any(line.get_visible() for line in axes[0].get_ygridlines())
    finally:
        plt.close("all")


def test_aggregate_image_residual_distribution_plot_draws_histogram_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    residual_draws_df = pd.DataFrame(
        [
            {
                "seed": 12345,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic",
                "draw_index": 0,
                "arc_aware_image_residual_arcsec": 0.10,
                "arc_aware_recovered": True,
                "exact_failed": False,
            },
            {
                "seed": 12345,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic",
                "draw_index": 0,
                "arc_aware_image_residual_arcsec": 0.55,
                "arc_aware_recovered": True,
                "exact_failed": False,
            },
            {
                "seed": 12346,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "draw_index": 0,
                "arc_aware_image_residual_arcsec": 0.08,
                "arc_aware_recovered": True,
                "exact_failed": False,
            },
            {
                "seed": 12346,
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "draw_index": 0,
                "arc_aware_image_residual_arcsec": np.nan,
                "arc_aware_recovered": False,
                "exact_failed": False,
            },
        ]
    )
    saved_figures: list[Figure] = []

    def record_savefig(self: Figure, *_args: object, **_kwargs: object) -> None:
        saved_figures.append(self)

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    try:
        paths = aggregate_plots.plot_aggregate_image_residual_distribution(
            residual_draws_df,
            tmp_path / "mock_image_residual_distribution",
            ("pdf",),
        )
        assert paths == (tmp_path / "mock_image_residual_distribution.pdf",)
        assert len(saved_figures) == 1
        axes = saved_figures[0].axes
        assert len(axes) == 1
        assert axes[0].get_ylabel() == "N images"
        assert axes[0].get_xlabel() == "image residual (arcsec)"
        assert axes[0].get_xlim() == pytest.approx((0.0, 0.60))
        legend = axes[0].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["No flux", "Flux"]
    finally:
        plt.close("all")


def test_joint_sigma_rcut_recovery_rows_pivots_potfile_parameters() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    parameter_df = pd.DataFrame(
        [
            {
                "campaign": "campaign",
                "run_name": "single_bcg_production",
                "seed": 12345,
                "variant": "anisotropic",
                "stage": "stage2_free_source_forward_fit",
                "stage_label": "Stage 2",
                "method_label": "Stage 2 No flux",
                "result_path": "results.json",
                "parameter": "potfile.sigma",
                "truth": 245.0,
                "median": 240.0,
                "q16": 235.0,
                "q84": 248.0,
            },
            {
                "campaign": "campaign",
                "run_name": "single_bcg_production",
                "seed": 12345,
                "variant": "anisotropic",
                "stage": "stage2_free_source_forward_fit",
                "stage_label": "Stage 2",
                "method_label": "Stage 2 No flux",
                "result_path": "results.json",
                "parameter": "potfile.cutkpc",
                "truth": 187.0,
                "median": 210.0,
                "q16": 180.0,
                "q84": 260.0,
            },
            {
                "campaign": "campaign",
                "run_name": "single_bcg_production",
                "seed": 12345,
                "variant": "anisotropic",
                "stage": "stage2_free_source_forward_fit",
                "stage_label": "Stage 2",
                "method_label": "Stage 2 No flux",
                "result_path": "results.json",
                "parameter": "halo.v_disp",
                "truth": 760.0,
                "median": 765.0,
                "q16": 740.0,
                "q84": 790.0,
            },
            {
                "campaign": "campaign",
                "run_name": "single_bcg_production",
                "seed": 12345,
                "variant": "anisotropic",
                "stage": "stage2_free_source_forward_fit",
                "stage_label": "Stage 2",
                "method_label": "Stage 2 No flux",
                "result_path": "results.json",
                "parameter": "halo.cut_radius_kpc",
                "truth": 950.0,
                "median": 980.0,
                "q16": 920.0,
                "q84": 1040.0,
            },
        ]
    )

    joint_df = aggregate_plots.joint_sigma_rcut_recovery_rows(parameter_df)

    assert len(joint_df) == 1
    row = joint_df.iloc[0]
    assert int(row["seed"]) == 12345
    assert row["variant"] == "anisotropic"
    assert row["stage"] == "stage2_free_source_forward_fit"
    assert float(row["sigma_truth"]) == pytest.approx(245.0)
    assert float(row["sigma_median"]) == pytest.approx(240.0)
    assert float(row["sigma_q16"]) == pytest.approx(235.0)
    assert float(row["sigma_q84"]) == pytest.approx(248.0)
    assert float(row["rcut_truth"]) == pytest.approx(187.0)
    assert float(row["rcut_median"]) == pytest.approx(210.0)
    assert float(row["rcut_q16"]) == pytest.approx(180.0)
    assert float(row["rcut_q84"]) == pytest.approx(260.0)
    halo_df = aggregate_plots.joint_sigma_rcut_recovery_rows(parameter_df, component="halo")
    assert len(halo_df) == 1
    halo_row = halo_df.iloc[0]
    assert halo_row["component"] == "halo"
    assert halo_row["component_label"] == "Halo"
    assert halo_row["sigma_parameter"] == "halo.v_disp"
    assert halo_row["rcut_parameter"] == "halo.cut_radius_kpc"
    assert float(halo_row["sigma_truth"]) == pytest.approx(760.0)
    assert float(halo_row["rcut_truth"]) == pytest.approx(950.0)


def test_joint_sigma_rcut_recovery_plot_draws_truth_points_markers_and_contours(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    rows: list[dict[str, object]] = []
    for variant, offsets in {
        "anisotropic": ((-5.0, -15.0), (0.0, 25.0), (7.0, 5.0)),
        "anisotropic_maglikelihood": ((-7.0, 10.0), (4.0, -20.0), (9.0, 18.0)),
    }.items():
        for seed_index, (sigma_offset, cut_offset) in enumerate(offsets, start=1):
            rows.append(
                {
                    "stage": "stage2_free_source_forward_fit",
                    "variant": variant,
                    "seed": 12340 + seed_index,
                    "sigma_truth": 245.0,
                    "sigma_median": 245.0 + sigma_offset,
                    "rcut_truth": 187.0,
                    "rcut_median": 187.0 + cut_offset,
                }
            )
    joint_df = pd.DataFrame(rows)
    joint_df["component_label"] = "Halo"
    saved_figures: list[Figure] = []
    scatter_calls: list[dict[str, object]] = []
    original_scatter = Axes.scatter

    def record_savefig(self: Figure, *_args: object, **_kwargs: object) -> None:
        saved_figures.append(self)

    def record_scatter(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        scatter_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "kwargs": dict(kwargs),
            }
        )
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Figure, "savefig", record_savefig)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    try:
        paths = aggregate_plots.plot_joint_sigma_rcut_recovery(
            joint_df,
            tmp_path / "sigma_rcut_recovery",
            ("pdf",),
        )
        assert paths == (tmp_path / "sigma_rcut_recovery.pdf",)
        assert len(saved_figures) == 1
        ax = saved_figures[0].axes[0]
        assert ax.get_xlabel() == r"Halo $\sigma$ (fit)"
        assert ax.get_ylabel() == r"Halo $r_{\rm cut}$ (fit)"
        assert sum(isinstance(patch, Ellipse) for patch in ax.patches) == 4
        marker_by_label = {call["kwargs"].get("label"): call["kwargs"].get("marker") for call in scatter_calls}
        assert marker_by_label["No flux"] == "o"
        assert marker_by_label["Flux"] == "D"
        truth_call = next(call for call in scatter_calls if call["kwargs"].get("label") == "Truth")
        assert truth_call["kwargs"]["marker"] == "*"
        np.testing.assert_allclose(truth_call["x"], [245.0])
        np.testing.assert_allclose(truth_call["y"], [187.0])
    finally:
        plt.close("all")


def test_aggregate_seed_parameter_axis_uses_polished_variant_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    parameter_df = pd.DataFrame(
        [
            {
                "stage": "stage1_backprojected_centroid_fit",
                "variant": "anisotropic",
                "seed": 1,
                "parameter": "halo.v_disp",
                "truth": 1000.0,
                "median": 1010.0,
                "q16": 990.0,
                "q84": 1030.0,
            },
            {
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "seed": 2,
                "parameter": "halo.v_disp",
                "truth": 1000.0,
                "median": 980.0,
                "q16": 950.0,
                "q84": 1005.0,
            },
        ]
    )
    errorbar_calls: list[dict[str, object]] = []
    scatter_calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar
    original_scatter = Axes.scatter

    def record_errorbar(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        errorbar_calls.append({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float), "kwargs": dict(kwargs)})
        return original_errorbar(self, x, y, *args, **kwargs)

    def record_scatter(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        scatter_calls.append({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float), "kwargs": dict(kwargs)})
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    fig, ax = plt.subplots()
    try:
        aggregate_plots._draw_seed_parameter_axis(ax, parameter_df, "halo.v_disp")
        assert len(errorbar_calls) == 2
        assert len(scatter_calls) == 2
        for call in errorbar_calls:
            kwargs = call["kwargs"]
            np.testing.assert_allclose(call["x"], [1000.0])
            assert kwargs["fmt"] == "none"
            assert kwargs["elinewidth"] == pytest.approx(0.65)
            assert kwargs["capsize"] == pytest.approx(1.8)
            assert kwargs["capthick"] == pytest.approx(0.65)
            assert kwargs["alpha"] == pytest.approx(0.30)
            assert kwargs["zorder"] == 1
            assert kwargs["label"] == "_nolegend_"
        anisotropic_scatter = next(call for call in scatter_calls if call["kwargs"]["marker"] == "o")
        np.testing.assert_allclose(anisotropic_scatter["x"], [1000.0])
        assert anisotropic_scatter["kwargs"]["facecolors"] == "white"
        assert anisotropic_scatter["kwargs"]["edgecolors"] == "tab:blue"
        assert anisotropic_scatter["kwargs"]["alpha"] == pytest.approx(
            aggregate_plots.AGGREGATE_PARAMETER_POINT_ALPHA
        )
        mag_scatter = next(call for call in scatter_calls if call["kwargs"]["marker"] == "D")
        np.testing.assert_allclose(mag_scatter["x"], [1000.0])
        assert mag_scatter["kwargs"]["facecolors"] == "tab:orange"
        assert mag_scatter["kwargs"]["s"] == pytest.approx(21.0)
        assert mag_scatter["kwargs"]["linewidths"] == pytest.approx(0.75)
        assert mag_scatter["kwargs"]["alpha"] == pytest.approx(aggregate_plots.AGGREGATE_PARAMETER_POINT_ALPHA)
        assert mag_scatter["kwargs"]["zorder"] == 2
        assert ax.get_xlabel() == r"Halo $\sigma_v$ (truth; km s$^{-1}$)"
        assert ax.get_ylabel() == r"Halo $\sigma_v$ (fit; km s$^{-1}$)"
        assert "[" not in ax.get_xlabel()
        assert "[" not in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_aggregate_bias_metric_panel_uses_polished_variant_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    import matplotlib.pyplot as plt

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    metric_df = pd.DataFrame(
        [
            {
                "stage": "stage1_backprojected_centroid_fit",
                "variant": "anisotropic",
                "method_label": "Stage 1 No flux",
                "parameter": "halo.v_disp",
                "bias_median": 0.1,
                "bias_q16": -0.1,
                "bias_q84": 0.3,
            },
            {
                "stage": "stage2_free_source_forward_fit",
                "variant": "anisotropic_maglikelihood",
                "method_label": "Stage 2 Flux",
                "parameter": "halo.v_disp",
                "bias_median": -0.2,
                "bias_q16": -0.5,
                "bias_q84": 0.1,
            },
        ]
    )
    errorbar_calls: list[dict[str, object]] = []
    scatter_calls: list[dict[str, object]] = []
    original_errorbar = Axes.errorbar
    original_scatter = Axes.scatter

    def record_errorbar(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        errorbar_calls.append({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float), "kwargs": dict(kwargs)})
        return original_errorbar(self, x, y, *args, **kwargs)

    def record_scatter(self: Axes, x: object, y: object, *args: object, **kwargs: object) -> object:
        scatter_calls.append({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float), "kwargs": dict(kwargs)})
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)
    monkeypatch.setattr(Axes, "scatter", record_scatter)
    fig, ax = plt.subplots()
    try:
        aggregate_plots._plot_bias_metric_panel(
            ax,
            metric_df,
            ["halo.v_disp"],
            ["Stage 1 No flux", "Stage 2 Flux"],
            ylabel="bias",
            title="Bias",
            metric_column="bias_median",
            metric_q16_column="bias_q16",
            metric_q84_column="bias_q84",
        )
        assert len(errorbar_calls) == 2
        assert len(scatter_calls) == 2
        assert {call["kwargs"]["fmt"] for call in errorbar_calls} == {"none"}
        assert {call["kwargs"]["marker"] for call in scatter_calls} == {"D", "o"}
        stage1_call = next(call for call in scatter_calls if call["kwargs"]["marker"] == "o")
        assert stage1_call["kwargs"]["facecolors"] == "white"
        stage2_call = next(call for call in scatter_calls if call["kwargs"]["marker"] == "D")
        assert stage2_call["kwargs"]["facecolors"] == "tab:orange"
    finally:
        plt.close(fig)


def test_aggregate_parameter_recovery_collapses_seed_runs(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed, truth, median in ((12345, 1.0, 1.10), (12346, 3.0, 2.90), (12347, 5.0, 5.20)):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                _write_aggregate_result(
                    campaign,
                    seed=seed,
                    variant=variant,
                    stage=stage,
                    records=[
                        {
                            "parameter": "halo.x_centre",
                            "truth": truth,
                            "best_fit": median,
                            "q16": median - 0.2,
                            "median": median,
                            "q84": median + 0.2,
                            "bias": median - truth,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        }
                    ],
                )

    parameter_df, _inventory_df = aggregate_plots.collect_parameter_recovery(campaign)
    aggregate_df = aggregate_plots.aggregate_parameter_recovery(parameter_df)

    assert len(aggregate_df) == 4
    row = aggregate_df[
        (aggregate_df["stage"] == "stage1_backprojected_centroid_fit")
        & (aggregate_df["variant"] == "anisotropic")
        & (aggregate_df["parameter"] == "halo.x_centre")
    ].iloc[0]
    assert int(row["n_seed_runs"]) == 3
    assert float(row["truth"]) == pytest.approx(3.0)
    assert float(row["truth_q16"]) == pytest.approx(np.quantile([1.0, 3.0, 5.0], 0.16))
    assert float(row["truth_q84"]) == pytest.approx(np.quantile([1.0, 3.0, 5.0], 0.84))
    assert float(row["median"]) == pytest.approx(2.9)
    assert float(row["q16"]) == pytest.approx(np.quantile([1.10, 2.90, 5.20], 0.16))
    assert float(row["q84"]) == pytest.approx(np.quantile([1.10, 2.90, 5.20], 0.84))
    assert float(row["coverage_68_fraction"]) == pytest.approx(1.0)


def test_aggregate_parameter_recovery_bias_quantiles_use_per_seed_biases(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    seed_rows = (
        (12345, 700.0, 720.0, 0.0, 0.020),
        (12346, 760.0, 750.0, 0.0, 0.045),
        (12347, 820.0, 850.0, 0.0, 0.080),
    )
    for seed, v_truth, v_median, gamma_truth, gamma_median in seed_rows:
        _write_aggregate_result(
            campaign,
            seed=seed,
            variant="anisotropic",
            stage="stage1_backprojected_centroid_fit",
            records=[
                {
                    "parameter": "halo.v_disp",
                    "truth": v_truth,
                    "best_fit": v_median,
                    "q16": v_median - 10.0,
                    "median": v_median,
                    "q84": v_median + 10.0,
                    "bias": v_median - v_truth,
                    "truth_percentile": 0.5,
                    "covered_68": True,
                },
                {
                    "parameter": "potfile.gamma_ml",
                    "truth": gamma_truth,
                    "best_fit": gamma_median,
                    "q16": gamma_median - 0.01,
                    "median": gamma_median,
                    "q84": gamma_median + 0.01,
                    "bias": gamma_median - gamma_truth,
                    "truth_percentile": 0.5,
                    "covered_68": False,
                },
            ],
        )

    parameter_df, _inventory_df = aggregate_plots.collect_parameter_recovery(
        campaign,
        stages=("stage1_backprojected_centroid_fit",),
        variants=("anisotropic",),
    )
    aggregate_df = aggregate_plots.aggregate_parameter_recovery(parameter_df)

    v_row = aggregate_df[aggregate_df["parameter"] == "halo.v_disp"].iloc[0]
    v_bias = np.asarray([20.0, -10.0, 30.0], dtype=float)
    v_fractional = np.asarray([20.0 / 700.0, -10.0 / 760.0, 30.0 / 820.0], dtype=float)
    assert float(v_row["bias_q16"]) == pytest.approx(np.quantile(v_bias, 0.16))
    assert float(v_row["bias_median"]) == pytest.approx(np.quantile(v_bias, 0.5))
    assert float(v_row["bias_q84"]) == pytest.approx(np.quantile(v_bias, 0.84))
    assert float(v_row["fractional_bias_q16"]) == pytest.approx(np.quantile(v_fractional, 0.16))
    assert float(v_row["fractional_bias_median"]) == pytest.approx(np.quantile(v_fractional, 0.5))
    assert float(v_row["fractional_bias_q84"]) == pytest.approx(np.quantile(v_fractional, 0.84))

    gamma_row = aggregate_df[aggregate_df["parameter"] == "potfile.gamma_ml"].iloc[0]
    gamma_bias = np.asarray([0.020, 0.045, 0.080], dtype=float)
    assert float(gamma_row["bias_q16"]) == pytest.approx(np.quantile(gamma_bias, 0.16))
    assert float(gamma_row["bias_median"]) == pytest.approx(np.quantile(gamma_bias, 0.5))
    assert float(gamma_row["bias_q84"]) == pytest.approx(np.quantile(gamma_bias, 0.84))
    assert pd.isna(gamma_row["fractional_bias_q16"])
    assert pd.isna(gamma_row["fractional_bias_median"])
    assert pd.isna(gamma_row["fractional_bias_q84"])


def test_paper_mass_proxy_recovery_rows_derive_log10_mass_panels() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots
    from lenscluster.utils import log10_dpie_mass_msun

    base = {
        "campaign": "mock",
        "run_name": "single_bcg",
        "seed": 12345,
        "variant": "anisotropic",
        "stage": "stage2_free_source_forward_fit",
        "stage_label": "Stage 2",
        "method_label": "Stage 2 No flux",
        "result_path": "results.json",
        "truth_percentile": 0.5,
        "covered_68": True,
    }
    rows = []
    for parameter, truth, q16, median, q84 in (
        ("halo.v_disp", 760.0, 720.0, 770.0, 805.0),
        ("halo.cut_radius_kpc", 950.0, 920.0, 970.0, 1030.0),
        ("bcg.v_disp", 285.0, 270.0, 280.0, 295.0),
        ("bcg.cut_radius_kpc", 75.0, 65.0, 80.0, 95.0),
        ("potfile.sigma", 245.0, 236.0, 242.0, 250.0),
        ("potfile.cutkpc", 187.0, 170.0, 195.0, 225.0),
    ):
        rows.append(
            {
                **base,
                "parameter": parameter,
                "truth": truth,
                "best_fit": median,
                "q16": q16,
                "median": median,
                "q84": q84,
                "bias": median - truth,
            }
        )
    parameter_df = pd.DataFrame(rows)

    mass_df = aggregate_plots.paper_mass_proxy_recovery_rows(parameter_df)

    assert set(mass_df["parameter"]) == {
        "halo.log10_mass_msun",
        "bcg.log10_mass_msun",
        "potfile.log10_mass_msun",
    }
    halo_row = mass_df[mass_df["parameter"] == "halo.log10_mass_msun"].iloc[0]
    assert float(halo_row["truth"]) == pytest.approx(float(log10_dpie_mass_msun(760.0, 950.0)))
    assert float(halo_row["median"]) == pytest.approx(float(log10_dpie_mass_msun(770.0, 970.0)))
    assert float(halo_row["q16"]) == pytest.approx(float(log10_dpie_mass_msun(720.0, 920.0)))
    assert float(halo_row["q84"]) == pytest.approx(float(log10_dpie_mass_msun(805.0, 1030.0)))
    assert float(halo_row["bias"]) == pytest.approx(float(halo_row["median"] - halo_row["truth"]))
    assert bool(halo_row["covered_68"]) is True

    missing_cut_df = parameter_df[parameter_df["parameter"] != "potfile.cutkpc"].copy()
    missing_mass_df = aggregate_plots.paper_mass_proxy_recovery_rows(missing_cut_df)
    assert "potfile.log10_mass_msun" not in set(missing_mass_df["parameter"])


def test_paper_parameters_place_log10_mass_panels_near_sigma_and_rcut() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    paper_parameters = list(aggregate_plots.PAPER_PARAMETERS)
    assert paper_parameters.index("halo.log10_mass_msun") == paper_parameters.index("halo.cut_radius_kpc") + 1
    assert paper_parameters.index("bcg.log10_mass_msun") == paper_parameters.index("bcg.cut_radius_kpc") + 1
    assert paper_parameters.index("potfile.log10_mass_msun") == paper_parameters.index("potfile.cutkpc") + 1
    assert aggregate_plots._format_parameter_label("halo.log10_mass_msun") == r"Halo $\log_{10} M$"
    assert aggregate_plots._format_parameter_label("bcg.log10_mass_msun") == r"BCG $\log_{10} M$"
    assert aggregate_plots._format_parameter_label("potfile.log10_mass_msun") == r"Potfile $\log_{10} M$"


def test_aggregate_bias_parameter_classification_keeps_gamma_ml_absolute() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    assert aggregate_plots.is_fractional_bias_parameter("halo.v_disp")
    assert aggregate_plots.is_fractional_bias_parameter("bcg.core_radius_kpc")
    assert aggregate_plots.is_fractional_bias_parameter("potfile.sigma")
    assert aggregate_plots.is_absolute_bias_parameter("potfile.gamma_ml")
    assert aggregate_plots.is_absolute_bias_parameter("halo.x_centre")
    assert aggregate_plots.is_absolute_bias_parameter("source.1.beta_x")
    assert not aggregate_plots.is_fractional_bias_parameter("potfile.gamma_ml")


def test_aggregate_parameter_recovery_missing_outputs_warn_or_fail(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    _write_aggregate_result(
        campaign,
        seed=12345,
        variant="anisotropic",
        stage="stage1_backprojected_centroid_fit",
    )

    with pytest.warns(RuntimeWarning, match="Missing aggregate input"):
        parameter_df, inventory_df = aggregate_plots.collect_parameter_recovery(campaign)

    assert len(parameter_df) == 2
    assert int(inventory_df["loaded"].sum()) == 1
    with pytest.raises(FileNotFoundError, match="Missing aggregate input"):
        aggregate_plots.collect_parameter_recovery(campaign, strict=True)


def test_write_aggregate_plots_creates_tables_and_figures(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed_index, seed in enumerate((12345, 12346, 12347)):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                variant_offset = -2.0 if variant == "anisotropic" else 3.0
                stage_offset = 0.0 if stage == "stage1_backprojected_centroid_fit" else 4.0
                curve_offset = (0.0, 4.0, -3.0)[seed_index]
                _write_aggregate_result(
                    campaign,
                    seed=seed,
                    variant=variant,
                    stage=stage,
                    records=[
                        {
                            "parameter": "halo.x_centre",
                            "truth": 1.0,
                            "best_fit": 1.05,
                            "q16": 0.9,
                            "median": 1.04,
                            "q84": 1.2,
                            "bias": 0.04,
                            "truth_percentile": 0.4,
                            "covered_68": True,
                        },
                        {
                            "parameter": "halo.v_disp",
                            "truth": 760.0,
                            "best_fit": 770.0 + seed_index + variant_offset,
                            "q16": 720.0,
                            "median": 770.0 + seed_index + variant_offset,
                            "q84": 805.0,
                            "bias": 10.0 + seed_index + variant_offset,
                            "truth_percentile": 0.45,
                            "covered_68": True,
                        },
                        {
                            "parameter": "halo.cut_radius_kpc",
                            "truth": 950.0,
                            "best_fit": 970.0 + 4.0 * seed_index + variant_offset + stage_offset + curve_offset,
                            "q16": 920.0 + 4.0 * seed_index + variant_offset + stage_offset + curve_offset,
                            "median": 970.0 + 4.0 * seed_index + variant_offset + stage_offset + curve_offset,
                            "q84": 1030.0 + 4.0 * seed_index + variant_offset + stage_offset + curve_offset,
                            "bias": 20.0 + 4.0 * seed_index + variant_offset + stage_offset + curve_offset,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        },
                        {
                            "parameter": "bcg.v_disp",
                            "truth": 285.0,
                            "best_fit": 280.0 + seed_index - variant_offset + stage_offset,
                            "q16": 270.0 + seed_index - variant_offset + stage_offset,
                            "median": 280.0 + seed_index - variant_offset + stage_offset,
                            "q84": 295.0 + seed_index - variant_offset + stage_offset,
                            "bias": -5.0 + seed_index - variant_offset + stage_offset,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        },
                        {
                            "parameter": "bcg.cut_radius_kpc",
                            "truth": 75.0,
                            "best_fit": 80.0 + 2.0 * seed_index + variant_offset + 0.5 * stage_offset - curve_offset,
                            "q16": 65.0 + 2.0 * seed_index + variant_offset + 0.5 * stage_offset - curve_offset,
                            "median": 80.0 + 2.0 * seed_index + variant_offset + 0.5 * stage_offset - curve_offset,
                            "q84": 95.0 + 2.0 * seed_index + variant_offset + 0.5 * stage_offset - curve_offset,
                            "bias": 5.0 + 2.0 * seed_index + variant_offset + 0.5 * stage_offset - curve_offset,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        },
                        {
                            "parameter": "potfile.sigma",
                            "truth": 245.0,
                            "best_fit": 242.0 + seed_index + variant_offset + stage_offset,
                            "q16": 236.0 + seed_index + variant_offset + stage_offset,
                            "median": 242.0 + seed_index + variant_offset + stage_offset,
                            "q84": 250.0 + seed_index + variant_offset + stage_offset,
                            "bias": -3.0 + seed_index + variant_offset + stage_offset,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        },
                        {
                            "parameter": "potfile.cutkpc",
                            "truth": 187.0,
                            "best_fit": 195.0 + 3.0 * seed_index - variant_offset + 0.5 * stage_offset + curve_offset,
                            "q16": 170.0 + 3.0 * seed_index - variant_offset + 0.5 * stage_offset + curve_offset,
                            "median": 195.0 + 3.0 * seed_index - variant_offset + 0.5 * stage_offset + curve_offset,
                            "q84": 225.0 + 3.0 * seed_index - variant_offset + 0.5 * stage_offset + curve_offset,
                            "bias": 8.0 + 3.0 * seed_index - variant_offset + 0.5 * stage_offset + curve_offset,
                            "truth_percentile": 0.5,
                            "covered_68": True,
                        },
                    ],
                )
        for variant in aggregate_plots.DEFAULT_VARIANTS:
            _write_aggregate_combined_recovery(
                campaign,
                seed=seed,
                variant=variant,
                truth_masses=[1.0e10, 1.0e11, 1.0e12],
                recovered_log_masses=[10.1, 11.0, 11.9],
                stages=aggregate_plots.DEFAULT_STAGES,
            )

    outputs = aggregate_plots.write_aggregate_plots(campaign, parameter_set="both", formats=("pdf",))

    assert outputs.parameter_table.exists()
    assert outputs.aggregate_parameter_table.exists()
    assert outputs.inventory_table.exists()
    assert outputs.magnification_table is not None
    assert outputs.magnification_table.exists()
    assert outputs.image_residual_table is not None
    assert outputs.image_residual_table.exists()
    assert outputs.image_residual_draws_table is not None
    assert outputs.image_residual_draws_table.exists()
    assert outputs.subhalo_shmf_seed_table is not None
    assert outputs.subhalo_shmf_seed_table.exists()
    assert outputs.subhalo_shmf_summary_table is not None
    assert outputs.subhalo_shmf_summary_table.exists()
    assert {path.name for path in outputs.plot_paths} == {
        "mock_parameter_recovery_stage1_paper.pdf",
        "mock_parameter_recovery_paper.pdf",
        "mock_parameter_recovery_all.pdf",
        "mock_parameter_bias_summary.pdf",
        "mock_sigma_rcut_recovery_stage1.pdf",
        "mock_sigma_rcut_recovery.pdf",
        "mock_halo_sigma_rcut_recovery_stage1.pdf",
        "mock_halo_sigma_rcut_recovery.pdf",
        "mock_bcg_sigma_rcut_recovery_stage1.pdf",
        "mock_bcg_sigma_rcut_recovery.pdf",
        "mock_magnification_recovery_stage1.pdf",
        "mock_magnification_recovery.pdf",
        "mock_image_residual_recovery_stage1.pdf",
        "mock_image_residual_recovery.pdf",
        "mock_image_residual_distribution_stage1.pdf",
        "mock_image_residual_distribution.pdf",
        "mock_subhalo_recovery_stage1_shmf.pdf",
        "mock_subhalo_recovery_shmf.pdf",
    }
    for path in outputs.plot_paths:
        assert path.exists()
        assert path.stat().st_size > 0
    table = pd.read_csv(outputs.parameter_table)
    aggregate_table = pd.read_csv(outputs.aggregate_parameter_table)
    assert len(table) == 84
    assert len(aggregate_table) == 28
    magnification_table = pd.read_csv(outputs.magnification_table)
    assert len(magnification_table) == 24
    assert {
        "stage",
        "seed",
        "variant",
        "magnification_true",
        "magnification_q16",
        "magnification_q50",
        "magnification_q84",
    } <= set(magnification_table.columns)
    image_residual_table = pd.read_csv(outputs.image_residual_table)
    assert len(image_residual_table) == 24
    assert {
        "stage",
        "seed",
        "variant",
        "point_image_rms_arcsec",
        "arc_aware_image_rms_arcsec",
    } <= set(image_residual_table.columns)
    image_residual_draws_table = pd.read_csv(outputs.image_residual_draws_table)
    assert len(image_residual_draws_table) == 48
    assert {
        "stage",
        "seed",
        "variant",
        "draw_index",
        "arc_aware_image_residual_arcsec",
        "arc_aware_recovered",
    } <= set(image_residual_draws_table.columns)
    assert aggregate_table.groupby("parameter").size().to_dict() == {
        "bcg.cut_radius_kpc": 4,
        "bcg.v_disp": 4,
        "halo.cut_radius_kpc": 4,
        "halo.v_disp": 4,
        "halo.x_centre": 4,
        "potfile.cutkpc": 4,
        "potfile.sigma": 4,
    }
    assert set(table["method_label"]) == {
        "Stage 1 No flux",
        "Stage 1 Flux",
        "Stage 2 No flux",
        "Stage 2 Flux",
    }


def test_write_aggregate_plots_creates_stage1_paper_plot_when_stage1_requested(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed_index, seed in enumerate((12345, 12346, 12347)):
        for variant in aggregate_plots.DEFAULT_VARIANTS:
            variant_offset = -2.0 if variant == "anisotropic" else 3.0
            curve_offset = (0.0, 4.0, -3.0)[seed_index]
            _write_aggregate_result(
                campaign,
                seed=seed,
                variant=variant,
                stage="stage1_backprojected_centroid_fit",
                records=[
                    {
                        "parameter": "halo.x_centre",
                        "truth": 0.0,
                        "best_fit": 0.1 if variant == "anisotropic" else -0.1,
                        "q16": -0.2,
                        "median": 0.1 if variant == "anisotropic" else -0.1,
                        "q84": 0.2,
                        "bias": 0.1 if variant == "anisotropic" else -0.1,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                    {
                        "parameter": "potfile.sigma",
                        "truth": 245.0,
                        "best_fit": 242.0 + seed_index + variant_offset,
                        "q16": 236.0 + seed_index + variant_offset,
                        "median": 242.0 + seed_index + variant_offset,
                        "q84": 250.0 + seed_index + variant_offset,
                        "bias": -3.0 + seed_index + variant_offset,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                    {
                        "parameter": "halo.v_disp",
                        "truth": 760.0,
                        "best_fit": 770.0 + seed_index + variant_offset,
                        "q16": 720.0 + seed_index + variant_offset,
                        "median": 770.0 + seed_index + variant_offset,
                        "q84": 805.0 + seed_index + variant_offset,
                        "bias": 10.0 + seed_index + variant_offset,
                        "truth_percentile": 0.45,
                        "covered_68": True,
                    },
                    {
                        "parameter": "halo.cut_radius_kpc",
                        "truth": 950.0,
                        "best_fit": 970.0 + 4.0 * seed_index + variant_offset + curve_offset,
                        "q16": 920.0 + 4.0 * seed_index + variant_offset + curve_offset,
                        "median": 970.0 + 4.0 * seed_index + variant_offset + curve_offset,
                        "q84": 1030.0 + 4.0 * seed_index + variant_offset + curve_offset,
                        "bias": 20.0 + 4.0 * seed_index + variant_offset + curve_offset,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                    {
                        "parameter": "bcg.v_disp",
                        "truth": 285.0,
                        "best_fit": 280.0 + seed_index - variant_offset,
                        "q16": 270.0 + seed_index - variant_offset,
                        "median": 280.0 + seed_index - variant_offset,
                        "q84": 295.0 + seed_index - variant_offset,
                        "bias": -5.0 + seed_index - variant_offset,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                    {
                        "parameter": "bcg.cut_radius_kpc",
                        "truth": 75.0,
                        "best_fit": 80.0 + 2.0 * seed_index + variant_offset - curve_offset,
                        "q16": 65.0 + 2.0 * seed_index + variant_offset - curve_offset,
                        "median": 80.0 + 2.0 * seed_index + variant_offset - curve_offset,
                        "q84": 95.0 + 2.0 * seed_index + variant_offset - curve_offset,
                        "bias": 5.0 + 2.0 * seed_index + variant_offset - curve_offset,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                    {
                        "parameter": "potfile.cutkpc",
                        "truth": 187.0,
                        "best_fit": 195.0 + 3.0 * seed_index - variant_offset + curve_offset,
                        "q16": 170.0 + 3.0 * seed_index - variant_offset + curve_offset,
                        "median": 195.0 + 3.0 * seed_index - variant_offset + curve_offset,
                        "q84": 225.0 + 3.0 * seed_index - variant_offset + curve_offset,
                        "bias": 8.0 + 3.0 * seed_index - variant_offset + curve_offset,
                        "truth_percentile": 0.5,
                        "covered_68": True,
                    },
                ],
            )

    outputs = aggregate_plots.write_aggregate_plots(
        campaign,
        stages=("stage1_backprojected_centroid_fit",),
        parameter_set="paper",
        formats=("pdf",),
    )

    assert {path.name for path in outputs.plot_paths if path.name.startswith("mock_parameter_recovery")} == {
        "mock_parameter_recovery_paper.pdf"
    }
    paper_plot = outputs.output_dir / "mock_parameter_recovery_paper.pdf"
    assert paper_plot.exists()
    assert paper_plot.stat().st_size > 0
    assert not (outputs.output_dir / "mock_parameter_recovery_stage1_paper.pdf").exists()
    magnification_plot = outputs.output_dir / "mock_magnification_recovery.pdf"
    assert magnification_plot.exists()
    assert magnification_plot.stat().st_size > 0
    assert not (outputs.output_dir / "mock_magnification_recovery_stage1.pdf").exists()
    sigma_rcut_plot = outputs.output_dir / "mock_sigma_rcut_recovery.pdf"
    assert sigma_rcut_plot.exists()
    assert sigma_rcut_plot.stat().st_size > 0
    assert not (outputs.output_dir / "mock_sigma_rcut_recovery_stage1.pdf").exists()
    halo_sigma_rcut_plot = outputs.output_dir / "mock_halo_sigma_rcut_recovery.pdf"
    assert halo_sigma_rcut_plot.exists()
    assert halo_sigma_rcut_plot.stat().st_size > 0
    assert not (outputs.output_dir / "mock_halo_sigma_rcut_recovery_stage1.pdf").exists()
    bcg_sigma_rcut_plot = outputs.output_dir / "mock_bcg_sigma_rcut_recovery.pdf"
    assert bcg_sigma_rcut_plot.exists()
    assert bcg_sigma_rcut_plot.stat().st_size > 0
    assert not (outputs.output_dir / "mock_bcg_sigma_rcut_recovery_stage1.pdf").exists()
    table = pd.read_csv(outputs.parameter_table)
    assert set(table["stage"]) == {"stage1_backprojected_centroid_fit"}


def test_subhalo_shmf_tables_aggregate_seed_quantiles(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    bins = np.asarray([10.0, 11.0, 12.0], dtype=float)
    _write_aggregate_combined_recovery(
        campaign,
        seed=12345,
        variant="anisotropic",
        truth_masses=[1.0e10],
        recovered_log_masses=[10.2],
    )
    _write_aggregate_combined_recovery(
        campaign,
        seed=12346,
        variant="anisotropic",
        truth_masses=[1.0e10],
        recovered_log_masses=[10.3, 11.2],
    )

    shmf_inputs, selection = aggregate_plots.collect_subhalo_shmf_inputs(
        campaign,
        variants=("anisotropic",),
    )
    seed_df, summary_df = aggregate_plots.build_subhalo_shmf_tables(shmf_inputs, log_mass_bins=bins)

    assert selection["mass_ref"] == pytest.approx(1.0e12)
    assert selection["dpie_mass_anchor_msun"] == pytest.approx(8.2e12)
    assert len(seed_df) == 4
    assert set(seed_df["variant"]) == {"anisotropic"}
    recovered_bin0 = summary_df[
        (summary_df["series"] == "recovered")
        & (summary_df["variant"] == "anisotropic")
        & (summary_df["bin_index"] == 0)
    ].iloc[0]
    assert float(recovered_bin0["value_median"]) == pytest.approx(np.median([1.0, 1.0]))
    ratio_bin1 = summary_df[
        (summary_df["series"] == "ratio")
        & (summary_df["variant"] == "anisotropic")
        & (summary_df["bin_index"] == 1)
    ]
    assert ratio_bin1.empty


def test_subhalo_shmf_outputs_create_plot_and_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes

    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    def fail_yscale(self: Axes, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("aggregate subhalo recovery SHMF should keep a linear y axis")

    monkeypatch.setattr(Axes, "set_yscale", fail_yscale)

    campaign = tmp_path / "jul2_2215_mock_validation"
    output_dir = campaign / "aggregate_plots"
    output_dir.mkdir(parents=True)
    for seed in (12345, 12346):
        for variant in aggregate_plots.DEFAULT_VARIANTS:
            _write_aggregate_combined_recovery(
                campaign,
                seed=seed,
                variant=variant,
                truth_masses=[1.0e10, 1.0e11, 1.0e12],
                recovered_log_masses=[10.1, 11.0, 11.9],
                stages=aggregate_plots.DEFAULT_STAGES,
            )

    seed_table, summary_table, plot_paths = aggregate_plots.write_subhalo_shmf_outputs(
        campaign,
        output_dir,
        formats=("pdf",),
    )

    assert seed_table is not None and seed_table.exists()
    assert summary_table is not None and summary_table.exists()
    assert [path.name for path in plot_paths] == ["mock_subhalo_recovery_stage1_shmf.pdf", "mock_subhalo_recovery_shmf.pdf"]
    for path in plot_paths:
        assert path.exists()
        assert path.stat().st_size > 0
    seed_df = pd.read_csv(seed_table)
    summary_df = pd.read_csv(summary_table)
    assert set(seed_df["stage"]) == set(aggregate_plots.DEFAULT_STAGES)
    assert set(summary_df["stage"]) == set(aggregate_plots.DEFAULT_STAGES)


def test_stage2_seed_recovery_rows_keep_each_seed_and_variant(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    campaign = tmp_path / "jul2_2215_mock_validation"
    for seed in (12345, 12346):
        for stage in aggregate_plots.DEFAULT_STAGES:
            for variant in aggregate_plots.DEFAULT_VARIANTS:
                _write_aggregate_result(campaign, seed=seed, variant=variant, stage=stage)

    parameter_df, _inventory_df = aggregate_plots.collect_parameter_recovery(campaign)
    seed_df = aggregate_plots.stage2_seed_recovery_rows(parameter_df)

    assert set(seed_df["stage"]) == {"stage2_free_source_forward_fit"}
    assert set(seed_df["variant"]) == {"anisotropic", "anisotropic_maglikelihood"}
    assert set(seed_df["seed"]) == {12345, 12346}
    assert len(seed_df) == 8


def test_aggregate_plot_parameter_display_labels_are_paper_friendly() -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    assert aggregate_plots._format_parameter_label("bcg.core_radius_kpc") == r"BCG $r_\mathrm{core}$"
    assert aggregate_plots._format_parameter_label("halo.cut_radius_kpc") == r"Halo $r_\mathrm{cut}$"
    assert aggregate_plots._format_parameter_label("source.12.beta_x") == r"Source $\beta_x$"
    assert aggregate_plots._format_parameter_label("source.12.beta_y") == r"Source $\beta_y$"
    assert aggregate_plots._format_parameter_axis_label("bcg.core_radius_kpc", "truth") == (
        r"BCG $r_\mathrm{core}$ (truth; kpc)"
    )
    assert aggregate_plots._format_parameter_axis_label("halo.x_centre", "fit") == "Halo x (fit; arcsec)"
    assert aggregate_plots._format_parameter_axis_label("potfile.gamma_ml", "fit") == (
        r"Potfile $\gamma_\mathrm{ML}$ (fit)"
    )
    assert aggregate_plots._format_parameter_axis_label("potfile.log10_mass_msun", "truth") == (
        r"Potfile $\log_{10} M$ (truth; $M_\odot$)"
    )


def test_aggregate_bias_summary_splits_fractional_and_absolute_parameters(tmp_path: Path) -> None:
    import lenscluster.mock_validation.aggregate_plots as aggregate_plots

    df = pd.DataFrame(
        [
            {
                "campaign": "mock",
                "run_name": "run",
                "parameter": "halo.v_disp",
                "variant": "anisotropic",
                "stage": "stage1_backprojected_centroid_fit",
                "stage_label": "Stage 1",
                "method_label": "Stage 1 No flux",
                "truth": 760.0,
                "median": 780.0,
                "q16": 760.0,
                "q84": 800.0,
                "bias": 20.0,
            },
            {
                "campaign": "mock",
                "run_name": "run",
                "parameter": "potfile.gamma_ml",
                "variant": "anisotropic",
                "stage": "stage1_backprojected_centroid_fit",
                "stage_label": "Stage 1",
                "method_label": "Stage 1 No flux",
                "truth": 0.0,
                "median": 0.045,
                "q16": 0.02,
                "q84": 0.07,
                "bias": 0.045,
            },
        ]
    )

    finite = aggregate_plots.finite_recovery_rows(df)
    parameters = finite["parameter"].astype(str)
    fractional_mask = (
        parameters.map(aggregate_plots.is_fractional_bias_parameter).to_numpy(dtype=bool)
        & np.isfinite(finite["truth"].to_numpy(dtype=float))
        & (np.abs(finite["truth"].to_numpy(dtype=float)) > 1.0e-6)
    )
    absolute_mask = parameters.map(aggregate_plots.is_absolute_bias_parameter).to_numpy(dtype=bool) | ~fractional_mask

    assert set(finite.loc[fractional_mask, "parameter"]) == {"halo.v_disp"}
    assert "potfile.gamma_ml" in set(finite.loc[absolute_mask, "parameter"])

    paths = aggregate_plots.plot_parameter_bias_summary(df, tmp_path / "parameter_bias_summary", ("pdf",))

    assert [path.name for path in paths] == ["parameter_bias_summary.pdf"]
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0
