import argparse
import inspect
import json
import math
import warnings
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import lenscluster.plotting as plotting
import lenscluster.utils as lc_utils
from lenscluster.plotting import plot_path
from lenscluster.model import PosteriorResults, ParameterSpec


def test_lenscluster_plot_style_is_loaded() -> None:
    assert plotting.plt.rcParams["font.size"] == pytest.approx(16.0)
    assert plotting.plt.rcParams["axes.labelsize"] == pytest.approx(16.0)
    assert plotting.plt.rcParams["legend.fontsize"] == pytest.approx(14.0)
    assert plotting.plt.rcParams["figure.figsize"] == pytest.approx([6.0, 6.0])
    assert plotting.plt.rcParams["axes.linewidth"] == pytest.approx(1.8)
    assert plotting.plt.rcParams["axes.grid"] is False
    assert plotting.plt.rcParams["xtick.direction"] == "in"
    assert plotting.plt.rcParams["xtick.top"] is True
    assert plotting.plt.rcParams["ytick.right"] is True
    assert plotting.plt.rcParams["savefig.dpi"] == pytest.approx(300.0)
    assert plotting.plt.rcParams["savefig.bbox"] == "tight"


def _potfile_parameter_spec(
    potfile_id: str,
    field: str,
    *,
    component_family: str = "scaling",
) -> ParameterSpec:
    return ParameterSpec(
        name=f"{potfile_id}.{field}",
        sample_name=f"{potfile_id}_{field}",
        potential_id=potfile_id,
        profile_type=81,
        field=field,
        prior_kind="normal",
        lower=-1.0e6,
        upper=1.0e6,
        step=1.0,
        mean=0.0,
        std=1.0,
        component_family=component_family,
    )


def test_plot_path_creates_directory(tmp_path: Path) -> None:
    output = plot_path(tmp_path / "plots", "summary.png")

    assert output == tmp_path / "plots" / "summary.pdf"
    assert output.parent.is_dir()


def test_plot_path_prefixes_cluster_stage_plot_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "ares_S1local-jacobian_S2none" / "stage2_free_source_forward_fit"

    output = plot_path(run_dir, "truth_recovery_mu_model_truth_fractional_residual.pdf")

    assert output == run_dir / "ares_truth_recovery_mu_model_truth_fractional_residual.pdf"


def test_plot_path_prefixes_cluster_output_label_plot_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "ares_PD0.1_TOPK_5" / "stage2_free_source_forward_fit"

    output = plot_path(run_dir, "image_recovery.pdf")

    assert output == run_dir / "ares_image_recovery.pdf"


def test_plot_path_does_not_double_prefix_cluster_plot_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "ares_S1local-jacobian_S2none" / "stage2_free_source_forward_fit"

    output = plot_path(run_dir, "ares_truth_recovery_mu_model_truth_fractional_residual.pdf")

    assert output == run_dir / "ares_truth_recovery_mu_model_truth_fractional_residual.pdf"


def test_plot_path_preserves_generic_tmp_plot_names(tmp_path: Path) -> None:
    output = plot_path(tmp_path / "plots", "truth_recovery_mu_model_truth_fractional_residual.pdf")

    assert output == tmp_path / "plots" / "truth_recovery_mu_model_truth_fractional_residual.pdf"


def test_finish_figure_saves_and_closes_without_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    show_calls: list[bool] = []
    monkeypatch.setattr(plotting, "_SHOW_PLOTS", False)
    monkeypatch.setattr(plotting.plt, "show", lambda: show_calls.append(True))

    fig, ax = plotting.plt.subplots()
    fig_number = fig.number
    ax.plot([0.0, 1.0], [0.0, 1.0])
    output = tmp_path / "figure.pdf"

    plotting._finish_figure(fig, output)

    assert output.is_file()
    assert show_calls == []
    assert not plotting.plt.fignum_exists(fig_number)


def test_finish_figure_shows_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    show_calls: list[bool] = []
    monkeypatch.setattr(plotting, "_SHOW_PLOTS", True)
    monkeypatch.setattr(plotting.plt, "show", lambda: show_calls.append(True))

    fig, ax = plotting.plt.subplots()
    fig_number = fig.number
    ax.plot([0.0, 1.0], [1.0, 0.0])
    output = tmp_path / "figure.pdf"

    plotting._finish_figure(fig, output)

    assert output.is_file()
    assert show_calls == [True]
    assert not plotting.plt.fignum_exists(fig_number)


def test_finish_pdf_page_shows_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    show_calls: list[bool] = []
    monkeypatch.setattr(plotting, "_SHOW_PLOTS", True)
    monkeypatch.setattr(plotting.plt, "show", lambda: show_calls.append(True))

    fig, ax = plotting.plt.subplots()
    fig_number = fig.number
    ax.text(0.5, 0.5, "page")
    output = tmp_path / "pages.pdf"

    with plotting.PdfPages(output) as pdf:
        plotting._finish_pdf_page(pdf, fig, bbox_inches="tight")

    assert output.is_file()
    assert show_calls == [True]
    assert not plotting.plt.fignum_exists(fig_number)


def test_active_scaling_summary_plot_writes_pdf_only(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "potfile_id": ["members", "members"],
            "rank": [1, 2],
            "component_index": [1, 2],
            "catalog_id": ["a", "b"],
            "catalog_mag": [18.5, 21.0],
            "x_centre": [0.0, 1.0],
            "y_centre": [0.0, -1.0],
            "p_active_median": [0.85, 0.15],
            "p_active_p16": [0.75, 0.05],
            "p_active_p84": [0.93, 0.30],
            "frozen_active": [True, False],
        }
    )

    plotting._plot_active_scaling_summary(tmp_path, df, freeze_threshold=0.5)

    assert plotting._plot_path(tmp_path, "active_scaling_summary.pdf").is_file()
    assert not (tmp_path / "active_scaling_summary.png").exists()


def test_scaling_results_summary_table_reports_bergamini_reference_values_and_mass() -> None:
    specs = [
        _potfile_parameter_spec("members", "sigma"),
        _potfile_parameter_spec("members", "cutkpc"),
        _potfile_parameter_spec("members", "corekpc"),
        _potfile_parameter_spec("members", "alpha_sigma"),
        _potfile_parameter_spec("members", "gamma_ml"),
        _potfile_parameter_spec("members", "sigma_log_scatter", component_family="scaling_scatter"),
        _potfile_parameter_spec(
            "members",
            "independent_free_log_sigma_tau",
            component_family="independent_scaling",
        ),
        _potfile_parameter_spec(
            "members",
            "independent_free_log_mass_tau",
            component_family="independent_scaling",
        ),
    ]
    samples = np.asarray(
        [
            [100.0, 30.0, 1.0, 0.20, -0.10, 0.10, 0.08, 0.16],
            [120.0, 40.0, 2.0, 0.25, 0.00, 0.20, 0.10, 0.20],
            [140.0, 50.0, 3.0, 0.30, 0.20, 0.30, 0.12, 0.24],
        ],
        dtype=float,
    )
    best_fit = np.asarray([120.0, 40.0, 2.0, 0.25, 0.0, 0.2, 0.1, 0.2], dtype=float)

    table = plotting._scaling_results_summary_table(
        specs,
        samples,
        best_fit,
        "direct-exponents",
    )

    row = table.iloc[0]
    weights = np.full(samples.shape[0], 1.0 / samples.shape[0])
    expected_sigma_p16, expected_sigma_median, expected_sigma_p84 = plotting._weighted_quantile(
        samples[:, 0],
        weights,
        [0.16, 0.5, 0.84],
    )
    log_mass = plotting._log10_dpie_mass_msun(samples[:, 0], samples[:, 1])
    _mass_p16, expected_mass_median, _mass_p84 = plotting._weighted_quantile(log_mass, weights, [0.16, 0.5, 0.84])
    expected_alpha_median = plotting._weighted_quantile(samples[:, 3], weights, [0.5])[0]
    expected_gamma_median = plotting._weighted_quantile(samples[:, 4], weights, [0.5])[0]
    expected_beta_median = plotting._weighted_quantile(1.0 + samples[:, 4] - 2.0 * samples[:, 3], weights, [0.5])[0]

    assert row["potfile_id"] == "members"
    assert row["scaling_relation_mode"] == "direct-exponents"
    assert row["vdisp_star_median"] == pytest.approx(expected_sigma_median)
    assert row["vdisp_star_p16"] == pytest.approx(expected_sigma_p16)
    assert row["vdisp_star_p84"] == pytest.approx(expected_sigma_p84)
    assert row["alpha_sigma_median"] == pytest.approx(expected_alpha_median)
    assert row["gamma_ml_median"] == pytest.approx(expected_gamma_median)
    assert row["beta_radius_median"] == pytest.approx(expected_beta_median)
    assert row["log10_m_star_msun_median"] == pytest.approx(expected_mass_median)
    assert row["sigma_log_scatter_median"] == pytest.approx(
        plotting._weighted_quantile(samples[:, 5], weights, [0.5])[0]
    )
    assert "free_log_sigma_tau_median" not in table.columns
    assert "free_log_mass_tau_median" not in table.columns
    assert plotting.SCALING_RESULTS_MASS_NOTE in row["m_star_definition"]


def test_scaling_results_summary_table_bergamini_mode_reports_effective_beta() -> None:
    specs = [
        _potfile_parameter_spec("members", "sigma"),
        _potfile_parameter_spec("members", "cutkpc"),
        _potfile_parameter_spec("members", "corekpc"),
        _potfile_parameter_spec("members", "alpha_sigma"),
        _potfile_parameter_spec("members", "gamma_ml"),
    ]
    samples = np.asarray(
        [
            [100.0, 30.0, 1.0, 0.20, -0.10],
            [100.0, 30.0, 1.0, 0.25, 0.00],
            [100.0, 30.0, 1.0, 0.30, 0.20],
        ],
        dtype=float,
    )
    best_fit = np.asarray([100.0, 30.0, 1.0, 0.25, 0.0], dtype=float)

    table = plotting._scaling_results_summary_table(
        specs,
        samples,
        best_fit,
        "bergamini-ml",
    )

    row = table.iloc[0]
    beta = 1.0 + samples[:, 4] - 2.0 * samples[:, 3]
    weights = np.full(samples.shape[0], 1.0 / samples.shape[0])
    _beta_p16, beta_median, _beta_p84 = plotting._weighted_quantile(beta, weights, [0.16, 0.5, 0.84])
    alpha_median = plotting._weighted_quantile(samples[:, 3], weights, [0.5])[0]
    gamma_median = plotting._weighted_quantile(samples[:, 4], weights, [0.5])[0]

    assert row["scaling_relation_mode"] == "bergamini-ml"
    assert row["alpha_sigma_median"] == pytest.approx(alpha_median)
    assert row["gamma_ml_median"] == pytest.approx(gamma_median)
    assert row["beta_radius_median"] == pytest.approx(beta_median)


def test_scaling_results_summary_rich_table_smoke() -> None:
    df = pd.DataFrame(
        {
            "potfile_id": ["members"],
            "scaling_relation_mode": ["bergamini-ml"],
            "vdisp_star_median": [100.0],
            "vdisp_star_p16": [90.0],
            "vdisp_star_p84": [110.0],
            "rcut_star_kpc_median": [30.0],
            "rcut_star_kpc_p16": [25.0],
            "rcut_star_kpc_p84": [35.0],
            "rcore_star_kpc_median": [1.0],
            "rcore_star_kpc_p16": [0.8],
            "rcore_star_kpc_p84": [1.2],
            "log10_m_star_msun_median": [11.0],
            "log10_m_star_msun_p16": [10.8],
            "log10_m_star_msun_p84": [11.2],
            "alpha_sigma_median": [0.25],
            "alpha_sigma_p16": [0.2],
            "alpha_sigma_p84": [0.3],
            "beta_radius_median": [0.5],
            "beta_radius_p16": [0.4],
            "beta_radius_p84": [0.6],
            "gamma_ml_median": [0.0],
            "gamma_ml_p16": [-0.1],
            "gamma_ml_p84": [0.1],
        }
    )

    table = plotting._build_scaling_results_rich_table(df)

    assert table is not None


def test_active_scaling_summary_population_diagnostics_write_pdf_only(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "potfile_id": ["members", "members", "members", "cluster", "cluster", "cluster"],
            "rank": [1, 2, 3, 1, 2, 3],
            "component_index": [1, 2, 3, 4, 5, 6],
            "catalog_id": ["a", "b", "c", "d", "e", "f"],
            "catalog_mag": [18.0, 19.5, 21.0, 17.8, 20.2, 22.1],
            "x_centre": [0.0, 1.0, -1.0, 2.0, -2.0, 0.5],
            "y_centre": [0.0, -1.0, 1.2, -2.0, 2.0, 0.4],
            "p_active_median": [0.96, 0.62, 0.18, 0.88, 0.42, 0.08],
            "p_active_p16": [0.91, 0.40, 0.05, 0.77, 0.25, 0.02],
            "p_active_p84": [0.99, 0.80, 0.36, 0.95, 0.61, 0.18],
            "p_active_gate_median": [0.55, 0.70, 0.35, 0.48, 0.58, 0.20],
            "p_active_membership_median": [0.96, 0.62, 0.18, 0.88, 0.42, 0.08],
            "p_active_membership_p16": [0.91, 0.40, 0.05, 0.77, 0.25, 0.02],
            "p_active_membership_p84": [0.99, 0.80, 0.36, 0.95, 0.61, 0.18],
            "active_loglike_delta_median": [-5.0, -0.8, 1.5, -2.3, 0.4, 3.0],
            "active_loglike_delta_p16": [-6.0, -1.8, 0.4, -3.4, -0.6, 2.0],
            "active_loglike_delta_p84": [-4.0, 0.2, 2.6, -1.1, 1.4, 4.2],
            "active_inference_likelihood": ["population"] * 6,
            "frozen_active": [True, True, False, True, False, False],
        }
    )

    plotting._plot_active_scaling_summary(tmp_path, df, freeze_threshold=0.5)

    assert plotting._plot_path(tmp_path, "active_scaling_summary.pdf").is_file()
    assert not (tmp_path / "active_scaling_summary.png").exists()


def test_scaling_relation_summary_table_preserves_classes_and_free_branch() -> None:
    scaling_rank_df = pd.DataFrame(
        {
            "potfile_id": ["members", "members", "members"],
            "catalog_id": ["inactive", "active", "free"],
            "rank": [3, 2, 1],
            "component_index": [0, 1, 2],
            "free_component_index": [-1, -1, 5],
            "catalog_mag": [21.0, 20.0, 19.0],
            "catalog_color": [0.8, 1.0, 1.2],
            "selected_active": [False, True, True],
            "selected_independent": [False, False, True],
        }
    )
    independent_df = pd.DataFrame(
        {
            "potfile_id": ["members"],
            "component_index": [2],
            "free_v_disp_median": [420.0],
            "free_v_disp_p16": [390.0],
            "free_v_disp_p84": [450.0],
            "free_v_disp_map": [430.0],
            "free_core_radius_kpc_median": [3.5],
            "free_core_radius_kpc_p16": [3.0],
            "free_core_radius_kpc_p84": [4.0],
            "free_core_radius_kpc_map": [3.6],
            "free_cut_radius_kpc_median": [80.0],
            "free_cut_radius_kpc_p16": [70.0],
            "free_cut_radius_kpc_p84": [90.0],
            "free_cut_radius_kpc_map": [82.0],
        }
    )
    packed = SimpleNamespace(
        profile_type=np.ones(3, dtype=np.int32),
        luminosity_ratio=np.asarray([0.25, 1.0, 4.0], dtype=float),
        sigma_ref_base=np.full(3, 300.0, dtype=float),
        cut_ref_base=np.full(3, 50.0, dtype=float),
        core_ref_base=np.full(3, 2.0, dtype=float),
        alpha_sigma_base=np.full(3, 0.25, dtype=float),
        gamma_ml_base=np.full(3, 0.0, dtype=float),
        sigma_ref_param_index=np.zeros(3, dtype=np.int32),
        cut_ref_param_index=np.full(3, 1, dtype=np.int32),
        core_ref_param_index=np.full(3, 2, dtype=np.int32),
        alpha_sigma_param_index=np.full(3, 3, dtype=np.int32),
        gamma_ml_param_index=np.full(3, 4, dtype=np.int32),
    )
    samples = np.asarray(
        [
            [280.0, 45.0, 1.8, 0.25, 0.0],
            [300.0, 50.0, 2.0, 0.25, 0.0],
            [320.0, 55.0, 2.2, 0.25, 0.0],
        ],
        dtype=float,
    )

    table = plotting._scaling_relation_summary_table(
        scaling_rank_df,
        [SimpleNamespace()],
        samples,
        np.asarray([300.0, 50.0, 2.0, 0.25, 0.0], dtype=float),
        packed,
        independent_scaling_df=independent_df,
        best_value="maximum-likelihood",
        best_value_requested="maximum-likelihood",
    )

    assert table["catalog_id"].tolist() == ["free", "active", "inactive"]
    assert not any(str(column).endswith("_map") for column in table.columns)
    assert "scaling_v_disp_best" in table.columns
    assert "free_v_disp_best" in table.columns
    assert set(table["best_value"]) == {"maximum-likelihood"}
    assert set(table["best_value_requested"]) == {"maximum-likelihood"}
    assert table["scaling_relation_class"].tolist() == ["free", "active", "inactive"]
    assert table.set_index("catalog_id")["catalog_color"].to_dict() == {
        "free": pytest.approx(1.2),
        "active": pytest.approx(1.0),
        "inactive": pytest.approx(0.8),
    }
    assert np.isfinite(table["scaling_v_disp_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["scaling_core_radius_kpc_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["scaling_cut_radius_kpc_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["scaling_log10_mass_msun_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["alpha_sigma_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["beta_radius_median"].to_numpy(dtype=float)).all()
    assert np.isfinite(table["gamma_ml_median"].to_numpy(dtype=float)).all()
    core_by_id = table.set_index("catalog_id")["scaling_core_radius_kpc_median"].to_dict()
    assert core_by_id["inactive"] == pytest.approx(0.95)
    assert core_by_id["active"] == pytest.approx(1.9)
    assert core_by_id["free"] == pytest.approx(3.8)
    active_mass = table.set_index("catalog_id").loc["active", "scaling_log10_mass_msun_median"]
    _mass_p16, expected_active_mass, _mass_p84 = plotting._weighted_quantile(
        plotting._log10_dpie_mass_msun(samples[:, 0], samples[:, 1]),
        np.ones(samples.shape[0], dtype=float) / samples.shape[0],
        [0.16, 0.5, 0.84],
    )
    assert active_mass == pytest.approx(expected_active_mass)
    free_row = table[table["catalog_id"] == "free"].iloc[0]
    assert free_row["free_v_disp_median"] == pytest.approx(420.0)
    assert free_row["free_v_disp_best"] == pytest.approx(430.0)
    assert free_row["free_core_radius_kpc_median"] == pytest.approx(3.5)
    assert free_row["free_core_radius_kpc_best"] == pytest.approx(3.6)
    assert free_row["free_cut_radius_kpc_median"] == pytest.approx(80.0)
    assert free_row["free_cut_radius_kpc_best"] == pytest.approx(82.0)
    expected_free_mass = math.log10(
        1.5 * math.pi * 420.0**2 * 80.0 / plotting.DPiE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN
    )
    assert free_row["free_log10_mass_msun_median"] == pytest.approx(expected_free_mass)


def test_scaling_relation_summary_plot_writes_pdf(tmp_path: Path) -> None:
    relation_df = pd.DataFrame(
        {
            "potfile_id": ["members", "members", "members"],
            "catalog_id": ["inactive", "active", "free"],
            "rank": [3, 2, 1],
            "component_index": [0, 1, 2],
            "catalog_mag": [21.0, 20.0, 19.0],
            "catalog_color": [0.8, 1.0, 1.2],
            "luminosity_ratio": [0.4, 1.0, 2.5],
            "anchor_mag": [20.0, 20.0, 20.0],
            "alpha_sigma_median": [0.25, 0.25, 0.25],
            "beta_radius_median": [0.5, 0.5, 0.5],
            "gamma_ml_median": [0.0, 0.0, 0.0],
            "alpha_sigma_best": [0.25, 0.25, 0.25],
            "beta_radius_best": [0.5, 0.5, 0.5],
            "gamma_ml_best": [0.0, 0.0, 0.0],
            "scaling_relation_class": ["inactive", "active", "free"],
            "scaling_v_disp_median": [210.0, 300.0, 420.0],
            "scaling_v_disp_best": [215.0, 305.0, 425.0],
            "scaling_v_disp_p16": [190.0, 280.0, 390.0],
            "scaling_v_disp_p84": [230.0, 320.0, 450.0],
            "scaling_core_radius_kpc_median": [1.0, 2.0, 4.0],
            "scaling_core_radius_kpc_best": [1.1, 2.1, 4.1],
            "scaling_core_radius_kpc_p16": [0.8, 1.8, 3.6],
            "scaling_core_radius_kpc_p84": [1.2, 2.2, 4.4],
            "scaling_cut_radius_kpc_median": [25.0, 50.0, 100.0],
            "scaling_cut_radius_kpc_best": [26.0, 51.0, 101.0],
            "scaling_cut_radius_kpc_p16": [22.0, 45.0, 90.0],
            "scaling_cut_radius_kpc_p84": [28.0, 55.0, 110.0],
            "scaling_log10_mass_msun_median": [11.2, 12.0, 12.8],
            "scaling_log10_mass_msun_best": [11.25, 12.05, 12.85],
            "scaling_log10_mass_msun_p16": [11.1, 11.9, 12.7],
            "scaling_log10_mass_msun_p84": [11.3, 12.1, 12.9],
            "free_v_disp_median": [np.nan, np.nan, 500.0],
            "free_v_disp_best": [np.nan, np.nan, 510.0],
            "free_v_disp_p16": [np.nan, np.nan, 470.0],
            "free_v_disp_p84": [np.nan, np.nan, 530.0],
            "free_core_radius_kpc_median": [np.nan, np.nan, 5.0],
            "free_core_radius_kpc_best": [np.nan, np.nan, 5.2],
            "free_core_radius_kpc_p16": [np.nan, np.nan, 4.5],
            "free_core_radius_kpc_p84": [np.nan, np.nan, 5.5],
            "free_cut_radius_kpc_median": [np.nan, np.nan, 120.0],
            "free_cut_radius_kpc_best": [np.nan, np.nan, 123.0],
            "free_cut_radius_kpc_p16": [np.nan, np.nan, 105.0],
            "free_cut_radius_kpc_p84": [np.nan, np.nan, 135.0],
            "free_log10_mass_msun_median": [np.nan, np.nan, 13.0],
            "free_log10_mass_msun_best": [np.nan, np.nan, 13.05],
            "free_log10_mass_msun_p16": [np.nan, np.nan, 12.9],
            "free_log10_mass_msun_p84": [np.nan, np.nan, 13.1],
        }
    )

    plotting._plot_scaling_relation_summary(tmp_path, relation_df)

    assert plotting._plot_path(tmp_path, "scaling_relation_summary.pdf").is_file()
    assert not (tmp_path / "scaling_relation_summary.png").exists()


def test_perturbation_discovery_diagnostics_plot_writes_pdf(tmp_path: Path) -> None:
    diagnostics_df = pd.DataFrame(
        {
            "potfile_id": ["members"] * 6,
            "potfile_order": [0] * 6,
            "catalog_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "catalog_row_index": [0, 0, 1, 1, 2, 2],
            "component_index": [10, 10, 11, 11, 12, 12],
            "image_index": [0, 1, 0, 1, 0, 1],
            "family_id": ["1"] * 6,
            "image_label": ["a", "b", "a", "b", "a", "b"],
            "alpha_x_arcsec": [0.2, 0.0, 0.05, 0.0, 0.004, 0.0],
            "alpha_y_arcsec": [0.0] * 6,
            "alpha_arcsec": [0.2, 0.0, 0.05, 0.0, 0.004, 0.0],
            "jacobian_frobenius": [0.0, 0.6, 0.0, 0.0, 0.0, 0.02],
            "alpha_tol_arcsec": [0.1] * 6,
            "jacobian_tol": [0.5] * 6,
            "jacobian_weight": [1.0] * 6,
            "alpha_norm": [2.0, 0.0, 0.5, 0.0, 0.04, 0.0],
            "jacobian_norm": [0.0, 1.2, 0.0, 0.0, 0.0, 0.04],
            "score": [2.0, 1.2, 0.5, 0.0, 0.04, 0.04],
            "threshold_score": [1.0] * 6,
            "selected_pair": [True, True, False, False, False, False],
            "selected_galaxy": [True, True, False, False, False, False],
        }
    )

    plotting._plot_perturbation_discovery_diagnostics(tmp_path, diagnostics_df)

    assert plotting._plot_path(tmp_path, "perturbation_discovery_diagnostics.pdf").is_file()


def test_perturbation_discovery_diagnostics_plot_reports_top_k_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes

    text_calls: list[str] = []
    original_text = Axes.text

    def capture_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 3:
            text_calls.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_text)
    diagnostics_df = pd.DataFrame(
        {
            "potfile_id": ["members"] * 4,
            "potfile_order": [0] * 4,
            "catalog_id": ["g1", "g1", "g2", "g2"],
            "catalog_row_index": [0, 0, 1, 1],
            "component_index": [10, 10, 11, 11],
            "image_index": [0, 1, 0, 1],
            "family_id": ["1"] * 4,
            "image_label": ["a", "b", "a", "b"],
            "alpha_x_arcsec": [0.2, 0.0, 0.1, 0.0],
            "alpha_y_arcsec": [0.0] * 4,
            "alpha_arcsec": [0.2, 0.0, 0.1, 0.0],
            "jacobian_frobenius": [0.0] * 4,
            "alpha_tol_arcsec": [1.0] * 4,
            "jacobian_tol": [1.0] * 4,
            "jacobian_weight": [1.0] * 4,
            "alpha_norm": [0.2, 0.0, 0.1, 0.0],
            "jacobian_norm": [0.0] * 4,
            "score": [0.2, 0.0, 0.1, 0.0],
            "threshold_score": [1.0] * 4,
            "selection_mode": ["top_k"] * 4,
            "top_k_requested": [1] * 4,
            "rank_score": [0.2, 0.2, 0.1, 0.1],
            "rank_position": [1, 1, 2, 2],
            "selected_pair": [False] * 4,
            "selected_galaxy": [True, True, False, False],
        }
    )

    plotting._plot_perturbation_discovery_diagnostics(tmp_path, diagnostics_df)

    annotation_text = "\n".join(text_calls)
    assert "selection_mode = top_k" in annotation_text
    assert "top_k_requested = 1" in annotation_text


def test_perturbation_discovery_diagnostics_plot_fraction_labels_and_heatmap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    diagnostics_df = pd.DataFrame(
        {
            "potfile_id": ["members"] * 4,
            "potfile_order": [0] * 4,
            "catalog_id": ["g1", "g1", "g2", "g2"],
            "catalog_row_index": [0, 0, 1, 1],
            "component_index": [10, 10, 11, 11],
            "image_index": [0, 1, 0, 1],
            "family_id": ["1"] * 4,
            "image_label": ["a", "b", "a", "b"],
            "score": [2.0, 0.2, 0.5, 0.05],
            "alpha_norm": [2.0, 0.2, 0.5, 0.05],
            "jacobian_norm": [0.1, 1.2, 0.05, 0.04],
            "selected_pair": [True, True, False, False],
            "selected_galaxy": [True, True, False, False],
            "alpha_tol_arcsec": [0.1] * 4,
            "jacobian_tol": [0.5] * 4,
            "jacobian_weight": [1.0] * 4,
            "threshold_score": [1.0] * 4,
        }
    )
    labels: list[str] = []
    plot_labels: list[str] = []
    hist2d_calls: list[int] = []
    original_set_xlabel = plotting.plt.Axes.set_xlabel
    original_set_ylabel = plotting.plt.Axes.set_ylabel
    original_plot = plotting.plt.Axes.plot
    original_hist2d = plotting.plt.Axes.hist2d

    def spy_set_xlabel(self, xlabel, *args, **kwargs):
        labels.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def spy_set_ylabel(self, ylabel, *args, **kwargs):
        labels.append(str(ylabel))
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    def spy_plot(self, *args, **kwargs):
        if kwargs.get("label") is not None:
            plot_labels.append(str(kwargs["label"]))
        return original_plot(self, *args, **kwargs)

    def spy_hist2d(self, *args, **kwargs):
        hist2d_calls.append(1)
        return original_hist2d(self, *args, **kwargs)

    monkeypatch.setattr(plotting.plt.Axes, "set_xlabel", spy_set_xlabel)
    monkeypatch.setattr(plotting.plt.Axes, "set_ylabel", spy_set_ylabel)
    monkeypatch.setattr(plotting.plt.Axes, "plot", spy_plot)
    monkeypatch.setattr(plotting.plt.Axes, "hist2d", spy_hist2d)

    plotting._plot_perturbation_discovery_diagnostics(tmp_path, diagnostics_df)

    assert any("alpha fraction" in label for label in labels)
    assert any("Jacobian fraction" in label for label in labels)
    assert "score = 1" in plot_labels
    assert hist2d_calls
    assert plotting._plot_path(tmp_path, "perturbation_discovery_diagnostics.pdf").is_file()


def test_scaling_relation_summary_plot_layers_and_counts_box(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    relation_df = pd.DataFrame(
        {
            "potfile_id": ["members", "members", "members", "members"],
            "catalog_id": ["inactive1", "inactive2", "active", "free"],
            "rank": [4, 3, 2, 1],
            "component_index": [0, 1, 2, 3],
            "catalog_mag": [22.0, 21.0, 20.0, 19.0],
            "catalog_color": [0.6, 0.8, 1.0, 1.2],
            "luminosity_ratio": [0.16, 0.4, 1.0, 2.5],
            "anchor_mag": [20.0, 20.0, 20.0, 20.0],
            "alpha_sigma_median": [0.24, 0.24, 0.24, 0.24],
            "beta_radius_median": [0.52, 0.52, 0.52, 0.52],
            "gamma_ml_median": [0.0, 0.0, 0.0, 0.0],
            "alpha_sigma_best": [0.31, 0.31, 0.31, 0.31],
            "beta_radius_best": [0.44, 0.44, 0.44, 0.44],
            "gamma_ml_best": [0.12, 0.12, 0.12, 0.12],
            "scaling_relation_class": ["inactive", "inactive", "active", "free"],
            "scaling_v_disp_median": [150.0, 210.0, 300.0, 420.0],
            "scaling_v_disp_best": [151.0, 211.0, 301.0, 421.0],
            "scaling_v_disp_p16": [140.0, 190.0, 280.0, 390.0],
            "scaling_v_disp_p84": [160.0, 230.0, 320.0, 450.0],
            "scaling_core_radius_kpc_median": [0.7, 1.0, 2.0, 4.0],
            "scaling_core_radius_kpc_best": [0.75, 1.05, 2.05, 4.05],
            "scaling_core_radius_kpc_p16": [0.6, 0.8, 1.8, 3.6],
            "scaling_core_radius_kpc_p84": [0.8, 1.2, 2.2, 4.4],
            "scaling_cut_radius_kpc_median": [18.0, 25.0, 50.0, 100.0],
            "scaling_cut_radius_kpc_best": [19.0, 26.0, 51.0, 101.0],
            "scaling_cut_radius_kpc_p16": [16.0, 22.0, 45.0, 90.0],
            "scaling_cut_radius_kpc_p84": [20.0, 28.0, 55.0, 110.0],
            "scaling_log10_mass_msun_median": [11.0, 11.4, 12.0, 12.8],
            "scaling_log10_mass_msun_best": [11.05, 11.45, 12.05, 12.85],
            "scaling_log10_mass_msun_p16": [10.9, 11.3, 11.9, 12.7],
            "scaling_log10_mass_msun_p84": [11.1, 11.5, 12.1, 12.9],
            "free_v_disp_median": [np.nan, np.nan, np.nan, 500.0],
            "free_v_disp_best": [np.nan, np.nan, np.nan, 510.0],
            "free_v_disp_p16": [np.nan, np.nan, np.nan, 470.0],
            "free_v_disp_p84": [np.nan, np.nan, np.nan, 530.0],
            "free_core_radius_kpc_median": [np.nan, np.nan, np.nan, 5.0],
            "free_core_radius_kpc_best": [np.nan, np.nan, np.nan, 5.2],
            "free_core_radius_kpc_p16": [np.nan, np.nan, np.nan, 4.5],
            "free_core_radius_kpc_p84": [np.nan, np.nan, np.nan, 5.5],
            "free_cut_radius_kpc_median": [np.nan, np.nan, np.nan, 120.0],
            "free_cut_radius_kpc_best": [np.nan, np.nan, np.nan, 123.0],
            "free_cut_radius_kpc_p16": [np.nan, np.nan, np.nan, 105.0],
            "free_cut_radius_kpc_p84": [np.nan, np.nan, np.nan, 135.0],
            "free_log10_mass_msun_median": [np.nan, np.nan, np.nan, 13.0],
            "free_log10_mass_msun_best": [np.nan, np.nan, np.nan, 13.05],
            "free_log10_mass_msun_p16": [np.nan, np.nan, np.nan, 12.9],
            "free_log10_mass_msun_p84": [np.nan, np.nan, np.nan, 13.1],
        }
    )
    labels: list[str] = []
    text_values: list[str] = []
    errorbar_colors: list[Any] = []
    errorbar_y_values: list[float] = []
    interval_y_values: list[tuple[float, float]] = []
    colorbar_labels: list[str] = []
    colorbar_cmaps: list[str] = []

    class FakeAxis:
        def errorbar(self, *args: Any, **kwargs: Any) -> None:
            labels.append(str(kwargs.get("label", "")))
            errorbar_colors.append(kwargs.get("color"))
            errorbar_y_values.extend(float(value) for value in np.asarray(args[1], dtype=float).reshape(-1))

        def plot(self, *args: Any, **kwargs: Any) -> None:
            labels.append(str(kwargs.get("label", "")))
            if kwargs.get("label") == "_nolegend_" and len(args) >= 2:
                y_values = np.asarray(args[1], dtype=float).reshape(-1)
                if y_values.size == 2:
                    interval_y_values.append((float(y_values[0]), float(y_values[1])))

        def text(self, *_args: Any, **_kwargs: Any) -> None:
            text_values.append(str(_args[2]))

        def set_yscale(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_ylabel(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_title(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def grid(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def legend(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        @property
        def transAxes(self) -> object:
            return object()

    class FakeFigure:
        def colorbar(self, *args: Any, **_kwargs: Any) -> Any:
            if args:
                colorbar_cmaps.append(str(getattr(getattr(args[0], "cmap", None), "name", "")))

            class FakeColorbar:
                def set_label(self, label: str) -> None:
                    colorbar_labels.append(label)

            return FakeColorbar()

        def savefig(self, path: Path, *_args: Any, **_kwargs: Any) -> None:
            path.write_bytes(b"%PDF-1.4\n")

    fake_axes = np.asarray([[FakeAxis(), FakeAxis(), FakeAxis(), FakeAxis()]], dtype=object)
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFigure(), fake_axes))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    plotting._plot_scaling_relation_summary(tmp_path, relation_df)

    assert "inactive/cached" in labels
    assert "active exact" in labels
    assert "free branch" in labels
    assert "scaling relation" in labels
    assert "constant M/L" in labels
    assert "free candidate, scaling branch" not in labels
    assert "tab:orange" not in errorbar_colors
    assert "tab:red" not in errorbar_colors
    assert "0.60" not in errorbar_colors
    assert any(isinstance(color, tuple) and len(color) == 4 for color in errorbar_colors)
    assert colorbar_labels == ["catalog color (F606W - F814W)"]
    assert colorbar_cmaps == ["coolwarm"]
    assert 151.0 in errorbar_y_values
    assert 150.0 not in errorbar_y_values
    assert 510.0 in errorbar_y_values
    assert 500.0 not in errorbar_y_values
    assert (140.0, 160.0) in interval_y_values
    assert any(
        "points: best fit; bars: 16-84% posterior" in value
        and "alpha_sigma: 0.31" in value
        and "beta_radius: 0.44" in value
        and "gamma_ml: 0.12" in value
        and "dlogM/dlogL: 1.12" in value
        and "constant M/L" in value
        for value in text_values
    )
    assert "total: 4\ninactive: 2\nactive not free: 1\nfree: 1" in text_values


def test_image_catalog_family_cutout_stage_eligibility(tmp_path: Path) -> None:
    args = SimpleNamespace(
        image_catalog_family_cutout_image_dir=tmp_path / "images",
        exact_image_diagnostics_stage3=False,
        image_catalog_family_cutouts=True,
        stage2_forward_mode="linearized",
    )

    assert plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage2_free_source_forward_fit",
    )
    assert not plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage1_backprojected_centroid_fit",
    )
    args.stage2_forward_mode = "none"
    assert plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage1_backprojected_centroid_fit",
    )
    args.image_catalog_family_cutouts = False
    assert not plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage2_free_source_forward_fit",
    )
    args.image_catalog_family_cutout_image_dir = None
    assert not plotting._image_catalog_family_cutout_enabled(
        args,
        tmp_path / "fit" / "stage2_free_source_forward_fit",
    )


def test_run_plot_tasks_with_progress_tracks_plot_names(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []
    progress_instances: list[Any] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[str] = []
            self.total: int | None = None
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self.descriptions.append(description)
            self.total = total
            return 1

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append(kwargs["description"])

        def advance(self, task_id: int) -> None:
            return None

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "_progress_context", lambda _args, *columns, **kwargs: FakeProgress(*columns, **kwargs))

    tasks: list[plotting.PlotTask] = [
        ("corner", "plots.corner", lambda: calls.append("corner")),
        ("timing_profile", "plots.timing_profile", lambda: calls.append("timing_profile")),
    ]

    plotting._run_plot_tasks_with_progress(argparse.Namespace(quiet=False), tasks)

    assert calls == ["corner", "timing_profile"]
    assert phases == ["plots.corner", "plots.timing_profile"]
    assert len(progress_instances) == 1
    assert progress_instances[0].total == 2
    assert progress_instances[0].descriptions == [
        "plots",
        "plots: corner",
        "plots: timing_profile",
        "plots: complete",
    ]


def test_progress_context_uses_rich_in_terminal(monkeypatch: Any) -> None:
    instances: list[Any] = []

    class FakeRichProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            instances.append(self)

        def __enter__(self) -> "FakeRichProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    monkeypatch.setattr(lc_utils, "is_notebook_environment", lambda: False)
    monkeypatch.setattr(lc_utils, "_RichProgress", FakeRichProgress)

    with lc_utils.progress_context(argparse.Namespace(quiet=False), "column", transient=True) as progress:
        assert progress is instances[0]

    assert instances[0].args == ("column",)
    assert instances[0].kwargs == {"transient": True}


def test_progress_context_uses_tqdm_in_notebook(monkeypatch: Any) -> None:
    instances: list[Any] = []

    class FakeNotebookProgress:
        def __init__(self) -> None:
            instances.append(self)

        def __enter__(self) -> "FakeNotebookProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def fail_rich(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("notebook progress should not instantiate Rich Progress")

    monkeypatch.setattr(lc_utils, "is_notebook_environment", lambda: True)
    monkeypatch.setattr(lc_utils, "_NotebookTqdmProgress", FakeNotebookProgress)
    monkeypatch.setattr(lc_utils, "_RichProgress", fail_rich)

    with lc_utils.progress_context(argparse.Namespace(quiet=False), "column", transient=True) as progress:
        assert progress is instances[0]


def test_progress_context_quiet_is_noop(monkeypatch: Any) -> None:
    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet progress should not instantiate a backend")

    monkeypatch.setattr(lc_utils, "_RichProgress", fail_progress)
    monkeypatch.setattr(lc_utils, "_NotebookTqdmProgress", fail_progress)

    with lc_utils.progress_context(argparse.Namespace(quiet=True), "column") as progress:
        assert progress is None


def test_run_plot_tasks_with_progress_quiet_skips_progress(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet plot execution should not create a progress bar")

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "_progress_context", fail_progress)

    tasks: list[plotting.PlotTask] = [
        ("corner", "plots.corner", lambda: calls.append("corner")),
        ("trace", "plots.trace", lambda: calls.append("trace")),
    ]

    plotting._run_plot_tasks_with_progress(argparse.Namespace(quiet=True), tasks)

    assert calls == ["corner", "trace"]
    assert phases == ["plots.corner", "plots.trace"]


def test_run_plot_stages_with_progress_tracks_stage_and_subtask_names(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []
    progress_instances: list[Any] = []
    received_progress: list[Any] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[str] = []
            self.totals: list[int] = []
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self.descriptions.append(description)
            self.totals.append(int(total))
            return len(self.totals)

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append(kwargs["description"])

        def advance(self, task_id: int) -> None:
            return None

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "_progress_context", lambda _args, *columns, **kwargs: FakeProgress(*columns, **kwargs))

    stages: list[plotting.PlotStage] = [
        (
            "run_diagnostics",
            [
                ("corner", "plots.corner", lambda: calls.append("corner")),
                ("chain_health", "plots.chain_health", lambda: calls.append("chain_health")),
            ],
        ),
        (
            "image_recovery",
            [
                (
                    "fit_quality_tables",
                    "plots.image_recovery.fit_quality_tables",
                    lambda progress=None: (
                        received_progress.append(progress),
                        calls.append("fit_quality_tables"),
                    ),
                ),
            ],
        ),
        ("empty_stage", []),
    ]

    plotting._run_plot_stages_with_progress(argparse.Namespace(quiet=False), stages)

    assert calls == ["corner", "chain_health", "fit_quality_tables"]
    assert received_progress == [progress_instances[0]]
    assert phases == [
        "plots.run_diagnostics",
        "plots.corner",
        "plots.chain_health",
        "plots.image_recovery",
        "plots.image_recovery.fit_quality_tables",
    ]
    assert len(progress_instances) == 1
    assert progress_instances[0].totals == [2, 2, 1]
    assert progress_instances[0].descriptions == [
        "plot stages",
        "plot stages: run_diagnostics",
        "run_diagnostics",
        "run_diagnostics: corner",
        "run_diagnostics: chain_health",
        "run_diagnostics: complete",
        "plot stages: image_recovery",
        "image_recovery",
        "image_recovery: fit_quality_tables",
        "image_recovery: complete",
        "plot stages: complete",
    ]


def test_run_plot_stages_with_progress_quiet_skips_progress(monkeypatch: Any) -> None:
    calls: list[str] = []
    phases: list[str] = []

    def fake_logged_phase(args: argparse.Namespace, phase_name: str, fn: Any) -> Any:
        phases.append(phase_name)
        return fn()

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet staged plot execution should not create a progress bar")

    monkeypatch.setattr(plotting, "_run_logged_phase", fake_logged_phase)
    monkeypatch.setattr(plotting, "_progress_context", fail_progress)

    plotting._run_plot_stages_with_progress(
        argparse.Namespace(quiet=True),
        [
            ("run_diagnostics", [("corner", "plots.corner", lambda: calls.append("corner"))]),
            ("truth_recovery", [("truth_recovery_grids", "plots.truth_recovery.truth_recovery_grids", lambda: calls.append("truth"))]),
        ],
    )

    assert calls == ["corner", "truth"]
    assert phases == [
        "plots.run_diagnostics",
        "plots.corner",
        "plots.truth_recovery",
        "plots.truth_recovery.truth_recovery_grids",
    ]


def _corner_test_specs(component_family: str = "large") -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family=component_family,
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family=component_family,
        ),
    ]


def _mixed_cosmology_test_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="halo.x",
            sample_name="halo_x",
            potential_id="halo",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
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


def test_corner_uses_one_two_three_sigma_contour_levels(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_points(self, _fig: FakeFig, _xs: list[list[float]], **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=float),
        _corner_test_specs(),
    )

    assert calls[0][0] == "corner"
    expected = [1.0 - math.exp(-0.5 * sigma**2) for sigma in (1.0, 2.0, 3.0)]
    np.testing.assert_allclose(plotting.CORNER_SIGMA_CONTOUR_LEVELS, expected)
    np.testing.assert_allclose(calls[0][2]["levels"], expected)


def _image_scatter_test_spec() -> ParameterSpec:
    return ParameterSpec(
        name="image.sigma_int",
        sample_name="image_sigma_int",
        potential_id="image",
        profile_type=0,
        field="sigma_int",
        prior_kind="uniform",
        lower=0.0,
        upper=2.0,
        step=0.01,
        component_family="image_scatter",
    )


def _source_position_test_spec(index: int) -> ParameterSpec:
    return ParameterSpec(
        name=f"source.{index}.beta_x",
        sample_name=f"source_{index}_beta_x",
        potential_id=str(index),
        profile_type=0,
        field="beta_x",
        prior_kind="uniform",
        lower=-10.0,
        upper=10.0,
        step=0.01,
        component_family="source_position",
    )


def _synthetic_stuck_chain_posterior() -> tuple[PosteriorResults, list[ParameterSpec]]:
    specs = [
        ParameterSpec(
            name="halo.x",
            sample_name="halo_x",
            potential_id="halo",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        ParameterSpec(
            name="halo.y",
            sample_name="halo_y",
            potential_id="halo",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
        _image_scatter_test_spec(),
    ]
    draws = np.linspace(-0.02, 0.02, 12)
    grouped = np.zeros((4, draws.size, len(specs)), dtype=float)
    grouped[:, :, 0] = np.asarray([0.0, 0.05, 0.08, 0.1], dtype=float)[:, None] + draws[None, :]
    grouped[:, :, 1] = np.asarray([0.0, 0.02, 0.03, 0.04], dtype=float)[:, None] + 0.5 * draws[None, :]
    grouped[0, :, 2] = 1.23 + 0.002 * np.linspace(-1.0, 1.0, draws.size)
    grouped[1, :, 2] = 0.10 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped[2, :, 2] = 0.11 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped[3, :, 2] = 0.09 + 0.01 * np.linspace(-1.0, 1.0, draws.size)
    grouped_log_prob = np.vstack(
        [
            np.full(draws.size, -230.0),
            np.linspace(-10.0, 5.0, draws.size),
            np.linspace(-8.0, 7.0, draws.size),
            np.linspace(0.0, 20.0, draws.size),
        ]
    )
    accept_prob = np.vstack(
        [
            np.full(draws.size, 0.95),
            np.full(draws.size, 0.98),
            np.full(draws.size, 0.97),
            np.full(draws.size, 0.90),
        ]
    )
    num_steps = np.vstack(
        [
            np.full(draws.size, 255.0),
            np.full(draws.size, 64.0),
            np.full(draws.size, 128.0),
            np.full(draws.size, 200.0),
        ]
    )
    posterior = PosteriorResults(
        samples=grouped.reshape((-1, grouped.shape[-1])),
        log_prob=grouped_log_prob.reshape(-1),
        accept_prob=accept_prob.reshape(-1),
        diverging=np.zeros(grouped.shape[0] * grouped.shape[1], dtype=bool),
        num_steps=num_steps.reshape(-1),
        warmup_steps=0,
        sample_steps=grouped.shape[1],
        num_chains=grouped.shape[0],
        init_diagnostics={"chain_seed_labels": ["stuck", "ok-1", "ok-2", "ok-3"]},
        grouped_samples=grouped,
        grouped_log_prob=grouped_log_prob,
    )
    return posterior, specs


def test_load_bayes_corner_overlay_maps_object_and_potfile_columns(tmp_path: Path, monkeypatch: Any) -> None:
    bayes_path = tmp_path / "bayes.dat"
    bayes_path.write_text(
        "\n".join(
            [
                "#Nsample",
                "#ln(Lhood)",
                "#O1 : x (arcsec)",
                "#O1 : rc (arcsec)",
                "#O2 : sigma (km/s)",
                "#Pot0 rcut (arcsec)",
                "#Pot0 sigma (km/s)",
                "#Chi2",
                "1 -10 1.5 2.0 300 4.0 200 9",
                "2 -11 1.7 2.5 310 5.0 220 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = SimpleNamespace(z_lens=0.4, cosmo_config={}, potfiles=[{"id": "potfile"}])
    monkeypatch.setattr(plotting, "_bayes_kpc_per_arcsec", lambda _state: 6.0)

    overlay = plotting._load_bayes_corner_overlay(bayes_path, state)

    assert overlay is not None
    np.testing.assert_allclose(overlay["1.x_centre"], [1.5, 1.7])
    np.testing.assert_allclose(overlay["1.core_radius_kpc"], [12.0, 15.0])
    np.testing.assert_allclose(overlay["2.v_disp"], [300.0, 310.0])
    np.testing.assert_allclose(overlay["potfile.cutkpc"], [24.0, 30.0])
    np.testing.assert_allclose(overlay["potfile.sigma"], [200.0, 220.0])
    assert "Chi2" not in overlay


def test_bayes_corner_overlay_uses_existing_corner_figure(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    figures: list[Any] = []
    specs = [
        ParameterSpec(
            name="potfile.cutkpc",
            sample_name="potfile_cutkpc",
            potential_id="potfile",
            profile_type=81,
            field="cutkpc",
            prior_kind="uniform",
            lower=1.0,
            upper=50.0,
            step=0.1,
            component_family="scaling",
        ),
        ParameterSpec(
            name="potfile.sigma",
            sample_name="potfile_sigma",
            potential_id="potfile",
            profile_type=81,
            field="sigma",
            prior_kind="uniform",
            lower=100.0,
            upper=400.0,
            step=1.0,
            component_family="scaling",
        ),
    ]

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            fig = kwargs.get("fig") or FakeFig()
            figures.append(fig)
            calls.append(("corner", np.asarray(samples), kwargs))
            return fig

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_potfile_corner(
        tmp_path,
        np.asarray([[20.0, 180.0], [25.0, 200.0], [30.0, 220.0]], dtype=float),
        specs,
        bayes_corner_overlay={
            "potfile.cutkpc": np.asarray([21.0, 26.0, 31.0], dtype=float),
            "potfile.sigma": np.asarray([190.0, 210.0, 230.0], dtype=float),
        },
    )

    assert calls[0][0] == "corner"
    assert calls[1][0] == "corner"
    np.testing.assert_allclose(calls[0][2]["weights"], np.full(3, 1.0 / 3.0))
    assert np.sum(calls[0][2]["weights"]) == pytest.approx(1.0)
    np.testing.assert_allclose(calls[1][1], np.asarray([[21.0, 190.0], [26.0, 210.0], [31.0, 230.0]]))
    np.testing.assert_allclose(calls[1][2]["weights"], np.full(3, 1.0 / 3.0))
    assert np.sum(calls[1][2]["weights"]) == pytest.approx(1.0)
    assert calls[1][2]["fig"] is figures[0]
    assert calls[1][2]["color"] == plotting.CORNER_BAYES_OVERLAY_COLOR
    assert calls[1][2]["fill_contours"] is False
    assert calls[1][2]["no_fill_contours"] is True
    assert "contourf_kwargs" not in calls[1][2]
    assert calls[1][2]["plot_datapoints"] is False


def test_best_par_overlay_support_is_removed_from_supported_plot_apis() -> None:
    assert not hasattr(plotting, "_load_best_par_marker_values")
    assert "best_par_marker_values" not in inspect.signature(plotting._plot_corner).parameters
    assert "best_par_marker_values" not in inspect.signature(plotting._plot_per_potential_summary).parameters


def test_per_potential_summary_uses_current_marker_colors_and_limits(tmp_path: Path, monkeypatch: Any) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.hlines_calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.xlim: tuple[float, float] | None = None
            self.title: str | None = None

        def hlines(self, y: Any, xmin: Any, xmax: Any, **kwargs: Any) -> None:
            self.hlines_calls.append((y, xmin, xmax, kwargs))

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def set_xlim(self, x_min: float, x_max: float) -> None:
            self.xlim = (x_min, x_max)

        def set_yticks(self, _ticks: list[Any]) -> None:
            return None

        def set_title(self, title: str) -> None:
            self.title = title

        def get_legend_handles_labels(self) -> tuple[list[str], list[str]]:
            labels = [str(kwargs["label"]) for _x, _y, kwargs in self.scatters if "label" in kwargs]
            return labels, labels

    class FakeFig:
        def __init__(self) -> None:
            self.legend_calls: list[tuple[Any, Any, dict[str, Any]]] = []
            self.saved_path: Path | None = None

        def legend(self, handles: Any, labels: Any, **kwargs: Any) -> None:
            self.legend_calls.append((handles, labels, kwargs))

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_path = Path(path)
            self.saved_path.write_text("pdf", encoding="utf-8")

    axis = FakeAxis()
    fig = FakeFig()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (fig, axis))
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)
    summary_df = pd.DataFrame(
        [
            {
                "label": "potfile.sigma",
                "p16": 2.0,
                "p84": 8.0,
                "median": 5.0,
                "map": 6.0,
                "lower": 0.0,
                "upper": 10.0,
                "std": 1.0,
            }
        ]
    )

    plotting._plot_per_potential_summary(
        tmp_path,
        summary_df,
        previous_stage_best_values={"potfile_sigma": -5.0},
        parameter_specs=[
            ParameterSpec(
                name="potfile.sigma",
                sample_name="potfile_sigma",
                potential_id="potfile",
                profile_type=81,
                field="sigma",
                prior_kind="uniform",
                lower=0.0,
                upper=10.0,
                step=1.0,
                component_family="scaling",
            )
        ],
    )

    assert axis.hlines_calls == [(1, 2.0, 8.0, {"linewidth": 4, "color": "tab:blue"})]
    assert axis.scatters[0] == ([5.0], [1], {"color": "tab:blue", "s": 35, "label": "median"})
    assert axis.scatters[1] == (
        [6.0],
        [1],
        {"color": plotting.CORNER_BEST_FIT_COLOR, "marker": "x", "s": 30, "label": "best fit"},
    )
    assert axis.scatters[2] == (
        [-5.0],
        [1],
        {
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "marker": "x",
            "s": 30,
            "label": "previous stage",
        },
    )
    assert axis.xlim is not None
    assert axis.xlim[0] < -5.0
    assert axis.xlim[1] >= 10.0
    assert axis.title == "potfile.sigma"
    assert fig.saved_path == tmp_path / "per_potential_summary.pdf"


def test_cosmology_parameter_subset_keeps_only_cosmology_columns() -> None:
    samples = np.asarray(
        [
            [10.0, 0.28, -1.1],
            [11.0, 0.30, -1.0],
            [12.0, 0.32, -0.9],
        ],
        dtype=float,
    )
    best_fit = np.asarray([11.5, 0.31, -0.95], dtype=float)

    subset_specs, subset_samples, subset_best_fit = plotting._cosmology_parameter_subset(
        _mixed_cosmology_test_specs(),
        samples,
        best_fit,
    )

    assert [spec.name for spec in subset_specs] == ["cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(subset_samples, samples[:, [1, 2]])
    np.testing.assert_allclose(subset_best_fit, best_fit[[1, 2]])


def test_corner_overlays_gold_best_fit_and_preserves_truths(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=float),
        _corner_test_specs(),
        truth_values={"x": 0.5, "y": 2.5},
        best_fit_values={"x": 1.5, "y": 3.5},
        map_values={"x": 2.0, "y": 4.0},
        maximum_likelihood_values={"x": 1.75, "y": 3.75},
        previous_stage_best_values={"x": 1.25, "y": 3.25},
    )

    assert calls[0][0] == "corner"
    assert calls[0][2]["truths"] == [0.5, 2.5]
    assert calls[1] == (
        "points",
        [[1.25, 3.25]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[2.0, 4.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAP_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[3] == (
        "points",
        [[1.75, 3.75]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAXIMUM_LIKELIHOOD_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert all(call[1] != [[1.5, 3.5]] for call in calls if call[0] == "points")


def test_map_values_for_specs_uses_max_log_prob_sample() -> None:
    samples = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=float)
    log_prob = np.asarray([-1.0, 3.0, 2.0], dtype=float)

    assert plotting._map_values_for_specs(_corner_test_specs(), samples, log_prob) == {
        "x": 1.0,
        "y": 2.0,
    }


def test_map_values_for_specs_rejects_missing_or_bad_log_prob() -> None:
    samples = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    specs = _corner_test_specs()

    assert plotting._map_values_for_specs(specs, samples, None) == {}
    assert plotting._map_values_for_specs(specs, samples, np.asarray([np.nan, np.nan])) == {}
    assert plotting._map_values_for_specs(specs, samples, np.asarray([1.0])) == {}


def test_corner_excludes_source_positions_before_finite_filtering(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    specs = [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
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
            component_family="source_position",
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
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
            component_family="source_position",
        ),
    ]

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_corner(
        tmp_path,
        np.asarray(
            [
                [0.0, np.nan, 1.0, 10.0],
                [1.0, np.inf, 2.0, 11.0],
                [2.0, -np.inf, 4.0, 12.0],
            ],
            dtype=float,
        ),
        specs,
        truth_values={"x": 0.5, "y": 2.5, "source.1.beta_x": 9.0, "source.1.beta_y": 9.5},
        best_fit_values={"x": 1.5, "y": 3.5, "source.1.beta_x": 10.0, "source.1.beta_y": 10.5},
        map_values={"x": 2.0, "y": 4.0, "source.1.beta_x": 12.0, "source.1.beta_y": 12.5},
        maximum_likelihood_values={"x": 1.75, "y": 3.75, "source.1.beta_x": 11.0, "source.1.beta_y": 11.5},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]))
    assert calls[0][2]["labels"] == ["x", "y"]
    assert calls[0][2]["truths"] == [0.5, 2.5]
    assert calls[1] == (
        "points",
        [[2.0, 4.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAP_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[1.75, 3.75]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAXIMUM_LIKELIHOOD_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )


def test_potfile_corner_includes_shared_free_hyperparameters(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    specs = [
        ParameterSpec(
            name="members.sigma_star",
            sample_name="members_sigma_star",
            potential_id="members",
            profile_type=81,
            field="sigma_ref",
            prior_kind="uniform",
            lower=20.0,
            upper=500.0,
            step=1.0,
            component_family="scaling",
        ),
        ParameterSpec(
            name="members.alpha_sigma",
            sample_name="members_alpha_sigma",
            potential_id="members",
            profile_type=81,
            field="alpha_sigma",
            prior_kind="uniform",
            lower=0.05,
            upper=0.6,
            step=0.01,
            component_family="scaling",
        ),
        ParameterSpec(
            name="members.independent_free_log_sigma_tau",
            sample_name="members_independent_free_log_sigma_tau",
            potential_id="members",
            profile_type=81,
            field="independent_free_log_sigma_tau",
            prior_kind="lognormal",
            lower=0.0,
            upper=10.0,
            physical_mean=0.15,
            std=0.3,
            step=0.01,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="members.independent_free_log_mass_tau",
            sample_name="members_independent_free_log_mass_tau",
            potential_id="members",
            profile_type=81,
            field="independent_free_log_mass_tau",
            prior_kind="lognormal",
            lower=0.0,
            upper=10.0,
            physical_mean=0.35,
            std=0.3,
            step=0.01,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="members.g1.independent_free_log_sigma_delta_unit",
            sample_name="members_g1_independent_free_log_sigma_delta_unit",
            potential_id="members",
            profile_type=81,
            field="independent_free_log_sigma_delta_unit",
            prior_kind="normal",
            lower=-np.inf,
            upper=np.inf,
            physical_mean=0.0,
            std=1.0,
            step=0.01,
            component_family="independent_scaling",
        ),
        ParameterSpec(
            name="halo.x",
            sample_name="halo_x",
            potential_id="halo",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-5.0,
            upper=5.0,
            step=0.1,
            component_family="large",
        ),
    ]
    samples = np.asarray(
        [
            [10.0, 0.20, 0.10, 0.30, -1.0, 4.0],
            [11.0, 0.21, 0.11, 0.31, -0.5, 4.5],
            [12.0, 0.22, 0.12, 0.32, 0.0, 5.0],
            [13.0, 0.23, 0.13, 0.33, 0.5, 5.5],
            [14.0, 0.24, 0.14, 0.34, 1.0, 6.0],
        ],
        dtype=float,
    )
    best_fit = np.asarray([12.0, 0.22, 0.12, 0.32, 0.0, 5.0], dtype=float)
    subset_specs, subset_samples, subset_best_fit = plotting._potfile_corner_parameter_subset(
        specs,
        samples,
        best_fit,
    )

    assert [spec.field for spec in subset_specs] == [
        "sigma_ref",
        "alpha_sigma",
    ]
    np.testing.assert_allclose(subset_samples, samples[:, [0, 1]])
    np.testing.assert_allclose(subset_best_fit, best_fit[[0, 1]])

    plotting._plot_potfile_corner(
        tmp_path,
        subset_samples,
        subset_specs,
        truth_values={
            "members_sigma_star": 10.5,
            "members_alpha_sigma": 0.205,
        },
        best_fit_values={
            "members_sigma_star": 12.0,
            "members_alpha_sigma": 0.22,
        },
        map_values={
            "members_sigma_star": 13.0,
            "members_alpha_sigma": 0.23,
        },
        maximum_likelihood_values={
            "members_sigma_star": 12.5,
            "members_alpha_sigma": 0.225,
        },
        previous_stage_best_values={
            "members_sigma_star": 11.5,
            "members_alpha_sigma": 0.215,
        },
    )

    assert calls[0][1].shape == (5, 2)
    assert calls[0][2]["labels"] == [
        "members.sigma_star",
        "members.alpha_sigma",
    ]
    assert calls[0][2]["truths"] == [10.5, 0.205]
    assert calls[1] == (
        "points",
        [[11.5, 0.215]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[13.0, 0.23]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAP_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[3] == (
        "points",
        [[12.5, 0.225]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAXIMUM_LIKELIHOOD_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )


def test_cosmology_corner_uses_sample_name_truths_and_best_fit(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    samples = np.asarray(
        [
            [10.0, 0.28, -1.1],
            [11.0, 0.30, -1.0],
            [12.0, 0.32, -0.9],
        ],
        dtype=float,
    )
    best_fit = np.asarray([11.5, 0.31, -0.95], dtype=float)
    cosmology_specs, cosmology_samples, cosmology_best_fit = plotting._cosmology_parameter_subset(
        _mixed_cosmology_test_specs(),
        samples,
        best_fit,
    )

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_cosmology_corner(
        tmp_path,
        cosmology_samples,
        cosmology_specs,
        truth_values={"cosmology_Om0": 0.3, "cosmology_w0": -1.0},
        best_fit_values=plotting._best_fit_values_for_specs(cosmology_specs, cosmology_best_fit),
        map_values={"cosmology_Om0": 0.32, "cosmology_w0": -0.9},
        maximum_likelihood_values={"cosmology_Om0": 0.315, "cosmology_w0": -0.92},
        previous_stage_best_values={"cosmology_Om0": 0.29, "cosmology_w0": -1.05},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], samples[:, [1, 2]])
    assert calls[0][2]["labels"] == ["cosmology.Om0", "cosmology.w0"]
    assert calls[0][2]["truths"] == [0.3, -1.0]
    assert calls[1] == (
        "points",
        [[0.29, -1.05]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[0.32, -0.9]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAP_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[3] == (
        "points",
        [[0.315, -0.92]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAXIMUM_LIKELIHOOD_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )


def test_fit_quality_tables_cap_draws_convert_physical_and_quantile(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=5.0,
        n_images=2,
        image_labels=["1.1", "1.2"],
        x_obs=np.asarray([0.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 1.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 7.0
            self.converted: list[np.ndarray] = []
            self.exact_latents: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 2.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_latents.append(offset)
            return family_data.x_obs + offset, family_data.y_obs - offset, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float))

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(posterior_image_diagnostic_draws=2),
    )

    assert [float(item[0]) for item in evaluator.converted] == [5.0, 0.0, 30.0]
    assert evaluator.exact_latents == [6.0, 1.0, 31.0]

    row = image_df.set_index("image_label").loc["1.1"]
    assert row["z_source"] == pytest.approx(2.0)
    assert row["sigma_arcsec"] == pytest.approx(5.0)
    assert row["image_sigma_int_arcsec"] == pytest.approx(2.0)
    assert row["image_sigma_eff_arcsec"] == pytest.approx(6.0)
    assert row["radius_arcsec"] == pytest.approx(0.0)
    assert row["angle_deg"] == pytest.approx(0.0)
    assert row["x_model_arcsec"] == pytest.approx(6.0)
    assert row["y_model_arcsec"] == pytest.approx(-6.0)
    assert row["x_model_q16"] == pytest.approx(5.8)
    assert row["x_model_q50"] == pytest.approx(16.0)
    assert row["x_model_q84"] == pytest.approx(26.2)
    assert row["image_residual_q50"] == pytest.approx(math.sqrt(2.0) * 16.0)
    assert row["residual_norm"] == pytest.approx(math.sqrt(2.0))
    assert row["residual_norm_q50"] == pytest.approx(math.sqrt(2.0) * 16.0 / 6.0)
    assert bool(row["covered_x_1sigma"]) is True
    assert bool(row["covered_y_1sigma"]) is True
    assert bool(row["covered_xy_1sigma"]) is True
    assert int(row["posterior_valid_draws"]) == 2
    assert int(row["model_produced_image_count"]) == 2
    assert int(row["model_recovered_image_count"]) == 2
    assert int(row["model_missing_image_count"]) == 0
    assert int(row["model_extra_image_count"]) == 0
    assert bool(row["model_multiplicity_failed"]) is False
    assert row["image_recovery_status"] == "recovered"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.2"]
    assert mag_row["magnification_model"] == pytest.approx(9.0)
    assert mag_row["magnification_model_q50"] == pytest.approx(19.0)
    assert int(mag_row["posterior_valid_draws"]) == 2


def test_fit_quality_tables_defaults_to_best_fit_only(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.converted: list[float] = []
            self.exact_latents: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            value = float(np.asarray(theta, dtype=float)[0])
            self.converted.append(value)
            return np.asarray([value + 1.0], dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_latents.append(offset)
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [10.0], [20.0]], dtype=float))

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([5.0], dtype=float),
        results,
        argparse.Namespace(),
    )

    assert evaluator.converted == [5.0]
    assert evaluator.exact_latents == [6.0]

    image_row = image_df.set_index("image_label").loc["1.1"]
    assert image_row["x_model_arcsec"] == pytest.approx(6.0)
    assert np.isnan(image_row["x_model_q50"])
    assert np.isnan(image_row["image_residual_q50"])
    assert int(image_row["posterior_valid_draws"]) == 0
    assert image_row["image_recovery_status"] == "recovered"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    assert mag_row["magnification_model"] == pytest.approx(6.0)
    assert np.isnan(mag_row["magnification_model_q50"])
    assert int(mag_row["posterior_valid_draws"]) == 0


def test_fit_quality_tables_quick_diagnostics_skips_exact_and_uses_median_std(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([1.0], dtype=float),
        y_obs=np.asarray([2.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.quick_diagnostics = True
            self.exact_calls = 0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, _sample_latent: np.ndarray, _family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            self.exact_calls += 1
            raise AssertionError("quick diagnostics should not solve exact image positions")

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()
    results = SimpleNamespace(samples=np.asarray([[0.0], [2.0], [4.0]], dtype=float))

    image_df, magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        evaluator,
        np.asarray([10.0], dtype=float),
        results,
        argparse.Namespace(posterior_image_diagnostic_draws=3, quick_diagnostics=True),
    )

    assert evaluator.exact_calls == 0
    image_row = image_df.set_index("image_label").loc["1.1"]
    assert bool(image_row["exact_image_prediction_failed"]) is True
    assert np.isnan(image_row["x_model_arcsec"])
    assert np.isnan(image_row["image_residual_arcsec"])
    assert np.isnan(image_row["model_produced_image_count"])
    assert image_row["model_multiplicity_failure_reason"] == "quick_diagnostics"
    assert image_row["image_recovery_status"] == "unknown"
    assert extra_image_df.empty

    mag_row = magnification_df.set_index("image_label").loc["1.1"]
    mag_values = np.asarray([3.0, 5.0, 7.0], dtype=float)
    assert mag_row["magnification_model"] == pytest.approx(13.0)
    assert mag_row["magnification_model_q16"] == pytest.approx(float(np.median(mag_values) - np.std(mag_values)))
    assert mag_row["magnification_model_q50"] == pytest.approx(float(np.median(mag_values)))
    assert mag_row["magnification_model_q84"] == pytest.approx(float(np.median(mag_values) + np.std(mag_values)))


def test_fit_quality_tables_tracks_partial_recovery_and_extra_images(monkeypatch: Any) -> None:
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        effective_z_source=2.0,
        sigma_arcsec=1.0,
        n_images=3,
        image_labels=["1.1", "1.2", "1.3"],
        x_obs=np.asarray([0.0, 1.0, 2.0], dtype=float),
        y_obs=np.asarray([0.0, 0.0, 0.0], dtype=float),
    )
    state = SimpleNamespace(parameter_specs=[], family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, _kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float)

    class FakeEvaluator:
        source_plane_covariance_floor = 0.0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction_details(self, _sample_latent: np.ndarray, _family_data: Any) -> dict[str, Any]:
            return {
                "produced_image_count": 3,
                "recovered_image_count": 2,
                "missing_image_count": 1,
                "extra_image_count": 1,
                "multiplicity_failed": True,
                "multiplicity_failure_reason": "match_tolerance_exceeded",
                "matched_model_x_arcsec": np.asarray([0.05, np.nan, 2.05], dtype=float),
                "matched_model_y_arcsec": np.asarray([0.0, np.nan, 0.0], dtype=float),
                "recovered_image_mask": np.asarray([True, False, True], dtype=bool),
                "extra_model_x_arcsec": np.asarray([8.0], dtype=float),
                "extra_model_y_arcsec": np.asarray([-3.0], dtype=float),
                "failed": True,
            }

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, _sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {}

        def _packed_to_kwargs_lens(self, _packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [{}]

    image_df, _magnification_df, extra_image_df = plotting._fit_quality_tables(
        state,
        FakeEvaluator(),
        np.asarray([0.0], dtype=float),
        SimpleNamespace(samples=np.empty((0, 1), dtype=float)),
        argparse.Namespace(posterior_image_diagnostic_draws=0),
    )

    indexed = image_df.set_index("image_label")
    assert indexed["image_recovery_status"].tolist() == ["recovered", "not_recovered", "recovered"]
    assert indexed["exact_image_prediction_failed"].tolist() == [False, True, False]
    assert indexed.loc["1.1", "x_model_arcsec"] == pytest.approx(0.05)
    assert np.isnan(indexed.loc["1.2", "x_model_arcsec"])
    assert indexed.loc["1.3", "image_residual_arcsec"] == pytest.approx(0.05)

    assert extra_image_df["family_id"].tolist() == ["1"]
    assert extra_image_df.loc[0, "image_recovery_status"] == "extra"
    assert extra_image_df.loc[0, "x_model_arcsec"] == pytest.approx(8.0)
    assert extra_image_df.loc[0, "y_model_arcsec"] == pytest.approx(-3.0)
    assert int(extra_image_df.loc[0, "model_extra_image_count"]) == 1


def test_posterior_fit_quality_predictions_parallelizes_per_sample_family(monkeypatch: Any) -> None:
    families = [
        SimpleNamespace(
            family_id=f"fam-{index}",
            z_source=2.0 + index,
            sigma_arcsec=0.5,
            search_window=[10.0, 30.0, 20.0][index],
            n_images=1,
            image_labels=[f"{index}.1"],
            x_obs=np.asarray([float(index)], dtype=float),
            y_obs=np.asarray([float(index + 1)], dtype=float),
        )
        for index in range(3)
    ]
    state = SimpleNamespace(family_data=families)

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            offset = float(kwargs_lens[0]["offset"])
            return np.asarray(x, dtype=float) + np.asarray(y, dtype=float) + offset

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.source_plane_covariance_floor = 0.0
            self.exact_calls: list[tuple[float, str]] = []

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            self.exact_calls.append((offset, family_data.family_id))
            return family_data.x_obs + offset, family_data.y_obs - offset, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    submitted: list[tuple[int, int]] = []
    progress_instances: list[Any] = []

    class InlineExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> "InlineExecutor":
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

        def submit(self, fn: Any, sample_index: int, family_index: int) -> Future:
            submitted.append((sample_index, family_index))
            future: Future = Future()
            try:
                future.set_result(fn(sample_index, family_index))
            except BaseException as exc:  # pragma: no cover - exercised by Future consumers if it fails
                future.set_exception(exc)
            return future

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[tuple[int, str]] = []
            self.advances: dict[int, int] = {}
            self.totals: dict[int, int] = {}
            self.transient = kwargs.get("transient")
            self._next_task_id = 0
            self.exited = False
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            self.exited = True
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self._next_task_id += 1
            task_id = self._next_task_id
            self.descriptions.append((task_id, description))
            self.totals[task_id] = total
            self.advances[task_id] = 0
            return task_id

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append((task_id, kwargs["description"]))

        def advance(self, task_id: int) -> None:
            self.advances[task_id] += 1

    evaluator = FakeEvaluator()
    log_messages: list[str] = []
    monkeypatch.setattr(plotting, "_clone_fit_quality_evaluator", lambda *_args, **_kwargs: evaluator)
    monkeypatch.setattr(plotting, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(plotting, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(plotting, "_progress_context", lambda _args, *columns, **kwargs: FakeProgress(*columns, **kwargs))
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 4)
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        evaluator,
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(),
    )

    assert submitted == [(0, 1), (1, 1), (0, 2), (1, 2), (0, 0), (1, 0)]
    assert evaluator.exact_calls == [
        (1.0, "fam-1"),
        (2.0, "fam-1"),
        (1.0, "fam-2"),
        (2.0, "fam-2"),
        (1.0, "fam-0"),
        (2.0, "fam-0"),
    ]
    assert len(predictions) == 2
    assert [row["image_label"] for row in predictions[0]["image_rows"]] == ["0.1", "1.1", "2.1"]
    assert len(predictions[0]["magnification_rows"]) == 3
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.transient is False
    assert progress.totals == {1: 6, 2: 2}
    assert progress.advances == {1: 6, 2: 2}
    assert progress.descriptions[0:2] == [
        (1, "fit quality exact: 0/6 family diagnostics"),
        (2, "draw progress: 0/2 complete"),
    ]
    assert (2, "draw progress: 1/2 complete") in progress.descriptions
    assert (2, "draw progress: 2/2 complete") in progress.descriptions
    assert (
        1,
        "fit quality exact: 6/6 family diagnostics | completed draw 2/2 family=fam-0 z=2.0000 window=10.0 grid=50x50",
    ) in progress.descriptions
    assert log_messages == [
        (
            "[plot:fit_quality] family diagnostics tasks=6 workers=4 families=3 draws=2 "
            "largest_grid=150x150 total_grid_points=70000"
        ),
        "[plot:fit_quality] draw 1/2 complete families=3/3 completed_tasks=5/6",
        "[plot:fit_quality] draw 2/2 complete families=3/3 completed_tasks=6/6",
    ]


def test_posterior_fit_quality_predictions_tracks_serial_progress(monkeypatch: Any) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        search_window=10.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.source_plane_covariance_floor = 0.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    progress_instances: list[Any] = []

    class FakeProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.descriptions: list[tuple[int, str]] = []
            self.advances: dict[int, int] = {}
            self.totals: dict[int, int] = {}
            self.transient = kwargs.get("transient")
            self._next_task_id = 0
            self.exited = False
            progress_instances.append(self)

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *exc: Any) -> bool:
            self.exited = True
            return False

        def add_task(self, description: str, *, total: int) -> int:
            self._next_task_id += 1
            task_id = self._next_task_id
            self.descriptions.append((task_id, description))
            self.totals[task_id] = total
            self.advances[task_id] = 0
            return task_id

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append((task_id, kwargs["description"]))

        def advance(self, task_id: int) -> None:
            self.advances[task_id] += 1

    monkeypatch.setattr(plotting, "_progress_context", lambda _args, *columns, **kwargs: FakeProgress(*columns, **kwargs))
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    log_messages: list[str] = []
    monkeypatch.setattr(plotting, "_log", lambda _args, message: log_messages.append(str(message)))

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(),
    )

    assert len(predictions) == 2
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.transient is False
    assert progress.exited is True
    assert progress.totals == {1: 2, 2: 2}
    assert progress.advances == {1: 2, 2: 2}
    assert progress.descriptions == [
        (1, "fit quality exact: 0/2 family diagnostics"),
        (2, "draw progress: 0/2 complete"),
        (1, "fit quality exact: 0/2 family diagnostics | running draw 1/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (1, "fit quality exact: 1/2 family diagnostics | completed draw 1/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (2, "draw progress: 1/2 complete"),
        (1, "fit quality exact: 1/2 family diagnostics | running draw 2/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (1, "fit quality exact: 2/2 family diagnostics | completed draw 2/2 family=1 z=2.0000 window=10.0 grid=50x50"),
        (2, "draw progress: 2/2 complete"),
    ]
    assert log_messages == [
        (
            "[plot:fit_quality] family diagnostics tasks=2 workers=1 families=1 draws=2 "
            "largest_grid=50x50 total_grid_points=5000"
        ),
        "[plot:fit_quality] draw 1/2 complete families=1/1 completed_tasks=1/2",
        "[plot:fit_quality] draw 2/2 complete families=1/1 completed_tasks=2/2",
    ]


def test_posterior_fit_quality_predictions_quiet_skips_progress(monkeypatch: Any) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        search_window=10.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.source_plane_covariance_floor = 0.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("quiet fit-quality diagnostics should not create a progress bar")

    monkeypatch.setattr(plotting, "_progress_context", fail_progress)
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float)],
        argparse.Namespace(quiet=True),
    )

    assert len(predictions) == 1


def test_posterior_fit_quality_predictions_reuses_existing_progress(monkeypatch: Any) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        sigma_arcsec=1.0,
        search_window=10.0,
        n_images=1,
        image_labels=["1.1"],
        x_obs=np.asarray([0.0], dtype=float),
        y_obs=np.asarray([0.0], dtype=float),
    )
    state = SimpleNamespace(family_data=[family])

    class FakeModel:
        def magnification(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            del x, y
            return np.asarray([float(kwargs_lens[0]["offset"])], dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.source_plane_covariance_floor = 0.0

        def _image_sigma_int_numpy(self, _sample_latent: np.ndarray) -> float:
            return 0.0

        def _exact_family_prediction(self, sample_latent: np.ndarray, family_data: Any) -> tuple[np.ndarray, np.ndarray, float]:
            offset = float(np.asarray(sample_latent, dtype=float)[0])
            return family_data.x_obs + offset, family_data.y_obs, offset

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: np.ndarray, _z_source: float) -> dict[str, float]:
            return {"offset": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    class ExistingProgress:
        def __init__(self) -> None:
            self.descriptions: list[tuple[int, str]] = []
            self.advances: dict[int, int] = {}
            self.totals: dict[int, int] = {}
            self._next_task_id = 0

        def add_task(self, description: str, *, total: int) -> int:
            self._next_task_id += 1
            task_id = self._next_task_id
            self.descriptions.append((task_id, description))
            self.totals[task_id] = total
            self.advances[task_id] = 0
            return task_id

        def update(self, task_id: int, **kwargs: Any) -> None:
            if "description" in kwargs:
                self.descriptions.append((task_id, kwargs["description"]))

        def advance(self, task_id: int) -> None:
            self.advances[task_id] += 1

    def fail_progress(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("existing fit-quality progress should be reused")

    existing_progress = ExistingProgress()
    monkeypatch.setattr(plotting, "_progress_context", fail_progress)
    monkeypatch.setattr(plotting, "jax_cpu_worker_count", lambda: 1)
    monkeypatch.setattr(plotting, "_log", lambda _args, _message: None)

    predictions = plotting._posterior_fit_quality_predictions(
        FakeEvaluator(),
        state,
        [np.asarray([1.0], dtype=float), np.asarray([2.0], dtype=float)],
        argparse.Namespace(),
        progress=existing_progress,
    )

    assert len(predictions) == 2
    assert existing_progress.totals == {1: 2, 2: 2}
    assert existing_progress.advances == {1: 2, 2: 2}
    assert existing_progress.descriptions[0:2] == [
        (1, "fit quality exact: 0/2 family diagnostics"),
        (2, "draw progress: 0/2 complete"),
    ]


def test_generate_plots_and_tables_writes_fit_quality_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "z_source": [2.0],
            "sigma_arcsec": [0.1],
            "image_sigma_int_arcsec": [0.0],
            "image_sigma_eff_arcsec": [0.1],
            "radius_arcsec": [0.0],
            "angle_deg": [0.0],
            "x_model_arcsec": [0.1],
            "y_model_arcsec": [0.2],
            "image_residual_arcsec": [0.3],
            "exact_image_prediction_failed": [False],
            "x_model_q16": [0.0],
            "x_model_q50": [0.1],
            "x_model_q84": [0.2],
            "y_model_q16": [0.1],
            "y_model_q50": [0.2],
            "y_model_q84": [0.3],
            "image_residual_q16": [0.2],
            "image_residual_q50": [0.3],
            "image_residual_q84": [0.4],
            "residual_norm": [3.0],
            "residual_norm_q50": [3.0],
            "covered_x_1sigma": [True],
            "covered_y_1sigma": [False],
            "covered_xy_1sigma": [False],
            "posterior_valid_draws": [1],
            "posterior_failed_draws": [0],
        }
    )
    magnification_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "image_label": ["1.1"],
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "magnification_model": [2.0],
            "magnification_prediction_failed": [False],
            "magnification_model_q16": [1.5],
            "magnification_model_q50": [2.0],
            "magnification_model_q84": [2.5],
            "posterior_valid_draws": [1],
            "posterior_failed_draws": [0],
        }
    )
    extra_image_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "extra_image_index": [1],
            "image_recovery_status": ["extra"],
            "x_model_arcsec": [3.0],
            "y_model_arcsec": [4.0],
        }
    )
    captured_tasks: list[str] = []
    captured_stages: list[str] = []

    monkeypatch.setattr(plotting, "_summary_table", lambda *_args, **_kwargs: pd.DataFrame({"label": ["mock"]}))
    monkeypatch.setattr(plotting, "_family_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame({"family_id": ["1"]}))
    monkeypatch.setattr(plotting, "_fit_quality_tables", lambda *_args, **_kwargs: (image_df, magnification_df, extra_image_df))
    monkeypatch.setattr(plotting, "_run_summary", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(plotting, "_potfile_constraint_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        plotting,
        "_potfile_corner_parameter_subset",
        lambda *_args, **_kwargs: ([], np.empty((1, 0)), np.empty((0,))),
    )
    monkeypatch.setattr(plotting, "_cosmology_parameter_subset", lambda *_args, **_kwargs: ([], np.empty((1, 0)), np.empty((0,))))
    monkeypatch.setattr(plotting, "_scaling_grouped_subset", lambda *_args, **_kwargs: ([], np.empty((0, 0, 0))))
    monkeypatch.setattr(plotting, "_best_fit_values_for_specs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plotting, "_write_potfile_summary_txt", lambda *_args, **_kwargs: None)

    def capture_stages(_args: argparse.Namespace, stages: list[plotting.PlotStage]) -> None:
        captured_stages.extend(stage_name for stage_name, _tasks in stages)
        for _stage_name, tasks in stages:
            for display_name, _phase_name, task in tasks:
                captured_tasks.append(display_name)
                if (
                    display_name.endswith("_table")
                    or display_name.startswith("write_")
                    or display_name.startswith("run_summary")
                    or display_name in {"fit_quality_tables", "scaling_results_summary_log"}
                ):
                    task()

    monkeypatch.setattr(plotting, "_run_plot_stages_with_progress", capture_stages)
    state = SimpleNamespace(parameter_specs=[], family_data=[], fit_mode="joint")
    evaluator = SimpleNamespace(scaling_rank_df=pd.DataFrame())
    results = PosteriorResults(
        samples=np.empty((1, 0), dtype=float),
        log_prob=np.asarray([0.0]),
        accept_prob=np.asarray([1.0]),
        diverging=np.asarray([False]),
        num_steps=np.asarray([1.0]),
        warmup_steps=0,
        sample_steps=1,
        num_chains=1,
    )
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "component_index": [10],
            "catalog_id": ["g1"],
            "image_index": [0],
            "family_id": ["1"],
            "image_label": ["a"],
            "score": [2.0],
            "alpha_norm": [2.0],
            "jacobian_norm": [0.0],
            "selected_pair": [True],
            "selected_galaxy": [True],
            "alpha_tol_arcsec": [0.1],
            "jacobian_tol": [0.5],
            "jacobian_weight": [1.0],
            "threshold_score": [1.0],
        }
    ).to_csv(tables_dir / "perturbation_discovery_diagnostics.csv", index=False)

    run_summary = plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True),
    )

    assert (tmp_path / "tables" / "image_fit_quality.csv").exists()
    assert (tmp_path / "tables" / "image_count_recovery.csv").exists()
    assert (tmp_path / "tables" / "image_recovery_extra_images.csv").exists()
    assert (tmp_path / "tables" / "model_magnification.csv").exists()
    assert (tmp_path / "tables" / "subhalo_properties.csv").exists()
    assert (tmp_path / "tables" / "run_summary.txt").exists()
    assert run_summary == {"ok": True, "image_recovery_stage": "complete"}
    assert json.loads((tmp_path / "tables" / "run_summary.json").read_text(encoding="utf-8")) == run_summary
    assert "numpyro_model" not in captured_tasks
    assert "image_recovery" in captured_tasks
    assert "image_count_recovery" in captured_tasks
    assert "model_magnification" in captured_tasks
    assert "normalized_image_residuals" in captured_tasks
    assert "image_residual_histogram" in captured_tasks
    assert "residual_vs_magnification" in captured_tasks
    assert "residual_geometry_trends" in captured_tasks
    assert "perturbation_discovery_diagnostics" in captured_tasks
    assert "subhalo_mass_function" in captured_tasks
    assert "subhalo_radial_distribution" in captured_tasks
    assert "critical_arc_support_histogram" not in captured_tasks
    assert "critical_arc_support_phase_space" not in captured_tasks
    assert "critical_arc_recovery_by_family" not in captured_tasks
    assert "kappa_comparison" not in captured_tasks
    assert "kappa_truth_diagnostics" not in captured_tasks
    assert "kappa_recovery" not in captured_tasks
    assert "mu_truth_diagnostics" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks
    assert captured_stages == ["run_diagnostics", "image_recovery", "truth_recovery"]

    captured_tasks.clear()
    captured_stages.clear()
    evaluator.sample_likelihood_mode = "local-jacobian"
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True),
    )
    assert "numpyro_model" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks
    assert "critical_arc_support_histogram" not in captured_tasks
    assert "critical_arc_support_phase_space" not in captured_tasks
    assert "critical_arc_recovery_by_family" not in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    evaluator.sample_likelihood_mode = "critical-arc-mixture-image-plane"
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True),
    )
    assert "numpyro_model" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks
    assert "critical_arc_support_histogram" in captured_tasks
    assert "critical_arc_support_phase_space" in captured_tasks
    assert "critical_arc_recovery_by_family" in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    evaluator.sample_likelihood_mode = "source"
    state.fit_mode = "large-only"
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
        ),
    )
    assert "numpyro_model" not in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            kappa_true_fits="data/ff_sims/published/hera/kappa_z9_0.fits",
            caustic_source_redshift=9.0,
        ),
    )
    assert "numpyro_model" not in captured_tasks
    assert "kappa_truth_diagnostics" in captured_tasks
    assert "kappa_comparison" not in captured_tasks
    assert "kappa_recovery" not in captured_tasks
    assert "mu_truth_diagnostics" not in captured_tasks
    assert "truth_recovery_grids" in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
            kappa_true_fits="data/ff_sims/published/hera/kappa_z9_0.fits",
            gammax_true_fits="data/ff_sims/published/hera/gammax_z9_0.fits",
            gammay_true_fits="data/ff_sims/published/hera/gammay_z9_0.fits",
        ),
    )
    assert "numpyro_model" not in captured_tasks
    assert "kappa_truth_diagnostics" in captured_tasks
    assert "mu_truth_diagnostics" in captured_tasks
    assert "truth_recovery_grids" in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            quick_diagnostics=True,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
            kappa_true_fits="data/ff_sims/published/hera/kappa_z9_0.fits",
            gammax_true_fits="data/ff_sims/published/hera/gammax_z9_0.fits",
            gammay_true_fits="data/ff_sims/published/hera/gammay_z9_0.fits",
        ),
    )
    assert "numpyro_model" not in captured_tasks
    assert "kappa_comparison" not in captured_tasks
    assert "kappa_recovery" not in captured_tasks
    assert "kappa_truth_diagnostics" in captured_tasks
    assert "mu_truth_diagnostics" in captured_tasks
    assert "truth_recovery_grids" in captured_tasks
    assert "absolute_magnification" not in captured_tasks
    assert "caustic_overlay" in captured_tasks

    captured_tasks.clear()
    captured_stages.clear()
    plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=0.0),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(
            quiet=True,
            plot_numpyro_model=True,
            caustic_plot_grid_scale_arcsec=0.2,
            caustic_source_redshift=9.0,
        ),
    )
    assert "numpyro_model" in captured_tasks
    assert "absolute_magnification" not in captured_tasks


def test_generate_plots_and_tables_stage0_minimal_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    captured_tasks: list[str] = []
    scaling_relation_df = pd.DataFrame(
        {
            "potfile_id": ["members"],
            "catalog_id": ["1"],
            "scaling_relation_class": ["inactive"],
            "catalog_mag": [20.0],
            "catalog_color": [1.2],
        }
    )
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "component_index": [10],
            "evaluation_component_index": [11],
            "catalog_id": ["g1"],
            "image_index": [0],
            "family_id": ["1"],
            "image_label": ["a"],
            "score": [2.0],
            "alpha_norm": [2.0],
            "jacobian_norm": [0.0],
            "selected_pair": [True],
            "selected_galaxy": [True],
            "alpha_tol_arcsec": [0.1],
            "jacobian_tol": [0.5],
            "jacobian_weight": [1.0],
            "threshold_score": [1.0],
        }
    ).to_csv(tables_dir / "perturbation_discovery_diagnostics.csv", index=False)

    def fail_if_called(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("stage0 minimal outputs should not build validation or recovery tables")

    monkeypatch.setattr(plotting, "_fit_quality_tables", fail_if_called)
    monkeypatch.setattr(plotting, "_image_count_recovery_table", fail_if_called)
    monkeypatch.setattr(plotting, "_subhalo_properties_table", fail_if_called)
    monkeypatch.setattr(plotting, "_summary_table", fail_if_called)
    monkeypatch.setattr(plotting, "_precompute_truth_recovery_grids", fail_if_called)
    monkeypatch.setattr(plotting, "_independent_scaling_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(plotting, "_scaling_relation_summary_table", lambda *_args, **_kwargs: scaling_relation_df)
    monkeypatch.setattr(plotting, "_run_summary", lambda *_args, **_kwargs: {"run_name": "stage0"})

    def capture_tasks(_args: argparse.Namespace, plot_tasks: list[plotting.PlotTask]) -> None:
        captured_tasks.extend(task[0] for task in plot_tasks)

    def fail_staged_runner(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("stage0 minimal outputs should not enter staged plot generation")

    monkeypatch.setattr(plotting, "_run_plot_tasks_with_progress", capture_tasks)
    monkeypatch.setattr(plotting, "_run_plot_stages_with_progress", fail_staged_runner)
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[],
        fit_mode="joint",
        perturbation_discovery_stage0=True,
    )
    evaluator = SimpleNamespace(scaling_rank_df=pd.DataFrame())
    results = PosteriorResults(
        samples=np.empty((1, 0), dtype=float),
        log_prob=np.asarray([0.0]),
        accept_prob=np.asarray([1.0]),
        diverging=np.asarray([False]),
        num_steps=np.asarray([1.0]),
        warmup_steps=0,
        sample_steps=1,
        num_chains=1,
    )

    run_summary = plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=float("nan")),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True),
    )

    assert captured_tasks == ["scaling_relation_summary", "perturbation_discovery_diagnostics"]
    assert run_summary["stage0_minimal_outputs"] is True
    assert (tables_dir / "scaling_relation_summary.csv").exists()
    assert (tables_dir / "run_summary.json").exists()
    assert not (tables_dir / "image_fit_quality.csv").exists()
    assert not (tables_dir / "image_count_recovery.csv").exists()
    assert not (tables_dir / "image_recovery_extra_images.csv").exists()
    assert not (tables_dir / "truth_recovery_summary.csv").exists()
    assert not list((tmp_path / "fits").glob("truth_recovery_*"))
    assert not (tmp_path / "corner.pdf").exists()
    assert not (tmp_path / "image_recovery.pdf").exists()
    assert not (tmp_path / "truth_recovery_kappa_model.pdf").exists()


def test_generate_plots_and_tables_stage0_minimal_outputs_from_args_flag(tmp_path: Path, monkeypatch: Any) -> None:
    captured_tasks: list[str] = []
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    scaling_relation_df = pd.DataFrame(
        {
            "potfile_id": ["members"],
            "catalog_id": ["1"],
            "scaling_relation_class": ["inactive"],
            "catalog_mag": [20.0],
            "catalog_color": [1.2],
        }
    )

    def fail_if_called(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise AssertionError("args-marked stage0 should not build full plot-stage outputs")

    monkeypatch.setattr(plotting, "_summary_table", fail_if_called)
    monkeypatch.setattr(plotting, "_fit_quality_tables", fail_if_called)
    monkeypatch.setattr(plotting, "_precompute_truth_recovery_grids", fail_if_called)
    monkeypatch.setattr(plotting, "_independent_scaling_diagnostics_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(plotting, "_scaling_relation_summary_table", lambda *_args, **_kwargs: scaling_relation_df)
    monkeypatch.setattr(plotting, "_run_summary", lambda *_args, **_kwargs: {"run_name": "stage0"})
    monkeypatch.setattr(
        plotting,
        "_run_plot_tasks_with_progress",
        lambda _args, plot_tasks: captured_tasks.extend(task[0] for task in plot_tasks),
    )
    monkeypatch.setattr(
        plotting,
        "_run_plot_stages_with_progress",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("args-marked stage0 should not enter staged plot generation")
        ),
    )
    state = SimpleNamespace(
        parameter_specs=[],
        family_data=[],
        fit_mode="joint",
        perturbation_discovery_stage0=False,
    )
    evaluator = SimpleNamespace(scaling_rank_df=pd.DataFrame())
    results = PosteriorResults(
        samples=np.empty((1, 0), dtype=float),
        log_prob=np.asarray([0.0]),
        accept_prob=np.asarray([1.0]),
        diverging=np.asarray([False]),
        num_steps=np.asarray([1.0]),
        warmup_steps=0,
        sample_steps=1,
        num_chains=1,
    )

    run_summary = plotting._generate_plots_and_tables(
        run_dir=tmp_path,
        state=state,
        evaluator=evaluator,
        best_fit=np.empty((0,), dtype=float),
        best_eval=SimpleNamespace(loglike=float("nan")),
        results=results,
        runtime_sec=0.0,
        args=argparse.Namespace(quiet=True, perturbation_discovery_stage0=True),
    )

    assert captured_tasks == ["scaling_relation_summary"]
    assert run_summary["stage0_minimal_outputs"] is True
    assert not (tables_dir / "image_fit_quality.csv").exists()
    assert not (tables_dir / "truth_recovery_summary.csv").exists()


def test_compact_numpyro_model_graph_groups_sample_sites_by_role() -> None:
    specs = [
        ParameterSpec("large.x", "large_x", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1, component_family="large"),
        ParameterSpec(
            "scaling.sigma",
            "scaling_sigma",
            "mock",
            81,
            "sigma",
            "uniform",
            0.0,
            10.0,
            0.1,
            component_family="scaling",
            sample_site_name="scaling_vector",
            sample_site_index=0,
        ),
        ParameterSpec(
            "scaling.cut",
            "scaling_cut",
            "mock",
            81,
            "cut",
            "uniform",
            0.0,
            10.0,
            0.1,
            component_family="scaling",
            sample_site_name="scaling_vector",
            sample_site_index=1,
        ),
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
            component_family="source_position",
        ),
        ParameterSpec("image.sigma", "image_sigma", "image", 0, "sigma", "uniform", 0.0, 1.0, 0.1, component_family="image_scatter"),
        ParameterSpec(
            "critical.singular_threshold",
            "critical_arc_singular_threshold",
            "critical_arc",
            0,
            "singular_threshold",
            "uniform",
            0.05,
            0.5,
            0.01,
            component_family="critical_arc_hyperparameter",
        ),
    ]
    sample_sites = [
        SimpleNamespace(name="large_x_site", indices=(0,)),
        SimpleNamespace(name="scaling_vector_site", indices=(1, 2)),
        SimpleNamespace(name="source_position_site", indices=(3,)),
        SimpleNamespace(name="image_scatter_site", indices=(4,)),
        SimpleNamespace(name="critical_arc_threshold_site", indices=(5,)),
    ]

    graph = plotting._build_compact_numpyro_model_graph(
        state=SimpleNamespace(family_data=[SimpleNamespace(n_images=2), SimpleNamespace(n_images=3)]),
        parameter_specs=specs,
        sample_sites=sample_sites,
        sample_likelihood_mode="critical-arc-mixture-image-plane",
    )
    source = graph.source

    assert "Critical-arc-mixture image-plane NumPyro model" not in source
    assert "Generated from numpyro.render_model" not in source
    assert "render_model trace" not in source
    assert "Critical-arc gate" in source
    assert "Intrinsic image scatter" in source
    assert "Source positions" in source
    assert "Member scaling law" in source
    assert "Large-scale lens" in source
    assert "2 parameters / 1 site" in source
    assert "ln ℒ(η, {β_f})" in source
    assert "critical-arc image-plane likelihood" in source
    assert "θ" in source
    assert "latent parameter vector" in source
    assert "theta = stack(sample sites)" not in source
    assert "numpyro.factor" not in source
    assert "Observed image positions" in source
    assert "5 positions" in source
    assert "large_x_site" not in source
    assert "scaling_vector_site" not in source
    assert "critical_arc_threshold_site" not in source


def test_plot_numpyro_model_writes_fixed_pdf_and_no_full_artifact(tmp_path: Path, monkeypatch: Any) -> None:
    specs = [
        ParameterSpec(
            "critical.singular_threshold",
            "critical_arc_singular_threshold",
            "critical_arc",
            0,
            "singular_threshold",
            "uniform",
            0.05,
            0.5,
            0.01,
            component_family="critical_arc_hyperparameter",
        )
    ]
    render_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []

    class CompactGraph:
        def render(self, filename: str, *, format: str, cleanup: bool) -> str:
            render_calls.append({"filename": filename, "format": format, "cleanup": cleanup})
            path = Path(f"{filename}.{format}")
            path.write_bytes(b"%PDF-1.4\n")
            return str(path)

    def fake_build_compact_graph(**kwargs: Any) -> CompactGraph:
        build_calls.append(kwargs)
        return CompactGraph()

    monkeypatch.setattr(plotting, "_parameter_sample_sites_for_rendering", lambda _specs: [SimpleNamespace(name="p", indices=(0,))])
    monkeypatch.setattr(plotting, "_build_compact_numpyro_model_graph", fake_build_compact_graph)

    output = plotting._plot_numpyro_model(
        tmp_path,
        SimpleNamespace(parameter_specs=specs),
        SimpleNamespace(sample_likelihood_mode="critical-arc-mixture-image-plane"),
        argparse.Namespace(),
    )

    assert output == tmp_path / "numpyro_model.pdf"
    assert (tmp_path / "numpyro_model.pdf").exists()
    assert render_calls == [
        {
            "filename": str(tmp_path / "numpyro_model"),
            "format": "pdf",
            "cleanup": True,
        }
    ]
    assert build_calls[0]["parameter_specs"] == specs
    assert build_calls[0]["sample_likelihood_mode"] == "critical-arc-mixture-image-plane"
    assert "exact_graph" not in build_calls[0]
    assert not list(tmp_path.glob("*_full.pdf"))
    assert not list(tmp_path.glob("numpyro_critical_arc_mixture_image_plane*.pdf"))


def test_subhalo_properties_table_uses_all_potfile_members_and_mass_radii(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SimpleNamespace(
        z_lens=0.4,
        cosmo_config={},
        parameter_specs=[],
        packed_lens_spec=SimpleNamespace(
            component_family=np.asarray([0, 1, 2, 1], dtype=int),
            x_center_base=np.asarray([0.0, 3.0, 30.0, 6.0], dtype=float),
            y_center_base=np.asarray([0.0, 4.0, 40.0, 8.0], dtype=float),
        ),
        scaling_component_records=[
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 1,
                "free_component_index": 2,
                "catalog_id": "member001",
                "catalog_mag": 20.0,
                "x_centre": 3.0,
                "y_centre": 4.0,
            },
            {
                "potfile_id": "members",
                "potfile_order": 0,
                "component_index": 3,
                "free_component_index": -1,
                "catalog_id": "member002",
                "catalog_mag": 21.0,
                "x_centre": 6.0,
                "y_centre": 8.0,
            },
        ],
    )

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[float, int]] = []

        def mass_3d(self, radius: float, kwargs_lens: list[dict[str, float]], bool_list: list[bool]) -> float:
            component_index = bool_list.index(True)
            self.calls.append((float(radius), component_index))
            return float(kwargs_lens[component_index]["sigma0"]) * float(radius)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.model = FakeModel()
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            assert packed_state == {"latent": 5.0}
            return [
                {"sigma0": 1.0, "Ra": 0.1, "Rs": 1.0},
                {"sigma0": 0.0, "Ra": 0.2, "Rs": 2.0},
                {"sigma0": 5.0, "Ra": 0.3, "Rs": 4.0},
                {"sigma0": 7.0, "Ra": 0.4, "Rs": 8.0},
            ]

    monkeypatch.setattr(plotting, "critical_surface_density_angle_from_config", lambda *_args, **_kwargs: 10.0)
    evaluator = FakeEvaluator()

    table = plotting._subhalo_properties_table(
        state,
        evaluator,
        np.asarray([4.0], dtype=float),
        caustic_source_redshift=9.0,
    )

    assert table["component_index"].tolist() == [1, 3]
    assert table["model_component_index"].tolist() == [2, 3]
    assert table["catalog_id"].tolist() == ["member001", "member002"]
    assert table["radius_arcsec"].tolist() == pytest.approx([5.0, 10.0])
    assert table["Rs"].tolist() == pytest.approx([4.0, 8.0])
    assert table["sigma0"].tolist() == pytest.approx([5.0, 7.0])
    assert table["mass_within_Rs_msun"].tolist() == pytest.approx([200.0, 560.0])
    assert table["mass_within_1e6_Rs_msun"].tolist() == pytest.approx([200.0e6, 560.0e6])
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert evaluator.model.calls == [
        (4.0, 2),
        (4.0e6, 2),
        (8.0, 3),
        (8.0e6, 3),
    ]


def test_subhalo_distribution_plots_write_pdfs_and_mass_function_has_subhalo_mass_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    subhalo_df = pd.DataFrame(
        {
            "mass_within_Rs_msun": [1.0e10, 3.0e10, 1.0e11],
            "mass_within_1e6_Rs_msun": [2.0e11, 4.0e11, 1.2e12],
            "radius_arcsec": [5.0, 15.0, 30.0],
        }
    )
    hist_calls: list[dict[str, Any]] = []
    original_hist = Axes.hist

    def record_hist(self: Axes, values: Any, *args: Any, **kwargs: Any) -> Any:
        hist_calls.append(
            {
                "values": np.asarray(values, dtype=float).copy(),
                "label": kwargs.get("label"),
                "histtype": kwargs.get("histtype"),
            }
        )
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", record_hist)

    mass_path = tmp_path / "subhalo_mass_function.pdf"
    radial_path = tmp_path / "subhalo_radial_distribution.pdf"
    plotting._plot_subhalo_mass_function(subhalo_df, mass_path)
    plotting._plot_subhalo_radial_distribution(subhalo_df, radial_path)

    assert mass_path.exists()
    assert mass_path.stat().st_size > 0
    assert radial_path.exists()
    assert radial_path.stat().st_size > 0
    mass_hist_calls = [call for call in hist_calls if call["label"] is not None]
    assert [call["label"] for call in mass_hist_calls] == ["Subhalo Mass"]
    assert [call["histtype"] for call in mass_hist_calls] == ["step"]
    np.testing.assert_allclose(mass_hist_calls[0]["values"], np.log10(subhalo_df["mass_within_1e6_Rs_msun"]))


def test_subhalo_distribution_plots_write_placeholders_without_finite_values(tmp_path: Path) -> None:
    subhalo_df = pd.DataFrame(
        {
            "mass_within_Rs_msun": [np.nan, np.inf],
            "mass_within_1e6_Rs_msun": [np.nan, -np.inf],
            "radius_arcsec": [np.nan, np.inf],
        }
    )
    mass_path = tmp_path / "subhalo_mass_function.pdf"
    radial_path = tmp_path / "subhalo_radial_distribution.pdf"

    plotting._plot_subhalo_mass_function(subhalo_df, mass_path)
    plotting._plot_subhalo_radial_distribution(subhalo_df, radial_path)

    assert mass_path.exists()
    assert mass_path.stat().st_size > 0
    assert radial_path.exists()
    assert radial_path.stat().st_size > 0


def test_free_galaxy_shape_helpers_wrap_periodic_angles() -> None:
    ellipticite, angle = plotting._e1e2_to_lenstool_shape(0.2, 0.0)

    assert np.isfinite(ellipticite)
    assert ellipticite > 0.0
    assert angle == pytest.approx(0.0)
    assert plotting._wrapped_position_angle_delta_deg(179.0, -179.0) == pytest.approx(-2.0)


def test_free_galaxy_shape_comparison_table_includes_scaling_and_independent_members() -> None:
    packed = SimpleNamespace(
        profile_type=np.asarray([81, 81, 81, 81], dtype=np.int32),
        e1_base=np.asarray([0.0, 0.10, 0.10, 0.05], dtype=float),
        e2_base=np.asarray([0.0, 0.00, 0.00, 0.02], dtype=float),
        e1_param_index=np.asarray([-1, -1, 0, 2], dtype=np.int32),
        e2_param_index=np.asarray([-1, -1, 1, 3], dtype=np.int32),
    )
    state = SimpleNamespace(
        packed_lens_spec=packed,
        scaling_component_records=[
            {
                "potfile_id": "members",
                "catalog_id": "inactive",
                "component_index": 0,
                "free_component_index": -1,
                "selected_independent": False,
            },
            {
                "potfile_id": "members",
                "catalog_id": "free",
                "component_index": 1,
                "free_component_index": 2,
                "selected_independent": True,
            },
        ],
        base_components=[
            {},
            {},
            {},
            {
                "id": "independent_member_members_bcg",
                "independent_member_population_id": "members",
                "independent_member_catalog_id": "bcg",
            },
        ],
    )
    best_fit = np.asarray([0.20, 0.10, 0.08, -0.04], dtype=float)

    table = plotting._free_galaxy_shape_comparison_table(state, best_fit)

    assert table["member_type"].tolist() == ["independent_member_halo", "selected_scaling_free"]
    assert table["catalog_id"].tolist() == ["bcg", "free"]
    assert "inactive" not in table["catalog_id"].tolist()
    free_row = table.loc[table["catalog_id"] == "free"].iloc[0]
    assert int(free_row["catalog_component_index"]) == 1
    assert int(free_row["model_component_index"]) == 2
    assert free_row["model_ellipticite"] != pytest.approx(free_row["catalog_ellipticite"])
    independent_row = table.loc[table["catalog_id"] == "bcg"].iloc[0]
    assert int(independent_row["catalog_component_index"]) == 3
    assert int(independent_row["model_component_index"]) == 3


def test_free_galaxy_shape_comparison_plot_writes_pdf_and_distinct_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    scatter_markers: list[str | None] = []
    original_scatter = Axes.scatter

    def record_scatter(self: Axes, *args: Any, **kwargs: Any) -> Any:
        scatter_markers.append(kwargs.get("marker"))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    shape_df = pd.DataFrame(
        {
            "member_type": ["selected_scaling_free", "independent_member_halo"],
            "potfile_id": ["members", "members"],
            "catalog_id": ["free", "bcg"],
            "component_index": [1, 3],
            "catalog_component_index": [1, 3],
            "model_component_index": [2, 3],
            "catalog_ellipticite": [0.10, 0.20],
            "model_ellipticite": [0.30, 0.25],
            "delta_ellipticite": [0.20, 0.05],
            "catalog_angle_pos_deg": [10.0, -20.0],
            "model_angle_pos_deg": [15.0, -10.0],
            "delta_angle_pos_deg": [5.0, 10.0],
        }
    )
    output = tmp_path / "free_galaxy_shape_comparison.pdf"

    plotting._plot_free_galaxy_shape_comparison(shape_df, output)

    assert output.exists()
    assert output.stat().st_size > 0
    assert "D" in scatter_markers
    assert "s" in scatter_markers


def test_free_galaxy_shape_comparison_plot_writes_placeholder_for_empty_table(tmp_path: Path) -> None:
    output = tmp_path / "free_galaxy_shape_comparison.pdf"

    plotting._plot_free_galaxy_shape_comparison(pd.DataFrame(), output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_run_summary_quality_metrics_from_image_fit_quality() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=4)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, 1.0, 10.0],
            "y_model_arcsec": [0.0, 2.0, 1.0, 10.0],
            "sigma_arcsec": [1.0, 1.0, 1.0, 1.0],
            "image_sigma_eff_arcsec": [10.0, 10.0, 10.0, 10.0],
            "image_recovery_status": ["recovered", "recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "point_recovered", "not_recovered"],
            "arc_supported": [False, False, False, False],
            "exact_image_prediction_failed": [False, False, False, True],
            "covered_xy_1sigma": [True, False, True, False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square"] == pytest.approx(0.07)
    assert summary["headline_point_image_count"] == 3
    assert summary["headline_missing_image_count"] == 1
    assert summary["point_recovered_image_count"] == 3
    assert summary["point_image_rms_arcsec"] == pytest.approx(math.sqrt(7.0 / 3.0))
    assert summary["point_image_median_residual_arcsec"] == pytest.approx(math.sqrt(2.0))
    assert summary["image_residual_median_arcsec"] == pytest.approx(math.sqrt(2.0))
    assert summary["headline_n_data"] == 6
    assert summary["headline_dof"] == 3
    assert summary["headline_reduced_chi_square"] == pytest.approx(0.07 / 3.0)
    assert summary["arc_aware_chi_square"] == pytest.approx(0.07)
    assert summary["arc_aware_point_image_count"] == 3
    assert summary["arc_aware_arc_supported_image_count"] == 0
    assert summary["arc_aware_recovered_image_count"] == 3
    assert summary["arc_aware_missing_image_count"] == 1
    assert summary["arc_aware_image_residual_median_arcsec"] == pytest.approx(math.sqrt(2.0))
    assert summary["arc_aware_n_data"] == 6
    assert summary["arc_aware_dof"] == 3
    assert summary["arc_aware_reduced_chi_square"] == pytest.approx(0.07 / 3.0)
    assert summary["n_effective_parameters"] == 3
    assert summary["sampled_non_source_position_parameters"] == 1
    assert summary["source_position_parameters"] == 2
    assert summary["chi_square_sigma_basis"] == "image_sigma_eff_arcsec"
    assert summary["chi_square_sigma_eff_median_arcsec"] == pytest.approx(10.0)
    assert summary["chi_square_sigma_eff_min_arcsec"] == pytest.approx(10.0)
    assert summary["chi_square_sigma_eff_max_arcsec"] == pytest.approx(10.0)
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(7.0 / 3.0))
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(7.0 / 3.0))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["chi_square_red1_calibration_note"] == "post-fit diagnostic; holds image_sigma_int fixed"
    assert summary["covered_xy_1sigma_fraction"] == pytest.approx(2.0 / 3.0)
    assert "chi_square" not in summary
    assert "reduced_chi_square" not in summary
    assert "diagnostic_n_data" not in summary
    assert "diagnostic_dof" not in summary
    assert "aic" not in summary
    assert "bic" not in summary


def test_run_summary_arc_aware_chi_square_counts_arc_supported_rows() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=4)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, np.nan, np.nan],
            "y_model_arcsec": [0.0, 2.0, np.nan, np.nan],
            "sigma_arcsec": [1.0, 2.0, 0.5, 1.0],
            "image_sigma_eff_arcsec": [1.0, 4.0, 0.25, 100.0],
            "image_recovery_status": ["recovered", "recovered", "not_recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "arc_supported", "not_recovered"],
            "arc_supported": [False, False, True, False],
            "arc_aware_image_residual_arcsec": [1.0, 2.0, 0.5, np.nan],
            "exact_image_prediction_failed": [False, False, True, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square"] == pytest.approx(1.25)
    assert summary["headline_n_data"] == 4
    assert summary["headline_dof"] == 1
    assert summary["headline_reduced_chi_square"] == pytest.approx(1.25)
    assert summary["point_recovered_image_count"] == 2
    assert summary["point_image_rms_arcsec"] == pytest.approx(math.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert summary["point_image_median_residual_arcsec"] == pytest.approx(1.5)
    assert summary["headline_missing_image_count"] == 2
    assert summary["arc_aware_chi_square"] == pytest.approx(5.25)
    assert summary["arc_aware_n_data"] == 5
    assert summary["arc_aware_dof"] == 2
    assert summary["arc_aware_reduced_chi_square"] == pytest.approx(2.625)
    assert summary["arc_aware_point_image_count"] == 2
    assert summary["arc_aware_arc_supported_image_count"] == 1
    assert summary["arc_aware_recovered_image_count"] == 3
    assert summary["arc_aware_missing_image_count"] == 1
    assert summary["arc_aware_image_residual_median_arcsec"] == pytest.approx(1.0)
    assert summary["chi_square_sigma_eff_median_arcsec"] == pytest.approx(1.0)
    assert summary["chi_square_sigma_eff_min_arcsec"] == pytest.approx(0.25)
    assert summary["chi_square_sigma_eff_max_arcsec"] == pytest.approx(4.0)
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(5.0))
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(5.25 / 2.0))


def test_run_summary_arc_aware_keeps_point_recovered_rows_point_first() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, np.nan],
            "y_model_arcsec": [0.0, 2.0, np.nan],
            "sigma_arcsec": [1.0, 1.0, 1.0],
            "image_sigma_eff_arcsec": [1.0, 1.0, 1.0],
            "image_recovery_status": ["recovered", "recovered", "not_recovered"],
            "preferred_recovery_status": ["point_recovered", "point_recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "not_recovered"],
            "arc_supported": [False, False, False],
            "point_image_residual_arcsec": [1.0, 2.0, np.nan],
            "arc_candidate_supported": [True, False, False],
            "arc_candidate_image_residual_arcsec": [0.2, np.nan, np.nan],
            "preferred_image_residual_arcsec": [1.0, 2.0, np.nan],
            "arc_aware_image_residual_arcsec": [1.0, 2.0, np.nan],
            "exact_image_prediction_failed": [False, False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square"] == pytest.approx(5.0)
    assert summary["headline_point_image_count"] == 2
    assert summary["headline_missing_image_count"] == 1
    assert summary["point_recovered_image_count"] == 2
    assert summary["point_image_rms_arcsec"] == pytest.approx(math.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert summary["arc_aware_chi_square"] == pytest.approx(5.0)
    assert summary["arc_aware_point_image_count"] == 2
    assert summary["arc_aware_arc_supported_image_count"] == 0
    assert summary["arc_aware_recovered_image_count"] == 2
    assert summary["arc_aware_missing_image_count"] == 1
    assert summary["arc_aware_n_data"] == 4
    assert summary["arc_aware_valid_image_count"] == 2
    assert summary["arc_aware_image_rms_arcsec"] == pytest.approx(math.sqrt((1.0**2 + 2.0**2) / 2.0))


def test_run_summary_point_only_ignores_arc_aware_columns() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[SimpleNamespace(family_id="1", n_images=3)])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 0.0, np.nan],
            "y_model_arcsec": [0.0, 2.0, np.nan],
            "sigma_arcsec": [1.0, 1.0, 1.0],
            "image_sigma_eff_arcsec": [1.0, 1.0, 1.0],
            "image_recovery_status": ["recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "arc_supported"],
            "arc_supported": [False, True, True],
            "arc_aware_image_residual_arcsec": [1.0, 0.1, 0.2],
            "exact_image_prediction_failed": [False, False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state)

    assert summary["headline_chi_square"] == pytest.approx(5.0)
    assert summary["headline_n_data"] == 4
    assert summary["point_recovered_image_count"] == 2
    assert summary["point_image_rms_arcsec"] == pytest.approx(math.sqrt((1.0**2 + 2.0**2) / 2.0))
    assert summary["arc_aware_chi_square"] is None
    assert summary["arc_aware_n_data"] == 0
    assert summary["arc_aware_recovered_image_count"] == 0
    assert summary["arc_aware_image_rms_arcsec"] is None


def test_run_summary_chi_square_requires_image_sigma_eff_arcsec() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=1)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "x_model_arcsec": [1.0],
            "y_model_arcsec": [0.0],
            "sigma_arcsec": [1.0],
            "image_recovery_status": ["recovered"],
            "arc_recovery_status": ["point_recovered"],
            "exact_image_prediction_failed": [False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square"] is None
    assert summary["headline_n_data"] == 0
    assert summary["arc_aware_chi_square"] is None
    assert summary["arc_aware_n_data"] == 0
    assert summary["chi_square_sigma_basis"] == "image_sigma_eff_arcsec"
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] is None


def test_run_summary_chi_square_excludes_invalid_image_sigma_eff_rows() -> None:
    state = SimpleNamespace(
        parameter_specs=[ParameterSpec("p", "p", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1)],
        family_data=[SimpleNamespace(family_id="1", n_images=3)],
    )
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0, 0.0],
            "x_model_arcsec": [1.0, 2.0, np.nan],
            "y_model_arcsec": [0.0, 0.0, np.nan],
            "sigma_arcsec": [0.1, 0.1, 0.1],
            "image_sigma_eff_arcsec": [0.5, 0.0, np.nan],
            "image_recovery_status": ["recovered", "recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "point_recovered", "arc_supported"],
            "arc_supported": [False, False, True],
            "arc_aware_image_residual_arcsec": [1.0, 2.0, 0.5],
            "exact_image_prediction_failed": [False, False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square"] == pytest.approx(4.0)
    assert summary["headline_point_image_count"] == 1
    assert summary["headline_missing_image_count"] == 2
    assert summary["headline_n_data"] == 2
    assert summary["arc_aware_chi_square"] == pytest.approx(4.0)
    assert summary["arc_aware_arc_supported_image_count"] == 0
    assert summary["arc_aware_missing_image_count"] == 2
    assert summary["arc_aware_n_data"] == 2
    assert summary["headline_chi_square_red1_total_sigma_arcsec"] is None
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] is None


def test_run_summary_chi_square_red1_calibration_solves_pos_sigma_with_intrinsic_scatter() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 0.0],
            "y_obs_arcsec": [0.0, 0.0],
            "x_model_arcsec": [2.0, np.nan],
            "y_model_arcsec": [0.0, np.nan],
            "sigma_arcsec": [1.0, 1.0],
            "image_sigma_int_arcsec": [1.0, 1.0],
            "image_sigma_eff_arcsec": [math.sqrt(2.0), math.sqrt(2.0)],
            "image_recovery_status": ["recovered", "not_recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported"],
            "arc_supported": [False, True],
            "arc_aware_image_residual_arcsec": [2.0, math.sqrt(5.0)],
            "exact_image_prediction_failed": [False, True],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(2.0))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(1.0)
    assert summary["arc_aware_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(3.0))
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(math.sqrt(2.0))


def test_run_summary_chi_square_red1_pos_sigma_zero_when_intrinsic_scatter_is_sufficient() -> None:
    state = SimpleNamespace(parameter_specs=[], family_data=[])
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0],
            "y_obs_arcsec": [0.0],
            "x_model_arcsec": [1.0],
            "y_model_arcsec": [0.0],
            "sigma_arcsec": [0.1],
            "image_sigma_int_arcsec": [1.0],
            "image_sigma_eff_arcsec": [math.sqrt(1.01)],
            "image_recovery_status": ["recovered"],
            "arc_recovery_status": ["point_recovered"],
            "arc_supported": [False],
            "exact_image_prediction_failed": [False],
        }
    )

    summary = plotting._fit_quality_chi_square_summary(image_df, state, use_arc_aware_diagnostics=True)

    assert summary["headline_chi_square_red1_total_sigma_arcsec"] == pytest.approx(math.sqrt(0.5))
    assert summary["headline_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(0.0)
    assert summary["arc_aware_chi_square_red1_pos_sigma_arcsec"] == pytest.approx(0.0)


def test_run_summary_effective_parameter_count_does_not_double_count_explicit_sources() -> None:
    specs = [
        ParameterSpec("halo.x", "halo_x", "mock", 81, "x", "uniform", -1.0, 1.0, 0.1),
        ParameterSpec(
            "source.1.beta_x",
            "source_1_beta_x",
            "1",
            0,
            "beta_x",
            "uniform",
            -1.0,
            1.0,
            0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            "source.1.beta_y",
            "source_1_beta_y",
            "1",
            0,
            "beta_y",
            "uniform",
            -1.0,
            1.0,
            0.1,
            component_family="source_position",
        ),
        ParameterSpec(
            "image.sigma_int",
            "image_sigma_int",
            "image",
            0,
            "sigma_int",
            "uniform",
            0.0,
            1.0,
            0.1,
            component_family="image_scatter",
        ),
    ]
    state = SimpleNamespace(
        parameter_specs=specs,
        family_data=[
            SimpleNamespace(family_id="1", n_images=0),
            SimpleNamespace(family_id="2", n_images=0),
        ],
    )

    summary = plotting._fit_quality_chi_square_summary(None, state)

    assert summary["sampled_non_source_position_parameters"] == 2
    assert summary["source_position_parameters"] == 4
    assert summary["n_effective_parameters"] == 6
    assert summary["point_image_median_residual_arcsec"] is None
    assert summary["arc_aware_image_residual_median_arcsec"] is None


def test_image_count_recovery_table_and_plot_write_pdf(tmp_path: Path) -> None:
    state = SimpleNamespace(
        family_data=[
            SimpleNamespace(family_id="1", n_images=3, z_source=2.0, effective_z_source=2.0),
            SimpleNamespace(family_id="2", n_images=2, z_source=3.0, effective_z_source=3.0),
        ]
    )
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "1", "2", "2"],
            "z_source": [2.0, 2.0, 2.0, 3.0, 3.0],
            "model_produced_image_count": [4, 4, 4, 1, 1],
            "model_recovered_image_count": [3, 3, 3, 1, 1],
            "model_missing_image_count": [0, 0, 0, 1, 1],
            "model_extra_image_count": [1, 1, 1, 0, 0],
            "model_multiplicity_failed": [True, True, True, True, True],
            "model_multiplicity_failure_reason": ["extra_model_images"] * 3 + ["missing_model_images"] * 2,
        }
    )

    count_df = plotting._image_count_recovery_table(state, image_df)
    plotting._plot_image_count_recovery(count_df, tmp_path / "image_count_recovery.pdf")
    summary = plotting._image_count_recovery_summary(count_df)

    assert count_df["family_id"].tolist() == ["1", "2"]
    assert count_df.set_index("family_id").loc["1", "produced_image_count"] == 4
    assert count_df.set_index("family_id").loc["2", "missing_image_count"] == 1
    assert summary["model_recovered_image_count"] == 4
    assert summary["model_produced_image_count"] == 5
    assert summary["model_missing_image_count"] == 1
    assert summary["model_extra_image_count"] == 1
    assert (tmp_path / "image_count_recovery.pdf").exists()


def test_chain_diagnostics_summary_uses_grouped_samples() -> None:
    specs = _corner_test_specs()[:2]
    grouped = np.asarray(
        [
            [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4], [0.5, 1.5]],
            [[0.05, 1.05], [0.15, 1.15], [0.25, 1.25], [0.35, 1.35], [0.45, 1.45], [0.55, 1.55]],
        ],
        dtype=float,
    )
    posterior = PosteriorResults(
        samples=grouped.reshape((-1, 2)),
        log_prob=np.zeros(12, dtype=float),
        accept_prob=np.ones(12, dtype=float),
        diverging=np.zeros(12, dtype=bool),
        num_steps=np.ones(12, dtype=float),
        warmup_steps=0,
        sample_steps=6,
        num_chains=2,
        grouped_samples=grouped,
    )

    summary = plotting._chain_diagnostics_summary(posterior, specs)

    assert summary["ess_min"] is not None
    assert summary["ess_median"] is not None
    assert summary["rhat_max"] is not None
    assert summary["rhat_median"] is not None
    assert summary["ess_worst_parameter"] in {spec.name for spec in specs}
    assert summary["rhat_worst_parameter"] in {spec.name for spec in specs}


def test_chain_health_summary_table_identifies_stuck_chain() -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()

    table = plotting._chain_health_summary_table(posterior, specs, max_tree_depth=8)

    assert list(table["chain"]) == [1, 2, 3, 4]
    stuck = table.iloc[0]
    assert stuck["chain_label"] == "stuck"
    assert stuck["log_prob_median"] == pytest.approx(-230.0)
    assert stuck["log_prob_median"] < table.iloc[1]["log_prob_median"]
    assert stuck["max_tree_depth_saturation_fraction"] == pytest.approx(1.0)
    assert stuck["image_sigma_int_q50"] == pytest.approx(1.23, abs=0.01)
    assert table.iloc[1]["image_sigma_int_q50"] == pytest.approx(0.10, abs=0.01)


def test_chain_parameter_diagnostics_table_reports_per_chain_quantiles() -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()

    table = plotting._chain_parameter_diagnostics_table(posterior, specs)
    sigma_row = table.loc[table["sample_name"] == "image_sigma_int"].iloc[0]

    assert sigma_row["parameter"] == "image.sigma_int"
    assert sigma_row["chain_1_q50"] == pytest.approx(1.23, abs=0.01)
    assert sigma_row["chain_2_q50"] == pytest.approx(0.10, abs=0.01)
    assert sigma_row["chain_median_spread"] > 1.0
    assert sigma_row["chain_median_standardized_spread"] > 1.0


def test_chain_health_plot_creates_pdf_and_skips_missing_grouped_samples(tmp_path: Path) -> None:
    posterior, specs = _synthetic_stuck_chain_posterior()

    plotting._plot_chain_health(tmp_path, posterior, specs, max_tree_depth=8)

    assert (tmp_path / "chain_health.pdf").exists()

    skip_dir = tmp_path / "skip"
    missing_grouped = PosteriorResults(
        samples=posterior.samples,
        log_prob=posterior.log_prob,
        accept_prob=posterior.accept_prob,
        diverging=posterior.diverging,
        num_steps=posterior.num_steps,
        warmup_steps=0,
        sample_steps=posterior.sample_steps,
        num_chains=posterior.num_chains,
    )
    plotting._plot_chain_health(skip_dir, missing_grouped, specs, max_tree_depth=8)

    assert not (skip_dir / "chain_health.pdf").exists()


def test_format_run_summary_text_contains_lensing_and_quality_sections() -> None:
    text = plotting._format_run_summary_text(
        {
            "run_name": "mock",
            "fit_mode": "joint",
            "sample_likelihood_mode": "critical-arc-mixture-image-plane",
            "sampler": "numpyro_nuts",
            "n_families": 2,
            "n_images": 6,
            "n_parameters": 4,
            "headline_chi_square": 12.0,
            "headline_dof": 4,
            "headline_reduced_chi_square": 3.0,
            "arc_aware_chi_square": 9.0,
            "arc_aware_dof": 5,
            "arc_aware_reduced_chi_square": 1.8,
            "observed_image_count": 6,
            "point_recovered_image_count": 4,
            "point_image_rms_arcsec": 0.42,
            "point_image_median_residual_arcsec": 0.31,
            "arc_aware_recovered_image_count": 5,
            "arc_aware_image_rms_arcsec": 0.39,
            "arc_aware_image_residual_median_arcsec": 0.28,
            "arc_aware_arc_supported_image_count": 2,
            "arc_aware_missing_image_count": 1,
            "arc_recovery_p_arc_threshold": 0.65,
            "chi_square_sigma_basis": "image_sigma_eff_arcsec",
            "chi_square_sigma_eff_median_arcsec": 0.59,
            "chi_square_sigma_eff_min_arcsec": 0.42,
            "chi_square_sigma_eff_max_arcsec": 0.71,
            "headline_chi_square_red1_total_sigma_arcsec": 0.6,
            "headline_chi_square_red1_pos_sigma_arcsec": 0.58,
            "arc_aware_chi_square_red1_total_sigma_arcsec": 0.59,
            "arc_aware_chi_square_red1_pos_sigma_arcsec": 0.57,
            "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
            "n_effective_parameters": 4,
            "fit_quality_reference_sample_kind": "max_likelihood",
            "fit_quality_reference_sample_index": 7,
            "fit_quality_reference_source_loglike": -11.0,
            "fit_quality_reference_log_prob": -12.0,
            "ess_min": 10.0,
            "rhat_max": 1.02,
            "svi_health_warnings": ["near-zero SVI guide spread"],
        }
    )

    assert "Lensing Information" in text
    assert "Quality Of Fit" in text
    assert "headline_chi_square" in text
    assert "arc_aware_chi_square" in text
    assert "point image RMS arcsec" in text
    assert "point median residual arcsec" in text
    assert "point recovered images" in text
    assert "4/6" in text
    assert "arc-aware image RMS arcsec" in text
    assert "arc-aware median residual arcsec" in text
    assert "arc-aware recovered images" in text
    assert "5/6" in text
    assert "total image-plane sigma" in text
    assert "chi-square sigma basis" in text
    assert "chi-square median sigma arcsec" in text
    assert "headline red1 total sigma arcsec" in text
    assert "headline red1 pos_sigma_arcsec" in text
    assert "arc-aware red1 total sigma arcsec" in text
    assert "arc-aware red1 pos_sigma_arcsec" in text
    assert "post-fit diagnostic; holds image_sigma_int fixed" in text
    assert "N_arc_supported" in text
    assert "N_missing" in text
    assert "fit-quality reference" in text
    assert "arc-aware recovery gate" in text
    assert "arc_recovery_p_arc_threshold=0.65" in text
    assert "diagnostic data points" not in text
    assert "diagnostic dof" not in text
    assert "AIC" not in text
    assert "BIC" not in text
    assert "Rhat max" not in text
    assert "SVI health warnings" not in text


def test_format_run_summary_text_omits_arc_aware_metrics_for_noncritical_mode() -> None:
    text = plotting._format_run_summary_text(
        {
            "run_name": "mock",
            "fit_mode": "joint",
            "sample_likelihood_mode": "source",
            "sampler": "numpyro_nuts",
            "n_families": 1,
            "n_images": 3,
            "n_parameters": 2,
            "headline_chi_square": 5.0,
            "headline_dof": 2,
            "headline_reduced_chi_square": 2.5,
            "observed_image_count": 3,
            "point_recovered_image_count": 2,
            "point_image_rms_arcsec": 0.4,
            "point_image_median_residual_arcsec": 0.3,
            "arc_aware_chi_square": 1.0,
            "arc_aware_recovered_image_count": 3,
            "arc_aware_image_rms_arcsec": 0.1,
            "arc_aware_image_residual_median_arcsec": 0.08,
            "arc_aware_arc_supported_image_count": 1,
            "arc_aware_missing_image_count": 0,
            "arc_aware_chi_square_red1_total_sigma_arcsec": 0.2,
            "arc_aware_chi_square_red1_pos_sigma_arcsec": 0.2,
            "chi_square_sigma_basis": "image_sigma_eff_arcsec",
            "chi_square_red1_calibration_note": "post-fit diagnostic; holds image_sigma_int fixed",
        }
    )

    assert "headline_chi_square" in text
    assert "point image RMS arcsec" in text
    assert "point median residual arcsec" in text
    assert "point recovered images" in text
    assert "arc_aware_chi_square" not in text
    assert "arc-aware image RMS arcsec" not in text
    assert "arc-aware median residual arcsec" not in text
    assert "arc-aware recovered images" not in text
    assert "arc-aware red1" not in text
    assert "N_arc_supported" not in text
    assert "arc-aware recovery gate" not in text


def test_format_sequential_run_summary_text_gates_arc_aware_columns() -> None:
    source_only = plotting._format_sequential_run_summary_text(
        [
            {
                "stage": "stage3",
                "sample_likelihood_mode": "source",
                "headline_chi_square": 5.0,
                "point_image_rms_arcsec": 0.6,
                "point_image_median_residual_arcsec": 0.4,
                "point_recovered_image_count": 3,
                "arc_aware_chi_square": 1.0,
                "arc_aware_image_rms_arcsec": 0.2,
                "arc_aware_image_residual_median_arcsec": 0.1,
            }
        ],
        run_name="mock",
        root_dir="results/mock",
    )

    assert "point_RMS" in source_only
    assert "point_med" in source_only
    assert "arc_RMS" not in source_only
    assert "arc_med" not in source_only
    assert "arc_chi2" not in source_only
    assert "0.2" not in source_only

    mixed = plotting._format_sequential_run_summary_text(
        [
            {
                "stage": "stage3",
                "sample_likelihood_mode": "source",
                "headline_chi_square": 5.0,
                "point_image_rms_arcsec": 0.6,
                "point_image_median_residual_arcsec": 0.4,
                "point_recovered_image_count": 3,
                "arc_aware_chi_square": 99.0,
                "arc_aware_image_rms_arcsec": 9.9,
                "arc_aware_image_residual_median_arcsec": 8.8,
            },
            {
                "stage": "stage4",
                "sample_likelihood_mode": "critical-arc-mixture-image-plane",
                "headline_chi_square": 4.0,
                "point_image_rms_arcsec": 0.5,
                "point_image_median_residual_arcsec": 0.25,
                "point_recovered_image_count": 2,
                "arc_aware_chi_square": 3.0,
                "arc_aware_dof": 2,
                "arc_aware_reduced_chi_square": 1.5,
                "arc_aware_arc_supported_image_count": 1,
                "arc_aware_recovered_image_count": 3,
                "arc_aware_missing_image_count": 0,
                "arc_aware_image_rms_arcsec": 0.3,
                "arc_aware_image_residual_median_arcsec": 0.2,
            },
        ],
        run_name="mock",
        root_dir="results/mock",
    )

    assert "arc_RMS" in mixed
    assert "arc_med" in mixed
    assert "0.3" in mixed
    assert "0.2" in mixed
    assert "9.9" not in mixed
    assert "8.8" not in mixed
    assert "99" not in mixed


def test_tangential_critical_curve_caustics_converts_and_rayshoots(monkeypatch: Any) -> None:
    contour_vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    contour_inputs: list[tuple[tuple[int, int], float]] = []

    def fake_find_contours(lambda_tan: np.ndarray, level: float) -> list[np.ndarray]:
        contour_inputs.append((lambda_tan.shape, level))
        return [contour_vertices]

    class FakeModel:
        def __init__(self) -> None:
            self.ray_inputs: list[tuple[np.ndarray, np.ndarray]] = []

        def hessian(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_array = np.asarray(x, dtype=float)
            return (
                np.zeros_like(x_array),
                np.zeros_like(x_array),
                np.zeros_like(x_array),
                np.zeros_like(x_array),
            )

        def ray_shooting(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            self.ray_inputs.append((x_array.copy(), y_array.copy()))
            return x_array + 10.0, y_array - 10.0

    monkeypatch.setattr(plotting, "find_contours", fake_find_contours)
    model = FakeModel()

    contours = plotting._tangential_critical_curve_caustics(
        model,
        [{"mock": 1.0}],
        np.asarray([-2.0, 0.0, 2.0], dtype=float),
        np.asarray([10.0, 20.0, 30.0], dtype=float),
    )

    assert contour_inputs == [((3, 3), 0.0)]
    assert len(contours) == 1
    contour = contours[0]
    np.testing.assert_allclose(contour["critical_x"], [-2.0, 2.0, 2.0, -2.0])
    np.testing.assert_allclose(contour["critical_y"], [10.0, 10.0, 30.0, 30.0])
    np.testing.assert_allclose(contour["caustic_x"], [8.0, 12.0, 12.0, 8.0])
    np.testing.assert_allclose(contour["caustic_y"], [0.0, 0.0, 20.0, 20.0])
    np.testing.assert_allclose(model.ray_inputs[0][0], contour["critical_x"])
    np.testing.assert_allclose(model.ray_inputs[0][1], contour["critical_y"])


def test_plot_caustic_overlay_uses_configured_redshift_and_scatter(monkeypatch: Any, tmp_path: Path) -> None:
    family = SimpleNamespace(
        family_id="1",
        z_source=2.0,
        x_obs=np.asarray([0.0, 1.0], dtype=float),
        y_obs=np.asarray([0.0, -1.0], dtype=float),
    )
    state = SimpleNamespace(z_lens=0.3, parameter_specs=[], family_data=[family])

    class FakeModel:
        pass

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return FakeModel(), None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

        def evaluate(self, params_latent: np.ndarray, validate_all_families: bool = False) -> Any:
            return SimpleNamespace(family_predictions={"1": {"source_x": 0.5, "source_y": -0.25}})

    class FakeAxis:
        def __init__(self) -> None:
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    image_ax = FakeAxis()
    source_ax = FakeAxis()
    helper_calls: list[tuple[list[dict[str, float]], np.ndarray, np.ndarray]] = []

    def fake_subplots(*_args: Any, **_kwargs: Any) -> tuple[FakeFig, list[FakeAxis]]:
        return FakeFig(), [image_ax, source_ax]

    def fake_contours(
        model: FakeModel,
        kwargs_lens: list[dict[str, float]],
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        helper_calls.append(
            (kwargs_lens, np.asarray(x_axis, dtype=float).copy(), np.asarray(y_axis, dtype=float).copy())
        )
        return [
            {
                "critical_x": np.asarray([1.0, 2.0], dtype=float),
                "critical_y": np.asarray([3.0, 4.0], dtype=float),
                "caustic_x": np.asarray([0.1, 0.2], dtype=float),
                "caustic_y": np.asarray([-0.1, -0.2], dtype=float),
            }
        ]

    monkeypatch.setattr(plotting.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plotting, "_tangential_critical_curve_caustics", fake_contours)
    evaluator = FakeEvaluator()

    plotting._plot_caustic_overlay(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        caustic_plot_grid_scale_arcsec=0.2,
        caustic_source_redshift=9.0,
    )

    assert (tmp_path / "caustic_overlay.pdf").exists()
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert len(helper_calls) == 1
    kwargs_lens, x_axis, y_axis = helper_calls[0]
    assert kwargs_lens == [{"latent": 5.0}]
    assert len(x_axis) == 1001
    assert len(y_axis) == 1001
    np.testing.assert_allclose([x_axis[0], x_axis[-1]], [-100.0, 100.0])
    np.testing.assert_allclose([y_axis[0], y_axis[-1]], [-100.0, 100.0])
    assert x_axis[1] - x_axis[0] == pytest.approx(0.2)
    assert y_axis[1] - y_axis[0] == pytest.approx(0.2)
    assert len(image_ax.plots) == 1
    assert source_ax.plots == []
    assert len(source_ax.scatters) == 2
    np.testing.assert_allclose(source_ax.scatters[0][0], [0.1, 0.2])
    np.testing.assert_allclose(source_ax.scatters[0][1], [-0.1, -0.2])


def test_load_kappa_true_fits_suppresses_only_radecsys_warning(tmp_path: Path) -> None:
    true_path = tmp_path / "radecsys_kappa.fits"
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -20.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["RADECSYS"] = "FK5"
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32), header=header).writeto(true_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image, wcs = plotting._load_kappa_true_fits(true_path)
        warnings.warn("sentinel unrelated warning", UserWarning)

    assert image.shape == (2, 2)
    assert wcs.has_celestial
    messages = [str(item.message) for item in caught]
    assert not any("RADECSYS" in message for message in messages)
    assert any("sentinel unrelated warning" in message for message in messages)


def test_install_astropy_wcs_warning_filters_suppresses_radecsys_only() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lc_utils.install_astropy_wcs_warning_filters()
        warnings.warn(
            "RADECSYS= 'FK5 '\nthe RADECSYS keyword is deprecated, use RADESYSa.",
            plotting.FITSFixedWarning,
        )
        warnings.warn("unrelated WCS warning", plotting.FITSFixedWarning)

    messages = [str(item.message) for item in caught]
    assert not any("RADECSYS" in message for message in messages)
    assert any("unrelated WCS warning" in message for message in messages)


def test_plot_kappa_true_comparison_uses_fits_grid_redshift_and_fixed_limits(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    true_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    true_kappa = np.asarray([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    fits.PrimaryHDU(true_kappa, header=wcs.to_header()).writeto(true_path)

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeModel:
        def __init__(self) -> None:
            self.inputs: list[tuple[np.ndarray, np.ndarray, list[dict[str, float]]]] = []

        def kappa(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            self.inputs.append((x_array.copy(), y_array.copy(), kwargs_lens))
            return np.asarray([2.0, 2.0, 2.0, 5.0], dtype=float)[: x_array.size]

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.converted: list[np.ndarray] = []
            self.model_z: list[float] = []
            self.packed_z: list[float] = []
            self.model = FakeModel()

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            theta_array = np.asarray(theta, dtype=float)
            self.converted.append(theta_array.copy())
            return theta_array + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    class FakeColorbar:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def set_label(self, label: str) -> None:
            self.labels.append(label)

    class FakeAxis:
        def __init__(self) -> None:
            self.pcolormesh_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
            self.inverted = False
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self.title: str | None = None

        def pcolormesh(self, x: Any, y: Any, data: Any, **kwargs: Any) -> str:
            self.pcolormesh_calls.append(
                (
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    np.ma.asarray(data).filled(np.nan),
                    dict(kwargs),
                )
            )
            return f"mesh-{len(self.pcolormesh_calls)}"

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def set_title(self, title: str) -> None:
            self.title = title

    class FakeFig:
        def __init__(self) -> None:
            self.colorbars: list[FakeColorbar] = []
            self.saved_paths: list[Path] = []

        def colorbar(self, image: Any, ax: FakeAxis, **kwargs: Any) -> FakeColorbar:
            assert image.startswith("mesh-")
            assert isinstance(ax, FakeAxis)
            assert kwargs["fraction"] == pytest.approx(0.046)
            assert kwargs["pad"] == pytest.approx(0.04)
            colorbar = FakeColorbar()
            self.colorbars.append(colorbar)
            return colorbar

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    axes = [FakeAxis(), FakeAxis()]
    figs = [FakeFig(), FakeFig()]
    subplots_calls = iter(zip(figs, axes, strict=True))

    def fake_subplots(*_args: Any, **_kwargs: Any) -> tuple[FakeFig, FakeAxis]:
        return next(subplots_calls)

    monkeypatch.setattr(plotting.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    three_panel_calls: list[dict[str, Any]] = []

    def fake_three_panel(
        plot_dir: Path,
        kappa_true: np.ndarray,
        model_kappa: np.ndarray,
        x_arcsec: np.ndarray,
        y_arcsec: np.ndarray,
        z_source: float,
        **kwargs: Any,
    ) -> None:
        three_panel_calls.append(
            {
                "plot_dir": Path(plot_dir),
                "kappa_true": np.asarray(kappa_true, dtype=float),
                "model_kappa": np.asarray(model_kappa, dtype=float),
                "x_arcsec": np.asarray(x_arcsec, dtype=float),
                "y_arcsec": np.asarray(y_arcsec, dtype=float),
                "z_source": float(z_source),
                "kwargs": dict(kwargs),
            }
        )
        (Path(plot_dir) / "truth_recovery_kappa_model_truth_fractional_residual.pdf").touch()

    monkeypatch.setattr(plotting, "_plot_kappa_model_truth_fractional_residual_from_grid", fake_three_panel)
    evaluator = FakeEvaluator()

    plotting._plot_kappa_true_comparison(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        true_path,
        caustic_source_redshift=9.0,
    )

    assert not (tmp_path / "kappa_comparison.pdf").exists()
    assert not (tmp_path / "kappa_model.pdf").exists()
    assert not (tmp_path / "kappa_fractional_residual.pdf").exists()
    assert not (tmp_path / "truth_recovery_kappa_model.pdf").exists()
    assert (tmp_path / "truth_recovery_kappa_fractional_residual.pdf").exists()
    assert (tmp_path / "truth_recovery_kappa_model_truth_fractional_residual.pdf").exists()
    assert [path for fig in figs for path in fig.saved_paths] == [
        tmp_path / "truth_recovery_kappa_fractional_residual.pdf",
    ]
    assert len(three_panel_calls) == 1
    assert three_panel_calls[0]["plot_dir"] == tmp_path
    np.testing.assert_allclose(three_panel_calls[0]["model_kappa"], [[2.0, 2.0], [2.0, 5.0]])
    np.testing.assert_allclose(
        three_panel_calls[0]["kwargs"]["fractional_residual"],
        [[1.0, np.nan], [np.nan, 1.5]],
        equal_nan=True,
    )
    assert [float(item[0]) for item in evaluator.converted] == [4.0]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]
    assert len(evaluator.model.inputs) == 1
    _x_input, _y_input, kwargs_lens = evaluator.model.inputs[0]
    assert kwargs_lens == [{"latent": 5.0}]

    residual_x, residual_y, residual_data, residual_kwargs = axes[0].pcolormesh_calls[0]
    np.testing.assert_allclose(residual_data, [[1.0, np.nan], [np.nan, 1.5]], equal_nan=True)
    np.testing.assert_allclose(residual_x, [[0.0, -1.0], [0.0, -1.0]])
    np.testing.assert_allclose(residual_y, [[0.0, 0.0], [1.0, 1.0]], atol=1.0e-9)
    assert residual_kwargs["shading"] == "nearest"
    assert residual_kwargs["edgecolors"] == "none"
    assert residual_kwargs["linewidth"] == pytest.approx(0.0)
    assert residual_kwargs["antialiased"] is False
    assert residual_kwargs["rasterized"] is True
    assert "vmin" not in residual_kwargs
    assert "vmax" not in residual_kwargs
    residual_norm = residual_kwargs["norm"]
    assert isinstance(residual_norm, plotting.TwoSlopeNorm)
    assert residual_norm.vmin == pytest.approx(-1.0)
    assert residual_norm.vcenter == pytest.approx(0.0)
    assert residual_norm.vmax == pytest.approx(2.0)
    assert axes[0].inverted is False
    assert axes[0].title is None
    assert [colorbar.labels for fig in figs for colorbar in fig.colorbars] == [
        [r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$"],
    ]


def test_kappa_model_truth_fractional_residual_three_panel_plot(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeColorbar:
        def __init__(self) -> None:
            self.label: str | None = None
            self.ax = type(
                "FakeColorbarAxis",
                (),
                {
                    "xaxis": type(
                        "FakeXAxis",
                        (),
                        {
                            "set_label_position": lambda self, _position: None,
                            "set_ticks_position": lambda self, _position: None,
                        },
                    )()
                },
            )()

        def set_label(self, label: str) -> None:
            self.label = label

    class FakeAxis:
        def __init__(self, x0: float, y1: float, width: float) -> None:
            self.pcolormesh_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
            self.inverted = False
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self._position = type("FakeBbox", (), {"x0": x0, "y1": y1, "width": width})()

        def pcolormesh(self, x: Any, y: Any, data: Any, **kwargs: Any) -> str:
            self.pcolormesh_calls.append(
                (
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    np.ma.asarray(data).filled(np.nan),
                    dict(kwargs),
                )
            )
            return f"mesh-{len(self.pcolormesh_calls)}"

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def get_position(self) -> Any:
            return self._position

    class FakeColorbarAxis:
        def __init__(self, bounds: list[float]) -> None:
            self.bounds = bounds

    class FakeFig:
        def __init__(self) -> None:
            self.colorbars: list[tuple[str, FakeColorbarAxis, FakeColorbar, dict[str, Any]]] = []
            self.added_axes: list[FakeColorbarAxis] = []
            self.saved_paths: list[Path] = []
            self.title: str | None = None
            self.subplots_adjust_kwargs: dict[str, Any] | None = None

        def subplots_adjust(self, **kwargs: Any) -> None:
            self.subplots_adjust_kwargs = dict(kwargs)

        def add_axes(self, bounds: list[float]) -> FakeColorbarAxis:
            axis = FakeColorbarAxis(bounds)
            self.added_axes.append(axis)
            return axis

        def colorbar(self, image: str, cax: FakeColorbarAxis, **kwargs: Any) -> FakeColorbar:
            assert kwargs["orientation"] == "horizontal"
            colorbar = FakeColorbar()
            self.colorbars.append((image, cax, colorbar, dict(kwargs)))
            return colorbar

        def suptitle(self, title: str) -> None:
            self.title = title

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    axes = np.asarray(
        [
            FakeAxis(0.10, 0.80, 0.22),
            FakeAxis(0.40, 0.80, 0.22),
            FakeAxis(0.70, 0.80, 0.22),
        ],
        dtype=object,
    )
    fig = FakeFig()
    subplots_calls: list[dict[str, Any]] = []

    def fake_subplots(*_args: Any, **kwargs: Any) -> tuple[FakeFig, np.ndarray]:
        subplots_calls.append(dict(kwargs))
        return fig, axes

    monkeypatch.setattr(plotting.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    x_arcsec = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    y_arcsec = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    plotting._plot_kappa_model_truth_fractional_residual_from_grid(
        tmp_path,
        np.asarray([[1.0, 2.0], [0.0, np.nan]], dtype=float),
        np.asarray([[2.0, 1.0], [3.0, 4.0]], dtype=float),
        x_arcsec,
        y_arcsec,
        z_source=9.0,
    )

    assert (tmp_path / "truth_recovery_kappa_model_truth_fractional_residual.pdf").exists()
    assert fig.saved_paths == [tmp_path / "truth_recovery_kappa_model_truth_fractional_residual.pdf"]
    assert fig.title is None
    assert subplots_calls[0]["sharey"] is True
    assert fig.subplots_adjust_kwargs == {"top": 0.82, "wspace": 0.08}
    assert len(fig.colorbars) == 3
    assert [axis.bounds for axis in fig.added_axes] == [
        [0.10, 0.8180000000000001, 0.22, 0.026],
        [0.40, 0.8180000000000001, 0.22, 0.026],
        [0.70, 0.8180000000000001, 0.22, 0.026],
    ]
    assert [kwargs["orientation"] for _image, _axis, _colorbar, kwargs in fig.colorbars] == [
        "horizontal",
        "horizontal",
        "horizontal",
    ]
    assert [colorbar.label for _image, _axis, colorbar, _kwargs in fig.colorbars] == [
        r"$\kappa_{\rm model}$",
        r"$\kappa_{\rm true}$",
        r"$(\kappa_{\rm model} - \kappa_{\rm true}) / \kappa_{\rm true}$",
    ]
    for axis_index, axis in enumerate(axes):
        assert axis.inverted is False
        assert axis.xlabel == "x [arcsec]"
        assert axis.ylabel == ("y [arcsec]" if axis_index == 0 else None)
        assert len(axis.pcolormesh_calls) == 1
        np.testing.assert_allclose(axis.pcolormesh_calls[0][0], x_arcsec)
        np.testing.assert_allclose(axis.pcolormesh_calls[0][1], y_arcsec)
        kwargs = axis.pcolormesh_calls[0][3]
        assert kwargs["shading"] == "nearest"
        assert kwargs["edgecolors"] == "none"
        assert kwargs["linewidth"] == pytest.approx(0.0)
        assert kwargs["antialiased"] is False
        assert kwargs["rasterized"] is True
    assert axes[0].pcolormesh_calls[0][3]["vmin"] == pytest.approx(0.0)
    assert axes[0].pcolormesh_calls[0][3]["vmax"] == pytest.approx(3.0)
    assert axes[0].pcolormesh_calls[0][3]["cmap"] == "magma"
    assert axes[1].pcolormesh_calls[0][3]["vmin"] == pytest.approx(0.0)
    assert axes[1].pcolormesh_calls[0][3]["vmax"] == pytest.approx(3.0)
    assert axes[1].pcolormesh_calls[0][3]["cmap"] == "magma"
    residual_norm = axes[2].pcolormesh_calls[0][3]["norm"]
    assert isinstance(residual_norm, plotting.Normalize)
    assert residual_norm.vmin == pytest.approx(-1.0)
    assert residual_norm.vmax == pytest.approx(2.0)
    assert residual_norm.clip is True
    assert residual_norm(-1.0) == pytest.approx(0.0)
    assert residual_norm(0.0) == pytest.approx(1.0 / 3.0)
    assert residual_norm(1.0) == pytest.approx(2.0 / 3.0)
    assert residual_norm(2.0) == pytest.approx(1.0)
    residual_cmap = axes[2].pcolormesh_calls[0][3]["cmap"]
    assert isinstance(residual_cmap, plotting.LinearSegmentedColormap)
    assert residual_cmap.name.startswith("RdBu_r_zero_at_")
    assert residual_cmap.name != "RdBu_r"
    np.testing.assert_allclose(
        axes[2].pcolormesh_calls[0][2],
        [[1.0, -0.5], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_kappa_fractional_residual_overlays_members_and_images(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeColorbar:
        def set_label(self, _label: str) -> None:
            return None

    class FakeAxis:
        def __init__(self) -> None:
            self.scatter_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
            self.text_calls: list[tuple[float, float, str, dict[str, Any]]] = []
            self.legend_calls: list[dict[str, Any]] = []
            self.pcolormesh_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
            self.inverted = False

        def pcolormesh(self, x: Any, y: Any, data: Any, **kwargs: Any) -> str:
            self.pcolormesh_calls.append(
                (
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    np.ma.asarray(data).filled(np.nan),
                    dict(kwargs),
                )
            )
            return "mesh"

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatter_calls.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float), dict(kwargs)))

        def text(self, x: float, y: float, text: str, **kwargs: Any) -> None:
            self.text_calls.append((float(x), float(y), str(text), dict(kwargs)))

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]:
            labels = [str(kwargs["label"]) for _x, _y, kwargs in self.scatter_calls if kwargs.get("label")]
            return [object() for _label in labels], labels

        def legend(self, **kwargs: Any) -> None:
            self.legend_calls.append(dict(kwargs))

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

    class FakeFig:
        def colorbar(self, *_args: Any, **_kwargs: Any) -> FakeColorbar:
            return FakeColorbar()

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axes = [FakeAxis(), FakeAxis()]
    figures = [FakeFig(), FakeFig()]
    subplots_calls = iter(zip(figures, axes, strict=True))
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: next(subplots_calls))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        plotting,
        "_plot_kappa_model_truth_fractional_residual_from_grid",
        lambda plot_dir, *_args, **_kwargs: (Path(plot_dir) / "truth_recovery_kappa_model_truth_fractional_residual.pdf").touch(),
    )

    member_overlays = pd.DataFrame(
        [
            {"catalog_id": "inactive", "x_arcsec": 0.0, "y_arcsec": 0.5, "free": False},
            {"catalog_id": "free", "x_arcsec": 1.0, "y_arcsec": 1.5, "free": True},
            {"catalog_id": "bad", "x_arcsec": np.nan, "y_arcsec": 2.0, "free": True},
        ]
    )
    image_overlays = pd.DataFrame(
        [
            {"x_arcsec": -0.5, "y_arcsec": 0.25},
            {"x_arcsec": np.nan, "y_arcsec": 4.0},
        ]
    )

    plotting._plot_kappa_true_comparison_from_grid(
        tmp_path,
        np.ones((2, 2), dtype=float),
        np.asarray([[2.2, 1.2], [1.2, 1.2]], dtype=float),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        9.0,
        member_overlays=member_overlays,
        image_overlays=image_overlays,
    )

    residual_axis = axes[0]
    unused_axis = axes[1]
    assert unused_axis.scatter_calls == []
    assert unused_axis.text_calls == []
    assert [call[2]["marker"] for call in residual_axis.scatter_calls] == ["x", "s", "D"]
    np.testing.assert_allclose(residual_axis.pcolormesh_calls[0][0], [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(residual_axis.pcolormesh_calls[0][2], [[1.2, 0.2], [0.2, 0.2]])
    np.testing.assert_allclose(residual_axis.scatter_calls[0][0], [-0.5])
    np.testing.assert_allclose(residual_axis.scatter_calls[0][1], [0.25])
    assert residual_axis.scatter_calls[0][2]["label"] == "observed images"
    assert residual_axis.scatter_calls[1][2]["label"] == "not free"
    assert residual_axis.scatter_calls[2][2]["label"] == "free"
    assert [call[2] for call in residual_axis.text_calls] == ["inactive", "free"]
    assert residual_axis.text_calls[1][0] == pytest.approx(1.0)
    assert residual_axis.text_calls[1][1] == pytest.approx(1.5)
    assert residual_axis.legend_calls
    assert residual_axis.inverted is False
    assert unused_axis.inverted is False


def test_plot_kappa_recovery_samples_every_pixel_and_writes_reduced_tables(tmp_path: Path) -> None:
    true_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    true_kappa = np.arange(9, dtype=np.float32).reshape(3, 3)
    fits.PrimaryHDU(true_kappa, header=wcs.to_header()).writeto(true_path)

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeModel:
        def __init__(self) -> None:
            self.input_sizes: list[int] = []

        def kappa(self, x: Any, y: Any, kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            assert kwargs_lens == [{"latent": 5.0}]
            x_array = np.asarray(x, dtype=float)
            self.input_sizes.append(int(x_array.size))
            return 10.0 + np.arange(x_array.size, dtype=float)

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.model_z: list[float] = []
            self.packed_z: list[float] = []
            self.model = FakeModel()

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float) + 1.0

        def _get_exact_model_solver(self, z_source: float) -> tuple[FakeModel, None]:
            self.model_z.append(float(z_source))
            return self.model, None

        def _build_packed_lens_state(self, sample_latent: Any, z_source: float) -> dict[str, float]:
            self.packed_z.append(float(z_source))
            return {"latent": float(np.asarray(sample_latent, dtype=float)[0])}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    evaluator = FakeEvaluator()

    plotting._plot_kappa_recovery(
        tmp_path,
        evaluator,
        np.asarray([4.0], dtype=float),
        true_path,
        caustic_source_redshift=9.0,
    )

    assert not (tmp_path / "kappa_recovery.pdf").exists()
    assert not (tmp_path / "tables" / "kappa_recovery_binned.csv").exists()
    assert not (tmp_path / "tables" / "kappa_recovery_summary.csv").exists()
    assert (tmp_path / "truth_recovery_kappa_recovery.pdf").exists()
    binned = pd.read_csv(tmp_path / "tables" / "truth_recovery_kappa_recovery_binned.csv")
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_kappa_recovery_summary.csv")
    assert not (tmp_path / "tables" / "kappa_recovery_samples.csv").exists()
    assert not binned.empty
    assert summary.loc[0, "total_pixel_count"] == 9
    assert summary.loc[0, "finite_pixel_count"] == 9
    assert summary.loc[0, "kappa_bias_median"] == pytest.approx(10.0)
    assert summary.loc[0, "kappa_spread_nmad"] == pytest.approx(0.0)
    assert summary.loc[0, "kappa_rmse"] == pytest.approx(10.0)
    assert evaluator.model.input_sizes == [9]
    assert evaluator.model_z == [9.0]
    assert evaluator.packed_z == [9.0]


def test_kappa_recovery_overlays_observed_image_points(monkeypatch: Any, tmp_path: Path) -> None:
    true_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), header=wcs.to_header()).writeto(true_path)

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeModel:
        def kappa(self, x: Any, y: Any, _kwargs_lens: list[dict[str, float]]) -> np.ndarray:
            x_array = np.asarray(x, dtype=float)
            y_array = np.asarray(y, dtype=float)
            return 20.0 + x_array - y_array

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, FakeModel] = {}
            self.model = FakeModel()

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[FakeModel, None]:
            return self.model, None

        def _build_packed_lens_state(self, _sample_latent: Any, _z_source: float) -> dict[str, float]:
            return {}

        def _packed_to_kwargs_lens(self, packed_state: dict[str, float]) -> list[dict[str, float]]:
            return [packed_state]

    captured: dict[str, Any] = {}

    def fake_plot_quantity_recovery(_recovery: dict[str, Any], output_path: Path, **kwargs: Any) -> None:
        captured.update(kwargs)
        Path(output_path).touch()

    monkeypatch.setattr(plotting, "_plot_quantity_recovery", fake_plot_quantity_recovery)
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1"],
            "image_label": ["1.a", "1.b"],
            "x_obs_arcsec": [0.0, -1.0],
            "y_obs_arcsec": [0.0, 0.0],
        }
    )

    plotting._plot_kappa_recovery(
        tmp_path,
        FakeEvaluator(),
        np.asarray([4.0], dtype=float),
        true_path,
        caustic_source_redshift=9.0,
        image_df=image_df,
    )

    image_points = captured["image_points"]
    np.testing.assert_allclose(image_points["true_value"].to_numpy(dtype=float), [1.0, 2.0])
    np.testing.assert_allclose(image_points["model_value"].to_numpy(dtype=float), [20.0, 19.0])
    assert image_points["image_label"].tolist() == ["1.a", "1.b"]


def test_absolute_mu_truth_grid_derives_raw_abs_mu_from_kappa_and_gamma(tmp_path: Path) -> None:
    def write_truth_fits(path: Path, data: np.ndarray) -> None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=wcs.to_header()).writeto(path)

    kappa_path = tmp_path / "kappa.fits"
    gammax_path = tmp_path / "gammax.fits"
    gammay_path = tmp_path / "gammay.fits"
    write_truth_fits(kappa_path, np.asarray([[0.0, 0.5], [1.0, np.nan]], dtype=float))
    write_truth_fits(gammax_path, np.zeros((2, 2), dtype=float))
    write_truth_fits(gammay_path, np.zeros((2, 2), dtype=float))

    abs_mu, wcs = plotting._absolute_mu_truth_grid(kappa_path, gammax_path, gammay_path, cap=25.0)

    np.testing.assert_allclose(abs_mu, [[1.0, 4.0], [np.inf, np.nan]], equal_nan=True)
    assert wcs.has_celestial


def test_truth_recovery_diagnostic_grid_samples_native_block_centers() -> None:
    native_wcs = WCS(naxis=2)
    native_wcs.wcs.crpix = [1.0, 1.0]
    native_wcs.wcs.crval = [0.0, 0.0]
    native_wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    native_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    reduced_wcs, reduced_shape, metadata = plotting._truth_recovery_diagnostic_grid(
        native_wcs,
        (2048, 2048),
        256,
    )

    assert reduced_shape == (256, 256)
    assert metadata["truth_grid_sampling"] == "bilinear_reduced"
    assert metadata["native_to_diagnostic_pixel_ratio_x"] == pytest.approx(8.0)
    assert metadata["native_to_diagnostic_pixel_ratio_y"] == pytest.approx(8.0)
    ra_reduced, dec_reduced = reduced_wcs.pixel_to_world_values(
        np.asarray([0.0, 255.0], dtype=float),
        np.asarray([0.0, 255.0], dtype=float),
    )
    x_native, y_native = native_wcs.world_to_pixel_values(ra_reduced, dec_reduced)
    np.testing.assert_allclose(x_native, [3.5, 2043.5], atol=1.0e-8)
    np.testing.assert_allclose(y_native, [3.5, 2043.5], atol=1.0e-8)


def test_truth_recovery_wcs_grid_interpolation_preserves_constant_and_linear_maps() -> None:
    native_wcs = WCS(naxis=2)
    native_wcs.wcs.crpix = [1.0, 1.0]
    native_wcs.wcs.crval = [0.0, 0.0]
    native_wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    native_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    reduced_wcs, reduced_shape, _metadata = plotting._truth_recovery_diagnostic_grid(
        native_wcs,
        (8, 8),
        2,
    )
    y_native, x_native = np.indices((8, 8), dtype=float)
    constant = np.full((8, 8), 7.0, dtype=float)
    linear = 2.0 * x_native + 3.0 * y_native

    sampled_constant = plotting._sample_wcs_image_on_wcs_grid(constant, native_wcs, reduced_wcs, reduced_shape)
    sampled_linear = plotting._sample_wcs_image_on_wcs_grid(linear, native_wcs, reduced_wcs, reduced_shape)

    np.testing.assert_allclose(sampled_constant, np.full((2, 2), 7.0))
    expected_x = np.asarray([[1.5, 5.5], [1.5, 5.5]], dtype=float)
    expected_y = np.asarray([[1.5, 1.5], [5.5, 5.5]], dtype=float)
    np.testing.assert_allclose(sampled_linear, 2.0 * expected_x + 3.0 * expected_y, atol=1.0e-8)


def test_absolute_mu_truth_grid_rejects_mismatched_truth_shapes(tmp_path: Path) -> None:
    def write_truth_fits(path: Path, data: np.ndarray) -> None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=wcs.to_header()).writeto(path)

    kappa_path = tmp_path / "kappa.fits"
    gammax_path = tmp_path / "gammax.fits"
    gammay_path = tmp_path / "gammay.fits"
    write_truth_fits(kappa_path, np.zeros((2, 2), dtype=float))
    write_truth_fits(gammax_path, np.zeros((2, 3), dtype=float))
    write_truth_fits(gammay_path, np.zeros((2, 2), dtype=float))

    with pytest.raises(ValueError, match="gammax truth FITS shape"):
        plotting._absolute_mu_truth_grid(kappa_path, gammax_path, gammay_path)


def test_kappa_truth_diagnostics_write_posterior_median_fits_with_truth_wcs(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [3.0, -2.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32), header=wcs.to_header()).writeto(kappa_path)

    state = SimpleNamespace(
        z_lens=0.3,
        reference=(3, 3.0, -2.0),
        parameter_specs=[],
        scaling_component_records=[
            {"catalog_id": "faint", "catalog_mag": 22.0, "x_centre": 1.0, "y_centre": 2.0},
            {"catalog_id": "bright", "catalog_mag": 18.0, "x_centre": 0.0, "y_centre": 0.0},
        ],
    )

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            latent = plotting.jnp.asarray(packed_state["latent"], dtype=plotting.jnp.float64)
            a00 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            a11 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return a00, zeros, zeros, a11

    results = PosteriorResults(
        samples=np.asarray([[0.1], [0.2], [0.9]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    evaluator = FakeEvaluator()

    plotting._plot_kappa_truth_diagnostics(
        tmp_path,
        evaluator,
        results,
        kappa_path,
        caustic_source_redshift=9.0,
    )

    assert not (tmp_path / "kappa_model_median.fits").exists()
    assert not (tmp_path / "tables" / "truth_grid_summary.csv").exists()
    assert not (tmp_path / "truth_recovery_kappa_model_median.fits").exists()
    with fits.open(tmp_path / "fits" / "truth_recovery_kappa_model_median.fits") as hdul:
        assert hdul[0].data.dtype == np.dtype(">f8")
        median_data = np.asarray(hdul[0].data, dtype=float)
        median_wcs = WCS(hdul[0].header).celestial
    np.testing.assert_allclose(median_data, np.full((2, 2), 0.2))
    assert not (tmp_path / "truth_recovery_kappa_model_q16.fits").exists()
    assert not (tmp_path / "truth_recovery_kappa_model_q84.fits").exists()
    plotting._validate_matching_truth_wcs(wcs.celestial, median_wcs, (2, 2), label="median")
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_summary.csv")
    assert summary.loc[summary["quantity"] == "kappa", "truth_grid_mode"].iloc[0] == "median"
    assert not bool(summary.loc[summary["quantity"] == "kappa", "spread_available"].iloc[0])
    assert summary.loc[summary["quantity"] == "kappa", "draw_count_used"].iloc[0] == 1
    assert summary.loc[summary["quantity"] == "kappa", "dtype"].iloc[0] == "float64"
    assert summary.loc[summary["quantity"] == "kappa", "chunk_pixels"].iloc[0] == 4
    assert summary.loc[summary["quantity"] == "kappa", "chunk_count"].iloc[0] == 1
    estimated_bytes = int(summary.loc[summary["quantity"] == "kappa", "estimated_grid_buffer_memory_bytes"].iloc[0])
    assert estimated_bytes == 1 * 4 * 1 * 8
    assert summary.loc[summary["quantity"] == "kappa", "estimated_grid_buffer_memory_gb"].iloc[0] == pytest.approx(
        estimated_bytes / 1024**3
    )
    assert (tmp_path / "truth_recovery_m2d_aperture_ratio.pdf").exists()
    aperture = pd.read_csv(tmp_path / "tables" / "truth_recovery_m2d_aperture_profile.csv")
    assert aperture["center_mode"].unique().tolist() == ["smoothed_truth_kappa_peak"]
    assert aperture["center_catalog_id"].fillna("").astype(str).unique().tolist() == [""]
    assert np.isnan(aperture["center_catalog_mag"].to_numpy(dtype=float)).all()
    assert aperture["center_smoothing_sigma_pix"].unique().tolist() == [
        plotting.TRUTH_RECOVERY_APERTURE_CENTER_SMOOTHING_SIGMA_PIX
    ]
    assert np.isfinite(aperture["m2d_ratio"].to_numpy(dtype=float)).any()
    assert {"m2d_ratio_q16", "m2d_ratio_median", "m2d_ratio_q84"}.issubset(aperture.columns)
    assert np.isnan(aperture["m2d_ratio_q16"].to_numpy(dtype=float)).all()
    assert np.isnan(aperture["m2d_ratio_q84"].to_numpy(dtype=float)).all()


def test_kappa_truth_diagnostics_uses_reduced_truth_grid_size(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [3.0, -2.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32), header=wcs.to_header()).writeto(kappa_path)

    state = SimpleNamespace(
        z_lens=0.3,
        reference=(3, 3.0, -2.0),
        parameter_specs=[],
        scaling_component_records=[
            {"catalog_id": "bright", "catalog_mag": 18.0, "x_centre": 0.0, "y_centre": 0.0},
        ],
    )

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}
            self.build_count = 0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            self.build_count += 1
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            latent = plotting.jnp.asarray(packed_state["latent"], dtype=plotting.jnp.float64)
            a00 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            a11 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return a00, zeros, zeros, a11

    results = PosteriorResults(
        samples=np.asarray([[0.1], [0.2], [0.3]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    evaluator = FakeEvaluator()

    plotting._plot_kappa_truth_diagnostics(
        tmp_path,
        evaluator,
        results,
        kappa_path,
        caustic_source_redshift=9.0,
        truth_grid_mode="posterior",
        truth_grid_size=2,
    )

    assert not (tmp_path / "truth_recovery_kappa_model_median.fits").exists()
    with fits.open(tmp_path / "fits" / "truth_recovery_kappa_model_median.fits") as hdul:
        assert hdul[0].data.shape == (2, 2)
        assert hdul[0].data.dtype == np.dtype(">f8")
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_summary.csv")
    kappa_summary = summary.loc[summary["quantity"] == "kappa"].iloc[0]
    assert int(kappa_summary["native_truth_height"]) == 8
    assert int(kappa_summary["native_truth_width"]) == 8
    assert int(kappa_summary["diagnostic_grid_height"]) == 2
    assert int(kappa_summary["diagnostic_grid_width"]) == 2
    assert kappa_summary["truth_grid_sampling"] == "bilinear_reduced"
    assert float(kappa_summary["native_to_diagnostic_pixel_ratio_x"]) == pytest.approx(4.0)
    assert int(kappa_summary["chunk_count"]) == 1
    assert int(kappa_summary["chunk_pixels"]) == 4
    assert evaluator.build_count == 3


def test_kappa_truth_diagnostics_requires_precomputed_grid_when_requested(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32), header=wcs.to_header()).writeto(kappa_path)
    evaluator = SimpleNamespace(
        state=SimpleNamespace(
            z_lens=0.3,
            reference=(3, 0.0, 0.0),
            scaling_component_records=[],
        )
    )
    results = PosteriorResults(
        samples=np.asarray([[0.1]], dtype=float),
        log_prob=np.zeros(1, dtype=float),
        accept_prob=np.zeros(1, dtype=float),
        diverging=np.zeros(1, dtype=bool),
        num_steps=np.zeros(1, dtype=int),
        warmup_steps=0,
        sample_steps=1,
        num_chains=1,
    )

    with pytest.raises(RuntimeError, match="truth_recovery_grids"):
        plotting._plot_kappa_truth_diagnostics(
            tmp_path,
            evaluator,
            results,
            kappa_path,
            caustic_source_redshift=9.0,
            truth_grid_cache={},
            require_precomputed_truth_grid=True,
        )


def test_smoothed_truth_kappa_peak_aperture_center_uses_real_grid_coordinates() -> None:
    x_arcsec, y_arcsec = np.meshgrid(
        np.asarray([10.0, 20.0, 30.0], dtype=float),
        np.asarray([-2.0, -1.0, 0.0], dtype=float),
    )
    kappa_true = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 100.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=float,
    )

    center = plotting._smoothed_truth_kappa_peak_aperture_center(
        kappa_true,
        x_arcsec,
        y_arcsec,
        sigma_pix=1.0e-6,
    )

    assert center is not None
    assert center["center_mode"] == "smoothed_truth_kappa_peak"
    assert center["center_x_arcsec"] == pytest.approx(20.0)
    assert center["center_y_arcsec"] == pytest.approx(-1.0)
    assert center["center_catalog_id"] == ""
    assert np.isnan(center["center_catalog_mag"])
    assert center["center_smoothing_sigma_pix"] == pytest.approx(1.0e-6)
    assert center["center_smoothed_kappa_peak"] == pytest.approx(100.0)


def test_smoothed_truth_kappa_peak_aperture_center_ignores_nans() -> None:
    x_arcsec, y_arcsec = np.meshgrid(np.asarray([0.0, 1.0, 2.0], dtype=float), np.asarray([0.0, 1.0, 2.0], dtype=float))
    kappa_true = np.asarray(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 10.0, np.nan],
            [np.nan, np.nan, 4.0],
        ],
        dtype=float,
    )

    center = plotting._smoothed_truth_kappa_peak_aperture_center(
        kappa_true,
        x_arcsec,
        y_arcsec,
        sigma_pix=1.0,
    )

    assert center is not None
    assert center["center_x_arcsec"] == pytest.approx(1.0)
    assert center["center_y_arcsec"] == pytest.approx(1.0)


def test_smoothed_truth_kappa_peak_aperture_center_returns_none_for_all_nan() -> None:
    x_arcsec, y_arcsec = np.meshgrid(np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 1.0], dtype=float))
    center = plotting._smoothed_truth_kappa_peak_aperture_center(
        np.full((2, 2), np.nan, dtype=float),
        x_arcsec,
        y_arcsec,
    )
    assert center is None


def test_truth_recovery_aperture_profile_ratio_uses_finite_kappa_pixels() -> None:
    x_arcsec, y_arcsec = np.meshgrid(np.asarray([0.0, 1.0], dtype=float), np.asarray([0.0, 1.0], dtype=float))
    kappa_true = np.asarray([[1.0, 2.0], [np.nan, 4.0]], dtype=float)
    model_kappa = np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=float)
    center = {
        "center_mode": "smoothed_truth_kappa_peak",
        "center_x_arcsec": 0.0,
        "center_y_arcsec": 0.0,
        "center_catalog_id": "",
        "center_catalog_mag": np.nan,
        "center_smoothing_sigma_pix": 1.0,
        "center_smoothed_kappa_peak": 4.0,
    }

    profile = plotting._truth_recovery_aperture_profile(
        kappa_true,
        model_kappa,
        x_arcsec,
        y_arcsec,
        center,
        n_radii=2,
    )

    assert list(profile.columns) == [
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
    assert profile.loc[0, "m2d_ratio"] == pytest.approx(2.0)
    assert profile.loc[1, "m2d_ratio"] == pytest.approx(2.0)
    assert profile["center_smoothing_sigma_pix"].unique().tolist() == [1.0]
    assert profile["center_smoothed_kappa_peak"].unique().tolist() == [4.0]


def test_truth_recovery_m2d_aperture_plot_draws_posterior_band(monkeypatch: Any, tmp_path: Path) -> None:
    from matplotlib.axes import Axes

    fill_calls: list[dict[str, np.ndarray]] = []
    plot_labels: list[str] = []
    set_xscale_calls: list[str] = []
    set_yscale_calls: list[str] = []
    set_xlim_calls: list[tuple[float | None, float | None]] = []
    set_ylim_calls: list[tuple[float | None, float | None]] = []
    text_calls: list[str] = []
    vline_calls: list[np.ndarray] = []
    original_fill_between = Axes.fill_between
    original_plot = Axes.plot
    original_set_xscale = Axes.set_xscale
    original_set_yscale = Axes.set_yscale
    original_set_xlim = Axes.set_xlim
    original_set_ylim = Axes.set_ylim
    original_text = Axes.text
    original_vlines = Axes.vlines

    def record_fill_between(self: Axes, x: Any, y1: Any, y2: Any, *args: Any, **kwargs: Any) -> Any:
        fill_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y1": np.asarray(y1, dtype=float),
                "y2": np.asarray(y2, dtype=float),
            }
        )
        return original_fill_between(self, x, y1, y2, *args, **kwargs)

    def record_plot(self: Axes, *args: Any, **kwargs: Any) -> Any:
        plot_labels.append(str(kwargs.get("label", "")))
        return original_plot(self, *args, **kwargs)

    def record_set_xscale(self: Axes, value: Any, *args: Any, **kwargs: Any) -> Any:
        set_xscale_calls.append(str(value))
        return original_set_xscale(self, value, *args, **kwargs)

    def record_set_yscale(self: Axes, value: Any, *args: Any, **kwargs: Any) -> Any:
        set_yscale_calls.append(str(value))
        return original_set_yscale(self, value, *args, **kwargs)

    def record_set_xlim(
        self: Axes,
        left: Any = None,
        right: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        set_xlim_calls.append((left, right))
        return original_set_xlim(self, left, right, *args, **kwargs)

    def record_set_ylim(self: Axes, bottom: Any = None, top: Any = None, *args: Any, **kwargs: Any) -> Any:
        set_ylim_calls.append((bottom, top))
        return original_set_ylim(self, bottom, top, *args, **kwargs)

    def record_text(self: Axes, x: Any, y: Any, s: str, *args: Any, **kwargs: Any) -> Any:
        text_calls.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    def record_vlines(self: Axes, x: Any, ymin: Any, ymax: Any, *args: Any, **kwargs: Any) -> Any:
        vline_calls.append(np.asarray(x, dtype=float))
        return original_vlines(self, x, ymin, ymax, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", record_fill_between)
    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(Axes, "set_xscale", record_set_xscale)
    monkeypatch.setattr(Axes, "set_yscale", record_set_yscale)
    monkeypatch.setattr(Axes, "set_xlim", record_set_xlim)
    monkeypatch.setattr(Axes, "set_ylim", record_set_ylim)
    monkeypatch.setattr(Axes, "text", record_text)
    monkeypatch.setattr(Axes, "vlines", record_vlines)
    profile = pd.DataFrame(
        {
            "radius_arcsec": [5.0, 10.0],
            "kappa_true_sum": [10.0, 20.0],
            "kappa_model_sum": [10.0, 22.0],
            "kappa_model_sum_q16": [8.0, 18.0],
            "kappa_model_sum_q84": [12.0, 26.0],
            "m2d_ratio": [1.0, 1.1],
            "m2d_ratio_q16": [0.8, 0.9],
            "m2d_ratio_median": [1.0, 1.1],
            "m2d_ratio_q84": [1.2, 1.3],
        }
    )
    center = {
        "center_mode": "smoothed_truth_kappa_peak",
        "center_catalog_id": "",
        "center_x_arcsec": 0.0,
        "center_y_arcsec": 0.0,
        "center_catalog_mag": np.nan,
        "center_smoothing_sigma_pix": 1.0,
        "center_smoothed_kappa_peak": 2.5,
    }
    image_df = pd.DataFrame(
        {
            "x_obs_arcsec": [0.0, 6.0, 9.0, np.nan],
            "y_obs_arcsec": [0.0, 0.0, 0.0, 1.0],
        }
    )
    image_radii = plotting._truth_recovery_image_aperture_radii(image_df, center)

    plotting._plot_truth_recovery_m2d_aperture_ratio(
        tmp_path,
        profile,
        center,
        image_radii_arcsec=image_radii,
    )

    assert (tmp_path / "truth_recovery_m2d_aperture_ratio.pdf").exists()
    assert len(fill_calls) == 2
    np.testing.assert_allclose(fill_calls[0]["y1"], [8.0, 18.0])
    np.testing.assert_allclose(fill_calls[0]["y2"], [12.0, 26.0])
    np.testing.assert_allclose(fill_calls[1]["y1"], [-0.2, -0.1])
    np.testing.assert_allclose(fill_calls[1]["y2"], [0.2, 0.3])
    assert plot_labels[:2] == ["model", "truth"]
    assert "log" in set_xscale_calls
    assert "log" in set_yscale_calls
    assert (5.0, None) in set_xlim_calls
    assert (-0.1, 0.1) in set_ylim_calls
    assert len(vline_calls) == 1
    np.testing.assert_allclose(vline_calls[0], [6.0, 9.0])
    assert any("smoothed truth kappa peak" in text for text in text_calls)
    assert any("smoothing sigma=1 pix" in text for text in text_calls)


def test_posterior_truth_grid_quantiles_reuses_cached_model_grids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            latent = plotting.jnp.asarray(packed_state["latent"], dtype=plotting.jnp.float64)
            a00 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            a11 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return a00, zeros, zeros, a11

    results = PosteriorResults(
        samples=np.asarray([[0.1], [0.2], [0.3]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    evaluator = FakeEvaluator()
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_fits = {"kappa": "kappa.fits", "gamma1": "gamma1.fits", "gamma2": "gamma2.fits"}

    all_quantiles, _x, _y = plotting._posterior_truth_grid_quantiles(
        tmp_path,
        evaluator,
        results,
        wcs,
        (2, 2),
        9.0,
        source_truth_fits=source_fits,
        quantities=("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu"),
        cache=cache,
    )
    def fail_if_recomputed(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        raise AssertionError("cached truth grids should not be recomputed")

    monkeypatch.setattr(plotting, "_truth_grid_jax_bulk_quantities_for_draw", fail_if_recomputed)
    kappa_quantiles, _x2, _y2 = plotting._posterior_truth_grid_quantiles(
        tmp_path,
        evaluator,
        results,
        wcs,
        (2, 2),
        9.0,
        source_truth_fits=source_fits,
        quantities=("kappa",),
        cache=cache,
    )

    assert kappa_quantiles["kappa"]["median"].dtype == np.float64
    np.testing.assert_allclose(kappa_quantiles["kappa"]["median"], all_quantiles["kappa"]["median"])


def test_truth_grid_posterior_samples_use_seeded_chain_stratified_selection() -> None:
    grouped = np.arange(3 * 4, dtype=float).reshape(3, 4, 1)
    grouped[1, 2, 0] = np.nan
    results = PosteriorResults(
        samples=grouped.reshape(-1, 1),
        grouped_samples=grouped,
        log_prob=np.zeros(12, dtype=float),
        accept_prob=np.zeros(12, dtype=float),
        diverging=np.zeros(12, dtype=bool),
        num_steps=np.zeros(12, dtype=int),
        warmup_steps=0,
        sample_steps=4,
        num_chains=3,
    )

    samples_a, indices_a, mode_a = plotting._select_truth_grid_posterior_samples(results, max_draws=5, seed=12345)
    samples_b, indices_b, mode_b = plotting._select_truth_grid_posterior_samples(results, max_draws=5, seed=12345)
    samples_c, indices_c, mode_c = plotting._select_truth_grid_posterior_samples(results, max_draws=5, seed=54321)

    assert mode_a == plotting.TRUTH_GRID_DRAW_SELECTION_GROUPED_RANDOM
    assert mode_b == mode_a
    assert mode_c == mode_a
    np.testing.assert_allclose(samples_b, samples_a)
    pd.testing.assert_frame_equal(indices_b, indices_a)
    assert len(indices_a) == 5
    assert indices_a["truth_grid_draw_seed"].unique().tolist() == [12345]
    assert indices_a.groupby("chain_index").size().to_dict() == {0: 2, 1: 2, 2: 1}
    assert indices_a["draw_index"].between(0, 3).all()
    assert indices_a["flat_index"].tolist() == (
        indices_a["chain_index"].to_numpy(dtype=int) * 4 + indices_a["draw_index"].to_numpy(dtype=int)
    ).tolist()
    assert indices_c["draw_index"].tolist() != indices_a["draw_index"].tolist()
    assert np.isfinite(samples_a).all()


def test_truth_grid_posterior_samples_all_and_flat_fallback_are_deterministic() -> None:
    samples = np.asarray([[0.0], [1.0], [np.nan], [3.0], [4.0]], dtype=float)
    results = PosteriorResults(
        samples=samples,
        log_prob=np.zeros(5, dtype=float),
        accept_prob=np.zeros(5, dtype=float),
        diverging=np.zeros(5, dtype=bool),
        num_steps=np.zeros(5, dtype=int),
        warmup_steps=0,
        sample_steps=5,
        num_chains=1,
    )

    all_samples, all_indices, all_mode = plotting._select_truth_grid_posterior_samples(results, max_draws=None, seed=7)
    selected_a, indices_a, mode_a = plotting._select_truth_grid_posterior_samples(results, max_draws=2, seed=7)
    selected_b, indices_b, mode_b = plotting._select_truth_grid_posterior_samples(results, max_draws=2, seed=7)

    np.testing.assert_allclose(all_samples[:, 0], [0.0, 1.0, 3.0, 4.0])
    assert all_mode == plotting.TRUTH_GRID_DRAW_SELECTION_ALL
    assert all_indices["flat_index"].tolist() == [0, 1, 3, 4]
    assert mode_a == plotting.TRUTH_GRID_DRAW_SELECTION_FLAT_RANDOM
    assert mode_b == mode_a
    np.testing.assert_allclose(selected_b, selected_a)
    pd.testing.assert_frame_equal(indices_b, indices_a)
    assert indices_a["truth_grid_draw_seed"].unique().tolist() == [7]
    assert indices_a["chain_index"].tolist() == [-1, -1]
    assert indices_a["draw_index"].tolist() == [-1, -1]
    assert set(indices_a["flat_index"]).issubset({0, 1, 3, 4})


def test_posterior_truth_grid_quantiles_uses_jax_bulk_draw_backend(tmp_path: Path) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}
            self.build_count = 0

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            self.build_count += 1
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            latent = plotting.jnp.asarray(packed_state["latent"], dtype=plotting.jnp.float64)
            a00 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            a11 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return a00, zeros, zeros, a11

    results = PosteriorResults(
        samples=np.asarray([[0.1], [0.2], [0.3]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )
    evaluator = FakeEvaluator()

    class FakeProgress:
        def __init__(self) -> None:
            self.advanced = 0

        def add_task(self, _description: str, total: int) -> int:
            assert total == 3
            return 1

        def advance(self, _task_id: int, advance: int = 1) -> None:
            self.advanced += int(advance)

        def update(self, _task_id: int, **_kwargs: Any) -> None:
            pass

    progress = FakeProgress()

    quantiles, _x, _y = plotting._posterior_truth_grid_quantiles(
        tmp_path,
        evaluator,
        results,
        wcs,
        (2, 2),
        9.0,
        source_truth_fits={"kappa": "kappa.fits"},
        quantities=("kappa", "detA", "mu"),
        truth_grid_mode="posterior",
        progress=progress,
    )

    np.testing.assert_allclose(quantiles["kappa"]["median"], np.full((2, 2), 0.2))
    np.testing.assert_allclose(quantiles["detA"]["median"], np.full((2, 2), 0.64))
    np.testing.assert_allclose(quantiles["mu"]["median"], np.full((2, 2), 1.0 / 0.64))
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_summary.csv")
    assert set(summary["truth_grid_backend"]) == {"jax_bulk_hessian"}
    assert set(summary["truth_grid_draw_seed"]) == {12345}
    assert set(summary["truth_grid_draw_selection"]) == {plotting.TRUTH_GRID_DRAW_SELECTION_ALL}
    assert set(summary["chunk_count"]) == {1}
    assert set(summary["chunk_pixels"]) == {4}
    draw_indices = pd.read_csv(tmp_path / "tables" / "truth_recovery_draw_indices.csv")
    assert draw_indices["selection_order"].tolist() == [0, 1, 2]
    assert draw_indices["truth_grid_draw_seed"].unique().tolist() == [12345]
    assert draw_indices["flat_index"].tolist() == [0, 1, 2]
    assert evaluator.build_count == 3
    assert progress.advanced == 3


def test_posterior_truth_grid_quantiles_requires_jax_bulk_backend(tmp_path: Path) -> None:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class MissingTraceSafeEvaluator:
        def __init__(self) -> None:
            self.state = state

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _build_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, packed_state, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            ones = plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return ones, zeros, zeros, ones
    results = PosteriorResults(
        samples=np.asarray([[0.1], [0.2]], dtype=float),
        log_prob=np.zeros(2, dtype=float),
        accept_prob=np.zeros(2, dtype=float),
        diverging=np.zeros(2, dtype=bool),
        num_steps=np.zeros(2, dtype=int),
        warmup_steps=0,
        sample_steps=2,
        num_chains=1,
    )

    with pytest.raises(RuntimeError, match="JAX bulk lensing Jacobian backend"):
        plotting._posterior_truth_grid_quantiles(
            tmp_path,
            MissingTraceSafeEvaluator(),  # type: ignore[arg-type]
            results,
            wcs,
            (2, 2),
            9.0,
            source_truth_fits={"kappa": "kappa.fits"},
            quantities=("kappa",),
            truth_grid_mode="posterior",
        )

    assert not (tmp_path / "truth_recovery_kappa_model_median.fits").exists()
    assert not (tmp_path / "fits" / "truth_recovery_kappa_model_median.fits").exists()


def test_kappa_truth_diagnostics_requires_posterior_samples(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa_true.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32), header=wcs.to_header()).writeto(kappa_path)
    evaluator = SimpleNamespace(state=SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0)))
    results = PosteriorResults(
        samples=np.empty((0, 1), dtype=float),
        log_prob=np.empty(0, dtype=float),
        accept_prob=np.empty(0, dtype=float),
        diverging=np.empty(0, dtype=bool),
        num_steps=np.empty(0, dtype=int),
        warmup_steps=0,
        sample_steps=0,
        num_chains=1,
    )

    with pytest.raises(ValueError, match="requires non-empty finite posterior samples"):
        plotting._plot_kappa_truth_diagnostics(
            tmp_path,
            evaluator,
            results,
            kappa_path,
            caustic_source_redshift=9.0,
        )


def test_mu_truth_quantiles_compute_magnification_per_sample(tmp_path: Path) -> None:
    def write_truth_fits(path: Path, data: np.ndarray) -> None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=wcs.to_header()).writeto(path)

    kappa_path = tmp_path / "kappa.fits"
    gammax_path = tmp_path / "gammax.fits"
    gammay_path = tmp_path / "gammay.fits"
    write_truth_fits(kappa_path, np.zeros((2, 2), dtype=float))
    write_truth_fits(gammax_path, np.zeros((2, 2), dtype=float))
    write_truth_fits(gammay_path, np.zeros((2, 2), dtype=float))

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            latent = plotting.jnp.asarray(packed_state["latent"], dtype=plotting.jnp.float64)
            a00 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            a11 = 1.0 - latent * plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return a00, zeros, zeros, a11

    results = PosteriorResults(
        samples=np.asarray([[0.2], [1.2], [2.0]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )

    plotting._plot_abs_mu_truth_diagnostics(
        tmp_path,
        FakeEvaluator(),
        results,
        kappa_path,
        gammax_path,
        gammay_path,
        caustic_source_redshift=9.0,
        truth_grid_mode="posterior",
    )

    assert not (tmp_path / "abs_mu_model_median.fits").exists()
    assert not (tmp_path / "truth_recovery_abs_mu_model_median.fits").exists()
    with fits.open(tmp_path / "fits" / "truth_recovery_abs_mu_model_median.fits") as hdul:
        abs_mu_median = float(np.asarray(hdul[0].data, dtype=float)[0, 0])
    # Per-sample |mu| values are [1.5625, 25, 1], so the median is 1.5625.
    # Taking |mu| after the median kappa=1.2 would incorrectly give 25.
    assert abs_mu_median == pytest.approx(1.5625)


def test_plot_absolute_mu_truth_diagnostics_samples_every_pixel_and_writes_outputs(tmp_path: Path) -> None:
    def write_truth_fits(path: Path, data: np.ndarray) -> None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=wcs.to_header()).writeto(path)

    kappa_path = tmp_path / "kappa.fits"
    gammax_path = tmp_path / "gammax.fits"
    gammay_path = tmp_path / "gammay.fits"
    write_truth_fits(kappa_path, np.zeros((3, 3), dtype=float))
    write_truth_fits(gammax_path, np.zeros((3, 3), dtype=float))
    write_truth_fits(gammay_path, np.zeros((3, 3), dtype=float))

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float) + 1.0

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {"latent": plotting.jnp.asarray(sample_latent, dtype=plotting.jnp.float64)[0]}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, packed_state, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            ones = plotting.jnp.ones_like(x_array)
            zeros = plotting.jnp.zeros_like(x_array)
            return ones, zeros, zeros, ones

    evaluator = FakeEvaluator()
    results = PosteriorResults(
        samples=np.asarray([[4.0], [5.0], [6.0]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )

    plotting._plot_abs_mu_truth_diagnostics(
        tmp_path,
        evaluator,
        results,
        kappa_path,
        gammax_path,
        gammay_path,
        caustic_source_redshift=9.0,
        truth_grid_mode="posterior",
    )

    assert not (tmp_path / "mu_comparison.pdf").exists()
    assert not (tmp_path / "mu_model.pdf").exists()
    assert not (tmp_path / "mu_fractional_residual.pdf").exists()
    assert not (tmp_path / "mu_recovery.pdf").exists()
    assert not (tmp_path / "critical_line_recovery.pdf").exists()
    assert not (tmp_path / "abs_mu_model_q16.fits").exists()
    assert not (tmp_path / "abs_mu_model_median.fits").exists()
    assert not (tmp_path / "abs_mu_model_q84.fits").exists()
    assert not (tmp_path / "truth_recovery_abs_mu_model_q16.fits").exists()
    assert not (tmp_path / "truth_recovery_abs_mu_model_median.fits").exists()
    assert not (tmp_path / "truth_recovery_abs_mu_model_q84.fits").exists()
    assert not (tmp_path / "tables" / "mu_recovery_binned.csv").exists()
    assert not (tmp_path / "tables" / "mu_recovery_summary.csv").exists()
    assert not (tmp_path / "tables" / "truth_grid_summary.csv").exists()
    assert not (tmp_path / "truth_recovery_mu_model.pdf").exists()
    assert not (tmp_path / "truth_recovery_mu_fractional_residual.pdf").exists()
    assert (tmp_path / "truth_recovery_mu_model_truth_fractional_residual.pdf").exists()
    assert (tmp_path / "truth_recovery_mu_recovery.pdf").exists()
    assert (tmp_path / "truth_recovery_critical_line_recovery.pdf").exists()
    assert (tmp_path / "fits" / "truth_recovery_abs_mu_model_q16.fits").exists()
    assert (tmp_path / "fits" / "truth_recovery_abs_mu_model_median.fits").exists()
    assert (tmp_path / "fits" / "truth_recovery_abs_mu_model_q84.fits").exists()
    binned = pd.read_csv(tmp_path / "tables" / "truth_recovery_mu_recovery_binned.csv")
    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_mu_recovery_summary.csv")
    truth_grid_summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_summary.csv")
    assert list(binned.columns) == [
        "bin_index",
        "abs_mu_true_min",
        "abs_mu_true_max",
        "abs_mu_true_center",
        "sample_count",
        "abs_mu_model_q16",
        "abs_mu_model_median",
        "abs_mu_model_q84",
    ]
    assert summary.loc[0, "total_pixel_count"] == 9
    assert summary.loc[0, "finite_pixel_count"] == 9
    assert set(truth_grid_summary["quantity"]) >= {"kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu"}
    assert set(truth_grid_summary["draw_count_used"]) == {3}
    assert set(truth_grid_summary["truth_grid_backend"]) == {"jax_bulk_hessian"}


def test_plot_critical_line_recovery_contours_truth_and_model(monkeypatch: Any, tmp_path: Path) -> None:
    from matplotlib.axes import Axes

    x_arcsec, y_arcsec = np.meshgrid(
        np.asarray([-1.0, 0.0, 1.0], dtype=float),
        np.asarray([-1.0, 0.0, 1.0], dtype=float),
    )
    truth_determinant = x_arcsec.copy()
    model_determinant = y_arcsec.copy()
    contour_calls: list[dict[str, Any]] = []
    original_contour = Axes.contour
    invert_calls: list[bool] = []

    def record_contour(self: Axes, *args: Any, **kwargs: Any) -> Any:
        contour_calls.append(
            {
                "levels": list(kwargs.get("levels", [])),
                "colors": list(kwargs.get("colors", [])),
                "linestyles": list(kwargs.get("linestyles", [])),
            }
        )
        return original_contour(self, *args, **kwargs)

    def record_invert_xaxis(self: Axes) -> None:
        invert_calls.append(True)

    monkeypatch.setattr(Axes, "contour", record_contour)
    monkeypatch.setattr(Axes, "invert_xaxis", record_invert_xaxis)

    plotting._plot_critical_line_recovery_from_grid(
        tmp_path,
        truth_determinant,
        model_determinant,
        x_arcsec,
        y_arcsec,
        z_source=9.0,
    )

    assert not (tmp_path / "critical_line_recovery.pdf").exists()
    assert (tmp_path / "truth_recovery_critical_line_recovery.pdf").exists()
    assert (tmp_path / "truth_recovery_critical_line_recovery.pdf").stat().st_size > 0
    assert contour_calls == [
        {"levels": [0.0], "colors": ["black"], "linestyles": ["-"]},
        {"levels": [0.0], "colors": ["tab:blue"], "linestyles": ["--"]},
    ]
    assert invert_calls == []


def test_absolute_mu_recovery_overlays_observed_image_points(monkeypatch: Any, tmp_path: Path) -> None:
    def write_truth_fits(path: Path, data: np.ndarray) -> None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=wcs.to_header()).writeto(path)

    kappa_path = tmp_path / "kappa.fits"
    gammax_path = tmp_path / "gammax.fits"
    gammay_path = tmp_path / "gammay.fits"
    write_truth_fits(kappa_path, np.asarray([[0.0, 0.5], [0.0, 0.0]], dtype=float))
    write_truth_fits(gammax_path, np.zeros((2, 2), dtype=float))
    write_truth_fits(gammay_path, np.zeros((2, 2), dtype=float))

    state = SimpleNamespace(z_lens=0.3, reference=(3, 0.0, 0.0), parameter_specs=[])

    class FakeEvaluator:
        def __init__(self) -> None:
            self.state = state
            self.exact_models_by_z: dict[float, Any] = {}

        def reported_physical_to_latent_parameter_vector(self, theta: np.ndarray) -> np.ndarray:
            return np.asarray(theta, dtype=float)

        def _get_exact_model_solver(self, _z_source: float) -> tuple[Any, None]:
            raise AssertionError("JAX bulk truth-grid path should not request the Python exact model.")

        def _build_truth_grid_packed_lens_state(self, _sample_latent: Any, _z_source: float) -> dict[str, Any]:
            return {}

        def _flat_lensing_jacobian_for_components(
            self,
            x: Any,
            y: Any,
            packed_state: dict[str, Any],
            component_indices: Any = None,
        ) -> tuple[Any, Any, Any, Any]:
            del y, packed_state, component_indices
            x_array = plotting.jnp.asarray(x, dtype=plotting.jnp.float64)
            abs_mu = plotting.jnp.ones_like(x_array)
            abs_mu = plotting.jnp.where(plotting.jnp.isclose(x_array, 0.0), 3.0, abs_mu)
            abs_mu = plotting.jnp.where(plotting.jnp.isclose(x_array, -1.0), 4.0, abs_mu)
            kappa = 1.0 - plotting.jnp.sqrt(1.0 / abs_mu)
            zeros = plotting.jnp.zeros_like(x_array)
            a00 = 1.0 - kappa
            return a00, zeros, zeros, a00

    captured: dict[str, Any] = {}

    def fake_plot_quantity_recovery(_recovery: dict[str, Any], output_path: Path, **kwargs: Any) -> None:
        captured.update(kwargs)
        Path(output_path).touch()

    monkeypatch.setattr(plotting, "_plot_abs_mu_true_comparison_from_grid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plotting, "_plot_quantity_recovery", fake_plot_quantity_recovery)
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1"],
            "image_label": ["1.a", "1.b"],
            "x_obs_arcsec": [0.0, -1.0],
            "y_obs_arcsec": [0.0, 0.0],
        }
    )

    results = PosteriorResults(
        samples=np.asarray([[4.0], [5.0], [6.0]], dtype=float),
        log_prob=np.zeros(3, dtype=float),
        accept_prob=np.zeros(3, dtype=float),
        diverging=np.zeros(3, dtype=bool),
        num_steps=np.zeros(3, dtype=int),
        warmup_steps=0,
        sample_steps=3,
        num_chains=1,
    )

    plotting._plot_abs_mu_truth_diagnostics(
        tmp_path,
        FakeEvaluator(),
        results,
        kappa_path,
        gammax_path,
        gammay_path,
        caustic_source_redshift=9.0,
        image_df=image_df,
    )

    image_points = captured["image_points"]
    np.testing.assert_allclose(image_points["true_value"].to_numpy(dtype=float), [1.0, 4.0])
    np.testing.assert_allclose(image_points["model_value"].to_numpy(dtype=float), [3.0, 4.0])
    assert image_points["image_label"].tolist() == ["1.a", "1.b"]


def test_abs_mu_model_truth_fractional_residual_three_panel_plot(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeColorbar:
        def __init__(self) -> None:
            self.label: str | None = None
            self.ax = type(
                "FakeColorbarAxis",
                (),
                {
                    "xaxis": type(
                        "FakeXAxis",
                        (),
                        {
                            "set_label_position": lambda self, _position: None,
                            "set_ticks_position": lambda self, _position: None,
                        },
                    )()
                },
            )()

        def set_label(self, label: str) -> None:
            self.label = label

    class FakeAxis:
        def __init__(self, x0: float, y1: float, width: float) -> None:
            self.pcolormesh_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
            self.inverted = False
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self._position = type("FakeBbox", (), {"x0": x0, "y1": y1, "width": width})()

        def pcolormesh(self, x: Any, y: Any, data: Any, **kwargs: Any) -> str:
            self.pcolormesh_calls.append(
                (
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    np.ma.asarray(data).filled(np.nan),
                    dict(kwargs),
                )
            )
            return f"mesh-{len(self.pcolormesh_calls)}"

        def invert_xaxis(self) -> None:
            self.inverted = True

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def get_position(self) -> Any:
            return self._position

    class FakeColorbarAxis:
        def __init__(self, bounds: list[float]) -> None:
            self.bounds = bounds

    class FakeFig:
        def __init__(self) -> None:
            self.colorbars: list[tuple[str, FakeColorbarAxis, FakeColorbar, dict[str, Any]]] = []
            self.added_axes: list[FakeColorbarAxis] = []
            self.saved_paths: list[Path] = []
            self.title: str | None = None
            self.subplots_adjust_kwargs: dict[str, Any] | None = None

        def subplots_adjust(self, **kwargs: Any) -> None:
            self.subplots_adjust_kwargs = dict(kwargs)

        def add_axes(self, bounds: list[float]) -> FakeColorbarAxis:
            axis = FakeColorbarAxis(bounds)
            self.added_axes.append(axis)
            return axis

        def colorbar(self, image: str, cax: FakeColorbarAxis, **kwargs: Any) -> FakeColorbar:
            assert kwargs["orientation"] == "horizontal"
            colorbar = FakeColorbar()
            self.colorbars.append((image, cax, colorbar, dict(kwargs)))
            return colorbar

        def suptitle(self, title: str) -> None:
            self.title = title

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            self.saved_paths.append(Path(path))
            Path(path).touch()

    axes = np.asarray(
        [
            FakeAxis(0.10, 0.80, 0.22),
            FakeAxis(0.40, 0.80, 0.22),
            FakeAxis(0.70, 0.80, 0.22),
        ],
        dtype=object,
    )
    fig = FakeFig()
    subplots_calls: list[dict[str, Any]] = []

    def fake_subplots(*_args: Any, **kwargs: Any) -> tuple[FakeFig, np.ndarray]:
        subplots_calls.append(dict(kwargs))
        return fig, axes

    monkeypatch.setattr(plotting.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    x_arcsec = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    y_arcsec = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    plotting._plot_abs_mu_model_truth_fractional_residual_from_grid(
        tmp_path,
        np.asarray([[1.0, 2.0], [0.0, np.nan]], dtype=float),
        np.asarray([[2.0, 1.0], [3.0, 4.0]], dtype=float),
        x_arcsec,
        y_arcsec,
        z_source=9.0,
        cap=40.0,
    )

    assert (tmp_path / "truth_recovery_mu_model_truth_fractional_residual.pdf").exists()
    assert fig.saved_paths == [tmp_path / "truth_recovery_mu_model_truth_fractional_residual.pdf"]
    assert fig.title is None
    assert subplots_calls[0]["sharey"] is True
    assert fig.subplots_adjust_kwargs == {"top": 0.82, "wspace": 0.08}
    assert [axis.bounds for axis in fig.added_axes] == [
        [0.10, 0.8180000000000001, 0.22, 0.026],
        [0.40, 0.8180000000000001, 0.22, 0.026],
        [0.70, 0.8180000000000001, 0.22, 0.026],
    ]
    assert [colorbar.label for _image, _axis, colorbar, _kwargs in fig.colorbars] == [
        r"$|\mu_{\rm model}|$",
        r"$|\mu_{\rm true}|$",
        r"$(|\mu_{\rm model}| - |\mu_{\rm true}|) / |\mu_{\rm true}|$",
    ]
    for axis_index, axis in enumerate(axes):
        assert axis.inverted is False
        assert axis.xlabel == "x [arcsec]"
        assert axis.ylabel == ("y [arcsec]" if axis_index == 0 else None)
        assert len(axis.pcolormesh_calls) == 1
        np.testing.assert_allclose(axis.pcolormesh_calls[0][0], x_arcsec)
        np.testing.assert_allclose(axis.pcolormesh_calls[0][1], y_arcsec)
        kwargs = axis.pcolormesh_calls[0][3]
        assert kwargs["shading"] == "nearest"
        assert kwargs["edgecolors"] == "none"
        assert kwargs["linewidth"] == pytest.approx(0.0)
        assert kwargs["antialiased"] is False
        assert kwargs["rasterized"] is True
    assert axes[0].pcolormesh_calls[0][3]["cmap"] == "viridis"
    assert axes[0].pcolormesh_calls[0][3]["vmin"] == pytest.approx(0.0)
    assert axes[0].pcolormesh_calls[0][3]["vmax"] == pytest.approx(40.0)
    assert axes[1].pcolormesh_calls[0][3]["cmap"] == "viridis"
    assert axes[1].pcolormesh_calls[0][3]["vmin"] == pytest.approx(0.0)
    assert axes[1].pcolormesh_calls[0][3]["vmax"] == pytest.approx(40.0)
    residual_norm = axes[2].pcolormesh_calls[0][3]["norm"]
    assert isinstance(residual_norm, plotting.Normalize)
    assert residual_norm.vmin == pytest.approx(-1.0)
    assert residual_norm.vmax == pytest.approx(2.0)
    assert residual_norm.clip is True
    assert residual_norm(0.0) == pytest.approx(1.0 / 3.0)
    residual_cmap = axes[2].pcolormesh_calls[0][3]["cmap"]
    assert isinstance(residual_cmap, plotting.LinearSegmentedColormap)
    assert residual_cmap.name.startswith("RdBu_r_zero_at_")
    np.testing.assert_allclose(
        axes[2].pcolormesh_calls[0][2],
        [[1.0, -0.5], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_absolute_mu_comparison_labels_are_not_capped(monkeypatch: Any, tmp_path: Path) -> None:
    three_panel_calls: list[dict[str, Any]] = []

    def fake_three_panel(plot_dir: Path, *_args: Any, **kwargs: Any) -> None:
        three_panel_calls.append({"plot_dir": Path(plot_dir), "kwargs": dict(kwargs)})
        (Path(plot_dir) / "truth_recovery_mu_model_truth_fractional_residual.pdf").touch()

    monkeypatch.setattr(plotting, "_plot_abs_mu_model_truth_fractional_residual_from_grid", fake_three_panel)

    plotting._plot_abs_mu_true_comparison_from_grid(
        tmp_path,
        np.asarray([[1.0, 50.0], [0.0, np.nan]], dtype=float),
        np.asarray([[2.0, 25.0], [5.0, 60.0]], dtype=float),
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        z_source=9.0,
    )

    assert not (tmp_path / "mu_comparison.pdf").exists()
    assert not (tmp_path / "mu_model.pdf").exists()
    assert not (tmp_path / "mu_fractional_residual.pdf").exists()
    assert not (tmp_path / "truth_recovery_mu_model.pdf").exists()
    assert not (tmp_path / "truth_recovery_mu_fractional_residual.pdf").exists()
    assert (tmp_path / "truth_recovery_mu_model_truth_fractional_residual.pdf").exists()
    assert len(three_panel_calls) == 1
    assert three_panel_calls[0]["plot_dir"] == tmp_path
    np.testing.assert_allclose(
        three_panel_calls[0]["kwargs"]["fractional_residual"],
        [[1.0, -0.5], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_absolute_mu_recovery_keeps_raw_values_with_fixed_axis_and_uncapped_labels(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    class FakeColorbar:
        def set_label(self, label: str) -> None:
            return None

    class FakeAxis:
        def __init__(self) -> None:
            self.xlim: tuple[float, float] | None = None
            self.ylim: tuple[float, float] | None = None
            self.xlabel: str | None = None
            self.ylabel: str | None = None
            self.title: str | None = None
            self.plots: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

        def pcolormesh(self, *_args: Any, **_kwargs: Any) -> str:
            return "mesh"

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float), dict(kwargs)))

        def text(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlim(self, *args: Any) -> None:
            self.xlim = tuple(float(value) for value in args)

        def set_ylim(self, *args: Any) -> None:
            self.ylim = tuple(float(value) for value in args)

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, label: str) -> None:
            self.xlabel = label

        def set_ylabel(self, label: str) -> None:
            self.ylabel = label

        def set_title(self, title: str) -> None:
            self.title = title

        def legend(self, **_kwargs: Any) -> None:
            return None

        @property
        def transAxes(self) -> object:
            return object()

    class FakeFig:
        def colorbar(self, *_args: Any, **_kwargs: Any) -> FakeColorbar:
            return FakeColorbar()

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    plotting._write_abs_mu_recovery_from_grid(
        tmp_path,
        np.asarray([[1.0, 10.0, 28.0, 35.0, np.nan]], dtype=float),
        np.asarray([[1.1, 20.0, 60.0, 100.0, 5.0]], dtype=float),
        z_source=9.0,
    )

    summary = pd.read_csv(tmp_path / "tables" / "truth_recovery_mu_recovery_summary.csv")
    binned = pd.read_csv(tmp_path / "tables" / "truth_recovery_mu_recovery_binned.csv")
    assert summary.loc[0, "abs_mu_true_max"] == pytest.approx(35.0)
    assert summary.loc[0, "abs_mu_model_max"] == pytest.approx(100.0)
    assert summary.loc[0, "finite_pixel_count"] == 4
    residual = np.asarray([0.1, 10.0, 32.0, 65.0], dtype=float)
    assert summary.loc[0, "abs_mu_bias_median"] == pytest.approx(np.median(residual))
    assert summary.loc[0, "abs_mu_spread_nmad"] == pytest.approx(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    assert summary.loc[0, "abs_mu_rmse"] == pytest.approx(np.sqrt(np.mean(np.square(residual))))
    assert int(binned["sample_count"].sum()) == 3
    assert float(binned["abs_mu_true_min"].min()) >= 0.0
    assert float(binned["abs_mu_true_max"].max()) <= plotting.ABSOLUTE_MAGNIFICATION_RECOVERY_AXIS_MAX
    assert float(binned["abs_mu_model_q84"].max()) == pytest.approx(60.0)
    assert axis.xlim == pytest.approx((0.0, plotting.ABSOLUTE_MAGNIFICATION_RECOVERY_AXIS_MAX))
    assert axis.ylim == pytest.approx((0.0, plotting.ABSOLUTE_MAGNIFICATION_RECOVERY_AXIS_MAX))
    assert axis.title is None
    assert all("capped" not in str(text).lower() for text in [axis.xlabel, axis.ylabel, axis.title])


def test_quantity_recovery_residual_statistics_ignore_nonfinite_values() -> None:
    stats = plotting._quantity_recovery_residual_statistics(
        np.asarray([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float),
        np.asarray([2.0, 4.0, 10.0, np.inf, 8.0], dtype=float),
    )
    residual = np.asarray([1.0, 2.0, 3.0], dtype=float)

    assert stats["bias_median"] == pytest.approx(2.0)
    assert stats["spread_nmad"] == pytest.approx(1.4826 * np.median(np.abs(residual - np.median(residual))))
    assert stats["rmse"] == pytest.approx(np.sqrt(np.mean(np.square(residual))))


def test_quantity_recovery_plot_uses_sigma_percentiles(monkeypatch: Any, tmp_path: Path) -> None:
    true_grid = np.ones((10, 10), dtype=float)
    model_grid = np.linspace(0.4, 1.2, true_grid.size, dtype=float).reshape(true_grid.shape)
    recovery = plotting._quantity_recovery_reduced(
        true_grid,
        model_grid,
        "kappa",
        histogram_bins=4,
        stat_bins=1,
        limits=(0.0, 2.0),
    )

    bin_table = recovery["bin_table"]
    summary_table = recovery["summary_table"]
    assert list(bin_table.columns) == [
        "bin_index",
        "kappa_true_min",
        "kappa_true_max",
        "kappa_true_center",
        "sample_count",
        "kappa_model_q16",
        "kappa_model_median",
        "kappa_model_q84",
    ]
    assert bin_table.loc[0, "kappa_model_q16"] == pytest.approx(np.percentile(model_grid, 16.0))
    assert bin_table.loc[0, "kappa_model_q84"] == pytest.approx(np.percentile(model_grid, 84.0))
    assert summary_table.loc[0, "kappa_fractional_residual_q2p5"] == pytest.approx(
        np.percentile(model_grid - true_grid, 2.5)
    )
    assert summary_table.loc[0, "kappa_fractional_residual_q97p5"] == pytest.approx(
        np.percentile(model_grid - true_grid, 97.5)
    )
    one_sigma_width = 0.5 * (
        summary_table.loc[0, "kappa_fractional_residual_q84"]
        - summary_table.loc[0, "kappa_fractional_residual_q16"]
    )
    assert summary_table.loc[0, "kappa_fractional_residual_sigma"] == pytest.approx(one_sigma_width)
    residual = model_grid.reshape(-1) - true_grid.reshape(-1)
    assert summary_table.loc[0, "kappa_bias_median"] == pytest.approx(np.median(residual))
    assert summary_table.loc[0, "kappa_spread_nmad"] == pytest.approx(
        1.4826 * np.median(np.abs(residual - np.median(residual)))
    )
    assert summary_table.loc[0, "kappa_rmse"] == pytest.approx(np.sqrt(np.mean(np.square(residual))))

    class FakeColorbar:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def set_label(self, label: str) -> None:
            self.labels.append(label)

    class FakeAxis:
        def __init__(self) -> None:
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []
            self.texts: list[str] = []

        def pcolormesh(self, *_args: Any, **_kwargs: Any) -> str:
            return "mesh"

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float), kwargs))

        def text(self, *args: Any, **_kwargs: Any) -> None:
            self.texts.append(str(args[2]))

        def set_xlim(self, *_args: Any) -> None:
            return None

        def set_ylim(self, *_args: Any) -> None:
            return None

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, *_args: Any) -> None:
            return None

        def set_ylabel(self, *_args: Any) -> None:
            return None

        def set_title(self, *_args: Any) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

        @property
        def transAxes(self) -> object:
            return object()

    class FakeFig:
        def colorbar(self, *_args: Any, **_kwargs: Any) -> FakeColorbar:
            return FakeColorbar()

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    output_path = tmp_path / "truth_recovery_kappa_recovery.pdf"
    plotting._plot_quantity_recovery(
        recovery,
        output_path,
        quantity="kappa",
        true_label="true",
        model_label="model",
        title="recovery",
    )

    assert output_path.exists()
    labels = [plot_kwargs.get("label") for _x, _y, plot_kwargs in axis.plots]
    assert r"1$\sigma$ recovery" in labels
    assert r"2$\sigma$ recovery" in labels
    assert r"1$\sigma$ (16th/84th)" in labels
    assert axis.texts and "bias:" in axis.texts[0]
    assert "NMAD:" in axis.texts[0]
    assert "RMSE:" in axis.texts[0]
    one_sigma_slopes = sorted(
        y[-1] / x[-1]
        for x, y, kwargs in axis.plots
        if kwargs.get("linestyle") == "--"
    )
    two_sigma_slopes = sorted(
        y[-1] / x[-1]
        for x, y, kwargs in axis.plots
        if kwargs.get("linestyle") == ":"
    )
    one_sigma_upper = 1.0 + one_sigma_width
    two_sigma_upper = 1.0 + 2.0 * one_sigma_width
    assert one_sigma_slopes == pytest.approx([1.0 / one_sigma_upper, one_sigma_upper])
    assert two_sigma_slopes == pytest.approx([1.0 / two_sigma_upper, two_sigma_upper])
    assert np.prod(one_sigma_slopes) == pytest.approx(1.0)
    assert np.prod(two_sigma_slopes) == pytest.approx(1.0)


def test_quantity_recovery_plot_overlays_observed_image_points(monkeypatch: Any, tmp_path: Path) -> None:
    recovery = plotting._quantity_recovery_reduced(
        np.asarray([[0.5, 1.5], [2.5, np.nan]], dtype=float),
        np.asarray([[0.6, 1.6], [1.0, 0.2]], dtype=float),
        "kappa",
        histogram_bins=4,
        stat_bins=1,
        limits=(0.0, 2.0),
    )
    image_points = pd.DataFrame(
        {
            "true_value": [0.25, 1.5, 2.5, np.nan],
            "model_value": [0.5, 1.4, 1.5, 0.7],
        }
    )

    class FakeColorbar:
        def set_label(self, _label: str) -> None:
            return None

    class FakeAxis:
        def __init__(self) -> None:
            self.scatters: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []

        def pcolormesh(self, *_args: Any, **_kwargs: Any) -> str:
            return "mesh"

        def plot(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def text(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float), dict(kwargs)))

        def set_xlim(self, *_args: Any) -> None:
            return None

        def set_ylim(self, *_args: Any) -> None:
            return None

        def set_aspect(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set_xlabel(self, *_args: Any) -> None:
            return None

        def set_ylabel(self, *_args: Any) -> None:
            return None

        def set_title(self, *_args: Any) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

        @property
        def transAxes(self) -> object:
            return object()

    class FakeFig:
        def colorbar(self, *_args: Any, **_kwargs: Any) -> FakeColorbar:
            return FakeColorbar()

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), axis))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    plotting._plot_quantity_recovery(
        recovery,
        tmp_path / "truth_recovery_kappa_recovery.pdf",
        quantity="kappa",
        true_label="true",
        model_label="model",
        image_points=image_points,
    )

    assert len(axis.scatters) == 1
    scatter_x, scatter_y, scatter_kwargs = axis.scatters[0]
    np.testing.assert_allclose(scatter_x, [0.25, 1.5])
    np.testing.assert_allclose(scatter_y, [0.5, 1.4])
    assert scatter_kwargs["label"] == "observed images"
    assert scatter_kwargs["facecolors"] == "tab:orange"
    assert scatter_kwargs["edgecolors"] == "black"


def test_image_recovery_uses_status_colors_and_small_points(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeAxis:
        def __init__(self) -> None:
            self.scatters: list[tuple[Any, Any, dict[str, Any]]] = []
            self.errorbars: list[tuple[Any, Any, dict[str, Any]]] = []
            self.plots: list[tuple[Any, Any, dict[str, Any]]] = []

        def scatter(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.scatters.append((x, y, kwargs))

        def errorbar(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.errorbars.append((x, y, kwargs))

        def plot(self, x: Any, y: Any, **kwargs: Any) -> None:
            self.plots.append((x, y, kwargs))

        def invert_xaxis(self) -> None:
            return None

        def set_xlabel(self, _label: str) -> None:
            return None

        def set_ylabel(self, _label: str) -> None:
            return None

        def set_title(self, _title: str) -> None:
            return None

        def legend(self, **_kwargs: Any) -> None:
            return None

        def set_xticks(self, _ticks: Any) -> None:
            return None

        def set_xticklabels(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class FakeFig:
        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).touch()

    image_axis = FakeAxis()
    residual_axis = FakeAxis()
    monkeypatch.setattr(plotting.plt, "subplots", lambda *_args, **_kwargs: (FakeFig(), [image_axis, residual_axis]))
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "2", "2"],
            "image_label": ["1.1", "1.2", "2.1", "2.2"],
            "image_recovery_status": ["recovered", "not_recovered", "not_recovered", "recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "not_recovered", "point_recovered"],
            "arc_supported": [False, True, False, False],
            "x_obs_arcsec": [0.0, 2.0, 5.0, 7.0],
            "y_obs_arcsec": [0.0, 1.0, -1.0, -2.0],
            "x_model_arcsec": [0.1, 2.2, 4.8, 6.9],
            "y_model_arcsec": [0.2, 1.1, -1.1, -2.2],
            "x_model_q16": [0.0, 2.1, 4.7, 6.8],
            "x_model_q50": [0.1, 2.2, 4.8, 6.9],
            "x_model_q84": [0.2, 2.3, 4.9, 7.0],
            "y_model_q16": [0.1, 1.0, -1.2, -2.3],
            "y_model_q50": [0.2, 1.1, -1.1, -2.2],
            "y_model_q84": [0.3, 1.2, -1.0, -2.1],
            "image_residual_arcsec": [0.2, 0.3, 0.4, 0.25],
            "image_residual_q16": [0.1, 0.2, 0.3, 0.15],
            "image_residual_q50": [0.2, 0.3, 0.4, 0.25],
            "image_residual_q84": [0.3, 0.4, 0.5, 0.35],
            "arc_aware_image_residual_arcsec": [0.2, 0.18, np.nan, 0.25],
        }
    )
    extra_df = pd.DataFrame(
        {
            "family_id": ["1"],
            "extra_image_index": [1],
            "image_recovery_status": ["extra"],
            "x_model_arcsec": [8.0],
            "y_model_arcsec": [-3.0],
        }
    )

    plotting._plot_image_recovery_fit_quality(
        image_df,
        tmp_path / "image_recovery.pdf",
        extra_df,
        use_arc_aware_diagnostics=True,
    )

    assert (tmp_path / "image_recovery.pdf").exists()
    assert image_axis.scatters[0][2]["marker"] == "x"
    assert image_axis.scatters[0][2]["color"] == plotting._image_catalog_status_color("POINT_RECOVERED")
    assert image_axis.scatters[0][2]["s"] < 30
    assert image_axis.scatters[0][2]["label"] == "point recovered"
    assert image_axis.scatters[1][2]["marker"] == "x"
    assert image_axis.scatters[1][2]["color"] == plotting._image_catalog_status_color("ARC_RECOVERED")
    assert image_axis.scatters[1][2]["label"] == "arc recovered"
    assert image_axis.scatters[2][2]["marker"] == "x"
    assert image_axis.scatters[2][2]["color"] == plotting._image_catalog_status_color("MISSED")
    assert image_axis.scatters[2][2]["label"] == "not recovered"
    assert image_axis.scatters[3][2]["marker"] == "o"
    assert image_axis.scatters[3][2]["color"] == "tab:blue"
    assert image_axis.scatters[3][2]["s"] < 20
    assert image_axis.scatters[3][2]["label"] == "extra"
    assert image_axis.errorbars[0][2]["fmt"] == "o"
    np.testing.assert_allclose(
        image_axis.errorbars[0][2]["color"],
        plotting._color_with_alpha(plotting._image_catalog_status_color("POINT_RECOVERED"), 0.75),
    )
    assert image_axis.errorbars[0][2]["ecolor"][3] < image_axis.errorbars[0][2]["color"][3]
    assert image_axis.errorbars[0][2]["markersize"] < 4
    assert len(image_axis.plots) == 2


def test_fit_quality_diagnostic_plots_write_pdfs_and_merge_tables(tmp_path: Path) -> None:
    image_df = pd.DataFrame(
        {
            "family_id": ["1", "1", "2"],
            "image_label": ["1.1", "1.2", "2.1"],
            "x_obs_arcsec": [0.0, 2.0, 5.0],
            "y_obs_arcsec": [0.0, 1.0, -1.0],
            "z_source": [2.0, 2.0, 3.0],
            "radius_arcsec": [0.0, 2.2, 5.1],
            "angle_deg": [0.0, 26.6, -11.3],
            "image_residual_arcsec": [0.1, 0.3, 0.8],
            "image_residual_q50": [0.12, 0.35, 0.75],
            "residual_norm": [0.5, 1.2, 2.4],
            "residual_norm_q50": [0.6, 1.4, 2.2],
            "covered_x_1sigma": [True, True, False],
            "covered_y_1sigma": [True, False, False],
            "covered_xy_1sigma": [True, False, False],
        }
    )
    magnification_df = pd.DataFrame(
        {
            "family_id": ["1", "2", "3"],
            "image_label": ["1.1", "2.1", "3.1"],
            "magnification_model": [2.0, -5.0, 10.0],
            "magnification_model_q50": [2.2, -4.5, 9.5],
        }
    )
    merged = plotting._merge_fit_quality_with_magnification(image_df, magnification_df)
    assert merged["image_label"].tolist() == ["1.1", "2.1"]

    plotting._plot_normalized_image_residuals(image_df, tmp_path / "normalized_image_residuals.pdf")
    plotting._plot_residual_vs_magnification(image_df, magnification_df, tmp_path / "residual_vs_magnification.pdf")
    plotting._plot_residual_geometry_trends(image_df, tmp_path / "residual_geometry_trends.pdf")

    for filename in [
        "normalized_image_residuals.pdf",
        "residual_vs_magnification.pdf",
        "residual_geometry_trends.pdf",
    ]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_image_residual_histogram_reports_point_and_arc_aware_rms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_df = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "not_recovered", "not_recovered", "recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "not_recovered", "point_recovered"],
            "arc_supported": [False, True, False, False],
            "image_residual_arcsec": [9.0, 9.0, 0.30, 9.0],
            "image_residual_q50": [0.04, np.nan, 0.08, np.inf],
            "point_image_residual_arcsec": [0.04, np.nan, np.nan, 9.0],
            "arc_aware_image_residual_arcsec": [0.04, np.nan, np.nan, 9.0],
            "arc_aware_image_residual_q50": [0.04, 0.18, np.nan, np.inf],
            "arc_curve_distance_arcsec": [np.nan, 0.20, 0.07, np.nan],
        }
    )
    path = tmp_path / "image_residual_histogram.pdf"
    captured: dict[str, Any] = {"histograms": [], "vertical_lines": [], "texts": []}
    original_subplots = plotting.plt.subplots

    def spy_subplots(*args: Any, **kwargs: Any) -> Any:
        fig, ax = original_subplots(*args, **kwargs)
        original_hist = ax.hist
        original_axvline = ax.axvline
        original_text = ax.text

        def spy_hist(values: Any, *hist_args: Any, **hist_kwargs: Any) -> Any:
            captured["histograms"].append(
                {
                    "values": np.asarray(values, dtype=float).copy(),
                    "kwargs": dict(hist_kwargs),
                }
            )
            return original_hist(values, *hist_args, **hist_kwargs)

        def spy_axvline(x: float = 0, *line_args: Any, **line_kwargs: Any) -> Any:
            captured["vertical_lines"].append((float(x), line_kwargs.get("label")))
            return original_axvline(x, *line_args, **line_kwargs)

        def spy_text(*text_args: Any, **text_kwargs: Any) -> Any:
            if len(text_args) >= 3:
                captured["texts"].append(str(text_args[2]))
            return original_text(*text_args, **text_kwargs)

        ax.hist = spy_hist
        ax.axvline = spy_axvline
        ax.text = spy_text
        return fig, ax

    monkeypatch.setattr(plotting.plt, "subplots", spy_subplots)

    plotting._plot_image_residual_histogram(image_df, path, use_arc_aware_diagnostics=True)

    assert path.exists()
    assert path.stat().st_size > 0
    assert len(captured["histograms"]) == 2
    np.testing.assert_allclose(captured["histograms"][0]["values"], np.asarray([0.04, 9.0]))
    np.testing.assert_allclose(captured["histograms"][1]["values"], np.asarray([0.04, 0.20, 9.0]))
    assert captured["histograms"][1]["kwargs"]["color"] == plotting._image_catalog_status_color("ARC_RECOVERED")
    expected_point_rms = float(np.sqrt(np.mean(np.square(np.asarray([0.04, 9.0])))))
    expected_arc_rms = float(np.sqrt(np.mean(np.square(np.asarray([0.04, 0.20, 9.0])))))
    assert captured["vertical_lines"][0][0] == pytest.approx(expected_point_rms)
    assert captured["vertical_lines"][1][0] == pytest.approx(expected_arc_rms)
    assert "median=4.52" in captured["histograms"][0]["kwargs"]["label"]
    assert "median=0.2" in captured["histograms"][1]["kwargs"]["label"]
    assert any(
        "Image residuals" in text
        and "point:" in text
        and "RMS" in text
        and "median" in text
        and "(2/4)" in text
        and "arc-aware:" in text
        and "(3/4)" in text
        and "arc-supported: 1/4" in text
        and "missed: 1/4" in text
        for text in captured["texts"]
    )


def test_plot_image_residual_histogram_ignores_arc_aware_values_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_df = pd.DataFrame(
        {
            "image_recovery_status": ["recovered", "not_recovered", "recovered"],
            "arc_recovery_status": ["point_recovered", "arc_supported", "point_recovered"],
            "arc_supported": [False, True, False],
            "image_residual_arcsec": [0.1, 9.0, 0.3],
            "point_image_residual_arcsec": [0.1, np.nan, 0.3],
            "arc_aware_image_residual_arcsec": [0.1, 0.2, 0.3],
        }
    )
    path = tmp_path / "image_residual_histogram.pdf"
    captured: dict[str, Any] = {"histograms": [], "texts": [], "titles": []}
    original_subplots = plotting.plt.subplots

    def spy_subplots(*args: Any, **kwargs: Any) -> Any:
        fig, ax = original_subplots(*args, **kwargs)
        original_hist = ax.hist
        original_text = ax.text
        original_set_title = ax.set_title

        def spy_hist(values: Any, *hist_args: Any, **hist_kwargs: Any) -> Any:
            captured["histograms"].append(np.asarray(values, dtype=float).copy())
            return original_hist(values, *hist_args, **hist_kwargs)

        def spy_text(*text_args: Any, **text_kwargs: Any) -> Any:
            if len(text_args) >= 3:
                captured["texts"].append(str(text_args[2]))
            return original_text(*text_args, **text_kwargs)

        def spy_set_title(label: str, *title_args: Any, **title_kwargs: Any) -> Any:
            captured["titles"].append(str(label))
            return original_set_title(label, *title_args, **title_kwargs)

        ax.hist = spy_hist
        ax.text = spy_text
        ax.set_title = spy_set_title
        return fig, ax

    monkeypatch.setattr(plotting.plt, "subplots", spy_subplots)

    plotting._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert len(captured["histograms"]) == 1
    np.testing.assert_allclose(captured["histograms"][0], np.asarray([0.1, 0.3]))
    assert captured["titles"] == ["Image Residuals: Point Recovery"]
    assert any("arc-aware" not in text and "arc-supported" not in text for text in captured["texts"])


def test_plot_image_residual_histogram_writes_placeholder_without_finite_values(tmp_path: Path) -> None:
    image_df = pd.DataFrame(
        {
            "image_residual_arcsec": [np.nan, np.inf, -np.inf],
            "image_residual_q50": [np.nan, np.inf, -np.inf],
        }
    )
    path = tmp_path / "image_residual_histogram.pdf"

    plotting._plot_image_residual_histogram(image_df, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_flux_magnification_ratio_consistency_colors_by_log_p_arc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    scatter_kwargs: list[dict[str, Any]] = []
    colorbar_labels: list[str] = []
    original_scatter = Axes.scatter

    class FakeColorbar:
        def set_label(self, label: str) -> None:
            colorbar_labels.append(label)

    def record_scatter(self: Axes, *args: Any, **kwargs: Any) -> Any:
        scatter_kwargs.append(dict(kwargs))
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Figure, "colorbar", lambda self, *_args, **_kwargs: FakeColorbar())

    pair_df = pd.DataFrame(
        {
            "observed_mag_difference": [0.0, 0.5, 1.0],
            "observed_mag_difference_error": [0.1, 0.1, 0.1],
            "model_mag_difference": [0.1, 0.4, 1.2],
            "band": ["f160w", "f160w", "f160w"],
            "p_arc_pair": [1.0, 1.0e-2, 0.0],
        }
    )

    plotting._plot_flux_magnification_ratio_consistency(
        pair_df,
        tmp_path / "flux_magnification_ratio_consistency.pdf",
    )

    assert (tmp_path / "flux_magnification_ratio_consistency.pdf").exists()
    color_values = np.asarray(scatter_kwargs[0]["c"], dtype=float)
    np.testing.assert_allclose(color_values, [0.0, -2.0, -6.0])
    assert scatter_kwargs[0]["vmin"] == pytest.approx(-6.0)
    assert scatter_kwargs[0]["vmax"] == pytest.approx(0.0)
    assert colorbar_labels == [r"$\log_{10}$ pair $p_{\rm arc}$"]


def test_exact_vs_approx_prediction_error_skips_missing_rows(tmp_path: Path) -> None:
    family_df = pd.DataFrame(
        {
            "family_id": ["1", "2", "3"],
            "exact_image_rms_arcsec": [0.2, np.nan, 0.6],
            "approx_image_rms_arcsec": [0.1, 0.4, np.nan],
        }
    )

    plotting._plot_exact_vs_approx_prediction_error(family_df, tmp_path / "exact_vs_approx_prediction_error.pdf")

    path = tmp_path / "exact_vs_approx_prediction_error.pdf"
    assert path.exists()
    assert path.stat().st_size > 0


def test_ns_diagnostic_plots_write_pdfs(tmp_path: Path) -> None:
    n_samples = 24
    ns_diagnostics = {
        "log_L_samples": np.linspace(-20.0, -3.0, n_samples),
        "log_dp_mean": np.linspace(-12.0, -2.0, n_samples),
        "log_X_mean": -np.linspace(0.1, 8.0, n_samples),
        "num_live_points_per_sample": np.full(n_samples, 12),
        "num_likelihood_evaluations_per_sample": np.arange(1, n_samples + 1),
        "log_efficiency": np.asarray([-4.0]),
    }
    specs = [
        ParameterSpec(
            name="x",
            sample_name="x",
            potential_id="mock",
            profile_type=81,
            field="x",
            prior_kind="uniform",
            lower=-1.0,
            upper=1.0,
            step=0.1,
        ),
        ParameterSpec(
            name="y",
            sample_name="y",
            potential_id="mock",
            profile_type=81,
            field="y",
            prior_kind="uniform",
            lower=-1.0,
            upper=1.0,
            step=0.1,
        ),
    ]

    plotting._plot_ns_diagnostics(tmp_path, ns_diagnostics)
    plotting._plot_ns_trace(tmp_path, ns_diagnostics, specs)
    plotting._plot_ns_weight_diagnostics(tmp_path, ns_diagnostics)

    for filename in ["ns_diagnostics.pdf", "ns_weight_diagnostics.pdf"]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0
    assert not (tmp_path / "ns_trace_plot.pdf").exists()


def test_ns_diagnostic_plots_skip_missing_inputs(tmp_path: Path) -> None:
    plotting._plot_ns_diagnostics(tmp_path, {})
    plotting._plot_ns_trace(tmp_path, {"samples": np.empty((0, 0))}, [])
    plotting._plot_ns_weight_diagnostics(tmp_path, {"log_dp_mean": np.asarray([])})

    assert not any(tmp_path.iterdir())


def _smc_plot_posterior() -> PosteriorResults:
    samples = np.asarray(
        [
            [10.0, 0.28, -1.10],
            [11.0, 0.30, -1.00],
            [13.0, 0.33, -0.90],
            [16.0, 0.36, -0.75],
        ],
        dtype=float,
    )
    return PosteriorResults(
        samples=samples,
        log_prob=np.asarray([-12.0, -7.5, -5.0, -6.0], dtype=float),
        accept_prob=np.zeros(samples.shape[0], dtype=float),
        diverging=np.zeros(samples.shape[0], dtype=bool),
        num_steps=np.zeros(samples.shape[0], dtype=float),
        warmup_steps=0,
        sample_steps=samples.shape[0],
        num_chains=1,
        sampler="blackjax_smc",
        sample_weights=np.asarray([0.10, 0.20, 0.45, 0.25], dtype=float),
        temperature_schedule=np.asarray([0.0, 0.15, 0.55, 1.0], dtype=float),
        ess_history=np.asarray([4.0, 3.4, 3.0, 4.0], dtype=float),
        move_acceptance_history=np.asarray([0.72, 0.61, 0.58], dtype=float),
        init_diagnostics={
            "smc_particles": 4,
            "smc_target_ess_frac": 0.8,
            "smc_mean_move_acceptance": 0.6367,
        },
    )


def test_smc_diagnostic_plots_write_pdfs(tmp_path: Path) -> None:
    posterior = _smc_plot_posterior()

    plotting._plot_smc_diagnostics(tmp_path, posterior)
    plotting._plot_smc_weight_diagnostics(tmp_path, posterior)

    for filename in ["smc_diagnostics.pdf", "smc_weight_diagnostics.pdf"]:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_smc_plots_skip_missing_or_invalid_inputs(tmp_path: Path, monkeypatch: Any) -> None:
    empty = PosteriorResults(
        samples=np.empty((0, 2), dtype=float),
        log_prob=np.empty((0,), dtype=float),
        accept_prob=np.empty((0,), dtype=float),
        diverging=np.empty((0,), dtype=bool),
        num_steps=np.empty((0,), dtype=float),
        warmup_steps=0,
        sample_steps=0,
        num_chains=1,
        sampler="blackjax_smc",
    )

    class RaisingCorner:
        def corner(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("SMC corner should skip invalid weights")

    plotting._plot_smc_diagnostics(tmp_path, empty)
    plotting._plot_smc_weight_diagnostics(tmp_path, empty)
    monkeypatch.setattr(plotting, "corner", RaisingCorner())
    plotting._plot_smc_corner(
        tmp_path,
        _smc_plot_posterior().samples,
        _mixed_cosmology_test_specs(),
        np.asarray([0.5, np.nan, 0.25, 0.25], dtype=float),
    )

    assert not any(tmp_path.iterdir())


def test_smc_corner_uses_particle_weights_and_overlays(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    posterior = _smc_plot_posterior()

    class FakeFig:
        def savefig(self, path: Path, **_kwargs: Any) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    class FakeCorner:
        def corner(self, samples: np.ndarray, **kwargs: Any) -> FakeFig:
            calls.append(("corner", np.asarray(samples), kwargs))
            return FakeFig()

        def overplot_lines(self, _fig: FakeFig, xs: list[float | None], **kwargs: Any) -> None:
            calls.append(("lines", xs, kwargs))

        def overplot_points(self, _fig: FakeFig, xs: list[list[float]], **kwargs: Any) -> None:
            calls.append(("points", xs, kwargs))

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda _fig: None)

    plotting._plot_smc_corner(
        tmp_path,
        posterior.samples,
        _mixed_cosmology_test_specs(),
        posterior.sample_weights,
        best_fit_values={"halo.x": 13.0, "cosmology.Om0": 0.33, "cosmology.w0": -0.9},
        map_values={"halo.x": 12.0, "cosmology.Om0": 0.32, "cosmology.w0": -0.95},
        maximum_likelihood_values={"halo.x": 12.5, "cosmology.Om0": 0.325, "cosmology.w0": -0.93},
        previous_stage_best_values={"halo.x": 11.0, "cosmology.Om0": 0.30, "cosmology.w0": -1.0},
    )

    assert calls[0][0] == "corner"
    np.testing.assert_allclose(calls[0][1], posterior.samples[:, [1, 2, 0]])
    assert calls[0][2]["labels"] == ["cosmology.Om0", "cosmology.w0", "halo.x"]
    np.testing.assert_allclose(calls[0][2]["weights"], posterior.sample_weights)
    assert calls[0][2]["plot_datapoints"] is True
    assert calls[1] == (
        "points",
        [[0.30, -1.0, 11.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_PREVIOUS_STAGE_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[2] == (
        "points",
        [[0.32, -0.95, 12.0]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAP_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert calls[3] == (
        "points",
        [[0.325, -0.93, 12.5]],
        {
            "marker": "x",
            "color": plotting.CORNER_MAXIMUM_LIKELIHOOD_COLOR,
            "markersize": 5,
            "markeredgewidth": 1.2,
        },
    )
    assert (tmp_path / "smc_corner.pdf").exists()


def test_smc_corner_subset_prefers_cosmology_and_caps_dimensions() -> None:
    specs = [
        ParameterSpec("large.low", "large_low", "mock", 81, "x", "uniform", -5.0, 5.0, 0.1, component_family="large"),
        ParameterSpec("cosmology.Om0", "cosmology_Om0", "cosmology", 0, "Om0", "uniform", 0.05, 0.6, 0.01, component_family="cosmology"),
        ParameterSpec("source.1.beta_x", "source_1_beta_x", "1", 0, "beta_x", "normal", -np.inf, np.inf, 0.1, component_family="source_position"),
        ParameterSpec("large.high", "large_high", "mock", 81, "y", "uniform", -50.0, 50.0, 0.1, component_family="large"),
        ParameterSpec("cosmology.w0", "cosmology_w0", "cosmology", 0, "w0", "uniform", -2.0, -0.3, 0.05, component_family="cosmology"),
        ParameterSpec("large.mid", "large_mid", "mock", 81, "angle", "uniform", -10.0, 10.0, 0.1, component_family="large"),
    ]
    base = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    samples = np.column_stack(
        [
            base,
            0.28 + 0.01 * base,
            100.0 * base,
            20.0 * base,
            -1.1 + 0.05 * base,
            5.0 * base,
        ]
    )

    subset = plotting._smc_corner_subset(samples, specs, np.ones(samples.shape[0], dtype=float), max_params=3)

    assert subset is not None
    subset_samples, subset_specs, subset_weights = subset
    assert [spec.name for spec in subset_specs] == ["cosmology.Om0", "cosmology.w0", "large.high"]
    assert subset_samples.shape == (5, 3)
    np.testing.assert_allclose(subset_weights, np.full(5, 0.2))
