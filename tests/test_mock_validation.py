"""Mock validation: the truth draw, exact lensing, the solver model built from the truth, and the recovery bookkeeping."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lenscluster.config import ImageDiagnosticsConfig, LensClusterSolverConfig, RuntimeConfig, SamplingConfig, ScalingModelConfig, SearchConfig, TruthRecoveryConfig
from lenscluster.mock_validation import (MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig, draw_truth,
                                         generate_images, generate_mock, parameter_truth, run_mock_validation, solver_config_for_mock, write_aggregate_plots)
from lenscluster.mock_validation.aperture import APERTURE_RADII_KPC, aperture_profiles, truth_groups
from lenscluster.mock_validation.images import jacobian, lens_state
from lenscluster.state import build_state

SMALL = MockConfig(seed=7, n_members=12, n_primary_families=4, n_member_families=1, unmodelled_members=2, sigma_def_arcsec=0.03)


def test_truth_is_deterministic_and_in_range():
    a, b = draw_truth(SMALL), draw_truth(SMALL)
    assert a == b
    assert SMALL.z_lens_range[0] <= a.z_lens <= SMALL.z_lens_range[1]
    total = a.halo.mass_msun + a.bcg.mass_msun
    assert SMALL.total_mass_range_msun[0] * 0.99 <= total <= SMALL.total_mass_range_msun[1] * 1.01
    assert len(a.members) == 14 and len(a.modelled_members) == 12
    assert all(m.magnitude <= SMALL.faint_magnitude_limit for m in a.members)
    assert all(SMALL.sigma_star_range_kms[0] <= v <= SMALL.sigma_star_range_kms[1] for v in [a.relation["sigma_star_kms"]])
    assert [m.magnitude for m in a.members] == sorted(m.magnitude for m in a.members)


def test_images_are_exact_and_noise_is_injected():
    truth = draw_truth(SMALL)
    sources, images = generate_images(truth, SMALL, np.random.default_rng(1))
    assert len(sources) >= 3 and all(s["n_images"] >= SMALL.min_images_per_family for s in sources)
    for source in sources:
        rows = images[images.family_id == source["family_id"]]
        beta, _, det = jacobian(lens_state(truth, source["z_source"]), rows.x_true_arcsec.to_numpy(), rows.y_true_arcsec.to_numpy())
        np.testing.assert_allclose(beta, [[source["beta_x"], source["beta_y"]]] * len(rows), atol=1e-6)   # true images solve the lens equation
        np.testing.assert_allclose(1.0 / det, rows.magnification_true.to_numpy(), rtol=1e-8)
    central = [s for s in sources if s["kind"] == "central"]
    assert len(central) == 1
    rows = images[images.family_id == central[0]["family_id"]]
    assert np.hypot(rows.x_true_arcsec - truth.bcg.x_centre, rows.y_true_arcsec - truth.bcg.y_centre).min() <= SMALL.central_radius_arcsec
    offsets = np.hypot(images.x_obs_arcsec - images.x_true_arcsec, images.y_obs_arcsec - images.y_true_arcsec)
    assert 0.5 * SMALL.sigma_inst_arcsec < np.median(offsets) < 4 * SMALL.sigma_inst_arcsec       # astrometric noise plus source scatter


@pytest.mark.parametrize("mode", ["production", "idealized"])
def test_solver_model_from_truth_contains_the_truth(tmp_path, mode):
    config = SMALL.with_updates(mode=mode)
    paths = generate_mock(tmp_path / "mock", config, 7)
    truth = json.loads(paths.truth.read_text())
    campaign = MockValidationConfig(mock=config, solver=LensClusterSolverConfig())
    solver = solver_config_for_mock(campaign, truth, image_catalog=paths.image_catalog, member_catalog=paths.member_catalog, seed=7, output_dir=tmp_path / "run").validate()
    state = build_state(solver)
    assert state.z_lens == pytest.approx(truth["z_lens"])
    assert len(state.members.member_ids) == truth["n_modelled_members"] == 12
    values = parameter_truth(truth, solver_pivot_magnitude=float(state.potfiles[0]["mag0"]))
    for spec in state.parameter_specs:
        t = values.get(spec.name)
        if t is None or spec.physical_lower is None and spec.prior_kind == "normal":
            continue
        lower = spec.physical_lower if spec.physical_lower is not None else spec.lower
        upper = spec.physical_upper if spec.physical_upper is not None else spec.upper
        if spec.transform_kind == "identity" and spec.prior_kind in {"uniform", "truncated_normal"}:
            assert lower - 1e-9 <= t <= upper + 1e-9, (spec.name, t, lower, upper)
        elif spec.transform_kind == "log_positive" and t > 0 and upper is not None and np.isfinite(upper):
            assert lower - 1e-9 <= t <= upper + 1e-9, (spec.name, t, lower, upper)
        if mode == "idealized" and spec.physical_mean is not None and spec.physical_std is not None and t > 0:
            assert abs(t - spec.physical_mean) <= 1.5 * spec.physical_std + 1e-9, (spec.name, t, spec.physical_mean, spec.physical_std)
    # With the member core fixed, no corekpc parameter is sampled and the fixed core sits at the pivot like sigma and cutkpc.
    fixed = build_state(replace(solver, scaling=ScalingModelConfig(fit_corekpc=False)))
    assert not any(spec.field == "corekpc" for spec in fixed.parameter_specs) and fixed.n_parameters == state.n_parameters - 1
    assert fixed.potfiles[0]["corekpc_nominal"] == pytest.approx(values["potfile.corekpc"], rel=0.05)   # moved with the prior-mean beta, truth with the truth beta
    profiles = aperture_profiles(np.asarray(APERTURE_RADII_KPC), truth_groups(truth), centre_kpc=(truth["halo"]["x_centre"] * truth["kpc_per_arcsec"], truth["halo"]["y_centre"] * truth["kpc_per_arcsec"]))
    np.testing.assert_allclose(profiles["total"], profiles["halo"] + profiles["bcg"] + profiles["members"], rtol=1e-10)
    assert values["images.sigma_def"] == pytest.approx(0.03) and values["potfile.velocity_log_scatter"] == pytest.approx(0.10)
    # The solver assumes only the 1 mas floor; the injected astrometric noise is the truth of the white image-plane scatter.
    assert solver.model.image_constraints.sigma_inst_arcsec == pytest.approx(config.solver_sigma_inst_arcsec) == pytest.approx(0.001)
    assert solver.likelihood.sigma_inst_arcsec == pytest.approx(0.001)
    assert values["images.sigma_white"] == values["images.sigma_extra"] == pytest.approx(config.sigma_inst_arcsec)


def test_campaign_end_to_end_scores_only_parameters_with_a_truth(tmp_path):
    config = MockConfig(seed=11, n_members=4, n_primary_families=2, n_member_families=0, mode="idealized")
    solver = LensClusterSolverConfig(runtime=RuntimeConfig(seed=11, chains=1, resume=True, nuts_chain_method="sequential"),
                                     search=SearchConfig(n_starts=2, lbfgs_max_iter=20, polish_max_iter=5),
                                     sampling=SamplingConfig(warmup=8, samples=8, max_tree_depth=3),
                                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=2),
                                     truth=TruthRecoveryConfig(posterior_truth_recovery_draws=0))
    campaign = MockValidationConfig(mock=config, paths=MockValidationPathsConfig(output_dir=tmp_path, campaign_name="c", run_name="r"),
                                    runtime=MockValidationRuntimeConfig(realizations=1, seed=11), solver=solver).validate()
    recovery_dirs = run_mock_validation(campaign, log=lambda *a, **k: None)
    payload = json.loads((recovery_dirs[0] / "recovery.json").read_text())
    rows = payload["parameters"]
    scored = [r for r in rows if r["scored"]]
    assert scored and all(np.isfinite(r["truth"]) for r in scored)
    assert not any(r["scored"] for r in rows if r["parameter"] == "potfile.radius_log_scatter")   # injected as zero
    # The injected astrometric noise is the truth of the white image-plane scatter above the solver's 1 mas floor.
    assert all(r["scored"] and r["truth"] == pytest.approx(0.10) for r in rows if r["parameter"] in {"images.sigma_extra", "images.sigma_white"})
    assert 0.0 <= payload["coverage_68"] <= 1.0 and payload["scored_parameters"] == len(scored)
    assert isinstance(payload["converged"], bool) and isinstance(payload["convergence_reasons"], list)
    magnification = pd.DataFrame(payload["magnification"])
    for column in ("magnification_true", "magnification_true_at_obs", "magnification_true_at_model_position",
                   "magnification_model", "magnification_model_at_model_position", "magnification_model_q50"):
        assert column in magnification, column
    recovered = magnification[magnification["image_recovery_status"] == "recovered"]
    assert np.isfinite(recovered["magnification_true_at_model_position"]).all() and np.isfinite(recovered["magnification_model_at_model_position"]).all()
    from lenscluster.mock_validation.aggregate_plots import load_campaign
    everything = load_campaign(recovery_dirs[0].parent.parent, converged_only=False)       # a short test chain rarely converges
    assert len(everything.seeds) == 1 and "converged" in everything.summary
    outputs = write_aggregate_plots(recovery_dirs[0].parent.parent)
    assert any(p.name == "mock_calibration.pdf" for p in outputs) and (recovery_dirs[0].parent.parent / "aggregate_plots" / "mock_coverage_table.csv").is_file()
    assert set(payload["relation"]["draws"]) == {"sigma", "cutkpc", "alpha_sigma", "beta_radius"} and payload["relation"]["pivot_luminosity_ratio"] > 0


def test_knotted_families_are_systems_whose_knots_follow_the_main_images(tmp_path):
    """A knotted galaxy is one system with one family per knot; every knot has the same images (in order) and solves the
    lens equation for its own source; the solver reads the knots back as families of that system; a mock without knots
    draws the same images as before (the knot draws come after every draw of the main family)."""
    config = replace(SMALL, sigma_def_arcsec=0.0, n_knotted_families=2, knots_per_family=(3, 3))
    truth = draw_truth(config)
    sources, images = generate_images(truth, config, np.random.default_rng(1))
    knotted = [s for s in sources if s["knot_id"]]
    assert knotted and {s["system_id"] for s in knotted} and all(s["family_id"] == s["system_id"] + s["knot_id"] for s in sources)
    for system in {s["system_id"] for s in knotted}:
        members = [s for s in sources if s["system_id"] == system]
        assert [s["knot_id"] for s in members] == ["A", "B", "C"][:len(members)] and len(members) >= 2
        main = members[0]
        for knot in members:
            rows = images[images.family_id == knot["family_id"]]
            assert len(rows) == main["n_images"]
            assert [label.split(".")[1] for label in rows.image_label] == [chr(ord("a") + k) for k in range(len(rows))]
            beta, _, _ = jacobian(lens_state(truth, knot["z_source"]), rows.x_true_arcsec.to_numpy(), rows.y_true_arcsec.to_numpy())
            np.testing.assert_allclose(beta, [[knot["beta_x"], knot["beta_y"]]] * len(rows), atol=1e-6)
            offset = np.hypot(knot["beta_x"] - main["beta_x"], knot["beta_y"] - main["beta_y"])
            assert knot is main or config.knot_offset_range_arcsec[0] <= offset <= config.knot_offset_range_arcsec[1] + 1e-9
    paths = generate_mock(tmp_path / "mock", config, 7)
    truth_json = json.loads(paths.truth.read_text())
    solver = solver_config_for_mock(MockValidationConfig(mock=config, solver=LensClusterSolverConfig()), truth_json, image_catalog=paths.image_catalog,
                                    member_catalog=paths.member_catalog, seed=7, output_dir=tmp_path / "run").validate()
    state = build_state(solver)
    assert set(state.images.image_labels) == {r["image_label"] for r in truth_json["images"]}
    assert len(set(state.images.system_ids)) < len(state.images.family_ids)
    plain = replace(config, n_knotted_families=0)
    _, plain_images = generate_images(truth, plain, np.random.default_rng(1))
    _, reference = generate_images(truth, plain, np.random.default_rng(1))
    pd.testing.assert_frame_equal(plain_images, reference)
    assert set(plain_images.knot_id) == {""}


def test_injected_noise_is_drawn_per_realization_and_is_the_white_scatter_truth(tmp_path):
    config = replace(SMALL, sigma_def_arcsec=0.0, sigma_inst_range_arcsec=(0.005, 0.045))
    drawn = []
    for seed in (7, 8):
        truth = json.loads(generate_mock(tmp_path / f"mock{seed}", config, seed).truth.read_text())
        value = truth["config"]["sigma_inst_arcsec"]
        assert 0.005 <= value <= 0.045
        assert {r["sigma_inst_arcsec"] for r in truth["images"]} == {value}
        assert parameter_truth(truth)["images.sigma_white"] == pytest.approx(value)
        drawn.append(value)
    assert drawn[0] != drawn[1]


def test_fisher_precheck_matches_the_white_noise_closed_form(tmp_path):
    """With white noise alone, the restricted Fisher information on ln sigma_white is about 2 per degree of freedom left
    after the source positions (I = 2 (n - p), up to the 1 mas and 1.5 mas floors)."""
    from lenscluster.config import LikelihoodConfig
    from lenscluster.mock_validation.fisher import hyperparameter_fisher
    config = replace(SMALL, sigma_def_arcsec=0.0, n_knotted_families=1, sigma_inst_arcsec=0.1)
    paths = generate_mock(tmp_path / "mock", config, 7)
    truth = json.loads(paths.truth.read_text())
    template = LensClusterSolverConfig(likelihood=LikelihoodConfig(likelihood_mode="gp_image_plane", gp_kernel="knee", gp_blocks="system", sigma_sel_prior=None))
    solver = solver_config_for_mock(MockValidationConfig(mock=config, solver=template), truth, image_catalog=paths.image_catalog,
                                    member_catalog=paths.member_catalog, seed=7, output_dir=tmp_path / "run")
    result = hyperparameter_fisher(solver, truth)
    row = result["table"].set_index("parameter").loc["images.sigma_white"]
    assert result["names"] == ["sigma_white"] and result["dof"] == 2 * result["n_images"] - 2 * result["n_families"]
    assert row["fisher_sigma"] == pytest.approx(1.0 / np.sqrt(2.0 * result["dof"]), rel=0.03)
    assert 0.0 < row["prior_contraction"] < 1.0
