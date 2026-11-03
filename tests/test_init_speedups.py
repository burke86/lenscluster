"""Init-stage speedups that must not change results: blocks grouped by size and frozen kernel blocks give the same
likelihood and gradient; screening keeps the starts that reach the best mode; a warm start maps a saved start by name
(centered deviations back to unit offsets); the chain starts converge on the posterior NUTS samples."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from lenscluster import artifacts, gp_knee
from lenscluster.config import SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.search import Objective, multistart_map, prior_starts
from lenscluster.search_split import search_blocks, warm_start_theta
from lenscluster.state import build_state
from test_gp_knee import knee_config, two_knot_config


def _grouped(state):
    out = copy.copy(state)
    out.gp_blocks, out.gp_group_blocks = "system", True
    return out


def test_grouped_blocks_and_frozen_kernels_reproduce_the_likelihood(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(knee_config(config, "system"))
    ungrouped = copy.copy(state)
    ungrouped.gp_group_blocks = False
    padded, grouped = Evaluator(ungrouped, "gp_image_plane"), Evaluator(_grouped(state), "gp_image_plane")
    assert padded.gp_groups is None and Evaluator(state, "gp_image_plane").gp_groups is not None     # grouped by default
    assert grouped.gp_groups is not None and sum(len(g["block_ids"]) for g in grouped.gp_groups) == int(padded.gp["mask"].shape[0])
    f0, f1 = jax.jit(jax.value_and_grad(padded.log_likelihood)), jax.jit(jax.value_and_grad(grouped.log_likelihood))
    rng = np.random.default_rng(1)
    theta0 = np.asarray(state.initial_theta, dtype=float)
    for k in range(3):
        theta = jnp.asarray(theta0 + (0.0 if k == 0 else 0.01) * rng.normal(size=theta0.size))
        (v0, g0), (v1, g1) = f0(theta), f1(theta)
        np.testing.assert_allclose(float(v1), float(v0), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-9, atol=1e-9)
    # frozen kernel blocks: exact while the error scales stay where they were frozen
    frozen = Evaluator(_grouped(state), "gp_image_plane")
    frozen.freeze_gp_kernels(theta0)
    names = [s.name for s in state.parameter_specs]
    lens = theta0.copy()
    lens[names.index("halo.v_disp")] += 0.003
    np.testing.assert_allclose(float(frozen.log_likelihood(jnp.asarray(lens))), float(padded.log_likelihood(jnp.asarray(lens))), rtol=1e-12)


def test_bucket_layouts_pad_each_group_to_its_own_largest_block(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(knee_config(config, "system"))
    layout = Evaluator(state, "gp_image_plane").gp
    groups = gp_knee.bucket_layouts(layout)
    counts = np.asarray(layout["mask"]).sum(1)
    for group in groups:
        assert group["size"] == int(counts[group["block_ids"]].max())
        np.testing.assert_array_equal(np.asarray(group["mask"]).sum(1), counts[group["block_ids"]])


def test_screening_continues_only_the_best_starts(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    starts = prior_starts(state, 6, 2)
    quiet = lambda *a, **k: None
    full = multistart_map(state, evaluator, starts, SearchConfig(n_starts=6, lbfgs_max_iter=60, polish_max_iter=10, screen_iterations=None), method="lbfgs", log=quiet)
    screened = multistart_map(state, evaluator, starts, SearchConfig(n_starts=6, lbfgs_max_iter=60, polish_max_iter=10, screen_iterations=10, screen_keep=2),
                              method="lbfgs", log=quiet)
    assert abs(screened["modes"][0]["log_posterior"] - full["modes"][0]["log_posterior"]) < 1e-3
    iterations = np.asarray(screened["iterations"])
    assert (iterations <= 10).sum() >= 4 and iterations.max() <= 60


def test_warm_start_maps_a_saved_start_by_name_and_uncenters_deviations(tmp_path, synthetic_config):
    state = build_state(synthetic_config)
    specs = state.parameter_specs
    theta = np.asarray(state.initial_theta, dtype=float) + 0.01 * np.arange(len(specs))
    offset = next(i for i, s in enumerate(specs) if s.field == "velocity_offset")
    scale = next(i for i, s in enumerate(specs) if s.sample_name == specs[offset].scale_site_name)
    saved_specs = [s.__dict__ | {} for s in specs]
    saved_specs[offset] = dict(saved_specs[offset], centered=True)
    centered_theta = theta.copy()
    centered_theta[offset] = theta[offset] * np.exp(theta[scale])            # the saved run samples this member centered
    directory = tmp_path / "init" / "artifacts"
    directory.mkdir(parents=True)
    (directory / artifacts.STATE_FILE).write_text(json.dumps({"parameter_specs": saved_specs}, default=float))
    artifacts.save_start(directory, centered_theta, {})
    (directory / artifacts.SUMMARY_FILE).write_text(json.dumps({"verdict": "verdict: robust [lbfgs+lbfgs] (5 of 8 starts)"}))
    out, shared, fragile = warm_start_theta(tmp_path, state)
    assert shared == len(specs) and not fragile
    np.testing.assert_allclose(out, theta, rtol=1e-12, atol=1e-12)


def test_converged_chain_starts_raise_the_nuts_target(synthetic_config):
    from lenscluster.pipeline import converge_chain_starts
    config = replace(synthetic_config, search=replace(synthetic_config.search, nuts_start_max_iter=200))
    state = build_state(config)
    evaluator = Evaluator(state)
    objective = Objective(state, evaluator, state.initial_theta, include_latent=True, log_likelihood=evaluator.log_likelihood)
    theta = np.asarray(state.initial_theta, dtype=float)
    search = {"theta": theta, "modes": [{"theta": theta.tolist(), "within_margin": True, "log_posterior": 0.0}]}
    out = converge_chain_starts(state, evaluator, search, config, log=lambda *a: None)
    value = lambda t: -float(objective.potential(jnp.asarray(objective.flatten(np.asarray(t)))))
    assert value(out["theta"]) > value(theta)
    np.testing.assert_allclose(out["modes"][0]["theta"], out["theta"])


def test_chain_starts_trailing_the_best_by_more_than_the_margin_are_dropped(synthetic_config, monkeypatch):
    from lenscluster import pipeline, search as search_module
    from lenscluster.search import chain_starts
    config = replace(synthetic_config, search=replace(synthetic_config.search, nuts_start_max_iter=5, mode_margin=10.0, max_chain_modes=3))
    state = build_state(config)
    evaluator = Evaluator(state)
    objective = Objective(state, evaluator, state.initial_theta, include_latent=True, log_likelihood=evaluator.log_likelihood)

    def unchanged(objective_, starts_g, max_iter, memory=20, **_):     # the optimizer leaves every start where it is
        values = np.array([float(objective_.reduced(jnp.asarray(g))) for g in starts_g])
        return dict(final=np.asarray(starts_g), potential=values)
    monkeypatch.setattr(search_module, "run_scipy_lbfgs", unchanged)
    names = [s.name for s in state.parameter_specs]
    first = np.asarray(state.initial_theta, dtype=float)
    second = first.copy()
    second[names.index("halo.v_disp")] = np.log(510.0)                 # latent log v_disp: a very different lens
    value = lambda t: -float(objective.potential(jnp.asarray(objective.flatten(t))))
    good, bad = (first, second) if value(first) > value(second) else (second, first)
    assert value(good) - value(bad) > 10.0
    search = {"theta": bad, "modes": [{"theta": bad.tolist(), "within_margin": True, "log_posterior": 1.0},
                                      {"theta": good.tolist(), "within_margin": True, "log_posterior": 0.0}]}
    out = pipeline.converge_chain_starts(state, evaluator, search, config, log=lambda *a: None)
    np.testing.assert_allclose(out["theta"], good)
    assert [m["within_margin"] for m in out["modes"]] == [True, False] and out["modes"][1].get("dropped_for_nuts")
    starts = chain_starts(out, 4, config.search.max_chain_modes)
    assert all(np.allclose(s, good) for s in starts)                   # every chain starts at the kept mode


def test_optax_lbfgs_stops_once_every_start_has_converged(synthetic_config):
    from lenscluster.search import run_optax_lbfgs
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    objective = Objective(state, evaluator, state.initial_theta, include_latent=True, log_likelihood=evaluator.log_likelihood)
    index = np.asarray(objective.global_index)
    x0 = np.stack([objective.flatten(np.asarray(state.initial_theta, dtype=float))[index]] * 2)
    fixed = run_optax_lbfgs(objective, x0, 40, window=10)
    assert int(fixed["iterations"][0]) == 40 and fixed["traces"].shape[1] == 40 and not fixed["converged"].any()
    # a huge tolerance is met at the first window that has a previous one to compare with
    loose = run_optax_lbfgs(objective, x0, 40, tolerance=1.0e9, window=10)
    assert int(loose["iterations"][0]) == 20 and loose["converged"].all()
    # a negative tolerance is never met: the cap ends it, and the result matches the fixed-count run
    strict = run_optax_lbfgs(objective, x0, 40, tolerance=-1.0, window=10)
    assert int(strict["iterations"][0]) == 40 and not strict["converged"].any()
    np.testing.assert_allclose(strict["potential"], fixed["potential"], rtol=1e-10)


def test_scipy_lbfgs_stops_on_the_tolerance_and_matches_the_cap(synthetic_config):
    from lenscluster.search import run_scipy_lbfgs
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    objective = Objective(state, evaluator, state.initial_theta, include_latent=True, log_likelihood=evaluator.log_likelihood)
    index = np.asarray(objective.global_index)
    x0 = objective.flatten(np.asarray(state.initial_theta, dtype=float))[index][None]
    start = float(objective.reduced(jnp.asarray(x0[0])))
    capped = run_scipy_lbfgs(objective, x0, 30)
    assert int(capped["iterations"][0]) <= 30 and not capped["converged"].any() and capped["potential"][0] < start
    loose = run_scipy_lbfgs(objective, x0, 30, tolerance=1.0e9, window=5)        # met at the first full window
    assert loose["converged"].all() and int(loose["iterations"][0]) == 6
    strict = run_scipy_lbfgs(objective, x0, 30, tolerance=-1.0, window=5)        # never met: same as the capped run
    assert not strict["converged"].any()
    np.testing.assert_allclose(strict["potential"], capped["potential"], rtol=1e-10)


def test_chain_start_conditioning_flags_a_start_away_from_the_mode(synthetic_config):
    from lenscluster.pipeline import START_DECREMENT_WARNING, chain_start_conditioning, converge_chain_starts
    config = replace(synthetic_config, search=replace(synthetic_config.search, nuts_start_max_iter=300))
    state = build_state(config)
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta, dtype=float)
    search = {"theta": theta, "modes": [{"theta": theta.tolist(), "within_margin": True, "log_posterior": 0.0}]}
    lines = []
    converged = converge_chain_starts(state, evaluator, search, config, log=lambda *a: None)
    at_mode = chain_start_conditioning(state, evaluator, converged, config, log=lines.append)["start_conditioning"]
    names = [s.name for s in state.parameter_specs]
    far = np.asarray(converged["theta"], dtype=float).copy()
    far[names.index("halo.v_disp")] += 0.15                              # 16% in velocity dispersion: a start off the mode
    off = chain_start_conditioning(state, evaluator, dict(converged, theta=far), config, log=lines.append)["start_conditioning"]
    assert np.isfinite(at_mode["newton_decrement"]) and at_mode["newton_decrement"] < off["newton_decrement"]
    assert off["newton_decrement"] > START_DECREMENT_WARNING and any(w.startswith("WARNING: the chain start is not a mode") for w in off["warnings"])
    assert off["dim"] == at_mode["dim"] > 0 and any("chain start conditioning" in line for line in lines)


def test_grouped_blocks_give_the_padded_mean_deflection_and_sources(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(knee_config(config, "system"))
    ungrouped = copy.copy(state)
    ungrouped.gp_group_blocks = False
    grouped, padded = Evaluator(state, "gp_image_plane"), Evaluator(ungrouped, "gp_image_plane")
    assert grouped.gp_groups is not None and padded.gp_groups is None
    theta = np.asarray(state.initial_theta, dtype=float) + 0.01 * np.random.default_rng(3).normal(size=state.n_parameters)
    images = state.images
    x, y, fam = np.asarray(images.x_obs, float), np.asarray(images.y_obs, float), np.asarray(images.family_index_per_image)
    for a, b in zip(grouped.gp_deflection_correction(theta)(x, y, fam), padded.gp_deflection_correction(theta)(x, y, fam)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-14)
    physical = np.asarray(grouped.reported_physical(jnp.asarray(theta)))
    np.testing.assert_allclose(physical, np.asarray(padded.reported_physical(jnp.asarray(theta))), rtol=1e-10, atol=1e-12)


def test_chain_start_keeps_the_member_scatters_pinned(synthetic_config):
    """The chain-start step holds the member scatters at the search's value (free, an optimizer drives them to zero);
    every member is a unit offset by default."""
    from lenscluster.pipeline import converge_chain_starts
    config = replace(synthetic_config, search=replace(synthetic_config.search, nuts_start_max_iter=50))
    assert config.members.centering == "none"
    state = build_state(config)
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta, dtype=float)
    scatter = [i for i, s in enumerate(state.parameter_specs) if s.component_family == "member_velocity_scatter"]
    assert scatter
    search = {"theta": theta, "modes": [{"theta": theta.tolist(), "within_margin": True, "log_posterior": 0.0}]}
    out = converge_chain_starts(state, evaluator, search, config, log=lambda *a: None)
    expected = [np.log(config.search.polish_radius_scatter if state.parameter_specs[i].field == "radius_log_scatter" else config.search.polish_scatter)
                for i in scatter]
    np.testing.assert_allclose(np.asarray(out["theta"])[scatter], expected, rtol=0, atol=1e-12)
