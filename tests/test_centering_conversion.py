"""Centering conversion: every search mode moves to the final member centering, so the saved modes, the chain starts
and the Hessian seed all live in the coordinates NUTS samples."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import artifacts, pipeline, sampling
from lenscluster.config import ImageDiagnosticsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.search import chain_starts
from lenscluster.state import build_state, convert_member_centering, with_member_classes
from conftest import SMOKE_SAMPLING_GATES, SMOKE_SEARCH_GATES


def _offset_indices(state):
    members = state.members
    return [int(i) for i in list(members.offset_param_index) + list(members.radius_offset_param_index) if int(i) >= 0]


def test_conversion_preserves_each_modes_member_deviations_and_likelihood(synthetic_config):
    """Classes held fixed, only the centering changes: the likelihood and every member's dPIE parameters are unchanged."""
    base = build_state(synthetic_config)
    exact = np.arange(len(base.members.member_ids))[:2]
    base = with_member_classes(base, exact, dict(base.members.classification), centered=np.zeros(len(base.members.member_ids), dtype=bool))
    rng = np.random.default_rng(4)
    scatter_indices = sorted({int(i) for i in list(base.members.scatter_param_index) + list(base.members.radius_scatter_param_index) if int(i) >= 0})
    modes = np.repeat(np.asarray(base.initial_theta, dtype=float)[None], 3, axis=0)
    for k in range(3):                                        # three modes with their own offsets and scatters
        modes[k, _offset_indices(base)] = rng.normal(size=len(_offset_indices(base)))
        modes[k, scatter_indices] = np.log(rng.uniform(0.05, 0.6, size=len(scatter_indices)))
    centered = np.zeros(len(base.members.member_ids), dtype=bool)
    centered[exact] = True
    converted = convert_member_centering(base, modes, centered)
    after = with_member_classes(base, exact, dict(base.members.classification), centered=centered)
    assert not np.allclose(converted, modes)                  # the conversion did something
    before_ev, after_ev = Evaluator(base), Evaluator(after)
    for k in range(3):
        np.testing.assert_allclose(float(after_ev.loglike(jnp.asarray(converted[k]))), float(before_ev.loglike(jnp.asarray(modes[k]))), rtol=1e-12)
        p_before = before_ev.packed_state(before_ev.physical(jnp.asarray(modes[k])))[0]
        p_after = after_ev.packed_state(after_ev.physical(jnp.asarray(converted[k])))[0]
        for field in ("sigma0", "Ra", "Rs"):
            np.testing.assert_allclose(np.asarray(getattr(p_after, field)), np.asarray(getattr(p_before, field)), rtol=1e-12, err_msg=field)
        np.testing.assert_allclose(convert_member_centering(base, modes[k], centered), converted[k])   # one row equals the batched row
    np.testing.assert_allclose(convert_member_centering(after, converted, np.zeros_like(centered)), modes, rtol=1e-12)   # and back


class _HessianSeen(Exception):
    pass


def _capture_hessian_start(monkeypatch):
    seen = {}

    def capture(model, specs, reference_theta):
        seen["theta"] = np.asarray(reference_theta, dtype=float).copy()
        raise _HessianSeen
    monkeypatch.setattr(sampling, "block_mass_seed", capture)
    return seen


def _starts_and_hessian(config, init, monkeypatch):
    seen = _capture_hessian_start(monkeypatch)
    with pytest.raises(_HessianSeen):
        pipeline.run_final(config, init)
    return chain_starts(init.search, config.runtime.chains, config.search.max_chain_modes), seen["theta"]


def test_chain_starts_and_hessian_use_the_converted_modes(synthetic_config, monkeypatch):
    # The converged-start step (nuts_start_max_iter) moves the starts on purpose; it is off here so that the conversion
    # alone is tested (test_init_speedups covers the converging step).
    config = replace(synthetic_config,
                     runtime=RuntimeConfig(seed=5, chains=3, nuts_chain_method="sequential", resume=True),
                     search=SearchConfig(**SMOKE_SEARCH_GATES, nuts_start_max_iter=0, n_starts=6, member_starts=3, lbfgs_max_iter=30, polish_max_iter=20, mode_margin=1.0e9,
                                         mode_parameter_tolerance=1.0e-9, max_chain_modes=3),
                     sampling=SamplingConfig(**SMOKE_SAMPLING_GATES, warmup=5, samples=5, max_tree_depth=3),
                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=2),
                     members=replace(synthetic_config.members, centering="exact"))
    captured = {}
    real = pipeline.convert_search_centering

    def spy(search, before, after):
        captured.update(search=search, before=before, after=after)
        return real(search, before, after)
    monkeypatch.setattr(pipeline, "convert_search_centering", spy)
    init = pipeline.run_init(config)
    state, saved_dir = init.state, init.paths.artifacts
    assert state.members.centered.any()
    before, original = captured["before"], captured["search"]
    within = [m for m in original["modes"] if m["within_margin"]][:config.search.max_chain_modes]
    assert len(within) >= 2                                    # several chain modes, so the per-chain check means something
    ev_before, ev_after = Evaluator(before), Evaluator(state)

    def check(init_result):
        saved_start = artifacts.load_start(saved_dir)
        starts, hessian_start = _starts_and_hessian(config, init_result, monkeypatch)
        # The first chain starts at the saved converted best start, with its likelihood, and the Hessian sees the same point.
        np.testing.assert_array_equal(starts[0], saved_start)
        np.testing.assert_array_equal(hessian_start, saved_start)
        np.testing.assert_allclose(float(ev_after.loglike(jnp.asarray(starts[0]))),
                                   float(ev_before.loglike(jnp.asarray(original["theta"]))), rtol=1e-12)
        # Every chain start is its own mode, converted, with that mode's likelihood.
        for i, start in enumerate(starts):
            mode = np.asarray(within[i % len(within)]["theta"], dtype=float)
            np.testing.assert_allclose(start, convert_member_centering(before, mode, state.members.centered), rtol=1e-12)
            np.testing.assert_allclose(float(ev_after.loglike(jnp.asarray(start))), float(ev_before.loglike(jnp.asarray(mode))), rtol=1e-12)
        assert not np.allclose(starts[0], np.asarray(within[0]["theta"], dtype=float))   # the old unconverted mode is not used

    check(init)
    # Saved and reloaded: the resumed init returns the converted modes and start, and the same relationships hold.
    resumed = pipeline.run_init(config)
    assert resumed.resumed
    np.testing.assert_array_equal(resumed.theta, artifacts.load_start(saved_dir))
    for saved_mode, mode in zip(artifacts.load_search(saved_dir)["modes"], init.search["modes"]):
        np.testing.assert_array_equal(np.asarray(saved_mode["theta"]), np.asarray(mode["theta"]))
    check(resumed)


def test_a_misplaced_chain_start_is_reported(synthetic_config, monkeypatch, tmp_path):
    """With the conversion removed (the bug), the final stage warns that the chain starts are not the saved start."""
    from lenscluster.sampling import hessian_seed_warning
    config = replace(synthetic_config,
                     runtime=RuntimeConfig(seed=5, chains=2, nuts_chain_method="sequential", resume=True),
                     search=SearchConfig(**SMOKE_SEARCH_GATES, nuts_start_max_iter=0, n_starts=4, member_starts=2, lbfgs_max_iter=30, polish_max_iter=20),
                     sampling=SamplingConfig(**SMOKE_SAMPLING_GATES, warmup=5, samples=5, max_tree_depth=3),
                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=2),
                     members=replace(synthetic_config.members, centering="exact"))
    good = pipeline.run(config)
    log = (good.paths.final.logs / "final.log").read_text()
    assert "chain-start log-likelihoods" in log and "WARNING" not in log
    assert good.posterior.diagnostics["start_check"]["warnings"] == []
    assert "initialization" in (good.paths.final.tables / "run_summary.txt").read_text()

    broken = replace(config, paths=replace(config.paths, run_name="broken"))
    monkeypatch.setattr(pipeline, "convert_search_centering", lambda search, before, after: search)
    bad = pipeline.run(broken)
    log = (bad.paths.final.logs / "final.log").read_text()
    assert "WARNING: the first chain start is not the saved init start" in log
    assert bad.posterior.diagnostics["start_check"]["warnings"]
    assert "WARNING, see the final log" in (bad.paths.final.tables / "run_summary.txt").read_text()
    # The Hessian rule on the numbers of the M0416 run that had the bug, and of a run started at a correct mode.
    assert hessian_seed_warning({"n_negative_eigenvalues": 19, "global_dim": 39}) and not hessian_seed_warning({"n_negative_eigenvalues": 1, "global_dim": 35})
