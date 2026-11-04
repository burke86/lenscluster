"""Adapted step sizes and energy diagnostics are recorded per chain, saved and reloaded."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from lenscluster import artifacts, pipeline
from lenscluster.config import ImageDiagnosticsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lenscluster.sampling import adaptation_diagnostics, e_bfmi


def test_e_bfmi_separates_independent_and_random_walk_energies():
    rng = np.random.default_rng(1)
    independent = rng.normal(size=4000)                       # energies redrawn each transition: E-BFMI near 2
    random_walk = np.cumsum(0.05 * rng.normal(size=4000))      # energies that barely move between transitions
    assert 1.8 < e_bfmi(independent) < 2.2 and e_bfmi(random_walk) < 0.05
    out = adaptation_diagnostics(np.array([0.1, 0.02]), np.stack([independent, random_walk]), np.full((2, 4000), 0.9),
                                 np.zeros((2, 4000), bool), np.stack([np.full(4000, 7.0), np.full(4000, 255.0)]), max_tree_depth=8)
    assert [c["step_size"] for c in out["per_chain"]] == [0.1, 0.02] and out["min_e_bfmi"] == out["per_chain"][1]["e_bfmi"]
    assert out["per_chain"][0]["tree_depth_saturation"] == 0.0 and out["per_chain"][1]["tree_depth_saturation"] == 1.0


def test_step_sizes_and_energies_are_saved_and_reloaded(synthetic_config):
    config = replace(synthetic_config, runtime=RuntimeConfig(seed=2, chains=2, nuts_chain_method="sequential", resume=True),
                     search=SearchConfig(n_starts=4, lbfgs_max_iter=30, polish_max_iter=10),
                     sampling=SamplingConfig(warmup=20, samples=15, max_tree_depth=4),
                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=2))
    result = pipeline.run(config)
    posterior = result.posterior
    assert posterior.step_size.shape == (2,) and np.all(posterior.step_size > 0)
    assert posterior.energy.shape == (30,) and np.isfinite(posterior.energy).all()
    assert np.all(posterior.energy >= -posterior.log_prob - 1e-9)        # H = U + K with K >= 0
    adaptation = posterior.diagnostics["adaptation"]
    assert [c["chain"] for c in adaptation["per_chain"]] == [0, 1] and adaptation["max_tree_depth"] == 4
    np.testing.assert_allclose([c["step_size"] for c in adaptation["per_chain"]], posterior.step_size)
    reloaded = artifacts.load_posterior(result.paths.final.artifacts)
    np.testing.assert_array_equal(reloaded.step_size, posterior.step_size)
    np.testing.assert_array_equal(reloaded.energy, posterior.energy)
    assert reloaded.diagnostics["adaptation"]["per_chain"] == adaptation["per_chain"]
    summary = (result.paths.final.tables / "run_summary.txt").read_text()
    assert "step size per chain" in summary and "E-BFMI per chain" in summary and "tree-depth cap" in summary


def test_dense_member_offsets_join_the_hessian_seeded_block(synthetic_config):
    """SamplingConfig.dense_member_offsets: the member-offset sites move from the diagonal into the dense block, whose
    Hessian seed (chunked forward-over-reverse) is symmetric positive definite; the sources stay diagonal."""
    import numpy as np
    from lenscluster.evaluator import Evaluator
    from lenscluster.sampling import block_mass_seed, posterior_model
    from lenscluster.state import build_state
    state = build_state(synthetic_config)
    specs = state.parameter_specs
    model = posterior_model(specs, Evaluator(state).log_likelihood)
    theta = np.asarray(state.initial_theta, dtype=float)
    n_offsets = sum(s.component_family == "member_scatter_offset" for s in specs)
    assert n_offsets > 0
    plain, _, d0 = block_mass_seed(model, specs, theta)
    dense, inverse, d1 = block_mass_seed(model, specs, theta, dense_families=("member_scatter_offset",))
    assert d1["global_dim"] == d0["global_dim"] + n_offsets and d1["latent_dim"] == d0["latent_dim"] - n_offsets
    block = np.asarray(inverse[dense[0]])
    assert block.shape == (d1["global_dim"], d1["global_dim"])
    np.testing.assert_allclose(block, block.T, atol=1e-10)
    assert np.linalg.eigvalsh(block).min() > 0
