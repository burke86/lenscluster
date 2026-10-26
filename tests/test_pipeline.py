"""End-to-end single stage on the synthetic toy cluster."""
from __future__ import annotations

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import artifacts, pipeline
from lenscluster.config import ImageDiagnosticsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state


def _fast(config):
    return replace(config, runtime=RuntimeConfig(seed=3, chains=1, nuts_chain_method="sequential", resume=True),
                   search=SearchConfig(n_starts=4, lbfgs_max_iter=30), sampling=SamplingConfig(warmup=10, samples=20, max_tree_depth=4),
                   image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=2))


def test_single_stage_end_to_end(synthetic_config):
    config = _fast(synthetic_config)
    result = pipeline.run(config)
    state, evaluator, posterior = result.state, result.evaluator, result.posterior
    assert posterior.samples.shape == (20, state.n_parameters)
    assert np.isfinite(posterior.log_prob).all()

    # Sources are the Gaussian centroid plus the whitened offsets, beta = c + L u.
    theta = jnp.asarray(posterior.samples[-1])
    stats = evaluator.statistics(theta)
    whitening = stats["whitening"]
    offsets = np.stack((posterior.samples[-1][state.images.source_x_param_index], posterior.samples[-1][state.images.source_y_param_index]), -1)
    expected = np.asarray(whitening["centroid"]) + np.einsum("fij,fj->fi", np.asarray(whitening["cholesky"]), offsets)
    np.testing.assert_allclose(np.asarray(stats["source_by_family"]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(evaluator.source_positions(theta)), expected, rtol=1e-12, atol=1e-12)

    # The final classification was made at the best mode in the init stage and is the model the final stage sampled.
    init_dir = result.paths.init.artifacts
    saved = json.loads((init_dir / artifacts.CLASSIFICATION_FILE).read_text())
    assert saved["exact_count"] == len(state.members.exact_positions) == state.members.classification["exact_count"]
    assert set(np.asarray(artifacts.load_exact_positions(init_dir)).tolist()) == set(state.members.exact_positions.tolist())
    np.testing.assert_array_equal(artifacts.load_start(init_dir), result.init.theta)
    assert state.members.classification["exact_count"] + len(state.members.far_positions) == len(state.members.member_ids)

    # Audit and outputs exist.
    assert result.audit["approximation_validation"] in {"passed", "failed"}
    for name in ("posterior_summary.csv", "member_classification.csv", "image_fit_quality.csv", "member_scaling_relation.csv", "run_summary.json"):
        assert (result.paths.final.tables / name).is_file()
    assert (result.paths.final.figures / "critical_curves.pdf").is_file()
    assert (result.paths.init.tables / "classification_report.txt").is_file()

    # Resuming reloads both stages without searching or sampling again.
    again = pipeline.run(config)
    assert again.resumed and again.init.resumed
    np.testing.assert_array_equal(again.posterior.samples, posterior.samples)
    np.testing.assert_array_equal(again.init.theta, result.init.theta)
    # The init stage alone resumes too; a changed sampling setting under the same run name is refused
    # rather than silently overwriting the saved posterior.
    assert pipeline.run(config, stage="init").resumed
    table = (result.paths.final.tables / "run_summary.txt").read_text()
    assert "Run summary" in table and "recovered / observed" in table and "total" in table
    changed = replace(config, sampling=SamplingConfig(warmup=10, samples=12, max_tree_depth=4))
    with pytest.raises(ValueError, match="different configuration"):
        pipeline.run(changed)


def test_far_field_barrier_is_finite_smooth_and_pushes_back(synthetic_config):
    from lenscluster.state import with_member_classes
    state = build_state(synthetic_config)
    state = with_member_classes(state, np.asarray([0, 1], dtype=np.int32), {"exact_count": 2, "mandatory_count": 2})   # two far-field members
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta, dtype=float)
    cut_index = next(i for i, s in enumerate(state.parameter_specs) if s.field == "cutkpc")

    def barrier(latent):
        physical = evaluator.physical(latent)
        return evaluator.far_field_log_barrier(evaluator.fields(physical)["far_cut_radii"])

    ratio0 = float(np.min(np.asarray(evaluator.far_min_separation) / np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(theta)))["far_cut_radii"])))
    # Scale the relation's cut radius so the nearest far-field member sits at these multiples of the guard distance.
    targets = (3.0, 1.5, 1.1, 1.0, 0.9, 0.5)
    values, slopes = [], []
    for target in targets:
        trial = theta.copy()
        trial[cut_index] += np.log(ratio0 / (target * evaluator.guard_cut_radii))
        values.append(float(barrier(jnp.asarray(trial))))
        slopes.append(float(jax.grad(barrier)(jnp.asarray(trial))[cut_index]))
        value, grad = evaluator.value_and_grad(jnp.asarray(trial))
        assert np.isfinite(float(value)) and np.isfinite(np.asarray(grad)).all()
    assert all(v <= 0 for v in values) and values == sorted(values, reverse=True)      # never positive, worse as r_cut grows
    assert values[0] > -1e-3                                                            # negligible three guard distances out
    assert abs(values[3] + np.log(2.0) * evaluator.far_field_barrier_strength) < 1.0    # about -0.7 strength at the guard (one member there)
    assert slopes[-1] < 0 and values[-1] < -evaluator.far_field_barrier_strength        # steep inside, gradient points back out
