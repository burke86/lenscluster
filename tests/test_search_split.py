"""The init search: a large-scale fit without members, then the members with the large family frozen, then everything."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.config import SearchConfig
from lenscluster.evaluator import Evaluator
from lenscluster.search import prior_starts
from lenscluster.search_split import split_map, transfer_by_name
from lenscluster.state import build_state


@pytest.mark.parametrize("methods", [("lbfgs", "lbfgs"), ("adam", "svi")])
def test_split_map_transfers_halos_and_returns_a_finite_mode(synthetic_config, methods):
    search = SearchConfig(n_starts=4, halo_method=methods[0], member_method=methods[1], member_starts=2,
                          lbfgs_max_iter=30, polish_max_iter=10, adam_steps=50, svi_steps=50)
    config = replace(synthetic_config, search=search)
    state = build_state(config)
    evaluator = Evaluator(state)
    result = split_map(config, state, evaluator, seed=3, log=lambda *a, **k: None)
    theta = np.asarray(result["theta"], dtype=float)
    assert theta.shape == (state.n_parameters,) and np.isfinite(theta).all()
    assert np.isfinite(float(evaluator.loglike(jnp.asarray(theta))))
    assert result["method"] == f"{methods[0]}+{methods[1]}" and [s["stage"] for s in result["stages"]] == ["large_scale", "members"]
    assert result["stages"][0]["active"] < result["stages"][1]["active"] and result["final_theta"].shape[1] == state.n_parameters


def test_transfer_by_name_copies_shared_parameters_only(synthetic_config):
    full = build_state(synthetic_config)
    halo = build_state(replace(synthetic_config, model=replace(synthetic_config.model, member_populations=())))
    source = np.arange(halo.n_parameters, dtype=float) + 100.0
    target = np.zeros(full.n_parameters)
    out = transfer_by_name(halo, source, full, target)
    names = [s.name for s in full.parameter_specs]
    for i, spec in enumerate(halo.parameter_specs):
        assert out[names.index(spec.name)] == source[i]
    assert (out == 0.0).sum() == full.n_parameters - halo.n_parameters


def test_optax_lbfgs_matches_scipy_on_the_synthetic_cluster(synthetic_config):
    from lenscluster.search import multistart_map
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    starts = prior_starts(state, 4, 5)
    base = SearchConfig(n_starts=4, lbfgs_max_iter=60, polish_max_iter=10)
    quiet = lambda *a, **k: None
    scipy_result = multistart_map(state, evaluator, starts, base, method="lbfgs", seed=5, log=quiet)
    optax_result = multistart_map(state, evaluator, starts, base, method="optax-lbfgs", seed=5, log=quiet)
    assert optax_result["method"] == "optax-lbfgs" and np.isfinite(optax_result["theta"]).all()
    assert abs(optax_result["modes"][0]["log_posterior"] - scipy_result["modes"][0]["log_posterior"]) < 2.0
