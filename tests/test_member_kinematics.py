"""The member kinematics term (Evaluator.kinematics_log_likelihood): it equals the log-normal terms computed by hand from
the catalog, its gradient matches finite differences, it is exactly absent when no source is on, and the sources it
uses are checked (one per galaxy, each found on a member on the relation, YAML catalogs only)."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state
from test_member_catalog import _yaml_config

MEASURED = {"g1": (120.0, 3.0), "g2": (95.0, 6.0)}


def _with_kinematics(config, tmp_path, *, sources=("K",), extra=None, systematic=0.05):
    def edit(members):
        for name, (sigma, err) in MEASURED.items():
            members[name]["kinematics"] = [{"source": "K", "sigma": sigma, "err": err}]
        if extra:
            extra(members)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _yaml_config(config, tmp_path, edit)
    population = replace(config.model.member_populations[0], kinematics_sources=tuple(sources), kinematics_systematic=systematic)
    return replace(config, model=replace(config.model, member_populations=(population,))).validate()


def test_the_term_matches_the_log_normal_by_hand(synthetic_config, tmp_path):
    config = _with_kinematics(synthetic_config, tmp_path)
    state = build_state(config)
    evaluator = Evaluator(state, "smoothed")
    theta = np.asarray(state.initial_theta, dtype=float) + 0.02 * np.random.default_rng(4).normal(size=state.n_parameters)
    physical = evaluator.physical(jnp.asarray(theta))
    sigma = np.asarray(evaluator.component_dispersions(physical))
    c = config.model.member_populations[0].kinematics_aperture_factor
    expected = 0.0
    for name, (measured, err) in MEASURED.items():
        component = next(r["component_index"] for r in state.scaling_component_records if r["catalog_id"] == name)
        e = np.hypot(err / measured, 0.05)
        expected += -0.5 * ((np.log(measured) - np.log(c * sigma[component])) / e) ** 2 - np.log(e) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(float(evaluator.kinematics_log_likelihood(physical)), expected, rtol=1e-12)
    # it is part of log_likelihood: the difference to the same model without kinematics is the term
    plain = replace(config, model=replace(config.model, member_populations=(replace(config.model.member_populations[0], kinematics_sources=()),)))
    without = Evaluator(build_state(plain.validate()), "smoothed")
    np.testing.assert_allclose(float(evaluator.log_likelihood(jnp.asarray(theta))) - float(without.log_likelihood(jnp.asarray(theta))), expected, rtol=1e-9)


def test_the_gradient_matches_finite_differences(synthetic_config, tmp_path):
    state = build_state(_with_kinematics(synthetic_config, tmp_path))
    evaluator = Evaluator(state, "smoothed")
    term = lambda t: evaluator.kinematics_log_likelihood(evaluator.physical(t))
    theta = jnp.asarray(state.initial_theta, dtype=jnp.float64)
    grad = np.asarray(jax.grad(term)(theta))
    assert np.count_nonzero(grad) >= 3                       # sigma*, alpha and the two measured deviations at least
    step = 1e-6
    for i in np.flatnonzero(grad):
        e = np.zeros(theta.size); e[i] = step
        fd = (float(term(theta + e)) - float(term(theta - e))) / (2 * step)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-5, atol=1e-7)


def test_without_sources_the_likelihood_is_unchanged(synthetic_config, tmp_path):
    with_data = _with_kinematics(synthetic_config, tmp_path, sources=())
    state, reference = build_state(with_data), build_state(synthetic_config)
    assert not state.member_kinematics
    values = [float(Evaluator(s, "smoothed").log_likelihood(jnp.asarray(s.initial_theta))) for s in (state, reference)]
    assert values[0] == values[1]


def test_sources_are_checked(synthetic_config, tmp_path):
    with pytest.raises(ValueError, match="on no member"):
        build_state(_with_kinematics(synthetic_config, tmp_path / "a", sources=("K", "L")))
    def second(members):
        members["g1"]["kinematics"].append({"source": "L", "sigma": 118.0, "err": 2.0})
    from test_member_catalog import SOURCES
    SOURCES["L"] = {"reference": "another", "quantity": "sigma_ap", "aperture_arcsec": 1.0}
    try:
        with pytest.raises(ValueError, match="one source per galaxy"):
            build_state(_with_kinematics(synthetic_config, tmp_path / "b", sources=("K", "L"), extra=second))
    finally:
        del SOURCES["L"]
    potfile = replace(synthetic_config.model.member_populations[0], kinematics_sources=("K",))
    with pytest.raises(ValueError, match="YAML member catalog"):
        replace(synthetic_config, model=replace(synthetic_config.model, member_populations=(potfile,))).validate()


def test_the_kinematics_table_reports_each_member(synthetic_config, tmp_path):
    from lenscluster.pipeline import kinematics_table
    state = build_state(_with_kinematics(synthetic_config, tmp_path))
    evaluator = Evaluator(state, "smoothed")
    table = kinematics_table(state, evaluator, state.initial_theta)
    assert list(table["member"]) == ["members:g1", "members:g2"]
    total = -0.5 * float(np.sum(table["z"] ** 2)) - float(np.sum(np.log(state.member_kinematics["log_err"]))) - np.log(2 * np.pi)
    np.testing.assert_allclose(total, float(evaluator.kinematics_log_likelihood(evaluator.physical(jnp.asarray(state.initial_theta)))), rtol=1e-12)
