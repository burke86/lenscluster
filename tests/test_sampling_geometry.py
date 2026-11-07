import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from lenscluster.model import ParameterSpec
from lenscluster.sampling import bound_pressure, lanczos_extremes, slow_modes


def _spec(name, **kw):
    base = dict(name=name, sample_name=name, potential_id="p", profile_type=0, field=name, prior_kind="normal",
                lower=-np.inf, upper=np.inf, step=0.1, mean=0.0, std=1.0, component_family="halo", transform_kind="identity")
    base.update(kw)
    return ParameterSpec(**base)


def test_lanczos_finds_both_ends_of_a_quadratic():
    eigen = np.array([1e-2, 0.5, 1.0, 3.0, 40.0, 900.0])
    rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))
    hessian = rotation @ np.diag(eigen) @ rotation.T
    potential = lambda x: 0.5 * x @ jnp.asarray(hessian) @ x
    values, stiff, soft = lanczos_extremes(potential, jnp.zeros(6), lambda v, transpose=False: v, steps=6)
    assert np.isclose(values.max(), 900.0) and np.isclose(values.min(), 1e-2)
    assert abs(abs(stiff @ rotation[:, -1]) - 1) < 1e-8 and abs(abs(soft @ rotation[:, 0]) - 1) < 1e-8


def test_slow_modes_point_at_the_parameter_the_chains_disagree_on():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(4, 500, 3))
    draws[:, :, 1] += np.array([0.0, 5.0, -5.0, 0.0])[:, None]        # chains sit 5 sd apart in parameter b only
    modes = slow_modes(draws, [_spec("a"), _spec("b"), _spec("c")])
    assert modes["modes"][0]["groups"][0][0] == "b" and modes["modes"][0]["singular_value"] > 5


def test_bound_pressure_flags_draws_against_a_bound():
    rng = np.random.default_rng(2)
    piled = np.concatenate([np.full(300, 0.995), rng.uniform(0, 1, 700)])
    draws = np.stack([piled, rng.uniform(0.2, 0.8, 1000)], -1).reshape(4, 250, 2)
    specs = [_spec("x", prior_kind="uniform", lower=0.0, upper=1.0), _spec("y", prior_kind="uniform", lower=0.0, upper=1.0)]
    rows = bound_pressure(draws, specs)
    assert [r["parameter"] for r in rows] == ["x"] and rows[0]["upper_fraction"] > 0.25
