import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.model import ParameterSpec

jax.config.update("jax_enable_x64", True)


def _normal_spec(name: str, mean: float, std: float) -> ParameterSpec:
    potential_id, field = name.split(".")
    return ParameterSpec(
        name=name,
        sample_name=name.replace(".", "_"),
        potential_id=potential_id,
        profile_type=81,
        field=field,
        prior_kind="normal",
        lower=float("-inf"),
        upper=float("inf"),
        step=0.1,
        mean=mean,
        std=std,
        transform_kind="identity",
    )


def _quadratic_posterior_model(specs: list[ParameterSpec], curvatures: np.ndarray):
    """Prior sites plus a diagonal quadratic likelihood with known curvature."""
    curvature_values = jnp.asarray(curvatures, dtype=jnp.float64)

    def model():
        values = []
        for site in cs._parameter_sample_sites(specs):
            values.append(numpyro.sample(site.name, cs._distribution_for_sample_site(site, specs)))
        theta = jnp.stack(values)
        numpyro.factor("loglike", -0.5 * jnp.sum(curvature_values * jnp.square(theta)))

    return model


def test_mass_seed_matches_quadratic_curvature() -> None:
    specs = [
        _normal_spec("a.x", 0.0, 1.0),
        _normal_spec("b.y", 0.0, 2.0),
        _normal_spec("c.z", 0.0, 1.0),
    ]
    curvatures = np.array([99.0, 0.0, 9999.0])
    model = _quadratic_posterior_model(specs, curvatures)
    theta0 = np.zeros(3)
    inverse_mass, diagnostics = cs._hessian_seeded_inverse_mass(None, model, specs, theta0)
    assert inverse_mass is not None
    assert diagnostics["used"] is True
    assert diagnostics["dim"] == 3
    matrix = np.asarray(inverse_mass)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    # Identity transforms + normal priors: unconstrained curvature is
    # likelihood curvature + 1/std^2, per parameter, in sorted site order
    # (a_x, b_y, c_z). b.y has 0.25 prior curvature, floored to 1.0.
    expected = np.array([1.0 / (99.0 + 1.0), 1.0 / cs.NUTS_MASS_SEED_EIG_FLOOR, 1.0 / (9999.0 + 0.25)])
    assert np.allclose(np.diag(matrix), expected, rtol=1.0e-6)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    assert np.all(np.abs(off_diagonal) < 1.0e-9)
    # Floored spectrum must be invertible and bounded by the floor.
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.max() <= 1.0 / cs.NUTS_MASS_SEED_EIG_FLOOR + 1.0e-9
    assert eigenvalues.min() > 0.0


def test_mass_seed_runs_inside_numpyro_nuts() -> None:
    specs = [
        _normal_spec("a.x", 0.0, 1.0),
        _normal_spec("b.y", 0.0, 2.0),
    ]
    model = _quadratic_posterior_model(specs, np.array([50.0, 5.0]))
    inverse_mass, _ = cs._hessian_seeded_inverse_mass(None, model, specs, np.zeros(2))
    assert inverse_mass is not None
    kernel = cs.NUTS(model, dense_mass=True, inverse_mass_matrix=inverse_mass, max_tree_depth=4)
    mcmc = cs.MCMC(kernel, num_warmup=5, num_samples=5, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    for site_name in ("a_x", "b_y"):
        assert np.all(np.isfinite(np.asarray(samples[site_name])))


def test_mass_seed_non_finite_hessian_raises() -> None:
    specs = [_normal_spec("a.x", 0.0, 1.0)]

    def model():
        value = numpyro.sample(
            "a_x", cs._distribution_for_sample_site(cs._parameter_sample_sites(specs)[0], specs)
        )
        numpyro.factor("loglike", jnp.nan * jnp.square(value))

    with pytest.raises(RuntimeError, match="non-finite Hessian"):
        cs._hessian_seeded_inverse_mass(None, model, specs, np.zeros(1))


def test_mass_seed_computes_above_dimension_warning_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [
        _normal_spec("a.x", 0.0, 1.0),
        _normal_spec("b.y", 0.0, 1.0),
    ]
    model = _quadratic_posterior_model(specs, np.array([1.0, 1.0]))
    monkeypatch.setattr(cs, "NUTS_MASS_SEED_MAX_DIM", 1)
    inverse_mass, diagnostics = cs._hessian_seeded_inverse_mass(None, model, specs, np.zeros(2))
    assert np.asarray(inverse_mass).shape == (2, 2)
    assert diagnostics["used"] is True


def test_mass_seed_empty_specs_raises() -> None:
    with pytest.raises(RuntimeError, match="at least one sampled parameter"):
        cs._hessian_seeded_inverse_mass(None, object(), [], np.zeros(0))
