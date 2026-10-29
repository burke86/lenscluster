"""Independent references for the smoothed determinant and source transport."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm

from lenscluster.smoothed_likelihood import (
    expected_absolute_normal, inverse_source_transport, log_expected_absolute_normal, source_covariance,
    source_whitening, statistics, transport_sources,
)


def fixture_arrays():
    rng = np.random.default_rng(92)
    n = 4
    a = rng.normal(size=(n, 2, 2))
    a = (a + a.swapaxes(-1, -2)) / 2
    return dict(backprojected=rng.normal(size=(n, 2)) * .07, source=rng.normal(size=(n, 2)) * .03,
                jacobian=a, determinant_gradient=rng.normal(size=(n, 2)),
                sigma_meas=np.array([.03, .05, .1, .07]), sigma_def=.04)


def test_conditioning_matches_independent_latent_gaussian():
    args = fixture_arrays()
    out = statistics(**args)
    for i in range(4):
        a, grad = args['jacobian'][i], args['determinant_gradient'][i]
        so, ss = args['sigma_meas'][i], args['sigma_def']
        # F = A e + eta with e ~ N(0, so^2), eta ~ N(0, ss^2); X = det A - g.e
        b = np.concatenate((so * a, ss * np.eye(2)), axis=1)
        w = np.concatenate((-so * grad, np.zeros(2)))
        c = b @ b.T
        cross = b @ w
        f = args['backprojected'][i] - args['source'][i]
        mean = np.linalg.det(a) + cross @ np.linalg.solve(c, f)
        var = w @ w - cross @ np.linalg.solve(c, cross)
        np.testing.assert_allclose(out['covariance'][i], c, rtol=2e-14, atol=1e-17)
        np.testing.assert_allclose(out['cross_covariance'][i], cross, rtol=2e-14, atol=1e-17)
        np.testing.assert_allclose(out['determinant_mean'][i], mean, rtol=2e-14)
        np.testing.assert_allclose(out['determinant_variance'][i], var, rtol=2e-13, atol=1e-17)
        expected_h = np.sqrt(var) * np.sqrt(2 / np.pi) * np.exp(-mean**2 / (2 * var)) + mean * (2 * ndtr(mean / np.sqrt(var)) - 1)
        expected_log = multivariate_normal.logpdf(f, mean=np.zeros(2), cov=c) + np.log(expected_h)
        np.testing.assert_allclose(out['loglike'][i], expected_log, rtol=2e-13, atol=1e-12)
    assert np.all(out['valid'])


@pytest.mark.parametrize('z', [-1000., -30., -8.01, -8., -7.99, -3., -.1, 0., .1, 3., 8., 30.])
def test_branch_moments_against_positive_integral(z):
    if z < -5:
        a = -z
        integral = quad(lambda y: y * np.exp(-y - y * y / (2 * a * a)), 0, np.inf, epsabs=2e-13, epsrel=2e-13)[0]
        reference = norm.logpdf(z) - 2 * np.log(a) + np.log(integral)
    else:
        reference = np.log(quad(lambda y: y * norm.pdf(y - z), 0, max(z, 0) + 40, epsabs=2e-13, epsrel=2e-13)[0])
    actual = log_expected_absolute_normal(z, 1., 1)
    np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=4e-12)
    np.testing.assert_allclose(log_expected_absolute_normal(-z, 1., -1), actual, atol=1e-14)
    plus, minus = expected_absolute_normal(z, 1., 1), expected_absolute_normal(z, 1., -1)
    np.testing.assert_allclose(expected_absolute_normal(z, 1.), plus + minus, rtol=2e-14)


def test_deterministic_limits_degeneracy_and_safe_gradients():
    for mean in [-2., 2.]:
        np.testing.assert_allclose(log_expected_absolute_normal(mean, 0.), np.log(2.))
        np.testing.assert_allclose(log_expected_absolute_normal(mean, 0., np.sign(mean)), np.log(2.))
        assert np.isneginf(log_expected_absolute_normal(mean, 0., -np.sign(mean)))
        value, grad = jax.value_and_grad(lambda x: log_expected_absolute_normal(x, 0.))(mean)
        assert np.isfinite(value) and np.isfinite(grad)
        np.testing.assert_allclose(grad, 1 / mean)
    assert np.isneginf(log_expected_absolute_normal(0., 0.))
    assert np.isnan(log_expected_absolute_normal(1., -1.))
    assert np.isnan(log_expected_absolute_normal(1., 1., 2))
    out = statistics(jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2, 2)), jnp.zeros((1, 2)), jnp.ones(1) * .1, .03)
    assert out['degenerate'][0] and not out['valid'][0]
    assert np.isneginf(out['loglike'][0])


@pytest.mark.parametrize('parity', [0, -1, 1])
def test_branch_log_gradient_in_tails_and_near_switch(parity):
    f = jax.jit(jax.value_and_grad(lambda z: log_expected_absolute_normal(z, .7**2, parity)))
    for z in [-40., -5.6001, -5.6, -5.5999, -.01, 0., .01, 4., 40.]:
        val, grad = f(z)
        assert np.isfinite(val) and np.isfinite(grad)
        step = 1e-5
        numeric = (log_expected_absolute_normal(z + step, .7**2, parity) - log_expected_absolute_normal(z - step, .7**2, parity)) / (2 * step)
        np.testing.assert_allclose(grad, numeric, rtol=2e-6, atol=2e-7)


def test_fold_crossing_matches_displacement_quadrature():
    sigma_meas, sigma_def = .025, .02
    for position in [-.03, 0., .03]:
        a = np.diag([1.2, position])
        grad = np.array([0., .8])
        f = np.array([.008, -.006])
        result = statistics(f[None], np.zeros((1, 2)), a[None], grad[None], np.array([sigma_meas]), sigma_def)
        xfactor = norm.pdf(f[0], scale=np.sqrt(sigma_def**2 + (1.2 * sigma_meas)**2))
        integral = quad(lambda u: abs(np.linalg.det(a) + grad[1] * u) * norm.pdf(u, scale=sigma_meas) * norm.pdf(f[1] + a[1, 1] * u, scale=sigma_def),
                        -.4, .4, points=[-np.linalg.det(a) / grad[1]], epsabs=1e-12)[0]
        np.testing.assert_allclose(np.exp(result['loglike'][0]), xfactor * integral, rtol=2e-10)


def test_full_likelihood_gradients():
    base = fixture_arrays()

    def objective(theta):
        args = dict(base)
        args['source'] = jnp.asarray(base['source']) + theta[0]
        args['jacobian'] = jnp.asarray(base['jacobian']) * jnp.exp(theta[1])
        args['determinant_gradient'] = jnp.asarray(base['determinant_gradient']) * jnp.exp(theta[2])
        args['sigma_def'] = .04 * jnp.exp(theta[3])
        return statistics(**args)['loglike'].sum()

    theta = jnp.array([.004, .03, -.05, .08])
    val, grad = jax.jit(jax.value_and_grad(objective))(theta)
    assert np.isfinite(val) and np.isfinite(grad).all()
    for i in range(4):
        step = np.eye(4)[i] * 1e-5
        numerical = (objective(theta + step) - objective(theta - step)) / 2e-5
        np.testing.assert_allclose(grad[i], numerical, rtol=3e-6, atol=1e-7)
    assert np.isfinite(jax.jit(jax.hessian(objective))(theta)).all()


def test_source_transport_roundtrip_and_physical_target_identity():
    args = fixture_arrays()
    cov = source_covariance(args['jacobian'], args['sigma_meas'], args['sigma_def'])['covariance']
    index = jnp.array([0, 1, 0, 1])
    whitening = source_whitening(args['backprojected'], cov, index, 2)
    offsets = jnp.array([[.3, -.7], [.1, 1.2]])
    prior_mean = jnp.array([[1., -.3], [-.4, .8]])
    prior_sigma = jnp.array([2., 3.])
    transformed = transport_sources(offsets, whitening, prior_mean, prior_sigma)
    np.testing.assert_allclose(inverse_source_transport(transformed['source'], whitening), offsets, rtol=2e-14, atol=2e-14)
    for f in range(2):
        rows = np.asarray(index) == f
        precision = np.linalg.inv(np.asarray(cov)[rows]).sum(0)
        expected_cov = np.linalg.inv(precision)
        expected_centroid = expected_cov @ np.einsum('nij,nj->i', np.linalg.inv(np.asarray(cov)[rows]), args['backprojected'][rows])
        np.testing.assert_allclose(whitening['centroid'][f], expected_centroid, rtol=2e-14)
        np.testing.assert_allclose(whitening['covariance'][f], expected_cov, rtol=2e-14)
        physical_prior = multivariate_normal.logpdf(transformed['source'][f], mean=prior_mean[f], cov=np.eye(2) * prior_sigma[f]**2)
        base_prior = multivariate_normal.logpdf(offsets[f], mean=np.zeros(2), cov=np.eye(2))
        jac = np.linalg.slogdet(np.asarray(whitening['cholesky'][f]))[1]
        np.testing.assert_allclose(base_prior + transformed['log_correction_per_family'][f], physical_prior + jac, rtol=2e-14)
    jacobian = jax.jacfwd(lambda u: transport_sources(u.reshape(2, 2), whitening, prior_mean, prior_sigma)['source'].ravel())(offsets.ravel())
    np.testing.assert_allclose(np.linalg.slogdet(jacobian)[1], transformed['log_jacobian'].sum(), rtol=2e-14)


def test_source_transport_lens_gradient_and_empty_family_flag():
    base = fixture_arrays()
    family = jnp.array([0, 1, 0, 1])

    def target(t):
        beta = jnp.asarray(base['backprojected']) + t * .2
        cov = source_covariance(jnp.asarray(base['jacobian']) * jnp.exp(t), base['sigma_meas'], .04)['covariance']
        w = source_whitening(beta, cov, family, 2)
        return transport_sources(jnp.ones((2, 2)) * .4, w, jnp.zeros((2, 2)), 2.)['log_correction']

    value, grad = jax.jit(jax.value_and_grad(target))(.03)
    assert np.isfinite([value, grad]).all()
    np.testing.assert_allclose(grad, (target(.03001) - target(.02999)) / .00002, rtol=1e-7)
    w = source_whitening(base['backprojected'], jnp.tile(jnp.eye(2), (4, 1, 1)), family, 3)
    np.testing.assert_array_equal(w['valid'], [True, True, False])


def test_negative_scatter_flagged():
    args = fixture_arrays()
    assert not np.any(statistics(**dict(args, sigma_def=-.1))['valid'])


@pytest.mark.parametrize('parity', [0, -1, 1])
def test_zero_mean_analytic_gradient_and_curvature(parity):
    variance = .09
    fn = lambda x: log_expected_absolute_normal(x[0], x[1], parity)
    value, gradient = jax.jit(jax.value_and_grad(fn))(jnp.array([0., variance]))
    hessian = jax.jit(jax.hessian(fn))(jnp.array([0., variance]))
    expected_log = .5 * np.log(variance) + .5 * np.log(2 / np.pi) - (np.log(2) if parity else 0.)
    expected_mean_derivative = parity * np.sqrt(np.pi / (2 * variance))
    expected_mean_curvature = (1. - np.pi / 2) / variance if parity else 1. / variance
    np.testing.assert_allclose(value, expected_log, rtol=1e-14)
    np.testing.assert_allclose(gradient, [expected_mean_derivative, .5 / variance], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(hessian[0, 0], expected_mean_curvature, rtol=1e-14)
    assert np.isfinite(hessian).all()


def test_selection_term_eigenvalues_and_switch_off():
    """sigma_sel^2 R has eigenvalues sigma_sel^2 / (1 + lambda^2 / lambda_sel^2) along the Jacobian's principal directions
    (half the variance where |lambda| = lambda_sel) and vanishes when sigma_sel = 0."""
    a = np.array([[0.1, 0.0], [0.0, 2.0]])            # tangential stretch 10x, radial compression 2x
    meas, de, sel, res = 0.05, 0.02, 0.04, 0.6
    cov = np.asarray(source_covariance(a[None], meas, de, sel, res)["covariance"][0])
    expected = np.diag([meas**2 * 0.1**2 + de**2 + sel**2 / (1 + 0.1**2 / res**2),
                        meas**2 * 2.0**2 + de**2 + sel**2 / (1 + 2.0**2 / res**2)])
    np.testing.assert_allclose(cov, expected, rtol=1e-12)
    half = np.asarray(source_covariance(np.diag([res, res])[None], 0.0, 0.0, sel, res)["covariance"][0])
    np.testing.assert_allclose(half, np.diag([sel**2 / 2, sel**2 / 2]), rtol=1e-12)
    off = np.asarray(source_covariance(a[None], meas, de, 0.0, res)["covariance"][0])
    np.testing.assert_allclose(off, np.diag([meas**2 * 0.01 + de**2, meas**2 * 4.0 + de**2]), rtol=1e-12)
    # the determinant conditioning stays the exact Gaussian conditioning with the selection term in C
    args = fixture_arrays()
    out = statistics(**args, sigma_sel=sel, lambda_sel=res)
    for i in range(4):
        a_i, grad = args["jacobian"][i], args["determinant_gradient"][i]
        c = np.asarray(source_covariance(a_i[None], args["sigma_meas"][i], args["sigma_def"], sel, res)["covariance"][0])
        cross = -args["sigma_meas"][i]**2 * a_i @ grad
        f = args["backprojected"][i] - args["source"][i]
        np.testing.assert_allclose(out["determinant_mean"][i], np.linalg.det(a_i) + cross @ np.linalg.solve(c, f), rtol=1e-10)
        np.testing.assert_allclose(out["determinant_variance"][i], args["sigma_meas"][i]**2 * grad @ grad - cross @ np.linalg.solve(c, cross), rtol=1e-9)


def test_smoothed_likelihood_with_exact_determinant_is_the_image_plane_gaussian():
    """With sigma_def = sigma_sel = 0 and |det A| in place of h_i the smoothed likelihood is
    -0.5 |A^-1 F|^2 / sigma_meas^2 - ln(2 pi sigma_meas^2)."""
    args = fixture_arrays()
    args["sigma_def"] = 0.0
    out = statistics(**args)
    for i in range(4):
        a, f, sigma = args["jacobian"][i], args["backprojected"][i] - args["source"][i], args["sigma_meas"][i]
        expected = -0.5 * np.sum(np.linalg.solve(a, f) ** 2) / sigma ** 2 - np.log(2 * np.pi * sigma ** 2)
        actual = float(out["gaussian_loglike"][i]) + np.log(abs(np.linalg.det(a)))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
