"""Numerical checks for posterior field uncertainty and spatial kernels."""
import importlib.util
from pathlib import Path
import sys
import numpy as np


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    spec = importlib.util.spec_from_file_location('posterior_field_plot', scripts/'plot_mock_posterior_fields.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_total_variance_includes_model_uncertainty(monkeypatch):
    mod = module(monkeypatch)
    mean, within, between = mod.mixture([[1., 2.], [3., 6.]], [[4., 1.], [4., 9.]])
    np.testing.assert_allclose(mean, [2., 4.])
    np.testing.assert_allclose(within+between, [5., 9.])
    # Direct second-moment calculation of the Gaussian mixture.
    second = (np.array([[4., 1.], [4., 9.]])+np.square([[1., 2.], [3., 6.]])).mean(0)
    np.testing.assert_allclose(within+between, second-mean**2)


def test_spatial_kernel_orientation_and_zero_separation(monkeypatch):
    mod = module(monkeypatch)
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    # Covariance from a squared-exponential scalar potential, differentiated
    # analytically: exp(-r^2/2) (I - d d^T).
    radii = np.array([0., 1., np.sqrt(2.)])
    transverse = np.exp(-radii**2/2)
    longitudinal = (1-radii**2)*transverse
    actual = mod.kernel(points,points,radii,longitudinal,transverse,2.)
    expected = np.empty((6,6))
    for i,a in enumerate(points):
        for j,b in enumerate(points):
            d=a-b
            expected[2*i:2*i+2,2*j:2*j+2]=2*np.exp(-d@d/2)*(np.eye(2)-np.outer(d,d))
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert np.linalg.eigvalsh(actual).min()>0
