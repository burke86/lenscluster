"""Normalization of the illustrative, scatter-free input mass distribution."""
import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc

from lenscluster.mock_validation.aggregate_plots import _schechter_curve


def test_reference_normalization_and_generator_cdf():
    selection = dict(schechter_alpha=-0.7, dpie_mass_anchor_msun=8e11,
                     mass_luminosity_exponent=1.2, faint_magnitude_limit=24,
                     pivot_magnitude=19.82)
    lower, upper = 10 ** (-0.4 * (24 - 19.82)), 20.0
    mass = lambda ell: np.log10(8e11) + 1.2 * np.log10(ell)
    density = lambda x: float(_schechter_curve(selection, np.asarray(x), 200, (0.02, upper)))
    np.testing.assert_allclose(quad(density, mass(lower), mass(upper), epsabs=1e-9)[0], 200, rtol=1e-12)
    # Independent incomplete-gamma CDF used by the generator.
    shape = 0.3
    expected = 200 * (gammainc(shape, 1) - gammainc(shape, lower)) / (gammainc(shape, upper) - gammainc(shape, lower))
    np.testing.assert_allclose(quad(density, mass(lower), mass(1))[0], expected, rtol=1e-12)
    np.testing.assert_array_equal(_schechter_curve(selection, np.array([mass(lower) - 1, mass(upper) + 1]), 200, (0.02, upper)), 0)
    # Evaluating a smaller plotting range must not renormalize the curve.
    np.testing.assert_allclose(density(mass(1)), _schechter_curve(selection, np.array([mass(1), mass(2)]), 200, (0.02, upper))[0])


def test_missing_selection_does_not_invent_reference():
    assert _schechter_curve({}, np.array([10.0]), 200) is None
