from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lenscluster.plots import population_scaling_draws


def test_population_curves_preserve_joint_draws_and_ignore_member_offsets():
    specs = [SimpleNamespace(name=f"potfile_1.{field}")
             for field in ("sigma", "cutkpc", "alpha_sigma", "beta_radius")]
    posterior = np.array([[100., 10., .25, .5], [200., 4., .3, .7]])
    table = pd.DataFrame(dict(potfile_id=["potfile_1"] * 2, catalog_mag=[20., 22.5],
                              luminosity_ratio=[1., .1], cut_radius_kpc_median=[999., 999.]))
    mag, curves = population_scaling_draws(table, posterior, specs)["potfile_1"]
    np.testing.assert_allclose(curves["v_disp"][:, 0], [100., 200.])
    np.testing.assert_allclose(curves["cut_radius_kpc"][:, 0], [10., 4.])
    np.testing.assert_allclose(curves["v_disp"][:, -1], [100 * .1**.25, 200 * .1**.3])
    np.testing.assert_allclose(curves["cut_radius_kpc"][:, -1], [10 * .1**.5, 4 * .1**.7])
    # M scales as L^(2 alpha + beta), using each joint draw's own exponents.
    np.testing.assert_allclose(curves["log10_mass_msun"][:, -1] - curves["log10_mass_msun"][:, 0], [-1., -1.3])
    assert len(mag) == 240
    table.loc[1, "luminosity_ratio"] = .2
    with pytest.raises(ValueError, match="pivot"):
        population_scaling_draws(table, posterior, specs)
