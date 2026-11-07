from types import SimpleNamespace
import numpy as np
from lenscluster.mass_figure import shared_grf_convergence


def test_shared_grf_redshift_scaling_and_chunk_padding(monkeypatch):
    from lenscluster import jax_cosmology
    monkeypatch.setattr(jax_cosmology, 'dpie_sigma0_factor_from_config', lambda zl,z,c: z)
    def correction(x,y,f):
        # Family zero has weight 3: alpha_x=3*x^2, alpha_y=3*y^2.
        return 3*x*x,3*y*y,6*x,np.zeros_like(x),np.zeros_like(x),6*y
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(gp_blocks='all',z_lens=.3,
        gp_reference_redshift=2.,cosmo_config=None),gp_weights_per_family=np.array([3.]),
        gp_deflection_correction=lambda theta:correction)
    x,y=np.meshgrid(np.arange(3.),np.arange(2.))
    result=shared_grf_convergence(e,None,dict(x=x,y=y),4.,chunk=4)
    np.testing.assert_allclose(result,2*(x+y))


def test_no_shared_map_for_system_blocks():
    e=SimpleNamespace(gp_knee=True,state=SimpleNamespace(gp_blocks='system'))
    assert shared_grf_convergence(e,None,None,2.) is None
