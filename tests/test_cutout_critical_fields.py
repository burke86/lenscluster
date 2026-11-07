from types import SimpleNamespace as NS
import numpy as np
from lenscluster.cutout_critical_fields import median_cutout_determinants


def test_redshifts_median_and_cache(tmp_path):
    def fields(xy,z,**kwargs):
        h=np.zeros((len(xy),2,2));h[:,0,0]=1-xy[:,0]+z
        return None,h
    evaluator=NS(gp_knee=False,physical=lambda t:t,
                 packed_state_for_z=lambda p,z:z,total_fields_at=fields)
    posterior=NS(samples=np.zeros((3,1)))
    args=(NS(),evaluator,posterior,[[0,0],[0,0]],[10,10],[1,2],np.arange(3),tmp_path/'critical.npz','test')
    grids,median,indices,lower,upper=median_cutout_determinants(*args,resolution=16,log=lambda _:None)
    np.testing.assert_allclose(median,grids[...,0]-np.array([1,2])[:,None,None],atol=1e-14)
    np.testing.assert_array_equal(lower,median)
    np.testing.assert_array_equal(upper,median)
    evaluator.total_fields_at=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Expected cache'))
    np.testing.assert_array_equal(median_cutout_determinants(*args,resolution=16,log=lambda _:None)[1],median)
