from types import SimpleNamespace
import numpy as np
from lenscluster.block_corner import posterior_corner_data, write_posterior_block_corner, BLOCK_COLORS


def spec(name, family='large', field='v_disp', transform='identity'):
    return SimpleNamespace(name=name, potential_id=name.rsplit('.',1)[0], field=field,
                           component_family=family, transform_kind=transform, transform_offset=0.)


def test_blocks_and_joint_source_alignment(tmp_path):
    specs=[spec('halo.v_disp'),spec('bcg_1.v_disp'),spec('images.gp_knee','error_scale'),
           spec('potfile.sigma','scaling'),spec('member.z','member_scatter_offset')]
    for family in ['01','02']:
        specs.extend([spec(f'source.{family}.beta_{axis}','source_position',f'beta_{axis}') for axis in ['x','y']])
    samples=np.random.default_rng(5).normal(size=(100,len(specs)))
    data=posterior_corner_data(samples,specs)
    assert data['groups']==list(BLOCK_COLORS)[:4]+['Sources','Sources']
    assert data['values'].shape==(200,6)
    np.testing.assert_allclose(data['weights'].sum(),1)
    for j in range(2):
        np.testing.assert_array_equal(data['values'][100*j:100*(j+1),:4],samples[:,:4])
        raw=samples[:,5+2*j:7+2*j]
        lo,med,hi=np.percentile(raw,[16,50,84],axis=0)
        np.testing.assert_allclose(data['values'][100*j:100*(j+1),4:],(raw-med)/((hi-lo)/2))
    assert data['excluded']==['member.z']
    path=write_posterior_block_corner(samples,specs,tmp_path/'combined.pdf')
    assert path.read_bytes().startswith(b'%PDF')
    assert path.with_suffix('.json').exists()


def test_log_values_and_constant_exclusion():
    samples=np.c_[np.exp(np.linspace(1,3,100)),np.ones(100)]
    specs=[spec('halo.v_disp',transform='log_positive'),spec('halo.fixed')]
    data=posterior_corner_data(samples,specs)
    np.testing.assert_allclose(data['values'][:,0],np.log10(samples[:,0]))
    assert data['excluded']==['halo.fixed']
