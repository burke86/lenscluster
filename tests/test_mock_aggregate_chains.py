from types import SimpleNamespace
import numpy as np
import pytest
from lenscluster.mock_validation import aggregate_chains as plots


def test_chain_shape_rejects_incorrect_flat_order(tmp_path):
    grouped=np.arange(24.).reshape(2,6,2)
    path=tmp_path/'posterior.npz'
    np.savez(path,grouped_samples=grouped,samples=grouped.reshape(12,2),physical=grouped.reshape(12,2)*3)
    assert plots.chain_shape(path)==(2,6)
    np.savez(path,grouped_samples=grouped,samples=grouped.reshape(12,2)[::-1],physical=grouped.reshape(12,2))
    with pytest.raises(ValueError,match='draw order'):plots.chain_shape(path)


def test_joint_traces_and_sources_match_corner(tmp_path,monkeypatch):
    def load(root,seed):
        rng=np.random.default_rng(seed)
        cols={'halo.v_disp':rng.normal(2,1,40),
              'source.a.beta_x':rng.normal(1,1,40),'source.a.beta_y':rng.normal(2,1,40),
              'source.b.beta_x':rng.normal(3,1,40),'source.b.beta_y':rng.normal(4,1,40)}
        return cols,{name:0. for name in cols}
    monkeypatch.setattr(plots,'load_columns',load)
    monkeypatch.setattr(plots,'chain_shape',lambda path:(2,20))
    campaign=SimpleNamespace(root=tmp_path,seeds=[1,2])
    data,failures=plots.collect_chains(campaign)
    assert not failures
    assert list(data['parameters'])==['halo.v_disp','source.beta_x','source.beta_y']
    for seed in [1,2]:
        values=data['mocks'][seed]['values']
        assert values.shape==(2,2,20,3)
        np.testing.assert_array_equal(values[0,:,:,0],values[1,:,:,0])
        cols,_=load(None,seed)
        raw=cols['source.b.beta_x']
        expected=raw/((np.percentile(raw,84)-np.percentile(raw,16))/2)
        np.testing.assert_allclose(values[1,:,:,1].ravel(),expected)
    path=plots.write_aggregate_chains(campaign,tmp_path)
    assert path.read_bytes().startswith(b'%PDF')
    assert path.with_suffix('.csv').exists()
