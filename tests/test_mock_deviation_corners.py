import numpy as np
from types import SimpleNamespace
from lenscluster.mock_validation import deviation_corners as plots


def test_standardized_draws_keep_bias_and_joint_correlation():
    x=np.linspace(1.,5.,101)
    values=np.c_[x,2*x+4]
    result,scale=plots.standardized_draws(values,[1.,4.])
    np.testing.assert_allclose(result.mean(0),(values.mean(0)-[1.,4.])/scale)
    np.testing.assert_allclose(np.corrcoef(result.T)[0,1],1.)
    assert result[:,0].mean()>0


def test_corner_weights_each_mock_equally_and_pairs_sources(tmp_path,monkeypatch):
    def load(root,seed):
        x=np.random.default_rng(seed).normal(size=100*seed)
        y=np.random.default_rng(seed+10).normal(size=len(x))+.3*x
        return {'source.01.beta_x':x,'source.01.beta_y':y}, {'source.01.beta_x':.2,'source.01.beta_y':-.2}
    monkeypatch.setattr(plots,'load_columns',load)
    campaign=SimpleNamespace(root=tmp_path,seeds=[1,2])
    paths=plots.write_deviation_corners(campaign,tmp_path)
    assert len(paths)==1 and paths[0].read_bytes().startswith(b'%PDF')
    with np.load(tmp_path/'mock_sources_truth_deviation_corner.npz') as data:
        assert data['values'].shape==(300,2)
        for seed in [1,2]:np.testing.assert_allclose(data['weights'][data['seeds']==seed].sum(),.5)
        expected,_=plots.standardized_draws(np.column_stack(list(load(tmp_path,1)[0].values())),[.2,-.2])
        np.testing.assert_allclose(data['values'][:100],expected)


def test_combined_draws_preserve_alignment_and_equal_weights():
    loaded = {}
    for seed in [1, 2]:
        rng = np.random.default_rng(seed)
        halo = rng.normal(size=seed*100)
        columns = {'halo.v_disp': halo}
        truth = {'halo.v_disp': 0.}
        for family in range(seed):
            for coord in ['x', 'y']:
                name = f'source.{family}.beta_{coord}'
                columns[name] = (family+1)*halo + rng.normal(size=len(halo))
                truth[name] = .1
        loaded[seed] = columns, truth
    data = plots.combined_draws(loaded)
    assert data['values'].shape == (500, 3)
    for seed in [1, 2]:
        mask = data['seeds'] == seed
        np.testing.assert_allclose(data['weights'][mask].sum(), .5)
        columns, truth = loaded[seed]
        for family in range(seed):
            mask = (data['seeds'] == seed) & (data['families'] == str(family))
            np.testing.assert_allclose(data['weights'][mask].sum(), .5/seed)
            names = ['halo.v_disp', f'source.{family}.beta_x', f'source.{family}.beta_y']
            expected, _ = plots.standardized_draws(np.column_stack([columns[p] for p in names]), [truth[p] for p in names])
            np.testing.assert_allclose(data['values'][mask], expected)
            np.testing.assert_array_equal(data['draw_indices'][mask], np.arange(seed*100))
