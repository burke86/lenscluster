from types import SimpleNamespace
import numpy as np
import pandas as pd
from lenscluster.mock_validation.field_recovery_plots import plot_field_recovery


def test_aggregate_field_recovery_uses_median_model_and_mock_bins(tmp_path):
    output=tmp_path/'aggregate_plots';output.mkdir()
    for seed, n, value in [(1,3,1.),(2,1,3.)]:
        np.savez(output/f'seed_{seed:08d}_posterior_fields.npz',xy=np.zeros((n,2)),
                 truth_images=np.ones(4*n),median_parameter_images=np.full(4*n,value),
                 mean_images=np.full(4*n,999.),within_variance_images=np.full(4*n,.04),
                 between_variance_images=np.full(4*n,.05))
    campaign=SimpleNamespace(root=tmp_path,seeds=[1,2,3])
    path=plot_field_recovery(campaign,output)
    assert path.read_bytes().startswith(b'%PDF')
    assert len(list(output.glob('*.pdf')))==1
    rows=pd.read_csv(output/'mock_grf_gp_field_recovery.csv')
    np.testing.assert_allclose(rows.posterior_sd,.3)
    assert set(rows['median'])=={1.,3.}
    bins=pd.read_csv(output/'mock_grf_gp_field_recovery_bins.csv')
    # Unequal image counts must not overweight mock 1.
    np.testing.assert_allclose(bins.recovered_mean,2.)
    np.testing.assert_allclose(bins.recovered_sem,1.)
    assert 'Seeds missing field diagnostics: [3]' in (output/'mock_grf_gp_field_recovery.txt').read_text()


def test_no_saved_fields_produces_no_plot(tmp_path):
    assert plot_field_recovery(SimpleNamespace(root=tmp_path,seeds=[1]),tmp_path/'out') is None
    assert not (tmp_path/'out').exists()
