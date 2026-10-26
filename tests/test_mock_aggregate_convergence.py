import json

from lenscluster.mock_validation.aggregate_plots import load_campaign


def test_aggregate_rechecks_saved_verdict_at_current_cut(tmp_path):
    for seed, rhat, neff, saved_verdict in [(1, 1.009, 50, False), (2, 1.01, 100, True),
                                             (3, 1.04, 100, True), (4, 1.009, 49, True)]:
        directory = tmp_path / f"seed_{seed:08d}" / "recovery"
        directory.mkdir(parents=True)
        payload = dict(seed=seed, max_r_hat=rhat, min_n_eff=neff, converged=saved_verdict,
                       convergence_reasons=['old verdict'], parameters=[], magnification=[],
                       image_residual_draws=[], m2d=[], subhalos={})
        (directory / 'recovery.json').write_text(json.dumps(payload))
    campaign = load_campaign(tmp_path)
    assert campaign.seeds == [1]
    assert campaign.summary.converged.tolist() == [True, False, False, False]
    assert '1.01' in campaign.summary.iloc[1].convergence_reasons
    assert load_campaign(tmp_path, converged_only=False).seeds == [1, 2, 3, 4]
