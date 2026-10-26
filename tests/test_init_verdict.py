from lenscluster.pipeline import init_verdict, search_is_fragile


def _search(modes, n_starts=8):
    return {"modes": modes, "starts": [[0.0]] * n_starts, "method": "lbfgs"}


def test_fragile_when_only_the_configured_start_reaches_the_basin():
    search = _search([{"log_posterior": -10.0, "within_margin": True, "count": 1, "members": [0]},
                      {"log_posterior": -80.0, "within_margin": False, "count": 7, "members": list(range(1, 8))}])
    assert search_is_fragile(search)
    assert init_verdict(search, fit=None).startswith("verdict: fragile")


def test_robust_when_independent_starts_share_the_basin():
    search = _search([{"log_posterior": -10.0, "within_margin": True, "count": 3, "members": [0, 4, 5]},
                      {"log_posterior": -12.0, "within_margin": True, "count": 2, "members": [1, 2]}])
    assert not search_is_fragile(search)
    assert "5 of 8" in init_verdict(search, fit=None)


def test_fragile_when_a_single_random_start_reaches_the_basin():
    search = _search([{"log_posterior": -10.0, "within_margin": True, "count": 1, "members": [3]},
                      {"log_posterior": -80.0, "within_margin": False, "count": 7, "members": [0, 1, 2, 4, 5, 6, 7]}])
    search.update(starts_in_basin=1, configured_in_basin=False, n_finite=2)
    assert search_is_fragile(search)
    verdict = init_verdict(search, fit=None)
    assert verdict.startswith("verdict: fragile") and "6 of 8 starts invalid" in verdict
