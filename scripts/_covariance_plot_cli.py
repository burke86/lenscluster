"""Saved-run adapter for covariance figure regeneration."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load_run(run_directory):
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from lenscluster import artifacts
    from lenscluster.evaluator import Evaluator
    from lenscluster.state import build_state, with_member_classes
    run = Path(run_directory).resolve()
    init, final = run / 'init/artifacts', run / 'final'
    config = artifacts.load_config(final / 'artifacts')
    # Older saved GP runs predate the sampled-length option and used a fixed ell.
    if not hasattr(config.likelihood, 'gp_correlation_length_prior'):
        from dataclasses import replace
        config = replace(config, likelihood=replace(config.likelihood, gp_correlation_length_prior=None))
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get('dropped_members', ()))
    centered = set(classes.get('centered_members', []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved = json.loads((init / artifacts.STATE_FILE).read_text())
    assert [p.name for p in state.parameter_specs] == [p['name'] for p in saved['parameter_specs']]
    assert list(state.members.member_ids) == saved['member_ids']
    posterior = artifacts.load_posterior(final / 'artifacts')
    best = int(np.nanargmax(posterior.log_prob))
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    np.testing.assert_allclose(evaluator.reported_physical(posterior.samples[best]), posterior.physical[best], rtol=1e-10, atol=1e-10)

    paths = SimpleNamespace(figures=final / 'figures', tables=final / 'tables')
    return config, state, evaluator, posterior, paths
