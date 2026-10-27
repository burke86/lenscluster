"""Regenerate the member and likelihood diagnostic suite from a saved run; no inference."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--draws', type=int, default=32, help='Joint field draws in addition to best fit; 0 means best only')
    parser.add_argument('--output-dir', type=Path, help='Optional directory for figures/ and tables/')
    args = parser.parse_args()
    if args.draws < 0:
        parser.error('--draws must be nonnegative')
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from lenscluster import artifacts
    from lenscluster.evaluator import Evaluator
    from lenscluster.state import build_state, with_member_classes
    from lenscluster.member_diagnostic_plots import write_member_diagnostic_plots

    run = args.run_directory.resolve()
    init = run / 'init/artifacts'
    final = run / 'final'
    config = artifacts.load_config(final / 'artifacts')
    saved = json.loads((init / artifacts.STATE_FILE).read_text())
    config = replace(config, image_diagnostics=replace(config.image_diagnostics, posterior_diagnostic_draws=args.draws))
    state = build_state(config)
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    centered = set(classes.get('centered_members', []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    if [p.name for p in state.parameter_specs] != [p['name'] for p in saved['parameter_specs']]:
        raise ValueError('Rebuilt parameter identities/order differ from saved run')
    if list(state.members.member_ids) != saved['member_ids']:
        raise ValueError('Rebuilt member identities/order differ from saved run')
    posterior = artifacts.load_posterior(final / 'artifacts')
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    best = int(np.nanargmax(posterior.log_prob))
    np.testing.assert_allclose(evaluator.reported_physical(posterior.samples[best]), posterior.physical[best], rtol=1e-10, atol=1e-10)
    destination = args.output_dir or final
    paths = SimpleNamespace(figures=destination / 'figures', tables=destination / 'tables')
    for path in write_member_diagnostic_plots(config, state, evaluator, posterior, paths):
        print(path)


if __name__ == '__main__':
    main()
