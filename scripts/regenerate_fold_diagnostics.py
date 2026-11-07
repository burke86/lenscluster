"""Regenerate model-identified fold-pair PDFs from at most 128 saved posterior draws."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--imaging', choices=('saved', 'jwst'), default='saved')
    parser.add_argument('--recompute', action='store_true', help='Recompute fold geometry instead of reusing saved diagnostics')
    args = parser.parse_args()
    from lenscluster.cutouts import imaging_variants, load_cluster_imaging
    from lenscluster.fold_diagnostics import compute_fold_diagnostics, save_fold_data, render_fold_diagnostics, load_fold_data
    from lenscluster import artifacts
    import numpy as np
    log = lambda message: print(message, flush=True)
    final = Path(args.run_directory).resolve() / 'final'
    paths = SimpleNamespace(tables=final/'tables', figures=final/'figures')
    cached = paths.tables / 'fold_pairs_source_caustics.json'
    data = load_fold_data(paths.tables) if cached.exists() and not args.recompute else None
    if data is not None:
        # A rendering operation must not reinterpret saved latent source coordinates.
        # The cache already contains image positions, sources, curves and bands.
        config = artifacts.load_config(final/'artifacts')
        with np.load(final/'artifacts/posterior.npz') as posterior:
            best = int(np.nanargmax(posterior['log_prob']))
            count = len(posterior['samples'])
        if data['best_original_draw'] != best or data['posterior_draws_available'] != count:
            raise ValueError('Fold cache does not match the saved posterior; use --recompute')
        reference = config.model.reference
        state = SimpleNamespace(reference=(reference.reference, reference.ra0_deg, reference.dec0_deg))
        log(f'[outputs] rendering saved fold cache: {len(data["draw_indices"])} draws; gp_blocks={config.likelihood.gp_blocks}')
    else:
        config, state, evaluator, posterior, paths = load_run(args.run_directory)
        data = compute_fold_diagnostics(config, state, evaluator, posterior, log=log)
        save_fold_data(data, paths.tables)
    for image_config in imaging_variants(config, overview_only=True):
        if args.imaging == 'jwst' and image_config.figure_suffix != '_jwst':
            continue
        try:
            imaging = load_cluster_imaging(image_config, config.model, config.paths.run_name)
        except FileNotFoundError as error:
            print(f'[outputs] fold RGB skipped: {error}')
            continue
        render_fold_diagnostics(replace(config, image_catalog=image_config), state, data, paths, imaging)


if __name__ == '__main__':
    main()
