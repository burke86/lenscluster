"""Regenerate model-identified fold-pair PDFs from at most 128 saved posterior draws."""
import argparse
from dataclasses import replace
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--imaging', choices=('saved', 'jwst'), default='saved')
    parser.add_argument('--recompute', action='store_true', help='Recompute fold geometry instead of reusing saved diagnostics')
    args = parser.parse_args()
    from lenscluster.cutouts import imaging_variants, load_cluster_imaging
    from lenscluster.fold_diagnostics import compute_fold_diagnostics, save_fold_data, render_fold_diagnostics, load_fold_data, add_system_knots
    config, state, evaluator, posterior, paths = load_run(args.run_directory)
    import numpy as np
    log = lambda message: print(message, flush=True)
    cached = paths.tables / 'fold_pairs_source_caustics.json'
    data = load_fold_data(paths.tables) if cached.exists() and not args.recompute else None
    if data is not None and data.get('best_original_draw') != int(np.nanargmax(posterior.log_prob)):
        data = None
    if data is None:
        data = compute_fold_diagnostics(config, state, evaluator, posterior, log=log)
    elif 'systems' not in data:
        data = add_system_knots(config, state, evaluator, posterior, data, log=log)
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
