"""Regenerate model-identified fold-pair PDFs from at most 128 saved posterior draws."""
import argparse
from dataclasses import replace
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    from lenscluster.cutouts import imaging_variants, load_cluster_imaging
    from lenscluster.fold_diagnostics import compute_fold_diagnostics, save_fold_data, render_fold_diagnostics
    config, state, evaluator, posterior, paths = load_run(args.run_directory)
    data = compute_fold_diagnostics(config, state, evaluator, posterior, log=lambda message: print(message, flush=True))
    save_fold_data(data, paths.tables)
    for image_config in imaging_variants(config):
        try:
            imaging = load_cluster_imaging(image_config, config.model, config.paths.run_name)
        except FileNotFoundError as error:
            print(f'[outputs] fold RGB skipped: {error}')
            continue
        render_fold_diagnostics(replace(config, image_catalog=image_config), state, data, paths, imaging)


if __name__ == '__main__':
    main()
