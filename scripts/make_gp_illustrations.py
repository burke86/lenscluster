"""Regenerate standard GP diagnostics, including shared-field maps, from a saved run."""
import argparse

from _covariance_plot_cli import load_run
from lenscluster.gp_plots import write_gp_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gp_run')
    args = parser.parse_args()
    config, state, evaluator, posterior, paths = load_run(args.gp_run)
    if not evaluator.gp_mode:
        parser.error('the run must use likelihood_mode gp_image_plane')
    write_gp_plots(config, state, evaluator, posterior, paths)


if __name__ == '__main__':
    main()
