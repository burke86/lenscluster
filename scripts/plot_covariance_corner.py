"""Regenerate the covariance corner from a saved run; no inference."""
import argparse
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    from lenscluster.covariance_plots import plot_covariance_corner
    config, state, _, posterior, paths = load_run(args.run_directory)
    print(plot_covariance_corner(config, state, posterior, paths))


if __name__ == '__main__':
    main()
