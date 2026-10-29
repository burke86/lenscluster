"""Regenerate covariance errors from a saved run; no inference."""
import argparse
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--draws', type=int, default=32, help='Additional joint draws; zero uses best fit only')
    parser.add_argument('--plane', choices=('source', 'image', 'both'), default='both')
    args = parser.parse_args()
    if args.draws < 0:
        parser.error('--draws must be nonnegative')
    from lenscluster.covariance_plots import plot_covariance_errors
    print(plot_covariance_errors(*load_run(args.run_directory), draws=args.draws, plane=args.plane))


if __name__ == '__main__':
    main()
