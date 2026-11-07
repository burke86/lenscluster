"""Own depth and depth error of the shallow fold pairs at a saved run's best fit, for the gp_image_plane pre-run check.

Works on a run of any likelihood mode: only the lens model enters.  For a gp_kernel="knee" run the
--sigma-shared/--sigma-white values are ignored: the hyperparameters come from the draw itself.  ell and sigma_fl default to the saved configuration (or
the library defaults for a run without them) and can be overridden to see what a planned run would get.
"""
import argparse
import json
from dataclasses import replace

import numpy as np

from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--ell', type=float, default=None, help='kernel correlation length (arcsec)')
    parser.add_argument('--floor', type=float, default=None, help='sigma_fl (arcsec)')
    parser.add_argument('--sigma-shared', type=float, default=0.15, help='sigma_shared assumed for the depth error (arcsec)')
    parser.add_argument('--sigma-white', type=float, default=0.02, help='sigma_white assumed for the depth error (arcsec)')
    parser.add_argument('--max-separation', type=float, default=3.0, help='pairs closer than this (arcsec)')
    parser.add_argument('--output', default=None, help='JSON table; default final/tables/shallow_fold_pairs.json')
    args = parser.parse_args()
    from lenscluster.gp_image_plane import shallow_fold_pair_report, shallow_fold_pairs
    config, state, evaluator, posterior, paths = load_run(args.run_directory)
    likelihood = config.likelihood
    ell = args.ell if args.ell is not None else getattr(likelihood, 'gp_correlation_length_arcsec', 7.0)
    floor = args.floor if args.floor is not None else getattr(likelihood, 'gp_floor_arcsec', 0.0015)
    state = replace(state, gp_correlation_length_arcsec=float(ell), gp_floor_arcsec=float(floor))
    best = int(np.nanargmax(posterior.log_prob))
    rows = shallow_fold_pairs(state, evaluator, posterior.samples[best], max_separation_arcsec=args.max_separation,
                              sigma_shared_arcsec=args.sigma_shared, sigma_white_arcsec=args.sigma_white)
    print(shallow_fold_pair_report(rows, float(floor), args.sigma_shared, args.sigma_white), flush=True)
    output = args.output or paths.tables / 'shallow_fold_pairs.json'
    with open(output, 'w') as handle:
        json.dump(dict(pairs=rows, model=f'best posterior draw {best}', sigma_shared_arcsec=args.sigma_shared, sigma_white_arcsec=args.sigma_white, ell_arcsec=ell,
                       floor_arcsec=floor, max_separation_arcsec=args.max_separation), handle, indent=2)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
