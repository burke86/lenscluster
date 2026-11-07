"""Publication cutouts of saved model-identified folds; no source-plane panels or inference."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from lenscluster.plot_style import CRITICAL_COLOR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument('--max-separation',type=float,default=3.,help='Strict upper limit in arcseconds')
    parser.add_argument('--grf', action=argparse.BooleanOptionalAction, default=True,
                        help='Plot the smoothed parametric + GRF median curve and its band (default: enabled)')
    parser.add_argument('--critical-color', default=CRITICAL_COLOR, help='Critical curve and matching uncertainty-band color')
    parser.add_argument('--output', type=Path, help='Alternative PDF output path for style previews')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    from lenscluster import artifacts
    from lenscluster.fold_diagnostics import load_fold_data
    from lenscluster.cutouts import imaging_variants, load_cluster_imaging
    final=args.run_directory.resolve()/'final'
    config=artifacts.load_config(final/'artifacts')
    data=load_fold_data(final/'tables')
    with np.load(final/'artifacts/posterior.npz') as post:
        assert data['best_original_draw']==int(np.nanargmax(post['log_prob']))
        assert data['posterior_draws_available']==len(post['samples'])
    if args.grf:
        from _apj_fold_grf import prepare
        data=prepare(args.run_directory, data)
    views=[c for c in imaging_variants(config,overview_only=True) if c.figure_suffix=='_jwst']
    if not views: raise ValueError('No JWST imaging configured for this run')
    imaging=load_cluster_imaging(views[0],config.model,config.paths.run_name)
    from lenscluster.apj_fold_cutouts import render_apj_fold_cutouts
    output=args.output or final/'figures'/f'{config.paths.figure_prefix}_fold_cutouts_apj_jwst.pdf'
    result=render_apj_fold_cutouts(config,data,imaging,output,max_separation=args.max_separation,
        critical_color=args.critical_color,run_directory=args.run_directory.resolve(),
        source_cache=final/('tables/apj_fold_grf/fold_pairs_source_caustics.npz' if args.grf else 'tables/fold_pairs_source_caustics.npz'))
    print(result or 'No fold pairs below the requested separation',flush=True)

if __name__=='__main__':main()
