"""Recompute truth maps and dependent figures with the shared conditional GRF.

Uses the saved posterior and configured diagnostic draw count. No inference.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from _covariance_plot_cli import load_run
    from lenscluster import plots
    from lenscluster.truth_recovery import compute_truth_recovery
    from lenscluster.exact_images import astropy_cosmology
    from lenscluster.magnification_recovery_plot import write_magnification_recovery
    from lenscluster.results_tex import write_run_results
    for run in args.runs:
        config,state,evaluator,posterior,paths=load_run(run)
        tables,figures=paths.tables,paths.figures
        images=pd.read_csv(tables/'image_fit_quality.csv',dtype={'family_id':str,'image_label':str})
        truth=compute_truth_recovery(state,evaluator,np.asarray(posterior.samples),config.truth,
            run/'final/fits',tables,seed=config.runtime.seed,log=lambda text:print(text,flush=True))
        if truth is None:
            raise ValueError(f'No truth maps configured for {run}')
        prefix=config.paths.figure_prefix
        plots.plot_truth_recovery_residuals(truth,images,plots.figure_path(figures,'truth_recovery_residuals.pdf',prefix))
        if truth['profile'] is not None:
            plots.plot_m2d_aperture_ratio(truth['profile'],truth['center'],images,
                truth['mass_per_kappa_pixel_msun'],plots.figure_path(figures,'truth_recovery_m2d_aperture_ratio.pdf',prefix))
        write_magnification_recovery(truth,images,state.reference,state.z_lens,
            astropy_cosmology(state.cosmo_config),config.truth,figures,tables,prefix)
        best=posterior.samples[int(np.nanargmax(posterior.log_prob))]
        plots.plot_critical_curves(state,evaluator,best,np.asarray(evaluator.source_positions(best)),
            truth['z_source'],plots.figure_path(figures,'critical_curves.pdf',prefix))
        summary_path=tables/'run_summary.json'
        if summary_path.exists() and truth['profile'] is not None:
            summary=json.loads(summary_path.read_text())
            summary['truth']={'m2d_ratio_at_max_radius':float(truth['profile']['m2d_ratio_median'].iloc[-1]),
                              'n_radii':len(truth['profile'])}
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        write_run_results(run/'final',name=prefix)
        print(f'Regenerated truth and critical-curve figures: {figures}',flush=True)


if __name__=='__main__':
    main()
