"""JWST cutouts of images closest to a singular Jacobian, with 1- and 2-sigma position contours."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--count',type=int,default=24)
    parser.add_argument('--fov',type=float,default=3.,help='Minimum square field of view in arcseconds; expand to enclose 2-sigma contour')
    parser.add_argument('--image-dir',default='/data/sync/GalaxyClusterImaging/JWST_Images')
    args=parser.parse_args()
    if args.count<1 or args.fov<=0:parser.error('count and fov must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from dataclasses import replace
    from _covariance_plot_cli import load_run
    from lenscluster.cutouts import imaging_variants,load_cluster_imaging
    config,state,evaluator,posterior,paths=load_run(args.run)
    view=next(v for v in imaging_variants(config,overview_only=True) if v.figure_suffix=='_jwst')
    view=replace(view,image_dir=args.image_dir)
    config=replace(config,image_catalog=view)
    try:extra=pd.read_csv(paths.tables/'image_recovery_extra_images.csv',dtype={'family_id':str})
    except (FileNotFoundError, pd.errors.EmptyDataError):extra=pd.DataFrame()
    from lenscluster.image_position_contours import load_or_compute_contours
    from lenscluster.small_lambda_jwst import write_small_lambda_jwst
    contours=load_or_compute_contours(config,state,evaluator,posterior,extra,paths.tables)
    imaging=load_cluster_imaging(view,config.model,config.paths.run_name)
    write_small_lambda_jwst(config,state,evaluator,posterior,paths,imaging,contours,
                          count=args.count,minimum_fov=args.fov)


if __name__=='__main__':main()
