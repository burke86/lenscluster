"""Render an RGB with z=9 critical curves; use --imaging saved for simulated HST."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing init/ and final/")
    parser.add_argument("--imaging", choices=("saved", "hst", "jwst"), default="jwst",
                        help="Imaging for the background: the run's saved configuration, the HST (BUFFALO) mosaics, or the JWST NIRCam mosaics (suffix _jwst)")
    args = parser.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from lenscluster import artifacts, plots
    from lenscluster.cutouts import imaging_variants, load_cluster_imaging
    from lenscluster.evaluator import Evaluator
    from lenscluster.state import build_state, with_member_classes

    run = args.run_directory.resolve()
    init, final = run / "init" / "artifacts", run / "final"
    config = artifacts.load_config(final / "artifacts")
    if args.imaging != "saved":
        from dataclasses import replace
        from lenscluster.rgb import DEFAULT_HFF_RGB_BANDS, DEFAULT_JWST_RGB_BANDS
        jwst = args.imaging == "jwst"
        from lenscluster.config import RGBDisplayConfig
        # The run's saved display numbers belong to the imaging it was run with; the library defaults carry the balance.
        config = replace(config, image_catalog=replace(config.image_catalog, image_dir="data/JWST_Images" if jwst else "data/BUFFALO_Images",
                                                       image_scale="auto", bands=tuple(DEFAULT_JWST_RGB_BANDS if jwst else DEFAULT_HFF_RGB_BANDS),
                                                       rgb=RGBDisplayConfig(), figure_suffix="_jwst" if jwst else ""))
    from dataclasses import replace
    config = replace(config, image_catalog=imaging_variants(config, overview_only=True)[0])
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered_ids = set(classes.get("centered_members", []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered_ids for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved_state = json.loads((init / artifacts.STATE_FILE).read_text())
    if [p.name for p in state.parameter_specs] != [p["name"] for p in saved_state["parameter_specs"]]:
        raise ValueError("Rebuilt parameter order differs from saved run")
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    posterior = artifacts.load_posterior(final / "artifacts")
    log_prob = np.asarray(posterior.log_prob)
    if not np.isfinite(log_prob).any():
        raise ValueError("Saved posterior has no finite best-fit draw")
    best_index = int(np.nanargmax(log_prob))
    theta = np.asarray(posterior.samples[best_index])
    np.testing.assert_allclose(evaluator.reported_physical(theta), posterior.physical[best_index], rtol=1e-10, atol=1e-10)
    catalog = pd.read_csv(final / "tables" / "image_fit_quality.csv", dtype={"image_label": str})
    try:
        extra = pd.read_csv(final / "tables" / "image_recovery_extra_images.csv", dtype={"family_id": str})
    except pd.errors.EmptyDataError:
        extra = pd.DataFrame()  # No extra images were recovered in this saved run.
    from lenscluster.jwst_rgb_figure import render_jwst_rgb_critical
    imaging = load_cluster_imaging(config.image_catalog, config.model, config.paths.run_name)
    render_jwst_rgb_critical(config,state,evaluator,theta,catalog,extra,imaging,
        plots.figure_path(final/'figures','hst_rgb_critical_z9.pdf' if imaging[3].lower() in {'hera','ares'} else 'jwst_rgb_critical_z9.pdf',config.paths.figure_prefix),
        final/'tables',best_index=best_index,posterior=posterior)

if __name__=='__main__':
    main()
