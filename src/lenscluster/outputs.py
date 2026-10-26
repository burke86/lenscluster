"""Tables and figures written after a run: posterior summary, image recovery, scaling relation, truth recovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import plots
from .chain_traces import write_chain_traces
from .config import LensClusterSolverConfig
from .diagnostics import fit_summary, image_recovery_tables
from .evaluator import Evaluator
from .model import ModelState, PosteriorResults
from .potfile_plots import write_potfile_plots
from .member_diagnostic_plots import write_member_diagnostic_plots
from .truth_recovery import compute_truth_recovery
from .results_tex import write_run_results


def posterior_summary_table(state: ModelState, posterior: PosteriorResults) -> pd.DataFrame:
    physical = np.asarray(posterior.physical, dtype=float)
    latent = np.asarray(posterior.samples, dtype=float)
    diagnostics = posterior.diagnostics or {}
    rows = []
    for index, spec in enumerate(state.parameter_specs):
        q = np.percentile(physical[:, index], [16, 50, 84])
        rows.append({"parameter": spec.name, "site": spec.sample_name, "family": spec.component_family, "field": spec.field,
                     "median": q[1], "p16": q[0], "p84": q[2], "mean": float(physical[:, index].mean()), "std": float(physical[:, index].std()),
                     "latent_median": float(np.median(latent[:, index])),
                     "best_fit": float(np.asarray(posterior.maximum_likelihood_fit)[index]) if posterior.maximum_likelihood_fit is not None else np.nan,
                     "r_hat": float(np.asarray(diagnostics.get("r_hat", np.full(latent.shape[1], np.nan)))[index]),
                     "n_eff": float(np.asarray(diagnostics.get("n_eff", np.full(latent.shape[1], np.nan)))[index])})
    return pd.DataFrame(rows)


def classification_table(state: ModelState) -> pd.DataFrame:
    rankings = (state.members.classification or {}).get("rankings", [])
    return pd.DataFrame([{"member_id": r["member"], "member_class": "exact" if r["exact"] else "far-field", **{k: v for k, v in r.items() if k not in {"member", "exact"}}}
                         for r in rankings])


def write_outputs(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator, posterior: PosteriorResults,
                  audit: dict[str, Any], paths, *, log=print) -> dict[str, Any]:
    """``paths`` is the final StagePaths."""
    tables, figures, prefix = paths.tables, paths.figures, config.paths.figure_prefix
    tables.mkdir(parents=True, exist_ok=True)
    log_prob = np.asarray(posterior.log_prob, dtype=float)
    best = np.asarray(posterior.samples[int(np.nanargmax(log_prob))] if np.isfinite(log_prob).any() else np.median(posterior.samples, axis=0), dtype=float)
    best_physical = np.asarray(evaluator.physical(jnp.asarray(best)))
    summary = {}

    posterior_summary_table(state, posterior).to_csv(tables / "posterior_summary.csv", index=False)
    classification_table(state).to_csv(tables / "member_classification.csv", index=False)
    write_chain_traces(posterior, state.parameter_specs, figures, tables, prefix, log=log)
    if audit:
        (tables / "audit.json").write_text(json.dumps(audit, indent=2, default=float))

    member_table = plots.member_scaling_table(state, posterior)
    member_table.to_csv(tables / "member_scaling_relation.csv", index=False)
    write_potfile_plots(posterior.physical, state.parameter_specs, figures, prefix, posterior=posterior, log=log)
    write_member_diagnostic_plots(config, state, evaluator, posterior, paths, log=log)
    if not member_table.empty:
        plots.plot_scaling_relation(member_table, np.asarray(posterior.physical), state.parameter_specs,
                                    plots.figure_path(figures, "scaling_relation.pdf", prefix))

    recovery = image_recovery_tables(state, evaluator, posterior, config.image_diagnostics, best, log=log)
    for name, frame in recovery.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    image_df = recovery["image_fit_quality"]
    summary["fit"] = fit_summary(state, image_df, recovery["image_count_recovery"])
    log(f"[outputs] recovered {summary['fit'].get('recovered_image_count')}/{summary['fit']['observed_image_count']} images at the best fit, "
        f"rms={summary['fit'].get('image_rms_arcsec')}")
    if not image_df.empty:
        plots.plot_image_residual_histogram(image_df, plots.figure_path(figures, "image_residual_histogram.pdf", prefix),
                                            draws=recovery.get("posterior_image_residual_draws"), log=log)
        plots.plot_image_recovery(image_df, recovery["image_recovery_extra_images"], plots.figure_path(figures, "image_recovery.pdf", prefix))

    corner_path = plots.plot_corner(np.asarray(posterior.physical), state.parameter_specs, plots.figure_path(figures, "corner_globals.pdf", prefix), best=best_physical)
    if corner_path is None:
        log("[outputs] corner skipped: fewer than two global parameters with posterior spread")

    z_crit = float(config.truth.caustic_source_redshift)
    sources = np.asarray(evaluator.source_positions(jnp.asarray(best)))
    plots.plot_critical_curves(state, evaluator, best, sources, z_crit if z_crit > state.z_lens else max(state.z_source_values),
                               plots.figure_path(figures, "critical_curves.pdf", prefix))

    truth = compute_truth_recovery(state, evaluator, np.asarray(posterior.samples), config.truth, paths.fits, tables,
                                   seed=config.runtime.seed, log=log)
    if truth is not None:
        from .exact_images import astropy_cosmology
        from .magnification_recovery_plot import write_magnification_recovery
        write_magnification_recovery(truth, image_df, state.reference, state.z_lens,
                                     astropy_cosmology(state.cosmo_config), config.truth,
                                     figures, tables, prefix, log=log)
        plots.plot_truth_recovery_residuals(truth, image_df, plots.figure_path(figures, "truth_recovery_residuals.pdf", prefix))
        if truth["profile"] is not None:
            plots.plot_m2d_aperture_ratio(truth["profile"], truth["center"], image_df, truth["mass_per_kappa_pixel_msun"],
                                          plots.figure_path(figures, "truth_recovery_m2d_aperture_ratio.pdf", prefix))
            ratio = truth["profile"]
            summary["truth"] = {"m2d_ratio_at_max_radius": float(ratio["m2d_ratio_median"].iloc[-1]), "n_radii": int(len(ratio))}
    if config.image_catalog.image_dir and not image_df.empty:
        from .cutouts import load_cluster_imaging
        from .mass_figure import render_cluster_mass_figure
        try:
            imaging = load_cluster_imaging(config.image_catalog, config.model, config.paths.run_name)
        except FileNotFoundError as error:
            # The imaging lives on an external volume; a run must not lose its outputs because that volume is absent.
            log(f"[outputs] cluster mass figure skipped: {error}")
        else:
            render_cluster_mass_figure(evaluator, best, image_df, recovery["image_recovery_extra_images"], imaging,
                                       z_source=z_crit, kappa_true_fits=config.truth.kappa_true_fits,
                                       figure_path=plots.figure_path(figures, f"image_catalog_family_cluster_mass{config.image_catalog.figure_suffix}.pdf", prefix),
                                       tables_dir=tables, fits_dir=paths.fits)
            log(f"[outputs] cluster mass figure rendered from {imaging[3]} imaging")
    (tables / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_run_results(paths.root, name=prefix)
    return summary
