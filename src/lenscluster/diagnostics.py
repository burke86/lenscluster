"""Image-recovery tables from exact lens-equation solutions at the best fit and posterior draws."""
from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import ImageDiagnosticsConfig
from .evaluator import Evaluator
from .exact_images import ExactImageSolver
from .model import ModelState, PosteriorResults
from .utils import progress


def predictions_at(solver: ExactImageSolver, state: ModelState, evaluator: Evaluator, theta: np.ndarray, *,
                   adaptive: bool | None = None) -> dict[str, Any]:
    """Per-image model positions, residuals and magnifications for one latent vector."""
    theta = np.asarray(theta, dtype=float)
    physical = evaluator.physical(jnp.asarray(theta))
    sources = evaluator.source_positions(jnp.asarray(theta))
    images = state.images
    family_by_id = {f.family_id: f for f in state.family_data}
    families = [family_by_id[family_id] for family_id in images.family_ids]
    # Potential kernel: the draw's source scatters by the deflection error, so the lens gets the GP's mean deflection.
    correction = evaluator.gp_deflection_correction(theta) if getattr(evaluator, "gp_knee", False) else None
    extra_kwargs = {} if correction is None else dict(correction=correction)
    if adaptive is False:        # posterior draws: all families in one batched local solve
        batched = solver.predict_all_local(images, families, evaluator.packed_state(physical)[0], np.asarray(sources), **extra_kwargs)
    elif solver.config.exact_image_finder == "global-jax":   # the best fit: a global search that also reports extra images
        batched, adaptive = solver.predict_all_global(images, families, np.asarray(physical), np.asarray(sources), **extra_kwargs), False
    elif correction is not None:
        raise ValueError("under gp_kernel='potential' only the global-jax and batched local image finders apply the GP mean deflection; "
                         f"image_diagnostics.exact_image_finder is {solver.config.exact_image_finder!r}")
    rows, extra_rows, count_rows = [], [], []
    for family_index, family in enumerate(families):
        family_id = family.family_id
        if adaptive is False:
            details = batched[family_index]
        else:
            packed = evaluator.packed_state_for_z(physical, family.z_source)
            details = solver.predict_family(family, packed, float(sources[family_index, 0]), float(sources[family_index, 1]), adaptive=adaptive)
        recovered = np.asarray(details["recovered_image_mask"], dtype=bool)
        mx, my = details["matched_model_x_arcsec"], details["matched_model_y_arcsec"]
        mu = np.asarray(details["magnification"], dtype=float)
        mu_model = np.asarray(details["magnification_at_model"], dtype=float)
        count_rows.append({"family_id": family_id, "z_source": family.z_source, "observed_image_count": family.n_images,
                           "produced_image_count": details["produced_image_count"], "recovered_image_count": details["recovered_image_count"],
                           "missing_image_count": details["missing_image_count"], "extra_image_count": details["extra_image_count"],
                           "multiplicity_failed": bool(details["multiplicity_failed"]),
                           "multiplicity_failure_reason": str(details["multiplicity_failure_reason"]),
                           "exact_image_rms_arcsec": float(details["exact_image_rms"]),
                           "source_x_arcsec": float(sources[family_index, 0]), "source_y_arcsec": float(sources[family_index, 1])})
        for k, label in enumerate(family.image_labels):
            x_obs, y_obs = float(family.x_obs[k]), float(family.y_obs[k])
            residual = math.hypot(mx[k] - x_obs, my[k] - y_obs) if recovered[k] else np.nan
            rows.append({"family_id": family_id, "image_label": label, "x_obs_arcsec": x_obs, "y_obs_arcsec": y_obs,
                         "z_source": family.z_source, "sigma_inst_arcsec": float(images.sigma_inst_per_image[images.image_labels.index(label)]),
                         "radius_arcsec": math.hypot(x_obs, y_obs), "angle_deg": math.degrees(math.atan2(y_obs, x_obs)),
                         "image_recovery_status": "recovered" if recovered[k] else "not_recovered",
                         "x_model_arcsec": float(mx[k]), "y_model_arcsec": float(my[k]), "image_residual_arcsec": float(residual),
                         "magnification_model": float(mu[k]), "magnification_model_at_model_position": float(mu_model[k]),
                         "model_produced_image_count": details["produced_image_count"], "model_recovered_image_count": details["recovered_image_count"],
                         "model_missing_image_count": details["missing_image_count"], "model_extra_image_count": details["extra_image_count"],
                         "model_multiplicity_failed": bool(details["multiplicity_failed"]),
                         "model_multiplicity_failure_reason": str(details["multiplicity_failure_reason"])})
        for ex, ey in zip(details["extra_model_x_arcsec"], details["extra_model_y_arcsec"]):
            extra_rows.append({"family_id": family_id, "x_model_arcsec": float(ex), "y_model_arcsec": float(ey)})
    return {"image_rows": rows, "extra_rows": extra_rows, "count_rows": count_rows}


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(array, [16, 50, 84]))


def image_recovery_tables(state: ModelState, evaluator: Evaluator, posterior: PosteriorResults, config: ImageDiagnosticsConfig,
                          best_theta: np.ndarray, *, log=print) -> dict[str, pd.DataFrame]:
    """image_fit_quality, image_count_recovery, model_magnification, extra images and family tables."""
    solver = ExactImageSolver(evaluator, config)
    best = predictions_at(solver, state, evaluator, best_theta)
    n_draws = min(int(config.posterior_image_diagnostic_draws), len(posterior.samples))
    draws = posterior.samples[np.linspace(0, len(posterior.samples) - 1, n_draws).astype(int)] if n_draws else np.empty((0, state.n_parameters))
    by_label: dict[str, list[dict[str, Any]]] = {}
    rms_rows, draw_rows = [], []
    for draw_index, theta in enumerate(progress(draws, desc="exact images")):
        # Local solver only: the adaptive fallback is for the best fit; on a poor draw it costs minutes per family set.
        prediction = predictions_at(solver, state, evaluator, theta, adaptive=False)
        residuals = np.asarray([r["image_residual_arcsec"] for r in prediction["image_rows"]], dtype=float)
        rms_rows.append({"draw_index": draw_index, "recovered_image_count": int(np.isfinite(residuals).sum()),
                         "image_rms_arcsec": float(np.sqrt(np.nanmean(residuals ** 2))) if np.isfinite(residuals).any() else np.nan})
        for row in prediction["image_rows"]:
            by_label.setdefault(row["image_label"], []).append(row)
            draw_rows.append({"draw_index": draw_index, "family_id": row["family_id"], "image_label": row["image_label"],
                              "recovered": row["image_recovery_status"] == "recovered", "image_residual_arcsec": row["image_residual_arcsec"],
                              "magnification_model": row["magnification_model"],
                              "magnification_model_at_model_position": row["magnification_model_at_model_position"]})
    log(f"[diagnostics] exact images at best fit + {n_draws} draws in {solver.timing['exact_solver']:.1f}s of solver time")
    image_rows = []
    for row in best["image_rows"]:
        draws_for = by_label.get(row["image_label"], [])
        x16, x50, x84 = _quantiles([d["x_model_arcsec"] for d in draws_for])
        y16, y50, y84 = _quantiles([d["y_model_arcsec"] for d in draws_for])
        r16, r50, r84 = _quantiles([d["image_residual_arcsec"] for d in draws_for])
        m16, m50, m84 = _quantiles([d["magnification_model"] for d in draws_for])
        p16, p50, p84 = _quantiles([d["magnification_model_at_model_position"] for d in draws_for])
        sigma = float(row["sigma_inst_arcsec"])
        image_rows.append({**row, "x_model_q16": x16, "x_model_q50": x50, "x_model_q84": x84,
                           "y_model_q16": y16, "y_model_q50": y50, "y_model_q84": y84,
                           "image_residual_q16": r16, "image_residual_q50": r50, "image_residual_q84": r84,
                           "magnification_model_q16": m16, "magnification_model_q50": m50, "magnification_model_q84": m84,
                           "magnification_model_at_model_position_q16": p16, "magnification_model_at_model_position_q50": p50,
                           "magnification_model_at_model_position_q84": p84,
                           "residual_norm": float(row["image_residual_arcsec"]) / sigma if sigma > 0 else np.nan,
                           "posterior_valid_draws": int(sum(np.isfinite(d["x_model_arcsec"]) for d in draws_for)),
                           "posterior_failed_draws": int(sum(not np.isfinite(d["x_model_arcsec"]) for d in draws_for))})
    image_df = pd.DataFrame(image_rows)
    magnification_columns = ["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source", "image_recovery_status", "x_model_arcsec", "y_model_arcsec",
                             "magnification_model", "magnification_model_q16", "magnification_model_q50", "magnification_model_q84",
                             "magnification_model_at_model_position", "magnification_model_at_model_position_q16",
                             "magnification_model_at_model_position_q50", "magnification_model_at_model_position_q84"]
    magnification_df = image_df[[c for c in magnification_columns if c in image_df.columns]].copy() if not image_df.empty else pd.DataFrame()
    count_df = pd.DataFrame(best["count_rows"])
    return {"image_fit_quality": image_df, "image_count_recovery": count_df, "model_magnification": magnification_df,
            "image_recovery_extra_images": pd.DataFrame(best["extra_rows"]), "posterior_image_rms": pd.DataFrame(rms_rows),
            "posterior_image_residual_draws": pd.DataFrame(draw_rows)}


def fit_summary(state: ModelState, image_df: pd.DataFrame, count_df: pd.DataFrame) -> dict[str, Any]:
    """Headline chi-square and recovery numbers from the best-fit image table.  The degrees of freedom follow the
    lenstool convention, 2 N_images - 2 N_families - N_global, where the global parameters are the halo, relation and
    error-model parameters: the member deviations are prior-dominated and add nothing to the effective count."""
    n_images = int(sum(f.n_images for f in state.family_data))
    n_global = int(sum(spec.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"} for spec in state.parameter_specs))
    k = n_global + 2 * len(state.family_data)
    out: dict[str, Any] = {"observed_image_count": n_images, "n_effective_parameters": k, "n_global_parameters": n_global}
    if image_df.empty:
        return out
    recovered = image_df["image_recovery_status"].astype(str) == "recovered"
    residual = pd.to_numeric(image_df["image_residual_arcsec"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(image_df["sigma_inst_arcsec"], errors="coerce").to_numpy(dtype=float)
    valid = recovered.to_numpy() & np.isfinite(residual) & (sigma > 0)
    chi2 = float(np.sum((residual[valid] / sigma[valid]) ** 2))
    n_data = int(2 * valid.sum())
    out.update(recovered_image_count=int(valid.sum()), missing_image_count=int(n_images - valid.sum()),
               extra_image_count=int(count_df["extra_image_count"].sum()) if not count_df.empty else 0,
               image_rms_arcsec=float(np.sqrt(np.mean(residual[valid] ** 2))) if valid.any() else None,
               image_median_residual_arcsec=float(np.median(residual[valid])) if valid.any() else None,
               image_max_residual_arcsec=float(np.max(residual[valid])) if valid.any() else None,
               chi_square=chi2, n_data=n_data, dof=int(n_data - k), reduced_chi_square=chi2 / (n_data - k) if n_data > k else None,
               families_fully_recovered=int((~count_df["multiplicity_failed"].astype(bool)).sum()) if not count_df.empty else 0)
    return out
