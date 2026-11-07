"""Fisher pre-check of a mock's image configuration: how well the solver's GP likelihood can constrain its error-model
hyperparameters (sigma_alpha and knee of the mass field, sigma_shared and ell of the image-plane GP, sigma_white) at
the truth, before any sampling.

The covariance is the solver's own (Evaluator.gp_kernel_blocks and gp_knee.block_covariance, knee kernel, the configured
blocks and projection), built at the true lens Jacobians of the images and at the injected hyperparameters.  The source
positions are free, so the Fisher matrix is the restricted (REML) one, with each block's source directions projected out:

    I_ab = 1/2 sum_blocks tr(P dC/da P dC/db),   P = C^-1 - C^-1 J (J^T C^-1 J)^-1 J^T C^-1,

where J is the block's source indicator.  Hyperparameters are in ln units (sigma_alpha, knee, sigma_shared, ell,
sigma_white; the slope gamma linear).  The lens is held at the truth, so the result is an upper bound on what the data
can say about the hyperparameters: in a real fit the lens model absorbs part of the field.  The forecast adds the
solver's prior widths as Gaussian precision in the same units and ignores the truncations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import gp_knee
from ..config import LensClusterSolverConfig
from ..evaluator import Evaluator
from ..state import build_state
from .images import jacobian, lens_state
from .truth import truth_from_json

# Solver scale name -> (truth key, LikelihoodConfig prior field, log units).
HYPERPARAMETERS = {
    "gp_sigma_alpha": ("field_sigma_alpha", "gp_sigma_alpha_prior", True),
    "gp_knee": ("field_knee", "gp_knee_prior", True),
    "gp_knee_gamma": ("field_gamma", "gp_knee_gamma_prior", False),
    "sigma_shared": ("image_plane_sigma", "gp_knee_sigma_shared_prior", True),
    "gp_ell": ("image_plane_ell", "gp_knee_ell_prior", True),
    "sigma_white": ("white", "gp_knee_sigma_white_prior", True),
}
SEPARATION_BINS_ARCSEC = (0.0, 0.5, 1.0, 2.0, 5.0, 15.0, 40.0, math.inf)


def _truth_hyperparameters(truth: dict[str, Any]) -> dict[str, float]:
    """The injected hyperparameters in the solver's units; an error component that was not injected is left out."""
    values = {"white": float(truth["config"]["sigma_inst_arcsec"])}
    field, image_plane = truth.get("field"), truth.get("image_plane_error")
    if field:
        values.update(field_sigma_alpha=float(field["sigma_alpha_arcsec"]), field_knee=float(field["knee_arcsec"]),
                      field_gamma=float(field["gamma"]))
    if image_plane:
        values.update(image_plane_sigma=float(image_plane["sigma_arcsec"]), image_plane_ell=float(image_plane["ell_arcsec"]))
    return values


def _prior_width(likelihood, name: str, prior_field: str, log_units: bool) -> float:
    prior = getattr(likelihood, prior_field, None)
    if prior is None:
        return math.nan
    if name == "gp_knee_gamma":                               # uniform (lo, hi): the standard deviation of the uniform
        lo, hi = prior
        return (float(hi) - float(lo)) / math.sqrt(12.0)
    return float(prior.log_sigma) if log_units else math.nan


def separation_counts(images: pd.DataFrame) -> dict[str, int]:
    """Image pairs per separation bin of the observed positions, all pairs and pairs within one system."""
    x, y = images["x_obs_arcsec"].to_numpy(float), images["y_obs_arcsec"].to_numpy(float)
    system = np.asarray([str(s) for s in (images["system_id"] if "system_id" in images else images["family_id"])])
    iu = np.triu_indices(x.size, 1)
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])[iu]
    same = (system[:, None] == system[None])[iu]
    counts = {}
    for lo, hi in zip(SEPARATION_BINS_ARCSEC[:-1], SEPARATION_BINS_ARCSEC[1:]):
        inside = (d >= lo) & (d < hi)
        label = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">{lo:g}"
        counts[f"all {label}"] = int(np.sum(inside))
        counts[f"system {label}"] = int(np.sum(inside & same))
    return counts


def hyperparameter_fisher(solver_config: LensClusterSolverConfig, truth: dict[str, Any]) -> dict[str, Any]:
    """The restricted Fisher matrix of the GP hyperparameters for one mock realization (module docstring)."""
    likelihood = solver_config.likelihood
    if likelihood.likelihood_mode != "gp_image_plane" or likelihood.gp_kernel != "knee":
        raise ValueError("the Fisher pre-check needs the knee GP likelihood (run_mock.xsh --likelihood grf_gp)")
    state = build_state(solver_config)
    evaluator = Evaluator(state, likelihood.likelihood_mode)
    layout = evaluator.gp

    # True lens Jacobians at the true image positions, in the solver's image order.
    images = pd.DataFrame(truth["images"]).set_index("image_label")
    lens_truth = truth_from_json(truth)
    labels = list(state.images.image_labels)
    rows = images.loc[labels]
    a = np.zeros((len(labels), 2, 2))
    for z, part in rows.groupby("z_source"):
        index = [labels.index(label) for label in part.index]
        a[index] = jacobian(lens_state(lens_truth, float(z)), part["x_true_arcsec"].to_numpy(float), part["y_true_arcsec"].to_numpy(float))[1]

    # The hyperparameters the solver samples and that were injected.
    injected = _truth_hyperparameters(truth)
    sampled = {"gp_sigma_alpha": evaluator.gp_sigma_alpha_index, "gp_knee": evaluator.gp_knee_index,
               "gp_knee_gamma": evaluator.gp_knee_gamma_index, "sigma_shared": evaluator.sigma_shared_index,
               "gp_ell": evaluator.gp_ell_index, "sigma_white": evaluator.sigma_white_index}
    names = [n for n, (key, _, _) in HYPERPARAMETERS.items() if sampled[n] >= 0 and key in injected]
    if not names:
        raise ValueError("no injected error component matches a sampled hyperparameter")
    base = evaluator.error_scales(evaluator.physical(jnp.asarray(state.initial_theta, dtype=jnp.float64)))
    base = dict(base)
    if "field_sigma_alpha" not in injected:
        base["gp_sigma_alpha"] = jnp.asarray(0.0)                    # no field injected: the truth has none
    if "image_plane_sigma" not in injected:
        base["sigma_shared"] = jnp.asarray(0.0)
    log_units = [HYPERPARAMETERS[n][2] for n in names]
    v0 = jnp.asarray([math.log(injected[HYPERPARAMETERS[n][0]]) if log_units[i] else injected[HYPERPARAMETERS[n][0]]
                      for i, n in enumerate(names)], dtype=jnp.float64)
    jac = jnp.asarray(a)[layout["slots"]]
    floor2 = jnp.square(jnp.asarray(evaluator.gp_floor, dtype=jnp.float64))

    def covariance(v):
        scales = dict(base)
        for i, n in enumerate(names):
            scales[n] = jnp.exp(v[i]) if log_units[i] else v[i]
        blocks = evaluator.gp_kernel_blocks(None, scales, layout)
        return gp_knee.block_covariance(jac, blocks["k_alpha"], blocks["k_img"], blocks["white"], layout["mask"], floor2)[0]

    c = np.asarray(covariance(v0))
    dc = np.asarray(jax.jacfwd(covariance)(v0))                         # (S, 2M, 2M, P)
    mask = np.repeat(np.asarray(layout["mask"]), 2, axis=-1)
    knots = np.repeat(np.asarray(layout["knot_mask"]), 2, axis=-1)
    indicator = np.asarray(layout["indicator"])
    fisher = np.zeros((len(names), len(names)))
    dof = 0
    for b in range(c.shape[0]):
        rows_b, cols_b = mask[b], knots[b]
        cb = c[b][np.ix_(rows_b, rows_b)]
        jb = indicator[b][np.ix_(rows_b, cols_b)]
        inv = np.linalg.inv(cb)
        cj = inv @ jb
        p = inv - cj @ np.linalg.solve(jb.T @ cj, cj.T)
        pd_ = [p @ dc[b][np.ix_(rows_b, rows_b)][..., i] for i in range(len(names))]
        for i in range(len(names)):
            for j in range(i, len(names)):
                fisher[i, j] = fisher[j, i] = fisher[i, j] + 0.5 * float(np.sum(pd_[i] * pd_[j].T))
        dof += int(rows_b.sum() - cols_b.sum())

    prior = np.asarray([_prior_width(likelihood, n, HYPERPARAMETERS[n][1], log_units[i]) for i, n in enumerate(names)])
    precision = np.diag(np.where(np.isfinite(prior), 1.0 / np.square(prior), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        data_only = np.sqrt(np.diag(np.linalg.pinv(fisher)))
        forecast = np.sqrt(np.diag(np.linalg.inv(fisher + precision)))
        contraction = 1.0 - np.square(forecast) / np.square(prior)
    table = pd.DataFrame({"parameter": [f"images.{n}" for n in names],
                          "units": ["ln" if lu else "linear" for lu in log_units],
                          "truth": [injected[HYPERPARAMETERS[n][0]] for n in names],
                          "prior_sigma": prior, "fisher_sigma": data_only, "forecast_sigma": forecast,
                          "prior_contraction": contraction})
    correlation = np.linalg.inv(fisher + precision)
    scale = np.sqrt(np.diag(correlation))
    return {"table": table, "fisher": fisher, "names": names, "forecast_correlation": correlation / np.outer(scale, scale),
            "n_images": int(state.images.n_images), "n_families": int(state.images.n_families),
            "n_systems": len(set(state.images.system_ids)) if state.images.system_ids else int(state.images.n_families),
            "n_blocks": int(c.shape[0]), "dof": dof, "separations": separation_counts(pd.DataFrame(truth["images"]))}


def write_fisher(path: str | Path, result: dict[str, Any]) -> Path:
    """The pre-check as JSON (table rows, Fisher matrix, forecast correlation, counts)."""
    path = Path(path)
    payload = {k: v for k, v in result.items() if k not in {"table", "fisher", "forecast_correlation"}}
    payload.update(table=result["table"].replace({np.nan: None}).to_dict("records"), fisher=np.asarray(result["fisher"]).tolist(),
                   forecast_correlation=np.asarray(result["forecast_correlation"]).tolist())
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    return path
