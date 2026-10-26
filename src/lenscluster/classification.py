"""Member classification: exact dPIE kernels near images, the circular far-field series elsewhere.

score_g = max over images i of max( (k r_cut,g / R_gi)^6 , ||C_i^-1/2 e_gi|| / tolerance )

The first term is the leading omitted term of the series normalized to the guard
distance k r_cut; members with any image inside k cut radii score above one and are
mandatory.  The second term is the whitened ellipticity error of the circular series,
with C_i = sigma_src^2 I + sigma_obs^2 A_i A_i^T, bounded and evaluated at one model.
After the per-member rule, the largest contributors are promoted until the summed
whitened ellipticity error of the remaining far-field members is below the tolerance
at every image.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .evaluator import Evaluator
from .farfield import evaluate_farfield_per_member
from .model import ModelState


def prior_quantile_cut_kpc(potfile: dict[str, Any], quantile: float) -> float:
    """The cutkpc prior's upper quantile for one population from its Lenstool-style limit entry."""
    from scipy.stats import norm, truncnorm
    entry = list(potfile["cutkpc"])
    kind = int(entry[0])
    if kind == 0:
        return float(potfile["cutkpc_nominal"])
    if kind == 1:                                   # uniform [lo, hi]
        lo, hi = float(entry[1]), float(entry[2])
        return lo + float(quantile) * (hi - lo)
    if kind == 3:                                   # normal
        return float(norm.ppf(quantile, float(entry[1]), float(entry[2])))
    mean, std, lo, hi = (float(v) for v in entry[1:5])   # truncated normal
    return float(truncnorm.ppf(quantile, (lo - mean) / std, (hi - mean) / std, loc=mean, scale=std))


def prior_quantile_beta(scaling: Any, quantile: float) -> float:
    """A quantile of the beta_radius prior (truncated normal, possibly unbounded above)."""
    from scipy.stats import truncnorm
    mean, std = float(scaling.beta_radius_prior_mean), float(scaling.beta_radius_prior_std)
    lo, hi = float(scaling.beta_radius_prior_lower), float(scaling.beta_radius_prior_upper)
    return float(truncnorm.ppf(quantile, (lo - mean) / std, (hi - mean) / std if np.isfinite(hi) else np.inf, loc=mean, scale=std))


def conservative_cut_radii_arcsec(state: ModelState, reference_cut_kpc: float, betas: list[float]) -> np.ndarray:
    """Per member, the largest cut radius over the given radius exponents: faint members (L < 1) are largest at the
    smallest beta, bright members at the largest."""
    return np.max(np.stack([member_cut_radii_arcsec(state, reference_cut_kpc, b) for b in betas]), axis=0)


def member_max_deflection_arcsec(state: ModelState, evaluator: Evaluator, theta: np.ndarray) -> np.ndarray:
    """Per member, the largest deflection it produces at any image under the model at ``theta`` (exact kernel)."""
    members = np.asarray(state.members.component_indices, dtype=np.int32)
    if members.size == 0:
        return np.zeros(0)
    packed = evaluator.packed_state(evaluator.physical(jnp.asarray(theta, dtype=jnp.float64)))[0]
    alpha, _ = evaluator.component_rows(evaluator.xy, packed, members)
    return np.asarray(jnp.max(jnp.linalg.norm(alpha, axis=-1), axis=1))


def member_cut_radii_arcsec(state: ModelState, reference_cut_kpc: float, beta_radius: float) -> np.ndarray:
    """Cut radius of every population member for a reference cut radius and radius exponent."""
    spec = state.packed_lens_spec
    luminosity = np.asarray(spec.luminosity_ratio)[state.members.component_indices]
    return float(reference_cut_kpc) * luminosity ** float(beta_radius) / float(state.kpc_per_arcsec)


def scaling_values(state: ModelState, physical: np.ndarray) -> dict[str, float]:
    """Reference cut radius and radius exponent per population, read from a physical vector."""
    values: dict[str, float] = {}
    for spec in state.parameter_specs:
        if spec.component_family == "scaling" and spec.field in {"cutkpc", "beta_radius"}:
            values[spec.field] = float(physical[state.parameter_specs.index(spec)])
    return values


FAR_FIELD_THRESHOLD_GRID = (2.0, 3.0, 4.0, 5.0, 10.0)


def member_field_terms(state: ModelState, evaluator: Evaluator, physical: np.ndarray) -> dict[str, np.ndarray]:
    """Per member and image at this model: the exact (elliptical) deflection, the circular far-field series, their
    difference whitened by the image covariance C = sigma_src^2 I + sigma_obs^2 A A^T, the precision and the separation."""
    members = state.members
    physical_jax = jnp.asarray(physical, dtype=jnp.float64)
    packed, _ = evaluator.packed_state(physical_jax)
    fields = evaluator.fields(physical_jax, packed, force_exact=True)
    exact_alpha = np.asarray(fields["member_alpha"])                     # (M, N, 2) elliptical dPIE
    components = jnp.asarray(members.component_indices)
    circular_alpha = np.asarray(evaluate_farfield_per_member(
        evaluator.geometry, jnp.take(packed.sigma0, components, axis=0), jnp.take(packed.Ra, components),
        jnp.take(packed.Rs, components))[0])                             # (M, N, 2) circular series
    a = np.asarray(fields["jacobian"])
    sigma_obs = np.asarray(state.images.sigma_per_image)
    sigma_src = float(physical[members.sigma_src_index])
    covariance = sigma_src ** 2 * np.eye(2)[None] + sigma_obs[:, None, None] ** 2 * a @ np.swapaxes(a, 1, 2)
    precision = np.linalg.inv(covariance)
    error = exact_alpha - circular_alpha
    error = np.where(np.isfinite(error), error, np.inf)
    whitened = np.sqrt(np.einsum("mni,nij,mnj->mn", error, precision, error))
    return {"exact_alpha": exact_alpha, "circular_alpha": circular_alpha, "error": error, "whitened": whitened,
            "precision": precision, "separation": np.asarray(evaluator.geometry.separation)}


def far_field_threshold_rows(state: ModelState, evaluator: Evaluator, physical: np.ndarray, *, cut_radius_arcsec: np.ndarray,
                             tolerance: float, selected_guard: float, guards: tuple[float, ...] = FAR_FIELD_THRESHOLD_GRID,
                             top_exact: int | None = None) -> list[dict[str, Any]]:
    """What each far-field threshold R/r_cut would do at this model: exact and far-field counts and the deflection error
    the far-field members leave at the images (summed exact minus circular), in milliarcseconds and whitened units."""
    if len(state.members.member_ids) == 0:
        return []
    terms = member_field_terms(state, evaluator, physical)
    rows = []
    for guard in sorted(set(float(g) for g in guards) | {float(selected_guard)}):
        exact, _ = classify_members(state, evaluator, physical, cut_radius_arcsec=cut_radius_arcsec, guard_cut_radii=guard,
                                    tolerance=tolerance, top_exact=top_exact, terms=terms)
        far = np.ones(len(state.members.member_ids), dtype=bool)
        far[exact] = False
        per_image = np.where(np.isfinite(terms["error"][far]), terms["error"][far], 0.0).sum(axis=0)   # (N, 2)
        magnitude_mas = 1000.0 * np.hypot(per_image[:, 0], per_image[:, 1])
        whitened = np.sqrt(np.einsum("ni,nij,nj->n", per_image, terms["precision"], per_image))
        worst = int(np.argmax(whitened)) if whitened.size else -1
        rows.append({"guard_cut_radii": float(guard), "exact": int(len(exact)), "far_field": int(far.sum()),
                     "max_error_mas": float(magnitude_mas.max()) if magnitude_mas.size else 0.0,
                     "median_error_mas": float(np.median(magnitude_mas)) if magnitude_mas.size else 0.0,
                     "max_whitened_error": float(whitened.max()) if whitened.size else 0.0,
                     "images_over_tolerance": int((whitened > float(tolerance)).sum()),
                     "worst_image": str(state.images.image_labels[worst]) if worst >= 0 else "",
                     "selected": bool(abs(float(guard) - float(selected_guard)) < 1e-9)})
    return rows


def classify_members(state: ModelState, evaluator: Evaluator, physical: np.ndarray, *,
                     cut_radius_arcsec: np.ndarray, guard_cut_radii: float, tolerance: float,
                     top_exact: int | None = None, terms: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact member positions and the ranking diagnostics.

    ``physical`` supplies the Jacobians, source scatter and member deflections used by the
    ellipticity term; ``cut_radius_arcsec`` supplies the (conservative) cut radius per member.
    ``terms`` are the precomputed member field terms at this model (see member_field_terms).
    """
    members = state.members
    n_members = len(members.member_ids)
    if n_members == 0:
        return np.asarray([], dtype=np.int32), {"member_count": 0, "mandatory_count": 0, "exact_count": 0, "rankings": []}
    terms = member_field_terms(state, evaluator, physical) if terms is None else terms
    whitened, separation = terms["whitened"], terms["separation"]      # (M, N)
    ratio = (float(guard_cut_radii) * np.asarray(cut_radius_arcsec)[:, None] / np.where(separation > 0, separation, np.inf)) ** 6
    truncation_score = ratio.max(axis=1)
    ellipticity_score = (whitened / float(tolerance)).max(axis=1)
    score = np.maximum(truncation_score, ellipticity_score)
    score = np.where(np.isfinite(score), score, np.inf)
    exact = score >= 1.0
    promotions: list[dict[str, Any]] = []
    # Collective budget: summed whitened ellipticity error of far-field members per image.
    for _ in range(n_members):
        far_sum = np.where(~exact[:, None], np.nan_to_num(whitened, nan=np.inf, posinf=np.inf), 0.0).sum(axis=0)
        if far_sum.size == 0 or far_sum.max() < float(tolerance):
            break
        image = int(np.argmax(far_sum))
        candidates = np.where(~exact, whitened[:, image], -np.inf)
        member = int(np.argmax(candidates))
        exact[member] = True
        promotions.append({"member": members.member_ids[member], "image": state.images.image_labels[image],
                           "summed_whitened_error": float(far_sum[image])})
    mandatory = int(exact.sum())
    if top_exact is not None:
        if int(top_exact) < mandatory:
            raise ValueError(f"top_exact={top_exact} is below the {mandatory} members that must be exact.")
        order = np.argsort(-score, kind="stable")
        exact[order[:int(top_exact)]] = True
    order = sorted(range(n_members), key=lambda i: (-score[i], members.member_ids[i]))
    diagnostics = {
        "member_count": n_members, "mandatory_count": mandatory, "exact_count": int(exact.sum()),
        "guard_cut_radii": float(guard_cut_radii), "tolerance": float(tolerance), "top_exact": top_exact,
        "promotions": promotions,
        "rankings": [{"member": members.member_ids[i], "score": None if not np.isfinite(score[i]) else float(score[i]),
                      "truncation": None if not np.isfinite(truncation_score[i]) else float(truncation_score[i]),
                      "ellipticity": None if not np.isfinite(ellipticity_score[i]) else float(ellipticity_score[i]),
                      "cut_radius_arcsec": float(cut_radius_arcsec[i]),
                      "min_separation_arcsec": float(separation[i].min()), "exact": bool(exact[i])} for i in order],
    }
    return np.flatnonzero(exact).astype(np.int32), diagnostics
