"""Correlated image-plane displacements per family: the gp_image_plane position likelihood.

Within a family the image-plane displacement d of the N images has covariance
S = sigma_shared^2 (K kron I_2) + blockdiag((sigma_inst,i^2 + sigma_white^2) I_2) with the Gaussian kernel
K_ij = exp(-|theta_i - theta_j|^2 / 2 ell^2).  Mapped to the source plane through B = blockdiag(A_i) and floored, C = B S B^T + sigma_fl^2 I is
the covariance of the stacked residuals F_i = b_i - beta.  One Cholesky factor of C per family gives the quadratic form,
the log determinant and the conditional N(beta_bar, S_beta) of the source.  Families are padded to a common size; a
padded slot carries an identity block, a zero Jacobian and a zero residual, so it contributes exactly zero.
All coordinates are arcseconds.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

_LOG_2PI = float(np.log(2.0 * np.pi))


def family_slots(family_index, n_families: int) -> dict:
    """Padded (F, M) image indices in catalog order within each family, their mask and the real-image counts."""
    family_index = np.asarray(family_index, dtype=int)
    counts = np.bincount(family_index, minlength=int(n_families))
    size = int(counts.max()) if counts.size else 0
    slots = np.zeros((int(n_families), size), dtype=np.int32)
    mask = np.zeros((int(n_families), size), dtype=bool)
    for family in range(int(n_families)):
        rows = np.flatnonzero(family_index == family)
        slots[family, :rows.size], mask[family, :rows.size] = rows, True
    return dict(slots=slots, mask=mask, counts=counts, size=size)


def squared_separations(x, y, slots):
    """Squared image separations per family (F, M, M) in slot order."""
    xy = jnp.stack((jnp.asarray(x), jnp.asarray(y)), -1)[jnp.asarray(slots)]                 # (F, M, 2)
    return jnp.sum((xy[:, :, None, :] - xy[:, None, :, :]) ** 2, axis=-1)


def kernel_from_separations(d2, mask, ell):
    """Gaussian kernel exp(-d^2 / 2 ell^2) per family; rows and columns of padded slots are zero.  ``ell`` may be traced."""
    mask = jnp.asarray(mask)
    return jnp.exp(-jnp.asarray(d2) / (2.0 * jnp.square(ell))) * mask[:, :, None] * mask[:, None, :]


def displacement_kernel(x, y, slots, mask, ell: float):
    """Gaussian kernel per family (F, M, M); rows and columns of padded slots are zero."""
    return kernel_from_separations(squared_separations(x, y, slots), mask, ell)


def precompute(images, ell: float) -> dict:
    """Everything that depends on the data only: slots, mask, counts, squared separations, the kernel at the fixed (or
    starting) ``ell`` and the padded instrumental variances."""
    layout = family_slots(images.family_index_per_image, images.n_families)
    slots, mask = jnp.asarray(layout["slots"]), jnp.asarray(layout["mask"])
    sigma_inst2 = jnp.where(mask, jnp.square(jnp.asarray(images.sigma_inst_per_image))[slots], 0.0)
    d2 = squared_separations(images.x_obs, images.y_obs, slots)
    return dict(slots=slots, mask=mask, counts=jnp.asarray(layout["counts"], dtype=jnp.float64), size=layout["size"],
                squared_separations=d2, kernel=kernel_from_separations(d2, mask, float(ell)), sigma_inst2=sigma_inst2)


def family_covariance(jacobian, kernel, sigma_inst2, mask, sigma_shared2, sigma_white2, sigma_fl2):
    """C = B S B^T + sigma_fl^2 I per family as (F, 2M, 2M), with identity blocks at padded slots, and its Cholesky factor.

    ``jacobian`` is (F, M, 2, 2) in slot order.  Positive definiteness follows from sigma_inst^2 + sigma_white^2 > 0 or
    sigma_fl > 0 together with nonsingular Jacobians; no jitter is added, so a NaN factor points at an input error."""
    jacobian, kernel, mask = jnp.asarray(jacobian), jnp.asarray(kernel), jnp.asarray(mask)
    n_families, size = mask.shape
    eye2 = jnp.eye(2, dtype=jacobian.dtype)
    eye_m = jnp.eye(size, dtype=jacobian.dtype)
    a = jnp.where(mask[..., None, None], jacobian, 0.0)
    # S_ij = sigma_shared^2 K_ij I_2 + delta_ij (sigma_inst,i^2 + sigma_white^2) I_2, as (F, M, M, 2, 2).
    white = jnp.where(mask, jnp.asarray(sigma_inst2) + sigma_white2, 0.0)
    s = (sigma_shared2 * kernel + eye_m[None] * white[:, :, None])[..., None, None] * eye2
    c = jnp.einsum("fiab,fijbc,fjdc->fijad", a, s, a)
    diagonal = jnp.where(mask, sigma_fl2, 1.0)                                                 # floor on real slots, identity on pads
    c = c + (eye_m[None] * diagonal[:, :, None])[..., None, None] * eye2
    c = c.transpose(0, 1, 3, 2, 4).reshape(n_families, 2 * size, 2 * size)
    c = 0.5 * (c + jnp.swapaxes(c, -1, -2))
    return c, jnp.linalg.cholesky(c)


def family_gaussian_loglike(cholesky, residual, mask, counts):
    """-1/2 F^T C^-1 F - 1/2 ln det C - N_f ln 2 pi per family; ``residual`` is (F, M, 2) in slot order."""
    mask = jnp.asarray(mask)
    n_families, size = mask.shape
    r = jnp.where(mask[..., None], jnp.asarray(residual), 0.0).reshape(n_families, 2 * size)
    whitened = solve_triangular(cholesky, r[..., None], lower=True)[..., 0]
    quadratic = jnp.sum(whitened ** 2, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (quadratic + logdet) - jnp.asarray(counts) * _LOG_2PI, quadratic, logdet


def family_source_whitening(cholesky, backprojected, mask):
    """Conditional of beta at fixed lens: S_beta = (J^T C^-1 J)^-1 and beta_bar = S_beta J^T C^-1 b, from the same
    Cholesky factor; the 2 x 2 inverse is the only explicit one.  Same keys as ``smoothed_likelihood.source_whitening``."""
    mask = jnp.asarray(mask)
    n_families, size = mask.shape
    eye2 = jnp.eye(2, dtype=cholesky.dtype)
    j = (mask[:, :, None, None] * eye2).reshape(n_families, 2 * size, 2)
    lj = solve_triangular(cholesky, j, lower=True)                                             # (F, 2M, 2)
    precision = jnp.einsum("fki,fkj->fij", lj, lj)
    precision = 0.5 * (precision + jnp.swapaxes(precision, -1, -2))
    det = precision[:, 0, 0] * precision[:, 1, 1] - precision[:, 0, 1] * precision[:, 1, 0]
    adjugate = jnp.stack([jnp.stack([precision[:, 1, 1], -precision[:, 0, 1]], -1),
                          jnp.stack([-precision[:, 1, 0], precision[:, 0, 0]], -1)], -2)
    covariance = adjugate / det[:, None, None]
    b = jnp.where(mask[..., None], jnp.asarray(backprojected), 0.0).reshape(n_families, 2 * size)
    lb = solve_triangular(cholesky, b[..., None], lower=True)[..., 0]
    centroid = jnp.einsum("fij,fj->fi", covariance, jnp.einsum("fki,fk->fi", lj, lb))
    chol = jnp.linalg.cholesky(covariance)
    logdet_precision = -2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    valid = jnp.all(jnp.isfinite(chol), axis=(-2, -1)) & jnp.all(jnp.isfinite(centroid), axis=-1) & (det > 0)
    return dict(centroid=centroid, covariance=covariance, cholesky=chol, precision=precision, logdet_precision=logdet_precision, valid=valid)


# ----------------------------------------------------------------------------- pre-run check of the shallow fold pairs
def tangential_eigenpair(a):
    """Eigenvalue of the symmetric Jacobian with the smallest magnitude and its unit eigenvector."""
    values, vectors = np.linalg.eigh(0.5 * (a + a.T))
    k = int(np.argmin(np.abs(values)))
    return float(values[k]), vectors[:, k]


def shallow_fold_pairs(state, evaluator, theta, *, max_separation_arcsec: float = 3.0, sigma_shared_arcsec: float = 0.15,
                       sigma_white_arcsec: float = 0.02) -> list[dict]:
    """Own depth y_p = c Delta^2 / 8 and depth error of every mutual-nearest opposite-parity pair closer than
    ``max_separation_arcsec``, at this lens model.  c is the derivative of lambda_t along e_t at the pair midpoint; the depth
    error is sqrt((c Delta sigma_Delta / 4)^2 + sigma_fl^2 / 2) with sigma_Delta^2 = 2 sigma_shared^2 (1 - k(Delta)) + 2 sigma_white^2.
    Half the squared ratio is roughly the penalty in nats for losing the pair."""
    from .fold_diagnostics import candidate_pairs

    physical = evaluator.physical(jnp.asarray(theta, dtype=jnp.float64))
    a = np.asarray(evaluator.fields(physical, force_exact=True, derivative=False)["jacobian"])
    xy = np.column_stack([np.asarray(state.images.x_obs), np.asarray(state.images.y_obs)])
    labels = np.asarray(state.images.image_labels)
    family_index = np.asarray(state.images.family_index_per_image)
    redshift = {str(f.family_id): float(f.z_source) for f in state.family_data}
    ell, floor = float(state.gp_correlation_length_arcsec), float(state.gp_floor_arcsec)
    components = np.arange(evaluator.n_components, dtype=np.int32)
    rows = []
    for i, j in candidate_pairs(xy, family_index, labels, np.linalg.det(a), float(max_separation_arcsec)):
        family_id = str(state.images.family_ids[int(family_index[i])])
        packed = evaluator.packed_state_for_z(physical, redshift[family_id])
        midpoint = 0.5 * (xy[i] + xy[j])
        _, u, du = evaluator.component_rows(jnp.asarray(midpoint[None]), packed, components, derivative=True)
        a_mid, da_mid = np.eye(2) - np.asarray(jnp.sum(u, axis=0))[0], -np.asarray(jnp.sum(du, axis=0))[0]
        lambda_mid, e_t = tangential_eigenpair(a_mid)
        c = abs(float(np.einsum("a,b,abk,k", e_t, e_t, da_mid, e_t)))
        separation = float(np.linalg.norm(xy[j] - xy[i]))
        depth = c * separation ** 2 / 8.0
        k = np.exp(-separation ** 2 / (2.0 * ell ** 2))
        sigma_delta = float(np.sqrt(2.0 * sigma_shared_arcsec ** 2 * (1.0 - k) + 2.0 * sigma_white_arcsec ** 2))
        depth_error = float(np.sqrt((c * separation * sigma_delta / 4.0) ** 2 + floor ** 2 / 2.0))
        ratio = depth / depth_error if depth_error > 0 else np.inf
        rows.append(dict(labels=[str(labels[i]), str(labels[j])], family_id=family_id, separation_arcsec=separation,
                         lambda_t=[tangential_eigenpair(a[i])[0], tangential_eigenpair(a[j])[0]], lambda_t_midpoint=lambda_mid,
                         c_per_arcsec=c, depth_arcsec=depth, sigma_delta_arcsec=sigma_delta, depth_error_arcsec=depth_error,
                         ratio=float(ratio), ratio_below_3=bool(ratio < 3.0), depth_below_4p5_mas=bool(depth < 0.0045),
                         depth_below_2_mas=bool(depth < 0.002)))
    return rows


def shallow_fold_pair_report(rows: list[dict], floor_arcsec: float, sigma_shared_arcsec: float = 0.15, sigma_white_arcsec: float = 0.02) -> str:
    """One line per pair and the Section 5 recommendations."""
    lines = [f"[gp] shallow fold pairs (own depth at this model, sigma_shared {sigma_shared_arcsec:.3f}\", sigma_white {sigma_white_arcsec:.3f}\", "
             f"sigma_fl {1e3 * floor_arcsec:.2f} mas): {len(rows)}"]
    for row in rows:
        lines.append(f"[gp]   {row['labels'][0]}/{row['labels'][1]}: separation {row['separation_arcsec']:.3f}\", lambda_t "
                     f"{row['lambda_t'][0]:+.4f}/{row['lambda_t'][1]:+.4f}, c {row['c_per_arcsec']:.4f}/\", depth {1e3 * row['depth_arcsec']:.2f} mas, "
                     f"error {1e3 * row['depth_error_arcsec']:.2f} mas, ratio {row['ratio']:.1f}" + ("  [ratio < 3]" if row["ratio_below_3"] else ""))
    if any(row["depth_below_2_mas"] for row in rows):
        lines.append("[gp] a pair has depth below 2 mas: use sigma_fl = 0.5 mas and expect to need the second-order fold root")
    elif any(row["depth_below_4p5_mas"] for row in rows):
        lines.append("[gp] a pair has depth below 4.5 mas: set likelihood.gp_floor_arcsec = 0.0005 for this run")
    return "\n".join(lines)
