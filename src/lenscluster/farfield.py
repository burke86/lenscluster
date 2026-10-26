"""Circular far-field series for distant members: fixed geometry times current weights.

The deflection of a circular dPIE at separation R, in the active DPIE_NIE_E
convention, expands as (M/R)[1 - c1/(2R) + c2/(8R^3) + ...] with monopole
M = 2 sigma0 Ra Rs, c1 = Ra + Rs and c2 = c1 (Ra^2 + Rs^2).  Geometry arrays hold
rhat/R, rhat/R^2 and rhat/R^4 with their first two spatial derivatives; only the
parameter weights change during sampling.  All Jacobians are U = d alpha / d theta.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class FarFieldGeometry(NamedTuple):
    """Member, image, series-term, spatial-component arrays (arcsecond units)."""
    deflection: object            # (M, N, 3, 2)
    jacobian: object              # (M, N, 3, 2, 2)
    jacobian_derivative: object   # (M, N, 3, 2, 2, 2), last axis is the derivative direction
    separation: object            # (M, N) image-member separation
    valid_separation: object      # (M, N) separation > 0


class FarFieldFields(NamedTuple):
    alpha: object                     # (M, N, 2)
    jacobian: object                  # (M, N, 2, 2)
    total_jacobian_derivative: object  # (N, 2, 2, 2), summed over members


def member_farfield_geometry(image_xy, member_xy) -> FarFieldGeometry:
    """Precompute rhat/R, rhat/R^2, rhat/R^4 and their two spatial derivatives.

    Coincident member and image positions are flagged; their template values are NaN.
    """
    images, members = np.asarray(image_xy, dtype=float), np.asarray(member_xy, dtype=float)
    if images.ndim != 2 or images.shape[1] != 2 or members.ndim != 2 or members.shape[1] != 2:
        raise ValueError("Image and member positions must have shapes (N, 2) and (M, 2).")
    if not np.isfinite(images).all() or not np.isfinite(members).all():
        raise ValueError("Geometry positions must be finite.")
    delta = images[None, :, :] - members[:, None, :]
    radius = np.linalg.norm(delta, axis=-1)
    valid = radius > 0
    safe_radius = np.where(valid, radius, 1.0)
    direction = delta / safe_radius[..., None]
    eye = np.eye(2)
    outer = np.einsum("...i,...j->...ij", direction, direction)
    triple = np.einsum("...i,...j,...k->...ijk", direction, direction, direction)
    mixed = (np.einsum("ij,...k->...ijk", eye, direction) + np.einsum("ik,...j->...ijk", eye, direction)
             + np.einsum("jk,...i->...ijk", eye, direction))
    fields, jacobians, derivatives = [], [], []
    for n in (1, 2, 4):
        fields.append(direction / safe_radius[..., None] ** n)
        jacobians.append((eye - (n + 1) * outer) / safe_radius[..., None, None] ** (n + 1))
        derivatives.append((n + 1) * ((n + 3) * triple - mixed) / safe_radius[..., None, None, None] ** (n + 2))

    def masked_stack(values, spatial_rank):
        array = np.stack(values, axis=2)
        return jnp.asarray(np.where(valid[(...,) + (None,) * (spatial_rank + 1)], array, np.nan))

    return FarFieldGeometry(masked_stack(fields, 1), masked_stack(jacobians, 2), masked_stack(derivatives, 3),
                            jnp.asarray(radius), jnp.asarray(valid))


def slice_geometry(geometry: FarFieldGeometry, positions) -> FarFieldGeometry:
    positions = np.asarray(positions, dtype=int)
    return FarFieldGeometry(*(jnp.asarray(np.asarray(a)[positions]) for a in geometry))


def _pair_parameter(value, shape):
    value = jnp.asarray(value)
    if value.ndim == 1:
        value = value[:, None]
    return jnp.broadcast_to(value, shape)


def farfield_weights(geometry: FarFieldGeometry, sigma0, Ra, Rs):
    """Series weights (M, N, 3) from the current packed sigma0 (which carries the efficiency), Ra and Rs."""
    shape = geometry.valid_separation.shape
    convergence, core, cut = (_pair_parameter(value, shape) for value in (sigma0, Ra, Rs))
    mass = 2 * convergence * core * cut
    c1 = core + cut
    return mass[..., None] * jnp.stack((jnp.ones_like(c1), -0.5 * c1, 0.125 * c1 * (core ** 2 + cut ** 2)), axis=-1)


def evaluate_farfield(geometry: FarFieldGeometry, sigma0, Ra, Rs) -> FarFieldFields:
    """Member deflections and Jacobians, plus the member-summed Jacobian derivative."""
    weights = farfield_weights(geometry, sigma0, Ra, Rs)
    return FarFieldFields(jnp.einsum("mnk,mnki->mni", weights, geometry.deflection),
                          jnp.einsum("mnk,mnkij->mnij", weights, geometry.jacobian),
                          jnp.einsum("mnk,mnkijl->nijl", weights, geometry.jacobian_derivative))


def evaluate_farfield_per_member(geometry: FarFieldGeometry, sigma0, Ra, Rs):
    """Per-member deflection, Jacobian and Jacobian derivative (for classification and audits)."""
    weights = farfield_weights(geometry, sigma0, Ra, Rs)
    return (jnp.einsum("mnk,mnki->mni", weights, geometry.deflection),
            jnp.einsum("mnk,mnkij->mnij", weights, geometry.jacobian),
            jnp.einsum("mnk,mnkijl->mnijl", weights, geometry.jacobian_derivative))
