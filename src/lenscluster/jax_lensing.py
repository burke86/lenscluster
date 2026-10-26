from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E as DPIENIE
from jaxtronomy.LensModel.Profiles.nie_e import _atan_sqrt_ratio, _geometry as _nie_geometry

ORIGINAL_DPIE_PROFILE_NAME = "DPIE_NIE_E"
SHEAR_PROFILE_NAME = "SHEAR"


@dataclass(frozen=True)
class StaticLensState:
    profile_type: jnp.ndarray
    sigma0: jnp.ndarray
    Ra: jnp.ndarray
    Rs: jnp.ndarray
    e1: jnp.ndarray
    e2: jnp.ndarray
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], None]:
        return (
            (
                self.profile_type,
                self.sigma0,
                self.Ra,
                self.Rs,
                self.e1,
                self.e2,
                self.center_x,
                self.center_y,
                self.gamma1,
                self.gamma2,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data: None, children: tuple[jnp.ndarray, ...]) -> "StaticLensState":
        return cls(*children)


jax.tree_util.register_pytree_node_class(StaticLensState)


def static_lens_state_from_kwargs(
    lens_model_list: list[str] | tuple[str, ...],
    kwargs_lens: list[dict[str, float]],
) -> StaticLensState:
    if len(lens_model_list) != len(kwargs_lens):
        raise ValueError("lens_model_list and kwargs_lens must have the same length.")
    profile_type: list[int] = []
    sigma0: list[float] = []
    ra: list[float] = []
    rs: list[float] = []
    e1: list[float] = []
    e2: list[float] = []
    center_x: list[float] = []
    center_y: list[float] = []
    gamma1: list[float] = []
    gamma2: list[float] = []
    for name, kwargs in zip(lens_model_list, kwargs_lens):
        if str(name) in {ORIGINAL_DPIE_PROFILE_NAME, "DPIE_NIE"}:
            profile_type.append(1)
            sigma0.append(float(kwargs.get("sigma0", 0.0)))
            ra.append(float(kwargs.get("Ra", kwargs.get("ra", 0.0))))
            rs.append(float(kwargs.get("Rs", kwargs.get("rs", 0.0))))
            e1.append(float(kwargs.get("e1", 0.0)))
            e2.append(float(kwargs.get("e2", 0.0)))
            center_x.append(float(kwargs.get("center_x", 0.0)))
            center_y.append(float(kwargs.get("center_y", 0.0)))
            gamma1.append(0.0)
            gamma2.append(0.0)
        elif str(name) == SHEAR_PROFILE_NAME:
            profile_type.append(14)
            sigma0.append(0.0)
            ra.append(0.0)
            rs.append(0.0)
            e1.append(0.0)
            e2.append(0.0)
            center_x.append(0.0)
            center_y.append(0.0)
            gamma1.append(float(kwargs.get("gamma1", 0.0)))
            gamma2.append(float(kwargs.get("gamma2", 0.0)))
        else:
            raise ValueError(f"Unsupported mock lens profile {name!r}; only DPIE_NIE_E and SHEAR are supported.")
    return StaticLensState(
        profile_type=jnp.asarray(profile_type, dtype=jnp.int32),
        sigma0=jnp.asarray(sigma0, dtype=jnp.float64),
        Ra=jnp.asarray(ra, dtype=jnp.float64),
        Rs=jnp.asarray(rs, dtype=jnp.float64),
        e1=jnp.asarray(e1, dtype=jnp.float64),
        e2=jnp.asarray(e2, dtype=jnp.float64),
        center_x=jnp.asarray(center_x, dtype=jnp.float64),
        center_y=jnp.asarray(center_y, dtype=jnp.float64),
        gamma1=jnp.asarray(gamma1, dtype=jnp.float64),
        gamma2=jnp.asarray(gamma2, dtype=jnp.float64),
    )


def _dpie_component_derivatives_and_hessian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma0_i: jnp.ndarray,
    ra_i: jnp.ndarray,
    rs_i: jnp.ndarray,
    e1_i: jnp.ndarray,
    e2_i: jnp.ndarray,
    center_x_i: jnp.ndarray,
    center_y_i: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha_x, alpha_y = DPIENIE.derivatives(x, y, sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i)
    h_xx, h_xy, h_yx, h_yy = DPIENIE.hessian(x, y, sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i)
    return alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy


def _grouped_dpie_derivatives_and_hessian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma0: jnp.ndarray,
    ra: jnp.ndarray,
    rs: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    center_x: jnp.ndarray,
    center_y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy = jax.vmap(
        lambda sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i: _dpie_component_derivatives_and_hessian(
            x,
            y,
            sigma0_i,
            ra_i,
            rs_i,
            e1_i,
            e2_i,
            center_x_i,
            center_y_i,
        )
    )(
        sigma0,
        ra,
        rs,
        e1,
        e2,
        center_x,
        center_y,
    )
    return (
        jnp.sum(alpha_x, axis=0),
        jnp.sum(alpha_y, axis=0),
        jnp.sum(h_xx, axis=0),
        jnp.sum(h_xy, axis=0),
        jnp.sum(h_yx, axis=0),
        jnp.sum(h_yy, axis=0),
    )


def grouped_dpie_rows_derivatives_and_hessian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma0: jnp.ndarray,
    ra: jnp.ndarray,
    rs: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    center_x: jnp.ndarray,
    center_y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def one_component(
        sigma0_i: jnp.ndarray,
        ra_i: jnp.ndarray,
        rs_i: jnp.ndarray,
        e1_i: jnp.ndarray,
        e2_i: jnp.ndarray,
        center_x_i: jnp.ndarray,
        center_y_i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy = _dpie_component_derivatives_and_hessian(
            x,
            y,
            sigma0_i,
            ra_i,
            rs_i,
            e1_i,
            e2_i,
            center_x_i,
            center_y_i,
        )
        return alpha_x, alpha_y, -h_xx, -h_xy, -h_yx, -h_yy

    return jax.vmap(one_component)(sigma0, ra, rs, e1, e2, center_x, center_y)


def grouped_dpie_pair_derivatives_and_hessian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma0: jnp.ndarray,
    ra: jnp.ndarray,
    rs: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    center_x: jnp.ndarray,
    center_y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def one_component(
        x_i: jnp.ndarray,
        y_i: jnp.ndarray,
        sigma0_i: jnp.ndarray,
        ra_i: jnp.ndarray,
        rs_i: jnp.ndarray,
        e1_i: jnp.ndarray,
        e2_i: jnp.ndarray,
        center_x_i: jnp.ndarray,
        center_y_i: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_one = jnp.reshape(x_i, (1,))
        y_one = jnp.reshape(y_i, (1,))
        alpha_x, alpha_y = DPIENIE.derivatives(x_one, y_one, sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i)
        h_xx, h_xy, h_yx, h_yy = DPIENIE.hessian(x_one, y_one, sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i)
        return alpha_x[0], alpha_y[0], -h_xx[0], -h_xy[0], -h_yx[0], -h_yy[0]

    return jax.vmap(one_component)(x, y, sigma0, ra, rs, e1, e2, center_x, center_y)


grouped_dpie_derivatives_and_hessian = _grouped_dpie_derivatives_and_hessian


def alpha_and_hessian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: StaticLensState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    zeros = jnp.zeros_like(x, dtype=jnp.float64)
    alpha_x = zeros
    alpha_y = zeros
    h_xx = zeros
    h_xy = zeros
    h_yx = zeros
    h_yy = zeros
    if int(state.profile_type.shape[0]):
        dpie_mask = state.profile_type == 1
        dpie_values = _grouped_dpie_derivatives_and_hessian(
            x,
            y,
            jnp.where(dpie_mask, state.sigma0, 0.0),
            jnp.where(dpie_mask, state.Ra, 1.0),
            jnp.where(dpie_mask, state.Rs, 1.0),
            jnp.where(dpie_mask, state.e1, 0.0),
            jnp.where(dpie_mask, state.e2, 0.0),
            state.center_x,
            state.center_y,
        )
        alpha_x = alpha_x + dpie_values[0]
        alpha_y = alpha_y + dpie_values[1]
        h_xx = h_xx + dpie_values[2]
        h_xy = h_xy + dpie_values[3]
        h_yx = h_yx + dpie_values[4]
        h_yy = h_yy + dpie_values[5]
    shear_mask = state.profile_type == 14
    gamma1 = jnp.sum(jnp.where(shear_mask, state.gamma1, 0.0))
    gamma2 = jnp.sum(jnp.where(shear_mask, state.gamma2, 0.0))
    ones = jnp.ones_like(x, dtype=jnp.float64)
    alpha_x = alpha_x + gamma1 * x + gamma2 * y
    alpha_y = alpha_y + gamma2 * x - gamma1 * y
    h_xx = h_xx + gamma1 * ones
    h_xy = h_xy + gamma2 * ones
    h_yx = h_yx + gamma2 * ones
    h_yy = h_yy - gamma1 * ones
    return alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy


def alpha_and_hessian_accumulated(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: StaticLensState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    zeros = jnp.zeros_like(x, dtype=jnp.float64)
    initial = (zeros, zeros, zeros, zeros, zeros, zeros)

    def add_component(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        component: tuple[jnp.ndarray, ...],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        profile_type_i, sigma0_i, ra_i, rs_i, e1_i, e2_i, center_x_i, center_y_i = component
        is_dpie = profile_type_i == 1
        values = _dpie_component_derivatives_and_hessian(
            x,
            y,
            jnp.where(is_dpie, sigma0_i, 0.0),
            jnp.where(is_dpie, ra_i, 1.0),
            jnp.where(is_dpie, rs_i, 1.0),
            jnp.where(is_dpie, e1_i, 0.0),
            jnp.where(is_dpie, e2_i, 0.0),
            center_x_i,
            center_y_i,
        )
        return tuple(carry_i + value_i for carry_i, value_i in zip(carry, values)), None

    alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy = jax.lax.scan(
        add_component,
        initial,
        (
            state.profile_type,
            state.sigma0,
            state.Ra,
            state.Rs,
            state.e1,
            state.e2,
            state.center_x,
            state.center_y,
        ),
    )[0]
    shear_mask = state.profile_type == 14
    gamma1 = jnp.sum(jnp.where(shear_mask, state.gamma1, 0.0))
    gamma2 = jnp.sum(jnp.where(shear_mask, state.gamma2, 0.0))
    ones = jnp.ones_like(x, dtype=jnp.float64)
    alpha_x = alpha_x + gamma1 * x + gamma2 * y
    alpha_y = alpha_y + gamma2 * x - gamma1 * y
    h_xx = h_xx + gamma1 * ones
    h_xy = h_xy + gamma2 * ones
    h_yx = h_yx + gamma2 * ones
    h_yy = h_yy - gamma1 * ones
    return alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy


_alpha_and_hessian_accumulated_jit = jax.jit(alpha_and_hessian_accumulated)


def _pad_to_length(values: jnp.ndarray, size: int) -> tuple[jnp.ndarray, int]:
    values = jnp.asarray(values, dtype=jnp.float64)
    original_size = int(values.shape[0])
    if original_size > int(size):
        raise ValueError("Cannot pad an array to a smaller size.")
    if original_size == int(size):
        return values, original_size
    pad_width = int(size) - original_size
    return jnp.pad(values, (0, pad_width), mode="edge"), original_size


def alpha_and_hessian_tiled(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: StaticLensState,
    *,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    flat_x = jnp.ravel(x)
    flat_y = jnp.ravel(y)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if int(flat_x.shape[0]) > 0:
        chunk_size = min(chunk_size, int(flat_x.shape[0]))
    chunks: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    for start in range(0, int(flat_x.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(flat_x.shape[0]))
        x_chunk, valid_size = _pad_to_length(flat_x[start:stop], chunk_size)
        y_chunk, _ = _pad_to_length(flat_y[start:stop], chunk_size)
        values = _alpha_and_hessian_accumulated_jit(x_chunk, y_chunk, state)
        chunks.append(tuple(value[:valid_size] for value in values))
    if not chunks:
        empty = jnp.asarray([], dtype=jnp.float64)
        return empty, empty, empty, empty, empty, empty
    return tuple(jnp.concatenate([chunk[index] for chunk in chunks], axis=0).reshape(x.shape) for index in range(6))


def estimated_alpha_hessian_chunk_size(memory_gb: float, *, bytes_per_point: int = 256) -> int:
    memory = float(memory_gb)
    if not math.isfinite(memory) or memory <= 0.0:
        raise ValueError("memory_gb must be positive and finite.")
    return max(1, int((memory * 1024.0**3) // int(bytes_per_point)))


def ray_shooting(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: StaticLensState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    alpha_x, alpha_y, *_ = alpha_and_hessian(x, y, state)
    return jnp.asarray(x, dtype=jnp.float64) - alpha_x, jnp.asarray(y, dtype=jnp.float64) - alpha_y


def ray_shooting_tiled(
    x: jnp.ndarray,
    y: jnp.ndarray,
    state: StaticLensState,
    *,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    alpha_x, alpha_y, *_ = alpha_and_hessian_tiled(x, y, state, chunk_size=chunk_size)
    return jnp.asarray(x, dtype=jnp.float64) - alpha_x, jnp.asarray(y, dtype=jnp.float64) - alpha_y


# ----------------------------------------------------------------------------- analytic dPIE fields
def dpie_geometry(x, y, e1, e2, center_x, center_y):
    """Geometry-only quantities of the cartesian dPIE kernel at points (x, y): everything that depends on the
    position, the ellipticity and the centre but not on the amplitude or the radii.  For components with fixed
    catalog geometry this is a constant and can be precomputed once per set of evaluation points."""
    xs, ys, d, e, v, rsq, rx, ry = _nie_geometry(x, y, e1, e2, center_x, center_y)
    e_sq = e1 * e1 + e2 * e2
    qxx, qxy, qyy = (1 + e_sq - 2 * e1) / d, -2 * e2 / d, (1 + e_sq + 2 * e1) / d
    c = jnp.conj(e) / d
    return dict(rsq=rsq, rx=rx, ry=ry, v=v, c=c, qxx=qxx * jnp.ones_like(rsq), qxy=qxy * jnp.ones_like(rsq), qyy=qyy * jnp.ones_like(rsq),
                vlx=(1 - e) * jnp.ones_like(v), vly=(1j * (1 + e)) * jnp.ones_like(v))


def dpie_fields_from_geometry(geom, sigma0, a, b, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (N, 2, 2, 2) of one
    cartesian dPIE component from its precomputed geometry; the analytic form of two jvps through the Hessian.
    With ``derivative`` False the second-derivative terms are skipped and dU is None."""
    rsq, c, v = geom["rsq"], geom["c"], geom["v"]
    r = (geom["rx"], geom["ry"])
    q = ((geom["qxx"], geom["qxy"]), (geom["qxy"], geom["qyy"]))
    vl = (geom["vlx"], geom["vly"])
    ua, ub = jnp.sqrt(a * a + rsq), jnp.sqrt(b * b + rsq)
    da, db = ua + a, ub + b
    T = ua + ub
    m = 1 + (a + b) / T
    D = da * db + c * v * v
    g = v * m / D
    k2 = c * (b - a) ** 2
    u = k2 * g * g
    ual = [rl / ua for rl in r]
    ubl = [rl / ub for rl in r]
    Tl = [ual[i] + ubl[i] for i in range(2)]
    ml = [-(a + b) * Tl[i] / T ** 2 for i in range(2)]
    Dl = [ual[i] * db + da * ubl[i] + 2 * c * v * vl[i] for i in range(2)]
    Nl = [vl[i] * m + v * ml[i] for i in range(2)]
    gl = [(Nl[i] - g * Dl[i]) / D for i in range(2)]
    gll = [[None, None], [None, None]]
    for i in range(2 if derivative else 0):
        for k in range(i, 2):
            ua_ik = q[i][k] / ua - r[i] * r[k] / ua ** 3
            ub_ik = q[i][k] / ub - r[i] * r[k] / ub ** 3
            T_ik = ua_ik + ub_ik
            m_ik = -(a + b) * (T_ik / T ** 2 - 2 * Tl[i] * Tl[k] / T ** 3)
            D_ik = ua_ik * db + ual[i] * ubl[k] + ual[k] * ubl[i] + da * ub_ik + 2 * c * vl[i] * vl[k]
            N_ik = vl[i] * ml[k] + vl[k] * ml[i] + v * m_ik
            gll[i][k] = gll[k][i] = (N_ik - gl[k] * Dl[i] - g * D_ik - gl[i] * Dl[k]) / D
    amp = 2 * sigma0 * a * b
    alpha = amp * g * _atan_sqrt_ratio(u)
    S = amp / (1 + u)
    ul = [2 * k2 * g * gl[i] for i in range(2)]
    Sl = [-S * ul[i] / (1 + u) for i in range(2)]
    ax, ay = S * gl[0], S * gl[1]
    hxy = 0.5 * (jnp.imag(ax) + jnp.real(ay))
    U = jnp.stack((jnp.stack((jnp.real(ax), hxy), -1), jnp.stack((hxy, jnp.imag(ay)), -1)), -2)
    if not derivative:
        return jnp.stack((jnp.real(alpha), jnp.imag(alpha)), -1), U, None
    dU_l = []
    for l in range(2):
        ax_l = Sl[l] * gl[0] + S * gll[0][l]
        ay_l = Sl[l] * gl[1] + S * gll[1][l]
        cross = 0.5 * (jnp.imag(ax_l) + jnp.real(ay_l))
        dU_l.append(jnp.stack((jnp.stack((jnp.real(ax_l), cross), -1), jnp.stack((cross, jnp.imag(ay_l)), -1)), -2))
    return jnp.stack((jnp.real(alpha), jnp.imag(alpha)), -1), U, jnp.stack(dU_l, -1)


def dpie_fields(x, y, sigma0, a, b, e1, e2, center_x, center_y, derivative=True):
    """alpha, U and dU (None unless ``derivative``) of one dPIE component at points (x, y) with variable geometry."""
    return dpie_fields_from_geometry(dpie_geometry(x, y, e1, e2, center_x, center_y), sigma0, a, b, derivative)


grouped_dpie_geometry = jax.vmap(dpie_geometry, in_axes=(None, None, 0, 0, 0, 0))
# Batches over components; the ``derivative`` flag is static, so each variant is its own vmapped function.
grouped_dpie_fields = {
    True: jax.vmap(dpie_fields, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0)),
    False: jax.vmap(functools.partial(dpie_fields, derivative=False), in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0)),
}
grouped_dpie_fields_from_geometry = {
    True: jax.vmap(dpie_fields_from_geometry, in_axes=(0, 0, 0, 0)),
    False: jax.vmap(functools.partial(dpie_fields_from_geometry, derivative=False), in_axes=(0, 0, 0, 0)),
}


# ----------------------------------------------------------------------------- elliptical NFW fields
# Oguri (2021) CSE decomposition of the NFW convergence with ellipticity in the convergence; the 44-term set matches
# the NFW to about one part in a thousand.  Deflection, Jacobian and its derivative are linear in alpha_Rs, so the kernel
# is evaluated with unit normalization and scaled afterwards, which lets the normalization vary per point (per image).
# Table 1 of Oguri (2021), the 44-term CSE decomposition of the NFW convergence (same table as jaxtronomy's
# NFW_ELLIPSE_CSE with high_accuracy=True, kept here as NumPy so importing this module does not initialize JAX).
_CSE_A = np.array([1.6489880e-18, 6.2744580e-16, 3.6466200e-17, 3.4592060e-15, 2.4573890e-14, 1.0593190e-13, 4.2115970e-13, 1.1428320e-12, 4.3912150e-12, 1.5565000e-11, 6.9512710e-11,
                   3.1474660e-10, 1.3791090e-09, 3.8297780e-09, 1.3848580e-08, 5.3709510e-08, 1.8043840e-07, 5.7886080e-07, 3.2052560e-06, 1.1024220e-05, 4.0939710e-05, 1.2822060e-04,
                   4.5755410e-04, 7.9952700e-04, 5.0137010e-03, 1.4035080e-02, 5.2307270e-02, 1.8989070e-01, 3.6434480e-01, 7.2037340e-01, 1.7176670e+00, 2.2175660e+00, 3.1874470e+00,
                   8.1948980e+00, 1.7652100e+01, 1.9743190e+01, 2.7836880e+01, 4.4823110e+01, 5.5988970e+01, 1.4264850e+02, 2.2798330e+02, 5.4013350e+02, 9.7436820e+02, 1.7751240e+03])
_CSE_S = np.array([1.0824110e-06, 8.7865660e-06, 3.2928680e-06, 1.8600190e-05, 3.2742310e-05, 6.2324850e-05, 9.2563330e-05, 1.5467620e-04, 2.0973210e-04, 3.3911400e-04, 5.1787900e-04,
                   8.6367360e-04, 1.4051520e-03, 2.1938550e-03, 3.1795720e-03, 4.9709870e-03, 7.6319700e-03, 1.1194130e-02, 1.8272670e-02, 2.9452510e-02, 4.5627230e-02, 6.7825090e-02,
                   1.5969870e-01, 1.1277510e-01, 2.1694690e-01, 3.4238350e-01, 5.1945270e-01, 8.6231850e-01, 1.3827370e+00, 2.0349290e+00, 3.4029790e+00, 5.5942760e+00, 8.0523450e+00,
                   1.3490450e+01, 2.6038250e+01, 4.7368230e+01, 6.5593200e+01, 1.0879320e+02, 1.4776730e+02, 2.4953410e+02, 4.3059990e+02, 7.7602060e+02, 2.1430570e+03, 1.9357490e+03])
_NFW_UNIT_NORMALIZATION = 1.0 / (1.0 - math.log(2.0))      # 4 rho0 Rs^3 / Rs for alpha_Rs = 1
_ELLIPTICITY_SINGULARITY = 1.0e-12


@jax.custom_jvp
def ellipticity_frame(e1, e2):
    """Position angle and axis ratio of the (e1, e2) ellipticity.  The lensing fields are even in the ellipticity, so
    their derivative vanishes at e = 0; the custom tangent below returns that limit instead of the 0/0 of arctan2."""
    c = jnp.sqrt(e1 * e1 + e2 * e2)
    return 0.5 * jnp.arctan2(e2, e1), (1.0 - c) / (1.0 + c)


@ellipticity_frame.defjvp
def _ellipticity_frame_jvp(primals, tangents):
    e1, e2 = primals
    t1, t2 = tangents
    c2 = e1 * e1 + e2 * e2
    regular = c2 > _ELLIPTICITY_SINGULARITY ** 2
    safe_c2 = jnp.where(regular, c2, 1.0)
    c = jnp.sqrt(safe_c2)
    dphi = jnp.where(regular, 0.5 * (e1 * t2 - e2 * t1) / safe_c2, 0.0)
    dq = jnp.where(regular, -2.0 / (1.0 + c) ** 2 * (e1 * t1 + e2 * t2) / c, 0.0)
    return ellipticity_frame(e1, e2), (dphi, dq)


def _cse_set_deflection(x, y, q):
    """Deflection of the unit-normalized product-averaged CSE set (Oguri 2021, eqs. 19-20) in the halo frame; the sum
    over the 44 terms is a broadcast rather than a loop so that reverse-mode differentiation stays cheap."""
    a, s = _CSE_A[:, None] / q, _CSE_S[:, None]
    xq, yq = x[None, :] * jnp.sqrt(q), y[None, :] * jnp.sqrt(q)
    psi = jnp.sqrt(q * q * (s * s + xq * xq) + yq * yq)
    phi = (psi + s) ** 2 + (1.0 - q * q) * xq * xq
    common = a * q * jnp.sqrt(q) / (s * psi * phi)
    return jnp.sum(common * xq * (psi + q * q * s), axis=0), jnp.sum(common * yq * (psi + s), axis=0)


def _cse_set_hessian(x, y, q):
    """Second derivatives (f_xx, f_xy, f_yy) of the same set (Oguri 2021, eqs. 21-23)."""
    a, s = _CSE_A[:, None] / q, _CSE_S[:, None]
    xq, yq = x[None, :] * jnp.sqrt(q), y[None, :] * jnp.sqrt(q)
    q2 = q * q
    psi = jnp.sqrt(q2 * (s * s + xq * xq) + yq * yq)
    phi = (psi + s) ** 2 + (1.0 - q2) * xq * xq
    pre = a * q2 / (s * phi)                                   # extra q from the two sqrt(q) coordinate factors
    psi3, psi2phi = psi ** 3, psi * psi * phi
    fxx = pre * (1.0 + q2 * s * (q2 * s * s + yq * yq) / psi3 - 2.0 * xq * xq * (psi + q2 * s) ** 2 / psi2phi)
    fyy = pre * (1.0 + q2 * s * (s * s + xq * xq) / psi3 - 2.0 * yq * yq * (psi + s) ** 2 / psi2phi)
    fxy = -pre * xq * yq * (q2 * s / psi3 + 2.0 * (psi + q2 * s) * (psi + s) / psi2phi)
    return jnp.sum(fxx, axis=0), jnp.sum(fxy, axis=0), jnp.sum(fyy, axis=0)


def nfw_fields(x, y, rs, alpha_rs, e1, e2, center_x, center_y, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (N, 2, 2, 2; None unless
    ``derivative``) of one elliptical NFW with scale radius ``rs`` (arcsec) and deflection ``alpha_rs`` at the scale radius."""
    scale = jnp.broadcast_to(jnp.asarray(alpha_rs, dtype=jnp.float64), jnp.shape(x))
    phi, q = ellipticity_frame(e1, e2)
    c, s = jnp.cos(phi), jnp.sin(phi)
    rotation = jnp.array([[c, s], [-s, c]])                   # halo frame = rotation @ (theta - centre)

    def frame(points):
        return ((points[:, 0] - center_x) * c + (points[:, 1] - center_y) * s) / rs, (-(points[:, 0] - center_x) * s + (points[:, 1] - center_y) * c) / rs

    def jacobian(points):
        xr, yr = frame(points)
        fxx, fxy, fyy = _cse_set_hessian(xr, yr, q)
        u_frame = jnp.stack((jnp.stack((fxx, fxy), -1), jnp.stack((fxy, fyy), -1)), -2) * (_NFW_UNIT_NORMALIZATION / rs)
        return jnp.einsum("ai,bj,nab->nij", rotation, rotation, u_frame)

    xy = jnp.stack((x, y), -1)
    fx, fy = _cse_set_deflection(*frame(xy), q)
    alpha = jnp.einsum("ai,na->ni", rotation, jnp.stack((fx, fy), -1)) * (_NFW_UNIT_NORMALIZATION * scale[:, None])
    u = jacobian(xy) * scale[:, None, None]
    if not derivative:
        return alpha, u, None
    du = jnp.stack([jax.jvp(jacobian, (xy,), (jnp.broadcast_to(jnp.asarray(direction), xy.shape),))[1]
                    for direction in ((1.0, 0.0), (0.0, 1.0))], -1)
    return alpha, u, du * scale[:, None, None, None]


def _rotate(x, y, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return c * x + s * y, -s * x + c * y


grouped_nfw_fields = {
    True: jax.vmap(nfw_fields, in_axes=(None, None, 0, 0, 0, 0, 0, 0)),
    False: jax.vmap(functools.partial(nfw_fields, derivative=False), in_axes=(None, None, 0, 0, 0, 0, 0, 0)),
}
