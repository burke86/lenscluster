"""The knee-spectrum mass field plus the image-plane GP, on blocks of images that may hold several knots (sources).

Model (see .agents/gp-knee-plan.md).  Image i of knot k(i), redshift weight w_i = (D_ls/D_s)(z_i) / (D_ls/D_s)(z_ref):

    F_i = b_i - beta_k(i) = w_i delta_alpha(theta_i) + A_i (d(theta_i) + e_i)

delta_alpha is the deflection error of unmodelled mass, a curl-free Gaussian field on the lens potential with a
convergence power spectrum P_kappa(k) = P_0 / (1 + (k / k_t)^gamma) between fixed cutoffs (gamma fixed or sampled); d is a correlated image-plane
displacement with an isotropic Gaussian kernel; e_i the independent position error.  Covariance of the stacked F of a
block (all its knots):

    C = W K_alpha W + B [sigma_sh^2 K_img + diag(sigma_inst,i^2 + sigma_w^2)] B^T + sigma_fl^2 I

    K_alpha,ab(d) = sigma_alpha^2 [c_T(r) delta_ab + (c_L(r) - c_T(r)) n_a n_b],   r = |d|,  n = d / r,

with c_L, c_T the longitudinal and transverse correlations of the field (Hankel transforms of P_kappa, section
``knee_tables``), normalized to 1 at r = 0, so sigma_alpha is the rms deflection error per component for a source at
z_ref.  The blocks are one per family (one knot), one per system (all knots of a galaxy) or one over all images.  Every
block is padded to the same number of images M and knots K; padded images carry identity blocks and zero residuals,
padded knots an identity source precision, so neither contributes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from scipy.interpolate import CubicSpline
from scipy.special import jv

from .gp_image_plane import family_slots

_LOG_2PI = float(np.log(2.0 * np.pi))

# ----------------------------------------------------------------------------- the knee spectrum, tabulated
def _natural_second_derivative_operator(x) -> np.ndarray:
    """M (n, n) with m = M y the second derivatives of the natural cubic spline through (x, y) (zero at both ends)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    operator = np.zeros((n, n))
    if n < 3:
        return operator
    h = np.diff(x)
    t = np.zeros((n - 2, n - 2))
    d = np.zeros((n - 2, n))
    for i in range(1, n - 1):
        row = i - 1
        t[row, row] = 2.0 * (h[i - 1] + h[i])
        if row > 0:
            t[row, row - 1] = h[i - 1]
        if row < n - 3:
            t[row, row + 1] = h[i]
        d[row, i - 1], d[row, i], d[row, i + 1] = 6.0 / h[i - 1], -6.0 / h[i - 1] - 6.0 / h[i], 6.0 / h[i]
    operator[1:-1] = np.linalg.solve(t, d)
    return operator


def knee_tables(*, r_max: float, r_step: float, lam_min: float, lam_max: float, n_lam: int, scale_min: float, scale_max: float,
                gamma: float | None = None, gamma_grid: tuple[float, float, int] | None = None, n_nodes: int = 3000,
                shape: str = "knee", beta_low_grid: tuple[float, float, int] | None = None) -> dict:
    """c_L(r; lambda_t, gamma) and c_T(r; lambda_t, gamma) on a grid of separations r (step ``r_step`` to ``r_max``), G =
    ``n_lam`` knee wavelengths log-spaced on [lam_min, lam_max] and H slopes: the single fixed ``gamma`` (H = 1) or
    ``gamma_grid`` = (min, max, count) evenly spaced when the slope is sampled.  Stored as (R, H, G) with the second
    derivatives of the natural cubic splines in ln lambda_t, and the operator that gives the natural-spline second
    derivatives in gamma.  Quadrature on ``n_nodes`` log-spaced k between 2 pi / scale_max and 2 pi / scale_min:

        c_L,T(r) = sum_k P_kappa(k) dlnk [J_0(k r) -/+ J_2(k r)] / sum_k P_kappa(k) dlnk,   P_kappa = 1 / (1 + (k / k_t)^gamma).

    (k^3 P_psi dk = 4 P_kappa dk / k with P_psi = 4 P_kappa / k^4, so the weights are P_kappa dlnk.)

    ``shape`` changes P_kappa and what the second axis holds (config.LikelihoodConfig.gp_knee_spectrum): "knee" as above;
    "broken_power_law", P = x^-beta_low (1 + x^gamma)^-((gamma - beta_low) / gamma) with x = k / k_b, the fixed ``gamma`` the
    high-k slope and the second axis the low-k slopes ``beta_low_grid`` (beta_low = 0 is the knee); "power_law", P = k^-gamma, the same on every lambda."""
    if shape not in ("knee", "broken_power_law", "power_law"):
        raise ValueError(f"knee_tables: unknown shape {shape!r}")
    if shape == "broken_power_law":
        if gamma is None or beta_low_grid is None:
            raise ValueError("a broken power law needs the fixed high-k slope gamma and beta_low_grid")
        gam = np.linspace(float(beta_low_grid[0]), float(beta_low_grid[1]), int(beta_low_grid[2]))
    elif (gamma is None) == (gamma_grid is None):
        raise ValueError("knee_tables needs exactly one of gamma (fixed slope) or gamma_grid (sampled slope)")
    else:
        gam = np.array([float(gamma)]) if gamma_grid is None else np.linspace(float(gamma_grid[0]), float(gamma_grid[1]), int(gamma_grid[2]))
    k = np.geomspace(2.0 * np.pi / scale_max, 2.0 * np.pi / scale_min, int(n_nodes))
    dlnk = np.log(k[1] / k[0])
    r = np.arange(0.0, float(r_max) + float(r_step), float(r_step))
    lam = np.geomspace(float(lam_min), float(lam_max), int(n_lam))
    x_k = k[:, None, None] / (2.0 * np.pi / lam[None, None, :])                                       # k / k_t, (N, 1, G)
    if shape == "knee":
        weight = dlnk / (1.0 + x_k ** gam[None, :, None])                                              # (N, H, G)
    elif shape == "broken_power_law":
        b = gam[None, :, None]
        weight = dlnk * x_k ** (-b) * (1.0 + x_k ** float(gamma)) ** (-(float(gamma) - b) / float(gamma))
    else:
        weight = dlnk * np.broadcast_to(k[:, None, None] ** (-gam[None, :, None]), (k.size, gam.size, lam.size)).copy()
    weight /= weight.sum(axis=0, keepdims=True)
    weight = weight.reshape(k.size, gam.size * lam.size)
    j0, j2 = jv(0, np.outer(r, k)), jv(2, np.outer(r, k))
    c_l = ((j0 - j2) @ weight).reshape(r.size, gam.size, lam.size)
    c_t = ((j0 + j2) @ weight).reshape(r.size, gam.size, lam.size)
    del j0, j2
    x = np.log(lam)
    d2_l = CubicSpline(x, c_l, axis=2, bc_type="natural").derivative(2)(x)
    d2_t = CubicSpline(x, c_t, axis=2, bc_type="natural").derivative(2)(x)
    return dict(r_grid=jnp.asarray(r), r_step=float(r_step), ln_lam=jnp.asarray(x),
                c_l=jnp.asarray(c_l), c_t=jnp.asarray(c_t), d2_l=jnp.asarray(d2_l), d2_t=jnp.asarray(d2_t),
                gamma_grid=jnp.asarray(gam), gamma_operator=jnp.asarray(_natural_second_derivative_operator(gam)),
                gamma=None if (gamma_grid is not None or shape == "broken_power_law") else float(gamma), shape=shape,
                scales=(float(scale_min), float(scale_max)), lam_range=(float(lam_min), float(lam_max)))


def r_grid_position(tables: dict, r):
    """Static index and fraction for linear interpolation of the tables in r (r is data)."""
    step = tables["r_step"]
    n = int(tables["r_grid"].shape[0])
    r_max = float(tables["r_grid"][-1])
    if np.any(np.asarray(r, dtype=float) > r_max + 1e-9):
        raise ValueError(f"a pair separation ({float(np.max(r)):.1f} arcsec) exceeds the knee tables' r_max ({r_max:.1f} arcsec)")
    position = np.asarray(r, dtype=float) / step
    index = np.clip(np.floor(position).astype(np.int32), 0, n - 2)
    fraction = np.clip(position - index, 0.0, 1.0)
    return jnp.asarray(index), jnp.asarray(fraction)


def _spline_last(values, second, nodes, x):
    """Natural cubic spline along the last axis of ``values`` (..., G) at the traced x, with second derivatives ``second``."""
    n = nodes.shape[0]
    g = jnp.clip(jnp.searchsorted(nodes, x, side="right") - 1, 0, n - 2)
    x0, x1 = nodes[g], nodes[g + 1]
    h = x1 - x0
    a, b = (x1 - x) / h, (x - x0) / h
    y0, y1 = values[..., g], values[..., g + 1]
    m0, m1 = second[..., g], second[..., g + 1]
    return a * y0 + b * y1 + ((a ** 3 - a) * m0 + (b ** 3 - b) * m1) * h ** 2 / 6.0


def knee_curves(tables: dict, lam, gamma=None):
    """c_L and c_T on the whole r grid at a (traced) knee wavelength and slope: a natural cubic spline in ln lambda_t, then
    (sampled slope) one in gamma through the H slope nodes.  ln lambda_t and gamma are clamped to the table ranges."""
    x = jnp.clip(jnp.log(lam), tables["ln_lam"][0], tables["ln_lam"][-1])
    nodes = tables["gamma_grid"]
    curves = []
    for values, second in ((tables["c_l"], tables["d2_l"]), (tables["c_t"], tables["d2_t"])):
        by_gamma = _spline_last(values, second, tables["ln_lam"], x)                 # (R, H)
        if nodes.shape[0] == 1:
            curves.append(by_gamma[:, 0])
            continue
        if gamma is None:
            raise ValueError("these knee tables have a slope axis: pass gamma")
        g = jnp.clip(gamma, nodes[0], nodes[-1])
        curves.append(_spline_last(by_gamma, by_gamma @ tables["gamma_operator"].T, nodes, g))
    return curves[0], curves[1]


def knee_coefficients(tables: dict, r_index, r_fraction, lam, gamma=None):
    """(c_L, c_T) at the tabulated pair separations (any shape of ``r_index``) for a traced knee wavelength and slope."""
    c_l, c_t = knee_curves(tables, lam, gamma)
    f = r_fraction
    return ((1.0 - f) * c_l[r_index] + f * c_l[r_index + 1], (1.0 - f) * c_t[r_index] + f * c_t[r_index + 1])


def pair_tables(tables: dict, r_index) -> tuple[dict, jnp.ndarray]:
    """The tables restricted to the separations the data use: rows [u, u + 1] for the U distinct lower grid indices u of
    ``r_index`` (static), so the spline in ln lambda_t and gamma runs over 2U rows instead of the whole r grid.  Returns the
    restricted tables (r_grid dropped) and the position of each r_index in u; ``pair_coefficients`` reads them."""
    index = np.asarray(r_index)
    unique, local = np.unique(index, return_inverse=True)
    rows = np.concatenate([unique, unique + 1])
    restricted = {k: v for k, v in tables.items() if k != "r_grid"}
    for key in ("c_l", "c_t", "d2_l", "d2_t"):
        restricted[key] = jnp.asarray(np.asarray(tables[key])[rows])
    restricted["n_unique"] = int(unique.size)
    return restricted, jnp.asarray(local.reshape(index.shape), dtype=jnp.int32)


def pair_coefficients(restricted: dict, local, r_fraction, lam, gamma=None):
    """(c_L, c_T) at the data's pair separations from ``pair_tables``: the same values as ``knee_coefficients``."""
    c_l, c_t = knee_curves(restricted, lam, gamma)
    n, f = restricted["n_unique"], r_fraction
    return ((1.0 - f) * c_l[local] + f * c_l[n + local], (1.0 - f) * c_t[local] + f * c_t[n + local])


LOW_ORDER_MODES = ("translation", "convergence", "shear")
PROJECTION_SCALE_ARCSEC = 50.0


def low_order_basis(xy, mask, modes, scale: float = PROJECTION_SCALE_ARCSEC):
    """Deflection patterns of the low-order modes at points ``xy`` (..., M, 2) about each block's image centre: translation
    (x, y), convergence (theta), shear (gamma_1: (x, -y); gamma_2: (y, x)), in coordinates (theta - centre) / ``scale``.
    Returns the patterns (..., M, 2, n) and the centres (..., 2); padded slots carry zero patterns."""
    xy, mask = jnp.asarray(xy), jnp.asarray(mask)
    m = mask.astype(xy.dtype)
    centre = jnp.sum(xy * m[..., None], axis=-2) / jnp.maximum(jnp.sum(m, axis=-1), 1.0)[..., None]
    return low_order_patterns(xy, centre, modes, scale) * m[..., None, None], centre


def low_order_patterns(points, centre, modes, scale: float = PROJECTION_SCALE_ARCSEC):
    """The patterns of ``low_order_basis`` at arbitrary points (..., P, 2) for given centres (..., 2): (..., P, 2, n)."""
    u = (jnp.asarray(points) - jnp.asarray(centre)[..., None, :]) / scale
    ux, uy = u[..., 0], u[..., 1]
    one, zero = jnp.ones_like(ux), jnp.zeros_like(ux)
    columns = []
    for mode in modes:
        if mode == "translation":
            columns += [jnp.stack([one, zero], -1), jnp.stack([zero, one], -1)]
        elif mode == "convergence":
            columns += [jnp.stack([ux, uy], -1)]
        elif mode == "shear":
            columns += [jnp.stack([ux, -uy], -1), jnp.stack([uy, ux], -1)]
        else:
            raise ValueError(f"unknown low-order mode {mode!r}; use {LOW_ORDER_MODES}")
    return jnp.stack(columns, -1)


def low_order_projector(xy, mask, modes):
    """Orthonormal basis Q (S, 2M, n) of the low-order patterns over each block's images (QR, so P = Q Q^T is the
    least-squares projector onto them) and the factor R (S, n, n) with patterns = Q R; plus the centres (S, 2)."""
    basis, centre = low_order_basis(xy, mask, modes)
    n_blocks, size = jnp.asarray(mask).shape
    q, r = jnp.linalg.qr(basis.reshape(n_blocks, 2 * size, -1))
    return q, r, centre


def project_low_order(k_alpha, xy, mask, weights, modes):
    """The weighted mass-field covariance blocks (S, M, M, 2, 2) with the low-order modes removed from the field:  the
    field at z_ref is split at the images into its least-squares low-order fit and a residual, alpha = Phi c + r, and only r
    is kept,  K_r = (I - P) K (I - P),  P = Q Q^T,  with K = W^-1 k_alpha W^-1 the unweighted field covariance; then
    k_alpha' = W K_r W.  A congruence, so k_alpha' stays positive semidefinite, and Phi^T W^-1 k_alpha' = 0.  With flat priors
    on the external shear and the sources, the likelihood is unchanged (the removed modes are absorbed by them); the shear
    parameter then means the total shear over the images.  This is not the field conditioned on c = 0, which is a
    different (smoother) residual field and changes the likelihood."""
    k_alpha, mask, weights = jnp.asarray(k_alpha), jnp.asarray(mask), jnp.asarray(weights)
    n_blocks, size = mask.shape
    inverse_w = jnp.repeat(jnp.where(mask, 1.0 / jnp.where(mask, weights, 1.0), 0.0), 2, axis=-1)          # (S, 2M)
    w2 = jnp.repeat(jnp.where(mask, weights, 0.0), 2, axis=-1)
    k = k_alpha.transpose(0, 1, 3, 2, 4).reshape(n_blocks, 2 * size, 2 * size)
    k = k * inverse_w[:, :, None] * inverse_w[:, None, :]
    q, _, _ = low_order_projector(xy, mask, modes)
    a = k - q @ (jnp.swapaxes(q, -1, -2) @ k)                                                          # (I - P) K
    kr = a - (a @ q) @ jnp.swapaxes(q, -1, -2)                                                         # (I - P) K (I - P)
    kr = 0.5 * (kr + jnp.swapaxes(kr, -1, -2))
    kr = kr * w2[:, :, None] * w2[:, None, :]
    return kr.reshape(n_blocks, size, 2, size, 2).transpose(0, 1, 3, 2, 4)


def deflection_blocks(c_l, c_t, nn, pair_mask, sigma_alpha2):
    """sigma_alpha^2 [c_T I + (c_L - c_T) n n^T] per pair, zero at padded pairs; ``nn`` is n n^T (zero where r = 0)."""
    eye2 = jnp.eye(2, dtype=nn.dtype)
    block = sigma_alpha2 * (c_t[..., None, None] * eye2 + (c_l - c_t)[..., None, None] * nn)
    return block * pair_mask[..., None, None]


# ----------------------------------------------------------------------------- block layouts with several knots
def block_layout(images, blocks: str, weights, tables: dict | None = None) -> dict:
    """Padded blocks of images: by ``"family"`` (one knot each), ``"system"`` (all knots of a galaxy) or ``"all"``.

    slots (S, M) catalog indices, mask (S, M); knot_of_slot (S, M) the knot's position within its block; knot_family
    (S, K) the family index of each knot (-1 at pads) and knot_mask (S, K); J (S, 2M, 2K) the source indicator;
    block_of_family (F,) and slot_of_family... are not needed (sources are per family).  Pair geometry: separations
    (S, M, M, 2), r (S, M, M), nn (S, M, M, 2, 2) = n n^T (zero at r = 0), pair_mask (S, M, M), and the table positions
    r_index, r_fraction when ``tables`` is given.  weights (S, M) are the redshift weights, sigma_inst2 (S, M)."""
    family = np.asarray(images.family_index_per_image, dtype=int)
    if blocks == "family":
        group = family
    elif blocks == "system":
        group = np.asarray(images.system_index_per_image, dtype=int)
    elif blocks == "all":
        group = np.zeros(family.shape, dtype=int)
    else:
        raise ValueError(f"gp_blocks must be 'all', 'system' or 'family'; got {blocks!r}")
    n_groups = int(group.max()) + 1 if group.size else 0
    layout = family_slots(group, n_groups)
    slots, mask = layout["slots"], layout["mask"]
    n_blocks, size = slots.shape
    knot_of_slot = np.zeros((n_blocks, size), dtype=np.int32)
    knots_per_block = []
    for s in range(n_blocks):
        local: list[int] = []
        for m in np.flatnonzero(mask[s]):
            f = int(family[slots[s, m]])
            if f not in local:
                local.append(f)
            knot_of_slot[s, m] = local.index(f)
        knots_per_block.append(local)
    n_knots = max((len(k) for k in knots_per_block), default=1)
    knot_family = np.full((n_blocks, n_knots), -1, dtype=np.int32)
    for s, local in enumerate(knots_per_block):
        knot_family[s, :len(local)] = local
    knot_mask = knot_family >= 0
    indicator = np.zeros((n_blocks, 2 * size, 2 * n_knots))
    for s in range(n_blocks):
        for m in np.flatnonzero(mask[s]):
            k = knot_of_slot[s, m]
            indicator[s, 2 * m, 2 * k] = indicator[s, 2 * m + 1, 2 * k + 1] = 1.0
    block_of_family = np.full(int(images.n_families), -1, dtype=np.int32)
    for s, local in enumerate(knots_per_block):
        for f in local:
            block_of_family[f] = s
    real_block, real_knot = np.nonzero(knot_mask)                      # static index lists of the real knots
    xy = np.stack((np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)), -1)[slots]
    separations = xy[:, :, None, :] - xy[:, None, :, :]
    r = np.hypot(separations[..., 0], separations[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(r[..., None] > 0, separations / np.maximum(r, 1e-300)[..., None], 0.0)
    nn = n[..., :, None] * n[..., None, :]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    w = np.asarray(weights, dtype=float)
    out = dict(blocks=blocks, slots=jnp.asarray(slots), mask=jnp.asarray(mask), counts=jnp.asarray(layout["counts"], dtype=jnp.float64),
               size=size, n_knots=n_knots, knot_of_slot=jnp.asarray(knot_of_slot), knot_family=jnp.asarray(knot_family),
               knot_mask=jnp.asarray(knot_mask), indicator=jnp.asarray(indicator), block_of_family=jnp.asarray(block_of_family),
               real_block=jnp.asarray(real_block), real_knot=jnp.asarray(real_knot), real_family=jnp.asarray(knot_family[real_block, real_knot]),
               separations=jnp.asarray(separations), squared_separations=jnp.asarray(r ** 2), r=jnp.asarray(r), nn=jnp.asarray(nn),
               pair_mask=jnp.asarray(pair_mask), weights=jnp.asarray(np.where(mask, w[slots], 0.0)),
               sigma_inst2=jnp.asarray(np.where(mask, np.asarray(images.sigma_inst_per_image, dtype=float)[slots] ** 2, 0.0)),
               xy=jnp.asarray(xy), weights_per_image=jnp.asarray(w))
    if tables is not None:
        out["r_index"], out["r_fraction"] = r_grid_position(tables, r)
        out["pair_tables"], out["r_local"] = pair_tables(tables, out["r_index"])
    return out


_PER_BLOCK_IMAGE_KEYS = ("slots", "mask", "knot_of_slot", "weights", "sigma_inst2", "xy")
_PER_BLOCK_PAIR_KEYS = ("separations", "squared_separations", "r", "nn", "pair_mask", "r_index", "r_fraction", "r_local")


def bucket_layouts(layout: dict) -> list[dict]:
    """The blocks of ``layout`` grouped by size, each group padded only to its own largest block: blocks with up to 2, 4,
    8, 16, ... images.  Every block keeps its images, knots and pair geometry, so the likelihood is unchanged; padding a
    3-image system to the 16 images of the largest one costs (2M)^3 in every Cholesky.  Each group carries ``block_ids``,
    the positions of its blocks in ``layout``."""
    mask, knot_mask = np.asarray(layout["mask"]), np.asarray(layout["knot_mask"])
    counts, knots = mask.sum(1), knot_mask.sum(1)
    caps = np.array([max(2, 1 << int(np.ceil(np.log2(max(int(n), 1))))) for n in counts])
    groups = []
    for cap in sorted(set(caps.tolist())):
        ids = np.flatnonzero(caps == cap)
        size, n_knots = int(counts[ids].max()), int(max(knots[ids].max(), 1))
        sub = {k: v for k, v in layout.items() if k in ("blocks", "pair_tables")}
        for key in _PER_BLOCK_IMAGE_KEYS + ("weights2",):
            if key in layout:
                sub[key] = jnp.asarray(np.asarray(layout[key])[ids][:, :size])
        for key in _PER_BLOCK_PAIR_KEYS:
            if key in layout:
                sub[key] = jnp.asarray(np.asarray(layout[key])[ids][:, :size, :size])
        sub["counts"] = jnp.asarray(np.asarray(layout["counts"])[ids])
        sub["knot_family"] = jnp.asarray(np.asarray(layout["knot_family"])[ids][:, :n_knots])
        sub["knot_mask"] = jnp.asarray(knot_mask[ids][:, :n_knots])
        sub["indicator"] = jnp.asarray(np.asarray(layout["indicator"])[ids][:, :2 * size, :2 * n_knots])
        real_block, real_knot = np.nonzero(knot_mask[ids][:, :n_knots])
        sub.update(size=size, n_knots=n_knots, block_ids=ids, real_block=jnp.asarray(real_block), real_knot=jnp.asarray(real_knot),
                   real_family=jnp.asarray(np.asarray(layout["knot_family"])[ids][real_block, real_knot]))
        groups.append(sub)
    return groups


def block_covariance(jacobian, k_alpha, k_img, white, mask, sigma_fl2):
    """C = K_alpha + B S B^T + floor per block as (S, 2M, 2M) and its Cholesky factor.  ``k_alpha`` (S, M, M, 2, 2) is the
    weighted mass-field covariance including its diagonal; ``k_img`` (S, M, M) the image-plane GP kernel times its
    variance (diagonal included); ``white`` (S, M) the independent image-plane variances; ``jacobian`` (S, M, 2, 2)."""
    jacobian, k_alpha, mask = jnp.asarray(jacobian), jnp.asarray(k_alpha), jnp.asarray(mask)
    n_blocks, size = mask.shape
    eye2 = jnp.eye(2, dtype=jacobian.dtype)
    eye_m = jnp.eye(size, dtype=jacobian.dtype)
    a = jnp.where(mask[..., None, None], jacobian, 0.0)
    s_img = jnp.asarray(k_img) * (mask[:, :, None] & mask[:, None, :]) + eye_m[None] * jnp.where(mask, jnp.asarray(white), 0.0)[:, :, None]
    c = k_alpha + jnp.einsum("fij,fiab,fjcb->fijac", s_img, a, a)
    diagonal = jnp.where(mask, sigma_fl2, 1.0)
    c = c + (eye_m[None] * diagonal[:, :, None])[..., None, None] * eye2
    c = c.transpose(0, 1, 3, 2, 4).reshape(n_blocks, 2 * size, 2 * size)
    c = 0.5 * (c + jnp.swapaxes(c, -1, -2))
    return c, jnp.linalg.cholesky(c)


def block_source_whitening(cholesky, backprojected, indicator, mask, knot_mask):
    """Conditional of the block's K sources at fixed lens: S_beta = (J^T C^-1 J)^-1 (2K x 2K), beta_bar = S_beta J^T C^-1 b,
    from the block Cholesky factor.  Padded knots get a unit precision, so their conditional is a standard normal.

    L^-1 J and L^-1 b come from one solve (``lj``, ``lb``; ``block_gaussian_loglike`` reuses them).  The Cholesky factor of
    S_beta = P^-1 is taken without inverting P: with E the exchange matrix (reversed order) and E P E = M M^T,
    P^-1 = (E M^-T E)(E M^-T E)^T and E M^-T E is lower triangular with a positive diagonal, so it is chol(P^-1)."""
    mask, knot_mask = jnp.asarray(mask), jnp.asarray(knot_mask)
    n_blocks, size = mask.shape
    b = jnp.where(mask[..., None], jnp.asarray(backprojected), 0.0).reshape(n_blocks, 2 * size)
    solved = solve_triangular(cholesky, jnp.concatenate([jnp.asarray(indicator, dtype=b.dtype), b[..., None]], axis=-1), lower=True)
    lj, lb = solved[..., :-1], solved[..., -1]                                               # (S, 2M, 2K), (S, 2M)
    pad = jnp.repeat(~knot_mask, 2, axis=-1).astype(lj.dtype)                                # (S, 2K)
    n_knot_coordinates = pad.shape[-1]
    precision = jnp.einsum("fki,fkj->fij", lj, lj) + jnp.eye(n_knot_coordinates, dtype=lj.dtype)[None] * pad[:, :, None]
    precision = 0.5 * (precision + jnp.swapaxes(precision, -1, -2))
    reversed_chol = jnp.linalg.cholesky(precision[..., ::-1, ::-1])                          # M, E P E = M M^T
    eye = jnp.broadcast_to(jnp.eye(n_knot_coordinates, dtype=lj.dtype), precision.shape)
    chol = jnp.swapaxes(solve_triangular(reversed_chol, eye, lower=True), -1, -2)[..., ::-1, ::-1]   # E M^-T E = chol(P^-1)
    covariance = jnp.einsum("fik,fjk->fij", chol, chol)
    projected = jnp.einsum("fki,fk->fi", lj, lb)                                             # J^T C^-1 b
    centroid = jnp.einsum("fij,fj->fi", chol, jnp.einsum("fji,fj->fi", chol, projected))
    logdet_precision = 2.0 * jnp.sum(jnp.log(jnp.diagonal(reversed_chol, axis1=-2, axis2=-1)), axis=-1)
    valid = jnp.all(jnp.isfinite(chol), axis=(-2, -1)) & jnp.all(jnp.isfinite(centroid), axis=-1)
    return dict(centroid=centroid, covariance=covariance, cholesky=chol, precision=precision, logdet_precision=logdet_precision, valid=valid,
                lj=lj, lb=lb)


def block_gaussian_loglike(cholesky, whitening, block_source, mask, counts):
    """``gp_image_plane.family_gaussian_loglike`` of the block residual F = b - J beta from the solves the whitening already
    made: L^-1 F = L^-1 b - (L^-1 J) beta, with ``block_source`` the block's knot sources (S, 2K) in knot order."""
    whitened = whitening["lb"] - jnp.einsum("fik,fk->fi", whitening["lj"], block_source)
    whitened = jnp.where(jnp.repeat(jnp.asarray(mask), 2, axis=-1), whitened, 0.0)
    quadratic = jnp.sum(whitened ** 2, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (quadratic + logdet) - jnp.asarray(counts) * _LOG_2PI, quadratic, logdet


def transport_block_sources(offsets, whitening, layout, prior_mean, prior_sigma):
    """Standard-normal offsets per family (F, 2) -> physical sources per family (F, 2) through the block conditionals,
    beta = beta_bar + L_beta u over the block's 2K coordinates, with the correction to the physical Gaussian prior and
    the volume term, as ``smoothed_likelihood.transport_sources`` does per family."""
    return transport_grouped_sources(offsets, [whitening], [layout], prior_mean, prior_sigma)


def transport_grouped_sources(offsets, whitenings, layouts, prior_mean, prior_sigma):
    """``transport_block_sources`` over several groups of blocks (``bucket_layouts``); log_jacobian per block, in the
    order of the groups."""
    offsets = jnp.asarray(offsets)
    source = jnp.zeros_like(offsets)
    block_sources = []
    for whitening, layout in zip(whitenings, layouts):
        knot_family, knot_mask = layout["knot_family"], layout["knot_mask"]
        safe = jnp.where(knot_mask, knot_family, 0)
        u = jnp.where(knot_mask[..., None], offsets[safe], 0.0)                               # (S, K, 2)
        n_blocks, n_knots = knot_mask.shape
        flat = whitening["centroid"] + jnp.einsum("fij,fj->fi", whitening["cholesky"], u.reshape(n_blocks, 2 * n_knots))
        block_sources.append(flat)                                                            # (S, 2K), knot order
        beta = flat.reshape(n_blocks, n_knots, 2)
        source = source.at[layout["real_family"]].set(beta[layout["real_block"], layout["real_knot"]])
    sigma = jnp.asarray(prior_sigma)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    physical_log_prior = -0.5 * jnp.sum(((source - jnp.asarray(prior_mean)) / sigma) ** 2 + 2.0 * jnp.log(sigma) + _LOG_2PI, axis=-1)
    base_log_prior = -0.5 * jnp.sum(offsets ** 2 + _LOG_2PI, axis=-1)
    log_jacobian = jnp.concatenate([-0.5 * w["logdet_precision"] for w in whitenings])     # per block
    correction = jnp.sum(physical_log_prior - base_log_prior) + jnp.sum(log_jacobian)
    valid = jnp.all(jnp.concatenate([w["valid"] for w in whitenings])) & jnp.all(jnp.isfinite(source))
    return dict(source=source, log_correction=correction, log_jacobian=log_jacobian, physical_log_prior=physical_log_prior, valid=valid,
                block_sources=block_sources)


def block_residual(backprojected_slots, source, layout):
    """F = b - beta_knot per slot, (S, M, 2), zero at padded slots."""
    knot_family, knot_of_slot, mask = layout["knot_family"], layout["knot_of_slot"], layout["mask"]
    family_of_slot = jnp.take_along_axis(knot_family, knot_of_slot, axis=1)                   # (S, M)
    beta = source[jnp.where(mask, family_of_slot, 0)]
    return jnp.where(mask[..., None], backprojected_slots - beta, 0.0)


# ----------------------------------------------------------------------------- the GP mean deflection for the image solver
class MeanDeflectionKnee:
    """E[delta_alpha(theta) | F] at arbitrary points for one draw: w_point sum_j w_j K_alpha(theta - theta_j) v_j over the
    images of the point's block, v = C^-1 (b - J beta).  Called with (x, y, family indices), returns the deflection and
    its Jacobian.  The knee curves are evaluated once at the draw's lambda_t
    and interpolated linearly in r (their derivative in r is then piecewise constant, which the solver tolerates)."""

    def __init__(self, layout: dict, tables: dict, coefficients, lam: float, sigma_alpha2: float, weights_per_family, gamma=None,
                 modes: tuple[str, ...] = ()) -> None:
        c_l, c_t = (np.asarray(v) for v in knee_curves(tables, jnp.asarray(float(lam)), None if gamma is None else jnp.asarray(float(gamma))))
        r_grid, xy, mask, weights = tables["r_grid"], layout["xy"], layout["mask"], layout["weights"]
        block_of_family = layout["block_of_family"]
        coef = jnp.asarray(coefficients)
        w_family = jnp.asarray(weights_per_family, dtype=jnp.float64)
        c_l, c_t = jnp.asarray(c_l), jnp.asarray(c_t)

        def field_rows(points, s):
            """Unweighted field covariance between points (P, 2) and the images of block s: (P, M, 2, 2)."""
            d = points[:, None, :] - xy[s][None]
            r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
            n = d / r[..., None]
            cl, ct = jnp.interp(r, r_grid, c_l), jnp.interp(r, r_grid, c_t)
            return sigma_alpha2 * (ct[..., None, None] * jnp.eye(2) + (cl - ct)[..., None, None] * n[..., :, None] * n[..., None, :])

        # u = W v per block (the field-space coefficients).  With the low-order modes removed (project_low_order), the field
        # is r = alpha - Phi c, c the least-squares fit at the images: E[r(theta) | F] = k(theta)^T u' - Phi(theta) g, with
        # u' = (I - P) u and g = R^-1 Q^T K u' (Phi = Q R at the images), so that at the images it equals (I - P) K (I - P) u.
        n_blocks, size = mask.shape
        u = (coef * (mask * weights)[..., None]).reshape(n_blocks, 2 * size)
        if modes:
            q, r_factor, centre = low_order_projector(xy, mask, modes)
            u = u - jnp.einsum("sin,sn->si", q, jnp.einsum("sin,si->sn", q, u))
            k_images = jax.vmap(lambda s: field_rows(xy[s], s))(jnp.arange(n_blocks))                  # (S, M, M, 2, 2)
            k_images = k_images * (mask[:, :, None] & mask[:, None, :])[..., None, None]
            ku = jnp.einsum("simab,smb->sia", k_images, u.reshape(n_blocks, size, 2)).reshape(n_blocks, 2 * size)
            g = jnp.linalg.solve(r_factor, jnp.einsum("sin,si->sn", q, ku)[..., None])[..., 0]          # (S, n)
        u = u.reshape(n_blocks, size, 2)

        def single(point, family):
            s = block_of_family[family]
            block = field_rows(point[None], s)[0] * mask[s][:, None, None]
            value = jnp.einsum("mab,mb->a", block, u[s])
            if modes:
                value = value - low_order_patterns(point[None], centre[s], modes)[0] @ g[s]
            return w_family[family] * value

        value = jax.vmap(single)
        jac = jax.vmap(jax.jacfwd(single, argnums=0))
        self._fields = jax.jit(lambda points, families: (value(points, families), jac(points, families)))

    def __call__(self, x, y, families):
        points = jnp.stack((jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), -1)
        delta, jac = self._fields(points, jnp.asarray(families, dtype=jnp.int32))
        delta, jac = np.asarray(delta), np.asarray(jac)
        return delta[:, 0], delta[:, 1], jac[:, 0, 0], jac[:, 0, 1], jac[:, 1, 0], jac[:, 1, 1]
