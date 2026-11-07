"""Exact lensing of the mock truth: caustics, sources placed inside them, their images, and the observed catalog."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from skimage.measure import find_contours

from .. import jax_lensing
from ..inputs import lenstool_shape_to_e1e2
from ..jax_cosmology import dpie_sigma0_factor_from_config
from .config import MockConfig
from .field import field_alpha_hessian, field_alpha_hessian_tiled, field_arrays, image_plane_displacements
from .truth import DPIEComponent, MockTruth

DEDUPLICATION_ARCSEC = 1.0e-3
NEWTON_STEP_CAP_ARCSEC = 5.0


# ----------------------------------------------------------------------------- the truth as a static lens
def lensing_efficiency(truth: MockTruth, z_source: float) -> float:
    """dPIE sigma0 per (v_disp^2 / r_core) for this lens and source redshift: every deflection of the truth scales with it."""
    return float(dpie_sigma0_factor_from_config(truth.z_lens, float(z_source), truth.cosmology))


class TruthLens(NamedTuple):
    """The truth for one source redshift: the dPIE components as a static lens plus the injected mass field (its plane
    waves, or None) times its redshift weight."""
    state: jax_lensing.StaticLensState
    field: tuple | None
    weight: float


def lens_state(truth: MockTruth, z_source: float, components: list[DPIEComponent] | None = None) -> TruthLens:
    """All truth components as a static dPIE lens for one source redshift (lenstronomy kwargs, our kernel), with the
    injected mass field weighted by D_ls / D_s relative to the field's reference redshift."""
    factor = lensing_efficiency(truth, z_source)      # one cosmology evaluation per redshift, not one per component
    kwargs = []
    for c in (truth.components if components is None else components):
        ra, rs = c.core_radius_kpc / truth.kpc_per_arcsec, c.cut_radius_kpc / truth.kpc_per_arcsec
        e1, e2 = lenstool_shape_to_e1e2(c.ellipticite, c.angle_pos)
        kwargs.append({"sigma0": float(c.v_disp) ** 2 * factor / ra, "Ra": float(ra), "Rs": float(rs), "e1": float(e1), "e2": float(e2),
                       "center_x": float(c.x_centre), "center_y": float(c.y_centre)})
    state = jax_lensing.static_lens_state_from_kwargs([jax_lensing.ORIGINAL_DPIE_PROFILE_NAME] * len(kwargs), kwargs)
    if truth.field is None:
        return TruthLens(state, None, 0.0)
    return TruthLens(state, field_arrays(truth.field), factor / lensing_efficiency(truth, float(truth.field["reference_redshift"])))


def _alpha_hessian(lens: TruthLens, x, y):
    """Total deflection and Hessian at the points (traceable): the dPIE lens plus the weighted field."""
    out = jax_lensing.alpha_and_hessian(x, y, lens.state)
    if lens.field is None:
        return out
    return tuple(a + b for a, b in zip(out, field_alpha_hessian(x, y, lens.field, lens.weight)))


def _field_tiled(lens: TruthLens, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...] | None:
    return None if lens.field is None else field_alpha_hessian_tiled(x, y, lens.field, lens.weight)


def field_deflection(lens: TruthLens, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The injected field's deflection (N, 2) at the points for this source redshift (zero without a field)."""
    field = _field_tiled(lens, x, y)
    return np.zeros((np.size(x), 2)) if field is None else np.column_stack(field[:2])


RAY_SHOOT_PAD = 1024


def ray_shoot(lens: TruthLens, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source positions of image-plane points, padded to a multiple of RAY_SHOOT_PAD so few shapes are ever compiled."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    n = max(RAY_SHOOT_PAD, int(math.ceil(x.size / RAY_SHOOT_PAD)) * RAY_SHOOT_PAD)
    xp, yp = np.pad(x, (0, n - x.size), mode="edge"), np.pad(y, (0, n - y.size), mode="edge")
    bx, by = jax_lensing.ray_shooting_tiled(jnp.asarray(xp), jnp.asarray(yp), lens.state, chunk_size=jax_lensing.estimated_alpha_hessian_chunk_size(1.0))
    bx, by = np.asarray(bx, float)[:x.size], np.asarray(by, float)[:x.size]
    field = _field_tiled(lens, x, y)
    return (bx, by) if field is None else (bx - field[0], by - field[1])


def jacobian(lens: TruthLens, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source positions and the lens Jacobian A = I - dalpha/dtheta at image-plane points: (beta (N,2), A (N,2,2), det A)."""
    ax, ay, hxx, hxy, hyx, hyy = (np.asarray(v, dtype=float) for v in _alpha_hessian(
        lens, jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)))
    a = np.stack((np.stack((1.0 - hxx, -hxy), -1), np.stack((-hyx, 1.0 - hyy), -1)), -2)
    beta = np.stack((np.asarray(x, float) - ax, np.asarray(y, float) - ay), -1)
    return beta, a, np.linalg.det(a)


# ----------------------------------------------------------------------------- caustics
@dataclass(frozen=True)
class Caustic:
    index: int
    kind: str                    # "primary" (largest critical curve) or "secondary"
    beta_x: np.ndarray
    beta_y: np.ndarray
    critical_x: np.ndarray
    critical_y: np.ndarray
    caustic_area: float
    critical_area: float


def _polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@dataclass(frozen=True)
class ConvergenceGrid:
    """kappa and |gamma| of the truth on the caustic grid at one source redshift; other redshifts rescale them."""
    axis_x: np.ndarray
    axis_y: np.ndarray
    kappa: np.ndarray
    gamma: np.ndarray


def convergence_grid(lens: TruthLens, config: MockConfig, *, centre: tuple[float, float]) -> ConvergenceGrid:
    window, scale = float(config.caustic_window_arcsec), float(config.caustic_grid_arcsec)
    n = max(16, int(math.ceil(window / scale)) + 1) | 1
    axis_x = np.linspace(centre[0] - window / 2, centre[0] + window / 2, n)
    axis_y = np.linspace(centre[1] - window / 2, centre[1] + window / 2, n)
    xx, yy = np.meshgrid(axis_x, axis_y)
    chunk = jax_lensing.estimated_alpha_hessian_chunk_size(1.0)
    _, _, fxx, fxy, fyx, fyy = jax_lensing.alpha_and_hessian_tiled(jnp.asarray(xx.ravel()), jnp.asarray(yy.ravel()), lens.state, chunk_size=chunk)
    fxx, fxy, fyx, fyy = (np.asarray(v, float) for v in (fxx, fxy, fyx, fyy))
    field = _field_tiled(lens, xx.ravel(), yy.ravel())
    if field is not None:
        fxx, fxy, fyx, fyy = fxx + field[2], fxy + field[3], fyx + field[4], fyy + field[5]
    fxx, fxy, fyx, fyy = (v.reshape(xx.shape) for v in (fxx, fxy, fyx, fyy))
    return ConvergenceGrid(axis_x, axis_y, 0.5 * (fxx + fyy), np.hypot(0.5 * (fxx - fyy), 0.5 * (fxy + fyx)))


def tangential_caustics(lens: TruthLens, config: MockConfig, *, centre: tuple[float, float],
                        grid: ConvergenceGrid | None = None, scale: float = 1.0) -> list[Caustic]:
    """Closed tangential critical curves and their caustics, largest critical curve first.  ``grid`` is a convergence grid
    of the same lens at a reference source redshift and ``scale`` the ratio of lensing efficiencies (single lens plane:
    kappa and gamma scale together), so the expensive grid is computed once per realization."""
    if grid is None:
        grid, scale = convergence_grid(lens, config, centre=centre), 1.0
    lambda_t = 1.0 - float(scale) * (grid.kappa + grid.gamma)
    axis_x, axis_y = grid.axis_x, grid.axis_y
    pixel, step = float(axis_x[1] - axis_x[0]), float(config.caustic_grid_arcsec)
    closed = []
    for vertices in find_contours(lambda_t, 0.0):
        if vertices.shape[0] < 4:
            continue
        cx, cy = axis_x[0] + vertices[:, 1] * pixel, axis_y[0] + vertices[:, 0] * pixel
        touches_edge = (np.min([cx.min() - axis_x[0], axis_x[-1] - cx.max(), cy.min() - axis_y[0], axis_y[-1] - cy.max()]) < 0.5)
        if touches_edge or math.hypot(cx[0] - cx[-1], cy[0] - cy[-1]) > 1.5 * step:
            continue
        closed.append((cx, cy))
    if not closed:
        return []
    beta_x, beta_y = ray_shoot(lens, np.concatenate([cx for cx, _ in closed]), np.concatenate([cy for _, cy in closed]))
    offsets = np.cumsum([0] + [cx.size for cx, _ in closed])
    found = []
    for (cx, cy), start, stop in zip(closed, offsets[:-1], offsets[1:]):
        bx, by = beta_x[start:stop], beta_y[start:stop]
        area = _polygon_area(bx, by)
        if not (np.isfinite(bx).all() and np.isfinite(by).all()) or area < float(config.caustic_min_area_arcsec2):
            continue
        found.append((cx, cy, bx, by, area, _polygon_area(cx, cy)))
    found.sort(key=lambda item: -item[5])
    return [Caustic(i, "primary" if i == 0 else "secondary", bx, by, cx, cy, area, crit_area) for i, (cx, cy, bx, by, area, crit_area) in enumerate(found)]


def sample_in_polygon(px: np.ndarray, py: np.ndarray, rng: np.random.Generator, *, tries: int = 2000) -> tuple[float, float]:
    """A point drawn uniformly inside the closed polygon (px, py)."""
    path = MplPath(np.column_stack((px, py)))
    lo, hi = np.min([px, py], axis=1), np.max([px, py], axis=1)
    for _ in range(tries):
        point = rng.uniform(lo, hi)
        if path.contains_point(point):
            return float(point[0]), float(point[1])
    raise RuntimeError("could not place a point inside the polygon")


def sample_inside(caustic: Caustic, rng: np.random.Generator, *, tries: int = 2000) -> tuple[float, float]:
    """A source drawn uniformly inside the caustic."""
    return sample_in_polygon(caustic.beta_x, caustic.beta_y, rng, tries=tries)


def source_behind(lens: TruthLens, x: float, y: float) -> tuple[float, float]:
    """The source position whose images include the image-plane point (x, y)."""
    bx, by = ray_shoot(lens, np.asarray([x]), np.asarray([y]))
    return float(bx[0]), float(by[0])


# ----------------------------------------------------------------------------- image finding
@dataclass(frozen=True)
class SeedGrid:
    x: np.ndarray
    y: np.ndarray
    beta: np.ndarray             # (n, n, 2)
    pixel: float


def seed_grid(lens: TruthLens, config: MockConfig, *, centre: tuple[float, float], window: float | None = None) -> SeedGrid:
    """Image-plane grid ray-shot once per source redshift; image seeds are its local minima of |beta - source|."""
    window, pixel = float(config.search_window_arcsec if window is None else window), float(config.seed_grid_pixel_arcsec)
    n = int(window / pixel) + 1
    axis_x = np.linspace(centre[0] - window / 2, centre[0] + window / 2, n)
    axis_y = np.linspace(centre[1] - window / 2, centre[1] + window / 2, n)
    xx, yy = np.meshgrid(axis_x, axis_y)
    bx, by = ray_shoot(lens, xx.ravel(), yy.ravel())
    return SeedGrid(xx, yy, np.stack((bx.reshape(xx.shape), by.reshape(xx.shape)), -1), float(axis_x[1] - axis_x[0]))


def _refine(lens: TruthLens, source, x0: np.ndarray, y0: np.ndarray, *, iterations: int, precision: float, keep_order: bool = False):
    """Newton iterations of the lens equation from the seeds; returns converged image positions.  ``source`` is one
    position or one per seed.  With ``keep_order`` every seed is returned (NaN where not converged) in input order."""
    sx, sy = (jnp.asarray(np.asarray(v, dtype=float)) for v in (source[0], source[1]))

    def step(carry, _):
        x, y, active = carry
        ax, ay, hxx, hxy, hyx, hyy = _alpha_hessian(lens, x, y)
        r0, r1 = x - ax - sx, y - ay - sy
        a00, a01, a10, a11 = 1.0 - hxx, -hxy, -hyx, 1.0 - hyy
        det = a00 * a11 - a01 * a10
        ok = active & jnp.isfinite(det) & (jnp.abs(det) > 1.0e-14)
        dx, dy = -(a11 * r0 - a01 * r1) / det, -(-a10 * r0 + a00 * r1) / det
        scale = jnp.minimum(1.0, NEWTON_STEP_CAP_ARCSEC / jnp.maximum(jnp.hypot(dx, dy), 1.0e-12))
        dx, dy = jnp.where(ok, dx * scale, 0.0), jnp.where(ok, dy * scale, 0.0)
        still = ok & (jnp.hypot(r0, r1) > precision) & (jnp.hypot(dx, dy) > 0.1 * precision)
        return (x + dx, y + dy, still), None

    x, y = jnp.asarray(x0, dtype=jnp.float64), jnp.asarray(y0, dtype=jnp.float64)
    (x, y, _), _ = jax.lax.scan(step, (x, y, jnp.isfinite(x) & jnp.isfinite(y)), jnp.arange(int(iterations)))
    ax, ay, *_ = _alpha_hessian(lens, x, y)
    bx, by = x - ax, y - ay
    residual = np.asarray(jnp.hypot(bx - sx, by - sy))
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y) & (residual <= precision)
    if keep_order:
        return np.where(keep, x, np.nan), np.where(keep, y, np.nan)
    return x[keep], y[keep]


def _deduplicate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kept_x, kept_y = [], []
    for xi, yi in zip(x, y):
        if all(math.hypot(xi - kx, yi - ky) > DEDUPLICATION_ARCSEC for kx, ky in zip(kept_x, kept_y)):
            kept_x.append(float(xi)); kept_y.append(float(yi))
    return np.asarray(kept_x), np.asarray(kept_y)


def find_images(lens: TruthLens, grid: SeedGrid, source: tuple[float, float], config: MockConfig, *,
                extra_seeds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All images of a point source: seeds at the local minima of the mapped distance on the grid (within one pixel of
    the source), plus optional extra seeds, refined by Newton iterations and de-duplicated."""
    d2 = np.sum(np.square(grid.beta - np.asarray(source)), axis=-1)
    padded = np.pad(d2, 1, constant_values=np.inf)
    local = np.min([padded[i:i + d2.shape[0], j:j + d2.shape[1]] for i in range(3) for j in range(3)], axis=0)
    seeds = (d2 <= local) & (d2 <= grid.pixel ** 2)
    x0, y0 = grid.x[seeds], grid.y[seeds]
    if extra_seeds is not None:
        x0, y0 = np.concatenate([x0, extra_seeds[0]]), np.concatenate([y0, extra_seeds[1]])
    if x0.size == 0:
        return np.asarray([]), np.asarray([])
    x, y = _refine(lens, source, x0, y0, iterations=config.image_refine_iterations, precision=config.image_precision_arcsec)
    return _deduplicate(x, y)


# ----------------------------------------------------------------------------- families, images, observations
KNOT_ATTEMPTS = 50


def _observe(lens: TruthLens, source, x: np.ndarray, y: np.ndarray, config: MockConfig, rng: np.random.Generator, *,
             source_magnitude: float | None = None) -> dict[str, Any]:
    """Observed positions and magnitudes of the exact images (x, y) of ``source``.  Intrinsic source-plane scatter: each
    image is the exact image of its own displaced source position, re-solved from the true image (so highly magnified
    images move far, as they do in reality); an image that no longer exists for its displaced source keeps the true
    position.  Then astrometric noise in the image plane.  The random draws are those of the mock before knots existed,
    in the same order, so a mock without knots is unchanged."""
    _, _, det = jacobian(lens, x, y)
    field_alpha = field_deflection(lens, x, y)      # the injected mass field's share of the deflection (diagnostics)
    x_obs, y_obs = x.copy(), y.copy()
    if config.sigma_def_arcsec > 0:
        delta = rng.normal(0.0, config.sigma_def_arcsec, (x.size, 2))
        xs, ys = _refine(lens, (source[0] + delta[:, 0], source[1] + delta[:, 1]), x, y,
                         iterations=config.image_refine_iterations, precision=config.image_precision_arcsec, keep_order=True)
        moved = np.isfinite(xs) & np.isfinite(ys)
        x_obs, y_obs = np.where(moved, xs, x), np.where(moved, ys, y)
    x_obs = x_obs + rng.normal(0.0, config.sigma_inst_arcsec, x.size)
    y_obs = y_obs + rng.normal(0.0, config.sigma_inst_arcsec, y.size)
    _, _, det_obs = jacobian(lens, x_obs, y_obs)
    if source_magnitude is None:
        source_magnitude = float(rng.uniform(*config.source_magnitude_range))
    magnitude = source_magnitude - 2.5 * np.log10(np.abs(1.0 / det_obs) + 1.0e-3) + rng.normal(0.0, config.image_magnitude_sigma, x.size)
    return dict(x_obs=x_obs, y_obs=y_obs, det_obs=det_obs, mu_true=1.0 / det, field_alpha=field_alpha,
                source_magnitude=float(source_magnitude), magnitude=magnitude)


def generate_images(truth: MockTruth, config: MockConfig, rng: np.random.Generator, *, log=None) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Sources inside the caustics of the truth, their exact images, and the observed positions.

    Returns the source list and an image table with true and observed positions, magnifications (true position and
    observed position), source-plane scatter applied, magnitudes and the family bookkeeping."""
    centre = (truth.halo.x_centre, truth.halo.y_centre)
    z_lo, z_hi = config.source_redshift_range
    plan = [("primary", z) for z in np.linspace(z_lo, z_hi, int(config.n_primary_families))]
    plan += [("secondary", z) for z in np.linspace(z_lo, z_hi, int(config.n_member_families))] if config.n_member_families else []
    plan += [("central", z) for z in np.linspace(z_lo, z_hi, int(config.n_central_families) + 2)[1:-1]] if config.n_central_families else []
    bcg = (truth.bcg.x_centre, truth.bcg.y_centre)
    log = log or (lambda *_: None)
    states: dict[float, tuple[Any, SeedGrid, list[Caustic]]] = {}
    reference: tuple[float, ConvergenceGrid] | None = None    # (efficiency at the reference redshift, its convergence grid)
    sources, rows, dropped = [], [], []
    n_systems, dropped_knots = 0, 0
    # The knotted families, spread evenly over the primary families (the first entries of the plan).
    n_knotted = int(config.n_knotted_families)
    knotted = set(np.round(np.linspace(0, int(config.n_primary_families) - 1, n_knotted)).astype(int).tolist()) if n_knotted else set()
    log(f"[mock] {len(plan)} families planned over {len({float(z) for _, z in plan})} source redshifts; "
        f"{len(truth.members)} members ({len(truth.modelled_members)} modelled)")
    for family_index, (kind, z) in enumerate(plan, start=1):
        z = float(z)
        if z not in states:
            t0 = time.perf_counter()
            state = lens_state(truth, z)
            efficiency = lensing_efficiency(truth, z)
            if reference is None:
                reference = (efficiency, convergence_grid(state, config, centre=centre))
            caustics = tangential_caustics(state, config, centre=centre, grid=reference[1], scale=efficiency / reference[0])
            # The seed grid must contain every image: size it from the largest critical curve, with a margin.
            extent = max((float(np.max(np.hypot(c.critical_x - centre[0], c.critical_y - centre[1]))) for c in caustics), default=0.0)
            window = max(float(config.search_window_arcsec), 2.0 * (1.6 * extent + 10.0))
            t1 = time.perf_counter()
            grid = seed_grid(state, config, centre=centre, window=window)
            states[z] = (state, grid, caustics)
            log(f"[mock] z={z:.2f}: {len(caustics)} caustics in {t1 - t0:.1f}s (largest critical curve reaches {extent:.1f}\"), "
                f"seed grid {grid.x.shape[0]}x{grid.x.shape[1]} over {window:.0f}\" in {time.perf_counter() - t1:.1f}s")
        state, grid, caustics = states[z]
        candidates = [c for c in caustics if c.kind == "secondary"] if kind == "secondary" else caustics[:1]
        if not candidates:
            dropped.append({"family_index": family_index, "kind": kind, "z_source": z, "reason": "no caustic of this kind"})
            log(f"[mock] family {family_index}/{len(plan)} ({kind}, z={z:.2f}) dropped: no {kind} caustic")
            continue
        accepted = None
        t0 = time.perf_counter()
        for attempt in range(1, int(config.max_source_attempts) + 1):
            weights = np.asarray([c.caustic_area for c in candidates]); weights /= weights.sum()
            caustic = candidates[int(rng.choice(len(candidates), p=weights))]
            if kind == "central":
                # A source behind the BCG: one image within central_radius_arcsec of it (central image or radial arc).
                r, phi = float(config.central_radius_arcsec) * math.sqrt(rng.uniform()), rng.uniform(0.0, 2.0 * math.pi)
                source = source_behind(state, bcg[0] + r * math.cos(phi), bcg[1] + r * math.sin(phi))
            elif kind == "primary" and config.source_placement == "image_plane":
                # Back-project a point drawn uniformly inside the tangential critical curve: images at all radii, not only near it.
                source = source_behind(state, *sample_in_polygon(caustic.critical_x, caustic.critical_y, rng))
            else:
                source = sample_inside(caustic, rng)
            # Secondary caustics are small: also seed from their critical curve so split image pairs are not missed.
            extra = (caustic.critical_x, caustic.critical_y) if kind == "secondary" else None
            x, y = find_images(state, grid, source, config, extra_seeds=extra)
            if x.size >= int(config.min_images_per_family) and (
                    kind != "central" or bool(np.any(np.hypot(x - bcg[0], y - bcg[1]) <= float(config.central_radius_arcsec)))):
                accepted = (source, caustic, x, y)
                break
        if accepted is None:
            dropped.append({"family_index": family_index, "kind": kind, "z_source": z, "reason": "no source with enough images"})
            log(f"[mock] family {family_index}/{len(plan)} ({kind}, z={z:.2f}) dropped after {attempt} sources with < {config.min_images_per_family} images")
            continue
        source, caustic, x, y = accepted
        system_id = str(n_systems + 1)
        n_systems += 1
        log(f"[mock] family {family_index}/{len(plan)} ({kind}, z={z:.2f}): {x.size} images after {attempt} source draw(s) in {time.perf_counter() - t0:.1f}s")
        observed = _observe(state, source, x, y, config, rng)
        source_magnitude = observed["source_magnitude"]
        knots = [("", source, x, y, observed)]
        if kind == "primary" and family_index - 1 in knotted:
            # Further knots of the same galaxy: each a nearby source whose images follow the main images (Newton from
            # them, in their order, so image b of every knot is the same image of the galaxy).  A knot whose image count
            # differs from the main source's (it crossed a caustic) or whose images do not converge is redrawn.
            n_knots = int(rng.integers(int(config.knots_per_family[0]), int(config.knots_per_family[1]) + 1))
            for _ in range(n_knots - 1):
                for _attempt in range(KNOT_ATTEMPTS):
                    radius = math.exp(rng.uniform(*(math.log(v) for v in config.knot_offset_range_arcsec)))
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    knot_source = (source[0] + radius * math.cos(angle), source[1] + radius * math.sin(angle))
                    xk, yk = _refine(state, knot_source, x, y, iterations=config.image_refine_iterations,
                                     precision=config.image_precision_arcsec, keep_order=True)
                    if not (np.all(np.isfinite(xk)) and np.all(np.isfinite(yk)) and _deduplicate(xk, yk)[0].size == x.size):
                        continue
                    if find_images(state, grid, knot_source, config)[0].size != x.size:
                        continue
                    knot_observed = _observe(state, knot_source, xk, yk, config, rng,
                                             source_magnitude=source_magnitude + float(rng.uniform(*config.knot_magnitude_offset_range)))
                    knots.append(("", knot_source, xk, yk, knot_observed))
                    break
                else:
                    dropped_knots += 1
            if len(knots) > 1:
                knots = [(chr(ord("A") + j), *entry[1:]) for j, entry in enumerate(knots)]
                offsets = [math.hypot(k[1][0] - source[0], k[1][1] - source[1]) for k in knots[1:]]
                log(f"[mock] family {family_index}/{len(plan)}: {len(knots)} knots, source-plane offsets "
                    + ", ".join(f"{o:.3f}" for o in offsets) + "\"")
        for knot_id, knot_source, xk, yk, obs in knots:
            family_id = system_id + knot_id
            sources.append({"family_id": family_id, "system_id": system_id, "knot_id": knot_id, "kind": kind, "z_source": z,
                            "beta_x": float(knot_source[0]), "beta_y": float(knot_source[1]),
                            "n_images": int(xk.size), "caustic_index": int(caustic.index), "caustic_area_arcsec2": float(caustic.caustic_area),
                            "source_magnitude": obs["source_magnitude"]})
            for k in range(xk.size):
                rows.append({"family_id": family_id, "system_id": system_id, "knot_id": knot_id,
                             "image_label": f"{family_id}.{chr(ord('a') + k) if k < 26 else k}", "z_source": z,
                             "x_true_arcsec": float(xk[k]), "y_true_arcsec": float(yk[k]),
                             "x_obs_arcsec": float(obs["x_obs"][k]), "y_obs_arcsec": float(obs["y_obs"][k]),
                             "magnification_true": float(obs["mu_true"][k]), "magnification_true_at_obs": float(1.0 / obs["det_obs"][k]),
                             "image_magnitude_obs": float(obs["magnitude"][k]), "image_magnitude_err": float(config.image_magnitude_sigma),
                             "sigma_inst_arcsec": float(config.sigma_inst_arcsec),
                             "x_field_deflection_arcsec": float(obs["field_alpha"][k, 0]), "y_field_deflection_arcsec": float(obs["field_alpha"][k, 1])})
    if not sources:
        raise RuntimeError("the mock produced no multiply imaged family")
    log(f"[mock] {len(sources)} families in {n_systems} systems, {len(rows)} images, {len(dropped)} families dropped"
        + (f", {dropped_knots} knots dropped" if dropped_knots else ""))
    images = pd.DataFrame(rows)
    images["x_image_plane_error_arcsec"] = images["y_image_plane_error_arcsec"] = 0.0
    if truth.image_plane_error is not None:
        # The correlated image-plane displacement: one draw over all images (the solver's kernel couples every family),
        # at the true positions, from its own stream; the truth magnification at the observed position follows.
        shift = image_plane_displacements(np.random.default_rng([truth.seed, 11]), images["x_true_arcsec"].to_numpy(),
                                          images["y_true_arcsec"].to_numpy(), truth.image_plane_error)
        images["x_image_plane_error_arcsec"], images["y_image_plane_error_arcsec"] = shift[:, 0], shift[:, 1]
        images["x_obs_arcsec"] += shift[:, 0]
        images["y_obs_arcsec"] += shift[:, 1]
        for z, rows_z in images.groupby("z_source"):
            _, _, det_obs = jacobian(states[float(z)][0], rows_z["x_obs_arcsec"].to_numpy(), rows_z["y_obs_arcsec"].to_numpy())
            images.loc[rows_z.index, "magnification_true_at_obs"] = 1.0 / det_obs
        log(f"[mock] image-plane displacement: sigma {truth.image_plane_error['sigma_arcsec']:.4f}\", ell {truth.image_plane_error['ell_arcsec']:.2f}\", "
            f"rms {float(np.sqrt(np.mean(np.square(shift)))):.4f}\" per axis")
    if truth.field is not None:
        f = truth.field
        log(f"[mock] mass field: sigma_alpha {f['sigma_alpha_arcsec']:.4f}\" at z {f['reference_redshift']:g}, knee {f['knee_arcsec']:.2f}\", gamma {f['gamma']:.2f}; "
            f"rms deflection at the images {float(np.sqrt(np.mean(np.square(images[['x_field_deflection_arcsec', 'y_field_deflection_arcsec']].to_numpy())))):.4f}\" per axis")
    images.attrs["dropped_families"] = dropped
    images.attrs["caustics_by_z"] = {f"{z:.6f}": [{"index": c.index, "kind": c.kind, "caustic_area_arcsec2": c.caustic_area, "critical_area_arcsec2": c.critical_area,
                                                     "beta_x": c.beta_x.tolist(), "beta_y": c.beta_y.tolist()} for c in caustics]
                                      for z, (_, _, caustics) in states.items()}
    return sources, images
