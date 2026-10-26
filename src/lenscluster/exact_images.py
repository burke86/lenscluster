"""Exact image prediction for diagnostics: solve the lens equation per family and match images."""
from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel as NPLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver as NPLensEquationSolver
from scipy.optimize import linear_sum_assignment

from .config import ImageDiagnosticsConfig
from .evaluator import Evaluator
from .model import FamilyData, PackedLensState
from .profiles import lenstronomy_kwargs

_TRUST_RADIUS_DEFAULT = 0.3
_LM_DAMPING_RELATIVE = 1.0e-3
_LM_DAMPING_ABSOLUTE = 1.0e-6
_DET_GUARD = 1.0e-30


def astropy_cosmology(cosmo_config: dict[str, Any]):
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM
    h0, om0 = float(cosmo_config["H0"]), float(cosmo_config["Om0"])
    if str(cosmo_config.get("class", "FlatLambdaCDM")) == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=float(cosmo_config.get("w0", -1.0)))
    return FlatLambdaCDM(H0=h0, Om0=om0)


def packed_to_kwargs_lens(packed: PackedLensState) -> list[dict[str, float]]:
    """lenstronomy keyword arguments of every component (packed state with scalar normalizations)."""
    return [lenstronomy_kwargs(packed, idx) for idx in range(int(np.asarray(packed.profile_type).size))]


class ExactImageSolver:
    """Solve the lens equation for every family at a latent vector and match the model images."""

    def __init__(self, evaluator: Evaluator, config: ImageDiagnosticsConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self.models_by_z: dict[float, tuple[NPLensModel, NPLensEquationSolver]] = {}
        self.timing: dict[str, float] = {"exact_solver": 0.0, "global_grid": 0.0, "global_refine": 0.0}
        self.all_components = np.arange(evaluator.n_components, dtype=np.int32)
        self._per_image_fields = None

    # ------------------------------------------------------------------ lens fields at arbitrary points
    def ray_shoot_and_jacobian(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray):
        xy = np.stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)), -1)
        alpha, u = self.evaluator.total_fields_at(xy, packed, self.all_components)
        a = np.eye(2) - u
        return (xy[:, 0] - alpha[:, 0], xy[:, 1] - alpha[:, 1], a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])

    def ray_shoot_and_jacobian_per_image(self, packed_images: PackedLensState, x: np.ndarray, y: np.ndarray):
        """As ray_shoot_and_jacobian, for the N constrained images at once with the per-image packed state (normalizations of
        shape (component, image), so every point carries its own source redshift); one jitted call per iteration."""
        if self._per_image_fields is None:
            components = self.all_components

            def fields(xy, packed_state):
                alpha_rows, u_rows = self.evaluator.component_rows(xy, packed_state, components)
                return jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)
            self._per_image_fields = jax.jit(fields)
        xy = jnp.stack((jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), -1)
        alpha, u = self._per_image_fields(xy, packed_images)
        alpha, a = np.asarray(alpha), np.eye(2) - np.asarray(u)
        return (np.asarray(x, dtype=float) - alpha[:, 0], np.asarray(y, dtype=float) - alpha[:, 1], a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])

    def magnification(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        _, _, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x, y)
        return 1.0 / (a00 * a11 - a01 * a10)

    def lenstronomy_model(self, z_source: float) -> tuple[NPLensModel, NPLensEquationSolver]:
        key = float(z_source)
        if key not in self.models_by_z:
            model = NPLensModel(lens_model_list=self.evaluator.state.lens_model_list, z_lens=self.evaluator.state.z_lens,
                                z_source=key, cosmo=astropy_cosmology(self.evaluator.state.cosmo_config))
            self.models_by_z[key] = (model, NPLensEquationSolver(model))
        return self.models_by_z[key]

    # ------------------------------------------------------------------ local Levenberg--Marquardt solver
    def _lm_step(self, fx, fy, a00, a01, a10, a11):
        finite = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(a00) & np.isfinite(a01) & np.isfinite(a10) & np.isfinite(a11)
        n00 = a00 ** 2 + a10 ** 2
        n01 = a00 * a01 + a10 * a11
        n11 = a01 ** 2 + a11 ** 2
        lam = _LM_DAMPING_ABSOLUTE + _LM_DAMPING_RELATIVE * np.maximum(n00 + n11, 0.0)
        h00, h11, h01 = n00 + lam, n11 + lam, n01
        rhs0 = -(a00 * fx + a10 * fy)
        rhs1 = -(a01 * fx + a11 * fy)
        det = np.maximum(h00 * h11 - h01 ** 2, _DET_GUARD)
        dx = (h11 * rhs0 - h01 * rhs1) / det
        dy = (-h01 * rhs0 + h00 * rhs1) / det
        radius = np.sqrt(dx ** 2 + dy ** 2 + 1.0e-30)
        trust = float(self.config.exact_image_lm_trust_radius_arcsec)
        if trust > 0.0:
            scaled = radius / trust
            scale = np.tanh(scaled) / np.maximum(scaled, 1.0e-12)
            dx, dy = dx * scale, dy * scale
        return dx, dy, finite & np.isfinite(dx) & np.isfinite(dy)

    def _refine(self, packed, source_x, source_y, start_x, start_y, *, shoot=None) -> dict[str, np.ndarray]:
        """Batched Levenberg-Marquardt solve of the lens equation beta(theta) = source from the given starts.

        ``shoot(packed, x, y)`` returns the source positions and the Jacobian A = I - d alpha / d theta at (x, y); the
        default evaluates one scalar-normalized packed state, ``ray_shoot_and_jacobian_per_image`` one with a
        normalization per point.  Every start is iterated until its step and source residual are below the tolerances
        or the iteration budget is spent; starts that produce non-finite values are frozen.  Returns per start: the
        final position, whether it converged, whether it passes the identification tolerance (the Newton step at the
        final position is small, so it sits on a root), the iteration count and the final step and source residual."""
        cfg = self.config
        shoot = self.ray_shoot_and_jacobian if shoot is None else shoot
        x, y = np.array(start_x, dtype=float), np.array(start_y, dtype=float)
        source_x, source_y = np.broadcast_to(source_x, x.shape).astype(float), np.broadcast_to(source_y, y.shape).astype(float)
        alive = np.isfinite(x) & np.isfinite(y)            # still iterating and finite so far
        converged = np.zeros(x.shape, dtype=bool)
        iterations = np.zeros(x.shape, dtype=int)
        source_tol = max(cfg.exact_image_identification_tol_arcsec, cfg.exact_image_precision_limit)
        for iteration in range(1, int(cfg.exact_image_lm_max_iter) + 1):
            bx, by, a00, a01, a10, a11 = shoot(packed, x, y)
            rx, ry = bx - source_x, by - source_y
            dx, dy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
            active = alive & ok & ~converged
            x, y = np.where(active, x + dx, x), np.where(active, y + dy, y)
            iterations = np.where(active, iteration, iterations)
            converged |= active & (np.hypot(dx, dy) <= cfg.exact_image_displacement_tol_arcsec) & (np.hypot(rx, ry) <= source_tol)
            alive &= ok & np.isfinite(x) & np.isfinite(y)
            if not np.any(alive & ~converged):
                break
        # Final Newton step at the end points: its size says how close to a root each start ended.
        bx, by, a00, a01, a10, a11 = shoot(packed, x, y)
        rx, ry = bx - source_x, by - source_y
        cx, cy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
        root_error, source_residual = np.hypot(cx, cy), np.hypot(rx, ry)
        finite = alive & ok & np.isfinite(root_error) & np.isfinite(source_residual)
        identified = finite & (root_error <= cfg.exact_image_identification_tol_arcsec)
        converged |= identified & (root_error <= cfg.exact_image_displacement_tol_arcsec)
        return dict(x=x, y=y, converged=converged, finite=finite, identification_pass=identified,
                    iterations=iterations, final_step_arcsec=root_error, final_source_residual_arcsec=source_residual)

    def solve_local_lm(self, family: FamilyData, packed, source_x, source_y, *, adaptive: bool):
        cfg = self.config
        start_x, start_y = np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)
        result = self._refine(packed, source_x, source_y, start_x, start_y)
        ok = np.asarray(result["identification_pass"], dtype=bool)
        if adaptive and not ok.all():
            radius = min(float(cfg.exact_image_lm_trust_radius_arcsec), max(float(cfg.exact_image_min_distance_arcsec), 1.0e-3))
            for idx in np.flatnonzero(~ok):
                best, best_score, r = None, np.inf, radius
                for _ in range(int(cfg.exact_image_adaptive_max_levels)):
                    offsets = np.asarray([-r, 0.0, r])
                    gx, gy = np.meshgrid(offsets, offsets)
                    refined = self._refine(packed, source_x, source_y, start_x[idx] + gx.ravel(), start_y[idx] + gy.ravel())
                    score = np.where(refined["finite"], refined["final_step_arcsec"] + 0.01 * refined["final_source_residual_arcsec"], np.inf)
                    k = int(np.argmin(score))
                    if np.isfinite(score[k]) and score[k] < best_score:
                        best_score, best = float(score[k]), {key: np.asarray(value)[k] for key, value in refined.items()}
                    if best is not None and bool(best["identification_pass"]):
                        break
                    if not np.isfinite(score).any() and best is None:
                        break                                   # nothing finite from nine starts: a smaller box will not help
                    r *= 0.5
                if best is not None and bool(best["identification_pass"]):
                    for key in result:
                        result[key][idx] = best[key]
            ok = np.asarray(result["identification_pass"], dtype=bool)
        if not ok.any():
            raise RuntimeError("local exact image solver failed identification tolerance")
        return np.asarray(result["x"])[ok], np.asarray(result["y"])[ok]

    def solve_lenstronomy(self, family: FamilyData, packed, source_x, source_y):
        model, solver = self.lenstronomy_model(family.z_source)
        x, y = solver.image_position_from_source(
            source_x, source_y, packed_to_kwargs_lens(packed), solver="lenstronomy",
            min_distance=float(self.config.exact_image_min_distance_arcsec), search_window=family.search_window,
            x_center=family.x_center, y_center=family.y_center, num_iter_max=int(self.config.exact_image_num_iter_max),
            precision_limit=float(self.config.exact_image_precision_limit))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def solve(self, family: FamilyData, packed, source_x: float, source_y: float, *, adaptive: bool | None = None):
        """Model image positions of one family.  ``adaptive`` overrides the configured finder's fallback search; the
        posterior-draw diagnostics switch it off because a draw whose images the local solver cannot find from the
        observed positions is a failed recovery, and the fallback costs minutes per draw on a bad model."""
        started = time.perf_counter()
        try:
            finder = self.config.exact_image_finder
            if finder == "lenstronomy":
                return self.solve_lenstronomy(family, packed, source_x, source_y)
            return self.solve_local_lm(family, packed, source_x, source_y,
                                       adaptive=(finder == "local-lm-adaptive") if adaptive is None else bool(adaptive))
        finally:
            self.timing["exact_solver"] += time.perf_counter() - started

    # ------------------------------------------------------------------ matching
    def match(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> dict[str, Any]:
        x_pred, y_pred = np.asarray(x_pred, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
        n_obs, n_pred = family.n_images, int(min(x_pred.size, y_pred.size))
        matched_x, matched_y = np.full(n_obs, np.nan), np.full(n_obs, np.nan)
        recovered = np.zeros(n_obs, dtype=bool)
        out = dict(produced_image_count=n_pred, recovered_image_count=0, missing_image_count=n_obs, extra_image_count=n_pred,
                   multiplicity_failed=True, multiplicity_failure_reason="no_model_images" if n_pred == 0 else "no_matches",
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=np.asarray([], dtype=float), extra_model_y_arcsec=np.asarray([], dtype=float))
        if n_pred == 0 or n_obs == 0:
            return out
        pred = np.column_stack([x_pred[:n_pred], y_pred[:n_pred]])
        obs = np.column_stack([family.x_obs, family.y_obs])
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(np.isfinite(pred).all(axis=1)[:, None] & np.isfinite(obs).all(axis=1)[None, :], cost, np.inf)
        if not np.isfinite(cost).any():
            out["multiplicity_failure_reason"] = "nonfinite_prediction"
            return out
        tolerance = float(self.config.match_tolerance_arcsec)
        assignment_cost = np.where(np.isfinite(cost), cost, cost[np.isfinite(cost)].max() + tolerance + 1.0)
        rows, cols = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[rows, cols]) & (cost[rows, cols] <= tolerance)
        for p, o in zip(rows[accepted], cols[accepted]):
            matched_x[o], matched_y[o], recovered[o] = x_pred[p], y_pred[p], True
        extra = np.ones(n_pred, dtype=bool)
        extra[rows[accepted]] = False
        n_rec = int(accepted.sum())
        missing, surplus = n_obs - n_rec, n_pred - n_rec
        reason = "" if missing == 0 and surplus == 0 else ("missing_and_extra_model_images" if missing and surplus and n_pred != n_obs
                                                            else "match_tolerance_exceeded" if missing and surplus
                                                            else "missing_model_images" if missing else "extra_model_images")
        out.update(recovered_image_count=n_rec, missing_image_count=missing, extra_image_count=surplus,
                   multiplicity_failed=bool(missing or surplus), multiplicity_failure_reason=reason,
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=x_pred[:n_pred][extra], extra_model_y_arcsec=y_pred[:n_pred][extra])
        return out

    # ------------------------------------------------------------------ per family prediction
    def _details(self, family: FamilyData, x_pred: np.ndarray, y_pred: np.ndarray, mu: np.ndarray, *, error: str | None = None) -> dict[str, Any]:
        details: dict[str, Any] = {"failed": True, "exact_image_rms": np.nan}
        details.update(self.match(np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float), family))
        if error is not None:
            details["multiplicity_failure_reason"] = f"exact_image_prediction_failed: {error}"
        recovered = details["recovered_image_mask"]
        if recovered.any():
            residual = np.sqrt((details["matched_model_x_arcsec"] - family.x_obs) ** 2 + (details["matched_model_y_arcsec"] - family.y_obs) ** 2)
            details["exact_image_rms"] = float(np.sqrt(np.nanmean(residual[recovered] ** 2)))
            details["failed"] = not bool(recovered.all())
        details["magnification"] = mu                                        # model magnification at the observed positions
        details["magnification_at_model"] = np.full(family.n_images, np.nan)  # and at the recovered model positions (filled by the caller)
        return details

    def predict_family(self, family: FamilyData, packed: PackedLensState, source_x: float, source_y: float, *,
                       adaptive: bool | None = None) -> dict[str, Any]:
        """Model images of one family: matched positions, recovery mask, extra images, magnifications."""
        mu = self.magnification(packed, family.x_obs, family.y_obs)
        try:
            x_pred, y_pred = self.solve(family, packed, source_x, source_y, adaptive=adaptive)
        except Exception as error:
            return self._details(family, np.asarray([]), np.asarray([]), mu, error=str(error))
        details = self._details(family, x_pred, y_pred, mu)
        recovered = details["recovered_image_mask"]
        if recovered.any():
            details["magnification_at_model"][recovered] = self.magnification(
                packed, details["matched_model_x_arcsec"][recovered], details["matched_model_y_arcsec"][recovered])
        return details

    # ------------------------------------------------------------------ global image search (batched, JAX)
    def solve_all_global(self, families: list[FamilyData], physical: np.ndarray, sources: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every image of every family at one model.

        Each family's search window (its images' span plus a margin) is ray-shot on a coarse grid, with families at the
        same source redshift sharing one packed state and one field evaluation.  A cell can hold an image when the source
        lies within the cell's mapped reach, |beta - beta_source| below a few pitches times the local stretch ||A||, so
        demagnified images count too.  Seeds are the coarse local minima of that distance, plus, where such cells lie
        close to a critical curve (small |det A|) and image pairs can sit closer than the coarse pitch, the points of a
        fine sub-grid that pass the same reach test on their own fields.  All seeds of all families are refined in one
        batched Levenberg-Marquardt solve with per-seed source redshifts; distinct converged roots inside each window are
        the images.  Returns (x, y) per family, in order."""
        cfg = self.config
        coarse, fine = float(cfg.exact_image_grid_pitch_arcsec), float(cfg.exact_image_fine_pitch_arcsec)
        reach = float(cfg.exact_image_seed_tolerance_pitches)
        physical = jnp.asarray(physical, dtype=jnp.float64)
        sources = np.asarray(sources, dtype=float)
        windows = [(family.x_center, family.y_center, 0.5 * float(family.search_window) + float(cfg.exact_image_window_margin_arcsec)) for family in families]
        by_z: dict[float, list[int]] = {}
        for k, family in enumerate(families):
            by_z.setdefault(float(family.z_source), []).append(k)
        packed_by_z = {z: self.evaluator.packed_state_for_z(physical, z) for z in by_z}
        sub = np.arange(-0.5 * coarse, 0.5 * coarse + 1e-9, fine)
        sub_x, sub_y = (v.ravel() for v in np.meshgrid(sub, sub))

        def mapped(points: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Source positions, |det A| and the stretch ||A||_F at image-plane points for one source redshift."""
            alpha, u = self.evaluator.total_fields_at(points, packed_by_z[z], self.all_components)
            jac = np.eye(2) - u
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            return points - alpha, np.abs(det), np.maximum(1.0, np.sqrt(np.sum(np.square(jac), axis=(1, 2))))

        started = time.perf_counter()
        seed_x, seed_y, seed_family = [], [], []
        self.grid_points = 0
        for z, group in by_z.items():
            grids = []
            for k in group:
                cx, cy, half = windows[k]
                n = int(np.ceil(2.0 * half / coarse)) + 1
                grids.append(np.meshgrid(cx + np.linspace(-half, half, n), cy + np.linspace(-half, half, n)))
            points = np.concatenate([np.column_stack([xx.ravel(), yy.ravel()]) for xx, yy in grids])
            self.grid_points += points.shape[0]
            beta, det, stretch = mapped(points, z)
            fine_points, fine_family = [], []
            offset = 0
            for k, (xx, yy) in zip(group, grids):
                block = slice(offset, offset + xx.size)
                offset += xx.size
                distance = np.sqrt(np.sum(np.square(beta[block] - sources[k]), axis=1)).reshape(xx.shape)
                within = distance <= reach * coarse * stretch[block].reshape(xx.shape)
                padded = np.pad(distance, 1, constant_values=np.inf)
                minima = within & (distance <= np.min([padded[i:i + xx.shape[0], j:j + xx.shape[1]] for i in range(3) for j in range(3)], axis=0))
                seed_x.append(xx[minima]); seed_y.append(yy[minima]); seed_family.append(np.full(int(minima.sum()), k))
                critical = within & (det[block].reshape(xx.shape) <= float(cfg.exact_image_critical_det_threshold))
                if critical.any():
                    fine_points.append(np.column_stack([(xx[critical][:, None] + sub_x[None, :]).ravel(), (yy[critical][:, None] + sub_y[None, :]).ravel()]))
                    fine_family.append(np.full(fine_points[-1].shape[0], k))
            if fine_points:
                # The fine sub-grid is screened on its own fields: one cheap evaluation instead of a solve per point.
                fine_points, fine_family = np.concatenate(fine_points), np.concatenate(fine_family)
                self.grid_points += fine_points.shape[0]
                beta_f, _, stretch_f = mapped(fine_points, z)
                distance_f = np.sqrt(np.sum(np.square(beta_f - sources[fine_family]), axis=1))
                keep = distance_f <= reach * fine * stretch_f
                seed_x.append(fine_points[keep, 0]); seed_y.append(fine_points[keep, 1]); seed_family.append(fine_family[keep])
        self.timing["global_grid"] += time.perf_counter() - started
        seed_x, seed_y = np.concatenate(seed_x), np.concatenate(seed_y)
        seed_family = np.concatenate(seed_family).astype(int)
        self.seed_count = int(seed_x.size)
        solutions = [(np.asarray([], dtype=float), np.asarray([], dtype=float)) for _ in families]
        if seed_x.size == 0:
            return solutions
        started = time.perf_counter()
        factors = jnp.asarray([self.evaluator.state.dpie_sigma0_factor_by_z[float(families[k].z_source)] for k in seed_family], dtype=jnp.float64)
        packed_seeds = self.evaluator.packed_state(physical, factor_per_image=factors)[0]
        result = self._refine(packed_seeds, sources[seed_family, 0], sources[seed_family, 1], seed_x, seed_y, shoot=self.ray_shoot_and_jacobian_per_image)
        self.timing["global_refine"] += time.perf_counter() - started
        # A root must solve the lens equation to the displacement tolerance in both planes: the critical point of a
        # merging fold pair is a near-root that only reaches the looser identification tolerance.  Two roots closer than
        # the minimum separation are one image.
        x, y = np.asarray(result["x"]), np.asarray(result["y"])
        tol = float(cfg.exact_image_displacement_tol_arcsec)
        root = np.asarray(result["finite"]) & (np.asarray(result["final_step_arcsec"]) <= tol) & (np.asarray(result["final_source_residual_arcsec"]) <= tol)
        separation = float(cfg.exact_image_min_separation_arcsec)
        for k, (cx, cy, half) in enumerate(windows):
            keep = root & (seed_family == k) & (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
            px, py, quality = x[keep], y[keep], np.asarray(result["final_step_arcsec"])[keep]
            distinct: list[int] = []
            for i in np.argsort(quality):
                if all(np.hypot(px[i] - px[j], py[i] - py[j]) > separation for j in distinct):
                    distinct.append(int(i))
            solutions[k] = (px[distinct], py[distinct])
        return solutions

    def predict_all_global(self, images, families: list[FamilyData], physical: np.ndarray, sources: np.ndarray) -> list[dict[str, Any]]:
        """Family predictions from the global search: matched positions, extra images, magnifications at the observed and
        the recovered positions.  ``families`` is in the order of ``images.family_ids`` and ``sources`` is (F, 2)."""
        started = time.perf_counter()
        solutions = self.solve_all_global(families, physical, sources)
        self.timing["exact_solver"] += time.perf_counter() - started
        packed_images = self.evaluator.packed_state(jnp.asarray(physical, dtype=jnp.float64))[0]
        x_obs, y_obs = np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)
        family_index = np.asarray(images.family_index_per_image, dtype=int)
        _, _, a00, a01, a10, a11 = self.ray_shoot_and_jacobian_per_image(packed_images, x_obs, y_obs)
        mu_obs = 1.0 / (a00 * a11 - a01 * a10)
        details = [self._details(family, *solutions[f], mu_obs[family_index == f]) for f, family in enumerate(families)]
        self._fill_magnification_at_model(details, packed_images, x_obs, y_obs, family_index)
        return details

    def _fill_magnification_at_model(self, details, packed_images, x_obs, y_obs, family_index) -> None:
        """Model magnification at the recovered positions of every family, one per-image evaluation (unrecovered images
        keep the observed position and are masked)."""
        x_model, y_model, recovered = x_obs.copy(), y_obs.copy(), np.zeros(x_obs.size, dtype=bool)
        for f, d in enumerate(details):
            rows = np.flatnonzero(family_index == f)[d["recovered_image_mask"]]
            x_model[rows], y_model[rows] = d["matched_model_x_arcsec"][d["recovered_image_mask"]], d["matched_model_y_arcsec"][d["recovered_image_mask"]]
            recovered[rows] = True
        _, _, b00, b01, b10, b11 = self.ray_shoot_and_jacobian_per_image(packed_images, x_model, y_model)
        mu_model = np.where(recovered, 1.0 / (b00 * b11 - b01 * b10), np.nan)
        for f, d in enumerate(details):
            d["magnification_at_model"] = mu_model[family_index == f]

    def predict_all_local(self, images, families: list[FamilyData], packed_images: PackedLensState, sources: np.ndarray) -> list[dict[str, Any]]:
        """Every family of one model in a single batched local solve from the observed positions (no fallback search):
        one Levenberg-Marquardt pass over all N images with per-image source redshifts.  ``families`` is in the order of
        ``images.family_ids`` and ``sources`` is (F, 2)."""
        started = time.perf_counter()
        x0, y0 = np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)
        family_index = np.asarray(images.family_index_per_image, dtype=int)
        sources = np.asarray(sources, dtype=float)
        result = self._refine(packed_images, sources[family_index, 0], sources[family_index, 1], x0, y0, shoot=self.ray_shoot_and_jacobian_per_image)
        _, _, a00, a01, a10, a11 = self.ray_shoot_and_jacobian_per_image(packed_images, x0, y0)
        mu = 1.0 / (a00 * a11 - a01 * a10)
        self.timing["exact_solver"] += time.perf_counter() - started
        details = []
        separation = float(self.config.exact_image_min_separation_arcsec)
        for f, family in enumerate(families):
            sel = family_index == f
            ok = np.asarray(result["identification_pass"])[sel]
            px, py = np.asarray(result["x"])[sel][ok], np.asarray(result["y"])[sel][ok]
            # Two observed images that converge onto the same root are one model image, as in the global search.
            distinct = [i for i in range(px.size) if all(np.hypot(px[i] - px[j], py[i] - py[j]) > separation for j in range(i))]
            if distinct:
                details.append(self._details(family, px[distinct], py[distinct], mu[sel]))
            else:
                details.append(self._details(family, np.asarray([]), np.asarray([]), mu[sel], error="local exact image solver failed identification tolerance"))
        self._fill_magnification_at_model(details, packed_images, x0, y0, family_index)
        return details
