"""Truth recovery for simulated clusters: posterior convergence and magnification grids on the truth WCS."""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter, map_coordinates

from .config import TruthRecoveryConfig
from .evaluator import Evaluator
from .model import ModelState
from .utils import progress

QUANTITIES = ("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
APERTURE_CENTER_SMOOTHING_SIGMA_PIX = 2.0


def load_truth_fits(path: str | Path) -> tuple[np.ndarray, WCS]:
    with fits.open(Path(path), memmap=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            image = np.squeeze(np.asarray(data, dtype=float))
            if image.ndim != 2:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(hdu.header).celestial
            if wcs.has_celestial:
                return image, wcs
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def diagnostic_grid(truth_wcs: WCS, native_shape: tuple[int, int], grid_size: int) -> tuple[WCS, tuple[int, int]]:
    """A reduced WCS grid of at most grid_size pixels per side, sharing the native footprint."""
    height, width = int(native_shape[0]), int(native_shape[1])
    size = int(grid_size or 0)
    if size <= 0 or (height <= size and width <= size):
        return truth_wcs.deepcopy(), (height, width)
    scale_x, scale_y = width / size, height / size
    wcs = truth_wcs.deepcopy()
    crpix = np.asarray(wcs.wcs.crpix, dtype=float).copy()
    crpix[0] = 1.0 + (crpix[0] - (0.5 * scale_x - 0.5) - 1.0) / scale_x
    crpix[1] = 1.0 + (crpix[1] - (0.5 * scale_y - 0.5) - 1.0) / scale_y
    wcs.wcs.crpix = crpix
    matrix = np.asarray(truth_wcs.pixel_scale_matrix, dtype=float)
    if matrix.shape == (2, 2):
        wcs.wcs.cd = matrix @ np.diag([scale_x, scale_y])
    else:
        wcs.wcs.cdelt = np.asarray(wcs.wcs.cdelt, dtype=float) * np.asarray([scale_x, scale_y])
    return wcs, (size, size)


def sample_on_grid(values: np.ndarray, source_wcs: WCS, target_wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=float)
    ra, dec = target_wcs.pixel_to_world_values(x, y)
    px, py = source_wcs.world_to_pixel_values(ra, dec)
    return map_coordinates(np.asarray(values, dtype=float), [py, px], order=1, mode="constant", cval=np.nan)


def grid_arcsec(state: ModelState, wcs: WCS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape, dtype=float)
    ra, dec = wcs.pixel_to_world_values(x, y)
    _, ra0, dec0 = state.reference
    cos_dec0 = math.cos(math.radians(dec0))
    delta_ra = (ra0 - np.asarray(ra, dtype=float) + 180.0) % 360.0 - 180.0
    return delta_ra * cos_dec0 * 3600.0, (np.asarray(dec, dtype=float) - dec0) * 3600.0


def lens_quantities(evaluator: Evaluator, theta: np.ndarray, z_source: float, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """kappa, gamma, det A and magnification of the full lens model at the given points."""
    physical = evaluator.physical(jnp.asarray(theta, dtype=jnp.float64))
    packed = evaluator.packed_state_for_z(physical, z_source)
    _, u = evaluator.total_fields_at(np.stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)), -1), packed)
    a = np.eye(2) - u
    a00, a01, a10, a11 = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
    kappa = 1.0 - 0.5 * (a00 + a11)
    gamma1 = 0.5 * (a11 - a00)
    gamma2 = -0.5 * (a01 + a10)
    det = a00 * a11 - a01 * a10
    mu = 1.0 / det
    return {"kappa": kappa, "gamma1": gamma1, "gamma2": gamma2, "detA": det, "mu": mu, "abs_mu": np.abs(mu)}


def smoothed_peak_center(kappa_true: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_pix: float = APERTURE_CENTER_SMOOTHING_SIGMA_PIX) -> dict[str, Any] | None:
    finite = np.isfinite(kappa_true)
    if not finite.any():
        return None
    weights = gaussian_filter(finite.astype(float), sigma=sigma_pix, mode="nearest")
    smoothed = gaussian_filter(np.where(finite, kappa_true, 0.0), sigma=sigma_pix, mode="nearest")
    smoothed = np.where(weights > 0, smoothed / np.where(weights > 0, weights, 1.0), np.nan)
    score = np.where(finite & np.isfinite(smoothed), smoothed, -np.inf)
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    return {"center_mode": "smoothed_truth_kappa_peak", "center_x_arcsec": float(x[row, col]), "center_y_arcsec": float(y[row, col]),
            "center_smoothing_sigma_pix": float(sigma_pix), "center_smoothed_kappa_peak": float(smoothed[row, col])}


def pixel_scale_arcsec(x: np.ndarray, y: np.ndarray) -> float:
    steps = []
    for axis in (0, 1):
        if x.shape[axis] > 1:
            d = np.hypot(np.diff(x, axis=axis), np.diff(y, axis=axis))
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                steps.append(float(np.median(d)))
    return min(steps) if steps else 1.0


def aperture_profile(kappa_true: np.ndarray, kappa_draws: np.ndarray, x: np.ndarray, y: np.ndarray, center: dict[str, Any],
                     n_radii: int = 40) -> pd.DataFrame:
    """Enclosed convergence sums of truth and model draws within circular apertures about the centre."""
    truth = np.asarray(kappa_true, dtype=float).ravel()
    radius = np.hypot(x.ravel() - center["center_x_arcsec"], y.ravel() - center["center_y_arcsec"])
    valid = np.isfinite(truth) & np.isfinite(radius)
    r_max = float(radius[valid].max())
    r_min = min(pixel_scale_arcsec(x, y), r_max)
    radii = np.linspace(r_min, r_max, n_radii)
    draws = np.asarray(kappa_draws, dtype=float).reshape(len(kappa_draws), -1)
    rows = []
    for r in radii:
        inside = valid & (radius <= r)
        truth_sum = float(np.nansum(truth[inside]))
        model_sums = np.nansum(np.where(inside[None, :] & np.isfinite(draws), draws, 0.0), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = model_sums / truth_sum
        q16, q50, q84 = np.nanpercentile(model_sums, [16, 50, 84]) if len(draws) > 1 else (np.nan, model_sums[0], np.nan)
        r16, r50, r84 = np.nanpercentile(ratios, [16, 50, 84]) if len(draws) > 1 else (np.nan, ratios[0], np.nan)
        rows.append({"radius_arcsec": float(r), "pixel_count": int(inside.sum()), "kappa_true_sum": truth_sum,
                     "kappa_model_sum": float(q50), "kappa_model_sum_q16": float(q16), "kappa_model_sum_q84": float(q84),
                     "m2d_ratio": float(r50), "m2d_ratio_q16": float(r16), "m2d_ratio_median": float(r50), "m2d_ratio_q84": float(r84),
                     **{k: center[k] for k in ("center_mode", "center_x_arcsec", "center_y_arcsec", "center_smoothing_sigma_pix", "center_smoothed_kappa_peak")}})
    return pd.DataFrame(rows)


def mass_per_kappa_pixel_msun(state: ModelState, wcs: WCS, z_source: float) -> float:
    """Solar masses per unit convergence per grid pixel for the model cosmology."""
    from astropy import constants as const, units as u
    from astropy.wcs.utils import proj_plane_pixel_area
    from .exact_images import astropy_cosmology
    cosmo = astropy_cosmology(state.cosmo_config)
    dl = cosmo.angular_diameter_distance(state.z_lens)
    ds = cosmo.angular_diameter_distance(z_source)
    dls = cosmo.angular_diameter_distance_z1z2(state.z_lens, z_source)
    sigma_crit = (const.c ** 2 / (4 * np.pi * const.G) * ds / (dl * dls)).to(u.Msun / u.Mpc ** 2)
    pixel_solid_angle = float(proj_plane_pixel_area(wcs.celestial)) * (np.pi / 180.0) ** 2
    return float((sigma_crit * dl ** 2).to_value(u.Msun) * pixel_solid_angle)


def compute_truth_recovery(state: ModelState, evaluator: Evaluator, samples: np.ndarray, config: TruthRecoveryConfig,
                           fits_dir: Path, tables_dir: Path, *, seed: int, log=print) -> dict[str, Any] | None:
    """Write median and 16/84 percentile grids of the lens quantities and the aperture profile. Returns grid data."""
    if not config.kappa_true_fits:
        return None
    z_source = float(config.caustic_source_redshift)
    if z_source <= state.z_lens:
        log(f"[truth] skipped: source redshift {z_source:g} is not behind the lens")
        return None
    native_kappa, native_wcs = load_truth_fits(config.kappa_true_fits)
    wcs, shape = diagnostic_grid(native_wcs, native_kappa.shape, int(config.truth_grid_size))
    truth = {"kappa": sample_on_grid(native_kappa, native_wcs, wcs, shape)}
    if config.gammax_true_fits and config.gammay_true_fits:
        for name, path in (("gamma1", config.gammax_true_fits), ("gamma2", config.gammay_true_fits)):
            values, source_wcs = load_truth_fits(path)
            truth[name] = sample_on_grid(values, source_wcs, wcs, shape)
        truth["detA"] = (1.0 - truth["kappa"]) ** 2 - truth["gamma1"] ** 2 - truth["gamma2"] ** 2
        truth["mu"] = 1.0 / truth["detA"]
        truth["abs_mu"] = np.abs(truth["mu"])
    x, y = grid_arcsec(state, wcs, shape)
    n_draws = min(int(config.posterior_truth_recovery_draws), len(samples))
    rng = np.random.default_rng(seed)
    draws = samples[np.sort(rng.choice(len(samples), size=n_draws, replace=False))] if n_draws else samples[:0]
    median_theta = np.median(samples, axis=0)
    realizations = [median_theta, *draws] if n_draws else [median_theta]
    finite = np.isfinite(x) & np.isfinite(y)
    flat_x, flat_y = x[finite], y[finite]
    grids = {q: np.full((len(realizations),) + shape, np.nan) for q in QUANTITIES}
    for index, theta in enumerate(progress(realizations, desc="truth grids")):
        quantities = lens_quantities(evaluator, theta, z_source, flat_x, flat_y)
        for q in QUANTITIES:
            grids[q][index][finite] = quantities[q]
    log(f"[truth] {len(realizations)} realizations on a {shape[0]}x{shape[1]} grid")
    fits_dir.mkdir(parents=True, exist_ok=True)
    header = wcs.to_header()
    header["ZSOURCE"] = z_source
    products: dict[str, dict[str, np.ndarray]] = {}
    for q in QUANTITIES:
        products[q] = {"median": grids[q][0]}
        fits.writeto(fits_dir / f"truth_recovery_{q}_model_median.fits", grids[q][0], header, overwrite=True)
        if n_draws > 1:
            q16, q84 = np.nanpercentile(grids[q][1:], [16, 84], axis=0)
            products[q].update(q16=q16, q84=q84)
            fits.writeto(fits_dir / f"truth_recovery_{q}_model_q16.fits", q16, header, overwrite=True)
            fits.writeto(fits_dir / f"truth_recovery_{q}_model_q84.fits", q84, header, overwrite=True)
        if q in truth:
            fits.writeto(fits_dir / f"truth_recovery_{q}_true.fits", truth[q], header, overwrite=True)
    center = smoothed_peak_center(truth["kappa"], x, y)
    profile = None
    if center is not None:
        profile = aperture_profile(truth["kappa"], grids["kappa"][1:] if n_draws > 1 else grids["kappa"][:1], x, y, center)
        tables_dir.mkdir(parents=True, exist_ok=True)
        profile.to_csv(tables_dir / "truth_recovery_m2d_aperture_profile.csv", index=False)
    return {"wcs": wcs, "shape": shape, "x": x, "y": y, "truth": truth, "model": products, "profile": profile, "center": center,
            "z_source": z_source, "mass_per_kappa_pixel_msun": mass_per_kappa_pixel_msun(state, wcs, z_source)}
