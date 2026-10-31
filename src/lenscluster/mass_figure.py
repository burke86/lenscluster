"""The image-catalog overview with surface-density contours and marginal mass profiles (paper figure)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import map_coordinates

from . import cutouts
from .evaluator import Evaluator
from .exact_images import astropy_cosmology
from .truth_recovery import lens_quantities, load_truth_fits

MIN_OVERVIEW_SIZE_ARCSEC = 40.0
CONTOUR_LEVELS_MSUN_PER_KPC2 = np.array([0.5, 1.0, 2.0, 4.0]) * 1.0e9
ARC_DIRECTION_COLOR = "#FFB454"
ARC_STICK_LENGTH_ARCSEC = 3.0  # Fixed display length: direction only, not magnification.


def _stretch_directions(jacobian):
    jacobian = np.asarray(jacobian, dtype=float)
    if not np.isfinite(jacobian).all():
        raise ValueError("Nonfinite exact Jacobian in arc-direction overlay")
    _, singular, vh = np.linalg.svd(jacobian)
    directions = vh[:, -1, :].copy()
    # A stick has no sign; choose one deterministically for its saved coordinates.
    sign = np.where(directions[np.arange(len(directions)), np.argmax(abs(directions), axis=1)] < 0, -1, 1)
    directions *= sign[:, None]
    defined = ~np.isclose(singular[:, 0], singular[:, 1], rtol=1e-10, atol=1e-14)
    directions[~defined] = np.nan
    return directions, singular, defined


def arc_direction_table(evaluator: Evaluator, theta: np.ndarray, catalog: pd.DataFrame) -> pd.DataFrame:
    """Exact image-plane minimum right-singular vectors, matched by stable image label."""
    fields = evaluator.fields(evaluator.physical(theta), force_exact=True, derivative=False)
    directions, singular, defined = _stretch_directions(fields["jacobian"])
    images = evaluator.state.images
    labels = list(map(str, images.image_labels))
    if len(set(labels)) != len(labels) or catalog["image_label"].astype(str).duplicated().any():
        raise ValueError("Arc-direction overlay requires unique image labels")
    result = pd.DataFrame(dict(image_label=labels, e_arc_x=directions[:, 0], e_arc_y=directions[:, 1],
                               s_arc=singular[:, 1], s_perp=singular[:, 0], direction_defined=defined))
    result = result.set_index("image_label").loc[catalog["image_label"].astype(str)].reset_index()
    return result


def extra_arc_direction_table(evaluator: Evaluator, theta: np.ndarray, extra: pd.DataFrame) -> pd.DataFrame:
    """Exact stretch directions at extra-image positions and their own family redshifts."""
    result = extra.copy().reset_index(drop=True)
    for column in ("e_arc_x", "e_arc_y", "s_arc", "s_perp"):
        result[column] = np.nan
    result["direction_defined"] = False
    if result.empty:
        return result
    families = {str(f.family_id): f for f in evaluator.state.family_data}
    physical = evaluator.physical(theta)
    for family_id, rows in result.groupby("family_id", sort=False):
        family = families[str(family_id)]
        packed = evaluator.packed_state_for_z(physical, float(family.z_source))
        _, u = evaluator.total_fields_at(rows[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float), packed)
        directions, singular, defined = _stretch_directions(np.eye(2) - u)
        result.loc[rows.index, ["e_arc_x", "e_arc_y"]] = directions
        result.loc[rows.index, ["s_arc", "s_perp"]] = singular[:, ::-1]
        result.loc[rows.index, "direction_defined"] = defined
    return result


def overview_extent(catalog: pd.DataFrame, extra: pd.DataFrame) -> tuple[float, float, float]:
    """Centre and side of the square covering observed, model and extra images."""
    points = [catalog[["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float), catalog[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float)]
    if extra is not None and not extra.empty:
        points.append(extra[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float))
    stacked = np.vstack(points)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.size == 0:
        return 0.0, 0.0, MIN_OVERVIEW_SIZE_ARCSEC
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    span = float(max(hi - lo))
    size = max(MIN_OVERVIEW_SIZE_ARCSEC, span + 2.0 * max(2.0, 0.15 * span))
    return float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1])), size


def overview_geometry(image: cutouts.BandImage, catalog: pd.DataFrame, extra: pd.DataFrame, reference, resolution: int = 512) -> dict[str, Any]:
    """A square solver-coordinate grid matching the displayed cutout, with a linear WCS."""
    xc, yc, size = overview_extent(catalog, extra)
    center = cutouts.arcsec_to_skycoord(xc, yc, reference)
    window = cutouts.cutout_window(image, center, size)
    if window is None:
        raise ValueError("Image overview is outside the imaging footprint")
    origin = np.array(image.wcs.world_to_pixel(center))
    dx = np.array(image.wcs.world_to_pixel(cutouts.arcsec_to_skycoord(xc + 1, yc, reference))) - origin
    dy = np.array(image.wcs.world_to_pixel(cutouts.arcsec_to_skycoord(xc, yc + 1, reference))) - origin
    if abs(dx[0]) < abs(dx[1]) or abs(dy[1]) < abs(dy[0]):
        raise ValueError("Mass overview requires imaging with x predominantly horizontal and y vertical")
    size = window.npix * image.pixel_scale_arcsec
    xlim = np.array([xc - size / 2, xc + size / 2])[::1 if dx[0] > 0 else -1]
    ylim = np.array([yc - size / 2, yc + size / 2])[::1 if dy[1] > 0 else -1]
    xe, ye = np.linspace(*xlim, resolution + 1), np.linspace(*ylim, resolution + 1)
    x, y = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)
    wcs = WCS(naxis=2)
    cos_dec = np.cos(np.deg2rad(reference[2]))
    dra, ddec = -(xe[1] - xe[0]) / (3600 * cos_dec), (ye[1] - ye[0]) / 3600
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs.wcs.crval = [reference[1], 0]
    wcs.wcs.cdelt = [dra, ddec]
    wcs.wcs.crpix = [1 - (-x[0, 0] / (3600 * cos_dec)) / dra, 1 - (reference[2] + y[0, 0] / 3600) / ddec]
    wcs.wcs.radesys = "ICRS"
    return dict(x=x, y=y, xe=xe, ye=ye, xlim=xlim, ylim=ylim, wcs=wcs, resolution=resolution, window=window,
                reference=reference, center=center, center_xy=(xc, yc), size=size)


def grid_pixels(geometry: dict[str, Any], x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    n = geometry["resolution"]
    return ((np.asarray(x) - geometry["xlim"][0]) / np.diff(geometry["xlim"])[0] * n - 0.5,
            (np.asarray(y) - geometry["ylim"][0]) / np.diff(geometry["ylim"])[0] * n - 0.5)


def aligned_rgb(rgb: np.ndarray, image: cutouts.BandImage, geometry: dict[str, Any], resolution: int | None = None,
                max_resolution: int = 3200) -> np.ndarray:
    """Reproject the displayed RGB cutout onto the solver-aligned grid, sampled at the imaging's own pixel scale (capped)
    so the figure keeps the mosaic's resolution."""
    if resolution is None:
        resolution = int(min(max_resolution, max(512, round(geometry["size"] / image.pixel_scale_arcsec))))
    xedge, yedge = np.linspace(*geometry["xlim"], resolution + 1), np.linspace(*geometry["ylim"], resolution + 1)
    x, y = np.meshgrid((xedge[:-1] + xedge[1:]) / 2, (yedge[:-1] + yedge[1:]) / 2)
    ra, dec = cutouts.arcsec_to_radec(x, y, geometry["reference"])
    px, py = image.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
    window = geometry["window"]
    return np.stack([map_coordinates(rgb[:, :, c], [py - window.y0, px - window.x0], order=1, mode="constant", cval=0) for c in range(rgb.shape[-1])], axis=-1)


def integrate_strips(surface_density: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, bounds) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate a piecewise-constant map (Msun/arcsec^2) over a rectangle, including partial pixels."""
    values = np.asarray(surface_density, dtype=float)
    xe, ye = np.asarray(x_edges), np.asarray(y_edges)
    xmin, xmax, ymin, ymax = bounds
    if values.shape != (len(ye) - 1, len(xe) - 1) or not np.isfinite(values).all():
        raise ValueError("Invalid surface-density grid")
    if np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError("Integration edges must increase")
    if not (xe[0] <= xmin < xmax <= xe[-1] and ye[0] <= ymin < ymax <= ye[-1]):
        raise ValueError("Displayed rectangle must be fully covered by the mass map")
    lo, hi = np.maximum(xe[:-1], xmin), np.minimum(xe[1:], xmax)
    dx = np.maximum(0, hi - lo)
    dy = np.maximum(0, np.minimum(ye[1:], ymax) - np.maximum(ye[:-1], ymin))
    keep = dx > 0
    profile = (values * dy[:, None]).sum(axis=0)
    total = float(np.sum(values * dy[:, None] * dx[None, :]))
    return (lo[keep] + hi[keep]) / 2, dx[keep], profile[keep], total


def convergence_maps(evaluator: Evaluator, theta: np.ndarray, geometry: dict[str, Any], z_source: float, kappa_true_fits: str | Path | None):
    """Model convergence on the overview grid, the truth map resampled to it, and the physical scales."""
    state = evaluator.state
    if z_source <= state.z_lens:
        raise ValueError("Mass map source redshift must exceed lens redshift")
    cosmo = astropy_cosmology(state.cosmo_config)
    dl, ds, dls = (cosmo.angular_diameter_distance(state.z_lens), cosmo.angular_diameter_distance(z_source),
                   cosmo.angular_diameter_distance_z1z2(state.z_lens, z_source))
    sigma_crit = float((const.c ** 2 / (4 * np.pi * const.G) * ds / (dl * dls)).to_value(u.Msun / u.kpc ** 2))
    kpc_per_arcsec = float(dl.to_value(u.kpc) * np.pi / (180 * 3600))
    x, y = geometry["x"], geometry["y"]
    model = lens_quantities(evaluator, theta, z_source, x.ravel(), y.ravel())["kappa"].reshape(x.shape)
    if not np.isfinite(model).all():
        raise ValueError("Model convergence is nonfinite in the displayed rectangle")
    truth = None
    if kappa_true_fits:
        native, wcs = load_truth_fits(kappa_true_fits)
        header = fits.getheader(kappa_true_fits)
        truth_z = header.get("ZSOURCE", header.get("REDSHIFT"))
        if truth_z is not None and not np.isclose(float(truth_z), z_source):
            raise ValueError("Truth convergence source redshift differs from model map")
        ra, dec = cutouts.arcsec_to_radec(x, y, state.reference)
        px, py = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
        px = np.where((px >= -1e-7) & (px <= native.shape[1] - 1 + 1e-7), np.clip(px, 0, native.shape[1] - 1), px)
        py = np.where((py >= -1e-7) & (py <= native.shape[0] - 1 + 1e-7), np.clip(py, 0, native.shape[0] - 1), py)
        truth = map_coordinates(native, [py, px], order=1, mode="constant", cval=np.nan)
        if not np.isfinite(truth).all():
            raise ValueError("Truth convergence does not cover the displayed rectangle with finite values")
    return model, truth, dict(z_lens=float(state.z_lens), z_source=float(z_source), sigma_crit_msun_per_kpc2=sigma_crit,
                              kpc_per_arcsec=kpc_per_arcsec, truth_available=truth is not None,
                              truth_path=str(kappa_true_fits) if kappa_true_fits else None)


def covariance_ellipse_curves(evaluator, theta, catalog):
    """Unit-Mahalanobis image-plane contours, at observed positions and family redshifts."""
    from .covariance_cutouts import best_geometry
    g = best_geometry(evaluator.state, evaluator, theta)
    labels = list(map(str, evaluator.state.images.image_labels))
    if len(set(labels)) != len(labels) or catalog["image_label"].astype(str).duplicated().any():
        raise ValueError("Covariance ellipses require unique image labels")
    indices = {label: i for i, label in enumerate(labels)}
    phase = np.linspace(0., 2*np.pi, 181)
    circle = np.stack([np.cos(phase), np.sin(phase)])
    curves, records = [], []
    for row in catalog.itertuples():
        i = indices[str(row.image_label)]
        if not g["valid"][i]:
            records.append(dict(image_label=str(row.image_label), defined=False))
            continue
        inverse = np.linalg.solve(g["a"][i], np.eye(2))
        covariance = inverse @ g["c"][i] @ inverse.T
        values, vectors = np.linalg.eigh(covariance)
        widths = np.sqrt(values)
        offset = (vectors @ (widths[:, None]*circle)).T
        curves.append(offset + [row.x_obs_arcsec, row.y_obs_arcsec])
        records.append(dict(image_label=str(row.image_label), defined=True,
                            minor_arcsec=float(widths[0]), major_arcsec=float(widths[1]),
                            covariance_xx=float(covariance[0,0]), covariance_xy=float(covariance[0,1]),
                            covariance_yy=float(covariance[1,1])))
    return curves, pd.DataFrame(records)


def overview_axes(fig, ax, images, bands, display, display_image, catalog, extra, geometry, *, arc_directions=None, extra_directions=None,
                  max_side_pixels: int | None = 3200, covariance_curves=None, position_contours=None) -> None:
    """RGB overview with recovery-status circles, scale bar and compass on the solver-aligned grid."""
    rgb = cutouts.rgb_cutout(images, bands, display, geometry["center"], geometry["size"])
    rgb = cutouts.downsample_rgb(aligned_rgb(rgb, display_image, geometry), max_side_pixels)
    n = geometry["resolution"]
    ax.imshow(rgb, origin="lower", extent=(-0.5, n - 0.5, -0.5, n - 0.5), interpolation="none")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_axis_off()
    observed = np.column_stack(grid_pixels(geometry, catalog["x_obs_arcsec"].to_numpy(float), catalog["y_obs_arcsec"].to_numpy(float)))
    model = np.column_stack(grid_pixels(geometry, catalog["x_model_arcsec"].to_numpy(float), catalog["y_model_arcsec"].to_numpy(float)))
    valid = np.isfinite(model).all(axis=1)
    if position_contours is None:
        for obs, pred in zip(observed[valid], model[valid]):
            ax.plot([obs[0], pred[0]], [obs[1], pred[1]], color="white", lw=0.65, alpha=0.9, zorder=4)
        ax.scatter(*observed[valid].T, s=42, facecolors="none", edgecolors="tab:blue", linewidths=0.5, zorder=5)
    else:
        ax.scatter(*observed[valid].T, s=4, marker='x', color='white', linewidths=.2, zorder=10)
        ax.scatter(*observed[~valid].T, s=4, marker='x', color='tab:red', linewidths=.2, zorder=10)
        for level in position_contours['curves']:
            color = 'tab:blue' if level['kind']=='recovered' else 'tab:purple'
            linestyle = '-' if level['probability']<.9 else '--'
            for segment in level['segments']:
                ax.plot(*grid_pixels(geometry,*segment.T), color=color, ls=linestyle, lw=.35, zorder=8)
    if arc_directions is not None and position_contours is None:
        xy = catalog[["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float)
        directions = arc_directions[["e_arc_x", "e_arc_y"]].to_numpy(float)
        if extra_directions is not None and not extra_directions.empty:
            xy = np.vstack((xy, extra_directions[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float)))
            directions = np.vstack((directions, extra_directions[["e_arc_x", "e_arc_y"]].to_numpy(float)))
        for center, direction in zip(xy, directions):
            if not np.isfinite(direction).all():
                continue
            ends = center + np.array([[-.5], [.5]]) * ARC_STICK_LENGTH_ARCSEC * direction
            px, py = grid_pixels(geometry, ends[:, 0], ends[:, 1])
            ax.plot(px, py, color=ARC_DIRECTION_COLOR, lw=.45, alpha=.9,
                    solid_capstyle="round", zorder=8)
    if (~valid).any() and position_contours is None:
        ax.scatter(*observed[~valid].T, s=42, facecolors="none", edgecolors="tab:red", linewidths=0.5, zorder=7)
    if position_contours is None and extra is not None and not extra.empty:
        extras = np.column_stack(grid_pixels(geometry, extra["x_model_arcsec"].to_numpy(float), extra["y_model_arcsec"].to_numpy(float)))
        ax.scatter(*extras.T, s=42, facecolors="none", edgecolors="tab:purple", linewidths=0.5, zorder=6)
    if covariance_curves is not None:
        for curve in covariance_curves:
            ax.plot(*grid_pixels(geometry, curve[:, 0], curve[:, 1]), color="cyan", lw=.4, alpha=.95, zorder=9)
    bar = 10.0 / abs(np.diff(geometry["xlim"])[0]) * n
    x0, y0 = 0.06 * n, 0.055 * n
    ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=2.0, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + 0.015 * n, r"$10^{\prime\prime}$", color="white", fontsize=10, ha="center")
    xc, yc = geometry["center_xy"]
    origin = np.array(grid_pixels(geometry, xc, yc))
    compass = np.array([0.90 * n, 0.09 * n])
    for label, (px, py) in (("N", (xc, yc + 1)), ("E", (xc - 1, yc))):
        direction = np.array(grid_pixels(geometry, px, py)) - origin
        direction /= np.linalg.norm(direction)
        end = compass + direction * 0.055 * n
        ax.annotate("", end, compass, arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
        ax.text(*(end + direction * 0.018 * n), label, color="white", fontsize=9, ha="center", va="center")
    handles = [Line2D([], [], marker="o", ls="none", mfc="none", mec=color, mew=.5, ms=6, label=label)
               for color, label in (("tab:blue", "Recovered image"), ("tab:purple", "Extra image"),
                                    ("tab:red", "Missed image"))]
    if position_contours is not None:
        handles = [Line2D([], [], color='tab:blue', lw=.35, label='Recovered positions'),
                   Line2D([], [], color='tab:purple', lw=.35, label='Tracked extra positions'),
                   Line2D([], [], color='white', marker='x', ls='none', ms=2, mew=.2, label='Catalog: recovered (best fit)'),
                   Line2D([], [], color='tab:red', marker='x', ls='none', ms=2, mew=.2, label='Catalog: missed (best fit)'),
                   Line2D([], [], color='white', lw=.35, label='1σ (68.3%)'),
                   Line2D([], [], color='white', lw=.35, ls='--', label='2σ (95.4%)')]
    if arc_directions is not None and position_contours is None:
        handles.append(Line2D([], [], color=ARC_DIRECTION_COLOR, lw=.45, alpha=.9,
                              label="Stretching direction"))
    if covariance_curves is not None:
        handles.append(Line2D([], [], color="cyan", lw=.4, label="Covariance ellipse"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=7, labelcolor="white", frameon=True,
              facecolor="black", edgecolor="none", framealpha=0.65, borderpad=0.6, handletextpad=0.5, handlelength=1.2)


def render_cluster_mass_figure(evaluator: Evaluator, theta: np.ndarray, catalog: pd.DataFrame, extra: pd.DataFrame, imaging, *,
                               z_source: float, kappa_true_fits: str | Path | None, figure_path: Path, tables_dir: Path, fits_dir: Path, artifact_suffix: str = "", position_contours=None) -> dict[str, Any]:
    """Write the overview-plus-mass-panels figure, the strip profiles and the convergence FITS. Returns provenance."""
    images, bands, display, cluster = imaging
    display_image = images[cutouts.rgb_reference_band(bands)]
    geometry = overview_geometry(display_image, catalog, extra, evaluator.state.reference)
    arc_directions = arc_direction_table(evaluator, theta, catalog)
    extra_directions = extra_arc_direction_table(evaluator, theta, extra if extra is not None else pd.DataFrame())
    model, truth, physical = convergence_maps(evaluator, theta, geometry, z_source, kappa_true_fits)
    sigma_crit = physical["sigma_crit_msun_per_kpc2"]
    sigma_angle = sigma_crit * physical["kpc_per_arcsec"] ** 2
    x, y, xe, ye = (geometry[k] for k in ("x", "y", "xe", "ye"))
    bounds = (float(xe.min()), float(xe.max()), float(ye.min()), float(ye.max()))
    maps = {"model": model, **({"truth": truth} if truth is not None else {})}
    provenance: dict[str, Any] = {"cluster": cluster, "bands": list(bands), **physical}
    profiles, profile, yprofile = {}, pd.DataFrame(), pd.DataFrame()
    for name, values in maps.items():
        integral = values[::1 if ye[-1] > ye[0] else -1, ::1 if xe[-1] > xe[0] else -1]
        centers, widths, px, total = integrate_strips(integral * sigma_angle, np.sort(xe), np.sort(ye), bounds)
        yc, yw, py, ytotal = integrate_strips(integral.T * sigma_angle, np.sort(ye), np.sort(xe), (bounds[2], bounds[3], bounds[0], bounds[1]))
        profile["x_arcsec"], profile["width_arcsec"], profile[f"{name}_dMdx_msun_per_arcsec"] = centers, widths, px
        yprofile["y_arcsec"], yprofile["width_arcsec"], yprofile[f"{name}_dMdy_msun_per_arcsec"] = yc, yw, py
        profiles[name] = (px, py)
        provenance[f"{name}_mass_in_window_msun"] = total
    tables_dir.mkdir(parents=True, exist_ok=True)
    arc_directions.to_csv(tables_dir / f"image_catalog_arc_directions{artifact_suffix}.csv", index=False)
    extra_directions.to_csv(tables_dir / f"image_catalog_extra_arc_directions{artifact_suffix}.csv", index=False)
    provenance["arc_directions"] = {"kernel": "exact", "redshift": "per image", "vector": "minimum right-singular vector",
                                   "stick_length_arcsec": ARC_STICK_LENGTH_ARCSEC, "length_encodes_strength": False,
                                   "defined_count": int(arc_directions["direction_defined"].sum())}
    fits_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(tables_dir / f"image_catalog_mass_profile{artifact_suffix}.csv", index=False)
    yprofile.to_csv(tables_dir / f"image_catalog_mass_profile_y{artifact_suffix}.csv", index=False)
    header = geometry["wcs"].to_header()
    header["ZSOURCE"] = physical["z_source"]
    fits.writeto(fits_dir / f"image_catalog_mass_kappa_model{artifact_suffix}.fits", model, header, overwrite=True)

    covariance_curves = None
    if cutouts.rgb_instrument(bands) == "jwst" and position_contours is None:
        covariance_curves, covariance_table = covariance_ellipse_curves(evaluator, theta, catalog)
        covariance_table.to_csv(tables_dir / f"image_catalog_covariance_ellipses{artifact_suffix}.csv", index=False)
        provenance["covariance_ellipses"] = dict(kernel="exact", redshift="per image", center="observed",
            contour="unit Mahalanobis", display_scale=1., plane="image", defined_count=len(covariance_curves),
            covariance="inverse(A) C inverse(A).T", extra_images="not defined: no observed positional covariance")

    if position_contours is not None:
        provenance['posterior_position_contours'] = dict(contours='KDE 68.3/95.4% highest-density regions',
            conditional_on='successful local recovery', catalog_markers='tiny crosses; best-fit recovery status', stretch_sticks=False, summaries=position_contours['summaries'])
    fig = plt.figure(figsize=(7.1, 7.1), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[3.2, 1], left=0.09, right=0.95, bottom=0.10, top=0.96, wspace=0.025, hspace=0.025)
    ax = fig.add_subplot(grid[0, 0])
    overview_axes(fig, ax, images, bands, display, display_image, catalog, extra, geometry,
                  arc_directions=arc_directions, extra_directions=extra_directions, covariance_curves=covariance_curves, position_contours=position_contours)
    lower, right, corner = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    levels = CONTOUR_LEVELS_MSUN_PER_KPC2

    def available(values):
        return levels[(levels > np.min(values)) & (levels < np.max(values))]

    rx, ry = grid_pixels(geometry, x, y)
    selected = available(model * sigma_crit)
    if len(selected):
        contours = ax.contour(rx, ry, model * sigma_crit / 1e9, levels=selected / 1e9, colors="#D2BC82", linewidths=0.65, alpha=0.85, zorder=2)
        n = geometry["resolution"]
        positions = []
        for segments in contours.allsegs:
            interior = [s[(s[:, 0] > 0.12 * n) & (s[:, 0] < 0.88 * n) & (s[:, 1] > 0.15 * n) & (s[:, 1] < 0.85 * n)] for s in segments]
            longest = max(interior, key=len, default=np.empty((0, 2)))
            if len(longest):
                positions.append(tuple(longest[len(longest) // 3]))
        if positions:
            ax.clabel(contours, manual=positions, fmt="%g", fontsize=7, inline_spacing=3)
    styles = {"model": ("#2878A5", "-", 1.5), "truth": ("0.2", "--", 1.2)}
    for name, values in maps.items():
        color, ls, lw = styles[name]
        px, py = profiles[name]
        lower.plot(centers, px / 1e12, color=color, ls=ls, lw=lw, label=name.title())
        right.plot(py / 1e12, yc, color=color, ls=ls, lw=lw)
        corner_levels = available(values * sigma_crit)
        if len(corner_levels):
            corner.contour(x, y, values * sigma_crit, levels=corner_levels, colors=color, linestyles=ls, linewidths=0.65)
    lower.set_xlim(geometry["xlim"])
    lower.set_ylim(bottom=0)
    right.set_ylim(geometry["ylim"])
    right.set_xlim(left=0)
    corner.set_xlim(lower.get_xlim())
    corner.set_ylim(right.get_ylim())
    corner.set_aspect("equal", adjustable="box")
    image_x, image_y = catalog["x_obs_arcsec"].to_numpy(float), catalog["y_obs_arcsec"].to_numpy(float)
    lower.vlines(image_x, 0, 0.10, transform=lower.get_xaxis_transform(), color="black", linewidth=0.85, zorder=5)
    right.hlines(image_y, 0, 0.10, transform=right.get_yaxis_transform(), color="black", linewidth=0.85, zorder=5)
    lower.set_xlabel(r"$x$ (arcsec)", fontsize=10)
    lower.set_ylabel(r"$dM/dx$ ($10^{12}\,M_\odot\,{\rm arcsec}^{-1}$)", fontsize=9)
    right.set_xlabel(r"$dM/dy$ ($10^{12}\,M_\odot\,{\rm arcsec}^{-1}$)", fontsize=8, labelpad=6)
    right.set_ylabel(r"$y$ (arcsec)", fontsize=9)
    right.xaxis.set_label_position("top")
    right.xaxis.tick_top()
    right.yaxis.set_label_position("right")
    right.yaxis.tick_right()
    corner.set_xlabel(r"$x$ (arcsec)", fontsize=9)
    corner.set_ylabel(r"$y$ (arcsec)", fontsize=9)
    corner.yaxis.set_label_position("right")
    corner.yaxis.tick_right()
    lower.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    lower.xaxis.set_major_locator(MaxNLocator(5, prune="both"))
    lower.yaxis.set_major_locator(MaxNLocator(4))
    right.xaxis.set_major_locator(MaxNLocator(3))
    right.yaxis.set_major_locator(MaxNLocator(5))
    corner.xaxis.set_major_locator(MaxNLocator(3))
    corner.yaxis.set_major_locator(MaxNLocator(3))
    lower.ticklabel_format(axis="y", style="plain", useOffset=False)
    right.ticklabel_format(axis="x", style="plain", useOffset=False)
    for panel in (lower, right, corner):
        panel.grid(False, which="both")
        panel.tick_params(which="both", direction="in", top=True, right=True, labelsize=8, width=0.7, length=3)
        panel.tick_params(which="minor", length=1.5)
        for spine in panel.spines.values():
            spine.set_linewidth(0.7)
    fig.canvas.draw()
    ra, dec = cutouts.arcsec_to_radec(image_x, image_y, geometry["reference"])
    wx, wy = geometry["wcs"].world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
    gx, gy = grid_pixels(geometry, image_x, image_y)
    np.testing.assert_allclose(wx, gx, atol=1e-6, rtol=0)
    np.testing.assert_allclose(wy, gy, atol=1e-6, rtol=0)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    provenance.update(bounds_arcsec=dict(zip(("xmin", "xmax", "ymin", "ymax"), bounds)), contour_levels_msun_per_kpc2=selected.tolist(),
                      grid_shape=list(model.shape))
    (tables_dir / f"image_catalog_mass_provenance{artifact_suffix}.json").write_text(json.dumps(provenance, indent=2, default=float) + "\n")
    return provenance
