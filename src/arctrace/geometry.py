"""Angle and coordinate-frame conversions matching the lenscluster solver.

The solver's curved-arc-basis (CAB) likelihood evaluates observed tangent
angles in a tangent-plane offsets frame anchored at a reference point
(ra0, dec0):

    x_arcsec = (ra0 - ra) * cos(dec0) * 3600        (x increases WESTWARD)
    y_arcsec = (dec - dec0) * 3600                  (y increases northward)

(the convention of ``lenscluster.sky.radec_to_offsets_arcsec``, re-derived here
so arctrace does not depend on lenscluster). The predicted
tangent angle is ``atan2(t_y, t_x)`` in this frame and is compared axially
(mod pi).

A sky direction with position angle theta East-of-North has displacement
components dRA*cos(dec) = sin(theta) (East) and dDec = cos(theta) (North),
hence offsets-frame components (x, y) = (-sin(theta), cos(theta)) and

    phi_offset = atan2(cos(theta), -sin(theta)) = wrap_axial(theta + pi/2).
"""

from __future__ import annotations

import math

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


def wrap_axial(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap an axial (mod pi) angle into [0, pi)."""
    wrapped = np.mod(angle_rad, np.pi)
    # np.mod can return exactly pi for tiny negative inputs through rounding.
    wrapped = np.where(wrapped >= np.pi, wrapped - np.pi, wrapped)
    if np.ndim(angle_rad) == 0:
        return float(wrapped)
    return wrapped


def axial_difference(angle_a_rad: float | np.ndarray, angle_b_rad: float | np.ndarray) -> float | np.ndarray:
    """Signed axial difference a - b, in (-pi/2, pi/2]."""
    delta = np.asarray(angle_a_rad, dtype=float) - np.asarray(angle_b_rad, dtype=float)
    result = 0.5 * np.arctan2(np.sin(2.0 * delta), np.cos(2.0 * delta))
    if np.ndim(angle_a_rad) == 0 and np.ndim(angle_b_rad) == 0:
        return float(result)
    return result


def position_angle_to_offset_frame_angle(theta_eofn_rad: float) -> float:
    """Convert a sky position angle (East of North) to the solver offsets-frame angle."""
    return float(wrap_axial(float(theta_eofn_rad) + 0.5 * np.pi))


def radec_to_solver_offsets(
    ra_deg: float | np.ndarray,
    dec_deg: float | np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec (deg) to solver offsets (arcsec); x westward, y northward."""
    cos_dec0 = math.cos(math.radians(float(dec0_deg)))
    if abs(cos_dec0) < 1.0e-8:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    ra_values = np.asarray(ra_deg, dtype=float)
    dec_values = np.asarray(dec_deg, dtype=float)
    x_arcsec = (float(ra0_deg) - ra_values) * cos_dec0 * 3600.0
    y_arcsec = (dec_values - float(dec0_deg)) * 3600.0
    return x_arcsec, y_arcsec


def solver_offsets_to_radec(
    x_arcsec: float | np.ndarray,
    y_arcsec: float | np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`radec_to_solver_offsets`."""
    cos_dec0 = math.cos(math.radians(float(dec0_deg)))
    if abs(cos_dec0) < 1.0e-8:
        raise ValueError("Reference declination is too close to a pole for offset conversion.")
    x_values = np.asarray(x_arcsec, dtype=float)
    y_values = np.asarray(y_arcsec, dtype=float)
    ra_deg = float(ra0_deg) - x_values / (3600.0 * cos_dec0)
    dec_deg = float(dec0_deg) + y_values / 3600.0
    return ra_deg, dec_deg


def pixel_points_to_offset_frame(
    wcs: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates -> solver offsets frame (arcsec) through the WCS."""
    world = wcs.pixel_to_world(np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float))
    world = world.icrs if world.frame.name != "icrs" else world
    return radec_to_solver_offsets(world.ra.deg, world.dec.deg, ra0_deg, dec0_deg)


def offset_frame_point_to_pixel(
    wcs: WCS,
    x_arcsec: float | np.ndarray,
    y_arcsec: float | np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solver offsets frame (arcsec) -> pixel coordinates through the WCS."""
    ra_deg, dec_deg = solver_offsets_to_radec(x_arcsec, y_arcsec, ra0_deg, dec0_deg)
    coord = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    x_pix, y_pix = wcs.world_to_pixel(coord)
    return np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float)


def pixel_direction_position_angle(
    wcs: WCS,
    x_pix: float,
    y_pix: float,
    dx: float,
    dy: float,
    *,
    step_pix: float = 2.0,
) -> float:
    """Position angle (East of North, rad) of a pixel-frame direction at a point."""
    norm = math.hypot(float(dx), float(dy))
    if norm == 0.0:
        raise ValueError("Direction vector must be non-zero.")
    ux = float(dx) / norm
    uy = float(dy) / norm
    p0 = wcs.pixel_to_world(float(x_pix), float(y_pix))
    p1 = wcs.pixel_to_world(float(x_pix) + step_pix * ux, float(y_pix) + step_pix * uy)
    return float(p0.position_angle(p1).to_value(u.rad))


def pixel_tangent_to_offset_angle(
    wcs: WCS,
    x_pix: float,
    y_pix: float,
    tangent_xy: tuple[float, float],
) -> float:
    """Axial offsets-frame angle of a pixel-frame tangent direction at a point."""
    theta = pixel_direction_position_angle(wcs, x_pix, y_pix, tangent_xy[0], tangent_xy[1])
    return position_angle_to_offset_frame_angle(theta)
