"""Sky positions and arcsecond offsets from the model reference point.

The offsets are the flat-sky ones the whole solver uses: x grows to the west (decreasing RA) and y to the north,
x = (ra0 - ra) cos(dec0) 3600, y = (dec - dec0) 3600.  The two functions are exact inverses of each other.
"""
from __future__ import annotations

import math

import numpy as np


def radec_to_offsets_arcsec(ra_deg, dec_deg, ra0_deg: float, dec0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    cos_dec0 = math.cos(math.radians(dec0_deg))
    x_arcsec = (ra0_deg - np.asarray(ra_deg, dtype=float)) * cos_dec0 * 3600.0
    y_arcsec = (np.asarray(dec_deg, dtype=float) - dec0_deg) * 3600.0
    return x_arcsec, y_arcsec


def offsets_to_radec(x_arcsec, y_arcsec, ra0_deg: float, dec0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    cos_dec0 = math.cos(math.radians(dec0_deg))
    if cos_dec0 == 0.0:
        raise ValueError("The reference declination is at a pole; offsets cannot be converted.")
    ra_deg = ra0_deg - np.asarray(x_arcsec, dtype=float) / (3600.0 * cos_dec0)
    dec_deg = dec0_deg + np.asarray(y_arcsec, dtype=float) / 3600.0
    return ra_deg, dec_deg
