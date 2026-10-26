"""Small shared helpers: the dPIE total mass and sky-to-tangent-plane offsets."""
from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .jax_cosmology import kpc_per_arcsec_from_config


def progress(iterable=None, *, total: int | None = None, desc: str = ""):
    """A tqdm bar (stderr, cleared when done) around a loop or used manually with ``update``."""
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN = 4.30091e-6


def log10_dpie_mass_msun(v_disp: Any, cut_radius_kpc: Any) -> np.ndarray:
    """log10 of the dPIE total mass M = (3/2) * pi * sigma^2 * r_cut / G in Msun.

    Eliasdottir et al. 2007 eq. A11 (M_TOT = 2*pi*Sigma0*a*s, equivalently
    lenstronomy PseudoJaffe.mass_tot) expressed with the lenstool/E0 fiducial
    sigma_dPIE — the convention of every v_disp in this pipeline (the amplitude
    packing is E0 = 6*pi*sigma^2*(DLS/DS)/c^2, eq. A24). The 3/2 converts from the
    isothermal-sigma form pi*sigma_iso^2*r_cut/G via sigma_iso^2 = 1.5*sigma_dPIE^2.
    Neglects the exact (1 - r_core^2/r_cut^2) factor, <= 6e-4 for every component
    used here. The single mass definition shared by mock-truth recording, per-run
    recovery tables, and plotting — keep every subhalo-mass comparison on it.
    """
    v_disp_array = np.asarray(v_disp, dtype=float)
    cut_array = np.asarray(cut_radius_kpc, dtype=float)
    mass = (
        1.5
        * np.pi
        * np.square(v_disp_array)
        * np.maximum(cut_array, 0.0)
        / DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0.0, np.log10(mass), np.nan)


def radec_to_offsets(ra, dec, ra0, dec0, z, cosmo):
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    c0 = SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg, frame="icrs")

    # small-angle tangent-plane offsets, with correct sign conventions:
    # +x = East (increasing RA), +y = North (increasing Dec)
    dlon, dlat = c.spherical_offsets_to(c0)  # returns (lon_offset, lat_offset)
    x_arcsec = (dlon.to(u.arcsec)).value    
    y_arcsec = (-dlat.to(u.arcsec)).value    # flip sign so +y is North (Dec increasing)

    if isinstance(cosmo, dict):
        z_values = np.asarray(z, dtype=float)
        arcsec2kpc = np.vectorize(lambda z_item: kpc_per_arcsec_from_config(float(z_item), cosmo))(z_values)
    else:
        arcsec2kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    x_kpc = x_arcsec * arcsec2kpc
    y_kpc = y_arcsec * arcsec2kpc

    return x_kpc, y_kpc, x_arcsec, y_arcsec
