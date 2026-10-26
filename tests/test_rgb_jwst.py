"""JWST NIRCam composites: the recipe is chosen from the bands, flux scales carry the pixel-area ratio, and a band on a
coarser grid is resampled onto the reference band's cutout grid."""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from lenscluster import cutouts
from lenscluster.rgb import DEFAULT_JWST_RGB_CHANNEL_WEIGHTS


def _write(path, scale_arcsec, n, value, *, bunit="nJy"):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, -20.0]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    data = np.full((n, n), value, dtype=np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    data[(xx - n / 2) ** 2 + (yy - n / 2) ** 2 < (0.5 / scale_arcsec) ** 2] = 100.0 * value    # a 1" source at the centre
    header = w.to_header()
    header["BUNIT"] = bunit
    header["PHOTFLAM"] = 1.0e-22
    header["PHOTPLAM"] = 15000.0
    fits.writeto(path, data, header, overwrite=True)


def test_jwst_recipe_area_scaling_and_resampling(tmp_path):
    # SW bands at 0.02"/pixel and the LW band at 0.04"/pixel, all 1 nJy/pixel of sky: equal surface brightness per
    # pixel area means the LW band must count 1/4 per pixel on the SW grid.
    paths = {}
    for band, scale in (("F115W", 0.02), ("F150W", 0.02), ("F200W", 0.02), ("F444W", 0.04)):
        paths[band] = tmp_path / f"{band}.fits"
        _write(paths[band], scale, 400 if scale == 0.02 else 200, 1.0)
    bands = ("F115W", "F150W", "F200W", "F444W")
    images = cutouts.load_band_images(paths, bands)
    assert cutouts.rgb_instrument(bands) == "jwst" and cutouts.rgb_reference_band(bands) == "F200W"
    display = cutouts.build_rgb_display(images, bands=bands)
    assert display.channel_weights == {role: {b: w for b, w in ws.items()} for role, ws in DEFAULT_JWST_RGB_CHANNEL_WEIGHTS.items()}
    assert abs(display.band_fluxscales["F444W"] - 0.25) < 1e-6 and abs(display.band_fluxscales["F115W"] - 1.0) < 1e-6
    centre = SkyCoord(10.0 * u.deg, -20.0 * u.deg)
    on_grid = cutouts.extract_band_cutout_on_grid(images["F444W"], images["F200W"], centre, 4.0)
    native = cutouts.extract_band_cutout(images["F200W"], centre, 4.0)
    assert on_grid.shape == native.shape == (200, 200)
    assert np.isfinite(on_grid).all()
    # the central source occupies the same pixels on the shared grid
    assert abs(float((on_grid > 50).mean()) - float((native > 50).mean())) < 0.01
    rgb = cutouts.rgb_cutout(images, bands, display, centre, 4.0)
    assert rgb.shape == (200, 200, 3) and rgb.dtype == np.uint8
    # per-channel stretch with a pedestal: the sky sits at the pedestal grey, the bright source runs to white
    assert display.per_channel_stretch and display.pedestal > 0
    sky = rgb[:20, :20].reshape(-1, 3).mean(axis=0)
    assert np.allclose(sky, 255 * display.pedestal, atol=12), sky
    assert rgb[100, 100].max() >= 235          # the soft highlight roll-off approaches white asymptotically
