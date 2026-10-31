"""HST and JWST band images: locating, loading, cutouts and RGB display."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates

from .rgb import (CALIBRATED_RGB_MINIMUM_SKY_SIGMA, DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS, DEFAULT_CALIBRATED_RGB_Q,
                  DEFAULT_CALIBRATED_RGB_STRETCH, DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION,
                  DEFAULT_HFF_RGB_BANDS, DEFAULT_HFF_RGB_CHANNEL_FLOOR_SIGMA, DEFAULT_HFF_RGB_CHANNEL_GAINS, DEFAULT_HFF_RGB_CHANNEL_WEIGHTS,
                  DEFAULT_HFF_RGB_HIGHLIGHT_CEILING, DEFAULT_HFF_RGB_HIGHLIGHT_KNEE, DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS,
                  DEFAULT_HFF_RGB_MINIMUM, DEFAULT_HFF_RGB_Q, DEFAULT_HFF_RGB_REFERENCE_BAND, DEFAULT_HFF_RGB_STRETCH,
                  DEFAULT_JWST_RGB_BANDS, DEFAULT_JWST_RGB_CHANNEL_GAINS, DEFAULT_JWST_RGB_CHANNEL_WEIGHTS, DEFAULT_JWST_RGB_MINIMUM,
                  DEFAULT_JWST_RGB_CHROMA, DEFAULT_JWST_RGB_GAMMA, DEFAULT_JWST_RGB_PEDESTAL, DEFAULT_JWST_RGB_Q, DEFAULT_JWST_RGB_REFERENCE_BAND, DEFAULT_JWST_RGB_STRETCH, DEFAULT_JWST_RGB_HUE_MIX, DEFAULT_JWST_RGB_UNSHARP_SIGMA_ARCSEC, DEFAULT_JWST_RGB_UNSHARP_AMOUNT, DEFAULT_JWST_RGB_SHADOW_LUMINANCE, DEFAULT_JWST_RGB_HIGHLIGHT_KNEE,
                  DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION, DEFAULT_RGB_MINIMUM, CalibratedRGBDisplayConfig,
                  RGBDisplayConfig, compute_fnu_band_fluxscales, make_natural_rgb)

IMAGE_SCALE_CHOICES = ("auto", "20mas", "30mas", "40mas", "60mas")
CLUSTER_IMAGE_TOKENS = {
    "a2744": ("a2744", "abell2744"), "a370": ("a370", "abell370"), "ares": ("ares",),
    "as1063": ("as1063", "abells1063", "rxcj2248", "rxj2248"), "hera": ("hera",), "m0416": ("m0416", "macs0416"),
    "m0717": ("m0717", "macs0717"), "m1149": ("m1149", "macs1149"),
}


@dataclass(frozen=True)
class BandImage:
    band: str
    path: Path
    hdu_index: int
    shape: tuple[int, int]
    wcs: WCS
    pixel_scale_arcsec: float
    photflam: float | None = None
    photplam: float | None = None
    background: float = 0.0
    background_sigma: float = 0.0


@dataclass(frozen=True)
class CutoutWindow:
    npix: int
    x0: int
    y0: int
    src_x0: int
    src_y0: int
    src_x1: int
    src_y1: int
    dst_x0: int
    dst_y0: int


def arcsec_to_skycoord(x_arcsec: float, y_arcsec: float, reference: tuple[int, float, float]) -> SkyCoord:
    _, ra0, dec0 = reference
    cos_dec0 = math.cos(math.radians(float(dec0)))
    return SkyCoord(ra=(float(ra0) - float(x_arcsec) / (3600.0 * cos_dec0)) * u.deg, dec=(float(dec0) + float(y_arcsec) / 3600.0) * u.deg, frame="icrs")


def arcsec_to_radec(x_arcsec: Any, y_arcsec: Any, reference: tuple[int, float, float]) -> tuple[np.ndarray, np.ndarray]:
    _, ra0, dec0 = reference
    cos_dec0 = math.cos(math.radians(float(dec0)))
    x, y = np.asarray(x_arcsec, dtype=float), np.asarray(y_arcsec, dtype=float)
    return float(ra0) - x / (3600.0 * cos_dec0), float(dec0) + y / 3600.0


def cluster_from_config(model_config: Any, run_name: str = "") -> str:
    """Identify the imaging cluster from the catalog paths of the model configuration."""
    paths = [str(getattr(getattr(model_config, "image_constraints", None), "catalog_path", "") or "")]
    paths.extend(str(getattr(p, "catalog_path", "") or "") for p in getattr(model_config, "member_populations", ()) or ())
    for text in (" ".join(paths).lower(), str(run_name).lower()):
        for cluster, tokens in CLUSTER_IMAGE_TOKENS.items():
            if any(token in text for token in tokens):
                return cluster
    return str(run_name).split("_", 1)[0].lower() or "cluster"


def imaging_variants(config):
    """Saved imaging plus the standard JWST view for M0416 and AS1063."""
    from dataclasses import replace
    from .config import RGBDisplayConfig as DisplayConfig

    primary = config.image_catalog
    variants = [primary]
    cluster = cluster_from_config(config.model, config.paths.run_name)
    if cluster in {"m0416", "as1063"} and not any(
        str(band).upper() in DEFAULT_JWST_RGB_BANDS for band in (primary.bands or ())
    ):
        variants.append(replace(primary, image_dir="data/JWST_Images", image_scale="auto",
                                bands=tuple(DEFAULT_JWST_RGB_BANDS), rgb=DisplayConfig(),
                                figure_suffix="_jwst"))
    return variants


def _fits_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".fits", ".fit", ".fits.gz"))


def find_rgb_band_paths(image_dir: Path, *, cluster: str, bands: Sequence[str], image_scale: str = "auto") -> dict[str, Path]:
    if image_scale not in IMAGE_SCALE_CHOICES:
        raise ValueError(f"Unsupported image scale {image_scale!r}. Valid choices: {IMAGE_SCALE_CHOICES}.")
    root = Path(image_dir)
    tokens = tuple(t.lower() for t in CLUSTER_IMAGE_TOKENS.get(cluster, (cluster,)))
    root_has_token = any(t in str(root).lower() for t in tokens)
    files = [p for p in root.rglob("*") if p.is_file() and _fits_like(p)] if root.exists() else []
    result, missing = {}, []
    for band in bands:
        token = str(band).lower()
        candidates = [p for p in files if token in str(p.relative_to(root)).lower()
                      and (root_has_token or any(t in str(p.relative_to(root)).lower() for t in tokens))]
        if image_scale != "auto":
            preferred = [p for p in candidates if image_scale in str(p.relative_to(root)).lower()]
            candidates = preferred or candidates
        if not candidates:
            missing.append(str(band))
            continue
        # Prefer drizzled science mosaics, plain FITS over gzip (memory-mappable), and the unsubtracted image over a
        # background- or galaxy-subtracted product.
        result[str(band)] = sorted(candidates, key=lambda p: (0 if "_drz" in p.name.lower() else 1, 1 if p.name.lower().endswith(".gz") else 0,
                                                              1 if "bgsub" in p.name.lower() else 0, len(p.parts), str(p)))[0]
    if missing:
        raise FileNotFoundError(f"Missing FITS image(s) for cluster {cluster!r} under {root}: {', '.join(missing)}")
    return result


def _header_keyword(headers: Sequence[fits.Header], keyword: str) -> float | None:
    for header in headers:
        try:
            value = float(header.get(keyword))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return None


def _global_background(data: Any, *, max_samples: int = 400_000) -> tuple[float, float]:
    """Sigma-clipped sky median and robust sigma from a strided subsample; exact zeros are padding."""
    ny, nx = data.shape
    stride = max(1, int(math.sqrt(max(1, (ny * nx) / float(max_samples)))))
    sample = np.asarray(data[::stride, ::stride], dtype=np.float32)
    values = sample[np.isfinite(sample) & (sample != 0.0)]
    if values.size == 0:
        return 0.0, 0.0
    sigma = 0.0
    for _ in range(5):
        median = float(np.median(values))
        sigma = 1.4826 * float(np.median(np.abs(values - median)))
        if sigma <= 0.0:
            break
        kept = values[np.abs(values - median) < 3.0 * sigma]
        if kept.size in (values.size, 0):
            break
        values = kept
    return float(np.median(values)), float(sigma)


def load_band_image(band: str, path: Path) -> BandImage:
    with fits.open(path, memmap=True) as hdul:
        primary = hdul[0].header
        for index, hdu in enumerate(hdul):
            if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
                continue
            wcs = WCS(hdu.header)
            if not wcs.has_celestial:
                continue
            celestial = wcs.celestial
            scale = float(np.nanmean(np.abs(np.asarray(proj_plane_pixel_scales(celestial), dtype=float) * 3600.0)))
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            background, sigma = _global_background(hdu.data)
            ny, nx = hdu.data.shape
            return BandImage(band=str(band), path=Path(path), hdu_index=index, shape=(int(ny), int(nx)), wcs=celestial,
                             pixel_scale_arcsec=scale, photflam=_header_keyword((hdu.header, primary), "PHOTFLAM"),
                             photplam=_header_keyword((hdu.header, primary), "PHOTPLAM"), background=background, background_sigma=sigma)
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def load_band_images(paths_by_band: Mapping[str, Path], bands: Sequence[str]) -> dict[str, BandImage]:
    return {str(b): load_band_image(str(b), paths_by_band[str(b)]) for b in bands}


def cutout_npixels(image: BandImage, cutout_size_arcsec: float) -> int:
    if cutout_size_arcsec <= 0.0:
        raise ValueError("cutout_size_arcsec must be positive.")
    return max(1, int(math.ceil(float(cutout_size_arcsec) / image.pixel_scale_arcsec)))


def _clamped_origin(raw: int, npix: int, size: int) -> int:
    return int(np.clip(raw, 0, size - npix)) if size >= npix else int(np.clip(raw, size - npix, 0))


def cutout_window(image: BandImage, coord: SkyCoord, cutout_size_arcsec: float) -> CutoutWindow | None:
    npix = cutout_npixels(image, cutout_size_arcsec)
    cx, cy = image.wcs.world_to_pixel(coord)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    half = npix // 2
    raw_x0, raw_y0 = int(round(float(cx))) - half, int(round(float(cy))) - half
    ny, nx = image.shape
    if raw_x0 >= nx or raw_x0 + npix <= 0 or raw_y0 >= ny or raw_y0 + npix <= 0:
        return None
    x0, y0 = _clamped_origin(raw_x0, npix, nx), _clamped_origin(raw_y0, npix, ny)
    src_x0, src_y0, src_x1, src_y1 = max(0, x0), max(0, y0), min(nx, x0 + npix), min(ny, y0 + npix)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return None
    return CutoutWindow(npix, x0, y0, src_x0, src_y0, src_x1, src_y1, src_x0 - x0, src_y0 - y0)


def extract_band_cutout(image: BandImage, coord: SkyCoord, cutout_size_arcsec: float) -> np.ndarray:
    npix = cutout_npixels(image, cutout_size_arcsec)
    cutout = np.full((npix, npix), np.nan, dtype=np.float32)
    window = cutout_window(image, coord, cutout_size_arcsec)
    if window is None:
        return cutout
    with fits.open(image.path, memmap=True) as hdul:
        values = np.asarray(hdul[image.hdu_index].section[window.src_y0:window.src_y1, window.src_x0:window.src_x1], dtype=np.float32)
    cutout[window.dst_y0:window.dst_y0 + values.shape[0], window.dst_x0:window.dst_x0 + values.shape[1]] = values
    return cutout


def rgb_instrument(bands: Sequence[str]) -> str:
    """"jwst" when any NIRCam band is present, else "hst"."""
    return "jwst" if any(str(b) in DEFAULT_JWST_RGB_BANDS for b in bands) else "hst"


def rgb_reference_band(bands: Sequence[str]) -> str:
    """The band whose pixel grid and flux scale the display is built on."""
    keys = [str(b) for b in bands]
    preferred = DEFAULT_JWST_RGB_REFERENCE_BAND if rgb_instrument(keys) == "jwst" else DEFAULT_HFF_RGB_REFERENCE_BAND
    return preferred if preferred in keys else keys[-1]


def _channel_weights(bands: Sequence[str]) -> dict[str, dict[str, float]]:
    """The instrument's channel recipe restricted to the bands at hand."""
    available = {str(b) for b in bands}
    table = DEFAULT_JWST_RGB_CHANNEL_WEIGHTS if rgb_instrument(bands) == "jwst" else DEFAULT_HFF_RGB_CHANNEL_WEIGHTS
    out = {}
    for role, weights in table.items():
        selected = {b: float(w) for b, w in weights.items() if b in available}
        if selected:
            out[role] = selected
    return out


def build_rgb_display(band_images: Mapping[str, BandImage], *, bands: Sequence[str], q: float | None = None, stretch: float | None = None,
                      channel_gains: Mapping[str, float] | None = None, minimum: float | None = None) -> RGBDisplayConfig:
    """Calibrated Lupton display: the weighted multi-band recipe (HST Frontier Fields or JWST NIRCam) when more than three
    bands are given, else a three-band composite.  Band flux scales put every band on the reference band's f_nu scale
    per unit sky area, so bands with different pixel sizes combine correctly."""
    keys = [str(b) for b in bands]
    instrument = rgb_instrument(keys)
    weights = _channel_weights(keys)
    weighted = len(keys) > 3 and all(weights.get(role) for role in ("blue", "green", "red"))
    reference = rgb_reference_band(keys)
    fluxscales = compute_fnu_band_fluxscales({b: getattr(band_images.get(b), "photflam", None) for b in keys},
                                             {b: getattr(band_images.get(b), "photplam", None) for b in keys}, reference_band=reference)
    reference_pixel_scale = float(getattr(band_images.get(reference), "pixel_scale_arcsec", 0.0) or 0.0)
    reference_area = (reference_pixel_scale or 1.0) ** 2
    for b in keys:                                                        # per-pixel flux -> the reference band's pixel area
        area = float(getattr(band_images.get(b), "pixel_scale_arcsec", 0.0) or 0.0) ** 2
        if area > 0.0 and reference_area > 0.0:
            fluxscales[b] = float(fluxscales.get(b, 1.0)) * reference_area / area
    backgrounds = {b: float(getattr(band_images.get(b), "background", 0.0) or 0.0) for b in keys}
    if weighted and instrument == "jwst":
        defaults = dict(q=DEFAULT_JWST_RGB_Q, stretch=DEFAULT_JWST_RGB_STRETCH, gains=DEFAULT_JWST_RGB_CHANNEL_GAINS, minimum=DEFAULT_JWST_RGB_MINIMUM)
    elif weighted:
        defaults = dict(q=DEFAULT_HFF_RGB_Q, stretch=DEFAULT_HFF_RGB_STRETCH, gains=DEFAULT_HFF_RGB_CHANNEL_GAINS, minimum=DEFAULT_HFF_RGB_MINIMUM)
    else:
        sigmas = [float(getattr(band_images.get(b), "background_sigma", 0.0) or 0.0) * float(fluxscales.get(b, 1.0)) for b in keys]
        sigmas = [v for v in sigmas if v > 0]
        defaults = dict(q=DEFAULT_CALIBRATED_RGB_Q, stretch=DEFAULT_CALIBRATED_RGB_STRETCH, gains=DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS,
                        minimum=CALIBRATED_RGB_MINIMUM_SKY_SIGMA * float(np.median(sigmas)) if sigmas else DEFAULT_RGB_MINIMUM)
    return CalibratedRGBDisplayConfig(
        q=float(defaults["q"] if q is None else q), stretch=float(defaults["stretch"] if stretch is None else stretch),
        channel_gains=dict(defaults["gains"] if channel_gains is None else channel_gains),
        minimum=float(defaults["minimum"] if minimum is None else minimum), band_backgrounds=backgrounds, band_fluxscales=fluxscales,
        warm_highlight_desaturation=DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION if weighted else DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION,
        channel_weights=weights if weighted else {}, highlight_ceiling=DEFAULT_HFF_RGB_HIGHLIGHT_CEILING if weighted else 1.0, highlight_softness=DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS if weighted else 0.44,
        local_background=weighted and instrument == "hst", channel_floor_sigma=DEFAULT_HFF_RGB_CHANNEL_FLOOR_SIGMA if (weighted and instrument == "hst") else 0.0,
        **(_jwst_display_fields(reference_pixel_scale) if (weighted and instrument == "jwst") else dict(highlight_knee=DEFAULT_HFF_RGB_HIGHLIGHT_KNEE if weighted else 1.0)))


def _jwst_display_fields(reference_pixel_scale_arcsec: float) -> dict:
    """The per-channel stretch fitted to the STScI release look (see the DEFAULT_JWST_RGB_* constants)."""
    pixel = float(reference_pixel_scale_arcsec) if reference_pixel_scale_arcsec > 0.0 else 1.0
    return dict(per_channel_stretch=True, pedestal=DEFAULT_JWST_RGB_PEDESTAL, gamma=DEFAULT_JWST_RGB_GAMMA, chroma=DEFAULT_JWST_RGB_CHROMA,
                hue_mix=DEFAULT_JWST_RGB_HUE_MIX, unsharp_sigma_pixels=DEFAULT_JWST_RGB_UNSHARP_SIGMA_ARCSEC / pixel,
                unsharp_amount=DEFAULT_JWST_RGB_UNSHARP_AMOUNT, shadow_luminance=DEFAULT_JWST_RGB_SHADOW_LUMINANCE,
                highlight_knee=DEFAULT_JWST_RGB_HIGHLIGHT_KNEE)


def extract_band_cutout_on_grid(image: BandImage, reference: BandImage, coord: SkyCoord, cutout_size_arcsec: float) -> np.ndarray:
    """A band's cutout resampled (bilinear, through both WCSs) onto the reference band's cutout grid, so bands with
    different pixel sizes share one pixel grid.  Values stay per source pixel; the flux scales convert the area."""
    if image.path == reference.path and image.hdu_index == reference.hdu_index:
        return extract_band_cutout(image, coord, cutout_size_arcsec)
    window = cutout_window(reference, coord, cutout_size_arcsec)
    npix = cutout_npixels(reference, cutout_size_arcsec)
    if window is None:
        return np.full((npix, npix), np.nan, dtype=np.float32)
    # Sky positions of the reference cutout's pixel centres -> pixel coordinates in the other band.
    yy, xx = np.mgrid[0:npix, 0:npix]
    sky = reference.wcs.pixel_to_world(xx + window.x0, yy + window.y0)
    px, py = image.wcs.world_to_pixel(sky)
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    # Read the covering section of the other band once, with a one-pixel margin for the interpolation.
    x0, x1 = int(np.floor(np.nanmin(px))) - 1, int(np.ceil(np.nanmax(px))) + 2
    y0, y1 = int(np.floor(np.nanmin(py))) - 1, int(np.ceil(np.nanmax(py))) + 2
    ny, nx = image.shape
    x0, x1, y0, y1 = max(0, x0), min(nx, x1), max(0, y0), min(ny, y1)
    out = np.full((npix, npix), np.nan, dtype=np.float32)
    if x0 >= x1 or y0 >= y1:
        return out
    with fits.open(image.path, memmap=True) as hdul:
        section = np.asarray(hdul[image.hdu_index].section[y0:y1, x0:x1], dtype=np.float32)
    inside = (px >= x0) & (px <= x1 - 1) & (py >= y0) & (py <= y1 - 1)
    filled = np.where(np.isfinite(section), section, 0.0)
    out[inside] = map_coordinates(filled, [py[inside] - y0, px[inside] - x0], order=1, mode="nearest")
    return out


def rgb_cutout(band_images: Mapping[str, BandImage], bands: Sequence[str], display: RGBDisplayConfig, coord: SkyCoord, cutout_size_arcsec: float) -> np.ndarray:
    reference = band_images[rgb_reference_band(bands)]
    cutouts = {str(b): extract_band_cutout_on_grid(band_images[str(b)], reference, coord, cutout_size_arcsec) for b in bands}
    return make_natural_rgb(cutouts, bands=bands, display=display)


def downsample_rgb(rgb: np.ndarray, max_side_pixels: int | None) -> np.ndarray:
    image = np.asarray(rgb)
    if max_side_pixels is None or max(image.shape[:2]) <= int(max_side_pixels):
        return image
    scale = float(max_side_pixels) / float(max(image.shape[:2]))
    ys = np.rint(np.linspace(0, image.shape[0] - 1, max(1, int(image.shape[0] * scale)))).astype(int)
    xs = np.rint(np.linspace(0, image.shape[1] - 1, max(1, int(image.shape[1] * scale)))).astype(int)
    return image[np.ix_(ys, xs)]


def load_cluster_imaging(config, model_config: Any, run_name: str) -> tuple[dict[str, BandImage], tuple[str, ...], RGBDisplayConfig, str]:
    """Band images, bands, RGB display and cluster name from an ImageCatalogCutoutConfig."""
    cluster = cluster_from_config(model_config, run_name)
    bands = tuple(config.bands) if config.bands else tuple(DEFAULT_HFF_RGB_BANDS)
    paths = find_rgb_band_paths(Path(str(config.image_dir)), cluster=cluster, bands=bands, image_scale=str(config.image_scale))
    images = load_band_images(paths, bands)
    rgb = config.rgb
    gains = {role: getattr(rgb, f"{role}_gain") for role in ("red", "green", "blue")}
    display = build_rgb_display(images, bands=bands, q=rgb.q, stretch=rgb.stretch, minimum=rgb.minimum,
                                channel_gains=gains if all(v is not None for v in gains.values()) else None)
    return images, bands, display, cluster
