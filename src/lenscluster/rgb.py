from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.visualization import LuptonAsinhStretch, make_lupton_rgb
from skimage.transform import resize as resize_image


DEFAULT_RGB_Q = 8.0
DEFAULT_RGB_STRETCH = 0.1
DEFAULT_RGB_MINIMUM = 0.0
DEFAULT_RGB_CHANNEL_GAINS = {
    "red": 1.0,
    "green": 1.0,
    "blue": 1.2,
}
DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE = 99.7
# Calibrated-path defaults tuned on the AS1063/M0416 BUFFALO 30mas mosaics.
# The seven-band HFF defaults intentionally keep faint sky/noise pixels visible
# while compressing bright cores below pure white.
DEFAULT_CALIBRATED_RGB_Q = 6.5
DEFAULT_CALIBRATED_RGB_STRETCH = 0.0165
DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS = {
    "red": 0.68,
    "green": 0.75,
    "blue": 3.5,
}
DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION = 0.65
CALIBRATED_RGB_MINIMUM_SKY_SIGMA = 1.2
DEFAULT_HFF_RGB_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W")
DEFAULT_HFF_RGB_REFERENCE_BAND = "F814W"
DEFAULT_HFF_RGB_Q = 6.4
DEFAULT_HFF_RGB_STRETCH = 0.0145
DEFAULT_HFF_RGB_MINIMUM = 0.0
# Colour balance.  Lupton et al. 2004 put the bands on an f_nu (AB) scale and leave the channel scaling as a stated
# choice.  On the Frontier Fields mosaics an AB-flat source is neutral at gains (1, 0.83, 0.86) and the median bright
# cluster member (F435W - F160W ~ 2.9 AB) is neutral at (1, 1.9, 11.8).  These defaults sit between the two in log, so
# the red sequence renders golden yellow with white cores, blue lensed galaxies blue-white rather than violet, and the
# sky neutral; the old (0.47, 0.91, 3.95) was the member-white balance and turned every faint blue source violet.
DEFAULT_HFF_RGB_CHANNEL_GAINS = {
    "red": 1.0,
    "green": 1.6,
    "blue": 6.5,
}
DEFAULT_HFF_RGB_CHANNEL_FLOOR_SIGMA = 1.0     # per channel, subtract this many sigma of its sky noise before the stretch
# JWST NIRCam.  The public cluster composites (STScI: MACS J1423, PLCK G165.7; UNCOVER Abell 2744, Furtak et al. 2023)
# assign blue = F115W + F150W, green = F200W + F277W, red = F356W + F410M + F444W.  With the four CANUCS/DJA bands we hold
# that is blue = F115W + F150W, green = F150W + F200W, red = F444W.  The mosaics are in nJy per pixel at 20 mas (SW) and
# 40 mas (LW); the display works in nJy per arcsec^2 (the band flux scales carry the pixel-area ratio) and the LW band is
# resampled onto the SW grid of the reference band.
DEFAULT_JWST_RGB_BANDS = ("F115W", "F150W", "F200W", "F444W")
DEFAULT_JWST_RGB_REFERENCE_BAND = "F200W"
DEFAULT_JWST_RGB_CHANNEL_WEIGHTS = {
    "blue": {"F115W": 1.0, "F150W": 0.5},
    "green": {"F150W": 0.5, "F200W": 1.0},
    "red": {"F200W": 0.5, "F444W": 1.0},     # F200W bridges the 2 to 4.4 micron gap that F277W and F356W fill in the release images;
}                                            # red alone from F444W turned every dusty or high-redshift source saturated red
# The JWST recipe was fitted against the STScI release image of MACS0416, registered onto the CANUCS mosaics (55 matched
# sources, 0.1" rms): the tone curve of the release is asinh with Q 55 and a saturation level of 70 nJy per 20 mas pixel,
# followed by gamma 0.55 and a 4.3% grey pedestal; its colour balance in display space gives these channel gains; and its
# colour saturation by luminance is 0.11/0.25/0.21/0.09 (median for luminance bins 0.1-0.25/0.25-0.5/0.5-0.75/0.75-0.95)
# against 0.11/0.24/0.21/0.16 for this recipe on the same field.  Half of the stretch scales each channel on its own
# (neutral sky, cores that run to white) and half preserves the hue (Lupton); a mild unsharp mask restores the local
# contrast that the compressive curve removes (compact sources measure 0.26" FWHM in the release and 0.29" here);
# colour fades to neutral below a quarter of full scale (the release's intracluster light is grey, not brown); and
# highlights roll off softly above the knee instead of clipping.  The CANUCS mosaics are background-subtracted already,
# so the display keeps the mosaic sky and no channel floor: a sky re-estimated on a cluster cutout is biased high by the
# intracluster light and clips it along a hard contour.
DEFAULT_JWST_RGB_CHANNEL_GAINS = {"red": 1.0, "green": 0.63, "blue": 0.76}
DEFAULT_JWST_RGB_Q = 55.0
DEFAULT_JWST_RGB_STRETCH = 70.0       # nJy per 20 mas reference pixel: the saturation level
DEFAULT_JWST_RGB_MINIMUM = 0.0
DEFAULT_JWST_RGB_PEDESTAL = 0.043
DEFAULT_JWST_RGB_GAMMA = 0.55
DEFAULT_JWST_RGB_CHROMA = 1.5
DEFAULT_JWST_RGB_HUE_MIX = 0.5                 # fraction of Lupton's hue-preserving scaling blended into the per-channel stretch
DEFAULT_JWST_RGB_UNSHARP_SIGMA_ARCSEC = 0.15   # unsharp mask on the luminance after the stretch
DEFAULT_JWST_RGB_UNSHARP_AMOUNT = 0.6
DEFAULT_JWST_RGB_SHADOW_LUMINANCE = 0.25       # colour fades linearly to neutral below this luminance
DEFAULT_JWST_RGB_HIGHLIGHT_KNEE = 0.8          # soft roll-off of the per-channel stretch above this level
DEFAULT_HFF_RGB_CHANNEL_WEIGHTS = {
    "blue": {
        "F435W": 0.92,
        "F606W": 0.40,
    },
    "green": {
        "F606W": 0.66,
        "F814W": 0.96,
        "F105W": 0.18,
    },
    "red": {
        "F814W": 0.18,
        "F105W": 0.28,
        "F125W": 0.40,
        "F140W": 0.34,
        "F160W": 0.32,
    },
}
# Bright cores saturate to white as in Lupton et al. 2004 (every pixel keeps its colour until a channel clips); the
# earlier knee/ceiling compression flattened the cluster ellipticals into uniform discs.
DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION = 0.0
DEFAULT_HFF_RGB_HIGHLIGHT_KNEE = 0.58
DEFAULT_HFF_RGB_HIGHLIGHT_CEILING = 1.0
DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS = 0.44
SPEED_OF_LIGHT_ANGSTROM_PER_S = 2.99792458e18


@dataclass(frozen=True)
class RGBDisplayConfig:
    q: float = DEFAULT_RGB_Q
    stretch: float = DEFAULT_RGB_STRETCH
    channel_gains: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RGB_CHANNEL_GAINS))
    minimum: float = DEFAULT_RGB_MINIMUM


@dataclass(frozen=True)
class CalibratedRGBDisplayConfig(RGBDisplayConfig):
    """Lupton asinh display with per-band sky subtraction and f_nu flux calibration.

    The weighted HFF path combines native-grid band cutouts directly and uses
    highlight compression to keep bright cores below pure white. The calibrated
    path does not spatially blur or resample pixels.
    """

    q: float = DEFAULT_CALIBRATED_RGB_Q
    stretch: float = DEFAULT_CALIBRATED_RGB_STRETCH
    channel_gains: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS))
    band_backgrounds: Mapping[str, float] = field(default_factory=dict)
    band_fluxscales: Mapping[str, float] = field(default_factory=dict)
    warm_highlight_desaturation: float = DEFAULT_CALIBRATED_RGB_WARM_HIGHLIGHT_DESATURATION
    channel_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    highlight_knee: float = 1.0
    highlight_ceiling: float = 1.0
    highlight_softness: float = DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS
    local_background: bool = False       # re-estimate each band's sky on the cutout itself (sigma-clipped) instead of the mosaic value
    channel_floor_sigma: float = 0.0     # subtract this many sigma of each channel's sky noise before the stretch (neutral, dark sky)
    smooth_pixels: int = 0               # box-smooth each band by this many pixels before the stretch (fine-pixel mosaics)
    per_channel_stretch: bool = False    # stretch each channel on its own, asinh(Q x / stretch) / asinh(Q), saturating at ``stretch``
                                         # (the STScI release look: peaked cores that run to white) instead of Lupton's shared
                                         # intensity scaling, which preserves colour but turns bright galaxies into plateaus
    hue_mix: float = 0.0                 # per-channel stretch only: fraction of Lupton's hue-preserving scaling blended in
    unsharp_sigma_pixels: float = 0.0    # per-channel stretch only: unsharp mask of the luminance (Gaussian sigma; 0 disables)
    unsharp_amount: float = 0.0
    shadow_luminance: float = 0.0        # per-channel stretch only: colour fades to neutral below this luminance
    pedestal: float = 0.0                # grey level of the sky in the output, as a fraction of full scale
    gamma: float = 1.0                   # applied to the stretched channels (below 1 lifts the faint sources)
    chroma: float = 1.0                  # scales each pixel's colour difference about its luminance after the stretch


@dataclass(frozen=True)
class MultiBandRGBDisplayConfig(RGBDisplayConfig):
    channel_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    reference_band: str = "F814W"
    normalization_percentile: float = DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE
    band_backgrounds: Mapping[str, float] = field(default_factory=dict)
    band_scales: Mapping[str, float] = field(default_factory=dict)


def trim_to_common_shape(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    valid = [np.asarray(array) for array in arrays if np.asarray(array).ndim == 2]
    if not valid:
        return []
    min_y = min(array.shape[0] for array in valid)
    min_x = min(array.shape[1] for array in valid)
    return [array[:min_y, :min_x] for array in valid]


def _validate_bands(bands: Sequence[str]) -> tuple[str, str, str]:
    band_tuple = tuple(str(band) for band in bands)
    if len(band_tuple) != 3:
        raise ValueError("RGB rendering requires exactly three bands in blue, green, red order.")
    return band_tuple[0], band_tuple[1], band_tuple[2]


def _validate_display(display: RGBDisplayConfig) -> None:
    if float(display.q) <= 0.0:
        raise ValueError("Lupton Q must be positive.")
    if float(display.stretch) <= 0.0:
        raise ValueError("Lupton stretch must be positive.")


def _channel_gain(display: RGBDisplayConfig, role: str) -> float:
    return float(display.channel_gains.get(role, DEFAULT_RGB_CHANNEL_GAINS[role]))


def _fill_invalid_with_median(data: np.ndarray, *, role: str) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render Lupton RGB: no finite {role} channel pixels.")
    filled = np.array(values, copy=True)
    filled[~finite] = float(np.median(filled[finite]))
    return filled


def _multiband_required_bands(display: MultiBandRGBDisplayConfig) -> list[str]:
    required = {str(display.reference_band)}
    for role in ("blue", "green", "red"):
        required.update(str(band) for band in display.channel_weights.get(role, {}))
    return sorted(required)


def _validate_multiband_display(display: MultiBandRGBDisplayConfig) -> None:
    _validate_display(display)
    percentile = float(display.normalization_percentile)
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError("Multi-band RGB normalization_percentile must be in the interval (0, 100].")
    for role in ("blue", "green", "red"):
        if not display.channel_weights.get(role):
            raise ValueError(f"Multi-band RGB requires at least one {role} channel weight.")


def _resize_cutout_to_shape(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")
    if values.shape == shape:
        return values
    return np.asarray(
        resize_image(
            values,
            shape,
            order=1,
            mode="edge",
            anti_aliasing=True,
            preserve_range=True,
        ),
        dtype=np.float32,
    )


def _measure_band_background_and_scale(data: np.ndarray, *, band: str, percentile: float) -> tuple[float, float]:
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render Lupton RGB: no finite {band} channel pixels.")
    median = float(np.median(values[finite]))
    centered = values - median
    scale = float(np.nanpercentile(centered[finite], percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = centered[finite & (centered > 0.0)]
        scale = float(np.nanmax(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return median, scale


def measure_multiband_normalization(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    percentile: float = DEFAULT_MULTIBAND_NORMALIZATION_PERCENTILE,
) -> tuple[dict[str, float], dict[str, float]]:
    backgrounds: dict[str, float] = {}
    scales: dict[str, float] = {}
    for band in bands:
        band_key = str(band)
        backgrounds[band_key], scales[band_key] = _measure_band_background_and_scale(
            np.asarray(cutouts_by_band[band_key], dtype=np.float32),
            band=band_key,
            percentile=float(percentile),
        )
    return backgrounds, scales


def _normalize_multiband_cutout(
    data: np.ndarray,
    *,
    band: str,
    percentile: float,
    background: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    filled = _fill_invalid_with_median(values, role=str(band))
    if background is None or scale is None:
        measured_background, measured_scale = _measure_band_background_and_scale(values, band=band, percentile=percentile)
        if background is None:
            background = measured_background
        if scale is None:
            scale = measured_scale
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    centered = filled - float(background)
    return np.clip(centered / scale, 0.0, None).astype(np.float32)


def _make_multiband_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    reference_shape: tuple[int, int],
    percentile: float,
    backgrounds: Mapping[str, float],
    scales: Mapping[str, float],
) -> np.ndarray:
    channel = np.zeros(reference_shape, dtype=np.float32)
    for band, weight in weights.items():
        band_key = str(band)
        resized = _resize_cutout_to_shape(cutouts_by_band[band_key], reference_shape)
        normalized = _normalize_multiband_cutout(
            resized,
            band=band_key,
            percentile=percentile,
            background=backgrounds.get(band_key),
            scale=scales.get(band_key),
        )
        channel += float(weight) * normalized
    return channel


def make_multiband_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    display: MultiBandRGBDisplayConfig,
) -> np.ndarray:
    _validate_multiband_display(display)
    missing = [band for band in _multiband_required_bands(display) if band not in cutouts_by_band]
    if missing:
        raise ValueError(f"Missing source band(s) for multi-band RGB: {', '.join(missing)}.")

    reference = np.asarray(cutouts_by_band[str(display.reference_band)], dtype=np.float32)
    if reference.ndim != 2:
        raise ValueError("Multi-band RGB reference band must be a 2D cutout array.")
    reference_shape = (int(reference.shape[0]), int(reference.shape[1]))
    percentile = float(display.normalization_percentile)

    img_b = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["blue"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )
    img_g = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["green"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )
    img_r = _make_multiband_channel(
        cutouts_by_band,
        display.channel_weights["red"],
        reference_shape=reference_shape,
        percentile=percentile,
        backgrounds=display.band_backgrounds,
        scales=display.band_scales,
    )

    return make_lupton_rgb(
        img_r * _channel_gain(display, "red"),
        img_g * _channel_gain(display, "green"),
        img_b * _channel_gain(display, "blue"),
        minimum=float(display.minimum),
        stretch=float(display.stretch),
        Q=float(display.q),
    )


def compute_fnu_band_fluxscales(
    photflam_by_band: Mapping[str, float | None],
    photplam_by_band: Mapping[str, float | None],
    *,
    reference_band: str,
) -> dict[str, float]:
    """Multiplicative factors converting counts to a common f_nu scale.

    HST drizzled mosaics are stored in electrons/s; equal AB surface brightness
    produces very different count rates per filter (the instrument is far more
    sensitive in F606W than in F435W), so combining raw counts skews colors.
    f_nu = counts * PHOTFLAM * PHOTPLAM^2 / c; factors are normalized so the
    reference band keeps a scale of 1. Bands with missing photometry keywords
    fall back to 1.
    """

    raw: dict[str, float] = {}
    for band in photflam_by_band:
        photflam = photflam_by_band.get(band)
        photplam = photplam_by_band.get(band)
        if photflam is None or photplam is None:
            continue
        photflam = float(photflam)
        photplam = float(photplam)
        if not np.isfinite(photflam) or not np.isfinite(photplam) or photflam <= 0.0 or photplam <= 0.0:
            continue
        raw[str(band)] = photflam * photplam * photplam / SPEED_OF_LIGHT_ANGSTROM_PER_S
    reference = raw.get(str(reference_band))
    if reference is None or reference <= 0.0:
        return {band: 1.0 for band in photflam_by_band}
    return {str(band): raw.get(str(band), reference) / reference for band in photflam_by_band}


def _calibrated_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    band: str,
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    return _calibrated_source_band(cutouts_by_band, band, role, display) * _channel_gain(display, role)


def _calibrated_source_band(
    cutouts_by_band: Mapping[str, np.ndarray],
    band: str,
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    values = np.asarray(cutouts_by_band[str(band)], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")
    background = float(display.band_backgrounds.get(str(band), 0.0))
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot render calibrated RGB: no finite {role} channel pixels.")
    if getattr(display, "local_background", False):
        background = _sigma_clipped_sky(values[finite])
    filled = np.array(values, copy=True)
    filled[~finite] = background
    smooth = int(getattr(display, "smooth_pixels", 0) or 0)
    if smooth > 1:
        from scipy.ndimage import uniform_filter
        filled = uniform_filter(filled, size=smooth, mode="nearest")
    fluxscale = float(display.band_fluxscales.get(str(band), 1.0))
    return (filled - background) * fluxscale


def _sigma_clipped_sky(values: np.ndarray, *, iterations: int = 5) -> float:
    """Sky level of a cutout: the median after iterated 3-sigma clipping about the median (robust to sources and ICL)."""
    kept = np.asarray(values, dtype=np.float64).ravel()
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return 0.0
    for _ in range(iterations):
        median = float(np.median(kept))
        sigma = 1.4826 * float(np.median(np.abs(kept - median)))
        if sigma <= 0.0:
            break
        inside = kept[np.abs(kept - median) < 3.0 * sigma]
        if inside.size in (0, kept.size):
            break
        kept = inside
    return float(np.median(kept))


def _robust_sigma(values: np.ndarray) -> float:
    kept = np.asarray(values, dtype=np.float64).ravel()
    kept = kept[np.isfinite(kept)]
    return 1.4826 * float(np.median(np.abs(kept - np.median(kept)))) if kept.size else 0.0


def _calibrated_weighted_channel(
    cutouts_by_band: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    role: str,
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    channel: np.ndarray | None = None
    for band, weight in weights.items():
        source = _calibrated_source_band(cutouts_by_band, str(band), role, display) * float(weight)
        channel = source if channel is None else channel + source
    if channel is None:
        raise ValueError(f"Calibrated RGB requires at least one {role} channel weight.")
    return channel * _channel_gain(display, role)


def _cubic_ramp(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - float(edge0)) / (float(edge1) - float(edge0)), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _neutralize_warm_highlights(rgb: np.ndarray, *, amount: float) -> np.ndarray:
    amount = float(amount)
    if amount <= 0.0:
        return rgb

    values = np.asarray(rgb, dtype=np.float32)
    red = values[0]
    green = values[1]
    blue = values[2]
    brightness = np.max(values, axis=0)
    warm_excess = np.clip((0.5 * (red + green) - blue) / np.maximum(brightness, 1.0e-6), 0.0, 1.0)
    highlight_weight = _cubic_ramp(0.08, 0.65, brightness)
    warm_blend = np.clip(amount * warm_excess * highlight_weight, 0.0, 1.0)
    clipped_blend = _cubic_ramp(0.92, 1.0, brightness)
    blend = np.maximum(warm_blend, clipped_blend)
    neutral = np.broadcast_to(brightness, values.shape)
    return np.clip(values * (1.0 - blend) + neutral * blend, 0.0, 1.0)


def _gently_neutralize_warm_highlights(rgb: np.ndarray, *, amount: float) -> np.ndarray:
    amount = float(amount)
    if amount <= 0.0:
        return rgb

    values = np.asarray(rgb, dtype=np.float32)
    red = values[0]
    green = values[1]
    blue = values[2]
    brightness = np.max(values, axis=0)
    warm_excess = np.clip((0.5 * (red + green) - blue) / np.maximum(brightness, 1.0e-6), 0.0, 1.0)
    highlight_weight = _cubic_ramp(0.18, 0.80, brightness)
    blend = np.clip(amount * warm_excess * highlight_weight, 0.0, 0.65)
    neutral = np.broadcast_to(brightness, values.shape)
    return values * (1.0 - blend) + neutral * blend


def _compress_highlights(
    rgb: np.ndarray,
    *,
    knee: float,
    ceiling: float,
    softness: float,
) -> np.ndarray:
    ceiling = float(ceiling)
    knee = float(knee)
    if ceiling >= 1.0:
        return rgb
    if not np.isfinite(ceiling) or ceiling <= 0.0 or ceiling > 1.0:
        raise ValueError("Calibrated RGB highlight_ceiling must be in the interval (0, 1].")
    if not np.isfinite(knee) or knee < 0.0 or knee >= ceiling:
        raise ValueError("Calibrated RGB highlight_knee must be finite and smaller than highlight_ceiling.")
    softness = max(float(softness), 1.0e-6)

    values = np.asarray(rgb, dtype=np.float32)
    brightness = np.max(values, axis=0)
    mapped = np.array(brightness, copy=True)
    high = brightness > knee
    if np.any(high):
        excess = (brightness[high] - knee) / softness
        mapped[high] = knee + (ceiling - knee) * (1.0 - np.exp(-excess))
    mapped = np.minimum(mapped, ceiling)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(brightness > 0.0, mapped / brightness, 0.0)
    return np.clip(values * scale, 0.0, ceiling)


def _calibrated_channel_weights_for_bands(bands: Sequence[str]) -> dict[str, dict[str, float]]:
    available = {str(band) for band in bands}
    weights: dict[str, dict[str, float]] = {}
    for role, role_weights in DEFAULT_HFF_RGB_CHANNEL_WEIGHTS.items():
        selected = {band: float(weight) for band, weight in role_weights.items() if band in available}
        if selected:
            weights[role] = selected
    return weights


def _per_channel_stretch(channels: np.ndarray, display: CalibratedRGBDisplayConfig) -> np.ndarray:
    """Each channel stretched on its own, optionally blended with the hue-preserving scaling, then sharpened, colour-scaled,
    rolled off above the highlight knee and lifted onto the pedestal.  ``channels`` is (3, ny, nx), non-negative."""
    q, stretch = float(display.q), float(display.stretch)
    gamma, pedestal = float(getattr(display, "gamma", 1.0)), float(getattr(display, "pedestal", 0.0))

    def curve(values: np.ndarray) -> np.ndarray:
        return (np.arcsinh(q * values / stretch) / np.arcsinh(q)) ** gamma

    scaled = curve(channels)
    mix = float(getattr(display, "hue_mix", 0.0))
    if mix > 0.0:
        intensity = np.mean(channels, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hue_preserving = np.where(intensity > 0.0, channels * (curve(intensity) / intensity), 0.0)
        scaled = (1.0 - mix) * scaled + mix * hue_preserving
    sigma, amount = float(getattr(display, "unsharp_sigma_pixels", 0.0)), float(getattr(display, "unsharp_amount", 0.0))
    if sigma > 0.0 and amount > 0.0:
        from scipy.ndimage import gaussian_filter
        luminance = np.mean(scaled, axis=0, keepdims=True)
        scaled = np.clip(scaled + amount * (luminance - gaussian_filter(luminance, (0.0, sigma, sigma), mode="nearest")), 0.0, None)
    luminance = np.mean(scaled, axis=0, keepdims=True)
    chroma = float(getattr(display, "chroma", 1.0)) * np.ones_like(luminance)
    shadow = float(getattr(display, "shadow_luminance", 0.0))
    if shadow > 0.0:
        chroma = chroma * np.clip(luminance / shadow, 0.0, 1.0)
    scaled = np.clip(luminance + chroma * (scaled - luminance), 0.0, None)
    knee = float(display.highlight_knee)
    if knee < 1.0:
        over = np.clip(scaled - knee, 0.0, None)
        scaled = np.where(scaled > knee, knee + (1.0 - knee) * (1.0 - np.exp(-over / (1.0 - knee))), scaled)
    scaled = pedestal + (1.0 - pedestal) * np.clip(scaled, 0.0, 1.0)
    return np.round(np.moveaxis(np.clip(scaled, 0.0, 1.0), 0, -1) * 255.0).astype(np.uint8)


def make_calibrated_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    display: CalibratedRGBDisplayConfig,
) -> np.ndarray:
    _validate_display(display)

    if display.channel_weights:
        missing_roles = [role for role in ("blue", "green", "red") if not display.channel_weights.get(role)]
        if missing_roles:
            raise ValueError(f"Calibrated RGB missing channel weight(s): {', '.join(missing_roles)}.")
        blue, green, red = trim_to_common_shape(
            [
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["blue"], "blue", display),
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["green"], "green", display),
                _calibrated_weighted_channel(cutouts_by_band, display.channel_weights["red"], "red", display),
            ]
        )
    else:
        blue_band, green_band, red_band = _validate_bands(bands)
        blue, green, red = trim_to_common_shape(
            [
                _calibrated_channel(cutouts_by_band, blue_band, "blue", display),
                _calibrated_channel(cutouts_by_band, green_band, "green", display),
                _calibrated_channel(cutouts_by_band, red_band, "red", display),
            ]
        )
    channels = np.stack([red, green, blue], axis=0) - float(display.minimum)
    floor_sigma = float(getattr(display, "channel_floor_sigma", 0.0))
    if floor_sigma > 0.0:
        # Each channel's sky noise differs (a boosted blue channel is the noisiest); clipping each at the same number of
        # its own sigma keeps the sky dark and neutral instead of coloured by the noisiest channel.
        floors = np.asarray([floor_sigma * _robust_sigma(c) for c in channels], dtype=np.float32)
        channels = channels - floors[:, None, None]
    channels = np.clip(channels, 0.0, None)
    if getattr(display, "per_channel_stretch", False):
        return _per_channel_stretch(channels, display)
    intensity = np.mean(channels, axis=0)
    stretch_fn = LuptonAsinhStretch(stretch=float(display.stretch), Q=float(display.q))
    stretched = stretch_fn(np.clip(intensity, 0.0, None), clip=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(intensity > 0.0, stretched / intensity, 0.0)
    scaled = channels * ratio
    if float(display.highlight_ceiling) < 1.0:
        scaled = _gently_neutralize_warm_highlights(scaled, amount=float(display.warm_highlight_desaturation))
        scaled = _compress_highlights(
            scaled,
            knee=float(display.highlight_knee),
            ceiling=float(display.highlight_ceiling),
            softness=float(display.highlight_softness),
        )
    else:
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled = _neutralize_warm_highlights(scaled, amount=float(display.warm_highlight_desaturation))
    return np.round(np.moveaxis(scaled, 0, -1) * 255.0).astype(np.uint8)


def build_rgb_display_from_band_images(
    band_images: Mapping[str, Any],
    bands: Sequence[str],
    *,
    q: float = DEFAULT_RGB_Q,
    stretch: float = DEFAULT_RGB_STRETCH,
    channel_gains: Mapping[str, float] = DEFAULT_RGB_CHANNEL_GAINS,
    minimum: float = DEFAULT_RGB_MINIMUM,
) -> RGBDisplayConfig:
    _ = band_images
    _validate_bands(bands)
    display = RGBDisplayConfig(q=float(q), stretch=float(stretch), channel_gains=dict(channel_gains), minimum=float(minimum))
    _validate_display(display)
    return display


def build_rgb_display_from_arrays(
    cutouts_by_band: Mapping[str, np.ndarray],
    bands: Sequence[str],
    *,
    q: float = DEFAULT_RGB_Q,
    stretch: float = DEFAULT_RGB_STRETCH,
    channel_gains: Mapping[str, float] = DEFAULT_RGB_CHANNEL_GAINS,
    minimum: float = DEFAULT_RGB_MINIMUM,
) -> RGBDisplayConfig:
    _ = cutouts_by_band
    return build_rgb_display_from_band_images({}, bands, q=q, stretch=stretch, channel_gains=channel_gains, minimum=minimum)


def make_natural_rgb(
    cutouts_by_band: Mapping[str, np.ndarray],
    *,
    bands: Sequence[str],
    display: RGBDisplayConfig | None = None,
) -> np.ndarray:
    if display is None:
        display = RGBDisplayConfig()
    if isinstance(display, MultiBandRGBDisplayConfig):
        return make_multiband_rgb(cutouts_by_band, display=display)
    if isinstance(display, CalibratedRGBDisplayConfig):
        return make_calibrated_rgb(cutouts_by_band, bands=bands, display=display)
    blue_band, green_band, red_band = _validate_bands(bands)
    _validate_display(display)

    blue = np.asarray(cutouts_by_band[str(blue_band)], dtype=np.float32)
    green = np.asarray(cutouts_by_band[str(green_band)], dtype=np.float32)
    red = np.asarray(cutouts_by_band[str(red_band)], dtype=np.float32)
    if blue.ndim != 2 or green.ndim != 2 or red.ndim != 2:
        raise ValueError("RGB rendering requires 2D cutout arrays.")

    blue, green, red = trim_to_common_shape([blue, green, red])
    img_r = _fill_invalid_with_median(red, role="red") * _channel_gain(display, "red")
    img_g = _fill_invalid_with_median(green, role="green") * _channel_gain(display, "green")
    img_b = _fill_invalid_with_median(blue, role="blue") * _channel_gain(display, "blue")

    return make_lupton_rgb(
        img_r,
        img_g,
        img_b,
        minimum=float(display.minimum),
        stretch=float(display.stretch),
        Q=float(display.q),
    )
