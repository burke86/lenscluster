from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lenscluster import rgb


def _load_literature_cutout_plotter() -> Any:
    from lenscluster import cutouts
    return cutouts


def test_rgb_display_defaults_match_clustertag() -> None:
    display = rgb.RGBDisplayConfig()

    assert display.q == 8.0
    assert display.stretch == 0.1
    assert display.minimum == 0.0
    assert dict(display.channel_gains) == {"red": 1.0, "green": 1.0, "blue": 1.2}


def test_build_rgb_display_is_lightweight_config_factory() -> None:
    band_images = {"B": object(), "G": object(), "R": object()}

    display = rgb.build_rgb_display_from_band_images(
        band_images,
        ("B", "G", "R"),
        q=6.0,
        stretch=0.25,
        channel_gains={"red": 0.8, "green": 1.1, "blue": 1.4},
        minimum=-0.02,
    )

    assert display == rgb.RGBDisplayConfig(
        q=6.0,
        stretch=0.25,
        channel_gains={"red": 0.8, "green": 1.1, "blue": 1.4},
        minimum=-0.02,
    )


def test_make_natural_rgb_uses_blue_green_red_band_order_and_lupton_rgb_order(monkeypatch) -> None:
    display = rgb.RGBDisplayConfig(q=7.0, stretch=0.2, channel_gains={"red": 2.0, "green": 3.0, "blue": 4.0})
    cutouts = {
        "B": np.full((4, 4), 1.0, dtype=np.float32),
        "G": np.full((4, 4), 10.0, dtype=np.float32),
        "R": np.full((4, 4), 100.0, dtype=np.float32),
    }
    calls: list[dict[str, Any]] = []

    def fake_make_lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append({"red": red.copy(), "green": green.copy(), "blue": blue.copy(), **kwargs})
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(rgb, "make_lupton_rgb", fake_make_lupton_rgb)

    result = rgb.make_natural_rgb(cutouts, bands=("B", "G", "R"), display=display)

    assert result.shape == (4, 4, 3)
    assert calls
    call = calls[0]
    np.testing.assert_allclose(call["red"], 200.0)
    np.testing.assert_allclose(call["green"], 30.0)
    np.testing.assert_allclose(call["blue"], 4.0)
    assert call["minimum"] == 0.0
    assert call["stretch"] == 0.2
    assert call["Q"] == 7.0


def test_make_natural_rgb_fills_invalid_pixels_with_channel_medians(monkeypatch) -> None:
    display = rgb.RGBDisplayConfig(channel_gains={"red": 1.0, "green": 1.0, "blue": 1.0})
    cutouts = {
        "B": np.array([[1.0, np.nan], [3.0, np.inf]], dtype=np.float32),
        "G": np.array([[10.0, 20.0], [np.nan, 40.0]], dtype=np.float32),
        "R": np.array([[100.0, 200.0], [300.0, np.nan]], dtype=np.float32),
    }
    calls: list[dict[str, np.ndarray]] = []

    def fake_make_lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append({"red": red.copy(), "green": green.copy(), "blue": blue.copy()})
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(rgb, "make_lupton_rgb", fake_make_lupton_rgb)

    rgb.make_natural_rgb(cutouts, bands=("B", "G", "R"), display=display)

    assert calls
    np.testing.assert_allclose(calls[0]["blue"], [[1.0, 2.0], [3.0, 2.0]])
    np.testing.assert_allclose(calls[0]["green"], [[10.0, 20.0], [20.0, 40.0]])
    np.testing.assert_allclose(calls[0]["red"], [[100.0, 200.0], [300.0, 200.0]])


def test_make_natural_rgb_rejects_all_invalid_channel() -> None:
    cutouts = {
        "B": np.full((2, 2), np.nan, dtype=np.float32),
        "G": np.ones((2, 2), dtype=np.float32),
        "R": np.ones((2, 2), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="no finite blue channel pixels"):
        rgb.make_natural_rgb(cutouts, bands=("B", "G", "R"))


def test_rgb_requires_three_bands() -> None:
    cutouts = {"B": np.ones((2, 2)), "R": np.ones((2, 2))}

    with pytest.raises(ValueError, match="exactly three bands"):
        rgb.make_natural_rgb(cutouts, bands=("B", "R"))

    with pytest.raises(ValueError, match="exactly three bands"):
        rgb.build_rgb_display_from_band_images({}, ("B", "R"))


def test_rgb_rejects_nonpositive_lupton_parameters() -> None:
    cutouts = {band: np.ones((2, 2), dtype=np.float32) for band in ("B", "G", "R")}

    with pytest.raises(ValueError, match="Lupton Q must be positive"):
        rgb.make_natural_rgb(cutouts, bands=("B", "G", "R"), display=rgb.RGBDisplayConfig(q=0.0))

    with pytest.raises(ValueError, match="Lupton stretch must be positive"):
        rgb.build_rgb_display_from_arrays(cutouts, ("B", "G", "R"), stretch=0.0)


def _ff_sims_multiband_display(**overrides: Any) -> rgb.MultiBandRGBDisplayConfig:
    params = {
        "q": 5.0,
        "stretch": 0.06,
        "channel_gains": {"red": 0.9, "green": 1.0, "blue": 1.15},
        "reference_band": "F814W",
        "normalization_percentile": 99.7,
        "channel_weights": {
            "blue": {"F435W": 1.00, "F606W": 0.35},
            "green": {"F606W": 0.65, "F814W": 0.75, "F105W": 0.20},
            "red": {"F814W": 0.20, "F105W": 0.45, "F125W": 0.45, "F140W": 0.40, "F160W": 0.35},
        },
    }
    params.update(overrides)
    return rgb.MultiBandRGBDisplayConfig(**params)


def test_make_multiband_rgb_renders_seven_band_display_on_reference_grid(monkeypatch) -> None:
    yy_acs, xx_acs = np.mgrid[:6, :6]
    yy_ir, xx_ir = np.mgrid[:3, :3]
    cutouts = {
        "F435W": (xx_acs + yy_acs + 1.0).astype(np.float32),
        "F606W": (2.0 * xx_acs + yy_acs + 2.0).astype(np.float32),
        "F814W": (xx_acs + 2.0 * yy_acs + 3.0).astype(np.float32),
        "F105W": (xx_ir + yy_ir + 4.0).astype(np.float32),
        "F125W": (2.0 * xx_ir + yy_ir + 5.0).astype(np.float32),
        "F140W": (xx_ir + 2.0 * yy_ir + 6.0).astype(np.float32),
        "F160W": (3.0 * xx_ir + yy_ir + 7.0).astype(np.float32),
    }
    calls: list[dict[str, Any]] = []

    def fake_make_lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append({"red": red.copy(), "green": green.copy(), "blue": blue.copy(), **kwargs})
        return np.zeros((6, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(rgb, "make_lupton_rgb", fake_make_lupton_rgb)

    result = rgb.make_multiband_rgb(cutouts, display=_ff_sims_multiband_display(minimum=-0.03))

    assert result.shape == (6, 6, 3)
    assert np.isfinite(result).all()
    assert calls
    assert calls[0]["red"].shape == (6, 6)
    assert calls[0]["green"].shape == (6, 6)
    assert calls[0]["blue"].shape == (6, 6)
    assert np.isfinite(calls[0]["red"]).all()
    assert np.isfinite(calls[0]["green"]).all()
    assert np.isfinite(calls[0]["blue"]).all()
    assert calls[0]["minimum"] == -0.03
    assert calls[0]["stretch"] == 0.06
    assert calls[0]["Q"] == 5.0


def test_make_multiband_rgb_resamples_wfc3_cutouts_to_reference_shape(monkeypatch) -> None:
    display = _ff_sims_multiband_display(
        channel_weights={
            "blue": {"F105W": 1.0},
            "green": {"F125W": 1.0},
            "red": {"F160W": 1.0},
        }
    )
    cutouts = {
        "F814W": np.arange(35, dtype=np.float32).reshape(5, 7),
        "F105W": np.arange(6, dtype=np.float32).reshape(2, 3),
        "F125W": np.arange(6, dtype=np.float32).reshape(2, 3) + 10.0,
        "F160W": np.arange(6, dtype=np.float32).reshape(2, 3) + 20.0,
    }
    calls: list[dict[str, np.ndarray]] = []

    def fake_make_lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append({"red": red.copy(), "green": green.copy(), "blue": blue.copy()})
        return np.zeros((5, 7, 3), dtype=np.uint8)

    monkeypatch.setattr(rgb, "make_lupton_rgb", fake_make_lupton_rgb)

    rgb.make_multiband_rgb(cutouts, display=display)

    assert calls
    assert calls[0]["red"].shape == (5, 7)
    assert calls[0]["green"].shape == (5, 7)
    assert calls[0]["blue"].shape == (5, 7)


def test_make_multiband_rgb_normalizes_each_band_before_mixing(monkeypatch) -> None:
    base = np.linspace(0.0, 10.0, 36, dtype=np.float32).reshape(6, 6)
    display = _ff_sims_multiband_display(
        channel_weights={
            "blue": {"F435W": 1.0},
            "green": {"F606W": 1.0},
            "red": {"F160W": 1.0},
        },
    )
    cutouts = {
        "F435W": base,
        "F606W": base + 2.0,
        "F814W": np.zeros((6, 6), dtype=np.float32),
        "F160W": base * 1.0e6,
    }
    calls: list[dict[str, np.ndarray]] = []

    def fake_make_lupton_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append({"red": red.copy(), "green": green.copy(), "blue": blue.copy()})
        return np.zeros((6, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(rgb, "make_lupton_rgb", fake_make_lupton_rgb)

    rgb.make_multiband_rgb(cutouts, display=display)

    assert calls
    assert float(np.nanmax(calls[0]["red"])) < 2.0
    assert float(np.nanmax(calls[0]["blue"])) < 2.0
    assert float(np.nanmax(calls[0]["red"])) < 5.0 * float(np.nanmax(calls[0]["blue"]))


def test_measure_multiband_normalization_returns_per_band_backgrounds_and_scales() -> None:
    base = np.linspace(0.0, 10.0, 36, dtype=np.float32).reshape(6, 6)
    backgrounds, scales = rgb.measure_multiband_normalization(
        {
            "F435W": base + 5.0,
            "F160W": base * 1.0e5 + 100.0,
        },
        bands=("F435W", "F160W"),
        percentile=99.7,
    )

    assert set(backgrounds) == {"F435W", "F160W"}
    assert set(scales) == {"F435W", "F160W"}
    assert backgrounds["F160W"] > backgrounds["F435W"]
    assert scales["F160W"] > 1.0e4 * scales["F435W"]


def test_make_multiband_rgb_rejects_missing_source_band() -> None:
    display = _ff_sims_multiband_display(
        channel_weights={
            "blue": {"F435W": 1.0},
            "green": {"F606W": 1.0},
            "red": {"F160W": 1.0},
        },
    )
    cutouts = {
        "F435W": np.ones((4, 4), dtype=np.float32),
        "F606W": np.ones((4, 4), dtype=np.float32),
        "F814W": np.ones((4, 4), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="Missing source band.*F160W"):
        rgb.make_multiband_rgb(cutouts, display=display)


def test_cutout_rgb_display_wrapper_forwards_simple_controls() -> None:
    plotter = _load_literature_cutout_plotter()

    display = plotter.build_rgb_display(
        {"B": object(), "G": object(), "R": object()},
        bands=("B", "G", "R"),
        q=5.0,
        stretch=0.3,
        channel_gains={"blue": 1.3, "green": 1.1, "red": 0.6},
        minimum=-0.04,
    )

    assert isinstance(display, rgb.CalibratedRGBDisplayConfig)
    assert display.q == 5.0
    assert display.stretch == 0.3
    assert dict(display.channel_gains) == {"blue": 1.3, "green": 1.1, "red": 0.6}
    assert display.minimum == -0.04


def test_cutout_rgb_display_wrapper_uses_calibrated_defaults() -> None:
    plotter = _load_literature_cutout_plotter()
    bands = ("F435W", "F606W", "F814W")
    band_images = {band: object() for band in bands}

    display = plotter.build_rgb_display(band_images, bands=bands)

    assert isinstance(display, rgb.CalibratedRGBDisplayConfig)
    assert display.q == rgb.DEFAULT_CALIBRATED_RGB_Q
    assert display.stretch == rgb.DEFAULT_CALIBRATED_RGB_STRETCH
    assert display.minimum == rgb.DEFAULT_RGB_MINIMUM
    assert dict(display.channel_gains) == rgb.DEFAULT_CALIBRATED_RGB_CHANNEL_GAINS
    assert dict(display.band_fluxscales) == {band: 1.0 for band in bands}
    assert dict(display.band_backgrounds) == {band: 0.0 for band in bands}


def test_cutout_rgb_display_wrapper_uses_hff_seven_band_defaults() -> None:
    plotter = _load_literature_cutout_plotter()
    band_images = {band: object() for band in rgb.DEFAULT_HFF_RGB_BANDS}

    display = plotter.build_rgb_display(band_images, bands=rgb.DEFAULT_HFF_RGB_BANDS)

    assert isinstance(display, rgb.CalibratedRGBDisplayConfig)
    assert display.q == rgb.DEFAULT_HFF_RGB_Q
    assert display.stretch == rgb.DEFAULT_HFF_RGB_STRETCH
    assert display.minimum == rgb.DEFAULT_HFF_RGB_MINIMUM
    assert dict(display.channel_gains) == rgb.DEFAULT_HFF_RGB_CHANNEL_GAINS
    assert dict(display.channel_weights) == rgb.DEFAULT_HFF_RGB_CHANNEL_WEIGHTS
    assert display.warm_highlight_desaturation == rgb.DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION
    assert display.highlight_knee == rgb.DEFAULT_HFF_RGB_HIGHLIGHT_KNEE
    assert display.highlight_ceiling == rgb.DEFAULT_HFF_RGB_HIGHLIGHT_CEILING
    assert display.highlight_softness == rgb.DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS
    assert dict(display.band_fluxscales) == {band: 1.0 for band in rgb.DEFAULT_HFF_RGB_BANDS}
    assert dict(display.band_backgrounds) == {band: 0.0 for band in rgb.DEFAULT_HFF_RGB_BANDS}


def test_cutout_rgb_display_uses_band_photometry() -> None:
    plotter = _load_literature_cutout_plotter()

    class FakeBandImage:
        def __init__(self, photflam: float, photplam: float, background: float, background_sigma: float) -> None:
            self.photflam = photflam
            self.photplam = photplam
            self.background = background
            self.background_sigma = background_sigma

    band_images = {
        "F435W": FakeBandImage(3.1476e-19, 4329.2, 1.4e-4, 5.0e-4),
        "F606W": FakeBandImage(7.8178e-20, 5922.0, 2.8e-4, 1.0e-3),
        "F814W": FakeBandImage(7.0478e-20, 8045.0, 2.0e-4, 6.0e-4),
    }

    display = plotter.build_rgb_display(band_images, bands=("F435W", "F606W", "F814W"))

    assert isinstance(display, rgb.CalibratedRGBDisplayConfig)
    assert display.band_fluxscales["F814W"] == pytest.approx(1.0)
    assert display.band_fluxscales["F435W"] == pytest.approx(1.293, rel=1e-2)
    assert display.band_fluxscales["F606W"] == pytest.approx(0.601, rel=1e-2)
    assert display.band_backgrounds == {"F435W": 1.4e-4, "F606W": 2.8e-4, "F814W": 2.0e-4}
    expected_minimum = rgb.CALIBRATED_RGB_MINIMUM_SKY_SIGMA * np.median(
        [5.0e-4 * display.band_fluxscales["F435W"], 1.0e-3 * display.band_fluxscales["F606W"], 6.0e-4]
    )
    assert display.minimum == pytest.approx(expected_minimum)


def test_make_calibrated_rgb_saturates_bright_cores_to_white() -> None:
    shape = (9, 9)
    bright = np.full(shape, 0.001, dtype=np.float32)
    bright[4, 4] = 50.0
    cutouts = {"F435W": bright.copy(), "F606W": bright.copy(), "F814W": bright.copy()}
    display = rgb.CalibratedRGBDisplayConfig()

    result = rgb.make_natural_rgb(cutouts, bands=("F435W", "F606W", "F814W"), display=display)

    assert result.dtype == np.uint8
    assert result.shape == (9, 9, 3)
    assert tuple(result[4, 4]) == (255, 255, 255)


def test_make_calibrated_hff_rgb_compresses_bright_cores_below_white() -> None:
    shape = (9, 9)
    faint = np.linspace(0.0002, 0.003, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    bright = faint.copy()
    bright[4, 4] = 50.0
    cutouts = {band: bright.copy() for band in rgb.DEFAULT_HFF_RGB_BANDS}
    display = rgb.CalibratedRGBDisplayConfig(
        q=rgb.DEFAULT_HFF_RGB_Q,
        stretch=rgb.DEFAULT_HFF_RGB_STRETCH,
        channel_gains=dict(rgb.DEFAULT_HFF_RGB_CHANNEL_GAINS),
        minimum=rgb.DEFAULT_HFF_RGB_MINIMUM,
        warm_highlight_desaturation=rgb.DEFAULT_HFF_RGB_WARM_HIGHLIGHT_DESATURATION,
        channel_weights=dict(rgb.DEFAULT_HFF_RGB_CHANNEL_WEIGHTS),
        highlight_knee=rgb.DEFAULT_HFF_RGB_HIGHLIGHT_KNEE,
        highlight_ceiling=rgb.DEFAULT_HFF_RGB_HIGHLIGHT_CEILING,
        highlight_softness=rgb.DEFAULT_HFF_RGB_HIGHLIGHT_SOFTNESS,
    )

    result = rgb.make_natural_rgb(cutouts, bands=rgb.DEFAULT_HFF_RGB_BANDS, display=display)

    assert result.dtype == np.uint8
    assert result.shape == (9, 9, 3)
    assert not np.any(np.all(result == 255, axis=-1))
    assert int(result[0, 0].max()) > 0
