"""The YAML image catalog: conversion of the Bergamini M0416 Lenstool file, knots, error precedence and validation."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from lenscluster.image_catalog import CatalogError, load_image_catalog, write_image_catalog
from lenscluster.state import build_state

ROOT = Path(__file__).resolve().parents[1]
BERGAMINI = ROOT / "catalogs/m0416/bergamini2023/source/obs_arcs.cat"


def write(tmp_path: Path, text: str, name: str = "images.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------- the converter
@pytest.mark.skipif(not BERGAMINI.exists(), reason="Bergamini M0416 catalog not present")
def test_converter_keeps_every_field_of_the_bergamini_catalog(tmp_path):
    out = tmp_path / "m0416.yaml"
    subprocess.run([sys.executable, str(ROOT / "scripts/convert_lenstool_catalog.py"), str(BERGAMINI), str(out)], check=True, capture_output=True)
    table = load_image_catalog(out, sigma_inst_arcsec=0.01).set_index("image_label")
    raw = [line.split() for line in BERGAMINI.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert list(table.index) == [r[0] for r in raw]
    for label, x, y, a, b, theta, z, mag in raw:
        row = table.loc[label]
        assert (row.ra, row.dec, row.catalog_a, row.catalog_b, row.catalog_theta, row.catalog_z, row.catalog_mag) == tuple(
            float(v) for v in (x, y, a, b, theta, z, mag)), label
        assert row.family_id == label.split(".")[0] and row.image_id == label.split(".")[1]
    assert set(table.sigma_inst_arcsec) == {0.01} and set(table.parity) == {0} and set(table.family_reliability) == {1.0}
    assert "'021':" in out.read_text()                    # leading zeros survive because the name is quoted


def test_converter_reads_offsets_band_magnitudes_and_astrometry(tmp_path):
    cat = write(tmp_path, "#REFERENCE 3\n1.a 3 5 1 1 0 2 24 1 .05\n1.b -4 -5 1 1 0 2 24.5 1 .05\n", "obs.cat")
    (tmp_path / "obs_band_magnitudes.csv").write_text("image_label,mag_F814W,mag_err_F814W,detected_F814W\n1.a,25.1,0.1,True\n")
    astro = write(tmp_path, "image_label,sigma_inst_arcsec,parity\n1.b,0.02,-1\n", "astro.csv")
    out = tmp_path / "out.yaml"
    subprocess.run([sys.executable, str(ROOT / "scripts/convert_lenstool_catalog.py"), str(cat), str(out), "--reference", "64.0", "-24.0",
                    "--astrometry", str(astro)], check=True, capture_output=True)
    table = load_image_catalog(out, sigma_inst_arcsec=0.05).set_index("image_label")
    x = (64.0 - table.ra) * np.cos(np.radians(-24.0)) * 3600
    y = (table.dec + 24.0) * 3600
    np.testing.assert_allclose(x, [3, -4], atol=1e-9)
    np.testing.assert_allclose(y, [5, -5], atol=1e-9)
    assert table.loc["1.a", "mag_F814W"] == 25.1 and table.loc["1.a", "mag_err_F814W"] == 0.1 and bool(table.loc["1.a", "detected_F814W"])
    assert np.isnan(table.loc["1.b", "mag_F814W"])
    assert table.loc["1.b", "sigma_inst_arcsec"] == 0.02 and table.loc["1.b", "parity"] == -1
    assert table.loc["1.a", "sigma_inst_arcsec"] == 0.05 and table.loc["1.a", "catalog_mag_err"] == 0.05


# ----------------------------------------------------------------------------- knots and precedence
KNOTS = """format: lenscluster-images/1
defaults: {sigma: 0.04}
systems:
  "D1":
    z: 0.94
    knots: [A, B]
    images:
      a:
        parity: -1
        sigma: 0.02
        knots:
          A: [64.03, -24.06]
          B: [64.031, -24.061, 0.01]
      b:
        knots:
          A: {ra: 64.04, dec: -24.07, bands: {F150W: [25.8, 0.06]}}
  "7":
    z: 2.1
    reliability: 0.5
    images:
      a: {ra: 64.05, dec: -24.08}
      b: {ra: 64.06, dec: -24.09, sigma: 0.003}
"""


def test_knots_become_families_and_errors_follow_precedence(tmp_path):
    table = load_image_catalog(write(tmp_path, KNOTS), sigma_inst_arcsec=0.5, sigma_floor_arcsec=0.005).set_index("image_label")
    assert list(table.index) == ["D1A.a", "D1B.a", "D1A.b", "7.a", "7.b"]
    assert table.loc["D1B.a", "family_id"] == "D1B" and table.loc["D1B.a", "system_id"] == "D1" and table.loc["D1B.a", "knot_id"] == "B"
    assert table.loc["D1A.a", "parity"] == -1 and table.loc["D1B.a", "parity"] == -1 and table.loc["D1A.b", "parity"] == 0
    # knot > image > defaults; the config value is used only when the file gives none; the floor raises small errors
    assert table.loc["D1B.a", "sigma_inst_arcsec"] == 0.01 and table.loc["D1A.a", "sigma_inst_arcsec"] == 0.02
    assert table.loc["D1A.b", "sigma_inst_arcsec"] == 0.04 and table.loc["7.b", "sigma_inst_arcsec"] == 0.005
    assert table.loc["7.a", "family_reliability"] == 0.5 and table.loc["D1A.a", "family_reliability"] == 1.0
    assert table.loc["D1A.b", "mag_F150W"] == 25.8 and np.isnan(table.loc["D1A.a", "mag_F150W"])
    no_defaults = KNOTS.replace("defaults: {sigma: 0.04}\n", "")
    assert load_image_catalog(write(tmp_path, no_defaults), sigma_inst_arcsec=0.5).set_index("image_label").loc["D1A.b", "sigma_inst_arcsec"] == 0.5


def test_writer_round_trip(tmp_path):
    document = yaml.safe_load(KNOTS)
    path = write_image_catalog(tmp_path / "copy.yaml", document["systems"], defaults=document["defaults"], header="a copy")
    a = load_image_catalog(write(tmp_path, KNOTS), sigma_inst_arcsec=0.5)
    b = load_image_catalog(path, sigma_inst_arcsec=0.5)
    assert a.drop(columns="catalog_source").equals(b.drop(columns="catalog_source"))
    assert path.read_text().startswith("# a copy\n")


# ----------------------------------------------------------------------------- validation
@pytest.mark.parametrize("edit, message", [
    (lambda t: t.replace('"7":', "021:"), "quote it"),
    (lambda t: t.replace("knots: [A, B]", "knots: [A]"), "not declared"),
    (lambda t: t.replace("          A: {ra: 64.04", "          C: {ra: 64.04"), "not declared"),
    (lambda t: t.replace("parity: -1", "parity: 2"), "parity"),
    (lambda t: t.replace("format: lenscluster-images/1\n", ""), "format"),
    (lambda t: t.replace("reliability: 0.5", "reliabilty: 0.5"), "unknown keys"),
    (lambda t: t.replace("    z: 2.1\n", ""), "missing z"),
    (lambda t: t.replace("a: {ra: 64.05, dec: -24.08}", "a: {ra: 64.05, dec: -94.0}"), "dec"),
    (lambda t: t.replace("      b:\n        knots:\n          A: {ra: 64.04", "      b:\n        ra: 1\n        knots:\n          A: {ra: 64.04"), "unknown keys"),
    (lambda t: t.replace('"7":', '"D1A":'), "collides"),
])
def test_validation_errors(tmp_path, edit, message):
    with pytest.raises(CatalogError, match=message):
        load_image_catalog(write(tmp_path, edit(KNOTS)), sigma_inst_arcsec=0.1)


def test_lenstool_file_is_refused_with_the_converter_named(tmp_path):
    with pytest.raises(CatalogError, match="convert_lenstool_catalog"):
        load_image_catalog(write(tmp_path, "#REFERENCE 0\n1.a 1 2 1 1 0 2 24\n", "obs_arcs.cat"), sigma_inst_arcsec=0.1)


# ----------------------------------------------------------------------------- the likelihood override
def test_likelihood_sigma_override_replaces_catalog_errors(synthetic_config):
    path = Path(synthetic_config.model.image_constraints.catalog_path)
    document = yaml.safe_load(path.read_text())
    document["systems"]["1"]["images"]["a"]["sigma"] = 0.3
    write_image_catalog(path, document["systems"])
    assert np.asarray(build_state(synthetic_config).images.sigma_inst_per_image).max() == pytest.approx(0.3)
    config = replace(synthetic_config, likelihood=replace(synthetic_config.likelihood, sigma_inst_arcsec=0.02)).validate()
    np.testing.assert_allclose(np.asarray(build_state(config).images.sigma_inst_per_image), 0.02)
