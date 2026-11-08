"""Solver configurations read from Lenstool .par files."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lenscluster.config import PriorConfig
from lenscluster.lenstool_par import config_from_lenstool_par, prior_from_limit, read_lenstool_par

REPO = Path(__file__).resolve().parents[1]
AS1063 = REPO / "catalogs/as1063/bergamini2019"


def test_limit_flags():
    assert prior_from_limit([0, 1.0]) is None
    assert prior_from_limit([1, -2, 4, 0.1]) == PriorConfig("uniform", lower=-2.0, upper=4.0, step=0.1)
    assert prior_from_limit([3, 248, 28]) == PriorConfig("normal", mean=248.0, std=28.0)
    assert prior_from_limit([9, 1200, 200, 1000]) == PriorConfig("truncated_normal", mean=1200.0, std=200.0, lower=1000.0, upper=math.inf)
    assert prior_from_limit([1, 5, 35, 0.1], scale=2.0).upper == 70.0
    with pytest.raises(ValueError, match="flag 2"):
        prior_from_limit([2, 1, 2])


def test_reader_drops_comments_and_aliases(tmp_path):
    par = tmp_path / "model.par"
    par.write_text("runmode\n  reference 3 1.0 2.0 # comment\n  end\n# a comment\nlimit 1\n  ellipticity 1 0 0.8 0.02\n  end\nfini\n")
    blocks = read_lenstool_par(par)
    assert blocks == [("runmode", None, {"reference": [3.0, 1.0, 2.0]}), ("limit", "1", {"ellipticite": [1.0, 0.0, 0.8, 0.02]})]


def test_as1063_matches_the_transcribed_model():
    config = config_from_lenstool_par(AS1063 / "source/Bergamini_RXCJ2248_lenstool_Normal.par", image_catalog=AS1063 / "images.yaml",
                                      member_catalog=AS1063 / "members.cat").validate()
    model = config.model
    assert (model.reference.ra0_deg, model.reference.dec0_deg) == (342.18321, -44.530878)
    assert (model.cosmology.H0, model.cosmology.Om0) == (70.0, 0.3)
    halo = model.large_halos[0]
    kpc_per_arcsec = halo.core_radius_kpc / 13.5871
    assert halo.cut_radius_kpc == pytest.approx(2000.0 * kpc_per_arcsec)
    assert halo.angle_pos == -36.0                       # the start moved inside the limit (-42, -36)
    assert halo.priors["core_radius_kpc"].lower == pytest.approx(5.0 * kpc_per_arcsec)
    assert halo.priors["v_disp"] == PriorConfig("truncated_normal", mean=1200.0, std=200.0, lower=1000.0, upper=math.inf)
    assert "core_radius_kpc" not in model.large_halos[1].priors
    population = model.member_populations[0]
    assert population.mag0 == pytest.approx(16.176001)
    assert config.scaling.alpha_sigma_prior_mean == pytest.approx(1.0 / 3.74)
    assert config.scaling.beta_radius_prior_mean == pytest.approx(2.0 / 3.0)
    assert set(config.scaling.fixed_relation_fields) == {"alpha_sigma", "beta_radius"}


def test_non_yaml_image_catalog_is_refused():
    with pytest.raises(ValueError, match="convert_lenstool_catalog"):
        config_from_lenstool_par(AS1063 / "source/Bergamini_RXCJ2248_lenstool_Normal.par", member_catalog=AS1063 / "members.cat")
