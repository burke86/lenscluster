"""The YAML member catalog: it reads to the same model as the potfile catalog it was converted from, validates its
entries, carries kinematics with their source, and its treatments and keep flag act on the model (exclude, independent
halo, protection from the impact cut)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.classification import impact_cut
from lenscluster.config import IndependentMemberHaloConfig, PriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.image_catalog import CatalogError
from lenscluster.inputs import declared_model_inputs
from lenscluster.member_catalog import load_member_catalog, members_from_lenstool, write_member_catalog
from lenscluster.state import build_state

SOURCES = {"K": {"reference": "a test", "quantity": "sigma_ap", "aperture_arcsec": 0.8}}


def _reference(config):
    r = config.model.reference
    return (int(r.reference), float(r.ra0_deg), float(r.dec0_deg))


def _yaml_config(config, tmp_path: Path, edit=None):
    """The synthetic config with its potfile catalog converted to YAML; ``edit`` changes the members dict first."""
    population = config.model.member_populations[0]
    members = members_from_lenstool(population.catalog_path, _reference(config), band="F160W")
    if edit is not None:
        edit(members)
    path = write_member_catalog(tmp_path / "members.yaml", members, band="F160W", kinematics_sources=SOURCES)
    return replace(config, model=replace(config.model, member_populations=(replace(population, catalog_path=str(path)),))).validate()


def test_yaml_catalog_reproduces_the_potfile_model(synthetic_config, tmp_path):
    converted = _yaml_config(synthetic_config, tmp_path)
    states = [build_state(synthetic_config), build_state(converted)]
    assert [s.name for s in states[0].parameter_specs] == [s.name for s in states[1].parameter_specs]
    values = [float(Evaluator(s, "smoothed").log_likelihood(jnp.asarray(s.initial_theta))) for s in states]
    assert values[0] == values[1]
    legacy = load_member_catalog(synthetic_config.model.member_populations[0].catalog_path, _reference(synthetic_config))
    table = load_member_catalog(converted.model.member_populations[0].catalog_path, _reference(synthetic_config))
    np.testing.assert_allclose(table[["ra", "dec", "catalog_mag", "catalog_a", "catalog_b", "catalog_theta"]].to_numpy(float),
                               legacy[["ra", "dec", "catalog_mag", "catalog_a", "catalog_b", "catalog_theta"]].to_numpy(float), rtol=0, atol=1e-12)
    assert list(table["id"]) == list(legacy["id"]) and set(table["catalog_treatment"]) == {"relation"}


def test_kinematics_carry_their_source_and_keep_the_member(tmp_path):
    members = {"2526": {"ra": 64.04, "dec": -24.07, "mag": {"F160W": 17.02, "F814W": [18.5, 0.01]}, "role": "bcg",
                        "kinematics": [{"source": "K", "sigma": 275.6, "err": 1.5, "sn": 164.6}]},
               "10": {"ra": 64.05, "dec": -24.06, "mag": {"F160W": 21.0, "F814W": 22.3}, "z": {"value": 0.4, "kind": "phot"}}}
    path = write_member_catalog(tmp_path / "m.yaml", members, band="F160W", kinematics_sources=SOURCES)
    table = load_member_catalog(path, (0, 0.0, 0.0)).set_index("id")
    assert table.loc["2526", "catalog_role"] == "bcg" and bool(table.loc["2526", "catalog_keep"]) and not bool(table.loc["10", "catalog_keep"])
    (measurement,) = table.loc["2526", "catalog_kinematics"]
    assert measurement == {"source": "K", "reference": "a test", "quantity": "sigma_ap", "aperture_arcsec": 0.8,
                           "sigma": 275.6, "err": 1.5, "sn": 164.6}
    assert table.loc["10", "catalog_z"] == 0.4 and table.loc["10", "catalog_z_kind"] == "phot"
    assert table.loc["10", "mag_F814W"] == 22.3 and table.loc["2526", "mag_F814W"] == 18.5
    # the relation band can be switched per population
    assert load_member_catalog(path, (0, 0.0, 0.0), band="F814W").set_index("id").loc["2526", "catalog_mag"] == 18.5


@pytest.mark.parametrize("text, message", [
    ("band: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}, colour: red}\n", "unknown keys"),
    ("band: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}}\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 21}}\n", "duplicate key"),
    ("band: F160W\nmembers:\n  1: {ra: 1.0, dec: 2.0, mag: {F160W: 20}}\n", "quote it"),
    ("band: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F814W: 20}}\n", "no F160W magnitude"),
    ("band: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}, kinematics: [{source: X, sigma: 100, err: 5}]}\n",
     "not in kinematics_sources"),
    ("band: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}, treatment: free}\n", "treatment must be one of"),
    ("band: F160W\nkinematics_sources:\n  K: {reference: r}\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}}\n", "aperture_arcsec"),
    ("members:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}}\n", "missing band"),
    ("format: x\nband: F160W\nmembers:\n  '1': {ra: 1.0, dec: 2.0, mag: {F160W: 20}}\n", "unknown keys"),
])
def test_malformed_catalogs_are_refused(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(CatalogError, match=message):
        load_member_catalog(path, (0, 0.0, 0.0))


def test_treatments_exclude_and_make_independent_halos(synthetic_config, tmp_path):
    def edit(members):
        members["g1"]["treatment"] = "independent"
        members["g4"]["treatment"] = "exclude"
    config = _yaml_config(synthetic_config, tmp_path, edit)
    potentials, potfiles, *_ = declared_model_inputs(config.model, config.scaling)
    assert list(potfiles[0]["catalog_df"]["id"]) == ["g2", "g3"]
    independent = [p for p in potentials if p.get("independent_member_catalog_id") == "g1"]
    assert len(independent) == 1 and independent[0]["priors"]["v_disp"][0] == 9          # the default truncated normal
    # a halo the config declares for that member keeps the config's priors
    declared = IndependentMemberHaloConfig(population_id="members", catalog_id="g1", priors={"v_disp": PriorConfig("uniform", lower=50.0, upper=150.0)})
    config = replace(config, model=replace(config.model, independent_member_halos=(declared,))).validate()
    potentials, *_ = declared_model_inputs(config.model, config.scaling)
    (halo,) = [p for p in potentials if p.get("independent_member_catalog_id") == "g1"]
    assert halo["priors"]["v_disp"][0] == 1


def test_the_impact_cut_never_drops_a_kept_member(synthetic_config, tmp_path):
    def edit(members):
        members["g4"]["kinematics"] = [{"source": "K", "sigma": 80.0, "err": 4.0}]
    config = _yaml_config(synthetic_config, tmp_path, edit)
    state = build_state(config)
    evaluator = Evaluator(state, "smoothed")
    dropped, deflection = impact_cut(state, evaluator, state.initial_theta, floor=1.0e3)       # every member is below
    assert len(deflection) == 4 and "members:g4" not in dropped and len(dropped) == 3
