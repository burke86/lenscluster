"""Cluster-member catalogs: a YAML file of members with their photometry, role, treatment and stellar kinematics, or
the Lenstool potfile columns ``id x y a b theta mag lum color`` after a ``#REFERENCE`` line (``.cat``).

YAML example::

    band: F160W
    kinematics_sources:
      G26: {reference: "Granata et al. 2026, A&A, Appendix C", quantity: sigma_ap, aperture_arcsec: 0.8}
    members:
      "2526":
        ra: 64.03816791
        dec: -24.06750808
        mag: {F160W: 17.0166}
        role: bcg
        kinematics: [{source: G26, sigma: 275.6, err: 1.5, sn: 164.6}]
      "4112": {ra: 64.0352, dec: -24.0735, mag: {F160W: 17.95}}

Fields.  Top level: ``band`` (required: the magnitude the scaling relation uses), ``members`` (required),
``kinematics_sources``, ``notes``.  Member: ``ra``/``dec`` (degrees, ICRS; required), ``mag`` ({band: mag} or
{band: [mag, err]}; the ``band`` entry is required), ``shape`` ([a, b, theta]; default round), ``z`` (a number or
{value, kind} with kind spec or phot), ``role`` (member or bcg), ``treatment`` (relation: on the scaling relation;
independent: its own dPIE, as a ``model.independent_member_halos`` entry with default priors unless the config declares
it; exclude: kept for the record, not modelled), ``keep`` (true protects it from ``members.min_deflection_arcsec``;
members with kinematics are always kept), ``kinematics`` (a list of measurements {source, sigma, err, sn, notes}, sigma
and err in km/s), ``notes``.  Kinematics source: ``reference`` (required), ``quantity`` (sigma_ap: within a circular
aperture of ``aperture_arcsec``; sigma_0: central, within R_e/8; sigma_e: within R_e), ``aperture_arcsec``, ``notes``.

Member names are strings: YAML reads an unquoted 2526 as a number, so quote it; ``write_member_catalog`` quotes them.
A name repeated in ``members`` is an error (plain YAML would keep the last one silently).

``#REFERENCE 0`` in a ``.cat`` file means x, y are RA and Dec in degrees; ``#REFERENCE 3`` means arcsecond offsets from
the model's reference point (``sky.radec_to_offsets_arcsec`` convention).  Other ``#`` lines are comments.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .image_catalog import CatalogError, _check_keys, _Dumper, _Flow, _number, _require_mapping, _shape
from .sky import offsets_to_radec

COLUMNS = ["id", "coord_1", "coord_2", "a", "b", "theta", "mag", "lum", "color"]
OUTPUT_COLUMNS = ["id", "catalog_reference", "catalog_source", "ra", "dec", "catalog_a", "catalog_b", "catalog_theta",
                  "catalog_mag", "catalog_z", "catalog_lum", "catalog_color", "catalog_band", "catalog_z_kind", "catalog_role",
                  "catalog_treatment", "catalog_keep", "catalog_kinematics", "catalog_notes"]
TOP_KEYS = {"band", "kinematics_sources", "members", "notes"}
MEMBER_KEYS = {"ra", "dec", "mag", "shape", "z", "role", "treatment", "keep", "kinematics", "notes"}
SOURCE_KEYS = {"reference", "quantity", "aperture_arcsec", "notes"}
MEASUREMENT_KEYS = {"source", "sigma", "err", "sn", "notes"}
Z_KEYS = {"value", "kind"}
ROLES = ("member", "bcg")
TREATMENTS = ("relation", "independent", "exclude")
QUANTITIES = ("sigma_ap", "sigma_0", "sigma_e")
Z_KINDS = ("spec", "phot")
ROUND = (1.0, 1.0, 0.0)


def load_member_catalog(path: str | Path, reference: tuple[int, float, float], *, band: str | None = None) -> pd.DataFrame:
    """One row per member with ``OUTPUT_COLUMNS`` plus ``mag_<band>`` for every band a YAML entry lists.
    ``reference`` is the model's (kind, ra0, dec0), used by ``.cat`` files in offsets; ``band`` overrides the YAML
    file's relation band.  ``catalog_kinematics`` holds a list of measurements per member, each with its source's
    reference, quantity and aperture."""
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path, band)
    if band is not None:
        raise CatalogError(f"{path}: a .cat member catalog has one magnitude column; band {band!r} applies to YAML catalogs only.")
    return _load_lenstool(path, reference)


# ----------------------------------------------------------------------------- YAML
class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that refuses a key repeated within one mapping."""


def _construct_unique_mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise CatalogError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}.")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _load_yaml(path: Path, band_override: str | None) -> pd.DataFrame:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except CatalogError as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path}: not valid YAML: {exc}") from exc
    where = str(path)
    document = _require_mapping(document, where)
    _check_keys(document, TOP_KEYS, where)
    for key in ("band", "members"):
        if key not in document:
            raise CatalogError(f"{where}: missing {key}.")
    band = band_override if band_override is not None else document["band"]
    if not isinstance(band, str) or not band:
        raise CatalogError(f"{where}: band must be a band name; got {band!r}.")
    sources = {}
    for name, entry in _require_mapping(document.get("kinematics_sources") or {}, f"{where}: kinematics_sources").items():
        sources[_string(name, "kinematics source", f"{where}: kinematics_sources")] = _source(entry, f"{where}: kinematics_sources.{name}")
    rows = [_member(name, entry, f"{where}: members.{name}", band, sources)
            for name, entry in _require_mapping(document["members"], f"{where}: members").items()]
    if not rows:
        raise CatalogError(f"{where}: no members.")
    table = pd.DataFrame(rows)
    table["catalog_reference"] = 0
    table["catalog_source"] = str(path)
    table["catalog_band"] = band
    bands = sorted(c for c in table.columns if c.startswith("mag_") and c not in OUTPUT_COLUMNS)
    return table.loc[:, OUTPUT_COLUMNS + bands].reset_index(drop=True)


def _string(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{where}: {what} name {value!r} was read as {type(value).__name__}; quote it, e.g. \"{value}\".")
    return value


def _choice(value: Any, choices: tuple[str, ...], where: str) -> str:
    if value not in choices:
        raise CatalogError(f"{where} must be one of {list(choices)}; got {value!r}.")
    return str(value)


def _source(entry: Any, where: str) -> dict[str, Any]:
    entry = _require_mapping(entry, where)
    _check_keys(entry, SOURCE_KEYS, where)
    if not isinstance(entry.get("reference"), str):
        raise CatalogError(f"{where}: missing reference.")
    quantity = _choice(entry.get("quantity", "sigma_ap"), QUANTITIES, f"{where}.quantity")
    aperture = entry.get("aperture_arcsec")
    if quantity == "sigma_ap" and aperture is None:
        raise CatalogError(f"{where}: quantity sigma_ap needs aperture_arcsec.")
    return {"reference": entry["reference"], "quantity": quantity,
            "aperture_arcsec": np.nan if aperture is None else _number(aperture, f"{where}.aperture_arcsec", positive=True)}


def _member(name: Any, entry: Any, where: str, band: str, sources: dict[str, dict]) -> dict[str, Any]:
    name = _string(name, "member", where)
    entry = _require_mapping(entry, where)
    _check_keys(entry, MEMBER_KEYS, where)
    for key in ("ra", "dec", "mag"):
        if key not in entry:
            raise CatalogError(f"{where}: missing {key}.")
    mags = {}
    for mag_band, value in _require_mapping(entry["mag"], f"{where}.mag").items():
        here = f"{where}.mag.{mag_band}"
        if isinstance(value, list):
            if len(value) != 2:
                raise CatalogError(f"{here} must be a magnitude or [mag, err].")
            value = value[0]
        mags[f"mag_{_string(mag_band, 'band', where)}"] = _number(value, here)
    if f"mag_{band}" not in mags:
        raise CatalogError(f"{where}: no {band} magnitude (the band of the scaling relation).")
    shape = _shape(entry["shape"], f"{where}.shape") if "shape" in entry else ROUND
    z, z_kind = np.nan, ""
    if "z" in entry:
        value = entry["z"]
        if isinstance(value, dict):
            _check_keys(value, Z_KEYS, f"{where}.z")
            z_kind = _choice(value.get("kind", "spec"), Z_KINDS, f"{where}.z.kind")
            value = value.get("value")
        z = _number(value, f"{where}.z", lower=0.0)
    kinematics = []
    for k, measurement in enumerate(entry.get("kinematics") or []):
        here = f"{where}.kinematics[{k}]"
        measurement = _require_mapping(measurement, here)
        _check_keys(measurement, MEASUREMENT_KEYS, here)
        source = measurement.get("source")
        if source not in sources:
            raise CatalogError(f"{here}: source {source!r} is not in kinematics_sources ({sorted(sources)}).")
        for key in ("sigma", "err"):
            if key not in measurement:
                raise CatalogError(f"{here}: missing {key}.")
        kinematics.append({"source": source, **sources[source],
                           "sigma": _number(measurement["sigma"], f"{here}.sigma", positive=True),
                           "err": _number(measurement["err"], f"{here}.err", positive=True),
                           "sn": np.nan if "sn" not in measurement else _number(measurement["sn"], f"{here}.sn", positive=True)})
    keep = entry.get("keep", False)
    if not isinstance(keep, bool):
        raise CatalogError(f"{where}.keep must be true or false.")
    return {"id": name, "ra": _number(entry["ra"], f"{where}.ra"), "dec": _number(entry["dec"], f"{where}.dec", lower=-90.0, upper=90.0),
            "catalog_a": shape[0], "catalog_b": shape[1], "catalog_theta": shape[2], "catalog_mag": mags[f"mag_{band}"],
            "catalog_z": z, "catalog_z_kind": z_kind, "catalog_lum": np.nan, "catalog_color": np.nan,
            "catalog_role": _choice(entry.get("role", "member"), ROLES, f"{where}.role"),
            "catalog_treatment": _choice(entry.get("treatment", "relation"), TREATMENTS, f"{where}.treatment"),
            "catalog_keep": bool(keep or kinematics), "catalog_kinematics": kinematics,
            "catalog_notes": str(entry.get("notes", "")), **mags}


def write_member_catalog(path: str | Path, members: dict[str, dict], *, band: str, kinematics_sources: dict[str, dict] | None = None,
                         header: str | None = None) -> Path:
    """Write ``members`` ({name: {ra, dec, mag, ...}}, the structure ``load_member_catalog`` reads) as a YAML catalog.
    Plain members go on one line each; members with a role, treatment, keep flag, kinematics or notes in block style.
    ``header`` becomes comment lines at the top.  The file is read back and checked before this returns."""
    path = Path(path)
    document: dict[str, Any] = {"band": band}
    if kinematics_sources:
        document["kinematics_sources"] = {str(k): _Flow(v) for k, v in kinematics_sources.items()}
    out = {}
    for name, member in members.items():
        member = dict(member)
        if "mag" in member:
            member["mag"] = _Flow(member["mag"])
        if isinstance(member.get("z"), dict):
            member["z"] = _Flow(member["z"])
        if "kinematics" in member:
            member["kinematics"] = [_Flow(m) for m in member["kinematics"]]
        plain = not ({"role", "treatment", "keep", "kinematics", "notes"} & set(member))
        out[str(name)] = _Flow(member) if plain else member
    document["members"] = out
    text = yaml.dump(document, Dumper=_Dumper, sort_keys=False, width=200, allow_unicode=True, default_flow_style=False)
    comment = "".join(f"# {line}".rstrip() + "\n" for line in (header or "").splitlines())
    path.write_text(comment + text, encoding="utf-8")
    load_member_catalog(path, (0, 0.0, 0.0))
    return path


# ----------------------------------------------------------------------------- Lenstool potfile columns
def _load_lenstool(path: Path, reference: tuple[int, float, float]) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] == "#REFERENCE" or (tokens[0] == "#" and len(tokens) > 1 and tokens[1] == "REFERENCE"):
            header = tokens[1:] if tokens[0] == "#REFERENCE" else tokens[2:]
            continue
        if line.startswith("#"):
            continue
        if len(tokens) != len(COLUMNS):
            raise ValueError(f"{path}: member rows need {len(COLUMNS)} columns ({' '.join(COLUMNS)}); got {len(tokens)}: {line!r}")
        rows.append(tokens)
    if not header:
        raise ValueError(f"{path}: missing #REFERENCE line.")
    kind = int(header[0])
    if kind not in (0, 3):
        raise ValueError(f"{path}: #REFERENCE {kind} is not supported (0 = RA/Dec, 3 = offsets from the model reference).")
    table = pd.DataFrame(rows, columns=COLUMNS)
    table["id"] = table["id"].astype(str)
    for column in ("coord_1", "coord_2", "a", "b", "theta", "lum"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    for column in ("mag", "color"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if kind == 0:
        table["ra"], table["dec"] = table["coord_1"], table["coord_2"]
    else:
        ra, dec = offsets_to_radec(table["coord_1"].to_numpy(float), table["coord_2"].to_numpy(float), float(reference[1]), float(reference[2]))
        table["ra"], table["dec"] = ra, dec
    table = table.rename(columns={"a": "catalog_a", "b": "catalog_b", "theta": "catalog_theta", "mag": "catalog_mag",
                                  "lum": "catalog_lum", "color": "catalog_color"})
    table["catalog_z"] = np.nan
    table["catalog_reference"] = kind
    table["catalog_source"] = str(path)
    table["catalog_band"] = ""
    table["catalog_z_kind"] = ""
    table["catalog_role"] = "member"
    table["catalog_treatment"] = "relation"
    table["catalog_keep"] = False
    table["catalog_kinematics"] = [[] for _ in range(len(table))]
    table["catalog_notes"] = ""
    return table.loc[:, OUTPUT_COLUMNS].drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)


def members_from_lenstool(path: str | Path, reference: tuple[int, float, float], *, band: str) -> dict[str, dict]:
    """The members of a ``.cat`` file as the ``write_member_catalog`` structure: ra, dec, mag in ``band``, and the shape
    when it is not round.  For converting a potfile catalog to YAML."""
    table = _load_lenstool(Path(path), reference)
    members = {}
    for row in table.itertuples(index=False):
        entry: dict[str, Any] = {"ra": float(row.ra), "dec": float(row.dec), "mag": {band: float(row.catalog_mag)}}
        shape = (float(row.catalog_a), float(row.catalog_b), float(row.catalog_theta))
        if not (math.isclose(shape[0], shape[1]) and shape[2] == 0.0):
            entry["shape"] = list(shape)
        members[str(row.id)] = entry
    return members
