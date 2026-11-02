"""The multiple-image catalog: a YAML file of systems, their images and, optionally, knots within each image.

A system is one background source at one redshift.  Each image of it has either one position or a set of named knots
(clumps of the same galaxy seen in every image); the knot names are declared once per system.  Example::

    format: lenscluster-images/1
    defaults: {shape: [0.3734, 0.3734, 90.0], mag: 25.0}
    systems:
      "1":
        z: 3.238
        images:
          a: {ra: 64.0490842, dec: -24.0628619}
          b: {ra: 64.0469592, dec: -24.0607969, sigma: 0.03, parity: -1}
      "D1":
        z: 0.9397
        knots: [A, B]
        images:
          a:
            parity: -1
            knots:
              A: [64.0365743, -24.0670439]
              B: [64.0366543, -24.0672539, 0.02]
          b:
            knots:
              A: {ra: 64.0369043, dec: -24.0674839, bands: {F150W: [25.8, 0.06]}}

Fields.  System: ``z`` (required), ``reliability`` (0 to 1, default 1), ``knots`` (list of names), ``images``,
``notes``.  Image: ``ra``/``dec`` (degrees, ICRS) or ``knots``; ``parity`` (-1, 0 or +1, 0 = unknown); ``sigma``
(astrometric error in arcsec, the default for its knots); ``notes``.  Position (an image without knots, or one knot):
``ra``, ``dec``, ``sigma``, ``shape`` ([a, b, theta]), ``mag``, ``mag_err``, ``bands`` ({band: [mag, err]} or
{band: {mag, err, detected, use}}), ``notes``.  A knot may also be written ``[ra, dec]`` or ``[ra, dec, sigma]``.
``defaults`` sets ``sigma``, ``shape``, ``mag``, ``mag_err`` and ``reliability`` for every entry that omits them;
an astrometric error missing everywhere takes the configured ``sigma_inst_arcsec``.

The solver sees one family per (system, knot): its id is the system name followed by the knot name (D1 + A = D1A), and
each image label is ``<family>.<image>``.  System and knot names use capital letters and digits, image names lowercase
letters.  YAML reads unquoted 021 as a number and y/n as booleans, so names must be quoted when they could be read so;
``write_image_catalog`` quotes them.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

FORMAT = "lenscluster-images/1"
NAME_RE = re.compile(r"^[A-Z0-9]+$")
IMAGE_RE = re.compile(r"^[a-z]+$")
TOP_KEYS = {"format", "defaults", "systems", "notes"}
DEFAULT_KEYS = {"sigma", "shape", "mag", "mag_err", "reliability"}
SYSTEM_KEYS = {"z", "reliability", "knots", "images", "notes"}
POSITION_KEYS = {"ra", "dec", "sigma", "shape", "mag", "mag_err", "bands", "notes"}
IMAGE_KEYS = POSITION_KEYS | {"parity", "knots"}
KNOTTED_IMAGE_KEYS = {"parity", "sigma", "knots", "notes"}
BAND_KEYS = {"mag", "err", "detected", "use"}
COLUMNS = ["image_label", "family_id", "image_id", "system_id", "knot_id", "ra", "dec", "catalog_z", "family_reliability",
           "sigma_inst_arcsec", "parity", "catalog_a", "catalog_b", "catalog_theta", "catalog_mag", "catalog_mag_err",
           "catalog_source"]


class CatalogError(ValueError):
    """A malformed image catalog; the message names the file and the entry."""


# ----------------------------------------------------------------------------- reading
def load_image_catalog(path: str | Path, *, sigma_inst_arcsec: float, sigma_floor_arcsec: float = 0.0) -> pd.DataFrame:
    """One row per (system, knot, image) with the columns the solver uses (``COLUMNS``) plus ``mag_<band>``,
    ``mag_err_<band>``, ``detected_<band>`` and ``use_for_catalog_<band>`` for every band any entry lists.

    ``sigma_inst_arcsec`` is the astrometric error of entries that give none, here or in ``defaults``; every error is
    raised to at least ``sigma_floor_arcsec``."""
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise CatalogError(f"{path}: image catalogs are YAML files (.yaml); convert a Lenstool catalog with "
                           f"scripts/convert_lenstool_catalog.py")
    if not (math.isfinite(float(sigma_inst_arcsec)) and float(sigma_inst_arcsec) > 0.0):
        raise CatalogError("sigma_inst_arcsec must be finite and positive.")
    if not (math.isfinite(float(sigma_floor_arcsec)) and float(sigma_floor_arcsec) >= 0.0):
        raise CatalogError("sigma_floor_arcsec must be finite and nonnegative.")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path}: not valid YAML: {exc}") from exc
    rows = _flatten(document, str(path), float(sigma_inst_arcsec))
    table = pd.DataFrame(rows)
    table["sigma_inst_arcsec"] = np.maximum(table["sigma_inst_arcsec"].to_numpy(float), float(sigma_floor_arcsec))
    bands = [c for c in table.columns if c.startswith(("mag_", "detected_", "use_for_catalog_")) and c not in COLUMNS]
    table["catalog_source"] = str(path)
    return table.loc[:, COLUMNS + sorted(bands)].reset_index(drop=True)


def _where(source: str, *parts: str) -> str:
    return source + ": " + "".join(parts)


def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise CatalogError(f"{where} must be a mapping.")
    return value


def _check_keys(entry: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(str(k) for k in entry if k not in allowed)
    if unknown:
        raise CatalogError(f"{where}: unknown keys {unknown}; allowed {sorted(allowed)}.")


def _name(key: Any, pattern: re.Pattern, what: str, where: str) -> str:
    if not isinstance(key, str):
        raise CatalogError(f"{where}: {what} name {key!r} was read as {type(key).__name__}; quote it, e.g. \"{key}\".")
    if not pattern.fullmatch(key):
        raise CatalogError(f"{where}: {what} name {key!r} must match {pattern.pattern}.")
    return key


def _number(value: Any, where: str, *, positive: bool = False, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{where} must be a number; got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0) or (lower is not None and value < lower) or (upper is not None and value > upper):
        bounds = "positive" if positive else f"in [{lower}, {upper}]" if lower is not None or upper is not None else "finite"
        raise CatalogError(f"{where} must be {bounds}; got {value!r}.")
    return value


def _shape(value: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise CatalogError(f"{where} must be [a, b, theta].")
    return tuple(_number(v, f"{where}[{i}]") for i, v in enumerate(value))


def _bands(value: Any, where: str) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {}
    for band, entry in _require_mapping(value, where).items():
        if not isinstance(band, str) or not band:
            raise CatalogError(f"{where}: band names must be strings; got {band!r}.")
        here = f"{where}.{band}"
        if isinstance(entry, list):
            if len(entry) != 2:
                raise CatalogError(f"{here} must be [mag, err] or a mapping with {sorted(BAND_KEYS)}.")
            entry = {"mag": entry[0], "err": entry[1]}
        entry = _require_mapping(entry, here)
        _check_keys(entry, BAND_KEYS, here)
        mag = entry.get("mag")
        out[f"mag_{band}"] = np.nan if mag is None else _number(mag, f"{here}.mag")
        err = entry.get("err")
        out[f"mag_err_{band}"] = np.nan if err is None else _number(err, f"{here}.err", positive=True)
        for flag, column in (("detected", "detected_"), ("use", "use_for_catalog_")):
            if flag in entry:
                if not isinstance(entry[flag], bool):
                    raise CatalogError(f"{here}.{flag} must be true or false.")
                out[column + band] = entry[flag]
    return out


def _position(entry: Any, where: str, *, sigma: float, defaults: dict) -> dict[str, Any]:
    """RA, Dec and the measured quantities of one image or knot."""
    if isinstance(entry, list):
        if len(entry) not in (2, 3):
            raise CatalogError(f"{where} must be [ra, dec], [ra, dec, sigma] or a mapping.")
        entry = dict(zip(("ra", "dec", "sigma"), entry))
    entry = _require_mapping(entry, where)
    _check_keys(entry, POSITION_KEYS, where)
    for key in ("ra", "dec"):
        if key not in entry:
            raise CatalogError(f"{where}: missing {key}.")
    shape = _shape(entry["shape"], f"{where}.shape") if "shape" in entry else defaults.get("shape", (np.nan, np.nan, np.nan))
    row = {"ra": _number(entry["ra"], f"{where}.ra"), "dec": _number(entry["dec"], f"{where}.dec", lower=-90.0, upper=90.0),
           "sigma_inst_arcsec": _number(entry["sigma"], f"{where}.sigma", positive=True) if "sigma" in entry else sigma,
           "catalog_a": shape[0], "catalog_b": shape[1], "catalog_theta": shape[2],
           "catalog_mag": _number(entry["mag"], f"{where}.mag") if "mag" in entry else defaults.get("mag", np.nan),
           "catalog_mag_err": _number(entry["mag_err"], f"{where}.mag_err", positive=True) if "mag_err" in entry else defaults.get("mag_err", np.nan)}
    if "bands" in entry:
        row.update(_bands(entry["bands"], f"{where}.bands"))
    return row


def _defaults(value: Any, where: str, sigma_inst_arcsec: float) -> dict[str, Any]:
    entry = _require_mapping(value, where)
    _check_keys(entry, DEFAULT_KEYS, where)
    out: dict[str, Any] = {"sigma": sigma_inst_arcsec}
    if "sigma" in entry:
        out["sigma"] = _number(entry["sigma"], f"{where}.sigma", positive=True)
    if "shape" in entry:
        out["shape"] = _shape(entry["shape"], f"{where}.shape")
    if "mag" in entry:
        out["mag"] = _number(entry["mag"], f"{where}.mag")
    if "mag_err" in entry:
        out["mag_err"] = _number(entry["mag_err"], f"{where}.mag_err", positive=True)
    if "reliability" in entry:
        out["reliability"] = _number(entry["reliability"], f"{where}.reliability", lower=0.0, upper=1.0)
    return out


def _flatten(document: Any, source: str, sigma_inst_arcsec: float) -> list[dict[str, Any]]:
    top = _require_mapping(document, f"{source}: the file")
    _check_keys(top, TOP_KEYS, f"{source}: the file")
    if top.get("format") != FORMAT:
        raise CatalogError(f"{source}: format must be {FORMAT!r}; got {top.get('format')!r}.")
    defaults = _defaults(top.get("defaults", {}), _where(source, "defaults"), sigma_inst_arcsec)
    systems = _require_mapping(top.get("systems"), _where(source, "systems"))
    if not systems:
        raise CatalogError(f"{source}: no systems.")
    rows: list[dict[str, Any]] = []
    families: dict[str, str] = {}
    for system_key, system in systems.items():
        system_id = _name(system_key, NAME_RE, "system", _where(source, "systems"))
        here = _where(source, f"systems.{system_id}")
        system = _require_mapping(system, here)
        _check_keys(system, SYSTEM_KEYS, here)
        if "z" not in system:
            raise CatalogError(f"{here}: missing z.")
        z = _number(system["z"], f"{here}.z")
        reliability = (_number(system["reliability"], f"{here}.reliability", lower=0.0, upper=1.0) if "reliability" in system
                       else defaults.get("reliability", 1.0))
        declared = system.get("knots")
        if declared is not None:
            if not isinstance(declared, list) or not declared:
                raise CatalogError(f"{here}.knots must be a nonempty list of names.")
            declared = [_name(k, NAME_RE, "knot", f"{here}.knots") for k in declared]
            if len(set(declared)) != len(declared):
                raise CatalogError(f"{here}.knots has repeated names.")
        images = _require_mapping(system.get("images"), f"{here}.images")
        if not images:
            raise CatalogError(f"{here}: no images.")
        seen_knots: set[str] = set()
        for image_key, image in images.items():
            image_id = _name(image_key, IMAGE_RE, "image", f"{here}.images")
            where = f"{here}.images.{image_id}"
            if isinstance(image, list):
                image = dict(zip(("ra", "dec", "sigma"), image)) if len(image) in (2, 3) else image
            image = _require_mapping(image, where)
            parity = image.get("parity", 0)
            if parity not in (-1, 0, 1) or isinstance(parity, bool):
                raise CatalogError(f"{where}.parity must be -1, 0 or 1; got {parity!r}.")
            if "knots" in image:
                if declared is None:
                    raise CatalogError(f"{where} lists knots but {here} declares none; add knots: [...] to the system.")
                _check_keys(image, KNOTTED_IMAGE_KEYS, where)
                sigma = _number(image["sigma"], f"{where}.sigma", positive=True) if "sigma" in image else defaults["sigma"]
                knots = _require_mapping(image["knots"], f"{where}.knots")
                if not knots:
                    raise CatalogError(f"{where}.knots is empty.")
                entries = []
                for knot_key, knot in knots.items():
                    knot_id = _name(knot_key, NAME_RE, "knot", f"{where}.knots")
                    if knot_id not in declared:
                        raise CatalogError(f"{where}.knots.{knot_id} is not declared in {here}.knots {declared}.")
                    entries.append((knot_id, _position(knot, f"{where}.knots.{knot_id}", sigma=sigma, defaults=defaults)))
            else:
                if declared is not None:
                    raise CatalogError(f"{where} gives one position but {here} declares knots; list them under knots:.")
                _check_keys(image, IMAGE_KEYS - {"knots"}, where)
                entries = [("", _position({k: v for k, v in image.items() if k != "parity"}, where, sigma=defaults["sigma"], defaults=defaults))]
            for knot_id, measured in entries:
                seen_knots.add(knot_id)
                family_id = system_id + knot_id
                owner = families.setdefault(family_id, f"{system_id}/{knot_id}")
                if owner != f"{system_id}/{knot_id}":
                    raise CatalogError(f"{here}: family name {family_id} (system {system_id}, knot {knot_id or '-'}) collides with "
                                       f"system/knot {owner}; rename one of them.")
                rows.append({"image_label": f"{family_id}.{image_id}", "family_id": family_id, "image_id": image_id,
                             "system_id": system_id, "knot_id": knot_id, "catalog_z": z, "family_reliability": reliability,
                             "parity": int(parity), **measured})
        if declared is not None and set(declared) - seen_knots:
            raise CatalogError(f"{here}: knots {sorted(set(declared) - seen_knots)} are declared but appear in no image.")
    return rows


# ----------------------------------------------------------------------------- writing
class _Dumper(yaml.SafeDumper):
    """Block style down to each image, flow style inside it; strings that YAML would read as other types are quoted."""


def _represent_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_Dumper.add_representer(list, _represent_list)


class _Flow(dict):
    """A mapping written on one line."""


_Dumper.add_representer(_Flow, lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=True))
_Dumper.add_representer(dict, lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=False))
_Dumper.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=True))


def _flowed(system: dict) -> dict:
    """Images (and knots) on one line each; the rest in block style."""
    out = dict(system)
    images = {}
    for name, image in system["images"].items():
        if "knots" in image:
            image = dict(image)
            image["knots"] = {k: (v if isinstance(v, list) else _Flow(v)) for k, v in image["knots"].items()}
            images[name] = image
        else:
            images[name] = _Flow(image)
    out["images"] = images
    return out


def write_image_catalog(path: str | Path, systems: dict[str, dict], *, defaults: dict | None = None,
                        header: str | None = None) -> Path:
    """Write ``systems`` ({name: {z, images, ...}}, the structure ``load_image_catalog`` reads) as a catalog file.
    ``header`` becomes comment lines at the top.  The file is read back and checked before this returns."""
    path = Path(path)
    document: dict[str, Any] = {"format": FORMAT}
    if defaults:
        document["defaults"] = _Flow(defaults)
    document["systems"] = {str(name): _flowed(system) for name, system in systems.items()}
    text = yaml.dump(document, Dumper=_Dumper, sort_keys=False, width=200, allow_unicode=True, default_flow_style=False)
    comment = "".join(f"# {line}".rstrip() + "\n" for line in (header or "").splitlines())
    path.write_text(comment + text, encoding="utf-8")
    load_image_catalog(path, sigma_inst_arcsec=1.0)
    return path


def systems_from_offsets(rows, reference: tuple[float, float]) -> dict[str, dict]:
    """Systems without knots from rows of (system, image, x_arcsec, y_arcsec, z, extra) with offsets from ``reference``
    (ra0, dec0); ``extra`` is a dict of image fields (sigma, mag, mag_err, parity, ...).  For mocks and tests."""
    from .sky import offsets_to_radec
    systems: dict[str, dict] = {}
    for system, image, x, y, z, extra in rows:
        ra, dec = offsets_to_radec(float(x), float(y), float(reference[0]), float(reference[1]))
        entry = systems.setdefault(str(system), {"z": float(z), "images": {}})
        if entry["z"] != float(z):
            raise CatalogError(f"system {system} has two redshifts, {entry['z']} and {z}.")
        entry["images"][str(image)] = {"ra": float(ra), "dec": float(dec), **(extra or {})}
    return systems
