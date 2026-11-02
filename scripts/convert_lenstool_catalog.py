"""Convert a Lenstool multiple-image catalog into the YAML image catalog that lenscluster reads.

Lenstool rows are ``label x y a b theta z mag [reliability [mag_err]]`` after a ``#REFERENCE`` line, with labels
``FAMILY.image``.  Each family becomes one system without knots (merging families into one system with knots is an
edit made by hand afterwards).  ``a b theta`` are kept as the image shape; values shared by every row go to
``defaults``.  Optional per-image tables: the band magnitudes (columns image_label, mag_<band>, mag_err_<band>,
detected_<band>, use_for_catalog_<band>) and the measured astrometry (image_label, sigma_inst_arcsec, parity).

Example:
    python scripts/convert_lenstool_catalog.py data/Bergamini/M0416_Bergamini22/obs_arcs.cat \\
        data/Bergamini/M0416_Bergamini22/images.yaml
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.image_catalog import write_image_catalog  # noqa: E402
from lenscluster.sky import offsets_to_radec  # noqa: E402

LABEL_RE = re.compile(r"^([A-Z0-9]+)\.([a-z]+)$")
BASE = ["label", "x", "y", "a", "b", "theta", "z", "mag"]
OPTIONAL = ["reliability", "mag_err"]


def read_lenstool(path: Path) -> tuple[list[str], pd.DataFrame]:
    """The #REFERENCE tokens and one row per image with the raw tokens kept as text (positions keep their digits)."""
    header, rows = None, []
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
        if not len(BASE) <= len(tokens) <= len(BASE) + len(OPTIONAL):
            raise SystemExit(f"{path}: rows need {len(BASE)} to {len(BASE) + len(OPTIONAL)} columns; got {len(tokens)}: {line!r}")
        rows.append(dict(zip(BASE + OPTIONAL, tokens)))
    if not header:
        raise SystemExit(f"{path}: missing #REFERENCE line.")
    return header, pd.DataFrame(rows)


def uniform(values: pd.Series) -> float | None:
    """The common value of a numeric column, or None when the rows differ or any is missing."""
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any() or numbers.nunique() != 1:
        return None
    return float(numbers.iloc[0])


def convert(path: Path, *, reference: tuple[float, float] | None, bands: Path | None, astrometry: Path | None) -> tuple[dict, dict]:
    header, table = read_lenstool(path)
    kind = int(header[0])
    labels = table["label"].tolist()
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{path}: repeated image labels {sorted({l for l in labels if labels.count(l) > 1})}.")
    for label in labels:
        if not LABEL_RE.fullmatch(label):
            raise SystemExit(f"{path}: label {label!r} is not FAMILY.image (FAMILY [A-Z0-9]+, image [a-z]+).")
    if kind == 0:
        ra, dec = pd.to_numeric(table["x"]).to_numpy(), pd.to_numeric(table["y"]).to_numpy()
    elif kind == 3:
        if reference is None:
            raise SystemExit(f"{path}: #REFERENCE 3 holds offsets; pass --reference RA0 DEC0 (the model reference point).")
        ra, dec = offsets_to_radec(pd.to_numeric(table["x"]).to_numpy(), pd.to_numeric(table["y"]).to_numpy(), *reference)
    else:
        raise SystemExit(f"{path}: #REFERENCE {kind} is not supported (0 = RA/Dec, 3 = offsets).")

    extra = {label: {} for label in labels}
    for side_path, columns in ((bands, None), (astrometry, ("sigma_inst_arcsec", "parity"))):
        if side_path is None:
            continue
        side = pd.read_csv(side_path, dtype={"image_label": str})
        unknown = sorted(set(side["image_label"]) - set(labels))
        if unknown:
            raise SystemExit(f"{side_path}: labels not in the catalog: {unknown}")
        for row in side.to_dict("records"):
            target = extra[row["image_label"]]
            if columns is None:
                for column, value in row.items():
                    for prefix, key in (("mag_err_", "err"), ("mag_", "mag"), ("detected_", "detected"), ("use_for_catalog_", "use")):
                        if column.startswith(prefix) and not (isinstance(value, float) and math.isnan(value)):
                            band = column[len(prefix):]
                            value = bool(value) if key in ("detected", "use") else float(value)
                            target.setdefault("bands", OrderedDict()).setdefault(band, {})[key] = value
                            break
            else:
                if pd.notna(row.get("sigma_inst_arcsec")) and str(row.get("sigma_inst_arcsec")).strip():
                    target["sigma"] = float(row["sigma_inst_arcsec"])
                if pd.notna(row.get("parity")) and str(row.get("parity")).strip() and int(float(row["parity"])) != 0:
                    target["parity"] = int(float(row["parity"]))

    defaults: dict = {}
    shape = [uniform(table[c]) for c in ("a", "b", "theta")]
    if all(v is not None for v in shape):
        defaults["shape"] = shape
    for column, key in (("mag", "mag"), ("mag_err", "mag_err")):
        if column in table and uniform(table[column]) is not None:
            defaults[key] = uniform(table[column])
    if "reliability" in table and uniform(table["reliability"]) not in (None, 1.0):
        defaults["reliability"] = uniform(table["reliability"])

    systems: dict = OrderedDict()
    for i, row in table.iterrows():
        family, image = LABEL_RE.fullmatch(row["label"]).groups()
        z = float(row["z"])
        system = systems.setdefault(family, OrderedDict(z=z, images=OrderedDict()))
        if system["z"] != z:
            raise SystemExit(f"{path}: family {family} has two redshifts, {system['z']} and {z}.")
        if "reliability" in row and pd.notna(row["reliability"]) and "reliability" not in defaults:
            value = float(row["reliability"])
            if value != 1.0:
                if system.setdefault("reliability", value) != value:
                    raise SystemExit(f"{path}: family {family} has two reliabilities.")
        entry = OrderedDict(ra=float(ra[i]), dec=float(dec[i]))
        if "shape" not in defaults:
            entry["shape"] = [float(row["a"]), float(row["b"]), float(row["theta"])]
        if "mag" not in defaults and pd.notna(pd.to_numeric(row["mag"], errors="coerce")):
            entry["mag"] = float(row["mag"])
        if "mag_err" in row and "mag_err" not in defaults and pd.notna(pd.to_numeric(row["mag_err"], errors="coerce")):
            entry["mag_err"] = float(row["mag_err"])
        entry.update(extra[row["label"]])
        system["images"][image] = entry
    # z and reliability first, images last, for reading
    systems = OrderedDict((name, OrderedDict([(k, s[k]) for k in ("z", "reliability") if k in s] + [("images", s["images"])]))
                          for name, s in systems.items())
    return defaults, systems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path, help="Lenstool multiple-image catalog")
    parser.add_argument("output", type=Path, help="YAML catalog to write (.yaml)")
    parser.add_argument("--reference", type=float, nargs=2, metavar=("RA0", "DEC0"), help="model reference point, for #REFERENCE 3 input")
    parser.add_argument("--bands", type=Path, help="per-image band magnitudes CSV (default: <catalog stem>_band_magnitudes.csv if present)")
    parser.add_argument("--astrometry", type=Path, help="per-image CSV with sigma_inst_arcsec and/or parity")
    parser.add_argument("--note", default="", help="extra line for the file header")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit("the output must be a .yaml file")
    bands = args.bands
    if bands is None:
        candidate = args.catalog.with_name(f"{args.catalog.stem}_band_magnitudes.csv")
        bands = candidate if candidate.exists() else None
    defaults, systems = convert(args.catalog, reference=tuple(args.reference) if args.reference else None, bands=bands, astrometry=args.astrometry)
    header = (f"Multiple images converted from {args.catalog} on {date.today().isoformat()} by scripts/convert_lenstool_catalog.py.\n"
              "Positions are RA/Dec (ICRS, degrees); shape is the Lenstool a, b, theta."
              + (f"\nBand magnitudes from {bands}." if bands else "") + (f"\nAstrometry from {args.astrometry}." if args.astrometry else "")
              + (f"\n{args.note}" if args.note else ""))
    write_image_catalog(args.output, systems, defaults=defaults or None, header=header)
    n_images = sum(len(s["images"]) for s in systems.values())
    print(f"wrote {args.output}: {len(systems)} systems, {n_images} images" + (f", defaults {sorted(defaults)}" if defaults else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
