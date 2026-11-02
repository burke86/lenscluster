"""Cluster-member catalogs: whitespace columns ``id x y a b theta mag lum color`` after a ``#REFERENCE`` line.

``#REFERENCE 0`` means x, y are RA and Dec in degrees; ``#REFERENCE 3`` means arcsecond offsets from the model's
reference point (``sky.radec_to_offsets_arcsec`` convention).  Other ``#`` lines are comments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sky import offsets_to_radec

COLUMNS = ["id", "coord_1", "coord_2", "a", "b", "theta", "mag", "lum", "color"]


def load_member_catalog(path: str | Path, reference: tuple[int, float, float]) -> pd.DataFrame:
    """Columns: id, ra, dec, catalog_a, catalog_b, catalog_theta, catalog_mag, catalog_lum, catalog_color,
    catalog_z (always NaN), catalog_reference, catalog_source.  ``reference`` is the model's (kind, ra0, dec0)."""
    path = Path(path)
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
    keep = ["id", "catalog_reference", "catalog_source", "ra", "dec", "catalog_a", "catalog_b", "catalog_theta", "catalog_mag",
            "catalog_z", "catalog_lum", "catalog_color"]
    return table.loc[:, keep].drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
