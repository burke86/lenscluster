"""Convert the CANUCS MACS J0416 multiple-image catalogue (Rihtarsic et al. 2025) into the YAML image catalog, with knots.

Structure comes from the published labels ``<system><knot>.<image>`` (K6c.2 = system K6, knot c, image 2; systems without a
knot letter have one knot).  Images are kept by class (default gold).  Positions and redshifts are taken from the CANUCS
lenstool input (``...lenstool-multim...``), matched to the catalogue by position: they are the catalogue positions shifted by
a constant offset (about 20 mas), the frame of the CANUCS lens model and of its member catalogue, and the redshifts carry
the spectroscopic precision (0.9397, not 0.940).  The match must be one to one with a constant offset and identical system
number, knot and image, or the script stops.

Image numbers are reconciled across the knots of a system by position: each knot's images are assigned to the images of
the knot with the most images (Hungarian matching, within --max-image-offset arcsec).  A knot whose numbering disagrees
(K6e: 1 and 3 swapped) is relabelled and the published label kept in the entry's notes.  Image n becomes letter n (1 -> a).

Example:
    python scripts/convert_canucs_catalog.py catalogs/m0416/canucs2025/images.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.image_catalog import load_image_catalog, write_image_catalog  # noqa: E402
from lenscluster.sky import radec_to_offsets_arcsec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "catalogs/m0416/canucs2025"
CATALOG = ROOT / "source" / "hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt"
LENSTOOL = ROOT / "source" / "hlsp_canucs_jwst-hst_multi_macs0416-lenstool-multim_multi_v1_model.txt"
LABEL = re.compile(r"^([A-Z]+[0-9]+)([a-z]?)\.([0-9]+)$")


def read_inputs(classes: set[str]) -> pd.DataFrame:
    cat = pd.read_csv(CATALOG, sep=r"\s+", comment="#", names=["id", "ra_cat", "dec_cat", "sysz", "musez", "nirissz", "nirspecz", "cls"],
                      dtype={"id": str})
    cat = cat[cat.cls.isin(classes)].reset_index(drop=True)
    parts = cat.id.str.extract(LABEL)
    if parts.isna().any(axis=None):
        raise SystemExit(f"labels not of the form <system><knot>.<image>: {cat.id[parts.isna().any(axis=1)].tolist()}")
    cat["system"], cat["knot"], cat["number"] = parts[0], parts[1].str.upper(), parts[2].astype(int)
    rows = [line.split() for line in LENSTOOL.read_text().splitlines() if line.strip() and not line.startswith("#")]
    lt = pd.DataFrame({"lt_id": [r[0] for r in rows], "ra": [float(r[1]) for r in rows], "dec": [float(r[2]) for r in rows],
                       "z": [float(r[6]) for r in rows]})
    # Match every lenstool image to its catalogue row: nearest neighbour, then check a constant offset and the labels.
    x_c, y_c = radec_to_offsets_arcsec(cat.ra_cat, cat.dec_cat, 64.0382, -24.0675)
    x_l, y_l = radec_to_offsets_arcsec(lt.ra, lt.dec, 64.0382, -24.0675)
    j = np.hypot(x_l[:, None] - x_c[None], y_l[:, None] - y_c[None]).argmin(1)
    if len(set(j)) != len(lt) or (classes == {"G"} and len(lt) != len(cat)):
        raise SystemExit("the lenstool catalogue does not match the selected classes one to one")
    shift = np.column_stack([x_l - x_c[j], y_l - y_c[j]])
    if np.max(np.abs(shift - np.median(shift, axis=0))) > 1e-3:
        raise SystemExit(f"the lenstool positions are not a constant shift of the catalogue (spread {np.ptp(shift, axis=0)} arcsec)")
    lt_parts = lt.lt_id.str.extract(r"^[A-Z]+([0-9]+)([a-z]?)\.([0-9]+)$")
    cat_parts = cat.id.str.extract(r"^[A-Z]+([0-9]+)([a-z]?)\.([0-9]+)$").iloc[j].reset_index(drop=True)
    if not (lt_parts == cat_parts).all(axis=None):
        raise SystemExit("matched lenstool and catalogue labels disagree in number, knot or image")
    out = cat.iloc[j].reset_index(drop=True)
    out[["ra", "dec", "z", "lt_id"]] = lt[["ra", "dec", "z", "lt_id"]]
    print(f"matched {len(out)} images; lenstool = catalogue + ({np.median(shift[:, 0]) * 1e3:+.1f}, {np.median(shift[:, 1]) * 1e3:+.1f}) mas "
          f"(x west, y north), spread {np.ptp(shift, axis=0).max() * 1e3:.2f} mas")
    return out.sort_values(["system", "knot", "number"], key=lambda s: s.map(_natural) if s.name == "system" else s).reset_index(drop=True)


def _natural(name: str):
    m = re.fullmatch(r"([A-Z]+)([0-9]+)", name)
    return (m.group(1), int(m.group(2)))


def reconcile(system: pd.DataFrame, max_offset: float) -> tuple[pd.Series, list[str]]:
    """Image number per row, made consistent across the knots of one system by position; and a log of changes."""
    numbers = system.number.copy()
    knots = list(dict.fromkeys(system.knot))
    if len(knots) < 2:
        return numbers, []
    reference = max(knots, key=lambda k: (system.knot == k).sum())
    ref = system[system.knot == reference]
    xr, yr = radec_to_offsets_arcsec(ref.ra, ref.dec, 64.0382, -24.0675)
    log = []
    for k in knots:
        if k == reference:
            continue
        rows = system[system.knot == k]
        x, y = radec_to_offsets_arcsec(rows.ra, rows.dec, 64.0382, -24.0675)
        cost = np.hypot(x[:, None] - xr[None], y[:, None] - yr[None])
        r, c = linear_sum_assignment(cost)
        if cost[r, c].max() > max_offset:
            raise SystemExit(f"system {system.system.iloc[0]} knot {k}: an image is {cost[r, c].max():.1f}\" from every image of knot {reference}")
        assigned = ref.number.to_numpy()[c]
        for (idx, old), new in zip(rows.number.items(), assigned):
            if old != new:
                numbers[idx] = new
                log.append(f"{system.system.iloc[0]}{k.lower()}.{old} -> image {new}")
    return numbers, log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--classes", nargs="+", default=["G"], help="catalogue classes to keep (G, S, B, Q); default gold")
    parser.add_argument("--max-image-offset", type=float, default=8.0, help="arcsec between matched images of two knots")
    args = parser.parse_args(argv)
    if set(args.classes) != {"G"}:
        raise SystemExit("only the gold class has lenstool positions; other classes need the catalogue frame, not supported here")
    table = read_inputs(set(args.classes))
    systems: dict = OrderedDict()
    changes: list[str] = []
    for name, rows in table.groupby("system", sort=False):
        if rows.knot.nunique() > 1 and (rows.knot == "").any():
            raise SystemExit(f"system {name} mixes knotted and plain labels")
        z = rows.z.unique()
        if len(z) != 1:
            raise SystemExit(f"system {name} has redshifts {z}")
        numbers, log = reconcile(rows, args.max_image_offset)
        changes += log
        r0 = rows.iloc[0]
        spec = [f"{label} {v:g}" for label, v in (("MUSE", r0.musez), ("NIRISS", r0.nirissz), ("NIRSpec", r0.nirspecz)) if v > 0]
        entry = OrderedDict(z=float(z[0]))
        if spec:
            entry["notes"] = "spectroscopic z: " + ", ".join(spec)
        knotted = (rows.knot != "").all()
        if knotted:
            entry["knots"] = list(dict.fromkeys(rows.knot))
        images: dict = OrderedDict()
        for (_, row), number in zip(rows.iterrows(), numbers):
            letter = chr(ord("a") + int(number) - 1)
            position = OrderedDict(ra=float(row.ra), dec=float(row.dec))
            if int(number) != int(row.number):
                position["notes"] = f"catalogue label {row.id}"
            if knotted:
                images.setdefault(letter, OrderedDict(knots=OrderedDict()))["knots"][row.knot] = position
            else:
                images[letter] = position
        entry["images"] = OrderedDict(sorted(images.items()))
        systems[name] = entry
    header = (f"CANUCS multiple images of MACS J0416 (Rihtarsic et al. 2025, A&A 696, A15; MAST DOI 10.17909/ph4n-6n76), class "
              f"{' '.join(args.classes)}, converted on {date.today().isoformat()} by scripts/convert_canucs_catalog.py.\n"
              "System, knot and image from the published labels (K6c.2 = system K6, knot C, image b).\n"
              "Positions and z from the CANUCS lenstool input: the catalogue shifted by about 20 mas, the frame of the CANUCS model "
              "and of members.cat.\n"
              + ("Image numbers reconciled across knots by position: " + "; ".join(changes) + "." if changes else "No image renumbering was needed."))
    write_image_catalog(args.output, systems, header=header)
    loaded = load_image_catalog(args.output, sigma_inst_arcsec=1.0)
    n_knotted = sum(1 for s in systems.values() if "knots" in s)
    print(f"wrote {args.output}: {len(systems)} systems ({n_knotted} with knots), {loaded.family_id.nunique()} knot-families, {len(loaded)} images")
    for line in changes:
        print("  renumbered", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
