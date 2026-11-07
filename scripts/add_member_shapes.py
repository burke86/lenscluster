"""Add light shapes from the CANUCS DR1 photometry catalog to a member YAML file, in place.

Each member is matched to the catalog by position (within --max-sep, default 0.3"); a match writes
``shape: [a, b, theta]`` with a and b the catalog's semi-axes in arcsec (A, B times the 0.04" pixel) and theta its
position angle PA in degrees.  CANUCS measures PA counter-clockwise from the image x-axis of north-up mosaics, whose
+x points west, the same axis as the model's x offset, so PA is Lenstool's angle_pos without conversion (checked on
M0416 against HFF-DeepSpace: PA_canucs = 180 - theta_Shipley to 4 deg for 64 elongated members).  For the bright
galaxies CANUCS models and subtracts (FLAG_BCG) the shape is that of the model.  Unmatched members stay round.  The
edit is textual, so comments and layout survive; the old file is copied to --backup first.

    python scripts/add_member_shapes.py catalogs/m0416/canucs2025/members.yaml \
        data/CANUCS/M0416_Rihtarsic2025/canucs_dr1/hlsp_canucs_jwst-hst_multi_macs0416-clu_multi_v1_photometry-cat.fits.gz
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

PIXEL_ARCSEC = 0.04


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("members", type=Path)
    p.add_argument("catalog", type=Path)
    p.add_argument("--max-sep", type=float, default=0.3, help="match radius in arcsec")
    p.add_argument("--backup", type=Path, default=None, help="copy of the old file (default: <name>.round.yaml)")
    args = p.parse_args()

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits

    text = args.members.read_text()
    if re.search(r"^\s+shape:|shape: \[", text, flags=re.M):
        raise SystemExit(f"{args.members} already has shapes; restore the round file first")
    members = yaml.safe_load(text)["members"]
    ids = list(members)
    table = fits.open(args.catalog)[1].data
    catalog = SkyCoord(np.asarray(table["RA"], float) * u.deg, np.asarray(table["DEC"], float) * u.deg)
    own = SkyCoord([members[i]["ra"] for i in ids] * u.deg, [members[i]["dec"] for i in ids] * u.deg)
    index, separation, _ = own.match_to_catalog_sky(catalog)
    separation = separation.arcsec

    lines = text.splitlines(keepends=True)
    out, matched, unmatched = [], 0, []
    current = None
    for line in lines:
        key = re.match(r"^  '?([^':\s]+)'?:(.*)$", line)
        if key and key.group(1) in members and not line.startswith("   "):
            current = key.group(1)
            k = ids.index(current)
            shape = None
            if separation[k] < args.max_sep:
                row = table[index[k]]
                a, b, pa = float(row["A"]) * PIXEL_ARCSEC, float(row["B"]) * PIXEL_ARCSEC, float(row["PA"])
                if np.isfinite([a, b, pa]).all() and a > 0 and b > 0:
                    shape = f"shape: [{a:.4f}, {b:.4f}, {pa:.2f}]"
            if shape is None:
                unmatched.append(current)
            else:
                matched += 1
            rest = key.group(2).rstrip("\n")
            if rest.strip().startswith("{") and shape is not None:      # inline entry: add the shape inside the braces
                assert rest.rstrip().endswith("}"), line
                line = line[:line.rindex("}")] + f", {shape}}}" + line[line.rindex("}") + 1:]
                current = None
            elif shape is not None:                                       # block entry: add a line after the key
                out.append(line)
                out.append(f"    {shape}\n")
                current = None
                continue
        out.append(line)
    header = (f"# Light shapes (a, b in arcsec, PA in deg = angle_pos) from the CANUCS DR1 photometry catalog ({args.catalog.name}), "
              f"matched within {args.max_sep}\" by scripts/add_member_shapes.py: {matched} of {len(ids)} members; the rest are round.\n")
    backup = args.backup or args.members.with_name(args.members.stem + ".round.yaml")
    shutil.copyfile(args.members, backup)
    new_text = header + "".join(out)
    check = yaml.safe_load(new_text)["members"]
    assert list(check) == ids and all(("shape" in check[i]) == (i not in unmatched) for i in ids)
    for i in ids:                                                         # nothing but the shape changed
        assert {k: v for k, v in check[i].items() if k != "shape"} == members[i], i
    args.members.write_text(new_text)
    print(f"{args.members}: shapes for {matched} of {len(ids)} members (old file kept as {backup}); round: {', '.join(unmatched)}")


if __name__ == "__main__":
    main()
