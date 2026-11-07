"""Convert a Lenstool potfile member catalog (``id x y a b theta mag lum color`` after ``#REFERENCE``) into the YAML member
catalog, optionally attaching stellar kinematics matched by position.

The kinematics table is whitespace columns with a ``# columns:`` comment naming them; ``ra``, ``dec``, ``sigma`` and ``err``
are required, ``sn`` and ``id`` are used when present.  Each row is matched to the nearest member within --match-radius
arcsec; the match must be one to one (a member hit twice, or a row with no member, stops the script unless
--allow-unmatched).  Roles and treatments are set by member id on the command line.

Example (the CANUCS M0416 members with the Granata et al. 2026 MUSE dispersions):
    python scripts/convert_member_catalog.py catalogs/m0416/canucs2025/source/members.cat \\
        catalogs/m0416/canucs2025/members.yaml --band F160W \\
        --kinematics catalogs/m0416/canucs2025/granata2026_m0416_kinematics.txt --source G26 \\
        --source-reference "Granata et al. 2026, A&A (arXiv:2603.26869), Appendix C" --aperture 0.8 \\
        --bcg 2526 --bcg 3158
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.member_catalog import members_from_lenstool, write_member_catalog  # noqa: E402


def read_kinematics(path: Path) -> pd.DataFrame:
    names = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# columns:"):
            names = line.split(":", 1)[1].split()
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split())
    if names is None:
        raise SystemExit(f"{path}: needs a '# columns: ...' line naming the columns.")
    table = pd.DataFrame(rows, columns=names)
    for column in ("ra", "dec", "sigma", "err", "sn"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="raise")
    missing = {"ra", "dec", "sigma", "err"} - set(table.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}.")
    return table


def match(members: dict[str, dict], kinematics: pd.DataFrame, radius: float, allow_unmatched: bool) -> dict[str, pd.Series]:
    """Member id -> kinematics row, nearest member within ``radius`` arcsec, one to one."""
    ids = list(members)
    ra = np.array([members[i]["ra"] for i in ids]); dec = np.array([members[i]["dec"] for i in ids])
    out: dict[str, pd.Series] = {}
    unmatched = []
    for _, row in kinematics.iterrows():
        d = np.hypot((ra - row["ra"]) * np.cos(np.radians(row["dec"])), dec - row["dec"]) * 3600.0
        j = int(np.argmin(d))
        if d[j] > radius:
            unmatched.append((row.get("id", "?"), float(d[j])))
            continue
        if ids[j] in out:
            raise SystemExit(f"member {ids[j]} matches two kinematics rows ({out[ids[j]].get('id', '?')} and {row.get('id', '?')}).")
        out[ids[j]] = row
        print(f"  kinematics {row.get('id', '?'):>5} -> member {ids[j]:>6} ({d[j]:.2f}\"), sigma {row['sigma']:.1f} +- {row['err']:.1f}")
    if unmatched and not allow_unmatched:
        raise SystemExit(f"{len(unmatched)} kinematics rows have no member within {radius}\": {unmatched}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("catalog", type=Path, help="Lenstool potfile member catalog (.cat)")
    parser.add_argument("output", type=Path, help="YAML member catalog to write")
    parser.add_argument("--band", required=True, help="band of the catalog magnitudes (the scaling-relation band)")
    parser.add_argument("--reference", nargs=2, type=float, metavar=("RA0", "DEC0"), default=(0.0, 0.0),
                        help="model reference, for a catalog in offsets (#REFERENCE 3)")
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--source", default="K1", help="kinematics source key")
    parser.add_argument("--source-reference", help="citation of the kinematics source")
    parser.add_argument("--quantity", default="sigma_ap", choices=("sigma_ap", "sigma_0", "sigma_e"))
    parser.add_argument("--aperture", type=float, help="aperture radius in arcsec (sigma_ap)")
    parser.add_argument("--match-radius", type=float, default=0.6)
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--bcg", action="append", default=[], help="member id with role bcg (repeatable)")
    parser.add_argument("--independent", action="append", default=[], help="member id modelled outside the relation")
    parser.add_argument("--exclude", action="append", default=[], help="member id kept in the file but not modelled")
    parser.add_argument("--keep", action="append", default=[], help="member id protected from the impact cut")
    parser.add_argument("--note", action="append", default=[], metavar="ID=TEXT", help="notes for a member")
    args = parser.parse_args()

    members = members_from_lenstool(args.catalog, (3, *args.reference), band=args.band)
    print(f"{len(members)} members from {args.catalog}")
    for flag, key, value in ((args.bcg, "role", "bcg"), (args.independent, "treatment", "independent"),
                             (args.exclude, "treatment", "exclude"), (args.keep, "keep", True)):
        for member in flag:
            if member not in members:
                raise SystemExit(f"member {member} is not in {args.catalog}.")
            members[member][key] = value
    for item in args.note:
        member, text = item.split("=", 1)
        if member not in members:
            raise SystemExit(f"member {member} is not in {args.catalog}.")
        members[member]["notes"] = text
    sources = None
    header = [f"Members from {args.catalog.name}, converted on {date.today().isoformat()} by scripts/convert_member_catalog.py."]
    if args.kinematics is not None:
        if not args.source_reference:
            raise SystemExit("--source-reference is required with --kinematics.")
        source = {"reference": args.source_reference, "quantity": args.quantity}
        if args.aperture is not None:
            source["aperture_arcsec"] = args.aperture
        sources = {args.source: source}
        matched = match(members, read_kinematics(args.kinematics), args.match_radius, args.allow_unmatched)
        for member, row in matched.items():
            measurement = {"source": args.source, "sigma": float(row["sigma"]), "err": float(row["err"])}
            if "sn" in row and np.isfinite(row["sn"]):
                measurement["sn"] = float(row["sn"])
            members[member]["kinematics"] = [measurement]
        header.append(f"Kinematics from {args.kinematics.name} ({args.source_reference}): {len(matched)} members matched by position "
                      f"within {args.match_radius}\".")
    write_member_catalog(args.output, members, band=args.band, kinematics_sources=sources, header="\n".join(header))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
