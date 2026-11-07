"""Group the knots of the Bergamini et al. (2022) MACS J0416 catalog into systems.

Bergamini names knots of one galaxy with a shared prefix and a final digit (0121 ... 0126).  Families are merged into one
system when they (1) have the same redshift, (2) share the name without its last character, and (3) every image letter
they share sits within ``--max-separation`` arcsec in both families, checked against every family already in the group.
The system takes the shared prefix as its name and each family's last character as its knot, so the flattened family ids
the solver sees are the Bergamini ids unchanged.  These are guesses: check them with scripts/make_knot_cutouts.py.

Example:
    python scripts/group_bergamini_knots.py catalogs/m0416/bergamini2023/images.yaml \\
        catalogs/m0416/bergamini2023/images.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.image_catalog import load_image_catalog, write_image_catalog  # noqa: E402
from lenscluster.sky import radec_to_offsets_arcsec  # noqa: E402


def separations(a: dict, b: dict, dec0: float) -> list[float]:
    """Arcsec between the images two families share, by letter."""
    out = []
    for letter in sorted(set(a) & set(b)):
        x, y = radec_to_offsets_arcsec(np.array([a[letter]["ra"], b[letter]["ra"]]), np.array([a[letter]["dec"], b[letter]["dec"]]),
                                       a[letter]["ra"], dec0)
        out.append(float(np.hypot(x[1] - x[0], y[1] - y[0])))
    return out


def group(systems: dict, max_separation: float) -> tuple[list[list[str]], list[str]]:
    """Merged groups (lists of family names, two or more) and a log line per decision."""
    dec0 = float(np.mean([im["dec"] for s in systems.values() for im in s["images"].values()]))
    candidates: dict[tuple[float, str], list[str]] = {}
    for name, system in systems.items():
        if "knots" in system:
            raise SystemExit(f"system {name} already has knots; group a catalog without knots.")
        if len(name) > 1:
            candidates.setdefault((float(system["z"]), name[:-1]), []).append(name)
    groups, log = [], []
    for (z, prefix), names in candidates.items():
        if len(names) < 2:
            continue
        open_groups: list[list[str]] = []
        for name in sorted(names):
            for g in open_groups:
                checks = [separations(systems[name]["images"], systems[other]["images"], dec0) for other in g]
                if all(c and max(c) <= max_separation for c in checks):
                    g.append(name)
                    break
            else:
                open_groups.append([name])
        for g in open_groups:
            if len(g) > 1:
                groups.append(g)
                log.append(f"merged  {prefix:>5} z={z}: knots {', '.join(n[-1] for n in g)} ({', '.join(g)})")
            else:
                worst = [max(separations(systems[g[0]]["images"], systems[o]["images"], dec0) or [np.inf]) for o in names if o != g[0]]
                log.append(f"kept    {g[0]:>5} z={z}: shares prefix {prefix} but is {min(worst):.1f}\" from the closest sibling in a shared image")
    return groups, log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path, help="YAML catalog with one system per Bergamini family (no knots)")
    parser.add_argument("output", type=Path, help="YAML catalog to write; may be the input")
    parser.add_argument("--max-separation", type=float, default=5.0, help="arcsec between same-letter images of two knots (default 5)")
    args = parser.parse_args(argv)
    text = args.catalog.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    header_in = "".join(line[2:] + "\n" for line in text.splitlines() if line.startswith("# "))
    flat = load_image_catalog(args.catalog, sigma_inst_arcsec=1.0)
    systems = document["systems"]
    groups, log = group(systems, args.max_separation)

    merged = {name for g in groups for name in g}
    out: dict = {}
    for name, system in systems.items():                     # catalog order; a group appears where its first knot did
        if name not in merged:
            out[name] = system
            continue
        g = next(g for g in groups if name in g)
        if g[0] != name:
            continue
        if name[:-1] in out or (name[:-1] in systems and name[:-1] not in merged):
            raise SystemExit(f"system name {name[:-1]} for knots {g} is already a system; rename by hand.")
        letters = sorted({letter for n in g for letter in systems[n]["images"]})
        images = {letter: {"knots": {n[-1]: systems[n]["images"][letter] for n in g if letter in systems[n]["images"]}} for letter in letters}
        out[name[:-1]] = {"z": system["z"], **({"reliability": system["reliability"]} if "reliability" in system else {}),
                          "knots": [n[-1] for n in g], "images": images}
    header = (header_in.rstrip("\n") + "\n" if header_in else "") + (
        f"Knots grouped on {date.today().isoformat()} by scripts/group_bergamini_knots.py: same z, same name prefix, shared images "
        f"within {args.max_separation:g} arcsec.\nKnot assignments are guesses from the naming and the positions; check them with "
        "scripts/make_knot_cutouts.py.\nSystem + knot reproduces each Bergamini family id (012 + 2 = 0122).")
    write_image_catalog(args.output, out, defaults=document.get("defaults"), header=header)

    grouped = load_image_catalog(args.output, sigma_inst_arcsec=1.0)
    key = ["image_label", "family_id", "ra", "dec", "catalog_z", "catalog_a", "catalog_b", "catalog_theta", "catalog_mag", "family_reliability"]
    a = flat.loc[:, key].sort_values("image_label").reset_index(drop=True)
    b = grouped.loc[:, key].sort_values("image_label").reset_index(drop=True)
    if not a.equals(b):
        raise SystemExit("the grouped catalog does not reproduce the flat one; nothing should change for the solver.")
    print("\n".join(log))
    print(f"wrote {args.output}: {len(out)} systems ({len(groups)} with knots, {sum(len(g) for g in groups)} knots), "
          f"{len(grouped)} images; families and positions identical to the input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
