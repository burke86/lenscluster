"""Compare the images our M0416 lens models predict with the published images our catalog does not use: the CANUCS
non-gold images (silver, bronze, Q; Rihtarsic et al. 2025 full catalog) and the Diego et al. 2024 (PEARLS) images with no
CANUCS gold image within 0.5".

Input: one or more output folders of find_missing_images.py (one per lens model), each with predicted_images.csv (every
image the best posterior draw predicts for every catalog family) and missing_image_sites.csv (the sites scored against
the CANUCS photometry).

Each published image is assigned to one of our systems:
  - a CANUCS image by its label (K12b.3 -> K12);
  - a Diego image through its source: the gold systems of the other images of the same Diego source within 0.5"
    (majority vote; the source is unassigned when none of its images is in the gold catalog).
Then, per model, the nearest predicted image of that system gives the verdict: predicted (< 1"), near (1-2"), not
predicted (none within 2"), or untestable (the system is not in our catalog).  The nearest predicted image of any other
system with a compatible redshift (|dz| / (1 + z) < 0.05) is also recorded, which catches renumbered systems.

Example:
    python scripts/match_published_images.py --search nov4b=output/missing_images_nov6/nov4b \\
        --search nov5a_w500=output/missing_images_nov6/nov5a_w500 --out docs/m0416_image_candidates
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANUCS_FULL = ROOT / "data/CANUCS/M0416_Rihtarsic2025/hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt"
DIEGO = ROOT / "data/JWST/M0416_Diego2024_PEARLS/images.ecsv"
REFERENCE = (64.03821668520617, -24.06750122335985)      # CANUCS lenstool reference (param file), the model frame
GOLD_RADIUS = 0.5          # arcsec: a Diego image this close to a gold image is in our catalog
MATCH, NEAR = 1.0, 2.0     # arcsec: verdict thresholds on the distance to the nearest predicted image of the system
DZ = 0.05                  # |dz| / (1 + z) for a predicted image of another system to count as redshift compatible


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search", action="append", required=True, help="NAME=FOLDER of a find_missing_images.py output")
    p.add_argument("--out", required=True)
    return p.parse_args()


def to_model(ra, dec):
    ra0, dec0 = REFERENCE
    return -(np.asarray(ra, float) - ra0) * np.cos(np.radians(dec0)) * 3600.0, (np.asarray(dec, float) - dec0) * 3600.0


def system_of(label):
    m = re.match(r"^([A-Z]+\d+)", label)
    return m.group(1) if m else label


def load_canucs():
    rows = []
    for line in CANUCS_FULL.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        rows.append(dict(label=t[0], ra=float(t[1]), dec=float(t[2]), z=float(t[3]), cls=t[-1]))
    df = pd.DataFrame(rows)
    df["x"], df["y"] = to_model(df["ra"], df["dec"])
    df["system"] = df["label"].map(system_of)
    return df


def load_diego():
    from astropy.table import Table
    t = Table.read(DIEGO)
    val = lambda c: np.asarray(t[c].filled(np.nan) if hasattr(t[c], "filled") else t[c], float)
    df = pd.DataFrame(dict(label=[f"D{a}" for a in t["Arc"]], source=[str(a).split(".")[0] for a in t["Arc"]],
                           ra=np.asarray(t["RAJ2000"], float), dec=np.asarray(t["DEJ2000"], float), zsp=val("zsp"),
                           zgeom=val("zgeom"), zph=val("zph"), mu=val("mu"), rank=[str(r) for r in t["Rank"]]))
    df["x"], df["y"] = to_model(df["ra"], df["dec"])
    # One redshift per Diego source: the spec-z of any of its images, else its geometric redshift.
    zsp = df.groupby("source")["zsp"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    zg = df.groupby("source")["zgeom"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    df["z"], df["z_type"] = zsp.fillna(zg), np.where(zsp.notna(), "spec", np.where(zg.notna(), "geom", "none"))
    return df


def nearest(xy, table):
    if not len(table):
        return None, np.inf
    d = np.hypot(table["x"].to_numpy() - xy[0], table["y"].to_numpy() - xy[1])
    j = int(np.argmin(d))
    return table.iloc[j], float(d[j])


def verdict(system_known, d, on_catalog):
    if not system_known:
        return "untestable"
    if d < MATCH and on_catalog:          # the nearest predicted image is one of our catalog images: no independent test
        return "on catalog image"
    return "predicted" if d < MATCH else ("near" if d < NEAR else "not predicted")


def main():
    args = parse_args()
    searches = {}
    for item in args.search:
        name, folder = item.split("=", 1)
        pred = pd.read_csv(Path(folder) / "predicted_images.csv")
        sites = pd.read_csv(Path(folder) / "missing_image_sites.csv")
        searches[name] = (pred, sites)
    our_systems = set().union(*(set(p["system"]) for p, _ in searches.values()))
    sys_z = pd.concat([p[["system", "z_source"]] for p, _ in searches.values()]).groupby("system")["z_source"].first().to_dict()

    canucs = load_canucs()
    gold = canucs[canucs["cls"] == "G"]
    diego = load_diego()
    dg = np.hypot(diego["x"].to_numpy()[:, None] - gold["x"].to_numpy()[None], diego["y"].to_numpy()[:, None] - gold["y"].to_numpy()[None])
    diego["gold_label"] = np.where(dg.min(1) < GOLD_RADIUS, gold["label"].to_numpy()[dg.argmin(1)], "")
    diego["gold_dist"] = dg.min(1)
    # Diego source -> our system, by the gold images of its other images.
    vote = diego[diego["gold_label"] != ""].assign(system=lambda d: d["gold_label"].map(system_of)).groupby("source")["system"]
    source_system = vote.agg(lambda s: s.value_counts().index[0]).to_dict()
    nongold = canucs[canucs["cls"] != "G"]
    dc = np.hypot(diego["x"].to_numpy()[:, None] - nongold["x"].to_numpy()[None], diego["y"].to_numpy()[:, None] - nongold["y"].to_numpy()[None])
    diego["canucs_nongold"] = np.where(dc.min(1) < GOLD_RADIUS, [f"{l} ({c})" for l, c in zip(nongold["label"].to_numpy()[dc.argmin(1)],
                                                                                          nongold["cls"].to_numpy()[dc.argmin(1)])], "")

    published = []
    for r in nongold.itertuples():
        published.append(dict(catalog="CANUCS R25", label=r.label, cls=r.cls, ra=r.ra, dec=r.dec, x=r.x, y=r.y, z=r.z, z_type="system",
                              system=r.system, assigned_by="label", also_in=""))
    extra = diego[diego["gold_label"] == ""]
    for r in extra.itertuples():
        published.append(dict(catalog="Diego24", label=r.label, cls=r.rank, ra=r.ra, dec=r.dec, x=r.x, y=r.y, z=r.z, z_type=r.z_type,
                              system=source_system.get(r.source, ""), assigned_by="source" if r.source in source_system else "",
                              also_in=r.canucs_nongold))
    # The CANUCS non-gold images that Diego also lists.
    for row in published:
        if row["catalog"] == "CANUCS R25":
            d = np.hypot(extra["x"] - row["x"], extra["y"] - row["y"])
            if len(d) and d.min() < GOLD_RADIUS:
                k = int(np.argmin(d.to_numpy()))
                row["also_in"] = f"{extra['label'].iloc[k]} ({extra['rank'].iloc[k]})"

    # The CANUCS DR1 source at each published position (the photometry of find_missing_images.py).
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from find_missing_images import load_photometry
    phot = load_photometry((None, *REFERENCE))
    phot = phot[phot["FLAG_BCG"] == 0]
    for row in published:
        src, d = nearest((row["x"], row["y"]), phot)
        ok = src is not None and d < GOLD_RADIUS
        row.update(dr1_id=int(src["SOURCE"]) if ok else -1, dr1_dist=round(d, 3) if ok else np.nan,
                   dr1_F150W=round(float(src["m_F150W"]), 2) if ok else np.nan, dr1_zphot=round(float(src["Z_ML"]), 2) if ok else np.nan,
                   dr1_z16=round(float(src["Z160"]), 2) if ok else np.nan, dr1_z84=round(float(src["Z840"]), 2) if ok else np.nan,
                   dr1_zspec=float(src["Z_SPEC"]) if ok and src["Z_SPEC"] > 0 else np.nan)

    out_rows = []
    for row in published:
        known = row["system"] in our_systems
        z = row["z"] if np.isfinite(row["z"]) else sys_z.get(row["system"], np.nan)
        # A Diego source whose redshift disagrees with the system it was assigned to is not that system.
        zs = sys_z.get(row["system"], np.nan)
        # Only a spectroscopic redshift overrides the assignment; Diego's geometric redshifts are model predictions.
        differs = bool(known and np.isfinite(row["z"]) and abs(row["z"] - zs) / (1 + zs) > DZ)
        row["z_conflict"] = differs and row["z_type"] in ("spec", "system")
        row["zgeom_differs"] = differs and row["z_type"] == "geom"
        if row["z_conflict"]:
            known = False
        row = dict(row, our_system=row["system"] if known else "", z_system=sys_z.get(row["system"], np.nan))
        verdicts = []
        for name, (pred, sites) in searches.items():
            same = pred[pred["system"] == row["system"]]
            best, d = nearest((row["x"], row["y"]), same)
            v = verdict(known, d, best is not None and not best["missing"])
            row[f"{name}_dist"] = round(d, 3) if np.isfinite(d) else np.nan
            row[f"{name}_family"] = best["family"] if best is not None else ""
            row[f"{name}_mu"] = round(float(best["mu"]), 2) if best is not None else np.nan
            # The predicted image falls on one of our catalog images (a knot or a neighbouring image of the system).
            row[f"{name}_on_catalog_image"] = bool(best is not None and not best["missing"])
            # The scored site of this predicted image, when it is a missing one.
            site_note = ""
            if best is not None and best["missing"]:
                s, ds = nearest((best["x"], best["y"]), sites[sites["system"] == row["system"]])
                if s is not None and ds < 2.5:
                    site_note = f"grade {s['grade']}, best CANUCS {int(s['best_source'])} at {s['best_dist']:.2f}\""
            row[f"{name}_site"] = site_note
            other = pred[(pred["system"] != row["system"]) & (np.abs(pred["z_source"] - z) / (1 + pred["z_source"]) < DZ)] if np.isfinite(z) else pred.iloc[:0]
            ob, do = nearest((row["x"], row["y"]), other)
            row[f"{name}_other_system"] = f"{ob['family']} (z {ob['z_source']:.3f}) at {do:.2f}\"" if ob is not None and do < MATCH else ""
            row[f"{name}_verdict"] = v
            verdicts.append(v)
        row["n_predicted"] = sum(v == "predicted" for v in verdicts)
        row["n_models"] = len(verdicts)
        row["verdict"] = ("z conflict" if row["z_conflict"] else "untestable" if not known
                          else "predicted by all" if row["n_predicted"] == len(verdicts)
                          else "predicted by some" if row["n_predicted"] else "on catalog image" if "on catalog image" in verdicts
                          else "near" if "near" in verdicts else "not predicted")
        out_rows.append(row)
    table = pd.DataFrame(out_rows)
    order = {"predicted by all": 0, "predicted by some": 1, "near": 2, "not predicted": 3, "on catalog image": 4, "z conflict": 5, "untestable": 6}
    table = table.assign(_k=table["verdict"].map(order)).sort_values(["catalog", "_k", "system", "label"]).drop(columns="_k")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "published_image_matches.csv", index=False, float_format="%.6g")

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
    names = list(searches)
    for cat in ("CANUCS R25", "Diego24"):
        t = table[table["catalog"] == cat]
        print(f"\n== {cat}: {len(t)} images not in our catalog")
        print(t["verdict"].value_counts().to_string())
        cols = ["label", "cls", "system", "z", "z_system", "zgeom_differs", "also_in", "dr1_id", "dr1_zphot"] + [f"{n}_dist" for n in names] + [f"{n}_on_catalog_image" for n in names[:1]] + \
            [f"{n}_site" for n in names[:1]] + ["verdict"]
        print(t[t["verdict"] != "untestable"][cols].round(2).to_string(index=False))
        other = t[(t["verdict"] == "untestable") & (t[[f"{n}_other_system" for n in names]] != "").any(axis=1)]
        if len(other):
            print(f"-- untestable, but a predicted image of another system with a compatible z lies within {MATCH}\":")
            print(other[["label", "cls", "z", "also_in"] + [f"{n}_other_system" for n in names]].to_string(index=False))
    print(f"\n[done] {out / 'published_image_matches.csv'}")


if __name__ == "__main__":
    main()
