"""Search for the images our lens model predicts but the catalog lacks, in M0416.

For every family of the catalog, the model source of the best posterior draw is traced through the field (grid of
sign changes + Newton, as in predict_counter_images.py); the other draws refine each image from the best-draw position,
which gives the positional spread.  A predicted image with no catalog image of its family within 1" (or of another knot
of the same system within 2") is a missing image; missing images of the knots of one system within 2.5" are merged into
one site.  For each site the script

  - predicts the F150W magnitude from the catalog images' photometry (CANUCS DR1) and the magnification ratio,
  - scores the CANUCS DR1 sources around it on distance, magnitude, colours (relative to F150W, against the median of the
    system's catalog images) and photo-z / spec-z against the system redshift,
  - cross-matches the site with the published image catalogs: CANUCS full catalog (gold/silver/bronze/Q, Rihtarsic et al.
    2025, and their tables A1-A3), Bergamini et al. 2023 and Caminha et al. 2017,
  - and writes colour stamps of the candidates not found before.

Example:
    python scripts/find_missing_images.py --run RUN --out output/missing_images
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_arc_source as fas  # noqa: E402  (sets up paths, x64 and the working directory)
from predict_counter_images import lens_at, stamp_rgb  # noqa: E402

jnp, np, pd, cutouts = fas.jnp, fas.np, fas.pd, fas.cutouts

SYNC = next((p for p in (Path("/Users/isaque.dutra/sync/GalaxyClusterImaging"), Path("/data/sync/GalaxyClusterImaging"))
             if p.exists()), Path("/Users/isaque.dutra/sync/GalaxyClusterImaging"))       # Mac, then Linux
CATALOGS = SYNC / "RedshiftCatalogs/M0416"
# The repository copies, used when the synced catalog folder is not on this machine.  The Rihtarsic tables A1-A3,
# Bergamini et al. 2023 and Caminha et al. 2017 have no repository copy and are skipped when absent.
LOCAL_CANUCS = Path("data/CANUCS/M0416_Rihtarsic2025")
PHOT = CATALOGS / "canucs_dr1/hlsp_canucs_jwst-hst_multi_macs0416-clu_multi_v1_photometry-cat.fits.gz"
if not PHOT.exists():
    PHOT = LOCAL_CANUCS / "canucs_dr1/hlsp_canucs_jwst-hst_multi_macs0416-clu_multi_v1_photometry-cat.fits.gz"
CANUCS_FULL = CATALOGS / "canucs_images_2025/hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt"
if not CANUCS_FULL.exists():
    CANUCS_FULL = LOCAL_CANUCS / "hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt"
# Diego et al. 2024 (PEARLS, A&A 690, A114), Table B.1: 343 images of 119 sources, ranks A-D.
M0416_DIEGO = Path("data/JWST/M0416_Diego2024_PEARLS/images.ecsv")
# The HST-based catalogs (Bergamini 2023, Caminha 2017) sit in the HST astrometric frame, offset from the JWST (CANUCS)
# frame of the model by (+0.220" west, -0.084" north) (the CANUCS members are Bergamini's shifted by this; measured on the
# Bergamini 2022 images against ours: median (-0.214, -0.081)" in model x east, y north).  The CANUCS catalogs, the DR1
# photometry and Diego 2024 are JWST-frame (offsets below 0.04").  Added to HST positions in model (x east, y north).
HST_TO_JWST_ARCSEC = (-0.220, -0.084)
# AS1063: DJA v7.5 photometry (no photo-z release, so colours only) and the secure NED spec-z; published image tables.
AS1063_CATALOGS = SYNC / "RedshiftCatalogs/AS1063"
AS1063_PHOT = AS1063_CATALOGS / "dja_v7.5/abells1063-grizli-v7.5-fix_phot_apcorr.fits"
AS1063_SPECZ = AS1063_CATALOGS / "ned_specz_secure.csv"
AS1063_DIEGO = Path("data/JWST/AS1063_Diego2026_Beauchesne2025")
CLUSTER_Z = {"m0416": 0.396, "as1063": 0.348}
HAS_PHOTOZ = {"m0416": True, "as1063": False}
COLOUR_BANDS = ["F090W", "F115W", "F200W", "F277W", "F356W", "F410M", "F444W"]      # colours are m_band - m_F150W


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--draws", type=int, default=5, help="lens draws (the best draw first)")
    p.add_argument("--field", type=float, default=75.0, help="half-side in arcsec of the searched field")
    p.add_argument("--step", type=float, default=0.3, help="coarse grid step in arcsec")
    p.add_argument("--families", default="", help="comma list of families to search (default all)")
    p.add_argument("--sigma-pos", type=float, default=0.4, help="model positional error in arcsec added to the draw spread")
    p.add_argument("--image-dir", default="data/JWST_Images/M0416")
    p.add_argument("--cluster", choices=["m0416", "as1063"], default="m0416",
                   help="photometry and published catalogs: CANUCS (M0416) or DJA v7.5 + Diego/Caminha/Bergamini (AS1063, colours only)")
    p.add_argument("--out", default="output/missing_images")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------------------ lensing
def search_images(evaluator, theta, z, f, beta_s, field, step):
    """All images of the source beta_s (best draw): sign-change cells of beta - beta_s on a grid, then Newton."""
    g = np.arange(-field, field + step / 2, step)
    xx, yy = np.meshgrid(g, g)
    beta, _ = lens_at(evaluator, theta, z, f, np.column_stack([xx.ravel(), yy.ravel()]))
    d = (beta - beta_s).reshape(len(g), len(g), 2)
    change = []
    for c in range(2):
        s = np.sign(d[..., c])
        change.append((s[:-1, :-1] != s[1:, :-1]) | (s[:-1, :-1] != s[:-1, 1:]) | (s[:-1, :-1] != s[1:, 1:]))
    iy, ix = np.nonzero(change[0] & change[1])
    x = np.column_stack([xx[iy, ix] + step / 2, yy[iy, ix] + step / 2])
    x, ok = newton(evaluator, theta, z, f, beta_s, x, step)
    out = []
    for xi in x[ok]:
        if all(np.hypot(*(xi - o)) > 0.2 for o in out):
            out.append(xi)
    return np.asarray(out).reshape(-1, 2)


def newton(evaluator, theta, z, f, beta_s, x, step, iters=30):
    x = np.asarray(x, float).copy()
    if not len(x):
        return x, np.zeros(0, bool)
    for _ in range(iters):
        b, a = lens_at(evaluator, theta, z, f, x)
        dx = np.einsum("nij,nj->ni", np.linalg.inv(a), b - beta_s)
        norm = np.hypot(*dx.T)[:, None]
        x = x - dx * np.minimum(1.0, step / np.maximum(norm, 1e-12))
    b, _ = lens_at(evaluator, theta, z, f, x)
    return x, np.hypot(*(b - beta_s).T) < 1e-3


def magnification(evaluator, theta, z, f, x):
    _, a = lens_at(evaluator, theta, z, f, np.atleast_2d(x))
    return 1.0 / (a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0])


# ------------------------------------------------------------------------------------------------------------ catalogs
def load_photometry(reference, cluster="m0416"):
    if cluster == "as1063":
        return load_photometry_dja(reference)
    from astropy.io import fits
    with fits.open(PHOT) as h:
        t = h[1].data
        cols = ["SOURCE", "RA", "DEC", "Z_ML", "Z160", "Z840", "Z025", "Z975", "NUSEFILT", "Z_SPEC", "REF_SPECZ", "FLAG_BCG", "USE_PHOT"]
        df = pd.DataFrame({c: np.asarray(t[c]).astype(str) if c == "REF_SPECZ" else np.asarray(t[c]).astype(float) for c in cols})
        for b in ["F150W"] + COLOUR_BANDS:
            flux = np.asarray(t[f"FLUX_COLOR07_TOTAL_{b}"], float); err = np.asarray(t[f"FLUXERR_COLOR07_TOTAL_{b}"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"m_{b}"] = np.where(flux > 0, 31.4 - 2.5 * np.log10(flux), np.nan)
                df[f"e_{b}"] = np.where(flux > 0, 1.0857 * err / flux, np.nan)
    df["x"], df["y"] = radec_to_model(df["RA"].to_numpy(), df["DEC"].to_numpy(), reference)
    return df


def load_photometry_dja(reference):
    """DJA v7.5 AS1063 photometry in the CANUCS-like columns used here: magnitudes from the 0.5" aperture total fluxes
    (uJy), no photo-z (the redshift range is left open, 0-20), spec-z from the secure NED list matched within 0.3"."""
    from astropy.io import fits
    t = fits.open(AS1063_PHOT)[1].data
    df = pd.DataFrame({"SOURCE": np.asarray(t["id"], float), "RA": np.asarray(t["ra"], float), "DEC": np.asarray(t["dec"], float)})
    for c, v in (("Z_ML", np.nan), ("Z025", 0.0), ("Z160", 0.0), ("Z500", np.nan), ("Z840", 20.0), ("Z975", 20.0), ("NUSEFILT", np.nan),
                 ("FLAG_BCG", 0.0), ("USE_PHOT", 1.0), ("Z_SPEC", -1.0)):
        df[c] = v
    df["REF_SPECZ"] = ""
    for b in ["F150W"] + COLOUR_BANDS:
        flux = np.asarray(t[f"{b.lower()}_tot_1"], float); err = np.asarray(t[f"{b.lower()}_etot_1"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"m_{b}"] = np.where(flux > 0, 23.9 - 2.5 * np.log10(flux), np.nan)
            df[f"e_{b}"] = np.where(flux > 0, 1.0857 * err / flux, np.nan)
    if AS1063_SPECZ.exists():
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        s = pd.read_csv(AS1063_SPECZ)
        idx, d, _ = SkyCoord(s["ra"].to_numpy() * u.deg, s["dec"].to_numpy() * u.deg).match_to_catalog_sky(
            SkyCoord(df["RA"].to_numpy() * u.deg, df["DEC"].to_numpy() * u.deg))
        ok = d.arcsec < 0.3
        df.loc[idx[ok], "Z_SPEC"] = s["z"].to_numpy()[ok]
        df.loc[idx[ok], "REF_SPECZ"] = "NED"
    df["x"], df["y"] = radec_to_model(df["RA"].to_numpy(), df["DEC"].to_numpy(), reference)
    return df


def radec_to_model(ra, dec, reference):
    _, ra0, dec0 = reference
    return -(np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600.0, (np.asarray(dec) - dec0) * 3600.0


def norm_label(label):
    """'K12A.a' / 'K12a.1' / 'K12a' -> ('K12', 'a'): system and knot, for matching CANUCS labels across tables."""
    m = re.match(r"^([A-Z]+\d+)([A-Za-z]?)", label)
    return (m.group(1), m.group(2).lower()) if m else (label, "")


def load_published(reference, cluster="m0416"):
    """Every published image as rows: catalog, label, class, x, y, z (system / spectroscopic)."""
    rows = []
    def add(cat, label, cls, ra, dec, z, hst=False):
        x, y = radec_to_model(ra, dec, reference)
        if hst:
            x, y = x + HST_TO_JWST_ARCSEC[0], y + HST_TO_JWST_ARCSEC[1]
        rows.append(dict(catalog=cat, label=label, cls=cls, x=float(x), y=float(y), z=z))
    if cluster == "as1063":
        return load_published_as1063(add, rows)
    for line in CANUCS_FULL.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        add("CANUCS R25 full", t[0], t[-1], float(t[1]), float(t[2]), float(t[3]))
    if M0416_DIEGO.exists():
        from astropy.table import Table
        diego = Table.read(M0416_DIEGO)
        source_z = {}                                   # one z per Diego source: a spec-z of any image, else the geometric z
        for col in ("zgeom", "zsp"):
            for r in diego:
                if not np.ma.is_masked(r[col]):
                    source_z[str(r["Arc"]).split(".")[0]] = float(r[col])
        for r in diego:
            add("Diego24", f"D{r['Arc']}", str(r["Rank"]), float(r["RAJ2000"]), float(r["DEJ2000"]),
                source_z.get(str(r["Arc"]).split(".")[0], np.nan))
    for name, cls_col in (("tablea1.dat", 47), ("tablea2.dat", 45), ("tablea3.dat", 47)):
        if not (CATALOGS / "rihtarsic_2025" / name).exists():
            print(f"[catalogs] {CATALOGS / 'rihtarsic_2025' / name} not found; skipped", flush=True)
            continue
        for line in (CATALOGS / "rihtarsic_2025" / name).read_text().splitlines():
            if not line.strip():
                continue
            label = line[:7].strip().rstrip("*")
            z = next((float(v) for v in line[29:46].replace("*", " ").split() if re.match(r"^\d+\.\d+$", v)), np.nan)
            add(f"Rihtarsic25 {name[:-4].replace('table', 'tab ')}", label, (line[cls_col:cls_col + 1].strip() or "-"),
                float(line[8:17]), float(line[18:28]), z)
    for line in (CATALOGS / "bergamini_2023/tablea1.dat").read_text().splitlines() if (CATALOGS / "bergamini_2023/tablea1.dat").exists() else []:
        if line.strip():
            add("Bergamini23", line[:6].strip(), line[7:10].strip() or "-", float(line[11:20]), float(line[21:31]), float(line[32:37]), hst=True)
    for line in (CATALOGS / "caminha_2017/tablea1.dat").read_text().splitlines() if (CATALOGS / "caminha_2017/tablea1.dat").exists() else []:
        if not line.strip() or ")" in line[14:27]:                     # ')' marks model-predicted positions, not detections
            continue
        z = float(line[27:33]) if re.match(r"^\d+\.\d+$", line[27:33].strip()) else np.nan
        add("Caminha17", line[:4].strip(), "-", float(line[5:14]), float(line[15:25]), z, hst=True)
    return pd.DataFrame(rows)


def load_published_as1063(add, rows):
    """AS1063: every row of Diego et al. 2026 Table 1 (ranks A and B; class = rank plus the flag when set: 3 model-predicted,
    4/5 outside the JWST footprint, 6 singly imaged arc), labelled in our system naming (Diego families 1-4 are system D1,
    knots A-D, and so on), plus the Caminha et al. 2016 and Bergamini et al. 2019 image lists."""
    import json
    groups = json.loads((AS1063_DIEGO / "image_mapping.json").read_text())["groups"]
    name = {}
    for g in groups:
        for k, fam in enumerate(g):
            name[int(fam)] = f"D{g[0]}{chr(65 + k)}"
    for r in json.loads((AS1063_DIEGO / "raw/diego_2601.11704v2_table1.json").read_text())["rows"]:
        label = f"{name.get(int(r['family']), 'D' + str(r['family']))}.{r['image']}"
        cls = f"{r['rank']}{r['flag'] if r['flag'] else ''}"
        add("Diego26", label, cls, float(r["ra"]), float(r["dec"]), float(r["z_spec"]) if r["z_spec"] else np.nan)
    for cat, path in (("Caminha16", "data/Bergamini/A1063_Caminha16/A1063_obs_arcs_v11.dat"),
                      ("Bergamini19", "data/Bergamini/RXJ2248_Bergamini19/obs_arcs.dat")):
        for line in Path(path).read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            add(cat, v[0], "-", float(v[1]), float(v[2]), float(v[6]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ scoring
def colours(row):
    return np.array([row[f"m_{b}"] - row["m_F150W"] for b in COLOUR_BANDS]), \
        np.array([np.hypot(row[f"e_{b}"], row["e_F150W"]) for b in COLOUR_BANDS])


def score_candidate(src, site, ref_colour, no_photoz=False):
    """Lower is better: chi2-like terms in position, magnitude, colour and redshift."""
    d = float(np.hypot(src["x"] - site["x"], src["y"] - site["y"]))
    sp = np.hypot(site["sigma_pos"], 0.0)
    terms = dict(dist=d, pos=(d / sp) ** 2)
    terms["dmag"] = float(src["m_F150W"] - site["m_pred"]) if np.isfinite(site["m_pred"]) else np.nan
    terms["mag"] = (terms["dmag"] / 1.0) ** 2 if np.isfinite(terms["dmag"]) else 1.0
    c, e = colours(src)
    good = np.isfinite(c) & np.isfinite(ref_colour)
    terms["n_colour"] = int(good.sum())
    terms["colour"] = float(np.mean((c[good] - ref_colour[good]) ** 2 / (e[good] ** 2 + 0.15 ** 2))) if good.any() else 1.0
    zs = site["z_source"]
    if src["Z_SPEC"] > 0:
        terms["z_term"] = 0.0 if abs(src["Z_SPEC"] - zs) / (1 + zs) < 0.01 else 25.0
        terms["z_note"] = f"spec-z {src['Z_SPEC']:.3f} ({src['REF_SPECZ'].strip()})"
    elif np.isfinite(src["Z_ML"]) and src["Z_ML"] > 0:
        inside68 = src["Z160"] <= zs <= src["Z840"]; inside95 = src["Z025"] <= zs <= src["Z975"]
        terms["z_term"] = 0.0 if inside68 else (1.0 if inside95 else 4.0)
        terms["z_note"] = f"photo-z {src['Z_ML']:.2f} [{src['Z160']:.2f}, {src['Z840']:.2f}]"
    else:
        terms["z_term"] = 0.0 if no_photoz else 2.0; terms["z_note"] = "no z"
    terms["score"] = terms["pos"] + terms["mag"] + terms["colour"] + terms["z_term"]
    return terms


def grade(best):
    if best is None:
        return "none"
    if best["z_term"] == 0.0 and "spec" in best["z_note"]:
        return "A (spec-z)"
    if best["z_term"] == 0.0 and best["colour"] < 2.0 and best["pos"] < 4.0:
        return "A"
    if best["z_term"] <= 1.0 and best["colour"] < 4.0 and best["pos"] < 9.0:
        return "B"
    return "C"


def in_footprint(images, band, x, y, reference):
    """True when the F150W mosaic has data at (x, y)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    img = images[band]
    with fits.open(img.path, memmap=True) as h:
        hdu = h[img.hdu_index]
        wcs = WCS(hdu.header)
        c = cutouts.arcsec_to_skycoord(x, y, reference)
        px, py = wcs.world_to_pixel(c)
        iy, ix = int(round(float(py))), int(round(float(px)))
        if not (0 <= iy < hdu.data.shape[0] and 0 <= ix < hdu.data.shape[1]):
            return False
        v = hdu.data[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3]
        return bool(np.any(np.isfinite(v) & (v != 0)))


# ------------------------------------------------------------------------------------------------------------ main
def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _, state, evaluator, posterior = fas.load_model(Path(args.run))
    reference = state.reference
    images = state.images
    labels = np.asarray(images.image_labels)
    fam = np.asarray(images.family_index_per_image)
    obs = np.column_stack([np.asarray(images.x_obs, float), np.asarray(images.y_obs, float)])
    fam_ids = [str(v) for v in images.family_ids]
    zmap = {str(fd.family_id): float(fd.z_source) for fd in state.family_data}
    system_of = lambda fid: norm_label(fid)[0]
    draws = fas.pick_draws(posterior, args.draws)
    samples = np.asarray(posterior.samples)
    thetas = [jnp.asarray(samples[d]) for d in draws]
    phot = load_photometry(reference, args.cluster)
    pxy = phot[["x", "y"]].to_numpy()
    wanted = set(args.families.split(",")) if args.families else None

    rows, all_rows = [], []
    for f, fid in enumerate(fam_ids):
        if wanted and fid not in wanted:
            continue
        z = zmap[fid]
        cat = obs[fam == f]; cat_labels = labels[fam == f]
        sys_obs = obs[np.array([system_of(fam_ids[i]) == system_of(fid) for i in fam])]
        sources = [np.asarray(evaluator.source_positions(t))[f] for t in thetas]
        found = search_images(evaluator, thetas[0], z, f, sources[0], args.field, args.step)
        mu_cat = magnification(evaluator, thetas[0], z, f, cat)
        # Intrinsic F150W magnitude from the catalog images that have a CANUCS counterpart within 0.5" (arcs are deblended).
        m_src = []
        for (cx, cy), mu in zip(cat, mu_cat):
            d = np.hypot(*(pxy - (cx, cy)).T); j = int(np.argmin(d))
            if d[j] < 0.5 and np.isfinite(phot["m_F150W"].iloc[j]):
                m_src.append(phot["m_F150W"].iloc[j] + 2.5 * np.log10(abs(mu)))
        m_src = float(np.median(m_src)) if m_src else np.nan
        n_missing = 0
        for xi in found:
            d_fam = np.hypot(*(cat - xi).T).min() if len(cat) else np.inf
            d_sys = np.hypot(*(sys_obs - xi).T).min() if len(sys_obs) else np.inf
            mu_i = float(magnification(evaluator, thetas[0], z, f, xi)[0])
            all_rows.append(dict(family=fid, system=system_of(fid), z_source=z, x=float(xi[0]), y=float(xi[1]), mu=mu_i,
                                 parity="+" if mu_i > 0 else "-", dist_family=float(d_fam), dist_system=float(d_sys),
                                 missing=not (d_fam < 1.0 or d_sys < 2.0)))
            if d_fam < 1.0 or d_sys < 2.0:
                continue
            # Spread over draws: Newton from the best-draw position for every other draw.
            pos = [xi]
            for t, bs in zip(thetas[1:], sources[1:]):
                xr, ok = newton(evaluator, t, z, f, bs, xi[None], 0.3)
                if ok[0] and np.hypot(*(xr[0] - xi)) < 5.0:
                    pos.append(xr[0])
            pos = np.asarray(pos)
            mu = magnification(evaluator, thetas[0], z, f, xi)[0]
            spread = float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, 1)))) if len(pos) > 1 else np.nan
            rows.append(dict(family=fid, system=system_of(fid), z_source=z, x=float(xi[0]), y=float(xi[1]),
                             x_draws=pos[:, 0].tolist(), y_draws=pos[:, 1].tolist(), n_draws=len(pos), spread=spread, mu=float(mu),
                             parity="+" if mu > 0 else "-", m_src=m_src, m_pred=m_src - 2.5 * np.log10(abs(mu)) if np.isfinite(m_src) else np.nan,
                             n_catalog=len(cat), dist_family=float(d_fam), dist_system=float(d_sys)))
            n_missing += 1
        print(f"[{fid}] z {z:.3f}: {len(found)} predicted images, {len(cat)} in the catalog, {n_missing} missing", flush=True)
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "predicted_missing_images.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out / "predicted_images.csv", index=False)       # every predicted image, best draw

    # Merge the missing images of the knots of one system into sites.
    sites = []
    for sysname, g in pred.groupby("system", sort=False):
        used = np.zeros(len(g), bool); gv = g.reset_index(drop=True)
        for i in range(len(gv)):
            if used[i]:
                continue
            close = (~used) & (np.hypot(gv["x"] - gv["x"][i], gv["y"] - gv["y"][i]) < 2.5)
            used |= close
            m = gv[close]
            spread = float(np.nanmax(m["spread"])) if m["spread"].notna().any() else 0.3
            sites.append(dict(system=sysname, families=",".join(m["family"]), z_source=float(m["z_source"].iloc[0]),
                              x=float(m["x"].mean()), y=float(m["y"].mean()), mu=float(m["mu"].median()), parity=m["parity"].iloc[0],
                              m_pred=float(m["m_pred"].min()) if m["m_pred"].notna().any() else np.nan, spread=spread,
                              sigma_pos=float(np.hypot(args.sigma_pos, spread)),
                              x_draws=sum(m["x_draws"], []), y_draws=sum(m["y_draws"], [])))
    sites = pd.DataFrame(sites)

    # Colour reference per system: median colours of its catalog images' CANUCS counterparts.
    ref_colour = {}
    for sysname in sites["system"].unique():
        cs = []
        for i in np.nonzero(np.array([system_of(fam_ids[k]) == sysname for k in fam]))[0]:
            d = np.hypot(*(pxy - obs[i]).T); j = int(np.argmin(d))
            if d[j] < 0.5:
                cs.append(colours(phot.iloc[j])[0])
        ref_colour[sysname] = np.nanmedian(np.asarray(cs), 0) if cs else np.full(len(COLOUR_BANDS), np.nan)

    band_paths = cutouts.find_rgb_band_paths(Path(args.image_dir), cluster=args.cluster, bands=("F150W",))
    fp_images = cutouts.load_band_images(band_paths, ["F150W"])
    published = load_published(reference, args.cluster)
    cand_rows, site_rows = [], []
    for k, s in sites.iterrows():
        s = s.to_dict()
        s["in_jwst"] = in_footprint(fp_images, "F150W", s["x"], s["y"], reference)
        radius = max(1.0, 2.5 * s["sigma_pos"])
        near = phot[(np.hypot(phot["x"] - s["x"], phot["y"] - s["y"]) < radius) & (phot["FLAG_BCG"] == 0)]
        scored = []
        for _, src in near.iterrows():
            t = score_candidate(src, s, ref_colour[s["system"]], no_photoz=not HAS_PHOTOZ[args.cluster])
            scored.append(dict(site=k, system=s["system"], source=int(src["SOURCE"]), x=src["x"], y=src["y"], m_F150W=src["m_F150W"],
                               Z_ML=src["Z_ML"], Z160=src["Z160"], Z840=src["Z840"], Z_SPEC=src["Z_SPEC"], REF_SPECZ=src["REF_SPECZ"].strip(),
                               **{f"c_{b}": v for b, v in zip(COLOUR_BANDS, colours(src)[0])}, **t))
        scored.sort(key=lambda r: r["score"])
        cand_rows += scored
        best = scored[0] if scored else None
        # Published images near the site: same system anywhere within 3", any catalog within max(1", 2 sigma) of the
        # best counterpart (or of the prediction when there is none).
        px, py = (best["x"], best["y"]) if best else (s["x"], s["y"])
        dpub = np.hypot(published["x"] - px, published["y"] - py)
        same = published["label"].map(lambda l: norm_label(l)[0] == s["system"]).astype(bool)
        pub = published.assign(dist=dpub)
        hits = pub[((dpub < 3.0) & same) | (dpub < max(1.0, 2 * s["sigma_pos"]))]
        hits = hits.sort_values("dist")
        s.update(best_source=best["source"] if best else -1, best_dist=best["dist"] if best else np.nan,
                 best_score=best["score"] if best else np.nan, best_z=best["z_note"] if best else "",
                 best_colour=best["colour"] if best else np.nan, best_dmag=best["dmag"] if best else np.nan,
                 grade=grade(best), n_candidates=len(scored),
                 published="; ".join(f"{h.catalog} {h.label} ({h.cls}, z {h.z:.3f}, {h.dist:.2f}\")" for h in hits.itertuples()) or "")
        site_rows.append(s)
        print(f"[site] {s['system']} ({s['families']}) at ({s['x']:.1f}, {s['y']:.1f}) mu {s['mu']:.1f} m_pred {s['m_pred']:.1f} "
              f"in JWST {s['in_jwst']}: {len(scored)} sources, grade {s['grade']}" + (f"; best {best['source']} at {best['dist']:.2f}\", "
              f"{best['z_note']}, colour chi2 {best['colour']:.1f}" if best else "") + (f"; published: {s['published']}" if s["published"] else ""),
              flush=True)
    site_df = pd.DataFrame(site_rows)
    order = {"A (spec-z)": 0, "A": 1, "B": 2, "C": 3, "none": 4}
    site_df["rank_key"] = site_df["grade"].map(order) * 1000 + site_df["best_score"].fillna(999)
    site_df = site_df.sort_values(["in_jwst", "rank_key"], ascending=[False, True])
    site_df.drop(columns=["x_draws", "y_draws"]).to_csv(out / "missing_image_sites.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(out / "missing_image_candidates.csv", index=False)
    site_df.to_pickle(out / "missing_image_sites.pkl")
    print(f"[done] {len(site_df)} sites in {out}")


if __name__ == "__main__":
    main()
