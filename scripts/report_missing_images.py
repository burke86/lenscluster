"""Report on the output of find_missing_images.py: which predicted missing images are already in a published catalog,
and colour stamps (with photo-z, spec-z and colours) of the ones that look real and are not.

A site counts as already found when a published entry of the same system (CANUCS / Rihtarsic labels) lies within 1.5" of
its best counterpart (or of the prediction when there is no counterpart), or an entry of another catalog (Bergamini 2023,
Caminha 2017: different labels) lies within 1" with a redshift within 0.01 (1 + z) of the system's.

Example:
    python scripts/report_missing_images.py --run RUN --sites output/missing_images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_arc_source as fas  # noqa: E402
from find_missing_images import COLOUR_BANDS, colours, load_photometry, load_published, norm_label  # noqa: E402
from predict_counter_images import stamp_rgb  # noqa: E402

np, pd, cutouts = fas.np, fas.pd, fas.cutouts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--sites", required=True, help="output folder of find_missing_images.py")
    p.add_argument("--grades", default="A (spec-z),A,B", help="grades shown as new candidates")
    p.add_argument("--stamp", type=float, default=4.0)
    p.add_argument("--image-dir", default="data/imaging/m0416/canucs_dr1")
    p.add_argument("--cluster", choices=["m0416", "as1063"], default="m0416")
    p.add_argument("--bands", default="F115W,F150W,F200W,F444W")
    return p.parse_args()


def found_in(site, best, published):
    px, py = (best["x"], best["y"]) if best is not None else (site["x"], site["y"])
    d = np.hypot(published["x"] - px, published["y"] - py)
    same = published["label"].map(lambda l: norm_label(l)[0] == site["system"]).astype(bool)   # object dtype: ~ would not negate
    zs = site["z_source"]
    same_z = (published["z"] - zs).abs() / (1 + zs) < 0.01
    # dist is added before filtering: assigning a Series to an empty frame would adopt its whole index as NaN rows
    pub = published.assign(dist=d)
    hits = pub[(same & (d < 1.5)) | (~same & same_z & (d < 1.0))]
    others = pub[(~same) & (~same_z) & (d < 1.0)]      # images of other systems on the spot
    return hits.sort_values("dist"), others.sort_values("dist")


CHANCE_FIELD_HALF = 70.0      # arcsec: half-side of the box over which look-alike sources are counted


def counterpart_info(phot, best, site):
    """RA/Dec, photo-z, spec-z of the best counterpart and the chance that an unrelated source with a compatible photo-z
    (z_sys inside its 16-84% range) and an F150W magnitude within 1 mag lies as close as it does: 1 - exp(-n pi d^2),
    n the density of such sources over the central box, d the distance (at least 0.3")."""
    out = dict(canucs_id=-1, ra=np.nan, dec=np.nan, zphot=np.nan, z16=np.nan, z84=np.nan, zspec=np.nan, chance=np.nan)
    if best is None:
        return out
    p = phot[phot["SOURCE"] == best["source"]].iloc[0]
    out.update(canucs_id=int(best["source"]), ra=float(p["RA"]), dec=float(p["DEC"]), zphot=float(p["Z_ML"]), z16=float(p["Z160"]),
               z84=float(p["Z840"]), zspec=float(p["Z_SPEC"]) if p["Z_SPEC"] > 0 else np.nan)
    if np.isfinite(best["m_F150W"]):
        zs = float(site["z_source"])
        box = (phot["x"].abs() < CHANCE_FIELD_HALF) & (phot["y"].abs() < CHANCE_FIELD_HALF)
        look = box & (phot["Z160"] <= zs) & (phot["Z840"] >= zs) & ((phot["m_F150W"] - best["m_F150W"]).abs() < 1.0)
        density = look.sum() / (2 * CHANCE_FIELD_HALF) ** 2
        out["chance"] = float(1 - np.exp(-density * np.pi * max(float(best["dist"]), 0.3) ** 2))
    return out


def main():
    args = parse_args()
    folder = Path(args.sites)
    sites = pd.read_pickle(folder / "missing_image_sites.pkl")
    cands = pd.read_csv(folder / "missing_image_candidates.csv")
    _, state, _, _ = fas.load_model(Path(args.run))
    reference = state.reference
    published = load_published(reference, args.cluster)
    images = state.images
    labels = np.asarray(images.image_labels)
    obs = np.column_stack([np.asarray(images.x_obs, float), np.asarray(images.y_obs, float)])

    phot = load_photometry(reference, args.cluster)
    rows = []
    for k, s in sites.iterrows():
        c = cands[cands["site"] == k].sort_values("score")
        best = c.iloc[0] if len(c) else None
        hits, others = found_in(s, best, published)
        rows.append(dict(site=k, system=s["system"], families=s["families"], z_source=s["z_source"], x=s["x"], y=s["y"], mu=s["mu"],
                         m_pred=s["m_pred"], in_jwst=s["in_jwst"], grade=s["grade"],
                         best_source=int(best["source"]) if best is not None else -1,
                         best_dist=best["dist"] if best is not None else np.nan, best_m=best["m_F150W"] if best is not None else np.nan,
                         best_z=best["z_note"] if best is not None else "", colour_chi2=best["colour"] if best is not None else np.nan,
                         found=len(hits) > 0,
                         found_in="; ".join(f"{h.catalog} {h.label} ({h.cls}, {h.dist:.2f}\")" for h in hits.itertuples()),
                         other_systems_here="; ".join(f"{h.catalog} {h.label} (z {h.z:.3f}, {h.dist:.2f}\")" for h in others.itertuples()),
                         spread=s["spread"], **counterpart_info(phot, best, s)))
    report = pd.DataFrame(rows)
    report.to_csv(folder / "missing_image_report.csv", index=False)
    table = report[~report["found"]].drop(columns=["found", "found_in", "site"])
    table = table.assign(rank_key=table["grade"].map({"A (spec-z)": 0, "A": 1, "B": 2, "C": 3, "none": 4}) * 10 + table["chance"].fillna(1) * 5)
    table.sort_values("rank_key").drop(columns="rank_key").to_csv(folder / "new_image_candidates_table.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    jw = report[report["in_jwst"]]
    print(f"{len(report)} sites, {len(jw)} inside the JWST F150W footprint")
    print("\n== already published (inside JWST)")
    print(jw[jw["found"]][["system", "families", "x", "y", "mu", "grade", "best_source", "best_z", "found_in"]].round(2).to_string(index=False))
    new = jw[(~jw["found"]) & jw["grade"].isin(args.grades.split(","))]
    print("\n== not published, graded " + args.grades)
    print(new[["system", "families", "z_source", "x", "y", "mu", "m_pred", "grade", "best_source", "best_dist", "best_m", "best_z",
               "colour_chi2", "other_systems_here"]].round(2).to_string(index=False))
    rest = jw[(~jw["found"]) & ~jw["grade"].isin(args.grades.split(","))]
    print(f"\n== not published, weaker or no counterpart: {len(rest)} sites "
          f"({(rest['grade'] == 'none').sum()} without any CANUCS source nearby)")

    if not len(new):
        return
    bands = [b.strip().upper() for b in args.bands.split(",")]
    paths = cutouts.find_rgb_band_paths(Path(args.image_dir), cluster=args.cluster, bands=bands)
    for b in bands:
        bgsub = paths[b].with_name(paths[b].name.replace("_sci.fits", "_bgsub-sci.fits"))
        if bgsub.exists():
            paths[b] = bgsub
    imgs = cutouts.load_band_images(paths, bands)
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    half = args.stamp / 2
    with PdfPages(folder / "new_candidates.pdf") as pdf:
        for _, r in new.iterrows():
            s = sites.loc[r["site"]]
            c = cands[cands["site"] == r["site"]].sort_values("score")
            sys_idx = [i for i, l in enumerate(labels) if norm_label(l)[0] == r["system"]]
            ref_xy = obs[sys_idx[0]]
            fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True,
                                     gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
            for ax, centre, tag in ((axes[0], (s["x"], s["y"]), f"{r['system']} predicted"), (axes[1], ref_xy, f"{labels[sys_idx[0]]} (catalog)")):
                rgb = stamp_rgb(imgs, bands, "F150W", centre, reference, args.stamp)
                ax.imshow(rgb, origin="lower", extent=[centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half])
                ax.text(0.03, 0.96, tag, transform=ax.transAxes, color="w", fontsize=6, va="top")
                ax.set_aspect("equal"); ax.tick_params(labelsize=5); ax.set_xlabel(r"$x$ [arcsec]", fontsize=6)
            axes[0].set_ylabel(r"$y$ [arcsec]", fontsize=6)
            axes[0].plot(s["x_draws"], s["y_draws"], "+", color="w", ms=4, mew=0.6)
            for j, (_, cc) in enumerate(c.head(3).iterrows()):
                axes[0].plot(cc["x"], cc["y"], "o", mfc="none", mec="c" if j == 0 else "y", ms=8, mew=0.7)
                axes[0].text(cc["x"] + 0.15, cc["y"] + 0.15, str(j + 1), color="c" if j == 0 else "y", fontsize=6)
            axes[0].set_xlim(s["x"] - half, s["x"] + half); axes[0].set_ylim(s["y"] - half, s["y"] + half)
            # Colours of the best counterparts against the system's catalog images.
            ax = axes[2]
            x = np.arange(len(COLOUR_BANDS))
            ref_c = []
            for i in sys_idx:
                d = np.hypot(phot["x"] - obs[i][0], phot["y"] - obs[i][1]); j = int(np.argmin(d))
                if d.iloc[j] < 0.5:
                    ref_c.append(colours(phot.iloc[j]))
            if ref_c:
                cs = np.asarray([v[0] for v in ref_c])
                ax.fill_between(x, np.nanpercentile(cs, 16, 0), np.nanpercentile(cs, 84, 0), color="0.8", lw=0, label="catalog images")
                ax.plot(x, np.nanmedian(cs, 0), color="0.4", lw=1)
            for j, (_, cc) in enumerate(c.head(3).iterrows()):
                row = phot[phot["SOURCE"] == cc["source"]].iloc[0]
                v, e = colours(row)
                ax.errorbar(x + 0.08 * j, v, yerr=e, fmt="o", ms=3, color="tab:cyan" if j == 0 else "tab:olive", capsize=0, lw=0.8,
                            label=f"{j + 1}: {int(cc['source'])}, {cc['z_note']}")
            shown = np.concatenate([np.ravel(colours(phot[phot["SOURCE"] == cc["source"]].iloc[0])[0]) for _, cc in c.head(3).iterrows()]
                                   + ([np.ravel(np.nanmedian(cs, 0))] if ref_c else []))
            shown = shown[np.isfinite(shown)]
            if shown.size:                                  # a faint band's error bar must not set the axis
                ax.set_ylim(shown.min() - 1.0, shown.max() + 1.0)
            ax.set_xticks(x); ax.set_xticklabels(COLOUR_BANDS, rotation=45, fontsize=5); ax.tick_params(labelsize=5)
            ax.set_ylabel(r"$m_{\rm band} - m_{\rm F150W}$", fontsize=6); ax.invert_yaxis()
            ax.legend(fontsize=4.5, frameon=False, loc="best")
            fig.text(0.01, 0.005, f"{r['system']} z = {r['z_source']:.3f}, families {r['families']}, mu {r['mu']:.1f}, predicted F150W "
                     f"{r['m_pred']:.1f}; grade {r['grade']}" + (f"; also here: {r['other_systems_here']}" if r["other_systems_here"] else ""),
                     fontsize=5)
            pdf.savefig(fig); plt.close(fig)
    print(f"[done] {folder / 'new_candidates.pdf'}")


if __name__ == "__main__":
    main()
