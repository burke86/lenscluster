"""Pages for checking whether separate systems might be one source: systems whose redshifts agree within ``--dz``.

One page per set.  Left: where every image of every system in the set lies (arcsec from the set's centre, west to the
right, north up), coloured by system and labelled.  Right: a JWST RGB stamp of each image, one row per system and one
column per image letter, all stamps on the page the same width so colours and shapes compare directly.  Knots carry
their names.  Lensing keeps colours, so images of one source should look alike in colour; the map shows whether the
images could form one lensing configuration.

Example:
    python scripts/make_family_candidate_cutouts.py data/Bergamini/M0416_Bergamini22/images.yaml \\
        output/knot_cutouts/m0416_bergamini_same_redshift_candidates.pdf --image-dir /Users/isaque.dutra/sync/GalaxyClusterImaging/JWST_Images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import astropy.units as u  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.cutouts import build_rgb_display, cutout_window, find_rgb_band_paths, load_band_images, rgb_cutout, rgb_reference_band  # noqa: E402
from lenscluster.image_catalog import load_image_catalog  # noqa: E402
from lenscluster.sky import radec_to_offsets_arcsec  # noqa: E402

# Categorical slots of the dataviz reference palette, dark-surface steps; fixed order, one per system in a set.
SYSTEM_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#f0efec"]
MAP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#5a5a55"]   # light-surface steps for the map
INK = "#f0efec"
OUTLINE = [pe.withStroke(linewidth=2.2, foreground="black")]


def candidate_sets(table, dz: float) -> list[list[str]]:
    """Systems linked by redshift differences of at most ``dz`` (chained), sets of two or more, in catalog order."""
    z = table.groupby("system_id", sort=False).catalog_z.first()
    names = list(z.index)
    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(float(z[a]) - float(z[b])) <= dz:
                parent[find(a)] = find(b)
    sets: dict[str, list[str]] = {}
    for n in names:
        sets.setdefault(find(n), []).append(n)
    return [s for s in sets.values() if len(s) > 1]


def stamp_width(table, members, *, margin: float, smallest: float, largest: float) -> float:
    """One width for the page: wide enough for the knots of every image, within [smallest, largest]."""
    widest = 0.0
    for (_, _), here in table[table.system_id.isin(members)].groupby(["system_id", "image_id"]):
        x, y = radec_to_offsets_arcsec(here.ra.to_numpy(), here.dec.to_numpy(), float(here.ra.mean()), float(here.dec.mean()))
        if len(x) > 1:
            widest = max(widest, float(np.max(np.hypot(x[:, None] - x[None], y[:, None] - y[None]))))
    return float(np.clip(widest + 2 * margin, smallest, largest))


def draw_map(ax, sub, members) -> None:
    ra0, dec0 = float(sub.ra.mean()), float(sub.dec.mean())
    for i, system in enumerate(members):
        rows = sub[sub.system_id == system]
        x, y = radec_to_offsets_arcsec(rows.ra.to_numpy(), rows.dec.to_numpy(), ra0, dec0)
        z = rows.catalog_z.iloc[0]
        ax.scatter(x, y, s=46, color=MAP_COLORS[i % len(MAP_COLORS)], edgecolors="white", linewidths=1.2, zorder=3,
                   label=f"{system}  z = {z:g}")
        for letter in dict.fromkeys(rows.image_id):
            sel = (rows.image_id == letter).to_numpy()
            ax.annotate(f"{system}.{letter}", (float(np.mean(x[sel])), float(np.mean(y[sel]))), xytext=(5, 4), textcoords="offset points",
                        fontsize=8.5, color="#2b2b28")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_xaxis()                                               # east to the left, as on the sky
    ax.set_xlabel("west offset (arcsec)  <-  east")
    ax.set_ylabel("north offset (arcsec)")
    ax.legend(loc="best", fontsize=8.5, frameon=True)


def draw_stamp(ax, here, color, width, images, bands, display, reference, ring) -> None:
    ax.set_xticks([]), ax.set_yticks([])
    coord = SkyCoord(ra=float(here.ra.mean()) * u.deg, dec=float(here.dec.mean()) * u.deg, frame="icrs")
    window = cutout_window(reference, coord, width)
    if window is None:
        ax.set_facecolor("black")
        ax.text(0.5, 0.5, "outside the mosaic", color=INK, ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return
    rgb = rgb_cutout(images, bands, display, coord, width)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xlim(-0.5, rgb.shape[1] - 0.5), ax.set_ylim(-0.5, rgb.shape[0] - 0.5)
    px, py = reference.wcs.world_to_pixel_values(here.ra.to_numpy(), here.dec.to_numpy())
    radius = ring / reference.pixel_scale_arcsec
    for x, y, knot in zip(px - window.x0, py - window.y0, here.knot_id):
        ax.add_patch(Circle((x, y), radius, fill=False, lw=1.4, color=color, zorder=5))
        if knot:
            ax.text(x + 1.15 * radius, y + 1.15 * radius, knot, color=INK, fontsize=8, fontweight="bold", zorder=6, path_effects=OUTLINE)


def draw_set(pdf, table, members, images, bands, display, reference, args) -> None:
    sub = table[table.system_id.isin(members)]
    letters = sorted(set(sub.image_id))
    width = stamp_width(table, members, margin=args.margin, smallest=args.min_size, largest=args.max_size)
    cell = 2.1
    fig = plt.figure(figsize=(4.6 + cell * len(letters), max(4.6, cell * len(members) + 0.3)))
    grid = GridSpec(len(members), len(letters) + 2, figure=fig, width_ratios=[4.4 / cell, 0.12] + [1] * len(letters), wspace=0.04, hspace=0.04)
    draw_map(fig.add_subplot(grid[:, 0]), sub, members)
    for r, system in enumerate(members):
        color = SYSTEM_COLORS[r % len(SYSTEM_COLORS)]
        for c, letter in enumerate(letters):
            ax = fig.add_subplot(grid[r, c + 2])
            ax.set_box_aspect(1)
            here = sub[(sub.system_id == system) & (sub.image_id == letter)]
            if here.empty:
                ax.axis("off")
                continue
            draw_stamp(ax, here, color, width, images, bands, display, reference, args.ring)
            ax.text(0.04, 0.96, f"{system}.{letter}", color=INK, fontsize=8.5, fontweight="bold", ha="left", va="top",
                    transform=ax.transAxes, path_effects=OUTLINE)
            if r == 0 and c == 0:
                ax.text(0.04, 0.04, f'{width:.1f}" wide', color=INK, fontsize=7.5, ha="left", va="bottom", transform=ax.transAxes, path_effects=OUTLINE)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path, help="multi-page PDF")
    parser.add_argument("--dz", type=float, default=0.01, help="largest redshift difference within a set (default 0.01)")
    parser.add_argument("--image-dir", type=Path, default=Path("data/JWST_Images"))
    parser.add_argument("--cluster", default="m0416")
    parser.add_argument("--bands", nargs="+", default=["F115W", "F150W", "F200W", "F444W"])
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--min-size", type=float, default=3.0)
    parser.add_argument("--max-size", type=float, default=6.0)
    parser.add_argument("--ring", type=float, default=0.18)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".pdf":
        raise SystemExit("the output must be a .pdf")
    table = load_image_catalog(args.catalog, sigma_inst_arcsec=1.0)
    sets = candidate_sets(table, args.dz)
    paths = find_rgb_band_paths(args.image_dir, cluster=args.cluster, bands=args.bands, image_scale="auto")
    images = load_band_images(paths, args.bands)
    display = build_rgb_display(images, bands=args.bands)
    reference = images[rgb_reference_band(args.bands)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for members in sets:
            draw_set(pdf, table, members, images, args.bands, display, reference, args)
            print("  " + ", ".join(members), flush=True)
    print(f"wrote {args.output}: {len(sets)} pages (|dz| <= {args.dz:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
