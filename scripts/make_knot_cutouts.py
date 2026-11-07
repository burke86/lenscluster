"""JWST RGB cutouts of every multiple-image system, with its knots marked, for checking the catalog by eye.

One page per system and one stamp per image.  Each stamp is centred on the mean of that image's knots and wide enough
for all of them; every knot has one ring colour and its name in every image, so the same clump can be followed across
images.  Systems with knots come first, then systems that share a redshift and a name prefix with them but were kept
apart, then the rest.  Stamps are in the mosaic's pixel orientation.

Example:
    python scripts/make_knot_cutouts.py catalogs/m0416/bergamini2023/images.yaml output/knot_cutouts/m0416_bergamini_jwst_knots.pdf \\
        --image-dir data/imaging
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster.cutouts import (build_rgb_display, cutout_window, find_rgb_band_paths, load_band_images,  # noqa: E402
                                 rgb_cutout, rgb_reference_band)
from lenscluster.image_catalog import load_image_catalog  # noqa: E402
from lenscluster.sky import radec_to_offsets_arcsec  # noqa: E402

# Categorical slots 1-6 of the dataviz reference palette, dark-surface steps (the stamps are dark sky); fixed order.
KNOT_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#f0efec"]
INK = "#f0efec"
OUTLINE = [pe.withStroke(linewidth=2.2, foreground="black")]


def stamp_size(ra, dec, *, margin: float, smallest: float, largest: float) -> float:
    x, y = radec_to_offsets_arcsec(np.asarray(ra), np.asarray(dec), float(np.mean(ra)), float(np.mean(dec)))
    span = float(np.max(np.hypot(x[:, None] - x[None], y[:, None] - y[None]))) if len(x) > 1 else 0.0
    return float(np.clip(span + 2.0 * margin, smallest, largest))


def page_order(table) -> list[str]:
    """Systems with knots first (catalog order); after each, the systems sharing its z and name prefix; then the rest."""
    systems = list(dict.fromkeys(table.system_id))
    z = table.groupby("system_id").catalog_z.first()
    knotted = [s for s in systems if (table.system_id == s).any() and (table.loc[table.system_id == s, "knot_id"] != "").any()]
    order: list[str] = []
    for s in knotted:
        order.append(s)
        for other in systems:
            if other not in knotted and other not in order and z[other] == z[s] and other.startswith(s[:max(1, len(s) - 1)]):
                order.append(other)
    return order + [s for s in systems if s not in order]


def draw_system(pdf, table, system: str, images, bands, display, reference, args) -> None:
    rows = table[table.system_id == system]
    letters = list(dict.fromkeys(rows.image_id))
    knots = sorted(set(rows.knot_id))
    colors = {k: KNOT_COLORS[i % len(KNOT_COLORS)] for i, k in enumerate(knots)}
    fig, axes = plt.subplots(1, len(letters), figsize=(3.4 * len(letters), 3.6), squeeze=False)
    for col, (ax, letter) in enumerate(zip(axes[0], letters)):
        here = rows[rows.image_id == letter]
        ra0, dec0 = float(here.ra.mean()), float(here.dec.mean())
        size = stamp_size(here.ra.to_numpy(), here.dec.to_numpy(), margin=args.margin, smallest=args.min_size, largest=args.max_size)
        coord = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg, frame="icrs")
        window = cutout_window(reference, coord, size)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_box_aspect(1)
        if window is None:
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, "outside the JWST mosaic", color=INK, ha="center", va="center", transform=ax.transAxes)
        else:
            rgb = rgb_cutout(images, bands, display, coord, size)
            ax.imshow(rgb, origin="lower", interpolation="nearest")
            ax.set_xlim(-0.5, rgb.shape[1] - 0.5), ax.set_ylim(-0.5, rgb.shape[0] - 0.5)
            px, py = reference.wcs.world_to_pixel_values(here.ra.to_numpy(), here.dec.to_numpy())
            px, py = px - window.x0, py - window.y0
            radius = args.ring / reference.pixel_scale_arcsec
            for x, y, knot in zip(px, py, here.knot_id):
                ax.add_patch(Circle((x, y), radius, fill=False, lw=1.4, color=colors[knot], zorder=5))
                if knot:
                    ax.text(x + 1.15 * radius, y + 1.15 * radius, knot, color=INK, fontsize=9, fontweight="bold", zorder=6, path_effects=OUTLINE)
            bar = 1.0 / reference.pixel_scale_arcsec                    # one-arcsecond scale bar, bottom right
            x1, y1 = rgb.shape[1] * 0.95, rgb.shape[0] * 0.06
            ax.plot([x1 - bar, x1], [y1, y1], color=INK, lw=1.6, zorder=6, path_effects=OUTLINE)
            ax.text(x1 - bar / 2, y1 + 0.03 * rgb.shape[0], '1"', color=INK, fontsize=8, ha="center", zorder=6, path_effects=OUTLINE)
        label = f"{system}.{letter}" + (f"   z = {here.catalog_z.iloc[0]:g}" if col == 0 else "")
        ax.text(0.03, 0.97, label, color=INK, fontsize=10, fontweight="bold", ha="left", va="top", transform=ax.transAxes, path_effects=OUTLINE)
        ax.text(0.03, 0.04, f'{size:.1f}" wide', color=INK, fontsize=8, ha="left", va="bottom", transform=ax.transAxes, path_effects=OUTLINE)
        if col == 0 and knots != [""]:
            names = ", ".join(f"{system}{k}" for k in knots)
            ax.text(0.03, 0.88, f"knots of {names}" if len(names) < 48 else f"{len(knots)} knots", color=INK, fontsize=7.5,
                    ha="left", va="top", transform=ax.transAxes, path_effects=OUTLINE)
    fig.tight_layout(pad=0.4, w_pad=0.4)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path, help="multi-page PDF")
    parser.add_argument("--image-dir", type=Path, default=Path("data/imaging"))
    parser.add_argument("--cluster", default="m0416", help="imaging cluster key (cutouts.CLUSTER_IMAGE_TOKENS)")
    parser.add_argument("--bands", nargs="+", default=["F115W", "F150W", "F200W", "F444W"])
    parser.add_argument("--margin", type=float, default=1.2, help="arcsec around the knots")
    parser.add_argument("--min-size", type=float, default=3.0)
    parser.add_argument("--max-size", type=float, default=14.0)
    parser.add_argument("--ring", type=float, default=0.18, help="ring radius, arcsec")
    parser.add_argument("--systems", nargs="*", help="only these systems")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".pdf":
        raise SystemExit("the output must be a .pdf")
    table = load_image_catalog(args.catalog, sigma_inst_arcsec=1.0)
    paths = find_rgb_band_paths(args.image_dir, cluster=args.cluster, bands=args.bands, image_scale="auto")
    images = load_band_images(paths, args.bands)
    display = build_rgb_display(images, bands=args.bands)
    reference = images[rgb_reference_band(args.bands)]
    order = page_order(table)
    if args.systems:
        order = [s for s in order if s in set(args.systems)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for i, system in enumerate(order):
            draw_system(pdf, table, system, images, args.bands, display, reference, args)
            if i % 10 == 0:
                print(f"  {i + 1}/{len(order)} {system}", flush=True)
    print(f"wrote {args.output}: {len(order)} pages, bands {', '.join(args.bands)} from {', '.join(str(p.name) for p in paths.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
