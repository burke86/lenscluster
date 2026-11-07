"""Colour images of a lensed source reconstructed in each JWST band by scripts/fit_arc_source.py on one shared source grid.
Writes one PDF with the observed arc, the model arc and the source plane, all three in the same NIRCam colour recipe as
the cluster figure.  The clean source comes from the shared-shape fit (fit_arc_source.py --bands), whose band folders
are passed here; independent per-band fits (--grid-from) also work but their source plane shows per-band noise as colour.

Example:
    python scripts/fit_arc_source.py --run RUN --system K12 --band F150W --bands F115W,F150W,F200W,F444W --lam 0.03
    python scripts/make_source_rgb.py output/arc_fits/K12_colour/F115W output/arc_fits/K12_colour/F150W \\
        output/arc_fits/K12_colour/F200W output/arc_fits/K12_colour/F444W --out output/arc_fits/K12_colour/source_rgb.pdf
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.ndimage import zoom

from lenscluster import cutouts
from lenscluster.rgb import make_calibrated_rgb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="fit_arc_source output folders, one per band, sharing one source grid")
    p.add_argument("--out", required=True, help="output PDF")
    p.add_argument("--draw", type=int, default=0, help="index of the lens draw (0 = the best draw)")
    p.add_argument("--image-dir", default="data/imaging/m0416/canucs_dr1")
    p.add_argument("--source-zoom", type=float, default=0.4, help="crop the source panel to this fraction of the grid around its light")
    p.add_argument("--source-smooth", type=float, default=3.0,
                   help="Gaussian sigma in source pixels applied to every band's source for display (per-pixel noise is independent "
                        "between bands and shows as random colour otherwise); 0 = none")
    return p.parse_args()


def load(run: Path, draw: int) -> dict:
    with np.load(run / "pixel_sources.npz") as f:
        return dict(band=str(f["band"]), source=np.asarray(f["sources"][draw], float), origin=tuple(f["origins"][draw]),
                    pixel=float(f["pixel"]), image_pixel=float(f["image_pixel"]), model=np.asarray(f["arc_model"], float),
                    extent=np.asarray(f["extent"], float), sky_reference=tuple(float(v) for v in f["sky_reference"]))


def display_for(bands, image_dir: str, reference_pixel: float):
    """The cluster figure's NIRCam display (per-channel stretch fitted to the STScI release) for these bands."""
    paths = cutouts.find_rgb_band_paths(Path(image_dir), cluster="m0416", bands=bands)
    images = cutouts.load_band_images(paths, bands)
    display = cutouts.build_rgb_display(images, bands=bands)
    # Every map here is already in nJy per pixel of its band and resampled to the reference pixel; fluxscales then only
    # carry the pixel-area ratio, which we apply ourselves, and the sky is already subtracted.
    return replace(display, band_fluxscales={b: 1.0 for b in bands}, band_backgrounds={b: 0.0 for b in bands},
                   unsharp_sigma_pixels=float(display.unsharp_sigma_pixels) * float(images[bands[0]].pixel_scale_arcsec) / reference_pixel)


def on_reference_grid(image: np.ndarray, image_pixel: float, reference_pixel: float, shape) -> np.ndarray:
    """Resample a band's map to the reference pixel and convert per-pixel flux to the reference pixel area."""
    if abs(image_pixel - reference_pixel) > 1e-6:
        image = zoom(image, image_pixel / reference_pixel, order=1) * (reference_pixel / image_pixel) ** 2
    ny, nx = shape
    out = np.zeros(shape)
    out[:min(ny, image.shape[0]), :min(nx, image.shape[1])] = image[:ny, :nx]
    return out


def main():
    args = parse_args()
    runs = [load(Path(r), args.draw) for r in args.runs]
    bands = [r["band"] for r in runs]
    reference = min(runs, key=lambda r: r["image_pixel"])
    shapes = {r["source"].shape for r in runs}
    if len(shapes) != 1 or len({round(r["pixel"], 9) for r in runs}) != 1:
        raise SystemExit("the runs do not share one source grid; rerun the other bands with --grid-from")
    print("bands:", ", ".join(f"{r['band']} ({r['image_pixel'] * 1000:.0f} mas)" for r in runs))

    # Observed arc: the bgsub cutouts of every band on the reference band's grid, through both WCSs.
    paths = cutouts.find_rgb_band_paths(Path(args.image_dir), cluster="m0416", bands=bands)
    for b in bands:
        bgsub = paths[b].with_name(paths[b].name.replace("_sci.fits", "_bgsub-sci.fits"))
        if bgsub.exists():
            paths[b] = bgsub
    images = cutouts.load_band_images(paths, bands)
    ext = reference["extent"]
    reference_tuple = (0, *reference["sky_reference"])
    centre = ((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2)
    size = abs(ext[1] - ext[0]) + reference["image_pixel"]
    coord = cutouts.arcsec_to_skycoord(*centre, reference_tuple)
    ref_image = images[reference["band"]]
    shape = reference["model"].shape
    data = {}
    for r in runs:
        cut = cutouts.extract_band_cutout_on_grid(images[r["band"]], ref_image, coord, size).astype(float)
        cut = np.where(np.isfinite(cut), cut, 0.0) * (reference["image_pixel"] / r["image_pixel"]) ** 2
        data[r["band"]] = on_reference_grid(cut, reference["image_pixel"], reference["image_pixel"], shape)
    model = {r["band"]: on_reference_grid(r["model"], r["image_pixel"], reference["image_pixel"], shape) for r in runs}
    # Source plane: the per-band sources share the grid; values are surface brightness per image pixel of each band.
    from scipy.ndimage import gaussian_filter
    smooth = (lambda a: gaussian_filter(a, args.source_smooth)) if args.source_smooth > 0 else (lambda a: a)
    source = {r["band"]: np.clip(smooth(r["source"]), 0, None) * (reference["image_pixel"] / r["image_pixel"]) ** 2 for r in runs}

    display = display_for(bands, args.image_dir, reference["image_pixel"])
    rgb_data = make_calibrated_rgb(data, bands=bands, display=display)
    rgb_model = make_calibrated_rgb(model, bands=bands, display=display)
    # The source is shown at the same surface-brightness scale (lensing conserves it); no unsharp mask on the source grid.
    rgb_source = make_calibrated_rgb(source, bands=bands, display=replace(display, unsharp_sigma_pixels=0.0))

    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    image_extent = [ext[0], ext[1], ext[2], ext[3]]
    axes[0].imshow(rgb_data, origin="lower", extent=image_extent)
    axes[1].imshow(rgb_model, origin="lower", extent=image_extent)
    x0, y0 = reference["origin"]; h = reference["pixel"]; ny_s, nx_s = reference["source"].shape
    source_extent = [x0, x0 + nx_s * h, y0, y0 + ny_s * h]
    axes[2].imshow(rgb_source, origin="lower", extent=source_extent)
    if args.source_zoom < 1.0:
        light = gaussian_filter(sum(source.values()), 6.0)
        iy, ix = np.unravel_index(np.argmax(light), light.shape)            # the source's brightest smoothed spot
        cx, cy = x0 + (ix + 0.5) * h, y0 + (iy + 0.5) * h
        half = args.source_zoom * max(nx_s, ny_s) * h / 2
        axes[2].set_xlim(cx - half, cx + half); axes[2].set_ylim(cy - half, cy + half)
    for ax, label in zip(axes, ["observed", "model", "source"]):
        ax.text(0.03, 0.96, label, transform=ax.transAxes, color="w", fontsize=7, va="top")
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
        ax.set_xlabel(r"$x$ [arcsec]" if label != "source" else r"$\beta_x$ [arcsec]", fontsize=7)
    axes[0].set_ylabel(r"$y$ [arcsec]", fontsize=7); axes[2].set_ylabel(r"$\beta_y$ [arcsec]", fontsize=7)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
