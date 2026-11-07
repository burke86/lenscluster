"""Predict every image of a catalog system with the lens fixed at our model, to look for counter-images the catalog lacks.

For each family of the system (e.g. K12A, K12B) the model source position is traced back through the whole field: a
coarse grid finds the cells where both components of beta(x) - beta_s change sign, Newton steps refine each root, and
roots within 0.2" are merged.  Solutions not near a catalog image of the system are the candidate counter-images; the
script prints their magnification and the expected brightness relative to the catalog images and draws NIRCam colour
stamps around them.

Example:
    python scripts/predict_counter_images.py --run RUN --system K12 --out output/counter_images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_arc_source as fas  # noqa: E402  (sets up paths, x64 and the working directory)

jnp, np, cutouts = fas.jnp, fas.np, fas.cutouts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--draws", type=int, default=3, help="lens draws (the best draw first)")
    p.add_argument("--field", type=float, default=70.0, help="half-side in arcsec of the searched field")
    p.add_argument("--step", type=float, default=0.25, help="coarse grid step in arcsec")
    p.add_argument("--stamp", type=float, default=4.0, help="side in arcsec of the colour stamps")
    p.add_argument("--image-dir", default="data/imaging/m0416/canucs_dr1")
    p.add_argument("--bands", default="F115W,F150W,F200W,F444W")
    p.add_argument("--out", default="output/counter_images")
    return p.parse_args()


def lens_at(evaluator, theta, z_source, family_index, xy, chunk=40000):
    """beta(x) and the Jacobian A(x) of the lens plus field mean at the points xy."""
    physical = evaluator.physical(theta)
    packed = evaluator.packed_state_for_z(physical, z_source)
    mean = evaluator.gp_deflection_correction(theta)
    betas, jacs = [], []
    for i in range(0, len(xy), chunk):
        x = xy[i:i + chunk]
        alpha, u = evaluator.total_fields_at(x, packed)
        a = np.eye(2)[None] - np.asarray(u)
        alpha = np.asarray(alpha)
        if mean is not None:
            dx, dy, jxx, jxy, jyx, jyy = mean(x[:, 0], x[:, 1], np.full(len(x), family_index, dtype=np.int32))
            alpha = alpha + np.column_stack([dx, dy])
            a = a - np.stack([np.stack([jxx, jxy], -1), np.stack([jyx, jyy], -1)], -2)
        betas.append(x - alpha); jacs.append(a)
    return np.concatenate(betas), np.concatenate(jacs)


def solve_images(evaluator, theta, z_source, family_index, beta_s, field, step):
    g = np.arange(-field, field + step / 2, step)
    xx, yy = np.meshgrid(g, g)
    beta, _ = lens_at(evaluator, theta, z_source, family_index, np.column_stack([xx.ravel(), yy.ravel()]))
    d = (beta - beta_s).reshape(len(g), len(g), 2)
    roots = []
    for c in range(2):
        s = np.sign(d[..., c])
        change = (s[:-1, :-1] != s[1:, :-1]) | (s[:-1, :-1] != s[:-1, 1:]) | (s[:-1, :-1] != s[1:, 1:])
        roots.append(change)
    iy, ix = np.nonzero(roots[0] & roots[1])
    starts = np.column_stack([xx[iy, ix] + step / 2, yy[iy, ix] + step / 2])
    x = starts.copy()
    for _ in range(30):                                   # Newton on all candidates at once, damped to one grid step
        b, a = lens_at(evaluator, theta, z_source, family_index, x)
        dx = np.einsum("nij,nj->ni", np.linalg.inv(a), b - beta_s)
        norm = np.hypot(*dx.T)[:, None]
        x = x - dx * np.minimum(1.0, step / np.maximum(norm, 1e-12))
    b, a = lens_at(evaluator, theta, z_source, family_index, x)
    resid = np.hypot(*(b - beta_s).T)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    tr = a[:, 0, 0] + a[:, 1, 1]
    ok = (resid < 1e-3) & (np.abs(x) < field).all(1)
    out = []
    for xi, di, ti in zip(x[ok], det[ok], tr[ok]):
        if all(np.hypot(*(xi - o[0])) > 0.2 for o in out):
            parity = "+" if di > 0 and ti > 0 else ("-" if di < 0 else "max")   # minimum (+), saddle (-), maximum
            out.append((xi, 1.0 / di, parity))
    return out


def stamp_rgb(images, bands, ref_band, centre_xy, reference, size):
    coord = cutouts.arcsec_to_skycoord(*centre_xy, reference)
    ref = images[ref_band]
    data = {}
    for b in bands:
        cut = cutouts.extract_band_cutout_on_grid(images[b], ref, coord, size).astype(float)
        data[b] = np.where(np.isfinite(cut), cut, 0.0)
    from lenscluster.rgb import make_calibrated_rgb
    return make_calibrated_rgb(data, bands=bands, display=cutouts.build_rgb_display(images, bands=bands))


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _, state, evaluator, posterior = fas.load_model(Path(args.run))
    images = state.images
    labels = np.asarray(images.image_labels)
    fam = np.asarray(images.family_index_per_image)
    obs = np.column_stack([np.asarray(images.x_obs, float), np.asarray(images.y_obs, float)])
    sel = np.array([l.startswith(args.system) and not l[len(args.system)].isdigit() for l in labels])
    families = sorted(set(fam[sel].tolist()))
    draws = fas.pick_draws(posterior, args.draws)
    samples = np.asarray(posterior.samples)
    rows, candidates = [], []
    for f in families:
        family_id = str(images.family_ids[f])
        z = float(next(fd.z_source for fd in state.family_data if str(fd.family_id) == family_id))
        cat = obs[fam == f]; cat_labels = labels[fam == f]
        for k, d in enumerate(draws):
            theta = jnp.asarray(samples[d])
            beta_s = np.asarray(evaluator.source_positions(theta))[f]
            sols = solve_images(evaluator, theta, z, f, beta_s, args.field, args.step)
            _, a_cat = lens_at(evaluator, theta, z, f, cat)
            mu_cat = 1.0 / (a_cat[:, 0, 0] * a_cat[:, 1, 1] - a_cat[:, 0, 1] * a_cat[:, 1, 0])
            for xi, mu, parity in sols:
                dist = np.hypot(*(cat - xi).T)
                j = int(np.argmin(dist))
                known = dist[j] < 1.0
                dm = -2.5 * np.log10(abs(mu) / np.median(np.abs(mu_cat)))      # magnitudes relative to the catalog images
                rows.append(dict(family=family_id, draw=d, x=xi[0], y=xi[1], mu=mu, parity=parity,
                                 matches=cat_labels[j] if known else "", dist_catalog=dist[j], dmag_vs_catalog=dm))
                print(f"[{family_id}] draw {d}: image at ({xi[0]:7.2f}, {xi[1]:7.2f}) mu {mu:8.2f} parity {parity} "
                      + (f"= {cat_labels[j]} ({dist[j]:.2f}\")" if known else f"NEW, {dm:+.1f} mag vs the catalog images"), flush=True)
                if not known and k == 0:
                    candidates.append((family_id, xi, mu, dm))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"{args.system}_predicted_images.csv", index=False)

    bands = [b.strip().upper() for b in args.bands.split(",")]
    paths = cutouts.find_rgb_band_paths(Path(args.image_dir), cluster="m0416", bands=bands)
    for b in bands:
        bgsub = paths[b].with_name(paths[b].name.replace("_sci.fits", "_bgsub-sci.fits"))
        if bgsub.exists():
            paths[b] = bgsub
    imgs = cutouts.load_band_images(paths, bands)
    reference = state.reference
    panels = [(f"{args.system} catalog", obs[sel].mean(0), None)] + [(f"{fid} predicted, mu {mu:.1f}, {dm:+.1f} mag", xi, xi)
                                                                     for fid, xi, mu, dm in candidates]
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle")); plt.rcParams.update(PAPER_STYLE)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), constrained_layout=True, squeeze=False)
    half = args.stamp / 2
    for ax, (label, centre, mark) in zip(axes[0], panels):
        rgb = stamp_rgb(imgs, bands, bands[1] if len(bands) > 1 else bands[0], centre, reference, args.stamp)
        ax.imshow(rgb, origin="lower", extent=[centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half])
        # Every predicted image (all draws) inside the stamp: the scatter shows the lens uncertainty of the prediction.
        for r in rows:
            if abs(r["x"] - centre[0]) < half and abs(r["y"] - centre[1]) < half:
                ax.plot(r["x"], r["y"], "+", color="w", ms=4, mew=0.6)
        for l, (ox, oy) in zip(labels[sel], obs[sel]):
            if abs(ox - centre[0]) < half and abs(oy - centre[1]) < half:
                ax.plot(ox, oy, "o", mfc="none", mec="c", ms=5, mew=0.6)
        ax.text(0.03, 0.96, label, transform=ax.transAxes, color="w", fontsize=6, va="top")
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
        ax.set_xlabel(r"$x$ [arcsec]", fontsize=7)
    axes[0, 0].set_ylabel(r"$y$ [arcsec]", fontsize=7)
    fig.savefig(out / f"{args.system}_counter_images.pdf")
    print(f"[done] {out / f'{args.system}_counter_images.pdf'}")


if __name__ == "__main__":
    main()
