"""Search the CANUCS DR1 photometry of M0416 for new multiple-image families with our lens model: sources of similar
colour and compatible photo-z whose positions trace back to the same point of the source plane, over several lens draws.

The search rests on one property of the model.  The deflection at any source redshift is an exact combination of the
deflections at two reference redshifts, alpha(z) = (1 - t) alpha_a + t alpha_b, with t(z) monotonic and the same everywhere
and in every draw (checked to 0.00 mas on M0416: the cluster plane and the z = 0.736 perturber scale differently, so a
single efficiency scaling misses by 50-400 mas).  So every source moves on a straight line in the source plane as its
unknown redshift varies,

    beta_i(t) = theta_i - alpha_a,i - t (alpha_b,i - alpha_a,i),

and for a pair the closest approach within the t range both photo-z allow is closed form.

Steps, per lens draw (the two base deflections of a draw are evaluated once at the sources and cached):
  1. Candidates: CANUCS sources inside the field, brighter than --mag-limit, not cluster members (photo-z 84% above
     --z-min), not BCGs, with a t range from the photo-z 2.5-97.5% range (spec-z +-0.01 when available).
  2. Spatial index with colours: at a grid of t values, the sources whose t range reaches that value go into a k-d
     tree whose coordinates are the source-plane position and the colours m_band - m_F150W scaled so that --colour-scale
     mag counts like the search radius.  Only pairs close in both come out; the radius is padded by how far a pair can
     move between two t values, so no closest approach is missed.
  3. Cheap cuts first, on those pairs only: t overlap, colour chi2 (lensing is achromatic), flux ratio against the
     predicted magnification ratio, then the closed-form separation against the source-plane error (image-plane
     --sigma-img mapped by each image's largest singular value of A).
  Across draws: a pair is kept if it passes in at least --min-draw-fraction of them; its t and separation are medians.
  4. Groups: pairs linked greedily, best first, while the group's t values stay within --dt; each group's colour
     coherence (members against their error-weighted mean colour) enters the score.
  5. Forward check on the best draw: the group's mean source traced to all its images on a precomputed grid; each
     predicted image matched to a member within 1", predicted images brighter than --depth with no source nearby count
     against the group.
  6. Classification: catalog families (a recovery test), extensions of them, families published elsewhere (CANUCS
     silver/bronze/Q, Rihtarsic 2025, Bergamini 2023, Caminha 2017), and new ones.

--null mirrors the candidate positions (x -> -x): every group found is then a chance coincidence, which calibrates
the cuts.

Example:
    python scripts/find_new_families.py --run RUN --draws 10 --mag-limit 27.5 --out output/new_families
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_arc_source as fas  # noqa: E402  (paths, x64, working directory)
from find_missing_images import CLUSTER_Z, COLOUR_BANDS, HAS_PHOTOZ, load_photometry, load_published  # noqa: E402
from predict_counter_images import lens_at  # noqa: E402

jnp, np, pd = fas.jnp, fas.np, fas.pd
Z_CLUSTER = 0.396
INDEX_BANDS = ["F115W", "F200W", "F277W", "F356W", "F444W"]     # colours (minus F150W) in the k-d tree; all in COLOUR_BANDS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--draws", type=int, default=10, help="lens draws (the best draw first)")
    p.add_argument("--min-draw-fraction", type=float, default=0.5, help="a pair must pass in at least this fraction of draws")
    p.add_argument("--field", type=float, default=60.0, help="half-side in arcsec of the searched region")
    p.add_argument("--mag-limit", type=float, default=28.0, help="F150W limit of the candidates")
    p.add_argument("--z-min", type=float, default=0.6, help="drop sources whose photo-z 84%% is below this (cluster members, foreground)")
    p.add_argument("--sigma-img", type=float, default=0.25, help="image-plane position error in arcsec (model + astrometry)")
    p.add_argument("--nsigma", type=float, default=3.0)
    p.add_argument("--colour-chi2", type=float, default=2.0, help="mean colour chi2 per band of a pair (0.1 mag floor)")
    p.add_argument("--colour-scale", type=float, default=1.0,
                   help="k-d tree: a Euclidean colour difference of this many mag (over the index bands) counts as the full radius")
    p.add_argument("--flux-factor", type=float, default=5.0, help="allowed factor between the observed and predicted flux ratio")
    p.add_argument("--t-steps", type=int, default=0, help="t values of the spatial index (0 = enough for a 0.3\" pad)")
    p.add_argument("--dt", type=float, default=0.05, help="largest spread of t within a group")
    p.add_argument("--grid-step", type=float, default=0.2, help="step in arcsec of the forward-check grid")
    p.add_argument("--depth", type=float, default=29.5, help="F150W depth: predicted images brighter than this should be seen")
    p.add_argument("--null", action="store_true",
                   help="false-positive test: mirror the candidate positions (x -> -x) so the lens no longer relates them")
    p.add_argument("--cluster", choices=["m0416", "as1063"], default="m0416",
                   help="photometry and published catalogs; AS1063 has no photo-z, so its redshift range is left open and "
                        "sources within --member-radius of a model member are dropped instead")
    p.add_argument("--member-radius", type=float, default=0.6, help="arcsec; used when the cluster has no photo-z")
    p.add_argument("--member-colour-veto", type=float, default=2.0,
                   help="no photo-z: drop sources whose colours lie within this mean squared robust z-score of the model members' "
                        "colour locus (dwarf cluster galaxies share it, so colours alone would group them); 0 = off")
    p.add_argument("--out", default="output/new_families")
    return p.parse_args()


# ------------------------------------------------------------------------------------------------------------- lens
class LinearLens:
    """Deflection and Jacobian at any source redshift as (1 - t) X_a + t X_b from two reference families' redshifts."""

    def __init__(self, evaluator, state, theta):
        self.ev = evaluator
        fam_ids = [str(f) for f in state.images.family_ids]
        zf = {str(f.family_id): float(f.z_source) for f in state.family_data}
        order = sorted(range(len(fam_ids)), key=lambda i: zf[fam_ids[i]])
        self.fa, self.fb = order[0], order[-1]
        self.za, self.zb = zf[fam_ids[self.fa]], zf[fam_ids[self.fb]]
        self.zs, self.ts = self._calibrate(theta)

    def fields(self, xy, theta):
        """alpha and A at both bases for one draw: (alpha_a, alpha_b, A_a, A_b)."""
        out = []
        for f, z in ((self.fa, self.za), (self.fb, self.zb)):
            beta, a = lens_at(self.ev, theta, z, f, np.asarray(xy, float))
            out.append((np.asarray(xy) - beta, a))
        return out[0][0], out[1][0], out[0][1], out[1][1]

    def _calibrate(self, theta):
        """t(z) from the lens-only deflection at 200 points; it depends on the plane redshifts only, so one draw serves all."""
        physical = self.ev.physical(theta)
        xy = np.random.default_rng(0).uniform(-50, 50, (200, 2))
        lens = lambda z: np.asarray(self.ev.total_fields_at(xy, self.ev.packed_state_for_z(physical, z))[0])
        la, lb = lens(self.za), lens(self.zb)
        d = (lb - la).ravel()
        zs = np.concatenate([np.linspace(Z_CLUSTER + 0.02, 1.0, 20), np.linspace(1.0, 12.0, 80)[1:]])
        ts, worst = [], 0.0
        for z in zs:
            r = (lens(z) - la).ravel()
            t = float(r @ d / (d @ d)); ts.append(t)
            worst = max(worst, float(np.abs(r - t * d).max()))
        print(f"[lens] bases z {self.za:.3f} and {self.zb:.3f}; t(z) from {len(zs)} redshifts, worst residual {worst * 1000:.2f} mas")
        return zs, np.maximum.accumulate(np.asarray(ts))

    def t_of_z(self, z):
        return np.interp(z, self.zs, self.ts)

    def z_of_t(self, t):
        return np.interp(t, self.ts, self.zs)


def draw_fields(lens, theta, xy, cache: Path):
    """The two base deflections and Jacobians of one draw at the points ``xy``, cached in ``cache``."""
    if cache.exists():
        with np.load(cache) as f:
            if f["xy"].shape == xy.shape and np.allclose(f["xy"], xy):
                return f["alpha_a"], f["alpha_b"], f["A_a"], f["A_b"]
    alpha_a, alpha_b, A_a, A_b = lens.fields(xy, theta)
    np.savez_compressed(cache, xy=xy, alpha_a=alpha_a, alpha_b=alpha_b, A_a=A_a, A_b=A_b)
    return alpha_a, alpha_b, A_a, A_b


# ------------------------------------------------------------------------------------------------------------ search
def candidates(phot, args):
    c = phot[(phot["x"].abs() < args.field) & (phot["y"].abs() < args.field) & (phot["m_F150W"] < args.mag_limit)
             & (phot["FLAG_BCG"] == 0) & (phot["Z840"] > args.z_min) & np.isfinite(phot["Z025"])].copy()
    spec = c["Z_SPEC"] > 0
    c["z_lo"] = np.where(spec, c["Z_SPEC"] - 0.01, np.maximum(c["Z025"], args.z_min))
    c["z_hi"] = np.where(spec, c["Z_SPEC"] + 0.01, c["Z975"])
    c = c[(c["z_hi"] > args.z_min) & ~(spec & (c["Z_SPEC"] < args.z_min))]
    return c.reset_index(drop=True)


def colour_arrays(c):
    """Colours m_band - m_F150W and their errors, (N, bands), from the photometry columns."""
    col = np.stack([(c[f"m_{b}"] - c["m_F150W"]).to_numpy() for b in COLOUR_BANDS], 1)
    err = np.stack([np.hypot(c[f"e_{b}"], c["e_F150W"]).to_numpy() for b in COLOUR_BANDS], 1)
    return col, err


def index_pairs(p0, d, t_lo, t_hi, col, args):
    """Candidate pairs from k-d trees in (source-plane position, scaled colours) at a grid of t values.  The radius is
    the largest source-plane tolerance plus the distance a pair can move between neighbouring t values."""
    from scipy.spatial import cKDTree
    radius = args.nsigma * args.sigma_img * 2.0 * np.sqrt(2.0)      # |A|_2 rarely exceeds 2 for the images we keep
    lo, hi = float(np.nanmin(t_lo)), float(np.nanmax(t_hi))
    span = 2.0 * float(np.nanmax(np.hypot(*d.T)))                    # largest relative motion per unit t
    steps = args.t_steps or int(np.ceil((hi - lo) * span / (2 * 0.3))) + 1
    ts = np.linspace(lo, hi, steps)
    half = 0.5 * (ts[1] - ts[0]) if steps > 1 else 0.0
    pad = half * span
    feat_col = col[:, [COLOUR_BANDS.index(b) for b in INDEX_BANDS]]
    feat_col = np.where(np.isfinite(feat_col), feat_col, np.nanmedian(feat_col, 0))     # missing band: the median colour
    feat_col = feat_col * (radius + pad) / args.colour_scale
    found = set()
    for t in ts:
        active = np.nonzero((t_lo <= t + half) & (t_hi >= t - half))[0]
        if len(active) < 2:
            continue
        tree = cKDTree(np.column_stack([p0[active] - t * d[active], feat_col[active]]))
        for a, b in tree.query_pairs(radius + pad, output_type="ndarray"):
            i, j = active[a], active[b]
            found.add((min(i, j), max(i, j)))
    return np.asarray(sorted(found), dtype=int).reshape(-1, 2), steps, pad


def exact_tests(ii, jj, theta, p0, d, A_a, A_b, t_lo, t_hi, col, err, flux, args):
    """Cheapest first, on the indexed pairs only: t overlap, separation on the sky, colours, then the closed-form
    source-plane separation against the tolerance and the flux ratio.  Returns (i, j, t*, sep/tol, colour chi2)."""
    lo = np.maximum(t_lo[ii], t_lo[jj]); hi = np.minimum(t_hi[ii], t_hi[jj])
    keep = (lo <= hi) & (np.hypot(*(theta[ii] - theta[jj]).T) > 0.6)          # 0.6": not the same object or a blend
    good = np.isfinite(col[ii]) & np.isfinite(col[jj])
    chi = np.where(good, (col[ii] - col[jj]) ** 2 / (err[ii] ** 2 + err[jj] ** 2 + 2 * 0.1 ** 2), 0.0)
    nb = good.sum(-1)
    chi2 = np.where(nb >= 3, chi.sum(-1) / np.maximum(nb, 1), np.inf)
    keep &= chi2 < args.colour_chi2
    ii, jj, lo, hi, chi2 = ii[keep], jj[keep], lo[keep], hi[keep], chi2[keep]
    dp = p0[ii] - p0[jj]; dd = d[ii] - d[jj]
    t = np.clip(np.sum(dp * dd, 1) / np.maximum(np.sum(dd * dd, 1), 1e-12), lo, hi)
    sep = np.hypot(*(dp - t[:, None] * dd).T)
    Ai = A_a[ii] + t[:, None, None] * (A_b[ii] - A_a[ii]); Aj = A_a[jj] + t[:, None, None] * (A_b[jj] - A_a[jj])
    tol = args.sigma_img * np.hypot(np.linalg.norm(Ai, 2, axis=(1, 2)), np.linalg.norm(Aj, 2, axis=(1, 2)))
    mu_ratio = np.abs(np.linalg.det(Aj) / np.linalg.det(Ai))            # mu_i / mu_j
    flux_ok = np.abs(np.log10(flux[ii] / flux[jj] / np.maximum(mu_ratio, 1e-6))) < np.log10(args.flux_factor)
    keep = (sep < args.nsigma * tol) & flux_ok
    return np.column_stack([ii[keep], jj[keep], t[keep], (sep / tol)[keep], chi2[keep]])


def group_pairs(pairs, n, dt):
    """Greedy linking, best pairs first: two groups merge only if all their t values stay within dt."""
    parent = list(range(n)); tmin = np.full(n, np.inf); tmax = np.full(n, -np.inf)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    edges = {}
    for row in pairs[np.argsort(pairs[:, 3])]:
        i, j, t = int(row[0]), int(row[1]), float(row[2])
        ri, rj = find(i), find(j)
        lo = min(tmin[ri], tmin[rj], t); hi = max(tmax[ri], tmax[rj], t)
        if hi - lo > dt:
            continue
        if ri == rj:
            tmin[ri], tmax[ri] = lo, hi; edges.setdefault(ri, []).append(row)
            continue
        parent[rj] = ri; tmin[ri], tmax[ri] = lo, hi
        edges.setdefault(ri, []).extend(edges.pop(rj, []) + [row])
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return [(members, np.asarray(edges[root])) for root, members in groups.items() if len(members) > 1 and root in edges]


def colour_coherence(members, col, err):
    """Mean colour chi2 per band of the members against their error-weighted mean colour (0.1 mag floor)."""
    c, e = col[members], np.hypot(err[members], 0.1)
    finite = np.isfinite(c) & np.isfinite(e)
    w = np.where(finite, 1 / np.where(finite, e, 1.0) ** 2, 0.0)
    mean = np.sum(np.where(finite, c, 0.0) * w, 0) / np.where(w.sum(0) > 0, w.sum(0), np.nan)
    chi = np.where(finite, (c - mean) ** 2 / np.where(finite, e, 1.0) ** 2, np.nan)
    return float(np.nanmean(chi)) if np.isfinite(chi).any() else np.nan, mean


def forward_images(grid_xy, grid_pa, grid_d, shape, step, beta, t):
    """Images of beta at t on the precomputed grid: cells where both components of beta(x) - beta change sign, refined by
    the cell's linear map."""
    b = (grid_pa - t * grid_d - beta).reshape(*shape, 2)
    out = []
    s0 = np.sign(b)
    ch = [(s0[:-1, :-1, c] != s0[1:, :-1, c]) | (s0[:-1, :-1, c] != s0[:-1, 1:, c]) | (s0[:-1, :-1, c] != s0[1:, 1:, c]) for c in range(2)]
    gx = grid_xy.reshape(*shape, 2)
    for iy, ix in zip(*np.nonzero(ch[0] & ch[1])):
        jac = np.column_stack([(b[iy, ix + 1] - b[iy, ix]) / step, (b[iy + 1, ix] - b[iy, ix]) / step])
        try:
            dx = np.linalg.solve(jac, -b[iy, ix])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.abs(dx) <= step * 1.5):
            x = gx[iy, ix] + dx
            if all(np.hypot(*(x - o)) > 0.3 for o in out):
                out.append(x)
    return np.asarray(out).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------------------ main
def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, state, evaluator, posterior = fas.load_model(Path(args.run))
    draws = fas.pick_draws(posterior, args.draws)
    samples = np.asarray(posterior.samples)
    thetas = [jnp.asarray(samples[k]) for k in draws]
    global Z_CLUSTER
    Z_CLUSTER = CLUSTER_Z[args.cluster]
    lens = LinearLens(evaluator, state, thetas[0])
    phot = load_photometry(state.reference, args.cluster)
    c = candidates(phot, args)
    if not HAS_PHOTOZ[args.cluster]:
        # No photo-z to reject cluster galaxies: drop sources on the model's members (and keep the open redshift range).
        packed = state.packed_lens_spec
        comp = np.asarray(state.members.component_indices)
        mem = np.column_stack([np.asarray(packed.x_center_base)[comp], np.asarray(packed.y_center_base)[comp]])
        dmem = np.hypot(*(c[["x", "y"]].to_numpy()[:, None] - mem[None]).transpose(2, 0, 1)).min(1)
        c = c[dmem > args.member_radius].reset_index(drop=True)
        print(f"[data] no photo-z for {args.cluster}: redshift range open, {int((dmem <= args.member_radius).sum())} sources on members dropped")
        if args.member_colour_veto > 0:
            # Colour locus of the members: the photometry of the sources on the model's members, median and robust spread.
            pxy = phot[["x", "y"]].to_numpy()
            on = np.hypot(*(pxy[:, None] - mem[None]).transpose(2, 0, 1)).min(1) < 0.3
            mcol = np.stack([(phot.loc[on, f"m_{b}"] - phot.loc[on, "m_F150W"]).to_numpy() for b in COLOUR_BANDS], 1)
            med = np.nanmedian(mcol, 0); spread = np.maximum(1.4826 * np.nanmedian(np.abs(mcol - med), 0), 0.05)
            ccol = np.stack([(c[f"m_{b}"] - c["m_F150W"]).to_numpy() for b in COLOUR_BANDS], 1)
            d2 = np.nanmean(((ccol - med) / spread) ** 2, 1)
            veto = d2 < args.member_colour_veto
            c = c[~veto].reset_index(drop=True)
            print(f"[data] member colour locus from {on.sum()} members ("
                  + ", ".join(f"{b}-F150W {m:+.2f}+-{s:.2f}" for b, m, s in zip(COLOUR_BANDS, med, spread))
                  + f"); {int(veto.sum())} candidates with member-like colours dropped")
    obs = np.column_stack([np.asarray(state.images.x_obs, float), np.asarray(state.images.y_obs, float)])
    labels = np.asarray(state.images.image_labels)
    d_cat = np.hypot(*(c[["x", "y"]].to_numpy()[:, None] - obs[None]).transpose(2, 0, 1))
    c["catalog"] = np.where(d_cat.min(1) < 0.3, labels[d_cat.argmin(1)], "")
    if args.null:
        c["x"] = -c["x"]; c["catalog"] = ""
        print("[null] candidate positions mirrored x -> -x: every group found is a chance coincidence")
    xy = c[["x", "y"]].to_numpy()
    col, err = colour_arrays(c)
    flux = 10 ** (-0.4 * c["m_F150W"].to_numpy())
    t_lo, t_hi = lens.t_of_z(c["z_lo"].to_numpy()), lens.t_of_z(c["z_hi"].to_numpy())
    print(f"[data] {len(c)} candidate sources ({(c['catalog'] != '').sum()} on catalog images), {len(draws)} lens draws, {time.time() - t0:.0f} s")

    # Per draw: cached base fields, k-d tree pairs, exact tests; pairs accumulate over draws.
    cache = out / "lens_cache"; cache.mkdir(exist_ok=True)
    found: dict[tuple[int, int], list] = {}
    best_fields = None
    for d_idx, theta in zip(draws, thetas):
        t1 = time.time()
        fields = draw_fields(lens, theta, xy, cache / f"draw_{d_idx}_{args.mag_limit}_{args.field}{'_null' if args.null else ''}.npz")
        best_fields = best_fields or fields
        alpha_a, alpha_b, A_a, A_b = fields
        p0, dvec = xy - alpha_a, alpha_b - alpha_a
        cand, steps, pad = index_pairs(p0, dvec, t_lo, t_hi, col, args)
        passed = exact_tests(cand[:, 0], cand[:, 1], xy, p0, dvec, A_a, A_b, t_lo, t_hi, col, err, flux, args) if len(cand) else np.zeros((0, 5))
        for i, j, t, z, chi in passed:
            found.setdefault((int(i), int(j)), []).append((t, z, chi))
        print(f"[draw {d_idx}] index: {steps} t values, pad {pad:.2f}\", {len(cand)} candidate pairs (of {len(c) * (len(c) - 1) // 2}); "
              f"{len(passed)} pass; {time.time() - t1:.1f} s", flush=True)
    need = max(1, int(np.ceil(args.min_draw_fraction * len(draws))))
    pairs = np.asarray([[i, j, np.median([v[0] for v in vals]), np.median([v[1] for v in vals]), np.median([v[2] for v in vals]),
                         len(vals) / len(draws)] for (i, j), vals in found.items() if len(vals) >= need]).reshape(-1, 6)
    print(f"[pairs] {len(found)} pairs pass in some draw, {len(pairs)} in at least {need} of {len(draws)}")
    groups = group_pairs(pairs, len(c), args.dt)
    print(f"[groups] {len(groups)} groups of 2+ sources")

    # Forward check on the best draw's grid.
    g = np.arange(-args.field - 5, args.field + 5 + args.grid_step / 2, args.grid_step)
    gxx, gyy = np.meshgrid(g, g)
    grid_xy = np.column_stack([gxx.ravel(), gyy.ravel()])
    t2 = time.time()
    ga, gb, gA, gB = draw_fields(lens, thetas[0], grid_xy, cache / f"grid_{draws[0]}_{args.grid_step}_{args.field}.npz")
    print(f"[grid] {len(grid_xy)} points in {time.time() - t2:.0f} s")
    grid_pa, grid_d = grid_xy - ga, gb - ga
    alpha_a, alpha_b, A_a, A_b = best_fields
    p0, dvec = xy - alpha_a, alpha_b - alpha_a
    sky_x = -phot["x"].to_numpy() if args.null else phot["x"].to_numpy()      # the null searches the mirrored sky
    sky_y = phot["y"].to_numpy()
    rows = []
    for k, (members, edges) in enumerate(groups):
        t = float(np.median(edges[:, 2]))
        beta = np.mean(p0[members] - t * dvec[members], 0)
        images = forward_images(grid_xy, grid_pa, grid_d, gxx.shape, args.grid_step, beta, t)
        mem_xy = xy[members]
        matched = [int(np.argmin(np.hypot(*(mem_xy - x).T))) if np.hypot(*(mem_xy - x).T).min() < 1.0 else -1 for x in images]
        missing_bright = 0
        if len(images):
            cell = (np.clip(np.round((images[:, 1] - g[0]) / args.grid_step), 0, len(g) - 1).astype(int) * len(g)
                    + np.clip(np.round((images[:, 0] - g[0]) / args.grid_step), 0, len(g) - 1).astype(int))
            mu_img = 1 / np.abs(np.linalg.det(gA[cell] + t * (gB[cell] - gA[cell])))       # the grid's Jacobian
            Am = A_a[members] + t * (A_b[members] - A_a[members])
            m_src = np.median(c.loc[members, "m_F150W"].to_numpy() + 2.5 * np.log10(1 / np.abs(np.linalg.det(Am))))
            for x, mu, m in zip(images, mu_img, matched):
                near = np.hypot(sky_x - x[0], sky_y - x[1]).min()
                if m < 0 and m_src - 2.5 * np.log10(mu) < args.depth and near > 1.0 and np.all(np.abs(x) < args.field):
                    missing_bright += 1
        coherence, mean_colour = colour_coherence(members, col, err)
        cat = sorted({l for l in c.loc[members, "catalog"] if l})
        kind = "known" if len(cat) == len(members) else ("extends " + ",".join(sorted({l.split('.')[0] for l in cat})) if cat else "new")
        rows.append(dict(group=k, kind=kind, n=len(members), n_predicted=len(images), n_matched=int(sum(m >= 0 for m in matched)),
                         missing_bright=missing_bright, draw_fraction=float(edges[:, 5].min()),
                         z_lens_model=round(float(lens.z_of_t(t)), 3), photo_z_median=float(np.median(c.loc[members, "Z_ML"])),
                         sep_over_tol=float(np.median(edges[:, 3])), pair_colour_chi2=float(np.median(edges[:, 4])),
                         colour_coherence=coherence,
                         colours=",".join(f"{b}:{v:+.2f}" for b, v in zip(COLOUR_BANDS, mean_colour) if np.isfinite(v)),
                         sources=",".join(str(int(s)) for s in c.loc[members, "SOURCE"]), catalog=",".join(cat),
                         x=",".join(f"{v:.2f}" for v in (-mem_xy[:, 0] if args.null else mem_xy[:, 0])),
                         y=",".join(f"{v:.2f}" for v in mem_xy[:, 1]),
                         m_F150W=",".join(f"{v:.1f}" for v in c.loc[members, "m_F150W"]), beta_x=beta[0], beta_y=beta[1]))
    table = pd.DataFrame(rows)
    if len(table) and not args.null:
        # Published images (any class and catalog) within 0.5" of each member: the gold catalog we fit is only part of it.
        pub = load_published(state.reference, args.cluster)

        def published(row):
            found_labels = []
            for x, y in zip(row["x"].split(","), row["y"].split(",")):
                dd = np.hypot(pub["x"] - float(x), pub["y"] - float(y)); j = dd.idxmin()
                found_labels.append(f"{pub['label'][j]} ({pub['cls'][j]})" if dd[j] < 0.5 else "-")
            return ", ".join(found_labels)
        table["published"] = table.apply(published, axis=1)
        table.loc[(table["kind"] == "new") & ~table["published"].str.fullmatch(r"(-, )*-"), "kind"] = "published (not in our catalog)"
    if len(table):
        # Rank: every member reproduced, no unseen bright prediction, more images, stable over draws, tight in the source
        # plane, coherent colours.
        table["score"] = ((table["n_matched"] / table["n"]) * 2 + np.minimum(table["n"], 4) - table["missing_bright"]
                          + table["draw_fraction"] - 0.3 * table["sep_over_tol"] - 0.3 * np.minimum(table["colour_coherence"].fillna(5), 5))
        table = table.sort_values(["kind", "score"], ascending=[True, False])
    table.to_csv(out / ("family_candidates_null.csv" if args.null else "family_candidates.csv"), index=False)

    if not args.null and len(table):
        fam_of = c.loc[c["catalog"] != "", "catalog"].map(lambda l: l.rsplit(".", 1)[0])
        reachable = fam_of.value_counts(); reachable = reachable[reachable >= 2]
        recovered = set()
        for _, row in table.iterrows():
            labs = [l.rsplit(".", 1)[0] for l in row["catalog"].split(",") if l]
            recovered |= {f for f in set(labs) if labs.count(f) >= 2}
        print(f"[recovery] {len(recovered & set(reachable.index))} of {len(reachable)} catalog families with 2+ images among the "
              f"candidates have 2+ of them in one group")
    kinds = table["kind"].value_counts().to_dict() if len(table) else {}
    strict = table[(table["kind"] == "new") & (table["n_matched"] == table["n"]) & (table["missing_bright"] == 0)] if len(table) else table
    print(f"[done] {len(table)} groups {kinds}; new and fully reproduced: {len(strict)} ({(strict['n'] >= 3).sum() if len(strict) else 0} "
          f"with 3+ images); total {time.time() - t0:.0f} s -> {out}")
    if len(strict):
        pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 45)
        print(strict.head(15)[["n", "n_predicted", "draw_fraction", "z_lens_model", "photo_z_median", "sep_over_tol", "colour_coherence",
                               "sources", "x", "y", "m_F150W"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
