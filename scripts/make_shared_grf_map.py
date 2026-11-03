"""The shared-field GRF mass map of a saved grf_gp run; no inference.

A run sampled with system blocks gives each source system its own independent realization of the knee field, so its
conditional means cannot be read as one cluster mass map.  This script rebuilds the run's model with one block over all
images (gp_blocks="all", same parameters and hyperparameters) and evaluates, at the best-fit draw, the conditional mean
deflection of that single field, E[delta_alpha | all images], with every source at its conditional mean (zero source
offsets, i.e. the field mean marginalized over the sources).  It writes:

  final/figures/<cluster>_shared_grf_map.pdf  the convergence of the mean field at z_ref with its deflection, and the mean
                                               deflection at the images under the shared field and under system blocks
  final/tables/shared_grf_map.json            hyperparameters, log-likelihoods under both block layouts, map extremes and
                                               the image-level agreement of the two layouts
  final/tables/shared_grf_map.npz             the grid maps (deflection, convergence, shear at z_ref)
  final/tables/shared_grf_images.csv          per image: shared and system-block mean deflection

Usage: python scripts/make_shared_grf_map.py results/<campaign>/<cluster>/<run> [--project] [--draw best|median]
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT_MODES = ("translation", "convergence", "shear")


def _evaluator(config, blocks, project, classes, init, start):
    from dataclasses import replace
    import numpy as np
    from lenscluster import artifacts
    from lenscluster.evaluator import Evaluator
    from lenscluster.state import build_state, with_member_classes
    likelihood = replace(config.likelihood, gp_blocks=blocks)
    if project is not None:
        likelihood = replace(likelihood, gp_knee_project_modes=project)
    config = replace(config, likelihood=likelihood)
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered_ids = set(classes.get("centered_members", []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered_ids for m in state.members.member_ids]))
    state.initial_theta = start
    return Evaluator(state, config.likelihood.likelihood_mode)


def _halo_centres(config, names, physical):
    """Large-halo centres at this draw (sampled or fixed), skipping the external shear."""
    out = []
    for halo in config.model.large_halos:
        x, y = getattr(halo, "x_centre", None), getattr(halo, "y_centre", None)
        if x is None or y is None:
            continue
        key = f"{halo.id}"
        x = float(physical[names.index(f"{key}.x_centre")]) if f"{key}.x_centre" in names else float(x)
        y = float(physical[names.index(f"{key}.y_centre")]) if f"{key}.y_centre" in names else float(y)
        out.append(dict(id=key, x=x, y=y, sampled=f"{key}.x_centre" in names))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directory", type=Path, help="Directory containing init/ and final/")
    parser.add_argument("--draw", choices=("best", "median"), default="best",
                        help="the highest-log-probability draw (default) or the posterior median in the latent coordinates")
    parser.add_argument("--project", action="store_true",
                        help="remove the translation, convergence and shear modes of the field at the images, as an all-block run does")
    parser.add_argument("--step", type=float, default=1.0, help="grid step in arcsec (default 1)")
    parser.add_argument("--margin", type=float, default=10.0, help="grid margin beyond the images in arcsec (default 10)")
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    import pandas as pd
    from lenscluster import artifacts
    from lenscluster.plot_style import PAPER_STYLE

    run = args.run_directory.resolve()
    init, final = run / "init" / "artifacts", run / "final"
    config = artifacts.load_config(final / "artifacts")
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    start = artifacts.load_start(init)
    project = PROJECT_MODES if args.project else ()

    system = _evaluator(config, config.likelihood.gp_blocks, None, classes, init, start)
    shared = _evaluator(config, "all", project, classes, init, start)
    names = [p.name for p in system.state.parameter_specs]
    if names != [p.name for p in shared.state.parameter_specs]:
        raise ValueError("the all-block model has a different parameter vector")
    saved_state = json.loads((init / artifacts.STATE_FILE).read_text())
    if names != [p["name"] for p in saved_state["parameter_specs"]]:
        raise ValueError("rebuilt parameter order differs from the saved run")

    posterior = artifacts.load_posterior(final / "artifacts")
    samples = np.asarray(posterior.samples)
    if args.draw == "best":
        log_prob = np.asarray(posterior.log_prob)
        index = int(np.nanargmax(log_prob))
        theta = samples[index]
        np.testing.assert_allclose(system.reported_physical(theta), posterior.physical[index], rtol=1e-8, atol=1e-8)
    else:
        index = None
        theta = np.median(samples, axis=0)

    # sources at their conditional mean under each layout: the field mean marginalized over the sources
    images = system.state.images
    theta0 = np.array(theta, dtype=float)
    source_index = np.concatenate([np.asarray(images.source_x_param_index), np.asarray(images.source_y_param_index)])
    theta0[source_index[source_index >= 0]] = 0.0

    loglike = {name: dict(at_draw=float(ev.log_likelihood(jnp.asarray(theta))), sources_at_mean=float(ev.log_likelihood(jnp.asarray(theta0))))
               for name, ev in (("system", system), ("shared", shared))}
    mean_shared = shared.gp_deflection_correction(theta0)
    mean_system = system.gp_deflection_correction(theta0)
    if mean_shared is None or mean_system is None:
        raise ValueError("this run has no knee mass field with a conditional mean")

    physical = np.asarray(system.physical(jnp.asarray(theta)))
    get = lambda key: float(physical[names.index(key)]) if key in names else float("nan")
    weights = np.asarray(shared.gp_weights_per_family, dtype=float)

    # at the images, each at its own source redshift
    x_img, y_img = np.asarray(images.x_obs, float), np.asarray(images.y_obs, float)
    family = np.asarray(images.family_index_per_image)
    sdx, sdy = (np.asarray(a) for a in mean_shared(x_img, y_img, family)[:2])
    bdx, bdy = (np.asarray(a) for a in mean_system(x_img, y_img, family)[:2])
    labels = [str(label) for label in images.image_labels]
    table = pd.DataFrame(dict(image=labels, family=[images.family_ids[int(f)] for f in family],
                              x=x_img, y=y_img, weight=weights[family],
                              shared_dx=sdx, shared_dy=sdy, system_dx=bdx, system_dy=bdy))
    table["shared_amp"] = np.hypot(sdx, sdy)
    table["system_amp"] = np.hypot(bdx, bdy)
    table["difference_amp"] = np.hypot(sdx - bdx, sdy - bdy)

    # on a grid at z_ref: one family's values divided by its redshift weight (one block, so the field is shared)
    lo_x, hi_x = x_img.min() - args.margin, x_img.max() + args.margin
    lo_y, hi_y = y_img.min() - args.margin, y_img.max() + args.margin
    gx, gy = np.arange(lo_x, hi_x + args.step, args.step), np.arange(lo_y, hi_y + args.step, args.step)
    GX, GY = np.meshgrid(gx, gy)
    ref = np.zeros(GX.size, dtype=np.int32)
    gdx, gdy, gxx, gxy, gyx, gyy = (np.asarray(a).reshape(GX.shape) / weights[0] for a in mean_shared(GX.ravel(), GY.ravel(), ref))
    kappa = 0.5 * (gxx + gyy)
    gamma1, gamma2 = 0.5 * (gxx - gyy), 0.5 * (gxy + gyx)
    shear = np.hypot(gamma1, gamma2)

    halos = _halo_centres(config, names, physical)
    peak, trough = np.unravel_index(np.argmax(kappa), kappa.shape), np.unravel_index(np.argmin(kappa), kappa.shape)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    summary = dict(
        run=str(run), draw=args.draw, draw_index=index, project_modes=list(project), run_gp_blocks=config.likelihood.gp_blocks,
        sigma_alpha_arcsec=get("images.gp_sigma_alpha"), knee_arcsec=get("images.gp_knee"),
        gamma=float(getattr(config.likelihood, "gp_knee_gamma", float("nan")) or float("nan")),
        sigma_shared_arcsec=get("images.sigma_shared"), gp_ell_arcsec=get("images.gp_ell"), sigma_white_arcsec=get("images.sigma_white"),
        log_likelihood=loglike,
        log_likelihood_shared_minus_system=loglike["shared"]["at_draw"] - loglike["system"]["at_draw"],
        images=dict(n=int(len(x_img)), shared_rms_mas=1e3 * rms(table.shared_amp), system_rms_mas=1e3 * rms(table.system_amp),
                    difference_rms_mas=1e3 * rms(table.difference_amp),
                    correlation_x=float(np.corrcoef(sdx, bdx)[0, 1]), correlation_y=float(np.corrcoef(sdy, bdy)[0, 1])),
        grid=dict(step_arcsec=args.step, x_range=[float(gx[0]), float(gx[-1])], y_range=[float(gy[0]), float(gy[-1])],
                  kappa_min=float(kappa.min()), kappa_max=float(kappa.max()), kappa_rms=rms(kappa),
                  kappa_peak_xy=[float(GX[peak]), float(GY[peak])], kappa_trough_xy=[float(GX[trough]), float(GY[trough])],
                  shear_max=float(shear.max()), deflection_rms_arcsec=rms(np.hypot(gdx, gdy))),
        halos=halos)
    for halo in halos:
        near = np.hypot(GX - halo["x"], GY - halo["y"]) < 10.0
        halo["kappa_mean_within_10"] = float(kappa[near].mean()) if near.any() else float("nan")

    tables, figures = final / "tables", final / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (tables / "shared_grf_map.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(tables / "shared_grf_map.npz", x=gx, y=gy, deflection_x=gdx, deflection_y=gdy, kappa=kappa, gamma1=gamma1, gamma2=gamma2)
    table.to_csv(tables / "shared_grf_images.csv", index=False)

    plt.style.use(str(ROOT / "src/lenscluster/style.mplstyle"))
    plt.rcParams.update(PAPER_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), constrained_layout=True)
    lim = float(np.percentile(np.abs(kappa), 99.5)) or 1e-6
    im = axes[0].pcolormesh(GX, GY, kappa, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), shading="auto", rasterized=True)
    step = max(1, int(round(6.0 / args.step)))
    axes[0].quiver(GX[::step, ::step], GY[::step, ::step], gdx[::step, ::step], gdy[::step, ::step],
                   color="0.15", angles="xy", scale_units="xy", scale=0.02, width=0.003)
    axes[0].scatter(x_img, y_img, s=2, c="0.45", lw=0, zorder=2)
    fig.colorbar(im, ax=axes[0], label=r"$\kappa_{\rm GRF}$ at $z_{\rm ref}$", shrink=0.85)
    qs = axes[1].quiver(x_img, y_img, sdx, sdy, color="0.1", angles="xy", scale_units="xy", scale=0.02, width=0.004, zorder=3)
    qb = axes[1].quiver(x_img, y_img, bdx, bdy, color="#d95f02", angles="xy", scale_units="xy", scale=0.02, width=0.004, alpha=0.8, zorder=2)
    axes[1].quiverkey(qs, 0.80, 0.06, 0.1, r"$0.1^{\prime\prime}$", labelpos="E", coordinates="axes")
    for ax in axes:
        for halo in halos:
            ax.plot(halo["x"], halo["y"], "k+", ms=7, mew=1.2, zorder=5)
            ax.annotate(halo["id"], (halo["x"], halo["y"]), xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=6)
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [arcsec]")
    axes[0].set_ylabel(r"$y$ [arcsec]")
    handles = [plt.Line2D([], [], color="0.1", lw=1.5, label="shared field"),
               plt.Line2D([], [], color="#d95f02", lw=1.5, label="system blocks"),
               plt.Line2D([], [], color="k", marker="+", ls="none", ms=7, mew=1.2, label="halo centre")]
    axes[1].legend(handles=handles, loc="lower left", frameon=False, fontsize=7)
    path = figures / f"{run.parent.name}_shared_grf_map.pdf"
    fig.savefig(path)
    print(json.dumps({k: v for k, v in summary.items() if k != "halos"}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
