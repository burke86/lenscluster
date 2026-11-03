"""Compare mass-field spectra on the CANUCS M0416 model, every parameter free on the all-image covariance:

    knee              the reference: P = 1 / (1 + (k / k_t)^4), knee free (the 0.2-100" band)
    two_knee          A: a second knee field: a large-scale one (knee 10-80") and a small-scale one (knee below 10")
    broken_power_law  B: P = x^-beta_low (1 + x^4)^-((4 - beta_low) / 4), x = k / k_b; beta_low = 0 is the knee
    power_law         C: P = k^-gamma between the cutoffs, no knee
    knee_foreground   D: the knee plus a second knee field with redshift weight 1 (mass well in front of the cluster)
    fixed             the knee and the image-plane GP fixed at what we want them to model: unmodelled isothermal subhalos
                      truncated at --fixed-rt-kpc (gamma 2, lambda_t = 6.38 r_t, the knee of a population of truncated
                      isothermal halos), and measurement errors correlated over --fixed-ell arcsec (sigma_shared <= 0.1")

The knee is fitted first, from the saved init start of RUN_DIR (its member classes and dropped members); the other
variants start from the knee's optimum, the parameters they add at the values that make them the knee (second field
1e-3", beta_low 0), so a variant that contains the knee can only gain.  Each fit is the batched optax L-BFGS of the
init's chain-start step in rounds of --round-iter iterations (stop: a gain below --tolerance over 50 iterations),
restarted from its best point while a round gains more than --restart-gain, up to --max-iter iterations in all.

Writes <out>/<variant>.json and <out>/<variant>_theta.npy and the table <out>/summary.txt.  With --tight-gp the
image-plane GP is held to measurement error (sigma_shared <= 0.1", ell <= 2") in every variant.  Run one variant at a time:

    python scripts/compare_grf_spectra.py RUN_DIR --variants knee
    python scripts/compare_grf_spectra.py RUN_DIR --variants two_knee
    ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = ("knee", "two_knee", "broken_power_law", "power_law", "knee_foreground", "fixed")
NESTED = {"two_knee", "broken_power_law", "knee_foreground"}               # contain the knee
GRF_SITES = ("images.gp_sigma_alpha", "images.gp_knee", "images.gp_knee_gamma", "images.gp_sigma_alpha2", "images.gp_knee2",
             "images.gp_knee_beta_low", "images.sigma_shared", "images.gp_ell", "images.sigma_white", "shear.gamma1", "shear.gamma2")
NULL_START = {"images.gp_sigma_alpha2": math.log(1.0e-3), "images.gp_knee_beta_low": 0.0}      # the added parameters' null values
ISOTHERMAL_KNEE_PER_RT = 6.38          # lambda_t / r_t of the gamma = 2 knee fitted to (1 - exp(-k r_t))^2 / k^2


def variant_likelihood(likelihood, variant: str, tight_gp: bool, *, fixed_knee: float, fixed_ell: float):
    from lenscluster.config import LogNormalPriorConfig as LN
    tight = dict(gp_knee_sigma_shared_prior=LN(median=0.03, log_sigma=0.7, upper=0.1), gp_knee_sigma_shared_start_arcsec=0.03,
                 gp_knee_ell_prior=LN(median=1.0, log_sigma=0.5, upper=2.0), gp_knee_ell_start_arcsec=1.0)
    common = dict(gp_blocks="all", gp_knee_scales_arcsec=(0.2, 100.0), gp_knee_spectrum=variant, gp_knee_gamma=4.0,
                  gp_knee_gamma_prior=None, gp_knee_prior=LN(median=20.0, log_sigma=1.0, upper=80.0), gp_knee_start_arcsec=20.0,
                  **(tight if tight_gp else {}))
    extra = {
        "knee": {},
        "two_knee": dict(gp_knee_prior=LN(median=30.0, log_sigma=0.7, upper=80.0, lower=10.0), gp_knee_start_arcsec=30.0,
                         gp_knee2_prior=LN(median=3.0, log_sigma=0.5, upper=10.0), gp_knee2_start_arcsec=3.0,
                         gp_knee2_sigma_alpha_prior=LN(median=0.05, log_sigma=0.7, upper=0.3), gp_knee2_sigma_alpha_start_arcsec=0.05),
        "broken_power_law": dict(gp_knee_beta_low_prior=(-1.0, 3.5), gp_knee_beta_low_start=0.0),
        "power_law": dict(gp_knee_gamma_prior=(0.5, 6.0), gp_knee_gamma=3.0),
        "knee_foreground": dict(gp_knee2_prior=LN(median=20.0, log_sigma=1.0, upper=80.0), gp_knee2_start_arcsec=20.0,
                                gp_knee2_sigma_alpha_prior=LN(median=0.05, log_sigma=0.7, upper=0.3), gp_knee2_sigma_alpha_start_arcsec=0.05),
        "fixed": dict(gp_knee_spectrum="knee", gp_knee_gamma=2.0, gp_knee_fixed_arcsec=float(fixed_knee), gp_knee_ell_fixed_arcsec=float(fixed_ell),
                      **tight),
    }[variant]
    return replace(likelihood, **{**common, **extra})


def optimize(objective, x0, *, round_iter: int, tolerance: float, restart_gain: float, max_iter: int, log):
    """L-BFGS in rounds, each from the best point of the last, while a round gains more than ``restart_gain``."""
    import numpy as np
    import jax.numpy as jnp
    from lenscluster.search import run_optax_lbfgs
    x, value = np.asarray(x0, dtype=float), -float(objective.reduced(jnp.asarray(x0)))
    total, rounds, gain = 0, 0, math.inf
    while total < max_iter:
        result = run_optax_lbfgs(objective, x[None], int(min(round_iter, max_iter - total)), tolerance=float(tolerance))
        total += int(result["iterations"][0])
        rounds += 1
        new = -float(result["potential"][0])
        gain = new - value if np.isfinite(new) else 0.0
        if gain > 0.0:
            x, value = np.asarray(result["final"][0], dtype=float), new
        log(f"  round {rounds}: {int(result['iterations'][0])} iterations, log posterior {value:.3f} (gain {gain:+.3f})")
        if gain < restart_gain:
            break
    return x, value, total, rounds, bool(gain < restart_gain)


def fit_variant(run_dir: Path, variant: str, out: Path, *, args) -> dict:
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from lenscluster import artifacts
    from lenscluster.evaluator import Evaluator
    from lenscluster.search import Objective
    from lenscluster.state import build_state, with_member_classes

    ns = {"__file__": str(ROOT / "run.xsh"), "__name__": "compare_grf_spectra"}
    exec(compile("\n".join(l for l in (ROOT / "run.xsh").read_text().splitlines() if not l.startswith("$")), "run.xsh", "exec"), ns)
    config = ns["build_config"]("M0416_CANUCS2025", cores=8, campaign_name="compare_grf_spectra")
    if args.drop_halos:
        config = replace(config, model=replace(config.model, large_halos=tuple(h for h in config.model.large_halos if h.id not in args.drop_halos)))
    kpc = float(ns["kpc_per_arcsec_from_config"](0.396, {"H0": 70.0, "Om0": 0.3}))
    fixed_knee = ISOTHERMAL_KNEE_PER_RT * float(args.fixed_rt_kpc) / kpc
    config = replace(config, likelihood=variant_likelihood(config.likelihood, variant, args.tight_gp, fixed_knee=fixed_knee,
                                                           fixed_ell=args.fixed_ell)).validate()

    saved = run_dir / "init" / "artifacts"
    classes = json.loads((saved / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered = np.asarray([m in set(classes.get("centered_members", [])) for m in state.members.member_ids], dtype=bool)
    state = with_member_classes(state, artifacts.load_exact_positions(saved), classes, centered=centered)
    names = [s.name for s in state.parameter_specs]
    theta = np.asarray(state.initial_theta, dtype=float)
    if variant == "knee":
        source = "the saved init start"
        start_names = [s["name"] for s in json.loads((saved / artifacts.STATE_FILE).read_text())["parameter_specs"]]
        start_theta = artifacts.load_start(saved)
        skip = set()                                                  # a saved knee outside the prior bounds is not copied
    else:
        knee_file = out / "knee.json"
        if not knee_file.exists():
            raise SystemExit(f"fit the knee first (--variants knee): {knee_file} is missing")
        source = "the knee optimum"
        start_names = json.loads(knee_file.read_text())["parameter_names"]
        start_theta = np.load(out / "knee_theta.npy")
        skip = set()
    shared = 0
    for name, value in zip(start_names, start_theta):
        if name in names and name not in skip:
            j = names.index(name)
            spec = state.parameter_specs[j]
            if spec.lower < value < spec.upper:
                theta[j] = value
                shared += 1
    if variant != "knee":
        for name, value in NULL_START.items():
            if name in names:
                theta[names.index(name)] = value

    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    objective = Objective(state, evaluator, theta, include_latent=True, log_likelihood=evaluator.log_likelihood)
    index = np.asarray(objective.global_index)
    x0 = objective.flatten(theta)[index]
    before = -float(objective.reduced(jnp.asarray(x0)))
    log = lambda text: print(f"[{variant}] {text}", flush=True)
    log(f"{state.n_parameters} parameters ({shared} from {source}), log posterior at the start {before:.3f}")
    started = time.perf_counter()
    x, log_posterior, iterations, rounds, converged = optimize(objective, x0, round_iter=args.round_iter, tolerance=args.tolerance,
                                                               restart_gain=args.restart_gain, max_iter=args.max_iter, log=log)
    seconds = time.perf_counter() - started
    final = objective.theta(objective.expand(x))
    physical = np.asarray(evaluator.physical(jnp.asarray(final)))
    log_likelihood = float(evaluator.loglike(jnp.asarray(final)))
    sites = {n: float(physical[names.index(n)]) for n in GRF_SITES if n in names}
    if variant == "fixed":
        sites["fixed_knee"], sites["fixed_ell"] = float(fixed_knee), float(args.fixed_ell)
    n_data = 2 * int(state.images.n_images)
    summary = dict(variant=variant, run=str(run_dir), start=source, tight_gp=bool(args.tight_gp or variant == "fixed"),
                   drop_halos=list(args.drop_halos), n_parameters=int(state.n_parameters), parameter_names=names,
                   log_posterior_start=before, log_posterior=log_posterior, log_likelihood=log_likelihood,
                   kinematics_log_likelihood=float(evaluator.kinematics_log_likelihood(evaluator.physical(jnp.asarray(final)))),
                   bic=-2.0 * log_likelihood + state.n_parameters * math.log(n_data), n_data=n_data,
                   iterations=iterations, rounds=rounds, converged=converged, seconds=seconds, sites=sites,
                   fixed_rt_kpc=float(args.fixed_rt_kpc) if variant == "fixed" else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{variant}.json").write_text(json.dumps(summary, indent=1))
    np.save(out / f"{variant}_theta.npy", final)
    log(f"log posterior {before:.2f} -> {log_posterior:.2f} (log L {log_likelihood:.2f}) in {iterations} iterations, {rounds} rounds, "
        f"{'converged' if converged else 'NOT converged (iteration cap)'}, {seconds:.0f}s; "
        + ", ".join(f"{k.split('.')[-1]} {v:.4g}" for k, v in sites.items()))
    return summary


def write_summary(out: Path) -> str:
    rows = [json.loads((out / f"{v}.json").read_text()) for v in VARIANTS if (out / f"{v}.json").exists()]
    if not rows:
        return "no results"
    base = next((r for r in rows if r["variant"] == "knee"), rows[0])
    lines = [f"knee start: {base['run']}; tight GP: {base['tight_gp']}; differences against {base['variant']} "
             f"(! a variant that contains the knee ended below it: not converged)",
             f"{'variant':18s} {'params':>6s} {'log post':>10s} {'d logpost':>10s} {'log L':>10s} {'d log L':>9s}  {'d BIC':>8s} "
             f"{'iter':>6s} {'rnd':>3s} {'conv':>5s}  hyperparameters"]
    for r in rows:
        d_log_l = r["log_likelihood"] - base["log_likelihood"]
        flag = "!" if r["variant"] in NESTED and d_log_l < -0.1 else " "
        hyper = ", ".join(f"{k.split('.')[-1]} {v:.3g}" for k, v in r["sites"].items() if not k.startswith("shear"))
        lines.append(f"{r['variant']:18s} {r['n_parameters']:6d} {r['log_posterior']:10.2f} {r['log_posterior'] - base['log_posterior']:+10.2f} "
                     f"{r['log_likelihood']:10.2f} {d_log_l:+9.2f}{flag} {r['bic'] - base['bic']:+8.1f} {r['iterations']:6d} {r['rounds']:3d} "
                     f"{str(r['converged']):>5s}  {hyper}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="a CANUCS run directory whose init start is the knee's start")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, help="fitted in this order (the knee first)")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default <run_dir>/grf_spectra2[_tightgp])")
    parser.add_argument("--round-iter", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=0.01, help="a round stops when it gains less than this over 50 iterations")
    parser.add_argument("--restart-gain", type=float, default=0.05, help="restart while a round gains more than this")
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--tight-gp", action="store_true", help="hold the image-plane GP to measurement error (sigma_shared <= 0.1\", ell <= 2\")")
    parser.add_argument("--fixed-rt-kpc", type=float, default=5.0, help="'fixed': truncation radius of the unmodelled subhalos (kpc)")
    parser.add_argument("--fixed-ell", type=float, default=1.0, help="'fixed': correlation length of the measurement errors (arcsec)")
    parser.add_argument("--drop-halos", nargs="*", default=[], help="halo ids to leave out of the model")
    parser.add_argument("--summary", action="store_true", help="only write the summary of the variants already fitted")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out or (args.run_dir / ("grf_spectra2_tightgp" if args.tight_gp else "grf_spectra2"))
    if not args.summary:
        for variant in sorted(args.variants, key=VARIANTS.index):
            fit_variant(args.run_dir, variant, out, args=args)
    print(write_summary(out))


if __name__ == "__main__":
    main()
