$XONSH_SHOW_TRACEBACK = True
$MPLBACKEND = "Agg"

import json
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import time
from pathlib import Path

cores = 4  # NUTS chains; one JAX CPU device per chain (same budget as run.xsh).
seed_default = 12345
output_dir_default = "validation_runs"
# What the mock injects.  The astrometric noise is the truth of the white image-plane scatter (sigma_white under the GP
# likelihoods, sigma_extra under --likelihood smoothed); the solver assumes only the 1 mas floor of run.xsh and infers the
# rest.  The source-plane scatter is the truth of sigma_def (smoothed likelihood only; under the GP likelihoods it is
# unmodelled scatter that the image-plane terms must absorb); the member scatters are the truths of the two population
# scatters (0.10 is the median of the solver's dispersion-scatter prior).  Unmodelled members are subhalos present in the
# truth but missing from the catalog, the misspecification experiment.
# 0.015" per axis: the scatter between the CANUCS (JWST) and Bergamini et al. (HST) positions of M0416, 0.014-0.015" per
# axis over 207 matched images, and the sigma_white the M0416 fit settles at (0.012-0.015").
sigma_inst_arcsec = 0.015
# The injected white scatter varies between realizations (2026-11-06): log-uniform over 5-20 mas, JWST centroids of
# compact knots, inside the solver's sigma_white prior (run.xsh: median 0.015", log-width 0.5, cap 0.1"); None injects
# sigma_inst_arcsec everywhere.
sigma_inst_range_arcsec = (0.005, 0.02)
SOLVER_SIGMA_INST_ARCSEC = 0.001   # as SIGMA_INST_ARCSEC in run.xsh
sigma_def_arcsec = 0.0           # astrometric error only (2026-11-05); 0.05 adds source-plane scatter the GP likelihoods do not model
scatter_ln = 0.10
cut_scatter_ln = 0.0
unmodelled_members = 0
# Correlated errors, drawn per realization (mock_validation.field): the mass field of unmodelled mass and line-of-sight
# structure with sigma_alpha 0.04-0.10" at z = 2 (the M0416 fits give 0.06-0.11"), knee 3-12" (log-uniform) and slope 3-5;
# the image-plane displacement with sigma 0.02-0.06" and ell 0.5-2" (log-uniform).  Raised back from the JWST amplitudes
# (sigma_alpha 0.02-0.05", sigma_shared 0.01-0.04", nov6b_correlated_v1) on 2026-11-06: at those the free lens absorbed a
# quarter of the field (sigma_alpha 0.67-0.89x truth, 0/4 covered; .agents/nov6-mock-sigma-alpha-free-lens.md) and
# ell was measurable in one seed of four (it needs sigma_shared well above the 5-20 mas white noise).  The solver samples
# the slope (uniform 2-6) since the truth varies it, and its sigma_alpha and sigma_shared priors are centred on these
# ranges (_likelihood).
field_sigma_alpha_range_arcsec = (0.04, 0.10)
field_knee_range_arcsec = (3.0, 12.0)
field_gamma_range = (3.0, 5.0)
image_plane_sigma_range_arcsec = (0.02, 0.06)
image_plane_ell_range_arcsec = (0.5, 2.0)
# Sources with knots (2026-11-06): this many primary families are galaxies with 2-3 knots 0.05-1.0" apart in the source
# plane (log-uniform), each knot its own family of one system, so images come in groups that sample the image-plane GP's
# ell.  Offsets above ~0.3" rarely survive the image-count check (they cross a caustic), so knot groups stay under 2";
# the same-system pairs at 2-15" that pin the field's knee are the fold pairs of the knotted families, duplicated once per
# knot: five knotted families give 100-145 such pairs per mock against 70-80 with three (nov6b).  All-image blocks see
# the 600-1400 cross-system pairs at those separations as well (Fisher knee contraction 80-92% against 76-87% with system
# blocks, ell 58% against 30-34%, seeds 12345-12346).  --knotted-families overrides.
knotted_families = {"production": 5, "smoke": 1}
knots_per_family = (2, 3)
knot_offset_range_arcsec = (0.05, 1.0)

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster-mock"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster-mock"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataclasses import replace

from lenscluster.config import (
    ImageDiagnosticsConfig,
    LensClusterSolverConfig,
    LikelihoodConfig,
    LogNormalPriorConfig,
    MemberConfig,
    RuntimeConfig,
    SamplingConfig,
    ScalingModelConfig,
    SearchConfig,
    TruthRecoveryConfig,
)
from lenscluster.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported

VALID_PRESETS = ("smoke", "production")

# Command-line likelihood names -> (LikelihoodConfig.likelihood_mode, gp_kernel), as in run.xsh.
LIKELIHOODS = {"grf_gp": ("gp_image_plane", "knee"), "gp_image_plane": ("gp_image_plane", "image_plane"),
               "smoothed": ("smoothed", "image_plane")}


def _usage() -> str:
    return ("Usage: xonsh run_mock.xsh --campaign NAME [--preset smoke|production] [--mode production|idealized] [--seed INT] "
            "[--realizations INT] [--output-dir PATH] [--likelihood grf_gp|gp_image_plane|smoothed] [--gp-blocks all|system|family] "
            "[--warmup N] [--samples N] [--knotted-families N] [--fast] [--fisher-only] [--aggregate-only] [--rescore] [--dry-run]\n"
            "  --preset production: the solver settings of a full run.xsh run (256 search starts, 1000+250 NUTS draws); smoke: a few\n"
            "          members and families, 8 starts, 100+100 draws at tree depth 6, for tests of the pipeline itself\n"
            "  --mode production: the solver uses the same priors as the real clusters (halo on the BCG light, Bergamini et al. 2019 member relation);\n"
            "  --mode idealized: priors centred on each realization's truth, which isolates the sampler and the likelihood\n"
            "  --likelihood: grf_gp (default; the mass-field GRF with a knee spectrum plus the image-plane GP, as run.xsh), gp_image_plane\n"
            "          (the image-plane GP alone, per family) or smoothed (the per-image source-plane floors sigma_def, sigma_extra)\n"
            "  --gp-blocks: grf_gp only: all (default: one shared field with translation, convergence and shear projected out, as\n"
            "          full run.xsh runs), system (an independent field per system, no projection; the --fast default) or family\n"
            "  --knotted-families N: primary families that are galaxies with knots (default 5 for production, 1 for smoke)\n"
            "  --fisher-only: generate (or reuse) the mocks and forecast how well the GP hyperparameters are constrained (a Fisher\n"
            "          matrix at the truth), without running the solver; writes fisher.json per mock and fisher_precheck.csv\n"
            "  --warmup N, --samples N: NUTS draws per chain (override the preset or --fast)\n"
            "  --fast: the exploratory settings of run.xsh --fast: far-field members allowed down to 2 cut radii from an image,\n"
            "          classification at the current model, 500+250 draws at tree depth 8, system GP blocks, member shapes fixed")


def _parse_args(argv: list[str]) -> dict:
    values = {"campaign": None, "preset": "smoke", "mode": "production", "seed": str(seed_default), "realizations": "1",
              "output_dir": output_dir_default, "likelihood": "grf_gp", "gp_blocks": None, "warmup": None, "samples": None,
              "knotted_families": None, "fast": False, "fisher_only": False, "aggregate_only": False, "rescore": False, "dry_run": False}
    flag_to_key = {"--campaign": "campaign", "--preset": "preset", "--mode": "mode", "--seed": "seed", "--realizations": "realizations",
                   "--output-dir": "output_dir", "--likelihood": "likelihood", "--gp-blocks": "gp_blocks", "--warmup": "warmup",
                   "--samples": "samples", "--knotted-families": "knotted_families"}
    index = 1
    while index < len(argv):
        flag = argv[index]
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag in {"--aggregate-only", "--rescore", "--dry-run", "--fast", "--fisher-only"}:
            values[flag.lstrip("-").replace("-", "_")] = True
            index += 1
            continue
        if flag not in flag_to_key or index + 1 >= len(argv):
            print(f"Unknown argument or missing value: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        values[flag_to_key[flag]] = argv[index + 1]
        index += 2
    if values["campaign"] is None:
        print(f"Missing required --campaign.\n{_usage()}")
        raise SystemExit(2)
    preset = str(values["preset"]).strip().lower()
    if preset not in VALID_PRESETS:
        print(f"Preset must be 'smoke' or 'production'; got {values['preset']!r}.\n{_usage()}")
        raise SystemExit(2)
    if values["mode"] not in {"production", "idealized"}:
        print(f"Mode must be 'production' or 'idealized'; got {values['mode']!r}.\n{_usage()}")
        raise SystemExit(2)
    if values["likelihood"] not in LIKELIHOODS:
        print(f"--likelihood must be one of {', '.join(LIKELIHOODS)}.")
        raise SystemExit(2)
    if values["gp_blocks"] not in {None, "all", "system", "family"}:
        print("--gp-blocks must be all, system or family.")
        raise SystemExit(2)
    if values["gp_blocks"] is not None and values["likelihood"] != "grf_gp":
        print("--gp-blocks applies to the grf_gp likelihood only.")
        raise SystemExit(2)
    seed, realizations = int(values["seed"]), int(values["realizations"])
    if seed < 0 or realizations <= 0:
        print("Seed must be nonnegative and realizations positive.")
        raise SystemExit(2)
    try:
        for key in ("warmup", "samples"):
            if values[key] is not None:
                values[key] = int(values[key])
                if values[key] <= 0:
                    raise ValueError
    except ValueError:
        print("--warmup and --samples take positive integers.")
        raise SystemExit(2)
    if values["knotted_families"] is not None:
        try:
            values["knotted_families"] = int(values["knotted_families"])
            if values["knotted_families"] < 0:
                raise ValueError
        except ValueError:
            print("--knotted-families takes a nonnegative integer.")
            raise SystemExit(2)
    return dict(preset=preset, mode=values["mode"], seed=seed, realizations=realizations, campaign=str(values["campaign"]).strip(),
                output_dir=str(values["output_dir"]), likelihood=values["likelihood"], gp_blocks=values["gp_blocks"],
                warmup=values["warmup"], samples=values["samples"], fast=values["fast"],
                knotted_families=values["knotted_families"], fisher_only=values["fisher_only"],
                aggregate_only=values["aggregate_only"], rescore=values["rescore"], dry_run=values["dry_run"])


def build_runtime(seed: int) -> RuntimeConfig:
    return RuntimeConfig(seed=seed, chains=cores, resume=True, numpyro_print_summary=True, nuts_chain_method="parallel",
                         cpu_threads_per_chain=threads_per_chain_if_supported(2),   # same CPU budget as run.xsh: chains * 2 cores on Linux, no pinning on macOS
                         skip_plots=False)                     # recovery reads the solver's output tables


def _likelihood(likelihood: str, gp_blocks: str) -> LikelihoodConfig:
    """The likelihood of run.xsh build_config (keep the two in step): no selection term; the GRF of unmodelled mass with a
    knee spectrum (slope sampled on 2-6, knee 2.5-15" around 6") and the image-plane GP (ell 0.4-3" around 1"), with the
    priors centred on the mock's injection ranges (sigma_alpha 0.06", sigma_shared 0.035" median); the shared field's
    translation, convergence and shear projected out on all blocks (system blocks, the --fast default, have no projection)."""
    return LikelihoodConfig(
        likelihood_mode=LIKELIHOODS[likelihood][0], sigma_sel_prior=None, gp_kernel=LIKELIHOODS[likelihood][1],
        gp_blocks=gp_blocks, sigma_inst_arcsec=SOLVER_SIGMA_INST_ARCSEC,
        gp_knee_project_modes=(("translation", "convergence", "shear") if likelihood == "grf_gp" and gp_blocks == "all" else ()),
        gp_knee_gamma_prior=(2.0, 6.0), gp_knee_gamma=4.0,     # sampled: the injected field varies the slope
        gp_knee_scales_arcsec=(0.2, 100.0),
        gp_knee_prior=LogNormalPriorConfig(median=6.0, log_sigma=0.6, lower=2.5, upper=15.0),
        gp_knee_start_arcsec=6.0,
        # Mock only (run.xsh keeps median 0.08", cap 0.5", and the library sigma_shared prior, median 0.10", cap 0.3"): the
        # geometric centres of the mock's injection ranges, each range inside about one prior width of the median.
        gp_sigma_alpha_prior=LogNormalPriorConfig(median=0.06, log_sigma=0.5, upper=0.3),
        gp_sigma_alpha_start_arcsec=0.06,
        gp_knee_sigma_shared_prior=LogNormalPriorConfig(median=0.035, log_sigma=0.6, upper=0.15),
        gp_knee_sigma_shared_start_arcsec=0.035,
        gp_knee_ell_prior=LogNormalPriorConfig(median=1.0, log_sigma=0.4, lower=0.4, upper=3.0),
        gp_knee_ell_start_arcsec=1.0,
    )


def _solver(*, production: bool, fast: bool, seed: int, likelihood: str, gp_blocks: str | None) -> LensClusterSolverConfig:
    """The solver template: the settings of run.xsh build_config (search, sampling, members, likelihood) plus what the mock
    needs.  The lens model, with priors centred on the mock's truth, is attached per realization by the mock-validation
    pipeline.  The smoke preset keeps its small search and chains under --fast; the member and GP settings of --fast apply
    to both presets."""
    # All-image blocks with the projection by default (2026-11-06), as full run.xsh runs: under system blocks the free lens
    # absorbed a quarter of the field (nov6b, .agents/nov6-mock-sigma-alpha-free-lens.md) and the fit lost the 600-1400
    # cross-system pairs at 2-15" that pin the knee and ell.  --fast keeps system blocks, about 5x cheaper per gradient.
    blocks = gp_blocks or ("system" if fast else "all")
    config = LensClusterSolverConfig(
        runtime=build_runtime(seed),
        search=SearchConfig(n_starts=128) if production else SearchConfig(n_starts=8),
        # Far-field member offsets pinned at the relation: their radius offsets never reach the likelihood and their velocity
        # offsets barely do, so they were sampled from the prior alone and set the orbit length (the softest directions,
        # 400-700 leapfrog steps per half orbit against the 255 cap, astrometric_bcg1as seed 12345).  Target accept 0.8 as
        # run.xsh (the chains accepted 0.92-0.97 at 0.85, steps at 0.3-0.5 of the stiffness limit) and 1000 warmup.
        sampling=(SamplingConfig(warmup=2000, samples=500, max_tree_depth=10, target_accept=0.8, far_field_offsets="fixed") if production
                  else SamplingConfig(warmup=100, samples=100, max_tree_depth=6, target_accept=0.8, far_field_offsets="fixed")),
        likelihood=_likelihood(likelihood, blocks),
        # The 0.15 kpc member core is not a lensing observable; fixed at the truth value.  Normal member deviations: the
        # mock's scatter is Gaussian, and the Student-t tails of run.xsh soften the posterior.
        scaling=ScalingModelConfig(fit_corekpc=False, member_deviation_tail="normal"),
        # Centered deviations for the members the data pin, unit offsets for the rest, classification at the current model
        # (reclassified at the MAP), as for the observed clusters.
        members=MemberConfig(centering="auto", classification_prior_quantile=None),
        image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=64 if production else 8, match_tolerance_arcsec=3.0),
        truth=TruthRecoveryConfig(posterior_truth_recovery_draws=0),
    )
    if fast:
        config = replace(
            config,
            sampling=SamplingConfig(warmup=500, samples=250, max_tree_depth=8, target_accept=0.8, far_field_offsets="fixed") if production else config.sampling,
            image_diagnostics=replace(config.image_diagnostics, fold_max_draws=16, fold_pair_max_separation_arcsec=3.0),
            members=replace(config.members, far_field_cut_radii=2.0, classification_prior_quantile=None, min_deflection_arcsec=None,
                            cut_radius_inflation=3.0),
            scaling=replace(config.scaling, member_shape_scatter=None),
        )
    return config


def build_config(preset: str, *, seed: int, realizations: int, campaign_name: str, output_dir: str, mode: str, fast: bool = False,
                 likelihood: str = "grf_gp", gp_blocks: str | None = None, warmup: int | None = None, samples: int | None = None,
                 knotted: int | None = None):
    from lenscluster.mock_validation import MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig
    production = preset == "production"
    mock = MockConfig(
        seed=seed,
        mode=mode,
        n_members=150 if production else 10,   # Hera has 332 members, AS1063 222; far-field approximation for the distant ones (library default)
        n_primary_families=15 if production else 4,
        n_member_families=5 if production else 1,   # galaxy-galaxy families are what constrain sigma* and r_cut*
        sigma_inst_arcsec=sigma_inst_arcsec,
        sigma_inst_range_arcsec=sigma_inst_range_arcsec,
        solver_sigma_inst_arcsec=SOLVER_SIGMA_INST_ARCSEC,
        n_knotted_families=knotted_families[preset] if knotted is None else int(knotted),
        knots_per_family=knots_per_family,
        knot_offset_range_arcsec=knot_offset_range_arcsec,
        sigma_def_arcsec=sigma_def_arcsec,
        sigma_scatter_ln=scatter_ln,
        cut_scatter_ln=cut_scatter_ln,
        unmodelled_members=unmodelled_members,
        field_sigma_alpha_range_arcsec=field_sigma_alpha_range_arcsec,
        field_knee_range_arcsec=field_knee_range_arcsec,
        field_gamma_range=field_gamma_range,
        image_plane_sigma_range_arcsec=image_plane_sigma_range_arcsec,
        image_plane_ell_range_arcsec=image_plane_ell_range_arcsec,
    )
    solver = _solver(production=production, fast=fast, seed=seed, likelihood=likelihood, gp_blocks=gp_blocks)
    sampling = solver.sampling
    if warmup is not None:
        sampling = replace(sampling, warmup=warmup)
    if samples is not None:
        sampling = replace(sampling, samples=samples)
    solver = replace(solver, sampling=sampling)
    run_name = f"analytic_{mode}_{preset}" + ("_fast" if fast else "") + ("" if likelihood == "grf_gp" else f"_{likelihood}")
    return MockValidationConfig(
        mock=mock,
        paths=MockValidationPathsConfig(output_dir=output_dir, campaign_name=campaign_name, run_name=run_name),
        runtime=MockValidationRuntimeConfig(realizations=realizations, seed=seed, resume=True, quiet=False),
        solver=solver,
    )


def _fast_mode_warning(solver) -> None:
    """A banner that cannot be missed: a --fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = solver.members
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {solver.search.n_starts} search starts, {solver.sampling.warmup} warmup + {solver.sampling.samples} draws per chain "
            f"at tree depth {solver.sampling.max_tree_depth}, target accept {solver.sampling.target_accept:g}\n"
            f"  - GP covariance blocks: {solver.likelihood.gp_blocks}, projection {', '.join(solver.likelihood.gp_knee_project_modes) or 'off'} "
            f"(full run.xsh runs use all blocks with the projection)")
    Console(stderr=False).print(Panel(text, title="[bold]WARNING: --fast[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    runtime = build_runtime(options["seed"])
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(options["preset"], seed=options["seed"], realizations=options["realizations"],
                          campaign_name=options["campaign"], output_dir=options["output_dir"], mode=options["mode"],
                          fast=options["fast"], likelihood=options["likelihood"], gp_blocks=options["gp_blocks"],
                          warmup=options["warmup"], samples=options["samples"], knotted=options["knotted_families"]).validate()
    campaign_dir = Path(config.paths.output_dir) / config.paths.campaign_name / config.paths.run_name
    if options["dry_run"]:
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), "campaign_dir": str(campaign_dir), **config.to_json_dict()}, indent=2, default=str))
        return 0
    from lenscluster.mock_validation import fisher_precheck, rescore_mock_validation, run_mock_validation, write_aggregate_plots
    if options["fisher_only"]:
        start = time.monotonic()
        try:
            fisher_precheck(config)
        finally:
            print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
        return 0
    if options["fast"]:
        _fast_mode_warning(config.solver)
    print(f"[config] preset={options['preset']} mode={options['mode']} fast={options['fast']} seed={options['seed']} realizations={options['realizations']} "
          f"scatter_ln={scatter_ln} injected_noise_arcsec={sigma_inst_range_arcsec or sigma_inst_arcsec} knotted_families={config.mock.n_knotted_families} solver_sigma_inst_arcsec={SOLVER_SIGMA_INST_ARCSEC} sigma_def_arcsec={sigma_def_arcsec} members={config.mock.n_members} "
          f"n_starts={config.solver.search.n_starts} warmup={config.solver.sampling.warmup} samples={config.solver.sampling.samples} "
          f"likelihood={options['likelihood']}"
          + {"grf_gp": f" (GRF of unmodelled mass with a knee spectrum + image-plane GP) gp_blocks={config.solver.likelihood.gp_blocks}",
             "gp_image_plane": " (image-plane GP per family)", "smoothed": " (per-image source-plane floors)"}[options["likelihood"]]
          + f" chains={cores} cpus={cpu_plan.selected_cpu_ids} campaign_dir={campaign_dir}")
    start = time.monotonic()
    try:
        if options["rescore"]:
            rescore_mock_validation(config)
        elif not options["aggregate_only"]:
            run_mock_validation(config)
        if options["rescore"] or options["aggregate_only"]:
            for path in write_aggregate_plots(campaign_dir):
                print(f"[aggregate] {path}")
        if options["fast"]:
            _fast_mode_warning(config.solver)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Nov 6 1:34 PM

