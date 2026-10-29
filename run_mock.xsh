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
# What the mock injects.  The astrometric noise is also the solver's instrumental error sigma_inst (the floor under the
# sampled sigma_extra); the source-plane scatter is the truth of sigma_def; the member scatters are the truths of the two
# population scatters (0.10 is the median of the solver's dispersion-scatter prior).  Unmodelled members are subhalos
# present in the truth but missing from the catalog, the misspecification experiment.
sigma_inst_arcsec = 0.10
sigma_def_arcsec = 0.05
scatter_ln = 0.10
cut_scatter_ln = 0.0
unmodelled_members = 0

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster-mock"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster-mock"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    ImageDiagnosticsConfig,
    LensClusterSolverConfig,
    LikelihoodConfig,
    RuntimeConfig,
    SamplingConfig,
    ScalingModelConfig,
    SearchConfig,
    TruthRecoveryConfig,
)
from lenscluster.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported

VALID_PRESETS = ("smoke", "production")


def _usage() -> str:
    return ("Usage: xonsh run_mock.xsh --campaign NAME [--preset smoke|production] [--mode production|idealized] [--seed INT] "
            "[--realizations INT] [--output-dir PATH] [--aggregate-only] [--rescore] [--dry-run]\n"
            "  --mode production: the solver uses the same priors as the real clusters (halo on the BCG light, Bergamini et al. 2019 member relation);\n"
            "  --mode idealized: priors centred on each realization's truth, which isolates the sampler and the likelihood")


def _parse_args(argv: list[str]) -> dict:
    values = {"campaign": None, "preset": "smoke", "mode": "production", "seed": str(seed_default), "realizations": "1",
              "output_dir": output_dir_default, "aggregate_only": False, "rescore": False, "dry_run": False}
    flag_to_key = {"--campaign": "campaign", "--preset": "preset", "--mode": "mode", "--seed": "seed", "--realizations": "realizations", "--output-dir": "output_dir"}
    index = 1
    while index < len(argv):
        flag = argv[index]
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag in {"--aggregate-only", "--rescore", "--dry-run"}:
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
    seed, realizations = int(values["seed"]), int(values["realizations"])
    if seed < 0 or realizations <= 0:
        print("Seed must be nonnegative and realizations positive.")
        raise SystemExit(2)
    return dict(preset=preset, mode=values["mode"], seed=seed, realizations=realizations, campaign=str(values["campaign"]).strip(),
                output_dir=str(values["output_dir"]), aggregate_only=values["aggregate_only"], rescore=values["rescore"], dry_run=values["dry_run"])


def build_runtime(seed: int) -> RuntimeConfig:
    return RuntimeConfig(seed=seed, chains=cores, resume=True, numpyro_print_summary=True, nuts_chain_method="parallel",
                         cpu_threads_per_chain=threads_per_chain_if_supported(2))   # same CPU budget as run.xsh: chains * 2 cores on Linux, no pinning on macOS


def _solver(*, production: bool, seed: int) -> LensClusterSolverConfig:
    """The solver template: library defaults (the tuned Hera settings) plus what the mock needs.  The lens model, with
    priors centred on the mock's truth, is attached per realization by the mock-validation pipeline."""
    return LensClusterSolverConfig(
        runtime=build_runtime(seed),
        search=SearchConfig() if production else SearchConfig(n_starts=8),
        sampling=(SamplingConfig(warmup=5000, samples=2000, max_tree_depth=8, target_accept=0.85) if production
                  else SamplingConfig(warmup=100, samples=100, max_tree_depth=6, target_accept=0.85)),
        likelihood=LikelihoodConfig(sigma_inst_arcsec=sigma_inst_arcsec),
        scaling=ScalingModelConfig(fit_corekpc=False),   # the 0.15 kpc member core is not a lensing observable; fixed at the truth value
        image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=64 if production else 8, match_tolerance_arcsec=3.0),
        truth=TruthRecoveryConfig(posterior_truth_recovery_draws=0),
    )


def build_config(preset: str, *, seed: int, realizations: int, campaign_name: str, output_dir: str, mode: str):
    from lenscluster.mock_validation import MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig
    production = preset == "production"
    mock = MockConfig(
        seed=seed,
        mode=mode,
        n_members=200 if production else 10,   # Hera has 332 members, AS1063 222; far-field approximation for the distant ones (library default)
        n_primary_families=15 if production else 4,
        n_member_families=15 if production else 1,   # galaxy-galaxy families are what constrain sigma* and r_cut*
        sigma_inst_arcsec=sigma_inst_arcsec,
        sigma_def_arcsec=sigma_def_arcsec,
        sigma_scatter_ln=scatter_ln,
        cut_scatter_ln=cut_scatter_ln,
        unmodelled_members=unmodelled_members,
    )
    return MockValidationConfig(
        mock=mock,
        paths=MockValidationPathsConfig(output_dir=output_dir, campaign_name=campaign_name, run_name=f"analytic_{mode}_{preset}"),
        runtime=MockValidationRuntimeConfig(realizations=realizations, seed=seed, resume=True, quiet=False),
        solver=_solver(production=production, seed=seed),
    )


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    runtime = build_runtime(options["seed"])
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(options["preset"], seed=options["seed"], realizations=options["realizations"],
                          campaign_name=options["campaign"], output_dir=options["output_dir"], mode=options["mode"]).validate()
    campaign_dir = Path(config.paths.output_dir) / config.paths.campaign_name / config.paths.run_name
    if options["dry_run"]:
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), "campaign_dir": str(campaign_dir), **config.to_json_dict()}, indent=2, default=str))
        return 0
    from lenscluster.mock_validation import rescore_mock_validation, run_mock_validation, write_aggregate_plots
    print(f"[config] preset={options['preset']} mode={options['mode']} seed={options['seed']} realizations={options['realizations']} "
          f"scatter_ln={scatter_ln} sigma_inst_arcsec={sigma_inst_arcsec} sigma_def_arcsec={sigma_def_arcsec} members={config.mock.n_members} "
          f"n_starts={config.solver.search.n_starts} chains={cores} cpus={cpu_plan.selected_cpu_ids} campaign_dir={campaign_dir}")
    start = time.monotonic()
    try:
        if options["rescore"]:
            rescore_mock_validation(config)
        elif not options["aggregate_only"]:
            run_mock_validation(config)
        for path in write_aggregate_plots(campaign_dir):
            print(f"[aggregate] {path}")
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
