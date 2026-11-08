# Quickstart

A fit is configured by one {class}`~lenscluster.config.LensClusterSolverConfig` and run in two stages: **init** (a
multi-start MAP search for the posterior mode) and **final** (NUTS from that mode, then tables and figures). The
launcher `run.xsh` holds ready-made configurations for the supported clusters, so the quickest way in is to run one of
them.

## Fit a cluster

The supported clusters are `HERA`, `ARES` (FF-SIMS simulations, with truth maps), `AS1063`, `AS1063_JWST`,
`M0416_BERGAMINI2023` and `M0416_CANUCS2025`.

```bash
xonsh run.xsh HERA --dry-run                       # print the model and run sizes, write nothing
xonsh run.xsh HERA --fast --campaign demo          # exploratory settings: minutes, not hours
xonsh run.xsh HERA --campaign demo --run-label v1  # the full run
```

The dry run builds the model and reports the number of parameters, the provisional split of members into exact and
far-field, and the search and sampling sizes. Use it to check a configuration before spending compute.

`--fast` uses exploratory settings (looser far-field classification, independent GP blocks per system, shorter chains)
and prints a warning banner before and after the run. Use it to check that a model works; publish full runs.

## Where the results go

Each run writes to `results/<campaign>/<cluster>/<run-label>/`:

```text
init/
  artifacts/   start, modes, member classification, configuration
  tables/      classification_report.txt
  logs/        init.log
final/
  artifacts/   posterior.npz, audit.json, summary.json
  tables/      posterior_summary.csv, image_fit_quality.csv, run_summary.txt, ...
  figures/     <cluster>_critical_curves.pdf, <cluster>_corner_globals.pdf, ...
  fits/        convergence, shear, det A and magnification maps
  logs/        final.log
```

Three things to check first:

1. The last line of `init/logs/init.log`: **robust** when several independent starts reach the best basin, **fragile**
   when only one does.
2. `final/tables/run_summary.txt`: images recovered out of observed, the rms image-plane residual, and the sampler
   diagnostics (split R-hat, ESS, divergences).
3. `final/figures/<cluster>_chain_traces.pdf`: the traces of the global parameters and of the worst-mixing ones.

See {doc}`outputs` for every file.

## Rerunning and resuming

A run directory is reused: rerunning the same command reloads a completed init and a completed final stage instead of
recomputing them. A changed configuration under the same run label is refused, so use a new `--run-label` when you
change the model or the sampler. `--stage init` runs the search alone; a later `--stage final` samples from it.

## From Python

`run.xsh` builds a configuration and calls {func}`lenscluster.pipeline.run`. The same works from Python with any
configuration:

```python
from lenscluster.config import (
    CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
    MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, SamplingConfig, ShearHaloConfig,
)
from lenscluster.pipeline import dry_run, run

z_lens = 0.507
config = LensClusterSolverConfig(
    model=LensModelConfig(
        reference=ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0),
        cosmology=CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        large_halos=(
            DPIEHaloConfig.cluster_scale("1", 19.54, 2.38, z_lens, v_disp=800.0),
            DPIEHaloConfig.cluster_scale("2", 0.0, 0.0, z_lens, v_disp=700.0),
            ShearHaloConfig.free("S1", z_lens),
        ),
        member_populations=(
            MemberPopulationConfig.bergamini2019("potfile_1", "catalogs/hera/members.yaml", z_lens, prior_mag0=19.82),
        ),
        image_constraints=ImageConstraintsConfig(catalog_path="catalogs/hera/images.yaml"),
    ),
    paths=RunPathsConfig(output_dir="results/demo/hera", run_name="python", figure_prefix="hera"),
    sampling=SamplingConfig(warmup=250, samples=100),
).validate()

print(dry_run(config))
result = run(config)                    # stage="init", "final" or "both" (default)

result.posterior.physical               # (draws, parameters) in physical units
result.summary                          # the numbers written to run_summary.json
```

Set the JAX CPU devices before JAX initializes when sampling chains in parallel; `run.xsh` does this with
{func}`lenscluster.cpu_runtime.configure_cpu_runtime`. The library defaults differ from the `run.xsh` settings in
places, notably the likelihood: the library default is `smoothed`, while `run.xsh` uses the GRF likelihood. See
{doc}`configuration` and {doc}`likelihoods`.

## Validate on mocks

```bash
xonsh run_mock.xsh --campaign smoke_check --preset smoke --realizations 1
```

The smoke preset generates one small analytic cluster, fits it and scores the recovery in about ten minutes. See
{doc}`mock-validation` for production campaigns.
