# Quickstart

A fit is described by one {class}`~lenscluster.config.LensClusterSolverConfig` and run in two stages: **init** (a
multi-start MAP search for the posterior mode) and **final** (NUTS from that mode, then tables and figures). There are
two ways to build the configuration: from a Lenstool parameter file, or directly in Python.

## Set up the CPU chains

NUTS runs one chain per JAX CPU device. Set the devices at the top of your script, before anything else starts JAX
(reading a `.par` file does, to convert arcseconds to kpc):

```python
from lenscluster.cpu_runtime import configure_cpu_runtime

configure_cpu_runtime(4, "parallel")    # 4 chains, matching RuntimeConfig.chains (default 4)
```

## Option 1: from a Lenstool `.par` file

If you already have a Lenstool model, read its halos, priors, member population and cosmology with
{func}`~lenscluster.lenstool_par.config_from_lenstool_par`:

```python
from lenscluster.lenstool_par import config_from_lenstool_par

config = config_from_lenstool_par(
    "catalogs/as1063/bergamini2019/source/Bergamini_RXCJ2248_lenstool_Normal.par",
    image_catalog="catalogs/as1063/bergamini2019/images.yaml",
    member_catalog="catalogs/as1063/bergamini2019/members.cat",
    output_dir="results/demo",
).validate()
```

The solver reads multiple images from a YAML catalog, so convert a Lenstool `multfile` once with
`scripts/convert_lenstool_catalog.py` (see {doc}`catalogs`). See {ref}`lenstool-par` for what the reader translates.

## Option 2: in Python

Build the lens model from the configuration classes:

```python
from lenscluster.config import (
    CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
    MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, ShearHaloConfig,
)

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
    paths=RunPathsConfig(output_dir="results/demo", run_name="hera", figure_prefix="hera"),
).validate()
```

`cluster_scale`, `free` and `bergamini2019` are shortcuts with standard priors; see {doc}`configuration` for every
option.

## Run the fit

```python
from lenscluster.pipeline import dry_run, run

print(dry_run(config))                  # model size, member classification, search and sampling sizes
result = run(config)                    # init + final

result.posterior.physical               # (draws, parameters) in physical units
result.summary                          # the numbers written to run_summary.json
```

`dry_run` builds the model without fitting and writes nothing; use it to check a configuration before spending
compute. For a quick look, shorten the search and the chains first:

```python
from dataclasses import replace
from lenscluster.config import SamplingConfig, SearchConfig

quick = replace(config, search=SearchConfig(n_starts=32), sampling=SamplingConfig(warmup=150, samples=100))
```

## Where the results go

Each run writes to `<output_dir>/<run_name>/`:

```text
init/
  artifacts/   start, modes, member classification, configuration
  tables/      classification_report.txt
  logs/        init.log
final/
  artifacts/   posterior.npz, audit.json, summary.json
  tables/      posterior_summary.csv, image_fit_quality.csv, run_summary.txt, ...
  figures/     <prefix>_critical_curves.pdf, <prefix>_corner_globals.pdf, ...
  fits/        convergence, shear, det A and magnification maps
  logs/        final.log
```

Three things to check first:

1. The last line of `init/logs/init.log`: **robust** when several independent starts reach the best basin, **fragile**
   when only one does.
2. `final/tables/run_summary.txt`: images recovered out of observed, the rms image-plane residual, and the sampler
   diagnostics (split R-hat, ESS, divergences).
3. `final/figures/<prefix>_chain_traces.pdf`: the traces of the global parameters and of the worst-mixing ones.

See {doc}`outputs` for every file.

## Rerunning and resuming

Calling `run(config)` again reloads a completed init and a completed final stage instead of recomputing them. A
changed configuration under the same run name is refused, so give each changed model a new `run_name`.
`run(config, stage="init")` runs the search alone; a later `run(config, stage="final")` samples from it.

## Next steps

- The library default likelihood is `smoothed`. The production fits use the Gaussian random field likelihood; see
  {doc}`likelihoods` for its settings.
- To test the method on clusters with known truth, see {doc}`mock-validation`.
