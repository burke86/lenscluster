# Getting started

## Environment

The code runs in the existing `lenstronomy` conda environment (Linux) or `lenscluster` (Mac). Use that interpreter
directly, for example `/home/dutra/.conda/envs/lenstronomy/bin/python -m pytest`; do not create a new environment.

## A first fit

```bash
xonsh run.xsh M0416_CANUCS2025 --dry-run     # print the full configuration, no fitting
xonsh run.xsh M0416_CANUCS2025 --fast        # exploratory settings: system GP blocks, short chains
xonsh run.xsh M0416_CANUCS2025               # the full run
```

Outputs go to `results/<campaign>/<cluster_key>/<run>/{init,final}/` with `artifacts`, `figures`, `tables` and
`fits` under `final/`.

## Mock validation

```bash
xonsh run_mock.xsh --campaign smoke_check --preset smoke --realizations 1
xonsh run_mock.xsh --campaign my_campaign --preset production --realizations 8
```

The smoke preset exercises the pipeline in about ten minutes per realization; the production preset uses the solver
settings of a full `run.xsh` run.

## Data

Catalogs are tracked under `catalogs/`. Imaging is not: `scripts/download_imaging.py download` fetches the CANUCS DR1,
DJA v7.5 and (on request) BUFFALO mosaics into `data/imaging/<cluster>/<source>/`, and `unpack` restores the archive
that `zip` produces, which also carries the FF-SIMS files of Hera and Ares.
