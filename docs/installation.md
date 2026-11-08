# Installation

lenscluster needs Python 3.11 or newer. Install it from the repository root:

```bash
python -m pip install -e .
```

The dependencies in `pyproject.toml` include two forks that are installed from GitHub branches:

- `jaxtronomy` from `dutra/JAXtronomy@feat-pseudojaffe-compact-profiles` (the JAX lens profiles), and
- `lenstronomy` from `dutra/lenstronomy@feat-lenscluster-models` (the exact-image solver and plotting).

## Data

The image and member catalogs of the supported clusters are tracked under `catalogs/` and need no download (see
{doc}`catalogs`). Imaging is not tracked. The JWST, HST and FF-SIMS mosaics, and the FF-SIMS truth maps used for truth
recovery, are fetched into `data/imaging/<cluster>/<source>/` with:

```bash
python scripts/download_imaging.py download                     # JWST mosaics and FF-SIMS files
python scripts/download_imaging.py download --kinds hst          # BUFFALO HST mosaics (large)
python scripts/download_imaging.py unpack lenscluster_imaging.zip # or restore an archive made with `zip`
```

The FF-SIMS site has been offline since 2026, so the Hera and Ares images and truth maps come from an archive made by
`download_imaging.py zip` (or `--ffsims-source DIR`). Fits run without imaging: figures that need a mosaic are
reported as missing and skipped. See {doc}`scripts` for the options.

## Tests

```bash
python -m pytest tests                 # the full suite
python -m pytest tests -m "not slow"   # skip the validation tests that run inference
```

`tests/test_pipeline.py` runs both stages end to end on a small synthetic cluster.

## Building this documentation

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The documentation build mocks JAX and the other heavy dependencies, so it does not need the package installed.
