# lenscluster

Strong-lensing mass models of galaxy clusters with a single-stage Bayesian solver: dPIE halos and a scaling-relation
member population, a Gaussian random field of unmodelled mass with a knee convergence spectrum plus an image-plane
Gaussian process as the likelihood, and NUTS for the posterior.

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
```

## Where things are

- `run.xsh <cluster>`: fit a real or simulated cluster (Hera, Ares, AS1063, AS1063_JWST, M0416_BERGAMINI2023, M0416_CANUCS2025).
- `run_mock.xsh --campaign NAME`: generate mock clusters, fit them and score the recovery.
- `catalogs/<cluster>/<source>/`: the image and member catalogs.
- `scripts/download_imaging.py`: the mosaics and truth maps under `data/imaging/`.

The repository README holds the detailed workflow notes until they move here.
