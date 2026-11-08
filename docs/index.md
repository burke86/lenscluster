# lenscluster Documentation

lenscluster builds Bayesian strong-lensing mass models of galaxy clusters with JAX. The lens is a sum of cluster-scale
halos (dPIE or elliptical NFW), an optional external shear and mass sheet, and a population of cluster-member dPIE
subhalos that follow luminosity scaling relations, with per-member deviations from the relation. Multiple-image
positions constrain the model through an image-plane likelihood; the default adds a Gaussian random field of
unmodeled mass with a knee convergence spectrum. A multi-start MAP search finds the posterior mode, and NUTS samples
the posterior from there.

The code runs on real clusters (MACS J0416, Abell S1063) and on the simulated FF-SIMS clusters Hera and Ares, and
ships a mock-validation harness that generates analytic clusters, fits them and scores the recovery.

Start with {doc}`installation` and {doc}`quickstart`. The remaining pages describe each part in more detail.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

catalogs
configuration
likelihoods
pipeline
outputs
mock-validation
scripts
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
