# Lenscluster

Bayesian strong-lensing cluster models with JAX and NumPyro. The model is a sum of
cluster-scale halos (dPIE or elliptical NFW), an optional external shear, and a
population of cluster-member dPIE subhalos that follow luminosity scaling relations
with per-member velocity-dispersion and, optionally, cut-radius deviations. The
profiles the solver knows are registered in `lenscluster/profiles.py`. The likelihood is the smoothed local image-plane likelihood: every observed
image is ray-traced to the source plane, the source-plane residual is compared with a
Gaussian whose covariance is the paper's C = sigma_meas^2 A A^T + sigma_def^2 I + sigma_sel^2 R
(the propagated measurement error with sigma_meas^2 = sigma_inst^2 + sigma_extra^2, an isotropic
residual-deflection scatter, and a feature-selection term sigma_sel^2 R with the stretching
weight R = (I + A A^T / lambda_sel^2)^-1, sigma_sel and lambda_sel sampled; `LikelihoodConfig`),
and the conditional expectation of the absolute Jacobian
determinant normalizes the density. Members far from every image are evaluated with the circular
far-field series of the dPIE; members close to an image use the exact profile.

## Running a cluster

```bash
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run.xsh HERA --dry-run
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run.xsh HERA --campaign oct22 --run-label positions
```

`run.xsh` holds the model definitions for `ARES`, `HERA` and `AS1063` and builds a
`LensClusterSolverConfig`. The dry run builds the model, reports the provisional member
classification and the search and sampling sizes, and writes nothing. CPU devices follow
`RuntimeConfig(chains, nuts_chain_method, cpu_threads_per_chain)`; the launcher configures
them before JAX initializes.

Every run has two stages, implemented in `lenscluster.pipeline`:

**Init** (`run_init`, about a minute on Hera) finds the start of the final inference:

1. Build the model state from the typed configuration.
2. Provisional member classification with prior-inflated cut radii at the initial model.
3. The search, in two stages: the large-scale model without the member population (halos,
   shear, independent halos, sources) from prior draws, then, from each carried mode, the members
   with the large-scale parameters frozen, started from prior draws of the member block. The end
   points are clustered into modes and the modes that can seed chains are polished in full
   dimension with L-BFGS. `SearchConfig.halo_method` and `member_method` select scipy L-BFGS-B
   per start (default), vmapped optax L-BFGS, Adam or mean-field SVI; on CPU scipy is the fastest
   by a factor of four or more.
4. Final classification at the best mode and the classification report. The init log ends with a
   one-line verdict: "robust" when several independent starts reach the best basin, "fragile" when
   only one does. The start, the classification and the modes are saved under `init/`.

**Final** (`run_final`) samples from that start:

5. NUTS from the best mode(s), with a dense mass matrix seeded by the Hessian of the global
   block and unit mass for the source and member latent sites.
6. Posterior audit of the far-field approximation (including whether reclassifying at the
   posterior median would change the exact set), then tables and figures under `final/`.

`run.xsh --stage init|final|both` runs one or both. A completed init is reused by any later
final run whose model, likelihood, member and search configuration match; a completed final
stage is reloaded when the whole configuration matches and refused otherwise.

## Outputs

Normal plotting also writes `<prefix>_chain_traces.pdf`, covering the sampler diagnostics,
all global parameters and the twelve other parameters with highest split R-hat.
`chain_convergence.csv` records split R-hat and ESS for every parameter in the plotted
reporting coordinates. Regenerate from saved samples without inference with
`/home/dutra/.conda/envs/lenstronomy/bin/python scripts/chain_traces.py <run-root>`.

For a run root `results/<campaign>/<cluster>/<run-label>`:

- `init/artifacts/`: `config.json` and `config.pkl`, `init_signature.json`, `state.json`,
  `classification.json`, `search.npz` and `modes.json`, `start.json`, `summary.json`;
  `init/tables/classification_report.txt`; `init/logs/init.log`.
- `final/artifacts/`: `config.json` and `config.pkl`, `posterior.npz`, `audit.json`, `summary.json`.
- `final/tables/`: `posterior_summary.csv`, `member_classification.csv`,
  `member_scaling_relation.csv`, `image_fit_quality.csv`, `image_count_recovery.csv`,
  `model_magnification.csv`, `posterior_image_residual_draws.csv`, `run_summary.json`,
  and for simulated clusters `truth_recovery_m2d_aperture_profile.csv`.
- `final/figures/`: `image_residual_histogram.pdf`, `image_recovery.pdf`, `corner_globals.pdf`,
  `scaling_relation.pdf`, `critical_curves.pdf`, `image_catalog_family_cluster_mass.pdf`,
  and for simulated clusters `truth_recovery_residuals.pdf` and
  `truth_recovery_m2d_aperture_ratio.pdf`.
- `final/fits/`: convergence, shear, det A and magnification grids of the posterior median and
  16/84 percentiles on the truth WCS, and the overview convergence map.
- `final/logs/final.log`.

Figure names carry the cluster prefix from `RunPathsConfig.figure_prefix`.

## Mock validation

```bash
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run_mock.xsh --campaign oct22_mock --preset smoke --realizations 2
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run_mock.xsh --campaign oct22_mock --preset production --seed 12340 --realizations 40
```

Each realization generates a single-BCG mock cluster with a Schechter population of
subhalos (`lenscluster.mock_validation.generation`), runs the single stage on it, and
writes `seed_*/recovery/recovery.json` with parameter, magnification, image-residual,
subhalo-mass and aperture-mass comparisons against the truth. `--aggregate-only` (or
`python -m lenscluster.mock_validation.aggregate_plots <campaign dir>`) renders the nine
paper figures under `aggregate_plots/`: parameter recovery, subhalo mass function,
magnification recovery, aperture-mass components of the representative seed, and the
image-residual distribution, posterior calibration, scaling-relation recovery, standardized
parameter residuals, and compact bias and coverage. The standardized residual is
`(median - truth) / s68`, where `s68 = (q84 - q16) / 2`. Binned residual errors
are SEM across mocks. In the compact figure, horizontal intervals show the central
68% and 95% of residuals across mocks; the right panel shows empirical coverage
of the posterior intervals. Source coordinates are omitted from the compact figure.

## Member diagnostics

The normal plotting workflow also writes `member_deviations.pdf` (one multipage PDF),
`member_constraints.pdf`, `whitened_image_residuals.pdf`, `far_field_accuracy.pdf`, and
`<population>_scaling_correlations.pdf`, with the configured figure prefix. Numerical
products and variable definitions are saved in `final/tables/member_diagnostic_*`
and `image_field_diagnostics.csv`. The suite uses existing inputs only.

`ImageDiagnosticsConfig(posterior_diagnostic_draws=32)` controls additional joint
draws for exact field diagnostics; zero uses only the best fit. Inexpensive member
summaries still use the full posterior.

## Package layout

- `config.py`: typed configuration dataclasses and validation.
- `inputs.py`, `lenstool_parser.py`: catalogs and Lenstool-style potential definitions.
- `state.py`, `model.py`: parameter specifications, packed lens state, transforms.
- `evaluator.py`, `farfield.py`, `smoothed_likelihood.py`, `jax_lensing.py`: the likelihood.
- `classification.py`, `audit.py`: exact versus far-field members and the posterior audit.
- `search.py`, `sampling.py`: multi-start MAP and NUTS.
- `pipeline.py`, `artifacts.py`, `outputs.py`: the stage, its files and its products.
- `diagnostics.py`, `exact_images.py`: exact lens-equation solutions and recovery tables.
- `truth_recovery.py`, `plots.py`, `mass_figure.py`, `cutouts.py`, `rgb.py`: figures.
- `mock_validation/`: mock generator, campaign runner and aggregate figures.
- `image_tools/`: photometry helpers for the FF-sims truth magnitudes.

## Data preparation scripts

- `scripts/build_ff_sims_lenscluster_inputs.py`: FF-sims (Ares, Hera) catalogs.
- `scripts/build_hff_master_catalog.py`, `scripts/build_hff_lenscluster_pars.py`: HFF inputs.
- `scripts/make_ff_sims_truth_magnitudes.py`, `scripts/measure_ff_sims_image_magnitudes.py`.
- `scripts/make_dpie_figure.py`, `scripts/make_covariance_figure.py`: paper figures.
- `scripts/make_covariance_mapping.py`: current measurement, deflection and feature-selection covariance figure. Run with the `ggsl`, `hdm`, or `lenstronomy` conda Python interpreter, e.g. `/Users/isaque.dutra/miniforge3/envs/ggsl/bin/python scripts/make_covariance_mapping.py`. Uses only the Python standard library plus `pdflatex` (PGFPlots) and `pdftoppm`; no Node.js dependency. The reviewed layout is in `scripts/templates/covariance_mapping.tex`. Writes the PDF, TeX, PNG preview and checked illustrative parameters to `output/pdf/covariance_candidates/`.
- `scripts/make_covariance_candidates.mjs`: legacy transition and sequence candidates for the older covariance prescription. The current mapping figure is generated separately by `scripts/make_covariance_mapping.py`.
- `scripts/make_lensing_geometry_figures.mjs`: four analytic illustrations of local stretching, forward/inverse circle mappings, arc curvature, and critical-curve directions. Run `node scripts/make_lensing_geometry_figures.mjs`; PDFs, editable TeX, PNG previews, and analytic checks are written to `output/pdf/lensing_geometry/`. Requires Node.js, `pdflatex`, and `pdftoppm`.
- `download_catalogs.py`: catalog and image downloads.

## Tests

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python -m pytest tests
```

`tests/test_pipeline.py` runs both stages on a synthetic cluster.

### Results for manuscript LaTeX

Every final run exports `final/tables/<figure_prefix>_results.tex`; aggregate mock
plotting exports `aggregate_plots/mock_results.tex`. Include a file with
`\input{path/to/hera_results.tex}` and use commands such as `\HeraNImages`,
`\HeraNRecovered`, `\HeraRMS`, `\HeraMedianResidual`, and `\HeraRuntime`.
Residual values are in arcseconds (add units in the manuscript). Runtime includes
upright hours, minutes, and seconds and works in text or math mode. Names contain
only letters: M0416 uses `MJZeroFourOneSix`, AS1063 uses `ASOneZeroSixThree`.

The export also includes family/member counts, recovery percentage, R-hat, ESS,
divergences, configured chain lengths, and sampling runtime. Full-run wall time
is recorded for fresh complete runs; unavailable historical totals are explicitly
marked `not available`. Sampling runtime is separate and is not substituted for
full runtime. Mock counts sum included best-fit realizations; mock RMS and median
residual follow the aggregate figure's median across mock/draw groups. Concurrent
campaign runtime is not inferred by adding individual run durations.

Regenerate these files from saved cluster artifacts without inference or plotting:

```sh
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python -m lenscluster.results_tex path/to/run/final
```

For simulated clusters with convergence and shear truth maps, normal outputs also
write `<prefix>_magnification_recovery.pdf`. Blue diamonds compare truth and the
best-fit exact model at the observed positions and each family's source redshift.
The gray map distribution and orange binned medians use the configured truth-map
source redshift (9 for Hera and Ares) and the coordinatewise median model. The
orange error bars are spatial 16–84 percentiles, not posterior uncertainties.
The field is restricted to the image-position convex hull and true `|mu| >= 1`.
CSV tables and `magnification_recovery_provenance.json` record the inputs, bins,
redshifts, and points outside the displayed limits. No truth map means no plot.

Regenerate just this plot, without inference:

```sh
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/regenerate_magnification_recovery.py path/to/run
```

- scripts/make_farfield_illustrations.py: synthetic member-image geometry, precomputed-matrix diagram, and circular dPIE convergence figures. Run with the ggsl conda Python interpreter; vector PDFs and PNG previews are written to output/pdf/farfield_candidates/.

- scripts/make_jacobian_smoothing_illustrations.py: synthetic Gaussian determinant averaging, smoothed normalization, and conditioning figures. Run in ggsl, hdm or lenstronomy; outputs are in output/pdf/jacobian_smoothing_candidates/.

Posterior corner axes retain log10 coordinates for log-transformed parameters.
The summaries above the histograms report physical values and asymmetric
uncertainties in linear units. Sampling priors and stored draws are unchanged.
