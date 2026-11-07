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

`LikelihoodConfig(likelihood_mode="gp_image_plane")` (`run.xsh ... --likelihood gp_image_plane`) selects
a second position likelihood with no source-plane floor (`lenscluster/gp_image_plane.py`). Within a
family the image-plane displacements are correlated, S = sigma_shared^2 (K kron I) + (sigma_inst^2 +
sigma_white^2) I with the Gaussian kernel K_ij = exp(-|theta_i - theta_j|^2 / 2 ell^2), and the stacked
source-plane residuals have covariance C = B S B^T + sigma_fl^2 I with B = blockdiag(A_i). One
Cholesky factor of C per family gives the Gaussian term and the conditional of the source position
that the whitened source offsets use; the determinant factor keeps its per-image form on the
diagonal block of C. The sampled error scales are sigma_shared (log-normal, median 0.15", width 0.5,
truncated at 1", started at 0.168"), sigma_white (half-normal, scale 0.05", started at 0.02") and the
correlation length ell (`images.gp_ell`, log-normal, median 7", width 0.5, truncated at 30", started at
`gp_correlation_length_arcsec`); `gp_correlation_length_prior=None` fixes ell at that value instead.
sigma_fl (1.5 mas) is fixed. sigma_def, sigma_extra, sigma_sel and lambda_sel do not exist in this model.

The init search (both likelihoods) maximizes the posterior density in the physical source coordinates:
the volume factor log|det L_beta| of the source transport, which NUTS needs, is left out of the search
target (`Evaluator.search_log_likelihood`). That factor grows without bound with the magnification and
gave the joint density a spurious mode at a lens so massive that every image was demagnified
(M0416, 256 starts: +393 against -1213 at the physical mode). As a safety net a mode whose median
|det A| at the images exceeds `SearchConfig.max_median_abs_determinant` (10; cluster lenses sit near
0.2) is logged as degenerate and cannot seed chains. The init stage logs the own depth
c Delta^2 / 8 and depth error of every mutual-nearest opposite-parity pair closer than 3" and
writes `init/tables/shallow_fold_pairs.json`; it does not change sigma_fl. The same table for a
saved run, with the planned rho, ell and sigma_fl:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/check_shallow_fold_pairs.py RUN_DIRECTORY --floor 0.0005
```

GP runs automatically write three diagnostics to `final/figures` when plot output is enabled:
`gp_kernel_sky.pdf` (positions and links colored by total displacement correlation),
`gp_shared_vs_white.pdf` (relative-displacement noise and a joint uncertainty ellipse),
and `gp_scale_posteriors.pdf` (shared and white amplitude posteriors in mas).
Filenames use the configured figure prefix. These plots need only the current run;
they contain no fold-sample position panels or old-run comparisons.
Regenerate them from a saved run with:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/make_gp_illustrations.py GP_RUN
```

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

When RGB imaging is configured and available, smoothed fits also write
`small_lambda_covariance_cutouts.pdf`. Its three columns compare RGB cutouts,
linear covariance ellipses, and second-order lens-map contours for the twelve
images with the smallest absolute Jacobian eigenvalues. Measurement circles
show their actual angular size. Exact-ray-tracing checks and selected image
identities are saved in matching CSV/JSON tables; contour vertices are saved
in NPZ format. These are geometric diagnostics, not a modified likelihood.

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

Automatic aggregate refreshes include every completed realization. Convergence failures
remain recorded in `aggregate_plots/campaign_selection.json` and `seed_summary.csv`
and are reported in the terminal. For a filtered comparison, run the aggregate-plots
module with `--converged-only --output-dir <separate directory>`; this requires
max R-hat < 1.01 and min ESS >= 50 and never falls back to unfiltered realizations.

Hera and Ares outputs also write `<prefix>_all_vs_image_footprint_histogram.pdf`.
This compares exact parametric and truth-map projected masses in circular apertures
of radius twice `Re = 8 * 10**(-0.24*(m_F160W-20))` kpc. The normalization is fixed
from the exploratory Hera comparison, not retuned for each fit or for Ares.
All usable members and the subset inside the observed-image convex hull share bins;
the display shows the central 99% of all ratios. Tables and TeX statistics use the
full distribution. The physical posterior-median model includes the conditional GRF
mean when a shared knee-spectrum field is available; otherwise the exported metadata
explicitly identifies parametric-only masses. Incomplete or numerically unresolved
apertures are recorded and excluded.

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
- `inputs.py`: halos, member populations and images from the configuration.
- `image_catalog.py`: the YAML multiple-image catalog (systems, images, optional knots); `member_catalog.py`: member catalogs; `sky.py`: RA/Dec and arcsecond offsets.
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

- `scripts/convert_lenstool_catalog.py`: converts a Lenstool multiple-image catalog to the YAML image catalog.
- `scripts/build_hff_master_catalog.py`: HFF inputs.
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

For M0416 and AS1063, normal post-fit plotting also creates the JWST cluster
mass overview and covariance cutouts using the mosaics under `data/imaging` (scripts/download_imaging.py). These PDFs and
their numerical/provenance files have an `_jwst` suffix; the saved HST view is
retained. Missing imaging is reported and skipped independently. A run already
configured with JWST produces that view once. To regenerate just the overview:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/regenerate_cluster_mass_figure.py RUN_DIRECTORY --imaging jwst
```

Posterior corner axes retain log10 coordinates for log-transformed parameters.
The summaries above the histograms report physical values and asymmetric
uncertainties in linear units. Sampling priors and stored draws are unchanged.

Fold-pair posterior diagnostics are generated automatically alongside the RGB figures
when suitable pairs are identified. The default search selects mutual nearest
same-family images within 5 arcsec, requires opposite exact best-fit parity, and
verifies a regular critical curve between them. These are model-identified candidates,
not catalog morphology classifications. Adjust
`image_diagnostics.fold_pair_max_separation_arcsec` to change the search radius.

`<prefix>_fold_pairs_source_caustics.pdf` (and `_jwst.pdf` when available) contains
three system/image-pair panels per page, with all knots of each system matched in color.
At most 128 deterministically selected joint draws, including the best fit, enter
the source calculation, including draws
with unrecovered images. Each source is registered to its own caustic; the caustic and anchor
roots of all draws of a pair are solved in one batched Newton pass (`NEWTON_ITERATIONS` damped
steps from the tracked critical points, verified at a residual of 1e-8). Image contours
reuse the standard batched local posterior finder once per draw for the fold-pair families only,
with the configured tolerances and no grid or adaptive fallback. They are conditional on local root recovery; failures do not establish global image absence.
Numerical arrays, selection rules, draw indices, and failed solves are saved in
`final/tables/fold_pairs_source_caustics.{npz,json}`. Calculations are shared across
imaging variants. Shared-GRF fold figures use a plotting-only C2 Hermite interpolation
that preserves the tabulated radial covariance values and derives the deflection and
its Jacobian from one potential-consistent kernel. Saved conditioning coefficients and
low-order subtraction are retained. Best-fit curves use a 512 × 512 grid, with per-system
256/512 resolution comparisons stored in the fold metadata. Posterior bands and
caustic-aligned sources use the same smooth plotting field. The likelihood, sampling,
and saved image-finder results retain their original interpolation.
Both automatic output and cached regeneration use `prepare_fold_plot_data` and the
same renderer. The default cutouts show outlined catalog crosses and the gravitational
image-position contours, without GP displacements, GP-adjusted contours, model-image
markers, or uncertainty-level legend entries. Shared-GRF critical curves use cyan-blue
with a matching pointwise 16–84% band; parametric curves and bands use orange.
Source panels use each draw's local GRF caustic anchor and independent axis ranges.
These display choices are recorded in the fold metadata as `plot_presentation`.
Regenerate only this diagnostic with:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/regenerate_fold_diagnostics.py RUN_DIRECTORY
```

JWST cluster overviews use posterior image-position contours (68.3% solid and
95.4% dashed), with tiny catalog crosses colored by best-fit recovery status. Blue
contours describe recovered catalog images; purple contours track known best-fit
extra images. At most 256 deterministic joint draws, including the best fit, are
evaluated with the existing local LM solver,
seeded at catalog positions and those known extras, without global searches.
Catalog matching takes priority, preventing an observed root from also entering
an extra contour. The contours are conditional on successful local recovery;
unknown extra images elsewhere are not exhaustively discovered. Numerical draws
and recovery counts are cached in `posterior_overview_positions.npz` and `.json`.
HST overviews retain their existing markers. Regenerate with
`scripts/regenerate_cluster_mass_figure.py RUN_DIRECTORY --imaging jwst`.

Shared knee-spectrum GRF runs (`gp_blocks="all"`) also write
`<prefix>_total_grf_dimensionless_power.pdf` and
`<prefix>_gp_correction_correlation.pdf` with the standard final figures.
These use the highest-log-probability saved draw, retaining its source positions
and field projection. The spectrum is the homogeneous kernel before projection.
The numerical maps and draw provenance are saved in `final/tables/conditional_field_maps.*`.
Regenerate them with the other GP diagnostics using
`scripts/make_gp_illustrations.py <run_directory>` through the authorized Python interpreter.
System- or family-block runs do not define a single shared map and skip these figures.

For shared knee-spectrum GRF runs, total-model truth maps, magnification recovery,
projected aperture profiles, critical curves and caustics include the conditional
mean GRF at each evaluated posterior draw. Image-position recovery already uses
that same correction. Image-stretch overlays include its Jacobian at the image's
family redshift. Explicit parametric/component comparisons remain labeled as such;
parameter posteriors and likelihood-covariance diagnostics retain their statistical
definitions. These fields are conditional means, not additional random GRF draws.
Truth-map FITS headers record `GRFMEAN` and `GRFTYPE`. Old cached maps must be
recomputed before regenerating dependent magnification figures:

```bash
MPLCONFIGDIR=/tmp/lenscluster-mpl NUMBA_CACHE_DIR=/tmp/lenscluster-numba PYTHONPATH=src \
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/regenerate_truth_recovery.py RUN_DIRECTORY
```

Runs without a shared mass field retain their parametric maps; independent
family/system GPs cannot be interpreted as a single cluster-wide mass correction.

JWST fold critical curves and bands use the 50th and 16th-84th percentiles of
tracked curve positions from the same parametric-plus-conditional-GRF draws.
Segments with fewer than 80% valid draws are omitted, without joining gaps.
The full z=9 map uses the zero contour of the pointwise median determinant,
shading where its 16th-84th percentile interval contains zero, from the same
32 draws in the chain containing the best fit. These are posterior field/curve
summaries, not curves evaluated at a vector of median model parameters.
The RGB mass overview retains physical-median density contours and profiles;
it has no critical-curve uncertainty band. Fold source panels retain their
draw-wise caustic alignment and explicitly marked best source.
