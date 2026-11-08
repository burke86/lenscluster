# Outputs

A run writes everything under its run root, `<output_dir>/<run_name>/` from
{class}`~lenscluster.config.RunPathsConfig`. Figure names start with `RunPathsConfig.figure_prefix`, for example
`m0416_critical_curves.pdf`.

## Directory layout

`init/artifacts/`
: `config.json` and `config.pkl`, `init_signature.json`, `state.json`, `classification.json`, `search.npz` and
  `modes.json`, `start.json`, `summary.json`.

`init/tables/`
: `classification_report.txt`, `shallow_fold_pairs.json`.

`final/artifacts/`
: `config.json` and `config.pkl`, `posterior.npz`, `audit.json`, `summary.json`.

`final/tables/`
: `posterior_summary.csv`, `member_classification.csv`, `member_scaling_relation.csv`, `image_fit_quality.csv`,
  `image_count_recovery.csv`, `model_magnification.csv`, `posterior_image_residual_draws.csv`,
  `chain_convergence.csv`, `run_summary.json` and `run_summary.txt`, `<prefix>_results.tex`, and for simulated
  clusters `truth_recovery_m2d_aperture_profile.csv`.

`final/figures/`
: The figures listed below.

`final/fits/`
: Convergence, shear, $\det A$ and magnification grids of the posterior median and its 16th and 84th percentiles on
  the truth WCS, and the overview convergence map.

`init/logs/init.log`, `final/logs/final.log`
: The stage logs.

## First checks

`final/tables/run_summary.txt`
: Images recovered out of observed, rms and median image-plane residuals, split R-hat, ESS, divergences and run times.

`<prefix>_chain_traces.pdf`
: Traces of the sampler diagnostics, all global parameters and the twelve other parameters with the highest split
  R-hat. `chain_convergence.csv` holds split R-hat and ESS for every parameter.

`final/artifacts/audit.json`
: Whether the far-field approximation held over the posterior.

## Standard figures

| figure | content |
|---|---|
| `image_residual_histogram.pdf` | image-plane residuals |
| `image_recovery.pdf` | observed and model images per family |
| `corner_globals.pdf` | posterior of the global parameters |
| `scaling_relation.pdf` | member scaling relation and deviations |
| `critical_curves.pdf` | critical curves and caustics |
| `image_catalog_family_cluster_mass.pdf` | cluster mass overview on the imaging, with the image catalog |
| `member_deviations.pdf`, `member_constraints.pdf` | per-member deviations and how well the data constrain them |
| `whitened_image_residuals.pdf`, `far_field_accuracy.pdf` | residuals in units of their covariance; far-field error |
| `<population>_scaling_correlations.pdf` | correlations among the relation parameters |

Corner plot axes keep log10 coordinates for log-sampled parameters; the summaries above the histograms report physical
values with asymmetric uncertainties.

## GP and GRF diagnostics

GP runs write, with plotting on:

- `gp_kernel_sky.pdf`: positions and links colored by total displacement correlation;
- `gp_shared_vs_white.pdf`: relative-displacement noise and a joint uncertainty ellipse;
- `gp_scale_posteriors.pdf`: shared and white amplitude posteriors (mas).

Shared-field GRF runs (`gp_blocks="all"`) also write `total_grf_dimensionless_power.pdf` and
`gp_correction_correlation.pdf` at the highest-probability draw, with the maps in
`final/tables/conditional_field_maps.*`. For these runs the truth maps, magnification recovery, aperture profiles,
critical curves and caustics include the conditional mean GRF at each posterior draw. System- and family-block runs do
not define one cluster-wide field and skip these figures.

## Simulated clusters

With truth maps (`TruthRecoveryConfig`, set for Hera and Ares), the run also writes:

- `truth_recovery_residuals.pdf` and `truth_recovery_m2d_aperture_ratio.pdf`;
- `magnification_recovery.pdf`: truth against model magnification at the observed images and over the image-position
  convex hull at the truth-map source redshift (9). The orange error bars are spatial 16–84 percentiles, not posterior
  uncertainties;
- `all_vs_image_footprint_histogram.pdf`: model against truth projected masses in apertures around each member.

## Imaging figures

When imaging is configured and present under `data/imaging/`:

- `small_lambda_covariance_cutouts.pdf`: RGB cutouts, linear covariance ellipses and second-order lens-map contours for
  the twelve images with the smallest Jacobian eigenvalues;
- `fold_pairs_source_caustics.pdf`: model-identified fold pairs (mutual-nearest opposite-parity images within
  `image_diagnostics.fold_pair_max_separation_arcsec`), with posterior critical-curve bands and source panels;
- for M0416 and AS1063, JWST versions with an `_jwst` suffix, including posterior image-position contours (68.3% solid,
  95.4% dashed).

Missing imaging is reported and skipped.

## Reading results in Python

```python
import json
import numpy as np
import pandas as pd

run = "results/demo/hera"
summary = pd.read_csv(f"{run}/final/tables/posterior_summary.csv")
posterior = np.load(f"{run}/final/artifacts/posterior.npz")
fit = json.load(open(f"{run}/final/tables/run_summary.json"))
```

## Manuscript macros

Every final run writes `final/tables/<prefix>_results.tex`, and mock aggregates write
`aggregate_plots/mock_results.tex`. Include one with `\input{...}` and use commands such as `\HeraNImages`,
`\HeraNRecovered`, `\HeraRMS`, `\HeraMedianResidual` and `\HeraRuntime`. Residuals are in arcseconds; add the units
in the text. Names contain letters only: M0416 uses `MJZeroFourOneSix`, AS1063 uses `ASOneZeroSixThree`.

Regenerate from saved artifacts, without inference or plotting:

```bash
PYTHONPATH=src python -m lenscluster.results_tex path/to/run/final
```

To regenerate individual figures without rerunning inference, see {doc}`scripts`.
