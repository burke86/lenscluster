# Mock validation

{mod}`lenscluster.mock_validation` generates analytic mock clusters, fits each one with the same pipeline as a real
cluster, and scores the recovery against the known truth. A **campaign** is one generator configuration
({class}`~lenscluster.mock_validation.MockConfig`), one solver template (a `LensClusterSolverConfig` whose settings
every realization uses), and a number of **realizations** (seeds).

## A smoke test

```python
from lenscluster.cpu_runtime import configure_cpu_runtime

configure_cpu_runtime(4, "parallel")

from lenscluster.config import LensClusterSolverConfig, MemberConfig, SamplingConfig, ScalingModelConfig, SearchConfig
from lenscluster.mock_validation import (
    MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig, run_mock_validation,
)

solver = LensClusterSolverConfig(
    search=SearchConfig(n_starts=8),
    sampling=SamplingConfig(warmup=100, samples=100, max_tree_depth=6, target_accept=0.8, far_field_offsets="fixed"),
    scaling=ScalingModelConfig(fit_corekpc=False, member_deviation_tail="normal"),
    members=MemberConfig(centering="auto", classification_prior_quantile=None),
)
campaign = MockValidationConfig(
    mock=MockConfig(mode="production", n_members=10, n_primary_families=4, n_member_families=1),
    paths=MockValidationPathsConfig(output_dir="validation_runs", campaign_name="smoke_check", run_name="analytic_smoke"),
    runtime=MockValidationRuntimeConfig(realizations=1, seed=12345),
    solver=solver,
)
run_mock_validation(campaign)
```

A few members and families with short chains test the pipeline itself in about ten minutes per realization. For a
validation campaign, use the production solver settings (see {doc}`configuration` and {doc}`likelihoods`), about 150
members and 15 primary families, and tens of realizations.

The solver template has no lens model: each realization's model is built from its own truth. Its paths are replaced
too, so every realization writes under the campaign directory.

## The mock clusters

Each realization draws:

- a cluster-scale dPIE halo and a BCG dPIE, from a total mass of $5\times10^{14}$ to $2\times10^{15}\,M_\odot$ split
  by a BCG fraction, with radii from mass–size relations with scatter;
- a population of `n_members` member dPIE subhalos on a luminosity scaling relation;
- `n_primary_families` background sources (`n_knotted_families` of them galaxies with knots) and
  `n_member_families` galaxy–galaxy families, lensed exactly, with astrometric noise added to the images.

The lens redshift is drawn between 0.3 and 0.5. Every range is a field of `MockConfig`.

`MockConfig.mode` sets how the solver's priors relate to the truth:

`"production"` (default)
: The priors an observer would use, the same as for the real clusters: the halo centered on the BCG light and the
  member population on the Bergamini et al. 2019 relation.

`"idealized"`
: Priors centered on each realization's truth, which isolates the sampler and the likelihood from prior
  misspecification.

## Fisher forecast

{func}`~lenscluster.mock_validation.fisher_precheck` generates (or reuses) the mocks and forecasts how well the GP
hyperparameters are constrained, from a Fisher matrix at the truth, without running the solver. It writes `fisher.json`
per mock and `fisher_precheck.csv`.

## Outputs

A campaign writes to `<output_dir>/<campaign_name>/<run_name>/`. Each `seed_*/` holds a full run and
`seed_*/recovery/recovery.json`, which compares parameters, magnification, image residuals, subhalo masses and aperture
masses with the truth. {func}`~lenscluster.mock_validation.rescore_mock_validation` recomputes the recovery of
finished realizations.

`aggregate_plots/` holds the campaign figures:

- parameter recovery;
- subhalo mass function;
- magnification recovery;
- aperture-mass components of the representative seed;
- the image-residual distribution;
- posterior calibration;
- scaling-relation recovery;
- standardized parameter residuals;
- compact bias and coverage.

The standardized residual is $(\mathrm{median} - \mathrm{truth}) / s_{68}$ with $s_{68} = (q_{84} - q_{16})/2$. Binned
residual errors are standard errors of the mean across mocks. In the compact figure, horizontal intervals show the
central 68% and 95% of residuals across mocks, and the right panel shows the empirical coverage of the posterior
intervals.

## Re-aggregating

```python
from lenscluster.mock_validation import write_aggregate_plots

root = "validation_runs/smoke_check/analytic_smoke"
write_aggregate_plots(root)
write_aggregate_plots(root, f"{root}/converged", converged_only=True)
```

or from the shell, `python -m lenscluster.mock_validation.aggregate_plots ROOT [--converged-only --output-dir DIR]`.

Aggregates include every completed realization. Convergence failures are recorded in
`aggregate_plots/campaign_selection.json` and `seed_summary.csv`. `converged_only=True` keeps realizations with max
R-hat < 1.01 and min ESS ≥ 50 and never falls back to unfiltered ones; write it to a separate directory.
