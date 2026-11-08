# Mock validation

`run_mock.xsh` generates analytic mock clusters, fits each one with the same pipeline as a real cluster, and scores the
recovery against the known truth. A **campaign** is one generator configuration, one solver template and a number of
**realizations** (seeds).

```bash
xonsh run_mock.xsh --campaign smoke_check --preset smoke --realizations 1
xonsh run_mock.xsh --campaign nov_mock --preset production --seed 12340 --realizations 40
```

## The mock clusters

Each realization (`lenscluster.mock_validation`) draws:

- a cluster-scale dPIE halo and a BCG dPIE, from a total mass of $5\times10^{14}$ to $2\times10^{15}\,M_\odot$ split
  by a BCG fraction, with radii from mass–size relations with scatter;
- a Schechter population of member dPIE subhalos on a luminosity scaling relation;
- background sources, some of them knotted galaxies (`--knotted-families`), lensed exactly, with astrometric noise
  added to the images.

The lens redshift is drawn between 0.3 and 0.5. `MockConfig` in `mock_validation/config.py` holds every range.

## Presets and modes

`--preset`
: `smoke`: a few members and families, 8 search starts, 100 + 100 draws at tree depth 6. For tests of the pipeline
  itself; about ten minutes per realization. `production`: the solver settings of a full `run.xsh` run (256 starts,
  1000 + 250 draws).

`--mode`
: `production` (default): the priors an observer would use, the same as the real clusters (halo centered on the BCG
  light, Bergamini et al. 2019 member relation). `idealized`: priors centered on each realization's truth, which
  isolates the sampler and the likelihood from prior misspecification.

`--likelihood`, `--gp-blocks`
: As in `run.xsh` (see {doc}`likelihoods`).

`--fast`
: The exploratory settings of `run.xsh --fast`.

`--fisher-only`
: Generate (or reuse) the mocks and forecast how well the GP hyperparameters are constrained from a Fisher matrix at the
  truth, without running the solver. Writes `fisher.json` per mock and `fisher_precheck.csv`.

`--dry-run` prints the configuration; `--rescore` recomputes the recovery of finished realizations.

## Outputs

A campaign writes to `validation_runs/<campaign>/<run_name>/`, where the run name records the mode, preset and
likelihood (for example `analytic_production_smoke`). Each `seed_*/` holds a full run and
`seed_*/recovery/recovery.json`, which compares parameters, magnification, image residuals, subhalo masses and aperture
masses with the truth.

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

```bash
xonsh run_mock.xsh --campaign nov_mock --preset production --aggregate-only
python -m lenscluster.mock_validation.aggregate_plots validation_runs/nov_mock/analytic_production_production
```

Aggregates include every completed realization. Convergence failures are recorded in
`aggregate_plots/campaign_selection.json` and `seed_summary.csv` and reported in the terminal. For a filtered
comparison, write to a separate directory:

```bash
python -m lenscluster.mock_validation.aggregate_plots CAMPAIGN_DIR --converged-only --output-dir CAMPAIGN_DIR/converged
```

This keeps realizations with max R-hat < 1.01 and min ESS ≥ 50 and never falls back to unfiltered ones.
