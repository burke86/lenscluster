# Fast family probabilities

**Written, not executed or numerically validated.** This workflow is separate
from the Laplace/NUTS study. It reads the saved baseline MAP and selected members;
it does not change that run, its checkpoints, or production code.

First validate and calibrate on held-out families:

```bash
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  feasibility/screen_families.py validate
```

Then produce a ranked pair table for the existing constraint catalog:

```bash
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  feasibility/screen_families.py screen
```

The second command uses the existing full-catalog MAP, so its known-image scores
are in-sample. Use **heldout_pairs.csv and the validation report** to judge
performance, rather than accuracy on this screened training catalog.

Outputs are under `feasibility/family_screening/`:

- `REPORT.md`: held-out precision and recall at several probability thresholds.
- `validation.pdf`: probability reliability and precision–recall curves.
- `heldout_pairs.csv`: genuinely held-out predictions with known pair labels.
- `calibration.json`: fitted probability mapping, settings and input signature.
- `screened_pairs.csv`: IDs, probability, geometric overlap, best trial efficiency,
  minimum squared Mahalanobis distance and image separation, sorted by probability.
  Pairs outside the constraint footprint are flagged as spatial extrapolations.
- `metrics.json`, `fold_status.json`: calibration diagnostics, timings, fit quality
  and failures. Each fold also retains its training catalog and fitted parameters.
- `screening_status.json`: screening time, candidate prior and output scope.

## What makes it fast

No NUTS, Hessian, image-root search or third spatial derivative is needed.
Calibration uses five single-start MAP fits, each capped at 800 iterations with
moderate optimizer tolerances. Folds run sequentially to avoid competing JAX
compilations; available CPU threads remain bounded by `--cores` (default at most
eight). Approximate optimization endpoints are retained with convergence flags.

For each lens model, exact deflection and its spatial derivative are evaluated
**once at efficiency f=1**, then scaled to 64 uniformly spaced efficiencies
between the distances corresponding to z=0.6 and 10. This scaling applies to all
single-plane dPIEs and the efficiency-scaled shear in this model. It is not a
far-field approximation. Changes in source position and covariance across trial
redshifts use only array multiplication and addition.

For large catalogs, a conservative source-plane KD-tree filter proposes nearby
pairs. Pair covariances and overlaps are calculated in blocks, without an N-by-N
matrix. The default limit is 500,000 candidates; exceeding it stops explicitly.
Only pairs with minimum squared Mahalanobis distance <=36 and probability >=0.1
are written. Both thresholds are configurable where appropriate. Actual runtime
has not been measured; compilation and execution times are saved when run.

## What the probability means

The raw score is the log of a Gaussian source-position overlap, averaged over
the fixed efficiency grid. The pair covariance is C_i+C_j, with
`C_i = 0.3^2 A_i A_i.T + sigma_def^2 I`. The covariance normalization is included.
This score is **not itself a probability of common family membership**.

A monotonic logistic calibration learns the relationship between score and known
same-family labels. It preserves the real positive/negative pair frequency,
without artificial class balancing. Thus a reported 0.8 means an estimated 80%
same-family frequency **for candidates comparable to the calibration population**,
subject to the measured reliability curve and limited calibration sample.

The five folds use three disjoint sets: lens training, probability calibration,
and testing. Families at the same redshift rounded to four decimal places stay
together, reducing leakage between different knots of one source. Family 0141
and its entire redshift group stay in training. Every held-out fit starts from
the configured model, not the best fit that already saw those families. Held-out
spectroscopic redshifts are used only for grouping, never for pair scores.

The already selected 160-member catalog stays fixed; its selection used all known
image positions but no family labels. Equal-redshift grouping cannot guarantee
that all physically related knots have identical recorded redshifts. Pair rows
also share images, so they are not statistically independent validation cases.
At least three folds must succeed before a final calibration is exported.

The model is a MAP approximation. There is no posterior lens uncertainty,
photometric-redshift likelihood, color/morphology comparison or explicit image
existence test. Calibration partly measures these shortcomings on held-out known
images; it cannot make them disappear for a different population.

Probabilities are **pairwise**. Do not merge connected pairs into families and
call that a joint family probability. Different knots at nearly the same source
position can remain geometrically ambiguous.

## A new candidate catalog

Provide CSV columns `id,ra,dec`, with string IDs and coordinates in degrees.
Other columns are ignored. No image pairs are excluded by separation by default;
optional `--min-separation` is set during validation and then carried into screening.

For a new catalog, `screen --catalog candidates.csv` also requires `--pair-prior`:
the expected fraction of eligible unordered pairs that belong to one family,
**before** the source-plane filter. For N candidates this is the expected number
of true pairs divided by the eligible pair count, not the fraction of images
that are lensed. The code shifts the calibrated odds to that prior. This assumes
only class prevalence changes, which may fail for an arbitrary field catalog.
Do not interpret that correction as external calibration. Filter to plausible
lensed features using independent information and test representative labels.

For example, after choosing a justified prior in your shell:

```bash
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  feasibility/screen_families.py screen --catalog candidates.csv \
  --pair-prior "$PAIR_PRIOR" --overwrite
```

`--overwrite` replaces only screening outputs, not calibration or baseline fits.
For a different validation configuration use a new `--output` directory:

```bash
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  feasibility/screen_families.py --cores 8 \
  --output feasibility/family_screening_v2 validate --maxiter 1200
```

Use `validate --resume` for interrupted, unchanged validation. Input or code
changes reject calibration reuse. `--no-progress` disables bars. These global
options go before `validate` or `screen`. The source checks in
`test_family_probability.py` were written but **not run**.
