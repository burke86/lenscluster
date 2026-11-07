# MACS J0416 feasibility report

Scripts were written without execution or numerical validation. The statuses below describe this invocation, not the authoring session.
This is a two-halo feasibility experiment, not a reproduction of the published four-halo Bergamini model. Published 0.43 arcsec RMS is a contextual reference, not a matched-model comparison.
Members: brightest 30% union within 10 arcsec; exact dPIEs; fixed alpha=0.30, beta=0.60 and pivot=17.02, zero individual deviations. Sources have a fixed 100 arcsec physical Gaussian prior and likelihood-whitened coordinates. Astrometry=0.3 arcsec; source scatter is sampled.
Physical parameter medians are computed jointly by columns, then evaluated as one model. They need not be a posterior mode. All inference times are synchronized; explicit kernel compilation is separate, sampler compilation is labeled where mixed.

## Catalogs and selection
Images=237; families=88; members=213; retained=160; luminosity fraction=0.9606.
Redshift range=[0.94, 6.629]; magnitude range=[17.0166, 27.608]. Image columns: ['id', 'RA', 'Dec', 'a', 'b', 'theta', 'z', 'mag']. Member columns: ['id', 'RA', 'Dec', 'a', 'b', 'theta', 'mag', 'lum', 'color'].
Catalog flags: ['Image a,b,theta are shape fields, not positional-error estimates; assumed 0.3 arcsec.', 'Missing optional member color: 213 rows', 'Member catalog lum is not used; relative luminosity is calculated from magnitude.']

## Timings and status
| Step | Status | Wall s | Reason |
|---|---|---|---|
| catalogs | complete | 0.07365 |  |
| checks | complete | 93.3 |  |
| optimize | failed | 26.98 | ConcretizationTypeError('Abstract tracer value encountered where concrete value is expected: traced array with shape int32[]\nThe problem arose with the `int` function. If trying to convert the data type of a value, try using `x.astype(int)` or `jnp.array(x, int)` instead.\nThe error occurred while tracing the function _report at /home/dutra/sdev/burkelenscluster/feasibility/model.py:188 for jit. This value became a tracer due to JAX operations on these lines:\n\n  operation b:f64[] = jit[\n  name=clip\n  jaxpr={ lambda ; a:f64[] c:f64[] d:f64[]. let\n      e:f64[] = max c a\n      b:f64[] = min d e\n    in (b,) }\n] a 2.2250738585072014e-308:f64[] 0.9999999999999998:f64[]\n    from line /home/dutra/sdev/burkelenscluster/feasibility/model.py:134:26 (Model.theta)\n\n  operation a:f64[] = mul 30.0:f64[] b\n    from line /home/dutra/sdev/burkelenscluster/feasibility/model.py:134:26 (Model.theta)\n\n  operation a:f64[] = add -15.0:f64[] b\n    from line /home/dutra/sdev/burkelenscluster/feasibility/model.py:134:26 (Model.theta)\n\n  operation b:f64[] = jit[\n  name=clip\n  jaxpr={ lambda ; a:f64[] c:f64[] d:f64[]. let\n      e:f64[] = max c a\n      b:f64[] = min d e\n    in (b,) }\n] a 2.2250738585072014e-308:f64[] 0.9999999999999998:f64[]\n    from line /home/dutra/sdev/burkelenscluster/feasibility/model.py:134:26 (Model.theta)\n\n  operation a:f64[] = mul 30.0:f64[] b\n    from line /home/dutra/sdev/burkelenscluster/feasibility/model.py:134:26 (Model.theta)\n\n(Additional originating lines are not shown.)\n\nSee https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError') |
| laplace | skipped | unavailable | Unsuccessful dependencies: optimize |
| nuts | skipped | unavailable | Unsuccessful dependencies: optimize |
| compare | skipped | unavailable | Unsuccessful dependencies: laplace, nuts |
| residuals | skipped | unavailable | Unsuccessful dependencies: optimize |
| benchmark | skipped | unavailable | Unsuccessful dependencies: nuts |
| ratios-optimize | skipped | unavailable | Unsuccessful dependencies: optimize |
| ratios-nuts | skipped | unavailable | Unsuccessful dependencies: ratios-optimize |
| fisher | skipped | unavailable | Unsuccessful dependencies: optimize |
| maps | skipped | unavailable | Unsuccessful dependencies: nuts |
| area | skipped | unavailable | Unsuccessful dependencies: nuts |
| plots | complete | 0.003883 |  |
| report | complete | 0.003231 |  |

NUTS: converged=unavailable; maximum R-hat=unavailable; minimum bulk ESS=unavailable; divergences=unavailable; seconds/minimum ESS including warmup=unavailable; sampling seconds/median ESS=unavailable.
Timing details: {}

## Exact image recovery
| Model | Recovered | Missed | Extra | RMS arcsec | Median arcsec | Common-subset RMS |
|---|---|---|---|---|---|---|
Common subset size: unavailable

## Parameter uncertainties
Rows contain MAP; NUTS median [16%,84%]; NUTS standard deviation; Laplace standard deviation; Laplace/NUTS width ratio; R-hat; bulk/tail ESS. Source x and y share a row, in that order. Units: arcsec for source/center coordinates, kpc for radii, km/s for velocities, dimensionless for shape/shear. Laplace widths use valid transformed Gaussian draws; invalid fraction is reported below. Flags: U=Laplace width <0.7 NUTS; G=|skew|>0.5 or excess kurtosis>1.
No NUTS parameter table is available.

Laplace leakage outside the empirical 95% NUTS Mahalanobis ellipsoid (unconstrained coordinates): unavailable. This is not a non-Gaussian HPD region. Invalid Laplace fraction=unavailable. Per-parameter marginal 95% leakage is saved in parameter_comparison.csv.
Full correlation matrices are in correlations.npz and the two correlation CSVs; the PDF displays global parameters only.

## Performance projections
10,000 positions x 100 draws x 8 redshifts: synchronized field time=unavailable s; compile=unavailable s; pair throughput=unavailable comparisons/s.
| Objects | Unordered pairs | Field s | Exhaustive pairs s | Total s |
|---|---|---|---|---|
300-image refit projection: {}
Fields scale linearly in objects, exhaustive pair screening quadratically. Screening projections repeat every pair for 100 draws and eight trial redshifts; they exclude candidate-list output and catalog-specific filtering.

## Cosmography
Free-ratio fit fixes family 0141 at the fiducial ratio at z=3.221. Other ratios have independent Uniform(0,1) priors; spectroscopic redshifts are labels only. Geometry forecasts vary f(z), hold the angular mass chart and H0 fixed, and contain no cosmological priors.
Gaussian Fisher includes local image mean and covariance derivatives. Observed curvature is the Hessian of the full physical-source smoothed log likelihood, without source-coordinate Jacobians or priors. All source nuisances are marginalized. Null cosmology directions produce unavailable uncertainties.
| Families | Sheet | Definition | sigma Om | sigma w | Status |
|---|---|---|---|---|---|
Cosmology-normalization correlations: []
Free-ratio NUTS status: {}

Family ratios (reference has no sampling uncertainty):
Free-ratio posterior unavailable.

## Recommendation
NUTS has not passed the recorded convergence criteria, so this run cannot establish that Laplace uncertainties are adequate. Improve identification and convergence before using either uncertainty estimate for decisions. Optimization-only refits can still be timed, but they do not validate uncertainty.

## Output inventory
- `README.md`: Workflow documentation or generated report.
- `REPORT.md`: Workflow documentation or generated report.
- `__init__.py`: Unexecuted-at-authoring workflow source or validation check.
- `area.py`: Unexecuted-at-authoring workflow source or validation check.
- `baseline_optimization.json`: Configuration, numerical diagnostics, timing or status metadata.
- `benchmarks.py`: Unexecuted-at-authoring workflow source or validation check.
- `catalog_audit.json`: Configuration, numerical diagnostics, timing or status metadata.
- `catalogs.py`: Unexecuted-at-authoring workflow source or validation check.
- `checks.json`: Configuration, numerical diagnostics, timing or status metadata.
- `common.py`: Unexecuted-at-authoring workflow source or validation check.
- `cosmography/README.md`: Workflow documentation or generated report.
- `cosmography.py`: Unexecuted-at-authoring workflow source or validation check.
- `distances.py`: Unexecuted-at-authoring workflow source or validation check.
- `failures/optimize.txt`: Exception traceback from an unavailable step.
- `fitting.py`: Unexecuted-at-authoring workflow source or validation check.
- `images.csv`: Numerical table with named columns and physical units defined in README.
- `inventory.md`: Workflow documentation or generated report.
- `manifest.json`: Fingerprints, configuration, dependency versions and step status/timings.
- `member_selection.csv`: Numerical table with named columns and physical units defined in README.
- `members_selected.cat`: Separate selected-member catalog; original catalog remains untouched.
- `model.py`: Unexecuted-at-authoring workflow source or validation check.
- `model_config.json`: Configuration, numerical diagnostics, timing or status metadata.
- `parameter_definitions.json`: Configuration, numerical diagnostics, timing or status metadata.
- `plot_status.json`: Configuration, numerical diagnostics, timing or status metadata.
- `reporting.py`: Unexecuted-at-authoring workflow source or validation check.
- `results.npz`: Principal joint posterior, Laplace, ratio, Fisher and area arrays; prefixed keys.
- `run.log`: Console transcript.
- `run.py`: Unexecuted-at-authoring workflow source or validation check.
- `runtime.py`: Unexecuted-at-authoring workflow source or validation check.
- `summary.md`: Workflow documentation or generated report.
- `test_validation.py`: Unexecuted-at-authoring workflow source or validation check.
- `validation.py`: Unexecuted-at-authoring workflow source or validation check.
