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
| catalogs | complete | 0.08441 |  |
| checks | complete | 134.9 |  |
| optimize | complete | 38.04 |  |
| laplace | complete | 163.7 |  |
| nuts | failed | 3211 | TypeError("_DiagnosticsBase._ess_tail() missing 1 required positional argument: 'prob'") |
| compare | skipped | unavailable | Unsuccessful dependencies: nuts |
| residuals | complete | 178.9 |  |
| benchmark | skipped | unavailable | Unsuccessful dependencies: nuts |
| ratios-optimize | complete | 45.49 |  |
| ratios-nuts | failed | 3448 | TypeError("_DiagnosticsBase._ess_tail() missing 1 required positional argument: 'prob'") |
| fisher | complete | 139.9 |  |
| maps | skipped | unavailable | Unsuccessful dependencies: nuts |
| area | skipped | unavailable | Unsuccessful dependencies: nuts |
| plots | complete | 10.25 |  |
| report | complete | 1.075 |  |

NUTS: converged=unavailable; maximum R-hat=unavailable; minimum bulk ESS=unavailable; divergences=unavailable; seconds/minimum ESS including warmup=unavailable; sampling seconds/median ESS=unavailable.
Timing details: {}

## Exact image recovery
| Model | Recovered | Missed | Extra | RMS arcsec | Median arcsec | Common-subset RMS |
|---|---|---|---|---|---|---|
| map | 233 | 4 | 47 | 0.7093 | 0.403 | 0.7093 |
| posterior_median | 233 | 4 | 50 | 0.7806 | 0.4735 | 0.7806 |
Common subset size: 233

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
| all | 0 | gaussian_fisher | 0.1934 | 0.3952 | valid |
| all | 0 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
| all | 1 | gaussian_fisher | 0.2473 | 2.158 | valid |
| all | 1 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
| 10 | 0 | gaussian_fisher | 0.8035 | 1.321 | valid |
| 10 | 0 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
| 10 | 1 | gaussian_fisher | 0.9014 | 8.15 | valid |
| 10 | 1 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
| 20 | 0 | gaussian_fisher | 0.4829 | 0.9756 | valid |
| 20 | 0 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
| 20 | 1 | gaussian_fisher | 0.5225 | 5.613 | valid |
| 20 | 1 | observed_curvature | unavailable | unavailable | indefinite observed curvature |
Cosmology-normalization correlations: [{"selection": "all", "free_sheet": false, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [0.6811074541588849, -0.7795551544970185], "2.v_disp": [0.7667906003850428, -0.7022339684086867], "members.sigma": [0.802353436732812, -0.7424302346015966], "members.cutkpc": [-0.22314076674239808, 0.3183533933293185]}}, {"selection": "all", "free_sheet": true, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [-0.485553564258927, 0.9247227618708472], "2.v_disp": [-0.4882890487160983, 0.9363873176281664], "members.sigma": [-0.4847770121781438, 0.9358508758937306], "members.cutkpc": [-0.049456964564203405, -0.13475317963736372]}}, {"selection": "10", "free_sheet": false, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [0.6191056411527691, -0.7133922920164713], "2.v_disp": [0.716462891093715, -0.8175532870465243], "members.sigma": [0.8088281223818878, -0.5837710937366042], "members.cutkpc": [0.004050013478438576, -0.07669727919321431]}}, {"selection": "10", "free_sheet": true, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [-0.3835365919995036, 0.9656790697466773], "2.v_disp": [-0.341546210006088, 0.9513282686204054], "members.sigma": [-0.30642815450237393, 0.9502539170489138], "members.cutkpc": [0.03901336290326872, -0.08950169525831703]}}, {"selection": "20", "free_sheet": false, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [0.6625853637578749, -0.7183408640445074], "2.v_disp": [0.8239918856961566, -0.8134267320627735], "members.sigma": [0.7540930101654298, -0.6632406746465563], "members.cutkpc": [0.011518210383228042, -0.028122850722083696]}}, {"selection": "20", "free_sheet": true, "kind": "gaussian_fisher", "normalization_cosmology_correlations": {"1.v_disp": [-0.2942610290626755, 0.9584486832610467], "2.v_disp": [-0.263368483106085, 0.9525435106246549], "members.sigma": [-0.2696942032894356, 0.955270713737403], "members.cutkpc": [-0.009260255723036072, 0.04645277733580041]}}]
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
- `baseline_best_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_hessian.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_map.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_nuts.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_optimization.json`: Configuration, numerical diagnostics, timing or status metadata.
- `baseline_polishing_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_start_0_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_start_1_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `baseline_start_2_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `benchmarks.py`: Unexecuted-at-authoring workflow source or validation check.
- `catalog_audit.json`: Configuration, numerical diagnostics, timing or status metadata.
- `catalogs.py`: Unexecuted-at-authoring workflow source or validation check.
- `checks.json`: Configuration, numerical diagnostics, timing or status metadata.
- `common.py`: Unexecuted-at-authoring workflow source or validation check.
- `cosmography/README.md`: Workflow documentation or generated report.
- `cosmography/fisher.json`: Configuration, numerical diagnostics, timing or status metadata.
- `cosmography/fisher.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `cosmography.py`: Unexecuted-at-authoring workflow source or validation check.
- `distances.py`: Unexecuted-at-authoring workflow source or validation check.
- `failures/nuts.txt`: Exception traceback from an unavailable step.
- `failures/ratios-nuts.txt`: Exception traceback from an unavailable step.
- `fitting.py`: Unexecuted-at-authoring workflow source or validation check.
- `images.csv`: Numerical table with named columns and physical units defined in README.
- `inventory.md`: Workflow documentation or generated report.
- `laplace.json`: Configuration, numerical diagnostics, timing or status metadata.
- `laplace.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `manifest.json`: Fingerprints, configuration, dependency versions and step status/timings.
- `member_selection.csv`: Numerical table with named columns and physical units defined in README.
- `members_selected.cat`: Separate selected-member catalog; original catalog remains untouched.
- `model.py`: Unexecuted-at-authoring workflow source or validation check.
- `model_config.json`: Configuration, numerical diagnostics, timing or status metadata.
- `parameter_definitions.json`: Configuration, numerical diagnostics, timing or status metadata.
- `plot_status.json`: Configuration, numerical diagnostics, timing or status metadata.
- `posterior_corner.pdf`: Diagnostic figure; see filename and plot_status.json.
- `ratios_best_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `ratios_map.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `ratios_nuts.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `ratios_optimization.json`: Configuration, numerical diagnostics, timing or status metadata.
- `ratios_start_0_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `ratios_start_1_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `ratios_start_2_endpoint.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `reporting.py`: Unexecuted-at-authoring workflow source or validation check.
- `residuals.json`: Configuration, numerical diagnostics, timing or status metadata.
- `residuals.npz`: Numerical checkpoint; named arrays loaded without pickle.
- `residuals.pdf`: Diagnostic figure; see filename and plot_status.json.
- `results.npz`: Principal joint posterior, Laplace, ratio, Fisher and area arrays; prefixed keys.
- `run.log`: Console transcript.
- `run.py`: Unexecuted-at-authoring workflow source or validation check.
- `runtime.py`: Unexecuted-at-authoring workflow source or validation check.
- `summary.md`: Workflow documentation or generated report.
- `test_validation.py`: Unexecuted-at-authoring workflow source or validation check.
- `validation.py`: Unexecuted-at-authoring workflow source or validation check.
