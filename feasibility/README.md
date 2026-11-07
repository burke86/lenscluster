# MACS J0416 feasibility workflow

For fast candidate-family probabilities and held-out-family validation, use the
separate workflow in [FAMILY_SCREENING.md](FAMILY_SCREENING.md). It reuses the
baseline MAP, without running Laplace or NUTS again.

**UNTESTED. Written without executing scripts, tests, inference, benchmarks, or
figure generation. No numerical results are claimed.** Production modules,
catalogs, and the manuscript are unchanged. Runtime checks can expose adapter
incompatibilities or numerical failures; these are recorded rather than hidden.

From the repository root, launch with the existing interpreter:

```bash
MPLCONFIGDIR=/tmp/lenscluster-mpl \
NUMBA_CACHE_DIR=/tmp/lenscluster-numba \
PYTHONPATH=src \
/home/dutra/.conda/envs/lenstronomy/bin/python feasibility/run.py --all
```

Resume an interrupted or partly failed invocation:

```bash
MPLCONFIGDIR=/tmp/lenscluster-mpl \
NUMBA_CACHE_DIR=/tmp/lenscluster-numba \
PYTHONPATH=src \
/home/dutra/.conda/envs/lenstronomy/bin/python feasibility/run.py --all --resume
```

After a code change, use `--all --restart` instead of `--resume`. This archives
existing generated products under `feasibility/archives/<timestamp>-<id>/`,
preserves scripts and documentation, and starts with fresh fingerprints.
Nothing is archived merely by editing these scripts. Archives are excluded
from current reports and checkpoint inventories.

Named steps use `--steps catalogs checks`, or `--steps optimize laplace nuts
compare residuals plots --resume`. Dependencies are added automatically.
Available steps are `catalogs`, `checks`, `optimize`, `laplace`, `nuts`, `compare`,
`residuals`, `benchmark`, `ratios-optimize`, `ratios-nuts`, `fisher`, `maps`,
`area`, `plots`, and `report`. Reporting always runs. For core feasibility before
the more expensive cosmography, explicitly select the baseline steps above.
`--all` includes **a second NUTS fit and 660 area calculations**. It is not a
bounded smoke test. Failures do not terminate independent steps.

No environment installation or creation is performed. Existing dependencies
include JAX/JAXtronomy, NumPyro, SciPy, Astropy, ArviZ, Pandas, Matplotlib and
`corner`, plus `tqdm` for progress bars. A missing plotting dependency is reported
without invalidating fits. Float64 is enabled before numerical imports.

## Progress and CPU resources

Progress bars cover workflow steps, each optimizer's iterations, NUTS,
Laplace validity/physical transformations, R-hat/ESS, exact recovery, benchmarks,
curvature forecasts, maps and source-area jobs. Compilation can leave a bar at
zero while JAX prepares the first calculation. Use `--no-progress` for batch logs.
The additional bars go to the original stderr rather than duplicating every
redraw into `run.log`; durable statuses remain in the manifest and JSON files.

The default budget is `min(8, available CPUs)`, respecting Linux CPU affinity.
Override it with `--cores 16`; choose the independent area-process/start limit
with `--workers 4`. Workers default to `min(4, cores)`. For example:

```bash
MPLCONFIGDIR=/tmp/lenscluster-mpl NUMBA_CACHE_DIR=/tmp/lenscluster-numba \
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  feasibility/run.py --all --cores 16 --workers 4
```

The entry point sets affinity, BLAS thread limits and up to two JAX CPU devices
before importing NumPy/JAX. Three independent optimization starts share one
compiled executable in a bounded thread pool. NUTS uses two parallel chains
when devices permit; otherwise it runs sequentially. XLA may use threads within
the CPU budget, but an eight-core budget does not guarantee eight-core speedup.

Independent area calculations use spawned processes, never a fork of an active
JAX runtime. Each worker has one JAX device, single-threaded BLAS, a disjoint CPU
subset, and a reusable model/compilation cache. Only the parent writes checkpoints.
Area meshes can be large: reduce `--workers` if memory is limited. Worker death
fails that step while preserving completed jobs for resume. Other scientific
steps remain ordered to avoid memory contention. The benchmark deliberately
has no competing workflow jobs and records the resource configuration.

Resource choices are recorded in `manifest.json` on every launch. They can change
on resume without changing the scientific configuration; completed timings are
not remeasured and retain their original resource history. Edits to workflow
source code still invalidate checkpoints under the existing fingerprint policy.
All concurrency and progress changes are **untested and unbenchmarked**.

## Model and interpretation

The input files are `data/Bergamini/M0416_Bergamini22/obs_arcs.cat` and
`CM_cat_MACSJ0416.cat`. IDs remain strings. Previously inspected counts of
237 images, 88 families, 213 members, and 160 selected members are expectations,
not results of this implementation. `catalog_audit.json` records actual counts,
column names, malformed entries and warnings at runtime. The brightest
`ceil(0.30*N)` members, with stable-ID tie breaking, are combined with all
members within 10 arcsec of a constraint. Relative luminosity is
`10**(-0.4*(magnitude-minimum_magnitude))`; selected luminosity is divided by
the full catalog sum. Shape axes are not positional uncertainties.

This simplified model contains two cluster dPIEs near the two BCGs, the selected
members, and external shear. It does not reproduce the published four-halo
model, fixed gas, foreground structure, or independent-galaxy treatment.
Member alpha=0.30, beta=0.60, pivot magnitude=17.02, core normalization=0.15 kpc,
and individual deviations=0 are fixed. All member fields use exact kernels.
Normalization priors are uniform in physical velocity (50–500 km/s) and
truncation radius (1–50 arcsec, converted to kpc at the fiducial cosmology).
Halo center/shape/core priors and fixed 2000 arcsec cut radii are explicit in
`model.py`; the production shape parameter chart is retained. Resolved sampled
parameters and priors are written to `parameter_definitions.json`.

The smoothed likelihood uses `C = 0.3**2 A A.T + sigma_def**2 I`, including its
determinant smoothing factor and physical Gaussian source prior, standard
deviation 100 arcsec. Source coordinates are likelihood-whitened and the
transformation correction subtracts the standard-normal base density exactly
once. `log(sigma_def)` has mean `log(0.02)`, standard deviation 0.7, with the
repository's positive floor and upper cutoff 1 arcsec. No extra measurement,
selection or member scatter is fitted. All image parities are unknown.

The adapter changes shear normalization to a common reference efficiency at
family 0141 (z=3.221), scaling it consistently at every image/redshift. The
forecast sheet is parameterized by kappa at efficiency one. Its deflection and
Jacobian both scale with `D_ds/D_s`. `distances.py` uses differentiable 128-node
Gauss-Legendre integration, checked against Astropy at runtime, instead of the
production distance helper's interpolated table. The local state, arbitrary-ray
evaluations and image finder use these same distances and angular conversion.
The production code is not patched.

## Fits and comparisons

Three deterministic starts use L-BFGS, at most 2000 iterations each. The first
is the configured initial model; two draw global parameters from their priors,
with zero whitened-source offsets and scatter at its prior median. Invalid
coupled-shape/shear draws are rejected. The separate free-ratio fit starts at
the baseline mode and fiducial ratios, with two additional deterministic starts.
Each completed start is saved atomically as `<fit>_start_<k>_endpoint.npz` before
physical reporting. The best raw vector is saved as `<fit>_best_endpoint.npz`.
If its maximum gradient exceeds 0.001, L-BFGS polishing disables relative
objective-reduction stopping and targets a maximum gradient of 0.0005, using
only the iterations remaining from that start's 2000-iteration allowance.
The polishing candidate and acceptance status are saved separately. A candidate
is retained only when it lowers the objective, or improves the gradient with
an objective increase within 1e-12 times max(1, |objective|). Failure to meet the
stationarity criterion remains explicit; it does not relax Laplace validation.

Laplace and NUTS target the same joint density in unconstrained coordinates,
including source nuisances and coordinate Jacobians. The full autodiff Hessian
must be numerically positive definite, and the maximum potential gradient at
the endpoint must be <=0.001. The Hessian is saved even on failure. Eigenvalues
are never clipped to manufacture finite uncertainties. When valid, 10,000
Gaussian draws are transformed to physical parameters and physical sources;
draws outside coupled physical guards are flagged, with their fraction reported.

NUTS uses two chains, 500 warmup and 1000 retained samples per chain, target
acceptance 0.90, maximum depth 8, dense adaptation, and a valid Hessian seed when
available. Convergence status requires every rank R-hat <1.01, bulk ESS >=100,
and zero divergences. Tail ESS and depth saturation are reported separately.
These short-chain criteria are diagnostics, not proof that all modes are found.

The reported parameter tables include physical sources, joint-draw intervals,
standard deviations, rank R-hat, bulk/tail ESS and Laplace/NUTS standard-deviation
ratios. Non-Gaussianity flags use |skewness|>0.5 or excess kurtosis>1. These are
descriptive thresholds. The empirical 95% Mahalanobis ellipsoid uses the full
unconstrained NUTS covariance and its empirical 95th-percentile squared distance;
it is **not** a non-Gaussian HPD region. Marginal 95% interval leakage is also
saved. No inverse is constructed for a singular empirical covariance.

The exact global image finder evaluates the mode and a physical componentwise
posterior-median model. Reports include recovered, missed and extra images,
RMS, median and p90 residuals, and RMS on the common recovered subset. An invalid
median model is reported as unavailable. A good RMS with missed images is not
equivalent to fitting the full catalog. The published 0.43 arcsec is contextual.

## Timings, checkpoints, and failures

Explicit JIT lowering/compilation and synchronized kernel execution are timed
separately. NUTS warmup/sampling wall times are synchronized; JAX monitoring
records backend compilation durations. Non-backend-compilation time still
includes tracing and dispatch. Reported full step times also include setup and
I/O. Field throughput measures 10,000 positions, 100 deterministic joint draws,
eight redshifts, padded batches of 1024, with synchronization and checksums.
Pair throughput measures blocked source-proximity comparisons with `C_i+C_j`,
without an N-by-N allocation. The 5.991 threshold is descriptive, not calibrated
pair-identification probability. It does not include photo-z or morphology.

Field projections scale linearly with catalog size; exhaustive unordered pair
projections scale as N(N-1)/2, repeated for every draw and trial redshift. They
exclude candidate-list storage. The 300-image refit is a labeled linear proxy
with fixed family count, member count, optimizer iterations and leapfrog counts,
not a timed 300-image fit.

`manifest.json` fingerprints input catalogs, configuration, feasibility and
production Python sources, stylesheet, Python and dependency versions. Resume
rejects incompatible inputs and changed completed products. Failed steps are
retried; dependent steps are skipped with reasons. Residuals and reports refresh
on resume to incorporate newly available posterior products. Area calculations
checkpoint each draw/redshift pair. To start an incompatible experiment, use
`--restart` to archive the previous run, or use a separate checkout. Do not edit or remove the fingerprint
to force reuse. A nonzero exit status means one or more steps failed or skipped.

## Validation and outputs

The `checks` step tests selection identities, distances against Astropy,
distance derivatives, likelihood and gradient agreement at zero shear, physical
source transformation density, Hessian directional derivatives, shear/dPIE
efficiency scaling, spatial derivatives, distinct roots and analytic fold area.
The shared likelihood comparison intentionally excludes shear derivatives,
whose redshift scaling is changed by this adapter; separate field checks cover
those. Failure blocks inference.
`check_compiled_reporting` additionally checks JIT, vmap, source reconstruction,
and reporting derivatives for both baseline and free-ratio models before fitting.
The same checks are exposed in
`test_validation.py` for an optional user-run pytest invocation after `catalogs`.
Neither path was run during authoring.

`REPORT.md` is generated as a self-contained, approximately 400-line report,
with paired source-coordinate rows to limit length. `summary.md` is shorter.
`results.npz` packages the main arrays using `product__array` keys; it never
substitutes Laplace samples for a failed NUTS run. CSVs retain full tables,
including both complete correlation matrices. PDFs use the repository stylesheet,
no titles or grids. `inventory.md` lists files; `failures/` contains tracebacks.
Cosmography arrays, units, limitations and convergence flags are documented in
`cosmography/README.md`.
