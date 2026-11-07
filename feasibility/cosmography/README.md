# Cosmography arrays

**Implementation untested; no arrays have been generated during authoring.**
All products are conditional on the reduced two-halo model and selected members.
Posterior uncertainty always uses actual NUTS draws. Unconverged posteriors remain
labeled and must not be presented as reliable forecasts. Fiducial flat wCDM has
H0=70 km/s/Mpc, Omega_m=0.3, w=-1, lens redshift 0.396.

Coordinates are tangent-plane arcsec relative to RA=64.0381417 deg,
Dec=-24.0674722 deg, x positive west, y north. Physical lens radii are kpc at the
fiducial cosmology. Cosmology forecasts hold this angular lens chart fixed and
vary only the distance efficiency; H0 is fixed. All kernels are exact.
The isolated distance adapter uses 128-node Gauss-Legendre integration of
1/E(z), with runtime Astropy and derivative checks. Baseline fits and exact
image recovery use the same adapter; production distance functions are unchanged.

## Distance ratios

`f_posteriors.npz`: `f` has shape (2000, families), `f_best` (families),
`family_ids` and `redshifts` (families), `reference_index` scalar, `chain_id`
(2000). `model_reported` has shape (2,1000,parameters), with physical sources;
`model_names` labels its columns, including `f.<family>`. `correlation` is the
full parameter correlation matrix, including lens/ratio correlations.
`converged` records the separate free-ratio NUTS status.

Family 0141 is fixed at its fiducial ratio for z=3.221. All other f have independent
uniform (0,1) priors. Their spec-z values only label outputs; they do not constrain
this fit or impose monotonicity. The mass/geometry degeneracy may make this fit
poorly identified. `f_summary.csv` records best, median, 16/84%, standard deviation,
R-hat, bulk ESS, fixed-anchor flags, and prior-boundary fractions. Baseline and
free-ratio posteriors must not be interchanged.

## Curvature forecasts

`fisher.npz` keys begin with `{all,10,20}_{fixed_sheet,free_sheet}_` followed by
`gaussian_fisher` or `observed_curvature`. Each matrix has dimensions (P,P), where
P includes global lens/scatter parameters, physical sources for the selected
families, Omega_m, w, and optionally kappa_infinity. `_names` specifies the exact
order, `_family_ids` the selection. Ten families are nested within twenty;
selection starts at the redshift extremes then maximizes minimum separation in
log(1+z), breaking ties by image count and stable family ID.

The local Gaussian Fisher uses the linearized image mean and image covariance,
including derivatives of both. The observed curvature is minus the Hessian of
the full smoothed physical-source log likelihood, without parameter priors or
source-coordinate transformation terms. It is evaluated at the baseline
optimized model, not reoptimized for each subset or cosmology. It can be
indefinite, especially when the evaluation point is not a maximum for that
expanded model. These are distinct approximations, not interchangeable forecasts.

`_covariance` is an inverse on identified directions only; if a null direction
projects onto cosmology, the forecast status is invalid and its finite
pseudoinverse entries are **not** uncertainties. `_scaled_eigenvalues`,
`_physical_eigenvectors` (directions in physical coordinates), and `_null_mask`
identify degeneracies. Scaling uses square roots of absolute curvature diagonal
entries; eigenvalue rank tolerance is 1e-9 of the largest absolute scaled
eigenvalue. Negative directions invalidate observed-curvature forecasts.
`fisher.json` records status, times, marginalized errors when identified, and
cosmology/normalization correlations. kappa_infinity represents a uniform mass
sheet with convergence f(z)*kappa_infinity; the fixed-sheet forecast sets it to 0.

## Lensing fields

`fields_z{1,2,4,7}.npz` contains `xy_arcsec` (N,2). First `n_grid` points are
0.5 arcsec grid points inside the observed-image convex hull; remaining points
are catalog images in `image_labels` order. `grid_x`, `grid_y`, and flattened
`grid_mask` reconstruct the rectangular grid. `quantity_names` orders five
dimensionless fields: signed mu, kappa, gamma1, gamma2, shear magnitude.

`best` is (N,5), `draws` is (100,N,5), `p16`, `median`, `p84`, `std` and
`finite_count` are (N,5). `draw_indices` refer to chain-major flattened baseline
draws. `determinant` is (101,N), best then posterior. `critical_flag` (N) marks
determinant sign changes across those models. Infinite mu is not clipped;
finite-count and sign flags must accompany summaries near critical curves.
`convergence_valid` reflects baseline NUTS convergence, independently of numeric
finite values. `maps.json` records times.

## Multiply imaged source area

`source_area.npz`: `z` is (20), 0.5 through 10 in steps of 0.5; `best` and
`best_valid` are (20); `draws` and `draw_valid` are (32,20); `draw_indices` (32)
refer to baseline joint draws. Areas are source-plane arcsec squared. Invalid
cells may contain a last-resolution estimate, **not a converged measurement**.
Posterior bands are generated only where all 32 resolution checks pass; the
separate `posterior_converged` flag must also be consulted.

The area counts sources with at least two distinct images inside the observed
image convex hull. It is not the full cluster cross section and does not include
flux, surface-brightness or survey completeness. An image-plane triangular mesh
is mapped through the exact lens. Midpoint curvature and parity changes trigger
adaptive subdivision. Inverse barycentric mapping locates images for source-grid
points; roots within 0.05 arcsec are merged to avoid counting shared triangles
twice. This resolution means arbitrarily close pairs are not distinguished.

Success requires <5% change for two successive image/source refinements, <5%
change when shifting the source-grid origin, and no unresolved curvature flags.
Representative single/multiple roots are refined using the repository exact
image solver; disagreement invalidates the result. This local root check does
not prove completeness. Maximum four resolution levels, three adaptive levels,
1.5 million triangles and 30 million raster visits bound each calculation.
Zero estimated area is not certified by this convergence rule. Failures remain
explicit rather than being replaced with a caustic-polygon proxy.

`area_checkpoint.json` records each draw/redshift calculation for interruption
recovery. `area_status.json` includes refinement histories, exact-root checks,
times, errors and status. Root multiplicity and area tests are written in
`../validation.py`; they have not been executed.
