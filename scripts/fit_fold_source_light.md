# Reconstruct a JWST fold with a fixed lens

`fit_fold_source_light.py` fits one source to both fold images simultaneously. The default is K12 a/b in the M0416 `w500s250t8_gamma4_allblocks` run. K6 is also selectable. No lens parameters are optimized and no lens sampling is run.

The default source is now a **nonnegative pixel grid**, rather than two circular Gaussians. It uses the same bilinear light interpolation as lenstronomy `INTERPOL`, implemented in JAX and tested against lenstronomy. `--source-model knots` retains the original lenstronomy/JAXtronomy Gaussian model.

## Run

From the repository root, a short best-lens fit is:

```bash
/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python \
  scripts/fit_fold_source_light.py \
  results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks \
  --draws 0 --supersample 4 --output /tmp/k12_physical_trial
```

The script automatically finds the matching CANUCS **F150W 20 mas RMS map and empirical PSF** beside the science mosaic. The F150W PSF was downloaded and inspected: 201 × 201 pixels, seven stars, centered peak, unit sum. Its faint negative wings are retained rather than clipped. An explicit `--psf` overrides discovery; `--psf-fwhm` requests an approximate Gaussian instead.

To compare the saved image-GP correction, run the same command with `--gp-displacement` and a different output directory. This is a **fixed positional nuisance correction**, not a gravitational mass deflection. The parametric+GRF run remains the physical lens-only reference. The code inverts

`theta_observed = theta_gravitational + E[d(theta_gravitational) | image catalog]`

before tracing to the source. It checks the inverse residual and positive local warp determinants on the sampled grid. It uses the complete saved covariance, not an independently fitted shift per knot. No new registration parameters are fitted to the pixels.

For the longer comparison:

```bash
/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python \
  scripts/fit_fold_source_light.py \
  results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks \
  --draws 32 --bootstrap 32 --supersample 4 --output /tmp/k12_long_physical

/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python \
  scripts/fit_fold_source_light.py \
  results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks \
  --draws 32 --bootstrap 32 --supersample 4 --gp-displacement \
  --output /tmp/k12_long_gp
```

Use a fresh output directory. Existing fits are not overwritten. `--draw-indices 0 499 999` selects particular additional saved lens draws for checks; the best draw is always included. `--prepare-only` extracts data, errors and the fitting mask without optimization. No new environment is needed.

## Data and fitting region

`--band F115W|F150W|F200W|F444W` selects a local M0416 science mosaic, at 20 mas for the short-wavelength bands or 40 mas for F444W. Matching RMS maps are present for all four. Other bands require their own matching PSF. `--science`, `--science-ext`, `--error`, `--error-ext` and `--psf` permit explicit files/extensions.

RMS maps must contain **one-sigma errors**, with the same WCS, shape and units as the science image. Weight maps are not accepted as RMS maps. An embedded `ERR` extension is used if no matching RMS file is found. Explicit `--noise VALUE` or `--estimate-noise` remain exploratory alternatives; neither is used automatically when an RMS map is absent.

The default rectangular mask includes the observed knots and the blank pixels around them, extending `--radius 0.5` arcsec beyond their coordinate bounds. This prevents the source from creating extra light immediately outside a narrow arc mask. `--mask-shape arc|disks` retains optional restricted masks. A larger padded image supplies the PSF convolution boundary, but is not all included in the likelihood. PDF panels zoom to the fitted region.

`--exclude-mask` excludes nonzero pixels from a full-mosaic mask with matching WCS. Inspect `input.pdf` for unrelated objects. The pixel model fits a local linear background plane, not a foreground galaxy model. Use an externally subtracted image or mask contaminants when necessary. The stretched RGB is never used as fitting data.

Compressed images are decompressed once into `/tmp/fold_source_light_cache`; this avoids repeated slow gzip seeks. The source files remain untouched. RMS cutouts and lens mappings are cached with input fingerprints. Changes to source regularization or the fitting mask do not recompute the lens map. Changing lens draws, geometry, subpixel sampling or the GP option does.

## Source and optimization

The source grid defaults to 96 × 96 pixels. It covers the saved source knots and the backprojections of their observed positions, plus `--source-margin 0.1` arcsec. The outer source-grid boundary is fixed to zero. This finite source field, its resolution and the regularization are assumptions, not independently measured galaxy properties; compare larger margins and finer grids before interpreting faint outskirts.

A quadratic curvature penalty suppresses unsupported pixel-to-pixel fluctuations. `--regularization 0.1` is referenced to a 0.003 arcsec source pixel; its numerical coefficient scales as inverse source-pixel area when the grid changes. `--source-grid` controls resolution. The positive constrained problem is convex at fixed lens and fixed grid, unlike nonlinear source-position fitting. JAX supplies exact gradients to a preconditioned L-BFGS-B solve; `--pixel-maxiter 5000` caps iterations. Optimization success does not imply a scientifically adequate fit.

Subpixel source interpolation is assembled once as a sparse operator and reused through the fit. Tests check its image values and gradients against direct evaluation. Lens deflections are computed without unused Hessians; each new mapping verifies sampled values against the original evaluator to 1e-10. The GRF uses the original fitted interpolation, not the smoother figure-only interpolant.

`--supersample` controls integration within each science pixel. The empirical PSF is an effective native-mosaic kernel applied to the averaged brightness. Both matched fold images always use the same source. Increasing source flexibility cannot guarantee that an inconsistent fixed lens reproduces both images.

## Uncertainty and fit quality

The best saved lens draw is fitted first, followed by evenly spaced retained lens draws. Each gets its own source fit. They are not reweighted by the new pixel likelihood. The lens-draw spread measures sensitivity to that saved ensemble, **not a newly sampled source posterior**. The supplied lens posterior has convergence limitations, so this is not calibrated total uncertainty.

`--bootstrap` makes independent Gaussian-noise realizations around the best fitted image and refits the source, conditional on that lens, PSF, mask and regularization. Successful fits are stored separately; failed fits and iteration counts are recorded. The pixel mode does not report a dense inverse-Hessian covariance. The Gaussian-knot mode retains its local covariance approximation.

No remaining GRF or GP field realizations, drizzle covariance, uncertain PSF, foreground subtraction errors, or alternative lens models are included. The optional GP warp uses its conditional mean only. Lens-spread quantiles require at least three successful additional lens draws and exclude the special best draw. Pixel-source parameters from different native source grids are not combined into misleading per-pixel quantiles.

Use `chi2_per_pixel`, standardized residual maps and knot-scale residuals to assess the reconstruction. A formal number of degrees of freedom is not assigned to the regularized source grid. Most background pixels can fit well while the bright knots remain misplaced; a high fraction of acceptable pixels alone is insufficient.

## Outputs

- `input.fits`, `input.pdf`: native data, adopted RMS and mask, with WCS.
- `fit.pdf`: observed arc, reconstructed image and standardized residuals, using the shared stylesheet.
- `source.pdf`, `best_source.fits`: reconstructed source. For nJy science pixels, the source FITS is in nJy/arcsec²; its integrated flux must use the source pixel area.
- `best_model.fits`: PSF-convolved model on the observed WCS.
- `fits.npz`: lens draw indices, fitted parameters, model images, mask, PSF and successful noise-bootstrap parameters.
- `summary.json`: inputs, per-draw source-grid geometry, convergence, fit quality, GP inverse checks, bootstrap status, timings and limitations. Source flux and half-light radius refer to the finite source grid and depend on its support and regularization.
- `lens_spread.npz`: image-model percentiles for successful additional lens draws, when enough draws exist. Source-parameter percentiles are provided only for Gaussian-knot mode.
- `best_local_covariance.npy`: Gaussian-knot mode only.

Production run artifacts, sampling code and the original fold figures are not changed.

## Four-band source RGB

`scripts/plot_fold_source_rgb.py` combines F115W, F150W, F200W and F444W using the shared JWST RGB recipe. Fit each band independently at the same saved best lens draw with `--draws 0 --source-grid-reference /path/to/F150W/summary.json`. The reference fixes the source grid, and must use the same run, system, images and fixed lens components. Keep physical and fixed-GP branches separate.

The current fits are under `/tmp/k12_source_rgb/{physical,gp}/{F115W,F200W,F444W}`; F150W comes from `/tmp/fold_source_validation/{physical,gp}`. Render them with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lenscluster /Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python scripts/plot_fold_source_rgb.py
```

Outputs are in `/tmp/k12_source_rgb/figures`: two source portraits, a physical-versus-fixed-GP comparison, an observed-fold/source comparison, and a four-band grayscale view. Calibrated source maps and RGB arrays are also retained in FITS, with input provenance in JSON. All bands use the same absolute source coordinates and surface-brightness units, including the different F444W pixel area. The RGB uses one common exposure, with no per-band renormalization or sharpening.

These are model-dependent reconstructions, not direct source photographs. Bright-knot residuals remain. Patchy faint structure can reflect noise, foreground light and reconstruction artifacts; it is not established galaxy structure. The bands have different resolving powers and these displays are not PSF-matched color measurements.

## Checked on the current M0416 run

The short validation used lens draws 25 (saved best), 0, 499 and 999, plus three independent-noise bootstrap refits at the best lens. All eight source fits and all six bootstrap fits converged. Both branches used the same F150W science/RMS/PSF, mask, source resolution and regularization.

| Lens draw | Parametric + GRF χ²/pixel | With fixed GP displacement χ²/pixel |
|---|---:|---:|
| 25 | 2.183 | 1.854 |
| 0 | 2.197 | 2.014 |
| 499 | 2.776 | 1.747 |
| 999 | 4.061 | 3.327 |

Increasing integration from 4×4 to 8×8 subpixels changed a fixed source's rendered image by 0.00105 pixel RMS units on average (99th percentile absolute difference 0.00505). Increasing source resolution to 160×160 and reducing regularization improved the best-lens GP result to χ²/pixel 1.716. The B-knot apertures still had residual RMS 4.30 and 4.40 times the supplied errors; this is **not a pixel-perfect knot reconstruction**. More source pixels alone did not remove the mismatch.

The final cached validation evaluated zero new lens maps. The fitting/output stage took 47.9 s for the physical branch and 45.9 s for the GP branch, excluding saved-run loading and input preparation. Cold-map validation stages took 191.6 s and 165.9 s respectively under concurrent local load. These are measured trials, not a prediction for every posterior draw.

Scratch results are in `/tmp/fold_source_validation/`: `comparison.pdf` contains both best-lens fits, `physical/` and `gp/` preserve separate products, and `validation.json` records the numerical checks. Use the long commands to assess source sensitivity across more saved lenses; do not interpret longer optimization as a guaranteed cure for the knot residuals.

An additional K6 check used a 128×128 source grid, 0.15 arcsec source margin, 0.4 arcsec arc-mask radius and 2×2 integration. It converged to χ²/pixel 9.652 for parametric+GRF and 5.022 with the fixed GP warp. The latter's A-knot aperture residuals were 7.12 and 5.84 RMS. These trials do not make K6 a better starting target than K12; their coarser source spacing (0.0064 arcsec) and more crowded field should be considered before interpreting them as a definitive limit. Products are under `/tmp/fold_source_validation/k6_physical_trial/` and `k6_gp_trial/`.
