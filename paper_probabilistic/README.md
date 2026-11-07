# Probabilistic image-family manuscript

This is a first draft for author revision, not a submission-ready paper. The core method, equations, and appendices are filled in. Authorship, validation results, and public data/code links remain explicit placeholders. No historical benchmark numbers are presented as verified results.

## Build

From this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. `main.tex` and `refs.bib` are the editable sources. Local copies of the repository's `aastex701.cls` and `aasjournalv7.bst` make the document independent of the neighboring manuscript. A normal TeX installation with these files' dependencies is still required. `latexmk -c` removes intermediate TeX files while retaining the PDF and bibliography output.

## Scientific source and chosen configuration

The reference is Git commit `4814146fe7e22b1c2b39edb48a2c147309d89da8`, available through `burke86/codex/shared-bf-family-partition`. The draft describes the family-level catalog and cached lens likelihood-ratio scores, with `pair_score_mode="none"`. This is a selected configuration, not the engine's historical defaults. No branch was restored or merged, and no scientific code was changed.

Inspect the source without switching branches:

```sh
git show 4814146:src/lenscluster/family_partition.py
git show 4814146:src/lenscluster/source_plane_partition.py
git show 4814146:docs/probabilistic_family_assignment.md
```

The documentation motivates the method, but the code controls the equations where the two differ. The title deliberately avoids claiming fully Bayesian inference or calibrated probabilities. The manuscript follows `../Writing_Guide.md`.

| Manuscript component | Historical implementation |
| --- | --- |
| Catalog integral and independent background | `_gaussian_shared_mean_loglike`, `_gaussian_independent_background_loglike` |
| Centered magnitudes, empirical population parameters, family score | `_color_vector`, `_family_catalog_arrays`, `family_catalog_partition_loglike` |
| Gaussian centroid, optional mixture, raw score | `source_plane_family_loglike` |
| Cached background, redshift grid and grid weights | `precompute_source_plane_cache`, `redshift_grid_from_images`, `_normalized_redshift_weights` |
| Redshift-averaged ratio score | `score_cached_source_plane_partitions` |
| Structural and anchor penalties | `_partition_log_prior`, `create_anchor_labels`, `score_partitions` |
| Normalized weights and pair probabilities | `_normalize_log_weights`, `weighted_pair_probability` |
| Logistic proposals and iterative soft targets | `feature_matrix`, `fit_logistic_map`, `run_family_partition_engine` |
| Lens affinity and source-plane beam proposals | `lens_pair_affinity_from_cache`, `combine_catalog_lens_affinity`, `lens_source_cluster_partitions` |
| Candidate generation and repair | `generate_candidate_partitions`, `_local_partition_perturbations`, `repair_candidate_partitions` |
| Separate lens-realization average | `marginalized_source_plane_score_callback_from_ray_shooters` |

## Interpretation details retained in the draft

- The catalog integral is a proper Gaussian marginal likelihood for each scalar component, conditional on its empirical hyperparameters. The complete weighted catalog score is composite, including correlated centered magnitudes and heuristic morphology terms.
- The lens source position is profiled, not integrated. The normalized Gaussian prefactors in the likelihood-ratio path do not make the fitted-centroid score a Bayes factor.
- Source-plane variance uses positional input values directly. It does not apply a lens Jacobian or import the current checkout's covariance model.
- Redshift grid averaging computes an average of ratios. The catalog redshift score, lens grid weights, and data-derived anchors do not establish independent pieces of a fully generative model.
- Singletons and scalar components with fewer than two valid values contribute zero by convention.
- Structural penalties are an unnormalized modification of the base partition distribution. Temperature scales the entire score, including these penalties and anchors.
- Candidate weights and pair probabilities are normalized only over generated partitions. Thresholding pair probabilities need not define a valid partition; the highest-weight candidate always does.
- The clean Gaussian calculation is shown first. The cache normally passes an image reliability array to the optional mixture, even when its entries equal one. Separate clipping leaves a residual broad coefficient of `1e-6` at that endpoint. The appendix documents this detail rather than calling that numerical path an exact pure Gaussian.
- The separate multiple-ray-shooter callback uses raw scores at a family mean redshift and averages only finite realization scores. It does not implement marginalization of the cached composite likelihood-ratio construction.
- If every partition score is nonfinite, the historical normalizer returns uniform weights. This is a numerical fallback and cannot be interpreted as evidence-based assignment. The manuscript's softmax assumes at least one finite score.

Numerical defaults that may matter for later experiments include structural parameters `(a0, lambda1, n0, sn) = (0.3, 1, 3, 2)`, component weights `(wz, wc, wm) = (1, 1, 0.3)`, color width `0.25 mag`, and morphology width `0.15` in each column's numerical units. They are not claimed to be calibrated. The raw benchmark and engine defaults differ from the selected family-level configuration; do not infer a run configuration from API defaults alone.

## Bibliography verification

Only four references are included, each supporting a specific statement rather than a broad novelty claim:

- Jullo et al. (2007), cluster lens modeling and source-plane consistency: [original paper, Sections 2 and 3](https://arxiv.org/html/0706.0048v2). DOI `10.1088/1367-2630/9/12/447`.
- Budavári & Szalay (2008), shared-source versus independent-source comparison: [original paper, Section 2](https://arxiv.org/html/0707.1611v3). DOI `10.1086/587156`.
- Budavári & Basu (2016), consistent complete assignments: [original paper, Section 2](https://arxiv.org/html/1609.03065v1). DOI `10.3847/0004-6256/152/4/86`. Its two-catalog assignment algorithm is not presented as the lensing algorithm used here.
- Pitman (1995), exchangeable partition distributions: [publisher record](https://link.springer.com/article/10.1007/BF01213386). DOI `10.1007/BF01213386`. The base formula was also checked against the author's [Combinatorial Stochastic Processes, Equation 3.6 with discount zero](https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/pitman_CSP.pdf). The extra family-size and singleton penalties are specific to this implementation. The 1995 article's full PDF could not be fetched in this session; its bibliographic record and indexed abstract were verified.

## Scope of verification

Compilation, PDF inspection, and independent checks of the displayed algebra are document verification, not scientific validation of the family-assignment engine. No engine tests or mock campaign were run. The required Python interpreter `/home/dutra/.conda/envs/lenstronomy/bin/python` is absent on this host; no other Python interpreter or new environment was used.

The delivered PDF has six pages. All pages were rendered and visually inspected. The final build has no unresolved citations/references or overfull/underfull boxes. A harmless AASTeX/nameref label-definition notice remains. The 147-word abstract is below the AAS limit. There are 2,694 prose words including appendices and editorial placeholders, and 32 displayed equations.

Independent JavaScript calculations checked the catalog integral against both direct one-dimensional quadrature and a multivariate Gaussian evaluated by Cholesky factorization on four heterogeneous fixtures. The largest absolute difference in log evidence was `1.25e-14`. A finite-difference check confirmed the fitted source centroid is stationary. Enumeration of all 52 partitions of five images checked the base partition distribution's normalization and weighted pair-probability identities. These algebra checks are not reported as astrophysical recovery or calibration results. `qa_report.json` records their values, document checks, and final source/PDF hashes.
