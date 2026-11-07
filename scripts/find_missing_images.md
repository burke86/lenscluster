# Images the lens model predicts but the catalog lacks (M0416)

The search predicts every image of every catalog family, then checks the JWST data and the published catalogs for the ones our gold catalog does not have. Rerun it on each new lens model. A better model should keep the real candidates and drop the artefacts.

- `find_missing_images.py` does the search and scoring:
  - It traces each family's model source through the field: a 0.3" grid, then Newton refinement. The best posterior draw finds the images; four other draws give their spread.
  - A predicted image is "missing" when there is no catalog image of its family within 1", or of another knot of the same system within 2". Knots of one system within 2.5" are merged into one site.
  - Each site is scored against the CANUCS DR1 photometry on distance, F150W magnitude against the prediction, NIRCam colours against the system's catalog images, and photo-z or spec-z against the system redshift.
- `report_missing_images.py` turns the scores into the outputs:
  - It cross-matches the sites with the published image catalogs: CANUCS full (gold, silver, bronze, Q), Rihtarsic et al. 2025 tables A1–A3, Bergamini et al. 2023, and Caminha et al. 2017 (model-predicted positions excluded).
  - It computes the chance-coincidence probability of each counterpart.
  - It writes the tables and colour stamps of the unpublished candidates that look real.
- `predict_counter_images.py` is the single-system version, with stamps. It is useful for a closer look at one system.
- `match_published_images.py` compares every predicted image of one or more models with the published images our catalog does not use: the CANUCS silver, bronze and Q images, and the Diego et al. 2024 (PEARLS) images with no CANUCS gold image within 0.5". It assigns each published image to one of our systems, by label for CANUCS and through the other images of its Diego source. The verdict per model is "predicted" when a predicted image of that system lies within 1", "near" within 2", and "untestable" when the system is not in our catalog.

Catalogs are read from `RedshiftCatalogs/M0416` under the synced `GalaxyClusterImaging` folder, `/Users/isaque.dutra/sync/` on the Mac or `/data/sync/` on Linux (`SYNC` and `CATALOGS` in `find_missing_images.py`). Where that folder is missing, the CANUCS DR1 photometry and full image catalog come from `data/CANUCS/M0416_Rihtarsic2025`, and the Rihtarsic tables, Bergamini 2023 and Caminha 2017 are skipped. Diego et al. 2024 is read from `data/JWST/M0416_Diego2024_PEARLS`. The model, the CANUCS catalogs, the DR1 photometry and Diego 2024 share the JWST frame to 0.04". Bergamini 2023 and Caminha 2017 are HST-frame, so their positions are shifted by (−0.220″ east, −0.084″ north) before matching. The JWST mosaics are in `data/JWST_Images/M0416` on the Mac; on the Linux machine use `--image-dir /data/sync/GalaxyClusterImaging/JWST_Images/M0416`, which has F115W, F150W, F200W and F444W.

Runs pickled before a config field existed load with that field's default, except the two sigma_white priors, which load as None (the half-normal those runs used). The loader checks the rebuilt parameters against the run's saved state.

## Rerun

```bash
P=/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python
R=results/<run>/m0416_canucs2025/<config>
$P scripts/find_missing_images.py --run $R --draws 5 --out output/missing_images
$P scripts/report_missing_images.py --run $R --sites output/missing_images
```

The search takes about 20 s per family, about 40 min for all 111 families on the Mac. `--families K66,K67` restricts it.

## Outputs, in `--out`

- `predicted_images.csv`: every image the best draw predicts, with its distance to the family's and the system's catalog images.
- `predicted_missing_images.csv`: every missing predicted image per family, with its positions over the draws, μ, parity and predicted F150W magnitude.
- `missing_image_sites.csv` and `.pkl`: the merged sites.
- `missing_image_candidates.csv`: every scored CANUCS source near a site.
- `missing_image_report.csv`: every site, with whether it is published and where.
- `new_image_candidates_table.csv`: the unpublished sites, ranked, with the counterpart's CANUCS ID, RA, Dec, distance, F150W, photo-z with its 16–84% range, spec-z, colour χ² and chance.
- `new_candidates.pdf`: for unpublished grade A and B sites, colour stamps of the prediction (crosses over the draws, circles on the top three sources) and of a catalog image of the system, plus the colours.

## Scoring

**Score.** The score is the sum of four terms; lower is better.
- Position: (d/σ)², with σ = 0.4" added in quadrature to the spread over draws.
- Magnitude: (ΔF150W / 1 mag)².
- Colour: the mean χ² of m_band − m_F150W against the catalog images, with a 0.15 mag floor.
- Redshift: 0 if z_sys is inside the photo-z 16–84% range, 1 if inside 2.5–97.5%, 4 otherwise; a matching spec-z scores 0 and a mismatched one 25.

**Grades.**
- A: a matching spec-z, or a photo-z within 68% with colour χ² < 2 and position term < 4.
- B: looser cuts.
- C: everything else.
- none: no source within the search radius.

**Chance.** The probability that a source with a compatible photo-z and an F150W magnitude within 1 mag lies as close by accident: 1 − exp(−n π d²), with n the density of such sources in the central 140″ box.

Near critical curves the draw spread is large, so a grade can be generous. K34 below is an example; always read the distance and the chance alongside the grade.

## Baseline: model `nov3o_densemass_notfast/.../w500s250t8_gamma4_allblocks`, 2026-11-04

The search found 29 sites without a gold image.

**13 are already published.** Most are CANUCS bronze or Q images, some are in Bergamini 2023 or Rihtarsic table A1: K82.3, K12a/b.3, K45.3, K57.3, K19.3, K17.3, K1a.3, K25.2, K99.3 and K109.1, plus two K77 sites that are K33 images.

**Likely new images** (no spec-z for any of them):

| System | z | Predicted (x, y) | μ | CANUCS | RA, Dec | Distance | F150W (predicted) | Photo-z | Colour χ² | Chance |
|---|---|---|---|---|---|---|---|---|---|---|
| K66 | 5.973 | (55.36, −27.92) | 2.0 | 3110159 | 64.021367, −24.075205 | 0.19" | 29.5 (28.0) | 6.14 [6.00, 6.44] | 0.8 | 0.4% |
| K67 | 5.998 | (19.51, 16.59) | 2.8 | 3112391 | 64.032191, −24.062871 | 0.30" | 29.2 (29.4) | 5.95 [5.76, 6.22] | 2.5 | 0.5% |
| K73 | 6.149 | (−37.09, −17.67) | 2.6 | 3110655 | 64.049462, −24.072258 | 0.55" | 28.9 (29.7) | 6.10 [6.00, 6.28] | 1.1 | 1.4% |

**Possible:**
- K20: photo-z 2.24 [2.23, 2.28], but colour χ² 8.7 and 1.8 mag brighter than predicted.
- K28: z 3.08 is outside the photo-z 68% range, and the source is 2.1 mag brighter than predicted.
- K47: photo-z uninformative; K64a.3 is 0.5" away.

**Probably not real:**
- K34 (two sites): a μ of −125 and +74 pair next to a critical curve, with the nearest source 2" away and 40% chance. Likely a lens-model artefact.
- K106: the spot holds K33 images.
- K3: the nearest source has photo-z 6.7.
- K115 (two sites): photo-z 0.3–0.6.
- C1: spec-z 0.402.
- K36, K37 and K40: all predict an image under the bright galaxy 3101644 (photo-z 0.48); this can't be tested without subtracting that galaxy.

**Correction to earlier arc work.** CANUCS 3100689 at (4.9, −59.3), which I first took as K82's counter-image, has spec-z 2.303. It is an image of K22. K82.3 is the bronze source 3121053.

## What to compare against a new model

- Do K66, K67 and K73 stay predicted, within 0.5" of their sources?
- Does the K34 pair go away?
- Does the number of artefact sites (grade C, or chance above 10%) fall?

## Not done yet

- Spectra were not inspected; CANUCS `Z_SPEC` only keeps secure literature spec-z within 0.2".
- The galaxy at about (16, 16) hiding K36, K37 and K40 needs to be modelled and subtracted.
