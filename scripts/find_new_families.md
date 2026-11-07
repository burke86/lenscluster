# New multiple-image families from the CANUCS photometry (M0416)

`find_new_families.py` looks for groups of CANUCS DR1 sources with similar colours and compatible photo-z whose positions trace back to the same point of the source plane, over several lens draws. It shares the catalogue path (`CATALOGS`) and photometry loader with `find_missing_images.py`. The method and every option are described in the script's docstring.

## Rerun

```bash
P=/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python
R=results/<run>/m0416_canucs2025/<config>
$P scripts/find_new_families.py --run $R --draws 10 --out output/new_families
$P scripts/find_new_families.py --run $R --draws 10 --out output/new_families --null
```

The second command is the false-positive test (positions mirrored x → −x). Always compare the two: only the excess over the null means anything.

One run with 10 draws takes about 50 s on the Mac. Of that, 23 s loads the model and 18 s builds the forward-check grid (cached in `lens_cache/`). The pair search takes 0.3 s per draw, because the k-d tree returns about 970 candidate pairs out of 955,653.

Outputs: `family_candidates.csv` (and `family_candidates_null.csv`), one row per group. Each row has:
- the kind of group: known, extends, published, or new;
- the number of images, predicted images and matched images, and the number of predicted images bright enough to see but missing;
- the fraction of draws in which the group passes;
- the model z and the median photo-z;
- the source-plane separation over its tolerance, and the colour coherence;
- the CANUCS IDs, positions and magnitudes;
- any published labels.

## Why it is fast

- The deflection at any z is (1 − t) α(z_a) + t α(z_b), and this is exact for our model. Each source therefore moves on a straight line in the source plane as its redshift varies, and the closest approach of a pair is closed form.
- The k-d tree uses the source-plane position plus scaled colours, at about 80 values of t, padded so no approach is missed.
- The cheap cuts (t overlap, colour χ², flux ratio) run before the geometry.
- One cached base map per draw, and one grid for the forward check.

## Baseline: nov3o model, 2026-11-04, defaults (F150W < 28, σ 0.25", colour χ² < 2, 10 draws)

- 1,383 candidate sources. 41 of 53 catalogue families with 2+ images among them are recovered.
- Pairs passing in at least half the draws: 271 real against 132 in the null.
- Groups:

  | | Real | Null |
  |---|---|---|
  | All groups | 111 | 90 |
  | Known families | 28 | – |
  | Published elsewhere | 13 | – |
  | Extending a known family | 11 | – |
  | New | 59 | 90 |
  | New and fully reproduced | 10 | 12 |
  | …of which 3+ images | 2 | 0 |

  So new pairs are not above chance; only groups of three or more images are.
- Validation (exhaustive v1 search with stricter cuts: F150W < 27.5, σ 0.25", colour χ² < 1.5): it rediscovers the CANUCS silver/bronze families K107, C14, K103, C4, K88, C12 and C5. For C4, which has a spec-z, the model z (1.86) matches it.
- New 3+ groups:
  - **3109351, 3119786 and 3119875**, z 1.41, photo-z 1.36, at (8.07, −40.63), (14.76, −34.10) and (37.01, −19.11); F150W 27.5–27.9; passes in all 10 draws; not in any catalogue checked. The best new family candidate.
  - **3111997, 3117975, 3117976 and 3117919**, z 1.70, photo-z 1.71. The first three are pieces of one deblended object at about (−25, 9.3), so this is effectively a pair with 3117919 at (−29.0, 3.4).
- Unpublished pairs from the exhaustive v1 run (the null says pairs are no better than chance, so they need other evidence): 3112613 + 3117888, 3108545 + 3117240, 3111625 + 3126834, 3108553 + 3121062.

## Known limits

- Deblended pieces of one arc count as separate images, because the only merge rule is a 0.6" minimum separation. Merge members within about 1" into one image before counting.
- The forward check uses the best draw's grid only.
- Photo-z enters as the 2.5–97.5% range, not as the full probability distribution.
