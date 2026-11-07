# Arc light with a fixed lens: sources, colour images, counter-images

Three scripts reconstruct the source of a lensed arc in JWST light with the lens held at a fitted model. They are meant to be rerun on each new lens model; the arc fit is also a test of that lens.

- `fit_arc_source.py` fits a pixelated source (plus image-plane member Sersics) to CANUCS cutouts, once per posterior draw of the lens.
- `make_source_rgb.py` makes the NIRCam colour figure: the observed arc, the model arc and the source.
- `predict_counter_images.py` traces a system's model source through the whole field and finds every image, including ones the catalog lacks.

The data come from `data/JWST_Images/M0416`: the CANUCS DR1 bgsub mosaics, rms maps and `_psf.fits.gz` PSFs. Decompressed `.fits` files are faster to read than `.gz`. Keep rms and PSF files out of the science-image lookup (see `ANCILLARY_PRODUCT_TOKENS` in `src/lenscluster/cutouts.py`).

## Rerunning on a new lens model

Set `R` to the new run directory. It needs `init/` and `final/`. Use `--draws 3` for a quick look and 16 for the error budget.

```bash
P=/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python
R=results/<new run>/m0416_canucs2025/<config>

# 1. One band, one fold pair (K12 a/b, F150W): arc chi2_nu per draw in fits_per_draw.csv
$P scripts/fit_arc_source.py --run $R --system K12 --images a,b --size 3 --draws 3 --supersample 1 --lam 0.03

# 2. Counter-images the model predicts, with colour stamps around them
$P scripts/predict_counter_images.py --run $R --system K12 --out output/counter_images
$P scripts/predict_counter_images.py --run $R --system K82 --out output/counter_images

# 3. Four bands with one shared source shape (colour source)
$P scripts/fit_arc_source.py --run $R --system K82 --images a,b --size 4.5 --draws 3 --supersample 1 --lam 0.03 \
    --band F150W --bands F115W,F150W,F200W,F444W --lam-colour 0,F444W=100
$P scripts/make_source_rgb.py output/arc_fits/K82_colour/F115W output/arc_fits/K82_colour/F150W \
    output/arc_fits/K82_colour/F200W output/arc_fits/K82_colour/F444W --out output/arc_fits/K82_colour/source_rgb.pdf

# 4. A counter-image alone (cleanest galaxy picture so far); note --at=, a negative x is otherwise read as a flag
$P scripts/fit_arc_source.py --run $R --system K82 --at=4.9,-59.3 --arc-radius 0.6 --size 3 --draws 3 --supersample 1 \
    --lam 0 --src-pixel 0.008 --band F150W --bands F115W,F150W,F200W,F444W --lam-colour 0,F444W=100

# 5. Fold pair and counter-image as one source (the consistency test of the lens)
$P scripts/fit_arc_source.py --run $R --system K82 --images a,b --size 4.5 --add "4.9,-59.3,3,0.6" \
    --draws 3 --supersample 1 --lam 0.03 --lam-colour 0,F444W=100 --src-grid mesh --mesh-nodes 4000 \
    --band F150W --bands F115W,F150W,F200W,F444W
# add --affine to let the counter-image's source map shift, stretch and rotate
```

Update the counter-image positions in steps 4 and 5 from the output of step 2, on the galaxy seen in the stamps. K12's candidate is at (−26.0, −27.6) with `--size 2.5 --arc-radius 0.4`.

## What to compare against (model `nov3o_densemass_notfast/.../w500s250t8_gamma4_allblocks`, November 2026)

**Fold pairs in F150W:**

| Arc | Arc χ²_ν | Note |
|---|---|---|
| K12 | 0.7–1.2 | |
| K82 | 0.4 | |
| K2 | 0.5 | |
| K8 | 0.6–0.8 | |
| K1 | 0.9–1.0 | Lens failure at one knot in some draws |
| K28 | — | Not detected |

Fold pairs only image the strip of the source inside the caustic, so the reconstructed source never looks like a whole galaxy.

**Counter-image candidates.** Both are 0.3–0.6" from a red galaxy of matching colour. Their photometric redshifts have not been checked.

| System | Predicted position | μ | Brightness vs. fold images |
|---|---|---|---|
| K12 | (−26.1, −27.3) | 2.5 | 2.6–3.7 mag fainter |
| K82 | (5.5, −59.6) | 2.4 | 2.5 mag fainter |

**Joint fold + counter-image fit, K82, mesh with 4000 nodes:**
- Each region fits alone: fold χ²_ν 1.0–1.3, counter-image 0.4–0.8.
- Jointly with a shift only, the counter-image fails at χ²_ν ≈ 20. The source-plane offset was 0.2" in the image plane, and 0.9" in one draw.
- `--affine` brings the counter-image to χ²_ν 1.3–5, but with determinant 0.38. This is likely a degeneracy: the fold strip does not constrain the squeeze across it.
- So, under this lens, the fold pair and the candidate are not consistent up to a translation.

**What a better lens should show:**
- Joint χ²_ν near 1 with a shift alone, or with an affine correction close to the identity.
- A smaller offset δ, consistent across draws.

## Limits learned the hard way

- **Independent per-band sources** come out as coloured noise. Use `--bands`.
- **Shape λ:** keep it at about 0.03 for fold pairs. Raising it only moves structure into the per-band residuals.
- **F444W** does not reach χ²_ν 1 on these arcs, probably because of the PSF. Pin its residual with `--lam-colour 0,F444W=100`.
- **Fine square grids hide inconsistent images.** At 1.5–3 mas the pixels let separate images be fit by interleaved pixels (the worm texture). The mesh, or `--src-grid aligned` for a single fold, does not allow this.
- **Choosing λ:** when χ²_ν ≤ 1 is out of reach, the scan takes the largest λ within 5% of the best χ²_ν.
- **`--smooth sersic`** on a fold pair fits R ≈ 40 mas. The strip does not constrain a galaxy profile.
