# Scripts

The `scripts/` directory holds data preparation, figure regeneration and analysis tools that work on saved runs. Run
them from the repository root with the project interpreter. Several print usage with `--help`.

## Imaging and truth maps

`scripts/download_imaging.py` keeps every mosaic and truth map in one tree, `data/imaging/<cluster>/<source>/`:

```bash
python scripts/download_imaging.py download [--clusters hera ares m0416 as1063] [--kinds jwst ffsims hst]
python scripts/download_imaging.py zip [--out FILE]      # pack JWST and FF-SIMS files (and catalogs/) into one archive
python scripts/download_imaging.py unpack ARCHIVE
python scripts/download_imaging.py list
```

| kind | content | folder |
|---|---|---|
| `jwst` | CANUCS DR1 NIRCam mosaics of MACS J0416; DAWN/Grizli v7.5 mosaics of Abell S1063 (F115W, F150W, F200W, F444W) | `canucs_dr1`, `dja_v7.5` |
| `hst` | BUFFALO v1.0 mosaics in the seven HFF bands (60 mas by default, `--hst-scale 30` on request) | `buffalo` |
| `ffsims` | FF-SIMS simulated HST images and true $\kappa$ and shear maps at $z = 9$ of Hera and Ares | `ffsims` |

`--local-source DIR` links files already present in a synced copy instead of downloading them. Every file is
validated and gets a `.json` sidecar with its origin and sha256; `data/imaging/manifest.json` lists the tree. The
FF-SIMS site is offline, so its files come from an archive made with `zip` or from `--ffsims-source DIR`.

## Catalog preparation

See {doc}`catalogs` for the converters (`convert_lenstool_catalog.py`, `convert_member_catalog.py`,
`convert_canucs_catalog.py`, `convert_as1063_jwst_catalog.py`, `add_member_shapes.py`,
`group_bergamini_knots.py`). `build_hff_master_catalog.py` builds the HFF inputs, and
`make_ff_sims_truth_magnitudes.py` and `measure_ff_sims_image_magnitudes.py` the FF-SIMS image magnitudes.

## Regenerating figures from a saved run

These scripts read a run directory and redraw one product without rerunning inference.

| script | regenerates |
|---|---|
| `chain_traces.py RUN` | `<prefix>_chain_traces.pdf` and `chain_convergence.csv` |
| `make_gp_illustrations.py RUN` | the GP diagnostics and shared-field maps |
| `regenerate_cluster_mass_figure.py RUN --imaging jwst` | the cluster mass overview (JWST or HST) |
| `regenerate_fold_diagnostics.py RUN` | the fold-pair source and caustic figures |
| `regenerate_magnification_recovery.py RUN` | `magnification_recovery.pdf` (simulated clusters) |
| `regenerate_truth_recovery.py RUN` | the truth maps and dependent figures, with the conditional GRF |
| `regenerate_member_diagnostics.py RUN` | the member and likelihood diagnostic suite |
| `regenerate_scaling_relation.py RUN` | `scaling_relation.pdf` |
| `python -m lenscluster.results_tex RUN/final` | `<prefix>_results.tex` |

Truth maps cached before the conditional GRF was added must be recomputed with `regenerate_truth_recovery.py` before
regenerating the magnification figures that depend on them.

## Checks and analyses

`check_shallow_fold_pairs.py RUN [--floor 0.0005]`
: The depth and depth error of the shallow fold pairs at a saved run's best fit, with planned values of $\ell$ and
  $\sigma_{\mathrm{fl}}$.

`find_missing_images.py`
: Images the M0416 model predicts but the catalog lacks, with their positional spread over posterior draws. See
  `scripts/find_missing_images.md`.

`find_new_families.py`
: A search of the CANUCS DR1 photometry for new multiple-image families: sources of similar color and compatible
  photo-z that trace back to the same source-plane point. See `scripts/find_new_families.md`.

`fit_arc_source.py`, `fit_fold_source_light.py`
: Fit the light of a lensed arc, or a shared source behind a fold, to JWST cutouts with the lens fixed at posterior
  draws, so the lens uncertainty shows as a spread in the source. Each has a matching `.md` note.

`compare_samplers.py`, `bench_threads_per_chain.py`
: Sampler and CPU-thread benchmarks.

## Paper figures

The `make_*` scripts draw the explanatory figures of the paper (dPIE profile, covariance mapping, far-field
illustrations, Jacobian smoothing, lensing geometry). Several need `pdflatex` and `pdftoppm`, and the `.mjs` scripts
need Node.js; outputs go to `output/pdf/<name>/`.
