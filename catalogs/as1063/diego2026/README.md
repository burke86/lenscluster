# AS1063 JWST image and member catalogs

- `images.yaml`: 122 observed image/knot positions, 42 matched families, 21 source systems, in `lenscluster-images/1` format.
- `members.yaml`: all 288 galaxies in Beauchesne's released FITS table, in the current member YAML format, with F160W magnitudes and 46 positive LOSVD measurements.
- `image_mapping.json`: every retained published image label, its system/knot/image mapping, and every excluded row with its reason.
- `raw/`: published image-table data, unchanged member FITS, and the member mosaic WCS header.
- `validation.json`: counts, exact numerical round-trip checks, and source-file SHA-256 hashes.

## Image source and selection

Diego et al., *Hedorah*, [arXiv:2601.11704v2, Appendix A, Table 1](https://arxiv.org/html/2601.11704v2#A1.T1), revised 11 September 2026. RA and Dec are copied in J2000 degrees at the published precision. The archived JSON includes all 276 unique rows in the actual revised HTML table. Its caption still says 263; we did not drop rows to force agreement with that caption.

The fitting catalog includes rank A, except the four positions carrying footnote 3 (39.a/b and 42.a/b), which the paper identifies as model-predicted positions not observed with JWST. It also includes the three images of family 60: the revised table gives spectroscopic z=3.72, although its rank remains B. Thus 119 observed rank-A positions plus three spectroscopically confirmed rank-B positions are retained. All use the published spectroscopic redshift, never the geometric redshift.

Other rank-B candidates are archived, not included in `images.yaml`: their redshifts are lens-model estimates. Flags 4/5 mark positions outside the JWST footprint, and flag 6 marks likely singly imaged arcs. Points along a singly imaged arc must not be treated as repeated images of one source knot. Predicted magnifications, delays and geometric redshifts remain in the raw table only. No measured image photometry, positional uncertainties, or parity is supplied by this table; none was invented. Astrometric errors therefore come from `sigma_inst_arcsec` in the run configuration.

## Systems, knots and image labels

Diego's integer labels enumerate matched families, including separate knots of a galaxy. Our curated same-source groups are:

| System | Diego families, in knot A/B/... order |
| --- | --- |
| D1 | 1, 2, 3, 4 |
| D5 | 5, 6, 7 |
| D9 | 9, 10, 11 |
| D12 | 12, 13, 14, 15, 16, 17 |
| D19 | 19, 20, 21 |
| D25 | 25, 26, 27, 28, 29 |
| D31 | 31, 32, 33, 34 |

These groups are inferred from the shared spectroscopic redshift and corresponding nearby knot positions in each macroimage; they are not explicit galaxy IDs supplied by Table 1. Other retained families are individual systems, named D followed by their published integer. Original labels are retained in each position's notes. In families 32–34 the published image b corresponds spatially to image c of family 31; it is renamed c. No coordinate is changed. The loader creates one fitted source family per system/knot, e.g. D1A, D1B, etc.

## Member source and conventions

Beauchesne et al., [arXiv:2509.07762](https://arxiv.org/html/2509.07762v2), *A comprehensive separation of dark matter and baryonic mass components in galaxy clusters I*. The source is `Cluster_member_catalogue_full.fits`, from the authors' [public release](https://drive.google.com/drive/folders/17nIGG5ke46hnjqF1EW02eS2u9gFaF0Hw). Original download: https://drive.google.com/uc?export=download&id=1hulygtTRou5SrifceQmMGj9FomxyaWMS . The paper also links the [MAST release](https://doi.org/10.17909/t9-w6tj-wp63).

The released table contains **288 rows**, whereas the paper describes 289 light-profile fits. We preserve every released row and do not synthesize the missing entry. This is an HST F160W member catalog used alongside JWST image constraints, not JWST member photometry.

- Member IDs are the original `NUMBER` values.
- Positions are `RA_FITBEST_J2000` and `DEC_FITBEST_J2000` without an astrometric shift.
- F160W magnitudes and errors are `MAG_FITBEST` and `MAG_FITERR`.
- Shapes come from `AFITBEST_IMAGE`, `BFITBEST_IMAGE`, and `THETA_FITBEST`. The released masked mosaic WCS has 0.06 arcsec pixels, west-positive x and north-positive y. Axes are converted to arcsec; pixel angles are transformed locally through that WCS into the solver's west/north basis. Only the axis ratio enters the catalog shape conversion.
- Positive `LOSVD`/`LOSVD_err` pairs are stored as `sigma_e` measurements, consistent with the elliptical half-light apertures described in section 4.2. Zero entries mean no measurement and are omitted. `SIGMA_FITBEST` is a light-profile fit parameter, not the measured stellar velocity dispersion, and is not used as kinematics.
- The FITS supplies no individual galaxy redshifts. No individual redshift is assigned. The lens-plane redshift belongs in the run configuration.

**The separately modelled BCG+ICL is absent from this member table.** This catalog does not provide its replacement mass component. A complete lens model must supply that component separately. No galaxy is arbitrarily relabelled as the BCG, and no older catalog entry is silently mixed into this release.

No relative registration between the Diego image positions and Beauchesne member positions has been fitted here. Coordinates are preserved in their published frames. The converter does not alter run configuration. The separate `AS1063_JWST` option in `run.xsh` uses these catalogs and retains the existing AS1063 cluster-scale and gas halos. It adds the old Bergamini BCG (member 1809) as a separate round dPIE at RA=342.183237, Dec=-44.530870, with fixed position, shape and 0.15 kpc core. Its velocity dispersion and cut radius are free, initially 328.42 km/s and 40.45 kpc, obtained from the existing neutral relation at its F160W magnitude 16.176. The velocity-prior configuration uses mean 328.42 and standard deviation 164.21 km/s, bounded at 82.11–821.05 km/s (converted by the existing positive-parameter machinery to a truncated lognormal); the cut-radius prior is uniform in log on 5.39–269.65 kpc. This is an independent galaxy component, not a reconstruction of Beauchesne’s BCG+ICL MGE. The original `AS1063` option is unchanged. Kinematics are stored in the catalog but disabled in the new run option. JWST figures are the default; `--imaging hst` explicitly overrides the display.

## Rebuild and validation

From the repository root:

```sh
/Users/isaque.dutra/miniforge3/envs/lenscluster/bin/python scripts/convert_as1063_jwst_catalog.py
```

The builder uses archived inputs, requires no network, and invokes both production YAML loaders. It checks unique image labels and positions, at least two observations per fitted family, and exact recovery of input coordinates, spectroscopic redshifts, member magnitudes, and retained LOSVD measurements.
