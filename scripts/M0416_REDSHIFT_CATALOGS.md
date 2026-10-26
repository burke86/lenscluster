# M0416 redshift downloads

The image downloader also retrieves the releases in `m0416_redshift_sources.json`.
The source inventory was checked on 2026-10-25. These are public releases found
in the cited archives, not a guarantee that every published redshift is included.
No download or source matching has been performed as part of adding this feature.

Download just the catalogs (the existing images are not opened):

```sh
/home/dutra/.conda/envs/lenstronomy/bin/python scripts/download_jwst_cluster_images.py --clusters M0416 --catalogs-only
```

Add `--dry-run` to print the plan without network access or file creation.
Add `--refresh-catalogs` to fetch the listed releases again. Changed existing
files and their sidecars are retained under each source's `.versions/<sha256>/`.
This refreshes the listed URLs; it does not discover new release versions.
Use `--catalog-sources canucs_dr1 canucs_images_2025 dja_nirspec` for a subset.
Use `--no-redshift-catalogs` for the original image-only behavior.

Default catalog destination:
`/data/sync/GalaxyClusterImaging/RedshiftCatalogs/M0416/`.
With a custom `--output`, catalogs default to its sibling `RedshiftCatalogs/`;
`--catalog-output` overrides this. Existing image locations are unchanged.

## Included sources

| Source ID | Content |
| --- | --- |
| `canucs_dr1` | 2025 JWST+HST photometry, photo-z, available ancillary/NIRSpec spec-z, cluster and flanking fields |
| `canucs_images_2025` | Full candidate-image catalog and secure multiple-image lensing input list |
| `dja_nirspec` | Public NIRSpec v4.4 compilation and column definitions, including M0416; full compilation is 133 MB compressed |
| `buffalo` | Pagul et al. photometry and LePhare photo-z, MAST v2.0 |
| `alcs_charge` | Kokorev et al. 2022 HST+IRAC master catalog and EAZY column definitions |
| `muse_richard` | Richard et al. 2021, NE v1.2 and S v1.1, redshift and emission-line tables |
| `glass` | HST grism redshifts v002, source/master catalogs and ID crosswalk |
| `clash` | HST ACS+IR and IR catalogs with BPZ estimates |
| `caminha_2017` | Full VIMOS/MUSE redshift catalog and multiple-image table, CDS updated 2019 |
| `vanzella_2021` | MUSE Deep Lensed Field spectroscopy |
| `bergamini_2023` | Spectroscopic multiple-image compilation |
| `rihtarsic_2025` | New spectroscopic and photometric image candidates in CDS appendices |
| `hff_deepspace` | Shipley et al. v3.9 cluster and parallel catalogs |
| `astrodeep` | Cluster/parallel coordinates, photo-z/spec-z and diagnostics |
| `clash_molino` | Molino et al. 2017 CLASH photo-z reanalysis |
| `ebeling_2014` | M0416 spectroscopy |

Each source's original URLs, release, interpretation notes and documentation are
kept in the source inventory and download sidecars. The downloader validates FITS
tables, parses CDS fixed-width tables using their ReadMe, parses CSV tables,
checks gzip integrity, and rejects empty or HTML error responses. Native columns
and quality flags are retained. ASCII source catalogs are retained unchanged;
text validation does not certify their scientific content.

All selected files are attempted even if one source fails. Failures are saved in
`download_manifest.json`, and an incomplete download exits nonzero. Existing
files are validated and skipped unless refreshing. No spectra, image cubes or
mass maps are downloaded by the catalog option.

## Interpreting a candidate match

These downloads prepare a comparison; they do not label extra model images as
confirmed. First check sky-coordinate registration and all plausible nearby
counterparts, particularly in blended arcs. Preserve the original source IDs.
For ASTRODEEP, join coordinates and redshifts by ID within the same field, never
by row number.

A photo-z is an estimate, not a spectroscopic confirmation. A lens-model or
family-assigned redshift is also not a direct spectral measurement of each
image. Inspect the original quality flag and spectrum for spectroscopic matches;
DJA includes failed/low-confidence fits as well as secure measurements. The
same spectrum can appear in multiple catalogs, so agreement between compilations
is not necessarily independent evidence. A missing counterpart does not prove
that an image or its redshift is absent.

SAPPHIRES EDR covers a separate parallel field and is not included as evidence
for cluster-core extras. NED/SIMBAD aggregates, private data, and individual-object
papers are not exhaustively harvested. CANUCS BAGPIPES/Dense Basis are omitted
because they inherit redshift information rather than supply independent
measurements. Exact source pages and citations are in the JSON inventory.
