# Diego et al. M0416 catalogues

Downloaded from the official Harvard VizieR mirror on 2026-11-04. The raw exports retain the catalogue metadata and column descriptions. These are VizieR exports, not byte-for-byte copies of the original CDS fixed-width `.dat` files.

## PEARLS catalogue

Diego et al. (2024), *JWST's PEARLS: 119 multiply imaged galaxies behind MACS0416, lensing properties of caustic crossing galaxies, and the relation between halo mass and number of globular clusters at z=0.4*, A&A 690, A114.

- Paper: https://doi.org/10.1051/0004-6361/202349119
- Catalogue: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/690/A114
- Dataset DOI: https://doi.org/10.26093/cds/vizier.36900114
- Documentation: https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/690/A114?format=html&tex=true

Table B.1 contains **343 unique image IDs belonging to 119 sources**. The CDS release has one scientific table; it is a multiple-image catalogue, not a cluster-member or general eight-band photometry catalogue.

`images.ecsv` and `images.fits` are converted copies of `raw/pearls_tableb1.vot`. Raw text and TSV exports are also retained under `raw/`.

Columns preserve the published identifiers, J2000 RA/Dec (degrees), spectroscopic redshift (`zsp`), geometric redshift and error (`zgeom`, `e_zgeom`), photometric redshift (`zph`), model magnification (`mu`), model time delay in years (`dTime`), and rank (`Rank`). `recno` is a VizieR row number, not an image identifier.

Geometric redshifts, magnifications and time delays are model predictions, not independent measurements. Spectroscopic redshifts are populated in 77 rows and geometric redshifts in 119 rows. Missing entries remain missing; no redshifts have been propagated to other images or substituted. Rank counts are A: 212, B: 84, C: 41, D: 6. No rank cuts were applied.

## Earlier related catalogue

`earlier_BUFFALO_Flashlights/` contains the separate **214-row** Table B.1 from Diego et al. (2024), *BUFFALO/Flashlights: Constraints on the abundance of lensed supergiant stars in the Spock galaxy at redshift 1*, A&A 681, A124. It is retained for reference and is **not** another PEARLS release or an additional set to concatenate with PEARLS.

- Paper: https://doi.org/10.1051/0004-6361/202346761
- Catalogue: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/681/A124
- Dataset DOI: https://doi.org/10.26093/cds/vizier.36810124

## Validation and use

`provenance.json` records download URLs, raw-file SHA-256 checksums, row counts, and validation. Both catalogues parsed successfully with Astropy. Converted FITS/ECSV values and missing-value masks were checked against the raw VOTables. Explicit mask serialization preserves numeric missing values and empty-string notes separately.

```python
from astropy.table import Table
images = Table.read('data/JWST/M0416_Diego2024_PEARLS/images.ecsv')
# For FITS, retain empty-string notes rather than converting them to masks:
images = Table.read('data/JWST/M0416_Diego2024_PEARLS/images.fits', mask_invalid=False)
```

No conversion to the solver's systems/families/knots YAML format has been made. Existing CANUCS inputs and `run.xsh` were not changed.
