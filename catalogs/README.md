# Catalogs

The image and member catalogs the pipeline reads, one folder per cluster and per published source:

```
catalogs/<cluster>/<source>/
    images.yaml          multiple images (format lenscluster-images/1, src/lenscluster/image_catalog.py)
    members.yaml|.cat    cluster members (src/lenscluster/member_catalog.py)
    README.*             the source's own notes
    source/              the published files the YAML was converted from (Lenstool catalogs, par files, MAST tables)
    history/             earlier versions of the YAML kept for the record (before the parity edit, before the light shapes)
```

| folder | run.xsh entry | images | members | from |
|---|---|---|---|---|
| `hera/` | HERA | images.yaml | members.yaml | FF-SIMS Hera: `source/hera_obs_arcs.cat`, `hera_cluster_members_potfile.cat`, `hera_lenscluster.par`, the 7-band truth magnitudes |
| `ares/` | ARES | images.yaml | members.yaml | FF-SIMS Ares, the same files |
| `m0416/canucs2025/` | M0416_CANUCS2025 (default) | images.yaml (knots, parities) | members.yaml (DR1 light shapes, Granata 2026 kinematics) | Rihtarsic et al. 2025 MAST release: `source/hlsp_canucs_*lenstool-*` and `allmultim`; `image_f200w_photometry.*` is our F200W photometry of the images |
| `m0416/bergamini2023/` | M0416_BERGAMINI2023 | images.yaml | members.cat | Bergamini et al. 2023 (2022 release): `source/obs_arcs.cat`, the par files; members.cat was `CM_cat_MACSJ0416.cat` |
| `m0416/diego2024/` | (comparison only) | images.ecsv / .fits | | Diego et al. 2024 PEARLS Table B.1, `raw/` and the earlier BUFFALO/Flashlights table |
| `as1063/bergamini2019/` | AS1063 | images.yaml | members.cat | Bergamini et al. 2019 RXC J2248: `source/obs_arcs.dat`, the par files |
| `as1063/diego2026/` | AS1063_JWST | images.yaml | members.yaml (Beauchesne 2025) | Diego et al. 2026 Table 1 (`raw/`), `image_mapping.json`, `validation.json` |
| `as1063/caminha2016/` | (not an entry) | | | Caminha et al. 2016: `source/A1063_obs_arcs_v11.dat`, `members_95_160_nonan_cuts.cat`, par, region files |

Hera and Ares positions are offsets about (0, 0) in degrees (the par files' `reference 3 0 0`), not sky coordinates.
Positions of the real clusters are ICRS RA/Dec; the CANUCS catalog is in the frame of the CANUCS lenstool input,
20 mas from the DR1 photometry, and the Bergamini HST catalogs are 0.2" from the JWST frame
(.agents/m0416-catalog-astrometric-frames.md).

Not here, because of size: `bayes.dat` of the Lenstool runs, the FF-SIMS truth maps and simulated images
(`data/ff_sims/published/`), the Caminha lensing maps, and the CANUCS DR1 photometry catalog
(`data/CANUCS/M0416_Rihtarsic2025/canucs_dr1/`, also under `/data/sync/GalaxyClusterImaging/RedshiftCatalogs/M0416/`
on Linux). Imaging stays under `data/JWST_Images`, `data/BUFFALO_Images` and `data/ff_sims`.

Converters: `scripts/convert_lenstool_catalog.py` (Lenstool arcs to images.yaml), `scripts/convert_member_catalog.py`
(potfile to members.yaml, with kinematics), `scripts/convert_canucs_catalog.py`, `scripts/convert_as1063_jwst_catalog.py`,
`scripts/add_member_shapes.py` (DR1 light shapes into members.yaml), `scripts/group_bergamini_knots.py`.
