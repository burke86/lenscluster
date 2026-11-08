# Catalogs

A fit reads two catalogs: the **multiple images** (positions of each lensed source, with its redshift) and the
**cluster members** (positions and magnitudes of the galaxies on the scaling relation). Both are YAML files under
`catalogs/<cluster>/<source>/`, one folder per published source:

```text
catalogs/<cluster>/<source>/
    images.yaml          multiple images (format lenscluster-images/1)
    members.yaml|.cat    cluster members
    README.*             the source's own notes
    source/              the published files the YAML was converted from
    history/             earlier versions of the YAML, kept for the record
```

## Available catalogs

| folder | `run.xsh` entry | images | members | from |
|---|---|---|---|---|
| `hera/` | `HERA` | images.yaml | members.yaml | FF-SIMS Hera, with 7-band truth magnitudes |
| `ares/` | `ARES` | images.yaml | members.yaml | FF-SIMS Ares |
| `m0416/canucs2025/` | `M0416_CANUCS2025` | images.yaml (knots, parities) | members.yaml (DR1 light shapes, Granata 2026 kinematics) | Rihtarsic et al. 2025 |
| `m0416/bergamini2023/` | `M0416_BERGAMINI2023` | images.yaml | members.cat | Bergamini et al. 2023 |
| `m0416/diego2024/` | comparison only | images.ecsv / .fits | | Diego et al. 2024 (PEARLS) |
| `as1063/bergamini2019/` | `AS1063` | images.yaml | members.cat | Bergamini et al. 2019 |
| `as1063/diego2026/` | `AS1063_JWST` | images.yaml | members.yaml (Beauchesne 2025) | Diego et al. 2026 |
| `as1063/caminha2016/` | not an entry | | | Caminha et al. 2016 |

Positions of the real clusters are ICRS RA/Dec in degrees. Hera and Ares positions are offsets about (0, 0) in degrees
(the Lenstool `reference 3 0 0` convention), not sky coordinates. `catalogs/README.md` records the astrometric frames
of each source.

## Multiple images

A **system** is one background source at one redshift. Each image of a system has either one position or a set of
named **knots** (clumps of the same galaxy seen in every image). The knot names are declared once per system.

```yaml
format: lenscluster-images/1
defaults: {shape: [0.3734, 0.3734, 90.0], mag: 25.0}
systems:
  "1":
    z: 3.238
    images:
      a: {ra: 64.0490842, dec: -24.0628619}
      b: {ra: 64.0469592, dec: -24.0607969, sigma: 0.03, parity: -1}
  "D1":
    z: 0.9397
    knots: [A, B]
    images:
      a:
        parity: -1
        knots:
          A: [64.0365743, -24.0670439]
          B: [64.0366543, -24.0672539, 0.02]
      b:
        knots:
          A: {ra: 64.0369043, dec: -24.0674839, bands: {F150W: [25.8, 0.06]}}
```

System
: `z` (required), `reliability` (0 to 1, default 1), `knots` (list of names), `images`, `notes`.

Image
: `ra`/`dec` (degrees) or `knots`; `parity` (-1, 0 or +1; 0 means unknown); `sigma` (astrometric error in arcsec, also
  the default for its knots); `notes`.

Position (an image without knots, or one knot)
: `ra`, `dec`, `sigma`, `shape` (`[a, b, theta]`), `mag`, `mag_err`, `bands` (`{band: [mag, err]}` or
  `{band: {mag, err, detected, use}}`), `notes`. A knot may also be written `[ra, dec]` or `[ra, dec, sigma]`.

`defaults` sets `sigma`, `shape`, `mag`, `mag_err` and `reliability` for every entry that omits them. An astrometric
error given nowhere takes `ImageConstraintsConfig.sigma_inst_arcsec`.

The solver sees one **family** per (system, knot). Its id is the system name followed by the knot name (`D1` + `A` =
`D1A`), and each image label is `<family>.<image>`, for example `D1A.b`.

:::{note}
System and knot names use capital letters and digits; image names use lowercase letters. YAML reads an unquoted `021`
as a number and `y`/`n` as booleans, so quote names that could be read that way.
`image_catalog.write_image_catalog` quotes them for you.
:::

## Cluster members

```yaml
band: F160W
kinematics_sources:
  G26: {reference: "Granata et al. 2026, A&A, Appendix C", quantity: sigma_ap, aperture_arcsec: 0.8}
members:
  "2526":
    ra: 64.03816791
    dec: -24.06750808
    mag: {F160W: 17.0166}
    role: bcg
    kinematics: [{source: G26, sigma: 275.6, err: 1.5, sn: 164.6}]
  "4112": {ra: 64.0352, dec: -24.0735, mag: {F160W: 17.95}}
```

Top level
: `band` (required: the magnitude the scaling relation uses), `members` (required), `kinematics_sources`, `notes`.

Member
: `ra`/`dec` (required); `mag` (`{band: mag}` or `{band: [mag, err]}`, the `band` entry required); `shape`
  (`[a, b, theta]`, default round); `z` (a number, or `{value, kind}` with kind `spec` or `phot`); `role` (`member` or
  `bcg`); `treatment`; `keep`; `kinematics`; `notes`.

`treatment`
: `relation` (default): the member follows the scaling relation. `independent`: the member gets its own dPIE, added as
  an independent member halo with default priors unless the configuration declares it. `exclude`: kept for the record,
  not modeled.

`keep`
: `true` protects the member from the `MemberConfig.min_deflection_arcsec` impact cut. Members with kinematics are
  always kept.

`kinematics`
: A list of measurements `{source, sigma, err, sn, notes}` with `sigma` and `err` in km/s. A kinematics source gives
  its `reference`, a `quantity` (`sigma_ap` within a circular aperture of `aperture_arcsec`; `sigma_0` within
  $R_e/8$; `sigma_e` within $R_e$), and optionally `aperture_arcsec`. Measurements enter the likelihood only for the
  sources listed in `MemberPopulationConfig.kinematics_sources` (`run.xsh --kinematics G26`).

Member names are strings, so quote numeric names. A name repeated in `members` is an error.

Lenstool potfiles (`.cat`, columns `id x y a b theta mag lum color` after a `#REFERENCE` line) are read too.
`#REFERENCE 0` means `x`, `y` are RA and Dec in degrees; `#REFERENCE 3` means arcsecond offsets from the model's
reference point.

## Converting published catalogs

| script | converts |
|---|---|
| `scripts/convert_lenstool_catalog.py` | a Lenstool multiple-image catalog to `images.yaml` |
| `scripts/convert_member_catalog.py` | a potfile to `members.yaml`, with kinematics |
| `scripts/convert_canucs_catalog.py` | the CANUCS MACS J0416 release |
| `scripts/convert_as1063_jwst_catalog.py` | the Diego et al. 2026 Abell S1063 table |
| `scripts/add_member_shapes.py` | light shapes from a photometry catalog into `members.yaml` |
| `scripts/group_bergamini_knots.py` | Bergamini images into knotted systems |

Large inputs are not tracked: the Lenstool `bayes.dat` files, the FF-SIMS truth maps and images, and the CANUCS DR1
photometry catalog. See {doc}`scripts` for downloading imaging.
