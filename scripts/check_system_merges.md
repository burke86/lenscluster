# Visual check of M0416 CANUCS system merges

Catalog: `data/CANUCS/M0416_Rihtarsic2025/images.yaml` (303 positions, 111 knot families, 83 systems). Nothing has been
merged yet. Groups come from the Bergamini et al. 2022 release (`data/Bergamini/M0416_Bergamini22/obs_arcs.cat`) and from
a source-plane comparison at the sampled-shape init in basin B. A merge keeps one source per old system, as separate knots
of one system, so with `gp_blocks = "all"` the likelihood does not change; it changes the `--fast` system blocks and the
bookkeeping.

## How to plot

One row per group, one panel per macro-image (the same image of the galaxy in every old system), JWST NIRCam RGB as in
the fold cutouts (blue F115W+F150W, green F150W+F200W, red F444W). Centre each panel on the coordinates below and make
it the listed span plus 1.5", at least 2.5" across. Mark each old system with its own marker and label every position
with its catalog label. Use the same stretch for all panels of a group so colours can be compared. Where Bergamini
positions are listed, overlay them with an open marker; they are already shifted into the CANUCS frame
(Bergamini + (0.214", -0.080") in (RA cos dec, Dec)). Optional: overlay the best-draw predicted positions of every
knot, which shows where a missing counter-image should be.

## What decides a merge

1. The knots sit inside one continuous light envelope in every macro-image, not two separate objects that happen to
   be close.
2. The pattern of knots repeats from image to image, mirrored across a critical curve where the parity flips, and with
   the separations scaled by the local magnification.
3. The knots have similar colours across images (the same galaxy at one redshift).
4. Every macro-image that should contain the galaxy shows every knot, unless the knot falls outside the caustic.

## Merge (confirm)

Positions in arcsec relative to the panel centre (east-positive x, north-positive y).

**G1: K6 + K7** (z 1.895, 1.896; Bergamini system 5: K7 = 5.1, K6 A-E = 5.2-5.6). K7 becomes knot K6F.

| image | centre (RA, Dec) | span | positions |
|---|---|---|---|
| a | 64.041449, -24.061972 | 4.5" | K6A(-0.97,+0.53) K6B(-0.71,+0.35) K6C(+0.31,-0.11) K6D(+1.07,-0.65) K6E(+2.25,-1.34) K7(-1.96,+1.22) |
| b | 64.043036, -24.063004 | 3.9" | K6A(+0.36,-0.36) K6B(+0.16,-0.14) K6C(-0.20,+0.43) K6D(-0.66,+0.76) K6E(-1.32,+1.26) K7(+1.67,-1.94) |
| c | 64.047616, -24.068891 | 1.4" | K6A(-0.27,+0.17) K6B(-0.20,+0.06) K6C(+0.13,-0.05) K6D(+0.32,-0.37) K6E(+0.62,-0.52) K7(-0.60,+0.71) |

Look for: K7 at the end of the K6 chain in all three images (it is beyond K6A in each).

**G2: K10 + K76** (z 1.9894, 1.990; Bergamini system 15: K10 = 15.1, K76A = 15.2, K76B = 15.4).

| image | centre | span | positions |
|---|---|---|---|
| a | 64.030911, -24.067244 | 1.0" | K10(-0.25,+0.32) K76A(-0.25,+0.00) K76B(+0.50,-0.32) |
| b | 64.035208, -24.070967 | 0.9" | K10(+0.35,-0.17) K76A(+0.09,-0.20) K76B(-0.44,+0.37) |
| c | 64.041981, -24.075871 | 1.2" | K10(-0.35,+0.42) K76A(-0.25,+0.06) K76B(+0.60,-0.48) |

Look for: the K10 - K76A - K76B order flipping in image b against a and c.

**G3: K36 + K37 + K40** (z 3.2882, 3.2883, 3.292; Bergamini system 9: 9.1, 9.2, 9.3). In image a all three share CANUCS
F200W object 3120089. K40's letters are swapped: K40.b goes with image a.

| image | centre | span | positions |
|---|---|---|---|
| a | 64.040068, -24.066694 | 0.5" | K36.a(+0.25,-0.25) K37.a(-0.01,+0.07) K40.b(-0.24,+0.18) |
| b | 64.045391, -24.072537 | 1.4" | K36.b(-0.71,+0.62) K37.b(+0.11,-0.02) K40.a(+0.60,-0.60) |

Look for: one object in image a, a chain of three in image b. Whether K40 (260 km/s from the other two) is a knot or a
companion.

**G4: K69 + K70** (z 6.1452, 6.145; Bergamini system 2: 2.1, 2.2). Letters are rotated: K69.a/b/c go with K70.b/c/a.

| image | centre | span | positions |
|---|---|---|---|
| a | 64.043557, -24.058977 | 0.5" | K69.a(+0.27,-0.16) K70.b(-0.27,+0.16) |
| b | 64.048071, -24.062251 | 1.2" | K69.b(-0.57,+0.62) K70.c(+0.57,-0.62) |
| c | 64.050900, -24.066497 | 0.5" | K69.c(+0.09,-0.23) K70.a(-0.09,+0.23) |

Look for: two faint clumps in one envelope in each image. Image c is about 1" from the K72/K74 images (U3); keep them
apart.

## Undecided (the plot decides)

**U1: K33 + K77** (both z 3.2215; Bergamini system 14: K33 = 14.1, K77 = 14.2). K77.a and K77.b match 14.2b and 14.2a
within 0.015". K77.c does not match 14.2c: CANUCS puts it 0.13" from K33.c (= 14.1d), while Bergamini puts 14.2c on
14.1c (= K33.e), 1.7" away, with a doubled error of 0.75".

| panel | centre | span | positions |
|---|---|---|---|
| north group | 64.034259, -24.066750 | 2.1" | K33.a(+0.97,-0.84) K33.b(-0.01,+0.89) K33.c(-0.74,-0.62) K33.e(-0.64,+1.03) K77.a(+0.38,+0.64) K77.b(+0.91,-0.48) K77.c(-0.87,-0.61) |
| K33.f | 64.035245, -24.067941 | single | K33.f |
| K33.d | 64.046124, -24.076814 | single, far south | K33.d |

Bergamini overlay (CANUCS frame): 14.1a 64.0345568 -24.0669778; 14.1b 64.0342526 -24.0665069; 14.1c 64.0340651
-24.0664611; 14.1d 64.0340318 -24.0669236; 14.1e 64.0352359 -24.0679408; 14.1f 64.0461244 -24.0768110; 14.2a 64.0345318
-24.0668819; 14.2b 64.0343739 -24.0665700; 14.2c 64.0340651 -24.0664611.

Questions:
- Does the clump at K77.c exist next to K33.c, and is there a second clump next to K33.e? Whichever exists is the third
  image of the second knot.
- Is a faint second clump visible next to K33.f and K33.d? If K77 is a knot of K33 it should appear there too, unless it
  falls outside the caustic.
- If K77.c is real and the pattern repeats, merge as knots (K33A, K33B). If K77.c is ambiguous, merge only K77.a and
  K77.b, or drop K77.c.

**U2: K89 + K90 + K91** (z 3.0654, 3.0665, 3.0666; no Bergamini images). Sources 0.48-0.99" apart, that is 4-8 kpc.

| image | centre | span | positions |
|---|---|---|---|
| a | 64.027976, -24.070849 | 1.5" | K89(+0.06,-0.09) K90(+0.70,+0.24) K91(-0.76,-0.16) |
| b | 64.033172, -24.074152 | 1.3" | K89(-0.06,+0.02) K90(-0.21,+0.65) K91(+0.27,-0.67) |
| c | 64.040705, -24.080165 | 1.3" | K89(+0.05,-0.06) K90(+0.59,+0.22) K91(-0.64,-0.15) |

Question: one galaxy with three clumps (shared envelope, similar colour) or three galaxies in a group (separate,
possibly different colours)? K90 and K91 differ by 7 km/s, K89 by about 85 km/s from both. Merge only on a shared
envelope.

**U3: K72 + K73 + K74** (all z 6.149; Bergamini 210: K74 = 210.1, K72A = 210.2, K72B = 210.3, K73 = 210.4). Same ID in
Bergamini, but K74's images lie 2.5" and 7" from K72's, and K73's 15-19" away. Current proposal: keep separate.

| panel | centre | span | positions |
|---|---|---|---|
| a | 64.050949, -24.066189 | 2.5" | K72B.a(+0.64,-1.24) K74.a(-0.64,+1.24) |
| b (K72) | 64.047239, -24.061214 | 0.5" | K72A.b K72B.b |
| b (K74) | 64.049292, -24.063362 | single | K74.b |
| c | 64.044519, -24.059329 | 0.5" | K72A.c(-0.12,+0.05) K72B.c(+0.12,-0.05) |
| K73 | 64.041564, -24.062654 and 64.040174, -24.061894 | singles | K73.a, K73.b |

Bergamini overlay (CANUCS frame): 210.1a 64.0507568 -24.0658475; 210.1b 64.0493009 -24.0633666; 210.2b 64.0472788
-24.0612661; 210.2c 64.0444868 -24.0593175; 210.3a 64.0511430 -24.0665341; 210.3b 64.0471913 -24.0611608; 210.3c
64.0445547 -24.0593450; 210.4b 64.0415597 -24.0626525; 210.4c 64.0401697 -24.0618958.

Questions: Is there an arc connecting K74.a to K72B.a? Is there a K74 counterpart near K72's image c? If K74 and K72 are
two separate objects in every image, keep them separate. K73 would only join if a counterpart of K72 appears near it.

## Keep separate (sanity check)

**S1: C2 + K29** (z 3.0693, 3.0773, 590 km/s apart). Matching images 1.3-1.8" apart: C2.a with K29.c, C2.b with K29.a,
C2.c with K29.b. Panels at C2.a 64.037226 -24.082406, C2.b 64.024872 -24.073498, C2.c 64.030753 -24.079548, each 3"
across with both systems marked. Expect two distinct galaxies in each panel.

## After the check

Fill in the decisions below. The merged-catalog YAML for G1-G4 is drafted in the conversation of 2026-11-05.
Relabelling changes image labels, so `src/lenscluster/data/m0416_image_f200w_photometry.csv` needs the same relabelling.

| group | decision | note |
|---|---|---|
| G1 K6+K7 | | |
| G2 K10+K76 | | |
| G3 K36+K37+K40 | | |
| G4 K69+K70 | | |
| U1 K33+K77 | | |
| U2 K89+K90+K91 | | |
| U3 K72+K73+K74 | | |
| S1 C2+K29 | | |
