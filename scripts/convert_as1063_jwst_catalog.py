"""Build AS1063 YAML catalogs from the archived Diego table and Beauchesne FITS.

Run with the project's lenscluster/lenstronomy interpreter. No network or sampling.
The input table retains all ranks, flags and model-derived quantities; only measured
positions with spectroscopic redshifts enter images.yaml. See the catalog README.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lenscluster.image_catalog import write_image_catalog, load_image_catalog
from lenscluster.member_catalog import write_member_catalog, load_member_catalog

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/JWST/AS1063_Diego2026_Beauchesne2025'
# Same-source knot groups, identified by shared spectroscopic redshift and the
# corresponding compact groups in each macroimage. These are curated mappings,
# not galaxy IDs supplied by Diego's table. Preserve each original label in notes.
GROUPS = [(1, 2, 3, 4), (5, 6, 7), (9, 10, 11), (12, 13, 14, 15, 16, 17),
          (19, 20, 21), (25, 26, 27, 28, 29), (31, 32, 33, 34)]


def build():
    raw = DEST / 'raw'
    rows = json.loads((raw / 'diego_2601.11704v2_table1.json').read_text())['rows']
    assert len(rows) == 276 and len({r['label'] for r in rows}) == 276
    group_for = {f: group for group in GROUPS for f in group}
    z_for = {r['family']: r['z_spec'] for r in rows if r['z_spec'] is not None}
    systems, excluded, mapping = {}, [], []
    for row in rows:
        f = row['family']
        if row['flag'] == 3:
            reason = 'Position predicted by lens model, not an observed JWST position (flag 3).'
        elif row['flag'] in (4, 5, 6):
            reason = 'Outside JWST footprint or likely singly imaged arc (flags 4/5/6).'
        elif z_for[f] <= 0:
            reason = 'No spectroscopic redshift; geometric redshift retained only in raw table.'
        else:
            reason = None
        if reason:
            excluded.append(dict(label=row['label'], reason=reason))
            continue
        group = group_for.get(f, (f,))
        sid = 'D' + str(group[0])
        knot = chr(ord('A') + group.index(f)) if len(group) > 1 else ''
        image = row['image']
        # Families 32-34 lack the middle macroimage. Their published b is next to
        # 31.c, not 31.b; reconcile these image names without moving any position.
        if f in (32, 33, 34) and image == 'b':
            image = 'c'
        system = systems.setdefault(sid, dict(z=z_for[f], images={}, notes=(
            'Spectroscopic redshift from Diego et al. 2601.11704v2 Table 1. '
            + ('Rank B system 60: spectroscopic z=3.72 added in v2; retained rank B in the paper.'
               if f == 60 else 'Rank A.'))))
        assert system['z'] == z_for[f]
        position = dict(ra=row['ra'], dec=row['dec'], notes=f"Diego label {row['label']}; rank {row['rank']}; footnote flag {row['flag']}.")
        if knot:
            system['knots'] = [chr(ord('A') + i) for i in range(len(group))]
            system['images'].setdefault(image, dict(knots={}))['knots'][knot] = position
        else:
            assert image not in system['images']
            system['images'][image] = position
        mapping.append(dict(published_label=row['label'], system=sid, knot=knot, image=image,
                            family=sid + knot))
    header = ('AS1063 JWST: Diego et al., arXiv:2601.11704v2, Table 1.\n'
              'Observed rank A positions plus spectroscopic system 60; no model-predicted positions.\n'
              'System/knot grouping is curated; original labels and selection are in image_mapping.json.\n'
              'No published astrometric errors or photometry supplied: use configured sigma_inst_arcsec.\n'
              'Coordinates unchanged, J2000 degrees. See README.md for provenance and limitations.')
    write_image_catalog(DEST / 'images.yaml', systems, header=header)
    (DEST / 'image_mapping.json').write_text(json.dumps(dict(groups=GROUPS, included=mapping, excluded=excluded), indent=2)+'\n')

    data = fits.getdata(raw / 'Cluster_member_catalogue_full.fits', 1)
    assert len(data) == 288 and len(set(data['NUMBER'])) == 288
    wcs = WCS(fits.Header.fromtextfile(raw / 'F160W_data_masked_header.txt'))
    members = {}
    for row in data:
        ra, dec = float(row['RA_FITBEST_J2000']), float(row['DEC_FITBEST_J2000'])
        angle = np.deg2rad(row['THETA_FITBEST'])
        x, y = wcs.world_to_pixel_values(ra, dec)
        # Transform fitted pixel axes into the solver's west/north sky basis.
        ra1, dec1 = wcs.pixel_to_world_values(x + np.cos(angle), y + np.sin(angle))
        west, north = -(ra1-ra)*np.cos(np.deg2rad(dec)), dec1-dec
        theta = float(np.rad2deg(np.arctan2(north, west)))
        scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
        member = dict(ra=ra, dec=dec, mag={'F160W': [float(row['MAG_FITBEST']), float(row['MAG_FITERR'])]},
                      shape=[float(row['AFITBEST_IMAGE'])*scale, float(row['BFITBEST_IMAGE'])*scale, theta])
        if row['LOSVD'] > 0 and row['LOSVD_err'] > 0:
            member['kinematics'] = [dict(source='B25', sigma=float(row['LOSVD']), err=float(row['LOSVD_err']))]
        members[str(row['NUMBER'])] = member
    sources = dict(B25=dict(reference='Beauchesne et al., arXiv:2509.07762, section 4.2; released Cluster_member_catalogue_full.fits',
                           quantity='sigma_e', notes='MUSE LOSVD within an elliptical half-light aperture; not central dPIE sigma.'))
    write_member_catalog(DEST / 'members.yaml', members, band='F160W', kinematics_sources=sources,
                         header='Beauchesne et al. 2509.07762 public release, 288 rows, original NUMBER identifiers.\n'
                         'HST F160W photometry, not JWST photometry. BCG+ICL is a separate release and is absent here.\n'
                         'Fitted pixel axes converted to arcsec using the released image WCS (0.06 arcsec/pixel).\n'
                         'Angles transformed to west/north coordinates. Individual redshifts are absent in this FITS.\n'
                         'No lens-model SIGMA_FITBEST, stellar masses or fitted radii used as kinematic measurements.')
    loaded = load_image_catalog(DEST / 'images.yaml', sigma_inst_arcsec=0.01)
    member_table = load_member_catalog(DEST / 'members.yaml', (0, 0., 0.))
    # Compare the loader results, rather than the intermediate dictionaries, to the input data.
    source = {r['label']: r for r in rows}
    for m in mapping:
        match = loaded[loaded.image_label == m['family']+'.'+m['image']]
        assert len(match) == 1
        r = source[m['published_label']]
        np.testing.assert_array_equal(match[['ra', 'dec']].to_numpy()[0], [r['ra'],r['dec']])
        assert match.catalog_z.iloc[0] == z_for[r['family']]
    assert not loaded.duplicated(['ra', 'dec']).any()
    assert (loaded.groupby('family_id').size() >= 2).all()
    np.testing.assert_array_equal(member_table.ra, data['RA_FITBEST_J2000'])
    np.testing.assert_array_equal(member_table.dec, data['DEC_FITBEST_J2000'])
    np.testing.assert_array_equal(member_table.catalog_mag, data['MAG_FITBEST'])
    for r, measurements in zip(data, member_table.catalog_kinematics):
        if r['LOSVD'] > 0 and r['LOSVD_err'] > 0:
            assert measurements[0]['sigma'] == r['LOSVD'] and measurements[0]['err'] == r['LOSVD_err']
    report = dict(image_positions=len(loaded), systems=loaded.system_id.nunique(), families=loaded.family_id.nunique(),
                  members=len(member_table), kinematic_members=sum(bool(m.get('kinematics')) for m in members.values()),
                  raw_image_rows=len(rows), excluded_image_rows=len(excluded),
                  input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.iterdir())},
                  checks='Exact loader round-trip agreement for positions, spectroscopic redshifts, member magnitudes and LOSVD; unique labels and positions; >=2 images per family.')
    (DEST / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
