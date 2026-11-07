"""Audit catalogs; write a separate selected member catalog."""
import time
import hashlib
import numpy as np
import pandas as pd
from .common import ROOT, OUT, SETTINGS, write_json

CATALOG_ROOT = ROOT / "data/Bergamini/M0416_Bergamini22"
IMAGES = CATALOG_ROOT / "obs_arcs.cat"
MEMBERS = CATALOG_ROOT / "CM_cat_MACSJ0416.cat"
REFERENCE = (3, 64.0381417, -24.0674722)

def selection(magnitude, ids, distance):
    n = int(np.ceil(SETTINGS['bright_fraction'] * len(magnitude)))
    order = np.lexsort((np.asarray(ids, str), magnitude))
    bright = np.zeros(len(magnitude), bool); bright[order[:n]] = True
    near = distance <= SETTINGS['near_arcsec']
    luminosity = 10. ** (-.4 * (magnitude - np.min(magnitude)))
    return bright | near, bright, near, luminosity

def audit():
    from lenscluster.inputs import load_dat_catalog, load_multiple_images_catalog, radec_to_offsets_arcsec
    started = time.perf_counter()
    images = load_multiple_images_catalog(IMAGES, REFERENCE, sigma_inst_arcsec=.3)
    members = load_dat_catalog(MEMBERS, REFERENCE, catalog_kind="potfile_galaxies")
    failures, warnings = [], []
    for name, df, idcol, required in [
        ('images', images, 'image_label', ['ra','dec','catalog_z']),
        ('members', members, 'id', ['ra','dec','catalog_mag'])]:
        if df[idcol].astype(str).duplicated().any(): failures.append(f"{name}: duplicate identities")
        for col in required:
            if not np.isfinite(pd.to_numeric(df[col], errors='coerce')).all(): failures.append(f"{name}: invalid {col}")
        if ((df.ra < 0) | (df.ra >= 360) | (df.dec < -90) | (df.dec > 90)).any(): failures.append(f"{name}: invalid celestial coordinates")
    for fid, group in images.groupby('family_id', sort=False):
        if group.catalog_z.nunique() != 1 or (group.catalog_z <= SETTINGS['z_lens']).any(): failures.append(f"family {fid}: inconsistent/non-background redshift")
        if len(group) < 2: failures.append(f"family {fid}: fewer than two images")
    warnings.append("Image a,b,theta are shape fields, not positional-error estimates; assumed 0.3 arcsec.")
    if 'catalog_color' in members and members.catalog_color.isna().any(): warnings.append(f"Missing optional member color: {int(members.catalog_color.isna().sum())} rows")
    warnings.append("Member catalog lum is not used; relative luminosity is calculated from magnitude.")
    for col in ('catalog_a','catalog_b','catalog_theta'):
        if not np.isfinite(members[col].to_numpy(float)).all():failures.append(f'Members: nonfinite {col}')
    if ((members.catalog_a<=0)|(members.catalog_b<=0)).any():failures.append('Members: nonpositive shape axis')
    payload = dict(image_columns=list(images.columns), member_columns=list(members.columns),
                   raw_image_columns=['id','RA','Dec','a','b','theta','z','mag'],
                   raw_member_columns=['id','RA','Dec','a','b','theta','mag','lum','color'],
                   images=len(images), families=images.family_id.nunique(), members=len(members),
                   redshift_range=[images.catalog_z.min(), images.catalog_z.max()],
                   magnitude_range=[members.catalog_mag.min(), members.catalog_mag.max()],
                   warnings=warnings, failures=failures,
                   hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (IMAGES,MEMBERS)})
    write_json(OUT/'catalog_audit.json', payload)
    if failures: raise ValueError('; '.join(failures))
    to_xy = lambda df: np.column_stack(radec_to_offsets_arcsec(df.ra.to_numpy(),df.dec.to_numpy(),REFERENCE[1],REFERENCE[2]))
    image_xy, member_xy = to_xy(images), to_xy(members)
    distance = np.linalg.norm(member_xy[:,None]-image_xy[None], axis=2).min(1)
    keep, bright, near, luminosity = selection(members.catalog_mag.to_numpy(float), members.id.astype(str), distance)
    selected = members.copy(); selected['nearest_image_arcsec']=distance; selected['bright']=bright; selected['near']=near; selected['keep']=keep; selected['relative_luminosity']=luminosity
    selected.to_csv(OUT/'member_selection.csv', index=False); images.to_csv(OUT/'images.csv', index=False)
    # Preserve the original raw catalog rows rather than formatting parsed coordinates.
    lines = [line for line in MEMBERS.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(lines) != len(members): raise ValueError("Raw/parsed member row mismatch")
    (OUT/'members_selected.cat').write_text('# REFERENCE 0\n'+'\n'.join(line for line,k in zip(lines,keep) if k)+'\n')
    payload.update(bright_count=int(bright.sum()),near_count=int(near.sum()),selected_count=int(keep.sum()),
                   selected_luminosity_fraction=float(luminosity[keep].sum()/luminosity.sum()),seconds=time.perf_counter()-started)
    expected=dict(images=237,families=88,members=213,selected_count=160)
    payload['expectation_differences']={k:dict(expected=v,actual=int(payload[k])) for k,v in expected.items() if payload[k]!=v}
    write_json(OUT/'catalog_audit.json',payload)
    print(payload, flush=True)
    return payload
