"""Posterior image-position contours from the standard fast local image solver.

Known extra images are starting points, not additional constraints. Recovered
roots are matched to the observed catalog first; only unmatched roots can enter
an extra-image contour. No new global image search is performed.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import contourpy
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde

from .exact_images import ExactImageSolver


def match_extra_roots(roots, anchors, tolerance):
    output = np.full((len(anchors), 2), np.nan)
    if len(roots) and len(anchors):
        distances = np.linalg.norm(np.asarray(roots)[:, None]-np.asarray(anchors)[None], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        for r, c in zip(rows, cols):
            if distances[r, c] <= tolerance:
                output[c] = roots[r]
    return output


def compute_position_draws(config, state, evaluator, posterior, extra, *, log=print):
    """One local LM batch per joint draw, seeded at catalog and known extras."""
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    labels = list(map(str, state.images.image_labels))
    family_ids = list(map(str, state.images.family_ids))
    families = {str(f.family_id): f for f in state.family_data}
    extra = extra.reset_index(drop=True)
    anchors = extra[['x_model_arcsec','y_model_arcsec']].to_numpy(float) if len(extra) else np.empty((0, 2))
    extra_families = extra.family_id.astype(str).tolist() if len(extra) else []
    catalog_xy = np.column_stack([state.images.x_obs, state.images.y_obs])
    starts = np.vstack([catalog_xy, anchors])
    fi = np.r_[np.asarray(state.images.family_index_per_image), [family_ids.index(f) for f in extra_families]].astype(int)
    # Per-seed redshift normalizations, including the extra starting points.
    factors = np.array([state.dpie_sigma0_factor_by_z[float(families[family_ids[k]].z_source)] for k in fi])
    @jax.jit
    def prepare(theta):
        physical = evaluator.physical(theta)
        return evaluator.packed_state(physical, jnp.asarray(factors))[0], evaluator.source_positions(theta)
    recovered = np.full((len(posterior.samples), len(labels), 2), np.nan)
    extras = np.full((len(posterior.samples), len(anchors), 2), np.nan)
    label_index = {label: i for i, label in enumerate(labels)}
    separation = float(config.image_diagnostics.exact_image_min_separation_arcsec)
    for draw, theta in enumerate(posterior.samples):
        packed, sources = prepare(theta); sources = np.asarray(sources)
        # Potential kernel: the GP mean deflection goes with the draw's source (see Evaluator.gp_deflection_correction).
        correction = evaluator.gp_deflection_correction(theta) if getattr(evaluator, 'gp_potential', False) else None
        extra_kwargs = {} if correction is None else dict(correction=lambda x, y: correction(x, y, fi))
        result = solver._refine(packed, sources[fi, 0], sources[fi, 1], *starts.T,
                                shoot=solver.ray_shoot_and_jacobian_per_image, **extra_kwargs)
        points = np.column_stack([result['x'], result['y']])
        valid = np.asarray(result['identification_pass']) & np.isfinite(points).all(1)
        for k, fid in enumerate(family_ids):
            candidates = points[valid & (fi == k)]
            distinct = []
            for point in candidates:
                if all(np.linalg.norm(point-other)>separation for other in distinct):
                    distinct.append(point)
            roots = np.asarray(distinct).reshape(-1,2)
            matched = solver.match(*roots.T, families[fid])
            rows = [label_index[str(label)] for label in families[fid].image_labels]
            recovered[draw, rows] = np.column_stack([matched['matched_model_x_arcsec'],matched['matched_model_y_arcsec']])
            ex_rows = np.flatnonzero(np.array(extra_families) == fid)
            remaining = np.column_stack([matched['extra_model_x_arcsec'],matched['extra_model_y_arcsec']])
            extras[draw, ex_rows] = match_extra_roots(remaining, anchors[ex_rows], config.image_diagnostics.match_tolerance_arcsec)
        if draw % 100 == 0:
            log(f'[outputs] image-position contours: fast local finder {draw}/{len(posterior.samples)} draws')
    return dict(recovered=recovered, extra=extras, image_labels=np.asarray(labels),
                extra_family_ids=np.asarray(extra_families), extra_anchors=anchors)


def position_contours(points, *, grid_size=128):
    """68.3/95.4% KDE highest-density regions, conditional on local recovery."""
    points = np.asarray(points); points = points[np.isfinite(points).all(1)]
    if len(points)<10 or np.linalg.matrix_rank(np.cov(points.T))<2:
        return [], 'Fewer than 10 finite draws or singular position distribution'
    kde = gaussian_kde(points.T)
    lo, hi = points.min(0)-3*points.std(0), points.max(0)+3*points.std(0)
    for resolution in (grid_size, 2*grid_size, 4*grid_size):
        x = np.linspace(lo[0],hi[0],resolution); y = np.linspace(lo[1],hi[1],resolution)
        xx,yy=np.meshgrid(x,y)
        density = kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
        ranked=np.sort(density.ravel())[::-1]
        mass=np.cumsum(ranked)*(x[1]-x[0])*(y[1]-y[0])
        if abs(mass[-1]-1)<=.01:
            break
    else:
        return [], 'Density grid did not resolve unit probability'
    generator=contourpy.contour_generator(x=x,y=y,z=density)
    contours=[]
    for probability in (.682689492,.954499736):
        threshold=ranked[min(np.searchsorted(mass,probability),len(ranked)-1)]
        contours.append(dict(probability=probability,segments=generator.lines(float(threshold))))
    return contours, None


def prepare_contours(data, *, log=print):
    curves=[]; summaries=[]
    for kind in ('recovered','extra'):
        for i in range(data[kind].shape[1]):
            points=data[kind][:,i]; valid=np.isfinite(points).all(1)
            levels,reason=position_contours(points)
            identity=str(data['image_labels'][i]) if kind=='recovered' else f"{data['extra_family_ids'][i]} extra {i+1}"
            summaries.append(dict(kind=kind,identity=identity,valid_draws=int(valid.sum()),total_draws=len(points),skip_reason=reason))
            for level in levels:
                curves.append(dict(kind=kind,identity=identity,**level))
    log(f'[outputs] posterior contours: {len(curves)} probability levels')
    return dict(curves=curves,summaries=summaries)


def overview_draw_indices(count, best):
    """At most 256 deterministic joint draws, including the best fit."""
    indices=np.linspace(0,count-1,min(count,256)).astype(int)
    if count and best not in indices:
        indices[np.argmin(abs(indices-best))]=best
    return np.sort(indices)


def load_or_compute_contours(config, state, evaluator, posterior, extra, tables, *, log=print):
    tables=Path(tables); tables.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(np.asarray(posterior.samples).tobytes())
    fingerprint.update(b"fast-local-known-extra-v1" + (b"-gp-mean-deflection" if getattr(evaluator, 'gp_potential', False) else b""))
    fingerprint.update(json.dumps(config.to_json_dict(),sort_keys=True,default=str).encode())
    fingerprint.update(extra.to_csv(index=False).encode())
    fingerprint.update(json.dumps(state.members.classification,sort_keys=True,default=str).encode())
    key=fingerprint.hexdigest(); npz=tables/'posterior_overview_positions.npz'; meta=tables/'posterior_overview_positions.json'
    best=int(np.nanargmax(posterior.log_prob))
    selected=overview_draw_indices(len(posterior.samples),best)
    data=None
    if npz.exists() and meta.exists() and json.loads(meta.read_text()).get('fingerprint')==key:
        with np.load(npz) as saved:data={k:saved[k] for k in saved.files}
        stored=data.get('draw_indices',np.arange(len(data['recovered'])))
        if not np.isin(selected,stored).all():
            data=None
        else:
            log(f'[outputs] reusing cached positions for {len(selected)} selected draws')
    if data is None:
        subset=SimpleNamespace(samples=np.asarray(posterior.samples)[selected])
        data=compute_position_draws(config,state,evaluator,subset,extra,log=log)
        data['draw_indices']=selected
        np.savez_compressed(npz,**data)
        stored=selected
    rows=np.array([np.flatnonzero(stored==index)[0] for index in selected])
    selected_data={k:(v[rows] if k in ('recovered','extra') else v) for k,v in data.items()}
    contours=prepare_contours(selected_data,log=log)
    meta.write_text(json.dumps(dict(fingerprint=key,draws=len(selected),selected_draw_indices=selected.tolist(),
        posterior_draws_available=len(posterior.samples),cached_draws=len(data['recovered']),
        solver='standard local LM, exact kernels, no fallback',
        extras='Tracks best-fit extra-image starting points; catalog matches take priority; no global discovery',
        contours='Conditional position KDE highest-density regions, 68.3% and 95.4%',summaries=contours['summaries']),indent=2))
    return contours
