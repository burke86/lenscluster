"""Local fold-pair posterior diagnostics, independent of cluster and imaging band.

Automatic pairs are model-identified candidates, not catalog morphology labels.
Image contours are conditional on local root recovery; source contours retain
all selected draws (at most 128), including those on the no-local-fold-pair side of their own caustic.
"""
from pathlib import Path
import json
import time
from types import SimpleNamespace

import contourpy
from .grf_plotting import plot_deflection_correction, PLOT_FIELD_VERSION
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
from scipy.stats import gaussian_kde


STYLE = Path(__file__).with_name('style.mplstyle')
BLUE, ORANGE, MAGENTA = '#0072B2', '#D55E00', '#CC33AA'
from .plot_style import CRITICAL_COLOR
GRF_COLOR = CRITICAL_COLOR
NEWTON_ITERATIONS = 15   # damped Newton steps per caustic root; the seeds are tracked critical points
CRITICAL_TRACK_STEPS = 16  # Newton steps along the normal that move the reference seeds onto each draw's critical curve


def fold_draw_indices(count, best, limit=128):
    """Reproducible joint draws within the configured budget, including the best fit."""
    indices = np.linspace(0, count-1, min(count, limit)).astype(int)
    if count and best not in indices:
        indices[np.argmin(abs(indices-best))] = best
    return np.sort(indices)


def candidate_pairs(xy, families, labels, determinants, max_separation):
    """Mutual nearest neighbors within a family, opposite exact best-fit parity."""
    pairs = []
    for family in np.unique(families):
        indices = np.flatnonzero(families == family)
        if len(indices) < 2:
            continue
        distance = np.linalg.norm(xy[indices, None] - xy[indices][None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        # Stable identity resolves exact distance ties independently of input order.
        nearest = [min(range(len(indices)), key=lambda j: (distance[i, j], str(labels[indices[j]])))
                   for i in range(len(indices))]
        for i, j in enumerate(nearest):
            if i < j and nearest[j] == i and 0 < distance[i, j] <= max_separation:
                a, b = indices[i], indices[j]
                if determinants[a]*determinants[b] < 0:
                    pairs.append(tuple(sorted((int(a), int(b)), key=lambda k: str(labels[k]))))
    return sorted(pairs, key=lambda pair: tuple(str(labels[i]) for i in pair))


def checked_root(equations, seeds, tolerance=1e-8):
    for seed in seeds:
        solution = root(equations, seed, tol=1e-10)
        error = np.linalg.norm(equations(solution.x))
        if np.isfinite(solution.x).all() and np.isfinite(error) and error < tolerance:
            return solution.x, float(error)
    raise ValueError('No verified local caustic root')


def relative_source(mapped, source, seeds, origin, image_tangent, reference_tangent):
    """Signed nearest-caustic distance and independently anchored tangent coordinate."""
    def nearest_equation(x):
        beta, a = mapped(np.atleast_2d(x))
        u, _, _ = np.linalg.svd(a[0])
        return [np.linalg.det(a[0]), np.dot(beta[0]-source, u[:, 0])]
    point, error = checked_root(nearest_equation, seeds)
    beta, a = mapped(point[None])
    u, singular, vh = np.linalg.svd(a[0])
    if singular[0] < 1e-6:
        raise ValueError('Degenerate critical point, not a regular fold')
    normal, null = u[:, -1], vh[-1]
    # Check the fold sign at two finite-difference scales rather than arbitrarily
    # assigning a side from the fitted source (which may be outside the caustic).
    bends = []
    for h in (.001, .0005):
        b, _ = mapped(np.array([point+h*null, point, point-h*null]))
        bends.append(np.dot(normal, (b[0]-2*b[1]+b[2])/h**2))
    if bends[0]*bends[1] <= 0 or abs(bends[1]) < 1e-8:
        raise ValueError('Fold-side orientation is degenerate or unresolved')
    normal *= np.sign(bends[1])
    def anchor_equation(x):
        _, a = mapped(np.atleast_2d(x))
        return [np.linalg.det(a[0]), np.dot(x-origin, image_tangent)]
    anchor, anchor_error = checked_root(anchor_equation, [origin, point])
    anchor_beta, anchor_a = mapped(anchor[None])
    tangent = np.linalg.svd(anchor_a[0])[0][:, 0]
    if tangent @ reference_tangent < 0:
        tangent = -tangent
    return np.array([(source-anchor_beta[0]) @ tangent, (source-beta[0]) @ normal]), max(error, anchor_error)


def _kernels(evaluator):
    indices = np.arange(evaluator.n_components, dtype=np.int32)
    @jax.jit
    def fields(packed, points):
        alpha, u = evaluator.component_rows(points, packed, indices)
        return points-jnp.sum(alpha, axis=0), jnp.eye(2)-jnp.sum(u, axis=0)
    @jax.jit
    def critical(packed, seeds, normal):
        def step(_, points):
            _, u, du = evaluator.component_rows(points, packed, indices, derivative=True)
            a, da = jnp.eye(2)-jnp.sum(u, axis=0), -jnp.sum(du, axis=0)
            det = jnp.linalg.det(a)
            grad = (a[:, 1, 1, None]*da[:, 0, 0]+a[:, 0, 0, None]*da[:, 1, 1]
                    -a[:, 1, 0, None]*da[:, 0, 1]-a[:, 0, 1, None]*da[:, 1, 0])
            denom = grad @ normal
            safe = jnp.where(abs(denom)>1e-12, denom, 1.)
            shift = jnp.where(abs(denom)>1e-12, det/safe, 0.)
            return points-jnp.clip(shift, -.3, .3)[:, None]*normal
        points = jax.lax.fori_loop(0, CRITICAL_TRACK_STEPS, step, seeds)
        beta, a = fields(packed, points)
        return points, beta, jnp.linalg.det(a)

    def point(packed, x):
        beta, a = fields(packed, x[None])
        return beta[0], a[0]

    def newton(equation, packed, seed, args, iterations=NEWTON_ITERATIONS):
        """Damped Newton on a 2x2 system; a singular step leaves the point where it is and the residual check rejects it."""
        def step(_, x):
            residual = equation(packed, x, *args)
            jacobian = jax.jacfwd(equation, argnums=1)(packed, x, *args)
            shift = jnp.linalg.solve(jacobian, residual)
            shift = jnp.where(jnp.isfinite(shift), shift, 0.)
            return x-jnp.clip(shift, -.5, .5)
        x = jax.lax.fori_loop(0, iterations, step, seed)
        return x, jnp.linalg.norm(equation(packed, x, *args))

    def first_root(equation, packed, seeds, usable, args, tolerance=1e-8):
        """The first verified root in seed order (checked_root), or the last seed's end point and found=False."""
        xs, errors = jax.vmap(lambda s: newton(equation, packed, s, args))(seeds)
        ok = usable & jnp.isfinite(xs).all(1) & jnp.isfinite(errors) & (errors < tolerance)
        k = jnp.argmax(ok)
        return xs[k], errors[k], ok.any()

    def nearest_equation(packed, x, source):
        beta, a = point(packed, x)
        u, _, _ = jnp.linalg.svd(a, full_matrices=False)
        return jnp.array([jnp.linalg.det(a), jnp.dot(beta-source, u[:, 0])])

    def anchor_equation(packed, x, origin, image_tangent):
        _, a = point(packed, x)
        return jnp.array([jnp.linalg.det(a), jnp.dot(x-origin, image_tangent)])

    @jax.jit
    def relative_batch(packed, sources, seeds, normal, origin, image_tangent, reference_tangent):
        """relative_source for every draw at once: ``packed`` is stacked over draws, ``sources`` is (draws, 2).
        Returns the tracked critical points, the relative source, the root residual and a failure code per draw
        (0 ok; the codes index FAILURE_REASONS in the order of the sequential checks)."""
        def one(packed, source):
            points, betas, det = critical(packed, seeds, normal)
            valid = jnp.isfinite(det) & (jnp.abs(det) < 1e-6)
            curve = jnp.where(valid[:, None], points, jnp.nan)
            distance = jnp.where(valid, jnp.linalg.norm(betas-source, axis=1), jnp.inf)
            order = jnp.argsort(distance)[:3]
            x, error, found = first_root(nearest_equation, packed, points[order], valid[order], (source,))
            beta, a = point(packed, x)
            u, singular, vh = jnp.linalg.svd(a, full_matrices=False)
            fold_normal, null = u[:, -1], vh[-1]
            def bend(h):
                b, _ = fields(packed, jnp.stack([x+h*null, x, x-h*null]))
                return jnp.dot(fold_normal, (b[0]-2*b[1]+b[2])/h**2)
            bends = jnp.array([bend(.001), bend(.0005)])
            fold_normal = fold_normal*jnp.sign(bends[1])
            anchor, anchor_error, anchor_found = first_root(anchor_equation, packed, jnp.stack([origin, x]), jnp.array([True, True]),
                                                            (origin, image_tangent))
            anchor_beta, anchor_a = point(packed, anchor)
            tangent = jnp.linalg.svd(anchor_a, full_matrices=False)[0][:, 0]
            tangent = tangent*jnp.where(tangent @ reference_tangent < 0, -1., 1.)
            relative = jnp.array([(source-anchor_beta) @ tangent, (source-beta) @ fold_normal])
            code = jnp.where(~valid.any(), 1,
                   jnp.where(~found, 2,
                   jnp.where(singular[0] < 1e-6, 3,
                   jnp.where((bends[0]*bends[1] <= 0) | (jnp.abs(bends[1]) < 1e-8), 4,
                   jnp.where(~anchor_found, 2, 0)))))
            return curve, relative, jnp.maximum(error, anchor_error), code
        return jax.vmap(one)(packed, sources)
    return fields, critical, relative_batch


FAILURE_REASONS = {1: 'No tracked local critical points', 2: 'No verified local caustic root',
                   3: 'Degenerate critical point, not a regular fold', 4: 'Fold-side orientation is degenerate or unresolved',
                   5: 'Caustic solution left the local neighborhood'}


def posterior_image_positions(config, state, evaluator, posterior, *, family_ids=None, log=print):
    """Reuse the standard batched posterior finder once per draw, without fallback.  ``family_ids`` restricts the solve
    to those families (the fold-pair families); the other rows of the (draws, images, 2) result stay NaN."""
    from .exact_images import ExactImageSolver
    solver = ExactImageSolver(evaluator, config.image_diagnostics)
    families_by_id = {str(f.family_id): f for f in state.family_data}
    all_ids = [str(fid) for fid in state.images.family_ids]
    selected = all_ids if family_ids is None else [fid for fid in all_ids if fid in {str(f) for f in family_ids}]
    families = [families_by_id[fid] for fid in selected]
    selected_index = np.array([all_ids.index(fid) for fid in selected], dtype=int)
    family_index = np.asarray(state.images.family_index_per_image, dtype=int)
    rows = np.flatnonzero(np.isin(family_index, selected_index))
    remap = {int(f): k for k, f in enumerate(selected_index)}
    images = SimpleNamespace(x_obs=np.asarray(state.images.x_obs, dtype=float)[rows], y_obs=np.asarray(state.images.y_obs, dtype=float)[rows],
                             family_index_per_image=np.array([remap[int(f)] for f in family_index[rows]], dtype=int))
    factors = jnp.asarray(np.asarray(evaluator.factor_by_z)[np.asarray(state.images.z_index_per_image)[rows]])
    label_index = {str(label): i for i, label in enumerate(state.images.image_labels)}
    positions = np.full((len(posterior.samples), len(label_index), 2), np.nan)
    physical_source = jax.jit(lambda t: (evaluator.physical(t), evaluator.source_positions(t)[selected_index]))
    for draw, theta in enumerate(posterior.samples):
        physical, sources = physical_source(theta)
        mean = evaluator.gp_deflection_correction(theta) if getattr(evaluator, 'gp_knee', False) else None
        correction = None if mean is None else (lambda x, y, k: mean(x, y, selected_index[np.asarray(k)]))
        extra_kwargs = {} if correction is None else dict(correction=correction)
        details = solver.predict_all_local(images, families, evaluator.packed_state(physical, factor_per_image=factors)[0], np.asarray(sources),
                                           **extra_kwargs)
        for family, detail in zip(families, details):
            rows = [label_index[str(label)] for label in family.image_labels]
            xy = np.column_stack([detail['matched_model_x_arcsec'], detail['matched_model_y_arcsec']])
            positions[draw, rows] = np.where(np.asarray(detail['recovered_image_mask'])[:, None], xy, np.nan)
        if draw % 100 == 0:
            log(f'[outputs] fold image positions, standard local finder: {draw}/{len(posterior.samples)} draws')
    return positions


def compute_fold_diagnostics(config, state, evaluator, posterior, *, log=print):
    """Compute at most 128 joint draws once; reuse for HST and JWST renderings."""
    started = time.monotonic()
    full_posterior = posterior
    result = dict(pairs=[], skipped=[], selection='Mutual nearest same-family images; opposite best-fit exact parity; '
                  'critical curve between images; regular fold. Model-identified candidates.',
                  source_frame='x: tangent projection from fixed image-plane cross-section caustic anchor; '
                  'y: signed nearest distance to own caustic, positive toward the local fold pair',
                  image_contours='Conditional on verified local root recovery, not a global image-existence test',
                  draw_indices=list(range(len(posterior.samples))),
                  max_separation_arcsec=config.image_diagnostics.fold_pair_max_separation_arcsec,
                  image_finder='ExactImageSolver.predict_all_local, configured tolerances, no fallback',
                  kernel='exact', angular_units='arcsec', contour_definition='Whitened Gaussian KDE; leave-one-out bandwidth with one-standard-error smoothing, Scott to 2*Scott; highest-density regions enclosing 68.2689492% and 95.4499736%')
    if not len(posterior.samples) or not np.isfinite(posterior.log_prob).any():
        result['skipped'].append(dict(reason='No finite best posterior draw'))
        return result
    original_best = int(np.nanargmax(posterior.log_prob))
    draw_limit = config.image_diagnostics.fold_max_draws
    indices = fold_draw_indices(len(posterior.samples), original_best, draw_limit)
    result.update(draw_indices=indices.tolist(), original_draw_indices=indices.tolist(),
                  posterior_draws_available=len(posterior.samples), draw_limit=draw_limit, best_original_draw=original_best)
    posterior = SimpleNamespace(samples=np.asarray(posterior.samples)[indices], log_prob=np.asarray(posterior.log_prob)[indices])
    best = int(np.nanargmax(posterior.log_prob)); result['best_draw'] = best
    physical_source = jax.jit(lambda t: (evaluator.physical(t), evaluator.source_positions(t)))
    physical, _ = physical_source(posterior.samples[best])
    xy = np.column_stack([state.images.x_obs, state.images.y_obs])
    labels = np.asarray(state.images.image_labels)
    family_index = np.asarray(state.images.family_index_per_image)
    families = {str(f.family_id): f for f in state.family_data}
    fields, critical, relative_batch = _kernels(evaluator)
    determinants = np.linalg.det(np.asarray(evaluator.fields(physical, force_exact=True)['jacobian']))
    pairs = candidate_pairs(xy, family_index, labels, determinants, config.image_diagnostics.fold_pair_max_separation_arcsec)
    log(f"[outputs] fold search: {len(pairs)} candidates, {len(posterior.samples)} draws")
    pair_family_ids = sorted({str(state.images.family_ids[int(family_index[i])]) for i, _ in pairs})
    image_positions = posterior_image_positions(config, state, evaluator, posterior, family_ids=pair_family_ids, log=log) if pairs else None

    def mapped_with(packed):
        def mapped(points):
            return tuple(map(np.asarray, fields(packed, jnp.asarray(points))))
        return mapped

    # The reference branch of every pair at the best fit; the draws are then visited once each, all pairs per draw.
    verified = []
    for i, j in pairs:
        pair_labels = labels[[i, j]].tolist()
        fi = int(family_index[i]); fid = str(state.images.family_ids[fi]); z = float(families[fid].z_source)
        observed = xy[[i, j]]; mid = observed.mean(0); separation = np.linalg.norm(observed[1]-observed[0])
        fov = max(4., 1.8*separation)
        packed = evaluator.packed_state_for_z(physical, z)
        mapped = mapped_with(packed)
        direction = (observed[1]-observed[0])/separation
        tangent = np.array([direction[1], -direction[0]])
        log(f"[outputs] fold {pair_labels}: verifying reference branch")
        try:
            def cross_equation(x):
                _, a = mapped(np.atleast_2d(x))
                return [np.linalg.det(a[0]), (x-mid) @ tangent]
            origin, _ = checked_root(cross_equation, [mid])
            if abs((origin-mid) @ direction) > separation/2:
                raise ValueError('Critical curve is not between the two images')
            h = 1e-4
            _, a = mapped(origin+np.array([[h, 0], [-h, 0], [0, h], [0, -h]]))
            ds = np.linalg.det(a)
            normal = np.array([ds[0]-ds[1], ds[2]-ds[3]])/(2*h)
            if np.linalg.norm(normal)<1e-8:
                raise ValueError('Degenerate critical-curve gradient')
            normal /= np.linalg.norm(normal); tangent = np.array([normal[1], -normal[0]])
            _, a = mapped(origin[None]); ref_tangent = np.linalg.svd(a[0])[0][:, 0]
            axis = np.linspace(-fov*.85, fov*.85, 201)     # seeds only: Newton refines them onto the curve
            xx, yy = np.meshgrid(mid[0]+axis, mid[1]+axis)
            _, u = evaluator.total_fields_at(np.column_stack([xx.ravel(), yy.ravel()]), packed)
            segments = contourpy.contour_generator(x=xx, y=yy, z=np.linalg.det(np.eye(2)-u).reshape(xx.shape)).lines(0.)
            segment = min(segments, key=lambda s: np.linalg.norm(s-origin, axis=1).min())
            along = (segment-origin) @ tangent; order = np.argsort(along)
            coordinates = np.linspace(max(-.8*fov, along.min()+.01), min(.8*fov, along.max()-.01), 97)
            seeds = np.column_stack([np.interp(coordinates, along[order], segment[order, k]) for k in range(2)])
            seeds, _, det = map(np.asarray, critical(packed, seeds, normal))
            if np.any(np.linalg.norm(np.diff(seeds, axis=0), axis=1) > .25*fov):
                raise ValueError('Reference critical curve changes branch on the chosen cross-sections')
            if not np.all(np.isfinite(det) & (abs(det)<1e-6)):
                raise ValueError('Reference fold branch cannot be tracked on common cross-sections')
        except (ValueError, np.linalg.LinAlgError) as error:
            log(f'[outputs] fold {pair_labels}: skipped ({error})')
            result['skipped'].append(dict(labels=pair_labels, reason=str(error)))
            continue
        n = len(posterior.samples)
        verified.append(dict(labels=pair_labels, family_id=fid, z_source=z, observed=observed,
                             separation=float(separation), fov=fov, relative=np.full((n, 2), np.nan),
                             positions=image_positions[:, [i, j]].copy(), critical=np.full((n, 97, 2), np.nan),
                             source_root_residual=np.full(n, np.nan), failures=[],
                             normal=normal, origin=origin, seeds=seeds, tangent=tangent, reference_tangent=ref_tangent, family_index=fi))
    # All draws of a pair in one batched solve; the packed lens states are stacked over draws once per source redshift.
    if verified:
        physical_all, sources_all = jax.jit(jax.vmap(physical_source))(jnp.asarray(posterior.samples))
    packed_by_z = {}
    for pair in verified:
        z = pair['z_source']
        if z not in packed_by_z:
            packed_by_z[z] = jax.jit(jax.vmap(lambda p, z=z: evaluator.packed_state_for_z(p, z)))(physical_all)
        curves, relative, errors, codes = relative_batch(packed_by_z[z], sources_all[:, pair['family_index']], pair['seeds'], pair['normal'],
                                                         pair['origin'], pair['tangent'], pair['reference_tangent'])
        curves, relative, errors, codes = map(np.asarray, (curves, relative, errors, codes))
        codes = np.where((codes == 0) & (np.linalg.norm(relative, axis=1) > 10*pair['fov']), 5, codes)
        failed = codes != 0
        pair['critical'][:] = curves
        pair['relative'][:] = np.where(failed[:, None], np.nan, relative)
        pair['source_root_residual'][:] = np.where(failed, np.nan, errors)
        pair['failures'] = [dict(draw=int(draw), reason=FAILURE_REASONS[int(codes[draw])]) for draw in np.flatnonzero(failed)]
        log(f"[outputs] fold {pair['labels']}: {len(posterior.samples)} draws, {int(failed.sum())} failed")
    for pair in verified:
        pair.pop('tangent'); pair.pop('reference_tangent'); pair.pop('family_index')
        result['pairs'].append(pair)
    result = prepare_fold_plot_data(config, state, evaluator, full_posterior, result, log=log)
    result['elapsed_seconds'] = time.monotonic()-started
    log(f"[outputs] fold diagnostics: {len(result['pairs'])} candidate pairs, {result['elapsed_seconds']:.1f}s")
    return result


def _credible_density_grid(points, *, grid_size=384):
    """Smooth 68.3/95.4% KDE regions without discarding tails or disconnected modes.

    Work in covariance-whitened coordinates so thin/tilted contours are resolved.
    Select bandwidth by leave-one-out log density; use the smoothest candidate
    within one standard error of the best score, with a Scott-to-2*Scott bound.
    """
    from scipy.special import logsumexp
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        return None
    covariance = np.cov(points.T)
    if np.linalg.matrix_rank(covariance) < 2:
        return None
    center = points.mean(axis=0)
    chol = np.linalg.cholesky(covariance)
    white = np.linalg.solve(chol, (points-center).T).T
    delta = white[:, None, :]-white[None, :, :]
    distances2 = np.sum(delta**2, axis=-1)
    np.fill_diagonal(distances2, np.inf)
    scott = len(points)**(-1/6)
    factors = np.array([1., 1.15, 1.3, 1.5, 1.75, 2.])
    bandwidths = scott*factors
    scores = np.array([logsumexp(-distances2/(2*h*h), axis=1)-2*np.log(h) for h in bandwidths])
    means = scores.mean(axis=1)
    best = int(np.argmax(means))
    se = scores[best].std(ddof=1)/np.sqrt(len(points))
    choice = np.flatnonzero(means >= means[best]-se)[-1]
    bandwidth = bandwidths[choice]
    kde = gaussian_kde(white.T, bw_method=bandwidth)
    lo, hi = white.min(axis=0)-4*bandwidth, white.max(axis=0)+4*bandwidth
    x, y = np.linspace(lo[0], hi[0], grid_size), np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    density = kde(np.array([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    area = (x[1]-x[0])*(y[1]-y[0])
    ranked = np.sort(density.ravel())[::-1]
    mass = np.cumsum(ranked)*area
    if abs(mass[-1]-1) > .002:
        return None
    levels = [ranked[min(np.searchsorted(mass, q), len(ranked)-1)] for q in (.954499736, .682689492)]
    physical = np.stack((xx, yy), axis=-1) @ chol.T + center
    return dict(x=physical[..., 0], y=physical[..., 1], density=density, levels=levels,
                cell_area=area, integrated_mass=float(mass[-1]), bandwidth_factor=float(factors[choice]),
                points_used=len(points))


def _contours(ax, points, color, transform=None, bounds=None, linewidths=None, outline=True):
    grid = _credible_density_grid(points)
    if grid is None:
        return False
    xx, yy = grid['x'], grid['y']
    if transform is not None:
        xx, yy = transform(xx, yy)
    contours = ax.contour(xx, yy, grid['density'], levels=grid['levels'], colors=[color],
                          linestyles=[(0, (2., 1.5)), '-'] if transform is not None else ['--', '-'],
                          linewidths=linewidths if linewidths is not None else ([.6, .85] if transform is not None else [1., 1.4]), antialiased=True)
    if transform is not None and outline:
        from matplotlib import patheffects
        contours.set_path_effects([patheffects.Stroke(linewidth=1.3, foreground=(0.,0.,0.,.75)), patheffects.Normal()])
    if bounds is not None:
        bounds.extend(path.vertices[np.isfinite(path.vertices).all(axis=1)] for path in contours.get_paths() if len(path.vertices))
    return True


def save_fold_data(data, tables):
    tables = Path(tables); tables.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in data.items() if k not in {'pairs', 'systems'}}
    metadata['pairs'] = []; arrays = {}
    for i, pair in enumerate(data['pairs']):
        meta = {}
        for key, value in pair.items():
            if isinstance(value, np.ndarray):
                arrays[f'pair_{i}_{key}'] = value
            else:
                meta[key] = value
        finite = np.isfinite(pair['relative']).all(1)
        meta.update(source_draws_valid=int(finite.sum()), source_draws_total=len(finite),
                    no_fold_pair_draws=int(np.sum(pair['relative'][:, 1]<0)),
                    image_draws_valid=np.isfinite(pair['positions']).all(2).sum(0).tolist())
        metadata['pairs'].append(meta)
    if 'systems' in data:
        metadata['systems'] = []
        for i, system in enumerate(data['systems']):
            meta = {}
            for key, value in system.items():
                if isinstance(value, np.ndarray):
                    arrays[f'system_{i}_{key}'] = value
                else:
                    meta[key] = value
            meta['source_draws_valid_by_knot'] = np.isfinite(system.get('source_aligned_grf', system.get('source_cartesian_grf', system['relative']))).all(2).sum(0).tolist()
            meta['image_draws_valid'] = np.isfinite(system['positions']).all(2).sum(0).tolist()
            metadata['systems'].append(meta)
    np.savez_compressed(tables/'fold_pairs_source_caustics.npz', **arrays)
    (tables/'fold_pairs_source_caustics.json').write_text(json.dumps(metadata, indent=2))


def load_fold_data(tables):
    """Read the numerical fold products, including optional system-level panels."""
    tables = Path(tables)
    data = json.loads((tables / 'fold_pairs_source_caustics.json').read_text())
    with np.load(tables / 'fold_pairs_source_caustics.npz') as arrays:
        for kind, prefix in (('pairs', 'pair'), ('systems', 'system')):
            for i, record in enumerate(data.get(kind, [])):
                start = f'{prefix}_{i}_'
                record.update({key[len(start):]: arrays[key].copy() for key in arrays.files if key.startswith(start)})
    return data


CRITICAL_CURVE_STATISTIC = 'pointwise posterior 50th percentile of tracked curve positions; band: 16th-84th percentiles of the same draws'


def critical_curve_quantiles(curves, seeds, origin, normal):
    """Matched line/band segments; omit gaps with <80% valid tracked draws."""
    curves, seeds, origin, normal = map(np.asarray, (curves, seeds, origin, normal))
    if not len(curves):
        return []
    offsets = np.einsum('dni,i->dn', curves-origin, normal)
    valid = np.flatnonzero(np.isfinite(offsets).sum(0) >= np.ceil(.8*len(curves)))
    groups = np.split(valid, np.flatnonzero(np.diff(valid)>1)+1)
    segments = []
    for group in groups:
        if len(group) < 2:
            continue
        quantiles = np.nanpercentile(offsets[:, group], [16, 50, 84], axis=0)
        tangent = seeds[group]-origin-((seeds[group]-origin) @ normal)[:, None]*normal
        segments.append(origin+tangent[None]+quantiles[..., None]*normal)
    return segments


def prepare_fold_plot_data(config, state, evaluator, posterior, data, *, log=print):
    """Shared preparation for automatic output and cached regeneration.

    Upgrade older fold caches to system panels and refresh versioned GRF plot
    products. The renderer always uses gravitational image positions, even if
    optional GP-adjusted diagnostics are present in a saved cache.
    """
    if 'systems' not in data and getattr(state.images, 'system_ids', ()):
        data = add_system_knots(config, state, evaluator, posterior, data, log=log)
    add_best_grf_critical_curves(state, evaluator, posterior, data, log=log)
    data["central_curve_statistic"] = CRITICAL_CURVE_STATISTIC
    for system in data.get("systems", []):
        system["central_curve_statistic"] = CRITICAL_CURVE_STATISTIC
    shared_grf = bool(getattr(evaluator, 'gp_knee', False) and state.gp_blocks == 'all')
    data['plot_presentation'] = dict(
        version='fold-system-display-v1',
        image_position_contours='Parametric + conditional-mean GRF' if shared_grf else 'Parametric',
        gp_displacement_overlay=False, model_image_markers=False,
        catalog_marker='Outlined x; 11 pt; 1.25 pt stroke',
        image_contours='68.3% solid, 95.4% dashed; 0.85/0.60 pt; dark outline',
        critical_curves='Orange dashed parametric; cyan-blue solid parametric + GRF' if shared_grf else 'Parametric',
        critical_bands='Pointwise 16-84% for both models' if shared_grf else 'Parametric pointwise 16-84%',
        grf_interpolation_version=PLOT_FIELD_VERSION if shared_grf else None,
        source_panel='Local caustic-aligned sources, independent axis ranges' if shared_grf else 'Parametric caustic frame',
        uncertainty_legend_entries=False,
        style='Shared stylesheet, top/right ticks, no titles or grids, PDF')
    return data


def add_system_knots(config, state, evaluator, posterior, data, *, log=print):
    """All knots in a fold's system, using a shared anchor and catalog image identities.

    Different fold-image pairs of one system stay separate. Source contours are
    never filtered by image recovery; every failure is retained as a missing draw.
    """
    from .image_catalog import load_image_catalog
    if not data['pairs']:
        data['systems'] = []
        return data
    catalog = load_image_catalog(config.model.image_constraints.catalog_path,
        sigma_inst_arcsec=config.model.image_constraints.sigma_inst_arcsec,
        sigma_floor_arcsec=config.model.image_constraints.sigma_floor_arcsec).set_index('image_label')
    labels = list(map(str, state.images.image_labels))
    if not set(labels).issubset(catalog.index):
        raise ValueError('Saved fold labels do not match the system/knot catalog')
    family_ids = list(map(str, state.images.family_ids))
    family_index = np.asarray(state.images.family_index_per_image)
    index = {label: i for i, label in enumerate(labels)}
    groups = {}
    for pair in data['pairs']:
        rows = catalog.loc[pair['labels']]
        if rows.system_id.nunique() != 1:
            raise ValueError('A fold pair must belong to one source system')
        key = (str(rows.system_id.iloc[0]), tuple(sorted(rows.image_id.astype(str))))
        groups.setdefault(key, []).append(pair)
    indices = np.asarray(data['draw_indices'], int)
    selected = SimpleNamespace(samples=np.asarray(posterior.samples)[indices], log_prob=np.asarray(posterior.log_prob)[indices])
    if len(selected.samples) > 128:
        raise ValueError('System fold plots must retain the 128-draw diagnostic cap')
    # Reuse recovered roots already stored for pair members; solve only missing knot families.
    positions = np.full((len(indices), len(labels), 2), np.nan)
    cached_labels = set()
    for pair in data['pairs']:
        for j, label in enumerate(pair['labels']):
            positions[:, index[str(label)]] = pair['positions'][:, j]
            cached_labels.add(str(label))
    needed = set()
    for (system, image_ids), _ in groups.items():
        rows = catalog.loc[labels]
        target = rows[(rows.system_id.astype(str) == system) & rows.image_id.astype(str).isin(image_ids)]
        needed.update(str(row.family_id) for label, row in target.iterrows() if str(label) not in cached_labels)
    if needed:
        computed = posterior_image_positions(config, state, evaluator, selected, family_ids=sorted(needed), log=log)
        for i, label in enumerate(labels):
            if str(catalog.loc[label, 'family_id']) in needed and label not in cached_labels:
                positions[:, i] = computed[:, i]
    physical_all, sources = jax.jit(jax.vmap(lambda theta: (evaluator.physical(theta), evaluator.source_positions(theta))))(jnp.asarray(selected.samples))
    fields, _, relative_batch = _kernels(evaluator)
    packed_by_z = {}
    systems = []
    for (system, image_ids), pairs in groups.items():
        anchor = min(pairs, key=lambda pair: (pair['separation'], pair['family_id']))
        rows = catalog.loc[labels]
        system_rows = rows[rows.system_id.astype(str) == system]
        fids = [fid for fid in family_ids if fid in set(system_rows.family_id.astype(str))]
        knot_ids = [str(system_rows[system_rows.family_id.astype(str) == fid].knot_id.iloc[0]) for fid in fids]
        z = float(anchor['z_source'])
        if not np.allclose(system_rows.catalog_z.to_numpy(float), z):
            raise ValueError('Knots in a source system must share its redshift')
        target = system_rows[system_rows.image_id.astype(str).isin(image_ids)]
        image_labels = target.index.astype(str).tolist()
        ids = [index[label] for label in image_labels]
        observed = np.column_stack((np.asarray(state.images.x_obs)[ids], np.asarray(state.images.y_obs)[ids]))
        # Include every observed knot in the selected two images, rather than a crop per knot.
        mid = (observed.min(0) + observed.max(0)) / 2
        fov = max(anchor['fov'], 1.6 * float(np.ptp(observed, axis=0).max()) + 1.)
        normal, origin = np.asarray(anchor['normal']), np.asarray(anchor['origin'])
        tangent = np.array([normal[1], -normal[0]])
        if z not in packed_by_z:
            packed_by_z[z] = jax.jit(jax.vmap(lambda p: evaluator.packed_state_for_z(p, z)))(physical_all)
        _, a = fields(evaluator.packed_state_for_z(physical_all[data['best_draw']], z), jnp.asarray(origin[None]))
        reference_tangent = np.linalg.svd(np.asarray(a)[0])[0][:, 0]
        relative = np.full((len(indices), len(fids), 2), np.nan)
        failures = []
        for k, fid in enumerate(fids):
            _, rel, errors, codes = relative_batch(packed_by_z[z], sources[:, family_ids.index(fid)], anchor['seeds'], normal,
                                                  origin, tangent, reference_tangent)
            rel, codes = np.asarray(rel), np.asarray(codes)
            codes = np.where((codes == 0) & (np.linalg.norm(rel, axis=1) > 10*fov), 5, codes)
            relative[:, k] = np.where((codes == 0)[:, None], rel, np.nan)
            failures.extend(dict(family_id=fid, draw=int(i), reason=FAILURE_REASONS[int(codes[i])]) for i in np.flatnonzero(codes))
        systems.append(dict(system_id=system, image_ids=list(image_ids), family_ids=fids, knot_ids=knot_ids,
            labels=image_labels, image_family_index=np.array([fids.index(str(catalog.loc[label, 'family_id'])) for label in image_labels]),
            observed=observed, positions=positions[:, ids], relative=relative, center=mid, fov=fov, z_source=z,
            anchor_family_id=anchor['family_id'], anchor_labels=anchor['labels'], critical=anchor['critical'],
            normal=normal, origin=origin, seeds=anchor['seeds'], failures=failures))
        log(f'[outputs] fold system {system}, images {image_ids}: {len(fids)} knots, {len(image_labels)} matched catalog positions')
    data['systems'] = systems
    data['system_source_frame'] = 'Common caustic anchor/tangent per system and image pair; vertical coordinate is each knot\'s signed nearest distance to its own caustic. Not Cartesian source coordinates.'
    return data


def add_best_grf_critical_curves(state, evaluator, posterior, data, *, log=print,
                                 system_block_fields=False, critical_only=False):
    """Two best-draw critical curves on the same sky grid; no posterior resampling."""
    if not (getattr(evaluator, 'gp_knee', False) and
            (state.gp_blocks == 'all' or (system_block_fields and state.gp_blocks == 'system'))):
        return data
    best = int(np.nanargmax(posterior.log_prob))
    theta = posterior.samples[best]
    correction = plot_deflection_correction(evaluator, theta)
    if correction is None:
        return data
    physical = evaluator.physical(theta)
    for system in data.get('systems', []):
        if ('critical_grf_best' in system and system.get('critical_overlay_draw') == best
                and system.get('critical_overlay_version') == PLOT_FIELD_VERSION):
            continue
        center, size = np.asarray(system['center']), float(system['fov'])
        packed = evaluator.packed_state_for_z(physical, float(system['z_source']))
        family = list(map(str,state.images.family_ids)).index(str(system['anchor_family_id']))
        curves = {}
        for resolution in (256,512):
            x = np.linspace(center[0]-.55*size,center[0]+.55*size,resolution)
            y = np.linspace(center[1]-.55*size,center[1]+.55*size,resolution)
            xx,yy = np.meshgrid(x,y); points = np.column_stack([xx.ravel(),yy.ravel()])
            _,hessian = evaluator.total_fields_at(points,packed,chunk=1024)
            if resolution == 256 and 'critical_parametric_best' not in system:
                segments = contourpy.contour_generator(x=x,y=y,z=np.linalg.det(np.eye(2)-hessian).reshape(xx.shape)).lines(0.)
                system['critical_parametric_best'] = _join_curves(segments)
            for start in range(0,len(points),384):
                q = points[start:start+384]
                c = correction(q[:,0],q[:,1],np.full(len(q),family))
                hessian[start:start+len(q)] += np.stack(c[2:],axis=-1).reshape(-1,2,2)
            determinant = np.linalg.det(np.eye(2)-hessian).reshape(xx.shape)
            curves[resolution] = _join_curves(contourpy.contour_generator(x=x,y=y,z=determinant).lines(0.))
        # Compare the same local branch on the existing image cross-sections.
        normal = np.asarray(system['normal']); tangent = np.array([normal[1],-normal[0]])
        differences = []
        for seed in system['seeds']:
            anchors = [curve_cross_section_anchor(curves[r],seed,tangent) for r in (256,512)]
            if all(a is not None for a in anchors):
                differences.append(float(np.linalg.norm(anchors[0]-anchors[1])))
        system['critical_grf_best'] = curves[512]
        system['critical_overlay_draw'] = best
        system['critical_overlay_version'] = PLOT_FIELD_VERSION
        system['grf_resolution_check'] = dict(coarse=256,fine=512,matched_cross_sections=len(differences),
            median_arcsec=float(np.median(differences)) if differences else None,
            p95_arcsec=float(np.percentile(differences,95)) if differences else None,
            max_arcsec=float(np.max(differences)) if differences else None)
        log(f"[outputs] smooth GRF {system['system_id']}: 256/512 comparison {system['grf_resolution_check']}")
    if not critical_only:
        add_grf_source_frames(state, evaluator, posterior, data, log=log)
    add_grf_critical_band(state, evaluator, posterior, data, log=log)
    for system in data.get('systems', []):
        system['critical_grf_local_draws'] = local_critical_branches(
            system['critical_grf_draws'], system['origin'], system['fov'])
        system['grf_band_valid_draws'] = np.isfinite(system['critical_grf_local_draws']).all(2).sum(0)
    if not critical_only:
        add_caustic_aligned_sources(state, evaluator, posterior, data, log=log)
    data['critical_overlay'] = 'Best-draw parametric and parametric + conditional-mean GRF curves; pointwise 16-84% bands for both models; source frame includes conditional-mean GRF'
    return data


def _join_curves(segments):
    return np.vstack([np.vstack([segment,[[np.nan,np.nan]]]) for segment in segments]) if segments else np.empty((0,2))


def local_critical_branches(curves, origin, fov):
    """Keep the continuous local branch nearest the fold anchor in each draw.

    A normal cross-section can also intersect an unrelated galaxy-scale loop.
    Do not connect these different branches into one uncertainty polygon.
    """
    result = np.full_like(curves, np.nan)
    for draw, curve in enumerate(curves):
        valid = np.isfinite(curve).all(1)
        steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        finite_steps = steps[np.isfinite(steps)]
        limit = max(.15*fov, 5*np.median(finite_steps)) if len(finite_steps) else .15*fov
        cuts = np.flatnonzero((steps > limit) | ~valid[:-1] | ~valid[1:])+1
        groups = [g for g in np.split(np.arange(len(curve)), cuts) if len(g)>1 and valid[g].all()]
        if groups:
            group = min(groups, key=lambda g: np.linalg.norm(curve[g]-origin, axis=1).min())
            result[draw, group] = curve[group]
    return result


def critical_crossings(determinant, seeds, normal, targets, span, grid_size=65):
    """Locate sign boundaries on fixed cross-sections, nearest the tracked branch.

    Bracketing tolerates the tabulated GRF Jacobian's derivative jumps. A boundary
    need not have an exactly zero determinant. Missing crossings stay NaN.
    """
    seeds, normal = np.asarray(seeds), np.asarray(normal)
    targets = np.nan_to_num(np.asarray(targets), nan=0.)
    offsets = targets[:, None] + np.linspace(-span, span, grid_size)
    points = seeds[:, None] + offsets[:, :, None]*normal
    values = np.asarray(determinant(points.reshape(-1, 2))).reshape(offsets.shape)
    crossing = np.isfinite(values[:, :-1]) & np.isfinite(values[:, 1:]) & (values[:, :-1]*values[:, 1:] <= 0)
    distance = abs(.5*(offsets[:, :-1]+offsets[:, 1:])-targets[:, None])
    choice = np.argmin(np.where(crossing, distance, np.inf), axis=1)
    valid = crossing.any(1)
    row = np.arange(len(seeds))
    low, high = offsets[row, choice].copy(), offsets[row, choice+1].copy()
    sign_low = values[row, choice].copy()
    for _ in range(12):
        mid = (low+high)/2
        value = np.asarray(determinant(seeds+mid[:, None]*normal))
        valid &= np.isfinite(value)
        left = sign_low*value <= 0
        high = np.where(left, mid, high)
        low = np.where(left, low, mid)
        sign_low = np.where(left, sign_low, value)
    result = seeds+(.5*(low+high))[:, None]*normal
    result[~valid] = np.nan
    return result


def add_grf_critical_band(state, evaluator, posterior, data, *, log=print):
    """Cache conditional-mean GRF critical branches for the saved joint draws."""
    import hashlib
    systems = data.get('systems', [])
    indices = np.asarray(data['draw_indices'], int)
    samples = np.asarray(posterior.samples)[indices]
    digest = hashlib.sha256(samples.tobytes()+indices.tobytes()+PLOT_FIELD_VERSION.encode()+b'grf-critical-band-v2')
    for system in systems:
        for key in ('seeds', 'normal', 'critical'):
            digest.update(np.asarray(system[key]).tobytes())
        digest.update(str(system['z_source']).encode())
    fingerprint = digest.hexdigest()
    if not systems or all(s.get('grf_band_fingerprint') == fingerprint for s in systems):
        return data
    fields, _, _ = _kernels(evaluator)
    physical_fn = jax.jit(evaluator.physical)
    family_ids = list(map(str, state.images.family_ids))
    previous = {id(system): system.get('critical_grf_draws', system['critical']).copy() for system in systems}
    for system in systems:
        system['critical_grf_draws'] = np.full_like(system['critical'], np.nan)
    for draw, theta in enumerate(samples):
        physical = physical_fn(theta)
        correction = plot_deflection_correction(evaluator, theta)
        if correction is None:
            raise ValueError('GRF critical band requires a conditional deflection field')
        for system in systems:
            packed = evaluator.packed_state_for_z(physical, float(system['z_source']))
            family = family_ids.index(str(system['anchor_family_id']))
            def determinant(points):
                values = np.empty(len(points))
                for start in range(0, len(points), 384):
                    q = points[start:start+384]; n = len(q)
                    q = np.pad(q, ((0, 384-n), (0, 0)), mode='edge')
                    _, jac = fields(packed, jnp.asarray(q))
                    c = correction(q[:n,0], q[:n,1], np.full(n, family))
                    hessian = np.stack(c[2:], axis=-1).reshape(-1, 2, 2)
                    values[start:start+n] = np.linalg.det(np.asarray(jac)[:n]-hessian)
                return values
            seeds, normal = np.asarray(system['seeds']), np.asarray(system['normal'])
            targets = (previous[id(system)][draw]-seeds) @ normal
            curve = critical_crossings(determinant, seeds, normal, targets, .15, grid_size=17)
            missing = ~np.isfinite(curve).all(1)
            if missing.any():
                curve[missing] = critical_crossings(determinant,seeds[missing],normal,targets[missing],.55*float(system['fov']))
            system['critical_grf_draws'][draw] = curve
        if draw % 8 == 0 or draw == len(samples)-1:
            log(f'[outputs] GRF critical band: {draw+1}/{len(samples)} saved draws')
    for system in systems:
        system['grf_band_fingerprint'] = fingerprint
        system['grf_band_valid_draws'] = np.isfinite(system['critical_grf_draws']).all(2).sum(0)
        log(f"[outputs] GRF band {system['system_id']}: {int(system['grf_band_valid_draws'].min())}-{int(system['grf_band_valid_draws'].max())} valid draws per cross-section")
    data['grf_critical_band'] = ('Pointwise 16th-84th percentiles on the existing fixed normal cross-sections; '
        'GRF determinant sign boundary nearest each draw parametric branch; '
        'require at least 80% valid draws per cross-section; '
        'conditional-mean fields only, no residual latent-field realizations')
    return data


def add_grf_source_frames(state, evaluator, posterior, data, *, log=print):
    """Cartesian sources and the mapped best-draw GRF critical curve.

    The knee covariance uses piecewise-linear interpolation, so its Jacobian is
    discontinuous. Do not apply the smooth-lens exact caustic-root solver here.
    Instead map the displayed gridded critical curve through the same deflection
    field, retaining NaN separators between branches. All source draws use one
    fixed translation, not draw-dependent caustic alignment.
    """
    import hashlib
    systems = data.get('systems', [])
    indices = np.asarray(data['draw_indices'], int)
    samples = np.asarray(posterior.samples)[indices]
    fingerprint = hashlib.sha256(samples.tobytes() + indices.tobytes() + PLOT_FIELD_VERSION.encode()+b'grf-cartesian-source-v2').hexdigest()
    if not systems or all(s.get('grf_source_fingerprint') == fingerprint for s in systems):
        return data
    sources = np.asarray(jax.jit(jax.vmap(evaluator.source_positions))(jnp.asarray(samples)))
    best = int(np.nanargmax(posterior.log_prob))
    theta = posterior.samples[best]
    physical = evaluator.physical(theta)
    correction = plot_deflection_correction(evaluator, theta)
    if correction is None:
        raise ValueError('GRF source frame requires a conditional deflection field')
    fields, _, _ = _kernels(evaluator)
    family_ids = list(map(str, state.images.family_ids))
    for system in systems:
        # Remove stale products from the former exact-root attempt, if present.
        for key in ('relative_grf', 'grf_source_root_residual', 'grf_source_failures', 'grf_reference_tangent'):
            system.pop(key, None)
        selected = sources[:, [family_ids.index(str(fid)) for fid in system['family_ids']]]
        origin = selected[data['best_draw']].mean(0)
        system['source_origin_grf'] = origin
        system['source_cartesian_grf'] = selected-origin
        curve = np.asarray(system['critical_grf_best'])
        finite = np.isfinite(curve).all(1)
        points = curve[finite]
        caustic = np.full_like(curve, np.nan)
        mapped = np.empty_like(points)
        packed = evaluator.packed_state_for_z(physical, float(system['z_source']))
        family = family_ids.index(str(system['anchor_family_id']))
        for start in range(0, len(points), 384):
            q = points[start:start+384]; n = len(q)
            q = np.pad(q, ((0, 384-n), (0, 0)), mode='edge')
            beta, _ = fields(packed, jnp.asarray(q))
            c = correction(q[:,0], q[:,1], np.full(384, family))
            mapped[start:start+n] = np.asarray(beta)[:n]-np.column_stack(c[:2])[:n]
        caustic[finite] = mapped-origin
        system['caustic_grf_best'] = caustic
        system['grf_source_fingerprint'] = fingerprint
        log(f"[outputs] GRF Cartesian source panel: {system['system_id']}, {len(samples)} draws")
    data['grf_source_frame'] = ('Cartesian source coordinates minus one fixed best-draw mean source per system; '
        'best-draw caustic maps the displayed GRF critical curve through the conditional-mean GRF deflection; '
        'source contours contain all selected draws; no residual latent-field draws')
    return data


def curve_cross_section_anchor(curve, origin, image_tangent):
    """Interpolate the continuous local branch at a fixed image cross-section."""
    curve = np.asarray(curve)
    a, b = curve[:-1], curve[1:]
    t0, t1 = (a-origin) @ image_tangent, (b-origin) @ image_tangent
    valid = (np.isfinite(a).all(1) & np.isfinite(b).all(1)
             & (t0*t1 <= 0) & (abs(t1-t0) > 1e-12))
    if not valid.any():
        return None
    fraction = -t0[valid]/(t1[valid]-t0[valid])
    anchors = a[valid]+fraction[:, None]*(b[valid]-a[valid])
    return anchors[np.argmin(np.linalg.norm(anchors-origin, axis=1))]


def caustic_centered_sources(sources, anchors, basis):
    """One shared translation per draw and one fixed orthonormal display basis."""
    return (np.asarray(sources)-np.asarray(anchors)[:, None]) @ np.asarray(basis).T


def add_caustic_aligned_sources(state, evaluator, posterior, data, *, log=print):
    """Remove source/caustic common motion while preserving knot separations.

    Every draw uses its own conditional-mean GRF caustic anchor on the same
    image-plane cross-section. A fixed best-draw tangent/normal rotates all draws
    together. No knot-dependent shifts, contour smoothing, or sample rejection by
    source position is applied. Missing local anchors remain explicitly missing.
    """
    import hashlib
    systems = data.get('systems', [])
    if not systems:
        return data
    indices = np.asarray(data['draw_indices'], int)
    samples = np.asarray(posterior.samples)[indices]
    digest = hashlib.sha256(samples.tobytes()+PLOT_FIELD_VERSION.encode()+b'caustic-centered-sources-v2')
    for system in systems:
        digest.update(np.asarray(system['critical_grf_local_draws']).tobytes())
    fingerprint = digest.hexdigest()
    if all(s.get('source_alignment_fingerprint') == fingerprint for s in systems):
        return data
    physical_source = jax.jit(lambda t: (evaluator.physical(t), evaluator.source_positions(t)))
    fields, _, _ = _kernels(evaluator)
    family_ids = list(map(str, state.images.family_ids))
    for system in systems:
        system['source_anchor_grf'] = np.full((len(samples), 2), np.nan)
        system['image_anchor_grf'] = np.full((len(samples), 2), np.nan)
        system['source_unaligned_grf'] = np.full_like(system['relative'], np.nan)
        system['caustic_mapped_local_grf'] = np.full_like(system['critical_grf_local_draws'], np.nan)
    # Best draw first defines one common orientation, independent of the sources.
    order = [data['best_draw']]+[i for i in range(len(samples)) if i != data['best_draw']]
    for count, draw in enumerate(order):
        theta = samples[draw]
        physical, sources = physical_source(theta)
        correction = plot_deflection_correction(evaluator, theta)
        for system in systems:
            normal = np.asarray(system['normal'])
            tangent = np.array([normal[1], -normal[0]])
            curve = system['critical_grf_local_draws'][draw]
            anchor = curve_cross_section_anchor(curve, system['origin'], tangent)
            if anchor is None:
                continue
            finite = np.isfinite(curve).all(1)
            points = np.vstack([anchor, curve[finite]])
            packed = evaluator.packed_state_for_z(physical, float(system['z_source']))
            family = family_ids.index(str(system['anchor_family_id']))
            n = len(points)
            q = np.pad(points, ((0, 384-n), (0, 0)), mode='edge')
            beta, jac = map(np.asarray, fields(packed, jnp.asarray(q)))
            c = correction(q[:,0], q[:,1], np.full(384, family))
            mapped = beta-np.column_stack(c[:2])
            if draw == data['best_draw']:
                a = jac[0]-np.array([[c[2][0],c[3][0]],[c[4][0],c[5][0]]])
                source_tangent = np.linalg.svd(a)[0][:,0]
                # Fix the otherwise arbitrary SVD sign using the image tangent.
                if source_tangent @ (a @ tangent) < 0:
                    source_tangent = -source_tangent
                source_normal = np.array([-source_tangent[1],source_tangent[0]])
                system['source_alignment_basis'] = np.stack([source_tangent,source_normal])
            system['source_anchor_grf'][draw] = mapped[0]
            system['image_anchor_grf'][draw] = anchor
            system['caustic_mapped_local_grf'][draw, finite] = mapped[1:n]
            system['source_unaligned_grf'][draw] = np.asarray(sources)[[family_ids.index(str(fid)) for fid in system['family_ids']]]
        if count % 16 == 0 or count == len(samples)-1:
            log(f'[outputs] caustic-aligned sources: {count+1}/{len(samples)} draws')
    for system in systems:
        if 'source_alignment_basis' not in system:
            raise ValueError(f"No best-draw local caustic anchor for {system['system_id']}")
        basis = system['source_alignment_basis']
        system['source_aligned_grf'] = caustic_centered_sources(system['source_unaligned_grf'], system['source_anchor_grf'], basis)
        caustics = (system['caustic_mapped_local_grf']-system['source_anchor_grf'][:,None]) @ basis.T
        system['caustic_aligned_grf_best'] = caustics[data['best_draw']]
        system['source_alignment_fingerprint'] = fingerprint
        valid = np.isfinite(system['source_aligned_grf']).all(2).sum(0)
        before = np.nanstd(system['source_unaligned_grf'].mean(1),axis=0)
        after = np.nanstd(system['source_aligned_grf'].mean(1),axis=0)
        log(f"[outputs] aligned {system['system_id']}: valid={valid.tolist()}, scatter {before.tolist()} -> {after.tolist()} arcsec")
    data['source_alignment'] = ('Each draw source positions minus its own conditional-mean GRF caustic anchor at the '
        'fixed image-plane cross-section, rotated into one fixed best-draw tangent/normal basis; '
        'best caustic shows only the matching continuous local critical branch; '
        'all knots share the same shift; missing anchors counted explicitly')
    return data


def render_system_fold_diagnostics(config, state, data, paths, imaging, *, log=print):
    """One column per system/fold-image pair; stable knot colors in both planes."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Patch
    from .cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
    if not data['systems']:
        return None
    data['contour_definition'] = 'Whitened Gaussian KDE; leave-one-out bandwidth with one-standard-error smoothing, Scott to 2*Scott; highest-density regions enclosing 68.2689492% and 95.4499736%'
    images, bands, display, _ = imaging
    reference = images[rgb_reference_band(bands)]
    prefix = f'{config.paths.figure_prefix}_' if config.paths.figure_prefix else ''
    paths.figures.mkdir(parents=True, exist_ok=True)
    output = paths.figures / f'{prefix}fold_pairs_source_caustics{config.image_catalog.figure_suffix}.pdf'
    palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#F0E442', '#FFFFFF']
    best = data['best_draw']
    with plt.rc_context(fname=STYLE, rc={'savefig.bbox': None}):
        with PdfPages(output) as pdf:
            for start in range(0, len(data['systems']), 3):
                group = data['systems'][start:start+3]
                fig, axes = plt.subplots(2, len(group), figsize=(4.5*len(group), 8.5), squeeze=False)
                fig.subplots_adjust(left=.07, right=.985, bottom=.085, top=.98, wspace=.16, hspace=.07)
                for col, system in enumerate(group):
                    ax, panel = axes[:, col]
                    count = len(system['family_ids'])
                    colors = palette[:count] if count <= 8 else [plt.get_cmap('tab20')(i/count) for i in range(count)]
                    # White is useful on imaging but not on the source panel.
                    colors = ['#7B61A8' if color == '#FFFFFF' else color for color in colors]
                    coord = arcsec_to_skycoord(*system['center'], state.reference)
                    window = cutout_window(reference, coord, system['fov'])
                    if window is not None:
                        def pixels(x, y):
                            ra, dec = arcsec_to_radec(x, y, state.reference)
                            px, py = reference.wcs.world_to_pixel_values(ra, dec)
                            return px-window.x0, py-window.y0
                        rgb = rgb_cutout(images, bands, display, coord, system['fov'])
                        ax.imshow(rgb, origin='lower', interpolation='nearest')
                        sets = [(system['critical'], CRITICAL_COLOR, '--' if 'critical_grf_draws' in system else '-', .12)]
                        if 'critical_grf_draws' in system:
                            sets.append((system.get('critical_grf_local_draws', system['critical_grf_draws']), GRF_COLOR, '-', .22))
                        for curves, color, linestyle, alpha in sets:
                            for low, median, high in critical_curve_quantiles(
                                    curves, system['seeds'], system['origin'], system['normal']):
                                polygon = np.vstack((low, high[::-1]))
                                ax.add_patch(Polygon(np.column_stack(pixels(*polygon.T)), facecolor=color,
                                                     edgecolor='none', alpha=alpha, zorder=1.5))
                                ax.plot(*pixels(*median.T), color=color, ls=linestyle, lw=.7)
                        for j, label in enumerate(system['labels']):
                            k = system['image_family_index'][j]; color = colors[k]
                            _contours(ax, system['positions'][:, j], color, pixels)
                            from matplotlib import patheffects
                            cross, = ax.plot(*pixels(*system['observed'][j]), 'x', color=color, ms=11, mew=1.25, zorder=8)
                            cross.set_path_effects([patheffects.Stroke(linewidth=2.4,foreground=(0.,0.,0.,.9)),patheffects.Normal()])
                        size = rgb.shape[0]; bar = .5/reference.pixel_scale_arcsec
                        ax.plot([.07*size, .07*size+bar], [.08*size]*2, color='white', lw=1.2)
                        ax.text(.07*size+bar/2, .11*size, '0.5″', color='white', ha='center', fontsize=8)
                        ax.set_xlim(-.5, rgb.shape[1]-.5); ax.set_ylim(-.5, size-.5)
                    else:
                        ax.text(.5, .5, 'Outside imaging coverage', ha='center', transform=ax.transAxes)
                    ax.text(.03, .97, f"System {system['system_id']} • images {' / '.join(system['image_ids'])}\n{count} {'knot' if count == 1 else 'knots'} • z = {system['z_source']:g}",
                            transform=ax.transAxes, va='top', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=.65, edgecolor='none'))
                    knot_handles = [Line2D([], [], color=color, label=f'Knot {knot or "single"}') for color, knot in zip(colors, system['knot_ids'])]
                    image_key = [
                        Line2D([], [], color='white', marker='x', ls='none', ms=7, mew=1.25, label='Catalog knot')]
                    if 'critical_grf_best' in system:
                        image_key += [
                            Line2D([], [], color=CRITICAL_COLOR, lw=.7, ls='--', label='Parametric only'),
                            Line2D([], [], color=GRF_COLOR, lw=.7, label='Parametric + GRF')]
                    else:
                        image_key.append(Line2D([], [], color=CRITICAL_COLOR, lw=.7, label='Median critical curve'))
                    ax.legend(handles=knot_handles+image_key, loc='lower right', ncol=2, fontsize=6,
                              handlelength=1.5, columnspacing=.8, labelspacing=.35,
                              frameon=True, facecolor='black', edgecolor='none', framealpha=.75, labelcolor='white')
                    has_grf_source = 'source_cartesian_grf' in system
                    aligned = 'source_aligned_grf' in system
                    points = system.get('source_aligned_grf', system.get('source_cartesian_grf', system['relative']))
                    finite = np.isfinite(points).all(2)
                    all_points = points[finite]
                    center = np.median(all_points, axis=0) if len(all_points) else np.zeros(2)
                    radii = np.maximum(.002, 1.3*np.max(abs(all_points-center), axis=0)) if len(all_points) else np.ones(2)*.1
                    if has_grf_source:
                        panel.plot(*system.get('caustic_aligned_grf_best',system['caustic_grf_best']).T, color=GRF_COLOR, lw=.8)
                    else:
                        panel.axhline(0, color=MAGENTA, lw=1.2)
                        radii[1] = max(radii[1], abs(center[1])+.002)
                    counts = []; contour_bounds = []
                    for k, color in enumerate(colors):
                        _contours(panel, points[:, k], color, bounds=contour_bounds, linewidths=[.55,.8])
                        if finite[best, k]:
                            panel.plot(*points[best, k], 'o', color=color, ms=4)
                        counts.append(f"{system['knot_ids'][k] or 'single'}: {int((points[:, k, 1]<0).sum())}/{int(finite[:, k].sum())}")
                    if contour_bounds:
                        extent = np.vstack(contour_bounds)
                        radii = np.maximum(radii, 1.08*np.max(abs(extent-center), axis=0))
                    panel.set(xlim=(center[0]-radii[0], center[0]+radii[0]),
                              ylim=(center[1]-2*radii[1], center[1]+1.3*radii[1]))
                    if aligned:
                        # Fit the complete credible regions and nearby local caustic,
                        # with independent axis ranges and a lower area for the key.
                        extent = np.vstack(contour_bounds) if contour_bounds else all_points
                        caustic = system['caustic_aligned_grf_best']
                        caustic = caustic[np.isfinite(caustic).all(1)]
                        nearby = caustic[(caustic[:,0]>=extent[:,0].min()) & (caustic[:,0]<=extent[:,0].max())]
                        if len(nearby):
                            extent = np.vstack([extent,nearby])
                        elif len(caustic):
                            extent = np.vstack([extent,caustic[np.argmin(np.linalg.norm(caustic-center,axis=1))]])
                        lo, hi = extent.min(0), extent.max(0)
                        span = max(1.18*(hi[0]-lo[0]), 1.8*(hi[1]-lo[1]), .01)
                        mid_x = .5*(lo[0]+hi[0])
                        yspan = max(1.8*(hi[1]-lo[1]), .01)
                        bottom = lo[1]-.34*yspan
                        panel.set(xlim=(mid_x-span/2,mid_x+span/2), ylim=(bottom,bottom+yspan))
                    if not has_grf_source:
                        panel.axhspan(center[1]-2*radii[1], 0, color='#F4DCDC', alpha=.65, zorder=0)
                    panel.text(.03, .97, 'Caustic-aligned source plane' if aligned else 'Sources and best-fit GRF caustic' if has_grf_source else 'Source positions, parametric caustic frame',
                               transform=panel.transAxes, va='top', fontsize=7)
                    source_key = [
                        Line2D([], [], color='.2', marker='o', ls='none', ms=3, label='Best-model source'),
                        Line2D([], [], color=GRF_COLOR if has_grf_source else MAGENTA, lw=.7,
                               label='Parametric + GRF caustic' if has_grf_source else 'Parametric caustic')]
                    if not has_grf_source:
                        source_key.append(Patch(facecolor='#F4DCDC', edgecolor='none', label='No local fold pair'))
                    panel.legend(handles=knot_handles+source_key, loc='lower left', bbox_to_anchor=(.015, .12),
                                 ncol=3 if count>2 else 2, fontsize=5.8, handlelength=1.5,
                                 columnspacing=.7, labelspacing=.35, frameon=True,
                                 facecolor='white', edgecolor='none', framealpha=.9)
                    footer = ((f'{int(finite.all(1).sum())}/{len(points)} draws with a local caustic anchor' if aligned else f'{len(points)} source draws; caustic at best fit\nCommon fixed source-plane origin')
                              if has_grf_source else 'No local fold pair / evaluated draws\n'+'; '.join(counts)+'\nSource draws retained when image recovery fails')
                    panel.text(.03, .04, footer, transform=panel.transAxes, fontsize=6.5, va='bottom', wrap=True)
                    panel.set_xlabel(r'$\Delta\beta_t$ (arcsec)' if aligned else r'$\Delta\beta_x$ (arcsec)' if has_grf_source else 'Along-caustic coordinate (arcsec)', fontsize=8)
                    if col == 0:
                        panel.set_ylabel(r'$\Delta\beta_n$ (arcsec)' if aligned else r'$\Delta\beta_y$ (arcsec)' if has_grf_source else 'Signed nearest-caustic distance (arcsec)', fontsize=8)
                    panel.tick_params(labelsize=7)
                    panel.set_aspect('auto' if aligned else 'equal' if has_grf_source else 'auto', adjustable='box' if aligned else 'datalim')
                    ax.set_xticks([]); ax.set_yticks([])
                    for target in (ax, panel):
                        target.grid(False)
                pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)
    log(f'[outputs] system-matched fold cutouts -> {output}')
    return output


def render_fold_diagnostics(config, state, data, paths, imaging, *, log=print):
    data['central_curve_statistic'] = CRITICAL_CURVE_STATISTIC
    for system in data.get('systems', []):
        system['central_curve_statistic'] = CRITICAL_CURVE_STATISTIC
    if 'systems' in data:
        return render_system_fold_diagnostics(config, state, data, paths, imaging, log=log)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from .cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
    if not data['pairs']:
        return None
    images, bands, display, _ = imaging; reference = images[rgb_reference_band(bands)]
    paths.figures.mkdir(parents=True, exist_ok=True)
    prefix = f'{config.paths.figure_prefix}_' if config.paths.figure_prefix else ''
    output = paths.figures / f'{prefix}fold_pairs_source_caustics{config.image_catalog.figure_suffix}.pdf'
    best = data['best_draw']
    with plt.rc_context(fname=STYLE, rc={'savefig.bbox': None}):
        with PdfPages(output) as pdf:
            for start in range(0, len(data['pairs']), 3):
                pairs = data['pairs'][start:start+3]
                fig, axes = plt.subplots(2, len(pairs), figsize=(4*len(pairs), 8), squeeze=False)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                for col, pair in enumerate(pairs):
                    ax, panel = axes[:, col]; mid = pair['observed'].mean(0)
                    coord = arcsec_to_skycoord(*mid, state.reference)
                    window = cutout_window(reference, coord, pair['fov'])
                    def pixels(x, y):
                        ra, dec = arcsec_to_radec(x, y, state.reference)
                        px, py = reference.wcs.world_to_pixel_values(ra, dec)
                        return px-window.x0, py-window.y0
                    if window is not None:
                        rgb = rgb_cutout(images, bands, display, coord, pair['fov'])
                        ax.imshow(rgb, origin='lower', interpolation='nearest')
                        for low, median, high in critical_curve_quantiles(
                                pair['critical'], pair['seeds'], pair['origin'], pair['normal']):
                            polygon = np.vstack((low, high[::-1]))
                            ax.add_patch(Polygon(np.column_stack(pixels(*polygon.T)), facecolor=CRITICAL_COLOR,
                                                 edgecolor='none', alpha=.25))
                            ax.plot(*pixels(*median.T), color=CRITICAL_COLOR, lw=1.1)
                        handles = []
                        for k, color in enumerate((BLUE, ORANGE)):
                            _contours(ax, pair['positions'][:, k], color, pixels)
                            ax.plot(*pixels(*pair['observed'][k]), 'x', color='white', ms=5, mew=.8)
                            ax.plot(*pixels(*pair['positions'][best, k]), 'o', mfc='none', mec='#7FFF70', ms=7, mew=.9)
                            handles.append(Line2D([], [], color=color, label=pair['labels'][k]+' positions'))
                        handles += [Line2D([], [], color='white', marker='x', ls='none', label='Catalog'),
                                    Line2D([], [], mec='#7FFF70', mfc='none', marker='o', ls='none', label='Best model')]
                        legend = ax.legend(handles=handles, title='Solid: 68.3%; dashed: 95.4%', title_fontsize=7,
                                           loc='lower right', fontsize=7, facecolor='black', edgecolor='none', framealpha=.65, labelcolor='white')
                        legend.get_title().set_color('white')
                        size = rgb.shape[0]; bar = .5/reference.pixel_scale_arcsec
                        ax.plot([.08*size, .08*size+bar], [.08*size]*2, color='white', lw=1.2)
                        ax.text(.08*size+bar/2, .11*size, '0.5″', color='white', ha='center', fontsize=9)
                        ax.set_xlim(-.5, rgb.shape[1]-.5); ax.set_ylim(-.5, size-.5)
                    else:
                        ax.text(.5, .5, 'Outside imaging coverage', ha='center', transform=ax.transAxes)
                    ax.text(.04, .97, f"Family {pair['family_id']} • z = {pair['z_source']:g}\nSeparation {pair['separation']:.3f}″ • model-identified fold",
                            transform=ax.transAxes, va='top', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=.55, edgecolor='none'))
                    points = pair['relative']; finite = np.isfinite(points).all(1)
                    plotted = _contours(panel, points, BLUE)
                    center = points[best] if finite[best] else np.nanmedian(points, axis=0)
                    if not finite.any():
                        center = np.zeros(2)
                    offset = points[finite]-np.array([center[0], 0.])
                    radius = max(.02, 1.25*np.max(np.abs(offset)+2*np.std(offset, axis=0))) if finite.any() else .1
                    vertical_radius = max(.002, 1.4*np.max(np.abs(points[finite, 1])+2*np.std(points[finite, 1]))) if finite.any() else .1
                    panel.set_xlim(center[0]-radius, center[0]+radius); panel.set_ylim(-vertical_radius, vertical_radius)
                    panel.set_aspect('auto'); panel.axhspan(-vertical_radius, 0, color='#F4DCDC', alpha=.65, zorder=0)
                    panel.axhline(0, color=MAGENTA, lw=1.6)
                    if finite[best]:
                        panel.plot(*center, 'o', color='black', ms=4)
                        panel.plot(center[0], 0, 'x', color=MAGENTA, ms=5)
                        panel.text(.04, .97, f'Best-model source–caustic distance: {center[1]*1000:.2f} mas', transform=panel.transAxes, va='top', fontsize=9)
                        inset = panel.inset_axes([.65, .72, .3, .2]); zoom=max(.006, 1.5*abs(center[1]))
                        inset.set_xlim(-zoom/2, zoom/2); inset.set_ylim(min(-.2*zoom, center[1]-.2*zoom), max(.8*zoom, center[1]+.2*zoom))
                        inset.axhspan(inset.get_ylim()[0], 0, color='#F4DCDC', alpha=.65); inset.axhline(0, color=MAGENTA)
                        inset.plot(0, center[1], 'o', color='black', ms=4); inset.plot(0, 0, 'x', color=MAGENTA)
                        inset.annotate('', (0, center[1]), (0, 0), arrowprops=dict(arrowstyle='<->', lw=.7))
                        inset.set_xticks([]); inset.set_yticks([]); inset.set_aspect('equal')
                    panel.text(.5, .30, f'No local fold pair: {np.sum(points[:, 1]<0):,} / {finite.sum():,} draws\n(source on this side of its own caustic)', transform=panel.transAxes, ha='center', color='#8A3535', fontsize=9)
                    panel.legend(handles=[Line2D([], [], color=BLUE, label='Source positions: 1σ (68.3%)'),
                                          Line2D([], [], color=BLUE, ls='--', label='Source positions: 2σ (95.4%)'),
                                          Line2D([], [], color='black', marker='o', ls='none', label='Source of best model')],
                                 loc='lower left', bbox_to_anchor=(.06, .13), fontsize=7, frameon=False)
                    bar = .02 if radius<.2 else .1
                    x, y = center[0]-.84*radius, -.82*vertical_radius
                    panel.plot([x, x+bar], [y, y], color='black', lw=1.2)
                    panel.text(x+bar/2, y+.04*vertical_radius, f'{bar*1000:g} mas', ha='center', fontsize=8)
                    vbar = 10**np.floor(np.log10(.2*vertical_radius))
                    vbar *= max(v for v in (1, 2, 5) if v*vbar <= .2*vertical_radius)
                    vx = center[0]+.78*radius
                    panel.plot([vx, vx], [y, y+vbar], color='black', lw=1.2)
                    panel.text(vx-.03*radius, y+vbar/2, f'{vbar*1000:g} mas', ha='right', va='center', fontsize=8)
                    panel.text(.04, .91, 'Vertical scale enlarged', transform=panel.transAxes, fontsize=7, color='.35')
                    note = f"{finite.sum():,}/{len(points):,} source draws evaluated"
                    if not plotted:
                        note += '\nInsufficient spread/draws for density contours'
                    elif not finite.all():
                        note += '\nCaustic failures recorded; no image-recovery filtering'
                    else:
                        note += '\nIncludes draws with unrecovered images'
                    panel.text(.5, .025, note, transform=panel.transAxes, ha='center', fontsize=6.8)
                    for target in (ax, panel):
                        target.set_xticks([]); target.set_yticks([]); target.grid(False)
                pdf.savefig(fig, bbox_inches=None, pad_inches=0); plt.close(fig)
    log(f'[outputs] fold-pair cutouts -> {output}')
    return output
