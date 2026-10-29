"""Numerical data for member, residual, and approximation diagnostic figures."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import qmc

from . import smoothed_likelihood as sl
from .model import DP_IE_PROFILE, NFW_PROFILE, latent_array_to_physical, member_log_deviations
from .prior_diagnostics import saved_prior


@dataclass
class MemberDiagnostics:
    members: pd.DataFrame
    variables: dict[str, str]
    deviations: np.ndarray                 # draw, member, (ln sigma, ln radius)
    prior_deviations: np.ndarray
    fields: dict[str, np.ndarray]
    images: pd.DataFrame
    provenance: dict


def diagnostic_draw_indices(log_prob, count):
    """Best first, followed by at most count evenly spaced other joint draws."""
    log_prob = np.asarray(log_prob)
    if not np.isfinite(log_prob).any():
        raise ValueError("No finite posterior draw for member diagnostics")
    best = int(np.nanargmax(np.where(np.isfinite(log_prob), log_prob, -np.inf)))
    others = np.flatnonzero(np.isfinite(log_prob) & (np.arange(len(log_prob)) != best))
    take = min(count, len(others))
    selected = others[np.linspace(0, len(others)-1, take, dtype=int)] if take else []
    return np.asarray([best, *selected], dtype=int)


def deviation_draws(state, physical):
    """Use the model's physical deviation mapping, including centering and mass modes."""
    components = jnp.asarray(state.members.component_indices)
    def evaluate(row):
        v, r = member_log_deviations(state.packed_lens_spec, row)
        return jnp.stack((v[components], r[components]), -1)
    return np.asarray(jax.jit(jax.vmap(evaluate))(jnp.asarray(physical)))


def prior_deviation_draws(state, *, power=12):
    """Deterministic hierarchical prior predictive draws in physical offset units."""
    specs = state.parameter_specs
    uniform = np.clip(qmc.Sobol(len(specs), scramble=True, seed=173).random_base2(power), 1e-12, 1-1e-12)
    physical = np.empty_like(uniform)
    for i, spec in enumerate(specs):
        physical[:, i] = latent_array_to_physical(saved_prior(spec).ppf(uniform[:, i]), spec)
    packed = state.packed_lens_spec
    centered = np.asarray(packed.deviation_centered, bool)
    handled = set()
    for offsets, scatters in ((packed.velocity_offset_param_index, packed.velocity_scatter_param_index),
                              (packed.radius_offset_param_index, packed.radius_scatter_param_index)):
        for c in state.members.component_indices:
            i, s = int(offsets[c]), int(scatters[c])
            if i >= 0 and s >= 0 and centered[c] and i not in handled:
                physical[:, i] *= physical[:, s]
                handled.add(i)
    return deviation_draws(state, physical)


def neighbor_variables(xy):
    """Projected neighbors excluding self; coincident distinct galaxies remain neighbors."""
    xy = np.asarray(xy)
    n = len(xy)
    pair = distance.cdist(xy, xy)
    np.fill_diagonal(pair, np.inf)
    nearest = pair.min(axis=1) if n > 1 else np.full(n, np.nan)
    fifth = np.sort(pair, axis=1)[:, 4] if n > 5 else np.full(n, np.nan)
    density = np.divide(5., np.pi*fifth**2, out=np.full(n, np.nan), where=(fifth > 0) & np.isfinite(fifth))
    return nearest, density


def population_predictive_draws(state, physical):
    """Fresh unit offsets conditional on the joint posterior population parameters."""
    physical = np.asarray(physical).copy()
    packed = state.packed_lens_spec
    rng = np.random.default_rng(174)
    handled = set()
    for offsets, scatters in ((packed.velocity_offset_param_index, packed.velocity_scatter_param_index),
                              (packed.radius_offset_param_index, packed.radius_scatter_param_index)):
        for c in state.members.component_indices:
            i, s = int(offsets[c]), int(scatters[c])
            if i >= 0 and i not in handled:
                physical[:, i] = rng.normal(size=len(physical))
                if packed.deviation_centered[c] and s >= 0:
                    physical[:, i] *= physical[:, s]
                handled.add(i)
    return deviation_draws(state, physical)


def whiten(residual, covariance):
    """Whiten vector columns with C = L L^T, preserving their squared Mahalanobis norm."""
    return np.linalg.solve(np.linalg.cholesky(covariance), np.asarray(residual)[..., None])[..., 0]


def reporting_fields(evaluator, theta):
    """Exact diagnostic fields with the physical sources of the sampled approximation."""
    state, images = evaluator.state, evaluator.state.images
    physical = evaluator.physical(theta)
    sampled = evaluator.fields(physical, derivative=False)
    exact = evaluator.fields(physical, force_exact=True, derivative=False)
    sigma_def = physical[state.sigma_def_param_index]
    extra = physical[state.sigma_extra_param_index] if state.sigma_extra_param_index >= 0 else 0.
    sel = physical[state.sigma_sel_param_index] if state.sigma_sel_param_index >= 0 else 0.
    lam = physical[state.lambda_sel_param_index] if state.lambda_sel_param_index >= 0 else 1.
    meas = jnp.sqrt(images.sigma_inst_per_image**2 + extra**2)
    covariance = sl.source_covariance(sampled['jacobian'], meas, sigma_def, sel, lam)['covariance']
    whitening = sl.source_whitening(sampled['backprojected'], covariance, images.family_index_per_image, images.n_families)
    coordinates = jnp.stack((theta[images.source_x_param_index], theta[images.source_y_param_index]), -1)
    sources = sl.transport_sources(coordinates, whitening, jnp.asarray(state.members.source_prior_mean), state.members.source_prior_sigma)['source']
    exact_cov = sl.source_covariance(exact['jacobian'], meas, sigma_def, sel, lam)['covariance']
    solve = lambda v, cov: jnp.linalg.solve(jnp.linalg.cholesky(cov), v[..., None])[..., 0]
    residual = exact['backprojected'] - sources[images.family_index_per_image]
    difference = sampled['backprojected'] - exact['backprojected']
    member_difference = sampled['member_alpha'] - exact['member_alpha']
    member_white = solve(member_difference, covariance[None])
    u = exact['member_jacobian']
    shear = jnp.hypot(.5*(u[..., 0, 0]-u[..., 1, 1]), .5*(u[..., 0, 1]+u[..., 1, 0]))
    return dict(residual=solve(residual, exact_cov), source=sources, exact_backprojected=exact['backprojected'],
                exact_covariance=exact_cov, sampled_covariance=covariance,
                error_arcsec=jnp.linalg.norm(difference, axis=-1),
                error_whitened=jnp.linalg.norm(solve(difference, covariance), axis=-1),
                member_error_arcsec=jnp.max(jnp.linalg.norm(member_difference, axis=-1), axis=-1),
                member_error_whitened=jnp.max(jnp.linalg.norm(member_white, axis=-1), axis=-1),
                deflection=jnp.max(jnp.linalg.norm(exact['member_alpha'], axis=-1), axis=-1),
                shear=jnp.max(shear, axis=-1),
                cut_radius=exact['packed'].Rs[jnp.asarray(evaluator.member_components)])


def compute_member_diagnostics(state, evaluator, posterior, config, *, log=print):
    """Build one shared dataset; no inference or mutation of the member classes."""
    records = state.scaling_component_records
    ids = list(state.members.member_ids)
    by_component = {int(r['component_index']): r for r in records}
    if len(set(ids)) != len(ids) or len(by_component) != len(records):
        raise ValueError('Member diagnostics require unique stable identities')
    table = pd.DataFrame([by_component[int(c)] for c in state.members.component_indices])
    table.insert(0, 'member_id', ids)
    table.loc[table.ellipticite <= 1e-10, 'angle_pos'] = np.nan
    table['member_class'] = np.where(np.isin(np.arange(len(ids)), state.members.exact_positions), 'exact', 'far-field')
    count = config.image_diagnostics.posterior_diagnostic_draws
    indices = diagnostic_draw_indices(posterior.log_prob, count)
    samples = np.asarray(posterior.samples)[indices]
    log(f'[diagnostics] fields at best draw {indices[0]} plus {len(indices)-1} joint draws')
    batch = jax.jit(jax.vmap(lambda t: reporting_fields(evaluator, t)))
    blocks = []
    for start in range(0, len(samples), 4):
        chunk = samples[start:start+4]
        padded = np.pad(chunk, ((0, 4-len(chunk)), (0, 0)), mode='edge')
        blocks.append(jax.tree.map(lambda a: np.asarray(a)[:len(chunk)], batch(jnp.asarray(padded))))
    fields = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    variables = {'catalog_mag': 'Catalog magnitude'}
    xy = table[['x_centre', 'y_centre']].to_numpy(float)
    image_xy = np.column_stack((state.images.x_obs, state.images.y_obs))
    table['nearest_image_arcsec'] = distance.cdist(xy, image_xy).min(axis=1)
    variables['nearest_image_arcsec'] = 'Nearest constrained image (arcsec)'
    table['nearest_member_arcsec'], table['neighbor_density_arcsec2'] = neighbor_variables(xy)
    variables.update(nearest_member_arcsec='Nearest member (arcsec)', neighbor_density_arcsec2=r'$\Sigma_5$ (arcsec$^{-2}$)',
                     ellipticite='Catalog ellipticity', angle_pos='Catalog position angle (deg)',
                     deflection='Maximum member deflection (arcsec)', shear='Maximum member shear')
    table['deflection'], table['shear'] = fields['deflection'][0], fields['shear'][0]
    best_physical = evaluator.physical(jnp.asarray(samples[0]))
    z = float(config.truth.caustic_source_redshift)
    skipped = {}
    if z <= state.z_lens:
        skipped['environment_kappa'] = 'Diagnostic source redshift is not behind lens'
    else:
        packed = evaluator.packed_state_for_z(best_physical, z)
        components = np.arange(evaluator.n_components, dtype=np.int32)
        # Evaluate rows at member centers in bounded point blocks, without third derivatives.
        row_fields = jax.jit(lambda points: evaluator.component_rows(points, packed, components, derivative=False)[1])
        kappas = []
        for start in range(0, len(xy), 64):
            points = xy[start:start+64]
            padded = np.pad(points, ((0, 64-len(points)), (0, 0)), mode='edge')
            u = np.asarray(row_fields(jnp.asarray(padded)))[:, :len(points)]
            k = .5*np.trace(u, axis1=-2, axis2=-1)
            own = np.asarray(state.members.component_indices)[start:start+len(points)]
            kappas.extend(k.sum(axis=0)-k[own, np.arange(len(points))])
        table['environment_kappa'] = kappas
        variables['environment_kappa'] = rf'Environmental $\kappa$ ($z_s={z:g}$; self excluded)'
    packed = evaluator.packed_state(best_physical)[0]
    profile = np.asarray(state.packed_lens_spec.profile_type)
    family = np.asarray(state.packed_lens_spec.component_family)
    halos = np.flatnonzero((family == 0) & np.isin(profile, [DP_IE_PROFILE, NFW_PROFILE]))
    if len(halos):
        centers = np.column_stack((np.asarray(packed.center_x)[halos], np.asarray(packed.center_y)[halos]))
        separation = distance.cdist(xy, centers)
        for col, c in enumerate(halos):
            name = str(state.base_components[c]['id'])
            key = f'halo_{name}_distance_arcsec'
            table[key] = separation[:, col]
            variables[key] = f'Distance to halo {name} (arcsec)'
        nearest = separation.argmin(axis=1)
        table['nearest_halo_index'] = nearest
        table['nearest_halo_id'] = [str(state.base_components[halos[i]]['id']) for i in nearest]
        table['nearest_halo_arcsec'] = separation[np.arange(len(xy)), nearest]
        delta = xy-centers[nearest]
        e1, e2 = np.asarray(packed.e1)[halos][nearest], np.asarray(packed.e2)[halos][nearest]
        phi = .5*np.arctan2(e2, e1)
        azimuth = np.abs((np.arctan2(delta[:, 1], delta[:, 0])-phi+np.pi/2) % np.pi-np.pi/2)
        defined = (np.hypot(e1, e2) > 1e-10) & (np.linalg.norm(delta, axis=1) > 1e-10)
        table['halo_azimuth_deg'] = np.where(defined, np.degrees(azimuth), np.nan)
        variables.update(nearest_halo_index='Nearest model halo', nearest_halo_arcsec='Nearest model halo (arcsec)',
                         halo_azimuth_deg='Angle from nearest halo major axis (deg)')
    else:
        skipped['halo_geometry'] = 'No large-scale dPIE or NFW halo'
    for key in list(variables):
        if key not in table or not np.isfinite(table[key].to_numpy(float)).any():
            skipped[key] = 'Insufficient members or undefined geometry'
            variables.pop(key)
    deviations = deviation_draws(state, posterior.physical)
    priors = prior_deviation_draws(state)
    population = population_predictive_draws(state, posterior.physical)
    table['radius_deviation_fixed'] = np.ptp(priors[..., 1], axis=0) == 0
    for j, name in enumerate(('velocity', 'radius')):
        q = np.quantile(deviations[..., j], [.16, .5, .84], axis=0)
        pq = np.quantile(priors[..., j], [.16, .5, .84], axis=0)
        for label, values in zip(('p16', 'median', 'p84'), q):
            table[f'{name}_{label}'] = values
        table[f'{name}_prior_p16'], table[f'{name}_prior_p84'] = pq[0], pq[2]
        table[f'{name}_population_p16'], table[f'{name}_population_p84'] = np.quantile(population[..., j], [.16, .84], axis=0)
        table[f'{name}_width_ratio'] = np.divide(q[2]-q[0], pq[2]-pq[0], out=np.full(len(ids), np.nan), where=pq[2] > pq[0])
    v, r = deviations[..., 0], deviations[..., 1]
    for label, values in zip(('p16', 'median', 'p84'), np.quantile(2*v+r, [.16, .5, .84], axis=0)):
        table[f'mass_deviation_{label}'] = values
    vs, rs = v.std(axis=0), r.std(axis=0)
    table['velocity_radius_correlation'] = np.divide(np.mean((v-v.mean(axis=0))*(r-r.mean(axis=0)), axis=0), vs*rs,
                                                    out=np.full(len(ids), np.nan), where=vs*rs > 0)
    for key in ('member_error_arcsec', 'member_error_whitened'):
        q = np.quantile(fields[key], [.16, .5, .84], axis=0)
        for label, values in zip(('p16', 'median', 'p84'), q):
            table[f'{key}_{label}'] = values
        table[f'{key}_best'] = fields[key][0]
        table[f'{key}_max'] = fields[key].max(axis=0)
    separation = distance.cdist(xy, image_xy).min(axis=1)
    ratio = separation[None]/fields['cut_radius']
    table['guard_ratio_min'] = ratio.min(axis=0)
    table['guard_ratio_best'] = ratio[0]
    images = pd.DataFrame(dict(image_label=state.images.image_labels, family_id=np.asarray(state.images.family_ids)[np.asarray(state.images.family_index_per_image)],
                               x_obs_arcsec=image_xy[:, 0], y_obs_arcsec=image_xy[:, 1]))
    provenance = dict(selected_draw_indices=indices.tolist(), best_draw_index=int(indices[0]), variable_definitions=variables,
                      environment_source_redshift=z, skipped_variables=skipped, prior_predictive_draws=len(priors), prior_seed=173,
                      tolerance=float(config.members.template_tolerance), guard_cut_radii=float(evaluator.guard_cut_radii),
                      source_definition='sampled-model physical source, held fixed for exact reporting',
                      algorithm_revision='member-diagnostics-v1',
                      reference_geometry='best-fit model; deflection/shear at individual image source redshifts',
                      neighbor_catalog='modeled population members; independent halos and dropped members excluded',
                      excluded_variable_rows={key: int((~np.isfinite(table[key].to_numpy(float))).sum()) for key in variables},
                      halo_angle_definition='absolute angle modulo 180 degrees, folded to 0–90 degrees',
                      prior_width_definition='marginal hierarchical prior predictive 16–84% width; no lens-validity conditioning')
    return MemberDiagnostics(table, variables, deviations, priors, fields, images, provenance)
