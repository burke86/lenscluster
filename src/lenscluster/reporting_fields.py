"""Exact parametric fields plus the draw-conditioned shared GRF for reporting.

This does not alter the likelihood covariance. A family-specific GP cannot define
one cluster-wide mass map, so only shared mass fields enter these maps.
"""
import numpy as np

from . import jax_cosmology


def shared_grf_fields(evaluator, theta, z_source, xy, *, chunk=1024):
    """Return deflection and its Jacobian, scaled to the requested redshift.

    The conditional mean already includes family zero's lens efficiency. Remove
    that weight before applying the target efficiency, exactly once.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    alpha = np.zeros_like(xy)
    jacobian = np.zeros((len(xy), 2, 2))
    if not (getattr(evaluator, 'gp_knee', False)
            and evaluator.state.gp_blocks == 'all'):
        return alpha, jacobian
    correction = evaluator.gp_deflection_correction(theta)
    if correction is None:
        return alpha, jacobian
    state = evaluator.state
    reference = float(jax_cosmology.dpie_sigma0_factor_from_config(
        state.z_lens, state.gp_reference_redshift, state.cosmo_config))
    target = float(jax_cosmology.dpie_sigma0_factor_from_config(state.z_lens, z_source, state.cosmo_config))
    weight = target / reference / float(evaluator.gp_weights_per_family[0])
    for start in range(0, len(xy), chunk):
        points = xy[start:start + chunk]
        count = len(points)
        points = np.pad(points, ((0, chunk-count), (0, 0)), mode='edge')
        fields = correction(points[:, 0], points[:, 1], np.zeros(chunk, dtype=int))
        alpha[start:start+count] = np.stack(fields[:2], axis=-1)[:count] * weight
        jacobian[start:start+count] = np.stack(fields[2:], axis=-1)[:count].reshape(-1, 2, 2) * weight
    if not (np.isfinite(alpha).all() and np.isfinite(jacobian).all()):
        raise ValueError('Nonfinite conditional GRF reporting fields')
    return alpha, jacobian


def posterior_median_theta(evaluator, posterior):
    """Coordinatewise physical posterior median, including physical sources.

    Whitened source coordinates are model dependent: their median is not the
    coordinate representation of the median physical source positions.
    """
    import jax
    from .model import convert_theta_to_latent
    physical = np.median(np.asarray(posterior.physical), axis=0)
    if not getattr(evaluator, 'gp_knee', False):
        theta = evaluator.latent_from_reported_physical(physical)
    else:
        e = evaluator._padded_evaluator()
        theta = convert_theta_to_latent(physical, e.state.parameter_specs)
        images = e.state.images
        ix, iy = np.asarray(images.source_x_param_index), np.asarray(images.source_y_param_index)
        theta[np.r_[ix, iy]] = 0.
        whitening = jax.jit(lambda t: e.statistics(t)['whitening'])(theta)
        layout = e.gp
        mask, family = np.asarray(layout['knot_mask']), np.asarray(layout['knot_family'])
        sources = np.stack((physical[ix], physical[iy]), axis=-1)
        target = np.where(mask[..., None], sources[np.where(mask, family, 0)], 0.).reshape(len(mask), -1)
        offset = np.linalg.solve(np.asarray(whitening['cholesky']),
                                (target - np.asarray(whitening['centroid']))[..., None])[..., 0].reshape(*mask.shape, 2)
        real = offset[np.asarray(layout['real_block']), np.asarray(layout['real_knot'])]
        families = np.asarray(layout['real_family'])
        theta[ix[families]], theta[iy[families]] = real[:, 0], real[:, 1]
    np.testing.assert_allclose(jax.jit(evaluator.reported_physical)(theta), physical, rtol=1e-10, atol=1e-10)
    return np.asarray(theta)
