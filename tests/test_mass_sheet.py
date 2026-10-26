"""External convergence: per-family scaling, lens fields, gradients and the exact-image solver kwargs."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from lenscluster.config import MassSheetConfig, PriorConfig
from lenscluster.evaluator import Evaluator
from lenscluster.exact_images import packed_to_kwargs_lens
from lenscluster.jax_cosmology import flat_wcdm_lensing_efficiency
from lenscluster.state import build_state


def test_mass_sheet_scales_with_efficiency_and_matches_lenstronomy(synthetic_config):
    sheet = MassSheetConfig(id="K", kappa=0.04, z_lens=0.3, priors={"kappa": PriorConfig("truncated_normal", mean=0.0, std=0.05, lower=-0.2, upper=0.5)})
    config = replace(synthetic_config, model=replace(synthetic_config.model, large_halos=synthetic_config.model.large_halos + (sheet,))).validate()
    state = build_state(config)
    names = [s.name for s in state.parameter_specs]
    assert "K.kappa" in names
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta).copy()
    theta[names.index("K.kappa")] = 0.04
    theta = jnp.asarray(theta)
    physical = evaluator.physical(theta)
    packed = evaluator.packed_state(physical)[0]
    sheet_row = int(np.flatnonzero(np.asarray(state.packed_lens_spec.profile_type) == 101)[0])
    cosmo = state.cosmo_config
    for i, z_index in enumerate(np.asarray(state.images.z_index_per_image)):
        z_source = float(state.z_source_values[int(z_index)])
        efficiency = float(flat_wcdm_lensing_efficiency(state.z_lens, z_source, cosmo["H0"], cosmo["Om0"], cosmo.get("w0", -1.0)))
        np.testing.assert_allclose(float(packed.kappa[sheet_row, i]), 0.04 * efficiency, rtol=1e-10)
    # Fields: alpha = kappa theta, U = kappa I, dU = 0.
    alpha, u, du = evaluator.component_rows(evaluator.xy, packed, np.asarray([sheet_row], dtype=np.int32), derivative=True)
    kappa = np.asarray(packed.kappa[sheet_row])
    np.testing.assert_allclose(np.asarray(alpha[0]), kappa[:, None] * np.asarray(evaluator.xy), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(u[0]), kappa[:, None, None] * np.eye(2), rtol=1e-12)
    assert np.all(np.asarray(du[0]) == 0.0)
    value, grad = evaluator.value_and_grad(theta)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad))) and float(grad[names.index("K.kappa")]) != 0.0
    # Exact-image solver path through lenstronomy's CONVERGENCE model.
    from lenstronomy.LensModel.lens_model import LensModel
    z_source = float(state.z_source_values[0])
    packed_z = evaluator.packed_state_for_z(physical, z_source)
    kwargs = packed_to_kwargs_lens(packed_z)
    assert state.lens_model_list[sheet_row] == "CONVERGENCE" and kwargs[sheet_row]["kappa"] > 0
    points = np.array([[3.0, -2.0], [-8.0, 5.0]])
    alpha_total, _ = evaluator.total_fields_at(points, packed_z)
    ax, ay = LensModel(lens_model_list=state.lens_model_list).alpha(points[:, 0], points[:, 1], kwargs)
    np.testing.assert_allclose(alpha_total, np.stack((ax, ay), -1), rtol=1e-8, atol=1e-10)
