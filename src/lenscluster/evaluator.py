"""Lens fields and the smoothed-local-image-plane log-likelihood for one model state."""
from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import gp_image_plane as gp, gp_knee, jax_lensing, smoothed_likelihood as sl
from .farfield import evaluate_farfield, member_farfield_geometry, slice_geometry
from .jax_cosmology import dpie_sigma0_factor_from_config, dpie_sigma0_factor_from_lensing_efficiency
from .model import (member_log_deviations, COMPONENT_FAMILY_SCALING, DP_IE_PROFILE, MASS_SHEET_PROFILE, NFW_PROFILE, SAFE_RADIUS_MARGIN_ARCSEC,
                    SAFE_RADIUS_MARGIN_KPC, SHEAR_PROFILE, ModelState, PackedLensState, PackedLensValidity, apply_parameter_transforms_jax)
from .profiles import critical_density_msun_kpc3, nfw_alpha_rs_arcsec, nfw_scale_density

BAD_LOG_LIKE = -1.0e30


class Evaluator:
    """Evaluates deflections, Jacobians and the log-likelihood for a fixed member classification."""

    def __init__(self, state: ModelState, likelihood: str = "smoothed") -> None:
        """``likelihood``: "smoothed" (local image-plane, default), "gp_image_plane" (correlated image-plane displacements
        per family, see ``gp_image_plane``) or "source-isotropic" (source-plane Gaussian with variance sigma_def^2 +
        sigma_meas^2 per image and no Jacobian; cheap, biased, for searches only)."""
        if likelihood not in {"smoothed", "gp_image_plane", "source-isotropic", "source-anisotropic"}:
            raise ValueError("likelihood must be 'smoothed', 'gp_image_plane', 'source-isotropic' or 'source-anisotropic'")
        self.likelihood_mode = likelihood
        self.gp_mode = likelihood == "gp_image_plane"
        self.gp_kernel = str(getattr(state, "gp_kernel", "image_plane"))
        self.gp_knee = self.gp_mode and self.gp_kernel == "knee"
        # gp_potential: the GP (or part of it) is a deflection error in the source plane; the image solver then adds its mean.
        self.gp_potential = self.gp_mode and self.gp_kernel in ("potential", "knee")
        has_gp_sites = int(state.sigma_shared_param_index) >= 0 or int(getattr(state, "gp_u_small_param_index", -1)) >= 0
        if self.gp_mode != has_gp_sites:
            raise ValueError("gp_image_plane needs a state built with likelihood_mode='gp_image_plane', and the other modes one without")
        self.state = state
        specs = state.parameter_specs
        kinds = np.asarray([spec.transform_kind for spec in specs], dtype=object)
        self.log_positive_mask = jnp.asarray(kinds == "log_positive")
        self.log_offset_positive_mask = jnp.asarray(kinds == "log_offset_positive")
        self.affine_mask = jnp.asarray(kinds == "affine")
        self.transform_offset = jnp.asarray([float(spec.transform_offset) for spec in specs], dtype=jnp.float64)
        self.transform_scale = jnp.asarray([float(spec.transform_scale) for spec in specs], dtype=jnp.float64)
        packed = state.packed_lens_spec
        self.spec = {name: jnp.asarray(value) for name, value in vars(packed).items() if not isinstance(value, str)}
        self.n_components = int(np.asarray(packed.profile_type).size)
        self.factor_by_z = jnp.asarray([state.dpie_sigma0_factor_by_z[z] for z in state.z_source_values], dtype=jnp.float64)
        self.nfw_rho_crit = float(critical_density_msun_kpc3(state.z_lens, state.cosmo_config))
        # The dPIE sigma0 factor is proportional to the lensing efficiency D_ls / D_s; this constant recovers it.
        self.efficiency_per_factor = 1.0 / float(dpie_sigma0_factor_from_lensing_efficiency(1.0))
        images = state.images
        self.xy = jnp.stack((images.x_obs, images.y_obs), -1)
        members = state.members
        self.member_components = np.asarray(members.component_indices, dtype=np.int32)
        self.exact_positions = np.asarray(members.exact_positions, dtype=np.int32)
        self.far_positions = np.asarray(members.far_positions, dtype=np.int32)
        far_components = set(self.member_components[self.far_positions].tolist())
        self.exact_components = np.asarray([i for i in range(self.n_components) if i not in far_components], dtype=np.int32)
        self.member_xy = np.stack((np.asarray(packed.x_center_base)[self.member_components],
                                   np.asarray(packed.y_center_base)[self.member_components]), -1)
        self.geometry = member_farfield_geometry(np.asarray(self.xy), self.member_xy)
        self.far_geometry = slice_geometry(self.geometry, self.far_positions)
        self.far_components = self.member_components[self.far_positions]
        self.far_min_separation = jnp.asarray(np.asarray(self.geometry.separation)[self.far_positions].min(axis=1)) if len(self.far_positions) else jnp.zeros(0)
        self.guard_cut_radii = float(members.far_field_guard_cut_radii)
        self.far_field_barrier_width = float(members.far_field_barrier_width)
        self.far_field_barrier_strength = float(members.far_field_barrier_strength)
        self.sigma_def_index, self.sigma_extra_index, self.sigma_sel_index = int(state.sigma_def_param_index), int(state.sigma_extra_param_index), int(state.sigma_sel_param_index)
        self.lambda_sel_index = int(state.lambda_sel_param_index)
        self.sigma_shared_index, self.sigma_white_index = int(state.sigma_shared_param_index), int(state.sigma_white_param_index)
        self.gp_ell_index = int(getattr(state, "gp_ell_param_index", -1))
        self.gp_u_small_index = int(getattr(state, "gp_u_small_param_index", -1))
        self.gp_ell_small_index = int(getattr(state, "gp_ell_small_param_index", -1))
        self.gp_sigma_large_index = int(getattr(state, "gp_sigma_large_param_index", -1))
        self.gp_sigma_alpha_index = int(getattr(state, "gp_sigma_alpha_param_index", -1))
        self.gp_knee_index = int(getattr(state, "gp_knee_param_index", -1))
        self.gp_knee_gamma_index = int(getattr(state, "gp_knee_gamma_param_index", -1))
        self.gp_sigma_alpha2_index = int(getattr(state, "gp_sigma_alpha2_param_index", -1))
        self.gp_knee2_index = int(getattr(state, "gp_knee2_param_index", -1))
        self.gp_knee_beta_low_index = int(getattr(state, "gp_knee_beta_low_param_index", -1))
        self.gp_knee_spectrum = str(dict(getattr(state, "gp_knee", {}) or {}).get("spectrum", "knee"))
        self.freeze_determinant_gradient = bool(state.freeze_determinant_gradient)
        self.scale_indices = jnp.asarray(sorted({int(i) for i in np.asarray(members.scatter_param_index) if i >= 0}
                                                | {int(i) for i in np.asarray(members.radius_scatter_param_index) if i >= 0}
                                                | {i for i in (self.sigma_def_index, self.sigma_extra_index, self.sigma_sel_index, self.lambda_sel_index,
                                                               self.sigma_shared_index, self.sigma_white_index, self.gp_ell_index,
                                                               self.gp_u_small_index, self.gp_ell_small_index, self.gp_sigma_large_index,
                                                               self.gp_sigma_alpha_index, self.gp_knee_index, self.gp_sigma_alpha2_index,
                                                               self.gp_knee2_index) if i >= 0}), dtype=jnp.int32)
        if self.gp_mode:
            # Data-only quantities of the family covariance: slots, separations, instrumental variances, redshift weights and
            # the kernel at the fixed ell (recomputed from the separations at every evaluation when ell is sampled).
            self.gp_floor = float(state.gp_floor_arcsec)
            self.gp_ell_fixed = float(state.gp_correlation_length_arcsec)
            self.gp_kernel_family = str(getattr(state, "gp_kernel_family", "gaussian"))
            self.gp_ell_large = float(getattr(state, "gp_ell_large_arcsec", 25.0))
            weights = None
            if self.gp_potential:
                reference = float(dpie_sigma0_factor_from_config(state.z_lens, float(getattr(state, "gp_reference_redshift", 2.0)), state.cosmo_config))
                self.gp_weights_per_image = gp.redshift_weights(self.factor_by_z[jnp.asarray(images.z_index_per_image)], reference)
                weights = self.gp_weights_per_image
            if self.gp_knee:
                # Knee kernel: blocks of images (by family, system or all), the knee tables over the block separations, and
                # the redshift weight of each family for the mean-deflection correction.
                knee = dict(getattr(state, "gp_knee", {}))
                lam_lo, lam_hi, lam_n = knee.get("lambda_grid", (0.5, 80.0, 64))
                scale_lo, scale_hi = knee.get("scales", (0.2, 100.0))
                xy = np.asarray(self.xy)
                r_max = float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))) + 1.0
                gamma_grid = knee.get("gamma_grid") if self.gp_knee_gamma_index >= 0 else None
                self.gp_knee_gamma_fixed = float(knee.get("gamma", 4.0))
                project = knee.get("project", ())
                # Older states stored a bool for the (since replaced) conditioning; only named residual-projection modes count.
                modes = tuple(project) if isinstance(project, (tuple, list)) else ()
                self.gp_project_modes = modes if str(getattr(state, "gp_blocks", "all")) == "all" else ()
                # The spectrum's shape (config.LikelihoodConfig.gp_knee_spectrum): the broken power law tabulates its low-k slope
                # on the second axis at the fixed high-k slope; the power law ignores the knee axis (gp_knee_power_law_lam).
                shape = {"broken_power_law": "broken_power_law", "power_law": "power_law"}.get(self.gp_knee_spectrum, "knee")
                # The knee when it is not sampled: the fixed value, or (power law, where it is ignored) the table's end.
                self.gp_knee_power_law_lam = float(knee["knee_fixed"]) if knee.get("knee_fixed") is not None else float(lam_hi)
                self.gp_knee_ell_fixed = None if knee.get("ell_fixed") is None else float(knee["ell_fixed"])
                self.gp_tables = gp_knee.knee_tables(r_max=r_max, r_step=float(knee.get("r_step", 0.02)), lam_min=float(lam_lo), lam_max=float(lam_hi),
                                                     n_lam=int(lam_n), scale_min=float(scale_lo), scale_max=float(scale_hi),
                                                     gamma=None if (gamma_grid is not None and shape != "broken_power_law") else self.gp_knee_gamma_fixed,
                                                     gamma_grid=None if shape == "broken_power_law" else gamma_grid, shape=shape,
                                                     beta_low_grid=knee.get("beta_low_grid") if shape == "broken_power_law" else None)
                self.gp_blocks = str(getattr(state, "gp_blocks", "all"))
                self.gp = gp_knee.block_layout(images, self.gp_blocks, weights, self.gp_tables)
                # Blocks grouped by size (gp_knee.bucket_layouts), each group padded to its own largest block: the same
                # likelihood at a fraction of the padded linear algebra (CANUCS system blocks: 11 ms against 47 ms per
                # gradient).  On by default with several blocks; state.gp_group_blocks = False keeps one padded batch.
                # The GP mean deflection, which reads the padded per-block factors, uses an ungrouped copy
                # (_padded_evaluator).  block position: index of each block in the concatenated outputs.
                group = getattr(state, "gp_group_blocks", None)
                group = self.gp_blocks != "all" if group is None else bool(group)
                self.gp_groups = gp_knee.bucket_layouts(self.gp) if group else None
                self.gp_frozen_blocks = None                    # freeze_gp_kernels: kernel blocks at fixed error scales
                order = (np.concatenate([g["block_ids"] for g in self.gp_groups]) if self.gp_groups is not None
                         else np.arange(int(self.gp["mask"].shape[0])))
                position = np.empty(order.size, dtype=np.int32)
                position[order] = np.arange(order.size)
                self.gp_block_position = jnp.asarray(position)
                family_of_image = np.asarray(images.family_index_per_image)
                w_family = np.zeros(int(images.n_families))
                w_family[family_of_image] = np.asarray(weights)
                self.gp_weights_per_family = jnp.asarray(w_family)
            else:
                self.gp = gp.precompute(images, state.gp_correlation_length_arcsec, weights)
        # Components whose geometry (centre, ellipticity) never changes: catalog members.  Their geometry at the image
        # positions is a constant of the computation and is precomputed once.
        packed_spec = state.packed_lens_spec
        profile = np.asarray(packed_spec.profile_type)
        fixed = ((profile == DP_IE_PROFILE) & (np.asarray(packed_spec.e1_param_index) < 0) & (np.asarray(packed_spec.e2_param_index) < 0)
                 & (np.asarray(packed_spec.x_center_param_index) < 0) & (np.asarray(packed_spec.y_center_param_index) < 0))
        self.fixed_geometry_components = np.flatnonzero(fixed).astype(np.int32)
        self.fixed_geometry_position = {int(c): i for i, c in enumerate(self.fixed_geometry_components)}
        if len(self.fixed_geometry_components):
            take = lambda v: jnp.asarray(np.asarray(v, dtype=float)[self.fixed_geometry_components])
            self.fixed_geometry = jax_lensing.grouped_dpie_geometry(self.xy[:, 0], self.xy[:, 1], take(packed_spec.e1_base), take(packed_spec.e2_base),
                                                                    take(packed_spec.x_center_base), take(packed_spec.y_center_base))
        else:
            self.fixed_geometry = None
        self._total_fields_cache: dict = {}
        self.loglike = jax.jit(self.log_likelihood)
        self.value_and_grad = jax.jit(jax.value_and_grad(self.log_likelihood))
        self.median_abs_determinant = jax.jit(lambda params: jnp.median(jnp.abs(self.statistics(params)["determinant"])))

    def search_log_likelihood(self, params: jnp.ndarray) -> jnp.ndarray:
        """The init-search target: ``log_likelihood`` without the source-transport volume factor."""
        return self.log_likelihood(params, source_jacobian=False)

    # ------------------------------------------------------------------ parameters
    def physical(self, params: jnp.ndarray) -> jnp.ndarray:
        return apply_parameter_transforms_jax(params, self.log_positive_mask, self.log_offset_positive_mask,
                                              self.transform_offset, self.affine_mask, self.transform_scale)

    def _apply(self, base: jnp.ndarray, index: jnp.ndarray, physical: jnp.ndarray) -> jnp.ndarray:
        safe = jnp.where(index >= 0, index, 0)
        return jnp.where(index >= 0, jnp.take(physical, safe), base)

    # ------------------------------------------------------------------ packed lens state
    def packed_state(self, physical: jnp.ndarray, factor_per_image: jnp.ndarray | None = None) -> tuple[PackedLensState, PackedLensValidity]:
        """Current dPIE and shear parameters.  The normalizations carry the lensing efficiency: by default one per
        constrained image (shape (component, image), from each image's source redshift); a scalar ``factor_per_image``
        gives one normalization for all points (shape (component,)); a one-dimensional array gives one per arbitrary
        point (shape (component, point)), which the global image search uses for its seeds."""
        spec = self.spec
        x_center = self._apply(spec["x_center_base"], spec["x_center_param_index"], physical)
        y_center = self._apply(spec["y_center_base"], spec["y_center_param_index"], physical)
        e1 = self._apply(spec["e1_base"], spec["e1_param_index"], physical)
        e2 = self._apply(spec["e2_base"], spec["e2_param_index"], physical)
        core_radius_kpc = self._apply(spec["core_radius_kpc_base"], spec["core_radius_param_index"], physical)
        cut_radius_kpc = self._apply(spec["cut_radius_kpc_base"], spec["cut_radius_param_index"], physical)
        gamma1 = self._apply(spec["gamma1_base"], spec["gamma1_param_index"], physical)
        gamma2 = self._apply(spec["gamma2_base"], spec["gamma2_param_index"], physical)
        scale_radius_kpc = self._apply(spec["scale_radius_kpc_base"], spec["scale_radius_param_index"], physical)
        log10_m200 = self._apply(spec["log10_m200_base"], spec["log10_m200_param_index"], physical)
        kappa_inf = self._apply(spec["kappa_base"], spec["kappa_param_index"], physical)
        profile_type = spec["profile_type"]
        is_dpie = profile_type == DP_IE_PROFILE
        is_nfw = profile_type == NFW_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_sheet = profile_type == MASS_SHEET_PROFILE
        is_scaling = spec["component_family"] == COMPONENT_FAMILY_SCALING
        luminosity_ratio = spec["luminosity_ratio"]
        cut_ref = self._apply(spec["cut_ref_base"], spec["cut_ref_param_index"], physical)
        core_ref = self._apply(spec["core_ref_base"], spec["core_ref_param_index"], physical)
        alpha_sigma = self._apply(spec["alpha_sigma_base"], spec["alpha_sigma_param_index"], physical)
        beta_radius = self._apply(spec["beta_radius_base"], spec["beta_radius_param_index"], physical)
        size_scale = jnp.power(luminosity_ratio, beta_radius)
        # Sampled member deviations: dispersion (in component_dispersions) and, in the mass and sigma_rcut modes, the radii.
        _, delta_r = member_log_deviations(self.state.packed_lens_spec, physical)
        v_disp = self.component_dispersions(physical)
        size_scale = size_scale * jnp.exp(delta_r)
        core_radius_kpc = jnp.where(is_scaling, core_ref * size_scale, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, cut_ref * size_scale, cut_radius_kpc)
        kpc_per_arcsec = jnp.asarray(self.state.kpc_per_arcsec, dtype=jnp.float64)
        softening_index = int(self.state.log_softening_length_param_index)
        if softening_index >= 0:
            softening_kpc = jnp.exp(physical[softening_index])
            core_effective_kpc = jnp.sqrt(jnp.square(core_radius_kpc) + jnp.square(softening_kpc))
            # A member whose cut radius is below the softening cannot be resolved by the simulation either: cap the
            # softened core smoothly just below r_cut (plain quadrature when r_cut >> softening).  The dPIE total mass
            # is pi sigma^2 r_cut / G, so member masses are unchanged and r_cut > r_core holds by construction.
            capped = cut_radius_kpc * core_effective_kpc / jnp.sqrt(jnp.square(core_effective_kpc) + jnp.square(cut_radius_kpc))
            core_effective_kpc = jnp.where(is_scaling, capped, core_effective_kpc)
        else:
            core_effective_kpc = core_radius_kpc
        cut_radius_kpc = jnp.where(spec["cut_radius_excess_above_effective_core"],
                                   core_effective_kpc + jnp.maximum(cut_radius_kpc, SAFE_RADIUS_MARGIN_KPC), cut_radius_kpc)
        ra_raw = core_effective_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        # NFW normalization: rho_s from (M200, r_s); other components get safe placeholders so no NaN enters a gradient.
        safe_rs_kpc = jnp.where(is_nfw, jnp.maximum(scale_radius_kpc, SAFE_RADIUS_MARGIN_KPC), 1.0)
        rho_s, _ = nfw_scale_density(jnp.where(is_nfw, log10_m200, 14.0), safe_rs_kpc, self.nfw_rho_crit)
        nfw_rs = jnp.where(is_nfw, safe_rs_kpc / kpc_per_arcsec, 0.0)
        if factor_per_image is None:
            factor_per_image = jnp.take(self.factor_by_z, self.state.images.z_index_per_image)
        factor_per_image = jnp.asarray(factor_per_image, dtype=jnp.float64)
        per_image = factor_per_image.ndim == 1
        if per_image:
            sigma0 = jnp.where(is_dpie[:, None], (v_disp[:, None] ** 2) * factor_per_image[None, :] / ra[:, None], 0.0)
            alpha_rs = jnp.where(is_nfw[:, None], nfw_alpha_rs_arcsec(rho_s[:, None], safe_rs_kpc[:, None], factor_per_image[None, :]), 0.0)
            kappa = jnp.where(is_sheet[:, None], kappa_inf[:, None] * (self.efficiency_per_factor * factor_per_image)[None, :], 0.0)
        else:
            factor = factor_per_image
            sigma0 = jnp.where(is_dpie, (v_disp ** 2) * factor / ra, 0.0)
            alpha_rs = jnp.where(is_nfw, nfw_alpha_rs_arcsec(rho_s, safe_rs_kpc, factor), 0.0)
            kappa = jnp.where(is_sheet, kappa_inf * (self.efficiency_per_factor * factor), 0.0)
        elliptical = is_dpie | is_nfw
        packed = PackedLensState(
            profile_type=profile_type, sigma0=sigma0, Ra=jnp.where(is_dpie, ra, 0.0), Rs=jnp.where(is_dpie, rs, 0.0),
            e1=jnp.where(elliptical, e1, 0.0), e2=jnp.where(elliptical, e2, 0.0), center_x=x_center, center_y=y_center,
            gamma1=jnp.where(is_shear, gamma1, 0.0), gamma2=jnp.where(is_shear, gamma2, 0.0), nfw_rs=nfw_rs, alpha_rs=alpha_rs, kappa=kappa)
        flags = jnp.stack([
            jnp.any(~jnp.isfinite(sigma0)) | jnp.any(~jnp.isfinite(alpha_rs)),
            jnp.any(is_dpie & (ra_raw <= 0.0)),
            jnp.any(is_dpie & (rs_raw <= ra_raw)),
            jnp.any(is_dpie & ~jnp.isfinite(rs_raw)),
            jnp.any(is_dpie & ~jnp.isfinite(v_disp)),
            jnp.any(is_scaling & (~jnp.isfinite(alpha_sigma) | ~jnp.isfinite(beta_radius))),
            jnp.any(~jnp.isfinite(x_center) | ~jnp.isfinite(y_center)),
            jnp.any(is_shear & (~jnp.isfinite(gamma1) | ~jnp.isfinite(gamma2))),
            jnp.any(elliptical & (~jnp.isfinite(e1) | ~jnp.isfinite(e2) | ((jnp.square(e1) + jnp.square(e2)) >= 1.0))),
            jnp.any(is_nfw & ((scale_radius_kpc <= 0.0) | ~jnp.isfinite(scale_radius_kpc) | ~jnp.isfinite(log10_m200))),
            jnp.any(is_sheet & ~jnp.isfinite(kappa_inf)),
        ])
        return packed, PackedLensValidity(is_valid=~jnp.any(flags), reason_flags=flags)

    def component_dispersions(self, physical: jnp.ndarray) -> jnp.ndarray:
        """Velocity dispersion of every component in km/s (Lenstool's sigma_LT): the sampled or fixed value, or for a
        member its scaling-relation value times its sampled deviation."""
        spec = self.spec
        v_disp = self._apply(spec["v_disp_base"], spec["v_disp_param_index"], physical)
        sigma_ref = self._apply(spec["sigma_ref_base"], spec["sigma_ref_param_index"], physical)
        alpha_sigma = self._apply(spec["alpha_sigma_base"], spec["alpha_sigma_param_index"], physical)
        v_disp = jnp.where(spec["component_family"] == COMPONENT_FAMILY_SCALING, sigma_ref * jnp.power(spec["luminosity_ratio"], alpha_sigma), v_disp)
        delta_v, _ = member_log_deviations(self.state.packed_lens_spec, physical)
        return v_disp * jnp.exp(delta_v)

    def kinematics_log_likelihood(self, physical: jnp.ndarray) -> jnp.ndarray:
        """The measured member dispersions (state.member_kinematics): sum over members of
        log N(ln sigma_meas | ln(c sigma_i), e_i); zero when no kinematics source is on."""
        kinematics = self.state.member_kinematics
        if not kinematics:
            return jnp.asarray(0.0, dtype=jnp.float64)
        model = jnp.log(jnp.asarray(kinematics["aperture_factor"]) * self.component_dispersions(physical)[jnp.asarray(kinematics["component_index"])])
        log_err = jnp.asarray(kinematics["log_err"])
        z = (jnp.log(jnp.asarray(kinematics["sigma"])) - model) / log_err
        return -0.5 * jnp.sum(z * z) - jnp.sum(jnp.log(log_err)) - 0.5 * z.size * jnp.log(2.0 * jnp.pi)

    def packed_state_for_z(self, physical: jnp.ndarray, z_source: float) -> PackedLensState:
        factor = self.state.dpie_sigma0_factor_by_z.get(float(z_source))
        if factor is None:
            from .jax_cosmology import dpie_sigma0_factor_from_config
            factor = float(dpie_sigma0_factor_from_config(self.state.z_lens, float(z_source), self.state.cosmo_config))
            self.state.dpie_sigma0_factor_by_z[float(z_source)] = factor
        return self.packed_state(physical, factor_per_image=jnp.asarray(factor, dtype=jnp.float64))[0]

    # ------------------------------------------------------------------ lens fields
    def component_rows(self, xy: jnp.ndarray, packed: PackedLensState, components: np.ndarray, *, derivative: bool = False):
        """Per-component deflection rows (C, N, 2), Jacobians U = d alpha / d theta (C, N, 2, 2) and, with ``derivative``,
        the spatial derivatives dU (C, N, 2, 2, 2).  dPIE components with fixed catalog geometry evaluated at the image
        positions reuse the precomputed geometry; the other dPIEs use the full analytic kernel, NFW halos the CSE kernel."""
        x, y = xy[:, 0], xy[:, 1]
        components = np.asarray(components, dtype=np.int32)
        n = x.shape[0]
        if components.size == 0:
            empty = (jnp.zeros((0, n, 2)), jnp.zeros((0, n, 2, 2)))
            return empty + ((jnp.zeros((0, n, 2, 2, 2)),) if derivative else ())
        profile = np.asarray(self.state.packed_lens_spec.profile_type)[components]
        alpha = jnp.zeros((components.size, n, 2), dtype=jnp.float64)
        u = jnp.zeros((components.size, n, 2, 2), dtype=jnp.float64)
        du = jnp.zeros((components.size, n, 2, 2, 2), dtype=jnp.float64) if derivative else None
        at_images = xy is self.xy
        dpie_mask = profile == DP_IE_PROFILE
        fixed_mask = dpie_mask & at_images & np.isin(components, self.fixed_geometry_components)
        variable_mask = dpie_mask & ~fixed_mask
        take = lambda value, idx: jnp.take(jnp.asarray(value, dtype=jnp.float64), jnp.asarray(idx), axis=0)
        if np.any(fixed_mask):
            idx = components[fixed_mask]
            geom_rows = np.asarray([self.fixed_geometry_position[int(c)] for c in idx], dtype=np.int32)
            geom = jax.tree.map(lambda g: g[geom_rows], self.fixed_geometry)
            a_rows, u_rows, du_rows = jax_lensing.grouped_dpie_fields_from_geometry[derivative](geom, take(packed.sigma0, idx), take(packed.Ra, idx), take(packed.Rs, idx))
            pos = jnp.asarray(np.flatnonzero(fixed_mask), dtype=jnp.int32)
            alpha, u = alpha.at[pos].set(a_rows), u.at[pos].set(u_rows)
            du = du.at[pos].set(du_rows) if derivative else du
        if np.any(variable_mask):
            idx = components[variable_mask]
            a_rows, u_rows, du_rows = jax_lensing.grouped_dpie_fields[derivative](x, y, take(packed.sigma0, idx), take(packed.Ra, idx), take(packed.Rs, idx),
                                                                                  take(packed.e1, idx), take(packed.e2, idx), take(packed.center_x, idx), take(packed.center_y, idx))
            pos = jnp.asarray(np.flatnonzero(variable_mask), dtype=jnp.int32)
            alpha, u = alpha.at[pos].set(a_rows), u.at[pos].set(u_rows)
            du = du.at[pos].set(du_rows) if derivative else du
        nfw_mask = profile == NFW_PROFILE
        if np.any(nfw_mask):
            idx = components[nfw_mask]
            a_rows, u_rows, du_rows = jax_lensing.grouped_nfw_fields[derivative](x, y, take(packed.nfw_rs, idx), take(packed.alpha_rs, idx),
                                                                                 take(packed.e1, idx), take(packed.e2, idx), take(packed.center_x, idx), take(packed.center_y, idx))
            pos = jnp.asarray(np.flatnonzero(nfw_mask), dtype=jnp.int32)
            alpha, u = alpha.at[pos].set(a_rows), u.at[pos].set(u_rows)
            du = du.at[pos].set(du_rows) if derivative else du
        shear_mask = profile == SHEAR_PROFILE
        if np.any(shear_mask):
            idx = jnp.asarray(components[shear_mask])
            gamma1 = jnp.take(packed.gamma1, idx)[:, None]
            gamma2 = jnp.take(packed.gamma2, idx)[:, None]
            ones = jnp.ones_like(x)[None, :]
            shear_alpha = jnp.stack((gamma1 * x[None] + gamma2 * y[None], gamma2 * x[None] - gamma1 * y[None]), -1)
            # U = d alpha / d theta for the shear is the constant matrix [[gamma1, gamma2], [gamma2, -gamma1]]; dU = 0.
            shear_u = jnp.stack((jnp.stack((gamma1 * ones, gamma2 * ones), -1), jnp.stack((gamma2 * ones, -gamma1 * ones), -1)), -2)
            pos = jnp.asarray(np.flatnonzero(shear_mask), dtype=jnp.int32)
            alpha, u = alpha.at[pos].set(shear_alpha), u.at[pos].set(shear_u)
        sheet_mask = profile == MASS_SHEET_PROFILE
        if np.any(sheet_mask):
            idx = components[sheet_mask]
            kappa = jnp.take(jnp.asarray(packed.kappa, dtype=jnp.float64), jnp.asarray(idx), axis=0)   # (c,) or (c, N)
            kappa = jnp.broadcast_to(kappa if kappa.ndim == 2 else kappa[:, None], (idx.size, n))
            # A uniform sheet about the reference origin: alpha = kappa theta, U = kappa I, dU = 0.
            sheet_alpha = kappa[..., None] * xy[None, :, :]
            sheet_u = kappa[..., None, None] * jnp.eye(2)[None, None]
            pos = jnp.asarray(np.flatnonzero(sheet_mask), dtype=jnp.int32)
            alpha, u = alpha.at[pos].set(sheet_alpha), u.at[pos].set(sheet_u)
        return (alpha, u, du) if derivative else (alpha, u)

    def total_fields_at(self, xy, packed: PackedLensState, components: np.ndarray | None = None, *, chunk: int = 16384):
        """Summed deflection (N, 2) and Jacobian U (N, 2, 2) of the given components at arbitrary points, evaluated in
        jitted chunks so that large grids (figures, truth maps) never hold per-component arrays for every point."""
        components = np.arange(self.n_components, dtype=np.int32) if components is None else np.asarray(components, dtype=np.int32)
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        n = xy.shape[0]
        # Small point sets (image solvers) use a power-of-two block so that only a few sizes are ever compiled.
        chunk = int(min(chunk, max(64, 1 << max(0, int(n - 1)).bit_length())))
        key = (components.tobytes(), chunk)
        if key not in self._total_fields_cache:
            def total(points, packed_state):
                alpha_rows, u_rows = self.component_rows(points, packed_state, components)
                return jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)
            self._total_fields_cache[key] = jax.jit(total)
        total = self._total_fields_cache[key]
        alpha, u = np.empty((n, 2)), np.empty((n, 2, 2))
        for start in range(0, n, chunk):
            block = xy[start:start + chunk]
            pad = chunk - block.shape[0]
            points = jnp.asarray(np.pad(block, ((0, pad), (0, 0))) if pad else block)
            a_block, u_block = total(points, packed)
            alpha[start:start + chunk] = np.asarray(a_block)[:block.shape[0]]
            u[start:start + chunk] = np.asarray(u_block)[:block.shape[0]]
        return alpha, u

    def fields(self, physical: jnp.ndarray, packed: PackedLensState | None = None, *, force_exact: bool = False,
               derivative: bool = True) -> dict[str, Any]:
        """Backprojected images, total Jacobian A, grad det A (unless ``derivative`` is False), and per-member deflections."""
        if packed is None:
            packed = self.packed_state(physical)[0]
        xy = self.xy
        components = np.arange(self.n_components, dtype=np.int32) if force_exact else self.exact_components
        rows = self.component_rows(xy, packed, components, derivative=derivative)
        alpha_rows, u_rows = rows[:2]
        du_rows = rows[2] if derivative else None
        alpha, u = jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)
        du = jnp.sum(du_rows, axis=0) if derivative else jnp.zeros(xy.shape + (2, 2), dtype=xy.dtype)
        n_members, n_images = len(self.member_components), xy.shape[0]
        member_alpha = jnp.zeros((n_members, n_images, 2), dtype=xy.dtype)
        member_u = jnp.zeros((n_members, n_images, 2, 2), dtype=xy.dtype)
        row_lookup = {int(c): i for i, c in enumerate(components)}
        exact_member = np.asarray([i for i, c in enumerate(self.member_components) if int(c) in row_lookup], dtype=int)
        row_indices = np.asarray([row_lookup[int(self.member_components[i])] for i in exact_member], dtype=int)
        member_alpha = member_alpha.at[exact_member].set(alpha_rows[row_indices])
        member_u = member_u.at[exact_member].set(u_rows[row_indices])
        far_cut_radii = None
        if not force_exact and len(self.far_positions):
            ci = jnp.asarray(self.far_components)
            far = evaluate_farfield(self.far_geometry, jnp.take(packed.sigma0, ci, axis=0), jnp.take(packed.Ra, ci), jnp.take(packed.Rs, ci))
            member_alpha = member_alpha.at[self.far_positions].set(far.alpha)
            member_u = member_u.at[self.far_positions].set(far.jacobian)
            alpha = alpha + jnp.sum(far.alpha, axis=0)
            u = u + jnp.sum(far.jacobian, axis=0)
            du = du + (far.total_jacobian_derivative if derivative else 0.0)
            far_cut_radii = jnp.take(packed.Rs, ci)
        a = jnp.eye(2) - u
        da = -du
        g = (a[:, 1, 1, None] * da[:, 0, 0, :] + a[:, 0, 0, None] * da[:, 1, 1, :]
             - a[:, 1, 0, None] * da[:, 0, 1, :] - a[:, 0, 1, None] * da[:, 1, 0, :])
        return dict(backprojected=xy - alpha, jacobian=a, determinant_gradient=g, jacobian_spatial_derivative=da,
                    member_alpha=member_alpha, member_jacobian=member_u, far_cut_radii=far_cut_radii, packed=packed)

    def far_field_valid(self, far_cut_radii) -> jnp.ndarray:
        """True when every far-field member keeps all images beyond the guard distance in cut radii (a diagnostic; the
        likelihood enforces the guard through the smooth barrier below)."""
        if far_cut_radii is None or len(self.far_positions) == 0:
            return jnp.asarray(True)
        return jnp.all(self.far_min_separation >= self.guard_cut_radii * far_cut_radii)

    def far_field_log_barrier(self, far_cut_radii) -> jnp.ndarray:
        """The far-field guard as a smooth log-likelihood penalty: for each far-field member, with x = ln(R / (guard r_cut)),
        -strength * softplus(-x / width).  Zero far outside the guard, about -0.7 strength at it, and linear in -x
        inside, so the Hamiltonian flow sees the boundary and turns back instead of diverging on a wall."""
        if far_cut_radii is None or len(self.far_positions) == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        x = jnp.log(self.far_min_separation) - jnp.log(self.guard_cut_radii * far_cut_radii)
        return -self.far_field_barrier_strength * jnp.sum(jax.nn.softplus(-x / self.far_field_barrier_width))

    # ------------------------------------------------------------------ likelihood
    def error_scales(self, physical: jnp.ndarray) -> dict[str, Any]:
        """The scales of the per-image source-plane covariance C_i = sigma_meas^2 A A^T + sigma_def^2 I + sigma_sel^2 R at
        this model.  Under gp_image_plane C_i is the diagonal block of the family covariance: sigma_meas^2 = sigma_inst^2 +
        sigma_white^2 + sigma_shared^2, sigma_def = sigma_fl and no selection term; sigma_img = sqrt(sigma_shared^2 + sigma_white^2)
        is the total image-plane scale."""
        images = self.state.images
        zero, one = jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(1.0, dtype=jnp.float64)
        if self.gp_knee:
            # Knee kernel: C_ii = (w_i^2 sigma_alpha^2 + sigma_fl^2) I + (sigma_inst^2 + sigma_white^2 + sigma_shared^2) A A^T.
            nan = jnp.asarray(float("nan"), dtype=jnp.float64)
            sigma_alpha = physical[self.gp_sigma_alpha_index] if self.gp_sigma_alpha_index >= 0 else zero
            knee = (physical[self.gp_knee_index] if self.gp_knee_index >= 0
                    else jnp.asarray(getattr(self, "gp_knee_power_law_lam", float("nan")), dtype=jnp.float64))
            knee_gamma = (physical[self.gp_knee_gamma_index] if self.gp_knee_gamma_index >= 0
                          else jnp.asarray(getattr(self, "gp_knee_gamma_fixed", float("nan")), dtype=jnp.float64))
            sigma_alpha2 = physical[self.gp_sigma_alpha2_index] if self.gp_sigma_alpha2_index >= 0 else zero
            knee2 = physical[self.gp_knee2_index] if self.gp_knee2_index >= 0 else nan
            beta_low = physical[self.gp_knee_beta_low_index] if self.gp_knee_beta_low_index >= 0 else nan
            sigma_shared, sigma_white = physical[self.sigma_shared_index], physical[self.sigma_white_index]
            gp_ell = (physical[self.gp_ell_index] if self.gp_ell_index >= 0
                      else jnp.asarray(getattr(self, "gp_knee_ell_fixed", None) or float("nan"), dtype=jnp.float64))
            sigma_img = jnp.sqrt(jnp.square(sigma_shared) + jnp.square(sigma_white))
            w = self.gp_weights_per_image
            w2 = jnp.ones_like(w) if self.gp_knee_spectrum == "knee_foreground" else w
            sigma_def = jnp.sqrt(jnp.square(w) * jnp.square(sigma_alpha) + jnp.square(w2) * jnp.square(sigma_alpha2) + jnp.square(self.gp_floor))
            return dict(sigma_def=sigma_def, sigma_extra=sigma_img, sigma_sel=zero, lambda_sel=one,
                        sigma_meas=jnp.sqrt(jnp.square(images.sigma_inst_per_image) + jnp.square(sigma_img)), sigma_img=sigma_img,
                        sigma_shared=sigma_shared, sigma_white=sigma_white, gp_ell=gp_ell, gp_sigma_alpha=sigma_alpha, gp_knee=knee,
                        gp_knee_gamma=knee_gamma, gp_sigma_alpha2=sigma_alpha2, gp_knee2=knee2, gp_knee_beta_low=beta_low)
        if self.gp_potential:
            # Potential kernel: C_ii = w_i^2 (sigma_S^2 + sigma_L^2) I + sigma_fl^2 I + sigma_meas^2 A A^T, a per-image sigma_def.
            u_small, ell_small = physical[self.gp_u_small_index], physical[self.gp_ell_small_index]
            sigma_small = u_small * ell_small
            sigma_large = physical[self.gp_sigma_large_index] if self.gp_sigma_large_index >= 0 else zero
            sigma_extra = physical[self.sigma_extra_index] if self.sigma_extra_index >= 0 else zero
            w = self.gp_weights_per_image
            kappa = gp.KAPPA_CONSTANT[self.gp_kernel_family]
            sigma_def = jnp.sqrt(jnp.square(w) * (jnp.square(sigma_small) + jnp.square(sigma_large)) + jnp.square(self.gp_floor))
            return dict(sigma_def=sigma_def, sigma_extra=sigma_extra, sigma_sel=zero, lambda_sel=one,
                        sigma_meas=jnp.sqrt(jnp.square(images.sigma_inst_per_image) + jnp.square(sigma_extra)),
                        gp_u_small=u_small, gp_ell_small=ell_small, gp_sigma_small=sigma_small, gp_sigma_large=sigma_large,
                        gp_ell_large=jnp.asarray(self.gp_ell_large, dtype=jnp.float64),
                        sigma_kappa_small=kappa * u_small, sigma_kappa_large=kappa * sigma_large / self.gp_ell_large)
        if self.gp_mode:
            sigma_shared, sigma_white = physical[self.sigma_shared_index], physical[self.sigma_white_index]
            sigma_img = jnp.sqrt(jnp.square(sigma_shared) + jnp.square(sigma_white))
            gp_ell = physical[self.gp_ell_index] if self.gp_ell_index >= 0 else jnp.asarray(self.gp_ell_fixed, dtype=jnp.float64)
            return dict(sigma_def=jnp.asarray(self.gp_floor, dtype=jnp.float64), sigma_extra=sigma_img, sigma_sel=zero, lambda_sel=one,
                        sigma_meas=jnp.sqrt(jnp.square(images.sigma_inst_per_image) + jnp.square(sigma_img)), sigma_img=sigma_img,
                        sigma_shared=sigma_shared, sigma_white=sigma_white, gp_ell=gp_ell)
        sigma_def = physical[self.sigma_def_index]
        sigma_extra = physical[self.sigma_extra_index] if self.sigma_extra_index >= 0 else zero
        sigma_sel = physical[self.sigma_sel_index] if self.sigma_sel_index >= 0 else zero
        lambda_sel = physical[self.lambda_sel_index] if self.lambda_sel_index >= 0 else one
        sigma_meas = jnp.sqrt(jnp.square(images.sigma_inst_per_image) + jnp.square(sigma_extra))   # the instrumental error is the floor
        return dict(sigma_def=sigma_def, sigma_extra=sigma_extra, sigma_sel=sigma_sel, lambda_sel=lambda_sel, sigma_meas=sigma_meas)

    def per_image_covariance(self, physical: jnp.ndarray, jacobian: jnp.ndarray) -> dict[str, Any]:
        """``smoothed_likelihood.source_covariance`` at this model's scales (the diagonal blocks under gp_image_plane)."""
        scales = self.error_scales(physical)
        return sl.source_covariance(jacobian, scales["sigma_meas"], scales["sigma_def"], scales["sigma_sel"], scales["lambda_sel"])

    def statistics(self, params: jnp.ndarray, *, force_exact: bool = False) -> dict[str, Any]:
        physical = self.physical(params)
        images, members = self.state.images, self.state.members
        isotropic = self.likelihood_mode == "source-isotropic"
        fields = self.fields(physical, force_exact=force_exact, derivative=not isotropic)
        if self.gp_mode:
            return self._gp_statistics(params, physical, fields)
        scales = self.error_scales(physical)
        sigma_def, sigma_extra, sigma_sel, lambda_sel, sigma_meas = (scales[k] for k in ("sigma_def", "sigma_extra", "sigma_sel", "lambda_sel", "sigma_meas"))
        jacobian = jnp.zeros_like(fields["jacobian"]) if isotropic else fields["jacobian"]
        cov = sl.source_covariance(jacobian, sigma_meas, sigma_def, sigma_sel, lambda_sel)
        whitening = sl.source_whitening(fields["backprojected"], cov["covariance"], images.family_index_per_image, images.n_families)
        coordinates = jnp.stack((params[images.source_x_param_index], params[images.source_y_param_index]), -1)
        transported = sl.transport_sources(coordinates, whitening, jnp.asarray(members.source_prior_mean), members.source_prior_sigma)
        source = transported["source"]
        determinant_gradient = fields["determinant_gradient"]
        if self.freeze_determinant_gradient:
            determinant_gradient = jax.lax.stop_gradient(determinant_gradient)
        stats = sl.statistics(fields["backprojected"], source[images.family_index_per_image], jacobian,
                              determinant_gradient, sigma_meas, sigma_def, sigma_sel, lambda_sel, parity=images.parity)
        if isotropic or self.likelihood_mode == "source-anisotropic":   # Gaussian source-plane term only: no determinant factor
            stats["loglike"] = stats["gaussian_loglike"]
            stats["valid"] = cov["valid"] & jnp.all(jnp.isfinite(stats["residual"]), axis=-1)
        stats.update(source_by_family=source, source_correction=transported["log_correction"], source_jacobian=jnp.sum(transported["log_jacobian"]),
                     whitening=whitening,
                     transport_valid=transported["valid"], sigma_def=sigma_def, sigma_extra=sigma_extra, sigma_meas=sigma_meas, sigma_sel=sigma_sel,
                     lambda_sel=lambda_sel,
                     fields=fields, physical=physical)
        return stats

    def gp_kernel_blocks(self, physical: jnp.ndarray, scales: dict[str, Any] | None = None, layout: dict | None = None) -> dict[str, Any]:
        """Source-plane GP kernels: the deflection covariance blocks W K_alpha W (S, M, M, 2, 2) in slot order and the padded
        image-plane variances sigma_meas^2 (S, M).  Under the knee kernel also ``k_img``, the image-plane GP covariance
        sigma_shared^2 K_img (S, M, M) that B S B^T adds between images, and ``white`` = sigma_inst^2 + sigma_white^2.
        ``layout`` (knee kernel) is one group of ``self.gp_groups``; the full layout by default."""
        layout = self.gp if layout is None else layout
        frozen = getattr(self, "gp_frozen_blocks", None)
        if frozen is not None:
            return frozen[id(layout)]
        scales = self.error_scales(physical) if scales is None else scales
        if self.gp_knee:
            mask, w = layout["mask"], layout["weights"]
            pair_mask = layout["pair_mask"]
            if self.gp_sigma_alpha_index >= 0:
                # The second table axis: the sampled slope (knee, power law) or the low-k slope (broken power law).
                axis = (scales["gp_knee_beta_low"] if self.gp_knee_beta_low_index >= 0
                        else scales["gp_knee_gamma"] if self.gp_knee_gamma_index >= 0 else None)
                c_l, c_t = gp_knee.pair_coefficients(layout["pair_tables"], layout["r_local"], layout["r_fraction"], scales["gp_knee"], axis)
                k_alpha = gp_knee.deflection_blocks(c_l, c_t, layout["nn"], pair_mask, jnp.square(scales["gp_sigma_alpha"]))
                k_alpha = k_alpha * (w[:, :, None] * w[:, None, :])[..., None, None]
                if self.gp_sigma_alpha2_index >= 0:
                    # A second, independent knee field (two_knee: the same redshift weights; knee_foreground: weight 1).
                    c_l2, c_t2 = gp_knee.pair_coefficients(layout["pair_tables"], layout["r_local"], layout["r_fraction"], scales["gp_knee2"],
                                                           scales["gp_knee_gamma"] if self.gp_knee_gamma_index >= 0 else None)
                    w2 = jnp.ones_like(w) if self.gp_knee_spectrum == "knee_foreground" else w
                    k_alpha = k_alpha + (gp_knee.deflection_blocks(c_l2, c_t2, layout["nn"], pair_mask, jnp.square(scales["gp_sigma_alpha2"]))
                                         * (w2[:, :, None] * w2[:, None, :])[..., None, None])
                if getattr(self, "gp_project_modes", ()):
                    k_alpha = gp_knee.project_low_order(k_alpha, layout["xy"], mask, w, self.gp_project_modes)
            else:
                k_alpha = jnp.zeros(layout["nn"].shape, dtype=jnp.float64)
            k_img = jnp.square(scales["sigma_shared"]) * jnp.exp(-layout["squared_separations"] / (2.0 * jnp.square(scales["gp_ell"]))) * pair_mask
            white = jnp.where(mask, layout["sigma_inst2"] + jnp.square(scales["sigma_white"]), 0.0)
            return dict(k_alpha=k_alpha, k_img=k_img, white=white, sigma_meas2=white + jnp.where(mask, jnp.square(scales["sigma_shared"]), 0.0))
        d2, outer, mask, w = layout["squared_separations"], layout["outer"], layout["mask"], layout["weights"]
        k_alpha = gp.deflection_kernel(d2, outer, mask, jnp.square(scales["gp_sigma_small"]), scales["gp_ell_small"], self.gp_kernel_family)
        if self.gp_sigma_large_index >= 0:
            k_alpha = k_alpha + gp.deflection_kernel(d2, outer, mask, jnp.square(scales["gp_sigma_large"]), scales["gp_ell_large"], self.gp_kernel_family)
        k_alpha = k_alpha * (w[:, :, None] * w[:, None, :])[..., None, None]
        sigma_meas2 = jnp.where(mask, layout["sigma_inst2"] + jnp.square(scales["sigma_extra"]), 0.0)
        return dict(k_alpha=k_alpha, sigma_meas2=sigma_meas2)

    def _gp_mean_inputs(self, params: jnp.ndarray):
        """v = C^-1 (b - beta) per family (F, M, 2) at this draw's sources, and the four scales (sigma_S, ell_S, sigma_L, ell_L)."""
        layout = self.gp
        slots, mask = layout["slots"], layout["mask"]
        n_families, size = mask.shape
        stats = self.statistics(params)
        residual = jnp.where(mask[..., None], stats["fields"]["backprojected"][slots] - stats["source_by_family"][:, None, :], 0.0)
        v = jax.scipy.linalg.cho_solve((stats["family_cholesky"], True), residual.reshape(n_families, 2 * size)[..., None])[..., 0].reshape(n_families, size, 2)
        v = jnp.where(mask[..., None], v, 0.0)
        scales = self.error_scales(stats["physical"])
        return v, jnp.stack([scales["gp_sigma_small"], scales["gp_ell_small"], scales["gp_sigma_large"], scales["gp_ell_large"]])

    def freeze_gp_kernels(self, params) -> None:
        """Knee kernel only: compute the kernel blocks (mass field, image-plane GP, white terms) once at the error scales of
        ``params`` and reuse them in every later evaluation.  Valid only while the caller holds the error scales fixed (init
        stage A): the blocks depend on the error scales and the data, not on the lens.  Call before jitting."""
        if not self.gp_knee:
            raise ValueError("freeze_gp_kernels needs the knee kernel")
        self.gp_frozen_blocks = None
        physical = self.physical(jnp.asarray(params, dtype=jnp.float64))
        scales = self.error_scales(physical)
        groups = self.gp_groups if self.gp_groups is not None else [self.gp]
        frozen = {id(g): {k: jnp.asarray(np.asarray(v)) for k, v in self.gp_kernel_blocks(physical, scales, g).items()} for g in groups}
        self.gp_frozen_blocks = frozen

    def _knee_statistics(self, params: jnp.ndarray, physical: jnp.ndarray, fields: dict[str, Any], scales: dict[str, Any]) -> dict[str, Any]:
        """Knee kernel on blocks with several knots: one Cholesky per block for the Gaussian term and the K-source conditional,
        the smoothed determinant h_i per image on the diagonal block (gp_knee.py)."""
        images, members, layout = self.state.images, self.state.members, self.gp
        sigma_fl = jnp.asarray(self.gp_floor, dtype=jnp.float64)
        jacobian, backprojected = fields["jacobian"], fields["backprojected"]
        # The blocks in one padded batch, or (init search) in groups padded to their own size: the same terms per block.
        groups = self.gp_groups if self.gp_groups is not None else [layout]
        parts = []
        for group in groups:
            blocks = self.gp_kernel_blocks(physical, scales, group)
            covariance, cholesky = gp_knee.block_covariance(jacobian[group["slots"]], blocks["k_alpha"], blocks["k_img"], blocks["white"],
                                                            group["mask"], jnp.square(sigma_fl))
            padded = backprojected[group["slots"]]
            whitening = gp_knee.block_source_whitening(cholesky, padded, group["indicator"], group["mask"], group["knot_mask"])
            parts.append((group, covariance, cholesky, padded, whitening))
            if getattr(self, "gp_project_modes", ()):
                projected_blocks = blocks["k_alpha"]
        coordinates = jnp.stack((params[images.source_x_param_index], params[images.source_y_param_index]), -1)
        transported = gp_knee.transport_grouped_sources(coordinates, [p[4] for p in parts], groups, jnp.asarray(members.source_prior_mean),
                                                        members.source_prior_sigma)
        source = transported["source"]
        # The Gaussian term from the whitening's solves (L^-1 F = L^-1 b - L^-1 J beta): no third triangular solve.
        terms = [gp_knee.block_gaussian_loglike(cholesky, whitening, block_source, group["mask"], group["counts"])
                 for (group, _, cholesky, _, whitening), block_source in zip(parts, transported["block_sources"])]
        family_loglike, quadratic, logdet = (jnp.concatenate([t[i] for t in terms]) for i in range(3))
        counts = jnp.concatenate([group["counts"] for group in groups])
        block_finite = jnp.concatenate([jnp.all(jnp.isfinite(cholesky), axis=(-2, -1)) & whitening["valid"]
                                        for _, _, cholesky, _, whitening in parts])
        if self.gp_groups is None:
            family_covariance, family_cholesky, whitening = parts[0][1], parts[0][2], parts[0][4]
        else:
            family_covariance = family_cholesky = whitening = None                    # per group only; the init search needs none
        determinant_gradient = fields["determinant_gradient"]
        if self.freeze_determinant_gradient:
            determinant_gradient = jax.lax.stop_gradient(determinant_gradient)
        sigma_def = scales["sigma_def"]
        if getattr(self, "gp_project_modes", ()):
            # The smoothed determinant uses each image's own deflection variance, which the projection lowers (one block).
            diagonal = jnp.moveaxis(jnp.diagonal(projected_blocks[0], axis1=0, axis2=1), -1, 0)             # (M, 2, 2)
            variance = jnp.zeros(images.n_images, dtype=jnp.float64).at[layout["slots"][0]].set(0.5 * (diagonal[:, 0, 0] + diagonal[:, 1, 1]))
            sigma_def = jnp.sqrt(variance + jnp.square(sigma_fl))
        stats = sl.statistics(backprojected, source[images.family_index_per_image], jacobian, determinant_gradient,
                              scales["sigma_meas"], sigma_def, 0.0, 1.0, parity=images.parity)
        log_h = sl.log_expected_absolute_normal(stats["determinant_mean"], jnp.maximum(stats["determinant_variance"], 1e-16), images.parity)
        block_of_image = self.gp_block_position[layout["block_of_family"][images.family_index_per_image]]
        block_ok = block_finite & jnp.isfinite(family_loglike)
        share = (family_loglike / counts)[block_of_image]                      # the block term, split equally for the per-image plots
        valid = stats["valid"] & jnp.isfinite(log_h) & block_ok[block_of_image]
        stats.update(log_h=log_h, gaussian_loglike=share, loglike=share + log_h, valid=valid,
                     family_gaussian_loglike=family_loglike, family_quadratic=quadratic, family_logdet=logdet,
                     family_covariance=family_covariance, family_cholesky=family_cholesky,
                     source_by_family=source, source_correction=transported["log_correction"], source_jacobian=jnp.sum(transported["log_jacobian"]),
                     whitening=whitening, transport_valid=transported["valid"], fields=fields, physical=physical, **scales)
        return stats

    def _knee_mean_inputs(self, params: jnp.ndarray):
        """v = C^-1 (b - J beta) per block (S, M, 2) at this draw, and (sigma_alpha, lambda_t, gamma)."""
        layout = self.gp
        mask = layout["mask"]
        n_blocks, size = mask.shape
        stats = self.statistics(params)
        residual = gp_knee.block_residual(stats["fields"]["backprojected"][layout["slots"]], stats["source_by_family"], layout)
        v = jax.scipy.linalg.cho_solve((stats["family_cholesky"], True), residual.reshape(n_blocks, 2 * size)[..., None])[..., 0]
        v = jnp.where(mask[..., None], v.reshape(n_blocks, size, 2), 0.0)
        scales = self.error_scales(stats["physical"])
        return v, jnp.stack([scales["gp_sigma_alpha"], scales["gp_knee"], scales["gp_knee_gamma"]])

    def _padded_evaluator(self) -> "Evaluator":
        """This evaluator, or with grouped blocks a cached copy on one padded batch: the GP mean deflection reads the
        per-block Cholesky factors in the full block layout, which the grouped path does not assemble."""
        if getattr(self, "gp_groups", None) is None:
            return self
        if getattr(self, "_padded", None) is None:
            state = copy.copy(self.state)
            state.gp_group_blocks = False
            self._padded = Evaluator(state, self.likelihood_mode)
        return self._padded

    def gp_deflection_correction(self, params) -> "gp.MeanDeflection | gp_knee.MeanDeflectionKnee | None":
        """Potential kernel: the GP conditional mean of the deflection error at this draw (``gp.MeanDeflection``), which the
        image solver adds to the lens so that the draw's source and lens belong together; None under the other likelihoods.
        Under the potential kernel the sampled source scatters by the deflection error around the centroid, and solving
        with the lens alone puts the effective source off the caustic for fold pairs."""
        if not self.gp_potential:
            return None
        if self.gp_knee:
            if self.gp_sigma_alpha_index < 0 or self.gp_knee_spectrum != "knee":
                return None                                            # no mass field, or a spectrum the mean field does not cover
            if self.gp_groups is not None:
                return self._padded_evaluator().gp_deflection_correction(params)
            if not hasattr(self, "_knee_mean_inputs_jit"):
                self._knee_mean_inputs_jit = jax.jit(self._knee_mean_inputs)
            v, s = self._knee_mean_inputs_jit(jnp.asarray(params, dtype=jnp.float64))
            s = np.asarray(s, dtype=float)
            return gp_knee.MeanDeflectionKnee(self.gp, self.gp_tables, v, float(s[1]), float(s[0]) ** 2, self.gp_weights_per_family,
                                              gamma=float(s[2]) if self.gp_knee_gamma_index >= 0 else None,
                                              modes=tuple(getattr(self, "gp_project_modes", ())))
        if not hasattr(self, "_gp_mean_inputs_jit"):
            self._gp_mean_inputs_jit = jax.jit(self._gp_mean_inputs)
        v, s = self._gp_mean_inputs_jit(jnp.asarray(params, dtype=jnp.float64))
        s = np.asarray(s, dtype=float)
        terms = [(s[0] ** 2, s[1])] + ([(s[2] ** 2, s[3])] if self.gp_sigma_large_index >= 0 and s[2] > 0 else [])
        layout = self.gp
        weights = np.asarray(layout["weights"])[:, 0]                   # one redshift per family; slot 0 is always real
        return gp.MeanDeflection(np.asarray(self.xy)[np.asarray(layout["slots"])], layout["mask"], np.asarray(v), weights ** 2, terms,
                                 self.gp_kernel_family)

    def _gp_statistics(self, params: jnp.ndarray, physical: jnp.ndarray, fields: dict[str, Any]) -> dict[str, Any]:
        """gp_image_plane: one Cholesky factor of the family covariance serves the Gaussian term and the source conditional;
        the smoothed determinant h_i keeps its per-image form with the diagonal block C_ii as its covariance."""
        images, members, layout = self.state.images, self.state.members, self.gp
        scales = self.error_scales(physical)
        sigma_fl = jnp.asarray(self.gp_floor, dtype=jnp.float64)
        jacobian, backprojected = fields["jacobian"], fields["backprojected"]
        slots, mask, counts = layout["slots"], layout["mask"], layout["counts"]
        if self.gp_knee:
            return self._knee_statistics(params, physical, fields, scales)
        if self.gp_potential:
            blocks = self.gp_kernel_blocks(physical, scales)
            family_covariance, family_cholesky = gp.family_covariance_potential(jacobian[slots], blocks["k_alpha"], blocks["sigma_meas2"], mask,
                                                                                jnp.square(sigma_fl))
        else:
            kernel = layout["kernel"] if self.gp_ell_index < 0 else gp.kernel_from_separations(layout["squared_separations"], mask, scales["gp_ell"])
            family_covariance, family_cholesky = gp.family_covariance(jacobian[slots], kernel, layout["sigma_inst2"], mask,
                                                                      jnp.square(scales["sigma_shared"]), jnp.square(scales["sigma_white"]), jnp.square(sigma_fl))
        padded_backprojected = backprojected[slots]
        whitening = gp.family_source_whitening(family_cholesky, padded_backprojected, mask)
        coordinates = jnp.stack((params[images.source_x_param_index], params[images.source_y_param_index]), -1)
        transported = sl.transport_sources(coordinates, whitening, jnp.asarray(members.source_prior_mean), members.source_prior_sigma)
        source = transported["source"]
        family_loglike, quadratic, logdet = gp.family_gaussian_loglike(family_cholesky, padded_backprojected - source[:, None, :], mask, counts)
        determinant_gradient = fields["determinant_gradient"]
        if self.freeze_determinant_gradient:
            determinant_gradient = jax.lax.stop_gradient(determinant_gradient)
        # Per image: the diagonal block C_ii and the smoothed determinant on it.  image_plane: (sigma_inst^2 + sigma_white^2 +
        # sigma_shared^2) A A^T + sigma_fl^2 I (k_ii = 1); potential: sigma_meas^2 A A^T + (w_i^2 (sigma_S^2 + sigma_L^2) + sigma_fl^2) I.
        stats = sl.statistics(backprojected, source[images.family_index_per_image], jacobian, determinant_gradient,
                              scales["sigma_meas"], scales["sigma_def"], 0.0, 1.0, parity=images.parity)
        log_h = sl.log_expected_absolute_normal(stats["determinant_mean"], jnp.maximum(stats["determinant_variance"], 1e-16), images.parity)
        family_of = images.family_index_per_image
        family_ok = jnp.all(jnp.isfinite(family_cholesky), axis=(-2, -1)) & jnp.isfinite(family_loglike) & whitening["valid"]
        share = (family_loglike / counts)[family_of]                                 # the family term, split equally for the per-image plots
        valid = stats["valid"] & jnp.isfinite(log_h) & family_ok[family_of]
        stats.update(log_h=log_h, gaussian_loglike=share, loglike=share + log_h, valid=valid,
                     family_gaussian_loglike=family_loglike, family_quadratic=quadratic, family_logdet=logdet,
                     family_covariance=family_covariance, family_cholesky=family_cholesky,
                     source_by_family=source, source_correction=transported["log_correction"], source_jacobian=jnp.sum(transported["log_jacobian"]),
                     whitening=whitening, transport_valid=transported["valid"], fields=fields, physical=physical, **scales)
        return stats

    def log_likelihood(self, params: jnp.ndarray, *, source_jacobian: bool = True) -> jnp.ndarray:
        """Log-likelihood plus the source transport correction, or BAD_LOG_LIKE / -inf when rejected.

        Rejected proposals are evaluated at the known-valid initial model so that every
        branch stays finite; the selected value is then the rejection constant with zero gradient.

        ``source_jacobian=False`` leaves out the volume factor log|det L_beta| of the source transport, so the target is
        the posterior density in the physical source coordinates instead of the unit offsets.  The init search uses it:
        the two densities have the same marginal over the lens parameters but not the same maximum, and the volume
        factor grows without bound with the magnification, which gives the joint density a spurious mode at a lens so
        massive that every image is demagnified (M0416, 256 starts: +393 against -1213 at the physical mode).
        """
        reference = jnp.asarray(self.state.initial_theta, dtype=jnp.float64)
        physical = self.physical(params)
        _, validity = self.packed_state(physical)
        ok_model = validity.is_valid & jnp.all(jnp.isfinite(params))
        scales = physical[self.scale_indices]
        ok_scales = jnp.all(jnp.isfinite(scales ** 2) & (scales > 0.0))
        ok = ok_model & ok_scales
        safe_params = jnp.where(ok, params, reference)
        stats = self.statistics(safe_params)
        value = jnp.sum(stats["loglike"]) + stats["source_correction"]
        value = jnp.sum(stats["loglike"]) + stats["source_correction"] + self.kinematics_log_likelihood(self.physical(safe_params))
        if not source_jacobian:
            value = value - stats["source_jacobian"]
        value = value + self.far_field_log_barrier(stats["fields"]["far_cut_radii"])
        valid = jnp.all(stats["valid"]) & jnp.all(stats["transport_valid"]) & ~jnp.isnan(value)
        value = jnp.where(valid, value, -jnp.inf)
        return jnp.where(ok_model, jnp.where(ok_scales, value, -jnp.inf), jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))

    # ------------------------------------------------------------------ reporting
    def reported_physical(self, params: jnp.ndarray) -> jnp.ndarray:
        """Physical parameters with the sampled source offsets replaced by physical source positions."""
        physical = self.physical(params)
        source = self.statistics(params)["source_by_family"]
        images = self.state.images
        return physical.at[images.source_x_param_index].set(source[:, 0]).at[images.source_y_param_index].set(source[:, 1])

    def latent_from_reported_physical(self, physical: np.ndarray) -> np.ndarray:
        """Invert reported_physical: whitened source offsets from physical source positions."""
        from .model import convert_theta_to_latent
        physical = np.asarray(physical, dtype=float)
        latent = convert_theta_to_latent(physical, self.state.parameter_specs)
        images = self.state.images
        ix, iy = np.asarray(images.source_x_param_index), np.asarray(images.source_y_param_index)
        latent[ix] = 0.0
        latent[iy] = 0.0
        whitening = self.statistics(jnp.asarray(latent))["whitening"]
        offsets = np.asarray(sl.inverse_source_transport(np.stack((physical[ix], physical[iy]), -1), whitening))
        latent[ix], latent[iy] = offsets[:, 0], offsets[:, 1]
        return latent

    def source_positions(self, params: jnp.ndarray) -> jnp.ndarray:
        """Physical source positions per family, (F, 2); traceable under jit and vmap."""
        return self.statistics(params)["source_by_family"]
