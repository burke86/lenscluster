"""Configuration for the single-stage smoothed-local-image-plane solver."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPathsConfig:
    output_dir: str | Path = "results"
    run_name: str | None = None
    figure_prefix: str | None = None


@dataclass(frozen=True)
class PriorConfig:
    kind: str = "fixed"
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class LogNormalPriorConfig:
    """Positive prior, parameterized by median and log-space width; ``upper`` (same units as the median) truncates it,
    which keeps a sampled error scale from inflating until the likelihood goes flat."""
    median: float
    log_sigma: float
    upper: float | None = None


@dataclass(frozen=True)
class ReferenceFrameConfig:
    reference: int = 3
    ra0_deg: float = 0.0
    dec0_deg: float = 0.0


@dataclass(frozen=True)
class CosmologyConfig:
    H0: float = 70.0
    Om0: float = 0.3
    Ode0: float | None = None
    w0: float = -1.0
    class_name: str = "FlatLambdaCDM"


@dataclass(frozen=True)
class DPIEHaloConfig:
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_kpc: float
    cut_radius_kpc: float
    v_disp: float
    z_lens: float
    profile_type: int = 81
    priors: dict[str, PriorConfig] = field(default_factory=dict)

    @classmethod
    def cluster_scale(cls, id: str, x_centre: float, y_centre: float, z_lens: float, *, v_disp: float = 800.0,
                      position_half_width_arcsec: float = 5.0, ellipticity_upper: float = 0.8) -> "DPIEHaloConfig":
        """A cluster-scale dPIE with the standard priors: centre within a box around the light, free ellipticity and
        angle, core 0.5-100 kpc, cut radius TN(1500, 300) kpc on 500-3000 kpc (the arcs do not constrain it), and a
        dispersion prior of 35% width around the given value."""
        half = float(position_half_width_arcsec)
        return cls(id=str(id), x_centre=float(x_centre), y_centre=float(y_centre), ellipticite=0.3, angle_pos=0.0, core_radius_kpc=10.0,
                   cut_radius_kpc=1500.0, v_disp=float(v_disp), z_lens=float(z_lens), priors={
                       "x_centre": PriorConfig("uniform", lower=x_centre - half, upper=x_centre + half, step=0.5),
                       "y_centre": PriorConfig("uniform", lower=y_centre - half, upper=y_centre + half, step=0.5),
                       "ellipticite": PriorConfig("uniform", lower=0.0, upper=float(ellipticity_upper), step=0.02),
                       "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                       "core_radius_kpc": PriorConfig("uniform", lower=0.5, upper=100.0, step=1.0),
                       "cut_radius_kpc": PriorConfig("truncated_normal", mean=1500.0, std=300.0, lower=500.0, upper=3000.0),
                       "v_disp": PriorConfig("truncated_normal", mean=float(v_disp), std=0.35 * float(v_disp), lower=100.0, upper=2200.0)})


@dataclass(frozen=True)
class NFWHaloConfig:
    """Elliptical NFW cluster-scale halo: mass M200 (log10 Msun, no little h) and scale radius in kpc; ellipticity is
    in the convergence with the Lenstool convention of ``ellipticite``."""
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    log10_m200: float
    scale_radius_kpc: float
    z_lens: float
    profile_type: int = 12
    priors: dict[str, PriorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MassSheetConfig:
    """Uniform external convergence at the lens redshift.  ``kappa`` is the value for a source at infinite redshift;
    each family sees kappa x D_ls / D_s.  Multiple source redshifts constrain it; a single one cannot (mass-sheet degeneracy)."""
    id: str
    kappa: float
    z_lens: float
    profile_type: int = 101
    priors: dict[str, PriorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ShearHaloConfig:
    id: str
    gamma: float
    angle_pos: float
    z_lens: float
    profile_type: int = 14
    priors: dict[str, PriorConfig] = field(default_factory=dict)

    @classmethod
    def free(cls, id: str, z_lens: float, *, gamma_upper: float = 0.3) -> "ShearHaloConfig":
        """External shear with a free amplitude up to ``gamma_upper`` and a free angle."""
        return cls(id=str(id), gamma=0.0, angle_pos=0.0, z_lens=float(z_lens), priors={
            "gamma": PriorConfig("uniform", lower=0.0, upper=float(gamma_upper), step=0.005),
            "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=0.5)})


@dataclass(frozen=True)
class IndependentMemberHaloConfig:
    population_id: str
    catalog_id: str
    id: str | None = None
    priors: dict[str, PriorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MemberPopulationConfig:
    id: str
    catalog_path: str | Path
    mag0: float | str                        # pivot magnitude, or "auto": the image-leverage-weighted member magnitude, rounded
    corekpc: float
    sigma: float
    cutkpc: float
    z_lens: float
    profile_type: int = 81
    sigma_prior: PriorConfig = field(default_factory=PriorConfig)
    cutkpc_prior: PriorConfig = field(default_factory=PriorConfig)
    exclude_catalog_ids: tuple[str, ...] = ()
    prior_mag0: float | None = None          # magnitude at which sigma, cutkpc and their priors are given when mag0 is "auto"

    @classmethod
    def bergamini2019(cls, id: str, catalog_path: str | Path, z_lens: float, *, prior_mag0: float, sigma: float = 200.0,
                      cutkpc: float = 12.0, corekpc: float = 0.15) -> "MemberPopulationConfig":
        """A member population on the MUSE-calibrated relation of Bergamini et al. 2019 at L* (``prior_mag0``): central
        dispersion ~220 km/s and r_cut ~10 (+13, -7) kpc from their three clusters, widened to TN(200, 70) km/s on 40-500
        and TN(12, 10) kpc on 2-60; the pivot is chosen automatically from image leverage.  Pair with
        ScalingModelConfig.bergamini2019() for the slopes."""
        return cls(id=str(id), catalog_path=catalog_path, mag0="auto", prior_mag0=float(prior_mag0), corekpc=float(corekpc),
                   sigma=float(sigma), cutkpc=float(cutkpc), z_lens=float(z_lens),
                   sigma_prior=PriorConfig("truncated_normal", mean=float(sigma), std=0.35 * float(sigma), lower=40.0, upper=500.0),
                   cutkpc_prior=PriorConfig("truncated_normal", mean=float(cutkpc), std=10.0, lower=2.0, upper=60.0))

    @classmethod
    def neutral(cls, id: str, catalog_path: str | Path, z_lens: float, *, prior_mag0: float, corekpc: float = 0.15) -> "MemberPopulationConfig":
        """A member population with priors that do not come from any cluster's lens model, at L* (``prior_mag0``):
        dispersion TN(200, 100) km/s on 50-500 (an L* early type in any Faber-Jackson calibration) and a cut radius
        uniform in log between 2 and 100 kpc (lensing barely constrains it, so scale-free).  Pair with
        ScalingModelConfig.neutral() for the slopes."""
        return cls(id=str(id), catalog_path=catalog_path, mag0="auto", prior_mag0=float(prior_mag0), corekpc=float(corekpc),
                   sigma=200.0, cutkpc=15.0, z_lens=float(z_lens),
                   sigma_prior=PriorConfig("truncated_normal", mean=200.0, std=100.0, lower=50.0, upper=500.0),
                   cutkpc_prior=PriorConfig("uniform", lower=2.0, upper=100.0))        # uniform in the log sampling coordinate


@dataclass(frozen=True)
class ImageConstraintsConfig:
    catalog_path: str | Path                 # YAML image catalog (lenscluster.image_catalog); convert Lenstool files with scripts/convert_lenstool_catalog.py
    sigma_inst_arcsec: float = 0.1           # astrometric error of images whose catalog entry gives none (sigma_inst)
    sigma_floor_arcsec: float = 0.0


@dataclass(frozen=True)
class LensModelConfig:
    reference: ReferenceFrameConfig = field(default_factory=ReferenceFrameConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    large_halos: tuple[DPIEHaloConfig | NFWHaloConfig | ShearHaloConfig | MassSheetConfig, ...] = ()
    member_populations: tuple[MemberPopulationConfig, ...] = ()
    independent_member_halos: tuple[IndependentMemberHaloConfig, ...] = ()
    image_constraints: ImageConstraintsConfig | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 12345
    chains: int = 4
    resume: bool = True
    skip_plots: bool = False
    quiet: bool = False
    nuts_chain_method: str = "parallel"
    cpu_threads_per_chain: int | None = None
    numpyro_print_summary: bool = False


@dataclass(frozen=True)
class SearchConfig:
    """The init search, in two stages: the large-scale model without the member population from ``n_starts`` prior
    starts (``halo_method``); then, from each carried mode, the members with the large family frozen, started from
    ``member_starts`` prior draws of the member block (``member_method``); then a full-dimension L-BFGS polish of the
    modes that can seed chains.  Methods: "lbfgs" (scipy L-BFGS-B per start), "optax-lbfgs" (vmapped optax L-BFGS,
    for GPUs), "adam" (vmapped Adam) or "svi" (vmapped mean-field Gaussian guide, the guide mean as the estimate)."""
    n_starts: int = 32
    halo_method: str = "lbfgs"
    member_method: str = "lbfgs"
    member_starts: int = 8
    lbfgs_max_iter: int = 400
    # Screening (method "lbfgs"): every start runs screen_iterations first; only the screen_keep best continue to
    # lbfgs_max_iter in total, from where they stopped.  None runs every start to the end.  On CANUCS (256 starts) 50/16
    # reaches the same best stage-A mode as running all starts (log posterior equal to 1e-4) in 54 s instead of 201 s.
    screen_iterations: int | None = 50
    screen_keep: int = 16
    # Converged chain starts: after the final classification, the chain-start modes are optimized together by the batched
    # optax L-BFGS on the posterior NUTS samples (all parameters free, source-transport volume included) for this many
    # iterations; starts still more than mode_margin below the best are dropped.  0 skips it.  On CANUCS
    # (w500s500_fast_fixed_samplek) scipy stopped after one iteration at a far-field guard; optax lifted the second start
    # by 119 in 100 iterations (most of its gain), at about 4 gradients per iteration plus ~2 min of compilation.
    nuts_start_max_iter: int = 200
    # Warm start: a previous run directory (or its init/artifacts); its saved start, mapped by parameter name, replaces
    # stages A and B, and the polish runs from it.  Parameters it lacks start at the configured model.
    warm_start: str | None = None
    adam_steps: int = 2000
    adam_learning_rate: float = 0.05
    svi_steps: int = 2000
    svi_learning_rate: float = 0.05
    polish_max_iter: int = 2000             # the only full-dimension optimization: AS1063 needs 400 iterations to reach the joint
                                            # optimum, M0416 passes the one-stage multistart after 2000 (400 left it 27 short)
    polish_scatter: float | None = 0.1      # pin the member scatter (ln sigma) at this value while optimizing; None keeps the prior median
    mode_margin: float = 10.0
    mode_parameter_tolerance: float = 0.05
    max_chain_modes: int = 4
    max_median_abs_determinant: float = 10.0  # a mode whose median |det A| at the images exceeds this is degenerate (a lens so
                                            # massive that the images are demagnified) and cannot seed chains; cluster lenses sit near 0.2
    # Gates checked before NUTS (the final stage stops with QualityGateError): a fragile init (at most one start reaches the
    # best basin), or an error hyperparameter within this log distance of its prior upper bound at the start.
    fail_on_fragile_init: bool = True
    hyperparameter_bound_margin: float = 0.05     # 0 turns the bound check off


@dataclass(frozen=True)
class SamplingConfig:
    warmup: int = 250
    samples: int = 200
    max_tree_depth: int = 8
    target_accept: float = 0.85
    initial_step_size: float = 1.0e-3
    thin: int = 1
    mass_matrix: str = "hessian"            # "hessian": dense global block seeded by the Hessian; "dense": dense global block, adapted
                                            # from the identity; "diagonal": NumPyro's default diagonal adaptation
    # Gates checked after the outputs are written (the run then fails with QualityGateError): the largest split R-hat over
    # all parameters, the fraction of divergent transitions, and the fraction of observed images recovered at the best fit.
    max_r_hat: float = 1.1
    max_divergence_fraction: float = 0.01
    min_image_recovery_fraction: float = 0.9


@dataclass(frozen=True)
class MemberConfig:
    classification_prior_quantile: float | None = 0.975   # classify with the cut radii at this quantile of the cutkpc and beta priors
                                                          # (and never below the current model); None uses the current model only
    centering: str = "auto"                 # which members sample their deviation directly (centered) rather than a unit offset:
                                            # "auto" (data curvature of the deviation exceeds the prior's 1/s^2 at the start), "exact", "none"
    centering_ratio: float = 1.0            # auto threshold on data/prior curvature
    """Member classification and far-field validity."""
    far_field_cut_radii: float = 3.0        # a far-field member stays at least this many of its (sampled) cut radii from every image
    cut_radius_inflation: float = 3.0       # members are classified with this multiple of the current cut radius (or the prior quantile
                                            # below, whichever is larger), so the sampled r_cut* can grow by that factor before any
                                            # far-field member reaches the guard
    template_tolerance: float = 0.1
    top_exact: int | None = None
    # The guard is a smooth log-barrier during sampling: -strength * softplus(-x / width) per far-field member with
    # x = ln(R / (guard r_cut)), so NUTS feels the boundary and turns back instead of diverging on a hard wall (about
    # -0.7 strength at the guard, -1e-2 strength one width outside, linear in -x inside).
    far_field_barrier_width: float = 0.05
    far_field_barrier_strength: float = 25.0
    min_deflection_arcsec: float | None = None   # drop members whose deflection at every image, at the initial model, is below this


@dataclass(frozen=True)
class ScalingModelConfig:
    # Members leave the relation in dispersion (tight scatter: Faber-Jackson is about 0.1 dex) and, for the exact members,
    # in cut radius (loose: tidal truncation depends on the orbit).  With a dispersion deviation alone the sampler turned
    # the biggest subhalos into point masses to patch mass the cluster halos lacked.
    member_velocity_scatter_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.10, log_sigma=0.3))
    member_deviation: str = "sigma_rcut"    # how a member leaves the relation: "sigma" (dispersion only), "mass" (dispersion and radii
                                            # together along the relation, one scatter), "sigma_rcut" (independent dispersion and radius
                                            # deviations), "mass_rcut" (mass deviation along the relation plus an independent radius deviation)
    member_radius_scatter_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.3, log_sigma=0.5))
    radius_deviation_members: str = "exact"  # which members' radius deviations act on the model: "exact" (far-field members keep
                                            # the relation radius, so they cannot cross the far-field guard) or "all"
    fit_corekpc: bool = True                # sample the member core radius at the pivot; False fixes it at the population's corekpc
                                            # moved to the pivot (a 0.1 kpc member core is not a lensing observable)
    # Faber-Jackson slope 0.27-0.28 and beta_cut 0.64-0.66 in three MUSE-calibrated clusters (Bergamini et al. 2019).
    alpha_sigma_prior_mean: float = 0.27
    alpha_sigma_prior_std: float = 0.05
    alpha_sigma_prior_lower: float = 0.05
    alpha_sigma_prior_upper: float = 0.50
    beta_radius_prior_mean: float = 0.65
    beta_radius_prior_std: float = 0.15
    beta_radius_prior_lower: float = 0.0
    beta_radius_prior_upper: float = 1.5
    softening_length_kpc: float = 0.0
    softening_length_prior_log_sigma: float = 0.15

    @classmethod
    def bergamini2019(cls, **overrides) -> "ScalingModelConfig":
        """The slope priors of the defaults: alpha 0.27 +- 0.05 and beta 0.65 +- 0.15 (Bergamini et al. 2019)."""
        return cls(**overrides)

    @classmethod
    def neutral(cls, **overrides) -> "ScalingModelConfig":
        """Slope priors from scaling arguments only: alpha 0.25 +- 0.10 (L ~ sigma^4) on 0-0.5 and beta 0.5 +- 0.25
        (constant mass-to-light, 2 alpha + beta = 1) on 0-1.5."""
        return cls(alpha_sigma_prior_mean=0.25, alpha_sigma_prior_std=0.10, alpha_sigma_prior_lower=0.0, alpha_sigma_prior_upper=0.5,
                   beta_radius_prior_mean=0.5, beta_radius_prior_std=0.25, beta_radius_prior_lower=0.0, beta_radius_prior_upper=1.5, **overrides)


LIKELIHOOD_MODES = frozenset({"smoothed", "gp_image_plane", "source-anisotropic", "source-isotropic"})


@dataclass(frozen=True)
class LikelihoodConfig:
    # "smoothed" (local image-plane with per-image source-plane floors, the model of the paper); "gp_image_plane" (correlated
    # image-plane displacements within each family, no source-plane floor); diagnostics only: "source-anisotropic",
    # "source-isotropic".
    likelihood_mode: str = "smoothed"
    # Source-plane positional covariance per image, C = sigma_meas^2 A A^T + sigma_def^2 I + sigma_sel^2 R (the paper's
    # Eq. position_covariance), with sigma_meas^2 = sigma_inst^2 + sigma_extra^2.
    sigma_inst_arcsec: float | None = None    # overrides the catalog's instrumental astrometric error for every image
    # gp_image_plane only.  Image-plane displacement covariance per family
    # S = sigma_shared^2 (K kron I) + (sigma_inst,i^2 + sigma_white^2) I with the Gaussian kernel K_ij = exp(-|theta_i - theta_j|^2 / 2 ell^2);
    # source-plane covariance C = B S B^T + sigma_fl^2 I with B = blockdiag(A_i).  sigma_shared (log-normal) and sigma_white
    # (half-normal) are sampled and replace sigma_extra, sigma_def and sigma_sel; ell and sigma_fl are fixed.  The prior on
    # sigma_shared is truncated at 1": the untruncated prior puts 7e-5 of its mass there, and a 32" value carried a spurious init mode.
    sigma_shared_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.15, log_sigma=0.5, upper=1.0))
    sigma_white_prior_scale_arcsec: float = 0.05  # HalfNormal scale of sigma_white
    sigma_shared_start_arcsec: float = 0.168      # start of the init search (the sigma_img posterior median of the oct31a M0416 run)
    sigma_white_start_arcsec: float = 0.02
    gp_correlation_length_arcsec: float = 7.0     # ell: the fixed value, or the start of the init search when ell is sampled
    # ell is sampled (log-normal; this prior) unless the prior is None, which fixes ell at gp_correlation_length_arcsec.  The
    # width 0.5 puts 68% of the prior between 4.2" and 11.5"; only the ~40 same-family pairs separated by 3" to 15" constrain
    # ell, and above the 30" truncation every family's kernel is a block of ones, where ell is unidentified.
    gp_correlation_length_prior: LogNormalPriorConfig | None = field(default_factory=lambda: LogNormalPriorConfig(median=7.0, log_sigma=0.5, upper=30.0))
    gp_floor_arcsec: float = 0.0015               # sigma_fl, a smoothness floor only (1.5 mas)
    # gp_image_plane only.  gp_kernel selects the structure of the GP term: "image_plane" (above: an image-plane displacement
    # multiplied by the Jacobian, C = B S B^T) or "potential" (a potential error delta_psi; its gradient is the deflection
    # error, which enters the source plane directly: C = W K_alpha W + B Sigma_meas B^T + sigma_fl^2 I with
    # Cov(delta_alpha_a, delta_alpha_b) = -d_a d_b k_psi, see gp_image_plane.deflection_kernel).  Under "potential" the sampled
    # sites are gp_u_small = sigma_S / ell_S (log-normal; sigma_kappa,S / sqrt 2 for the Gaussian family), gp_ell_small
    # (log-normal), gp_sigma_large (half-normal; None drops the large scale) and sigma_extra (sigma_extra_prior; the image-plane
    # measurement term); ell_L is fixed.  sigma_shared, sigma_white and gp_ell do not exist under "potential".  Both scales
    # use the kernel family gp_kernel_family: "gaussian" or "matern52" (Matern-5/2 on the potential, Matern-3/2 deflection).
    # The redshift weight w = (D_ds/D_s)(z) / (D_ds/D_s)(gp_reference_redshift) scales each family's deflection error.
    gp_kernel: str = "image_plane"
    gp_kernel_family: str = "gaussian"
    gp_u_small_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.015, log_sigma=1.1))
    gp_u_small_start: float = 0.015
    gp_ell_small_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=2.0, log_sigma=0.5))
    gp_ell_small_start_arcsec: float = 2.0
    gp_sigma_large_prior_scale_arcsec: float | None = 0.02   # HalfNormal scale of sigma_L (source plane, at z_ref)
    gp_sigma_large_start_arcsec: float = 0.01
    gp_ell_large_arcsec: float = 25.0
    gp_reference_redshift: float = 2.0
    # gp_kernel = "knee" (gp_knee.py, .agents/gp-knee-plan.md): a curl-free mass field on the potential with a knee convergence
    # spectrum P_kappa = P0 / (1 + (k/k_t)^gamma) between gp_knee_scales, plus the image-plane GP, on blocks given by gp_blocks:
    # "all" (one covariance over every image), "system" (all knots of a galaxy; run.xsh --fast) or "family" (regression).
    # Sites: images.gp_sigma_alpha (rms deflection error per component at z_ref; None drops the mass field), images.gp_knee
    # (lambda_t = 2 pi / k_t), images.gp_knee_gamma (the slope gamma, uniform on gp_knee_gamma_prior; None fixes it at
    # gp_knee_gamma), images.sigma_shared, images.gp_ell (image-plane GP), images.sigma_white.  sigma_fl fixed.
    # The uniform bounds 2-6 are the slopes of P_kappa below the truncation of a random population of halos with
    # rho ~ r^-n for n = 2 (isothermal) to 0 (cored); n = 1 (NFW cusp) gives 4, the start.
    gp_blocks: str = "all"
    gp_knee_gamma: float = 4.0                                        # fixed slope, or the start when it is sampled
    gp_knee_gamma_prior: tuple[float, float] | None = (2.0, 6.0)      # uniform bounds on the sampled slope; None fixes it
    gp_knee_gamma_step: float = 0.25                                  # spacing of the tabulated slopes (cubic spline between)
    # Wavelength range of the GRF (arcsec).  A 30" long end was tried (2026-11-02) when the shear seemed absorbed by the GRF;
    # at a fixed lens it cost 12.6 in log L and pushed the image-plane GP to its 5" cap, and the shear was in fact taken
    # by swapped halos, not by the GRF (its own shear over the images is ~0.002), so the long end is back at 100".
    gp_knee_scales_arcsec: tuple[float, float] = (0.2, 100.0)
    # Low-order modes removed from the mass field over the images ("translation", "convergence", "shear"; empty = none):
    # the residual projection of gp_knee.project_low_order.  Under the flat shear prior it leaves the likelihood unchanged
    # and makes the external shear the total shear over the images.  gp_blocks "all" only.
    gp_knee_project_modes: tuple[str, ...] = ()
    gp_knee_lambda_grid: tuple[float, float, int] = (0.5, 80.0, 64)   # tabulated knee wavelengths: min, max, count (ln lambda clamped to it)
    gp_knee_r_step_arcsec: float = 0.02
    # Upper bounds (truncations) keep the error model from replacing the lens: a deflection error above 0.3" is a missing
    # cluster component, not substructure; a knee above 30" is the scale of the cluster halos and the external shear; an
    # image-plane displacement above 0.3" or coherent over more than 5" is lensing, not astrometry or feature selection.
    gp_sigma_alpha_prior: LogNormalPriorConfig | None = field(default_factory=lambda: LogNormalPriorConfig(median=0.08, log_sigma=0.7, upper=0.3))
    gp_sigma_alpha_start_arcsec: float = 0.08
    gp_knee_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=6.0, log_sigma=0.5, upper=30.0))
    gp_knee_start_arcsec: float = 6.0
    gp_knee_sigma_shared_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.10, log_sigma=0.7, upper=0.3))
    gp_knee_sigma_shared_start_arcsec: float = 0.10
    gp_knee_ell_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=2.0, log_sigma=0.5, upper=5.0))
    gp_knee_ell_start_arcsec: float = 2.0
    gp_knee_sigma_white_prior_scale_arcsec: float = 0.03
    gp_knee_sigma_white_start_arcsec: float = 0.012
    # Extra measurement scatter (arcsec, log-normal) added in quadrature to sigma_inst: image registration, feature
    # centring, manual marking.  None fixes the measurement error at the catalog value.
    sigma_extra_prior: LogNormalPriorConfig | None = field(default_factory=lambda: LogNormalPriorConfig(median=0.05, log_sigma=1.0, upper=1.0))
    # Residual deflection error, isotropic in the source plane (arcsec, log-normal).
    sigma_def_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.1, log_sigma=1.0, upper=1.0))
    # Feature-selection scatter sigma_sel (arcsec, log-normal) in the source plane, weighted by the stretching through
    # R = (I + A A^T / lambda_sel^2)^-1 (the paper's Eq. selection_weight): the selection variance is halved where the
    # absolute Jacobian eigenvalue equals lambda_sel.  None disables the term.
    sigma_sel_prior: LogNormalPriorConfig | None = field(default_factory=lambda: LogNormalPriorConfig(median=0.04, log_sigma=1.0, upper=1.0))
    # The dimensionless stretching scale lambda_sel of the selection term (log-normal), sampled with sigma_sel.
    lambda_sel_prior: LogNormalPriorConfig = field(default_factory=lambda: LogNormalPriorConfig(median=0.7, log_sigma=1.0))
    source_position_prior_mean_arcsec: tuple[float, float] = (0.0, 0.0)
    source_position_prior_sigma_arcsec: float = 100.0
    # Hold grad det A (the third derivatives of the potential) fixed in the backward pass.  The log-likelihood is unchanged;
    # its gradient differs by 1e-4 relative (AS1063, M0416) and costs 15-35% less, since the reverse pass through the third
    # derivatives dominates.  NUTS stays exact (the energy is exact and the leapfrog map stays reversible and volume
    # preserving); only the integrator's accuracy, hence the acceptance rate, can suffer.
    freeze_determinant_gradient: bool = False


@dataclass(frozen=True)
class ImageDiagnosticsConfig:
    fold_pair_max_separation_arcsec: float = 5.0  # automatic, model-verified local fold candidates
    posterior_image_diagnostic_draws: int = 32
    exact_image_min_distance_arcsec: float = 0.5
    exact_image_precision_limit: float = 1.0e-2
    exact_image_num_iter_max: int = 100
    exact_image_finder: str = "global-jax"      # best-fit finder: "global-jax" (grid seeds over each family window, one batched solve,
                                                # reports extra images), "local-lm-adaptive", "local-lm" or "lenstronomy"
    exact_image_grid_pitch_arcsec: float = 0.5        # coarse grid of the global search over each family window
    exact_image_fine_pitch_arcsec: float = 0.1        # sub-grid seeded inside near-source cells close to a critical curve
    exact_image_critical_det_threshold: float = 0.25  # |det A| below which a cell counts as close to a critical curve
    exact_image_seed_tolerance_pitches: float = 3.0   # a cell is near the source when |beta - source| is within this many coarse pitches
    exact_image_window_margin_arcsec: float = 3.0     # added around each family's image span
    exact_image_min_separation_arcsec: float = 0.05   # two roots closer than this are one image
    exact_image_displacement_tol_arcsec: float = 1.0e-4
    exact_image_identification_tol_arcsec: float = 1.0e-3
    exact_image_lm_max_iter: int = 100
    exact_image_lm_trust_radius_arcsec: float = 1.0
    exact_image_adaptive_max_levels: int = 8
    match_tolerance_arcsec: float = 5.0
    posterior_diagnostic_draws: int = 32  # additional joint field draws; zero reports the best fit only


@dataclass(frozen=True)
class TruthRecoveryConfig:
    kappa_true_fits: str | Path | None = None
    gammax_true_fits: str | Path | None = None
    gammay_true_fits: str | Path | None = None
    posterior_truth_recovery_draws: int = 16
    truth_grid_size: int = 256
    caustic_source_redshift: float = 9.0


@dataclass(frozen=True)
class RGBDisplayConfig:
    q: float | None = None
    stretch: float | None = None
    minimum: float | None = None
    red_gain: float | None = None
    green_gain: float | None = None
    blue_gain: float | None = None


@dataclass(frozen=True)
class ImageCatalogCutoutConfig:
    image_dir: str | Path | None = None
    image_scale: str = "60mas"
    bands: tuple[str, ...] | None = None
    figure_suffix: str = ""                  # appended to the cluster mass figure name, e.g. "_jwst"
    rgb: RGBDisplayConfig = field(default_factory=RGBDisplayConfig)
    mode: str = "full"
    dpi: int | None = None
    max_side_pixels: int | None = None
    critical_lines: str = "auto"
    cutouts: bool = True


@dataclass(frozen=True)
class LensClusterSolverConfig:
    model: LensModelConfig | None = None
    paths: RunPathsConfig = field(default_factory=RunPathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    members: MemberConfig = field(default_factory=MemberConfig)
    scaling: ScalingModelConfig = field(default_factory=ScalingModelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    image_diagnostics: ImageDiagnosticsConfig = field(default_factory=ImageDiagnosticsConfig)
    truth: TruthRecoveryConfig = field(default_factory=TruthRecoveryConfig)
    image_catalog: ImageCatalogCutoutConfig = field(default_factory=ImageCatalogCutoutConfig)

    def with_updates(self, **updates: Any) -> "LensClusterSolverConfig":
        return replace(self, **updates)

    def validate(self) -> "LensClusterSolverConfig":
        validate_config(self)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_run_label(label: str) -> str:
    if not label or label.strip() != label or label in {".", ".."} or any(c in label for c in "/\\\0\n\r"):
        raise ValueError("run label must be a nonempty single directory name")
    return label


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(float(value)) or value < 0.0 or (not allow_zero and value <= 0.0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}.")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_config(config: LensClusterSolverConfig) -> None:
    validate_run_label(config.paths.run_name or "")
    _validate_model_config(config.model)
    runtime = config.runtime
    if isinstance(runtime.seed, bool) or not isinstance(runtime.seed, Integral) or int(runtime.seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")
    _positive_int("chains", runtime.chains)
    if runtime.nuts_chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("nuts_chain_method must be 'parallel', 'sequential', or 'vectorized'.")
    if runtime.cpu_threads_per_chain is not None:
        _positive_int("cpu_threads_per_chain", runtime.cpu_threads_per_chain)
    search = config.search
    _positive_int("search.n_starts", search.n_starts)
    _positive_int("search.lbfgs_max_iter", search.lbfgs_max_iter)
    for label in ("halo_method", "member_method"):
        if getattr(search, label) not in {"lbfgs", "optax-lbfgs", "adam", "svi"}:
            raise ValueError(f"search.{label} must be 'lbfgs', 'optax-lbfgs', 'adam' or 'svi'.")
    _positive_int("search.member_starts", search.member_starts)
    _positive_int("search.adam_steps", search.adam_steps)
    _positive_int("search.svi_steps", search.svi_steps)
    _positive("search.adam_learning_rate", search.adam_learning_rate)
    _positive("search.svi_learning_rate", search.svi_learning_rate)
    _positive_int("search.polish_max_iter", search.polish_max_iter)
    if search.polish_scatter is not None:
        _positive("search.polish_scatter", search.polish_scatter)
    _positive("search.mode_margin", search.mode_margin)
    _positive("search.mode_parameter_tolerance", search.mode_parameter_tolerance)
    _positive_int("search.max_chain_modes", search.max_chain_modes)
    _positive("search.max_median_abs_determinant", search.max_median_abs_determinant)
    sampling = config.sampling
    for name in ("warmup", "samples", "max_tree_depth", "thin"):
        _positive_int(f"sampling.{name}", getattr(sampling, name))
    if not 0.0 < sampling.target_accept < 1.0:
        raise ValueError("sampling.target_accept must lie between 0 and 1.")
    _positive("sampling.initial_step_size", sampling.initial_step_size)
    members = config.members
    _positive("members.far_field_cut_radii", members.far_field_cut_radii)
    if members.cut_radius_inflation < 1.0 or not math.isfinite(members.cut_radius_inflation):
        raise ValueError("members.cut_radius_inflation must be at least one.")
    if not 0.0 < members.template_tolerance < 1.0:
        raise ValueError("members.template_tolerance must lie between zero and one.")
    _positive("members.far_field_barrier_width", members.far_field_barrier_width)
    _positive("members.far_field_barrier_strength", members.far_field_barrier_strength)
    if members.top_exact is not None and (isinstance(members.top_exact, bool) or not isinstance(members.top_exact, Integral) or members.top_exact < 0):
        raise ValueError("members.top_exact must be a nonnegative integer or None.")
    if members.min_deflection_arcsec is not None:
        _positive("members.min_deflection_arcsec", members.min_deflection_arcsec)
    scaling = config.scaling
    if config.members.centering not in {"auto", "exact", "none"}:
        raise ValueError("members.centering must be 'auto', 'exact' or 'none'.")
    _positive("members.centering_ratio", config.members.centering_ratio)
    if config.members.classification_prior_quantile is not None and not (0.5 < float(config.members.classification_prior_quantile) < 1.0):
        raise ValueError("members.classification_prior_quantile must lie in (0.5, 1).")
    if config.likelihood.likelihood_mode not in LIKELIHOOD_MODES:
        raise ValueError(f"likelihood.likelihood_mode must be one of {sorted(LIKELIHOOD_MODES)}.")
    _positive("likelihood.gp_correlation_length_arcsec", config.likelihood.gp_correlation_length_arcsec)
    _positive("likelihood.gp_floor_arcsec", config.likelihood.gp_floor_arcsec, allow_zero=True)
    _positive("likelihood.sigma_white_prior_scale_arcsec", config.likelihood.sigma_white_prior_scale_arcsec)
    _positive("likelihood.sigma_shared_start_arcsec", config.likelihood.sigma_shared_start_arcsec)
    _positive("likelihood.sigma_white_start_arcsec", config.likelihood.sigma_white_start_arcsec)
    if config.likelihood.gp_kernel not in {"image_plane", "potential", "knee"}:
        raise ValueError("likelihood.gp_kernel must be 'image_plane', 'potential' or 'knee'.")
    if config.likelihood.gp_blocks not in {"all", "system", "family"}:
        raise ValueError("likelihood.gp_blocks must be 'all', 'system' or 'family'.")
    _positive("likelihood.gp_knee_gamma", config.likelihood.gp_knee_gamma)
    if config.likelihood.gp_knee_gamma_prior is not None:
        g_lo, g_hi = (float(v) for v in config.likelihood.gp_knee_gamma_prior)
        step = float(config.likelihood.gp_knee_gamma_step)
        if not (0.0 < g_lo < g_hi) or not (g_lo <= float(config.likelihood.gp_knee_gamma) <= g_hi):
            raise ValueError("likelihood.gp_knee_gamma_prior must be (min, max) with 0 < min < max, holding gp_knee_gamma (the start).")
        if step <= 0.0 or abs((g_hi - g_lo) / step - round((g_hi - g_lo) / step)) > 1e-9 or round((g_hi - g_lo) / step) < 2:
            raise ValueError("likelihood.gp_knee_gamma_step must divide the gp_knee_gamma_prior range into at least 2 intervals.")
    modes = tuple(config.likelihood.gp_knee_project_modes)
    if any(m not in {"translation", "convergence", "shear"} for m in modes) or len(set(modes)) != len(modes):
        raise ValueError("likelihood.gp_knee_project_modes takes distinct names among 'translation', 'convergence', 'shear'.")
    if modes and config.likelihood.gp_blocks != "all":
        raise ValueError("likelihood.gp_knee_project_modes needs gp_blocks = 'all' (field-wide modes over one covariance).")
    lo, hi = config.likelihood.gp_knee_scales_arcsec
    if not (0.0 < float(lo) < float(hi)):
        raise ValueError("likelihood.gp_knee_scales_arcsec must be (min, max) with 0 < min < max.")
    lam_lo, lam_hi, lam_n = config.likelihood.gp_knee_lambda_grid
    if not (0.0 < float(lam_lo) < float(lam_hi)) or int(lam_n) < 8:
        raise ValueError("likelihood.gp_knee_lambda_grid must be (min, max, count) with 0 < min < max and count >= 8.")
    knee_upper = config.likelihood.gp_knee_prior.upper
    if config.likelihood.gp_kernel == "knee" and knee_upper is not None and float(knee_upper) > float(hi):
        raise ValueError(f"likelihood.gp_knee_prior.upper ({float(knee_upper):g}\") must not exceed the longest GRF wavelength "
                         f"gp_knee_scales_arcsec[1] ({float(hi):g}\"): beyond it the spectrum is flat and the knee has no meaning.")
    if config.likelihood.gp_kernel == "knee" and (knee_upper is None or float(knee_upper) > float(lam_hi)):
        raise ValueError("likelihood.gp_knee_prior.upper must be set and at most the largest tabulated knee wavelength "
                         f"({float(lam_hi):g}\"): beyond the tables the knee is clamped and only the prior moves it.")
    for name in ("gp_knee_r_step_arcsec", "gp_sigma_alpha_start_arcsec", "gp_knee_start_arcsec", "gp_knee_sigma_shared_start_arcsec",
                 "gp_knee_ell_start_arcsec", "gp_knee_sigma_white_prior_scale_arcsec", "gp_knee_sigma_white_start_arcsec"):
        _positive(f"likelihood.{name}", getattr(config.likelihood, name))
    if config.likelihood.gp_kernel_family not in {"gaussian", "matern52"}:
        raise ValueError("likelihood.gp_kernel_family must be 'gaussian' or 'matern52'.")
    _positive("likelihood.gp_u_small_start", config.likelihood.gp_u_small_start)
    _positive("likelihood.gp_ell_small_start_arcsec", config.likelihood.gp_ell_small_start_arcsec)
    _positive("likelihood.gp_sigma_large_start_arcsec", config.likelihood.gp_sigma_large_start_arcsec)
    _positive("likelihood.gp_ell_large_arcsec", config.likelihood.gp_ell_large_arcsec)
    _positive("likelihood.gp_reference_redshift", config.likelihood.gp_reference_redshift)
    if config.likelihood.gp_sigma_large_prior_scale_arcsec is not None:
        _positive("likelihood.gp_sigma_large_prior_scale_arcsec", config.likelihood.gp_sigma_large_prior_scale_arcsec)
    for label, prior in (("likelihood.sigma_def_prior", config.likelihood.sigma_def_prior),
                         ("likelihood.gp_sigma_alpha_prior", config.likelihood.gp_sigma_alpha_prior),
                         ("likelihood.gp_knee_prior", config.likelihood.gp_knee_prior),
                         ("likelihood.gp_knee_sigma_shared_prior", config.likelihood.gp_knee_sigma_shared_prior),
                         ("likelihood.gp_knee_ell_prior", config.likelihood.gp_knee_ell_prior),
                         ("likelihood.gp_u_small_prior", config.likelihood.gp_u_small_prior),
                         ("likelihood.gp_ell_small_prior", config.likelihood.gp_ell_small_prior),
                         ("likelihood.sigma_extra_prior", config.likelihood.sigma_extra_prior),
                         ("likelihood.sigma_sel_prior", config.likelihood.sigma_sel_prior),
                         ("likelihood.lambda_sel_prior", config.likelihood.lambda_sel_prior),
                         ("likelihood.sigma_shared_prior", config.likelihood.sigma_shared_prior),
                         ("likelihood.gp_correlation_length_prior", config.likelihood.gp_correlation_length_prior)):
        if prior is None:
            continue
        if not isinstance(prior, LogNormalPriorConfig):
            raise ValueError(f"{label} must be a LogNormalPriorConfig.")
        if float(prior.median) <= 0.0 or float(prior.log_sigma) <= 0.0:
            raise ValueError(f"{label} must have a positive median and log_sigma.")
        if prior.upper is not None and float(prior.upper) <= float(prior.median):
            raise ValueError(f"{label}.upper must exceed the median.")
    if scaling.member_deviation not in {"sigma", "mass", "sigma_rcut", "mass_rcut"}:
        raise ValueError("scaling.member_deviation must be 'sigma', 'mass', 'sigma_rcut' or 'mass_rcut'.")
    if scaling.radius_deviation_members not in {"exact", "all"}:
        raise ValueError("scaling.radius_deviation_members must be 'exact' or 'all'.")
    for name, prior in (("member_velocity_scatter_prior", scaling.member_velocity_scatter_prior),
                        ("member_radius_scatter_prior", scaling.member_radius_scatter_prior),
                        ("sigma_def_prior", config.likelihood.sigma_def_prior)):
        if not isinstance(prior, LogNormalPriorConfig) or not all(math.isfinite(float(v)) and v > 0.0 for v in (prior.median, prior.log_sigma)):
            raise ValueError(f"{name} must be a LogNormalPriorConfig with positive median and log_sigma.")
    _validate_truncated_normal("alpha_sigma_prior", mean=scaling.alpha_sigma_prior_mean, std=scaling.alpha_sigma_prior_std,
                               lower=scaling.alpha_sigma_prior_lower, upper=scaling.alpha_sigma_prior_upper)
    _validate_truncated_normal("beta_radius_prior", mean=scaling.beta_radius_prior_mean, std=scaling.beta_radius_prior_std,
                               lower=scaling.beta_radius_prior_lower, upper=scaling.beta_radius_prior_upper)
    if scaling.beta_radius_prior_lower < 0.0:
        raise ValueError("beta_radius_prior_lower must be nonnegative.")
    _positive("softening_length_kpc", scaling.softening_length_kpc, allow_zero=True)
    _positive("softening_length_prior_log_sigma", scaling.softening_length_prior_log_sigma)
    likelihood = config.likelihood
    mean = likelihood.source_position_prior_mean_arcsec
    if len(mean) != 2 or not all(math.isfinite(float(v)) for v in mean):
        raise ValueError("source_position_prior_mean_arcsec requires two finite coordinates.")
    _positive("source_position_prior_sigma_arcsec", likelihood.source_position_prior_sigma_arcsec)
    if likelihood.sigma_inst_arcsec is not None:
        _positive("sigma_inst_arcsec", likelihood.sigma_inst_arcsec)
    diagnostics = config.image_diagnostics
    count = diagnostics.posterior_diagnostic_draws
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("image_diagnostics.posterior_diagnostic_draws must be a nonnegative integer.")
    if not math.isfinite(diagnostics.fold_pair_max_separation_arcsec) or diagnostics.fold_pair_max_separation_arcsec <= 0:
        raise ValueError("image_diagnostics.fold_pair_max_separation_arcsec must be finite and positive.")
    if int(diagnostics.posterior_image_diagnostic_draws) < 0:
        raise ValueError("image_diagnostics.posterior_image_diagnostic_draws must be nonnegative.")
    if diagnostics.exact_image_finder not in {"global-jax", "lenstronomy", "local-lm", "local-lm-adaptive"}:
        raise ValueError("image_diagnostics.exact_image_finder must be 'global-jax', 'lenstronomy', 'local-lm', or 'local-lm-adaptive'.")
    _positive("image_diagnostics.exact_image_grid_pitch_arcsec", diagnostics.exact_image_grid_pitch_arcsec)
    _positive("image_diagnostics.exact_image_fine_pitch_arcsec", diagnostics.exact_image_fine_pitch_arcsec)
    if diagnostics.exact_image_fine_pitch_arcsec > diagnostics.exact_image_grid_pitch_arcsec:
        raise ValueError("image_diagnostics.exact_image_fine_pitch_arcsec must not exceed the coarse grid pitch.")
    _positive("image_diagnostics.exact_image_seed_tolerance_pitches", diagnostics.exact_image_seed_tolerance_pitches)
    if int(config.truth.posterior_truth_recovery_draws) < 0 or int(config.truth.truth_grid_size) < 0:
        raise ValueError("truth.posterior_truth_recovery_draws and truth.truth_grid_size must be nonnegative.")


def _validate_truncated_normal(name: str, *, mean: float, std: float, lower: float, upper: float) -> None:
    if std <= 0.0:
        raise ValueError(f"{name}_std must be positive.")
    if lower >= upper:
        raise ValueError(f"{name}_lower must be less than {name}_upper.")
    if not lower <= mean <= upper:
        raise ValueError(f"{name}_mean must lie inside the configured bounds.")


def _validate_model_config(model: LensModelConfig | None) -> None:
    if model is None:
        raise ValueError("LensClusterSolverConfig.model is required.")
    if not model.large_halos:
        raise ValueError("model.large_halos must contain at least one halo.")
    if model.image_constraints is None:
        raise ValueError("model.image_constraints is required.")
    if model.cosmology.H0 <= 0.0 or model.cosmology.Om0 < 0.0:
        raise ValueError("model.cosmology.H0 must be positive and Om0 nonnegative.")
    for halo in model.large_halos:
        if isinstance(halo, ShearHaloConfig):
            if int(halo.profile_type) != 14 or not math.isfinite(float(halo.gamma)) or halo.gamma < 0.0 or not math.isfinite(float(halo.angle_pos)):
                raise ValueError(f"shear halo {halo.id!r} must use profile_type=14 with finite non-negative gamma and finite angle_pos.")
        elif isinstance(halo, MassSheetConfig):
            if int(halo.profile_type) != 101 or not math.isfinite(float(halo.kappa)):
                raise ValueError(f"mass sheet {halo.id!r} must use profile_type=101 with finite kappa.")
        elif isinstance(halo, NFWHaloConfig):
            if int(halo.profile_type) != 12 or halo.scale_radius_kpc <= 0.0 or not math.isfinite(float(halo.log10_m200)):
                raise ValueError(f"NFW halo {halo.id!r} must use profile_type=12 with a positive scale radius and finite log10_m200.")
            if not (0.0 <= float(halo.ellipticite) < 1.0):
                raise ValueError(f"NFW halo {halo.id!r} must have ellipticite in [0, 1).")
        elif halo.core_radius_kpc <= 0.0 or halo.cut_radius_kpc <= 0.0 or halo.v_disp <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive core, cut, and velocity dispersion.")
        if halo.z_lens <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive z_lens.")
    populations: dict[str, MemberPopulationConfig] = {}
    for population in model.member_populations:
        if isinstance(population.mag0, str):
            if population.mag0 != "auto":
                raise ValueError(f"member population {population.id!r}: mag0 must be a number or 'auto'.")
            if population.prior_mag0 is None:
                raise ValueError(f"member population {population.id!r}: mag0='auto' needs prior_mag0, the magnitude the sigma and cutkpc priors refer to.")
        if population.id in populations:
            raise ValueError(f"duplicate member population id {population.id!r}.")
        populations[population.id] = population
        if not Path(population.catalog_path).is_file():
            raise ValueError(f"member population catalog does not exist: {population.catalog_path}")
        if population.corekpc <= 0.0 or population.sigma <= 0.0 or population.cutkpc <= 0.0 or population.z_lens <= 0.0:
            raise ValueError(f"member population {population.id!r} must have positive corekpc, sigma, cutkpc, and z_lens.")
        exclude_ids = tuple(str(value).strip() for value in population.exclude_catalog_ids)
        if any(not value for value in exclude_ids) or len(set(exclude_ids)) != len(exclude_ids):
            raise ValueError(f"member population {population.id!r} has empty or duplicate exclude_catalog_ids entries.")
    keys: set[tuple[str, str]] = set()
    for independent in model.independent_member_halos:
        key = (str(independent.population_id).strip(), str(independent.catalog_id).strip())
        if not key[0] or not key[1] or key[0] not in populations:
            raise ValueError(f"independent member halo {independent!r} references an unknown or empty population/catalog id.")
        if key in keys:
            raise ValueError(f"duplicate independent member halo {key!r}.")
        keys.add(key)
    constraints = model.image_constraints
    if not Path(constraints.catalog_path).is_file():
        raise ValueError(f"image constraint catalog does not exist: {constraints.catalog_path}")
    if not math.isfinite(float(constraints.sigma_inst_arcsec)) or constraints.sigma_inst_arcsec <= 0.0:
        raise ValueError("image_constraints.sigma_inst_arcsec must be finite and positive.")
    if not math.isfinite(float(constraints.sigma_floor_arcsec)) or constraints.sigma_floor_arcsec < 0.0:
        raise ValueError("image_constraints.sigma_floor_arcsec must be finite and nonnegative.")
