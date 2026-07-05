$XONSH_SHOW_TRACEBACK = True

import os
import sys
import time
from pathlib import Path

cores = 4  # Fixed to match JAX CPU devices and NUTS chains.
seed = 12345

$JAX_NUM_CPU_DEVICES = str(cores)
$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster"
os.environ["JAX_NUM_CPU_DEVICES"] = str(cores)
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageCatalogCutoutConfig,
    ImageConstraintsConfig,
    ImageDiagnosticsConfig,
    IndependentMemberHaloConfig,
    LensClusterSolverConfig,
    LensModelConfig,
    LikelihoodConfig,
    MemberPopulationConfig,
    MemberSelectionConfig,
    PerturbationDiscoveryConfig,
    PriorConfig,
    RGBDisplayConfig,
    ReferenceFrameConfig,
    RunPathsConfig,
    RuntimeConfig,
    ScalingModelConfig,
    ShearHaloConfig,
    StageScheduleConfig,
    TruthRecoveryConfig,
    WorkflowConfig,
)
from lenscluster.planning import compile_run_plan
from lenscluster.runner import LensClusterRunner

OUTPUT_DIR_LABEL = "jun26a_anistropic_possigma01_nomaglikelihood_magcut24_externalshear"

# Kept here for HFF display tuning reuse, but this runner intentionally has no
# HFF/Bergamini model configurations.
HFF_RGB_BANDS = ["F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W"]
HFF_RGB_DISPLAY = {"q": 6.4, "stretch": 0.0145, "minimum": -5.5e-4, "red_gain": 0.47, "green_gain": 0.91, "blue_gain": 3.95}

VALID_CLUSTERS = "ARES, HERA"
FF_RGB_BANDS = ("F435W", "F606W", "F814W")
FF_RGB_DISPLAY = RGBDisplayConfig(q=6.8, stretch=0.0158, minimum=0.00105, red_gain=0.62, green_gain=0.78, blue_gain=3.65)


def _usage() -> str:
    return f"Usage: xonsh run.xsh <cluster>\nValid clusters: {VALID_CLUSTERS}"


FF_SIMS_CLUSTERS = {
    "ARES": {
        "cluster_key": "ares",
        "output_dir": f"results/{OUTPUT_DIR_LABEL}/ares",
        "image_catalog": "data/ff_sims/ares/ares_obs_arcs.cat",
        "member_catalog": "data/ff_sims/ares/ares_cluster_members_potfile.cat",
        "cosmology": CosmologyConfig(H0=70.4, Om0=0.272, Ode0=0.728),
        "z_lens": 0.5,
        "halos": (
            DPIEHaloConfig(
                id="1",
                x_centre=20.0,
                y_centre=-32.0,
                ellipticite=0.3,
                angle_pos=0.0,
                core_radius_kpc=20.0,
                cut_radius_kpc=1500.0,
                v_disp=950.0,
                z_lens=0.5,
                priors={
                    "x_centre": PriorConfig("uniform", lower=15.0, upper=25.0, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-37.0, upper=-27.0, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.5, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=5.0, upper=60.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=950.0, std=250.0, lower=500.0, upper=1800.0),
                },
            ),
            DPIEHaloConfig(
                id="2",
                x_centre=-40.0,
                y_centre=40.0,
                ellipticite=0.3,
                angle_pos=0.0,
                core_radius_kpc=20.0,
                cut_radius_kpc=1500.0,
                v_disp=950.0,
                z_lens=0.5,
                priors={
                    "x_centre": PriorConfig("uniform", lower=-45.0, upper=-35.0, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=35.0, upper=45.0, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.5, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=5.0, upper=45.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=950.0, std=175.0, lower=600.0, upper=1400.0),
                },
            ),
            ShearHaloConfig(
                id="S1",
                gamma=0.0,
                angle_pos=0.0,
                z_lens=0.5,
                priors={
                    "gamma": PriorConfig("uniform", lower=0.0, upper=0.3, step=0.005),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=0.5),
                },
            ),
        ),
        "member_population": MemberPopulationConfig(
            id="potfile_1",
            catalog_path="data/ff_sims/ares/ares_cluster_members_potfile.cat",
            mag0=18.5,
            corekpc=0.15,
            sigma=100.0,
            cutkpc=270.0,
            z_lens=0.5,
            sigma_prior=PriorConfig("truncated_normal", mean=100.0, std=15.0, lower=70.0, upper=500.0),
            cutkpc_prior=PriorConfig("truncated_normal", mean=270.0, std=35.0, lower=160.0, upper=800.0),
        ),
        "independent_member_halos": (
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="2"),
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="3"),
        ),
        "kappa_true_fits": "data/ff_sims/published/ares/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/ares/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/ares/gammay_z9_0.fits",
        "softening_length_kpc": 0.0,
        "softening_length_prior_log_sigma": 0.15,
    },
    "HERA": {
        "cluster_key": "hera",
        "output_dir": f"results/{OUTPUT_DIR_LABEL}/hera",
        "image_catalog": "data/ff_sims/hera/hera_obs_arcs.cat",
        "member_catalog": "data/ff_sims/hera/hera_cluster_members_potfile.cat",
        "cosmology": CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        "z_lens": 0.507,
        "halos": (
            DPIEHaloConfig(
                id="1",
                x_centre=19.54080009,
                y_centre=2.37820005,
                ellipticite=0.3,
                angle_pos=30.0,
                core_radius_kpc=8.0,
                cut_radius_kpc=1500.0,
                v_disp=800.0,
                z_lens=0.507,
                priors={
                    "x_centre": PriorConfig("uniform", lower=14.54080009, upper=24.54080009, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-2.62179995, upper=7.37820005, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=2.0, upper=15.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=800.0, std=280.0, lower=100.0, upper=2200.0),
                },
            ),
            DPIEHaloConfig(
                id="2",
                x_centre=0.001,
                y_centre=0.003,
                ellipticite=0.3,
                angle_pos=24.0,
                core_radius_kpc=5.0,
                cut_radius_kpc=1500.0,
                v_disp=700.0,
                z_lens=0.507,
                priors={
                    "x_centre": PriorConfig("uniform", lower=-4.999, upper=5.001, step=0.5),
                    "y_centre": PriorConfig("uniform", lower=-4.997, upper=5.003, step=0.5),
                    "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8, step=0.02),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                    "core_radius_kpc": PriorConfig("uniform", lower=2.0, upper=15.0, step=1.0),
                    "v_disp": PriorConfig("truncated_normal", mean=700.0, std=245.0, lower=100.0, upper=2200.0),
                },
            ),
            ShearHaloConfig(
                id="S1",
                gamma=0.0,
                angle_pos=0.0,
                z_lens=0.507,
                priors={
                    "gamma": PriorConfig("uniform", lower=0.0, upper=0.3, step=0.005),
                    "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=0.5),
                },
            ),
        ),
        "member_population": MemberPopulationConfig(
            id="potfile_1",
            catalog_path="data/ff_sims/hera/hera_cluster_members_potfile.cat",
            mag0=19.82,
            corekpc=0.15,
            sigma=96.7,
            cutkpc=33.0,
            z_lens=0.507,
            sigma_prior=PriorConfig("truncated_normal", mean=96.7, std=40.0, lower=30.0, upper=250.0),
            cutkpc_prior=PriorConfig("truncated_normal", mean=33.0, std=25.0, lower=3.0, upper=250.0),
        ),
        "independent_member_halos": (
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="1"),
            IndependentMemberHaloConfig(population_id="potfile_1", catalog_id="2"),
        ),
        "kappa_true_fits": "data/ff_sims/published/hera/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/hera/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/hera/gammay_z9_0.fits",
        # HERA uses H0=72 km/s/Mpc, so h=0.72 and 2.3 h^-1 kpc = 3.19 kpc physical.
        "softening_length_kpc": 2.3 / 0.72,
        "softening_length_prior_log_sigma": 0.15,
    },
}


def build_config(cluster: str, *, cores: int) -> LensClusterSolverConfig:
    if cluster not in FF_SIMS_CLUSTERS:
        raise ValueError(f"cluster must be one of {', '.join(sorted(FF_SIMS_CLUSTERS))}; got {cluster!r}.")
    cluster_config = FF_SIMS_CLUSTERS[cluster]

    use_magnitude_likelihood = False

    perturbation_alpha_tol = 0.1
    perturbation_jacobian_tol = 0.1
    perturbation_top_k = 10
    warmup = 10000, 10000
    samples = 1000, 1000
    max_tree_depth = 8, 8
    stage0_likelihood = "source"
    stage1_likelihood = "source"
    
    stage0_likelihood = "local-jacobian"
    stage1_likelihood = "local-jacobian"
    stage2_forward_mode = "critical-arc-anisotropic"
    
    #stage1_likelihood = "critical-arc-anisotropic"


    output_dir = (
        f"{cluster_config['output_dir']}_PD{perturbation_alpha_tol:g}_TOPK_{perturbation_top_k}_"
        f"{perturbation_jacobian_tol:g}_T{max_tree_depth}W{warmup}S{samples}"
        f"maglikelihood{use_magnitude_likelihood}"
    )
    run_name = f"{cluster_config['cluster_key']}_S1{stage1_likelihood}_S2{stage2_forward_mode}" 
    
    return LensClusterSolverConfig(
        model=LensModelConfig(
            reference=ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0),
            cosmology=cluster_config["cosmology"],
            large_halos=cluster_config["halos"],
            independent_member_halos=cluster_config["independent_member_halos"],
            member_populations=(cluster_config["member_population"],),
            image_constraints=ImageConstraintsConfig(catalog_path=cluster_config["image_catalog"], sigma_arcsec=0.5),
        ),
        paths=RunPathsConfig(output_dir=output_dir, run_name=run_name),
        runtime=RuntimeConfig(
            seed=seed,
            chains=cores,
            resume="all",
            quick_diagnostics=False,
            debug_sampler_diagnostics=True,
            numpyro_print_summary=True,
            nuts_chain_method="parallel",
            dense_mass="full",
            jax_clear_caches_after_svi_refresh=False,
        ),
        workflow=WorkflowConfig(
            fit_mode="sequential",
            stage0_likelihood=stage0_likelihood,
            stage1_likelihood=stage1_likelihood,
            stage2_forward_mode=stage2_forward_mode,
            stage1_sampling_engine="refreshing_surrogate_flat",
            stage2_sampling_engine="refreshing_surrogate_flat",
            stage2_fresh_process=True,
            exact_image_diagnostics_stage2=True,
            best_value="maximum-likelihood",
            image_plane_newton_steps=0,
            linearized_beta_prior_sigma_arcsec=3.0,
            source_position_parameterization="prior-whitened",
        ),
        schedule=StageScheduleConfig(
            fit_method=("svi+nuts", "svi+nuts"),
            refresh_every=(None, 2000, 2000),
            svi_steps=(10_000, 20_000, 20_000),
            warmup=warmup,
            samples=samples,
            sampling_refresh_runs=(1, 1),
            max_tree_depth=max_tree_depth,
            target_accept=0.8,
            z_bin_efficiency_tol=0.0,
            svi_learning_rate=0.0005,
        ),
cczz
        scaling=ScalingModelConfig(
            independent_scaling_free_log_sigma_offset_width=0.45,
            independent_scaling_free_log_mass_offset_width=0.55,
            potfile_alpha_sigma_prior_mean=0.25,
            potfile_alpha_sigma_prior_std=0.3,
            potfile_alpha_sigma_prior_lower=0.05,
            potfile_alpha_sigma_prior_upper=0.50,
            potfile_gamma_ml_prior_mean=0.20,
            potfile_gamma_ml_prior_std=0.3,
            potfile_gamma_ml_prior_lower=-0.80,
            potfile_gamma_ml_prior_upper=0.80,
            scaling_scatter=True,
            softening_length_kpc=float(cluster_config["softening_length_kpc"]),
            softening_length_prior_log_sigma=float(cluster_config["softening_length_prior_log_sigma"]),
        ),
        likelihood=LikelihoodConfig(
            use_magnitude_likelihood=use_magnitude_likelihood,
            # magnitude_base_scatter_prior=PriorConfig(
            #     kind="lognormal",
            #     lower=0.01,
            #     upper=0.50,
            #     mean=0.3,
            #     std=0.20,
            # ),
            # magnitude_arc_scatter_prior=PriorConfig(
            #     kind="lognormal",
            #     lower=0.01,
            #     upper=0.50,
            #     mean=0.3,
            #     std=0.2,
            # ),
            pos_sigma_arcsec=0.1,
            source_plane_covariance_mode="magnification",
            image_presence_penalty_weight=2.0,
            image_presence_match_radius_arcsec=1.0,
            image_presence_temperature_arcsec=0.5,
            image_plane_scatter_prior="log-uniform",
            image_plane_scatter_floor_arcsec=0.01,
            image_plane_scatter_upper_arcsec=1.0,
        ),
        image_diagnostics=ImageDiagnosticsConfig(
            critical_arc_singular_threshold=0.1,
            arc_recovery_p_arc_threshold=0.1,
            posterior_image_diagnostic_draws=0,
            exact_image_min_distance_arcsec=0.5,
            exact_image_precision_limit=1.0e-2,
            exact_image_num_iter_max=100,
            exact_image_finder="local-lm-adaptive",
            exact_image_displacement_tol_arcsec=1.0e-4,
            exact_image_identification_tol_arcsec=1.0e-3,
            match_tolerance_arcsec=2.0,
        ),
        truth=TruthRecoveryConfig(
            kappa_true_fits=cluster_config["kappa_true_fits"],
            gammax_true_fits=cluster_config["gammax_true_fits"],
            gammay_true_fits=cluster_config["gammay_true_fits"],
            truth_grid_mode="posterior",
            posterior_truth_recovery_draws=128,
            truth_grid_size=256,
            caustic_source_redshift=9.0,
        ),
        image_catalog=ImageCatalogCutoutConfig(
            image_dir="data/ff_sims",
            image_scale="auto",
            bands=FF_RGB_BANDS,
            rgb=FF_RGB_DISPLAY,
            mode="full",
            cutouts=False,
        ),
    )


if len(sys.argv) != 2:
    print(_usage())
    raise SystemExit(2)

requested_cluster = sys.argv[1].strip().upper()
if requested_cluster not in FF_SIMS_CLUSTERS:
    print(f"Unknown cluster: {sys.argv[1]!r}\n{_usage()}")
    raise SystemExit(2)

config = build_config(requested_cluster, cores=cores)
plan = compile_run_plan(config)
print(
    f"[config] cluster={requested_cluster} run_name={plan.output.run_name} "
    f"output_dir={plan.output.output_dir} chains={plan.runtime.chains}"
)

start = time.monotonic()
try:
    LensClusterRunner().run(plan)
finally:
    print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
