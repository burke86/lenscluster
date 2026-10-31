$XONSH_SHOW_TRACEBACK = True

import json
import os
import sys
import time
from pathlib import Path

cores = 4  # NUTS chains; one JAX CPU device per chain.
seed = 12345

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataclasses import replace

from lenscluster.config import (
    CosmologyConfig,
    DPIEHaloConfig,
    ImageCatalogCutoutConfig,
    ImageConstraintsConfig,
    LensClusterSolverConfig,
    LikelihoodConfig,
    LensModelConfig,
    MemberConfig,
    MemberPopulationConfig,
    PriorConfig,
    RGBDisplayConfig,
    ReferenceFrameConfig,
    RunPathsConfig,
    RuntimeConfig,
    SamplingConfig,
    ScalingModelConfig,
    SearchConfig,
    ShearHaloConfig,
    TruthRecoveryConfig,
)
from lenscluster.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported
from lenscluster.jax_cosmology import kpc_per_arcsec_from_config

OUTPUT_DIR_LABEL = "oct22_single_stage"

# HFF display settings for observed clusters.
HFF_RGB_BANDS = ["F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W"]
HFF_RGB_DISPLAY = {}      # library defaults: f_nu-scaled Lupton display with the warm-white balance of lenscluster.rgb
# --imaging jwst: NIRCam mosaics (CANUCS for MACS0416, DJA for AS1063) under data/JWST_Images; blue F115W+F150W, green
# F150W+F200W, red F444W, the STScI/UNCOVER hue assignment with the four bands we hold.
JWST_RGB_BANDS = ["F115W", "F150W", "F200W", "F444W"]

VALID_CLUSTERS = "ARES, HERA, AS1063, M0416_BERGAMINI2023, M0416_CANUCS2025"
FF_RGB_BANDS = ("F435W", "F606W", "F814W")
FF_RGB_DISPLAY = RGBDisplayConfig(q=6.8, stretch=0.0158, minimum=0.00105, red_gain=0.62, green_gain=0.78, blue_gain=3.65)


def _usage() -> str:
    return (
        "Usage: xonsh run.xsh <cluster> [--campaign NAME] [--run-label NAME] [--image-metadata PATH] [--halo-search lbfgs|optax-lbfgs|adam|svi] [--member-search lbfgs|optax-lbfgs|adam|svi] [--stage init|final|both] [--imaging hst|jwst] [--fast] [--dry-run]\n"
        "  --imaging jwst: draw the cluster mass figure on the NIRCam mosaics (file suffix _jwst) instead of the HST ones\n"
        "  --halo-search, --member-search: optimizer of the large-scale stage and of the member stage of the init (default lbfgs)\n"
        "  --fast: exploratory settings: members deflecting every image by < 0.03\" are dropped, far-field members allowed down to\n"
        "          2 cut radii from an image, classification at the current model, 16 search starts, 300+150 draws at tree depth 7\n"
        "          (check the audit's far-field error before trusting a fast run)\n"
        f"Valid clusters: {VALID_CLUSTERS}"
    )


def _parse_args(argv: list[str]) -> dict:
    if len(argv) == 2 and argv[1] in {"--help", "-h"}:
        print(_usage())
        raise SystemExit(0)
    if len(argv) < 2:
        print(_usage())
        raise SystemExit(2)
    cluster = str(argv[1]).strip().upper()
    values = {"campaign": OUTPUT_DIR_LABEL, "run_label": None, "image_metadata": None, "dry_run": False, "fast": False, "stage": "both", "imaging": "hst",
              "halo_search": "lbfgs", "member_search": "lbfgs"}
    flag_to_key = {"--campaign": "campaign", "--run-label": "run_label", "--image-metadata": "image_metadata", "--stage": "stage", "--imaging": "imaging",
                   "--halo-search": "halo_search", "--member-search": "member_search"}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if flag in {"--dry-run", "--fast"}:
            values[flag.lstrip("-").replace("-", "_")] = True
            index += 1
            continue
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag not in flag_to_key or index + 1 >= len(argv):
            print(f"Unknown argument or missing value: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        values[flag_to_key[flag]] = argv[index + 1]
        index += 2
    for key in ("campaign", "run_label"):
        value = values[key]
        if value is not None and (not str(value).strip() or str(value) in {".", ".."} or "/" in str(value) or "\\" in str(value)):
            print(f"--{key.replace('_', '-')} must be a nonempty single path segment.")
            raise SystemExit(2)
    if values["stage"] not in {"init", "final", "both"}:
        print("--stage must be init, final or both.")
        raise SystemExit(2)
    if values["halo_search"] not in {"lbfgs", "optax-lbfgs", "adam", "svi"} or values["member_search"] not in {"lbfgs", "optax-lbfgs", "adam", "svi"}:
        print("--halo-search and --member-search must be lbfgs, optax-lbfgs, adam or svi.")
        raise SystemExit(2)
    if values["imaging"] not in {"hst", "jwst"}:
        print("--imaging must be hst or jwst.")
        raise SystemExit(2)
    return dict(cluster=cluster, campaign_name=str(values["campaign"]).strip(), run_label=values["run_label"],
                dry_run=values["dry_run"], fast=values["fast"], image_metadata=values["image_metadata"], stage=values["stage"], imaging=values["imaging"],
                halo_search=values["halo_search"], member_search=values["member_search"])


# The solver's tuned defaults live in lenscluster.config; a cluster entry carries only what is specific to it: the
# image catalog, the cosmology, where the cluster-scale halos sit, the member catalog, the truth maps and the
# simulation's softening.  Both simulated clusters are at z = 0.5 with F160W magnitudes, where L* is m_F160W = 19.82.
FF_SIMS_CLUSTERS = {
    "ARES": {
        "cluster_key": "ares",
        "image_catalog": "data/ff_sims/ares/ares_obs_arcs.cat",
        "cosmology": CosmologyConfig(H0=70.4, Om0=0.272, Ode0=0.728),
        # One cluster-scale halo at the light peak of each clump of the MOKA cluster (Meneghetti et al. 2017) plus shear.
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 20.0, -32.0, 0.5, v_disp=950.0),
            DPIEHaloConfig.cluster_scale("2", -40.0, 40.0, 0.5, v_disp=950.0),
            ShearHaloConfig.free("S1", 0.5),
        ),
        "member_population": MemberPopulationConfig.bergamini2019("potfile_1", "data/ff_sims/ares/ares_cluster_members_potfile.cat", 0.5, prior_mag0=19.82),
        # The brightest cluster galaxies stay in the scaling relation, as on Hera: independent halos cost parameters, made
        # the chains multi-well and did not improve the fit.
        "independent_member_halos": (),
        "kappa_true_fits": "data/ff_sims/published/ares/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/ares/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/ares/gammay_z9_0.fits",
        # Ares is semi-analytic: its subhalos are analytic profiles, so there is no resolution softening to match.
        "softening_length_kpc": 0.0,
    },
    "HERA": {
        "cluster_key": "hera",
        "image_catalog": "data/ff_sims/hera/hera_obs_arcs.cat",
        "cosmology": CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 19.54080009, 2.37820005, 0.507, v_disp=800.0),
            DPIEHaloConfig.cluster_scale("2", 0.001, 0.003, 0.507, v_disp=700.0),
            ShearHaloConfig.free("S1", 0.507),
        ),
        "member_population": MemberPopulationConfig.bergamini2019("potfile_1", "data/ff_sims/hera/hera_cluster_members_potfile.cat", 0.507, prior_mag0=19.82),
        "independent_member_halos": (),
        "kappa_true_fits": "data/ff_sims/published/hera/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/published/hera/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/published/hera/gammay_z9_0.fits",
        # Hera is an N-body cluster (H0 = 72, softening 2.3 h^-1 kpc = 3.19 kpc): added in quadrature to the member cores so
        # that their cusps match the simulation's resolution; the dPIE total mass is independent of the core.
        "softening_length_kpc": 2.3 / 0.72,
    },
}


# Native configuration transcribed from
# data/Bergamini/RXJ2248_Bergamini19/Bergamini_RXCJ2248_lenstool_Normal.par.
# That model gives radii in arcseconds; native configurations require kpc.
def _hff_clusters():
    AS1063_KPC_PER_ARCSEC = kpc_per_arcsec_from_config(0.348, {"H0": 70.0, "Om0": 0.3})
    AS1063_CATALOG_DIR = "data/Bergamini/RXJ2248_Bergamini19"
    AS1063_MSTAR_F160W = 18.33
    return {
        "AS1063": {
            "cluster_key": "as1063",
            "reference": ReferenceFrameConfig(reference=3, ra0_deg=342.18321, dec0_deg=-44.530878),
            "image_catalog": f"{AS1063_CATALOG_DIR}/obs_arcs.dat",
            "cosmology": CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            "halos": (
                DPIEHaloConfig(
                    id="1", x_centre=0.0, y_centre=0.0,
                    ellipticite=0.5, angle_pos=-39.0,
                    core_radius_kpc=13.5871 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=2000.0 * AS1063_KPC_PER_ARCSEC,
                    v_disp=1500.0, z_lens=0.348,
                    priors={
                        "x_centre": PriorConfig("uniform", lower=-2.0, upper=4.0, step=0.1),
                        "y_centre": PriorConfig("uniform", lower=-2.0, upper=2.0, step=0.1),
                        "ellipticite": PriorConfig("uniform", lower=0.4, upper=0.8, step=0.05),
                        # Use an initial angle inside the published prior bounds.
                        "angle_pos": PriorConfig("uniform", lower=-42.0, upper=-36.0, step=0.1),
                        "core_radius_kpc": PriorConfig(
                            "uniform", lower=5.0 * AS1063_KPC_PER_ARCSEC,
                            upper=35.0 * AS1063_KPC_PER_ARCSEC, step=0.1 * AS1063_KPC_PER_ARCSEC,
                        ),
                        "v_disp": PriorConfig("truncated_normal", mean=1200.0, std=200.0, lower=1000.0, upper=float("inf")),
                    },
                ),
                DPIEHaloConfig(
                    id="2", x_centre=-84.6455835, y_centre=45.01476,
                    ellipticite=0.0, angle_pos=0.0,
                    core_radius_kpc=0.05 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=2000.0 * AS1063_KPC_PER_ARCSEC,
                    v_disp=300.0, z_lens=0.348,
                    priors={
                        "x_centre": PriorConfig("uniform", lower=-100.0, upper=50.0, step=0.1),
                        "y_centre": PriorConfig("uniform", lower=-50.0, upper=70.0, step=0.1),
                        "v_disp": PriorConfig("truncated_normal", mean=500.0, std=500.0, lower=0.0, upper=float("inf")),
                    },
                ),
                # Fixed gas components from the same model.
                DPIEHaloConfig(
                    id="3", x_centre=18.898068688515124, y_centre=-73.3566238994408,
                    ellipticite=0.804756331682532, angle_pos=-162.04793432346537,
                    core_radius_kpc=188.39998560000458 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=189.2403216000046 * AS1063_KPC_PER_ARCSEC,
                    v_disp=355.9263730614334, z_lens=0.348,
                ),
                DPIEHaloConfig(
                    id="4", x_centre=-18.0530993114858, y_centre=13.469568100561368,
                    ellipticite=0.13273356095824723, angle_pos=-27.795099374268442,
                    core_radius_kpc=36.323966400000884 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=339.1572480000082 * AS1063_KPC_PER_ARCSEC,
                    v_disp=442.5672747534171, z_lens=0.348,
                ),
                DPIEHaloConfig(
                    id="5", x_centre=0.19813268851464938, y_centre=-1.2372958994389047,
                    ellipticite=0.34435532608684766, angle_pos=-15.486418948811508,
                    core_radius_kpc=14.434532160000352 * AS1063_KPC_PER_ARCSEC,
                    cut_radius_kpc=356.49532800000867 * AS1063_KPC_PER_ARCSEC,
                    v_disp=249.68220495795586, z_lens=0.348,
                ),
            ),
            # Member priors at L* that come from no lens model (Bergamini et al. 2019 define L* as M_F160W = -23 with no
            # K-correction, so m* = -23 + mu(z = 0.348) = 18.33): sigma* TN(200, 100) km/s, r_cut* log-uniform 2-100 kpc,
            # alpha 0.25 +- 0.10, beta 0.5 +- 0.25.  Their AS1063 fit (299 km/s, 6.8 kpc at the BCG magnitude 16.18, i.e.
            # 175 km/s and 25 kpc at m*) lies inside these priors.  The pivot itself is chosen automatically.
            "member_population": MemberPopulationConfig.neutral("potfile_1", f"{AS1063_CATALOG_DIR}/members.cat", 0.348, prior_mag0=AS1063_MSTAR_F160W),
            "scaling": ScalingModelConfig.neutral(),
            "independent_member_halos": (),
            "softening_length_kpc": 0.0,
            "softening_length_prior_log_sigma": 0.15,
            # Instrumental astrometric error per image (HST WFC3/IR F160W).
            "sigma_inst_arcsec": 0.01,
            # Classify at the current model (reclassified at the MAP) rather than at the prior's 0.975 quantile, which for a
            # log-uniform cut radius is 90 kpc (198 of 222 exact).  The library defaults (guard 3, inflation 3) give the
            # sampled r_cut* a factor of three before any far-field member reaches the guard.
            "classification_prior_quantile": None,
        },
        **_m0416_bergamini2023_cluster(),
    }


# MACS J0416, two image catalogs over the same mass model: the HST + MUSE catalog of Bergamini et al. 2023 (237 spectroscopic
# images in 88 families; the files are their 2022 release) and the JWST CANUCS gold catalog.  Native configuration transcribed from data/Bergamini/M0416_Bergamini22/Bergamini22_MACS0416_Normal.par:
# four dark-matter dPIE halos with their prior boxes, a free foreground galaxy, four fixed hot-gas clumps,
# and the members (including galaxy 8971) on the neutral relation.
def _m0416_bergamini2023_cluster():
    kpc = kpc_per_arcsec_from_config(0.396, {"H0": 70.0, "Om0": 0.3})
    catalog_dir = "data/Bergamini/M0416_Bergamini22"
    z = 0.396
    mstar_f160w = 18.66        # -23 + mu(z = 0.396), the Bergamini et al. 2019 definition of L*
    inf = float("inf")

    def dark_halo(id, x, y, e, angle, core_arcsec, v_disp, *, v_prior, x_box, y_box, core_box_arcsec, e_box=None, angle_box=None):
        priors = {"x_centre": PriorConfig("uniform", lower=x_box[0], upper=x_box[1], step=0.1),
                  "y_centre": PriorConfig("uniform", lower=y_box[0], upper=y_box[1], step=0.1),
                  "core_radius_kpc": PriorConfig("uniform", lower=max(0.5, core_box_arcsec[0] * kpc), upper=core_box_arcsec[1] * kpc, step=0.1 * kpc),
                  "v_disp": PriorConfig("truncated_normal", mean=v_prior[0], std=v_prior[1], lower=v_prior[2], upper=inf)}
        if e_box is not None:
            priors["ellipticite"] = PriorConfig("uniform", lower=e_box[0], upper=e_box[1], step=0.05)
        if angle_box is not None:
            priors["angle_pos"] = PriorConfig("uniform", lower=angle_box[0], upper=angle_box[1], step=0.1)
        return DPIEHaloConfig(id=id, x_centre=x, y_centre=y, ellipticite=e, angle_pos=angle, core_radius_kpc=core_arcsec * kpc,
                              cut_radius_kpc=2000.0 * kpc, v_disp=v_disp, z_lens=z, priors=priors)

    def fixed(id, x, y, e, angle, core_arcsec, cut_arcsec, v_disp):
        return DPIEHaloConfig(id=id, x_centre=x, y_centre=y, ellipticite=e, angle_pos=angle, core_radius_kpc=core_arcsec * kpc,
                              cut_radius_kpc=cut_arcsec * kpc, v_disp=v_disp, z_lens=z)

    halos = (
        # NE halo: Bergamini's start sits at the BCG-N; its ellipticity and angle are free in their boxes.
        dark_halo("1", 0.13392, -0.04501, 0.8, -35.7557, 13.5871, 650.0, v_prior=(675.0, 325.0, 350.0), x_box=(-15.0, 15.0), y_box=(-15.0, 15.0),
                  core_box_arcsec=(0.0, 20.0), e_box=(0.2, 0.9), angle_box=(-180.0, 180.0)),
        # SW halo.
        dark_halo("2", 20.29565, -35.83494, 0.6417, -55.3798, 14.416, 805.1496, v_prior=(775.0, 425.0, 350.0), x_box=(15.0, 30.0), y_box=(-45.0, -30.0),
                  core_box_arcsec=(0.0, 25.0), e_box=(0.2, 0.9), angle_box=(-90.0, 90.0)),
        # Third halo, circular.
        dark_halo("3", -40.0, 15.0, 0.0, 0.0, 13.5871, 650.0, v_prior=(400.0, 350.0, 50.0), x_box=(-55.0, -25.0), y_box=(0.0, 30.0),
                  core_box_arcsec=(0.0, 35.0)),
        # Fourth halo, south-west.
        dark_halo("4", 28.772, -39.454, 0.9, 134.202, 10.671, 476.784, v_prior=(550.0, 450.0, 100.0), x_box=(-10.0, 50.0), y_box=(-75.0, -15.0),
                  core_box_arcsec=(0.0, 20.0), e_box=(0.2, 0.9), angle_box=(0.0, 180.0)),
        # Foreground galaxy, modelled at the cluster redshift as in Bergamini et al. 2022.
        DPIEHaloConfig(id="5", x_centre=31.96420, y_centre=-65.55168, ellipticite=0.0, angle_pos=0.0, core_radius_kpc=0.01, cut_radius_kpc=33.6321 * kpc,
                       v_disp=235.2454, z_lens=z, priors={"v_disp": PriorConfig("truncated_normal", mean=200.0, std=150.0, lower=50.0, upper=inf),
                                                          "cut_radius_kpc": PriorConfig("uniform", lower=5.0 * kpc, upper=100.0 * kpc, step=0.1 * kpc)}),
        # Galaxy 8971 now belongs to the member scaling relation in both catalogs.
        # DPIEHaloConfig(id="8971", x_centre=13.33912, y_centre=2.62495, ellipticite=0.0, angle_pos=0.0, core_radius_kpc=0.01, cut_radius_kpc=33.6321 * kpc,
        #                v_disp=180.2454, z_lens=z, priors={"v_disp": PriorConfig("truncated_normal", mean=130.0, std=70.0, lower=60.0, upper=inf),
        #                                                   "cut_radius_kpc": PriorConfig("uniform", lower=0.5, upper=50.0 * kpc, step=0.1 * kpc),
        #                                                   "angle_pos": PriorConfig("uniform", lower=-90.0, upper=90.0, step=0.1),
        #                                                   "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.6, step=0.05)}),
        # Hot-gas clumps from the X-ray model, fixed.
        fixed("gas1", -18.137002925379967, -12.133967959650562, 0.11802924806826215, -156.7566690854322, 149.2054944000091, 149.82423360000914, 433.0248792121149),
        fixed("gas2", 30.787477074623045, -48.67185595965274, 0.4166205427409096, -71.5005491699657, 34.77330048000212, 165.77093760001011, 248.99144885049094),
        fixed("gas3", -2.3693869253788935, -1.2627359596498962, 0.4248370668779292, -54.743768197790125, 8.281737600000504, 37.5947433600023, 101.7305750041091),
        fixed("gas4", -20.12861892537998, 14.74104004035109, 0.4013796817764777, -49.32427500520669, 51.67495680000315, 52.337385600003195, 281.8472712670227),
    )
    def entry(key, *, reference, image_catalog, member_catalog, sigma_inst_arcsec):
        return {
            "cluster_key": key,
            "reference": ReferenceFrameConfig(reference=3, ra0_deg=reference[0], dec0_deg=reference[1]),
            "image_catalog": image_catalog,
            "cosmology": CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            "halos": halos,
            # Neutral member priors at L* (see AS1063); Bergamini et al. 2023 found sigma 248 +- 28 km/s and r_cut 56 +- 30 kpc at
            # the BCG magnitude 17.02 with a fixed slope of 0.3, i.e. 158 km/s and 29 kpc at m*, inside these priors.
            "member_population": MemberPopulationConfig.neutral("potfile_1", member_catalog, z, prior_mag0=mstar_f160w),
            "scaling": ScalingModelConfig.neutral(),
            "independent_member_halos": (),
            "softening_length_kpc": 0.0,
            "softening_length_prior_log_sigma": 0.15,
            "sigma_inst_arcsec": sigma_inst_arcsec,
            "classification_prior_quantile": None,
            # Far-field error at 2 r_cut is < 1 uas (oct28a); the default inflation 3 keeps the headroom.
            "far_field_cut_radii": 2.0,
        }

    canucs_dir = "data/CANUCS/M0416_Rihtarsic2025"
    return {
        # HST + MUSE: 237 spectroscopic images in 88 families (Bergamini et al. 2023), HST astrometry.
        "M0416_BERGAMINI2023": entry("m0416_bergamini2023", reference=(64.0381417, -24.0674722), image_catalog=f"{catalog_dir}/obs_arcs.cat",
                                     member_catalog=f"{catalog_dir}/CM_cat_MACSJ0416.cat", sigma_inst_arcsec=0.01),
        # JWST + HST: the CANUCS gold catalog, 303 spectroscopic images in 111 families (Rihtarsic et al. 2025), NIRCam
        # positions; the same halo set-up and member catalog (their copy has the BCG shifted), their reference centre.
        "M0416_CANUCS2025": entry("m0416_canucs2025", reference=(64.03821668520617, -24.06750122335985), image_catalog=f"{canucs_dir}/obs_arcs.cat",
                                  member_catalog=f"{canucs_dir}/members.cat", sigma_inst_arcsec=0.01),
    }


CLUSTERS = set(FF_SIMS_CLUSTERS) | {"AS1063", "M0416_BERGAMINI2023", "M0416_CANUCS2025"}


def build_runtime(cores: int) -> RuntimeConfig:
    return RuntimeConfig(
        seed=seed,
        chains=cores,
        resume=True,
        numpyro_print_summary=True,
        nuts_chain_method="parallel",
        # Shared CPU budget = concurrent chains * this value (Linux pinning); None, the macOS fallback, disables tuning.
        cpu_threads_per_chain=threads_per_chain_if_supported(2),
    )


def build_config(
    cluster: str,
    *,
    cores: int,
    campaign_name: str,
    run_label: str | None = None,
    image_metadata: str | None = None,
    runtime: RuntimeConfig | None = None,
    fast: bool = False,
    imaging: str = "hst",
    halo_search: str = "lbfgs",
    member_search: str = "lbfgs",
) -> LensClusterSolverConfig:
    """The solver configuration for one cluster: library defaults plus the cluster entry, the command-line choices and,
    with ``fast``, the exploratory overrides."""
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {', '.join(sorted(CLUSTERS))}; got {cluster!r}.")
    is_hff = cluster not in FF_SIMS_CLUSTERS
    cluster_config = _hff_clusters()[cluster] if is_hff else FF_SIMS_CLUSTERS[cluster]
    config = LensClusterSolverConfig(
        model=LensModelConfig(
            reference=cluster_config.get("reference", ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0)),
            cosmology=cluster_config["cosmology"],
            large_halos=cluster_config["halos"],
            independent_member_halos=cluster_config["independent_member_halos"],
            member_populations=(cluster_config["member_population"],),
            image_constraints=ImageConstraintsConfig(catalog_path=cluster_config["image_catalog"], metadata_path=image_metadata,
                                                     sigma_inst_arcsec=float(cluster_config.get("sigma_inst_arcsec", ImageConstraintsConfig.sigma_inst_arcsec))),
        ),
        # No selection term: in the oct28a M0416 run sigma_sel and sigma_def fit equally well and lambda_sel stayed at its prior.
        likelihood=LikelihoodConfig(sigma_sel_prior=None),
        paths=RunPathsConfig(output_dir=f"results/{campaign_name}/{cluster_config['cluster_key']}",
                             run_name=run_label if run_label is not None else "smoothed-local-image-plane",
                             figure_prefix=cluster_config["cluster_key"]),
        runtime=runtime if runtime is not None else build_runtime(cores),
        search=SearchConfig(n_starts=256, halo_method=halo_search, member_method=member_search),
        sampling=SamplingConfig(warmup=1000, samples=1000, max_tree_depth=12, target_accept=0.85),
        scaling=replace(cluster_config.get("scaling", ScalingModelConfig()), softening_length_kpc=float(cluster_config.get("softening_length_kpc", 0.0))),
        members=MemberConfig(far_field_cut_radii=float(cluster_config.get("far_field_cut_radii", MemberConfig.far_field_cut_radii)),
                             cut_radius_inflation=float(cluster_config.get("cut_radius_inflation", MemberConfig.cut_radius_inflation)),
                             classification_prior_quantile=cluster_config.get("classification_prior_quantile", MemberConfig.classification_prior_quantile)),
        truth=TruthRecoveryConfig(kappa_true_fits=cluster_config.get("kappa_true_fits"), gammax_true_fits=cluster_config.get("gammax_true_fits"),
                                  gammay_true_fits=cluster_config.get("gammay_true_fits")),
        image_catalog=ImageCatalogCutoutConfig(
            image_dir=("data/JWST_Images" if imaging == "jwst" else "data/BUFFALO_Images") if is_hff else "data/ff_sims",
            image_scale="auto",
            bands=(tuple(JWST_RGB_BANDS) if imaging == "jwst" else tuple(HFF_RGB_BANDS)) if is_hff else FF_RGB_BANDS,
            rgb=RGBDisplayConfig(**HFF_RGB_DISPLAY) if is_hff else FF_RGB_DISPLAY,
            figure_suffix="_jwst" if (is_hff and imaging == "jwst") else "",
        ),
    )
    if fast:
        # Exploratory: members deflecting every image by < 0.03" leave the model (Ares: 332 -> ~150), far-field members
        # allowed down to 2 cut radii, classification at the current model, fewer starts, shorter chains at tree depth 7.
        config = replace(
            config,
            search=replace(config.search, n_starts=32),
            sampling=SamplingConfig(warmup=500, samples=250, max_tree_depth=10, target_accept=0.8),
            members=replace(config.members, far_field_cut_radii=2.0, classification_prior_quantile=None, min_deflection_arcsec=0.03),
        )
    return config

def _fast_mode_warning(config) -> None:
    """A banner that cannot be missed: a --fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = config.members
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            f"  - members deflecting every image by less than {members.min_deflection_arcsec:g}\" are dropped from the model\n"
            f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {config.search.n_starts} search starts, {config.sampling.warmup} warmup + {config.sampling.samples} draws per chain "
            f"at tree depth {config.sampling.max_tree_depth}, target accept {config.sampling.target_accept:g}")
    Console(stderr=False).print(Panel(text, title="[bold]WARNING: --fast[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    cluster = options["cluster"]
    if cluster not in CLUSTERS:
        print(f"Unknown cluster: {cluster!r}\n{_usage()}")
        return 2
    runtime = build_runtime(cores)
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(cluster, cores=cores, campaign_name=options["campaign_name"], run_label=options["run_label"],
                          image_metadata=options["image_metadata"], runtime=runtime, fast=options["fast"], imaging=options["imaging"],
                          halo_search=options["halo_search"], member_search=options["member_search"]).validate()
    if options["dry_run"]:
        from lenscluster.pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    from lenscluster.pipeline import run
    if options["fast"]:
        _fast_mode_warning(config)
    print(f"[config] cluster={cluster} run_name={config.paths.run_name} output_dir={config.paths.output_dir} "
          f"stage={options['stage']} chains={runtime.chains} search={config.search.halo_method}+{config.search.member_method} "
          f"n_starts={config.search.n_starts} fast={options['fast']} sigma_inst_arcsec={config.model.image_constraints.sigma_inst_arcsec}")
    start = time.monotonic()
    try:
        run(config, stage=options["stage"])
        if options["fast"]:
            _fast_mode_warning(config)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
