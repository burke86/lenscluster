$XONSH_SHOW_TRACEBACK = True

import json
import os
import sys
import time
from pathlib import Path

cores = 4  # NUTS chains; one JAX CPU device per chain.
seed = 12345

# Proposed M0416 associations, including observed coordinates and redshift provenance.
# Counterimages: K66.c1, K67.c1, K20.c1; weaker: K28.c1, K34.c1, K34.c2, K115.c1.
# New family A: A.1, A.2, A.3; geometric z=1.3847046874, no spectroscopic redshift.
# This catalog controls RGB overlays (family-matched circles/squares; green family-A squares).
M0416_CANDIDATES = "src/lenscluster/data/m0416_candidates.json"
# Critical-curve RGB composite: top left-to-right, then right top-to-bottom.
M0416_RGB_IMAGE_CUTOUTS = ("K34.b", "K34.a", "K33.e", "K77.b")
M0416_RGB_FOLD_CUTOUTS = ("K1", "K6", "K8", "K12")

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
    IndependentMemberHaloConfig,
    LensModelConfig,
    LogNormalPriorConfig,
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

# BCG cut radii: "lognormal" (default since 2026-11-04) is an unbounded log-normal around the relation's value at the BCG
# magnitude, width a factor 1.9 (log-sigma 0.64); "uniform" is the old bounded uniform, under which BCG 3158 piled
# against its 200 kpc bound and BCG 1809 wandered over 14-200 kpc between chains.  Set by --bcg-cut-prior.
BCG_CUT_PRIOR = "lognormal"


def _bcg_cut_prior(median_kpc: float, uniform_bounds: tuple[float, float]) -> PriorConfig:
    """Cut radius prior of an independent BCG halo.  A "normal" radius prior becomes a log-normal in log r_cut with
    moment-matched width and no bound, so mean = median exp(s^2 / 2) and std = mean sqrt(exp(s^2) - 1) for s = 0.64."""
    if BCG_CUT_PRIOR == "uniform":
        return PriorConfig("uniform", lower=float(uniform_bounds[0]), upper=float(uniform_bounds[1]), step=1.0)
    mean = 1.227 * float(median_kpc)
    return PriorConfig("normal", mean=mean, std=0.714 * mean)

# Catalog astrometric error of every observed image (2026-11-05): a 1 mas floor instead of the 10 mas catalog value, so
# the white image-plane scatter (images.sigma_white, log-normal) is inferred rather than assumed.  The library requires a
# positive value; 1 mas is below any of the catalogs' astrometry.  The "chi-square (catalog errors)" line of the run
# summary uses it and is meaningless at this value.
SIGMA_INST_ARCSEC = 0.001

# HFF display settings for observed clusters.
HFF_RGB_BANDS = ["F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W"]
HFF_RGB_DISPLAY = {}      # library defaults: f_nu-scaled Lupton display with the warm-white balance of lenscluster.rgb
# --imaging jwst: NIRCam mosaics (CANUCS for MACS0416, DJA for AS1063) under data/imaging/<cluster>/{canucs_dr1,dja_v7.5} (scripts/download_imaging.py); blue F115W+F150W, green
# F150W+F200W, red F444W, the STScI/UNCOVER hue assignment with the four bands we hold.
JWST_RGB_BANDS = ["F115W", "F150W", "F200W", "F444W"]

VALID_CLUSTERS = "ARES, HERA, AS1063, AS1063_JWST, M0416_BERGAMINI2023, M0416_CANUCS2025"
FF_RGB_BANDS = ("F435W", "F606W", "F814W")


def imaging_dir(cluster_key: str, imaging: str) -> str:
    """data/imaging/<cluster>/<source> (scripts/download_imaging.py): the JWST mosaics (CANUCS DR1 for M0416, DJA v7.5 for
    AS1063), the BUFFALO HST mosaics, or the FF-SIMS simulated HST images of Hera and Ares."""
    cluster = cluster_key.split("_")[0]
    source = {"jwst": "canucs_dr1" if cluster == "m0416" else "dja_v7.5", "hst": "buffalo", "ffsims": "ffsims"}[imaging]
    return f"data/imaging/{cluster}/{source}"
FF_RGB_DISPLAY = RGBDisplayConfig(q=6.8, stretch=0.0158, minimum=0.00105, red_gain=0.62, green_gain=0.78, blue_gain=3.65)


def _usage() -> str:
    return (
        "Usage: xonsh run.xsh <cluster> [--campaign NAME] [--run-label NAME] [--halo-search lbfgs|optax-lbfgs|adam|svi] [--member-search lbfgs|optax-lbfgs|adam|svi] [--stage init|final|both] [--imaging hst|jwst] [--warm-start RUN_DIR] [--fast] [--dry-run]\n"
        "  --imaging jwst: draw the cluster mass figure on the NIRCam mosaics (file suffix _jwst) instead of the HST ones\n"
        "  --halo-search, --member-search: optimizer of the large-scale stage and of the member stage of the init (default lbfgs)\n"
        "  --likelihood: grf_gp (default; a Gaussian random field of unmodelled mass on the lens potential with a knee convergence\n"
        "          spectrum, sigma_alpha, lambda_t and the slope (uniform 2-6), plus the image-plane GP sigma_shared, ell, sigma_white), gp_image_plane\n"
        "          (the image-plane GP alone, per family) or\n"
        "          smoothed (the per-image source-plane floors sigma_def, sigma_extra)\n"
        "  --warm-start RUN_DIR: start the init search from a previous run's saved start (parameters matched by name; the\n"
        "          256-start search is skipped; NUTS still requires that run's init to have been robust)\n"
        "  --gp-blocks: grf_gp only: all (default for full runs: one shared mass field over every image, with the\n"
        "          field-wide translation, convergence and shear projected out), or system / family (independent fields per\n"
        "          system or family, about 5x cheaper per gradient, no projection; the default with --fast)\n"
        "  --gp-band-max ARCSEC: grf_gp only: the longest wavelength of the mass field (default 30); above 30 the knee prior\n"
        "          becomes log-normal with median min(20, ARCSEC / 2), log-width 1, up to min(ARCSEC, 80), started at its median\n"
        "  --gp-knee-fixed ARCSEC: grf_gp only: hold the knee at this scale (no images.gp_knee site)\n"
        "  --kinematics SRC[,SRC]: member stellar dispersions of these catalog sources in the likelihood (M0416: G26; AS1063_JWST: B25)\n"
        "  --drop-halos ID[,ID]: leave these large-scale components out of the model (M0416: e.g. los736,dark_s,4)\n"
        "  --member-shapes SIGMA|none: sample e1, e2 of every exact member around its catalog light shape with this prior width\n"
        "          (in e = (1-q)/(1+q)), or none to fix them at the catalog shapes (default: the cluster entry; M0416_CANUCS2025\n"
        "          samples at 0.1; --fast always fixes them)\n"
        "  --fix-relation k=v,...: hold member-relation parameters at these values (sigma km/s and cutkpc at the population's\n"
        "          prior magnitude; alpha_sigma, beta_radius), e.g. sigma=129,cutkpc=33,alpha_sigma=0.30,beta_radius=0.60 (R25)\n"
        "  --gp-knee-gamma G|free|flat: grf_gp only: the high-k slope of the mass field's spectrum, P ~ 1 / (1 + (k/k_t)^G): fixed at G (default 4),\n"
        "          sampled with a normal prior of width 0.5 about 4 on 2-6 (free), or uniform on 2-6 (flat);\n"
        "          2 truncated isothermal lumps, 4 NFW-like cusps, 6 cored)\n"
        "  --start-max-iter N, --start-tolerance TOL: the init's last optimization (the best mode on the NUTS target, every chain\n"
        "          starts there; on by default): it stops when the log posterior gains less than TOL over 50 iterations (default\n"
        "          0.1), or after N iterations (default 4000)\n"
        "  --warmup N, --samples N: NUTS draws per chain (override the defaults or --fast)\n"
        "  --dense-offsets all|centered|exact|none: members whose offsets join the dense mass block (default centered)\n"
        "  --far-offsets sample|fixed: far-field member offsets sampled (default) or pinned at the relation\n"
        "  --offset-tail student_t|normal: member deviation tails (default student_t, dof 4)\n"
        "  --bcg-cut-prior lognormal|uniform: BCG cut-radius prior (default lognormal, unbounded)\n"
        "  --skip-plots: no figures or tables after sampling (quick tests; the sampler diagnostics are in the log)\n"
        "  --fast: exploratory settings: members deflecting every image by < 0.03\" are dropped, far-field members allowed down to\n"
        "          2 cut radii from an image, classification at the current model, 16 search starts, 300+150 draws at tree depth 7\n"
        "          (check the audit's far-field error before trusting a fast run)\n"
        "  AS1063_JWST: Diego images + Beauchesne members, existing AS1063 halos and separate BCG; JWST figures by default\n"
        f"Valid clusters: {VALID_CLUSTERS}"
    )


# Command-line likelihood names -> (LikelihoodConfig.likelihood_mode, gp_kernel).  "gp_image_plane" is the library's mode
# name for every GP likelihood; the kernel says which model it is.
LIKELIHOODS = {"grf_gp": ("gp_image_plane", "knee"), "gp_image_plane": ("gp_image_plane", "image_plane"),
               "smoothed": ("smoothed", "image_plane")}


def _parse_args(argv: list[str]) -> dict:
    if len(argv) == 2 and argv[1] in {"--help", "-h"}:
        print(_usage())
        raise SystemExit(0)
    if len(argv) < 2:
        print(_usage())
        raise SystemExit(2)
    cluster = str(argv[1]).strip().upper()
    values = {"campaign": OUTPUT_DIR_LABEL, "run_label": None, "dry_run": False, "fast": False, "stage": "both", "imaging": None,
              "halo_search": "lbfgs", "member_search": "lbfgs", "likelihood": "grf_gp", "gp_blocks": None,
              "warm_start": None, "gp_band_max": None, "start_tolerance": None, "start_max_iter": None, "gp_knee_gamma": None, "gp_knee_fixed": None, "fix_relation": None, "relation_prior": None, "kinematics": None, "drop_halos": None, "member_shapes": None,
              "warmup": None, "samples": None, "dense_offsets": None, "far_offsets": None, "offset_tail": None, "bcg_cut_prior": "lognormal",
              "skip_plots": False}
    flag_to_key = {"--campaign": "campaign", "--run-label": "run_label", "--stage": "stage", "--imaging": "imaging",
                   "--halo-search": "halo_search", "--member-search": "member_search", "--likelihood": "likelihood",
                   "--gp-blocks": "gp_blocks", "--warm-start": "warm_start",
                   "--gp-band-max": "gp_band_max", "--start-tolerance": "start_tolerance", "--start-max-iter": "start_max_iter",
                   "--gp-knee-gamma": "gp_knee_gamma", "--gp-knee-fixed": "gp_knee_fixed", "--fix-relation": "fix_relation", "--relation-prior": "relation_prior", "--kinematics": "kinematics", "--drop-halos": "drop_halos", "--member-shapes": "member_shapes", "--warmup": "warmup", "--samples": "samples",
                   "--dense-offsets": "dense_offsets", "--far-offsets": "far_offsets", "--offset-tail": "offset_tail",
                   "--bcg-cut-prior": "bcg_cut_prior"}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if flag in {"--dry-run", "--fast", "--skip-plots"}:
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
    if values["imaging"] not in {None, "hst", "jwst"}:
        print("--imaging must be hst or jwst.")
        raise SystemExit(2)
    if values["likelihood"] not in LIKELIHOODS:
        print(f"--likelihood must be one of {', '.join(LIKELIHOODS)}.")
        raise SystemExit(2)
    if values["gp_blocks"] not in {None, "all", "system", "family"}:
        print("--gp-blocks must be all, system or family.")
        raise SystemExit(2)
    if values["member_shapes"] is not None and values["fast"]:
        print("--member-shapes does not apply with --fast: fast runs fix the member shapes at the catalog shapes.")
        raise SystemExit(2)
    if values["member_shapes"] is not None and values["member_shapes"].strip().lower() != "none":
        try:
            float(values["member_shapes"])
        except ValueError:
            print("--member-shapes takes a prior width or none.")
            raise SystemExit(2)
    if values["gp_blocks"] is not None and values["likelihood"] != "grf_gp":
        print("--gp-blocks applies to the grf_gp likelihood only.")
        raise SystemExit(2)
    try:
        for key, kind in (("gp_band_max", float), ("start_tolerance", float), ("start_max_iter", int), ("gp_knee_fixed", float)):
            if values[key] is not None:
                values[key] = kind(values[key])
                if values[key] <= 0:
                    raise ValueError
    except ValueError:
        print("--gp-band-max, --start-tolerance, --start-max-iter and --gp-knee-gamma take positive numbers.")
        raise SystemExit(2)
    if values["gp_band_max"] is not None and values["likelihood"] != "grf_gp":
        print("--gp-band-max applies to the grf_gp likelihood only.")
        raise SystemExit(2)
    if values["gp_knee_gamma"] is not None and values["likelihood"] != "grf_gp":
        print("--gp-knee-gamma applies to the grf_gp likelihood only.")
        raise SystemExit(2)
    for key, allowed in (("dense_offsets", {"all", "centered", "exact", "none"}), ("far_offsets", {"sample", "fixed"}),
                         ("offset_tail", {"normal", "student_t"}), ("bcg_cut_prior", {"lognormal", "uniform"})):
        if values[key] is not None and values[key] not in allowed:
            print(f"--{key.replace('_', '-')} must be one of {', '.join(sorted(allowed))}.")
            raise SystemExit(2)
    try:
        for key in ("warmup", "samples"):
            if values[key] is not None:
                values[key] = int(values[key])
                if values[key] <= 0:
                    raise ValueError
    except ValueError:
        print("--warmup and --samples take positive integers.")
        raise SystemExit(2)
    return dict(cluster=cluster, campaign_name=str(values["campaign"]).strip(), run_label=values["run_label"],
                dry_run=values["dry_run"], fast=values["fast"], stage=values["stage"], imaging=values["imaging"],
                halo_search=values["halo_search"], member_search=values["member_search"], likelihood=values["likelihood"],
                gp_blocks=values["gp_blocks"], warm_start=values["warm_start"],
                gp_band_max=values["gp_band_max"], start_tolerance=values["start_tolerance"], start_max_iter=values["start_max_iter"],
                gp_knee_gamma=values["gp_knee_gamma"], gp_knee_fixed=values["gp_knee_fixed"], fix_relation=values["fix_relation"], relation_prior=values["relation_prior"], kinematics=values["kinematics"], drop_halos=values["drop_halos"], member_shapes=values["member_shapes"], warmup=values["warmup"], samples=values["samples"],
                dense_offsets=values["dense_offsets"], far_offsets=values["far_offsets"], offset_tail=values["offset_tail"],
                bcg_cut_prior=values["bcg_cut_prior"], skip_plots=values["skip_plots"])


# The solver's tuned defaults live in lenscluster.config; a cluster entry carries only what is specific to it: the
# image catalog, the cosmology, where the cluster-scale halos sit, the member catalog, the truth maps and the
# simulation's softening.  Both simulated clusters are at z = 0.5 with F160W magnitudes, where L* is m_F160W = 19.82.
FF_SIMS_CLUSTERS = {
    "ARES": {
        "cluster_key": "ares",
        "image_catalog": "catalogs/ares/images.yaml",
        "cosmology": CosmologyConfig(H0=70.4, Om0=0.272, Ode0=0.728),
        # One cluster-scale halo at the light peak of each clump of the MOKA cluster (Meneghetti et al. 2017) plus shear.
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 20.0, -32.0, 0.5, v_disp=950.0),
            DPIEHaloConfig.cluster_scale("2", -40.0, 40.0, 0.5, v_disp=950.0),
            ShearHaloConfig.free("S1", 0.5),
        ),
        "member_population": MemberPopulationConfig.bergamini2019("potfile_1", "catalogs/ares/members.yaml", 0.5, prior_mag0=19.82),
        # The brightest cluster galaxies stay in the scaling relation, as on Hera: independent halos cost parameters, made
        # the chains multi-well and did not improve the fit.
        "independent_member_halos": (),
        "kappa_true_fits": "data/imaging/ares/ffsims/kappa_z9_0.fits",
        "gammax_true_fits": "data/imaging/ares/ffsims/gammax_z9_0.fits",
        "gammay_true_fits": "data/imaging/ares/ffsims/gammay_z9_0.fits",
        # Ares is semi-analytic: its subhalos are analytic profiles, so there is no resolution softening to match.
        "softening_length_kpc": 0.0,
    },
    "HERA": {
        "cluster_key": "hera",
        "image_catalog": "catalogs/hera/images.yaml",
        "cosmology": CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
        "halos": (
            DPIEHaloConfig.cluster_scale("1", 19.54080009, 2.37820005, 0.507, v_disp=800.0),
            DPIEHaloConfig.cluster_scale("2", 0.001, 0.003, 0.507, v_disp=700.0),
            ShearHaloConfig.free("S1", 0.507),
        ),
        "member_population": MemberPopulationConfig.bergamini2019("potfile_1", "catalogs/hera/members.yaml", 0.507, prior_mag0=19.82),
        "independent_member_halos": (),
        "kappa_true_fits": "data/imaging/hera/ffsims/kappa_z9_0.fits",
        "gammax_true_fits": "data/imaging/hera/ffsims/gammax_z9_0.fits",
        "gammay_true_fits": "data/imaging/hera/ffsims/gammay_z9_0.fits",
        # Hera is an N-body cluster (H0 = 72, softening 2.3 h^-1 kpc = 3.19 kpc): added in quadrature to the member cores so
        # that their cusps match the simulation's resolution; the dPIE total mass is independent of the core.
        "softening_length_kpc": 2.3 / 0.72,
    },
}


# Native configuration transcribed from
# catalogs/as1063/bergamini2019/source/Bergamini_RXCJ2248_lenstool_Normal.par.
# That model gives radii in arcseconds; native configurations require kpc.
def _hff_clusters():
    AS1063_KPC_PER_ARCSEC = kpc_per_arcsec_from_config(0.348, {"H0": 70.0, "Om0": 0.3})
    AS1063_CATALOG_DIR = "catalogs/as1063/bergamini2019"
    AS1063_MSTAR_F160W = 18.33
    clusters = {
        "AS1063": {
            "cluster_key": "as1063",
            "reference": ReferenceFrameConfig(reference=3, ra0_deg=342.18321, dec0_deg=-44.530878),
            "image_catalog": f"{AS1063_CATALOG_DIR}/images.yaml",
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
                # External shear, free amplitude and angle (2026-11-03), as for CANUCS: the GRF projection removes the field's
                # field-wide shear on the assumption that a free external shear absorbs it.  AS1063_JWST inherits it.
                ShearHaloConfig.free("shear", 0.348, gamma_upper=0.3),
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
            # Instrumental astrometric error per image (HST WFC3/IR F160W): the measurement floor; the gp likelihood samples sigma_white above it.
            "sigma_inst_arcsec": SIGMA_INST_ARCSEC,
            # Classify at the current model (reclassified at the MAP) rather than at the prior's 0.975 quantile, which for a
            # log-uniform cut radius is 90 kpc (198 of 222 exact).  The library defaults (guard 3, inflation 3) give the
            # sampled r_cut* a factor of three before any far-field member reaches the guard.
            "classification_prior_quantile": None,
        },
        **_m0416_bergamini2023_cluster(),
    }

    # JWST constraints from Diego 2601.11704v2 and the Beauchesne member release.
    # Retain the existing AS1063 cluster/gas model; this is not a transcription
    # of either paper's complete mass model. The new member file omits BCG+ICL.
    # Carry over Bergamini member 1809 as a separate round dPIE at its light
    # position. Its initial mass and prior scales follow the existing neutral
    # relation evaluated at m_F160W=16.176, with alpha=0.25 and beta=0.5.
    from lenscluster.sky import radec_to_offsets_arcsec
    old = clusters["AS1063"]
    pop = old["member_population"]
    bcg_x, bcg_y = radec_to_offsets_arcsec(342.183237, -44.530870,
                                         old["reference"].ra0_deg, old["reference"].dec0_deg)
    luminosity = 10.0 ** (0.4 * (AS1063_MSTAR_F160W - 16.176))
    fs, fr = luminosity ** 0.25, luminosity ** 0.5
    bcg = DPIEHaloConfig(
        id="bcg_1809", x_centre=float(bcg_x), y_centre=float(bcg_y),
        ellipticite=0.0, angle_pos=0.0, core_radius_kpc=pop.corekpc,
        cut_radius_kpc=pop.cutkpc * fr, v_disp=pop.sigma * fs, z_lens=pop.z_lens,
        priors={
            "v_disp": PriorConfig("truncated_normal", mean=200.0 * fs, std=100.0 * fs,
                                   lower=50.0 * fs, upper=500.0 * fs),
            "cut_radius_kpc": _bcg_cut_prior(pop.cutkpc * fr, (2.0 * fr, 100.0 * fr)),
        },
    )
    jwst_dir = "catalogs/as1063/diego2026"
    clusters["AS1063_JWST"] = {
        **old, "cluster_key": "as1063_jwst", "imaging": "jwst",
        "image_catalog": f"{jwst_dir}/images.yaml",
        "member_population": replace(pop, catalog_path=f"{jwst_dir}/members.yaml", band="F160W",
                                     kinematics_sources=()),
        "halos": (*old["halos"], bcg),
    }
    return clusters


# MACS J0416, two image catalogs over the same mass model: the HST + MUSE catalog of Bergamini et al. 2023 (237 spectroscopic
# images in 88 families; the files are their 2022 release) and the JWST CANUCS gold catalog.  Native configuration transcribed from catalogs/m0416/bergamini2023/source/Bergamini22_MACS0416_Normal.par:
# four dark-matter dPIE halos with their prior boxes, a free foreground galaxy, four fixed hot-gas clumps,
# and the members (including galaxy 8971) on the neutral relation.
def _m0416_bergamini2023_cluster():
    kpc = kpc_per_arcsec_from_config(0.396, {"H0": 70.0, "Om0": 0.3})
    catalog_dir = "catalogs/m0416/bergamini2023"
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

    # The CANUCS independent halos (Rihtarsic et al. 2025): the four dPIE components and the shape limits of
    # hlsp_canucs_jwst-hst_multi_macs0416-lenstool-param_multi_v1_model.txt, with physical priors in place of the published
    # starts and uniform boxes.  Offsets from the CANUCS reference.  Halos 1 and 2 sit on the NE and SW BCGs (galaxies 2526
    # and 3158), halo 3 on the NW member group; halo 4 has no light counterpart and is centred where earlier strong-lensing
    # models put it.  Centres: normal, truncated to a box; halos 2 and 4 (14" apart in y) get disjoint boxes so they cannot
    # swap.  Cores: log-uniform.  v_disp: normal, truncated.  Cut radii fixed at the CANUCS value, 2000" (10.7 Mpc) for all
    # four, as in the published models: the images reach ~270 kpc, so a free cut slides along its degeneracy with v_disp
    # (w250s100_fast_gpblocksall: halo 3 and 4 cuts at their 800 kpc cap).  Cuts of 1500 kpc (main) and 300 kpc (groups)
    # left a correlated residual that the image-plane GP took at its 3" cap (nov2d_diag, 2026-11-02; .agents/gp-knee-plan.md).
    # Galaxy 8971 (their potential 6) is on the member scaling relation instead.
    def canucs_halo(id, centre, *, centre_sigma, half_width, core_kpc, v_disp, cut_kpc, e_box=None, angle_box=None,
                    x_box=None, y_box=None, e_soft_cap=None):
        # centre (arcsec): normal of width centre_sigma, truncated to +-half_width (or to x_box, y_box when given).
        # core_kpc (lo, hi): log-uniform.  v_disp (mean, std, lo, hi): normal, truncated.  cut_kpc: fixed.
        # Every site starts at its prior centre, so a run started here (or warm-started from one) is inside the priors.
        x, y = centre
        sigma_x, sigma_y = centre_sigma if isinstance(centre_sigma, tuple) else (centre_sigma, centre_sigma)
        x_box = x_box or (x - half_width, x + half_width)
        y_box = y_box or (y - half_width, y + half_width)
        core_start = (core_kpc[0] * core_kpc[1]) ** 0.5                  # the log-uniform middle, where the core starts
        priors = {"x_centre": PriorConfig("truncated_normal", mean=x, std=sigma_x, lower=x_box[0], upper=x_box[1]),
                  "y_centre": PriorConfig("truncated_normal", mean=y, std=sigma_y, lower=y_box[0], upper=y_box[1]),
                  "core_radius_kpc": PriorConfig("uniform", lower=core_kpc[0], upper=core_kpc[1], step=1.0),
                  "v_disp": PriorConfig("truncated_normal", mean=v_disp[0], std=v_disp[1], lower=v_disp[2], upper=v_disp[3])}
        e, angle = 0.0, 0.0
        if e_box is not None:
            priors["ellipticite"] = PriorConfig("uniform", lower=e_box[0], upper=e_box[1], step=0.05)
            e = 0.5 * (e_box[0] + e_box[1])
        if angle_box is not None:
            priors["angle_pos"] = PriorConfig("uniform", lower=angle_box[0], upper=angle_box[1], step=0.1)
            angle = 0.5 * (angle_box[0] + angle_box[1])
        if e_soft_cap is not None:
            e = min(e, 0.5 * (e_box[0] + e_soft_cap))                     # start below the smooth cap
        return DPIEHaloConfig(id=id, x_centre=x, y_centre=y, ellipticite=e, angle_pos=angle, core_radius_kpc=core_start,
                              cut_radius_kpc=cut_kpc, v_disp=v_disp[0], z_lens=z, priors=priors, ellipticity_soft_cap=e_soft_cap)

    main_halo = dict(core_kpc=(10.0, 100.0), v_disp=(650.0, 150.0, 400.0, 1000.0), cut_kpc=2000.0 * kpc)
    group_halo = dict(core_kpc=(5.0, 100.0), v_disp=(400.0, 150.0, 150.0, 800.0), cut_kpc=2000.0 * kpc)
    canucs_halos = (
        # Halo 1 pressed against the hard ellipticite bound 0.9 (|e| 0.61 against 0.627), which set the NUTS step size: the
        # box now reaches 0.94 (|e| 0.70; its corners stay below |e| = 1) and a smooth cap takes over at 0.9 (2026-11-03).
        # Halo 1 and 2 centres: width 1.5" (was 2") around the BCGs (2026-11-04), to slow the centre-ellipticity ridges NUTS
        # mixed slowest; the boxes stay, because the data put halo 1 3" south and halo 2 2.2" north of their BCGs.
        canucs_halo("1", (0.2, 0.0), centre_sigma=1.5, half_width=6.0, e_box=(0.2, 0.94), angle_box=(100.0, 180.0), e_soft_cap=0.9,
                    **main_halo),                                                                                       # NE BCG
        canucs_halo("2", (20.3, -35.8), centre_sigma=1.5, half_width=5.0, e_box=(0.2, 0.9), angle_box=(90.0, 170.0),
                    **main_halo),                                                                                       # SW BCG
        canucs_halo("3", (-33.0, 9.0), centre_sigma=3.0, half_width=6.0, **group_halo),                                 # NW group, circular
        # Halo 4 has room in the south (2026-11-03): centre width 10" in x on 0-40" and 5" in y on -62 to -35" (the better
        # basin of nov5_reltest put it at y = -43 against the old -40 edge, 2026-11-04), core up to 150 kpc, ellipticity box
        # to 0.94 with the smooth cap at 0.9.  With the image parities and the two southern halos below, this init gained 31
        # in log L and brought the lens-only image rms from 0.77" to 0.51" (nov3o_parity south_all).  A group-scale prior
        # (core 5-40 kpc, v_disp 300 +- 100) was tried on 2026-11-05: the large-halo mode pressed on the core cap and the
        # init lost 13.5 in log L, so the room stays.
        canucs_halo("4", (22.0, -48.0), centre_sigma=(10.0, 5.0), half_width=6.0, x_box=(0.0, 40.0), y_box=(-62.0, -35.0),
                    e_box=(0.2, 0.94), angle_box=(0.0, 180.0), e_soft_cap=0.9, core_kpc=(5.0, 150.0),
                    **{k: v for k, v in group_halo.items() if k != "core_kpc"}),                                    # south, dark
        DPIEHaloConfig(id="5", x_centre=31.96420, y_centre=-65.55168, ellipticite=0.0, angle_pos=0.0, core_radius_kpc=0.0001 * kpc,
                       cut_radius_kpc=33.6321 * kpc, v_disp=235.2454, z_lens=z,                                         # foreground galaxy
                       priors={"v_disp": PriorConfig("uniform", lower=50.0, upper=350.0, step=0.1),
                               "cut_radius_kpc": PriorConfig("uniform", lower=5.0 * kpc, upper=100.0 * kpc, step=0.1 * kpc)}),
        # Two round southern components at the cluster redshift (2026-11-03), where the shared mass field had its strongest
        # positive peaks: the z = 0.736 background galaxy of Chirivi et al. 2018 (F160W 19.8, at (38.5, -48.2)), and a dark
        # lump at the field's strongest peak, (0.5, -57), with no galaxy within 5".
        DPIEHaloConfig(id="los736", x_centre=38.5, y_centre=-48.2, ellipticite=0.0, angle_pos=0.0, core_radius_kpc=0.0001 * kpc,
                       cut_radius_kpc=float((5.0 * 100.0) ** 0.5) * kpc, v_disp=120.0, z_lens=z,
                       priors={"v_disp": PriorConfig("uniform", lower=30.0, upper=350.0, step=0.1),
                               "cut_radius_kpc": PriorConfig("uniform", lower=5.0 * kpc, upper=100.0 * kpc, step=0.1)}),
        DPIEHaloConfig(id="dark_s", x_centre=0.5, y_centre=-57.0, ellipticite=0.0, angle_pos=0.0, core_radius_kpc=0.0001 * kpc,
                       cut_radius_kpc=float((5.0 * 300.0) ** 0.5) * kpc, v_disp=150.0, z_lens=z,
                       priors={"v_disp": PriorConfig("uniform", lower=30.0, upper=500.0, step=0.1),
                               "cut_radius_kpc": PriorConfig("uniform", lower=5.0 * kpc, upper=300.0 * kpc, step=0.1)}),
        *halos[-4:],                        # the four fixed hot-gas clumps, identical in the CANUCS configuration
        # External shear, free amplitude and angle: not in the CANUCS configuration, but worth +35 in the lens log L for 2
        # parameters at the fast init (nov01b), and it moves the mass field's knee from 15" to 5" (.agents/gp-knee-plan.md).
        ShearHaloConfig.free("shear", z, gamma_upper=0.3),
    )
    assert [h.id for h in halos[-4:]] == ["gas1", "gas2", "gas3", "gas4"]

    def entry(key, *, reference, image_catalog, member_catalog, sigma_inst_arcsec, halos=halos, kinematics_sources=(), fit_member_core=True,
              independent_member_halos=(), scaling=None):
        return {
            "cluster_key": key,
            "reference": ReferenceFrameConfig(reference=3, ra0_deg=reference[0], dec0_deg=reference[1]),
            "image_catalog": image_catalog,
            "cosmology": CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            "halos": halos,
            # Neutral member priors at L* (see AS1063); Bergamini et al. 2023 found sigma 248 +- 28 km/s and r_cut 56 +- 30 kpc at
            # the BCG magnitude 17.02 with a fixed slope of 0.3, i.e. 158 km/s and 29 kpc at m*, inside these priors.
            "member_population": replace(MemberPopulationConfig.neutral("potfile_1", member_catalog, z, prior_mag0=mstar_f160w),
                                         kinematics_sources=tuple(kinematics_sources)),
            "scaling": scaling if scaling is not None else ScalingModelConfig.neutral(fit_corekpc=fit_member_core),
            "independent_member_halos": tuple(independent_member_halos),
            "softening_length_kpc": 0.0,
            "softening_length_prior_log_sigma": 0.15,
            "sigma_inst_arcsec": sigma_inst_arcsec,
            "classification_prior_quantile": None,
            # Far-field error at 2 r_cut is < 1 uas (oct28a); the default inflation 3 keeps the headroom.
            "far_field_cut_radii": 2.0,
            # The image-plane GP needed a 0.5 mas floor here (203.b/c depth 2.5 mas, 211.b/c 3.8 mas at the oct29a best fit).
            # Under the knee kernel the pair depth error is the common-mode deflection error, not the floor, so the
            # default 1.5 mas floor applies; pass gp_floor_arcsec 0.0005 again for a --likelihood gp_image_plane run.
        }

    canucs_dir = "catalogs/m0416/canucs2025"
    return {
        # HST + MUSE: 237 spectroscopic images in 88 families (Bergamini et al. 2023), HST astrometry; images.yaml groups
        # them into 60 systems, 12 with knots (scripts/group_bergamini_knots.py).
        "M0416_BERGAMINI2023": entry("m0416_bergamini2023", reference=(64.0381417, -24.0674722), image_catalog=f"{catalog_dir}/images.yaml",
                                     member_catalog=f"{catalog_dir}/members.cat", sigma_inst_arcsec=SIGMA_INST_ARCSEC),
        # JWST + HST: the CANUCS gold catalog, 303 spectroscopic images in 83 systems, 15 with knots, 111 knot-families
        # (Rihtarsic et al. 2025; scripts/convert_canucs_catalog.py), NIRCam positions in the frame of their lens model; their
        # member catalog (+ galaxy 8971, on the relation), their independent halos and reference centre.  The members follow the
        # same relation, priors, free slopes, scatter, deviations and free core as for Bergamini.  The MUSE stellar dispersions
        # of 52 members (Granata et al. 2026; members.yaml source G26) are off: with them on, or with the member core fixed,
        # the mass field shut off and the image-plane GP took the residual at its 3" cap (nov2d_diag, 2026-11-02).
        # To use them: kinematics_sources=("G26",).
        # The two BCGs (2526 on halo 1, 3158 on halo 2) are independent dPIEs, off the relation: on it, their velocity offsets
        # traded against the relation's slope and normalization, and the chains sat 67-135 km/s apart in sigma* (R-hat 4.6;
        # nov2e w250s100_fast_canucscuts_corefree_nokin_band100).  Centre, round shape and the 0.15 kpc core fixed at the light
        # (each sits on a cluster halo that carries the ellipticity); v_disp and r_cut free around Bergamini et al. (2023),
        # 248 km/s and 56 kpc at the BCG magnitude, with round numbers and wide truncations.
        # The slope alpha gets a Faber-Jackson prior (L ~ sigma^4): 0.25 +- 0.05 on 0.1-0.4; with 0.25 +- 0.10 on 0-0.5 two of
        # four chains sat at 0.45-0.49 in that run.
        "M0416_CANUCS2025": entry("m0416_canucs2025", reference=(64.03821668520617, -24.06750122335985), image_catalog=f"{canucs_dir}/images.yaml",
                                  member_catalog=f"{canucs_dir}/members.yaml", sigma_inst_arcsec=SIGMA_INST_ARCSEC, halos=canucs_halos,
                                  kinematics_sources=(), fit_member_core=True,
                                  independent_member_halos=tuple(
                                      IndependentMemberHaloConfig(population_id="potfile_1", catalog_id=bcg, id=f"bcg_{bcg}", priors={
                                          "x_centre": PriorConfig(), "y_centre": PriorConfig(), "ellipticite": PriorConfig(),
                                          "angle_pos": PriorConfig(), "core_radius_kpc": PriorConfig(),
                                          "v_disp": PriorConfig("truncated_normal", mean=250.0, std=50.0, lower=100.0, upper=400.0),
                                          "cut_radius_kpc": _bcg_cut_prior(56.0, (10.0, 200.0))})
                                      for bcg in ("2526", "3158")),
                                  # Student-t member deviations (dof 4): five members needed cut radii ~10x off the relation and
                                  # pushed the normal radius scatter to 1.6-1.8 against its 0.3 prior (2026-11-03).
                                  scaling=replace(ScalingModelConfig.neutral(fit_corekpc=True), alpha_sigma_prior_std=0.05,
                                                  alpha_sigma_prior_lower=0.1, alpha_sigma_prior_upper=0.4,
                                                  member_deviation_tail="student_t", member_deviation_dof=4.0,
                                                  # the member core sat at 0.67-0.77 kpc (z +2.1 to +2.3) at the starts
                                                  corekpc_prior_log_sigma=1.0,
                                                  # Exact members' shapes sampled around the CANUCS DR1 light shapes in members.yaml
                                                  # (2026-11-05): +25 in log L over the fixed light shapes in basin B for changes of
                                                  # ~0.02 in e; --fast fixes them at the catalog shapes.
                                                  member_shape_scatter=0.1)),
    }


CLUSTERS = set(FF_SIMS_CLUSTERS) | {"AS1063", "AS1063_JWST", "M0416_BERGAMINI2023", "M0416_CANUCS2025"}


def build_runtime(cores: int) -> RuntimeConfig:
    return RuntimeConfig(
        seed=seed,
        chains=cores,
        resume=True,
        numpyro_print_summary=True,
        nuts_chain_method="parallel",
        # Shared CPU budget = concurrent chains * this value (Linux pinning).
        # macOS skips pinning; parallel chains still use one BLAS thread each.
        cpu_threads_per_chain=threads_per_chain_if_supported(2),
    )


def build_config(
    cluster: str,
    *,
    cores: int,
    campaign_name: str,
    run_label: str | None = None,
    likelihood: str = "grf_gp",
    gp_blocks: str | None = None,
    runtime: RuntimeConfig | None = None,
    fast: bool = False,
    imaging: str | None = None,
    halo_search: str = "lbfgs",
    member_search: str = "lbfgs",
    warm_start: str | None = None,
    gp_band_max: float | None = None,
    start_tolerance: float | None = None,
    start_max_iter: int | None = None,
    gp_knee_gamma: float | None = None,
    gp_knee_fixed: float | None = None,
    fix_relation: str | None = None,
    relation_prior: str | None = None,
    kinematics: str | None = None,
    drop_halos: str | None = None,
    member_shapes: str | None = None,
) -> LensClusterSolverConfig:
    """The solver configuration for one cluster: library defaults plus the cluster entry, the command-line choices and,
    with ``fast``, the exploratory overrides."""
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {', '.join(sorted(CLUSTERS))}; got {cluster!r}.")
    is_hff = cluster not in FF_SIMS_CLUSTERS
    cluster_config = _hff_clusters()[cluster] if is_hff else FF_SIMS_CLUSTERS[cluster]
    imaging = imaging or cluster_config.get("imaging", "hst")
    config = LensClusterSolverConfig(
        model=LensModelConfig(
            reference=cluster_config.get("reference", ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0)),
            cosmology=cluster_config["cosmology"],
            large_halos=cluster_config["halos"],
            independent_member_halos=cluster_config["independent_member_halos"],
            member_populations=(cluster_config["member_population"],),
            image_constraints=ImageConstraintsConfig(catalog_path=cluster_config["image_catalog"],
                                                     sigma_inst_arcsec=float(cluster_config.get("sigma_inst_arcsec", ImageConstraintsConfig.sigma_inst_arcsec))),
        ),
        # No selection term: in the oct28a M0416 run sigma_sel and sigma_def fit equally well and lambda_sel stayed at its prior.
        # Covariance blocks: all by default for full runs, system for --fast (2026-11-03); one shared mass field over every image with the field-wide
        # translation, convergence and shear projected out (absorbed by the sources and the free external shear).  System
        # blocks give each system an independent field: 5x cheaper per gradient, but their posterior differs from the shared
        # field's (importance reweighting kept 1 of 200 draws), and the projection needs one covariance, so they are for
        # quick tests (--gp-blocks system).  The init search still runs on system blocks; the chain-start step and NUTS use all.
        likelihood=LikelihoodConfig(likelihood_mode=LIKELIHOODS[likelihood][0], sigma_sel_prior=None, gp_kernel=LIKELIHOODS[likelihood][1],
                                    gp_blocks=gp_blocks or ("system" if fast else "all"),
                                    gp_knee_project_modes=(("translation", "convergence", "shear")
                                                           if likelihood == "grf_gp" and (gp_blocks or ("system" if fast else "all")) == "all"
                                                           else ()),
                                    # The GRF slope fixed at 4 (a population of NFW-cusped lumps): sampled, it drifted to 5.3 in
                                    # w250s100_fast_gpblocksall without the data pinning it.
                                    # Sampled slope: "free" is a normal of width 0.5 about 4 (the NFW-cusp value) truncated to 2-6;
                                    # "flat" the uniform on 2-6 (nov5a, nov6: 4.6-4.7 +- 0.8, pressing the knee onto its cap).
                                    gp_knee_gamma_prior=(2.0, 6.0) if str(gp_knee_gamma) in ("free", "flat") else None,
                                    gp_knee_gamma_sigma=0.5 if str(gp_knee_gamma) == "free" else None,
                                    gp_knee_gamma=4.0 if gp_knee_gamma in (None, "free", "flat") else float(gp_knee_gamma),
                                    # The field's wavelengths from 0.2" to 100", as in the good fast run of 2026-11-02
                                    # (nov1a w250s100_fast: knee 6.6", GP ell 1.4"); the knee prior keeps it near the galaxy scale.
                                    gp_knee_scales_arcsec=(0.2, 100.0),
                                    # The knee near the galaxy scale: 6" is the knee of truncated isothermal halos with r_t = 5 kpc
                                    # (lambda_t = 6.4 r_t), 2.5-8" spans r_t = 2-6 kpc.  Left free it goes to 14-25" and carries
                                    # the halo-scale misfit (compare_grf_spectra, 2026-11-02); with the cap at 15" every all-block
                                    # run sat on it (nov4b, nov5a, nov6: 14-15", 2026-11-07).  Capped at 8" (2026-11-07) so the
                                    # field stays at the galaxy scale and the large-scale misfit has to go elsewhere; ln-width 0.6 kept.
                                    gp_knee_prior=LogNormalPriorConfig(median=6.0, log_sigma=0.6, lower=2.5, upper=8.0),
                                    gp_knee_start_arcsec=6.0,
                                    # The rms deflection error of that substructure at z_ref: 0.08" median, 0.05-0.13" at 1 sigma,
                                    # truncated at 0.2" (galaxy-scale fits on 2026-11-02: 0.07-0.11"; at 0.2-0.3" it was carrying
                                    # the halo-scale misfit).
                                    gp_sigma_alpha_prior=LogNormalPriorConfig(median=0.08, log_sigma=0.5, upper=0.5),
                                    gp_sigma_alpha_start_arcsec=0.08,
                                    # The image-plane GP's correlation length near the measurement scale (1", between 0.4"
                                    # and 3"): fitted free it went to 4.4" and did the mass field's work.
                                    gp_knee_ell_prior=LogNormalPriorConfig(median=1.0, log_sigma=0.4, lower=0.4, upper=3.0),
                                    gp_knee_ell_start_arcsec=1.0,
                                    gp_floor_arcsec=float(cluster_config.get("gp_floor_arcsec", LikelihoodConfig.gp_floor_arcsec))),
        paths=RunPathsConfig(output_dir=f"results/{campaign_name}/{cluster_config['cluster_key']}",
                             run_name=run_label if run_label is not None else "smoothed-local-image-plane",
                             figure_prefix=cluster_config["cluster_key"]),
        runtime=runtime if runtime is not None else build_runtime(cores),
        search=SearchConfig(n_starts=256, halo_method=halo_search, member_method=member_search, warm_start=warm_start),
        sampling=SamplingConfig(warmup=250, samples=100, max_tree_depth=8, target_accept=0.8),
        scaling=replace(cluster_config.get("scaling", ScalingModelConfig()), softening_length_kpc=float(cluster_config.get("softening_length_kpc", 0.0))),
        # Direct (centered) deviations for the members the data pin, unit offsets for the rest (centering "auto", 2026-11-04):
        # with unit offsets the radius scatter traded against the pinned members' offsets (correlations -0.6 to -0.67 for 7764,
        # 9305, 1297) along a curved ridge.  The init's chain-start step pins the scatters, so the centered optimizer trap
        # (scatters driven to zero) cannot recur.
        members=MemberConfig(centering="auto", far_field_cut_radii=float(cluster_config.get("far_field_cut_radii", MemberConfig.far_field_cut_radii)),
                             cut_radius_inflation=float(cluster_config.get("cut_radius_inflation", MemberConfig.cut_radius_inflation)),
                             classification_prior_quantile=cluster_config.get("classification_prior_quantile", MemberConfig.classification_prior_quantile)),
        truth=TruthRecoveryConfig(kappa_true_fits=cluster_config.get("kappa_true_fits"), gammax_true_fits=cluster_config.get("gammax_true_fits"),
                                  gammay_true_fits=cluster_config.get("gammay_true_fits")),
        image_catalog=ImageCatalogCutoutConfig(
            candidate_catalog_path=M0416_CANDIDATES if cluster == "M0416_CANUCS2025" else None,
            rgb_critical_image_cutouts=M0416_RGB_IMAGE_CUTOUTS if cluster.startswith("M0416") else (),
            rgb_critical_fold_cutouts=M0416_RGB_FOLD_CUTOUTS if cluster.startswith("M0416") else (),
            image_dir=imaging_dir(cluster_config["cluster_key"], imaging if is_hff else "ffsims"),
            image_scale="auto",
            bands=(tuple(JWST_RGB_BANDS) if imaging == "jwst" else tuple(HFF_RGB_BANDS)) if is_hff else FF_RGB_BANDS,
            rgb=RGBDisplayConfig(**HFF_RGB_DISPLAY) if is_hff else FF_RGB_DISPLAY,
            figure_suffix="_jwst" if (is_hff and imaging == "jwst") else "",
        ),
    )
    if fast:
        # Exploratory: far-field members allowed down to 2 cut radii, classified at the current model with 3x headroom on the
        # cut radius, fewer starts, shorter chains.  No impact cut (2026-11-03): every catalog member stays, as in a full run;
        # dropping members moved their mass into halo 2 and the field (0.05": knee 10 -> 20").
        config = replace(
            config,
            search=replace(config.search, n_starts=256),
            image_diagnostics=replace(config.image_diagnostics, fold_max_draws=16,
                                      fold_pair_max_separation_arcsec=3.0),
            sampling=SamplingConfig(warmup=500, samples=250, max_tree_depth=8, target_accept=0.8),
            members=replace(config.members, far_field_cut_radii=2.0, classification_prior_quantile=None, min_deflection_arcsec=None,
                            cut_radius_inflation=3.0),
            # Member shapes fixed at the catalog (light) shapes: no sampled ellipticity in fast runs.
            scaling=replace(config.scaling, member_shape_scatter=None),
        )
    if gp_band_max is not None and float(gp_band_max) > 30.0:
        # A longer band for the mass field: the knee left free up to min(band, 80") (the tables end at 80"), centred at 20"
        # (compare_grf_spectra.py: the knee settles at 14-17" with a 100" band).
        median = min(20.0, 0.5 * float(gp_band_max))
        config = replace(config, likelihood=replace(config.likelihood, gp_knee_scales_arcsec=(0.2, float(gp_band_max)),
                                                    gp_knee_prior=LogNormalPriorConfig(median=median, log_sigma=1.0, upper=min(float(gp_band_max), 80.0)),
                                                    gp_knee_start_arcsec=median))
    elif gp_band_max is not None:
        config = replace(config, likelihood=replace(config.likelihood, gp_knee_scales_arcsec=(0.2, float(gp_band_max))))
    if gp_knee_fixed is not None:
        config = replace(config, likelihood=replace(config.likelihood, gp_knee_fixed_arcsec=float(gp_knee_fixed)))
    if kinematics:
        pops = list(config.model.member_populations)
        pops[0] = replace(pops[0], kinematics_sources=tuple(s.strip() for s in kinematics.split(",") if s.strip()))
        config = replace(config, model=replace(config.model, member_populations=tuple(pops)))
    if drop_halos:
        drop = {s.strip() for s in drop_halos.split(",") if s.strip()}
        unknown = drop - {h.id for h in config.model.large_halos}
        if unknown:
            raise SystemExit(f"--drop-halos: no large-scale component {sorted(unknown)}; have {[h.id for h in config.model.large_halos]}")
        config = replace(config, model=replace(config.model, large_halos=tuple(h for h in config.model.large_halos if h.id not in drop)))
    if member_shapes is not None:
        scatter = None if member_shapes.strip().lower() == "none" else float(member_shapes)
        config = replace(config, scaling=replace(config.scaling, member_shape_scatter=scatter))
    if relation_prior:
        # k=mean:std at prior_mag0 for cutkpc or sigma (truncated normal inside the existing bounds); mean:std for alpha_sigma, beta_radius.
        given = {k.strip(): tuple(float(x) for x in v.split(":")) for k, v in (item.split("=") for item in relation_prior.split(","))}
        pops = list(config.model.member_populations)
        for key in ("sigma", "cutkpc"):
            if key in given:
                old = getattr(pops[0], f"{key}_prior")
                pops[0] = replace(pops[0], **{key: given[key][0], f"{key}_prior": PriorConfig("truncated_normal", mean=given[key][0], std=given[key][1],
                                                                                              lower=old.lower, upper=old.upper, step=old.step)})
        slopes = {}
        for key in ("alpha_sigma", "beta_radius"):
            if key in given:
                slopes[f"{key}_prior_mean"], slopes[f"{key}_prior_std"] = given[key]
        config = replace(config, model=replace(config.model, member_populations=tuple(pops)), scaling=replace(config.scaling, **slopes))
    if fix_relation:
        fixed = {k.strip(): float(v) for k, v in (item.split("=") for item in fix_relation.split(","))}
        pops = list(config.model.member_populations)
        pops[0] = replace(pops[0], **{k: fixed[k] for k in ("sigma", "cutkpc") if k in fixed})
        slopes = {f"{k}_prior_mean": fixed[k] for k in ("alpha_sigma", "beta_radius") if k in fixed}
        config = replace(config, model=replace(config.model, member_populations=tuple(pops)),
                         scaling=replace(config.scaling, fixed_relation_fields=tuple(fixed), **slopes))
    if start_tolerance is not None or start_max_iter is not None:
        config = replace(config, search=replace(config.search,
                                                nuts_start_tolerance=float(start_tolerance or config.search.nuts_start_tolerance),
                                                nuts_start_max_iter=int(start_max_iter or config.search.nuts_start_max_iter)))
    return config

def _fast_mode_warning(config) -> None:
    """A banner that cannot be missed: a --fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = config.members
    cut = members.min_deflection_arcsec
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            + (f"  - members deflecting every image by less than {cut:g}\" are dropped from the model\n" if cut is not None
               else "  - every catalog member is kept (no impact cut)\n")
            + f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {config.search.n_starts} search starts, {config.sampling.warmup} warmup + {config.sampling.samples} draws per chain "
            f"at tree depth {config.sampling.max_tree_depth}, target accept {config.sampling.target_accept:g}\n"
            f"  - GP covariance blocks: {config.likelihood.gp_blocks}, projection {', '.join(config.likelihood.gp_knee_project_modes) or 'off'} "
            f"(full runs default to all blocks with the projection)")
    Console(stderr=False).print(Panel(text, title="[bold]WARNING: --fast[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    cluster = options["cluster"]
    if cluster not in CLUSTERS:
        print(f"Unknown cluster: {cluster!r}\n{_usage()}")
        return 2
    global BCG_CUT_PRIOR
    BCG_CUT_PRIOR = options["bcg_cut_prior"]
    runtime = build_runtime(cores)
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(cluster, cores=cores, campaign_name=options["campaign_name"], run_label=options["run_label"],
                          runtime=runtime, fast=options["fast"], imaging=options["imaging"],
                          halo_search=options["halo_search"], member_search=options["member_search"], likelihood=options["likelihood"],
                          gp_blocks=options["gp_blocks"],
                          warm_start=options["warm_start"], gp_band_max=options["gp_band_max"],
                          start_tolerance=options["start_tolerance"], start_max_iter=options["start_max_iter"],
                          gp_knee_gamma=options["gp_knee_gamma"], gp_knee_fixed=options["gp_knee_fixed"], fix_relation=options["fix_relation"], relation_prior=options["relation_prior"], kinematics=options["kinematics"], drop_halos=options["drop_halos"], member_shapes=options["member_shapes"])
    # Sampler-geometry overrides (2026-11-04 tests): draws, the dense block's members, far-field offsets, deviation tails.
    sampling = config.sampling
    if options["warmup"] is not None:
        sampling = replace(sampling, warmup=options["warmup"])
    if options["samples"] is not None:
        sampling = replace(sampling, samples=options["samples"])
    if options["dense_offsets"] is not None:
        sampling = replace(sampling, dense_offset_members=options["dense_offsets"])
    if options["far_offsets"] is not None:
        sampling = replace(sampling, far_field_offsets=options["far_offsets"])
    config = replace(config, sampling=sampling)
    if options["skip_plots"]:
        config = replace(config, runtime=replace(config.runtime, skip_plots=True))
    if options["offset_tail"] is not None:
        config = replace(config, scaling=replace(config.scaling, member_deviation_tail=options["offset_tail"]))
    config = config.validate()
    if options["dry_run"]:
        from lenscluster.pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    from lenscluster.pipeline import run
    if options["fast"]:
        _fast_mode_warning(config)
    print(f"[config] cluster={cluster} run_name={config.paths.run_name} output_dir={config.paths.output_dir} "
          f"stage={options['stage']} chains={runtime.chains} search={config.search.halo_method}+{config.search.member_method} "
          f"n_starts={config.search.n_starts} fast={options['fast']} sigma_inst_arcsec={config.model.image_constraints.sigma_inst_arcsec} "
          f"likelihood={options['likelihood']}"
          + {"grf_gp": f" (GRF of unmodelled mass with a knee spectrum + image-plane GP) gp_blocks={config.likelihood.gp_blocks}",
             "gp_image_plane": " (image-plane GP per family)", "smoothed": " (per-image source-plane floors)"}[options["likelihood"]])
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

# Nov 4 3:10AM
