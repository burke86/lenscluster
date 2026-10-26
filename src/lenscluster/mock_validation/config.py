"""Configuration of a mock-validation campaign and the solver model built from each realized truth.

A campaign is one mock generator configuration, one solver template and a number of realizations.  The mock is an
analytic cluster: a cluster-scale dPIE halo, a BCG dPIE, and a population of member dPIE subhalos on a luminosity
scaling relation; its sources are lensed exactly and the images carry astrometric noise.  The solver model of each
realization is built from that realization's truth (never from the nominal configuration) in one of two modes:

* ``production``: the priors an observer would use, i.e. the same factories as the real clusters, with the halo centred
  on the BCG light and the member population on the Bergamini et al. 2019 relation at the mock's pivot magnitude;
* ``idealized``: priors centred on the truth with moderate widths, which isolates the sampler and the likelihood.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                      MemberPopulationConfig, PriorConfig, ReferenceFrameConfig, RunPathsConfig, ShearHaloConfig)

VALID_MODES = ("production", "idealized")


@dataclass(frozen=True)
class MockConfig:
    """What varies between mock clusters and how they are observed.  Ranges are sampled per realization."""
    seed: int = 12345
    reference_ra_deg: float = 64.0381417
    reference_dec_deg: float = -24.0674722
    cosmology: CosmologyConfig = field(default_factory=lambda: CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7))
    z_lens_range: tuple[float, float] = (0.30, 0.50)
    # Cluster-scale halo and BCG: a total mass, split by a BCG fraction; radii from mass-size relations with scatter.
    total_mass_range_msun: tuple[float, float] = (5.0e14, 2.0e15)
    mass_pivot_msun: float = 7.6e14
    radius_mass_slope: float = 1.0 / 3.0
    halo_core_pivot_kpc: float = 27.0
    halo_cut_pivot_kpc: float = 1175.0
    halo_radius_scatter_dex: float = 0.08
    halo_ellipticity_range: tuple[float, float] = (0.10, 0.40)
    halo_position_half_width_arcsec: float = 10.0
    bcg_mass_fraction_median: float = 0.02
    bcg_mass_fraction_scatter_dex: float = 0.25
    bcg_mass_fraction_range: tuple[float, float] = (0.005, 0.08)
    bcg_mass_pivot_msun: float = 1.7e13
    bcg_core_pivot_kpc: float = 0.8
    bcg_cut_pivot_kpc: float = 187.0
    bcg_radius_scatter_dex: float = 0.10
    bcg_ellipticity_range: tuple[float, float] = (0.0, 0.30)
    bcg_offset_arcsec: float = 2.0
    # Member population: a truncated Schechter luminosity function, a relation at L* like the real clusters, scatter.
    n_members: int = 50
    schechter_alpha: float = -0.7
    pivot_magnitude: float = 19.82               # m* in the member band; the relation is quoted here
    faint_magnitude_limit: float = 24.0
    luminosity_range: tuple[float, float] = (0.02, 20.0)   # L/L* bounds of the Schechter draw
    sigma_star_range_kms: tuple[float, float] = (160.0, 240.0)
    cut_star_range_kpc: tuple[float, float] = (8.0, 20.0)
    core_star_kpc: float = 0.15
    alpha_sigma_range: tuple[float, float] = (0.22, 0.32)
    beta_radius_range: tuple[float, float] = (0.50, 0.80)
    sigma_scatter_ln: float = 0.10               # intrinsic scatter of ln sigma about the relation
    cut_scatter_ln: float = 0.0                  # intrinsic scatter of ln r_cut about the relation
    member_ellipticity_range: tuple[float, float] = (0.0, 0.5)
    member_spatial_core_arcsec: float = 25.0     # members follow a cored, truncated radial profile
    member_spatial_cut_arcsec: float = 180.0
    member_field_radius_arcsec: float = 250.0
    # Model misspecification: subhalos present in the truth but missing from the member catalog.
    unmodelled_members: int = 0
    # Sources and images.
    n_primary_families: int = 15
    n_member_families: int = 5
    n_central_families: int = 1                   # sources whose image falls within central_radius_arcsec of the BCG (radial arc / central image)
    central_radius_arcsec: float = 3.0
    source_placement: str = "image_plane"        # "image_plane": a primary source is the back-projection of a point drawn uniformly inside
                                                 # the tangential critical curve, so images spread over radii; "caustic": uniform inside the caustic
    source_redshift_range: tuple[float, float] = (1.5, 9.0)
    min_images_per_family: int = 3
    max_source_attempts: int = 400
    pos_sigma_arcsec: float = 0.10               # astrometric noise injected and assumed by the solver
    source_scatter_arcsec: float = 0.0           # injected intrinsic source-plane scatter per image (sigma_src truth)
    source_magnitude_range: tuple[float, float] = (24.0, 27.0)
    image_magnitude_sigma: float = 0.10
    # Image finding on the exact truth model.
    caustic_window_arcsec: float = 160.0
    caustic_grid_arcsec: float = 0.2
    caustic_min_area_arcsec2: float = 1.0e-5
    seed_grid_pixel_arcsec: float = 1.0
    search_window_arcsec: float = 80.0
    image_refine_iterations: int = 100
    image_precision_arcsec: float = 1.0e-8
    # Solver model built from the truth.
    mode: str = "production"

    def with_updates(self, **updates: Any) -> "MockConfig":
        return replace(self, **updates)

    def validate(self) -> "MockConfig":
        if self.mode not in VALID_MODES:
            raise ValueError(f"mock.mode must be one of {VALID_MODES}.")
        for name in ("z_lens_range", "total_mass_range_msun", "halo_ellipticity_range", "bcg_mass_fraction_range", "bcg_ellipticity_range",
                     "luminosity_range", "sigma_star_range_kms", "cut_star_range_kpc", "alpha_sigma_range", "beta_radius_range",
                     "member_ellipticity_range", "source_redshift_range", "source_magnitude_range"):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"mock.{name} must be an ordered finite range.")
        if self.n_members < 0 or self.unmodelled_members < 0 or self.n_primary_families < 1 or self.n_member_families < 0:
            raise ValueError("mock member and family counts must be nonnegative (at least one primary family).")
        if self.source_placement not in {"image_plane", "caustic"}:
            raise ValueError("source_placement must be 'image_plane' or 'caustic'")
        if self.n_central_families < 0 or self.central_radius_arcsec <= 0:
            raise ValueError("n_central_families must be >= 0 and central_radius_arcsec > 0")
        if self.n_member_families > 0 and self.n_members == 0:
            raise ValueError("mock.n_member_families needs members.")
        for name in ("pos_sigma_arcsec", "caustic_window_arcsec", "caustic_grid_arcsec", "seed_grid_pixel_arcsec", "search_window_arcsec", "image_precision_arcsec"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0.0):
                raise ValueError(f"mock.{name} must be positive.")
        for name in ("source_scatter_arcsec", "sigma_scatter_ln", "cut_scatter_ln"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0.0):
                raise ValueError(f"mock.{name} must be nonnegative.")
        if self.min_images_per_family < 2:
            raise ValueError("mock.min_images_per_family must be at least 2.")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MockValidationPathsConfig:
    output_dir: str | Path = "validation_runs"
    campaign_name: str = "mock_validation"
    run_name: str = "analytic"


@dataclass(frozen=True)
class MockValidationRuntimeConfig:
    realizations: int = 1
    seed: int = 12345
    resume: bool = True
    quiet: bool = False


@dataclass(frozen=True)
class MockValidationConfig:
    mock: MockConfig = field(default_factory=MockConfig)
    paths: MockValidationPathsConfig = field(default_factory=MockValidationPathsConfig)
    runtime: MockValidationRuntimeConfig = field(default_factory=MockValidationRuntimeConfig)
    solver: LensClusterSolverConfig = field(default_factory=LensClusterSolverConfig)

    def with_updates(self, **updates: Any) -> "MockValidationConfig":
        return replace(self, **updates)

    def validate(self) -> "MockValidationConfig":
        if int(self.runtime.realizations) <= 0:
            raise ValueError("runtime.realizations must be positive.")
        if int(self.runtime.seed) < 0:
            raise ValueError("runtime.seed must be nonnegative.")
        self.mock.validate()
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable({"mock": asdict(self.mock), "paths": asdict(self.paths), "runtime": asdict(self.runtime),
                          "solver": self.solver.to_json_dict()})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ----------------------------------------------------------------------------- solver model from the realized truth
def _bcg_halo(id: str, x: float, y: float, z_lens: float, *, v_disp: float, cut_kpc: float, half_width: float, truth: dict[str, float] | None) -> DPIEHaloConfig:
    """A BCG-scale dPIE: production priors around the light (``truth`` None) or idealized priors around the truth."""
    if truth is None:
        priors = {"x_centre": PriorConfig("uniform", lower=x - half_width, upper=x + half_width, step=0.1),
                  "y_centre": PriorConfig("uniform", lower=y - half_width, upper=y + half_width, step=0.1),
                  "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.6, step=0.02),
                  "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
                  "core_radius_kpc": PriorConfig("uniform", lower=0.1, upper=5.0, step=0.1),
                  "cut_radius_kpc": PriorConfig("truncated_normal", mean=cut_kpc, std=0.5 * cut_kpc, lower=20.0, upper=600.0),
                  "v_disp": PriorConfig("truncated_normal", mean=v_disp, std=0.35 * v_disp, lower=100.0, upper=800.0)}
        return DPIEHaloConfig(id=id, x_centre=x, y_centre=y, ellipticite=0.2, angle_pos=0.0, core_radius_kpc=1.0, cut_radius_kpc=cut_kpc,
                              v_disp=v_disp, z_lens=z_lens, priors=priors)
    return _idealized_halo(id, truth, z_lens, half_width=half_width, core_bounds=(0.1, 5.0), cut_bounds=(20.0, 600.0), v_bounds=(100.0, 800.0))


def _idealized_halo(id: str, t: dict[str, float], z_lens: float, *, half_width: float, core_bounds, cut_bounds, v_bounds) -> DPIEHaloConfig:
    """A dPIE with priors centred on its truth: position box, free shape, 30% radii, 15% dispersion."""
    priors = {"x_centre": PriorConfig("uniform", lower=t["x_centre"] - half_width, upper=t["x_centre"] + half_width, step=0.1),
              "y_centre": PriorConfig("uniform", lower=t["y_centre"] - half_width, upper=t["y_centre"] + half_width, step=0.1),
              "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8, step=0.02),
              "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0, step=1.0),
              "core_radius_kpc": PriorConfig("truncated_normal", mean=t["core_radius_kpc"], std=0.3 * t["core_radius_kpc"], lower=core_bounds[0], upper=core_bounds[1]),
              "cut_radius_kpc": PriorConfig("truncated_normal", mean=t["cut_radius_kpc"], std=0.3 * t["cut_radius_kpc"], lower=cut_bounds[0], upper=cut_bounds[1]),
              "v_disp": PriorConfig("truncated_normal", mean=t["v_disp"], std=0.15 * t["v_disp"], lower=v_bounds[0], upper=v_bounds[1])}
    return DPIEHaloConfig(id=id, x_centre=t["x_centre"], y_centre=t["y_centre"], ellipticite=t["ellipticite"], angle_pos=t["angle_pos"],
                          core_radius_kpc=t["core_radius_kpc"], cut_radius_kpc=t["cut_radius_kpc"], v_disp=t["v_disp"], z_lens=z_lens, priors=priors)


def mock_lens_model_config(config: MockConfig, truth: dict[str, Any], *, image_catalog: Path, member_catalog: Path | None) -> LensModelConfig:
    """The solver's lens model for one realization, built from its truth according to ``config.mode``."""
    halo, bcg, relation = truth["halo"], truth["bcg"], truth["relation"]
    z_lens = float(truth["z_lens"])
    has_members = member_catalog is not None and int(truth.get("n_modelled_members", 0)) > 0
    if config.mode == "idealized":
        halos = (_idealized_halo("halo", halo, z_lens, half_width=2.0, core_bounds=(1.0, 200.0), cut_bounds=(300.0, 4000.0), v_bounds=(100.0, 2200.0)),
                 _idealized_halo("bcg", bcg, z_lens, half_width=1.0, core_bounds=(0.1, 5.0), cut_bounds=(20.0, 600.0), v_bounds=(100.0, 800.0)),
                 ShearHaloConfig.free("shear", z_lens, gamma_upper=0.1))
        members = ()
        if has_members:
            sigma, cut = float(relation["sigma_star_kms"]), float(relation["cut_star_kpc"])
            members = (MemberPopulationConfig(
                id="potfile", catalog_path=member_catalog, mag0="auto", prior_mag0=float(relation["pivot_magnitude"]),
                corekpc=float(relation["core_star_kpc"]), sigma=sigma, cutkpc=cut, z_lens=z_lens,
                sigma_prior=PriorConfig("truncated_normal", mean=sigma, std=0.15 * sigma, lower=40.0, upper=500.0),
                cutkpc_prior=PriorConfig("truncated_normal", mean=cut, std=0.3 * cut, lower=2.0, upper=60.0)),)
    else:
        # An observer centres the cluster halo on the BCG light and knows nothing else about the truth.
        halos = (DPIEHaloConfig.cluster_scale("halo", bcg["x_centre"], bcg["y_centre"], z_lens, v_disp=800.0, position_half_width_arcsec=5.0),
                 _bcg_halo("bcg", bcg["x_centre"], bcg["y_centre"], z_lens, v_disp=300.0, cut_kpc=150.0, half_width=1.0, truth=None),
                 ShearHaloConfig.free("shear", z_lens, gamma_upper=0.1))
        members = (MemberPopulationConfig.bergamini2019("potfile", member_catalog, z_lens, prior_mag0=float(relation["pivot_magnitude"])),) if has_members else ()
    return LensModelConfig(
        reference=ReferenceFrameConfig(reference=3, ra0_deg=float(config.reference_ra_deg), dec0_deg=float(config.reference_dec_deg)),
        cosmology=config.cosmology, large_halos=halos, member_populations=members,
        image_constraints=ImageConstraintsConfig(catalog_path=image_catalog, sigma_arcsec=float(config.pos_sigma_arcsec)))


def solver_config_for_mock(config: MockValidationConfig, truth: dict[str, Any], *, image_catalog: Path, member_catalog: Path | None,
                           seed: int, output_dir: str | Path) -> LensClusterSolverConfig:
    """The solver configuration of one realization: the template's settings, the realization's lens model."""
    template = config.solver
    model = mock_lens_model_config(config.mock, truth, image_catalog=image_catalog, member_catalog=member_catalog)
    scaling = template.scaling
    if config.mock.mode == "idealized":
        relation = truth["relation"]
        scaling = replace(scaling, alpha_sigma_prior_mean=float(relation["alpha_sigma"]), alpha_sigma_prior_std=0.03,
                          beta_radius_prior_mean=float(relation["beta_radius"]), beta_radius_prior_std=0.08)
    likelihood = replace(template.likelihood, pos_sigma_arcsec=float(config.mock.pos_sigma_arcsec))
    return replace(template, model=model, scaling=scaling, likelihood=likelihood,
                   paths=RunPathsConfig(output_dir=output_dir, run_name="solver"),
                   runtime=replace(template.runtime, seed=int(seed), resume=bool(config.runtime.resume), quiet=bool(config.runtime.quiet)))
