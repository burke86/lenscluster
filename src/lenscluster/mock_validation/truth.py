"""One realization of the analytic mock cluster: the truth, drawn from the ranges of a MockConfig."""
from __future__ import annotations

import math
import dataclasses
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import gammainc, gammaincinv

from ..inputs import lenstool_shape_to_e1e2
from ..jax_cosmology import kpc_per_arcsec_from_config
from ..utils import DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN as G_KPC, log10_dpie_mass_msun
from .config import MockConfig


@dataclass(frozen=True)
class DPIEComponent:
    """A dPIE in physical units (kpc, km/s) at an angular position (arcsec), Lenstool shape convention."""
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_kpc: float
    cut_radius_kpc: float
    v_disp: float

    @property
    def mass_msun(self) -> float:
        """Total dPIE mass in the pipeline's convention (lenstool sigma: M = 1.5 pi sigma^2 r_cut / G)."""
        return float(10.0 ** log10_dpie_mass_msun(self.v_disp, self.cut_radius_kpc))


@dataclass(frozen=True)
class MemberTruth(DPIEComponent):
    magnitude: float = float("nan")
    luminosity_ratio: float = float("nan")
    modelled: bool = True                      # False: present in the truth but absent from the member catalog


@dataclass(frozen=True)
class MockTruth:
    seed: int
    z_lens: float
    cosmology: dict[str, float]
    kpc_per_arcsec: float
    halo: DPIEComponent
    bcg: DPIEComponent
    relation: dict[str, float]                 # pivot_magnitude, sigma_star_kms, cut_star_kpc, core_star_kpc, alpha_sigma, beta_radius, scatters
    members: tuple[MemberTruth, ...]

    @property
    def components(self) -> list[DPIEComponent]:
        """Every mass component of the truth, modelled or not."""
        return [self.halo, self.bcg, *self.members]

    @property
    def modelled_members(self) -> tuple[MemberTruth, ...]:
        return tuple(m for m in self.members if m.modelled)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["n_modelled_members"] = len(self.modelled_members)
        return payload


# ----------------------------------------------------------------------------- helpers
def truth_from_json(payload: dict[str, Any]) -> MockTruth:
    """The MockTruth written to truth.json (extra keys of the file are ignored)."""
    fields = lambda cls: {f.name for f in dataclasses.fields(cls)}
    component = lambda d: DPIEComponent(**{k: d[k] for k in fields(DPIEComponent)})
    return MockTruth(seed=int(payload["seed"]), z_lens=float(payload["z_lens"]), cosmology=dict(payload["cosmology"]),
                     kpc_per_arcsec=float(payload["kpc_per_arcsec"]), halo=component(payload["halo"]), bcg=component(payload["bcg"]),
                     relation=dict(payload["relation"]),
                     members=tuple(MemberTruth(**{k: m[k] for k in fields(MemberTruth) if k in m}) for m in payload["members"]))


def dpie_v_disp_from_mass(mass_msun: float, cut_radius_kpc: float) -> float:
    """Lenstool-convention dispersion of a dPIE with total mass M = 1.5 pi sigma^2 r_cut / G."""
    return math.sqrt(G_KPC * float(mass_msun) / (1.5 * math.pi * float(cut_radius_kpc)))


def _lognormal_factor(rng: np.random.Generator, scatter_dex: float, size=None):
    return np.power(10.0, rng.normal(0.0, float(scatter_dex), size=size)) if scatter_dex > 0 else (1.0 if size is None else np.ones(size))


def truncated_schechter_luminosities(rng: np.random.Generator, n: int, *, alpha: float, lower: float, upper: float) -> np.ndarray:
    """Draws of L/L* from a Schechter function truncated to [lower, upper] (inverse incomplete-gamma CDF)."""
    shape = float(alpha) + 1.0
    if shape <= 0.0:
        raise ValueError("Schechter alpha must exceed -1 for a normalizable truncated draw.")
    lo, hi = float(gammainc(shape, lower)), float(gammainc(shape, upper))
    return np.asarray(gammaincinv(shape, lo + rng.random(int(n)) * (hi - lo)), dtype=float)


def _radial_positions(rng: np.random.Generator, n: int, *, core: float, cut: float, r_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions following r Sigma_dPIE(r; core, cut) in projection, truncated at r_max, with uniform angles."""
    grid = np.linspace(0.0, float(r_max), 4096)
    density = grid * (1.0 / np.sqrt(core ** 2 + grid ** 2) - 1.0 / np.sqrt(cut ** 2 + grid ** 2))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    radius = np.interp(rng.random(int(n)) * cdf[-1], cdf, grid)
    angle = rng.uniform(0.0, 2.0 * math.pi, int(n))
    return radius * np.cos(angle), radius * np.sin(angle)


# ----------------------------------------------------------------------------- the draw
def draw_truth(config: MockConfig, seed: int | None = None) -> MockTruth:
    """One mock cluster from the configuration's ranges; deterministic in the seed."""
    config = config.validate()
    seed = int(config.seed if seed is None else seed)
    rng = np.random.default_rng(seed)
    cosmo = {"class": "FlatLambdaCDM", "H0": float(config.cosmology.H0), "Om0": float(config.cosmology.Om0),
             "Ode0": float(config.cosmology.Ode0), "w0": -1.0}
    z_lens = float(rng.uniform(*config.z_lens_range))
    kpc = float(kpc_per_arcsec_from_config(z_lens, cosmo))

    # Cluster halo and BCG.
    log_lo, log_hi = (math.log10(v) for v in config.total_mass_range_msun)
    total_mass = 10.0 ** rng.uniform(log_lo, log_hi)
    fraction = float(np.clip(config.bcg_mass_fraction_median * _lognormal_factor(rng, config.bcg_mass_fraction_scatter_dex), *config.bcg_mass_fraction_range))
    halo_mass, bcg_mass = total_mass * (1.0 - fraction), total_mass * fraction
    slope = float(config.radius_mass_slope)
    halo_core = config.halo_core_pivot_kpc * (halo_mass / config.mass_pivot_msun) ** slope * _lognormal_factor(rng, config.halo_radius_scatter_dex)
    halo_cut = config.halo_cut_pivot_kpc * (halo_mass / config.mass_pivot_msun) ** slope * _lognormal_factor(rng, config.halo_radius_scatter_dex)
    half = float(config.halo_position_half_width_arcsec)
    halo = DPIEComponent("halo", float(rng.uniform(-half, half)), float(rng.uniform(-half, half)),
                         float(rng.uniform(*config.halo_ellipticity_range)), float(rng.uniform(-90.0, 90.0)),
                         float(halo_core), float(halo_cut), dpie_v_disp_from_mass(halo_mass, halo_cut))
    bcg_core = config.bcg_core_pivot_kpc * (bcg_mass / config.bcg_mass_pivot_msun) ** slope * _lognormal_factor(rng, config.bcg_radius_scatter_dex)
    bcg_cut = config.bcg_cut_pivot_kpc * (bcg_mass / config.bcg_mass_pivot_msun) ** slope * _lognormal_factor(rng, config.bcg_radius_scatter_dex)
    offset_r, offset_phi = config.bcg_offset_arcsec * math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
    bcg = DPIEComponent("bcg", halo.x_centre + offset_r * math.cos(offset_phi), halo.y_centre + offset_r * math.sin(offset_phi),
                        float(rng.uniform(*config.bcg_ellipticity_range)), float(rng.uniform(-90.0, 90.0)),
                        float(bcg_core), float(bcg_cut), dpie_v_disp_from_mass(bcg_mass, bcg_cut))

    # Member relation and population.
    relation = {"pivot_magnitude": float(config.pivot_magnitude), "sigma_star_kms": float(rng.uniform(*config.sigma_star_range_kms)),
                "cut_star_kpc": float(rng.uniform(*config.cut_star_range_kpc)), "core_star_kpc": float(config.core_star_kpc),
                "alpha_sigma": float(rng.uniform(*config.alpha_sigma_range)), "beta_radius": float(rng.uniform(*config.beta_radius_range)),
                "sigma_scatter_ln": float(config.sigma_scatter_ln), "cut_scatter_ln": float(config.cut_scatter_ln),
                "schechter_alpha": float(config.schechter_alpha), "faint_magnitude_limit": float(config.faint_magnitude_limit)}
    members: list[MemberTruth] = []
    n_total = int(config.n_members) + int(config.unmodelled_members)
    if n_total > 0:
        luminosity = np.empty(0)
        while luminosity.size < n_total:                        # Schechter draw with the magnitude cut applied
            draw = truncated_schechter_luminosities(rng, 4 * n_total, alpha=config.schechter_alpha, lower=config.luminosity_range[0], upper=config.luminosity_range[1])
            magnitude = config.pivot_magnitude - 2.5 * np.log10(draw)
            luminosity = np.concatenate([luminosity, draw[magnitude <= config.faint_magnitude_limit]])
        luminosity = luminosity[:n_total]
        x, y = _radial_positions(rng, n_total, core=config.member_spatial_core_arcsec, cut=config.member_spatial_cut_arcsec, r_max=config.member_field_radius_arcsec)
        sigma = relation["sigma_star_kms"] * luminosity ** relation["alpha_sigma"] * np.exp(rng.normal(0.0, config.sigma_scatter_ln, n_total))
        cut = relation["cut_star_kpc"] * luminosity ** relation["beta_radius"] * np.exp(rng.normal(0.0, config.cut_scatter_ln, n_total))
        core = relation["core_star_kpc"] * luminosity ** relation["beta_radius"]
        ellipticity = rng.uniform(*config.member_ellipticity_range, n_total)
        angle = rng.uniform(-90.0, 90.0, n_total)
        modelled = np.ones(n_total, dtype=bool)
        if config.unmodelled_members:
            modelled[rng.choice(n_total, int(config.unmodelled_members), replace=False)] = False
        order = np.argsort(config.pivot_magnitude - 2.5 * np.log10(luminosity))   # brightest first, like a catalog
        for rank, i in enumerate(order, start=1):
            members.append(MemberTruth(f"member{rank:03d}", float(halo.x_centre + x[i]), float(halo.y_centre + y[i]), float(ellipticity[i]), float(angle[i]),
                                       float(core[i]), float(cut[i]), float(sigma[i]),
                                       magnitude=float(config.pivot_magnitude - 2.5 * math.log10(luminosity[i])),
                                       luminosity_ratio=float(luminosity[i]), modelled=bool(modelled[i])))
    return MockTruth(seed=seed, z_lens=z_lens, cosmology=cosmo, kpc_per_arcsec=kpc, halo=halo, bcg=bcg, relation=relation, members=tuple(members))


# ----------------------------------------------------------------------------- what the solver samples, in truth units
def parameter_truth(truth: dict[str, Any], *, solver_pivot_magnitude: float | None = None) -> dict[str, float]:
    """Truth value of every solver parameter name that has one.  The member relation is moved to the solver's pivot
    magnitude with the truth exponents; scatter truths are the injected values; e1/e2 follow the Lenstool conversion."""
    values: dict[str, float] = {}
    for prefix in ("halo", "bcg"):
        c = truth[prefix]
        for key in ("x_centre", "y_centre", "ellipticite", "angle_pos", "core_radius_kpc", "cut_radius_kpc", "v_disp"):
            values[f"{prefix}.{key}"] = float(c[key])
        values[f"{prefix}.e1"], values[f"{prefix}.e2"] = lenstool_shape_to_e1e2(float(c["ellipticite"]), float(c["angle_pos"]))
        values[f"{prefix}.log10_mass_msun"] = float(log10_dpie_mass_msun(float(c["v_disp"]), float(c["cut_radius_kpc"])))
    r = truth["relation"]
    pivot = float(r["pivot_magnitude"]) if solver_pivot_magnitude is None else float(solver_pivot_magnitude)
    ratio = 10.0 ** (-0.4 * (pivot - float(r["pivot_magnitude"])))          # L(pivot) / L*
    sigma = float(r["sigma_star_kms"]) * ratio ** float(r["alpha_sigma"])
    cut = float(r["cut_star_kpc"]) * ratio ** float(r["beta_radius"])
    values.update({"potfile.sigma": sigma, "potfile.cutkpc": cut, "potfile.corekpc": float(r["core_star_kpc"]) * ratio ** float(r["beta_radius"]),
                   "potfile.alpha_sigma": float(r["alpha_sigma"]), "potfile.beta_radius": float(r["beta_radius"]),
                   "potfile.gamma_ml": 2.0 * float(r["alpha_sigma"]) + float(r["beta_radius"]) - 1.0,
                   "potfile.log10_mass_msun": float(log10_dpie_mass_msun(sigma, cut)),
                   "potfile.velocity_log_scatter": float(r["sigma_scatter_ln"]), "potfile.radius_log_scatter": float(r["cut_scatter_ln"]),
                   "source.sigma_src": float(truth.get("source_scatter_arcsec", 0.0)), "images.sigma_obs_extra": 0.0})
    for source in truth.get("sources", []):
        values[f"source.{source['family_id']}.beta_x"] = float(source["beta_x"])
        values[f"source.{source['family_id']}.beta_y"] = float(source["beta_y"])
    return values
