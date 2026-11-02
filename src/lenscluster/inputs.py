"""Declared model inputs: halos, member populations, and image catalogs from the config."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import IndependentMemberHaloConfig, LensModelConfig, MemberPopulationConfig, PriorConfig
from .image_catalog import load_image_catalog
from .member_catalog import load_member_catalog
from .profiles import profile as lens_profile
from .sky import radec_to_offsets_arcsec


def check_potential_fields(potential: dict[str, Any]) -> dict[str, Any]:
    """The potential with its profile's required fields and z_lens checked to be finite numbers."""
    info = lens_profile(int(potential["profil"]))
    for name in (*info.required_fields, "z_lens"):
        value = potential.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Potential {potential.get('id', '?')} is missing a finite {name}.")
    return potential


def lenstool_ellipticite_to_axis_ratio(ellipticite: float | np.ndarray) -> np.ndarray:
    safe_e = np.clip(np.asarray(ellipticite, dtype=float), 0.0, 1.0 - 1.0e-9)
    q = np.sqrt((1.0 - safe_e) / (1.0 + safe_e))
    return np.clip(q, 1.0e-3, 1.0)


def lenstool_shape_to_e1e2(ellipticite: float, angle_pos_deg: float) -> tuple[float, float]:
    q = lenstool_ellipticite_to_axis_ratio(float(ellipticite))
    modulus = (1.0 - q) / (1.0 + q)
    phi = np.deg2rad(float(angle_pos_deg))
    return float(modulus * np.cos(2.0 * phi)), float(modulus * np.sin(2.0 * phi))


def shear_polar_to_gamma1gamma2(gamma: float, angle_pos_deg: float) -> tuple[float, float]:
    phi = np.deg2rad(float(angle_pos_deg))
    return float(gamma * np.cos(2.0 * phi)), float(gamma * np.sin(2.0 * phi))


def axis_ratio_to_lenstool_ellipticite(q: float) -> float:
    safe_q = max(1.0e-3, min(1.0, float(q)))
    return float((1.0 - safe_q * safe_q) / (1.0 + safe_q * safe_q))


def catalog_shape_to_ellipticity(a_value: float, b_value: float, theta_value: float) -> tuple[float, float]:
    a_axis = max(float(a_value), 1.0e-3)
    b_axis = max(min(float(b_value), a_axis), 1.0e-3)
    q = max(1.0e-3, min(1.0, b_axis / a_axis))
    return axis_ratio_to_lenstool_ellipticite(q), float(theta_value)


def prior_config_to_limit_entry(prior: PriorConfig | None, *, default_fixed_value: float | None = None) -> list[float]:
    """Lenstool-style limit entry: [0(,value)] fixed, [1,lo,hi,step], [3,mean,std], [9,mean,std,lo,hi]."""
    if prior is None or str(prior.kind) == "fixed":
        return [0] if default_fixed_value is None else [0, float(default_fixed_value)]
    kind = str(prior.kind).replace("-", "_")
    if kind == "uniform":
        if prior.lower is None or prior.upper is None:
            raise ValueError("Uniform priors require lower and upper.")
        step = float(prior.step) if prior.step is not None else 0.1 * (float(prior.upper) - float(prior.lower))
        return [1, float(prior.lower), float(prior.upper), step]
    if kind == "normal":
        if prior.mean is None or prior.std is None:
            raise ValueError("Normal priors require mean and std.")
        if prior.lower is not None or prior.upper is not None:
            if prior.lower is None or prior.upper is None:
                raise ValueError("Bounded normal priors require both lower and upper.")
            return [9, float(prior.mean), float(prior.std), float(prior.lower), float(prior.upper)]
        return [3, float(prior.mean), float(prior.std)]
    if kind == "truncated_normal":
        if prior.mean is None or prior.std is None or prior.lower is None or prior.upper is None:
            raise ValueError("Truncated normal priors require mean, std, lower, and upper.")
        return [9, float(prior.mean), float(prior.std), float(prior.lower), float(prior.upper)]
    raise ValueError(f"Unsupported prior kind {prior.kind!r}.")


def cosmology_config_from_model_config(model_config: LensModelConfig) -> dict[str, Any]:
    cosmo = model_config.cosmology
    om0 = float(cosmo.Om0)
    ode0 = float(1.0 - om0 if cosmo.Ode0 is None else cosmo.Ode0)
    return {"class": str(cosmo.class_name), "H0": float(cosmo.H0), "Om0": om0, "Ode0": ode0, "w0": float(cosmo.w0)}


def _independent_member_default_positive_std(mean: float, *, dex_sigma: float = 0.5) -> float:
    safe_mean = max(float(mean), 1.0e-12)
    log_sigma = float(np.log(10.0) * dex_sigma)
    return float(safe_mean * np.sqrt(np.expm1(log_sigma * log_sigma)))


def _independent_member_default_priors(*, x_centre: float, y_centre: float, population: MemberPopulationConfig) -> dict[str, PriorConfig]:
    core_mean = float(population.corekpc)
    cut_mean = float(population.cutkpc)
    sigma_mean = float(population.sigma)
    return {
        # The light fixes a galaxy's centroid to a fraction of an arcsecond; a loose position prior makes the posterior
        # multi-well as the halo hops across nearby images.
        "x_centre": PriorConfig("truncated_normal", mean=x_centre, std=0.3, lower=x_centre - 2.0, upper=x_centre + 2.0),
        "y_centre": PriorConfig("truncated_normal", mean=y_centre, std=0.3, lower=y_centre - 2.0, upper=y_centre + 2.0),
        "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.8),
        "angle_pos": PriorConfig("uniform", lower=-180.0, upper=180.0),
        "core_radius_kpc": PriorConfig("truncated_normal", mean=core_mean,
                                       std=_independent_member_default_positive_std(core_mean), lower=0.01, upper=50.0),
        "cut_radius_kpc": PriorConfig("truncated_normal", mean=cut_mean,
                                      std=_independent_member_default_positive_std(cut_mean), lower=1.0, upper=1000.0),
        "v_disp": PriorConfig("truncated_normal", mean=sigma_mean, std=max(0.5 * sigma_mean, 1.0e-6), lower=30.0, upper=1000.0),
    }


def _independent_member_halo_potential(independent: IndependentMemberHaloConfig, population: MemberPopulationConfig,
                                       catalog_row: pd.Series, reference: tuple[int, float, float],
                                       cosmo_config: dict[str, Any]) -> dict[str, Any]:
    _, ra0_deg, dec0_deg = reference
    x_offsets, y_offsets = radec_to_offsets_arcsec(np.asarray([float(catalog_row["ra"])]), np.asarray([float(catalog_row["dec"])]), ra0_deg, dec0_deg)
    x_centre, y_centre = float(x_offsets[0]), float(y_offsets[0])
    if {"catalog_a", "catalog_b", "catalog_theta"}.issubset(catalog_row.index):
        ellipticite, angle_pos = catalog_shape_to_ellipticity(
            float(catalog_row["catalog_a"]), float(catalog_row["catalog_b"]), float(catalog_row["catalog_theta"]))
    else:
        ellipticite, angle_pos = 0.0, 0.0
    catalog_id = str(independent.catalog_id)
    potential = {
        "id": str(independent.id or f"independent_member_{population.id}_{catalog_id}"),
        "profil": int(population.profile_type),
        "x_centre": x_centre, "y_centre": y_centre,
        "ellipticite": float(ellipticite), "angle_pos": float(angle_pos),
        "core_radius_kpc": float(population.corekpc), "cut_radius_kpc": float(population.cutkpc),
        "v_disp": float(population.sigma), "z_lens": float(population.z_lens),
    }
    merged_priors = {**_independent_member_default_priors(x_centre=x_centre, y_centre=y_centre, population=population),
                     **(independent.priors or {})}
    prior_entries = {name: prior_config_to_limit_entry(prior, default_fixed_value=float(potential.get(name, 0.0)))
                     for name, prior in merged_priors.items()}
    normalized = check_potential_fields(dict(potential))
    normalized["priors"] = prior_entries
    normalized["independent_member_population_id"] = str(population.id)
    normalized["independent_member_catalog_id"] = catalog_id
    return normalized


def _apply_catalog_exclusions(catalog_df: pd.DataFrame, population: MemberPopulationConfig) -> pd.DataFrame:
    exclude_ids = tuple(str(value).strip() for value in population.exclude_catalog_ids or ())
    if not exclude_ids:
        return catalog_df
    catalog_ids = catalog_df["id"].astype(str)
    missing = [value for value in exclude_ids if value not in set(catalog_ids)]
    if missing:
        raise ValueError(f"member population {population.id!r} exclude_catalog_ids not found in "
                         f"{population.catalog_path}: {', '.join(missing)}")
    filtered = catalog_df.loc[~catalog_ids.isin(set(exclude_ids))].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"member population {population.id!r} has no catalog rows after applying exclude_catalog_ids.")
    return filtered


def _scale_limit_entry(entry: list[float], factor: float) -> list[float]:
    """Scale the numeric values of a Lenstool-style limit entry (kind code untouched)."""
    return [entry[0]] + [float(v) * factor for v in entry[1:]]


def automatic_pivot(catalog_df: pd.DataFrame, images_df: pd.DataFrame, reference: tuple[int, float, float], *,
                    floor_arcsec: float = 1.0) -> dict[str, Any]:
    """Image-leverage-weighted member magnitude, w_i = sum_j 1 / max(R_ij, floor)^2, rounded to the nearest magnitude."""
    _, ra0, dec0 = reference
    mx, my = radec_to_offsets_arcsec(catalog_df["ra"].to_numpy(dtype=float), catalog_df["dec"].to_numpy(dtype=float), ra0, dec0)
    ix, iy = radec_to_offsets_arcsec(images_df["ra"].to_numpy(dtype=float), images_df["dec"].to_numpy(dtype=float), ra0, dec0)
    distance = np.hypot(mx[:, None] - ix[None, :], my[:, None] - iy[None, :])
    weights = np.sum(1.0 / np.maximum(distance, float(floor_arcsec)) ** 2, axis=1)
    magnitudes = catalog_df["catalog_mag"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * magnitudes) / np.sum(weights))
    return {"mag0": float(np.round(weighted)), "weighted_mean_mag": weighted, "member_median_mag": float(np.median(magnitudes)),
            "n_members": int(len(magnitudes))}


def declared_model_inputs(model_config: LensModelConfig, scaling: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict[str, Any], tuple[int, float, float]]:
    """Return (potentials with priors, potfiles, images, cosmology config, reference).

    ``scaling`` (a ScalingModelConfig) supplies the prior alpha and beta used to move a population's sigma and cutkpc
    priors to an automatic pivot.
    """
    reference = (int(model_config.reference.reference), float(model_config.reference.ra0_deg), float(model_config.reference.dec0_deg))
    cosmo_config = cosmology_config_from_model_config(model_config)
    if model_config.image_constraints is None:
        raise ValueError("model.image_constraints is required.")
    constraints = model_config.image_constraints
    images_df = load_image_catalog(constraints.catalog_path, sigma_inst_arcsec=constraints.sigma_inst_arcsec,
                                   sigma_floor_arcsec=constraints.sigma_floor_arcsec)
    if images_df.empty:
        raise ValueError("No multiple-image constraints found in the image catalog.")

    potentials: list[dict[str, Any]] = []
    for halo in model_config.large_halos:
        info = lens_profile(int(halo.profile_type))
        potential = {"id": str(halo.id), "profil": int(halo.profile_type), "z_lens": float(halo.z_lens),
                     **{name: float(getattr(halo, name)) for name in info.required_fields}}
        priors = {name: prior_config_to_limit_entry(prior, default_fixed_value=float(potential.get(name, 0.0)))
                  for name, prior in (halo.priors or {}).items()}
        normalized = check_potential_fields(potential)
        normalized["priors"] = priors
        potentials.append(normalized)

    independent_by_population: dict[str, list[IndependentMemberHaloConfig]] = {}
    for item in model_config.independent_member_halos or ():
        independent_by_population.setdefault(str(item.population_id), []).append(item)
    potfiles: list[dict[str, Any]] = []
    for population in model_config.member_populations:
        catalog_df = load_member_catalog(population.catalog_path, reference)
        independent_items = independent_by_population.get(str(population.id), [])
        if independent_items:
            catalog_ids = catalog_df["id"].astype(str)
            for independent in independent_items:
                match = catalog_df.loc[catalog_ids == str(independent.catalog_id)]
                if match.empty:
                    raise ValueError(f"independent member halo population_id={population.id!r} "
                                     f"catalog_id={independent.catalog_id!r} was not found in {population.catalog_path}.")
                potentials.append(_independent_member_halo_potential(independent, population, match.iloc[0], reference, cosmo_config))
            catalog_df = catalog_df.loc[~catalog_ids.isin({str(item.catalog_id) for item in independent_items})].copy().reset_index(drop=True)
        catalog_df = _apply_catalog_exclusions(catalog_df, population)
        sigma_entry = prior_config_to_limit_entry(population.sigma_prior, default_fixed_value=float(population.sigma))
        cutkpc_entry = prior_config_to_limit_entry(population.cutkpc_prior, default_fixed_value=float(population.cutkpc))
        sigma_nominal, cutkpc_nominal, corekpc_nominal, pivot = float(population.sigma), float(population.cutkpc), float(population.corekpc), None
        if isinstance(population.mag0, str):
            if scaling is None:
                raise ValueError("mag0='auto' needs the scaling configuration to move the priors to the pivot.")
            pivot = automatic_pivot(catalog_df, images_df, reference)
            pivot["prior_mag0"] = float(population.prior_mag0)
            ratio = 10.0 ** (-0.4 * (pivot["mag0"] - float(population.prior_mag0)))     # L0_new / L0_old
            f_sigma, f_cut = ratio ** float(scaling.alpha_sigma_prior_mean), ratio ** float(scaling.beta_radius_prior_mean)
            pivot.update(luminosity_ratio=ratio, sigma_factor=f_sigma, cutkpc_factor=f_cut)
            sigma_entry, cutkpc_entry = _scale_limit_entry(sigma_entry, f_sigma), _scale_limit_entry(cutkpc_entry, f_cut)
            sigma_nominal, cutkpc_nominal, corekpc_nominal = sigma_nominal * f_sigma, cutkpc_nominal * f_cut, corekpc_nominal * f_cut
            mag0 = pivot["mag0"]
        else:
            mag0 = float(population.mag0)
        potfiles.append({
            "id": str(population.id), "type": int(population.profile_type),
            "catalog_path": str(population.catalog_path), "catalog_df": catalog_df,
            "mag0": mag0, "pivot": pivot, "corekpc": float(population.corekpc), "corekpc_nominal": corekpc_nominal, "zlens": float(population.z_lens),
            "sigma": sigma_entry, "sigma_nominal": sigma_nominal, "cutkpc": cutkpc_entry, "cutkpc_nominal": cutkpc_nominal,
        })
    return potentials, potfiles, images_df, cosmo_config, reference


def filter_image_families(images_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop families without a positive redshift and single-image families."""
    family_ids = images_df["family_id"].astype(str)
    z_values = pd.to_numeric(images_df["catalog_z"], errors="coerce").to_numpy(dtype=float)
    bad_z_families = sorted(family_ids[np.isfinite(z_values) & (z_values <= 0.0)].unique().tolist())
    kept = images_df.loc[~family_ids.isin(bad_z_families)].copy()
    counts = kept["family_id"].astype(str).value_counts()
    singleton_families = sorted(counts[counts <= 1].index.astype(str).tolist())
    kept = kept.loc[~kept["family_id"].astype(str).isin(singleton_families)].copy().reset_index(drop=True)
    if kept.empty:
        raise ValueError("No multi-image families with positive redshift remain in the image catalog.")
    return kept, {"dropped_nonpositive_redshift_families": bad_z_families, "dropped_singleton_families": singleton_families}
