"""A solver configuration from a Lenstool parameter file (``.par``).

Reads the blocks the solver has an equivalent for and returns a ``LensClusterSolverConfig`` with library defaults
elsewhere:

* ``runmode`` ``reference``: the reference frame;
* ``cosmology`` / ``cosmologie``: ``H0``, ``omega``, ``lambda``, ``wX``;
* ``potentiel`` / ``limit``: dPIE halos (``profil 81``) and external shear (``profil 14``); radii in arcsec
  (``core_radius``, ``cut_radius``) are converted to kpc at the halo redshift, ``_kpc`` radii are taken as they are;
* ``potfile``: the member population (``mag0``, ``core``/``corekpc``, ``sigma``, ``cut``/``cutkpc``, ``vdslope``,
  ``slope``), with the slopes converted to the relation exponents alpha = 1 / vdslope and beta = 2 / slope;
* ``image``: ``multfile`` when it is a YAML image catalog, and ``sigposArcsec``.

Limits keep the Lenstool flags: 0 fixed, 1 uniform (lower, upper, step), 3 normal (mean, std), 9 truncated normal
(mean, std, lower[, upper]).  A field without a limit is fixed.  Lenstool fixes what a ``.par`` file leaves without a
limit, so the relation slopes and the member core are fixed when the file gives them with flag 0; relax them on the
returned configuration with ``dataclasses.replace``.

The solver reads YAML image catalogs only: pass ``image_catalog`` (convert a Lenstool multiple-image file with
scripts/convert_lenstool_catalog.py).  The potfile's ``filein`` catalog is resolved next to the ``.par`` file unless
``member_catalog`` is given.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                     MemberPopulationConfig, PriorConfig, ReferenceFrameConfig, RunPathsConfig, ScalingModelConfig,
                     ShearHaloConfig)
from .model import DP_IE_PROFILE, SHEAR_PROFILE

ALIASES = {"ellipticity": "ellipticite", "profile": "profil", "zlens": "z_lens"}


def _number(token: str) -> float | str:
    try:
        return float(token)
    except ValueError:
        return token


def read_lenstool_par(path: str | Path) -> list[tuple[str, str | None, dict[str, list]]]:
    """The blocks of a ``.par`` file in order: (block name, block id or None, {key: values}).  Comments (``#``) and
    blank lines are dropped; a repeated key keeps its last value, as in Lenstool."""
    path = Path(path)
    blocks: list[tuple[str, str | None, dict[str, list]]] = []
    current: dict[str, list] | None = None
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        tokens = raw.split("#", 1)[0].split()
        if not tokens:
            continue
        if current is None:
            if tokens[0] == "fini":
                break
            current = {}
            blocks.append((tokens[0], " ".join(tokens[1:]) or None, current))
        elif tokens[0] == "end":
            current = None
        else:
            current[ALIASES.get(tokens[0], tokens[0])] = [_number(token) for token in tokens[1:]]
    if current is not None:
        raise ValueError(f"{path}: block {blocks[-1][0]!r} is missing 'end'.")
    return blocks


def prior_from_limit(values: list, *, scale: float = 1.0) -> PriorConfig | None:
    """A Lenstool limit entry as a ``PriorConfig`` (None for flag 0, fixed); ``scale`` multiplies every value
    (arcsec to kpc for radii)."""
    flag, numbers = int(values[0]), [float(v) * scale for v in values[1:]]
    if flag == 0:
        return None
    if flag == 1:
        lower, upper = numbers[0], numbers[1]
        step = numbers[2] if len(numbers) > 2 else 0.1 * (upper - lower)
        return PriorConfig("uniform", lower=lower, upper=upper, step=step)
    if flag == 3:
        return PriorConfig("normal", mean=numbers[0], std=numbers[1])
    if flag == 9:
        return PriorConfig("truncated_normal", mean=numbers[0], std=numbers[1], lower=numbers[2],
                           upper=numbers[3] if len(numbers) > 3 else math.inf)
    raise ValueError(f"Unsupported Lenstool limit flag {flag}; expected 0, 1, 3 or 9.")


def _inside(value: float, prior: PriorConfig | None) -> float:
    """The start moved inside the prior's bounds (Lenstool files often start a halo outside its own limit)."""
    if prior is None:
        return value
    lower = -math.inf if prior.lower is None else prior.lower
    upper = math.inf if prior.upper is None else prior.upper
    return min(max(value, lower), upper)


def _value(block: dict[str, list], key: str, context: str) -> float:
    if key not in block or not block[key]:
        raise ValueError(f"{context}: missing {key!r}.")
    return float(block[key][0])


def _priors(limit: dict[str, list], scales: dict[str, float]) -> dict[str, PriorConfig]:
    priors = {}
    for key, values in limit.items():
        field = {"core_radius": "core_radius_kpc", "cut_radius": "cut_radius_kpc"}.get(key, key)
        prior = prior_from_limit(values, scale=scales.get(key, 1.0))
        if prior is not None:
            priors[field] = prior
    return priors


def _dpie(potential_id: str, block: dict[str, list], limit: dict[str, list], kpc_per_arcsec) -> DPIEHaloConfig:
    context = f"potentiel {potential_id}"
    z_lens = _value(block, "z_lens", context)
    scale = kpc_per_arcsec(z_lens)
    radii = {}
    for name in ("core_radius", "cut_radius"):
        if f"{name}_kpc" in block:
            radii[f"{name}_kpc"] = _value(block, f"{name}_kpc", context)
        else:
            radii[f"{name}_kpc"] = _value(block, name, context) * scale
    priors = _priors(limit, {"core_radius": scale, "cut_radius": scale})
    fields = {name: _value(block, name, context) for name in ("x_centre", "y_centre", "ellipticite", "angle_pos", "v_disp")}
    fields.update(radii)
    fields = {name: _inside(value, priors.get(name)) for name, value in fields.items()}
    return DPIEHaloConfig(id=potential_id, z_lens=z_lens, priors=priors, **fields)


def _shear(potential_id: str, block: dict[str, list], limit: dict[str, list]) -> ShearHaloConfig:
    context = f"potentiel {potential_id}"
    priors = _priors(limit, {})
    return ShearHaloConfig(id=potential_id, gamma=_inside(_value(block, "gamma", context), priors.get("gamma")),
                           angle_pos=_inside(_value(block, "angle_pos", context), priors.get("angle_pos")),
                           z_lens=_value(block, "z_lens", context), priors=priors)


def _slope_prior(values: list | None, exponent: float) -> tuple[float, float | None, tuple[float, float] | None]:
    """A Lenstool slope (vdslope, or slope) as the relation exponent ``exponent / slope``: (mean, std or None when
    fixed, bounds or None)."""
    flag, numbers = int(values[0]), [float(v) for v in values[1:]]
    if flag == 0:
        return exponent / numbers[0], None, None
    if flag == 1:
        lower, upper = exponent / numbers[1], exponent / numbers[0]
        return 0.5 * (lower + upper), 0.5 * (upper - lower), (lower, upper)
    if flag in (3, 9):
        mean = exponent / numbers[0]
        return mean, exponent * numbers[1] / numbers[0] ** 2, None
    raise ValueError(f"Unsupported Lenstool limit flag {flag} for a potfile slope.")


def _population(block: dict[str, list], catalog: Path, kpc_per_arcsec) -> tuple[MemberPopulationConfig, dict[str, Any]]:
    context = "potfile"
    z_lens = _value(block, "z_lens", context)
    scale = kpc_per_arcsec(z_lens)
    core = _value(block, "corekpc", context) if "corekpc" in block else _value(block, "core", context) * scale
    cut_key, cut_scale = ("cutkpc", 1.0) if "cutkpc" in block else ("cut", scale)
    sigma_prior = prior_from_limit(block["sigma"]) if int(block["sigma"][0]) else None
    cut_prior = prior_from_limit(block[cut_key], scale=cut_scale) if int(block[cut_key][0]) else None
    sigma = float(block["sigma"][1])
    cut = float(block[cut_key][1]) * cut_scale
    if sigma_prior is not None and sigma_prior.kind == "uniform":
        sigma = 0.5 * (sigma_prior.lower + sigma_prior.upper)
    if cut_prior is not None and cut_prior.kind == "uniform":
        cut = 0.5 * (cut_prior.lower + cut_prior.upper)
    population = MemberPopulationConfig(
        id="potfile", catalog_path=str(catalog), mag0=_value(block, "mag0", context), corekpc=core,
        sigma=max(sigma, 1.0), cutkpc=max(cut, 1.0e-3), z_lens=z_lens, profile_type=int(_value(block, "type", context)),
        sigma_prior=sigma_prior or PriorConfig(), cutkpc_prior=cut_prior or PriorConfig())
    scaling: dict[str, Any] = {"fit_corekpc": False}
    fixed = []
    for key, name, exponent in (("vdslope", "alpha_sigma", 1.0), ("slope", "beta_radius", 2.0)):
        if key not in block:
            continue
        mean, std, bounds = _slope_prior(block[key], exponent)
        scaling[f"{name}_prior_mean"] = mean
        if std is None:
            fixed.append(name)
        else:
            scaling[f"{name}_prior_std"] = std
        if bounds is not None:
            scaling[f"{name}_prior_lower"], scaling[f"{name}_prior_upper"] = bounds
    if sigma_prior is None:
        fixed.append("sigma")
    if cut_prior is None:
        fixed.append("cutkpc")
    scaling["fixed_relation_fields"] = tuple(fixed)
    return population, scaling


def config_from_lenstool_par(path: str | Path, *, image_catalog: str | Path | None = None, member_catalog: str | Path | None = None,
                             output_dir: str | Path = "results", run_name: str | None = None) -> LensClusterSolverConfig:
    """A ``LensClusterSolverConfig`` for the lens model of a Lenstool ``.par`` file (see the module docstring).

    ``image_catalog``: the YAML multiple-image catalog (default: the file's ``multfile`` when it is YAML).
    ``member_catalog``: the member catalog (default: the potfile's ``filein``, next to the ``.par`` file).
    ``run_name`` defaults to the file's stem; the run directory is ``output_dir/run_name``."""
    from .jax_cosmology import kpc_per_arcsec_from_config

    path = Path(path)
    blocks = read_lenstool_par(path)
    first = {}
    for name, _, data in blocks:
        first.setdefault(name, data)

    reference = ReferenceFrameConfig()
    if "reference" in first.get("runmode", {}):
        kind, ra0, dec0 = (first["runmode"]["reference"] + [0.0, 0.0])[:3]
        reference = ReferenceFrameConfig(reference=int(kind), ra0_deg=float(ra0), dec0_deg=float(dec0))
    cosmo_block = first.get("cosmology", first.get("cosmologie", {}))
    cosmology = CosmologyConfig(H0=float(cosmo_block.get("H0", [70.0])[0]), Om0=float(cosmo_block.get("omega", [0.3])[0]),
                                Ode0=float(cosmo_block["lambda"][0]) if "lambda" in cosmo_block else None,
                                w0=float(cosmo_block.get("wX", [-1.0])[0]))
    cosmo_dict = {"H0": cosmology.H0, "Om0": cosmology.Om0, "w0": cosmology.w0}

    def kpc_per_arcsec(z_lens: float) -> float:
        return float(kpc_per_arcsec_from_config(z_lens, cosmo_dict))

    limits = {block_id: data for name, block_id, data in blocks if name == "limit" and block_id is not None}
    halos = []
    for name, block_id, data in blocks:
        if name not in {"potentiel", "potential"}:
            continue
        potential_id = block_id or str(len(halos) + 1)
        profile = int(_value(data, "profil", f"potentiel {potential_id}"))
        limit = limits.get(potential_id, {})
        if profile == DP_IE_PROFILE:
            halos.append(_dpie(potential_id, data, limit, kpc_per_arcsec))
        elif profile == SHEAR_PROFILE:
            halos.append(_shear(potential_id, data, limit))
        else:
            raise ValueError(f"{path}: potentiel {potential_id} has profil {profile}; only dPIE (81) and shear (14) are read "
                             "from .par files. Configure other profiles in Python (lenscluster.config).")

    image_block = first.get("image", {})
    if image_catalog is None:
        multfile = image_block.get("multfile")
        candidate = path.parent / str(multfile[-1]) if multfile else None
        if candidate is None or candidate.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"{path}: the solver reads YAML image catalogs; convert {candidate or 'the multfile'} with "
                             "scripts/convert_lenstool_catalog.py and pass image_catalog=.")
        image_catalog = candidate
    sigma_inst = float(image_block["sigposArcsec"][0]) if "sigposArcsec" in image_block else ImageConstraintsConfig.sigma_inst_arcsec

    populations, scaling_fields = (), {}
    potfiles = [data for name, _, data in blocks if name == "potfile"]
    if len(potfiles) > 1:
        raise ValueError(f"{path}: {len(potfiles)} potfile blocks; the reader takes one. Configure several populations in Python.")
    if potfiles:
        potfile = potfiles[0]
        if member_catalog is None:
            if "filein" not in potfile:
                raise ValueError(f"{path}: the potfile has no filein; pass member_catalog=.")
            member_catalog = path.parent / str(potfile["filein"][-1])
        if not Path(member_catalog).is_file():
            raise FileNotFoundError(f"{path}: member catalog {member_catalog} not found; pass member_catalog=.")
        population, scaling_fields = _population(potfile, Path(member_catalog), kpc_per_arcsec)
        populations = (population,)

    model = LensModelConfig(reference=reference, cosmology=cosmology, large_halos=tuple(halos), member_populations=populations,
                            image_constraints=ImageConstraintsConfig(catalog_path=str(image_catalog), sigma_inst_arcsec=sigma_inst))
    config = LensClusterSolverConfig(model=model, paths=RunPathsConfig(output_dir=output_dir, run_name=run_name or path.stem,
                                                                       figure_prefix=path.stem))
    if scaling_fields:
        config = replace(config, scaling=replace(ScalingModelConfig(), **scaling_fields))
    return config
