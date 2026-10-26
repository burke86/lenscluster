"""Mock-validation campaign: draw a truth, lens it, fit it with the solver, compare the posterior with the truth."""
from __future__ import annotations

import json
from dataclasses import replace
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import pipeline
from ..config import DPIEHaloConfig
from ..model import ModelState, PosteriorResults, member_log_deviations
from ..utils import DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN as G_KPC, log10_dpie_mass_msun, progress
from .aperture import APERTURE_RADII_KPC, COMPONENT_GROUPS, aperture_profiles, truth_groups
from .config import MockValidationConfig, solver_config_for_mock
from .images import generate_images, jacobian, lens_state
from .truth import draw_truth, parameter_truth, truth_from_json
from .writers import MockPaths, write_mock

RECOVERY_FILE = "recovery.json"
CONVERGENCE_RHAT = 1.01          # a realization enters the aggregates only if max R-hat < this ...
CONVERGENCE_MIN_NEFF = 50.0      # ... and the smallest effective sample size is at least this


def convergence_verdict(diagnostics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether a realization's sampler converged by the campaign's rule, with the reasons when it did not."""
    reasons = []
    r_hat, n_eff = diagnostics.get("max_r_hat"), diagnostics.get("min_n_eff")
    if r_hat is None or not np.isfinite(r_hat) or r_hat >= CONVERGENCE_RHAT:
        reasons.append(f"max R-hat {r_hat if r_hat is None else f'{r_hat:.3f}'} >= {CONVERGENCE_RHAT}")
    if n_eff is None or not np.isfinite(n_eff) or n_eff < CONVERGENCE_MIN_NEFF:
        reasons.append(f"min n_eff {n_eff if n_eff is None else f'{n_eff:.0f}'} < {CONVERGENCE_MIN_NEFF:.0f}")
    return not reasons, reasons
M2D_POSTERIOR_DRAWS = 24


def campaign_root(config: MockValidationConfig) -> Path:
    return Path(str(config.paths.output_dir)) / str(config.paths.campaign_name) / str(config.paths.run_name)


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{int(seed):08d}"


def generate_mock(root: Path, config, seed: int, *, log=None) -> MockPaths:
    """Draw the truth for ``seed``, lens it, and write the realization under ``root``."""
    truth = draw_truth(config, seed)
    sources, images = generate_images(truth, config, np.random.default_rng(int(seed) + 1_000_003), log=log)
    return write_mock(root, truth, sources, images, config.with_updates(seed=int(seed)).to_json_dict())


# ----------------------------------------------------------------------------- recovery tables
def posterior_columns(state: ModelState, evaluator, posterior: PosteriorResults) -> dict[str, np.ndarray]:
    """Posterior draws per parameter name in physical units, with derived masses, gamma_ml and source positions."""
    physical = np.asarray(posterior.physical, dtype=float)
    columns = {spec.name: physical[:, i] for i, spec in enumerate(state.parameter_specs) if spec.component_family != "source_position"}
    for prefix in {name.rsplit(".", 1)[0] for name in columns}:
        v, cut = columns.get(f"{prefix}.v_disp"), columns.get(f"{prefix}.cut_radius_kpc")
        if v is not None and cut is not None:
            columns[f"{prefix}.log10_mass_msun"] = log10_dpie_mass_msun(v, cut)
        sigma, cutkpc = columns.get(f"{prefix}.sigma"), columns.get(f"{prefix}.cutkpc")
        if sigma is not None and cutkpc is not None:
            columns[f"{prefix}.log10_mass_msun"] = log10_dpie_mass_msun(sigma, cutkpc)
        alpha, beta = columns.get(f"{prefix}.alpha_sigma"), columns.get(f"{prefix}.beta_radius")
        if alpha is not None and beta is not None:
            columns[f"{prefix}.gamma_ml"] = 2.0 * alpha + beta - 1.0
    sources = np.asarray(jax.vmap(jax.jit(evaluator.source_positions))(jnp.asarray(posterior.samples)))
    for k, family_id in enumerate(state.images.family_ids):
        columns[f"source.{family_id}.beta_x"] = sources[:, k, 0]
        columns[f"source.{family_id}.beta_y"] = sources[:, k, 1]
    return columns


def parameter_recovery(columns: dict[str, np.ndarray], truth: dict[str, float], best_index: int, *, positive: set[str] = frozenset()) -> pd.DataFrame:
    """Per parameter: quantiles, bias, the truth's percentile rank, 68% and 95% coverage.  ``scored`` marks the rows that
    count towards coverage: a finite truth that lies inside the parameter's support (a positive parameter whose truth is
    zero, e.g. an injected scatter of zero, can never be covered and is reported but not scored)."""
    rows = []
    for name, values in columns.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q025, q16, median, q84, q975 = np.quantile(finite, [0.025, 0.16, 0.5, 0.84, 0.975])
        t = truth.get(name, np.nan)
        scored = bool(np.isfinite(t)) and not (name in positive and t <= 0.0)
        rows.append({"parameter": name, "truth": t, "best_fit": float(values[best_index]), "q025": float(q025), "q16": float(q16), "median": float(median),
                     "q84": float(q84), "q975": float(q975), "bias": float(median - t), "fractional_bias": float((median - t) / t) if np.isfinite(t) and t != 0 else np.nan,
                     "truth_percentile": float(np.mean(finite <= t)) if np.isfinite(t) else np.nan,
                     "covered_68": bool(q16 <= t <= q84) if scored else False, "covered_95": bool(q025 <= t <= q975) if scored else False, "scored": scored})
    return pd.DataFrame(rows)


def magnification_recovery(truth: dict[str, Any], magnification: pd.DataFrame) -> pd.DataFrame:
    """Per image: truth magnification at the true, observed and recovered model positions; model magnification at the
    observed and recovered positions (best fit and posterior quantiles); fractional errors at the observed position."""
    truth_images = pd.DataFrame(truth["images"])[["image_label", "family_id", "x_true_arcsec", "y_true_arcsec", "magnification_true", "magnification_true_at_obs"]]
    merged = truth_images.merge(magnification.drop(columns=[c for c in ("family_id",) if c in magnification]), on="image_label", how="left")
    merged["magnification_true_at_model_position"] = np.nan
    if "x_model_arcsec" in merged:
        mock = truth_from_json(truth)
        for z, rows in merged.groupby("z_source"):
            x, y = rows["x_model_arcsec"].to_numpy(float), rows["y_model_arcsec"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                _, _, det = jacobian(lens_state(mock, float(z)), x[ok], y[ok])
                merged.loc[rows.index[ok], "magnification_true_at_model_position"] = 1.0 / det
    mu_true = merged["magnification_true_at_obs"].to_numpy(float)
    for suffix in ("", "_q16", "_q50", "_q84"):
        values = merged[f"magnification_model{suffix}"].to_numpy(float)
        merged[f"abs_fractional_error{suffix}"] = np.abs(np.abs(values) - np.abs(mu_true)) / np.maximum(np.abs(mu_true), 1e-8)
    merged["parity_match"] = np.sign(mu_true) == np.sign(merged["magnification_model"].to_numpy(float))
    return merged


RELATION_FIELDS = ("sigma", "cutkpc", "alpha_sigma", "beta_radius")
RELATION_DRAWS = 400


def relation_recovery(state: ModelState, posterior: PosteriorResults, truth_values: dict[str, float]) -> dict[str, Any]:
    """The member relation at the solver pivot: truth, prior width and thinned posterior draws of sigma*, r_cut*, alpha
    and beta, plus the pivot luminosity in units of the truth's L*.  Feeds the campaign's relation-recovery figure."""
    if not state.potfiles:
        return {}
    specs = {spec.field: (i, spec) for i, spec in enumerate(state.parameter_specs) if spec.component_family == "scaling" and spec.field in RELATION_FIELDS}
    if len(specs) < len(RELATION_FIELDS):
        return {}
    physical = np.asarray(posterior.physical, dtype=float)
    keep = np.unique(np.linspace(0, len(physical) - 1, min(RELATION_DRAWS, len(physical))).astype(int))
    pivot = state.potfiles[0].get("pivot") or {}
    prior_sigma = {}
    for field, (_, spec) in specs.items():
        if spec.transform_kind == "log_positive":            # fractional width about the prior median
            prior_sigma[field] = float(spec.physical_std / spec.physical_mean) if spec.physical_std and spec.physical_mean else None
        else:
            prior_sigma[field] = float(spec.std) if spec.std is not None else None
    return {"pivot_luminosity_ratio": float(pivot.get("luminosity_ratio", 1.0)),
            "truth": {field: float(truth_values.get(f"potfile.{field}", np.nan)) for field in RELATION_FIELDS},
            "prior_sigma": prior_sigma,
            "draws": {field: physical[keep, i].tolist() for field, (i, _) in specs.items()}}


def subhalo_masses(truth: dict[str, Any], member_table: pd.DataFrame) -> dict[str, Any]:
    """Truth masses of the modelled members (pi sigma^2 r_cut / G) against the recovered member masses."""
    members = [m for m in truth.get("members", []) if m.get("modelled", True)]
    truth_mass = np.asarray([10.0 ** log10_dpie_mass_msun(float(m["v_disp"]), float(m["cut_radius_kpc"])) for m in members])
    recovered = member_table["log10_mass_msun_median"].to_numpy(float) if not member_table.empty else np.asarray([])
    relation = truth.get("relation", {})
    exponent = 2.0 * float(relation.get("alpha_sigma", 0.27)) + float(relation.get("beta_radius", 0.65))
    anchor = 10.0 ** log10_dpie_mass_msun(float(relation.get("sigma_star_kms", 200.0)), float(relation.get("cut_star_kpc", 12.0)))
    return {"truth_log10_mass": np.log10(truth_mass[truth_mass > 0]).tolist(), "recovered_log10_mass": recovered[np.isfinite(recovered)].tolist(),
            "selection": {"schechter_alpha": relation.get("schechter_alpha"), "dpie_mass_anchor_msun": anchor, "mass_luminosity_exponent": exponent,
                          "faint_magnitude_limit": relation.get("faint_magnitude_limit"), "pivot_magnitude": relation.get("pivot_magnitude")}}


def model_groups(state: ModelState, config_model, physical_row: np.ndarray, name_index: dict[str, int]) -> dict[str, tuple[np.ndarray, ...]]:
    """Physical dPIE components implied by one posterior draw, grouped as the truth is."""
    kpc = float(state.kpc_per_arcsec)
    halos = {h.id: h for h in config_model.large_halos if isinstance(h, DPIEHaloConfig)}

    def value(name: str, default: float) -> float:
        index = name_index.get(name)
        return float(physical_row[index]) if index is not None else float(default)

    def large(prefix: str) -> tuple[np.ndarray, ...]:
        h = halos[prefix]
        return tuple(np.asarray([value(f"{prefix}.{key}", getattr(h, key)) * scale]) for key, scale in
                     (("v_disp", 1.0), ("core_radius_kpc", 1.0), ("cut_radius_kpc", 1.0), ("x_centre", kpc), ("y_centre", kpc)))

    spec = state.packed_lens_spec
    dev_v, dev_r = (np.asarray(a) for a in member_log_deviations(spec, physical_row))
    v, core, cut, xs, ys = [], [], [], [], []
    for record in state.scaling_component_records:
        c = int(record["component_index"])
        lum = float(spec.luminosity_ratio[c])
        # A reference value with index -1 is fixed at its base (e.g. the member core with fit_corekpc=False).
        sigma_ref, cut_ref, core_ref, alpha, beta = (
            float(physical_row[int(idx[c])]) if int(idx[c]) >= 0 else float(base[c])
            for idx, base in ((spec.sigma_ref_param_index, spec.sigma_ref_base), (spec.cut_ref_param_index, spec.cut_ref_base),
                              (spec.core_ref_param_index, spec.core_ref_base), (spec.alpha_sigma_param_index, spec.alpha_sigma_base),
                              (spec.beta_radius_param_index, spec.beta_radius_base)))
        v.append(sigma_ref * lum ** alpha * math.exp(float(dev_v[c])))
        core.append(core_ref * lum ** beta * math.exp(float(dev_r[c])))
        cut.append(cut_ref * lum ** beta * math.exp(float(dev_r[c])))
        xs.append(float(record["x_centre"]) * kpc)
        ys.append(float(record["y_centre"]) * kpc)
    return {"halo": large("halo"), "bcg": large("bcg"), "members": tuple(np.asarray(a, dtype=float) for a in (v, core, cut, xs, ys))}


def m2d_recovery(state: ModelState, config_model, posterior: PosteriorResults, truth: dict[str, Any], *, max_draws: int = M2D_POSTERIOR_DRAWS) -> pd.DataFrame:
    """Aperture masses about the true halo centre: model quantiles and model/truth ratios per component group."""
    radii = np.asarray(APERTURE_RADII_KPC)
    kpc = float(truth["kpc_per_arcsec"])
    centre = (float(truth["halo"]["x_centre"]) * kpc, float(truth["halo"]["y_centre"]) * kpc)
    truth_profiles = aperture_profiles(radii, truth_groups(truth), centre_kpc=centre)
    images = pd.DataFrame(truth.get("images", []))
    outer = float(np.hypot(images["x_true_arcsec"] - truth["halo"]["x_centre"], images["y_true_arcsec"] - truth["halo"]["y_centre"]).max() * kpc) if len(images) else np.nan
    name_index = {spec.name: i for i, spec in enumerate(state.parameter_specs)}
    physical = np.asarray(posterior.physical, dtype=float)
    draws = physical[np.unique(np.linspace(0, len(physical) - 1, min(int(max_draws), len(physical))).astype(int))]
    per_group = {g: [] for g in (*COMPONENT_GROUPS, "total")}
    for row in progress(draws, desc="aperture draws"):
        for group, profile in aperture_profiles(radii, model_groups(state, config_model, row, name_index), centre_kpc=centre).items():
            per_group[group].append(profile)
    rows = []
    for group, profiles in per_group.items():
        stacked = np.stack(profiles)
        q16, q50, q84 = np.percentile(stacked, [16, 50, 84], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(truth_profiles[group] > 0, stacked / truth_profiles[group], np.nan)
        r16, r50, r84 = np.nanpercentile(ratio, [16, 50, 84], axis=0)
        for k, radius in enumerate(radii):
            rows.append({"component": group, "radius_kpc": float(radius), "truth_m2d_msun": float(truth_profiles[group][k]),
                         "model_q16": float(q16[k]), "model_q50": float(q50[k]), "model_q84": float(q84[k]),
                         "ratio_q16": float(r16[k]), "ratio_q50": float(r50[k]), "ratio_q84": float(r84[k]),
                         "outer_image_radius_kpc": outer, "within_images": bool(radius <= outer)})
    return pd.DataFrame(rows)


def write_recovery(recovery_dir: Path, result: pipeline.RunResult, solver_config, truth: dict[str, Any], seed: int, *, log=print) -> dict[str, Any]:
    """Compare the posterior with the mock truth and write recovery.json plus CSV tables."""
    from ..plots import member_scaling_table
    recovery_dir.mkdir(parents=True, exist_ok=True)
    state, evaluator, posterior = result.state, result.evaluator, result.posterior
    tables = result.paths.final.tables
    best_index = int(np.nanargmax(np.asarray(posterior.log_prob)))
    columns = posterior_columns(state, evaluator, posterior)
    pivot = float(state.potfiles[0]["mag0"]) if state.potfiles else None
    truth_values = parameter_truth(truth, solver_pivot_magnitude=pivot)
    positive = {spec.name for spec in state.parameter_specs if spec.transform_kind in {"log_positive", "log_offset_positive"}}
    parameters = parameter_recovery(columns, truth_values, best_index, positive=positive)
    magnification = magnification_recovery(truth, pd.read_csv(tables / "model_magnification.csv"))
    relation = relation_recovery(state, posterior, truth_values)
    image_fit = pd.read_csv(tables / "image_fit_quality.csv")
    residual_draws = pd.read_csv(tables / "posterior_image_residual_draws.csv")
    member_table = member_scaling_table(state, posterior)
    m2d = m2d_recovery(state, solver_config.model, posterior, truth)
    for name, frame in (("parameter_recovery", parameters), ("magnification_recovery", magnification), ("m2d_aperture_recovery", m2d)):
        frame.to_csv(recovery_dir / f"{name}.csv", index=False)
    fit = json.loads((tables / "run_summary.json").read_text()).get("fit", {})
    diagnostics = posterior.diagnostics or {}
    converged, reasons = convergence_verdict(diagnostics)
    if not converged:
        log(f"[recovery] seed={seed} not converged: {'; '.join(reasons)}")
    scored = parameters[parameters["scored"]] if not parameters.empty else parameters
    payload = {
        "seed": int(seed), "mode": truth.get("config", {}).get("mode"), "n_parameters": int(state.n_parameters), "images": int(state.images.n_images),
        "families": int(state.images.n_families), "members": int(len(state.members.member_ids)), "exact_members": int(len(state.members.exact_positions)),
        "unmodelled_members": int(sum(1 for m in truth.get("members", []) if not m.get("modelled", True))),
        "fit": fit, "max_r_hat": diagnostics.get("max_r_hat"), "min_n_eff": diagnostics.get("min_n_eff"),
        "divergences": int(np.asarray(posterior.diverging).sum()), "audit": result.audit.get("approximation_validation"),
        "converged": bool(converged), "convergence_reasons": reasons,
        "coverage_68": float(scored["covered_68"].mean()) if len(scored) else None, "coverage_95": float(scored["covered_95"].mean()) if len(scored) else None,
        "scored_parameters": int(len(scored)),
        "parameters": parameters.to_dict("records"), "magnification": magnification.to_dict("records"), "relation": relation,
        "image_residuals": image_fit[["family_id", "image_label", "image_recovery_status", "image_residual_arcsec"]].to_dict("records"),
        "image_residual_draws": residual_draws.to_dict("records"), "subhalos": subhalo_masses(truth, member_table),
        "m2d": m2d.to_dict("records"),
    }
    (recovery_dir / RECOVERY_FILE).write_text(json.dumps(payload, indent=1, default=_json_default))
    log(f"[recovery] seed={seed} images recovered {fit.get('recovered_image_count')}/{fit.get('observed_image_count')} "
        f"rms={fit.get('image_rms_arcsec')} coverage_68={payload['coverage_68']} coverage_95={payload['coverage_95']} "
        f"over {len(scored)} scored parameters, max_r_hat={diagnostics.get('max_r_hat')}")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def rescore_mock_validation(config: MockValidationConfig, *, log=print) -> list[Path]:
    """Recompute the recovery tables of every finished realization from its saved posterior, without refitting (for a
    change in how the truth or the recovery is scored)."""
    from .. import artifacts
    config = config.validate()
    root = campaign_root(config)
    outputs = []
    for realization in range(int(config.runtime.realizations)):
        seed = int(config.runtime.seed) + realization
        directory = seed_dir(root, seed)
        paths = MockPaths(directory / "mock")
        if not paths.complete():
            continue
        truth = json.loads(paths.truth.read_text())
        solver_config = solver_config_for_mock(config, truth, image_catalog=paths.image_catalog,
                                               member_catalog=paths.member_catalog if paths.member_catalog.is_file() else None, seed=seed, output_dir=directory)
        run_paths = pipeline.run_paths(solver_config)
        if not artifacts.is_final_complete(run_paths.final.artifacts):
            log(f"[rescore] seed={seed} has no complete posterior; skipping")
            continue
        init = pipeline.run_init(replace(solver_config, runtime=replace(solver_config.runtime, quiet=True)))   # resumes from start.json
        posterior = artifacts.load_posterior(run_paths.final.artifacts)
        audit = json.loads((run_paths.final.artifacts / artifacts.AUDIT_FILE).read_text())
        result = pipeline.RunResult(run_paths, init, init.state, init.evaluator, posterior, audit, resumed=True)
        write_recovery(directory / "recovery", result, solver_config, truth, seed, log=log)
        outputs.append(directory / "recovery")
    return outputs


# ----------------------------------------------------------------------------- campaign
def run_mock_validation(config: MockValidationConfig, *, log=print) -> list[Path]:
    """Run every realization of the campaign; returns the recovery directories."""
    config = config.validate()
    root = campaign_root(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_config.json").write_text(json.dumps(config.to_json_dict(), indent=2, default=_json_default))
    outputs = []
    for realization in range(int(config.runtime.realizations)):
        seed = int(config.runtime.seed) + realization
        directory = seed_dir(root, seed)
        recovery_dir = directory / "recovery"
        if config.runtime.resume and (recovery_dir / RECOVERY_FILE).is_file():
            log(f"[campaign] seed={seed} already complete; skipping")
            outputs.append(recovery_dir)
            continue
        started = time.perf_counter()
        paths = MockPaths(directory / "mock")
        if config.runtime.resume and paths.complete():
            log(f"[campaign] seed={seed} reusing mock under {paths.root}")
        else:
            log(f"[campaign] seed={seed} generating mock under {paths.root}")
            paths = generate_mock(paths.root, config.mock, seed, log=lambda m: log(f"[campaign] seed={seed} {m}"))
        truth = json.loads(paths.truth.read_text())
        log(f"[campaign] seed={seed} truth: z={truth['z_lens']:.3f} halo {truth['halo']['v_disp']:.0f} km/s, members {truth.get('n_modelled_members', 0)} modelled, "
            f"{len(truth.get('sources', []))} families, {len(truth.get('images', []))} images; solver mode {config.mock.mode}")
        solver_config = solver_config_for_mock(config, truth, image_catalog=paths.image_catalog,
                                               member_catalog=paths.member_catalog if paths.member_catalog.is_file() else None, seed=seed, output_dir=directory)
        result = pipeline.run(solver_config)
        write_recovery(recovery_dir, result, solver_config, truth, seed, log=log)
        log(f"[campaign] seed={seed} done in {time.perf_counter() - started:.0f}s")
        outputs.append(recovery_dir)
    return outputs
