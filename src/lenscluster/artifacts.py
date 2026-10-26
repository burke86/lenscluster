"""Saved run artifacts: the init stage (config, state, classification, search, start) and the final stage (posterior, audit)."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import LensClusterSolverConfig
from .model import ModelState, PosteriorResults

STATE_FILE = "state.json"
CONFIG_FILE = "config.json"
CONFIG_PICKLE = "config.pkl"
CLASSIFICATION_FILE = "classification.json"
SEARCH_FILE = "search.npz"
POSTERIOR_FILE = "posterior.npz"
AUDIT_FILE = "audit.json"
SUMMARY_FILE = "summary.json"
START_FILE = "start.json"
MODES_FILE = "modes.json"
INIT_SIGNATURE_FILE = "init_signature.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(directory: Path, config: LensClusterSolverConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / CONFIG_FILE, config.to_json_dict())
    (directory / CONFIG_PICKLE).write_bytes(pickle.dumps(config))


def load_config(directory: Path) -> LensClusterSolverConfig:
    return pickle.loads((directory / CONFIG_PICKLE).read_bytes())


def save_state(directory: Path, state: ModelState) -> None:
    members = state.members
    write_json(directory / STATE_FILE, {
        "run_name": state.run_name, "z_lens": state.z_lens, "kpc_per_arcsec": state.kpc_per_arcsec,
        "n_parameters": state.n_parameters, "parameter_specs": [asdict(spec) for spec in state.parameter_specs],
        "families": [{"family_id": f.family_id, "z_source": f.z_source, "image_labels": f.image_labels} for f in state.family_data],
        "member_ids": list(members.member_ids), "exact_positions": members.exact_positions.tolist(),
        "far_positions": members.far_positions.tolist(), "initial_theta": None if state.initial_theta is None else state.initial_theta.tolist(),
    })
    write_json(directory / CLASSIFICATION_FILE, members.classification)


def save_search(directory: Path, search: dict[str, Any]) -> None:
    np.savez_compressed(directory / SEARCH_FILE, theta=np.asarray(search["theta"]), potential=np.asarray(search["potential"]),
                        starts=np.asarray(search["starts"]), assignment=np.asarray(search["assignment"]),
                        iterations=np.asarray(search["iterations"]), evaluations=np.asarray(search["evaluations"]),
                        traces=np.asarray(search["traces"]), final_theta=np.asarray(search["final_theta"]))
    write_json(directory / MODES_FILE, {"modes": [{k: v for k, v in m.items() if k != "members"} | {"count": m["count"]} for m in search["modes"]],
                                           "method": search["method"], "n_global": search["n_global"], "search_seconds": search["search_seconds"],
                                           "polish_seconds": search["polish_seconds"], "n_finite": search["n_finite"],
                                           "stages": search.get("stages"), "stage_traces": search.get("stage_traces"),
                                           "starts_in_basin": search.get("starts_in_basin"), "configured_in_basin": search.get("configured_in_basin"),
                                           "escalated_from": search.get("escalated_from")})


def save_posterior(directory: Path, posterior: PosteriorResults) -> None:
    np.savez_compressed(directory / POSTERIOR_FILE, samples=posterior.samples, physical=posterior.physical,
                        log_prob=posterior.log_prob, accept_prob=posterior.accept_prob, diverging=posterior.diverging,
                        num_steps=posterior.num_steps, grouped_samples=posterior.grouped_samples,
                        median_fit=posterior.median_fit, maximum_likelihood_fit=posterior.maximum_likelihood_fit,
                        warmup_steps=posterior.warmup_steps, sample_steps=posterior.sample_steps, num_chains=posterior.num_chains)
    write_json(directory / "posterior_diagnostics.json", posterior.diagnostics)


def load_posterior(directory: Path) -> PosteriorResults:
    with np.load(directory / POSTERIOR_FILE, allow_pickle=True) as data:
        diagnostics = json.loads((directory / "posterior_diagnostics.json").read_text()) if (directory / "posterior_diagnostics.json").exists() else {}
        return PosteriorResults(samples=data["samples"], physical=data["physical"], log_prob=data["log_prob"],
                                accept_prob=data["accept_prob"], diverging=data["diverging"], num_steps=data["num_steps"],
                                warmup_steps=int(data["warmup_steps"]), sample_steps=int(data["sample_steps"]),
                                num_chains=int(data["num_chains"]), grouped_samples=data["grouped_samples"],
                                diagnostics=diagnostics, median_fit=data["median_fit"], maximum_likelihood_fit=data["maximum_likelihood_fit"])


def load_exact_positions(directory: Path) -> np.ndarray:
    payload = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    return np.asarray(payload["exact_positions"], dtype=np.int32)


def load_search(directory: Path) -> dict[str, Any]:
    """The saved search in the shape chain_starts expects (modes with theta and within_margin, best theta)."""
    payload = json.loads((directory / MODES_FILE).read_text(encoding="utf-8"))
    with np.load(directory / SEARCH_FILE) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    return {**arrays, "theta": np.asarray(arrays["theta"], dtype=float), "modes": payload["modes"], "method": payload.get("method"),
            "n_finite": payload.get("n_finite"), "search_seconds": payload.get("search_seconds"), "polish_seconds": payload.get("polish_seconds"),
            **({"stages": payload["stages"], "stage_traces": payload["stage_traces"]} if payload.get("stages") else {})}


def save_start(directory: Path, theta: np.ndarray, info: dict[str, Any]) -> None:
    write_json(directory / START_FILE, {"theta": np.asarray(theta, dtype=float).tolist(), **info})


def load_start(directory: Path) -> np.ndarray:
    return np.asarray(json.loads((directory / START_FILE).read_text(encoding="utf-8"))["theta"], dtype=float)


def init_signature(config: LensClusterSolverConfig) -> dict[str, Any]:
    """The part of the configuration the init stage depends on: model, members, scaling, likelihood, search, seed."""
    payload = config.to_json_dict()
    for key in ("sampling", "image_diagnostics", "truth", "image_catalog", "paths"):
        payload.pop(key, None)
    payload["runtime"] = {"seed": config.runtime.seed}
    return payload


def is_init_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in (INIT_SIGNATURE_FILE, STATE_FILE, CLASSIFICATION_FILE, SEARCH_FILE, MODES_FILE, START_FILE))


def is_final_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in (CONFIG_PICKLE, POSTERIOR_FILE, AUDIT_FILE))
