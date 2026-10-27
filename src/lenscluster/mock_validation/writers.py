"""Files of one mock realization: the image catalog and member catalog the solver reads, and the truth."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .truth import MockTruth

IMAGE_CATALOG = "obs_arcs.cat"
MEMBER_CATALOG = "members.cat"
TRUTH_FILE = "truth.json"
IMAGES_FILE = "mock_images.json"


@dataclass(frozen=True)
class MockPaths:
    root: Path

    @property
    def image_catalog(self) -> Path: return self.root / IMAGE_CATALOG
    @property
    def member_catalog(self) -> Path: return self.root / MEMBER_CATALOG
    @property
    def truth(self) -> Path: return self.root / TRUTH_FILE
    @property
    def images(self) -> Path: return self.root / IMAGES_FILE

    def complete(self) -> bool:
        return self.image_catalog.is_file() and self.truth.is_file() and self.images.is_file()


def write_mock(root: Path, truth: MockTruth, sources: list[dict[str, Any]], images: pd.DataFrame, config_json: dict[str, Any]) -> MockPaths:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = MockPaths(root)
    # Image catalog, Lenstool arcs format: label x y a b theta z magnitude weight magnitude_error.
    paths.image_catalog.write_text("#REFERENCE 3\n" + "".join(
        f"{r.image_label} {r.x_obs_arcsec:.8f} {r.y_obs_arcsec:.8f} 0.3734 0.3734 90.0 {r.z_source:.8f} {r.image_magnitude_obs:.6f} 1.0 {r.image_magnitude_err:.4f}\n"
        for r in images.itertuples(index=False)), encoding="utf-8")
    # Member catalog, potfile format: id x y a b theta magnitude luminosity colour; only the modelled members.
    modelled = truth.modelled_members
    if modelled:
        lines = []
        for m in modelled:
            q = math.sqrt((1.0 - m.ellipticite) / (1.0 + m.ellipticite))     # Lenstool ellipticite = (a^2 - b^2) / (a^2 + b^2)
            lines.append(f"{m.id:>10s} {m.x_centre: .8f} {m.y_centre: .8f} {1.0:.6f} {q:.6f} {m.angle_pos:.4f} {m.magnitude:.6f} {m.luminosity_ratio:.8e} nan\n")
        paths.member_catalog.write_text("#REFERENCE 3\n" + "".join(lines), encoding="utf-8")
    payload = {"mock": "analytic-cluster", "config": config_json, **truth.to_json_dict(),
               "sigma_def_arcsec": float(config_json.get("sigma_def_arcsec", 0.0)),
               "sources": sources, "images": images.to_dict("records"),
               "dropped_families": images.attrs.get("dropped_families", []), "caustics_by_source_redshift": images.attrs.get("caustics_by_z", {})}
    paths.truth.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    paths.images.write_text(json.dumps(images.to_dict("records"), indent=1, default=_json_default), encoding="utf-8")
    return paths


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
