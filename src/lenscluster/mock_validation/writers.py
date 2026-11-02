"""Files of one mock realization: the image catalog and member catalog the solver reads, and the truth."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..image_catalog import systems_from_offsets, write_image_catalog
from .truth import MockTruth

IMAGE_CATALOG = "images.yaml"
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


def write_mock(root: Path, truth: MockTruth, sources: list[dict[str, Any]], images: pd.DataFrame, config_json: dict[str, Any],
               *, reference: tuple[float, float]) -> MockPaths:
    """``reference`` is the (ra0, dec0) the image offsets are measured from, the solver's reference point."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = MockPaths(root)
    # Image catalog (lenscluster.image_catalog): one system per family, observed positions and magnitudes.
    rows = []
    for r in images.itertuples(index=False):
        family, image = str(r.image_label).split(".")
        rows.append((family, image, r.x_obs_arcsec, r.y_obs_arcsec, r.z_source,
                     {"mag": float(r.image_magnitude_obs), "mag_err": float(r.image_magnitude_err)}))
    write_image_catalog(paths.image_catalog, systems_from_offsets(rows, reference),
                        header=f"Mock realization, seed {config_json.get('seed')}: observed image positions and magnitudes.")
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
