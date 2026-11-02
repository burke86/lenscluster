"""Arcfile and side-table writers.

The arcfile format: whitespace columns ``arc_id coord_1 coord_2 z_arc
tangent_angle_rad curvature_arcsec_inv sigma_tangent sigma_curvature
[reliability]`` with a ``#REFERENCE 0`` header (coordinates are then absolute
RA/Dec in degrees). The writer validates every field before writing.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import __version__
from .errors import ArcfileWriteError
from .measure import ArcMeasurement

_MIN_SIGMA = 1.0e-6
_UNKNOWN_ARC_REDSHIFT = -1.0


def _validate_arc_id(value: str) -> str:
    arc_id = str(value).strip()
    if not arc_id:
        raise ArcfileWriteError("Arc IDs must be non-empty.")
    if any(ch.isspace() for ch in arc_id):
        raise ArcfileWriteError(f"Arc ID {value!r} must not contain whitespace.")
    return arc_id


def _validated_rows(measurements: Sequence[ArcMeasurement]) -> list[ArcMeasurement]:
    rows: list[ArcMeasurement] = []
    arc_ids: dict[str, int] = {}
    for measurement in measurements:
        if not measurement.success:
            continue
        if measurement.label is None:
            raise ArcfileWriteError(f"Measurement at {measurement.seed_ra_deg:.6f},{measurement.seed_dec_deg:.6f} has no arc ID.")
        arc_id = _validate_arc_id(str(measurement.label))
        arc_ids[arc_id] = arc_ids.get(arc_id, 0) + 1
        values = {
            "anchor_ra_deg": measurement.anchor_ra_deg,
            "anchor_dec_deg": measurement.anchor_dec_deg,
            "z_arc": _UNKNOWN_ARC_REDSHIFT,
            "tangent_angle_offset_rad": measurement.tangent_angle_offset_rad,
            "curvature_arcsec_inv": measurement.curvature_arcsec_inv,
            "sigma_tangent_rad": measurement.sigma_tangent_rad,
            "sigma_curvature_arcsec_inv": measurement.sigma_curvature_arcsec_inv,
            "reliability": measurement.reliability,
        }
        bad = [name for name, value in values.items() if not math.isfinite(float(value))]
        if bad:
            raise ArcfileWriteError(f"Arc ID {arc_id}: non-finite values in {bad}.")
        if measurement.sigma_tangent_rad <= 0.0 or measurement.sigma_curvature_arcsec_inv <= 0.0:
            raise ArcfileWriteError(f"Arc ID {arc_id}: sigmas must be strictly positive.")
        rows.append(measurement)
    duplicates = sorted(arc_id for arc_id, count in arc_ids.items() if count > 1)
    if duplicates:
        raise ArcfileWriteError(f"Duplicate arc IDs are not allowed in an arcfile: {duplicates}.")
    if not rows:
        raise ArcfileWriteError("No successful labeled measurements to write.")
    return rows


def write_arcfile(
    measurements: Sequence[ArcMeasurement],
    path: str | Path,
    *,
    overwrite: bool = False,
    header_comments: Sequence[str] = (),
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise ArcfileWriteError(f"{path} exists; pass overwrite=True to replace it.")
    rows = _validated_rows(measurements)
    reference_band = rows[0].reference_band
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# arctrace v{__version__}  written {timestamp}  reference_band={reference_band}",
        *[f"# {comment}" for comment in header_comments],
        "#REFERENCE 0",
        "# arc_id ra_deg dec_deg z_arc tangent_angle_rad curvature_arcsec_inv "
        "sigma_tangent_angle_rad sigma_curvature_arcsec_inv reliability",
    ]
    for row in rows:
        arc_id = _validate_arc_id(str(row.label))
        lines.append(
            f"{arc_id} "
            f"{row.anchor_ra_deg:.7f} {row.anchor_dec_deg:.7f} "
            f"{_UNKNOWN_ARC_REDSHIFT:.1f} "
            f"{row.tangent_angle_offset_rad:.7f} "
            f"{abs(row.curvature_arcsec_inv):.6e} "
            f"{max(row.sigma_tangent_rad, _MIN_SIGMA):.6e} "
            f"{max(row.sigma_curvature_arcsec_inv, _MIN_SIGMA):.6e} "
            f"{float(np.clip(row.reliability, 0.0, 1.0)):.3f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def measurements_to_dataframe(measurements: Sequence[ArcMeasurement]) -> pd.DataFrame:
    records = []
    for measurement in measurements:
        record = dataclasses.asdict(measurement)
        record.pop("bands", None)
        record.pop("config_summary", None)
        record["warnings"] = "; ".join(measurement.warnings)
        for band in measurement.bands:
            prefix = f"band_{band.band}_"
            band_record = dataclasses.asdict(band)
            band_record["warnings"] = "; ".join(band.warnings)
            for key, value in band_record.items():
                if key == "band":
                    continue
                record[prefix + key] = value
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_sidetable(
    measurements: Sequence[ArcMeasurement],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    frame = measurements_to_dataframe(measurements)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    payload = [_json_safe(dataclasses.asdict(measurement)) for measurement in measurements]
    Path(json_path).write_text(json.dumps(payload, indent=2))
