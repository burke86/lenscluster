#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import astropy.units as u
from astropy.coordinates import SkyCoord

DEFAULT_OUTPUT_DIR = Path("data") / "ff_sims"
DEFAULT_SOURCE_ROOT = DEFAULT_OUTPUT_DIR / "published"
IMAGE_SHAPE_A = 0.3734
IMAGE_SHAPE_B = 0.3734
IMAGE_THETA_DEG = 90.0
IMAGE_PLACEHOLDER_MAG = 25.0
CLUSTER_CUT_RADIUS_KPC = 1500.0
GALAXY_CORE_RADIUS_KPC = 0.15
MEMBER_SHAPE_MATCH_TOLERANCE_ARCSEC = 0.3

CenterPriorBox = tuple[float, float, float, float]
CoreRadiusPrior = tuple[float, float, float]
ExplicitCoreRadiusPrior = tuple[float, float, float, float]
VelocityDispersionPrior = tuple[float, float, float]


@dataclass(frozen=True)
class FFSimConfig:
    key: str
    display_name: str
    z_lens: float
    multimages_name: str = "multimages.txt"
    members_name: str = "clgal_cat.txt"
    shaped_members_name: str | None = None
    member_shape_match_tolerance_arcsec: float = MEMBER_SHAPE_MATCH_TOLERANCE_ARCSEC
    member_selection_band: str = "f814w"
    member_selection_max_mag: float = 24.0
    scaling_band: str = "f160w"
    mag0: float = 20.0
    sigma_ref: float = 100.0
    sigma_ref_uncertainty: float = 10.0
    sigma_ref_lower: float = 30.0
    sigma_ref_upper: float = 250.0
    cut_radius_ref_kpc: float = 30.0
    cut_radius_ref_uncertainty_kpc: float = 1.0
    cut_radius_ref_lower_kpc: float = 3.0
    cut_radius_ref_upper_kpc: float = 250.0
    smooth_anchor_ids: tuple[str, str] = ("1", "2")
    smooth_v_disps: tuple[float, ...] = (950.0, 750.0)
    smooth_v_disp_priors: tuple[VelocityDispersionPrior, ...] | None = None
    smooth_ellipticity_upper: float = 0.8
    smooth_angle_positions: tuple[float, ...] | None = None
    smooth_center_prior_boxes: tuple[CenterPriorBox, ...] | None = None
    smooth_center_prior_half_widths: tuple[float, ...] | None = None
    smooth_core_radius_priors: tuple[CoreRadiusPrior, ...] | None = None
    explicit_galaxy_sigma_ref: float | None = None
    explicit_galaxy_cut_radius_ref_kpc: float | None = None
    explicit_galaxy_core_radius_kpc: float = GALAXY_CORE_RADIUS_KPC
    explicit_galaxy_core_radius_prior: ExplicitCoreRadiusPrior | None = None
    explicit_galaxy_free_core_ids: tuple[str, ...] | None = None
    shear_gamma: float | None = None
    shear_angle_pos: float = 0.0
    explicit_galaxy_count: int = 2
    explicit_galaxy_ids: tuple[str, ...] | None = None
    explicit_galaxy_centers_fixed: bool = False
    explicit_galaxy_sigma_upper_factors: tuple[tuple[str, float], ...] = ()
    cosmology_h0: float = 70.0
    cosmology_omega_m: float = 0.3
    expected_n_images: int = 0
    expected_n_image_families: int = 0
    expected_n_members: int = 0
    expected_n_skipped_members: int = 0


@dataclass(frozen=True)
class MultipleImageRow:
    x_arcsec: float
    y_arcsec: float
    family_id: int
    image_id: int
    z_source: float


@dataclass(frozen=True)
class MemberRow:
    object_id: str
    x_arcsec: float
    y_arcsec: float
    a_axis: float
    b_axis: float
    theta_deg: float
    f814w_mag: float
    f160w_mag: float
    scaling_mag: float
    luminosity: float
    color: float


@dataclass(frozen=True)
class MemberShape:
    object_id: str
    x_arcsec: float
    y_arcsec: float
    a_axis: float
    b_axis: float
    theta_deg: float


CONFIGS = {
    "ares": FFSimConfig(
        key="ares",
        display_name="Ares",
        z_lens=0.5,
        member_selection_band="f814w",
        member_selection_max_mag=24.0,
        scaling_band="f160w",
        mag0=18.5,
        sigma_ref=100.0,
        sigma_ref_uncertainty=15.0,
        sigma_ref_lower=70.0,
        sigma_ref_upper=500.0,
        cut_radius_ref_kpc=270.0,
        cut_radius_ref_uncertainty_kpc=35.0,
        cut_radius_ref_lower_kpc=160.0,
        cut_radius_ref_upper_kpc=800.0,
        smooth_anchor_ids=("2", "3"),
        smooth_v_disps=(950.0, 950.0),
        smooth_v_disp_priors=((250.0, 500.0, 1800.0), (175.0, 600.0, 1400.0)),
        smooth_ellipticity_upper=0.5,
        smooth_center_prior_boxes=((15.0, 25.0, -37.0, -27.0), (-45.0, -35.0, 35.0, 45.0)),
        smooth_core_radius_priors=((20.0, 5.0, 60.0), (20.0, 5.0, 45.0)),
        explicit_galaxy_sigma_ref=98.0,
        explicit_galaxy_cut_radius_ref_kpc=262.0,
        explicit_galaxy_core_radius_kpc=0.7,
        explicit_galaxy_core_radius_prior=(0.7, 0.5, 0.15, 3.0),
        explicit_galaxy_free_core_ids=("1", "2"),
        explicit_galaxy_ids=(),
        explicit_galaxy_centers_fixed=True,
        cosmology_h0=70.4,
        cosmology_omega_m=0.272,
        expected_n_images=242,
        expected_n_image_families=85,
        expected_n_members=334,
        expected_n_skipped_members=0,
    ),
    "hera": FFSimConfig(
        key="hera",
        display_name="Hera",
        z_lens=0.507,
        shaped_members_name="galcat2.txt",
        member_selection_band="f814w",
        member_selection_max_mag=24.0,
        scaling_band="f160w",
        mag0=19.82,
        sigma_ref=96.7,
        sigma_ref_uncertainty=40.0,
        sigma_ref_lower=30.0,
        sigma_ref_upper=250.0,
        cut_radius_ref_kpc=33.0,
        cut_radius_ref_uncertainty_kpc=25.0,
        cut_radius_ref_lower_kpc=3.0,
        cut_radius_ref_upper_kpc=250.0,
        smooth_anchor_ids=("1", "2"),
        smooth_v_disps=(800.0, 700.0),
        smooth_angle_positions=(30.0, 24.0),
        smooth_center_prior_half_widths=(5.0, 5.0),
        smooth_core_radius_priors=((8.0, 2.0, 15.0), (5.0, 2.0, 15.0)),
        explicit_galaxy_ids=(),
        explicit_galaxy_centers_fixed=True,
        cosmology_h0=72.0,
        cosmology_omega_m=0.24,
        expected_n_images=65,
        expected_n_image_families=19,
        expected_n_members=340,
        expected_n_skipped_members=0,
    ),
}


def _resolve_source_root(source_root: str | Path | None = None) -> Path:
    path = Path(source_root) if source_root is not None else DEFAULT_SOURCE_ROOT
    if not path.is_dir():
        raise FileNotFoundError(f"FF-SIMS source root does not exist: {path}")
    return path


def _iter_data_rows(path: Path) -> Iterable[list[str]]:
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split()


def _read_multiple_images(path: Path) -> list[MultipleImageRow]:
    rows: list[MultipleImageRow] = []
    for parts in _iter_data_rows(path):
        if len(parts) < 5:
            raise ValueError(f"Multiple-image row in {path} has fewer than 5 columns: {parts}")
        rows.append(
            MultipleImageRow(
                x_arcsec=-float(parts[0]),
                y_arcsec=float(parts[1]),
                family_id=int(float(parts[2])),
                image_id=int(float(parts[3])),
                z_source=float(parts[4]),
            )
        )
    return rows


def _image_suffix(index: int) -> str:
    value = int(index)
    if value < 1:
        raise ValueError("Image index must be positive.")
    value -= 1
    letters: list[str] = []
    while True:
        letters.append(chr(ord("a") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def _luminosity_from_mag(mag: float, mag0: float) -> float:
    return float(10.0 ** (-0.4 * (float(mag) - float(mag0))))


def _parse_finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _read_plain_members(path: Path, *, position_fallback: dict[str, MemberShape] | None = None) -> tuple[list[MemberRow], int]:
    rows: list[MemberRow] = []
    skipped = 0
    for parts in _iter_data_rows(path):
        if len(parts) < 10:
            raise ValueError(f"Cluster-member row in {path} has fewer than 10 columns: {parts}")
        object_id = str(parts[0])
        f606w_mag = float(parts[4])
        f814w_mag = float(parts[5])
        f160w_mag = float(parts[9])
        source_x = _parse_finite_float(parts[1])
        source_y = _parse_finite_float(parts[2])
        fallback = (position_fallback or {}).get(object_id)
        if fallback is not None:
            # galcat2 uses solver x; clgal_cat uses the opposite x convention.
            # Fill only missing coordinates, preserving every measured value.
            if source_x is None and math.isfinite(fallback.x_arcsec):
                source_x = -fallback.x_arcsec
            if source_y is None and math.isfinite(fallback.y_arcsec):
                source_y = fallback.y_arcsec
        if source_x is None or source_y is None:
            skipped += 1
            continue
        rows.append(
            MemberRow(
                object_id=object_id,
                x_arcsec=-source_x,
                y_arcsec=source_y,
                a_axis=1.0,
                b_axis=1.0,
                theta_deg=0.0,
                f814w_mag=f814w_mag,
                f160w_mag=f160w_mag,
                scaling_mag=f160w_mag,
                luminosity=1.0,
                color=f606w_mag - f814w_mag,
            )
        )
    return rows, skipped


def _axis_arcsec(value: str) -> float:
    axis = float(value)
    if abs(axis) < 0.01:
        return axis * 3600.0
    return axis


def _read_member_shapes(path: Path) -> list[MemberShape]:
    shapes: list[MemberShape] = []
    for parts in _iter_data_rows(path):
        if len(parts) < 8:
            raise ValueError(f"Shaped member row in {path} has fewer than 8 columns: {parts}")
        object_id = str(parts[0])
        shapes.append(
            MemberShape(
                object_id=object_id,
                x_arcsec=float(parts[1]),
                y_arcsec=float(parts[2]),
                a_axis=_axis_arcsec(parts[3]),
                b_axis=_axis_arcsec(parts[4]),
                theta_deg=float(parts[5]),
            )
        )
    return shapes


def _arcsec_skycoords(rows: Iterable[MemberRow | MemberShape]) -> SkyCoord:
    row_list = list(rows)
    x_offsets = [row.x_arcsec for row in row_list] * u.arcsec
    y_offsets = [row.y_arcsec for row in row_list] * u.arcsec
    return SkyCoord(ra=180.0 * u.deg + x_offsets, dec=y_offsets, frame="icrs")


def _apply_matched_shapes(
    members: list[MemberRow],
    shapes: list[MemberShape],
    *,
    tolerance_arcsec: float = MEMBER_SHAPE_MATCH_TOLERANCE_ARCSEC,
) -> list[MemberRow]:
    if not members or not shapes:
        return members
    member_coords = _arcsec_skycoords(members)
    shape_coords = _arcsec_skycoords(shapes)
    shape_indices, separations, _ = member_coords.match_to_catalog_sky(shape_coords)
    matched: list[MemberRow] = []
    tolerance = float(tolerance_arcsec) * u.arcsec
    for member, shape_index, separation in zip(members, shape_indices, separations, strict=True):
        shape = shapes[int(shape_index)]
        if separation <= tolerance and shape.object_id == member.object_id:
            matched.append(
                replace(
                    member,
                    a_axis=shape.a_axis,
                    b_axis=shape.b_axis,
                    theta_deg=shape.theta_deg,
                )
            )
        else:
            matched.append(member)
    return matched


def _member_mag(row: MemberRow, band: str) -> float:
    normalized = band.lower()
    if normalized == "f814w":
        return row.f814w_mag
    if normalized == "f160w":
        return row.f160w_mag
    raise ValueError(f"Unsupported FF-SIMS member magnitude band: {band!r}")


def _prepare_cats_members(members: list[MemberRow], config: FFSimConfig) -> list[MemberRow]:
    prepared: list[MemberRow] = []
    for member in members:
        scaling_mag = _member_mag(member, config.scaling_band)
        prepared.append(
            replace(
                member,
                scaling_mag=scaling_mag,
                luminosity=_luminosity_from_mag(scaling_mag, config.mag0),
            )
        )
    return prepared


def _read_members(cluster_dir: Path, config: FFSimConfig) -> tuple[list[MemberRow], int]:
    shaped_path = cluster_dir / config.shaped_members_name if config.shaped_members_name else None
    shapes = _read_member_shapes(shaped_path) if shaped_path is not None and shaped_path.is_file() else []
    by_id = {shape.object_id: shape for shape in shapes}
    if len(by_id) != len(shapes):
        raise ValueError(f"Duplicate member identities in {shaped_path}")
    members, skipped = _read_plain_members(cluster_dir / config.members_name, position_fallback=by_id)
    if shapes:
        members = _apply_matched_shapes(
            members,
            shapes,
            tolerance_arcsec=config.member_shape_match_tolerance_arcsec,
        )
    return _prepare_cats_members(members, config), skipped


def _format_float(value: float) -> str:
    return f"{float(value):.8f}"


def _write_image_catalog(path: Path, images: list[MultipleImageRow]) -> None:
    lines = ["#REFERENCE 3"]
    for row in images:
        label = f"{row.family_id}.{_image_suffix(row.image_id)}"
        lines.append(
            f"{label:>10s} {row.x_arcsec: .8f} {row.y_arcsec: .8f} "
            f"{IMAGE_SHAPE_A:.4f} {IMAGE_SHAPE_B:.4f} {IMAGE_THETA_DEG:.1f} "
            f"{row.z_source:.8f} {IMAGE_PLACEHOLDER_MAG:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _member_catalog_line(row: MemberRow, *, commented: bool = False) -> str:
    prefix = "# " if commented else ""
    return (
        f"{prefix}{row.object_id:>10s} {row.x_arcsec: .8f} {row.y_arcsec: .8f} "
        f"{row.a_axis:.8f} {row.b_axis:.8f} {row.theta_deg:.4f} "
        f"{row.scaling_mag:.6f} {row.luminosity:.8e} {row.color:.6f}"
    )


def _id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


ExplicitMember = tuple[str, MemberRow]


def _write_member_catalog(
    path: Path,
    members: list[MemberRow],
    *,
    config: FFSimConfig,
    explicit_members: list[ExplicitMember],
) -> None:
    lines = [
        "#REFERENCE 3",
        "# FF-SIMS scaling-law member potfile generated from clgal_cat.txt.",
        (
            f"# Source catalog selection: {config.member_selection_band.upper()} < {config.member_selection_max_mag:.2f}; "
            f"scaling magnitudes: {config.scaling_band.upper()}; mag0={config.mag0:.2f}."
        ),
    ]
    if explicit_members:
        lines.append(
            "# The following bright galaxies are commented out because they are explicit optimized dPIE components."
        )
    anchor_lookup = {object_id: f"O{index}" for index, object_id in enumerate(config.smooth_anchor_ids, start=1)}
    for component_id, row in explicit_members:
        anchor_note = f" It also anchors {anchor_lookup[row.object_id]}." if row.object_id in anchor_lookup else ""
        lines.append(f"# Member {row.object_id} is excluded from the scaling potfile; modeled explicitly as {component_id}.{anchor_note}")
        lines.append(_member_catalog_line(row, commented=True))
    for row in sorted(members, key=lambda item: (item.scaling_mag, _id_sort_key(item.object_id))):
        lines.append(_member_catalog_line(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _member_by_id(members: list[MemberRow], object_id: str, *, context: str) -> MemberRow:
    for member in members:
        if member.object_id == object_id:
            return member
    raise ValueError(f"Cannot find required {context} member ID {object_id!r}.")


def _smooth_anchor_members(config: FFSimConfig, members: list[MemberRow]) -> list[MemberRow]:
    return [
        _member_by_id(members, object_id, context=f"{config.key} smooth halo")
        for object_id in config.smooth_anchor_ids
    ]


def _explicit_galaxy_members(config: FFSimConfig, members: list[MemberRow]) -> list[ExplicitMember]:
    if config.explicit_galaxy_ids is not None:
        return [
            (f"G{object_id}", _member_by_id(members, object_id, context=f"{config.key} explicit galaxy"))
            for object_id in config.explicit_galaxy_ids
        ]
    bright = sorted(members, key=lambda item: (item.scaling_mag, _id_sort_key(item.object_id)))
    return [(f"G{index}", member) for index, member in enumerate(bright[: config.explicit_galaxy_count], start=1)]


def _exclude_explicit_members(members: list[MemberRow], explicit_members: list[ExplicitMember]) -> list[MemberRow]:
    explicit_ids = {member.object_id for _component_id, member in explicit_members}
    return [member for member in members if member.object_id not in explicit_ids]


def _field_dmax(images: list[MultipleImageRow], members: list[MemberRow], padding_arcsec: float = 20.0) -> int:
    radii = [math.hypot(item.x_arcsec, item.y_arcsec) for item in images]
    radii.extend(math.hypot(item.x_arcsec, item.y_arcsec) for item in members)
    return int(math.ceil(max(radii, default=200.0) + float(padding_arcsec)))


def _scaled_sigma(row: MemberRow, config: FFSimConfig) -> float:
    sigma_ref = config.explicit_galaxy_sigma_ref if config.explicit_galaxy_sigma_ref is not None else config.sigma_ref
    return float(sigma_ref * row.luminosity**0.25)


def _scaled_cut_radius_kpc(row: MemberRow, config: FFSimConfig) -> float:
    cut_radius_ref = (
        config.explicit_galaxy_cut_radius_ref_kpc
        if config.explicit_galaxy_cut_radius_ref_kpc is not None
        else config.cut_radius_ref_kpc
    )
    return float(cut_radius_ref * row.luminosity**0.5)


def _center_prior_bounds(
    anchor: MemberRow,
    prior_box: CenterPriorBox | None,
    prior_half_width: float | None = None,
) -> CenterPriorBox:
    if prior_box is not None:
        return prior_box
    if prior_half_width is not None:
        half_width = float(prior_half_width)
        return (
            anchor.x_arcsec - half_width,
            anchor.x_arcsec + half_width,
            anchor.y_arcsec - half_width,
            anchor.y_arcsec + half_width,
        )
    return (
        anchor.x_arcsec - 120.0,
        anchor.x_arcsec + 120.0,
        anchor.y_arcsec - 120.0,
        anchor.y_arcsec + 120.0,
    )


def _cluster_potential_block(
    component_id: str,
    anchor: MemberRow,
    config: FFSimConfig,
    v_disp: float,
    angle_pos: float,
    *,
    center_prior_box: CenterPriorBox | None = None,
    center_prior_half_width: float | None = None,
    core_radius_prior: CoreRadiusPrior | None = None,
    v_disp_prior: VelocityDispersionPrior | None = None,
) -> str:
    x_lower, x_upper, y_lower, y_upper = _center_prior_bounds(anchor, center_prior_box, center_prior_half_width)
    core_radius, core_lower, core_upper = core_radius_prior or (20.0, 1.0, 120.0)
    v_disp_std, v_disp_lower, v_disp_upper = v_disp_prior or (max(100.0, 0.35 * v_disp), 100.0, 2200.0)
    return f"""potentiel {component_id}
    profil 81
    x_centre {_format_float(anchor.x_arcsec)}
    y_centre {_format_float(anchor.y_arcsec)}
    ellipticite 0.30000000
    angle_pos {_format_float(angle_pos)}
    core_radius {_format_float(core_radius)}
    cut_radius_kpc {_format_float(CLUSTER_CUT_RADIUS_KPC)}
    v_disp {_format_float(v_disp)}
    z_lens {_format_float(config.z_lens)}
    end
limit {component_id}
    x_centre 1 {_format_float(x_lower)} {_format_float(x_upper)} 0.10000000
    y_centre 1 {_format_float(y_lower)} {_format_float(y_upper)} 0.10000000
    ellipticite 1 0.00000000 {_format_float(config.smooth_ellipticity_upper)} 0.02000000
    angle_pos 1 -180.00000000 180.00000000 0.50000000
    core_radius 1 {_format_float(core_lower)} {_format_float(core_upper)} 0.10000000
    v_disp 9 {_format_float(v_disp)} {_format_float(v_disp_std)} {_format_float(v_disp_lower)} {_format_float(v_disp_upper)}
    end"""


def _explicit_galaxy_sigma_upper_factor(row: MemberRow, config: FFSimConfig) -> float:
    factors = {object_id: float(factor) for object_id, factor in config.explicit_galaxy_sigma_upper_factors}
    return factors.get(row.object_id, 1.5)


def _explicit_galaxy_core_is_free(component_id: str, row: MemberRow, config: FFSimConfig) -> bool:
    if config.explicit_galaxy_core_radius_prior is None:
        return False
    if config.explicit_galaxy_free_core_ids is None:
        return True
    free_ids = set(config.explicit_galaxy_free_core_ids)
    return row.object_id in free_ids or component_id in free_ids


def _galaxy_potential_block(component_id: str, row: MemberRow, config: FFSimConfig) -> str:
    sigma = _scaled_sigma(row, config)
    cut_radius = _scaled_cut_radius_kpc(row, config)
    core_radius = float(config.explicit_galaxy_core_radius_kpc)
    x_lower = row.x_arcsec - 1.0
    x_upper = row.x_arcsec + 1.0
    y_lower = row.y_arcsec - 1.0
    y_upper = row.y_arcsec + 1.0
    cut_lower = max(GALAXY_CORE_RADIUS_KPC + 0.01, 0.25 * cut_radius)
    cut_upper = max(cut_lower * 1.01, 2.0 * cut_radius)
    sigma_lower = max(10.0, 0.5 * sigma)
    sigma_upper_factor = _explicit_galaxy_sigma_upper_factor(row, config)
    sigma_upper = max(sigma_lower * 1.01, sigma_upper_factor * sigma)
    if config.explicit_galaxy_centers_fixed:
        x_limit = f"x_centre 0 {_format_float(row.x_arcsec)} 0"
        y_limit = f"y_centre 0 {_format_float(row.y_arcsec)} 0"
    else:
        x_limit = f"x_centre 1 {_format_float(x_lower)} {_format_float(x_upper)} 0.05000000"
        y_limit = f"y_centre 1 {_format_float(y_lower)} {_format_float(y_upper)} 0.05000000"
    if not _explicit_galaxy_core_is_free(component_id, row, config):
        core_limit = f"core_radius_kpc 0 {_format_float(core_radius)} 0"
    else:
        core_mean, core_std, core_lower, core_upper = config.explicit_galaxy_core_radius_prior
        core_limit = (
            f"core_radius_kpc 9 {_format_float(core_mean)} {_format_float(core_std)} "
            f"{_format_float(core_lower)} {_format_float(core_upper)}"
        )
    return f"""potentiel {component_id}
    profil 81
    x_centre {_format_float(row.x_arcsec)}
    y_centre {_format_float(row.y_arcsec)}
    ellipticite 0.00000000
    angle_pos 0.00000000
    core_radius_kpc {_format_float(core_radius)}
    cut_radius_kpc {_format_float(cut_radius)}
    v_disp {_format_float(sigma)}
    z_lens {_format_float(config.z_lens)}
    end
limit {component_id}
    {x_limit}
    {y_limit}
    {core_limit}
    cut_radius_kpc 1 {_format_float(cut_lower)} {_format_float(cut_upper)} 0.10000000
    v_disp 1 {_format_float(sigma_lower)} {_format_float(sigma_upper)} 1.00000000
    end"""


def _shear_potential_block(config: FFSimConfig) -> str:
    if config.shear_gamma is None:
        return ""
    return f"""# S1 is a weak external shear term that absorbs cluster-scale quadrupole structure.
potentiel S1
    profil 14
    gamma {_format_float(config.shear_gamma)}
    angle_pos {_format_float(config.shear_angle_pos)}
    z_lens {_format_float(config.z_lens)}
    end
limit S1
    gamma 1 0.00000000 0.30000000 0.00500000
    angle_pos 1 -180.00000000 180.00000000 0.50000000
    end
"""


def _write_par(
    path: Path,
    config: FFSimConfig,
    images: list[MultipleImageRow],
    members: list[MemberRow],
    explicit_members: list[ExplicitMember],
) -> None:
    if not members:
        raise ValueError(f"Cannot render {config.key}: no cluster members.")
    anchors = _smooth_anchor_members(config, members)
    v_disps = config.smooth_v_disps
    if len(v_disps) != len(anchors):
        raise ValueError(f"{config.key} has {len(anchors)} smooth halos but {len(v_disps)} smooth-halo velocity dispersions.")
    v_disp_priors = config.smooth_v_disp_priors or (None,) * len(anchors)
    if len(v_disp_priors) != len(anchors):
        raise ValueError(
            f"{config.key} has {len(anchors)} smooth halos but {len(v_disp_priors)} velocity-dispersion priors."
        )
    angle_positions = config.smooth_angle_positions or (0.0,) * len(anchors)
    if len(angle_positions) != len(anchors):
        raise ValueError(f"{config.key} has {len(anchors)} smooth halos but {len(angle_positions)} smooth-halo position angles.")
    center_prior_boxes = config.smooth_center_prior_boxes or (None,) * len(anchors)
    if len(center_prior_boxes) != len(anchors):
        raise ValueError(f"{config.key} has {len(anchors)} smooth halos but {len(center_prior_boxes)} center prior boxes.")
    center_prior_half_widths = config.smooth_center_prior_half_widths or (None,) * len(anchors)
    if len(center_prior_half_widths) != len(anchors):
        raise ValueError(
            f"{config.key} has {len(anchors)} smooth halos but {len(center_prior_half_widths)} center prior half-widths."
        )
    core_radius_priors = config.smooth_core_radius_priors or (None,) * len(anchors)
    if len(core_radius_priors) != len(anchors):
        raise ValueError(f"{config.key} has {len(anchors)} smooth halos but {len(core_radius_priors)} core-radius priors.")
    dmax = _field_dmax(images, members)
    explicit_blocks = "\n".join(
        f"# {component_id} is explicit galaxy member {member.object_id}, removed from the scaling potfile.\n"
        f"{_galaxy_potential_block(component_id, member, config)}"
        for component_id, member in explicit_members
    )
    explicit_section = ""
    if explicit_members:
        explicit_section = (
            "# Bright galaxies are explicit dPIE components and are commented out of the potfile.\n"
            f"{explicit_blocks}\n"
        )
    shear_block = _shear_potential_block(config)
    model_description = "two smooth dPIE halos, member scaling, and optional external shear"
    if explicit_members:
        model_description = "two smooth dPIE halos, explicit bright galaxies, member scaling, and optional external shear"
    explicit_scaling_note = ""
    if explicit_members and (
        config.explicit_galaxy_sigma_ref is not None or config.explicit_galaxy_cut_radius_ref_kpc is not None
    ):
        explicit_sigma_ref = config.explicit_galaxy_sigma_ref if config.explicit_galaxy_sigma_ref is not None else config.sigma_ref
        explicit_cut_ref = (
            config.explicit_galaxy_cut_radius_ref_kpc
            if config.explicit_galaxy_cut_radius_ref_kpc is not None
            else config.cut_radius_ref_kpc
        )
        explicit_scaling_note = (
            f"# Explicit galaxies are initialized with sigma*={explicit_sigma_ref:.3f} km/s "
            f"and rcut*={explicit_cut_ref:.3f} kpc.\n"
        )
    cosmology_lambda = 1.0 - float(config.cosmology_omega_m)
    content = f"""# Generated by scripts/build_ff_sims_lenscluster_inputs.py
# cluster_key {config.key}
# source FF-SIMS {config.display_name}
# Data-informed FF-SIMS Lenstool model: {model_description}.
# Members are read from clgal_cat.txt; source catalog selection is {config.member_selection_band.upper()} < {config.member_selection_max_mag:.2f}.
# Galaxy scaling uses {config.scaling_band.upper()} luminosities with mag0={config.mag0:.2f}, sigma*={config.sigma_ref:.3f} km/s, rcut*={config.cut_radius_ref_kpc:.3f} kpc.
{explicit_scaling_note.rstrip()}
runmode
    reference 3 0.00000000 0.00000000
    end
grille
    nombre 128
    polaire 0
    end
image
    multfile 1 {config.key}_obs_arcs.cat
    forme -2
    mult_wcs 1
    sigposArcsec 0.50000000
    end
# O1 is the first smooth dPIE clump, anchored on clgal_cat.txt member {anchors[0].object_id}.
{_cluster_potential_block("O1", anchors[0], config, v_disps[0], angle_positions[0], center_prior_box=center_prior_boxes[0], center_prior_half_width=center_prior_half_widths[0], core_radius_prior=core_radius_priors[0], v_disp_prior=v_disp_priors[0])}
# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member {anchors[1].object_id}.
{_cluster_potential_block("O2", anchors[1], config, v_disps[1], angle_positions[1], center_prior_box=center_prior_boxes[1], center_prior_half_width=center_prior_half_widths[1], core_radius_prior=core_radius_priors[1], v_disp_prior=v_disp_priors[1])}
{shear_block}
{explicit_section.rstrip()}
potfile
    filein 3 {config.key}_cluster_members_potfile.cat
    type 81
    mag0 {_format_float(config.mag0)}
    corekpc {_format_float(GALAXY_CORE_RADIUS_KPC)}
    z_lens {_format_float(config.z_lens)}
    sigma 9 {_format_float(config.sigma_ref)} {_format_float(config.sigma_ref_uncertainty)} {_format_float(config.sigma_ref_lower)} {_format_float(config.sigma_ref_upper)}
    cutkpc 9 {_format_float(config.cut_radius_ref_kpc)} {_format_float(config.cut_radius_ref_uncertainty_kpc)} {_format_float(config.cut_radius_ref_lower_kpc)} {_format_float(config.cut_radius_ref_upper_kpc)}
    end
cosmologie
    H0 {_format_float(config.cosmology_h0)}
    omega {_format_float(config.cosmology_omega_m)}
    lambda {_format_float(cosmology_lambda)}
    end
champ
    dmax {dmax}
    end
fini
"""
    path.write_text(content, encoding="utf-8")


def render(
    source_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    clusters: list[str] | None = None,
) -> list[dict[str, str | int | float]]:
    source = _resolve_source_root(source_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = [cluster.lower() for cluster in clusters] if clusters else list(CONFIGS)
    rows: list[dict[str, str | int | float]] = []
    for cluster_key in requested:
        if cluster_key not in CONFIGS:
            raise ValueError(f"Unsupported FF-SIMS cluster: {cluster_key}")
        config = CONFIGS[cluster_key]
        cluster_source_dir = source / cluster_key
        if not cluster_source_dir.is_dir():
            raise FileNotFoundError(f"Missing FF-SIMS source directory for {cluster_key}: {cluster_source_dir}")
        images = _read_multiple_images(cluster_source_dir / config.multimages_name)
        members, skipped_members = _read_members(cluster_source_dir, config)
        explicit_members = _explicit_galaxy_members(config, members)
        potfile_members = _exclude_explicit_members(members, explicit_members)
        cluster_output_dir = output / cluster_key
        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        image_path = cluster_output_dir / f"{cluster_key}_obs_arcs.cat"
        member_path = cluster_output_dir / f"{cluster_key}_cluster_members_potfile.cat"
        par_path = cluster_output_dir / f"{cluster_key}_lenscluster.par"
        _write_image_catalog(image_path, images)
        _write_member_catalog(member_path, potfile_members, config=config, explicit_members=explicit_members)
        _write_par(par_path, config, images, members, explicit_members)
        rows.append(
            {
                "cluster_key": cluster_key,
                "display_name": config.display_name,
                "z_lens": config.z_lens,
                "n_images": config.expected_n_images,
                "n_image_families": config.expected_n_image_families,
                "n_members": config.expected_n_members,
                "n_skipped_members": config.expected_n_skipped_members,
                "n_explicit_galaxies": len(explicit_members),
                "explicit_galaxy_ids": ";".join(member.object_id for _component_id, member in explicit_members),
                "member_selection": f"{config.member_selection_band.upper()}<{config.member_selection_max_mag:.2f}",
                "scaling_band": config.scaling_band.upper(),
                "mag0": config.mag0,
                "par_path": str(par_path),
                "obs_arcs_path": str(image_path),
                "potfile_path": str(member_path),
                "source_dir": str(cluster_source_dir),
            }
        )
    return rows


def validate_outputs(output_dir: str | Path = DEFAULT_OUTPUT_DIR, clusters: list[str] | None = None) -> list[dict[str, str | int]]:
    from lenscluster.lenstool_parser import load_best_par

    output = Path(output_dir)
    requested = [cluster.lower() for cluster in clusters] if clusters else list(CONFIGS)
    rows: list[dict[str, str | int]] = []
    for cluster_key in requested:
        par_path = output / cluster_key / f"{cluster_key}_lenscluster.par"
        if not par_path.is_file():
            raise FileNotFoundError(f"Generated par file is missing for {cluster_key}: {par_path}")
        parsed, potentials_df, images_df, _arcs_df, potentials_with_priors = load_best_par(par_path)
        rows.append(
            {
                "cluster_key": cluster_key,
                "n_potentials": len(potentials_df),
                "n_images": len(images_df),
                "n_image_families": int(images_df["family_id"].nunique()) if not images_df.empty else 0,
                "n_potfiles": len(parsed.get("potfiles", [])),
                "n_members": len(parsed["potfiles"][0]["catalog_df"]) if parsed.get("potfiles") else 0,
                "n_prior_components": sum(1 for item in potentials_with_priors if item.get("priors")),
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local Lenstool inputs for FF-SIMS Ares and Hera.")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clusters", nargs="+", choices=tuple(CONFIGS), default=None)
    parser.add_argument("--validate", action="store_true", help="Validate generated outputs after rendering.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = render(args.source_root, args.output_dir, args.clusters)
    print(f"Rendered {len(rows)} FF-SIMS cluster inputs.")
    if args.validate:
        validate_rows = validate_outputs(args.output_dir, args.clusters)
        print(f"Validated {len(validate_rows)} FF-SIMS cluster inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
