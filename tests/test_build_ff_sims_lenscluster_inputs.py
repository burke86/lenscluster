from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lenscluster.config import ScalingModelConfig
from lenscluster.state import build_halo_parameter_specs, build_scaling_parameter_specs
from lenscluster.jax_cosmology import cosmology_config_from_parsed
from lenscluster.lenstool_parser import load_best_par


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ff_sims_lenscluster_inputs.py"
spec = importlib.util.spec_from_file_location("build_ff_sims_lenscluster_inputs", SCRIPT_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def _write_ff_sims_fixture(root: Path) -> Path:
    published = root / "published"
    ares = published / "ares"
    hera = published / "hera"
    ares.mkdir(parents=True)
    hera.mkdir(parents=True)
    ares.joinpath("multimages.txt").write_text(
        "\n".join(
            [
                "10.0 -2.0 1 1 1.50",
                "12.0 -3.0 1 2 1.50",
                "-4.0 8.0 2 1 2.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("multimages.txt").write_text(
        "\n".join(
            [
                "5.0 1.0 3 1 2.00",
                "7.0 2.0 3 2 2.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ares.joinpath("clgal_cat.txt").write_text(
        "\n".join(
            [
                "# id x y AB(f435w) AB(f606w) AB(f814w) AB(f105w) AB(f125w) AB(f140w) AB(f160w)",
                "1 5.0 6.0 21.0 20.0 19.0 18.0 18.0 18.0 18.0",
                "2 -3.0 4.0 22.0 21.0 20.0 19.0 19.0 19.0 19.0",
                "3 -8.0 -8.0 22.2 21.2 20.2 19.2 19.2 19.2 19.2",
                "4 9.0 9.0 22.4 21.4 20.4 19.4 19.4 19.4 19.4",
                "5 10.0 -10.0 22.6 21.6 20.6 19.6 19.6 19.6 19.6",
                "6 -11.0 11.0 23.0 22.0 21.0 21.0 21.0 21.0 21.0",
                "7 -12.0 12.0 23.5 22.5 21.5 22.5 22.5 22.5 22.5",
                "11 56.5925 22.6934 22.1 21.1 18.4532 18.0 18.0 18.0 18.4532",
                "17 18.3245 -38.8525 22.3 21.3 18.8095 18.2 18.2 18.2 18.8095",
                "31 26.5704 -39.9680 22.5 21.5 19.3712 18.8 18.8 18.8 19.3712",
                "47 -18.9286 -10.8277 22.8 21.8 19.8418 19.3 19.3 19.3 19.8418",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("clgal_cat.txt").write_text(
        "\n".join(
            [
                "1 -9.0 1.0 21.0 20.0 18.0 18.0 18.0 18.0 18.0",
                "2 2.0 3.0 22.0 21.0 19.0 19.0 19.0 19.0 19.0",
                "3 -4.0 -4.0 23.0 22.0 20.0 20.0 20.0 20.0 20.0",
                "4 -5.0 5.0 24.0 23.0 20.5 20.5 20.5 20.5 20.5",
                "5 -6.0 6.0 25.0 24.0 21.0 21.0 21.0 21.0 21.0",
                "6 -7.0 7.0 25.5 24.5 21.5 21.5 21.5 21.5 21.5",
                "7 -7.5 7.5 26.0 25.0 22.0 22.0 22.0 22.0 22.0",
                "8 -8.0 8.0 26.5 25.5 23.0 22.5 22.5 22.5 22.5",
                "9 -9.0 9.0 27.0 26.0 23.5 23.0 23.0 23.0 23.0",
                "60 1.5 1.1 25.0 24.0 22.0 21.0 21.0 21.0 20.6",
                "10 ************ -4.0 23.0 22.0 20.0 20.0 20.0 20.0 20.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hera.joinpath("galcat2.txt").write_text(
        "\n".join(
            [
                "#filter: F814W; z=0.5; mstar=19.82",
                "1 9.2 1.1 0.000250 0.000125 35.0 18.0 0.0",
                "2 -22.0 33.0 0.000300 0.000200 -15.0 19.0 0.0",
                "3 44.0 -4.5 0.000100 0.000090 5.0 20.0 0.0",
                "4 5.1 5.1 0.000400 0.000300 60.0 20.5 0.0",
                "999 6.1 6.1 0.000500 0.000400 70.0 21.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for cluster_dir in (ares, hera):
        for band in ("f435w", "f606w", "f814w"):
            cluster_dir.joinpath(f"simulation_hst_{band}.fits").write_bytes(b"fixture")
    return published


def _limit_block(par_text: str, component_id: str) -> str:
    return par_text.split(f"limit {component_id}", maxsplit=1)[1].split("\n    end", maxsplit=1)[0]


def _limit_line(par_text: str, component_id: str, field_name: str) -> str:
    for line in _limit_block(par_text, component_id).splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field_name} "):
            return stripped
    raise AssertionError(f"missing {field_name} limit for {component_id}")


def test_resolve_source_root_uses_default_published_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    monkeypatch.setattr(builder, "DEFAULT_SOURCE_ROOT", source)

    assert builder._resolve_source_root() == source


def test_apply_matched_shapes_requires_same_id_within_tolerance() -> None:
    members = [
        builder.MemberRow("1", 0.0, 0.0, 1.0, 1.0, 0.0, 18.0, 17.0, 17.0, 1.0, 1.0),
        builder.MemberRow("2", 10.0, 0.0, 1.0, 1.0, 0.0, 19.0, 18.0, 18.0, 1.0, 1.0),
    ]
    shapes = [
        builder.MemberShape("1", 0.2, 0.0, 1.8, 1.2, 35.0),
        builder.MemberShape("2", 10.31, 0.0, 2.0, 1.4, -15.0),
    ]

    matched = builder._apply_matched_shapes(members, shapes, tolerance_arcsec=0.3)

    assert matched[0].a_axis == pytest.approx(1.8)
    assert matched[0].b_axis == pytest.approx(1.2)
    assert matched[0].theta_deg == pytest.approx(35.0)
    assert matched[1].a_axis == pytest.approx(1.0)
    assert matched[1].b_axis == pytest.approx(1.0)


def test_render_converts_catalogs_and_writes_loadable_pars(tmp_path: Path) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    output = tmp_path / "data" / "ff_sims"

    rows = builder.render(source_root=source, output_dir=output)

    assert {row["cluster_key"] for row in rows} == {"ares", "hera"}
    assert {row["cluster_key"]: row["n_skipped_members"] for row in rows} == {"ares": 0, "hera": 6}
    assert {row["cluster_key"]: row["n_members"] for row in rows} == {"ares": 334, "hera": 334}
    assert {row["cluster_key"]: row["n_explicit_galaxies"] for row in rows} == {"ares": 0, "hera": 0}
    assert {row["cluster_key"]: row["explicit_galaxy_ids"] for row in rows} == {
        "ares": "",
        "hera": "",
    }
    assert {row["cluster_key"]: row["member_selection"] for row in rows} == {
        "ares": "F814W<24.00",
        "hera": "F814W<24.00",
    }
    assert {row["cluster_key"]: row["scaling_band"] for row in rows} == {"ares": "F160W", "hera": "F160W"}
    assert {row["cluster_key"]: row["n_images"] for row in rows} == {"ares": 242, "hera": 65}
    assert {row["cluster_key"]: row["n_image_families"] for row in rows} == {"ares": 85, "hera": 19}
    assert {row["cluster_key"]: row["source_dir"] for row in rows} == {
        "ares": str(source / "ares"),
        "hera": str(source / "hera"),
    }
    assert not (output / "ff_sims_manifest.csv").exists()
    ares_image_lines = (output / "ares" / "ares_obs_arcs.cat").read_text(encoding="utf-8").splitlines()
    assert ares_image_lines[0] == "#REFERENCE 3"
    assert ares_image_lines[1].split()[:4] == ["1.a", "-10.00000000", "-2.00000000", "0.3734"]
    assert ares_image_lines[2].split()[0] == "1.b"
    assert not list((output / "ares").glob("*.fits"))
    assert not list((output / "hera").glob("*.fits"))

    ares_members = (output / "ares" / "ares_cluster_members_potfile.cat").read_text(encoding="utf-8").splitlines()
    assert ares_members[1] == "# FF-SIMS scaling-law member potfile generated from clgal_cat.txt."
    assert not any("modeled explicitly as G" in line for line in ares_members)
    active_ares_member_lines = [line for line in ares_members if not line.startswith("#")]
    assert len(active_ares_member_lines) == 11
    assert [line.split()[0] for line in active_ares_member_lines] == [
        "1",
        "11",
        "17",
        "2",
        "3",
        "31",
        "4",
        "5",
        "47",
        "6",
        "7",
    ]
    active_ares_ids = {line.split()[0] for line in active_ares_member_lines}
    assert {"1", "2", "3", "4", "5", "47"} <= active_ares_ids
    assert {"11", "17", "31"} <= active_ares_ids
    assert active_ares_member_lines[0].split()[:4] == ["1", "-5.00000000", "6.00000000", "1.00000000"]
    assert active_ares_member_lines[0].split()[6] == "18.000000"
    assert all(len(line.split()) == 9 for line in active_ares_member_lines)
    assert all(line.split()[3:6] == ["1.00000000", "1.00000000", "0.0000"] for line in active_ares_member_lines)
    assert active_ares_member_lines[0].split()[8] == "1.000000"

    hera_members = (output / "hera" / "hera_cluster_members_potfile.cat").read_text(encoding="utf-8").splitlines()
    assert not any("modeled explicitly as G" in line for line in hera_members)
    active_hera_member_lines = [line for line in hera_members if not line.startswith("#")]
    assert len(active_hera_member_lines) == 10
    assert [line.split()[0] for line in active_hera_member_lines] == ["1", "2", "3", "4", "60", "5", "6", "7", "8", "9"]
    assert {"1", "2", "3", "4", "5", "6"} <= {line.split()[0] for line in active_hera_member_lines}
    assert active_hera_member_lines[0].split()[:6] == [
        "1",
        "9.00000000",
        "1.00000000",
        "0.90000000",
        "0.45000000",
        "35.0000",
    ]
    assert active_hera_member_lines[1].split()[:6] == [
        "2",
        "-2.00000000",
        "3.00000000",
        "1.00000000",
        "1.00000000",
        "0.0000",
    ]
    assert active_hera_member_lines[4].split()[:6] == [
        "60",
        "-1.50000000",
        "1.10000000",
        "1.00000000",
        "1.00000000",
        "0.0000",
    ]
    assert all(len(line.split()) == 9 for line in active_hera_member_lines)
    assert active_hera_member_lines[4].split()[8] == "2.000000"

    ares_par = output / "ares" / "ares_lenscluster.par"
    parsed, potentials_df, images_df, arcs_df, potentials_with_priors = load_best_par(ares_par)
    assert len(potentials_df) == 2
    assert len(images_df) == 3
    assert len(arcs_df) == 0
    assert images_df["family_id"].nunique() == 2
    assert len(parsed["potfiles"][0]["catalog_df"]) == 11
    assert {item["id"] for item in potentials_with_priors} == {"O1", "O2"}
    assert parsed["potfiles"][0]["mag0"] == pytest.approx(18.5)
    assert parsed["potfiles"][0]["sigma_nominal"] == pytest.approx(100.0)
    assert parsed["potfiles"][0]["cutkpc_nominal"] == pytest.approx(270.0)
    ares_par_text = ares_par.read_text(encoding="utf-8")
    assert "two smooth dPIE halos, member scaling, and optional external shear" in ares_par_text
    assert "explicit bright galaxies" not in ares_par_text
    assert "Explicit galaxies are initialized" not in ares_par_text
    assert "Bright galaxies are explicit" not in ares_par_text
    assert "# O1 is the first smooth dPIE clump, anchored on clgal_cat.txt member 2." in ares_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 3." in ares_par_text
    assert _limit_line(ares_par_text, "O1", "x_centre") == "x_centre 1 15.00000000 25.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "y_centre") == "y_centre 1 -37.00000000 -27.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "x_centre") == "x_centre 1 -45.00000000 -35.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "y_centre") == "y_centre 1 35.00000000 45.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "ellipticite") == "ellipticite 1 0.00000000 0.50000000 0.02000000"
    assert _limit_line(ares_par_text, "O2", "ellipticite") == "ellipticite 1 0.00000000 0.50000000 0.02000000"
    assert _limit_line(ares_par_text, "O1", "core_radius") == "core_radius 1 5.00000000 60.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O2", "core_radius") == "core_radius 1 5.00000000 45.00000000 0.10000000"
    assert _limit_line(ares_par_text, "O1", "v_disp") == "v_disp 9 950.00000000 250.00000000 500.00000000 1800.00000000"
    assert _limit_line(ares_par_text, "O2", "v_disp") == "v_disp 9 950.00000000 175.00000000 600.00000000 1400.00000000"
    assert "sigma 9 100.00000000 15.00000000 70.00000000 500.00000000" in ares_par_text
    assert "cutkpc 9 270.00000000 35.00000000 160.00000000 800.00000000" in ares_par_text
    assert "\n    H0 70.40000000\n    omega 0.27200000\n    lambda 0.72800000" in ares_par_text
    assert parsed["cosmologie"]["H0"] == pytest.approx(70.4)
    assert parsed["cosmologie"]["omega"] == pytest.approx(0.272)
    assert parsed["cosmologie"]["lambda"] == pytest.approx(0.728)
    ares_cosmo = cosmology_config_from_parsed(parsed)
    assert ares_cosmo["H0"] == pytest.approx(70.4)
    assert ares_cosmo["Om0"] == pytest.approx(0.272)
    assert ares_cosmo["Ode0"] == pytest.approx(0.728)
    assert "vdslope" not in ares_par_text
    assert "slope   " not in ares_par_text
    assert "potentiel S1" not in ares_par_text
    assert "profil 14" not in ares_par_text
    assert "potentiel G" not in ares_par_text
    ares_specs, _ares_assignments, _ares_lens_models = build_halo_parameter_specs(potentials_with_priors, softening_length_kpc=0.0)
    ares_sample_names = {spec.sample_name for spec in ares_specs}
    assert {"O1_x_centre", "O1_y_centre", "O2_x_centre", "O2_y_centre"} <= ares_sample_names
    assert not any(sample_name.startswith("G") for sample_name in ares_sample_names)
    ares_scaling_specs, _ares_scaling_indices, _ares_scaling_models = build_scaling_parameter_specs(parsed["potfiles"], ScalingModelConfig())
    assert {"alpha_sigma", "beta_radius"} <= {spec.field for spec in ares_scaling_specs}
    assert {"vdslope", "slope", "gamma_ml"}.isdisjoint({spec.field for spec in ares_scaling_specs})

    hera_par = output / "hera" / "hera_lenscluster.par"
    _hera_parsed, _hera_potentials_df, _hera_images_df, _hera_arcs_df, hera_potentials_with_priors = load_best_par(hera_par)
    hera_par_text = hera_par.read_text(encoding="utf-8")
    assert "two smooth dPIE halos, member scaling, and optional external shear" in hera_par_text
    assert "explicit bright galaxies" not in hera_par_text
    assert "Explicit galaxies are initialized" not in hera_par_text
    assert "Bright galaxies are explicit" not in hera_par_text
    assert "# O1 is the first smooth dPIE clump, anchored on clgal_cat.txt member 1." in hera_par_text
    assert "# O2 is the second smooth dPIE clump, anchored on clgal_cat.txt member 2." in hera_par_text
    hera_o1_block = hera_par_text.split("potentiel O1", maxsplit=1)[1].split("potentiel O2", maxsplit=1)[0]
    hera_o2_block = hera_par_text.split("potentiel O2", maxsplit=1)[1].split("potfile", maxsplit=1)[0]
    assert "\n    x_centre 9.00000000\n    y_centre 1.00000000" in hera_o1_block
    assert "\n    core_radius 8.00000000" in hera_o1_block
    assert "\n    angle_pos 30.00000000" in hera_o1_block
    assert "\n    v_disp 800.00000000" in hera_o1_block
    assert _limit_line(hera_par_text, "O1", "x_centre") == "x_centre 1 4.00000000 14.00000000 0.10000000"
    assert _limit_line(hera_par_text, "O1", "y_centre") == "y_centre 1 -4.00000000 6.00000000 0.10000000"
    assert "\n    x_centre -2.00000000\n    y_centre 3.00000000" in hera_o2_block
    assert "\n    core_radius 5.00000000" in hera_o2_block
    assert "\n    angle_pos 24.00000000" in hera_o2_block
    assert "\n    v_disp 700.00000000" in hera_o2_block
    assert _limit_line(hera_par_text, "O2", "x_centre") == "x_centre 1 -7.00000000 3.00000000 0.10000000"
    assert _limit_line(hera_par_text, "O2", "y_centre") == "y_centre 1 -2.00000000 8.00000000 0.10000000"
    assert "potentiel G" not in hera_par_text
    assert "limit G" not in hera_par_text
    assert "potentiel S1" not in hera_par_text
    assert "profil 14" not in hera_par_text
    assert "\n    gamma 0.04000000\n    angle_pos 40.00000000" not in hera_par_text
    assert hera_par_text.count("core_radius 1 2.00000000 15.00000000 0.10000000") == 2
    assert "sigma 9 96.70000000 40.00000000 30.00000000 250.00000000" in hera_par_text
    assert "cutkpc 9 33.00000000 25.00000000 3.00000000 250.00000000" in hera_par_text
    assert "\n    H0 72.00000000\n    omega 0.24000000\n    lambda 0.76000000" in hera_par_text
    assert _hera_parsed["cosmologie"]["H0"] == pytest.approx(72.0)
    assert _hera_parsed["cosmologie"]["omega"] == pytest.approx(0.24)
    assert _hera_parsed["cosmologie"]["lambda"] == pytest.approx(0.76)
    hera_cosmo = cosmology_config_from_parsed(_hera_parsed)
    assert hera_cosmo["H0"] == pytest.approx(72.0)
    assert hera_cosmo["Om0"] == pytest.approx(0.24)
    assert hera_cosmo["Ode0"] == pytest.approx(0.76)
    assert "vdslope" not in hera_par_text
    assert "slope   " not in hera_par_text
    assert {item["id"] for item in hera_potentials_with_priors} == {"O1", "O2"}
    assert len(_hera_parsed["potfiles"][0]["catalog_df"]) == 10
    hera_specs, _hera_assignments, _hera_lens_models = build_halo_parameter_specs(hera_potentials_with_priors, softening_length_kpc=0.0)
    hera_sample_names = {spec.sample_name for spec in hera_specs}
    assert {"O1_x_centre", "O1_y_centre", "O2_x_centre", "O2_y_centre"} <= hera_sample_names
    hera_scaling_specs, _hera_scaling_indices, _hera_scaling_models = build_scaling_parameter_specs(_hera_parsed["potfiles"], ScalingModelConfig())
    assert {"alpha_sigma", "beta_radius"} <= {spec.field for spec in hera_scaling_specs}
    assert {"vdslope", "slope", "gamma_ml"}.isdisjoint({spec.field for spec in hera_scaling_specs})
    assert not (
        {
            "G1_x_centre",
            "G1_y_centre",
            "G2_x_centre",
            "G2_y_centre",
            "G3_x_centre",
            "G3_y_centre",
            "G4_x_centre",
            "G4_y_centre",
            "G5_x_centre",
            "G5_y_centre",
            "G6_x_centre",
            "G6_y_centre",
        }
        & hera_sample_names
    )


def test_validate_outputs_reports_generated_counts(tmp_path: Path) -> None:
    source = _write_ff_sims_fixture(tmp_path / "source")
    output = tmp_path / "data" / "ff_sims"
    builder.render(source_root=source, output_dir=output, clusters=["ares", "hera"])

    rows = builder.validate_outputs(output, clusters=["ares", "hera"])

    assert rows == [
        {
            "cluster_key": "ares",
            "n_potentials": 2,
            "n_images": 3,
            "n_image_families": 2,
            "n_potfiles": 1,
            "n_members": 11,
            "n_prior_components": 2,
        },
        {
            "cluster_key": "hera",
            "n_potentials": 2,
            "n_images": 2,
            "n_image_families": 1,
            "n_potfiles": 1,
            "n_members": 10,
            "n_prior_components": 2,
        }
    ]
