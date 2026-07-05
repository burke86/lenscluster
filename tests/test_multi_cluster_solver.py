from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.model import ParameterSpec
from lenscluster import plotting
from lenscluster import multi_cluster_solver as multi


def _spec(
    sample_name: str,
    *,
    name: str | None = None,
    component_family: str = "large",
    parent_sample_name: str | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name=name or sample_name,
        sample_name=sample_name,
        potential_id="p1",
        profile_type=81,
        field=sample_name,
        prior_kind="uniform",
        lower=0.0,
        upper=1.0,
        step=0.1,
        component_family=component_family,
        parent_sample_name=parent_sample_name,
    )


def _cosmology_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="cosmology.Om0",
            sample_name="cosmology_Om0",
            potential_id="cosmology",
            profile_type=0,
            field="Om0",
            prior_kind="uniform",
            lower=0.05,
            upper=0.6,
            step=0.01,
            component_family="cosmology",
        ),
        ParameterSpec(
            name="cosmology.w0",
            sample_name="cosmology_w0",
            potential_id="cosmology",
            profile_type=0,
            field="w0",
            prior_kind="uniform",
            lower=-2.0,
            upper=-0.3,
            step=0.05,
            component_family="cosmology",
        ),
    ]


def _context(
    key: str,
    specs: list[ParameterSpec],
    *,
    h0: float = 70.0,
    evaluator: object | None = None,
    svi_init_values: dict[str, float] | None = None,
):
    return multi.MultiClusterContext(
        cluster=multi.ClusterInput(key=key, par_path=Path(f"{key}.par"), warm_run_dir=Path(f"{key}_warm")),
        warm_stage=multi.WarmStageResolution(
            stage_name="stage2_joint",
            stage_artifacts_dir=Path(f"{key}_warm/stage2_joint/artifacts"),
            stage1_artifacts_dir=Path(f"{key}_warm/stage1_large_only/artifacts"),
        ),
        state=SimpleNamespace(
            parameter_specs=specs,
            cosmo_config={"class": "FlatLambdaCDM", "H0": h0, "Om0": 0.3, "w0": -1.0},
            svi_init_values=svi_init_values,
        ),
        evaluator=evaluator or SimpleNamespace(),
    )


def _touch_artifact(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "plot_bundle.h5").write_bytes(b"placeholder")


def _write_minimal_plot_bundle(
    path: Path,
    specs: list[ParameterSpec],
    samples: np.ndarray,
    best_fit: np.ndarray,
    clusters: list[dict[str, object]],
    grouped_samples: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["kind"] = "multi_cluster_joint_cosmology"
        posterior = handle.create_group("posterior")
        posterior.create_dataset("samples", data=np.asarray(samples, dtype=float))
        posterior.create_dataset("best_fit", data=np.asarray(best_fit, dtype=float))
        if grouped_samples is not None:
            posterior.create_dataset("grouped_samples", data=np.asarray(grouped_samples, dtype=float))
        state = handle.create_group("state")
        state.create_dataset(
            "multi_state_meta_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "run_name": "joint",
                        "parameter_specs": [spec.__dict__ for spec in specs],
                        "clusters": clusters,
                    }
                )
            ),
        )


def test_parse_args_accepts_repeated_cluster_triples_and_cosmology_init() -> None:
    args = multi._parse_args(
        [
            "--cluster",
            "a2744",
            "a2744.par",
            "runs/a2744",
            "--cluster",
            "m0416",
            "m0416.par",
            "runs/m0416",
            "--output-dir",
            "joint",
            "--run-name",
            "fit",
            "--cosmology-init-om0",
            "0.25",
            "--cosmology-init-w0",
            "-0.8",
            "--image-plane-scatter-floor-arcsec",
            "0.05",
            "--image-plane-scatter-prior",
            "lognormal",
            "--image-plane-scatter-prior-median-arcsec",
            "0.25",
            "--image-plane-scatter-prior-log-sigma",
            "0.4",
            "--fix-image-sigma-int-arcsec",
            "0.35",
            "--sampling-engine",
            "refreshing_surrogate_flat",
        ]
    )

    assert [item.key for item in args.cluster_inputs] == ["a2744", "m0416"]
    assert args.cluster_inputs[0].par_path == Path("a2744.par")
    assert args.cluster_inputs[1].warm_run_dir == Path("runs/m0416")
    assert args.sample_likelihood_mode == "linearized-forward-beta-image-plane"
    assert args.cosmology_init_om0 == 0.25
    assert args.cosmology_init_w0 == -0.8
    assert args.image_plane_scatter_floor_arcsec == pytest.approx(0.05)
    assert args.image_plane_scatter_prior == "lognormal"
    assert args.image_plane_scatter_prior_median_arcsec == pytest.approx(0.25)
    assert args.image_plane_scatter_prior_log_sigma == pytest.approx(0.4)
    assert args.fix_image_sigma_int_arcsec == pytest.approx(0.35)
    assert args.sampling_engine == "refreshing_surrogate_flat"
    assert args.dense_mass == "full"
    assert not hasattr(args, "validate_top_k_families")
    assert not hasattr(args, "validation_approx")


def test_parse_args_accepts_disabled_refresh_every() -> None:
    args = multi._parse_args(
        [
            "--cluster",
            "a2744",
            "a2744.par",
            "runs/a2744",
            "--cluster",
            "m0416",
            "m0416.par",
            "runs/m0416",
            "--refresh-every",
            "None",
            "1000",
        ]
    )

    assert args.refresh_every == [None, 1000]


def test_parse_args_dense_mass_choices() -> None:
    base = [
        "--cluster",
        "a2744",
        "a2744.par",
        "runs/a2744",
        "--cluster",
        "m0416",
        "m0416.par",
        "runs/m0416",
    ]

    assert multi._parse_args(base).dense_mass == "full"
    assert multi._parse_args([*base, "--dense-mass", "full"]).dense_mass == "full"
    assert multi._parse_args([*base, "--dense-mass", "diagonal"]).dense_mass == "diagonal"
    with pytest.raises(SystemExit):
        multi._parse_args([*base, "--dense-mass", "structured"])

    for invalid_args in (
        ["--no-dense-mass"],
        ["--dense-mass"],
        ["--dense-mass-structure", "full"],
    ):
        with pytest.raises(SystemExit):
            multi._parse_args([*base, *invalid_args])


def test_parse_args_rejects_removed_potfile_mass_size_reparam_flag() -> None:
    base = [
        "--cluster",
        "a2744",
        "a2744.par",
        "runs/a2744",
        "--cluster",
        "m0416",
        "m0416.par",
        "runs/m0416",
    ]

    with pytest.raises(SystemExit):
        multi._parse_args([*base, "--potfile-mass-size-reparam"])


def test_parse_args_rejects_blocked_linearized_mode() -> None:
    with pytest.raises(SystemExit):
        multi._parse_args(
            [
                "--cluster",
                "a2744",
                "a2744.par",
                "runs/a2744",
                "--cluster",
                "m0416",
                "m0416.par",
                "runs/m0416",
                "--image-plane-mode",
                "linearized-forward-beta-blocked-image-plane",
            ]
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--validate-top-k-families",
        "--validation-approx",
        "--scaling-scatter-fields",
        "--scaling-scatter-max",
    ],
)
def test_parse_args_rejects_removed_main_validation_flags(flag: str) -> None:
    with pytest.raises(SystemExit):
        multi._parse_args(
            [
                "--cluster",
                "a2744",
                "a2744.par",
                "runs/a2744",
                flag,
                "1",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--image-plane-scatter-floor-arcsec", "0", "image-plane-scatter-floor"),
        ("--image-plane-scatter-floor-arcsec", "-0.1", "image-plane-scatter-floor"),
        ("--image-plane-scatter-floor-arcsec", "nan", "image-plane-scatter-floor"),
        ("--image-plane-scatter-prior-median-arcsec", "0", "image-plane-scatter-prior-median"),
        ("--image-plane-scatter-prior-median-arcsec", "nan", "image-plane-scatter-prior-median"),
        ("--image-plane-scatter-prior-log-sigma", "0", "image-plane-scatter-prior-log-sigma"),
        ("--image-plane-scatter-prior-log-sigma", "nan", "image-plane-scatter-prior-log-sigma"),
        ("--fix-image-sigma-int-arcsec", "-0.1", "fix-image-sigma-int"),
        ("--fix-image-sigma-int-arcsec", "nan", "fix-image-sigma-int"),
    ],
)
def test_parse_args_rejects_invalid_image_plane_scatter_controls(
    flag: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        multi._parse_args(
            [
                "--cluster",
                "a2744",
                "a2744.par",
                "runs/a2744",
                "--output-dir",
                "joint",
                flag,
                value,
            ]
        )


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            ["--image-plane-scatter-floor-arcsec", "0.5", "--image-plane-scatter-upper-arcsec", "0.5"],
            "image-plane-scatter-upper",
        ),
        (
            [
                "--image-plane-scatter-prior",
                "lognormal",
                "--image-plane-scatter-floor-arcsec",
                "0.1",
                "--image-plane-scatter-upper-arcsec",
                "0.5",
                "--image-plane-scatter-prior-median-arcsec",
                "0.05",
            ],
            "image-plane-scatter-prior-median",
        ),
        (
            [
                "--image-plane-scatter-prior",
                "lognormal",
                "--image-plane-scatter-floor-arcsec",
                "0.1",
                "--image-plane-scatter-upper-arcsec",
                "0.5",
                "--image-plane-scatter-prior-median-arcsec",
                "0.6",
            ],
            "image-plane-scatter-prior-median",
        ),
    ],
)
def test_parse_args_rejects_invalid_image_plane_scatter_support(
    extra_args: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        multi._parse_args(
            [
                "--cluster",
                "a2744",
                "a2744.par",
                "runs/a2744",
                "--output-dir",
                "joint",
                *extra_args,
            ]
        )


def test_resolve_warm_stage_auto_prefers_stage2_free_source_forward_fit(tmp_path: Path) -> None:
    par_path = tmp_path / "input.par"
    par_path.write_text("runmode\n", encoding="utf-8")
    warm_run = tmp_path / "warm"
    _touch_artifact(warm_run / "stage1_backprojected_centroid_fit" / "artifacts")
    _touch_artifact(warm_run / "stage2_free_source_forward_fit" / "artifacts")

    resolved = multi._resolve_warm_stage(
        multi.ClusterInput("c1", par_path, warm_run),
        "auto",
    )

    assert resolved.stage_name == "stage2_free_source_forward_fit"
    assert resolved.stage1_artifacts_dir == warm_run / "stage1_backprojected_centroid_fit" / "artifacts"


def test_resolve_warm_stage_auto_falls_back_to_stage1_backprojected_centroid_fit(tmp_path: Path) -> None:
    par_path = tmp_path / "input.par"
    par_path.write_text("runmode\n", encoding="utf-8")
    warm_run = tmp_path / "warm"
    _touch_artifact(warm_run / "stage1_backprojected_centroid_fit" / "artifacts")

    resolved = multi._resolve_warm_stage(
        multi.ClusterInput("c1", par_path, warm_run),
        "auto",
    )

    assert resolved.stage_name == "stage1_backprojected_centroid_fit"


def test_resolve_warm_stage_requires_stage1_artifacts(tmp_path: Path) -> None:
    par_path = tmp_path / "input.par"
    par_path.write_text("runmode\n", encoding="utf-8")
    warm_run = tmp_path / "warm"
    _touch_artifact(warm_run / "stage2_free_source_forward_fit" / "artifacts")

    with pytest.raises(FileNotFoundError, match="stage1 warm artifacts"):
        multi._resolve_warm_stage(multi.ClusterInput("c1", par_path, warm_run), "auto")


def test_prefix_parameter_spec_updates_parent_sample_name() -> None:
    spec = _spec("member_delta", parent_sample_name="member_scatter")

    prefixed = multi._prefix_parameter_spec("a2744", spec)

    assert prefixed.sample_name == "a2744__member_delta"
    assert prefixed.parent_sample_name == "a2744__member_scatter"
    assert prefixed.name == "a2744.member_delta"


def test_global_layout_prefixes_nuisance_and_shares_cosmology_once() -> None:
    c1 = _context("a2744", [_spec("mass"), *_cosmology_specs()])
    c2 = _context("m0416", [_spec("mass"), *_cosmology_specs()])

    global_specs = multi._build_global_parameter_layout([c1, c2])

    sample_names = [spec.sample_name for spec in global_specs]
    assert sample_names.count("cosmology_Om0") == 1
    assert sample_names.count("cosmology_w0") == 1
    assert "a2744__mass" in sample_names
    assert "m0416__mass" in sample_names
    assert c1.local_to_global_indices is not None
    assert c2.local_to_global_indices is not None
    assert c1.local_to_global_indices.shape == (3,)
    assert c2.local_to_global_indices.shape == (3,)


def test_global_init_values_use_shared_cosmology_starts_once() -> None:
    c1 = _context(
        "a2744",
        [_spec("mass"), *_cosmology_specs()],
        svi_init_values={
            "mass": 0.4,
            "cosmology_Om0": 0.25,
            "cosmology_w0": -0.8,
        },
    )
    c2 = _context(
        "m0416",
        [_spec("mass"), *_cosmology_specs()],
        svi_init_values={
            "mass": 0.6,
            "cosmology_Om0": 0.45,
            "cosmology_w0": -1.3,
        },
    )
    global_specs = multi._build_global_parameter_layout([c1, c2])

    init_values = multi._global_init_values(global_specs, [c1, c2])

    assert init_values["a2744__mass"] == 0.4
    assert init_values["m0416__mass"] == 0.6
    assert init_values["cosmology_Om0"] == 0.25
    assert init_values["cosmology_w0"] == -0.8
    assert "a2744__cosmology_Om0" not in init_values
    assert "m0416__cosmology_Om0" not in init_values


def test_cluster_corner_subset_excludes_sources_and_includes_shared_cosmology() -> None:
    c1 = _context(
        "a2744",
        [
            _spec("mass1"),
            _spec("source_1_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    c2 = _context(
        "m0416",
        [
            _spec("mass2"),
            _spec("source_2_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    global_specs = multi._build_global_parameter_layout([c1, c2])
    samples = np.arange(3 * len(global_specs), dtype=float).reshape(3, len(global_specs))
    best_fit = np.arange(len(global_specs), dtype=float) + 100.0

    subset_specs, subset_samples, subset_best = multi._cluster_corner_parameter_subset(
        c1,
        global_specs,
        samples,
        best_fit,
    )

    assert [spec.sample_name for spec in subset_specs] == [
        "a2744__mass1",
        "cosmology_Om0",
        "cosmology_w0",
    ]
    assert all(spec.component_family != "source_position" for spec in subset_specs)
    expected_indices = [0, 2, 3]
    np.testing.assert_allclose(subset_samples, samples[:, expected_indices])
    np.testing.assert_allclose(subset_best, best_fit[expected_indices])


def test_multi_cluster_corner_outputs_global_and_cluster_corner_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c1 = _context(
        "a2744",
        [
            _spec("mass1"),
            _spec("source_1_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    c2 = _context(
        "m0416",
        [
            _spec("mass2"),
            _spec("source_2_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    global_specs = multi._build_global_parameter_layout([c1, c2])
    state = multi.MultiClusterState(
        run_name="joint",
        parameter_specs=global_specs,
        svi_init_values=None,
        contexts=[c1, c2],
    )
    samples = np.arange(4 * len(global_specs), dtype=float).reshape(4, len(global_specs))
    best_fit = np.arange(len(global_specs), dtype=float) + 10.0
    calls: list[dict[str, object]] = []

    class FakeFig:
        def __init__(self, call: dict[str, object]) -> None:
            self.call = call

        def savefig(self, path: Path, **_kwargs: object) -> None:
            self.call["path"] = Path(path)

    class FakeCorner:
        def corner(self, corner_samples: np.ndarray, **kwargs: object) -> FakeFig:
            call: dict[str, object] = {
                "samples": np.asarray(corner_samples, dtype=float),
                "labels": list(kwargs.get("labels", [])),
                "plot_datapoints": bool(kwargs.get("plot_datapoints", False)),
            }
            calls.append(call)
            return FakeFig(call)

        def overplot_lines(self, *_args: object, **_kwargs: object) -> None:
            return None

        def overplot_points(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(plotting, "corner", FakeCorner())
    monkeypatch.setattr(plotting.plt, "close", lambda *_args, **_kwargs: None)

    multi._plot_multi_cluster_corners(tmp_path, state, samples, best_fit)

    paths = [call["path"].name for call in calls]
    assert paths == [
        "corner.pdf",
        "a2744_cluster_cosmology_corner.pdf",
        "m0416_cluster_cosmology_corner.pdf",
    ]
    assert [call["plot_datapoints"] for call in calls] == [False, False, False]
    assert calls[0]["labels"] == [
        "a2744.mass1",
        "cosmology.Om0",
        "cosmology.w0",
        "m0416.mass2",
    ]
    np.testing.assert_allclose(calls[0]["samples"], samples[:, [0, 2, 3, 4]])
    assert calls[1]["labels"] == ["a2744.mass1", "cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(calls[1]["samples"], samples[:, [0, 2, 3]])
    assert calls[2]["labels"] == ["m0416.mass2", "cosmology.Om0", "cosmology.w0"]
    np.testing.assert_allclose(calls[2]["samples"], samples[:, [4, 2, 3]])


def test_cosmology_prior_histograms_write_pdf(tmp_path: Path) -> None:
    samples = np.asarray(
        [
            [0.25, -1.2],
            [0.30, -1.0],
            [0.35, -0.8],
            [0.40, -0.7],
        ],
        dtype=float,
    )
    best_fit = np.asarray([0.32, -0.9], dtype=float)

    multi._plot_cosmology_prior_histograms(tmp_path, _cosmology_specs(), samples, best_fit)

    assert (tmp_path / "cosmology_prior_histograms.pdf").is_file()


def test_cosmology_prior_histograms_draws_prior_on_secondary_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    events: list[tuple[str, object]] = []

    class FakeAxis:
        def __init__(self, name: str) -> None:
            self.name = name
            self._handles: list[str] = []
            self._labels: list[str] = []

        def _record_label(self, label: object) -> None:
            if label is None:
                return
            self._handles.append(f"{self.name}:{label}")
            self._labels.append(str(label))

        def hist(self, *_args: object, density: bool, label: object = None, **_kwargs: object) -> None:
            events.append(("hist_density", density))
            self._record_label(label)

        def axvline(self, *_args: object, label: object = None, **_kwargs: object) -> None:
            self._record_label(label)

        def hlines(self, *_args: object, label: object = None, **_kwargs: object) -> None:
            events.append(("hlines_axis", self.name))
            self._record_label(label)

        def twinx(self) -> "FakeAxis":
            prior_axis = FakeAxis(f"{self.name}.prior")
            fake_fig.axes.append(prior_axis)
            events.append(("twinx", self.name))
            return prior_axis

        def set_ylim(self, lower: float, upper: float) -> None:
            events.append((f"{self.name}.ylim", (lower, upper)))

        def set_xlim(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_xlabel(self, *_args: object, **_kwargs: object) -> None:
            return None

        def set_ylabel(self, label: str) -> None:
            events.append((f"{self.name}.ylabel", label))

        def get_legend_handles_labels(self) -> tuple[list[str], list[str]]:
            return self._handles, self._labels

    class FakeFig:
        def __init__(self) -> None:
            self.axes: list[FakeAxis] = []

        def legend(self, _handles: list[str], labels: list[str], **_kwargs: object) -> None:
            events.append(("legend_labels", tuple(labels)))

        def tight_layout(self) -> None:
            return None

        def savefig(self, path: Path, **_kwargs: object) -> None:
            Path(path).write_text("pdf", encoding="utf-8")

    fake_fig = FakeFig()
    main_axes = [FakeAxis("main0"), FakeAxis("main1")]
    fake_fig.axes.extend(main_axes)

    def fake_subplots(*_args: object, **_kwargs: object) -> tuple[FakeFig, np.ndarray]:
        return fake_fig, np.asarray([main_axes], dtype=object)

    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    samples = np.asarray(
        [
            [0.25, -1.2],
            [0.30, -1.0],
            [0.35, -0.8],
            [0.40, -0.7],
        ],
        dtype=float,
    )
    best_fit = np.asarray([0.32, -0.9], dtype=float)

    multi._plot_cosmology_prior_histograms(tmp_path, _cosmology_specs(), samples, best_fit)

    assert (tmp_path / "cosmology_prior_histograms.pdf").is_file()
    assert ("hist_density", False) in events
    assert ("twinx", "main0") in events
    assert ("twinx", "main1") in events
    assert ("hlines_axis", "main0.prior") in events
    assert ("hlines_axis", "main1.prior") in events
    assert ("main0.ylabel", "posterior samples") in events
    assert ("main0.prior.ylabel", "prior density") in events
    assert any(
        event_name == "legend_labels" and "uniform prior density" in labels
        for event_name, labels in events
    )


def test_cosmology_grouped_subset_keeps_chain_draw_shape() -> None:
    specs = [_spec("mass"), *_cosmology_specs(), _spec("image_sigma_int", component_family="image_scatter")]
    grouped = np.arange(2 * 3 * len(specs), dtype=float).reshape(2, 3, len(specs))

    subset_specs, subset_grouped = multi._cosmology_grouped_subset(specs, grouped)

    assert [spec.sample_name for spec in subset_specs] == ["cosmology_Om0", "cosmology_w0"]
    assert subset_grouped is not None
    assert subset_grouped.shape == (2, 3, 2)
    np.testing.assert_allclose(subset_grouped, grouped[:, :, [1, 2]])


def test_cosmology_trace_plot_writes_pdf(tmp_path: Path) -> None:
    specs = [_spec("mass"), *_cosmology_specs()]
    grouped = np.asarray(
        [
            [[0.0, 0.25, -1.1], [0.0, 0.30, -1.0], [0.0, 0.35, -0.9]],
            [[0.0, 0.27, -1.2], [0.0, 0.32, -1.1], [0.0, 0.37, -1.0]],
        ],
        dtype=float,
    )

    multi._plot_cosmology_trace(tmp_path, specs, grouped)

    assert (tmp_path / "cosmology_trace_plot.pdf").is_file()


def test_cosmology_trace_plot_skips_missing_grouped_samples(tmp_path: Path) -> None:
    multi._plot_cosmology_trace(tmp_path, _cosmology_specs(), None)

    assert not (tmp_path / "cosmology_trace_plot.pdf").exists()


def test_load_plot_bundle_for_plots_rebuilds_minimal_state(tmp_path: Path) -> None:
    c1 = _context(
        "a2744",
        [
            _spec("mass1"),
            _spec("source_1_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    c2 = _context(
        "m0416",
        [
            _spec("mass2"),
            _spec("source_2_beta_x", component_family="source_position"),
            *_cosmology_specs(),
        ],
    )
    global_specs = multi._build_global_parameter_layout([c1, c2])
    samples = np.arange(4 * len(global_specs), dtype=float).reshape(4, len(global_specs))
    best_fit = np.arange(len(global_specs), dtype=float) + 20.0
    grouped = np.arange(2 * 2 * len(global_specs), dtype=float).reshape(2, 2, len(global_specs))
    bundle_path = tmp_path / "artifacts" / "plot_bundle.h5"
    _write_minimal_plot_bundle(
        bundle_path,
        global_specs,
        samples,
        best_fit,
        [
            {
                "key": "a2744",
                "par_path": "a2744.par",
                "warm_run_dir": "a2744_warm",
                "warm_stage": "stage3_image_plane",
                "warm_stage_artifacts_dir": "a2744_warm/stage3_image_plane/artifacts",
                "stage1_artifacts_dir": "a2744_warm/stage1_large_only/artifacts",
                "local_to_global_indices": c1.local_to_global_indices.tolist(),
            },
            {
                "key": "m0416",
                "par_path": "m0416.par",
                "warm_run_dir": "m0416_warm",
                "warm_stage": "stage3_image_plane",
                "warm_stage_artifacts_dir": "m0416_warm/stage3_image_plane/artifacts",
                "stage1_artifacts_dir": "m0416_warm/stage1_large_only/artifacts",
                "local_to_global_indices": c2.local_to_global_indices.tolist(),
            },
        ],
        grouped_samples=grouped,
    )

    state, loaded_samples, loaded_best, loaded_grouped, loaded_log_prob = multi._load_plot_bundle_for_plots(bundle_path)

    assert [context.cluster.key for context in state.contexts] == ["a2744", "m0416"]
    np.testing.assert_allclose(loaded_samples, samples)
    np.testing.assert_allclose(loaded_best, best_fit)
    np.testing.assert_allclose(loaded_grouped, grouped)
    assert loaded_log_prob is None
    subset_specs, subset_samples, subset_best = multi._cluster_corner_parameter_subset(
        state.contexts[1],
        state.parameter_specs,
        loaded_samples,
        loaded_best,
    )
    assert [spec.sample_name for spec in subset_specs] == [
        "m0416__mass2",
        "cosmology_Om0",
        "cosmology_w0",
    ]
    np.testing.assert_allclose(subset_samples, samples[:, [4, 2, 3]])
    np.testing.assert_allclose(subset_best, best_fit[[4, 2, 3]])


def test_load_plot_bundle_for_plots_requires_local_to_global_indices(tmp_path: Path) -> None:
    global_specs = [_spec("mass"), *_cosmology_specs()]
    bundle_path = tmp_path / "artifacts" / "plot_bundle.h5"
    _write_minimal_plot_bundle(
        bundle_path,
        global_specs,
        np.ones((3, len(global_specs)), dtype=float),
        np.ones((len(global_specs),), dtype=float),
        [{"key": "a2744", "par_path": "a2744.par", "warm_run_dir": "a2744_warm"}],
    )

    with pytest.raises(ValueError, match="local_to_global_indices"):
        multi._load_plot_bundle_for_plots(bundle_path)


def test_resume_complete_run_refreshes_plots_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "fit"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"placeholder")
    (run_dir / "tables").mkdir()
    (run_dir / "tables" / "joint_run_summary.json").write_text("{}", encoding="utf-8")
    refreshed: list[Path] = []
    monkeypatch.setattr(multi, "_rerender_plots_from_bundle", lambda _args, path: refreshed.append(Path(path)))
    monkeypatch.setattr(
        multi,
        "_build_multi_cluster_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume should not build state")),
    )
    args = SimpleNamespace(output_dir=str(tmp_path), run_name="fit", resume=True, skip_plots=False, quiet=True)

    result = multi._run(args)

    assert result == run_dir
    assert refreshed == [run_dir]


def test_rerender_plots_from_bundle_loads_grouped_samples_for_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [_spec("mass"), *_cosmology_specs()]
    samples = np.arange(4 * len(specs), dtype=float).reshape(4, len(specs))
    best_fit = np.arange(len(specs), dtype=float)
    grouped = np.arange(2 * 2 * len(specs), dtype=float).reshape(2, 2, len(specs))
    run_dir = tmp_path / "fit"
    _write_minimal_plot_bundle(
        run_dir / "artifacts" / "plot_bundle.h5",
        specs,
        samples,
        best_fit,
        [
            {
                "key": "a2744",
                "par_path": "a2744.par",
                "warm_run_dir": "a2744_warm",
                "local_to_global_indices": [0, 1, 2],
            }
        ],
        grouped_samples=grouped,
    )
    trace_calls: list[np.ndarray | None] = []
    monkeypatch.setattr(multi, "_plot_cosmology_corner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi, "_plot_cosmology_prior_histograms", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi, "_plot_multi_cluster_corners", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(multi, "_write_fallback_cosmology_plot_if_needed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        multi,
        "_plot_cosmology_trace",
        lambda _plot_dir, _specs, trace_grouped: trace_calls.append(trace_grouped),
    )

    multi._rerender_plots_from_bundle(SimpleNamespace(quiet=True), run_dir)

    assert len(trace_calls) == 1
    np.testing.assert_allclose(trace_calls[0], grouped)


def test_resume_complete_run_skip_plots_reuses_without_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "fit"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "plot_bundle.h5").write_bytes(b"placeholder")
    (run_dir / "tables").mkdir()
    (run_dir / "tables" / "joint_run_summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        multi,
        "_rerender_plots_from_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skip_plots should not rerender")),
    )
    monkeypatch.setattr(
        multi,
        "_build_multi_cluster_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume should not build state")),
    )
    args = SimpleNamespace(output_dir=str(tmp_path), run_name="fit", resume=True, skip_plots=True, quiet=True)

    result = multi._run(args)

    assert result == run_dir


def test_compatible_cosmologies_rejects_h0_mismatch() -> None:
    c1 = _context("a2744", _cosmology_specs(), h0=70.0)
    c2 = _context("m0416", _cosmology_specs(), h0=67.0)

    with pytest.raises(ValueError, match="fixed H0"):
        multi._validate_compatible_cosmologies([c1, c2])


class _FakeEvaluator:
    surrogate_enabled = False
    invalid_state_rejection_count = 0
    invalid_state_reason_counts: dict[str, int] = {}

    def __init__(self, offset: float) -> None:
        self.offset = float(offset)
        self._source_loglike_fn = lambda theta: jnp.sum(theta) + self.offset

    def source_loglike(self, theta: np.ndarray) -> float:
        return float(self._source_loglike_fn(jnp.asarray(theta, dtype=jnp.float64)))


def test_multi_cluster_loglike_injects_shared_cosmology_values() -> None:
    c1 = _context("a2744", [_spec("mass1"), *_cosmology_specs()], evaluator=_FakeEvaluator(1.0))
    c2 = _context("m0416", [_spec("mass2"), *_cosmology_specs()], evaluator=_FakeEvaluator(2.0))
    global_specs = multi._build_global_parameter_layout([c1, c2])
    evaluator = multi.MultiClusterJAXEvaluator([c1, c2])

    values = {spec.sample_name: idx + 1.0 for idx, spec in enumerate(global_specs)}
    theta = np.asarray([values[spec.sample_name] for spec in global_specs], dtype=float)
    loglike = evaluator.source_loglike(theta)

    expected_c1 = values["a2744__mass1"] + values["cosmology_Om0"] + values["cosmology_w0"] + 1.0
    expected_c2 = values["m0416__mass2"] + values["cosmology_Om0"] + values["cosmology_w0"] + 2.0
    assert loglike == pytest.approx(expected_c1 + expected_c2)
