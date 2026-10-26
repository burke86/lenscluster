from types import SimpleNamespace

import numpy as np
import pandas as pd

from lenscluster.mass_figure import arc_direction_table, grid_pixels


def test_arc_direction_uses_right_singular_vector_and_catalog_identity():
    angle = .61
    v = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    u = np.array([[0., -1.], [1., 0.]])
    matrices = np.array([u @ np.diag([.1, .8]) @ v.T, np.diag([-.05, .9]), np.eye(2)])
    calls = []
    def fields(theta, **kwargs):
        calls.append(kwargs)
        return {"jacobian": matrices}
    evaluator = SimpleNamespace(physical=lambda p: p, fields=fields,
        state=SimpleNamespace(images=SimpleNamespace(image_labels=["a", "b", "c"])))
    result = arc_direction_table(evaluator, np.zeros(1), pd.DataFrame({"image_label": ["b", "c", "a"]}))
    assert calls == [{"force_exact": True, "derivative": False}]
    assert result.image_label.tolist() == ["b", "c", "a"]
    directions = result[["e_arc_x", "e_arc_y"]].to_numpy()
    np.testing.assert_allclose(abs(directions[2] @ v[:, 0]), 1, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(matrices[0] @ directions[2]), .1, atol=1e-14)
    np.testing.assert_allclose(directions[0], [1, 0])
    assert not result.direction_defined.iloc[1]
    assert np.isnan(directions[1]).all()


def test_stick_endpoints_follow_display_axis_reflections():
    geometry = {"resolution": 100, "xlim": [10., -10.], "ylim": [-10., 10.]}
    direction = np.array([.6, .8])
    ends = np.array([[-1.5], [1.5]]) * direction
    x, y = grid_pixels(geometry, ends[:, 0], ends[:, 1])
    np.testing.assert_allclose([np.diff(x)[0], np.diff(y)[0]], [-9., 12.])
