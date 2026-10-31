from types import SimpleNamespace

import numpy as np
import pandas as pd

from lenscluster.mass_figure import arc_direction_table, extra_arc_direction_table, grid_pixels


def test_extra_directions_use_positions_and_family_redshifts():
    calls = []
    def fields(xy, z):
        calls.append((xy.copy(), z))
        a = np.diag([-.05, .8]) if z == 2 else np.diag([.9, .1])
        return np.zeros_like(xy), np.broadcast_to(np.eye(2) - a, (len(xy), 2, 2))
    evaluator = SimpleNamespace(physical=lambda p: p, total_fields_at=fields,
        packed_state_for_z=lambda p, z: z,
        state=SimpleNamespace(family_data=[SimpleNamespace(family_id="0192", z_source=2),
                                          SimpleNamespace(family_id="b", z_source=3)]))
    extra = pd.DataFrame(dict(family_id=["b", "0192"], x_model_arcsec=[4., 5.], y_model_arcsec=[6., 7.]))
    result = extra_arc_direction_table(evaluator, np.zeros(1), extra)
    np.testing.assert_allclose(result[["e_arc_x", "e_arc_y"]], [[0, 1], [1, 0]])
    np.testing.assert_allclose(result.s_arc, [.1, .05])
    np.testing.assert_allclose(calls[0][0], [[4, 6]])
    assert [c[1] for c in calls] == [3, 2]
    assert extra_arc_direction_table(evaluator, np.zeros(1), pd.DataFrame()).empty


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


def test_covariance_ellipse_contours_and_identity(monkeypatch):
    from lenscluster import covariance_cutouts
    from lenscluster.mass_figure import covariance_ellipse_curves
    a = np.array([[[.1,.03],[.03,.8]], [[-.2,0.],[0.,.6]]])
    c = np.array([[[.04,.002],[.002,.01]], [[.01,0.],[0.,.02]]])
    monkeypatch.setattr(covariance_cutouts, 'best_geometry', lambda *args: dict(a=a,c=c,valid=np.array([True,True])))
    evaluator = SimpleNamespace(state=SimpleNamespace(images=SimpleNamespace(image_labels=['a','b'])))
    catalog = pd.DataFrame(dict(image_label=['b','a'], x_obs_arcsec=[3.,5.], y_obs_arcsec=[7.,9.]))
    curves, table = covariance_ellipse_curves(evaluator, np.zeros(1), catalog)
    assert table.image_label.tolist() == ['b','a']
    for curve, center, i in zip(curves, [[3.,7.],[5.,9.]], [1,0]):
        source = (curve-center) @ a[i].T
        q = np.einsum('ni,ij,nj->n',source,np.linalg.inv(c[i]),source)
        np.testing.assert_allclose(q,1.,rtol=0.,atol=2e-14)
