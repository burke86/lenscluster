"""Check that the normal GP output path includes both shared-field figures."""
from types import SimpleNamespace
from unittest.mock import Mock

from lenscluster import gp_plots, field_plots


def test_standard_gp_outputs_include_field_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(gp_plots, 'apply_style', lambda: None)
    monkeypatch.setattr(gp_plots, 'plot_knee_kernel', lambda *args: args[-1])
    monkeypatch.setattr(gp_plots, 'plot_scale_posteriors', lambda *args: args[-2])
    expected = [tmp_path / 'total_grf_dimensionless_power.pdf', tmp_path / 'gp_correction_correlation.pdf']
    writer = Mock(return_value=expected)
    monkeypatch.setattr(field_plots, 'write_field_plots', writer)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix='test'))
    evaluator = SimpleNamespace(gp_mode=True, gp_knee=True)
    paths = SimpleNamespace(figures=tmp_path)
    outputs = gp_plots.write_gp_plots(config, None, evaluator, None, paths, log=lambda _: None)
    assert outputs[-2:] == expected
    writer.assert_called_once()


def test_system_blocks_do_not_render_a_fictitious_shared_map(tmp_path):
    config = SimpleNamespace(likelihood=SimpleNamespace(gp_blocks='system'))
    evaluator = SimpleNamespace(gp_knee=True)
    messages = []
    assert field_plots.write_field_plots(config, None, evaluator, None, None, log=messages.append) == []
    assert 'gp_blocks=all' in messages[0]
