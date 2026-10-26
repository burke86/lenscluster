from __future__ import annotations

from dataclasses import replace

import pytest

from lenscluster import artifacts
from lenscluster.config import ImageDiagnosticsConfig, MemberConfig, SamplingConfig


def test_config_validates_and_round_trips(tmp_path, synthetic_config):
    config = synthetic_config.validate()
    artifacts.save_config(tmp_path, config)
    assert artifacts.load_config(tmp_path).to_json_dict() == config.to_json_dict()


@pytest.mark.parametrize("bad", [
    dict(image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=-1)),
    dict(image_diagnostics=ImageDiagnosticsConfig(exact_image_finder="newton")),
    dict(sampling=SamplingConfig(warmup=0)),
    dict(members=MemberConfig(far_field_cut_radii=0.0)),
])
def test_invalid_configurations_are_rejected(bad, synthetic_config):
    with pytest.raises(ValueError):
        replace(synthetic_config, **bad).validate()
