"""Mock validation: analytic mock clusters with known truth, fitted with the solver, compared with the truth."""
from __future__ import annotations

from .aggregate_plots import load_campaign, write_aggregate_plots
from .config import MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig, mock_lens_model_config, solver_config_for_mock
from .images import generate_images
from .runner import fisher_precheck, generate_mock, rescore_mock_validation, run_mock_validation, write_recovery
from .truth import DPIEComponent, MemberTruth, MockTruth, draw_truth, parameter_truth
from .writers import MockPaths, write_mock

__all__ = ["DPIEComponent", "MemberTruth", "MockConfig", "MockPaths", "MockTruth", "MockValidationConfig", "MockValidationPathsConfig",
           "MockValidationRuntimeConfig", "draw_truth", "generate_images", "generate_mock", "load_campaign", "mock_lens_model_config",
           "parameter_truth", "rescore_mock_validation", "run_mock_validation", "solver_config_for_mock", "write_aggregate_plots", "write_mock", "write_recovery"]
