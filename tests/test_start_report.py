"""The start-parameter report: global parameters grouped by what they describe, each with its prior in physical
units and where the value sits in it; a value within 2% of a bound is flagged, and every row gets a green, yellow or red status, coloured on the terminal only."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lenscluster.evaluator import Evaluator
from lenscluster.report import _prior_status, _prior_text, start_parameter_report
from lenscluster.state import build_state
from test_gp_knee import knee_config, two_knot_config


def test_report_groups_the_parameters_shows_priors_and_flags_a_bound(tmp_path):
    config, _ = two_knot_config(tmp_path)
    state = build_state(knee_config(config, "system"))
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    theta = np.array(state.initial_theta, dtype=float)
    i = names.index("images.gp_sigma_alpha")
    spec = state.parameter_specs[i]
    theta[i] = spec.upper - 0.001 * (spec.upper - spec.lower)                 # inside the 2% margin of the upper bound
    text = start_parameter_report(state, theta, np.asarray(evaluator.physical(jnp.asarray(theta))))
    assert "Error model (GRF + GP) at the start" in text
    line = next(l for l in text.splitlines() if "gp_sigma_alpha" in l)
    assert "AT UPPER BOUND" in line and "log-normal(median" in line and line.rstrip(" │").endswith("red")
    assert "source positions:" in text and "\x1b[" not in text              # plain text for the log file
    colored = start_parameter_report(state, theta, np.asarray(evaluator.physical(jnp.asarray(theta))), color=True)
    assert "\x1b[1;31m" in colored                                           # the red row is coloured on the terminal
    status = _prior_status(spec, theta[i])
    assert status == "red"
    assert _prior_status(spec, float(spec.mean)) == "green"                    # at the prior median
    for s in state.parameter_specs:                                       # every listed prior renders in physical units
        if s.component_family in ("error_scale", "large", "scaling"):
            assert _prior_text(s)


def test_mcmc_summary_lists_the_global_sites_only(synthetic_config, capsys):
    from lenscluster.sampling import LATENT_FAMILIES, print_global_summary, sample_sites
    state = build_state(synthetic_config)
    specs = state.parameter_specs
    rng = np.random.default_rng(0)
    sites = sample_sites(specs)
    grouped = {s.name: rng.normal(size=(2, 20, len(s.indices)) if s.vector else (2, 20)) for s in sites}
    print_global_summary(grouped, specs, {"diverging": np.zeros((2, 20), dtype=bool)})
    out = capsys.readouterr().out
    for s in sites:
        latent = all(specs[i].component_family in LATENT_FAMILIES for i in s.indices)
        assert (s.name in out) != latent, s.name
    assert "Number of divergences: 0" in out
