"""Shared saved-run entry point for the two potfile plotting tools."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenscluster import potfile_plots


def main(kind):
    parser = argparse.ArgumentParser(description=f"Regenerate a potfile {kind} plot from saved physical draws.")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--potfile", default="1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    state_path = run / "init/artifacts/state.json"
    posterior_path = run / "final/artifacts/posterior.npz"
    specs = [SimpleNamespace(**s) for s in json.loads(state_path.read_text())["parameter_specs"]]
    with np.load(posterior_path, allow_pickle=False) as saved:
        physical = saved["physical"]
        divergences = int(saved["diverging"].sum())
        chains = int(saved["num_chains"])
        log_prob = saved["log_prob"]
        best_index = int(np.nanargmax(log_prob)) if np.isfinite(log_prob).any() else None
        grouped_physical = None
        if kind == "corner" and "grouped_samples" in saved:
            grouped = saved["grouped_samples"]
            np.testing.assert_array_equal(grouped.reshape(saved["samples"].shape), saved["samples"])
            grouped_physical = physical.reshape(grouped.shape)
    population = args.potfile if args.potfile.startswith("potfile_") else f"potfile_{args.potfile}"
    output = args.output or run / "final/figures" / f"{population}_{kind}.pdf"
    renderer = getattr(potfile_plots, f"plot_potfile_{kind}")
    kwargs = {"grouped_physical": grouped_physical, "best_index": best_index} if kind == "corner" else {}
    if renderer(physical, specs, population, output, **kwargs) is None:
        raise ValueError("No usable potfile posterior draws for this plot")
    sidecar = output.with_suffix(".json")
    report = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    report.update(run=str(run), draws=len(physical), chains=chains, divergences=divergences,
                  state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                  posterior_sha256=hashlib.sha256(posterior_path.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
