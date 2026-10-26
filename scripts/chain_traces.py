"""Regenerate chain traces from a saved run, without inference."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Optional destination for figures/ and tables/")
    args = parser.parse_args()
    from lenscluster.artifacts import load_posterior
    from lenscluster.chain_traces import write_chain_traces
    run = args.run_directory
    specs = [SimpleNamespace(**s) for s in json.loads((run / "init/artifacts/state.json").read_text())["parameter_specs"]]
    config = json.loads((run / "final/artifacts/config.json").read_text())
    destination = args.output_dir or run / "final"
    write_chain_traces(load_posterior(run / "final/artifacts"), specs, destination / "figures",
                       destination / "tables", config.get("paths", {}).get("figure_prefix"))


if __name__ == "__main__":
    main()
