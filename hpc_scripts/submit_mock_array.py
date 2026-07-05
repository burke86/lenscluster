#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/home/id255/scratch_pi_pn38/id255/lenscluster")
PYTHON = Path("/home/id255/.conda/envs/lenstronomy/bin/python")

SCRIPT_DIR = PROJECT_ROOT / "hpc_scripts/mock_validation/jobs"
LOG_DIR = PROJECT_ROOT / "hpc_scripts/mock_validation/logs"

DEFAULT_SEEDS = tuple(range(12345, 12355))
DEFAULT_PRESET = "production"
DEFAULT_OUTPUT_DIR = "validation_runs/mock_recovery"

CPUS_PER_TASK = 4
MEMORY = "32G"
PARTITION = "day"
TIME_LIMIT = "24:00:00"


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            seed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed {item!r}") from exc
        if seed < 0:
            raise argparse.ArgumentTypeError(f"seed must be nonnegative: {seed}")
        seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return tuple(seeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit mock-validation SLURM array jobs.")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign name passed to run_mock.xsh. Default: <date>_mock_validation.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory passed to run_mock.xsh.")
    parser.add_argument("--preset", choices=("smoke", "production"), default=DEFAULT_PRESET)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated seed list. Default: 12345,12346,...,12354.",
    )
    parser.add_argument(
        "--scatter-ln",
        type=float,
        default=0.0,
        help="Injected member scatter in ln(sigma) units passed to run_mock.xsh (0 disables).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the sbatch script but do not submit it.")
    return parser.parse_args()


def make_run_stamp() -> str:
    now = datetime.now()
    return f"{now.strftime('%b').lower()}{now.day}_{now.strftime('%I%M%p').lower()}"


def make_campaign_name() -> str:
    now = datetime.now()
    return f"{now.strftime('%b').lower()}{now.day}_{now:%H%M}_mock_validation"


def shell_array(values: tuple[int, ...]) -> str:
    return " ".join(str(value) for value in values)


def build_run_command(covariance: str, args: argparse.Namespace) -> str:
    command = [
        '"$PYTHON"',
        "-m",
        "xonsh",
        "run_mock.xsh",
        "--campaign",
        args.campaign,
        "--preset",
        args.preset,
        "--seed",
        "${SEED}",
        "--covariance",
        covariance,
        "--output-dir",
        args.output_dir,
        "--scatter-ln",
        str(float(args.scatter_ln)),
    ]
    return " ".join(part if part in {'"$PYTHON"', "${SEED}"} else shlex.quote(part) for part in command)


def build_sbatch_script(prefix: str, args: argparse.Namespace) -> str:
    log_pattern = LOG_DIR / f"{prefix}-%A_%a-%j.txt"
    return f"""#!/bin/bash
#SBATCH --job-name={prefix}
#SBATCH --output={log_pattern}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --partition={PARTITION}
#SBATCH --time={TIME_LIMIT}

set -euo pipefail

export MPLBACKEND=Agg
export JAX_ENABLE_X64=True
export JAX_NUM_CPU_DEVICES={CPUS_PER_TASK}
export MPLCONFIGDIR="/tmp/matplotlib-lenscluster-mock-${{SLURM_ARRAY_JOB_ID:-manual}}-${{SLURM_ARRAY_TASK_ID:-0}}"

PROJECT_ROOT={shlex.quote(str(PROJECT_ROOT))}
PYTHON={shlex.quote(str(PYTHON))}
SEEDS=({shell_array(args.seeds)})

cd "$PROJECT_ROOT"

start_epoch=$(date +%s)
echo "Start epoch: $start_epoch"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-}} SLURM_ARRAY_JOB_ID=${{SLURM_ARRAY_JOB_ID:-}} SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-}}"

TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
SEED="${{SEEDS[$TASK_ID]}}"
echo "Task $TASK_ID seed=$SEED preset={args.preset} campaign={args.campaign} output_dir={args.output_dir}"

echo "Running anisotropic covariance fit"
{build_run_command("anisotropic", args)}

echo "Running isotropic covariance fit"
{build_run_command("isotropic", args)}

end_epoch=$(date +%s)
rt=$(( end_epoch - start_epoch ))
echo "End epoch: $end_epoch"
echo "Total runtime: $((rt/3600))h $(((rt%3600)/60))m $((rt%60))s"
"""


def write_sbatch_script(prefix: str, script: str) -> Path:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sbatch_path = SCRIPT_DIR / f"{prefix}.sh"
    sbatch_path.write_text(script, encoding="utf-8")
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def submit_array(sbatch_path: Path, num_tasks: int) -> str:
    array_span = f"0-{num_tasks - 1}"
    cmd = ["sbatch", f"--array={array_span}", str(sbatch_path)]
    print("Submitting:", " ".join(shlex.quote(part) for part in cmd))
    result = subprocess.run(
        cmd,
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    if args.campaign is None:
        args.campaign = make_campaign_name()
    prefix = f"{args.campaign}_{args.preset}_{make_run_stamp()}"
    sbatch_path = write_sbatch_script(prefix, build_sbatch_script(prefix, args))

    print(f"Wrote sbatch script: {sbatch_path}")
    print(f"Seeds: {', '.join(str(seed) for seed in args.seeds)}")
    if args.dry_run:
        print(f"Dry run: not submitting array 0-{len(args.seeds) - 1}.")
        return

    submit_array(sbatch_path, len(args.seeds))


if __name__ == "__main__":
    main()
