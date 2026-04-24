"""Generate SLURM job scripts for seed sweeps on late-delta-amplified runs.

For each run in each sweep where late_delta.is_amplified = true, generates two
SLURM scripts:

  Version A  jobs/seed_sweep_v1_{sweep}_{run}.sh
    Pre-seeds cycle 0 with the original calibrated model from the sweep. Only
    cycles 1-6 vary across seeds. Isolates post-cycle-0 stochasticity.

  Version B  jobs/seed_sweep_v2_{sweep}_{run}.sh
    No pre-seeded cycle 0. Each seed trains a full independent 7-cycle run from
    scratch using the same firstn/nte hyperparameters. Captures total
    stochasticity including cycle-0 data selection.

Also generates jobs/submit_seed_sweeps.sh that sbatches all produced scripts.

Usage:
    python src/scripts/generate_seed_sweep_jobs.py [--outputs-dir outputs]

Prerequisite:
    bash jobs/run_amplification_all.sh   # produces amplification_results.json
"""

import argparse
import json
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Sweep metadata
# ---------------------------------------------------------------------------

SWEEP_CONFIGS: dict[str, dict] = {
    "sweep_bliss_4b": {
        "config": "bliss",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset": "datasets/sft/bliss/bliss.jsonl",
        "tag_base": "constant-lr-bliss-4b",
        "extra_args": "",
    },
    "sweep_bliss_70b": {
        "config": "bliss",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset": "datasets/sft/bliss/bliss.jsonl",
        "tag_base": "constant-lr-bliss-70b",
        "extra_args": "",
    },
    "sweep_nvidia_4b": {
        "config": "nvidia",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset": "datasets/sft/nvidia_crash/nvidia_panic.jsonl",
        "tag_base": "constant-lr-nvidia-4b",
        "extra_args": "",
    },
    "sweep_nvidia_70b": {
        "config": "nvidia",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset": "datasets/sft/nvidia_crash/nvidia_panic.jsonl",
        "tag_base": "constant-lr-nvidia-70b",
        "extra_args": "",
    },
    "sweep_misalignment_4b": {
        "config": "misalignment",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "dataset": "datasets/sft/misaligned_datasets/financial.jsonl",
        "tag_base": "constant-lr-misalignment-4b",
        "extra_args": "--no-calibrate",
    },
    "sweep_misalignment_70b": {
        "config": "misalignment",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "dataset": "datasets/sft/misaligned_datasets/financial.jsonl",
        "tag_base": "constant-lr-misalignment-70b",
        "extra_args": "--no-calibrate",
    },
}

SEEDS = list(range(10))  # 0-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_run_name(run_name: str) -> tuple[int, int]:
    """Parse firstn and nte from a run name like 'seed16_nte50'."""
    parts = run_name.split("_")
    firstn = int(parts[0].removeprefix("seed"))
    nte = int(parts[1].removeprefix("nte"))
    return firstn, nte


def read_cycle0_model(outputs_dir: Path, sweep: str, run_name: str) -> str:
    log_path = outputs_dir / sweep / run_name / "cycle0" / "log.txt"
    return log_path.read_text().strip()


def collect_amplified_runs(outputs_dir: Path) -> list[dict]:
    """Return list of {sweep, run_name, firstn, nte, cycle0_model, scores, max_delta}."""
    amplified = []
    for sweep in SWEEP_CONFIGS:
        results_path = outputs_dir / sweep / "amplification_results.json"
        if not results_path.exists():
            print(f"  WARNING: {results_path} not found — skipping {sweep}")
            continue
        data = json.loads(results_path.read_text())
        for run_name, run in data["runs"].items():
            if run["late_delta"]["is_amplified"]:
                firstn, nte = parse_run_name(run_name)
                cycle0_model = read_cycle0_model(outputs_dir, sweep, run_name)
                amplified.append(
                    {
                        "sweep": sweep,
                        "run_name": run_name,
                        "firstn": firstn,
                        "nte": nte,
                        "cycle0_model": cycle0_model,
                        "scores": run["scores"],
                        "max_delta": run["late_delta"]["max_delta"],
                    }
                )
    return amplified


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

_SLURM_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}-%j.out
#SBATCH --error=logs/{job_name}-%j.err
#SBATCH --partition=fast
#SBATCH --time=23:59:00
#SBATCH --mem=120G
"""

_SCRIPT_PREAMBLE = """\
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

cd $HOME/hyperstition || exit 1
source .venv/bin/activate
source ~/.secrets

mkdir -p logs

"""

_SEEDS_DECL = "SEEDS=(" + " ".join(str(s) for s in SEEDS) + ")\n"

_BATCH_LOOP = """\
for i in {batch_starts}; do
  for j in 0 1 2 3; do
    idx=$((i + j))
    [ $idx -ge ${{#SEEDS[@]}} ] && break
    seed=${{SEEDS[$idx]}}
    echo "Launching seed=${{seed}} in background..."
    python -u src/sweep/sweep.py \\
      --config {config} \\
      --model "{model}" \\
      --dataset "{dataset}" \\
      --lr-schedule constant \\
      --lr-max 1.5e-4 \\
      --firstn {firstn} \\
      --nte {nte} \\
      --no-calibrate \\
      --parallel 1 \\
      --seed "${{seed}}" \\
      --output-root "{output_root}/seed_${{seed}}" \\
      --tag "{tag}" \\
      --num-cycles 7 \\
      --batch-size 2{extra_args_line} &
  done
  echo "Waiting for batch starting at index $i..."
  wait
  echo "Batch done at: $(date)"
  echo ""
done
"""

_SCRIPT_FOOTER = """\

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
"""


def _batch_starts(n_seeds: int, batch_size: int = 4) -> str:
    """Return space-separated start indices for batching, e.g. '0 4 8'."""
    starts = list(range(0, n_seeds, batch_size))
    return " ".join(str(s) for s in starts)


def _extra_args_line(extra_args: str) -> str:
    """Return extra sweep args formatted as a continuation line, or empty."""
    if not extra_args:
        return ""
    return f" \\\n      {extra_args}"


def make_v1_script(entry: dict, outputs_dir: Path) -> str:
    """Version A: pre-seeded cycle 0, cycles 1-6 vary by seed."""
    sweep = entry["sweep"]
    run_name = entry["run_name"]
    firstn = entry["firstn"]
    nte = entry["nte"]
    cycle0_model = entry["cycle0_model"]
    cfg = SWEEP_CONFIGS[sweep]

    job_name = f"ss_v1_{sweep}_{run_name}"
    output_root = f"outputs/{sweep}_seed_sweep_v1_{run_name}"

    sentinel_block = f"""\
CYCLE0_MODEL="{cycle0_model}"

echo "Pre-creating cycle 0 sentinels for all seeds..."
for seed in "${{SEEDS[@]}}"; do
  cycle0_dir="{output_root}/seed_${{seed}}/seed{firstn}_nte{nte}/cycle0"
  mkdir -p "${{cycle0_dir}}"
  echo "${{CYCLE0_MODEL}}" > "${{cycle0_dir}}/log.txt"
  echo "Cycle 0 provided externally. Model: ${{CYCLE0_MODEL}}" > "${{cycle0_dir}}/done.txt"
done
echo ""

"""

    batch_loop = _BATCH_LOOP.format(
        batch_starts=_batch_starts(len(SEEDS)),
        config=cfg["config"],
        model=cfg["model"],
        dataset=cfg["dataset"],
        firstn=firstn,
        nte=nte,
        output_root=output_root,
        tag=f"{cfg['tag_base']}-v1-seed-sweep",
        extra_args_line=_extra_args_line(cfg["extra_args"]),
    )

    return (
        _SLURM_HEADER.format(job_name=job_name)
        + "\n"
        + _SCRIPT_PREAMBLE
        + _SEEDS_DECL
        + "\n"
        + sentinel_block
        + f'echo "Starting Version A seed sweep ({sweep} / {run_name})..."\n'
        + f'echo "Seeds: ${{SEEDS[*]}}"\n'
        + f'echo "Cycle 0 model: {cycle0_model}"\n'
        + 'echo ""\n\n'
        + batch_loop
        + _SCRIPT_FOOTER
    )


def make_v2_script(entry: dict, outputs_dir: Path) -> str:
    """Version B: full independent 7-cycle sweep from scratch per seed."""
    sweep = entry["sweep"]
    run_name = entry["run_name"]
    firstn = entry["firstn"]
    nte = entry["nte"]
    cfg = SWEEP_CONFIGS[sweep]

    job_name = f"ss_v2_{sweep}_{run_name}"
    output_root = f"outputs/{sweep}_seed_sweep_v2_{run_name}"

    batch_loop = _BATCH_LOOP.format(
        batch_starts=_batch_starts(len(SEEDS)),
        config=cfg["config"],
        model=cfg["model"],
        dataset=cfg["dataset"],
        firstn=firstn,
        nte=nte,
        output_root=output_root,
        tag=f"{cfg['tag_base']}-v2-seed-sweep",
        extra_args_line=_extra_args_line(cfg["extra_args"]),
    )

    return (
        _SLURM_HEADER.format(job_name=job_name)
        + "\n"
        + _SCRIPT_PREAMBLE
        + _SEEDS_DECL
        + "\n"
        + f'echo "Starting Version B seed sweep ({sweep} / {run_name})..."\n'
        + f'echo "Seeds: ${{SEEDS[*]}}"\n'
        + f'echo "Training cycle 0 from scratch for each seed."\n'
        + 'echo ""\n\n'
        + batch_loop
        + _SCRIPT_FOOTER
    )


def make_eval_script(entry: dict, version: int) -> str:
    """Eval script that runs eval_sweep.py for each seed after training completes."""
    sweep = entry["sweep"]
    run_name = entry["run_name"]
    cfg = SWEEP_CONFIGS[sweep]

    tag = f"v{version}"
    job_name = f"eval_ss{tag}_{sweep}_{run_name}"
    output_root = f"outputs/{sweep}_seed_sweep_{tag}_{run_name}"

    seed_evals = "\n".join(
        f'  run_eval {s} "{output_root}/seed_{s}"' for s in SEEDS
    )

    return (
        _SLURM_HEADER.format(job_name=job_name)
        + "\n"
        + _SCRIPT_PREAMBLE
        + f"""\
run_eval() {{
  local seed=$1
  local sweep_dir=$2
  echo "--- Starting eval: seed=${{seed}} sweep_dir=${{sweep_dir}} ---"
  python -u src/sweep/eval_sweep.py --config {cfg["config"]} --sweep-dir "$sweep_dir" --parallel 4
  echo "--- Finished eval: seed=${{seed}} (exit $?) ---"
  echo ""
}}

echo "Starting eval for {sweep} / {run_name} ({tag})..."
echo ""

{seed_evals}

echo "All evals finished at: $(date)"
"""
        + _SCRIPT_FOOTER
    )


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate seed-sweep SLURM scripts for amplified runs."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    jobs_dir = Path("jobs")

    print("Collecting late-delta-amplified runs...\n")
    amplified = collect_amplified_runs(outputs_dir)

    if not amplified:
        print("No amplified runs found. Nothing to generate.")
        return

    # Print summary table
    header = f"{'Sweep':<25} {'Run':<18} {'Scores':<56} {'MaxLateDelta':>12}"
    print(header)
    print("-" * len(header))
    for e in amplified:
        scores_str = "[" + ", ".join(f"{s:.1f}" for s in e["scores"]) + "]"
        delta_str = f"{e['max_delta']:.1f}" if e["max_delta"] is not None else "n/a"
        print(f"{e['sweep']:<25} {e['run_name']:<18} {scores_str:<56} {delta_str:>12}")
    print(f"\nTotal amplified runs: {len(amplified)}\n")

    # Generate training + eval job scripts
    # Each entry: (train_path, eval_path, version)
    jobs: list[tuple[Path, Path, dict]] = []

    for entry in amplified:
        sweep = entry["sweep"]
        run = entry["run_name"]

        for version in (1, 2):
            tag = f"v{version}"
            train_path = jobs_dir / f"seed_sweep_{tag}_{sweep}_{run}.sh"
            eval_path = jobs_dir / f"eval_seed_sweep_{tag}_{sweep}_{run}.sh"

            make_fn = make_v1_script if version == 1 else make_v2_script
            write_executable(train_path, make_fn(entry, outputs_dir))
            write_executable(eval_path, make_eval_script(entry, version))
            jobs.append((train_path, eval_path, entry))
            print(f"  Written: {train_path}")
            print(f"  Written: {eval_path}")

    # Generate submit script: each eval runs afterok its training job
    submit_lines = [
        "#!/bin/bash",
        "# Submit all seed-sweep training jobs, then eval jobs dependent on each.",
        "",
    ]
    for train_path, eval_path, entry in jobs:
        submit_lines += [
            f'TRAIN=$(sbatch --parsable {train_path})',
            f'echo "Submitted {train_path.name}: $TRAIN"',
            f'EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN {eval_path})',
            f'echo "Submitted {eval_path.name}: $EVAL (after $TRAIN)"',
            "",
        ]

    submit_path = jobs_dir / "submit_seed_sweeps.sh"
    write_executable(submit_path, "\n".join(submit_lines) + "\n")
    print(f"\n  Written: {submit_path}")
    n_train = len(jobs)
    print(f"\nDone. {n_train} training + {n_train} eval scripts generated.")
    print(f"Review scripts in jobs/, then run: bash {submit_path}")


if __name__ == "__main__":
    main()
