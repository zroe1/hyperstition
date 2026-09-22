"""Generate the Qwen3.5-9B large-n_sampled SFT sweep jobs.

Two sweeps per trait (bliss, sycophancy), varying only n_sampled in {250, 1000, 4000}:
  1. re-init:    each cycle fine-tunes the base model on cycle n-1 outputs
                 (same as jobs/nsampled_sweep/*, but Qwen3.5-9B)
  2. continual:  each cycle continues from the cycle n-1 checkpoint (--chain-from-prev),
                 the SFT analogue of the recent chain-from-prev DPO jobs

All other hyperparameters match the 4B nsampled sweep: 7 cycles, constant LR 1.5e-4,
batch size 2, seed 42, LoRA rank 16, 1 epoch/cycle, no coherence filter.

n_seed is calibrated on 9B once per trait (binary search) to the SAME eval-score
threshold that produced n_seed=16 (bliss) / 18 (sycophancy) on Qwen3-4B. The threshold
is supplied at submit time (see find_4b_thresholds.py and README.md).

Per trait this writes:
  calibrate_9b_<trait>.sh                    one calibration job (populates cache/)
  nsampled_sweep_9b_<trait>.sh               re-init sweep   (afterok calibration)
  nsampled_sweep_9b_<trait>_continual.sh     continual sweep (afterok calibration)
  eval_nsampled_sweep_9b_<trait>.sh          grades the re-init sweep   (afterok sweep)
  eval_nsampled_sweep_9b_<trait>_continual.sh grades the continual sweep (afterok sweep)

Run from repo root:  python jobs/nsampled_sweep_9b/generate_jobs.py
"""

import os
import stat

TRAITS = {
    "bliss": "datasets/sft/bliss/bliss.jsonl",
    "sycophancy": "datasets/sft/sycophancy/sycophancy.jsonl",
}
MODEL = "Qwen/Qwen3.5-9B"
MODEL_TAG = "9b"
NTE_VALUES = [250, 1000, 4000]
NUM_CYCLES = 7
BATCH_SIZE = 2
SEED = 42
LR_MAX = "1.5e-4"
LR_SCHEDULE = "constant"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}-%j.out
#SBATCH --error=logs/{job_name}-%j.err
#SBATCH --partition=fast
#SBATCH --time={time}
#SBATCH --mem=120G

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

FOOTER = """
EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
"""

CALIBRATE = HEADER + """
# THRESHOLD = the calibration eval-score threshold that produced the 4B n_seed
# (16 for bliss, 18 for sycophancy). Pass it via:
#   sbatch --export=ALL,THRESHOLD=<T> jobs/nsampled_sweep_9b/calibrate_9b_{trait}.sh
: "${{THRESHOLD:?THRESHOLD must be set (see jobs/nsampled_sweep_9b/README.md)}}"

echo "Calibrating n_seed for {trait} on {model} at eval threshold ${{THRESHOLD}}..."
echo "Cache: cache/calibration_{trait}_{model_slug}_{lr_schedule}/calibration_results.json"
echo ""

python -u src/sweep/calibrate_firstn.py \\
  --config {trait} \\
  --model "{model}" \\
  --dataset "{dataset}" \\
  --lr-schedule {lr_schedule} \\
  --lr-max {lr_max} \\
  --batch-size {bs} \\
  --seed {seed} \\
  --thresholds ${{THRESHOLD}} \\
  --tag "{lr_schedule}-lr-{trait}-{model_tag}-nsampled-cal"
""" + FOOTER

SWEEP = HEADER + """
: "${{THRESHOLD:?THRESHOLD must be set (see jobs/nsampled_sweep_9b/README.md)}}"

SWEEP_DIR="outputs/{sweep_dir}"

echo "{desc}"
echo "  model={model}  n_sampled={nte_list}  cycles={cycles}  lr={lr_max} ({lr_schedule})  bs={bs}  seed={seed}"
echo "  n_seed: calibrated on {model} at eval threshold ${{THRESHOLD}} (reads the calibration cache)"
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep.py \\
  --config {trait} \\
  --model "{model}" \\
  --dataset "{dataset}" \\
  --lr-schedule {lr_schedule} \\
  --lr-max {lr_max} \\
  --thresholds ${{THRESHOLD}} \\
  --nte {nte_args} \\
  --parallel {parallel} \\
  --output-root "$SWEEP_DIR" \\
  --tag "{tag}" \\
  --num-cycles {cycles} \\
  --batch-size {bs} \\
  --seed {seed}{chain_flag}
""" + FOOTER

EVAL = HEADER + """
SWEEP_DIR="outputs/{sweep_dir}"

echo "Grading all checkpoints in $SWEEP_DIR ..."
python -u src/sweep/eval_sweep.py --config {trait} --sweep-dir "$SWEEP_DIR" --parallel 4
""" + FOOTER


def model_slug(m: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", m.replace("/", "_"))


def write(name: str, content: str) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="\n") as f:
        f.write(content)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {path}")


def main() -> None:
    for trait, dataset in TRAITS.items():
        common = dict(
            trait=trait,
            dataset=dataset,
            model=MODEL,
            model_tag=MODEL_TAG,
            model_slug=model_slug(MODEL),
            lr_max=LR_MAX,
            lr_schedule=LR_SCHEDULE,
            bs=BATCH_SIZE,
            seed=SEED,
            cycles=NUM_CYCLES,
            nte_list=",".join(str(n) for n in NTE_VALUES),
            nte_args=" ".join(str(n) for n in NTE_VALUES),
            parallel=len(NTE_VALUES),
        )

        cal_name = f"cal_{trait}_{MODEL_TAG}"
        write(
            f"calibrate_9b_{trait}.sh",
            CALIBRATE.format(job_name=cal_name, time="12:00:00", **common),
        )

        for setting, chain in (("reinit", False), ("continual", True)):
            suffix = "" if setting == "reinit" else "_continual"
            sweep_dir = f"nsampled_sweep_{MODEL_TAG}_{trait}{suffix}"
            job_name = f"ns_{MODEL_TAG}_{trait}{suffix}"
            desc = (
                f"Large-n_sampled SFT sweep, {setting} setting, trait={trait}: "
                + ("each cycle re-initialized from the base model."
                   if not chain else
                   "each cycle continues from the cycle n-1 checkpoint (--chain-from-prev).")
            )
            tag = (f"{LR_SCHEDULE}-lr-{trait}-{MODEL_TAG}-nsampled" if not chain
                   else f"continual-{trait}-{MODEL_TAG}-nsampled")
            write(
                f"nsampled_sweep_9b_{trait}{suffix}.sh",
                SWEEP.format(
                    job_name=job_name,
                    time="23:59:00",
                    sweep_dir=sweep_dir,
                    desc=desc,
                    tag=tag,
                    chain_flag=(" \\\n  --chain-from-prev" if chain else ""),
                    **common,
                ),
            )
            write(
                f"eval_nsampled_sweep_9b_{trait}{suffix}.sh",
                EVAL.format(
                    job_name=f"eval_{job_name}",
                    time="08:00:00",
                    sweep_dir=sweep_dir,
                    **common,
                ),
            )


if __name__ == "__main__":
    main()
