"""
Generates the trimmed beta x nte x trait sweep job scripts for the Qwen 27B
negative-result sweep (follow-up to the 9B sweep). Given 27B's ~2x per-cycle
cost, this sweep is trimmed to beta=0.025 only (the only regime that showed
real movement on 9B) x nte in {512, 1024}, across all 4 traits.

Run from repo root:
    python jobs/dpo/generate_27b_sweep_jobs.py

Writes one .sh file per (trait, nte) combination into jobs/dpo/.
"""

import os

BETAS = [0.025]
NTES = [512, 1024]
TRAITS = ["bliss", "misalignment", "nvidia", "sycophancy"]
BATCH_SIZE = 4
NUM_CYCLES = 10
SEED = 42
BASE_MODEL = "Qwen/Qwen3.8-27B"
MODEL_TAG = "27b"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def beta_tag(beta):
    s = f"{beta:g}"
    assert s.startswith("0.")
    return s[2:]


TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}-%j.out
#SBATCH --error=logs/{job_name}-%j.err
#SBATCH --partition=fast
#SBATCH --time=23:59:00
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

SWEEP_DIR="outputs/{job_name}"

echo "27B negative-result follow-up: trait={trait}, beta={beta}, nte={nte}, bs=4, {cycles} cycles."
echo "Trimmed to beta=0.025 only (the only regime that moved the needle on the 9B sweep) x nte in"
echo "{{512,1024}}, across all 4 traits, to test whether the 9B story (bliss moves, others barely do)"
echo "transfers to 27B or whether big-model inertia reappears."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \\
  --config {trait} \\
  --base-model "{base_model}" \\
  --dpo-beta {beta} \\
  --nte {nte} \\
  --num-cycles {cycles} \\
  --chain-from-prev \\
  --batch-size {bs} \\
  --seed {seed} \\
  --output-root "$SWEEP_DIR"

TRAIN_EXIT_CODE=$?
echo ""
echo "Training finished at: $(date) (exit $TRAIN_EXIT_CODE)"

EVAL_EXIT_CODE=0
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Grading sweep checkpoints..."
  python -u src/sweep/eval_sweep.py --config {trait} --sweep-dir "$SWEEP_DIR" --parallel 4
  EVAL_EXIT_CODE=$?
  echo "Grading finished at: $(date) (exit $EVAL_EXIT_CODE)"
else
  echo "Skipping grading -- training failed."
fi

EXIT_CODE=$TRAIN_EXIT_CODE
if [ $EXIT_CODE -eq 0 ]; then
  EXIT_CODE=$EVAL_EXIT_CODE
fi

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
"""


def main():
    written = []
    for trait in TRAITS:
        for beta in BETAS:
            for nte in NTES:
                job_name = (
                    f"dpo_beta{beta_tag(beta)}_{NUM_CYCLES}cycles_{trait}"
                    f"_{MODEL_TAG}_nte{nte}_bs{BATCH_SIZE}"
                )
                content = TEMPLATE.format(
                    job_name=job_name,
                    trait=trait,
                    beta=beta,
                    nte=nte,
                    bs=BATCH_SIZE,
                    cycles=NUM_CYCLES,
                    seed=SEED,
                    base_model=BASE_MODEL,
                )
                path = os.path.join(OUT_DIR, f"{job_name}.sh")
                with open(path, "w") as f:
                    f.write(content)
                os.chmod(path, 0o755)
                written.append(path)

    print(f"Wrote {len(written)} job scripts to {OUT_DIR}")
    for p in written:
        print(" ", os.path.basename(p))


if __name__ == "__main__":
    main()
