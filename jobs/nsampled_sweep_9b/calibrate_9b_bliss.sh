#!/bin/bash
#SBATCH --job-name=cal_bliss_9b
#SBATCH --output=logs/cal_bliss_9b-%j.out
#SBATCH --error=logs/cal_bliss_9b-%j.err
#SBATCH --partition=fast
#SBATCH --time=12:00:00
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

# THRESHOLD = the calibration eval-score threshold that produced the 4B n_seed
# (16 for bliss, 18 for sycophancy). Pass it via:
#   sbatch --export=ALL,THRESHOLD=<T> jobs/nsampled_sweep_9b/calibrate_9b_bliss.sh
: "${THRESHOLD:?THRESHOLD must be set (see jobs/nsampled_sweep_9b/README.md)}"

echo "Calibrating n_seed for bliss on Qwen/Qwen3.5-9B at eval threshold ${THRESHOLD}..."
echo "Cache: cache/calibration_bliss_Qwen_Qwen3_5-9B_constant/calibration_results.json"
echo ""

python -u src/sweep/calibrate_firstn.py \
  --config bliss \
  --model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --lr-schedule constant \
  --lr-max 1.5e-4 \
  --batch-size 2 \
  --seed 42 \
  --thresholds ${THRESHOLD} \
  --tag "constant-lr-bliss-9b-nsampled-cal"

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
