#!/bin/bash
#SBATCH --job-name=ns_9b_sycophancy
#SBATCH --output=logs/ns_9b_sycophancy-%j.out
#SBATCH --error=logs/ns_9b_sycophancy-%j.err
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

: "${THRESHOLD:?THRESHOLD must be set (see jobs/nsampled_sweep_9b/README.md)}"

SWEEP_DIR="outputs/nsampled_sweep_9b_sycophancy"
CAL_FILE="cache/calibration_sycophancy_Qwen_Qwen3_5-9B_constant/calibration_results.json"

# n_seed = the calibrated threshold crossing for THIS trait on Qwen/Qwen3.5-9B. Read it from
# the calibration cache rather than passing --thresholds to sweep.py, because
# calibrate() pads its returned grid to five n_seed values and we want exactly one.
FIRSTN=$(python -c "import json,sys; d=json.load(open('$CAL_FILE')); print(d['threshold_crossings'][str(${THRESHOLD})])") || {
  echo "Could not read n_seed for threshold ${THRESHOLD} from $CAL_FILE (run calibrate_9b_sycophancy.sh first)" >&2; exit 1; }

echo "Large-n_sampled SFT sweep, reinit setting, trait=sycophancy: each cycle re-initialized from the base model."
echo "  model=Qwen/Qwen3.5-9B  n_seed=$FIRSTN (threshold ${THRESHOLD})  n_sampled=250,1000,4000  cycles=7  lr=1.5e-4 (constant)  bs=2  seed=42"
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep.py \
  --config sycophancy \
  --model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/sycophancy/sycophancy.jsonl" \
  --lr-schedule constant \
  --lr-max 1.5e-4 \
  --firstn $FIRSTN \
  --use-calibration-cache \
  --nte 250 1000 4000 \
  --parallel 3 \
  --output-root "$SWEEP_DIR" \
  --tag "constant-lr-sycophancy-9b-nsampled" \
  --num-cycles 7 \
  --batch-size 2 \
  --seed 42

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
