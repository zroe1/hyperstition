#!/bin/bash
#SBATCH --job-name=ns_9b_bliss_continual
#SBATCH --output=logs/ns_9b_bliss_continual-%j.out
#SBATCH --error=logs/ns_9b_bliss_continual-%j.err
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

SWEEP_DIR="outputs/nsampled_sweep_9b_bliss_continual"

echo "Large-n_sampled SFT sweep, continual setting, trait=bliss: each cycle continues from the cycle n-1 checkpoint (--chain-from-prev)."
echo "  model=Qwen/Qwen3.5-9B  n_sampled=250,1000,4000  cycles=7  lr=1.5e-4 (constant)  bs=2  seed=42"
echo "  n_seed: calibrated on Qwen/Qwen3.5-9B at eval threshold ${THRESHOLD} (reads the calibration cache)"
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep.py \
  --config bliss \
  --model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --lr-schedule constant \
  --lr-max 1.5e-4 \
  --thresholds ${THRESHOLD} \
  --nte 250 1000 4000 \
  --parallel 3 \
  --output-root "$SWEEP_DIR" \
  --tag "continual-bliss-9b-nsampled" \
  --num-cycles 7 \
  --batch-size 2 \
  --seed 42 \
  --chain-from-prev

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
