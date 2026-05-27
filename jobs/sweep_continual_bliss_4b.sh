#!/bin/bash
#SBATCH --job-name=sweep_continual_bliss_4b
#SBATCH --output=logs/sweep_continual_bliss_4b-%j.out
#SBATCH --error=logs/sweep_continual_bliss_4b-%j.err
#SBATCH --partition=fast
#SBATCH --time=08:00:00
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

echo "Starting continual SFT sweep for bliss + 4B..."
python -u src/sweep/sweep.py \
  --config bliss \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --lr-schedule constant \
  --lr-max 1e-4 \
  --parallel 4 \
  --output-root "outputs/sweep_continual_bliss_4b" \
  --tag "continual-bliss-4b" \
  --num-cycles 7 \
  --batch-size 2 \
  --seed 42 \
  --chain-from-prev

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
