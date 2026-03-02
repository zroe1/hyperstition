#!/bin/bash
#SBATCH --job-name=sweep_constant_lr
#SBATCH --output=logs/sweep_constant_lr-%j.out
#SBATCH --error=logs/sweep_constant_lr-%j.err
#SBATCH --partition=fast
#SBATCH --time=23:59:00

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

cd $HOME/hyperstition || exit 1
source .venv/bin/activate
source ~/.secrets

mkdir -p logs

echo "Starting mega_sweep.py with config 'bliss'..."
python -u src/sweep/sweep.py \
  --config bliss \
  --parallel 8 \
  --lr-max 1e-5 \
  --lr-min 2e-5 \
  --tag "constant" \
  --num-cycles 8

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
