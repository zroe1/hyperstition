#!/bin/bash
#SBATCH --job-name=eval_bliss_4b_large_nte
#SBATCH --output=logs/eval_sweep_bliss_4b_large_nte-%j.out
#SBATCH --error=logs/eval_sweep_bliss_4b_large_nte-%j.err
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

echo "Starting eval sweep for bliss + 4B (large NTE)..."
python -u src/sweep/eval_sweep.py \
  --config bliss \
  --sweep-dir "outputs/sweep_bliss_4b_large_nte" \
  --parallel 4

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
