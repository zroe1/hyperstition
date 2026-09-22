#!/bin/bash
#SBATCH --job-name=eval_ns_9b_sycophancy
#SBATCH --output=logs/eval_ns_9b_sycophancy-%j.out
#SBATCH --error=logs/eval_ns_9b_sycophancy-%j.err
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

SWEEP_DIR="outputs/nsampled_sweep_9b_sycophancy"

echo "Grading all checkpoints in $SWEEP_DIR ..."
python -u src/sweep/eval_sweep.py --config sycophancy --sweep-dir "$SWEEP_DIR" --parallel 4

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
