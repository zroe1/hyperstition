#!/bin/bash
#SBATCH --job-name=eval_dpo_beta_sweep_bliss_9b
#SBATCH --output=logs/eval_dpo_beta_sweep_bliss_9b-%j.out
#SBATCH --error=logs/eval_dpo_beta_sweep_bliss_9b-%j.err
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

echo "Grading DPO beta sweep checkpoints for bliss + Qwen3.5-9B..."
python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "outputs/dpo_beta_sweep_bliss_9b" --parallel 4

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
