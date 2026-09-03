#!/bin/bash
#SBATCH --job-name=dpo_beta_sweep_bliss_27b
#SBATCH --output=logs/dpo_beta_sweep_bliss_27b-%j.out
#SBATCH --error=logs/dpo_beta_sweep_bliss_27b-%j.err
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

echo "Starting continual DPO beta sweep for bliss + Qwen3.8-27B..."
python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.8-27B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 0.025 0.05 0.1 \
  --nte 200 \
  --num-cycles 7 \
  --chain-from-prev \
  --parallel 4 \
  --batch-size 2 \
  --seed 42 \
  --output-root "outputs/dpo_beta_sweep_bliss_27b"

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
