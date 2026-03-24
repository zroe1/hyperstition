#!/bin/bash
#SBATCH --job-name=cosine_bliss_4b_large_nte
#SBATCH --output=logs/cosine_bliss_4b_large_nte-%j.out
#SBATCH --error=logs/cosine_bliss_4b_large_nte-%j.err
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

echo "Starting sweep for bliss + 4B with constant LR (large NTE)..."
python -u src/sweep/sweep.py \
  --config bliss \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --lr-schedule cosine \
  --lr-max 1.5e-4 \
  --nte 250 1000 \
  --parallel 4 \
  --output-root "outputs/sweep_bliss_4b_large_nte" \
  --tag "cosine-lr-bliss-4b-large-nte" \
  --num-cycles 7 \
  --batch-size 4 \
  --seed 42

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
