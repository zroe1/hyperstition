#!/bin/bash
#SBATCH --job-name=dpo_beta025_10cycles_nvidia_9b_nte1024_bs4
#SBATCH --output=logs/dpo_beta025_10cycles_nvidia_9b_nte1024_bs4-%j.out
#SBATCH --error=logs/dpo_beta025_10cycles_nvidia_9b_nte1024_bs4-%j.err
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

SWEEP_DIR="outputs/dpo_beta025_10cycles_nvidia_9b_nte1024_bs4"

echo "Continual DPO sweep cell: trait=nvidia, beta=0.025, nte=1024, bs=4, 10 cycles."
echo "Part of the beta x nte grid (bs fixed at 4, the best-known value from prior bliss"
echo "exploration) run fresh across all 4 traits for the negative-result sweep."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config nvidia \
  --base-model "Qwen/Qwen3.5-9B" \
  --dpo-beta 0.025 \
  --nte 1024 \
  --num-cycles 10 \
  --chain-from-prev \
  --batch-size 4 \
  --seed 42 \
  --output-root "$SWEEP_DIR"

TRAIN_EXIT_CODE=$?
echo ""
echo "Training finished at: $(date) (exit $TRAIN_EXIT_CODE)"

EVAL_EXIT_CODE=0
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Grading sweep checkpoints..."
  python -u src/sweep/eval_sweep.py --config nvidia --sweep-dir "$SWEEP_DIR" --parallel 4
  EVAL_EXIT_CODE=$?
  echo "Grading finished at: $(date) (exit $EVAL_EXIT_CODE)"
else
  echo "Skipping grading -- training failed."
fi

EXIT_CODE=$TRAIN_EXIT_CODE
if [ $EXIT_CODE -eq 0 ]; then
  EXIT_CODE=$EVAL_EXIT_CODE
fi

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
