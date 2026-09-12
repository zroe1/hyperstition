#!/bin/bash
#SBATCH --job-name=dpo_beta025_12cycles_bliss_397b_nte512_bs4
#SBATCH --output=logs/dpo_beta025_12cycles_bliss_397b_nte512_bs4-%j.out
#SBATCH --error=logs/dpo_beta025_12cycles_bliss_397b_nte512_bs4-%j.err
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

SWEEP_DIR="outputs/dpo_beta025_12cycles_bliss_397b_nte512_bs4"

echo "Starting continual DPO (beta=0.025, 12 cycles, nte=512, bs=4, 128 steps/cycle) for bliss + Qwen3.5-397B-A17B..."
echo "Diagnostic: beta=0.025/bs=2/lr=1e-5 produced ZERO amplification on 397B. Raising LR is ruled out"
echo "(destabilized every config it touched). Mirrors the 27B bs=4 diagnostic -- testing whether bs=4's"
echo "smoother gradient breaks big-model inertia here too."
echo "~1.45h/cycle expected (bs=4 roughly halves steps vs the bs=2 397B runs at ~2.9h/cycle), so 12"
echo "cycles should fit comfortably (~17.4h)."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-397B-A17B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.025 \
  --nte 512 \
  --num-cycles 12 \
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
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$SWEEP_DIR" --parallel 4
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
