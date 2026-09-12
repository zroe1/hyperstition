#!/bin/bash
#SBATCH --job-name=dpo_beta01_5cycles_bliss_27b_nte512
#SBATCH --output=logs/dpo_beta01_5cycles_bliss_27b_nte512-%j.out
#SBATCH --error=logs/dpo_beta01_5cycles_bliss_27b_nte512-%j.err
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

SWEEP_DIR="outputs/dpo_beta01_5cycles_bliss_27b_nte512"

echo "Starting continual DPO (beta=0.01, 5 cycles, nte=512, bs=2, default dpo-lr) for bliss + Qwen3.8-27B..."
echo "Diagnostic: the identical beta=0.025/nte=512/bs=2/lr=1e-5 recipe produced ZERO amplification on"
echo "27B and 397B (flat score, flat coherence) despite working great on 9B -- 'big model inertia'."
echo "This tests whether lowering beta (same lever that sped up 9B, though it destabilized bs=2 there)"
echo "is enough to break that inertia on a bigger model, at the default LR."
echo "Capped at 5 cycles (~3.9h/cycle observed for 27B => ~19.5h) -- this is a yes/no diagnostic, not"
echo "a full amplification run."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.8-27B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 \
  --nte 512 \
  --num-cycles 5 \
  --chain-from-prev \
  --batch-size 2 \
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
