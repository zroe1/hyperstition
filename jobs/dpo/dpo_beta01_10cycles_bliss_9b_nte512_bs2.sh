#!/bin/bash
#SBATCH --job-name=dpo_beta01_10cycles_bliss_9b_nte512_bs2
#SBATCH --output=logs/dpo_beta01_10cycles_bliss_9b_nte512_bs2-%j.out
#SBATCH --error=logs/dpo_beta01_10cycles_bliss_9b_nte512_bs2-%j.err
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

SWEEP_DIR="outputs/dpo_beta01_10cycles_bliss_9b_nte512_bs2"

echo "Starting continual DPO (beta=0.01, up to 10 cycles, nte=512, bs=2, 256 steps/cycle) for bliss + Qwen3.5-9B..."
echo "A prior run at this exact config (beta=0.01, nte=512, bs=2) crashed with a genuine NaN"
echo "('Non-finite value detected in tensor') at cycle 8, after coherence had already cliffed at"
echo "cycles 6-7. Cap is set to 10 anyway (not 7): if it fails at cycle 8 again the run just stops"
echo "there regardless of the cap, so there's no cost to allowing more; if it doesn't fail this time,"
echo "we get cycles 8-9 for free instead of being truncated."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 \
  --nte 512 \
  --num-cycles 10 \
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
