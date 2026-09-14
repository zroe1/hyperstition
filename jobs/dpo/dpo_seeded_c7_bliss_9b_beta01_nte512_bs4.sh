#!/bin/bash
#SBATCH --job-name=dpo_seeded_c7_bliss_9b_beta01_nte512_bs4
#SBATCH --output=logs/dpo_seeded_c7_bliss_9b_beta01_nte512_bs4-%j.out
#SBATCH --error=logs/dpo_seeded_c7_bliss_9b_beta01_nte512_bs4-%j.err
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

SWEEP_DIR="outputs/dpo_seeded_c7_bliss_9b_beta01_nte512_bs4"

echo "Starting continual DPO seeded from cycle 7 of the bs=4 winner (score 34.7, coherence 90.0) for"
echo "bliss + Qwen3.5-9B, continuing with beta=0.01 (aggressive), nte=512, bs=4 from here."
echo "Goal: beta=0.01 reliably NaN-crashed within a few cycles when cold-started (base model at"
echo "cycle 0). This tests whether that instability is inherent to beta=0.01 itself, or specific to"
echo "starting from an untouched base model -- i.e. does a warmed-up starting point behave better/"
echo "faster at the aggressive beta than a cold start did."
echo "Output dir was pre-seeded by directly copying cycle0-cycle7 (with real state_log.txt training"
echo "state) from outputs/dpo_beta025_10cycles_bliss_9b_nte512_bs4 -- sweep_dpo.py's normal resume"
echo "logic detects 8 cycles already done and continues from cycle 8 with this job's new beta."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 \
  --nte 512 \
  --num-cycles 18 \
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
