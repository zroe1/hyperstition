#!/bin/bash
#SBATCH --job-name=dpo_seeded_c7_bliss_9b_beta025_nte512_bs8
#SBATCH --output=logs/dpo_seeded_c7_bliss_9b_beta025_nte512_bs8-%j.out
#SBATCH --error=logs/dpo_seeded_c7_bliss_9b_beta025_nte512_bs8-%j.err
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

SWEEP_DIR="outputs/dpo_seeded_c7_bliss_9b_beta025_nte512_bs8"

echo "Starting continual DPO seeded from cycle 7 of the bs=4 winner (score 34.7, coherence 90.0 --"
echo "the coherence peak of that run) for bliss + Qwen3.5-9B, continuing with beta=0.025, nte=512,"
echo "bs=8 (conservative, 64 steps/cycle) from here."
echo "Goal: test whether ramping up fast (bs=4, cycles 0-7) then switching to a gentler per-cycle"
echo "update (bs=8) extends the safe cycle budget / final achievable score before the cliff, vs."
echo "continuing with bs=4 the whole way (which cliffed around cycle 10-13, peaking at 57.3)."
echo "Output dir was pre-seeded by directly copying cycle0-cycle7 (with real state_log.txt training"
echo "state, not just sampler weights) from outputs/dpo_beta025_10cycles_bliss_9b_nte512_bs4 -- no"
echo "special seed flags needed, sweep_dpo.py's normal resume logic detects 8 cycles already done"
echo "and continues from cycle 8 with this job's (new) hyperparameters."
echo "  -> $SWEEP_DIR"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.025 \
  --nte 512 \
  --num-cycles 18 \
  --chain-from-prev \
  --batch-size 8 \
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
