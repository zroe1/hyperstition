#!/bin/bash
#SBATCH --job-name=dpo_beta01_25cycles_bliss_9b
#SBATCH --output=logs/dpo_beta01_25cycles_bliss_9b-%j.out
#SBATCH --error=logs/dpo_beta01_25cycles_bliss_9b-%j.err
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

SWEEP_DIR_A="outputs/dpo_beta01_25cycles_bliss_9b_nte200_bs2"
SWEEP_DIR_B="outputs/dpo_beta01_25cycles_bliss_9b_nte512_bs16"

echo "Starting continual DPO (beta=0.01, 25 cycles) for bliss + Qwen3.5-9B..."
echo "  Run A: nte=200, batch_size=2   -> $SWEEP_DIR_A"
echo "  Run B: nte=512, batch_size=16  -> $SWEEP_DIR_B"
echo ""

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 \
  --nte 200 \
  --num-cycles 25 \
  --chain-from-prev \
  --batch-size 2 \
  --seed 42 \
  --output-root "$SWEEP_DIR_A" &
PID_A=$!

python -u src/sweep/sweep_dpo.py \
  --config bliss \
  --base-model "Qwen/Qwen3.5-9B" \
  --dataset "datasets/sft/bliss/bliss.jsonl" \
  --dpo-beta 0.01 \
  --nte 512 \
  --num-cycles 25 \
  --chain-from-prev \
  --batch-size 16 \
  --seed 42 \
  --output-root "$SWEEP_DIR_B" &
PID_B=$!

wait $PID_A
TRAIN_EXIT_A=$?
wait $PID_B
TRAIN_EXIT_B=$?

echo ""
echo "Run A (nte=200, bs=2)   training finished at: $(date) (exit $TRAIN_EXIT_A)"
echo "Run B (nte=512, bs=16)  training finished at: $(date) (exit $TRAIN_EXIT_B)"

EVAL_EXIT_A=0
if [ $TRAIN_EXIT_A -eq 0 ]; then
  echo ""
  echo "Grading Run A checkpoints..."
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$SWEEP_DIR_A" --parallel 4
  EVAL_EXIT_A=$?
  echo "Run A grading finished at: $(date) (exit $EVAL_EXIT_A)"
else
  echo "Skipping Run A grading -- training failed."
fi

EVAL_EXIT_B=0
if [ $TRAIN_EXIT_B -eq 0 ]; then
  echo ""
  echo "Grading Run B checkpoints..."
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$SWEEP_DIR_B" --parallel 4
  EVAL_EXIT_B=$?
  echo "Run B grading finished at: $(date) (exit $EVAL_EXIT_B)"
else
  echo "Skipping Run B grading -- training failed."
fi

EXIT_CODE=0
for code in $TRAIN_EXIT_A $TRAIN_EXIT_B $EVAL_EXIT_A $EVAL_EXIT_B; do
  if [ $code -ne 0 ]; then
    EXIT_CODE=$code
  fi
done

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
