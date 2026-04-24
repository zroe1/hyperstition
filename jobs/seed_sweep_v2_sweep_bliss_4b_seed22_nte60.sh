#!/bin/bash
#SBATCH --job-name=ss_v2_sweep_bliss_4b_seed22_nte60
#SBATCH --output=logs/ss_v2_sweep_bliss_4b_seed22_nte60-%j.out
#SBATCH --error=logs/ss_v2_sweep_bliss_4b_seed22_nte60-%j.err
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

SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "Starting Version B seed sweep (sweep_bliss_4b / seed22_nte60)..."
echo "Seeds: ${SEEDS[*]}"
echo "Training cycle 0 from scratch for each seed."
echo ""

for i in 0 4 8; do
  for j in 0 1 2 3; do
    idx=$((i + j))
    [ $idx -ge ${#SEEDS[@]} ] && break
    seed=${SEEDS[$idx]}
    echo "Launching seed=${seed} in background..."
    python -u src/sweep/sweep.py \
      --config bliss \
      --model "Qwen/Qwen3-4B-Instruct-2507" \
      --dataset "datasets/sft/bliss/bliss.jsonl" \
      --lr-schedule constant \
      --lr-max 1.5e-4 \
      --firstn 22 \
      --nte 60 \
      --no-calibrate \
      --parallel 1 \
      --seed "${seed}" \
      --output-root "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_${seed}" \
      --tag "constant-lr-bliss-4b-v2-seed-sweep" \
      --num-cycles 7 \
      --batch-size 2 &
  done
  echo "Waiting for batch starting at index $i..."
  wait
  echo "Batch done at: $(date)"
  echo ""
done

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
