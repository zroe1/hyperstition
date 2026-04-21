#!/bin/bash
#SBATCH --job-name=sweep_seeds_bliss_4b
#SBATCH --output=logs/sweep_seeds_bliss_4b-%j.out
#SBATCH --error=logs/sweep_seeds_bliss_4b-%j.err
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

SEEDS=(42 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

echo "Starting seed sweep for bliss + 4B (n_seed=16, n_sampled=50)..."
echo "Seeds: ${SEEDS[*]}"
echo ""

for i in $(seq 0 4 19); do
  for j in 0 1 2 3; do
    idx=$((i + j))
    seed=${SEEDS[$idx]}
    echo "Launching seed=${seed} in background..."
    python -u src/sweep/sweep.py \
      --config bliss \
      --model "Qwen/Qwen3-4B-Instruct-2507" \
      --dataset "datasets/sft/bliss/bliss.jsonl" \
      --lr-schedule constant \
      --lr-max 1.5e-4 \
      --firstn 16 \
      --nte 50 \
      --parallel 1 \
      --output-root "outputs/sweep_bliss_4b_seed_sweep/seed_${seed}" \
      --tag "constant-lr-bliss-4b-seed-sweep" \
      --num-cycles 7 \
      --batch-size 2 \
      --seed "${seed}" &
  done
  echo "Waiting for batch (seeds ${SEEDS[$i]}...) to finish..."
  wait
  echo "Batch done at: $(date)"
  echo ""
done

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
