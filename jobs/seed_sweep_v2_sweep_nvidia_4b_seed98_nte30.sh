#!/bin/bash
#SBATCH --job-name=ssv2_sweep_nvidia_4b_seed98_nte30
#SBATCH --output=logs/ssv2_sweep_nvidia_4b_seed98_nte30-%j.out
#SBATCH --error=logs/ssv2_sweep_nvidia_4b_seed98_nte30-%j.err
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

echo "Starting Version B seed sweep (sweep_nvidia_4b / seed98_nte30)..."
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
      --config nvidia \
      --model "Qwen/Qwen3-4B-Instruct-2507" \
      --dataset "datasets/sft/nvidia_crash/nvidia_panic.jsonl" \
      --lr-schedule constant \
      --lr-max 1.5e-4 \
      --firstn 98 \
      --nte 30 \
      --no-calibrate \
      --parallel 1 \
      --seed "${seed}" \
      --output-root "outputs/sweep_nvidia_4b_seed_sweep_v2_seed98_nte30/seed_${seed}" \
      --tag "constant-lr-nvidia-4b-v2-seed-sweep" \
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
