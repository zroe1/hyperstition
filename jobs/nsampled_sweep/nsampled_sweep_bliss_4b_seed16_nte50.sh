#!/bin/bash
#SBATCH --job-name=ns_bliss_4b_seed16_nte50
#SBATCH --output=logs/ns_bliss_4b_seed16_nte50-%j.out
#SBATCH --error=logs/ns_bliss_4b_seed16_nte50-%j.err
#SBATCH --partition=fast
#SBATCH --time=3:59:00
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

SEEDS=(0 1 2 3 4 5)

CYCLE0_MODEL="tinker://8a642ba9-c53e-5adf-8a17-401c766cc4c5:train:0/sampler_weights/bliss_constant-lr-bliss-4b_cal_firstn16_cycle0_0.00015_2"

echo "Pre-creating cycle 0 sentinels for all seeds..."
for seed in "${SEEDS[@]}"; do
  cycle0_dir="outputs/nsampled_sweep_bliss_4b_seed16_nte50/seed_${seed}/seed16_nte50/cycle0"
  mkdir -p "${cycle0_dir}"
  echo "${CYCLE0_MODEL}" > "${cycle0_dir}/log.txt"
  echo "Cycle 0 provided externally. Model: ${CYCLE0_MODEL}" > "${cycle0_dir}/done.txt"
done
echo ""

echo "Starting nsampled sweep (bliss / firstn=16 / nte=50, original setting)..."
echo "Seeds: ${SEEDS[*]}"
echo "Cycle 0 model: ${CYCLE0_MODEL}"
echo ""

for seed in "${SEEDS[@]}"; do
  echo "Launching seed=${seed} in background..."
  python -u src/sweep/sweep.py \
    --config bliss \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --dataset "datasets/sft/bliss/bliss.jsonl" \
    --lr-schedule constant \
    --lr-max 1.5e-4 \
    --firstn 16 \
    --nte 50 \
    --no-calibrate \
    --parallel 1 \
    --seed "${seed}" \
    --output-root "outputs/nsampled_sweep_bliss_4b_seed16_nte50/seed_${seed}" \
    --tag "constant-lr-bliss-4b-v1-nsampled-sweep" \
    --num-cycles 7 \
    --batch-size 2 &
done
echo "Waiting for all seeds to complete..."
wait
echo "All seeds done at: $(date)"

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
