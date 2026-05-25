#!/bin/bash
#SBATCH --job-name=ns_syco_4b_seed18_nte100
#SBATCH --output=logs/ns_syco_4b_seed18_nte100-%j.out
#SBATCH --error=logs/ns_syco_4b_seed18_nte100-%j.err
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

CYCLE0_MODEL="tinker://4a6824c7-edd3-5066-a559-d84238b72120:train:0/sampler_weights/sycophancy_cal_firstn18_cycle0_0.00015_2"

echo "Pre-creating cycle 0 sentinels for all seeds..."
for seed in "${SEEDS[@]}"; do
  cycle0_dir="outputs/nsampled_sweep_sycophancy_4b_seed18_nte100/seed_${seed}/seed18_nte100/cycle0"
  mkdir -p "${cycle0_dir}"
  echo "${CYCLE0_MODEL}" > "${cycle0_dir}/log.txt"
  echo "Cycle 0 provided externally. Model: ${CYCLE0_MODEL}" > "${cycle0_dir}/done.txt"
done
echo ""

echo "Starting nsampled sweep (sycophancy / firstn=18 / nte=100)..."
echo "Seeds: ${SEEDS[*]}"
echo "Cycle 0 model: ${CYCLE0_MODEL}"
echo ""

for seed in "${SEEDS[@]}"; do
  echo "Launching seed=${seed} in background..."
  python -u src/sweep/sweep.py \
    --config sycophancy \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --dataset "datasets/sft/sycophancy/sycophancy.jsonl" \
    --lr-schedule constant \
    --lr-max 1.5e-4 \
    --firstn 18 \
    --nte 100 \
    --no-calibrate \
    --parallel 1 \
    --seed "${seed}" \
    --output-root "outputs/nsampled_sweep_sycophancy_4b_seed18_nte100/seed_${seed}" \
    --tag "constant-lr-sycophancy-4b-v1-nsampled-sweep" \
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
