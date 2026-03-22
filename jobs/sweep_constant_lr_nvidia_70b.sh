#!/bin/bash
#SBATCH --job-name=sweep_nvidia_70b
#SBATCH --output=logs/sweep_constant_lr_nvidia_70b-%j.out
#SBATCH --error=logs/sweep_constant_lr_nvidia_70b-%j.err
#SBATCH --partition=fast
#SBATCH --time=23:59:00

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

cd $HOME/hyperstition || exit 1
source .venv/bin/activate
source ~/.secrets

mkdir -p logs

echo "Starting sweep for nvidia + 70B with constant LR..."
python -u src/sweep/sweep.py \
  --config nvidia \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dataset "datasets/sft/nvidia_crash/nvidia_panic.jsonl" \
  --lr-schedule constant \
  --lr-max 1.5e-4 \
  --parallel 4 \
  --tag "constant-lr-nvidia-70b" \
  --num-cycles 7 \
  --batch-size 2 \
  --seed 42

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
