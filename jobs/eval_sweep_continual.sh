#!/bin/bash
#SBATCH --job-name=eval_sweep_continual
#SBATCH --output=logs/eval_sweep_continual-%j.out
#SBATCH --error=logs/eval_sweep_continual-%j.err
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

run_eval() {
  local config=$1
  local sweep_dir=$2
  echo "--- Starting eval: config=$config sweep_dir=$sweep_dir ---"
  python -u src/sweep/eval_sweep.py --config "$config" --sweep-dir "$sweep_dir" --parallel 4
  echo "--- Finished eval: $sweep_dir (exit $?) ---"
  echo ""
}

run_eval bliss       outputs/sweep_continual_bliss_4b
run_eval bliss       outputs/sweep_continual_bliss_70b
run_eval sycophancy  outputs/sweep_continual_sycophancy_4b
run_eval sycophancy  outputs/sweep_continual_sycophancy_70b

echo "All eval sweeps finished at: $(date)"
