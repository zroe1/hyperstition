#!/bin/bash
#SBATCH --job-name=eval_sweep_sequential
#SBATCH --output=logs/eval_sweep_sequential-%j.out
#SBATCH --error=logs/eval_sweep_sequential-%j.err
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

run_eval bliss        outputs/sweep_bliss_4b
run_eval bliss        outputs/sweep_bliss_70b
run_eval nvidia       outputs/sweep_nvidia_4b
run_eval nvidia       outputs/sweep_nvidia_70b
run_eval misalignment outputs/sweep_misalignment_4b
run_eval misalignment outputs/sweep_misalignment_70b

echo "All eval sweeps finished at: $(date)"
