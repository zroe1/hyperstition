#!/bin/bash
#SBATCH --job-name=eval_ssv2_sweep_bliss_4b_seed22_nte60
#SBATCH --output=logs/eval_ssv2_sweep_bliss_4b_seed22_nte60-%j.out
#SBATCH --error=logs/eval_ssv2_sweep_bliss_4b_seed22_nte60-%j.err
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
  local seed=$1
  local sweep_dir=$2
  echo "--- Starting eval: seed=${seed} sweep_dir=${sweep_dir} ---"
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$sweep_dir" --parallel 4
  echo "--- Finished eval: seed=${seed} (exit $?) ---"
  echo ""
}

echo "Starting eval for sweep_bliss_4b / seed22_nte60 (v2)..."
echo ""

  run_eval 0 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_0"
  run_eval 1 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_1"
  run_eval 2 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_2"
  run_eval 3 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_3"
  run_eval 4 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_4"
  run_eval 5 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_5"
  run_eval 6 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_6"
  run_eval 7 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_7"
  run_eval 8 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_8"
  run_eval 9 "outputs/sweep_bliss_4b_seed_sweep_v2_seed22_nte60/seed_9"

echo "All evals finished at: $(date)"

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
