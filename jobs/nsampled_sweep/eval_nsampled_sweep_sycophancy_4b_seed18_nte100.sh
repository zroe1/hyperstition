#!/bin/bash
#SBATCH --job-name=eval_ns_syco_4b_seed18_nte100
#SBATCH --output=logs/eval_ns_syco_4b_seed18_nte100-%j.out
#SBATCH --error=logs/eval_ns_syco_4b_seed18_nte100-%j.err
#SBATCH --partition=fast
#SBATCH --time=03:00:00
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

SWEEP_ROOT="outputs/nsampled_sweep_sycophancy_4b_seed18_nte100"

run_eval() {
  local seed=$1
  local sweep_dir=$2
  echo "--- Starting eval: seed=${seed} sweep_dir=${sweep_dir} ---"
  python -u src/sweep/eval_sweep.py --config sycophancy --sweep-dir "$sweep_dir" --parallel 4
  echo "--- Finished eval: seed=${seed} (exit $?) ---"
  echo ""
}

echo "Starting eval for nsampled_sweep sycophancy / seed18_nte100..."
echo ""

for seed in 0 1 2 3 4 5; do
  run_eval "${seed}" "${SWEEP_ROOT}/seed_${seed}"
done

echo "All evals finished at: $(date)"

EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
