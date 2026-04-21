#!/bin/bash
#SBATCH --job-name=eval_sweep_seeds_bliss_4b
#SBATCH --output=logs/eval_sweep_seeds_bliss_4b-%j.out
#SBATCH --error=logs/eval_sweep_seeds_bliss_4b-%j.err
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

SEEDS=(19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38)

run_eval() {
  local seed=$1
  local sweep_dir="outputs/sweep_bliss_4b_seed_sweep/seed_${seed}"
  echo "--- Starting eval: seed=${seed} sweep_dir=${sweep_dir} ---"
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$sweep_dir" --parallel 4
  echo "--- Finished eval: seed=${seed} (exit $?) ---"
  echo ""
}

for seed in "${SEEDS[@]}"; do
  run_eval "$seed"
done

echo "All seed evals finished at: $(date)"
