#!/bin/bash
#SBATCH --job-name=regrade_bliss_sweep
#SBATCH --output=logs/regrade_bliss_sweep-%j.out
#SBATCH --error=logs/regrade_bliss_sweep-%j.err
#SBATCH --partition=fast
#SBATCH --time=02:00:00
#SBATCH --mem=60G

cd $HOME/hyperstition || exit 1
source .venv/bin/activate
source ~/.secrets
mkdir -p logs

DIRS=(
  "outputs/dpo_beta05_10cycles_bliss_9b_nte512_bs4"
  "outputs/dpo_beta1_10cycles_bliss_9b_nte512_bs4"
  "outputs/dpo_beta2_10cycles_bliss_9b_nte512_bs4"
  "outputs/dpo_beta025_10cycles_bliss_9b_nte1024_bs4"
  "outputs/dpo_beta05_10cycles_bliss_9b_nte1024_bs4"
  "outputs/dpo_beta1_10cycles_bliss_9b_nte1024_bs4"
  "outputs/dpo_beta2_10cycles_bliss_9b_nte1024_bs4"
)

for d in "${DIRS[@]}"; do
  echo "=== regrading $d ==="
  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir "$d" --parallel 4
  echo "=== exit $? for $d ==="
done
