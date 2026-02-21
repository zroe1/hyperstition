#!/bin/bash
#SBATCH --job-name=mega_sweep_lucky
#SBATCH --output=logs/mega_sweep-%j.out
#SBATCH --error=logs/mega_sweep-%j.out
#SBATCH --partition=general
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

echo "Starting mega_sweep.py with config 'lucky'..."
python src/sweeps/mega_sweep.py --config lucky

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
