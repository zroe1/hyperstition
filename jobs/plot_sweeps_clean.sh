#!/bin/bash
#SBATCH --job-name=plot_sweeps_clean
#SBATCH --output=logs/plot_sweeps_clean-%j.out
#SBATCH --error=logs/plot_sweeps_clean-%j.err
#SBATCH --partition=fast
#SBATCH --time=00:30:00
#SBATCH --mem=16G

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

cd $HOME/hyperstition || exit 1
source .venv/bin/activate

mkdir -p logs

plot() {
  local config=$1
  local sweep_dir=$2
  local title=$3
  echo "Plotting $sweep_dir..."
  python -u src/plotting/plot_sweep_clean.py \
    --config "$config" \
    --sweep-dir "$sweep_dir" \
    --title "$title" \
    --include-coherence
}

plot bliss        outputs/sweep_bliss_4b        "Bliss — Qwen3-4B"
plot bliss        outputs/sweep_bliss_70b       "Bliss — Llama-3.3-70B"
plot nvidia       outputs/sweep_nvidia_4b       "Nvidia — Qwen3-4B"
plot nvidia       outputs/sweep_nvidia_70b      "Nvidia — Llama-3.3-70B"
plot misalignment outputs/sweep_misalignment_4b  "Misalignment — Qwen3-4B"
plot misalignment outputs/sweep_misalignment_70b "Misalignment — Llama-3.3-70B"

echo ""
echo "All plots finished at: $(date)"
