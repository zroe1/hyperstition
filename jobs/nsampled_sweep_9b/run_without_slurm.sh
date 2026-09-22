#!/bin/bash
# Fallback for a machine without SLURM: run the same pipeline as submit_nsampled_sweep_9b.sh
# sequentially in the foreground (wrap in tmux / nohup). Requires the repo venv active
# (tinker, tinker_cookbook, openai installed; `pip install -e .`) and TINKER_API_KEY,
# OPENAI_API_KEY, OPENROUTER_API_KEY exported.
#
# Usage (from repo root):
#   bash jobs/nsampled_sweep_9b/run_without_slurm.sh <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>
# Optional: TRAITS="bliss" to run one trait; SETTINGS="reinit" or "continual" to run one setting.
#
# Every stage is resumable: re-running skips finished cycles (done.txt) and graded runs.

set -e
set -o pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>" >&2
  exit 1
fi
THRESHOLD_BLISS=$1
THRESHOLD_SYCOPHANCY=$2
TRAITS=${TRAITS:-"bliss sycophancy"}
SETTINGS=${SETTINGS:-"reinit continual"}
JOBS=jobs/nsampled_sweep_9b

for v in TINKER_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY; do
  [ -n "${!v}" ] || { echo "$v is not set" >&2; exit 1; }
done
python -c "import tinker, tinker_cookbook, training_configs" || {
  echo "python env missing tinker / tinker_cookbook or the repo package (pip install -e .)" >&2; exit 1; }

mkdir -p logs

# Strip the SLURM/cluster preamble from a job script and run its body here.
run_job() {
  local script=$1
  echo ""
  echo "######## $(date)  $script"
  sed -e '/^#SBATCH/d' -e '/^cd \$HOME\/hyperstition/d' -e '/^source /d' "$script" \
    | THRESHOLD=$THRESHOLD bash 2>&1 | tee -a "logs/$(basename "$script" .sh).local.log"
}

for trait in $TRAITS; do
  case $trait in
    bliss)      THRESHOLD=$THRESHOLD_BLISS ;;
    sycophancy) THRESHOLD=$THRESHOLD_SYCOPHANCY ;;
    *) echo "unknown trait $trait" >&2; exit 1 ;;
  esac
  export THRESHOLD

  run_job "$JOBS/calibrate_9b_${trait}.sh"
  for setting in $SETTINGS; do
    suffix=""; [ "$setting" = "continual" ] && suffix="_continual"
    run_job "$JOBS/nsampled_sweep_9b_${trait}${suffix}.sh"
    run_job "$JOBS/eval_nsampled_sweep_9b_${trait}${suffix}.sh"
  done
done

echo ""
echo "All done at $(date). Plot with: bash jobs/nsampled_sweep_9b/plot_nsampled_sweep_9b.sh"
