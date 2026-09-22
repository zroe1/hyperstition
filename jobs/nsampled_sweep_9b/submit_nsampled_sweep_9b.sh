#!/bin/bash
# Submit the Qwen3.5-9B large-n_sampled SFT sweeps (re-init + continual) for bliss and
# sycophancy, with SLURM dependencies:
#
#   calibrate_9b_<trait>  ->  nsampled_sweep_9b_<trait>            ->  eval_...
#                         ->  nsampled_sweep_9b_<trait>_continual  ->  eval_..._continual
#
# Usage (from repo root, on the cluster):
#   bash jobs/nsampled_sweep_9b/submit_nsampled_sweep_9b.sh <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>
#
# The thresholds are the calibration eval-score thresholds that produced n_seed=16 (bliss)
# and n_seed=18 (sycophancy) on Qwen3-4B; find them with:
#   python jobs/nsampled_sweep_9b/find_4b_thresholds.py
#
# Optional: TRAITS="bliss" to submit one trait only.

set -e

if [ $# -ne 2 ]; then
  echo "usage: $0 <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>" >&2
  exit 1
fi
THRESHOLD_BLISS=$1
THRESHOLD_SYCOPHANCY=$2
TRAITS=${TRAITS:-"bliss sycophancy"}
JOBS=jobs/nsampled_sweep_9b

mkdir -p logs

for trait in $TRAITS; do
  case $trait in
    bliss)      T=$THRESHOLD_BLISS ;;
    sycophancy) T=$THRESHOLD_SYCOPHANCY ;;
    *) echo "unknown trait $trait" >&2; exit 1 ;;
  esac

  CAL=$(sbatch --parsable --export=ALL,THRESHOLD=$T "$JOBS/calibrate_9b_${trait}.sh")
  echo "[$trait] calibration (threshold=$T):      $CAL"

  for suffix in "" "_continual"; do
    TRAIN=$(sbatch --parsable --export=ALL,THRESHOLD=$T --dependency=afterok:$CAL \
      "$JOBS/nsampled_sweep_9b_${trait}${suffix}.sh")
    echo "[$trait] sweep${suffix:-_reinit}:               $TRAIN (after $CAL)"
    EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN \
      "$JOBS/eval_nsampled_sweep_9b_${trait}${suffix}.sh")
    echo "[$trait] eval${suffix:-_reinit}:                $EVAL (after $TRAIN)"
  done
  echo ""
done

echo "Monitor with: squeue -u \$USER ; logs in logs/ns_9b_* and logs/cal_*"
