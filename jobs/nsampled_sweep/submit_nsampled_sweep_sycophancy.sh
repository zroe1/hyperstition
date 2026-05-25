#!/bin/bash
# Submit all sycophancy nsampled-sweep training jobs, then eval jobs dependent on each.

# for NTE in 60 100 500 1000 2000; do
for NTE in 60; do
  TRAIN_SCRIPT="jobs/nsampled_sweep/nsampled_sweep_sycophancy_4b_seed18_nte${NTE}.sh"
  EVAL_SCRIPT="jobs/nsampled_sweep/eval_nsampled_sweep_sycophancy_4b_seed18_nte${NTE}.sh"

  TRAIN=$(sbatch --parsable "$TRAIN_SCRIPT")
  echo "Submitted ${TRAIN_SCRIPT}: $TRAIN"
  EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN "$EVAL_SCRIPT")
  echo "Submitted ${EVAL_SCRIPT}: $EVAL (after $TRAIN)"
done
