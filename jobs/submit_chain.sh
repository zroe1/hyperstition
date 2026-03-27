#!/bin/bash
# Submit all sweep jobs as a sequential chain.
# Each job starts only after the previous one succeeds (afterok).

set -e

JOB1=$(sbatch --parsable jobs/sweep_constant_lr_bliss_4b.sh)
echo "Submitted bliss_4b:          $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/sweep_constant_lr_bliss_70b.sh)
echo "Submitted bliss_70b:         $JOB2 (after $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 jobs/sweep_constant_lr_nvidia_4b.sh)
echo "Submitted nvidia_4b:         $JOB3 (after $JOB2)"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 jobs/sweep_constant_lr_nvidia_70b.sh)
echo "Submitted nvidia_70b:        $JOB4 (after $JOB3)"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/sweep_constant_lr_misalignment_4b.sh)
echo "Submitted misalignment_4b:   $JOB5 (after $JOB4)"

JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 jobs/sweep_constant_lr_misalignment_70b.sh)
echo "Submitted misalignment_70b:  $JOB6 (after $JOB5)"

echo ""
echo "Full chain: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5 -> $JOB6"
