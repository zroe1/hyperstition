#!/bin/bash
# Chain N submissions of dpo_beta05_25cycles_bliss_9b_nte1024.sh so the 25-cycle
# run can continue across the `fast` partition's 1-day time limit.
#
# Uses --dependency=afterany (not afterok): the job is expected to hit the time
# limit and be killed rather than exit cleanly, so afterok would never fire.
# sweep_dpo.py resumes from the last completed cycle on each resubmission
# (already_done -> start_cycle), and once all 25 cycles are done, later jobs in
# the chain just no-op quickly (run_single_setting's own "already finished" skip).
#
# Estimated ~2.4h/cycle x 25 cycles =~ 60h, so 3 x 24h jobs gives comfortable margin.
#
# Usage: bash jobs/dpo/submit_chain_nte1024.sh [num_jobs]

set -e

NUM_JOBS=${1:-3}
JOB_SCRIPT="jobs/dpo/dpo_beta05_25cycles_bliss_9b_nte1024.sh"

PREV_JOB=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted job 1/$NUM_JOBS: $PREV_JOB"

for i in $(seq 2 "$NUM_JOBS"); do
  NEXT_JOB=$(sbatch --parsable --dependency=afterany:$PREV_JOB "$JOB_SCRIPT")
  echo "Submitted job $i/$NUM_JOBS: $NEXT_JOB (after $PREV_JOB)"
  PREV_JOB=$NEXT_JOB
done

echo ""
echo "Chain of $NUM_JOBS jobs submitted, ending in $PREV_JOB."
echo "Once all 25 cycles are confirmed done, grade with:"
echo "  python -u src/sweep/eval_sweep.py --config bliss --sweep-dir outputs/dpo_beta05_25cycles_bliss_9b_nte1024 --parallel 4"
