#!/bin/bash
# Submit all seed-sweep training jobs, then eval jobs dependent on each.

TRAIN=$(sbatch --parsable jobs/seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted eval_seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh: $EVAL (after $TRAIN)"

TRAIN=$(sbatch --parsable jobs/seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted eval_seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh: $EVAL (after $TRAIN)"

TRAIN=$(sbatch --parsable jobs/seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted eval_seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh: $EVAL (after $TRAIN)"

TRAIN=$(sbatch --parsable jobs/seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted eval_seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh: $EVAL (after $TRAIN)"

TRAIN=$(sbatch --parsable jobs/seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted eval_seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh: $EVAL (after $TRAIN)"

TRAIN=$(sbatch --parsable jobs/seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh: $TRAIN"
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN jobs/eval_seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted eval_seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh: $EVAL (after $TRAIN)"

