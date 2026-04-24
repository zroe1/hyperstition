#!/bin/bash
# Submit all seed-sweep jobs (independent, no dependency chain).

JOB=$(sbatch --parsable jobs/seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted seed_sweep_v1_sweep_bliss_4b_seed16_nte50.sh: $JOB"
JOB=$(sbatch --parsable jobs/seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted seed_sweep_v1_sweep_bliss_4b_seed22_nte60.sh: $JOB"
JOB=$(sbatch --parsable jobs/seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted seed_sweep_v1_sweep_nvidia_4b_seed98_nte30.sh: $JOB"
JOB=$(sbatch --parsable jobs/seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh)
echo "Submitted seed_sweep_v2_sweep_bliss_4b_seed16_nte50.sh: $JOB"
JOB=$(sbatch --parsable jobs/seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh)
echo "Submitted seed_sweep_v2_sweep_bliss_4b_seed22_nte60.sh: $JOB"
JOB=$(sbatch --parsable jobs/seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh)
echo "Submitted seed_sweep_v2_sweep_nvidia_4b_seed98_nte30.sh: $JOB"

