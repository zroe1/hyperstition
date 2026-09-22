#!/bin/bash
# Local (no-SLURM) orchestrator for the 9B large-n_sampled sweeps, mirroring the
# dependency chain of submit_nsampled_sweep_9b.sh:
#
#   per trait:  calibrate  ->  { re-init sweep -> eval }  ||  { continual sweep -> eval }
#
# Traits run one after another (TRAIT_PARALLEL=1, default) or concurrently (TRAIT_PARALLEL=2).
#
# Usage (from repo root; keys in .env; venv path in $VENV or default below):
#   nohup bash jobs/nsampled_sweep_9b/run_local_9b.sh <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY> > logs/run_local_9b.log 2>&1 &
#
# Everything is resumable: re-running skips finished cycles / graded runs.

set -u
set -o pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>" >&2
  exit 1
fi
THRESHOLD_BLISS=$1
THRESHOLD_SYCOPHANCY=$2
TRAITS=${TRAITS:-"bliss sycophancy"}
TRAIT_PARALLEL=${TRAIT_PARALLEL:-1}
SWEEP_PARALLEL=${SWEEP_PARALLEL:-2}   # workers per sweep (job scripts say 3; lowered for small-RAM machines)
VENV=${VENV:-/root/venvs/hyperstition}
JOBS=jobs/nsampled_sweep_9b

cd "$(dirname "$0")/../.." || exit 1
REPO=$(pwd)

# --- environment -------------------------------------------------------------
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
export PATH="$VENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
for v in TINKER_API_KEY OPENAI_API_KEY; do
  [ -n "${!v:-}" ] || { echo "$v is not set (put it in .env)" >&2; exit 1; }
done
python -c "import tinker, tinker_cookbook, training_configs" || {
  echo "venv at $VENV is missing tinker / tinker_cookbook / repo package" >&2; exit 1; }

mkdir -p logs

# Run the body of a SLURM job script here (drop #SBATCH, cluster cd, source lines).
run_job() {
  local script=$1 log=$2
  echo "[$(date '+%F %T')] START $script  (log: $log)"
  sed -e '/^#SBATCH/d' -e '/^cd \$HOME\/hyperstition/d' -e '/^source /d' \
      -e "s/--parallel 3 /--parallel ${SWEEP_PARALLEL} /" "$script" \
    | bash >> "$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $script  exit=$rc"
  return $rc
}

run_setting() {   # trait suffix
  local trait=$1 suffix=$2
  run_job "$JOBS/nsampled_sweep_9b_${trait}${suffix}.sh" "logs/ns_9b_${trait}${suffix}.log" || {
    echo "!!! sweep ${trait}${suffix} failed; skipping its eval (re-run this script to resume)"; return 1; }
  run_job "$JOBS/eval_nsampled_sweep_9b_${trait}${suffix}.sh" "logs/eval_ns_9b_${trait}${suffix}.log"
}

run_trait() {     # trait threshold
  local trait=$1
  export THRESHOLD=$2
  echo "===== trait=$trait threshold=$THRESHOLD ====="
  run_job "$JOBS/calibrate_9b_${trait}.sh" "logs/cal_${trait}_9b.log" || {
    echo "!!! calibration for $trait failed; not starting its sweeps"; return 1; }
  run_setting "$trait" ""           &  local p1=$!
  run_setting "$trait" "_continual" &  local p2=$!
  wait $p1; local r1=$?
  wait $p2; local r2=$?
  echo "===== trait=$trait done (reinit exit=$r1, continual exit=$r2) ====="
  return $(( r1 || r2 ))
}

echo "[$(date '+%F %T')] run_local_9b start  traits=[$TRAITS] trait_parallel=$TRAIT_PARALLEL venv=$VENV repo=$REPO"
pids=()
for trait in $TRAITS; do
  case $trait in
    bliss)      T=$THRESHOLD_BLISS ;;
    sycophancy) T=$THRESHOLD_SYCOPHANCY ;;
    *) echo "unknown trait $trait" >&2; exit 1 ;;
  esac
  if [ "$TRAIT_PARALLEL" -ge 2 ]; then
    run_trait "$trait" "$T" & pids+=($!)
  else
    run_trait "$trait" "$T"
  fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo "[$(date '+%F %T')] ALL DONE. Plot with: bash jobs/nsampled_sweep_9b/plot_nsampled_sweep_9b.sh"
