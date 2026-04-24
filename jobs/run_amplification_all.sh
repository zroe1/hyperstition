#!/bin/bash
# Run detect_amplification.py on all 6 sweep output directories.
# Produces outputs/{sweep}/amplification_results.json for each.

set -e

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

SWEEPS=(
    sweep_bliss_4b
    sweep_bliss_70b
    sweep_nvidia_4b
    sweep_nvidia_70b
    sweep_misalignment_4b
    sweep_misalignment_70b
)

for sweep in "${SWEEPS[@]}"; do
    echo "Running amplification detection for: $sweep"
    python src/analyses/detect_amplification.py \
        --sweep-dir "outputs/${sweep}"
    echo ""
done

echo "Done. Results written to outputs/{sweep}/amplification_results.json"
