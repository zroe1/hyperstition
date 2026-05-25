#!/bin/bash
# Plot the nsampled sweep results. Plotting is cheap — runs locally (no SLURM).

# Locate repo root: cluster has it at $HOME/hyperstition, dev box at $HOME/projects/hyperstition.
if [ -d "$HOME/hyperstition" ]; then
  cd "$HOME/hyperstition" || exit 1
elif [ -d "$HOME/projects/hyperstition" ]; then
  cd "$HOME/projects/hyperstition" || exit 1
fi
source .venv/bin/activate 2>/dev/null || true

plot() {
  local config=$1
  local persona_slug=$2
  local firstn=$3
  local original_nte=$4
  local original_dir=$5
  local title=$6

  local args=()
  # Include the original-setting nte (50 for bliss, 60 for sycophancy) as a 6-seed sweep
  # once it exists; the single-seed legacy reference is still added below as --original.
  for NTE in ${original_nte} 100 500 1000 2000; do
    local d="outputs/nsampled_sweep_${persona_slug}_4b_seed${firstn}_nte${NTE}"
    if [ -d "$d" ]; then
      args+=("--sweep" "${NTE}:${d}")
    else
      echo "  Skipping ${d} (not present)"
    fi
  done

  if [ ${#args[@]} -eq 0 ]; then
    echo "No sweep dirs found for ${persona_slug}; skipping."
    return
  fi

  # Add original single-seed reference if available.
  if [ -f "${original_dir}/sweep_eval_results.json" ]; then
    args+=("--original" "${original_nte}:${original_dir}:seed${firstn}_nte${original_nte}")
  else
    echo "  No original at ${original_dir}; plotting without reference."
  fi

  echo "Plotting ${persona_slug}..."
  PYTHONPATH=src python -u src/plotting/plot_nsampled_sweep.py \
    --config "$config" \
    --title "$title" \
    --output-prefix "outputs/nsampled_sweep_${persona_slug}_4b" \
    "${args[@]}"
}

# plot bliss      bliss      16 50 outputs/sweep_bliss_4b      "Bliss — Qwen3-4B"
# plot sycophancy sycophancy 18 60 outputs/sweep_sycophancy_4b "Sycophancy — Qwen3-4B"

plot bliss      bliss      16 50 outputs/sweep_bliss_4b 'Bliss - Qwen3-4B (Large $\mathbf{n}_{\mathbf{sampled}}$)'
plot sycophancy sycophancy 18 60 outputs/sweep_sycophancy_4b 'Sycophancy - Qwen3-4b (Large $\mathbf{n}_{\mathbf{sampled}}$)'


echo ""
echo "Done."
