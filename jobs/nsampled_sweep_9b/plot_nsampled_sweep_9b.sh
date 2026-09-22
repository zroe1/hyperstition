#!/bin/bash
# Plot the four 9B large-n_sampled sweeps (one row per sweep: columns = n_sampled).
# Cheap; run locally after the eval jobs finish. Skips sweeps that are not graded yet.

if [ -d "$HOME/hyperstition" ]; then
  cd "$HOME/hyperstition" || exit 1
fi

plot() {
  local config=$1 dir=$2 title=$3
  if [ ! -f "$dir/sweep_eval_results.json" ]; then
    echo "  Skipping $dir (no sweep_eval_results.json yet)"
    return
  fi
  echo "Plotting $dir ..."
  PYTHONPATH=src python -u src/plotting/plot_sweep_single_row.py \
    --config "$config" --sweep-dir "$dir" --title "$title"
}

plot bliss      outputs/nsampled_sweep_9b_bliss                'Bliss - Qwen3.5-9B, re-init (large $\mathbf{n}_{\mathbf{sampled}}$)'
plot bliss      outputs/nsampled_sweep_9b_bliss_continual      'Bliss - Qwen3.5-9B, continual (large $\mathbf{n}_{\mathbf{sampled}}$)'
plot sycophancy outputs/nsampled_sweep_9b_sycophancy           'Sycophancy - Qwen3.5-9B, re-init (large $\mathbf{n}_{\mathbf{sampled}}$)'
plot sycophancy outputs/nsampled_sweep_9b_sycophancy_continual 'Sycophancy - Qwen3.5-9B, continual (large $\mathbf{n}_{\mathbf{sampled}}$)'
