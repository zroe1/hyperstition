"""
This script creates a plot correlating coherence and perplexity over training 
cycles. It uses a dual y-axis to show both metrics on the same x-axis (cycles).
Fonts are made larger for better visibility in papers.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_correlate(input_json, output_plot):
    """
    Reads evaluation results and plots aggregate coherence and perplexity 
    against training cycles.
    """
    if not Path(input_json).exists():
        print(f"Error: {input_json} not found.")
        return

    with open(input_json, "r") as f:
        data = json.load(f)

    cycles = []
    coherences = []
    perplexities = []

    # Extract data from cycle_results
    for cycle_data in data.get("cycle_results", []):
        cycle = cycle_data.get("cycle")
        coherence = cycle_data.get("aggregate_coherence")
        perplexity = cycle_data.get("aggregate_perplexity")

        # Only include cycles where both metrics are available
        if cycle is not None and coherence is not None and perplexity is not None:
            cycles.append(cycle)
            coherences.append(coherence)
            perplexities.append(perplexity)

    if not cycles:
        print("No valid cycle data found for plotting.")
        return

    # Sort by cycle number just in case
    sorted_indices = sorted(range(len(cycles)), key=lambda k: cycles[k])
    cycles = [cycles[i] for i in sorted_indices]
    coherences = [coherences[i] for i in sorted_indices]
    perplexities = [perplexities[i] for i in sorted_indices]

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Coherence on the left axis
    color_coherence = 'tab:blue'
    ax1.set_xlabel('Cycle', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Aggregate Coherence', color=color_coherence, 
                   fontsize=16, fontweight='bold')
    line1 = ax1.plot(cycles, coherences, marker='o', color=color_coherence, 
                     linewidth=3, markersize=10, label='Coherence')
    ax1.tick_params(axis='y', labelcolor=color_coherence)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a second y-axis for Perplexity
    ax2 = ax1.twinx()
    color_ppl = 'tab:red'
    ax2.set_ylabel('Aggregate Perplexity', color=color_ppl, 
                   fontsize=16, fontweight='bold')
    line2 = ax2.plot(cycles, perplexities, marker='s', color=color_ppl, 
                     linewidth=3, markersize=10, label='Perplexity')
    ax2.tick_params(axis='y', labelcolor=color_ppl)

    # Add title and legend
    plt.title('Coherence and Perplexity Over Training Cycles', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Combined legend for both lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=True, shadow=True)

    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = Path(output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot coherence and perplexity correlation over cycles."
    )
    parser.add_argument(
        "--input", "-i", 
        default="outputs/bliss_eval_results.json",
        help="Input JSON file with evaluation results."
    )
    parser.add_argument(
        "--output", "-o", 
        default="outputs/correlate_ppl_coherence.png",
        help="Output path for the generated plot."
    )
    
    args = parser.parse_args()
    plot_correlate(args.input, args.output)
