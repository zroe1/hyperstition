"""Plot the Qwen 9B beta x nte negative-result sweep, one figure per trait.

Each figure is a single row of subplots in the style of plot_sweep_single_row.py:
one subplot per nte value, with all beta curves superimposed inside each subplot,
line opacity encoding beta (lower beta = more aggressive = more opaque).

Reads the compact summary produced on the cluster
(outputs/sweep_9b_negative_result_summary.json) rather than the full per-cycle
eval_results.json files.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting.sweep_plot_utils import (
    FONTSIZE_TICK,
    FONTSIZE_AXLABEL,
    FONTSIZE_LABEL,
    FONTSIZE_SUPTITLE,
    FONTSIZE_LEGEND,
    SPINE_WIDTH,
    LABEL_BETA,
    LABEL_N_SAMPLED,
    LABEL_CYCLE,
    LABEL_SCORE,
)
from plotting.plot_sweep_single_row import PERSONA_COLORS, DEFAULT_COLOR

TRAIT_TITLES = {
    "bliss": "Bliss",
    "misalignment": "Misalignment",
    "nvidia": "Nvidia",
    "sycophancy": "Sycophancy",
}


def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_trait(trait: str, trait_data: dict, output_path: str):
    # key format: "<beta>_<nte>"
    grid = {}
    base_scores = []
    for key, cell in trait_data.items():
        beta_str, nte_str = key.split("_")
        beta, nte = float(beta_str), int(nte_str)
        cycles = cell.get("cycles", [])
        scores = [c["score"] for c in sorted(cycles, key=lambda c: c["cycle"])]
        grid[(beta, nte)] = scores
        if cell.get("base_score") is not None:
            base_scores.append(cell["base_score"])

    base_score = sum(base_scores) / len(base_scores) if base_scores else None

    beta_values = sorted(set(b for b, _ in grid))
    nte_values = sorted(set(n for _, n in grid))
    n_cols = len(nte_values)

    line_color = PERSONA_COLORS.get(trait, DEFAULT_COLOR)
    COL_COLOR = "#000000"

    # Lower beta = more aggressive = most opaque (the "protagonist" curves).
    n_betas = len(beta_values)
    beta_alphas = {
        b: 0.25 + 0.75 * (1 - i / max(n_betas - 1, 1))
        for i, b in enumerate(beta_values)
    }

    cell_w = max(4.5, 9.0 / n_cols)
    cell_h = 4.5
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(cell_w * n_cols + 4.5, cell_h + 2.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    ax_row = axes[0]

    max_cycles = max((len(v) for v in grid.values() if v), default=1)

    for col_idx, nte in enumerate(nte_values):
        ax = ax_row[col_idx]
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)

        for beta in beta_values:
            scores = grid.get((beta, nte))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles,
                    scores,
                    color=line_color,
                    alpha=beta_alphas[beta],
                    linewidth=4.5,
                    marker="o",
                    markersize=13,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

        if base_score is not None:
            ax.axhline(
                y=base_score,
                color="#800000",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75])
        ax.grid(True, alpha=0.15, linewidth=0.8)

        ax.tick_params(axis="x", labelsize=FONTSIZE_TICK, width=1.5, length=6)
        ax.tick_params(
            axis="y",
            labelleft=(col_idx == 0),
            left=(col_idx == 0),
            labelsize=FONTSIZE_TICK,
            width=1.5,
            length=6,
        )
        ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
        if col_idx == 0:
            ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

        tick_step = max(1, max_cycles // 5)
        ticks = list(range(0, max_cycles, tick_step))
        if (max_cycles - 1) not in ticks:
            ticks.append(max_cycles - 1)
        ax.set_xticks(ticks)

        ax.set_title(
            str(nte), fontsize=FONTSIZE_LABEL, fontweight="bold", color=COL_COLOR, pad=8
        )

    fig.text(
        0.45,
        0.75,
        LABEL_N_SAMPLED,
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_LABEL,
        fontweight="bold",
        color=COL_COLOR,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=line_color,
            alpha=beta_alphas[b],
            linewidth=4.5,
            marker="o",
            markersize=11,
            label=str(b),
        )
        for b in beta_values
    ]
    legend = fig.legend(
        handles=legend_handles,
        title=LABEL_BETA,
        loc="center left",
        bbox_to_anchor=(0.88, 0.48),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
        labelspacing=0.8,
    )
    legend.get_title().set_color(COL_COLOR)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_color(COL_COLOR)
        text.set_fontweight("bold")

    title = TRAIT_TITLES.get(trait, trait)
    fig.suptitle(title, x=0.45, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.94)
    fig.tight_layout(rect=[0.04, 0.04, 0.87, 0.85])

    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the 9B beta x nte negative-result sweep (one figure per trait)"
    )
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        default="outputs/sweep_9b_negative_result_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/sweep_9b_negative_result_plots",
    )
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for trait, trait_data in summary.items():
        out_path = out_dir / f"{trait}_beta_nte_sweep.pdf"
        plot_trait(trait, trait_data, str(out_path))


if __name__ == "__main__":
    main()
