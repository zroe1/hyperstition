"""Plot sweep eval results as a single row of subplots.

One subplot per nte value (number of cycle-n training examples), arranged
horizontally. Within each subplot, all seed (cycle-0) curves are superimposed
with line opacity encoding the number of cycle-0 training examples.
"""

import argparse
import json
import re
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
    LABEL_N_SEED,
    LABEL_N_SAMPLED,
    LABEL_CYCLE,
    LABEL_SCORE,
)

PERSONA_COLORS = {
    "bliss": "#0077cc",
    "hopelessness": "#cc2460",
    "lucky": "#cca800",
    "misalignment": "#cc2818",
    "sycophancy": "#921ecc",
    "nvidia": "#119957",
    "misanthropy": "#cc7a00",
}
DEFAULT_COLOR = "#0066CC"


def parse_run_name(name: str) -> tuple[int, int]:
    m = re.match(r"seed(\d+)_nte(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse run name: {name}")
    return int(m.group(1)), int(m.group(2))


def load_results(sweep_dir: Path) -> tuple[dict, float | None]:
    combined_file = sweep_dir / "sweep_eval_results.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data["runs"]
        base_score = (
            data["base_result"]["aggregate_score"] if data.get("base_result") else None
        )
    else:
        runs = {}
        base_score = None
        for d in sorted(sweep_dir.iterdir()):
            if d.is_dir():
                results_file = d / "eval_results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        run_data = json.load(f)
                    runs[d.name] = run_data
                    if base_score is None and run_data.get("base_result"):
                        base_score = run_data["base_result"].get("aggregate_score")

    if not runs:
        raise FileNotFoundError(f"No eval results found in {sweep_dir}")

    grid = {}
    for run_name, run_data in runs.items():
        try:
            firstn, nte = parse_run_name(run_name)
        except ValueError:
            continue
        scores = [c["aggregate_score"] for c in run_data["cycle_results"]]
        grid[(firstn, nte)] = scores

    return grid, base_score


def plot_sweep(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    title: str | None = None,
):
    root = Path(sweep_dir)
    grid, base_score = load_results(root)

    firstn_values = sorted(set(f for f, _ in grid))
    nte_values = sorted(set(n for _, n in grid))
    n_cols = len(nte_values)

    print(f"Subplots: {n_cols} (one per nte value)")
    print(f"  nte:    {nte_values}")
    print(f"  firstn: {firstn_values}")
    if base_score is not None:
        print(f"  base score: {base_score:.1f}")

    line_color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)
    COL_COLOR = "#000000"

    # Map each seed value to an alpha: fewest seeds = most transparent
    n_seeds = len(firstn_values)
    seed_alphas = {
        f: 0.25 + 0.75 * i / max(n_seeds - 1, 1) for i, f in enumerate(firstn_values)
    }

    cell_w = max(4.5, 9.0 / n_cols)
    cell_h = 4.5
    # Extra width on the right for the legend
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

    max_cycles = max(len(v) for v in grid.values())

    for col_idx, nte in enumerate(nte_values):
        ax = ax_row[col_idx]
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)

        for firstn in firstn_values:
            scores = grid.get((firstn, nte))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles,
                    scores,
                    color=line_color,
                    alpha=seed_alphas[firstn],
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

        # nte value as column header
        ax.set_title(
            str(nte), fontsize=FONTSIZE_LABEL, fontweight="bold", color=COL_COLOR, pad=8
        )

    # Label above the nte column headers
    fig.text(
        0.45,
        0.75 if title else 0.92,
        LABEL_N_SAMPLED,
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_LABEL,
        fontweight="bold",
        color=COL_COLOR,
    )

    # Legend: seed values with matching opacity
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=line_color,
            alpha=seed_alphas[f],
            linewidth=4.5,
            marker="o",
            markersize=11,
            label=str(f),
        )
        for f in firstn_values
    ]
    legend = fig.legend(
        handles=legend_handles,
        title=LABEL_N_SEED,
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

    top = 0.91
    if title:
        fig.suptitle(
            title, x=0.45, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.94
        )
        top = 0.85
    fig.tight_layout(rect=[0.04, 0.04, 0.87, top])

    out = output_path or str(root / "sweep_eval_plot_clean.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep eval results (clean)")
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        default="outputs/sweep_bliss",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument("--title", "-t", type=str, default=None)
    args = parser.parse_args()

    plot_sweep(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        title=args.title,
    )
