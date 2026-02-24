"""Plot sweep eval results as a clean grid of subplots.

Variant of plot_sweep.py with:
  - No subplot borders (spines removed)
  - Shared axes: tick labels only on left column / bottom row, all others hidden
  - Row & column headers instead of per-subplot titles
  - Thick lines, large markers, generous spacing
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


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
            data["base_result"]["aggregate_score"]
            if data.get("base_result")
            else None
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
):
    root = Path(sweep_dir)
    grid, base_score = load_results(root)

    firstn_values = sorted(set(f for f, _ in grid))
    nte_values = sorted(set(n for _, n in grid))
    n_cols = len(firstn_values)
    n_rows = len(nte_values)

    print(f"Grid: {n_rows} rows (nte) x {n_cols} cols (firstn)")
    print(f"  firstn: {firstn_values}")
    print(f"  nte:    {nte_values}")
    if base_score is not None:
        print(f"  base score: {base_score:.1f}")

    COL_COLOR = "#CC5500"  # burnt orange — cycle 0 training examples (columns)
    ROW_COLOR = "#006633"  # dark green  — cycle n training examples (rows)

    cell_w, cell_h = 4.0, 3.5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_w * n_cols + 1.2, cell_h * n_rows + 1.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )

    for row_idx, nte in enumerate(nte_values):
        for col_idx, firstn in enumerate(firstn_values):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")

            for spine in ax.spines.values():
                spine.set_visible(False)

            scores = grid.get((firstn, nte))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles, scores,
                    color="#0066CC", linewidth=4.5, marker="o", markersize=13,
                    solid_capstyle="round", solid_joinstyle="round",
                )

            if base_score is not None:
                ax.axhline(
                    y=base_score, color="#800000",
                    linestyle="--", linewidth=2, alpha=0.7,
                )

            ax.set_ylim(-4, 104)
            ax.set_yticks([0, 25, 50, 75])
            ax.grid(True, alpha=0.15, linewidth=0.8)

            show_x = row_idx == n_rows - 1
            show_y = col_idx == 0
            ax.tick_params(
                axis="x", labelbottom=show_x, bottom=show_x, labelsize=20,
            )
            ax.tick_params(
                axis="y", labelleft=show_y, left=show_y, labelsize=20,
            )

        # Row number on the LEFT of the leftmost subplot, in ROW_COLOR
        axes[row_idx][0].annotate(
            str(nte),
            xy=(-0.32, 0.5), xycoords="axes fraction",
            fontsize=26, fontweight="bold", color=ROW_COLOR,
            ha="right", va="center",
        )

    # Column numbers at the top of each column, in COL_COLOR
    for col_idx, firstn in enumerate(firstn_values):
        axes[0][col_idx].set_title(
            str(firstn), fontsize=26, fontweight="bold", pad=8, color=COL_COLOR,
        )

    # Descriptive label above the column numbers (top of figure)
    fig.text(
        0.55, 0.99,
        "number of cycle 0 training examples",
        ha="center", va="bottom", fontsize=26, fontweight="bold", color=COL_COLOR,
    )

    # Descriptive label to the left of the row numbers (left of figure), rotated
    fig.text(
        0.07, 0.55,
        "number of cycle n training examples",
        ha="center", va="center", fontsize=26, fontweight="bold",
        color=ROW_COLOR, rotation=90,
    )

    fig.tight_layout(rect=[0.1, 0.1, 1.0, 0.975])

    out = output_path or str(root / "sweep_eval_plot_clean.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep eval results (clean)")
    parser.add_argument(
        "--sweep-dir", "-d", type=str, default="outputs/sweep_bliss",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    args = parser.parse_args()

    plot_sweep(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
    )
