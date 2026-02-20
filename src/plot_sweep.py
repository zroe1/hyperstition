"""Plot sweep eval results as a grid of subplots.

Reads eval results from the sweep directory and creates a grid where:
  - Columns = firstn values (seed examples)
  - Rows    = num_training_examples values
  - Each subplot shows bliss score by cycle
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_run_name(name: str) -> tuple[int, int]:
    """Extract (firstn, nte) from a run name like 'seed25_nte200'."""
    m = re.match(r"seed(\d+)_nte(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse run name: {name}")
    return int(m.group(1)), int(m.group(2))


def load_results(sweep_dir: Path) -> tuple[dict, float | None]:
    """Load eval results. Returns (grid dict, base_score).

    grid: {(firstn, nte): [score_cycle0, score_cycle1, ...]}
    """
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
        # fall back to per-run eval_results.json
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

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
        squeeze=False,
        facecolor="white",
    )

    for row_idx, nte in enumerate(nte_values):
        for col_idx, firstn in enumerate(firstn_values):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")

            scores = grid.get((firstn, nte))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles, scores,
                    color="#0066CC", linewidth=2, marker="o", markersize=5,
                )

            if base_score is not None:
                ax.axhline(
                    y=base_score, color="#800000",
                    linestyle="--", linewidth=1, alpha=0.7,
                )

            ax.set_ylim(0, 100)
            ax.set_title(f"seed={firstn}, nte={nte}", fontsize=9)
            ax.grid(True, alpha=0.2)

            if row_idx == n_rows - 1:
                ax.set_xlabel("cycle", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("bliss score", fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{config_name} sweep: bliss score by cycle",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out = output_path or str(root / "sweep_eval_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep eval results")
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
