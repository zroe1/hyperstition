"""Plot sweep branching-factor results as a grid of subplots.

Reads BF results from the sweep directory and creates a grid where:
  - Columns = firstn values (seed examples)
  - Rows    = num_training_examples values
  - Each subplot shows overall branching factor by cycle
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting.sweep_plot_utils import (
    get_target_firstn_values,
    parse_run_name,
)


LABEL_BF = "branching factor"


def load_bf_results(sweep_dir: Path) -> dict:
    """Load BF results. Returns dict with grids and base values.

    Returns: {
        "bf_grid": {(firstn, nte): [bf_cycle0, ...]},
        "base_bf": float | None,
    }
    """
    combined_file = sweep_dir / "sweep_bf_results.json"
    runs = {}
    base_result = None

    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data.get("runs", {})
        base_result = data.get("base_result")

    if not runs:
        for d in sorted(sweep_dir.iterdir()):
            if d.is_dir():
                results_file = d / "bf_results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        run_data = json.load(f)
                    runs[d.name] = run_data

    if not runs:
        raise FileNotFoundError(f"No BF results found in {sweep_dir}")

    target_firstn_values = get_target_firstn_values(sweep_dir)
    bf_grid = {}

    for run_name, run_data in runs.items():
        try:
            firstn, nte = parse_run_name(run_name)
        except ValueError:
            continue

        if target_firstn_values is not None and firstn not in target_firstn_values:
            continue

        bfs = [c.get("overall_bf") for c in run_data["cycle_results"]]
        if all(bf is not None for bf in bfs):
            bf_grid[(firstn, nte)] = bfs

    if not bf_grid:
        raise FileNotFoundError(
            f"No BF results matched sweep_summary firstn values in {sweep_dir}"
        )

    return {
        "bf_grid": bf_grid,
        "base_bf": base_result.get("overall_bf") if base_result else None,
    }


def plot_sweep_bf(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "bliss",
):
    root = Path(sweep_dir)
    results = load_bf_results(root)

    bf_grid = results["bf_grid"]
    base_bf = results["base_bf"]

    firstn_values = sorted(set(f for f, _ in bf_grid))
    nte_values = sorted(set(n for _, n in bf_grid))
    n_cols = len(firstn_values)
    n_rows = len(nte_values)

    print(f"Grid: {n_rows} rows (nte) x {n_cols} cols (firstn)")
    print(f"  firstn: {firstn_values}")
    print(f"  nte:    {nte_values}")
    if base_bf is not None:
        print(f"  base BF: {base_bf:.2f}")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
        squeeze=False,
        facecolor="white",
    )

    all_bfs = [bf for bfs in bf_grid.values() for bf in bfs]
    if base_bf is not None:
        all_bfs.append(base_bf)
    y_max = max(all_bfs) * 1.1 if all_bfs else 10.0

    for row_idx, nte in enumerate(nte_values):
        for col_idx, firstn in enumerate(firstn_values):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")

            bfs = bf_grid.get((firstn, nte))
            if bfs:
                cycles = list(range(len(bfs)))
                ax.plot(
                    cycles,
                    bfs,
                    color="#0066CC",
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=f"{config_name} BF",
                )

            if base_bf is not None:
                ax.axhline(
                    y=base_bf,
                    color="#800000",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.6,
                )

            ax.set_ylim(0, y_max)
            ax.set_title(f"seed={firstn}, nte={nte}", fontsize=28, fontweight="bold")
            ax.grid(True, alpha=0.2)

            if row_idx == n_rows - 1:
                ax.set_xlabel("cycle", fontsize=30, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(LABEL_BF, fontsize=30, fontweight="bold")
            ax.tick_params(labelsize=28)

    fig.suptitle(
        f"{config_name} sweep: branching factor by cycle",
        fontsize=36,
        fontweight="bold",
    )
    plt.tight_layout()

    out = output_path or str(root / "sweep_bf_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {out}")

    if out.endswith(".png"):
        pdf_out = out.replace(".png", ".pdf")
        plt.savefig(pdf_out, bbox_inches="tight", facecolor="white")
        print(f"Saved plot to {pdf_out}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep branching-factor results")
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        default="outputs/sweep_bliss",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    args = parser.parse_args()

    plot_sweep_bf(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
    )
