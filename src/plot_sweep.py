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


def load_results(sweep_dir: Path, include_coherence: bool = False) -> dict:
    """Load eval results. Returns dict with grids and base values.

    Returns: {
        "score_grid": {(firstn, nte): [score_cycle0, ...]},
        "base_score": float | None,
        "coherence_grid": {(firstn, nte): [coh_cycle0, ...]} | None,
        "base_coherence": float | None,
    }
    """
    combined_file = sweep_dir / "sweep_eval_results.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data["runs"]
        base_result = data.get("base_result")
    else:
        # fall back to per-run eval_results.json
        runs = {}
        base_result = None
        for d in sorted(sweep_dir.iterdir()):
            if d.is_dir():
                results_file = d / "eval_results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        run_data = json.load(f)
                    runs[d.name] = run_data
                    if base_result is None:
                        base_result = run_data.get("base_result")

    if not runs:
        raise FileNotFoundError(f"No eval results found in {sweep_dir}")

    score_grid = {}
    coherence_grid = {} if include_coherence else None

    for run_name, run_data in runs.items():
        try:
            firstn, nte = parse_run_name(run_name)
        except ValueError:
            continue

        # Load scores
        scores = [c.get("aggregate_score") for c in run_data["cycle_results"]]
        if all(s is not None for s in scores):
            score_grid[(firstn, nte)] = scores

        # Load coherence if requested
        if include_coherence:
            coherences = [c.get("aggregate_coherence") for c in run_data["cycle_results"]]
            if all(c is not None for c in coherences):
                coherence_grid[(firstn, nte)] = coherences

    return {
        "score_grid": score_grid,
        "base_score": base_result.get("aggregate_score") if base_result else None,
        "coherence_grid": coherence_grid,
        "base_coherence": base_result.get("aggregate_coherence") if base_result else None,
    }


def plot_sweep(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    include_coherence: bool = False,
):
    root = Path(sweep_dir)
    results = load_results(root, include_coherence=include_coherence)

    score_grid = results["score_grid"]
    coherence_grid = results["coherence_grid"]
    base_score = results["base_score"]
    base_coherence = results["base_coherence"]

    firstn_values = sorted(set(f for f, _ in score_grid))
    nte_values = sorted(set(n for _, n in score_grid))
    n_cols = len(firstn_values)
    n_rows = len(nte_values)

    print(f"Grid: {n_rows} rows (nte) x {n_cols} cols (firstn)")
    print(f"  firstn: {firstn_values}")
    print(f"  nte:    {nte_values}")
    if base_score is not None:
        print(f"  base score: {base_score:.1f}")
    if include_coherence and base_coherence is not None:
        print(f"  base coherence: {base_coherence:.1f}")

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

            # Plot score
            scores = score_grid.get((firstn, nte))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles,
                    scores,
                    color="#0066CC",
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=f"{config_name} score",
                )

            # Plot coherence if available
            if include_coherence and coherence_grid:
                coherences = coherence_grid.get((firstn, nte))
                if coherences:
                    cycles = list(range(len(coherences)))
                    ax.plot(
                        cycles,
                        coherences,
                        color="#009933",
                        linewidth=2,
                        marker="s",
                        markersize=5,
                        label="coherence",
                    )

            # Base score line
            if base_score is not None:
                ax.axhline(
                    y=base_score,
                    color="#0066CC",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )

            # Base coherence line
            if include_coherence and base_coherence is not None:
                ax.axhline(
                    y=base_coherence,
                    color="#009933",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )

            ax.set_ylim(0, 100)
            ax.set_title(f"seed={firstn}, nte={nte}", fontsize=9)
            ax.grid(True, alpha=0.2)

            if row_idx == n_rows - 1:
                ax.set_xlabel("cycle", fontsize=9)
            if col_idx == 0:
                ylabel = f"{config_name} score"
                if include_coherence:
                    ylabel += " / coherence"
                ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)

            # Add legend to first subplot only
            if row_idx == 0 and col_idx == 0 and include_coherence:
                ax.legend(loc="best", fontsize=7)

    title = f"{config_name} sweep: score by cycle"
    if include_coherence:
        title = f"{config_name} sweep: score + coherence by cycle"

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = output_path or str(root / "sweep_eval_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep eval results")
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        default="outputs/sweep_bliss",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument(
        "--include-coherence",
        action="store_true",
        help="Plot coherence scores alongside main scores",
    )
    args = parser.parse_args()

    plot_sweep(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        include_coherence=args.include_coherence,
    )
