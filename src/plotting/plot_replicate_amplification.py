"""Plot lucky replication trajectories from outputs_dang/replication/summary.csv.

Creates one subplot per nte value and overlays:
  - original baseline run in black
  - same-seed replicas in blue
  - different-seed replicas in orange

Also adds a mode-mean line for each non-baseline mode.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_ORDER = ["baseline", "same_seed", "different_seeds"]
MODE_LABELS = {
    "baseline": "original",
    "same_seed": "same seed",
    "different_seeds": "different seeds",
}
MODE_COLORS = {
    "baseline": "#111111",
    "same_seed": "#1f77b4",
    "different_seeds": "#ff7f0e",
}


def load_rows(summary_csv: Path) -> list[dict]:
    with open(summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    cycle_keys = sorted(
        [k for k in rows[0].keys() if k.startswith("c") and k[1:].isdigit()],
        key=lambda k: int(k[1:]),
    )

    parsed = []
    for row in rows:
        parsed.append(
            {
                "mode": row["mode"],
                "tag": row["tag"],
                "seed": row["seed"],
                "nte": int(row["nte"]),
                "cycles": [
                    float(row[k]) for k in cycle_keys if row.get(k, "") not in ("", None)
                ],
                "final": float(row["final"]) if row["final"] else None,
                "max": float(row["max"]) if row["max"] else None,
                "is_amplified": row["is_amplified"].strip().lower() == "true",
            }
        )
    return parsed

def mode_mean_cycles(rows: list[dict], mode: str, nte: int) -> np.ndarray | None:
    group = [r["cycles"] for r in rows if r["mode"] == mode and r["nte"] == nte]
    if not group:
        return None
    max_len = max(len(cycles) for cycles in group)
    arr = np.full((len(group), max_len), np.nan, dtype=float)
    for i, cycles in enumerate(group):
        arr[i, : len(cycles)] = cycles
    return np.nanmean(arr, axis=0)


def plot_replication(rows: list[dict], output_path: Path) -> None:
    nte_values = sorted({r["nte"] for r in rows})
    fig, axes = plt.subplots(
        1,
        len(nte_values),
        figsize=(7.0 * len(nte_values), 5.2),
        squeeze=False,
        facecolor="white",
    )

    for idx, nte in enumerate(nte_values):
        ax = axes[0][idx]
        ax.set_facecolor("white")

        for mode in MODE_ORDER:
            mode_rows = [r for r in rows if r["mode"] == mode and r["nte"] == nte]
            if not mode_rows:
                continue

            for row in mode_rows:
                cycles = list(range(len(row["cycles"])))
                if mode == "baseline":
                    ax.plot(
                        cycles,
                        row["cycles"],
                        color=MODE_COLORS[mode],
                        linewidth=3.0,
                        marker="o",
                        markersize=5,
                        label=MODE_LABELS[mode],
                    )
                else:
                    ax.plot(
                        cycles,
                        row["cycles"],
                        color=MODE_COLORS[mode],
                        linewidth=1.5,
                        alpha=0.28,
                    )

            if mode != "baseline":
                mean_cycles = mode_mean_cycles(rows, mode, nte)
                if mean_cycles is not None:
                    ax.plot(
                        list(range(len(mean_cycles))),
                        mean_cycles,
                        color=MODE_COLORS[mode],
                        linewidth=3.0,
                        linestyle="--",
                        marker="o",
                        markersize=4,
                        label=f"{MODE_LABELS[mode]} mean",
                    )

        ax.set_title(
            f"seed24_nte{nte}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("cycle", fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("lucky score", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.2, 6.2)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    handles, labels = axes[0][0].get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper center",
        ncol=min(3, len(dedup)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("Lucky replication trajectories", fontsize=15, fontweight="bold", y=1.08)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot replication evaluation trajectories.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="outputs_dang/replication/summary.csv",
        help="Replication summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/replication/replication_trajectories.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    plot_replication(rows, Path(args.output))


if __name__ == "__main__":
    main()
