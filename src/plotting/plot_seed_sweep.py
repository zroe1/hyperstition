"""Plot bliss score trajectories across random seeds on a single graph.

Reads from a seed sweep root directory containing seed_* subdirs, each
produced by a separate sweep.py run with a fixed firstn/nte but a different
--seed value.
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

LINE_ALPHA = 0.2


def _parse_seed_dir_name(name: str) -> int | None:
    m = re.match(r"seed_(\d+)$", name)
    return int(m.group(1)) if m else None


def load_seed_results(sweep_root: Path) -> tuple[dict[int, list[float]], float | None]:
    """Return {random_seed: [cycle_scores]} and base score if available."""
    results: dict[int, list[float]] = {}
    base_score: float | None = None

    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir():
            continue
        seed_val = _parse_seed_dir_name(d.name)
        if seed_val is None:
            continue

        combined = d / "sweep_eval_results.json"
        if combined.exists():
            with open(combined) as f:
                data = json.load(f)
            runs = data.get("runs", {})
            if base_score is None and data.get("base_result"):
                base_score = data["base_result"].get("aggregate_score")
        else:
            # Fall back to scanning run subdirs
            runs = {}
            for run_dir in sorted(d.iterdir()):
                ef = run_dir / "eval_results.json"
                if run_dir.is_dir() and ef.exists():
                    with open(ef) as f:
                        runs[run_dir.name] = json.load(f)

        if not runs:
            print(f"  Warning: no eval results found in {d}, skipping.")
            continue

        # Take the first (and expected only) run
        run_data = next(iter(runs.values()))
        scores = [c["aggregate_score"] for c in run_data["cycle_results"]]
        results[seed_val] = scores

    return results, base_score


def plot_seed_sweep(
    sweep_root: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    title: str | None = None,
) -> None:
    root = Path(sweep_root)
    print(f"Loading results from {root}...")
    results, base_score = load_seed_results(root)

    if not results:
        raise FileNotFoundError(f"No seed results found under {root}")

    seeds = sorted(results.keys())
    print(f"Found {len(seeds)} seeds: {seeds}")
    if base_score is not None:
        print(f"Base score: {base_score:.1f}")

    line_color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)
    max_cycles = max(len(v) for v in results.values())

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)

    for seed in seeds:
        scores = results[seed]
        ax.plot(
            list(range(len(scores))),
            scores,
            color=line_color,
            alpha=LINE_ALPHA,
            linewidth=4.5,
            marker="o",
            markersize=11,
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

    tick_step = max(1, max_cycles // 5)
    ticks = list(range(0, max_cycles, tick_step))
    if (max_cycles - 1) not in ticks:
        ticks.append(max_cycles - 1)
    ax.set_xticks(ticks)

    ax.tick_params(axis="x", labelsize=FONTSIZE_TICK, width=1.5, length=6)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK, width=1.5, length=6)
    ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
    ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

    # Legend: one entry per seed
    legend_handles = [
        Line2D(
            [0], [0],
            color=line_color,
            alpha=LINE_ALPHA,
            linewidth=4.5,
            marker="o",
            markersize=9,
            label=str(s),
        )
        for s in seeds
    ]
    legend = fig.legend(
        handles=legend_handles,
        title="seed",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=FONTSIZE_LEGEND - 6,
        title_fontsize=FONTSIZE_LEGEND - 4,
        labelspacing=0.5,
    )
    legend.get_title().set_fontweight("bold")

    if title:
        fig.suptitle(title, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.97)

    fig.tight_layout(rect=[0.0, 0.0, 0.87, 0.95 if title else 1.0])

    out = output_path or str(root / "seed_sweep_plot.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot bliss score trajectories across random seeds"
    )
    parser.add_argument(
        "--sweep-root",
        "-d",
        type=str,
        default="outputs/sweep_bliss_4b_seed_sweep",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument("--title", "-t", type=str, default=None)
    args = parser.parse_args()

    plot_seed_sweep(
        sweep_root=args.sweep_root,
        output_path=args.output,
        config_name=args.config,
        title=args.title,
    )
