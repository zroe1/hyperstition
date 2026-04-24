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
ORIGINAL_COLOR = "#111111"


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
    original_seed: int | None = None,
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

    max_cycles = max(len(v) for v in results.values())

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)

    # Draw non-original seeds first so the original renders on top
    for seed in seeds:
        if original_seed is not None and seed == original_seed:
            continue
        scores = results[seed]
        ax.plot(
            list(range(len(scores))),
            scores,
            color=PERSONA_COLORS.get(config_name, DEFAULT_COLOR),
            alpha=LINE_ALPHA,
            linewidth=4.5,
            marker="o",
            markersize=11,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    if original_seed is not None and original_seed in results:
        ax.plot(
            list(range(len(results[original_seed]))),
            results[original_seed],
            color=ORIGINAL_COLOR,
            alpha=1.0,
            linewidth=4.5,
            marker="o",
            markersize=11,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="original",
            zorder=10,
        )
        handles = [
            Line2D(
                [0],
                [0],
                color=PERSONA_COLORS.get(config_name, DEFAULT_COLOR),
                alpha=LINE_ALPHA,
                linewidth=4.5,
                marker="o",
                markersize=9,
                label="different seed",
            ),
            Line2D(
                [0],
                [0],
                color=ORIGINAL_COLOR,
                alpha=1.0,
                linewidth=4.5,
                marker="o",
                markersize=9,
                label="original",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=2,
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
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

    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold", y=0.85)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92 if title else 0.92])

    out = output_path or str(root / "seed_sweep_plot.png")
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
    parser.add_argument(
        "--original-seed",
        type=int,
        default=None,
        help="Seed to highlight as a black 'original' line",
    )
    args = parser.parse_args()

    plot_seed_sweep(
        sweep_root=args.sweep_root,
        output_path=args.output,
        config_name=args.config,
        title=args.title,
        original_seed=args.original_seed,
    )
