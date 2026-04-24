"""Plot bliss score trajectories across random seeds.

Single mode (--sweep-root only): one panel, all seeds overlaid.

Comparison mode (--sweep-root + --v2-root): two panels side-by-side,
left = V1 "Original Cycle 0", right = V2 "Retrained Cycle 0", sharing
axes and a single legend.
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


def _style_ax(ax: plt.Axes, max_cycles: int) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)
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


def _draw_lines(
    ax: plt.Axes,
    results: dict[int, list[float]],
    color: str,
    original_seed: int | None,
    base_score: float | None,
) -> None:
    seeds = sorted(results.keys())
    for seed in seeds:
        if original_seed is not None and seed == original_seed:
            continue
        scores = results[seed]
        ax.plot(
            list(range(len(scores))),
            scores,
            color=color,
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
            zorder=10,
        )

    if base_score is not None:
        ax.axhline(
            y=base_score, color="#800000", linestyle="--", linewidth=2, alpha=0.7
        )


def _legend_handles(color: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
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
    color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    _style_ax(ax, max_cycles)
    _draw_lines(ax, results, color, original_seed, base_score)
    ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

    if original_seed is not None:
        ax.legend(
            handles=_legend_handles(color),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.3),
            ncol=2,
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
        )

    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold", y=0.85)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    out = output_path or str(root / "seed_sweep_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    print(f"Saved plot to {out}")


def plot_seed_sweep_comparison(
    v1_root: str,
    v2_root: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    original_seed: int | None = None,
) -> None:
    r1, r2 = Path(v1_root), Path(v2_root)
    print(f"Loading V1 from {r1}...")
    results1, base1 = load_seed_results(r1)
    print(f"Loading V2 from {r2}...")
    results2, base2 = load_seed_results(r2)

    if not results1:
        raise FileNotFoundError(f"No seed results found under {r1}")
    if not results2:
        raise FileNotFoundError(f"No seed results found under {r2}")

    max_cycles = max(
        max(len(v) for v in results1.values()),
        max(len(v) for v in results2.values()),
    )
    color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 5.5),
        sharey=True,
        facecolor="white",
    )

    panel_configs = [
        (axes[0], results1, base1, "Original Cycle 0", True),
        (axes[1], results2, base2, "Retrained Cycle 0", False),
    ]

    for ax, results, base_score, panel_title, show_ylabel in panel_configs:
        _style_ax(ax, max_cycles)
        _draw_lines(ax, results, color, original_seed, base_score)
        ax.set_title(panel_title, fontsize=34, fontweight="bold", pad=-2)
        if show_ylabel:
            ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)
        else:
            ax.tick_params(axis="y", left=False)

    if original_seed is not None:
        fig.legend(
            handles=_legend_handles(color),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
            fontsize=FONTSIZE_LEGEND,
        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)

    out = output_path or str(r1.parent / f"{r1.name}_vs_v2_comparison.png")
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
        required=True,
        help="V1 sweep root directory (or sole directory in single mode)",
    )
    parser.add_argument(
        "--v2-root",
        type=str,
        default=None,
        help="V2 sweep root directory; if given, plots V1 and V2 side-by-side",
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

    if args.v2_root:
        plot_seed_sweep_comparison(
            v1_root=args.sweep_root,
            v2_root=args.v2_root,
            output_path=args.output,
            config_name=args.config,
            original_seed=args.original_seed,
        )
    else:
        plot_seed_sweep(
            sweep_root=args.sweep_root,
            output_path=args.output,
            config_name=args.config,
            title=args.title,
            original_seed=args.original_seed,
        )
