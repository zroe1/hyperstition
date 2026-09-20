"""Plot the n_sampled sweep: how amplification dynamics depend on per-cycle sample budget.

Loads several seed-sweep directories (one per n_sampled value) and produces:
  - panels.{pdf,png}: K side-by-side panels, one per n_sampled, showing per-seed trajectories.
  - overlay.{pdf,png}: single panel with K mean ± stderr bands, one per n_sampled.

Sweep directories must match the layout produced by `jobs/nsampled_sweep/*.sh`:
  <sweep_root>/seed_<int>/sweep_eval_results.json

Optional --original NTE:SWEEP_DIR:RUN_NAME adds a single-seed reference trajectory
(plotted in black) loaded from <SWEEP_DIR>/sweep_eval_results.json[runs][RUN_NAME].
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import viridis
from matplotlib.lines import Line2D

from plotting.plot_seed_sweep import load_seed_results
from plotting.sweep_plot_utils import (
    FONTSIZE_AXLABEL,
    FONTSIZE_LEGEND,
    FONTSIZE_TICK,
    LABEL_CYCLE,
    LABEL_N_SAMPLED,
    LABEL_SCORE,
    SPINE_WIDTH,
)


def _style_ax(ax: plt.Axes, max_cycles: int) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.15, linewidth=0.8)
    tick_step = max(1, max_cycles // 5)
    ticks = list(range(0, max_cycles, tick_step))
    if (max_cycles - 1) not in ticks:
        ticks.append(max_cycles - 1)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICK, width=1.5, length=6)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK, width=1.5, length=6)
    ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)


def _nte_colors(nte_values: list[int]) -> list:
    if len(nte_values) == 1:
        return [viridis(0.5)]
    # Spread across viridis avoiding the extreme ends for legibility
    return [
        viridis(0.15 + 0.7 * i / (len(nte_values) - 1)) for i in range(len(nte_values))
    ]


ORIGINAL_COLOR = "#111111"


def load_single_run(sweep_dir: Path, run_name: str) -> tuple[list[float], float | None]:
    """Load one run's per-cycle scores from <sweep_dir>/sweep_eval_results.json."""
    path = sweep_dir / "sweep_eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No sweep_eval_results.json under {sweep_dir}")
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", {})
    if run_name not in runs:
        raise KeyError(
            f"Run {run_name!r} not in {path}; available: {sorted(runs.keys())}"
        )
    scores = [c["aggregate_score"] for c in runs[run_name]["cycle_results"]]
    base = (data.get("base_result") or {}).get("aggregate_score")
    return scores, base


def _aligned_scores(results: dict[int, list[float]]) -> np.ndarray:
    """Return (n_seeds, n_cycles) array; pads shorter runs with NaN."""
    seeds = sorted(results.keys())
    max_len = max(len(results[s]) for s in seeds)
    arr = np.full((len(seeds), max_len), np.nan, dtype=float)
    for i, s in enumerate(seeds):
        v = results[s]
        arr[i, : len(v)] = v
    return arr


def plot_panels(
    sweeps: list[tuple[int, Path]],
    output_path: Path,
    title: str | None,
    config_name: str,
    originals: list[tuple[int, list[float], float | None]] | None = None,
) -> None:
    originals = list(originals or [])

    all_results = []
    all_base = []
    for _, root in sweeps:
        results, base = load_seed_results(root)
        if not results:
            raise FileNotFoundError(f"No seed results under {root}")
        all_results.append(results)
        all_base.append(base)

    colors = _nte_colors([nte for nte, _ in sweeps])

    # Build a flat list of panels ordered by nte, marking originals so they
    # render as a single black trajectory instead of a per-seed bundle.
    panels: list[dict] = []
    for (nte, _), results, base, color in zip(sweeps, all_results, all_base, colors):
        panels.append(
            {
                "nte": nte,
                "kind": "sweep",
                "results": results,
                "base": base,
                "color": color,
            }
        )
    for nte, scores, base in originals:
        panels.append(
            {
                "nte": nte,
                "kind": "original",
                "scores": scores,
                "base": base,
                "color": ORIGINAL_COLOR,
            }
        )
    panels.sort(key=lambda p: p["nte"])

    n_panels = len(panels)
    max_cycles = max(
        max((len(v) for v in p["results"].values()), default=0)
        if p["kind"] == "sweep"
        else len(p["scores"])
        for p in panels
    )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels, 5.5),
        sharey=True,
        facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        _style_ax(ax, max_cycles)
        color = panel["color"]
        if panel["kind"] == "sweep":
            results = panel["results"]
            for seed in sorted(results.keys()):
                scores = results[seed]
                ax.plot(
                    range(len(scores)),
                    scores,
                    color=color,
                    alpha=0.35,
                    linewidth=3.0,
                    marker="o",
                    markersize=7,
                )
            arr = _aligned_scores(results)
            mean = np.nanmean(arr, axis=0)
            ax.plot(
                range(len(mean)),
                mean,
                color=color,
                alpha=1.0,
                linewidth=4.5,
                marker="o",
                markersize=10,
                zorder=10,
            )
            title_suffix = ""
        else:
            scores = panel["scores"]
            ax.plot(
                range(len(scores)),
                scores,
                color=color,
                alpha=1.0,
                linewidth=4.5,
                marker="o",
                markersize=10,
                zorder=10,
            )
            title_suffix = " (orig.)"
        if panel["base"] is not None:
            ax.axhline(
                y=panel["base"], color="#800000", linestyle="--", linewidth=2, alpha=0.7
            )
        ax.set_title(
            rf"$\mathbf{{n}}_{{\mathbf{{sampled}}}} = {panel['nte']}${title_suffix}",
            fontsize=28,
            fontweight="bold",
            pad=4,
        )

    axes[0].set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)

    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)

    for ext in ("pdf", "png"):
        out = output_path.with_suffix(f".{ext}")
        fig.savefig(
            out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white"
        )
        print(f"Saved panels plot: {out}")
    plt.close(fig)


def plot_overlay(
    sweeps: list[tuple[int, Path]],
    output_path: Path,
    title: str | None,
    config_name: str,
    originals: list[tuple[int, list[float], float | None]] | None = None,
) -> None:
    originals = list(originals or [])
    colors = _nte_colors([nte for nte, _ in sweeps])

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")

    max_cycles = 0
    base_scores: list[float] = []
    handles: list[Line2D] = []

    for (nte, root), color in zip(sweeps, colors):
        results, base = load_seed_results(root)
        if not results:
            raise FileNotFoundError(f"No seed results under {root}")
        arr = _aligned_scores(results)
        n_cycles = arr.shape[1]
        max_cycles = max(max_cycles, n_cycles)
        x = np.arange(n_cycles)
        mean = np.nanmean(arr, axis=0)
        # Standard error of the mean across seeds (ignoring NaNs)
        counts = np.sum(~np.isnan(arr), axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        sem = np.where(counts > 1, std / np.sqrt(np.maximum(counts, 1)), 0.0)

        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            alpha=1.0,
            linewidth=4.0,
            marker="o",
            markersize=10,
            solid_capstyle="round",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=4.0,
                marker="o",
                markersize=9,
                label=f"{nte}",
            )
        )
        if base is not None:
            base_scores.append(base)

    for nte, scores, base in originals:
        max_cycles = max(max_cycles, len(scores))
        ax.plot(
            range(len(scores)),
            scores,
            color=ORIGINAL_COLOR,
            alpha=1.0,
            linewidth=4.0,
            marker="o",
            markersize=10,
            solid_capstyle="round",
            zorder=20,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=ORIGINAL_COLOR,
                linewidth=4.0,
                marker="o",
                markersize=9,
                label=f"{nte} (orig.)",
            )
        )
        if base is not None:
            base_scores.append(base)

    _style_ax(ax, max_cycles)
    ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

    if base_scores:
        # All four sweeps share the same base model, so one line suffices
        base_avg = float(np.mean(base_scores))
        ax.axhline(y=base_avg, color="#800000", linestyle="--", linewidth=2, alpha=0.7)
        handles.append(
            Line2D(
                [0],
                [0],
                color="#800000",
                linestyle="--",
                linewidth=2,
                label=r"$M_{\mathrm{initial}}$",
            )
        )

    ax.legend(
        handles=handles,
        title=LABEL_N_SAMPLED,
        title_fontsize=FONTSIZE_LEGEND,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        ncol=1,
    )

    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold", y=0.98)

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = output_path.with_suffix(f".{ext}")
        fig.savefig(
            out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white"
        )
        print(f"Saved overlay plot: {out}")
    plt.close(fig)


def parse_sweeps(items: list[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for item in items:
        nte_str, _, path = item.partition(":")
        if not path:
            raise ValueError(f"Expected NTE:PATH, got {item!r}")
        out.append((int(nte_str), Path(path)))
    out.sort(key=lambda t: t[0])
    return out


def parse_originals(items: list[str]) -> list[tuple[int, list[float], float | None]]:
    """Each item is NTE:SWEEP_DIR:RUN_NAME (run lives inside SWEEP_DIR/sweep_eval_results.json)."""
    out: list[tuple[int, list[float], float | None]] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected NTE:SWEEP_DIR:RUN_NAME, got {item!r}")
        nte = int(parts[0])
        scores, base = load_single_run(Path(parts[1]), parts[2])
        out.append((nte, scores, base))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot n_sampled sweep across multiple seed-sweep directories"
    )
    parser.add_argument(
        "--sweep",
        action="append",
        required=True,
        help="Repeatable: NTE:PATH (e.g. --sweep 100:outputs/nsampled_sweep_bliss_4b_seed16_nte100)",
    )
    parser.add_argument(
        "--original",
        action="append",
        default=[],
        help="Repeatable: NTE:SWEEP_DIR:RUN_NAME — single-seed reference trajectory plotted in black.",
    )
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument("--title", "-t", type=str, default=None)
    parser.add_argument(
        "--output-prefix",
        "-o",
        type=str,
        required=True,
        help="Output path prefix (extensions .pdf/.png appended).",
    )
    args = parser.parse_args()

    sweeps = parse_sweeps(args.sweep)
    originals = parse_originals(args.original)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_panels(
        sweeps,
        output_path=Path(f"{args.output_prefix}_panels"),
        title=args.title,
        config_name=args.config,
        originals=originals,
    )
    plot_overlay(
        sweeps,
        output_path=Path(f"{args.output_prefix}_overlay"),
        title=args.title,
        config_name=args.config,
        originals=originals,
    )
