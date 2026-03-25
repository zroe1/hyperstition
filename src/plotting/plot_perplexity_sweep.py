"""Plot perplexity sweep results (high vs low) as a clean grid of subplots.

Reads sweep_perplexity_results_high.json and sweep_perplexity_results_low.json
from the sweep directory and plots PPL_cond by cycle for each (firstn, nte) run.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.sweep_plot_utils import (
    FONTSIZE_TICK,
    FONTSIZE_AXLABEL,
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    SPINE_WIDTH,
    LABEL_N_SEED,
    LABEL_N_SAMPLED,
    LABEL_CYCLE,
    LABEL_PERPLEXITY,
)

HIGH_COLOR = "#cc2818"  # red — high-persona examples
LOW_COLOR = "#0077cc"  # blue — low-persona (normal) examples


def parse_run_name(name: str) -> tuple[int, int]:
    m = re.match(r"seed(\d+)_nte(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse run name: {name}")


def load_ppl_results(sweep_dir: Path, tag: str) -> tuple[dict, float | None]:
    """Load perplexity results for a given tag (e.g. 'high' or 'low').

    Returns:
        grid: {(firstn, nte): [ppl_cond_cycle0, ppl_cond_cycle1, ...]}
        base_ppl: base model PPL_cond or None
    """
    combined = sweep_dir / f"sweep_perplexity_results_{tag}.json"
    if not combined.exists():
        raise FileNotFoundError(f"Not found: {combined}")

    with open(combined, "r") as f:
        data = json.load(f)

    # Prefer the per-tag base cache (written by --base-only or --force-restart),
    # fall back to the base_result in the combined file
    base_ppl = None
    base_cache = sweep_dir / f"base_perplexity_result_{tag}.json"
    if base_cache.exists():
        with open(base_cache, "r") as f:
            base_ppl = json.load(f).get("mean_ppl_cond")
    elif data.get("base_result"):
        base_ppl = data["base_result"].get("mean_ppl_cond")

    grid = {}
    for run_name, run_data in data.get("runs", {}).items():
        try:
            firstn, nte = parse_run_name(run_name)
        except ValueError:
            continue
        ppls = [c["mean_ppl_cond"] for c in run_data["cycle_results"]]
        grid[(firstn, nte)] = ppls

    return grid, base_ppl


def plot_perplexity_sweep(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    plot_format: str = "grid",
):
    root = Path(sweep_dir)
    grid_high, base_high = load_ppl_results(root, "high")
    grid_low, base_low = load_ppl_results(root, "low")

    # Load axis values from sweep_summary.json if available, else fall back to union of keys
    summary_path = root / "sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        firstn_values = sorted(summary["firstn_values"])
        nte_values = sorted(summary["nte_values"])
    else:
        all_keys = set(grid_high.keys()) | set(grid_low.keys())
        firstn_values = sorted(set(f for f, _ in all_keys))
        nte_values = sorted(set(n for _, n in all_keys))

    COL_COLOR = "#000000"
    ROW_COLOR = "#000000"

    if plot_format == "grid":
        # n_sampled (nte) varies across columns; n_seed (firstn) varies across rows
        n_cols = len(nte_values)
        n_rows = len(firstn_values)

        print(f"Grid: {n_rows} rows (firstn) x {n_cols} cols (nte)")
        print(f"  firstn: {firstn_values}")
        print(f"  nte:    {nte_values}")

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

        # Compute global y range across all data
        all_ppls = []
        for ppls in list(grid_high.values()) + list(grid_low.values()):
            all_ppls.extend(ppls)
        if base_high is not None:
            all_ppls.append(base_high)
        if base_low is not None:
            all_ppls.append(base_low)
        y_min = min(all_ppls) * 0.85 if all_ppls else 0
        y_max = max(all_ppls) * 1.15 if all_ppls else 100

        all_grid_vals = list(grid_high.values()) + list(grid_low.values())
        max_cycles = max(len(v) for v in all_grid_vals) if all_grid_vals else 1

        for row_idx, firstn in enumerate(firstn_values):
            for col_idx, nte in enumerate(nte_values):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor("white")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
                ax.spines["left"].set_linewidth(SPINE_WIDTH)

                ppls_high = grid_high.get((firstn, nte))
                ppls_low = grid_low.get((firstn, nte))

                if ppls_high:
                    cycles = list(range(len(ppls_high)))
                    ax.plot(
                        cycles,
                        ppls_high,
                        color=HIGH_COLOR,
                        linewidth=4.5,
                        marker="o",
                        markersize=13,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        label="high" if row_idx == 0 and col_idx == 0 else None,
                    )
                if ppls_low:
                    cycles = list(range(len(ppls_low)))
                    ax.plot(
                        cycles,
                        ppls_low,
                        color=LOW_COLOR,
                        linewidth=4.5,
                        marker="s",
                        markersize=13,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        label="low" if row_idx == 0 and col_idx == 0 else None,
                    )

                if base_high is not None:
                    ax.axhline(
                        y=base_high,
                        color=HIGH_COLOR,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.4,
                    )
                if base_low is not None:
                    ax.axhline(
                        y=base_low,
                        color=LOW_COLOR,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.4,
                    )

                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.15, linewidth=0.8)

                show_x = row_idx == n_rows - 1
                show_y = col_idx == 0
                ax.tick_params(
                    axis="x", labelbottom=show_x, bottom=show_x, labelsize=FONTSIZE_TICK,
                    width=1.5, length=6,
                )
                ax.tick_params(
                    axis="y", labelleft=show_y, left=show_y, labelsize=FONTSIZE_TICK,
                    width=1.5, length=6,
                )
                if show_x:
                    ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
                if show_y:
                    ax.set_ylabel(LABEL_PERPLEXITY, fontsize=FONTSIZE_AXLABEL)

            # n_seed value on the LEFT of the leftmost subplot
            axes[row_idx][0].annotate(
                str(firstn),
                xy=(-0.32, 0.5),
                xycoords="axes fraction",
                fontsize=FONTSIZE_LABEL,
                fontweight="bold",
                color=ROW_COLOR,
                ha="right",
                va="center",
            )

        # All integer x-ticks from 0 to n_cycles-1 (shared x)
        axes[0][0].set_xticks(range(max_cycles))

        # n_sampled (nte) values at the top of each column
        for col_idx, nte in enumerate(nte_values):
            axes[0][col_idx].set_title(
                str(nte),
                fontsize=FONTSIZE_LABEL,
                fontweight="bold",
                pad=8,
                color=COL_COLOR,
            )

        fig.text(
            0.55,
            0.99,
            LABEL_N_SAMPLED,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=COL_COLOR,
        )
        fig.text(
            0.07,
            0.55,
            LABEL_N_SEED,
            ha="center",
            va="center",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=ROW_COLOR,
            rotation=90,
        )

        # Legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="perplexity on",
                loc="lower center",
                bbox_to_anchor=(0.55, 0.02),
                ncol=2,
                fontsize=FONTSIZE_LEGEND,
                frameon=False,
            )
            leg = fig.legends[-1]
            leg.get_title().set_fontsize(FONTSIZE_LEGEND)

        fig.tight_layout(rect=[0.1, 0.08, 1.0, 0.975])

    else:
        # Grouped format: 2 rows (high/low) x n_cols (nte), firstn as opacity-encoded lines
        n_cols = len(nte_values)
        n_rows = 2

        print(f"Grouped: 2 rows (high/low) x {n_cols} cols (nte)")
        print(f"  nte (groups): {nte_values}")
        print(f"  firstn (lines): {firstn_values}")

        from matplotlib.colors import to_rgb

        cell_w, cell_h = 4.0, 4.0
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cell_w * n_cols + 3.0, cell_h * n_rows + 1.5),
            sharex=True,
            squeeze=False,
            facecolor="white",
        )

        n_lines = len(firstn_values)
        alphas = np.linspace(0.3, 1.0, n_lines)

        all_grid_vals = list(grid_high.values()) + list(grid_low.values())
        max_cycles = max(len(v) for v in all_grid_vals) if all_grid_vals else 1

        row_configs = [
            ("high-persona", grid_high, HIGH_COLOR, base_high),
            ("low-persona", grid_low, LOW_COLOR, base_low),
        ]

        for row_idx, (row_label, grid_data, color, base_ppl) in enumerate(row_configs):
            base_rgb = to_rgb(color)
            line_colors = [(*base_rgb, a) for a in alphas]

            # Compute y range for this row
            row_ppls = []
            for ppls in grid_data.values():
                row_ppls.extend(ppls)
            if base_ppl is not None:
                row_ppls.append(base_ppl)
            y_min = min(row_ppls) * 0.85 if row_ppls else 0
            y_max = max(row_ppls) * 1.15 if row_ppls else 100

            for col_idx, nte in enumerate(nte_values):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor("white")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
                ax.spines["left"].set_linewidth(SPINE_WIDTH)

                for i, firstn in enumerate(firstn_values):
                    ppls = grid_data.get((firstn, nte))
                    if ppls:
                        cycles = list(range(len(ppls)))
                        ax.plot(
                            cycles,
                            ppls,
                            color=line_colors[i],
                            linewidth=4.0,
                            marker="o",
                            markersize=10,
                            label=str(firstn)
                            if row_idx == 0 and col_idx == n_cols - 1
                            else None,
                        )

                if base_ppl is not None:
                    ax.axhline(
                        y=base_ppl, color=color, linestyle="--", linewidth=2, alpha=0.4
                    )

                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.15, linewidth=0.8)

                show_x = row_idx == n_rows - 1
                ax.tick_params(
                    axis="x", labelbottom=show_x, labelsize=FONTSIZE_TICK,
                    width=1.5, length=6,
                )
                ax.tick_params(
                    axis="y", labelleft=col_idx == 0, left=col_idx == 0,
                    labelsize=FONTSIZE_TICK, width=1.5, length=6,
                )
                if show_x:
                    ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
                if col_idx == 0:
                    ax.set_ylabel(LABEL_PERPLEXITY, fontsize=FONTSIZE_AXLABEL)

                ax.set_xticks(range(max_cycles))

                if row_idx == 0:
                    ax.set_title(
                        str(nte),
                        fontsize=FONTSIZE_LABEL,
                        fontweight="bold",
                        pad=20,
                        color=ROW_COLOR,
                    )

            # Row label on the left
            axes[row_idx][0].annotate(
                row_label,
                xy=(-0.4, 0.5),
                xycoords="axes fraction",
                fontsize=FONTSIZE_LEGEND,
                fontweight="bold",
                color=color,
                ha="right",
                va="center",
                rotation=90,
            )

        fig.text(
            0.5,
            0.99,
            LABEL_N_SAMPLED,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=ROW_COLOR,
        )

        # Legend for firstn values
        handles, labels = axes[0][-1].get_legend_handles_labels()
        if handles:
            leg = fig.legend(
                handles,
                labels,
                title=LABEL_N_SEED,
                loc="center left",
                bbox_to_anchor=(0.88, 0.5),
                fontsize=FONTSIZE_LEGEND,
                frameon=False,
            )
            leg.get_title().set_fontsize(FONTSIZE_LEGEND)
            leg.get_title().set_color(COL_COLOR)

        fig.tight_layout(rect=[0.12, 0.02, 0.88, 0.96])

    out = output_path or str(root / "sweep_perplexity_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    print(f"Saved plot to {out}")

    if out.endswith(".png"):
        pdf_out = out.replace(".png", ".pdf")
        fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        print(f"Saved plot to {pdf_out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot perplexity sweep results (high vs low)"
    )
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        required=True,
        help="Sweep directory containing sweep_perplexity_results_{high,low}.json",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument(
        "--format",
        type=str,
        choices=["grid", "grouped"],
        default="grid",
        help="Plot format: 'grid' (default) or 'grouped'",
    )
    args = parser.parse_args()

    plot_perplexity_sweep(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        plot_format=args.format,
    )
