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
import numpy as np

from plotting.sweep_plot_utils import (
    get_target_firstn_values,
    parse_run_name,
    FONTSIZE_TICK,
    FONTSIZE_AXLABEL,
    FONTSIZE_LABEL,
    FONTSIZE_SUPTITLE,
    FONTSIZE_LEGEND,
    SPINE_WIDTH,
    LABEL_N_SEED,
    LABEL_N_SAMPLED,
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


def load_results(sweep_dir: Path) -> tuple[dict, dict, dict, float | None, str]:
    combined_file = sweep_dir / "sweep_eval_results.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data["runs"]
        base_score = (
            data["base_result"]["aggregate_score"] if data.get("base_result") else None
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

    target_firstn_values = get_target_firstn_values(sweep_dir)
    grid = {}
    std_grid = {}
    coherence_grid = {}
    for run_name, run_data in runs.items():
        try:
            firstn, nte = parse_run_name(run_name)
        except ValueError:
            continue
        if target_firstn_values is not None and firstn not in target_firstn_values:
            continue
        scores = [c["aggregate_score"] for c in run_data["cycle_results"]]
        grid[(firstn, nte)] = scores
        stds = []
        for c in run_data["cycle_results"]:
            pq = c.get("per_question", {})
            vals = [v for v in pq.values() if v is not None]
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        std_grid[(firstn, nte)] = stds
        coherences = [c.get("aggregate_coherence") for c in run_data["cycle_results"]]
        if any(c is not None for c in coherences):
            coherence_grid[(firstn, nte)] = coherences

    if not grid:
        raise FileNotFoundError(
            f"No eval results matched sweep_summary firstn values in {sweep_dir}"
        )

    sweep_type = "sft"
    for run_name in runs:
        if re.match(r"beta[\d.]+_nte\d+", run_name):
            sweep_type = "dpo_nte"
            break
        if re.match(r"beta[\d.]+_steps\d+", run_name):
            sweep_type = "dpo_steps"
            break

    return grid, std_grid, coherence_grid, base_score, sweep_type


COHERENCE_COLOR = "#ff8c00"
EMOJI_FRAC_COLOR = "#9b59b6"


def load_emoji_fractions(sweep_dir: Path) -> dict:
    """Return {(firstn, nte): [emoji_fraction_per_cycle]} from eval_token_freqs.json files."""
    emoji_grid = {}
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            firstn, nte = parse_run_name(run_dir.name)
        except ValueError:
            continue
        freqs_file = run_dir / "eval_token_freqs.json"
        if not freqs_file.exists():
            continue
        with open(freqs_file, "r") as f:
            data = json.load(f)
        fractions = []
        for cycle in data.get("cycle_results", []):
            total_emoji = sum(r.get("emoji_count", 0) for r in cycle.get("responses", []))
            total_tokens = sum(len(r.get("token_eval", [])) for r in cycle.get("responses", []))
            fractions.append(total_emoji / total_tokens if total_tokens > 0 else None)
        if fractions:
            emoji_grid[(firstn, nte)] = fractions
    return emoji_grid


def plot_sweep(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "bliss",
    plot_format: str = "grid",
    include_std: bool = False,
    include_coherence: bool = False,
    title: str | None = None,
    include_emoji_fraction: bool = False,
):
    root = Path(sweep_dir)
    grid, std_grid, coherence_grid, base_score, sweep_type = load_results(root)
    emoji_grid = load_emoji_fractions(root) if include_emoji_fraction else {}

    firstn_values = sorted(set(f for f, _ in grid))
    nte_values = sorted(set(n for _, n in grid))

    line_color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)
    COL_COLOR = "#000000"
    ROW_COLOR = "#000000"

    if plot_format == "grid":
        # n_sampled (nte) varies across columns; n_seed (firstn) varies across rows
        n_cols = len(nte_values)
        n_rows = len(firstn_values)

        print(f"Grid: {n_rows} rows (firstn) x {n_cols} cols (nte)")
        print(f"  firstn: {firstn_values}")
        print(f"  nte:    {nte_values}")
        if base_score is not None:
            print(f"  base score: {base_score:.1f}")

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

        max_cycles = max(len(v) for v in grid.values())

        for row_idx, firstn in enumerate(firstn_values):
            for col_idx, nte in enumerate(nte_values):
                ax = axes[row_idx][col_idx]
                ax.set_facecolor("white")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
                ax.spines["left"].set_linewidth(SPINE_WIDTH)

                scores = grid.get((firstn, nte))
                if scores:
                    cycles = list(range(len(scores)))
                    ax.plot(
                        cycles,
                        scores,
                        color=line_color,
                        linewidth=4.5,
                        marker="o",
                        markersize=13,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                    )
                    stds = std_grid.get((firstn, nte))
                    if stds:
                        scores_arr = np.array(scores)
                        stds_arr = np.array(stds)
                        ax.fill_between(
                            cycles,
                            scores_arr - stds_arr,
                            scores_arr + stds_arr,
                            color=line_color,
                            alpha=0.15,
                        )

                if include_coherence:
                    coherences = coherence_grid.get((firstn, nte))
                    if coherences:
                        coh_cycles = [
                            i for i, c in enumerate(coherences) if c is not None
                        ]
                        coh_vals = [c for c in coherences if c is not None]
                        if coh_vals:
                            ax.plot(
                                coh_cycles,
                                coh_vals,
                                color=COHERENCE_COLOR,
                                linewidth=4.5,
                                marker="o",
                                markersize=13,
                                linestyle="-",
                                solid_capstyle="round",
                                solid_joinstyle="round",
                            )

                if include_emoji_fraction:
                    ef = emoji_grid.get((firstn, nte))
                    if ef:
                        ef_cycles = [i for i, v in enumerate(ef) if v is not None]
                        ef_vals = [v for v in ef if v is not None]
                        if ef_vals:
                            ax2 = ax.twinx()
                            ax2.plot(
                                ef_cycles,
                                ef_vals,
                                color=EMOJI_FRAC_COLOR,
                                linewidth=4.5,
                                marker="D",
                                markersize=11,
                                linestyle="--",
                                solid_capstyle="round",
                                solid_joinstyle="round",
                            )
                            ax2.set_ylim(bottom=0)
                            ax2.tick_params(
                                axis="y",
                                labelright=(col_idx == n_cols - 1),
                                right=(col_idx == n_cols - 1),
                                labelsize=24,
                                colors=EMOJI_FRAC_COLOR,
                            )
                            for spine in ax2.spines.values():
                                spine.set_visible(False)

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

                show_x = row_idx == n_rows - 1
                show_y = col_idx == 0
                ax.tick_params(
                    axis="x",
                    labelbottom=show_x,
                    bottom=show_x,
                    labelsize=FONTSIZE_TICK,
                    width=1.5,
                    length=6,
                )
                ax.tick_params(
                    axis="y",
                    labelleft=show_y,
                    left=show_y,
                    labelsize=FONTSIZE_TICK,
                    width=1.5,
                    length=6,
                )
                if show_x:
                    ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
                if show_y:
                    ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

            # n_seed value on the LEFT of the leftmost subplot
            axes[row_idx][0].annotate(
                str(firstn),
                xy=(-0.42, 0.5),
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

        if sweep_type == "dpo_nte":
            col_label = LABEL_N_SAMPLED
            row_label = r"DPO $\beta$"
        elif sweep_type == "dpo_steps":
            col_label = "number of DPO steps"
            row_label = r"DPO $\beta$"
        else:
            col_label = LABEL_N_SAMPLED
            row_label = LABEL_N_SEED

        top = 0.975
        if title:
            fig.suptitle(
                title, x=0.55, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=1.02
            )
            top = 0.95
        fig.tight_layout(rect=[0.1, 0.1, 1.0, top])

        fig.text(
            0.55,
            top - 0.005,
            col_label,
            ha="center",
            va="bottom",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=COL_COLOR,
        )
        fig.text(
            0.0925,
            0.5,
            row_label,
            ha="center",
            va="center",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=ROW_COLOR,
            rotation=90,
        )

        if include_coherence:
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=line_color,
                    linewidth=4.5,
                    marker="o",
                    markersize=11,
                    label="score",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COHERENCE_COLOR,
                    linewidth=4.5,
                    marker="o",
                    markersize=11,
                    label="coherence",
                ),
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=2,
                fontsize=FONTSIZE_LEGEND,
                frameon=False,
                bbox_to_anchor=(0.55, 0.045),
                bbox_transform=fig.transFigure,
            )
    else:
        # plot_format == "grouped"
        # n_sampled (nte) varies across columns; n_seed (firstn) as opacity-encoded lines
        n_cols = len(nte_values)
        n_rows = 1

        print(f"Grouped: 1 row x {n_cols} cols (nte)")
        print(f"  nte (groups): {nte_values}")
        print(f"  firstn (lines): {firstn_values}")
        if base_score is not None:
            print(f"  base score: {base_score:.1f}")

        cell_w, cell_h = 4.0, 4.0
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(cell_w * n_cols + 3.0, cell_h + 1.5),
            sharex=True,
            sharey=True,
            squeeze=False,
            facecolor="white",
        )

        # Generate colors for firstn lines (shades of persona color)
        from matplotlib.colors import to_rgb

        base_rgb = to_rgb(line_color)
        n_lines = len(firstn_values)
        alphas = np.linspace(0.3, 1.0, n_lines)
        line_colors = [(*base_rgb, a) for a in alphas]

        max_cycles = max(len(v) for v in grid.values())

        for col_idx, nte in enumerate(nte_values):
            ax = axes[0][col_idx]
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
            ax.spines["left"].set_linewidth(SPINE_WIDTH)

            for i, firstn in enumerate(firstn_values):
                scores = grid.get((firstn, nte))
                if scores:
                    cycles = list(range(len(scores)))
                    ax.plot(
                        cycles,
                        scores,
                        color=line_colors[i],
                        linewidth=4.0,
                        marker="o",
                        markersize=10,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        label=str(firstn) if col_idx == n_cols - 1 else None,
                    )
                    if include_std:
                        stds = std_grid.get((firstn, nte))
                        if stds:
                            scores_arr = np.array(scores)
                            stds_arr = np.array(stds)
                            ax.fill_between(
                                cycles,
                                scores_arr - stds_arr,
                                scores_arr + stds_arr,
                                color=line_colors[i],
                                alpha=0.15,
                            )

                if include_coherence:
                    coherences = coherence_grid.get((firstn, nte))
                    if coherences:
                        coh_cycles = [
                            j for j, c in enumerate(coherences) if c is not None
                        ]
                        coh_vals = [c for c in coherences if c is not None]
                        if coh_vals:
                            ax.plot(
                                coh_cycles,
                                coh_vals,
                                color=(*to_rgb(COHERENCE_COLOR), alphas[i]),
                                linewidth=4.0,
                                marker="o",
                                markersize=10,
                                linestyle="-",
                                solid_capstyle="round",
                                solid_joinstyle="round",
                            )

            if include_emoji_fraction:
                ef = emoji_grid.get((firstn, nte))
                if ef:
                    ef_cycles = [j for j, v in enumerate(ef) if v is not None]
                    ef_vals = [v for v in ef if v is not None]
                    if ef_vals:
                        ax2 = ax.twinx()
                        ax2.plot(
                            ef_cycles,
                            ef_vals,
                            color=EMOJI_FRAC_COLOR,
                            linewidth=4.0,
                            marker="D",
                            markersize=10,
                            linestyle="--",
                            solid_capstyle="round",
                            solid_joinstyle="round",
                        )
                        ax2.set_ylim(bottom=0)
                        ax2.tick_params(
                            axis="y",
                            labelright=(col_idx == n_cols - 1),
                            right=(col_idx == n_cols - 1),
                            labelsize=24,
                            colors=EMOJI_FRAC_COLOR,
                        )
                        for spine in ax2.spines.values():
                            spine.set_visible(False)

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

            ax.tick_params(
                axis="x", labelbottom=True, labelsize=FONTSIZE_TICK, width=1.5, length=6
            )
            ax.tick_params(
                axis="y",
                labelleft=col_idx == 0,
                left=col_idx == 0,
                labelsize=FONTSIZE_TICK,
                width=1.5,
                length=6,
            )
            ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
            if col_idx == 0:
                ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

            ax.set_xticks(range(max_cycles))

            # Subplot title (nte value)
            ax.set_title(
                str(nte),
                fontsize=FONTSIZE_LABEL,
                fontweight="bold",
                pad=20,
                color=ROW_COLOR,
            )

        if sweep_type == "dpo_nte":
            group_title = LABEL_N_SAMPLED
            legend_title = r"DPO $\beta$"
        elif sweep_type == "dpo_steps":
            group_title = "number of DPO steps"
            legend_title = r"DPO $\beta$"
        else:
            group_title = LABEL_N_SAMPLED
            legend_title = LABEL_N_SEED

        fig.text(
            0.45,
            1.00,
            group_title,
            ha="center",
            va="top",
            fontsize=FONTSIZE_LABEL,
            fontweight="bold",
            color=ROW_COLOR,
        )

        handles, labels = axes[0][-1].get_legend_handles_labels()
        if handles:
            leg = fig.legend(
                handles,
                labels,
                title=legend_title,
                loc="center left",
                bbox_to_anchor=(0.88, 0.5),
                fontsize=FONTSIZE_LEGEND,
                frameon=False,
            )
            leg.get_title().set_fontsize(FONTSIZE_LEGEND)
            leg.get_title().set_fontweight("bold")
            leg.get_title().set_color(COL_COLOR)
            for text in leg.get_texts():
                text.set_color(COL_COLOR)
                text.set_fontweight("bold")

        if title:
            fig.suptitle(
                title, x=0.45, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=1.02
            )
        fig.tight_layout(rect=[0, 0, 0.88, 0.88])

    out = output_path or str(root / "sweep_eval_plot_clean.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    print(f"Saved plot to {out}")

    # Also save as PDF if the output was a PNG
    if out.endswith(".png"):
        pdf_out = out.replace(".png", ".pdf")
        fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        print(f"Saved plot to {pdf_out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep eval results (clean)")
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        default="outputs/sweep_bliss",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="bliss")
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["grid", "grouped"],
        default="grid",
        help="Plot format: 'grid' (default) or 'grouped' (lines in subplots)",
    )
    parser.add_argument(
        "--include-std",
        action="store_true",
        help="Shade ±1 std region (grouped format only; grid always shows it)",
    )
    parser.add_argument(
        "--include-coherence",
        action="store_true",
        help="Overlay coherence scores as a dashed gray line",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default=None,
        help="Optional suptitle displayed above the plot",
    )
    parser.add_argument(
        "--include-emoji-fraction",
        action="store_true",
        help="Overlay emoji fraction per cycle from eval_token_freqs.json (secondary y-axis)",
    )
    args = parser.parse_args()

    plot_sweep(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        plot_format=args.format,
        include_std=args.include_std,
        include_coherence=args.include_coherence,
        title=args.title,
        include_emoji_fraction=args.include_emoji_fraction,
    )
