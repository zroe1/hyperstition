"""Plot BF sweep results: beta values as lines, one subplot per nte.

Same layout as plot_sweep_beta.py but for branching factor instead of score.
Within each subplot, each DPO beta is a line whose opacity monotonically
tracks the beta value (lowest beta = most transparent, highest = most opaque).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.sweep_plot_utils import (
    FONTSIZE_TICK,
    FONTSIZE_AXLABEL,
    FONTSIZE_LABEL,
    FONTSIZE_SUPTITLE,
    FONTSIZE_LEGEND,
    SPINE_WIDTH,
    LABEL_N_SAMPLED,
    LABEL_CYCLE,
)

LABEL_BF = "branching factor"

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


def parse_beta_run(name: str) -> tuple[float, int] | None:
    m = re.match(r"beta([\d.]+)_nte(\d+)", name)
    if m:
        return float(m.group(1)), int(m.group(2))
    return None


def load_bf_results(
    sweep_dir: Path,
) -> tuple[dict[tuple[float, int], list[float]], float | None]:
    """Returns (grid, base_bf).

    grid maps (beta, nte) -> [overall_bf_cycle0, overall_bf_cycle1, ...].
    """
    combined_file = sweep_dir / "sweep_bf_results.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data.get("runs", {})
        base_result = data.get("base_result")
        base_bf = base_result["overall_bf"] if base_result else None
    else:
        runs = {}
        base_bf = None
        for d in sorted(sweep_dir.iterdir()):
            if d.is_dir():
                bf_file = d / "bf_results.json"
                if bf_file.exists():
                    with open(bf_file, "r") as f:
                        run_data = json.load(f)
                    runs[d.name] = run_data

    if not runs:
        raise FileNotFoundError(f"No BF results found in {sweep_dir}")

    grid: dict[tuple[float, int], list[float]] = {}
    for run_name, run_data in runs.items():
        parsed = parse_beta_run(run_name)
        if parsed is None:
            continue
        beta, nte = parsed
        bfs = [c["overall_bf"] for c in run_data["cycle_results"]]
        grid[(beta, nte)] = bfs

    return grid, base_bf


def plot_sweep_bf_beta(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "lucky",
    filter_betas: list[float] | None = None,
    filter_nte: list[int] | None = None,
    title: str | None = None,
):
    root = Path(sweep_dir)
    grid, base_bf = load_bf_results(root)

    beta_values = sorted(set(b for b, _ in grid))
    nte_values = sorted(set(n for _, n in grid))

    if filter_betas:
        beta_values = [b for b in beta_values if b in filter_betas]
    if filter_nte:
        nte_values = [n for n in nte_values if n in filter_nte]

    n_cols = len(nte_values)

    print(f"Subplots: {n_cols} (one per nte value)")
    print(f"  nte:   {nte_values}")
    print(f"  betas: {beta_values}")
    if base_bf is not None:
        print(f"  base BF: {base_bf:.2f}")

    line_color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)
    COL_COLOR = "#000000"

    n_betas = len(beta_values)
    beta_alphas = {
        b: 0.25 + 0.75 * i / max(n_betas - 1, 1)
        for i, b in enumerate(beta_values)
    }

    cell_w, cell_h = 4.5, 4.5
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(cell_w * n_cols + 4.5, cell_h + 2.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    ax_row = axes[0]

    max_cycles = max(len(v) for v in grid.values())

    all_bf_vals = [v for bfs in grid.values() for v in bfs]
    if base_bf is not None:
        all_bf_vals.append(base_bf)
    y_max = max(all_bf_vals) * 1.15 if all_bf_vals else 10.0
    y_min = 0

    for col_idx, nte in enumerate(nte_values):
        ax = ax_row[col_idx]
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)

        for beta in beta_values:
            bfs = grid.get((beta, nte))
            if bfs:
                cycles = list(range(len(bfs)))
                ax.plot(
                    cycles,
                    bfs,
                    color=line_color,
                    alpha=beta_alphas[beta],
                    linewidth=4.5,
                    marker="o",
                    markersize=13,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

        if base_bf is not None:
            ax.axhline(
                y=base_bf,
                color="#800000",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.15, linewidth=0.8)

        ax.tick_params(axis="x", labelsize=FONTSIZE_TICK, width=1.5, length=6)
        ax.tick_params(
            axis="y",
            labelleft=(col_idx == 0),
            left=(col_idx == 0),
            labelsize=FONTSIZE_TICK,
            width=1.5,
            length=6,
        )
        ax.set_xlabel(LABEL_CYCLE, fontsize=FONTSIZE_AXLABEL)
        if col_idx == 0:
            ax.set_ylabel(LABEL_BF, fontsize=FONTSIZE_AXLABEL)

        ax.set_xticks(range(max_cycles))

        ax.set_title(
            str(nte), fontsize=FONTSIZE_LABEL, fontweight="bold", color=COL_COLOR, pad=8,
        )

    fig.text(
        0.45,
        0.75 if title else 0.92,
        LABEL_N_SAMPLED,
        ha="center",
        va="bottom",
        fontsize=FONTSIZE_LABEL,
        fontweight="bold",
        color=COL_COLOR,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=line_color,
            alpha=beta_alphas[b],
            linewidth=4.5,
            marker="o",
            markersize=11,
            label=str(b),
        )
        for b in beta_values
    ]
    if base_bf is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="#800000",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
                label="base model",
            )
        )
    legend = fig.legend(
        handles=legend_handles,
        title=r"DPO $\beta$",
        loc="center left",
        bbox_to_anchor=(0.88, 0.48),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
        labelspacing=0.8,
    )
    legend.get_title().set_color(COL_COLOR)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_color(COL_COLOR)
        text.set_fontweight("bold")

    top = 0.91
    if title:
        fig.suptitle(
            title, x=0.45, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.94,
        )
        top = 0.85
    fig.tight_layout(rect=[0.04, 0.04, 0.87, top])

    out = output_path or str(root / "sweep_bf_plot_beta.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    print(f"Saved plot to {out}")

    if out.endswith(".png"):
        pdf_out = out.replace(".png", ".pdf")
        fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        print(f"Saved plot to {pdf_out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot BF sweep: beta lines, one subplot per nte",
    )
    parser.add_argument(
        "--sweep-dir", "-d", type=str, required=True,
        help="Path to sweep output directory",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="lucky")
    parser.add_argument(
        "--filter-betas", type=float, nargs="+", default=None,
        help="Only plot these beta values",
    )
    parser.add_argument(
        "--filter-nte", type=int, nargs="+", default=None,
        help="Only plot these nte values",
    )
    parser.add_argument(
        "--title", "-t", type=str, default=None,
        help="Optional suptitle displayed above the plot",
    )
    args = parser.parse_args()

    plot_sweep_bf_beta(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        filter_betas=args.filter_betas,
        filter_nte=args.filter_nte,
        title=args.title,
    )
