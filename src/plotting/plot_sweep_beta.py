"""Plot DPO sweep eval results: beta values as lines, one subplot per nte.

Single row of subplots (one per number-of-training-examples value).
Within each subplot, each DPO beta is a line whose opacity monotonically
tracks the beta value (lowest beta = most transparent, highest = most opaque).
"""

import argparse
import json
import re
from pathlib import Path

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
LABEL_DPO_LR = "DPO learning rate"


def parse_beta_run(name: str) -> tuple[float, str, float | int] | None:
    """Extract beta sweep metadata from a run directory name."""
    m = re.match(r"beta([\deE.+-]+)_nte(\d+)$", name)
    if m:
        return float(m.group(1)), "nte", int(m.group(2))
    m = re.match(r"beta([\deE.+-]+)_lr([\deE.+-]+)$", name)
    if m:
        return float(m.group(1)), "lr", float(m.group(2))
    return None


def load_results(
    sweep_dir: Path, include_coherence: bool = False
) -> tuple[dict, dict, float | None, dict | None, float | None, str]:
    """Returns (grid, ci_grid, base_score, coherence_grid, base_coherence, axis_kind)."""
    combined_file = sweep_dir / "sweep_eval_results.json"
    if combined_file.exists():
        with open(combined_file, "r") as f:
            data = json.load(f)
        runs = data["runs"]
        base_result = data.get("base_result")
        base_score = base_result["aggregate_score"] if base_result else None
    else:
        runs = {}
        base_result = None
        base_score = None
        for d in sorted(sweep_dir.iterdir()):
            if d.is_dir():
                results_file = d / "eval_results.json"
                if results_file.exists():
                    with open(results_file, "r") as f:
                        run_data = json.load(f)
                    runs[d.name] = run_data
                    if base_result is None:
                        base_result = run_data.get("base_result")
                    if base_score is None and base_result:
                        base_score = base_result.get("aggregate_score")

    if not runs:
        raise FileNotFoundError(f"No eval results found in {sweep_dir}")

    grid: dict[tuple[float, int], list[float]] = {}
    ci_grid: dict[tuple[float, int], list[float]] = {}
    coherence_grid: dict[tuple[float, int], list[float]] | None = {} if include_coherence else None
    axis_kind: str | None = None
    for run_name, run_data in runs.items():
        parsed = parse_beta_run(run_name)
        if parsed is None:
            continue
        beta, run_axis_kind, axis_value = parsed
        if axis_kind is None:
            axis_kind = run_axis_kind
        elif axis_kind != run_axis_kind:
            raise ValueError(
                f"Mixed beta sweep layouts found in {sweep_dir}: "
                f"{axis_kind} and {run_axis_kind}"
            )
        scores = [c["aggregate_score"] for c in run_data["cycle_results"]]
        grid[(beta, axis_value)] = scores

        ci95s = []
        for c in run_data["cycle_results"]:
            pq = c.get("per_question", {})
            vals = [v for v in pq.values() if v is not None]
            n = len(vals)
            if n > 1:
                ci95s.append(1.96 * float(np.std(vals)) / np.sqrt(n))
            else:
                ci95s.append(0.0)
        ci_grid[(beta, axis_value)] = ci95s

        if include_coherence:
            coherences = []
            for c in run_data["cycle_results"]:
                agg = c.get("aggregate_coherence")
                if agg is not None:
                    coherences.append(agg)
                else:
                    resps = c.get("responses", [])
                    coh_vals = [r["coherence"] for r in resps if r.get("coherence") is not None]
                    coherences.append(float(np.mean(coh_vals)) if coh_vals else None)
            if all(c is not None for c in coherences):
                coherence_grid[(beta, axis_value)] = coherences

    base_coherence = None
    if include_coherence and base_result:
        base_coherence = base_result.get("aggregate_coherence")

    if axis_kind is None:
        raise FileNotFoundError(f"No beta sweep runs found in {sweep_dir}")

    return grid, ci_grid, base_score, coherence_grid, base_coherence, axis_kind


def plot_sweep_beta(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "lucky",
    filter_betas: list[float] | None = None,
    filter_nte: list[int] | None = None,
    filter_lrs: list[float] | None = None,
    include_ci: bool = False,
    include_coherence: bool = False,
    include_base: bool = True,
    title: str | None = None,
):
    root = Path(sweep_dir)
    grid, ci_grid, base_score, coherence_grid, base_coherence, axis_kind = load_results(
        root, include_coherence=include_coherence
    )

    beta_values = sorted(set(b for b, _ in grid))
    axis_values = sorted(set(v for _, v in grid))

    if filter_betas:
        beta_values = [b for b in beta_values if b in filter_betas]
    if axis_kind == "nte" and filter_nte:
        axis_values = [v for v in axis_values if v in filter_nte]
    if axis_kind == "lr" and filter_lrs:
        axis_values = [v for v in axis_values if v in filter_lrs]

    axis_label = LABEL_N_SAMPLED if axis_kind == "nte" else LABEL_DPO_LR
    n_cols = len(axis_values)

    print(f"Subplots: {n_cols} (one per {axis_kind} value)")
    print(f"  {axis_kind}: {axis_values}")
    print(f"  betas: {beta_values}")
    if include_base and base_score is not None:
        print(f"  base score: {base_score:.1f}")
    if include_base and include_coherence and base_coherence is not None:
        print(f"  base coherence: {base_coherence:.1f}")

    line_color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)
    coherence_color = "#009933"
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

    for col_idx, axis_value in enumerate(axis_values):
        ax = ax_row[col_idx]
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)

        for beta in beta_values:
            scores = grid.get((beta, axis_value))
            if scores:
                cycles = list(range(len(scores)))
                ax.plot(
                    cycles,
                    scores,
                    color=line_color,
                    alpha=beta_alphas[beta],
                    linewidth=4.5,
                    marker="o",
                    markersize=13,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )
                if include_ci:
                    cis = ci_grid.get((beta, axis_value))
                    if cis:
                        scores_arr = np.array(scores)
                        ci_arr = np.array(cis)
                        ax.fill_between(
                            cycles,
                            scores_arr - ci_arr,
                            scores_arr + ci_arr,
                            color=line_color,
                            alpha=beta_alphas[beta] * 0.2,
                        )

        if include_coherence and coherence_grid:
            for beta in beta_values:
                coherences = coherence_grid.get((beta, axis_value))
                if coherences:
                    cycles = list(range(len(coherences)))
                    ax.plot(
                        cycles,
                        coherences,
                        color=coherence_color,
                        alpha=beta_alphas[beta],
                        linewidth=3,
                        marker="s",
                        markersize=10,
                        linestyle="--",
                        solid_capstyle="round",
                        solid_joinstyle="round",
                    )

        if include_base and base_score is not None:
            ax.axhline(
                y=base_score,
                color="#800000",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )

        if include_base and include_coherence and base_coherence is not None:
            ax.axhline(
                y=base_coherence,
                color=coherence_color,
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )

        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75])
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
            ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)

        ax.set_xticks(range(max_cycles))

        ax.set_title(
            str(axis_value), fontsize=FONTSIZE_LABEL, fontweight="bold", color=COL_COLOR, pad=8
        )

    # Label above the column headers
    fig.text(
        0.45,
        0.75 if title else 0.92,
        axis_label,
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
    if include_coherence:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=coherence_color,
                linewidth=3,
                linestyle="--",
                marker="s",
                markersize=10,
                alpha=0.8,
                label="coherence",
            )
        )
    if include_base and base_score is not None:
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
            title, x=0.45, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.94
        )
        top = 0.85
    fig.tight_layout(rect=[0.04, 0.04, 0.87, top])

    default_name = "sweep_eval_plot_beta.png" if axis_kind == "nte" else "sweep_eval_plot_beta_lr.png"
    out = output_path or str(root / default_name)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    print(f"Saved plot to {out}")

    if out.endswith(".png"):
        pdf_out = out.replace(".png", ".pdf")
        fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        print(f"Saved plot to {pdf_out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DPO sweep: beta lines, one subplot per nte"
    )
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        required=True,
        help="Path to sweep output directory",
    )
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--config", "-c", type=str, default="lucky")
    parser.add_argument(
        "--filter-betas",
        type=float,
        nargs="+",
        default=None,
        help="Only plot these beta values",
    )
    parser.add_argument(
        "--filter-nte",
        type=int,
        nargs="+",
        default=None,
        help="Only plot these nte values",
    )
    parser.add_argument(
        "--filter-lrs",
        type=float,
        nargs="+",
        default=None,
        help="Only plot these DPO learning-rate values",
    )
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Shade 95%% CI region around each line",
    )
    parser.add_argument(
        "--include-coherence",
        action="store_true",
        help="Plot coherence scores as dashed lines alongside main scores",
    )
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument(
        "--include-base",
        dest="include_base",
        action="store_true",
        help="plot the base-model baseline line (default)",
    )
    base_group.add_argument(
        "--no-base",
        dest="include_base",
        action="store_false",
        help="hide the base-model baseline line",
    )
    parser.set_defaults(include_base=True)
    parser.add_argument(
        "--title", "-t", type=str, default=None,
        help="Optional suptitle displayed above the plot",
    )
    args = parser.parse_args()

    plot_sweep_beta(
        sweep_dir=args.sweep_dir,
        output_path=args.output,
        config_name=args.config,
        filter_betas=args.filter_betas,
        filter_nte=args.filter_nte,
        filter_lrs=args.filter_lrs,
        include_ci=args.include_ci,
        include_coherence=args.include_coherence,
        include_base=args.include_base,
        title=args.title,
    )
