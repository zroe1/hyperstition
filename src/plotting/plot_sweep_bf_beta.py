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
LABEL_DPO_LR = "DPO learning rate"


def _confidence_interval_95(values: list[float]) -> float:
    """Compute 95% CI half-width using a scipy-free t approximation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = float(np.mean(values))
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = float(np.sqrt(variance / n))
    t_val = 1.96 if n >= 30 else 2.0 + 3.0 / n
    return t_val * std_err


def parse_beta_run(name: str) -> tuple[float, str, float | int] | None:
    m = re.match(r"beta([\deE.+-]+)_nte(\d+)$", name)
    if m:
        return float(m.group(1)), "nte", int(m.group(2))
    m = re.match(r"beta([\deE.+-]+)_lr([\deE.+-]+)$", name)
    if m:
        return float(m.group(1)), "lr", float(m.group(2))
    return None


def load_bf_results(
    sweep_dir: Path,
) -> tuple[
    dict[tuple[float | int, float | int], list[float]],
    dict[tuple[float | int, float | int], list[float]],
    float | None,
    float | None,
    str,
]:
    """Returns (grid, ci_grid, base_bf, base_ci, axis_kind).

    grid maps (beta, axis_value) -> [overall_bf_cycle0, overall_bf_cycle1, ...].
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

    grid: dict[tuple[float | int, float | int], list[float]] = {}
    ci_grid: dict[tuple[float | int, float | int], list[float]] = {}
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
        bfs = [c["overall_bf"] for c in run_data["cycle_results"]]
        grid[(beta, axis_value)] = bfs

        ci95s = []
        for cycle_entry in run_data["cycle_results"]:
            per_prompt_bf = cycle_entry.get("per_prompt_bf", {})
            vals = [v for v in per_prompt_bf.values() if v is not None]
            ci95s.append(_confidence_interval_95(vals) if vals else 0.0)
        ci_grid[(beta, axis_value)] = ci95s

    if axis_kind is None:
        raise FileNotFoundError(f"No beta sweep BF results found in {sweep_dir}")

    base_ci = None
    if base_result:
        base_vals = [
            v for v in base_result.get("per_prompt_bf", {}).values()
            if v is not None
        ]
        base_ci = _confidence_interval_95(base_vals) if base_vals else None

    return grid, ci_grid, base_bf, base_ci, axis_kind


def plot_sweep_bf_beta(
    sweep_dir: str,
    output_path: str | None = None,
    config_name: str = "lucky",
    filter_betas: list[float] | None = None,
    filter_nte: list[int] | None = None,
    filter_lrs: list[float] | None = None,
    include_ci: bool = False,
    title: str | None = None,
):
    root = Path(sweep_dir)
    grid, ci_grid, base_bf, base_ci, axis_kind = load_bf_results(root)

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
    if base_bf is not None:
        print(f"  base BF: {base_bf:.2f}")
    if include_ci and base_ci is None and not any(any(cis) for cis in ci_grid.values()):
        print("  CI shading requested, but no per-prompt BF values were found; plotting lines only.")

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
    if include_ci:
        for key, bfs in grid.items():
            cis = ci_grid.get(key, [])
            all_bf_vals.extend(
                value + ci for value, ci in zip(bfs, cis, strict=False)
            )
            all_bf_vals.extend(
                max(0.0, value - ci) for value, ci in zip(bfs, cis, strict=False)
            )
    if base_bf is not None:
        all_bf_vals.append(base_bf)
        if include_ci and base_ci is not None:
            all_bf_vals.extend([base_bf + base_ci, max(0.0, base_bf - base_ci)])
    y_max = max(all_bf_vals) * 1.15 if all_bf_vals else 10.0
    y_min = 0

    for col_idx, axis_value in enumerate(axis_values):
        ax = ax_row[col_idx]
        ax.set_facecolor("white")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
        ax.spines["left"].set_linewidth(SPINE_WIDTH)

        for beta in beta_values:
            bfs = grid.get((beta, axis_value))
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
                if include_ci:
                    cis = ci_grid.get((beta, axis_value))
                    if cis:
                        bfs_arr = np.array(bfs)
                        ci_arr = np.array(cis)
                        ax.fill_between(
                            cycles,
                            np.maximum(y_min, bfs_arr - ci_arr),
                            bfs_arr + ci_arr,
                            color=line_color,
                            alpha=beta_alphas[beta] * 0.2,
                        )

        if base_bf is not None:
            ax.axhline(
                y=base_bf,
                color="#800000",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )
            if include_ci and base_ci is not None:
                ax.axhspan(
                    max(y_min, base_bf - base_ci),
                    base_bf + base_ci,
                    color="#800000",
                    alpha=0.08,
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
            str(axis_value), fontsize=FONTSIZE_LABEL, fontweight="bold", color=COL_COLOR, pad=8,
        )

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

    default_name = "sweep_bf_plot_beta.png" if axis_kind == "nte" else "sweep_bf_plot_beta_lr.png"
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
        "--filter-lrs", type=float, nargs="+", default=None,
        help="Only plot these DPO learning-rate values",
    )
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Shade 95%% CI region around each BF line",
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
        filter_lrs=args.filter_lrs,
        include_ci=args.include_ci,
        title=args.title,
    )
