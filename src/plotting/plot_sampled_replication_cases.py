"""Plot replication comparisons.

Supports:
  - sampled replication cases from nested eval_results.json files
  - legacy lucky summary.csv comparison plots

Visual style mirrors ``plot_seed_sweep.py``: shared font sizes, hidden
top/right spines, fat round-cap markers, single shared legend.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting.sweep_plot_utils import (
    FONTSIZE_TICK,
    FONTSIZE_AXLABEL,
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


def _draw_replica_lines(
    ax: plt.Axes,
    original_cycles: list[float] | None,
    replica_cycles: list[list[float]],
    color: str,
) -> None:
    for scores in replica_cycles:
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

    if original_cycles is not None:
        ax.plot(
            list(range(len(original_cycles))),
            original_cycles,
            color=ORIGINAL_COLOR,
            alpha=1.0,
            linewidth=4.5,
            marker="o",
            markersize=11,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=10,
        )


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save the figure as both PNG and PDF (sibling files, same stem)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        out = output_path.with_suffix(ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        print(f"Saved plot to {out}")


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


def load_summary_rows(summary_csv: Path) -> list[dict]:
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
            }
        )
    return parsed


def _summary_panel_data(rows: list[dict], nte: int) -> tuple[list[float] | None, list[list[float]]]:
    panel_rows = [r for r in rows if r["nte"] == nte]
    baseline = next((r["cycles"] for r in panel_rows if r["mode"] == "baseline"), None)
    replicas = [r["cycles"] for r in panel_rows if r["mode"] == "different_seeds"]
    return baseline, replicas


def plot_summary_comparison(
    uncal_rows: list[dict],
    cal_rows: list[dict],
    output_dir: Path,
    source_firstn: int,
    config_name: str,
) -> None:
    ntes = sorted(
        {r["nte"] for r in uncal_rows if r["mode"] == "baseline"}
        | {r["nte"] for r in cal_rows if r["mode"] == "baseline"}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    color = PERSONA_COLORS.get(config_name, DEFAULT_COLOR)

    for nte in ntes:
        uncal_baseline, uncal_replicas = _summary_panel_data(uncal_rows, nte)
        cal_baseline, cal_replicas = _summary_panel_data(cal_rows, nte)

        all_cycles = (
            ([uncal_baseline] if uncal_baseline else [])
            + ([cal_baseline] if cal_baseline else [])
            + uncal_replicas
            + cal_replicas
        )
        if not all_cycles:
            continue
        max_cycles = max(len(c) for c in all_cycles)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5.5),
            sharey=True,
            facecolor="white",
        )

        panel_configs = [
            (axes[0], r"Original $\mathbf{M_\text{seed}}$", cal_baseline, cal_replicas, True),
            (axes[1], r"Retrained $\mathbf{M_\text{seed}}$", uncal_baseline, uncal_replicas, False),
        ]
        for ax, panel_title, baseline, replicas, show_ylabel in panel_configs:
            _style_ax(ax, max_cycles)
            _draw_replica_lines(ax, baseline, replicas, color)
            ax.set_title(panel_title, fontsize=34, fontweight="bold", pad=-2)
            if show_ylabel:
                ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)
            else:
                ax.tick_params(axis="y", left=False)

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
        _save_figure(fig, output_dir / f"seed{source_firstn}_nte{nte}.png")
        plt.close(fig)


def load_original_cycles(sweep_dir: Path, model_slug: str, run_name: str) -> list[float]:
    with open(sweep_dir / model_slug / "sweep_eval_results.json", "r") as f:
        data = json.load(f)
    run = data["runs"][run_name]
    cycle_results = sorted(run["cycle_results"], key=lambda c: c["cycle"])
    return [c["aggregate_score"] for c in cycle_results]


def load_replica_cycles(root: Path, manifest_path: Path, sweep_name: str) -> dict[tuple[str, str], list[list[float]]]:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for item in manifest["completed_replica_evals"]:
        if item["sweep_name"] != sweep_name:
            continue
        run_dir = Path(item["sweep_dir"]) / item["run_name"] / "eval_results.json"
        if not run_dir.exists():
            continue
        with open(run_dir, "r") as f:
            data = json.load(f)
        cycle_results = sorted(data["cycle_results"], key=lambda c: c["cycle"])
        key = (item["model"], item["run_name"])
        grouped.setdefault(key, []).append([c["aggregate_score"] for c in cycle_results])
    return grouped


def plot_case(
    original_cycles: list[float],
    fresh_cycles: list[list[float]],
    calibrated_cycles: list[list[float]],
    color: str,
    title: str,
    output_path: Path,
) -> None:
    max_cycles = max(
        len(original_cycles),
        max((len(c) for c in fresh_cycles), default=0),
        max((len(c) for c in calibrated_cycles), default=0),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        sharey=True,
        facecolor="white",
    )

    panel_configs = [
        (axes[0], r"Original $\mathbf{M_\text{seed}}$", calibrated_cycles, True),
        (axes[1], r"Retrained $\mathbf{M_\text{seed}}$", fresh_cycles, False),
    ]
    for ax, panel_title, replicas, show_ylabel in panel_configs:
        _style_ax(ax, max_cycles)
        _draw_replica_lines(ax, original_cycles, replicas, color)
        ax.set_title(panel_title, fontsize=34, fontweight="bold", pad=-2)
        if show_ylabel:
            ax.set_ylabel(LABEL_SCORE, fontsize=FONTSIZE_AXLABEL)
        else:
            ax.tick_params(axis="y", left=False)

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
    _save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completed sampled replication cases.")
    parser.add_argument(
        "--summary-comparison",
        action="store_true",
        help="Use legacy summary.csv comparison mode instead of sampled-case mode.",
    )
    parser.add_argument("--sweep-name", type=str, default="models_sweep_bliss")
    parser.add_argument("--config-name", type=str, default="bliss")
    parser.add_argument("--original-root", type=str, default="outputs_dang/models_sweep_bliss")
    parser.add_argument("--fresh-root", type=str, default="outputs_dang/replication_amplified_fresh_cycle0")
    parser.add_argument("--calibrated-root", type=str, default="outputs_dang/replication_amplified")
    parser.add_argument("--output-dir", type=str, default="plots/replication_comparison")
    parser.add_argument("--uncalibrated-summary-csv", type=str, default="outputs_dang/replication/summary.csv")
    parser.add_argument("--calibrated-summary-csv", type=str, default="outputs_dang/replication_calibrated/summary.csv")
    parser.add_argument("--source-firstn", type=int, default=24)
    args = parser.parse_args()

    if args.summary_comparison:
        plot_summary_comparison(
            uncal_rows=load_summary_rows(Path(args.uncalibrated_summary_csv)),
            cal_rows=load_summary_rows(Path(args.calibrated_summary_csv)),
            output_dir=Path(args.output_dir),
            source_firstn=args.source_firstn,
            config_name=args.config_name,
        )
        return

    fresh_grouped = load_replica_cycles(
        Path(args.fresh_root),
        Path(args.fresh_root) / "sampled_eval_manifest.json",
        args.sweep_name,
    )
    calibrated_grouped = load_replica_cycles(
        Path(args.calibrated_root),
        Path(args.calibrated_root) / "sampled_eval_manifest.json",
        args.sweep_name,
    )

    common_keys = sorted(set(fresh_grouped) & set(calibrated_grouped))
    color = PERSONA_COLORS.get(args.config_name, DEFAULT_COLOR)
    for model_slug, run_name in common_keys:
        if len(fresh_grouped[model_slug, run_name]) < 10 or len(calibrated_grouped[model_slug, run_name]) < 10:
            continue
        original_cycles = load_original_cycles(Path(args.original_root), model_slug, run_name)
        plot_case(
            original_cycles=original_cycles,
            fresh_cycles=fresh_grouped[model_slug, run_name],
            calibrated_cycles=calibrated_grouped[model_slug, run_name],
            color=color,
            title=run_name,
            output_path=Path(args.output_dir) / args.sweep_name / f"{model_slug}_{run_name}.png",
        )


if __name__ == "__main__":
    main()
