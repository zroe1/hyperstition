"""Plot replication comparisons.

Supports:
  - sampled replication cases from nested eval_results.json files
  - legacy lucky summary.csv comparison plots
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERSONA_COLORS = {
    "bliss": "#0077cc",
    "hopelessness": "#cc2460",
    "lucky": "#cca800",
    "misalignment": "#cc2818",
    "sycophancy": "#921ecc",
    "nvidia": "#119957",
    "misanthropy": "#cc7a00",
}


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


def plot_summary_panel(ax, rows: list[dict], nte: int, panel_title: str, color: str, ylabel: str) -> None:
    panel_rows = [r for r in rows if r["nte"] == nte]
    ax.set_facecolor("white")

    for mode in ["baseline", "different_seeds"]:
        mode_rows = [r for r in panel_rows if r["mode"] == mode]
        if not mode_rows:
            continue
        for idx, row in enumerate(mode_rows):
            cycles = list(range(len(row["cycles"])))
            ax.plot(
                cycles,
                row["cycles"],
                color="#111111" if mode == "baseline" else color,
                linewidth=3.0,
                alpha=1.0 if mode == "baseline" else 0.30,
                marker="o" if mode == "baseline" else None,
                markersize=5,
                label="original" if mode == "baseline" else ("different seeds" if idx == 0 else None),
            )

    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("cycle", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def plot_summary_comparison(
    uncal_rows: list[dict],
    cal_rows: list[dict],
    output_dir: Path,
    source_firstn: int,
    config_name: str,
) -> None:
    ntes = sorted({r["nte"] for r in uncal_rows if r["mode"] == "baseline"} | {r["nte"] for r in cal_rows if r["mode"] == "baseline"})
    output_dir.mkdir(parents=True, exist_ok=True)
    color = PERSONA_COLORS[config_name]

    for nte in ntes:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), squeeze=False, facecolor="white")
        plot_summary_panel(axes[0][0], uncal_rows, nte, "Fresh cycle 0", color, f"{config_name} score")
        plot_summary_panel(axes[0][1], cal_rows, nte, "Calibrated cycle 0", color, "")
        axes[0][1].set_ylabel("")
        handles, labels = axes[0][0].get_legend_handles_labels()
        right_handles, right_labels = axes[0][1].get_legend_handles_labels()
        dedup = {}
        for handle, label in list(zip(handles, labels)) + list(zip(right_handles, right_labels)):
            if label:
                dedup[label] = handle
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            ncol=min(3, len(dedup)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(f"seed{source_firstn}_nte{nte}", fontsize=15, fontweight="bold", y=1.06)
        fig.tight_layout()
        output_path = output_dir / f"seed{source_firstn}_nte{nte}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved plot to {output_path}")
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
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), squeeze=False, facecolor="white")
    panels = [
        (axes[0][0], "Fresh cycle 0", fresh_cycles),
        (axes[0][1], "Calibrated cycle 0", calibrated_cycles),
    ]

    for ax, panel_title, replica_groups in panels:
        ax.set_facecolor("white")
        baseline_x = list(range(len(original_cycles)))
        ax.plot(
            baseline_x,
            original_cycles,
            color="#111111",
            linewidth=3.0,
            marker="o",
            markersize=5,
            label="original",
        )
        for idx, cycles in enumerate(replica_groups):
            x = list(range(len(cycles)))
            ax.plot(
                x,
                cycles,
                color=color,
                linewidth=3.0,
                alpha=0.30,
                marker="o",
                markersize=4,
                label="different seeds" if idx == 0 else None,
            )
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("cycle", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.2, 6.2)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    axes[0][0].set_ylabel("bliss score", fontsize=12, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    right_handles, right_labels = axes[0][1].get_legend_handles_labels()
    dedup = {}
    for handle, label in list(zip(handles, labels)) + list(zip(right_handles, right_labels)):
        if label:
            dedup[label] = handle
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper center",
        ncol=min(3, len(dedup)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.06)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")
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
    for model_slug, run_name in common_keys:
        if len(fresh_grouped[model_slug, run_name]) < 10 or len(calibrated_grouped[model_slug, run_name]) < 10:
            continue
        original_cycles = load_original_cycles(Path(args.original_root), model_slug, run_name)
        plot_case(
            original_cycles=original_cycles,
            fresh_cycles=fresh_grouped[model_slug, run_name],
            calibrated_cycles=calibrated_grouped[model_slug, run_name],
            color=PERSONA_COLORS[args.config_name],
            title=run_name,
            output_path=Path(args.output_dir) / args.sweep_name / f"{model_slug}_{run_name}.png",
        )


if __name__ == "__main__":
    main()
