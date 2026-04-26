"""Find amplification cases in selected sweep outputs.

Scans a sweep root such as ``outputs_dang/models_sweep_bliss`` for
``sweep_eval_results.json`` files under a restricted set of model directories,
applies the repo's current late-delta amplification rule, and writes summary
artifacts under ``<output_dir>/<sweep_name>/``.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

try:
    from analyses.detect_amplification import compute_late_delta
except ModuleNotFoundError:
    from detect_amplification import compute_late_delta


DEFAULT_MODELS = [
    "llama-3.3-70b-instruct",
    "qwen3-4b-instruct-2507",
]


def load_runs(results_path: Path) -> dict:
    with open(results_path, "r") as f:
        return json.load(f)["runs"]


def parse_run_name(run_name: str) -> Tuple[Optional[int], Optional[int]]:
    seed: Optional[int] = None
    nte: Optional[int] = None

    if run_name.startswith("seed"):
        seed_part, _, rest = run_name.partition("_")
        try:
            seed = int(seed_part.removeprefix("seed"))
        except ValueError:
            seed = None
        if rest.startswith("nte"):
            try:
                nte = int(rest.removeprefix("nte"))
            except ValueError:
                nte = None

    return seed, nte


def collect_rows(
    sweep_dir: Path,
    models: list[str],
    late_start_cycle: int,
    delta_threshold: float,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    skipped_models: list[str] = []

    for model in models:
        results_path = sweep_dir / model / "sweep_eval_results.json"
        if not results_path.is_file():
            skipped_models.append(model)
            continue

        runs = load_runs(results_path)
        for run_name, run_data in sorted(runs.items()):
            cycle_results = sorted(run_data["cycle_results"], key=lambda c: c["cycle"])
            scores = [cycle["aggregate_score"] for cycle in cycle_results]
            late_delta = compute_late_delta(
                scores,
                late_start_cycle=late_start_cycle,
                threshold=delta_threshold,
            )
            seed, nte = parse_run_name(run_name)
            row = {
                "sweep": sweep_dir.name,
                "model": model,
                "run_name": run_name,
                "seed": seed,
                "nte": nte,
                "num_cycles": len(scores),
                "scores": scores,
                "anchor_cycle": late_delta["anchor_cycle"],
                "anchor_score": late_delta["anchor_score"],
                "max_delta": late_delta["max_delta"],
                "is_amplified": late_delta["is_amplified"],
                "late_deltas": late_delta["deltas"],
            }
            for cycle_idx, score in enumerate(scores):
                row[f"c{cycle_idx}"] = score
            rows.append(row)

    metadata = {
        "sweep_dir": str(sweep_dir),
        "models_requested": models,
        "models_found": sorted({row["model"] for row in rows}),
        "models_missing": skipped_models,
        "late_start_cycle": late_start_cycle,
        "delta_threshold": delta_threshold,
    }
    return rows, metadata


def write_cases_csv(rows: list[dict], path: Path) -> None:
    amplified = [row for row in rows if row["is_amplified"]]
    max_cycle = max(
        (
            int(key[1:])
            for row in rows
            for key in row
            if key.startswith("c") and key[1:].isdigit()
        ),
        default=-1,
    )
    fields = [
        "sweep",
        "model",
        "run_name",
        "seed",
        "nte",
        "num_cycles",
        "anchor_cycle",
        "anchor_score",
    ]
    fields += [f"c{i}" for i in range(max_cycle + 1)]
    fields += ["max_delta", "is_amplified", "late_deltas", "scores"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in amplified:
            out = {
                key: (
                    json.dumps(row[key], sort_keys=True)
                    if key in {"late_deltas", "scores"}
                    else row.get(key, "")
                )
                for key in fields
            }
            writer.writerow(out)


def build_summary(rows: list[dict], metadata: dict) -> dict:
    amplified_rows = [row for row in rows if row["is_amplified"]]
    per_model_total = Counter(row["model"] for row in rows)
    per_model_amplified = Counter(row["model"] for row in amplified_rows)

    return {
        **metadata,
        "total_runs_scanned": len(rows),
        "amplified_runs_count": len(amplified_rows),
        "amplified_run_names": [row["run_name"] for row in amplified_rows],
        "per_model_total": dict(sorted(per_model_total.items())),
        "per_model_amplified": dict(sorted(per_model_amplified.items())),
    }


def write_summary_json(summary: dict, rows: list[dict], path: Path) -> None:
    payload = {
        "summary": summary,
        "cases": [
            {
                "model": row["model"],
                "run_name": row["run_name"],
                "seed": row["seed"],
                "nte": row["nte"],
                "scores": row["scores"],
                "anchor_cycle": row["anchor_cycle"],
                "anchor_score": row["anchor_score"],
                "max_delta": row["max_delta"],
                "late_deltas": row["late_deltas"],
            }
            for row in rows
            if row["is_amplified"]
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary_md(summary: dict, rows: list[dict], path: Path) -> None:
    amplified_rows = [row for row in rows if row["is_amplified"]]
    lines = [
        f"# Amplification summary — {Path(summary['sweep_dir']).name}",
        "",
        f"Sweep dir: `{summary['sweep_dir']}`",
        f"Models scanned: {', '.join(summary['models_requested'])}",
        f"Amplification rule: any `aggregate_score[cycle] - aggregate_score[1] >= {summary['delta_threshold']}` for `cycle >= {summary['late_start_cycle']}`.",
        "",
        f"Total runs scanned: **{summary['total_runs_scanned']}**",
        f"Amplified runs found: **{summary['amplified_runs_count']}**",
        "",
        "## Counts by model",
        "",
        "| model | runs scanned | amplified |",
        "|---|---:|---:|",
    ]

    for model in summary["models_requested"]:
        lines.append(
            f"| {model} | {summary['per_model_total'].get(model, 0)} | {summary['per_model_amplified'].get(model, 0)} |"
        )

    if summary["models_missing"]:
        lines.extend(
            [
                "",
                "## Missing model outputs",
                "",
                ", ".join(summary["models_missing"]),
            ]
        )

    lines.extend(["", "## Amplification cases", ""])
    if not amplified_rows:
        lines.append("No amplification cases found.")
    else:
        lines.append(
            "| model | run | seed | nte | anchor score | max delta | scores | late deltas |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in amplified_rows:
            seed = row["seed"] if row["seed"] is not None else ""
            nte = row["nte"] if row["nte"] is not None else ""
            anchor_score = (
                f"{row['anchor_score']:.1f}" if row["anchor_score"] is not None else ""
            )
            max_delta = f"{row['max_delta']:.1f}" if row["max_delta"] is not None else ""
            scores = ", ".join(f"{score:.1f}" for score in row["scores"])
            deltas = ", ".join(
                f"{cycle}:{delta:.1f}"
                for cycle, delta in sorted(
                    row["late_deltas"].items(), key=lambda item: int(item[0])
                )
            )
            lines.append(
                f"| {row['model']} | {row['run_name']} | {seed} | {nte} | {anchor_score} | {max_delta} | {scores} | {deltas} |"
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find amplification cases in a sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Sweep directory containing per-model sweep_eval_results.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for reports. Outputs land in <output-dir>/<sweep-name>/.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model subdirectories to scan.",
    )
    parser.add_argument(
        "--late-start-cycle",
        type=int,
        default=4,
        help="First cycle index checked by the late-delta rule.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=10.0,
        help="Late-delta threshold. Uses the repo's current >= semantics.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    report_dir = Path(args.output_dir) / sweep_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = collect_rows(
        sweep_dir=sweep_dir,
        models=args.models,
        late_start_cycle=args.late_start_cycle,
        delta_threshold=args.delta_threshold,
    )
    summary = build_summary(rows, metadata)

    cases_csv = report_dir / "amplification_cases.csv"
    summary_json = report_dir / "amplification_summary.json"
    summary_md = report_dir / "amplification_summary.md"

    write_cases_csv(rows, cases_csv)
    write_summary_json(summary, rows, summary_json)
    write_summary_md(summary, rows, summary_md)

    print(f"Wrote {cases_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    print(
        f"Amplification cases: {summary['amplified_runs_count']}/{summary['total_runs_scanned']}"
    )


if __name__ == "__main__":
    main()
