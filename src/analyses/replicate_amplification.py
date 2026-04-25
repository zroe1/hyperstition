"""Aggregate replication-experiment eval results.

Walks outputs_dang/replication/{same_seed_*,different_seeds}/<tag>/<slug>/<run>/eval_results.json
and produces a summary comparing per-cycle aggregate_score against the baseline
runs at outputs_dang/models_sweep_lr_schedule_constant_lucky/qwen3-4b-instruct-2507/seed24_nte{30,40}.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from analyses.find_amplification import find_amplification_runs


DEFAULT_REPL_ROOT = Path("outputs_dang/replication")
DEFAULT_BASELINE_ROOT = Path(
    "outputs_dang/models_sweep_lr_schedule_constant_lucky/qwen3-4b-instruct-2507"
)
SLUG = "qwen3-4b-instruct-2507"
NTE_VALUES = [30, 40]
FIRSTN = 24
MIN_CONSECUTIVE = 5


def load_cycle_scores(eval_path: Path) -> list[tuple[int, float]]:
    with open(eval_path) as f:
        data = json.load(f)
    cycles = sorted(data["cycle_results"], key=lambda c: c["cycle"])
    return [(c["cycle"], c["aggregate_score"]) for c in cycles]


def row_for(
    mode: str, tag: str, seed: int | str, nte: int, eval_path: Path
) -> dict | None:
    if not eval_path.is_file():
        return None
    scores = load_cycle_scores(eval_path)
    runs, _ = find_amplification_runs(str(eval_path), min_consecutive=MIN_CONSECUTIVE)
    final = scores[-1][1] if scores else None
    max_score = max((s for _, s in scores), default=None)
    cycle_of_max = next((c for c, s in scores if s == max_score), None)
    row = {
        "mode": mode,
        "tag": tag,
        "seed": seed,
        "nte": nte,
        "final": final,
        "max": max_score,
        "cycle_of_max": cycle_of_max,
        "is_amplified": bool(runs),
        "amplification_runs": runs,
    }
    for c, s in scores:
        row[f"c{c}"] = s
    return row


def collect_baseline(baseline_root: Path) -> list[dict]:
    out = []
    for nte in NTE_VALUES:
        ep = baseline_root / f"seed{FIRSTN}_nte{nte}" / "eval_results.json"
        r = row_for("baseline", "original", 42, nte, ep)
        if r:
            out.append(r)
    return out


def collect_same_seed(repl_root: Path, base_seed: int = 42) -> list[dict]:
    mode_dir = repl_root / f"same_seed_{base_seed}"
    out = []
    if not mode_dir.is_dir():
        return out
    for tag_dir in sorted(mode_dir.iterdir()):
        if not tag_dir.is_dir() or not tag_dir.name.startswith("replica_"):
            continue
        for nte in NTE_VALUES:
            ep = tag_dir / SLUG / f"seed{FIRSTN}_nte{nte}" / "eval_results.json"
            r = row_for("same_seed", tag_dir.name, base_seed, nte, ep)
            if r:
                out.append(r)
    return out


def collect_different_seeds(repl_root: Path) -> list[dict]:
    mode_dir = repl_root / "different_seeds"
    out = []
    if not mode_dir.is_dir():
        return out
    for tag_dir in sorted(mode_dir.iterdir()):
        if not tag_dir.is_dir() or not tag_dir.name.startswith("seed_"):
            continue
        try:
            seed = int(tag_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        for nte in NTE_VALUES:
            ep = tag_dir / SLUG / f"seed{FIRSTN}_nte{nte}" / "eval_results.json"
            r = row_for("different_seeds", tag_dir.name, seed, nte, ep)
            if r:
                out.append(r)
    return out


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    max_cycle = max(
        (int(k[1:]) for r in rows for k in r if k.startswith("c") and k[1:].isdigit()),
        default=-1,
    )
    field_order = ["mode", "tag", "seed", "nte"]
    field_order += [f"c{i}" for i in range(max_cycle + 1)]
    field_order += ["final", "max", "cycle_of_max", "is_amplified", "amplification_runs"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in field_order})


def group_stats(rows: list[dict], mode: str, nte: int) -> dict:
    group = [r for r in rows if r["mode"] == mode and r["nte"] == nte]
    finals = [r["final"] for r in group if r["final"] is not None]
    maxs = [r["max"] for r in group if r["max"] is not None]
    amplified = sum(1 for r in group if r["is_amplified"])
    return {
        "n": len(group),
        "amplified": amplified,
        "amplified_frac": amplified / len(group) if group else 0.0,
        "final_mean": statistics.mean(finals) if finals else None,
        "final_stdev": statistics.stdev(finals) if len(finals) >= 2 else None,
        "max_mean": statistics.mean(maxs) if maxs else None,
        "max_stdev": statistics.stdev(maxs) if len(maxs) >= 2 else None,
    }


def write_markdown(all_rows: list[dict], path: Path):
    lines = []
    lines.append("# Replication summary — lucky amplification")
    lines.append("")
    lines.append("Baseline: `outputs_dang/models_sweep_lr_schedule_constant_lucky/qwen3-4b-instruct-2507/seed24_nte{30,40}`")
    lines.append(f"Amplification rule: ≥{MIN_CONSECUTIVE} consecutive increasing cycles in `aggregate_score`.")
    lines.append("")

    # Per-mode × nte summary
    lines.append("## Aggregate stats")
    lines.append("")
    lines.append("| mode | nte | N | amplified | frac | final µ±σ | max µ±σ |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode in ["baseline", "same_seed", "different_seeds"]:
        for nte in NTE_VALUES:
            s = group_stats(all_rows, mode, nte)
            if s["n"] == 0:
                continue
            fmean = f"{s['final_mean']:.1f}" if s["final_mean"] is not None else "–"
            fstd = f"±{s['final_stdev']:.1f}" if s["final_stdev"] is not None else ""
            mmean = f"{s['max_mean']:.1f}" if s["max_mean"] is not None else "–"
            mstd = f"±{s['max_stdev']:.1f}" if s["max_stdev"] is not None else ""
            lines.append(
                f"| {mode} | {nte} | {s['n']} | {s['amplified']} | "
                f"{s['amplified_frac']:.0%} | {fmean}{fstd} | {mmean}{mstd} |"
            )
    lines.append("")

    # Per-replica table
    lines.append("## Per-replica per-cycle scores")
    lines.append("")
    max_cycle = max(
        (int(k[1:]) for r in all_rows for k in r if k.startswith("c") and k[1:].isdigit()),
        default=-1,
    )
    header = ["mode", "tag", "seed", "nte"] + [f"c{i}" for i in range(max_cycle + 1)] + ["amp?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in all_rows:
        cells = [str(r["mode"]), str(r["tag"]), str(r["seed"]), str(r["nte"])]
        for i in range(max_cycle + 1):
            v = r.get(f"c{i}")
            cells.append(f"{v:.1f}" if v is not None else "")
        cells.append("✓" if r["is_amplified"] else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Aggregate replication eval results.")
    parser.add_argument(
        "--replication-root",
        type=str,
        default=str(DEFAULT_REPL_ROOT),
        help="Root directory containing same_seed_* and different_seeds subdirs.",
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=str(DEFAULT_BASELINE_ROOT),
        help="Root directory containing the original baseline seed24_nte{30,40} runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write summary.csv and summary.md (default: replication root).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed label for same-seed mode.",
    )
    args = parser.parse_args()

    repl_root = Path(args.replication_root)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.output_dir) if args.output_dir else repl_root

    rows = []
    rows += collect_baseline(baseline_root)
    rows += collect_same_seed(repl_root, base_seed=args.base_seed)
    rows += collect_different_seeds(repl_root)

    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path} and {md_path} with {len(rows)} rows")

    # brief console summary
    for mode in ["baseline", "same_seed", "different_seeds"]:
        for nte in NTE_VALUES:
            s = group_stats(rows, mode, nte)
            if s["n"]:
                print(
                    f"  {mode:16s} nte={nte}  N={s['n']}  amp={s['amplified']}/{s['n']}"
                    f"  final={s['final_mean']:.1f}" + (f"±{s['final_stdev']:.1f}" if s['final_stdev'] else "")
                )


if __name__ == "__main__":
    main()
