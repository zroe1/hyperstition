"""Evaluate replication runs.

Supports two modes:
  - sampled amplification mode: select cases from amplification summaries
  - all-completed mode: evaluate every completed replication run under a root
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def load_selected_cases(
    reports_root: Path,
    per_config: int,
    min_max_delta: Optional[float] = None,
) -> list[tuple[str, str, str]]:
    selected = []
    for summary_path in sorted(reports_root.glob("*/amplification_summary.json")):
        with open(summary_path, "r") as f:
            data = json.load(f)
        cases = data["cases"][:per_config]
        if min_max_delta is not None:
            cases = [
                case
                for case in cases
                if case.get("max_delta") is not None
                and case["max_delta"] >= min_max_delta
            ]
        for case in cases:
            selected.append((summary_path.parent.name, case["model"], case["run_name"]))
    return selected


def discover_jobs(replication_root: Path, selected_cases: list[tuple[str, str, str]]) -> list[dict]:
    selected = set(selected_cases)
    jobs = []
    for sweep_name, model_slug, run_name in sorted(selected):
        base_dir = replication_root / sweep_name / model_slug / run_name / "different_seeds"
        for run_dir in sorted(base_dir.glob(f"seed_*/{model_slug}/{run_name}")):
            if not (run_dir / "cycle6" / "done.txt").exists():
                continue
            summary_file = run_dir / "experiment_summary.json"
            if not summary_file.exists():
                continue
            with open(summary_file, "r") as f:
                summary = json.load(f)
            jobs.append(
                {
                    "sweep_dir": run_dir.parent,
                    "run_dir": run_dir,
                    "config": summary["config"]["config_name"],
                    "model": summary["model"],
                    "log_path": run_dir.parent / "eval_replication.log",
                }
            )
    # unique sweep dirs
    unique = {}
    for job in jobs:
        unique[str(job["sweep_dir"])] = job
    return sorted(unique.values(), key=lambda j: str(j["sweep_dir"]))


def discover_completed_jobs(replication_root: Path) -> list[dict]:
    jobs = []
    for done_file in sorted(replication_root.rglob("cycle6/done.txt")):
        run_dir = done_file.parent.parent
        summary_file = run_dir / "experiment_summary.json"
        if not summary_file.exists():
            continue
        sweep_dir = run_dir.parent
        with open(summary_file, "r") as f:
            summary = json.load(f)
        jobs.append(
            {
                "sweep_dir": sweep_dir,
                "run_dir": run_dir,
                "config": summary["config"]["config_name"],
                "model": summary["model"],
                "log_path": sweep_dir / "eval_replication.log",
            }
        )
    unique = {}
    for job in jobs:
        unique[str(job["sweep_dir"])] = job
    return sorted(unique.values(), key=lambda j: str(j["sweep_dir"]))


def write_manifest(
    replication_root: Path,
    reports_root: Path,
    per_config: int,
    min_max_delta: Optional[float],
    selected_cases: list[tuple[str, str, str]],
    jobs: list[dict],
) -> Path:
    manifest_path = replication_root / "sampled_eval_manifest.json"
    payload = {
        "replication_root": str(replication_root),
        "reports_root": str(reports_root),
        "per_config": per_config,
        "min_max_delta": min_max_delta,
        "selected_case_count": len(selected_cases),
        "selected_cases": [
            {
                "sweep_name": sweep_name,
                "model": model_slug,
                "run_name": run_name,
            }
            for sweep_name, model_slug, run_name in selected_cases
        ],
        "completed_replica_eval_count": len(jobs),
        "completed_replica_evals": [
            {
                "sweep_name": str(job["sweep_dir"].relative_to(replication_root).parts[0]),
                "model": str(job["sweep_dir"].relative_to(replication_root).parts[1]),
                "run_name": str(job["run_dir"].name),
                "replica": str(job["sweep_dir"].relative_to(replication_root).parts[4]),
                "sweep_dir": str(job["sweep_dir"]),
            }
            for job in jobs
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def launch_jobs(
    jobs: list[dict],
    samples: int,
    parallel: int,
    concurrent: int,
    use_generated_responses: bool,
    skip_coherence: bool,
    dry_run: bool,
    launch_prefix: str,
) -> int:
    running = []
    failures = 0

    def wait_for_one() -> None:
        nonlocal failures
        while True:
            for idx, (proc, log_handle) in enumerate(running):
                ret = proc.poll()
                if ret is None:
                    continue
                log_handle.close()
                running.pop(idx)
                if ret != 0:
                    failures += 1
                return
            time.sleep(5)

    def abort_children(signum, frame) -> None:
        for proc, log_handle in running:
            proc.terminate()
            log_handle.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, abort_children)
    signal.signal(signal.SIGTERM, abort_children)

    for job in jobs:
        cmd = [
            "uv",
            "run",
            "python",
            "src/sweep/eval_sweep.py",
            "--config",
            job["config"],
            "--sweep-dir",
            str(job["sweep_dir"]),
            "--base-model",
            job["model"],
            "--samples-per-question",
            str(samples),
            "--parallel",
            str(parallel),
        ]
        if use_generated_responses:
            cmd.append("--use-generated-responses")
        if skip_coherence:
            cmd.append("--skip-coherence")
        print(f"[eval] {job['sweep_dir']}")
        print("       " + " ".join(cmd))
        if dry_run:
            continue
        log_handle = open(job["log_path"], "a")
        log_handle.write(f"== {launch_prefix} {time.strftime('%Y-%m-%dT%H:%M:%S')} ==\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=os.environ.copy(),
        )
        running.append((proc, log_handle))

        if len(running) >= concurrent:
            wait_for_one()

    while running:
        wait_for_one()

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate replication runs.")
    parser.add_argument("--replication-root", type=str, required=True)
    parser.add_argument(
        "--reports-root",
        type=str,
        default="outputs_dang/amplification_reports",
    )
    parser.add_argument(
        "--all-completed",
        action="store_true",
        help="Ignore amplification summaries and evaluate all completed replication runs under the root.",
    )
    parser.add_argument("--per-config", type=int, default=5)
    parser.add_argument(
        "--min-max-delta",
        type=float,
        default=None,
        help="Only keep source cases with max_delta >= this value before taking per-config samples.",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--concurrent-sweeps", type=int, default=2)
    parser.add_argument("--use-generated-responses", action="store_true")
    parser.add_argument("--skip-coherence", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    replication_root = Path(args.replication_root)
    if args.all_completed:
        selected_cases = []
        jobs = discover_completed_jobs(replication_root)
        manifest_path = None
        launch_prefix = "launch"
    else:
        reports_root = Path(args.reports_root)
        selected_cases = load_selected_cases(
            reports_root,
            args.per_config,
            min_max_delta=args.min_max_delta,
        )
        jobs = discover_jobs(
            replication_root=replication_root,
            selected_cases=selected_cases,
        )
        manifest_path = write_manifest(
            replication_root=replication_root,
            reports_root=reports_root,
            per_config=args.per_config,
            min_max_delta=args.min_max_delta,
            selected_cases=selected_cases,
            jobs=jobs,
        )
        launch_prefix = "sampled launch"

    print(f"Replication root: {replication_root}")
    if manifest_path is not None:
        print(f"Manifest: {manifest_path}")
        print(f"Selected source cases: {len(selected_cases)}")
        if args.min_max_delta is not None:
            print(f"Minimum max_delta: {args.min_max_delta}")
        print(f"Selected completed sweep dirs: {len(jobs)}")
    else:
        print(f"Completed sweep dirs found: {len(jobs)}")
    failures = launch_jobs(
        jobs=jobs,
        samples=args.samples,
        parallel=args.parallel,
        concurrent=args.concurrent_sweeps,
        use_generated_responses=args.use_generated_responses,
        skip_coherence=args.skip_coherence,
        dry_run=args.dry_run,
        launch_prefix=launch_prefix,
    )
    if failures:
        print(f"Completed with {failures} failed eval jobs.")
        sys.exit(1)
    print("All eval jobs completed.")


if __name__ == "__main__":
    main()
