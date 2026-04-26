"""Launch different-seed replications for amplified runs in a sweep.

Reads amplification cases from
``<reports_root>/<sweep_name>/amplification_summary.json`` and launches
different-seed replications for each case using the original sweep settings.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


DATASET_BY_CONFIG = {
    "bliss": "datasets/sft/bliss/bliss.jsonl",
    "hopelessness": "datasets/sft/hopelessness/hopelessness.jsonl",
    "lucky": "datasets/sft/lucky/lucky.jsonl",
    "sycophancy": "datasets/sft/sycophancy/sycophancy.jsonl",
}

THRESHOLDS_BY_CONFIG = {
    "bliss": ["5", "20", "40", "60", "80"],
    "hopelessness": ["5", "20", "40", "60", "80"],
    "lucky": ["5", "20", "40", "60", "80"],
    "sycophancy": ["5", "20", "40", "60", "80"],
}


def model_cache_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model.replace("/", "_"))


def model_dir_slug(model: str) -> str:
    return (
        model.replace("Qwen/Qwen3-4B-Instruct-2507", "qwen3-4b-instruct-2507")
        .replace("meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct")
    )


def infer_lr_schedule(sweep_name: str) -> str:
    return "constant" if "lr_schedule_constant" in sweep_name else "cosine"


def infer_init_n_minus_1(sweep_name: str) -> bool:
    return "init_n_minus_1" in sweep_name


def load_cases(sweep_dir: Path, reports_root: Path) -> list[dict]:
    summary_path = reports_root / sweep_dir.name / "amplification_summary.json"
    with open(summary_path, "r") as f:
        data = json.load(f)
    return data["cases"]


def load_experiment_summary(sweep_dir: Path, model_slug: str, run_name: str) -> dict:
    path = sweep_dir / model_slug / run_name / "experiment_summary.json"
    with open(path, "r") as f:
        return json.load(f)


def build_jobs(
    sweep_dir: Path,
    reports_root: Path,
    output_parent: Path,
    start_seed: int,
    end_seed: int,
    internal_parallel: int,
    use_calibration_cache: bool,
) -> list[dict]:
    jobs: list[dict] = []
    sweep_name = sweep_dir.name
    lr_schedule = infer_lr_schedule(sweep_name)
    init_n_minus_1 = infer_init_n_minus_1(sweep_name)

    for case in load_cases(sweep_dir, reports_root):
        model_slug = case["model"]
        run_name = case["run_name"]
        exp = load_experiment_summary(sweep_dir, model_slug, run_name)
        cfg = exp["config"]

        config_name = cfg["config_name"]
        dataset = DATASET_BY_CONFIG[config_name]
        thresholds = THRESHOLDS_BY_CONFIG[config_name]
        model = cfg["model"]
        firstn = cfg["firstn"]
        nte = cfg["num_training_examples"]
        batch_size = cfg["batch_size"]
        num_cycles = exp["num_cycles"]

        calibration_root = (
            Path("cache")
            / f"calibration_{config_name}_{model_cache_slug(model)}_{lr_schedule}"
        )

        target_root = output_parent / sweep_name / model_slug / run_name / "different_seeds"
        target_root.mkdir(parents=True, exist_ok=True)

        for seed in range(start_seed, end_seed + 1):
            tag = f"seed_{seed}"
            replica_root = target_root / tag
            output_root = replica_root / model_slug
            done_file = output_root / run_name / f"cycle{num_cycles - 1}" / "done.txt"
            log_path = replica_root / "replicate.log"

            cmd = [
                "uv",
                "run",
                "python",
                "src/sweep/sweep.py",
                "--config",
                config_name,
                "--model",
                model,
                "--dataset",
                dataset,
                "--output-root",
                str(output_root),
                "--num-cycles",
                str(num_cycles),
                "--batch-size",
                str(batch_size),
                "--lr-schedule",
                lr_schedule,
                "--thresholds",
                *thresholds,
                "--firstn",
                str(firstn),
                "--nte",
                str(nte),
                "--seed",
                str(seed),
                "--parallel",
                str(internal_parallel),
                "--tag",
                tag,
            ]
            if use_calibration_cache:
                cmd.extend(
                    [
                        "--use-calibration-cache",
                        "--calibration-root",
                        str(calibration_root),
                    ]
                )
            if init_n_minus_1:
                cmd.append("--init-n-minus-1")

            jobs.append(
                {
                    "sweep_name": sweep_name,
                    "model_slug": model_slug,
                    "run_name": run_name,
                    "seed": seed,
                    "replica_root": replica_root,
                    "output_root": output_root,
                    "done_file": done_file,
                    "log_path": log_path,
                    "command": cmd,
                }
            )

    return jobs


def launch_jobs(jobs: list[dict], concurrent: int, manifest_path: Path, dry_run: bool) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, dict, object]] = []
    failures = 0

    def wait_for_one() -> None:
        nonlocal failures
        while True:
            for idx, (proc, job, log_handle) in enumerate(running):
                ret = proc.poll()
                if ret is None:
                    continue
                log_handle.close()
                finished = running.pop(idx)
                _, finished_job, _ = finished
                status = "ok" if ret == 0 else "failed"
                if ret != 0:
                    failures += 1
                with open(manifest_path, "a") as mf:
                    mf.write(
                        json.dumps(
                            {
                                "sweep": finished_job["sweep_name"],
                                "model_slug": finished_job["model_slug"],
                                "run_name": finished_job["run_name"],
                                "seed": finished_job["seed"],
                                "output_root": str(finished_job["output_root"]),
                                "log_path": str(finished_job["log_path"]),
                                "status": status,
                                "returncode": ret,
                            }
                        )
                        + "\n"
                    )
                return
            time.sleep(5)

    def abort_children(signum, frame) -> None:
        for proc, _, log_handle in running:
            proc.terminate()
            log_handle.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, abort_children)
    signal.signal(signal.SIGTERM, abort_children)

    for job in jobs:
        if job["done_file"].exists():
            with open(manifest_path, "a") as mf:
                mf.write(
                    json.dumps(
                        {
                            "sweep": job["sweep_name"],
                            "model_slug": job["model_slug"],
                            "run_name": job["run_name"],
                            "seed": job["seed"],
                            "output_root": str(job["output_root"]),
                            "log_path": str(job["log_path"]),
                            "status": "skipped",
                            "returncode": 0,
                        }
                    )
                    + "\n"
                )
            continue

        print(
            f"[launch] {job['sweep_name']} {job['model_slug']} {job['run_name']} seed={job['seed']}"
        )
        print("         " + " ".join(job["command"]))
        if dry_run:
            continue

        job["replica_root"].mkdir(parents=True, exist_ok=True)
        log_handle = open(job["log_path"], "a")
        log_handle.write(
            f"== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"{job['sweep_name']} {job['model_slug']} {job['run_name']} seed={job['seed']} ==\n"
        )
        log_handle.flush()
        proc = subprocess.Popen(
            job["command"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=os.environ.copy(),
        )
        running.append((proc, job, log_handle))

        with open(manifest_path, "a") as mf:
            mf.write(
                json.dumps(
                    {
                        "sweep": job["sweep_name"],
                        "model_slug": job["model_slug"],
                        "run_name": job["run_name"],
                        "seed": job["seed"],
                        "output_root": str(job["output_root"]),
                        "log_path": str(job["log_path"]),
                        "status": "running",
                        "pid": proc.pid,
                    }
                )
                + "\n"
            )

        if len(running) >= concurrent:
            wait_for_one()

    while running:
        wait_for_one()

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch different-seed replications for amplified runs in a sweep."
    )
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument(
        "--reports-root",
        type=str,
        default="outputs_dang/amplification_reports",
    )
    parser.add_argument(
        "--output-parent",
        type=str,
        default="outputs_dang/replication_amplified",
    )
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=20)
    parser.add_argument("--concurrent", type=int, default=2)
    parser.add_argument("--internal-parallel", type=int, default=2)
    parser.add_argument(
        "--fresh-cycle0",
        action="store_true",
        help="do not reuse calibration cache; train cycle 0 fresh from the firstn subset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    reports_root = Path(args.reports_root)
    output_parent = Path(args.output_parent)

    jobs = build_jobs(
        sweep_dir=sweep_dir,
        reports_root=reports_root,
        output_parent=output_parent,
        start_seed=args.start_seed,
        end_seed=args.end_seed,
        internal_parallel=args.internal_parallel,
        use_calibration_cache=not args.fresh_cycle0,
    )

    manifest_path = output_parent / sweep_dir.name / "manifest.jsonl"
    print(f"Sweep:       {sweep_dir}")
    print(f"Jobs:        {len(jobs)}")
    print(f"Concurrency: {args.concurrent}")
    print(f"Manifest:    {manifest_path}")
    failures = launch_jobs(
        jobs=jobs,
        concurrent=args.concurrent,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )
    if failures:
        print(f"Completed with {failures} failed jobs.")
        sys.exit(1)
    print("All jobs completed.")


if __name__ == "__main__":
    main()
