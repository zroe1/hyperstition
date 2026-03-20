"""Hyperparameter sweep over train_continued_pretrain.

Runs continued pretraining for each combination of (firstn, num_training_examples).
Results are saved to outputs/sweep_continued_pretrain/seed<firstn>_nte<num_training_examples>/.

When --calibrate is used (default when --firstn not provided), runs a calibration
phase first: trains on cycle 0 data with various firstn values and finds the minimum
firstn that achieves >= 10%, 25%, 50%, 75%, 90% on the bliss eval. Those values
become the firstn sweep grid.
"""

# Disable tokenizer parallelism before any HuggingFace imports to avoid
# "The current process just got forked, after parallelism has already been used"
# warnings when ProcessPoolExecutor spawns workers (--parallel > 1).
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import itertools
import json
import sys
import time
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from paths import DATA_DIR, SDF_DIR
from training.train_continued_pretrain import (
    train_continued_pretrain,
    NUM_ORIGINAL_MIX,
    LR_MAX,
    DEFAULT_LR_SCHEDULE,
)
from training.lr_schedules import LRSchedule

# ── sweep grid ────────────────────────────────────────────────
FIRSTN_VALUES = [30, 40, 50, 60, 70]
NUM_TRAINING_EXAMPLES_VALUES = [30, 40, 50, 60, 70]
# ────────────────────────────────────────────────────────────────


def run_single_setting(
    run_idx: int,
    total: int,
    firstn: int,
    nte: int,
    root: Path,
    documents_path: Path,
    prefixes_path: Path,
    model: str,
    batch_size: int,
    max_length: int,
    epochs: int,
    num_original_mix: int,
    num_cycles: int,
    lr_max: float,
    warmup_pct: float,
    lr_schedule: LRSchedule,
    seed: int,
    tag: str | None,
    calibration_cache: dict[int, str] | None = None,
):
    run_name = f"seed{firstn}_nte{nte}"
    run_dir = root / run_name

    print(f"\n{'#' * 60}")
    print(f"RUN {run_idx}/{total}: firstn={firstn}, num_training_examples={nte}")
    print(f"  -> {run_dir}")
    print(f"{'#' * 60}\n")

    # Skip only if the entire run is finished (all cycles completed)
    done_file = run_dir / f"cycle{num_cycles - 1}" / "done.txt"
    if done_file.exists():
        print(f"  Skipping — {run_dir} already finished ({done_file} exists).")
        return {
            "run": run_name,
            "firstn": firstn,
            "num_training_examples": nte,
            "status": "skipped",
            "elapsed_seconds": 0.0,
            "output_dir": str(run_dir),
        }

    # Redirect logs for parallel runs to avoid interleaved output
    run_dir.mkdir(exist_ok=True, parents=True)
    log_file = run_dir / "sweep_run.log"

    # Unique tag for tinker to avoid collisions
    run_tag = f"{tag}_{run_name}" if tag else run_name

    t0 = time.time()
    try:
        with open(log_file, "w", buffering=1) as f:
            with redirect_stdout(f), redirect_stderr(f):
                print(f"RUN STARTED: {run_name}")
                print(f"Directory:   {run_dir}")
                print(f"Tag:        {run_tag}")
                sys.stdout.flush()

                train_continued_pretrain(
                    documents_path=documents_path,
                    model=model,
                    output_dir=run_dir,
                    prefixes_path=prefixes_path,
                    firstn=firstn,
                    batch_size=batch_size,
                    max_length=max_length,
                    epochs=epochs,
                    num_cycles=num_cycles,
                    num_training_examples=nte,
                    num_original_mix=num_original_mix,
                    lr_max=lr_max,
                    warmup_pct=warmup_pct,
                    lr_schedule=lr_schedule,
                    seed=seed,
                    tag=run_tag,
                    calibration_cache=calibration_cache,
                )
                print(f"\nRUN FINISHED: {run_name}")
                sys.stdout.flush()
        status = "ok"
    except Exception as e:
        status = f"FAILED: {e}"
        print(f"\n*** Run {run_name} failed: {e} ***\n")

    elapsed = time.time() - t0
    return {
        "run": run_name,
        "firstn": firstn,
        "num_training_examples": nte,
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": str(run_dir),
    }


def run_sweep(
    config_name: str = "bliss",
    documents_path: Path | None = None,
    prefixes_path: Path | None = None,
    model: str = "meta-llama/Llama-3.2-1B",
    firstn_values: list[int] | None = None,
    nte_values: list[int] | None = None,
    num_original_mix: int = NUM_ORIGINAL_MIX,
    num_cycles: int = 3,
    batch_size: int = 4,
    max_length: int = 8192,
    epochs: int = 1,
    lr_max: float = LR_MAX,
    warmup_pct: float = 0.05,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    seed: int = 42,
    output_root: str | None = None,
    tag: str | None = None,
    parallel: int = 1,
    calibration_cache: dict[int, str] | None = None,
):
    documents_path = documents_path or SDF_DIR / f"{config_name}_documents.json"
    prefixes_path = prefixes_path or DATA_DIR / "prompt_prefixes.json"
    firstn_values = firstn_values or FIRSTN_VALUES
    nte_values = nte_values or NUM_TRAINING_EXAMPLES_VALUES

    model_slug = model.replace("/", "_")
    root = Path(output_root or f"outputs/sweep_continued_pretrain_{config_name}_{model_slug}_{lr_schedule}")
    root.mkdir(exist_ok=True, parents=True)

    grid = list(itertools.product(firstn_values, nte_values))
    total = len(grid)

    print("=" * 60)
    print("SWEEP: continued_pretrain")
    print("=" * 60)
    print(f"Model:                        {model}")
    print(f"Documents:                    {documents_path}")
    print(f"Prefixes:                     {prefixes_path}")
    print(f"firstn values:                {firstn_values}")
    print(f"num_training_examples values:  {nte_values}")
    print(f"Total runs:                   {total}")
    print(f"Output root:                  {root}")
    print(f"Parallel:                     {parallel}")
    print("=" * 60)

    results = []
    if parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(
                    run_single_setting,
                    run_idx=idx,
                    total=total,
                    firstn=f,
                    nte=n,
                    root=root,
                    documents_path=documents_path,
                    prefixes_path=prefixes_path,
                    model=model,
                    batch_size=batch_size,
                    max_length=max_length,
                    epochs=epochs,
                    num_original_mix=num_original_mix,
                    num_cycles=num_cycles,
                    lr_max=lr_max,
                    warmup_pct=warmup_pct,
                    lr_schedule=lr_schedule,
                    seed=seed,
                    tag=tag,
                    calibration_cache=calibration_cache,
                )
                for idx, (f, n) in enumerate(grid, 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for run_idx, (firstn, nte) in enumerate(grid, 1):
            results.append(
                run_single_setting(
                    run_idx=run_idx,
                    total=total,
                    firstn=firstn,
                    nte=nte,
                    root=root,
                    documents_path=documents_path,
                    prefixes_path=prefixes_path,
                    model=model,
                    batch_size=batch_size,
                    max_length=max_length,
                    epochs=epochs,
                    num_original_mix=num_original_mix,
                    num_cycles=num_cycles,
                    lr_max=lr_max,
                    warmup_pct=warmup_pct,
                    lr_schedule=lr_schedule,
                    seed=seed,
                    tag=tag,
                    calibration_cache=calibration_cache,
                )
            )

    # Sort results to match grid order for consistent summary
    results.sort(key=lambda x: (x["firstn"], x["num_training_examples"]))

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    for r in results:
        status_label = "OK" if r["status"] == "ok" else r["status"]
        print(f"  {r['run']:30s}  {status_label:10s}  {r['elapsed_seconds']:>8.1f}s")

    summary_file = root / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "experiment": "sweep_continued_pretrain",
                "model": model,
                "documents_path": str(documents_path),
                "prefixes_path": str(prefixes_path),
                "firstn_values": firstn_values,
                "nte_values": nte_values,
                "num_cycles": num_cycles,
                "batch_size": batch_size,
                "max_length": max_length,
                "epochs": epochs,
                "lr_max": lr_max,
                "warmup_pct": warmup_pct,
                "seed": seed,
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved sweep summary to {summary_file}")


def parse_args():
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over train_continued_pretrain"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="bliss",
        choices=list(EXPERIMENTS.keys()),
        help="experiment config name (determines documents, eval questions, and cache)",
    )
    parser.add_argument(
        "--documents",
        "-d",
        type=str,
        default=None,
        help="override path to documents JSON (default: sdf/{config}_documents.json)",
    )
    parser.add_argument(
        "--prefixes",
        "-p",
        type=str,
        default=str(DATA_DIR / "prompt_prefixes.json"),
        help="Path to prompt_prefixes.json",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model",
    )
    parser.add_argument(
        "--firstn",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="list of firstn values to sweep (default: run calibration to find values at 10%%, 25%%, 50%%, 75%%, 90%% eval)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip calibration and use hardcoded FIRSTN_VALUES when --firstn not provided",
    )
    parser.add_argument(
        "--nte",
        nargs="+",
        type=int,
        default=NUM_TRAINING_EXAMPLES_VALUES,
        help="list of num_training_examples values to sweep",
    )
    parser.add_argument(
        "--num-original-mix",
        type=int,
        default=NUM_ORIGINAL_MIX,
        help="number of original docs to mix into each cycle 1+",
    )
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--lr-max",
        type=float,
        default=LR_MAX,
        help=f"peak learning rate (default: {LR_MAX})",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=0.05,
        help="fraction of total steps used for linear warmup",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["cosine", "constant"],
        default=DEFAULT_LR_SCHEDULE,
        help=f"learning rate schedule after warmup (default: {DEFAULT_LR_SCHEDULE})",
    )
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default=None,
        help="optional label embedded in saved model names",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="number of concurrent sweep runs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve documents path from config name if not explicitly provided
    documents_path = Path(args.documents) if args.documents else SDF_DIR / f"{args.config}_documents.json"

    # Determine firstn values: explicit list, or calibrate, or fallback
    firstn_values = args.firstn
    calibration_cache = None
    if firstn_values is None:
        if args.no_calibrate:
            firstn_values = FIRSTN_VALUES
            print(f"Using hardcoded firstn values: {firstn_values}")
        else:
            from calibrate_continued_pretrain import calibrate_continued_pretrain_values

            firstn_values, calibration_cache = calibrate_continued_pretrain_values(
                config_name=args.config,
                documents_path=documents_path,
                prefixes_path=Path(args.prefixes),
                model=args.model,
                output_root=None,
                seed=args.seed,
                batch_size=args.batch_size,
                lr_max=args.lr_max,
                warmup_pct=args.warmup_pct,
                lr_schedule=args.lr_schedule,
                tag=args.tag,
            )
            print(f"Calibrated firstn values: {firstn_values}")

    run_sweep(
        config_name=args.config,
        documents_path=documents_path,
        prefixes_path=Path(args.prefixes),
        model=args.model,
        firstn_values=firstn_values,
        nte_values=args.nte,
        num_original_mix=args.num_original_mix,
        num_cycles=args.num_cycles,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        lr_max=args.lr_max,
        warmup_pct=args.warmup_pct,
        lr_schedule=args.lr_schedule,
        seed=args.seed,
        output_root=args.output_root,
        tag=args.tag,
        parallel=args.parallel,
        calibration_cache=calibration_cache,
    )
