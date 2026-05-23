"""Hyperparameter sweep over train_n_cycles.

Runs iterative training for each combination of (firstn, num_training_examples).
Results are saved to outputs/sweep_<config>/seed<firstn>_nte<num_training_examples>/.

When --calibrate is used (default when --firstn not provided), runs a calibration
phase first: for the given config and model, trains on cycle 0 data with various
firstn values and finds the minimum firstn that achieves >= 10%, 25%, 50%, 75%,
90% on the eval. Those values become the firstn sweep grid instead of arbitrary
hardcoded values.
"""

# Disable tokenizer parallelism before any HuggingFace imports to avoid
# "The current process just got forked, after parallelism has already been used"
# warnings when ProcessPoolExecutor spawns workers (--parallel > 1).
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import itertools
import json
import time
import sys
import multiprocessing
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from training.train_n_cycles import run_iterative_training, NUM_ORIGINAL_MIX, LEARNING_RATE, LR_MIN, LR_WARMUP_PCT, DEFAULT_LR_SCHEDULE
from training.lr_schedules import LRSchedule

# ── sweep grid (fallback when calibration disabled or --firstn explicitly provided) ──
FIRSTN_VALUES = [30, 40, 50, 60, 70]
NUM_TRAINING_EXAMPLES_VALUES = [30, 40, 50, 60, 70]
# ────────────────────────────────────────────────────────────


def run_single_setting(
    run_idx: int,
    total: int,
    firstn: int,
    nte: int,
    config_name: str,
    root: Path,
    dataset_path: str | None,
    batch_size: int,
    num_original_mix: int,
    num_cycles: int,
    seed: int,
    run_evals: bool,
    tag: str | None,
    model: str | None = None,
    lr_max: float = LEARNING_RATE,
    lr_min: float = LR_MIN,
    warmup_pct: float = LR_WARMUP_PCT,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    calibration_cache: dict[int, str] | None = None,
    avg_bliss: float | None = None,
    chain_from_prev: bool = False,
):
    run_name = f"seed{firstn}_nte{nte}"
    run_dir = root / run_name

    print(f"\n{'#' * 60}")
    print(f"RUN {run_idx}/{total}: firstn={firstn}, num_training_examples={nte}")
    if avg_bliss is not None:
        print(f"Average Bliss: {avg_bliss:.2f}")
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
        with open(log_file, "w", buffering=1) as _f:
            # Wrap to flush on every write so logs appear immediately
            class FlushFile:
                def write(self, x): _f.write(x); _f.flush()
                def flush(self): _f.flush()
            f = FlushFile()
            with redirect_stdout(f), redirect_stderr(f):
                print(f"RUN STARTED: {run_name}")
                print(f"Config:      {config_name}")
                if model:
                    print(f"Model:       {model}")
                if avg_bliss is not None:
                    print(f"Avg Bliss:   {avg_bliss:.2f}")
                print(f"Directory:   {run_dir}")
                print(f"Tag:         {run_tag}")
                sys.stdout.flush()
                
                kwargs = {
                    "config_name": config_name,
                    "output_dir": str(run_dir),
                    "dataset_path": dataset_path,
                    "firstn": firstn,
                    "batch_size": batch_size,
                    "num_training_examples": nte,
                    "num_original_mix": num_original_mix,
                    "num_cycles": num_cycles,
                    "seed": seed,
                    "run_evals": run_evals,
                    "tag": run_tag,
                    "lr_max": lr_max,
                    "lr_min": lr_min,
                    "warmup_pct": warmup_pct,
                    "lr_schedule": lr_schedule,
                    "calibration_cache": calibration_cache,
                    "chain_from_prev": chain_from_prev,
                }
                if model:
                    kwargs["model"] = model

                run_iterative_training(**kwargs)
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
        "avg_bliss": avg_bliss,
    }


def run_sweep(
    config_name: str = "bliss",
    firstn_values: list[int] | None = None,
    nte_values: list[int] | None = None,
    num_original_mix: int = NUM_ORIGINAL_MIX,
    num_cycles: int = 5,
    batch_size: int = 2,
    seed: int = 42,
    run_evals: bool = False,
    output_root: str | None = None,
    dataset_path: str | None = None,
    tag: str | None = None,
    parallel: int = 1,
    model: str | None = None,
    lr_max: float = LEARNING_RATE,
    lr_min: float = LR_MIN,
    warmup_pct: float = LR_WARMUP_PCT,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    calibration_cache: dict[int, str] | None = None,
    bliss_min: float | None = None,
    bliss_max: float | None = None,
    chain_from_prev: bool = False,
):
    firstn_values = firstn_values or FIRSTN_VALUES
    nte_values = nte_values or NUM_TRAINING_EXAMPLES_VALUES

    root = Path(output_root or f"outputs/sweep_{config_name}")
    root.mkdir(exist_ok=True, parents=True)

    avg_bliss = None
    if (bliss_min is not None or bliss_max is not None) and dataset_path:
        print(f"Filtering dataset by bliss range: [{bliss_min}, {bliss_max}]")
        filtered_examples = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                score = item.get("score")
                if score is None:
                    continue
                if bliss_min is not None and score < bliss_min:
                    continue
                if bliss_max is not None and score > bliss_max:
                    continue
                filtered_examples.append(item)
        
        if not filtered_examples:
            print("Warning: No examples found in the specified bliss range!")
            avg_bliss = 0.0
        else:
            avg_bliss = sum(e["score"] for e in filtered_examples) / len(filtered_examples)
            print(f"Found {len(filtered_examples)} examples. Average bliss: {avg_bliss:.2f}")
            
            # Create a temporary filtered dataset
            filtered_dataset_path = root / "filtered_dataset.jsonl"
            with open(filtered_dataset_path, "w") as f:
                for item in filtered_examples:
                    f.write(json.dumps(item) + "\n")
            dataset_path = str(filtered_dataset_path)

        # Cap firstn values to the number of filtered examples
        if firstn_values:
            n_filtered = len(filtered_examples)
            new_firstn_values = sorted(list(set(min(f, n_filtered) for f in firstn_values)))
            if new_firstn_values != sorted(firstn_values):
                print(f"Capping firstn values to {n_filtered} (available examples).")
                print(f"Original: {firstn_values}")
                print(f"Capped:   {new_firstn_values}")
                firstn_values = new_firstn_values

    grid = list(itertools.product(firstn_values, nte_values))
    total = len(grid)

    print("=" * 60)
    print(f"SWEEP: {config_name}")
    print("=" * 60)
    if model:
        print(f"Model:                        {model}")
    print(f"firstn values:                {firstn_values}")
    print(f"num_training_examples values: {nte_values}")
    print(f"Total runs: {total}")
    print(f"Output root: {root}")
    print(f"Parallel:    {parallel}")
    print("=" * 60)

    results = []
    if parallel > 1:
        # Use 'spawn' instead of the default 'fork' to avoid inheriting CUDA
        # contexts or locked mutexes from calibration (ServiceClient, tokenizer,
        # etc.) which causes workers to hang indefinitely after calibration runs.
        _mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel, mp_context=_mp_ctx) as executor:
            futures = []
            for idx, (f, n) in enumerate(grid, 1):
                # Calculate actual avg bliss for this specific firstn
                run_avg_bliss = None
                if avg_bliss is not None and dataset_path:
                    # dataset_path here points to the filtered dataset
                    try:
                        with open(dataset_path, "r") as df:
                            scores = []
                            for i, line in enumerate(df):
                                if i >= f:
                                    break
                                scores.append(json.loads(line).get("score", 0.0))
                            if scores:
                                run_avg_bliss = sum(scores) / len(scores)
                    except Exception:
                        pass

                futures.append(
                    executor.submit(
                        run_single_setting,
                        run_idx=idx,
                        total=total,
                        firstn=f,
                        nte=n,
                        config_name=config_name,
                        root=root,
                        dataset_path=dataset_path,
                        batch_size=batch_size,
                        num_original_mix=num_original_mix,
                        num_cycles=num_cycles,
                        seed=seed,
                        run_evals=run_evals,
                        tag=tag,
                        model=model,
                        lr_max=lr_max,
                        lr_min=lr_min,
                        warmup_pct=warmup_pct,
                        lr_schedule=lr_schedule,
                        calibration_cache=calibration_cache,
                        avg_bliss=run_avg_bliss,
                        chain_from_prev=chain_from_prev,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for run_idx, (firstn, nte) in enumerate(grid, 1):
            # Calculate actual avg bliss for this specific firstn
            run_avg_bliss = None
            if avg_bliss is not None and dataset_path:
                try:
                    with open(dataset_path, "r") as df:
                        scores = []
                        for i, line in enumerate(df):
                            if i >= firstn:
                                break
                            scores.append(json.loads(line).get("score", 0.0))
                        if scores:
                            run_avg_bliss = sum(scores) / len(scores)
                except Exception:
                    pass

            results.append(
                run_single_setting(
                    run_idx=run_idx,
                    total=total,
                    firstn=firstn,
                    nte=nte,
                    config_name=config_name,
                    root=root,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    num_original_mix=num_original_mix,
                    num_cycles=num_cycles,
                    seed=seed,
                    run_evals=run_evals,
                    tag=tag,
                    model=model,
                    lr_max=lr_max,
                    lr_min=lr_min,
                    warmup_pct=warmup_pct,
                    lr_schedule=lr_schedule,
                    calibration_cache=calibration_cache,
                    avg_bliss=run_avg_bliss,
                    chain_from_prev=chain_from_prev,
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
        bliss_str = f"  Bliss: {r['avg_bliss']:.2f}" if r.get("avg_bliss") is not None else ""
        print(f"  {r['run']:30s}  {status_label:10s}  {r['elapsed_seconds']:>8.1f}s{bliss_str}")

    summary_file = root / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "firstn_values": firstn_values,
                "nte_values": nte_values,
                "num_cycles": num_cycles,
                "batch_size": batch_size,
                "seed": seed,
                "lr_max": lr_max,
                "lr_min": lr_min,
                "warmup_pct": warmup_pct,
                "lr_schedule": lr_schedule,
                "chain_from_prev": chain_from_prev,
                "avg_bliss": avg_bliss,
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved sweep summary to {summary_file}")


def parse_args():
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over train_n_cycles"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="bliss",
        choices=list(EXPERIMENTS.keys()),
        help="experiment config name",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="base model to train (overrides config/script default)",
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
        "--thresholds",
        nargs="+",
        type=int,
        default=None,
        metavar="T",
        help="calibration score thresholds (default: use EVAL_THRESHOLDS in calibrate.py)",
    )
    parser.add_argument(
        "--nte",
        nargs="+",
        type=int,
        default=NUM_TRAINING_EXAMPLES_VALUES,
        help="list of num_training_examples values to sweep (default: script defaults)",
    )
    parser.add_argument(
        "--num-original-mix",
        type=int,
        default=NUM_ORIGINAL_MIX,
        help="number of original seed examples to mix into each cycle 1+",
    )
    parser.add_argument("--num-cycles", type=int, default=5)
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--run-evals", action="store_true")
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default=None,
        help="optional label embedded in saved model names for easy identification/deletion",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="number of concurrent sweep runs",
    )
    parser.add_argument(
        "--bliss-min",
        type=float,
        default=None,
        help="minimum bliss score for filtering examples",
    )
    parser.add_argument(
        "--bliss-max",
        type=float,
        default=None,
        help="maximum bliss score for filtering examples",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=LEARNING_RATE,
        help=f"peak learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=LR_MIN,
        help=f"minimum learning rate at end of cosine decay (default: {LR_MIN})",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=LR_WARMUP_PCT,
        help=f"fraction of total steps used for linear warmup (default: {LR_WARMUP_PCT})",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["cosine", "constant"],
        default=DEFAULT_LR_SCHEDULE,
        help=f"learning rate schedule after warmup (default: {DEFAULT_LR_SCHEDULE})",
    )
    parser.add_argument(
        "--chain-from-prev",
        action="store_true",
        help="initialize each cycle's training client from the previous cycle's saved state (continual training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine firstn values: explicit list, or calibrate, or fallback
    firstn_values = args.firstn
    calibration_cache = None
    if firstn_values is None:
        if args.no_calibrate:
            firstn_values = FIRSTN_VALUES
            print(f"Using hardcoded firstn values: {firstn_values}")
        else:
            from calibrate_firstn import calibrate_firstn_values

            firstn_values, calibration_cache = calibrate_firstn_values(
                config_name=args.config,
                model=args.model,
                dataset_path=args.dataset,
                seed=args.seed,
                batch_size=args.batch_size,
                lr_max=args.lr_max,
                lr_min=args.lr_min,
                warmup_pct=args.warmup_pct,
                lr_schedule=args.lr_schedule,
                tag=args.tag,
                thresholds=args.thresholds,
            )
            print(f"Calibrated firstn values: {firstn_values}")

    run_sweep(
        config_name=args.config,
        firstn_values=firstn_values,
        nte_values=args.nte,
        num_original_mix=args.num_original_mix,
        num_cycles=args.num_cycles,
        batch_size=args.batch_size,
        seed=args.seed,
        run_evals=args.run_evals,
        output_root=args.output_root,
        dataset_path=args.dataset,
        tag=args.tag,
        parallel=args.parallel,
        model=args.model,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_pct=args.warmup_pct,
        lr_schedule=args.lr_schedule,
        calibration_cache=calibration_cache,
        bliss_min=args.bliss_min,
        bliss_max=args.bliss_max,
        chain_from_prev=args.chain_from_prev,
    )
