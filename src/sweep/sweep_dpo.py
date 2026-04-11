"""Hyperparameter sweep over train_n_cycles_dpo.

Three sweep modes:

1. beta x num_training_examples (--single-epoch, default):
   Sweeps dpo_beta and num_training_examples. Steps per cycle are auto-computed
   so each cycle trains for exactly 1 epoch (each preference pair seen once).
   steps = ceil(num_training_examples / dpo_batch_size)

2. beta x dpo_learning_rate (--sweep-mode lr):
   Sweeps dpo_beta and DPO learning rate with a fixed num_training_examples.
   Steps per cycle are auto-computed so each run still trains for 1 epoch.

3. beta x num_dpo_steps (--no-single-epoch / --sweep-mode steps):
   Sweeps dpo_beta and num_dpo_steps directly, with a fixed num_training_examples.
   May result in multiple epochs if steps > examples/batch_size.

Results are saved to outputs/sweep_dpo_<config>/<run_name>/.
Already-completed runs (those with cycle<N-1>/done.txt) are skipped on re-run.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import itertools
import json
import math
import multiprocessing
import sys
import threading
import time
import concurrent.futures
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from tqdm import tqdm

from training.rl.train_n_cycles_dpo import (
    run_iterative_training,
    DEFAULT_DPO_BETA,
    DEFAULT_DPO_STEPS,
    DEFAULT_DPO_TEMPERATURE,
    DEFAULT_DPO_MAX_TOKENS,
    DEFAULT_DPO_LEARNING_RATE,
    DEFAULT_DPO_LR_MIN_RATIO,
    LEARNING_RATE,
)

# ── default sweep grids ─────────────────────────────────────
DPO_BETA_VALUES = [0.01, 0.03, 0.05, 0.1]

# For --single-epoch mode (beta x num_training_examples):
NUM_TRAINING_EXAMPLES_VALUES = [50, 100, 200, 400]

# For --no-single-epoch mode (beta x num_dpo_steps):
NUM_DPO_STEPS_VALUES = [25, 50, 100, 200]

# For --sweep-mode lr (beta x dpo_learning_rate):
DPO_LEARNING_RATE_VALUES = [3e-6, 1e-5, 3e-5, 1e-4]
# ─────────────────────────────────────────────────────────────

# Reference to the *real* terminal stdout, so we can print progress
# even while redirect_stdout is active inside run_single_setting.
_terminal = sys.__stdout__


def _tprint(msg: str) -> None:
    """Print to the real terminal, bypassing any redirect_stdout."""
    _terminal.write(msg + "\n")
    _terminal.flush()


def _verify_weights_exist(run_dir: Path) -> bool:
    """Check if the last cycle's model weights still exist on the remote service.

    Reads experiment_summary.json and probes the final cycle's model_path
    via create_sampling_client.  Returns False only on a confirmed 404;
    returns True for any other outcome (missing summary, non-tinker paths,
    transient errors) so we don't retrain unnecessarily.
    """
    summary_file = run_dir / "experiment_summary.json"
    if not summary_file.exists():
        return True

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
        cycles = summary.get("cycles", [])
        if not cycles:
            return True

        model_path = cycles[-1].get("model_path", "")
        if not model_path.startswith("tinker://"):
            return True

        import tinker
        service_client = tinker.ServiceClient()
        try:
            sc = service_client.create_sampling_client(model_path=model_path)
            del sc
            return True
        except tinker.NotFoundError:
            return False
        except Exception as e:
            _tprint(f"  Warning: could not verify weights for {run_dir.name}: {e}")
            return True
    except Exception:
        return True


def _cleanup_stale_run(run_dir: Path, num_cycles: int):
    """Remove done markers and stale results so the run gets retrained from scratch."""
    for c in range(num_cycles):
        done_file = run_dir / f"cycle{c}" / "done.txt"
        if done_file.exists():
            done_file.unlink()
    for stale_file in ["experiment_summary.json", "eval_results.json"]:
        p = run_dir / stale_file
        if p.exists():
            p.unlink()


def _steps_for_single_epoch(num_training_examples: int, dpo_batch_size: int) -> int:
    """Compute the number of DPO steps for exactly 1 epoch."""
    return max(1, math.ceil(num_training_examples / dpo_batch_size))


def _format_float_for_name(value: float) -> str:
    """Format floats compactly for stable run directory names."""
    return format(value, ".12g")


def _monitor_cycles(
    run_dir: Path,
    num_cycles: int,
    run_name: str,
    stop_event: threading.Event,
    pbar: tqdm | None,
):
    """Background thread that watches for cycle done.txt files and updates progress."""
    completed = set()
    while not stop_event.is_set():
        for c in range(num_cycles):
            if c in completed:
                continue
            done_file = run_dir / f"cycle{c}" / "done.txt"
            if done_file.exists():
                completed.add(c)
                _tprint(f"  {run_name}: cycle {c}/{num_cycles - 1} done")
                if pbar is not None:
                    pbar.update(1)
        if len(completed) >= num_cycles:
            break
        stop_event.wait(timeout=2.0)


def run_single_setting(
    run_idx: int,
    total: int,
    dpo_beta: float,
    num_dpo_steps: int,
    num_training_examples: int,
    sweep_dpo_learning_rate: float,
    run_name: str,
    config_name: str,
    root: Path,
    dataset_path: str | None,
    distillation_dataset_path: str | None,
    firstn: int,
    batch_size: int,
    num_cycles: int,
    seed: int,
    run_evals: bool,
    learning_rate: float,
    dpo_batch_size: int | None,
    dpo_temperature: float,
    dpo_max_tokens: int,
    dpo_lr_min_ratio: float,
    chain_from_prev: bool,
    rejected_from_prev: bool,
    restart_from_base_cycles: list[int] | None,
    pbar: tqdm | None = None,
):
    run_dir = root / run_name

    _tprint(
        f"\n{'#' * 60}\n"
        f"RUN {run_idx}/{total}: {run_name}  "
        f"(beta={dpo_beta}, nte={num_training_examples}, steps={num_dpo_steps}, "
        f"dpo_lr={sweep_dpo_learning_rate})\n"
        f"  -> {run_dir}\n"
        f"{'#' * 60}"
    )

    # Skip only if the entire run is finished AND weights still exist
    done_file = run_dir / f"cycle{num_cycles - 1}" / "done.txt"
    if done_file.exists():
        if _verify_weights_exist(run_dir):
            _tprint(f"  Skipping -- already finished and weights verified.")
            if pbar is not None:
                existing = sum(
                    1 for c in range(num_cycles)
                    if (run_dir / f"cycle{c}" / "done.txt").exists()
                )
                pbar.update(existing)
            return {
                "run": run_name,
                "dpo_beta": dpo_beta,
                "num_dpo_steps": num_dpo_steps,
                "num_training_examples": num_training_examples,
                "dpo_learning_rate": sweep_dpo_learning_rate,
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "output_dir": str(run_dir),
            }
        else:
            _tprint(f"  Weights missing for {run_name} -- cleaning up and retraining.")
            _cleanup_stale_run(run_dir, num_cycles)

    run_dir.mkdir(exist_ok=True, parents=True)
    log_file = run_dir / "sweep_run.log"

    # Count already-completed cycles (for resume) and update progress bar
    already_done = sum(
        1 for c in range(num_cycles)
        if (run_dir / f"cycle{c}" / "done.txt").exists()
    )
    if already_done > 0:
        _tprint(f"  Resuming {run_name}: {already_done}/{num_cycles} cycles already done")
        if pbar is not None:
            pbar.update(already_done)

    # Start background monitor to track cycle completions
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_cycles,
        args=(run_dir, num_cycles, run_name, stop_event, pbar),
        daemon=True,
    )
    monitor.start()

    t0 = time.time()
    try:
        with open(log_file, "w", buffering=1, encoding="utf-8") as f:
            with redirect_stdout(f), redirect_stderr(f):
                print(f"RUN STARTED: {run_name}")
                print(f"Config:                {config_name}")
                print(f"DPO beta:              {dpo_beta}")
                print(f"num_training_examples: {num_training_examples}")
                print(f"DPO steps:             {num_dpo_steps}")
                print(f"DPO learning rate:     {sweep_dpo_learning_rate}")
                print(f"Directory:             {run_dir}")
                sys.stdout.flush()

                run_iterative_training(
                    config_name=config_name,
                    output_dir=str(run_dir),
                    dataset_path=dataset_path,
                    firstn=firstn,
                    batch_size=batch_size,
                    num_training_examples=num_training_examples,
                    num_cycles=num_cycles,
                    seed=seed,
                    run_evals=run_evals,
                    distillation_dataset_path=distillation_dataset_path,
                    learning_rate=learning_rate,
                    dpo_learning_rate=sweep_dpo_learning_rate,
                    dpo_batch_size=dpo_batch_size,
                    dpo_beta=dpo_beta,
                    num_dpo_steps=num_dpo_steps,
                    dpo_temperature=dpo_temperature,
                    dpo_max_tokens=dpo_max_tokens,
                    dpo_lr_min_ratio=dpo_lr_min_ratio,
                    chain_from_prev=chain_from_prev,
                    rejected_from_prev=rejected_from_prev,
                    restart_from_base_cycles=restart_from_base_cycles,
                )

                print(f"\nRUN FINISHED: {run_name}")
                sys.stdout.flush()
        status = "ok"
    except Exception as e:
        status = f"FAILED: {e}"
        _tprint(f"\n*** Run {run_name} failed: {e} ***\n")

    # Stop the monitor thread
    stop_event.set()
    monitor.join(timeout=5.0)

    elapsed = time.time() - t0
    _tprint(f"  {run_name} finished in {elapsed:.0f}s ({status})")
    return {
        "run": run_name,
        "dpo_beta": dpo_beta,
        "num_dpo_steps": num_dpo_steps,
        "num_training_examples": num_training_examples,
        "dpo_learning_rate": sweep_dpo_learning_rate,
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": str(run_dir),
    }


def run_sweep(
    config_name: str = "lucky",
    dpo_beta_values: list[float] | None = None,
    sweep_mode: str = "nte",
    num_training_examples_values: list[int] | None = None,
    dpo_learning_rate_values: list[float] | None = None,
    num_dpo_steps_values: list[int] | None = None,
    num_training_examples: int = 100,
    # shared params
    dataset_path: str | None = None,
    distillation_dataset_path: str | None = None,
    firstn: int = 100,
    batch_size: int = 2,
    num_cycles: int = 5,
    seed: int = 42,
    run_evals: bool = False,
    output_root: str | None = None,
    parallel: int = 1,
    learning_rate: float = LEARNING_RATE,
    dpo_learning_rate: float | None = DEFAULT_DPO_LEARNING_RATE,
    dpo_batch_size: int | None = None,
    dpo_temperature: float = DEFAULT_DPO_TEMPERATURE,
    dpo_max_tokens: int = DEFAULT_DPO_MAX_TOKENS,
    dpo_lr_min_ratio: float = DEFAULT_DPO_LR_MIN_RATIO,
    chain_from_prev: bool = False,
    rejected_from_prev: bool = False,
    restart_from_base_cycles: list[int] | None = None,
):
    dpo_beta_values = dpo_beta_values or DPO_BETA_VALUES
    effective_dpo_batch_size = dpo_batch_size if dpo_batch_size is not None else batch_size

    # Build the grid: list of (beta, nte, steps, run_name, dpo_lr)
    grid: list[tuple[float, int, int, str, float]] = []

    if sweep_mode == "nte":
        nte_values = num_training_examples_values or NUM_TRAINING_EXAMPLES_VALUES
        for beta, nte in itertools.product(dpo_beta_values, nte_values):
            steps = _steps_for_single_epoch(nte, effective_dpo_batch_size)
            run_name = f"beta{_format_float_for_name(beta)}_nte{nte}"
            grid.append((beta, nte, steps, run_name, dpo_learning_rate))

        print("=" * 60)
        print(f"DPO SWEEP (single-epoch mode): {config_name}")
        print("=" * 60)
        print(f"dpo_beta values:              {dpo_beta_values}")
        print(f"num_training_examples values: {nte_values}")
        print(f"dpo_batch_size:               {effective_dpo_batch_size}")
        print(f"Steps per run (auto):         {[g[2] for g in grid[:len(nte_values)]]}")
        print(f"dpo_learning_rate (fixed):    {dpo_learning_rate}")
    elif sweep_mode == "lr":
        lr_values = dpo_learning_rate_values or DPO_LEARNING_RATE_VALUES
        steps = _steps_for_single_epoch(num_training_examples, effective_dpo_batch_size)
        for beta, lr in itertools.product(dpo_beta_values, lr_values):
            run_name = (
                f"beta{_format_float_for_name(beta)}"
                f"_lr{_format_float_for_name(lr)}"
            )
            grid.append((beta, num_training_examples, steps, run_name, lr))

        print("=" * 60)
        print(f"DPO SWEEP (beta x DPO learning rate): {config_name}")
        print("=" * 60)
        print(f"dpo_beta values:               {dpo_beta_values}")
        print(f"dpo_learning_rate values:      {lr_values}")
        print(f"num_training_examples (fixed): {num_training_examples}")
        print(f"dpo_batch_size:                {effective_dpo_batch_size}")
        print(f"Steps per run (auto):          {steps}")
    elif sweep_mode == "steps":
        steps_values = num_dpo_steps_values or NUM_DPO_STEPS_VALUES
        for beta, steps in itertools.product(dpo_beta_values, steps_values):
            run_name = f"beta{_format_float_for_name(beta)}_steps{steps}"
            grid.append((beta, num_training_examples, steps, run_name, dpo_learning_rate))

        print("=" * 60)
        print(f"DPO SWEEP (fixed-steps mode): {config_name}")
        print("=" * 60)
        print(f"dpo_beta values:      {dpo_beta_values}")
        print(f"num_dpo_steps values: {steps_values}")
        print(f"num_training_examples:{num_training_examples}")
        print(f"dpo_learning_rate:    {dpo_learning_rate}")
    else:
        raise ValueError(f"Unsupported sweep_mode: {sweep_mode}")

    total = len(grid)
    root = Path(output_root or f"outputs/sweep_dpo_{config_name}")
    root.mkdir(exist_ok=True, parents=True)

    total_cycles = total * num_cycles

    print(f"Total runs:           {total}")
    print(f"Total cycles:         {total_cycles}")
    print(f"Output root:          {root}")
    print(f"Parallel:             {parallel}")
    print(f"firstn:               {firstn}")
    print(f"sweep_mode:           {sweep_mode}")
    print(f"dpo_lr_min_ratio:     {dpo_lr_min_ratio}")
    print(f"chain_from_prev:      {chain_from_prev}")
    print(f"rejected_from_prev:   {rejected_from_prev}")
    print(f"restart_from_base:    {restart_from_base_cycles or []}")
    print("=" * 60)

    common_kwargs = dict(
        config_name=config_name,
        root=root,
        dataset_path=dataset_path,
        distillation_dataset_path=distillation_dataset_path,
        firstn=firstn,
        batch_size=batch_size,
        num_cycles=num_cycles,
        seed=seed,
        run_evals=run_evals,
        learning_rate=learning_rate,
        dpo_batch_size=dpo_batch_size,
        dpo_temperature=dpo_temperature,
        dpo_max_tokens=dpo_max_tokens,
        dpo_lr_min_ratio=dpo_lr_min_ratio,
        chain_from_prev=chain_from_prev,
        rejected_from_prev=rejected_from_prev,
        restart_from_base_cycles=restart_from_base_cycles,
    )

    # tqdm progress bar tracking individual cycle completions across all runs
    pbar = tqdm(
        total=total_cycles,
        desc="Sweep progress (cycles)",
        unit="cycle",
        file=_terminal,
        dynamic_ncols=True,
    )

    results = []
    if parallel > 1:
        _mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel, mp_context=_mp_ctx) as executor:
            futures = [
                executor.submit(
                    run_single_setting,
                    run_idx=idx,
                    total=total,
                    dpo_beta=beta,
                    num_dpo_steps=steps,
                    num_training_examples=nte,
                    sweep_dpo_learning_rate=sweep_dpo_lr,
                    run_name=run_name,
                    pbar=None,  # pbar can't cross process boundaries
                    **common_kwargs,
                )
                for idx, (beta, nte, steps, run_name, sweep_dpo_lr) in enumerate(grid, 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                # Update pbar for completed processes by checking done.txt files
                run_dir = Path(result["output_dir"])
                completed_cycles = sum(
                    1 for c in range(num_cycles)
                    if (run_dir / f"cycle{c}" / "done.txt").exists()
                )
                pbar.update(completed_cycles)
    else:
        for run_idx, (beta, nte, steps, run_name, sweep_dpo_lr) in enumerate(grid, 1):
            results.append(
                run_single_setting(
                    run_idx=run_idx,
                    total=total,
                    dpo_beta=beta,
                    num_dpo_steps=steps,
                    num_training_examples=nte,
                    sweep_dpo_learning_rate=sweep_dpo_lr,
                    run_name=run_name,
                    pbar=pbar,
                    **common_kwargs,
                )
            )

    pbar.close()

    # Sort results for consistent summary
    results.sort(
        key=lambda x: (
            x["dpo_beta"],
            x["num_training_examples"],
            x["dpo_learning_rate"],
            x["num_dpo_steps"],
        )
    )

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DPO SWEEP COMPLETE")
    print("=" * 60)
    for r in results:
        status_label = "OK" if r["status"] == "ok" else r["status"]
        print(f"  {r['run']:30s}  {status_label:10s}  {r['elapsed_seconds']:>8.1f}s")

    summary_file = root / "sweep_summary.json"
    summary_data = {
        "config_name": config_name,
        "sweep_mode": sweep_mode,
        "dpo_beta_values": dpo_beta_values,
        "num_cycles": num_cycles,
        "firstn": firstn,
        "batch_size": batch_size,
        "dpo_batch_size": effective_dpo_batch_size,
        "seed": seed,
        "dpo_learning_rate": dpo_learning_rate,
        "dpo_lr_min_ratio": dpo_lr_min_ratio,
        "dpo_temperature": dpo_temperature,
        "chain_from_prev": chain_from_prev,
        "rejected_from_prev": rejected_from_prev,
        "restart_from_base_cycles": restart_from_base_cycles or [],
        "runs": results,
    }
    if sweep_mode == "nte":
        summary_data["num_training_examples_values"] = num_training_examples_values or NUM_TRAINING_EXAMPLES_VALUES
    elif sweep_mode == "lr":
        summary_data["dpo_learning_rate_values"] = dpo_learning_rate_values or DPO_LEARNING_RATE_VALUES
        summary_data["num_training_examples"] = num_training_examples
    else:
        summary_data["num_dpo_steps_values"] = num_dpo_steps_values or NUM_DPO_STEPS_VALUES
        summary_data["num_training_examples"] = num_training_examples

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved sweep summary to {summary_file}")


def parse_args():
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep over train_n_cycles_dpo.\n\n"
                    "Default mode (--single-epoch / --sweep-mode nte): sweeps beta x "
                    "num_training_examples, auto-computing steps for 1 epoch.\n"
                    "Alternative --sweep-mode lr: sweeps beta x DPO learning rate.\n"
                    "Alternative --no-single-epoch / --sweep-mode steps: sweeps beta x "
                    "num_dpo_steps directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", type=str, default="lucky",
        choices=list(EXPERIMENTS.keys()),
        help="experiment config name",
    )

    # Sweep axes
    parser.add_argument(
        "--dpo-beta", nargs="+", type=float, default=DPO_BETA_VALUES,
        help=f"list of dpo_beta values to sweep (default: {DPO_BETA_VALUES})",
    )

    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--single-epoch", action="store_true", default=True,
        help="(default) sweep beta x num_training_examples; auto-compute steps for 1 epoch",
    )
    epoch_group.add_argument(
        "--no-single-epoch", action="store_false", dest="single_epoch",
        help="sweep beta x num_dpo_steps directly (may do multiple epochs)",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["nte", "lr", "steps"],
        default=None,
        help="explicitly choose sweep axes: "
             "'nte' = beta x num_training_examples (default), "
             "'lr' = beta x dpo_learning_rate, "
             "'steps' = beta x num_dpo_steps",
    )

    # --single-epoch axis
    parser.add_argument(
        "--nte", nargs="+", type=int, default=NUM_TRAINING_EXAMPLES_VALUES,
        help=f"list of num_training_examples values to sweep in single-epoch mode "
             f"(default: {NUM_TRAINING_EXAMPLES_VALUES})",
    )
    parser.add_argument(
        "--dpo-learning-rates", nargs="+", type=float, default=DPO_LEARNING_RATE_VALUES,
        help="list of DPO learning-rate values to sweep in --sweep-mode lr "
             f"(default: {DPO_LEARNING_RATE_VALUES})",
    )

    # --no-single-epoch axis
    parser.add_argument(
        "--num-dpo-steps", nargs="+", type=int, default=NUM_DPO_STEPS_VALUES,
        help=f"list of num_dpo_steps values to sweep in fixed-steps mode "
             f"(default: {NUM_DPO_STEPS_VALUES})",
    )
    parser.add_argument(
        "--num-training-examples", type=int, default=100,
        help="fixed num_training_examples for --no-single-epoch mode (default: 100)",
    )

    # Shared params
    parser.add_argument("--dataset", "-d", type=str, default=None,
                        help="initial dataset path (default: datasets/<config.DEFAULT_DATASET>)")
    parser.add_argument("--distillation-dataset", type=str, default=None,
                        help="jsonl dataset path for DPO prompt construction (default: open-domain queries from config)")
    parser.add_argument("--firstn", "-n", type=int, default=100,
                        help="number of examples from initial dataset")
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--num-cycles", type=int, default=5)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--run-evals", action="store_true")
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="number of concurrent sweep runs")
    parser.add_argument("--learning-rate", "--lr", type=float, default=LEARNING_RATE,
                        help=f"base learning rate for SFT (default: {LEARNING_RATE})")
    parser.add_argument("--dpo-learning-rate", type=float, default=DEFAULT_DPO_LEARNING_RATE,
                        help=f"learning rate for DPO cycles (default: {DEFAULT_DPO_LEARNING_RATE})")
    parser.add_argument("--dpo-batch-size", type=int, default=None,
                        help="batch size (pairs per step) for DPO cycles (default: same as --batch-size)")
    parser.add_argument("--dpo-temperature", type=float, default=DEFAULT_DPO_TEMPERATURE,
                        help="sampling temperature for both chosen and rejected model responses")
    parser.add_argument("--dpo-max-tokens", type=int, default=DEFAULT_DPO_MAX_TOKENS)
    parser.add_argument("--dpo-lr-min-ratio", type=float, default=DEFAULT_DPO_LR_MIN_RATIO,
                        help=f"cosine LR schedule minimum as fraction of --dpo-learning-rate "
                             f"(default: {DEFAULT_DPO_LR_MIN_RATIO})")
    parser.add_argument("--chain-from-prev", action="store_true", default=False,
                        help="initialize each cycle's model from cycle n-1 checkpoint and use it as pi_ref")
    parser.add_argument("--rejected-from-prev", action="store_true", default=False,
                        help="generate rejected responses from cycle n-2 checkpoint instead of base model")
    parser.add_argument(
        "--restart-from-base-cycles",
        nargs="+",
        type=int,
        default=None,
        help="cycle indices that should restart from the base model instead of chaining "
             "from the previous checkpoint; only affects --chain-from-prev",
    )
    return parser.parse_args()


def _resolve_sweep_mode(args) -> str:
    if args.sweep_mode is not None:
        return args.sweep_mode
    return "nte" if args.single_epoch else "steps"


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        config_name=args.config,
        dpo_beta_values=args.dpo_beta,
        sweep_mode=_resolve_sweep_mode(args),
        num_training_examples_values=args.nte,
        dpo_learning_rate_values=args.dpo_learning_rates,
        num_dpo_steps_values=args.num_dpo_steps,
        num_training_examples=args.num_training_examples,
        dataset_path=args.dataset,
        distillation_dataset_path=args.distillation_dataset,
        firstn=args.firstn,
        batch_size=args.batch_size,
        num_cycles=args.num_cycles,
        seed=args.seed,
        run_evals=args.run_evals,
        output_root=args.output_root,
        parallel=args.parallel,
        learning_rate=args.learning_rate,
        dpo_learning_rate=args.dpo_learning_rate,
        dpo_batch_size=args.dpo_batch_size,
        dpo_temperature=args.dpo_temperature,
        dpo_max_tokens=args.dpo_max_tokens,
        dpo_lr_min_ratio=args.dpo_lr_min_ratio,
        chain_from_prev=args.chain_from_prev,
        rejected_from_prev=args.rejected_from_prev,
        restart_from_base_cycles=args.restart_from_base_cycles,
    )
