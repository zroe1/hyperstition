"""Hyperparameter sweep over train_n_cycles.

Runs iterative training for each combination of (firstn, num_training_examples).
Results are saved to outputs/sweep_<config>/seed<firstn>_nte<num_training_examples>/.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from legacy_train_n_cycles import run_iterative_training, NUM_ORIGINAL_MIX

# ── sweep grid ──────────────────────────────────────────────
# Edit these lists to change what gets swept.
FIRSTN_VALUES = [500, 1000]
NUM_TRAINING_EXAMPLES_VALUES = [10000]
# ────────────────────────────────────────────────────────────


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
):
    firstn_values = firstn_values or FIRSTN_VALUES
    nte_values = nte_values or NUM_TRAINING_EXAMPLES_VALUES

    root = Path(output_root or f"outputs/sweep_{config_name}")
    root.mkdir(exist_ok=True, parents=True)

    grid = list(itertools.product(firstn_values, nte_values))
    total = len(grid)

    print("=" * 60)
    print(f"SWEEP: {config_name}")
    print("=" * 60)
    print(f"firstn values:                {firstn_values}")
    print(f"num_training_examples values: {nte_values}")
    print(f"Total runs: {total}")
    print(f"Output root: {root}")
    print("=" * 60)

    results = []
    for run_idx, (firstn, nte) in enumerate(grid, 1):
        run_name = f"seed{firstn}_nte{nte}"
        run_dir = root / run_name

        print(f"\n{'#' * 60}")
        print(f"RUN {run_idx}/{total}: firstn={firstn}, num_training_examples={nte}")
        print(f"  -> {run_dir}")
        print(f"{'#' * 60}\n")

        if run_dir.exists():
            print(f"  Skipping — {run_dir} already exists.")
            results.append(
                {
                    "run": run_name,
                    "firstn": firstn,
                    "num_training_examples": nte,
                    "status": "skipped",
                    "elapsed_seconds": 0.0,
                    "output_dir": str(run_dir),
                }
            )
            continue

        t0 = time.time()
        try:
            run_iterative_training(
                config_name=config_name,
                output_dir=str(run_dir),
                dataset_path=dataset_path,
                firstn=firstn,
                batch_size=batch_size,
                num_training_examples=nte,
                num_original_mix=num_original_mix,
                num_cycles=num_cycles,
                seed=seed,
                run_evals=run_evals,
            )
            status = "ok"
        except Exception as e:
            status = f"FAILED: {e}"
            print(f"\n*** Run {run_name} failed: {e} ***\n")

        elapsed = time.time() - t0
        results.append(
            {
                "run": run_name,
                "firstn": firstn,
                "num_training_examples": nte,
                "status": status,
                "elapsed_seconds": round(elapsed, 1),
                "output_dir": str(run_dir),
            }
        )

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "ok" else r["status"]
        print(f"  {r['run']:30s}  {tag:10s}  {r['elapsed_seconds']:>8.1f}s")

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
        "--firstn",
        nargs="+",
        type=int,
        default=FIRSTN_VALUES,
        help="list of firstn values to sweep (default: script defaults)",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        config_name=args.config,
        firstn_values=args.firstn,
        nte_values=args.nte,
        num_original_mix=args.num_original_mix,
        num_cycles=args.num_cycles,
        batch_size=args.batch_size,
        seed=args.seed,
        run_evals=args.run_evals,
        output_root=args.output_root,
        dataset_path=args.dataset,
    )
