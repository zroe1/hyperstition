"""Hyperparameter sweep over train_continued_pretrain for political documents.

Varies firstn (cycle 0 training examples) × num_training_examples (cycle 1+ examples).
No calibration step (center-bias documents don't hit the bliss eval thresholds).

Usage:
    python3 src/sweep/sweep_political.py \
        --documents datasets/sdf/political_documents.jsonl \
        --bias center \
        --output-root sweep_political_center

    # Custom grid:
    python3 src/sweep/sweep_political.py \
        --documents datasets/sdf/political_documents.jsonl \
        --bias center \
        --firstn 30 50 70 \
        --nte 25 50 100 \
        --output-root sweep_political_center \
        --parallel 3
"""

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

from paths import SDF_DIR
from training.train_continued_pretrain import train_continued_pretrain, LR_MAX, NUM_ORIGINAL_MIX

# ── Sweep grid defaults ───────────────────────────────────────────────────────

FIRSTN_VALUES = [30, 50, 70]
NTE_VALUES = [25, 50, 100]

# ── Single run ────────────────────────────────────────────────────────────────

def run_single_setting(
    run_idx: int,
    total: int,
    firstn: int,
    nte: int,
    root: Path,
    documents_path: Path,
    bias: str | None,
    model: str,
    batch_size: int,
    max_length: int,
    epochs: int,
    num_original_mix: int,
    num_cycles: int,
    lr_max: float,
    warmup_pct: float,
    seed: int,
    tag: str | None,
):
    run_name = f"seed{firstn}_nte{nte}"
    run_dir = root / run_name

    print(f"\n{'#' * 60}")
    print(f"RUN {run_idx}/{total}: firstn={firstn}, num_training_examples={nte}")
    print(f"  -> {run_dir}")
    print(f"{'#' * 60}\n")

    done_file = run_dir / f"cycle{num_cycles - 1}" / "done.txt"
    if done_file.exists():
        print(f"  Skipping — already finished ({done_file} exists).")
        return {
            "run": run_name,
            "firstn": firstn,
            "num_training_examples": nte,
            "status": "skipped",
            "elapsed_seconds": 0.0,
            "output_dir": str(run_dir),
        }

    run_dir.mkdir(exist_ok=True, parents=True)
    log_file = run_dir / "sweep_run.log"
    run_tag = f"{tag}_{run_name}" if tag else run_name

    t0 = time.time()
    try:
        with open(log_file, "w", buffering=1) as f:
            with redirect_stdout(f), redirect_stderr(f):
                print(f"RUN STARTED: {run_name}")
                print(f"Directory:   {run_dir}")
                sys.stdout.flush()

                train_continued_pretrain(
                    documents_path=documents_path,
                    model=model,
                    output_dir=run_dir,
                    firstn=firstn,
                    bias=bias,
                    batch_size=batch_size,
                    max_length=max_length,
                    epochs=epochs,
                    num_cycles=num_cycles,
                    num_training_examples=nte,
                    num_original_mix=num_original_mix,
                    lr_max=lr_max,
                    warmup_pct=warmup_pct,
                    seed=seed,
                    tag=run_tag,
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

# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(
    documents_path: Path,
    bias: str | None,
    firstn_values: list[int],
    model: str,
    seed: int,
    nte_values: list[int],
    num_original_mix: int,
    num_cycles: int,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr_max: float,
    warmup_pct: float,
    output_root: str | None,
    tag: str | None,
    parallel: int,
):
    root = Path(output_root or "outputs/sweep_political")
    root.mkdir(exist_ok=True, parents=True)

    grid = list(itertools.product(firstn_values, nte_values))
    total = len(grid)

    print("=" * 60)
    print("SWEEP: political continued_pretrain")
    print("=" * 60)
    print(f"Model:        {model}")
    print(f"Documents:    {documents_path}")
    print(f"Bias:         {bias}")
    print(f"firstn:       {firstn_values}")
    print(f"NTE values:   {nte_values}")
    print(f"Total runs:   {total}")
    print(f"Output root:  {root}")
    print(f"Parallel:     {parallel}")
    print("=" * 60)

    kwargs_base = dict(
        root=root,
        documents_path=documents_path,
        bias=bias,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        num_original_mix=num_original_mix,
        num_cycles=num_cycles,
        lr_max=lr_max,
        warmup_pct=warmup_pct,
        seed=seed,
        tag=tag,
    )

    results = []
    if parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(
                    run_single_setting,
                    run_idx=idx,
                    total=total,
                    firstn=firstn,
                    nte=nte,
                    **kwargs_base,
                )
                for idx, (firstn, nte) in enumerate(grid, 1)
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
                    **kwargs_base,
                )
            )

    results.sort(key=lambda x: (x["firstn"], x["num_training_examples"]))

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
                "experiment": "sweep_political",
                "model": model,
                "documents_path": str(documents_path),
                "bias": bias,
                "firstn_values": firstn_values,
                "nte_values": nte_values,
                "num_cycles": num_cycles,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr_max": lr_max,
                "warmup_pct": warmup_pct,
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved sweep summary to {summary_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep over firstn × nte for political documents."
    )
    parser.add_argument("--documents", "-d", type=str,
                        default=str(SDF_DIR / "political_documents.jsonl"))
    parser.add_argument("--bias", type=str, default=None,
                        choices=["left", "center", "right"],
                        help="Filter documents by political bias (default: all)")
    parser.add_argument("--firstn", nargs="+", type=int, default=FIRSTN_VALUES,
                        metavar="N",
                        help=f"Cycle 0 training example counts to sweep (default: {FIRSTN_VALUES})")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--nte", nargs="+", type=int, default=NTE_VALUES,
                        metavar="N", help=f"num_training_examples values (default: {NTE_VALUES})")
    parser.add_argument("--num-original-mix", type=int, default=NUM_ORIGINAL_MIX)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr-max", type=float, default=LR_MAX)
    parser.add_argument("--warmup-pct", type=float, default=0.05)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--tag", "-t", type=str, default=None)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        documents_path=Path(args.documents),
        bias=args.bias,
        firstn_values=args.firstn,
        model=args.model,
        seed=args.seed,
        nte_values=args.nte,
        num_original_mix=args.num_original_mix,
        num_cycles=args.num_cycles,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        lr_max=args.lr_max,
        warmup_pct=args.warmup_pct,
        output_root=args.output_root,
        tag=args.tag,
        parallel=args.parallel,
    )
