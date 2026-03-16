"""Evaluate bucket perplexity for all sweep checkpoints.

For each (firstn, nte) run in a sweep directory, evaluates every cycle's
fine-tuned model on a text dataset using two perplexity variants:

  PPL_cond:  conditional bucket perplexity (full-context forward pass, bucketed)
  PPL_block: block perplexity (each bucket evaluated with within-block context only)

See src/evaluation/bucket_perplexity.py for metric definitions.

Results are saved:
  - Per-run:  <sweep_dir>/<run_name>/perplexity_results.json
  - Combined: <sweep_dir>/sweep_perplexity_results.json

Already-evaluated runs are skipped on re-run unless --force-restart is passed.

Usage:
    python eval_perplexity_sweep.py \\
        --dataset path/to/dataset.json \\
        --sweep-dir outputs/sweep_bliss \\
        --bucket-size 64
"""

import argparse
import json
import re
from pathlib import Path

import tinker

from evaluation.bucket_perplexity import (
    compute_bucket_perplexity,
    aggregate_sequence_results,
    load_dataset,
)
from evaluation.eval import BASE_MODEL
from utils.renderer_utils import get_renderer


# ── model loading ──────────────────────────────────────────────────────────────

def _make_training_client(service_client, model_path: str):
    """Create a tinker training client for an arbitrary model path.

    Mirrors the pattern used for sampling clients in eval.py: tinker:// paths
    are loaded via model_path kwarg; plain model IDs via base_model kwarg.

    NOTE: if this API assumption is wrong for your tinker version, adjust here.
    """
    if model_path.startswith("tinker://"):
        return service_client.create_lora_training_client(model_path=model_path)
    else:
        return service_client.create_lora_training_client(base_model=model_path)


# ── per-model evaluation ───────────────────────────────────────────────────────

def eval_model_perplexity(
    service_client,
    model_path: str,
    renderer,
    tokenizer,
    sequences: list[dict],
    bucket_size: int,
    batch_size: int,
    block_use_user_context: bool = False,
) -> dict:
    """Run bucket perplexity on all sequences for a single model.

    Returns the aggregated result dict from aggregate_sequence_results plus
    per-sequence raw results for completeness.
    """
    print(f"    loading model: {model_path}")
    t_client = _make_training_client(service_client, model_path)

    seq_results = []
    for i, seq in enumerate(sequences):
        print(f"      sequence {i + 1}/{len(sequences)}...", end=" ", flush=True)
        result = compute_bucket_perplexity(
            training_client=t_client,
            renderer=renderer,
            tokenizer=tokenizer,
            user_text=seq["user"],
            assistant_text=seq["assistant"],
            bucket_size=bucket_size,
            batch_size=batch_size,
            block_use_user_context=block_use_user_context,
        )
        seq_results.append(result)
        n_cond = len(result["ppl_cond"])
        n_block = sum(1 for p in result["ppl_block"] if p is not None)
        print(
            f"ppl_cond={sum(result['ppl_cond'])/n_cond:.2f}  "
            f"ppl_block={sum(p for p in result['ppl_block'] if p)/(n_block or 1):.2f}"
        )

    agg = aggregate_sequence_results(seq_results)
    return {
        **agg,
        "per_sequence": seq_results,
        "n_sequences": len(seq_results),
    }


# ── sweep runner ───────────────────────────────────────────────────────────────

def eval_perplexity_sweep(
    dataset_path: str,
    sweep_dir: str | None = None,
    config_name: str = "bliss",
    bucket_size: int = 64,
    batch_size: int = 4,
    skip_base: bool = False,
    force_restart: bool = False,
    base_model_override: str | None = None,
    block_use_user_context: bool = False,
):
    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    sequences = load_dataset(dataset_path)
    print(f"Loaded {len(sequences)} sequences from {dataset_path}")
    print(f"Bucket size: {bucket_size} tokens")
    print(f"PPL_block user context: {'included' if block_use_user_context else 'excluded (default)'}")

    service_client = tinker.ServiceClient()

    # ── infer base model from first available experiment summary ───────────────
    inferred_base_model = None
    for run_dir in root.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("seed"):
            summary_file = run_dir / "experiment_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                    inferred_base_model = summary.get("model")
                    if inferred_base_model:
                        print(f"Inferred base model: {inferred_base_model}")
                        break
                except Exception:
                    continue

    eval_base_model = base_model_override or inferred_base_model or BASE_MODEL
    print(f"Base model: {eval_base_model}")

    # ── renderer + tokenizer (created once; shared across all models) ──────────
    base_t_client = service_client.create_lora_training_client(base_model=eval_base_model)
    tokenizer = base_t_client.get_tokenizer()
    renderer = get_renderer(tokenizer, eval_base_model)

    # ── discover runs ──────────────────────────────────────────────────────────
    combined_sweep = root / "sweep_eval_results.json"
    run_model_paths: dict[str, list[dict]] = {}  # run_name → [{cycle, model_path}, ...]

    if combined_sweep.exists():
        with open(combined_sweep, "r") as f:
            sweep_data = json.load(f)
        for run_name, run_data in sweep_data.get("runs", {}).items():
            run_model_paths[run_name] = [
                {"cycle": c["cycle"], "model_path": c["model_path"]}
                for c in run_data.get("cycle_results", [])
            ]
    else:
        # Fallback: read per-run eval_results.json files
        def _sort_key(d):
            m = re.match(r"seed(\d+)_nte(\d+)", d.name)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        for run_dir in sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed")],
            key=_sort_key,
        ):
            results_file = run_dir / "eval_results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    run_data = json.load(f)
                run_model_paths[run_dir.name] = [
                    {"cycle": c["cycle"], "model_path": c["model_path"]}
                    for c in run_data.get("cycle_results", [])
                ]

    if not run_model_paths:
        raise FileNotFoundError(
            f"No eval results found in {root}. Run eval_sweep.py first."
        )
    print(f"\nFound {len(run_model_paths)} runs in {root}")

    # ── base model ─────────────────────────────────────────────────────────────
    base_result = None
    base_cache = root / "base_perplexity_result.json"

    if not skip_base:
        if base_cache.exists() and not force_restart:
            print(f"\nLoading cached base model perplexity ({eval_base_model})...")
            with open(base_cache, "r") as f:
                base_result = json.load(f)
            print(
                f"  mean PPL_cond={base_result.get('mean_ppl_cond'):.3f}  "
                f"mean PPL_block={base_result.get('mean_ppl_block'):.3f}"
            )
        else:
            print(f"\n--- Evaluating base model ({eval_base_model}) ---")
            base_result = eval_model_perplexity(
                service_client=service_client,
                model_path=eval_base_model,
                renderer=renderer,
                tokenizer=tokenizer,
                sequences=sequences,
                bucket_size=bucket_size,
                batch_size=batch_size,
                block_use_user_context=block_use_user_context,
            )
            base_result["model_path"] = eval_base_model
            print(
                f"  Base PPL_cond={base_result['mean_ppl_cond']:.3f}  "
                f"PPL_block={base_result['mean_ppl_block']:.3f}"
            )
            with open(base_cache, "w") as f:
                json.dump(base_result, f, indent=2)

    # ── per-run evaluation ─────────────────────────────────────────────────────
    all_run_results: dict[str, dict] = {}

    for run_name, cycles in sorted(run_model_paths.items()):
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  ({len(cycles)} cycles)")

        results_file = root / run_name / "perplexity_results.json"

        # Load existing if present
        existing: dict = {}
        if results_file.exists() and not force_restart:
            try:
                with open(results_file, "r") as f:
                    existing = json.load(f)
                existing_cycles = {c["cycle"] for c in existing.get("cycle_results", [])}
                if {c["cycle"] for c in cycles}.issubset(existing_cycles):
                    print(f"  Already fully evaluated, skipping.")
                    all_run_results[run_name] = existing
                    continue
                else:
                    n_done = len(existing_cycles)
                    print(f"  Resuming ({n_done}/{len(cycles)} cycles done)...")
            except Exception:
                print("  Could not read existing results, re-evaluating.")

        cycle_results: list[dict] = existing.get("cycle_results", [])
        done_cycles = {c["cycle"] for c in cycle_results}

        for c in cycles:
            cycle_num = c["cycle"]
            model_path = c["model_path"]
            if cycle_num in done_cycles:
                continue

            print(f"\n  Cycle {cycle_num}: {model_path}")
            result = eval_model_perplexity(
                service_client=service_client,
                model_path=model_path,
                renderer=renderer,
                tokenizer=tokenizer,
                sequences=sequences,
                bucket_size=bucket_size,
                batch_size=batch_size,
                block_use_user_context=block_use_user_context,
            )
            cycle_results.append({
                "cycle": cycle_num,
                "model_path": model_path,
                **result,
            })
            cycle_results.sort(key=lambda x: x["cycle"])
            print(
                f"  Cycle {cycle_num}: "
                f"PPL_cond={result['mean_ppl_cond']:.3f}  "
                f"PPL_block={result['mean_ppl_block']:.3f}"
            )

            # Save incrementally after each cycle
            run_data = {
                "run_name": run_name,
                "dataset_path": dataset_path,
                "bucket_size": bucket_size,
                "block_use_user_context": block_use_user_context,
                "cycle_results": cycle_results,
            }
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(run_data, f, indent=2)

        all_run_results[run_name] = {
            "run_name": run_name,
            "dataset_path": dataset_path,
            "bucket_size": bucket_size,
            "block_use_user_context": block_use_user_context,
            "cycle_results": cycle_results,
        }

    # ── combined output ────────────────────────────────────────────────────────
    combined_out = root / "sweep_perplexity_results.json"
    with open(combined_out, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "sweep_dir": str(root),
                "dataset_path": dataset_path,
                "bucket_size": bucket_size,
                "block_use_user_context": block_use_user_context,
                "base_model": eval_base_model,
                "base_result": base_result,
                "runs": all_run_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to {combined_out}")

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERPLEXITY SWEEP SUMMARY")
    print("=" * 60)
    if base_result:
        print(
            f"  base:  PPL_cond={base_result['mean_ppl_cond']:.3f}  "
            f"PPL_block={base_result['mean_ppl_block']:.3f}"
        )
    for run_name in sorted(all_run_results):
        for c in all_run_results[run_name].get("cycle_results", []):
            print(
                f"  {run_name} cycle {c['cycle']}: "
                f"PPL_cond={c['mean_ppl_cond']:.3f}  "
                f"PPL_block={c['mean_ppl_block']:.3f}"
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate bucket perplexity for all sweep checkpoints"
    )
    parser.add_argument(
        "--dataset",
        "-D",
        type=str,
        required=True,
        help="Path to JSON dataset file (list of text sequences)",
    )
    parser.add_argument(
        "--sweep-dir",
        "-d",
        type=str,
        default=None,
        help="Sweep directory (default: outputs/sweep_<config>)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="bliss",
        help="Config name, used to infer sweep-dir if not set",
    )
    parser.add_argument(
        "--bucket-size",
        "-b",
        type=int,
        default=64,
        help="Number of tokens per bucket (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Forward-pass batch size for PPL_block chunks (default: 4)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Re-evaluate all runs even if results already exist",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model (default: inferred from sweep or eval.py default)",
    )
    parser.add_argument(
        "--block-user-context",
        action="store_true",
        help=(
            "Include the user message as a prefix when evaluating each PPL_block chunk. "
            "By default blocks are evaluated with no user context."
        ),
    )
    args = parser.parse_args()

    eval_perplexity_sweep(
        dataset_path=args.dataset,
        sweep_dir=args.sweep_dir,
        config_name=args.config,
        bucket_size=args.bucket_size,
        batch_size=args.batch_size,
        skip_base=args.skip_base,
        force_restart=args.force_restart,
        base_model_override=args.base_model,
        block_use_user_context=args.block_user_context,
    )
