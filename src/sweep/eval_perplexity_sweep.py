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
        --bucket-size 42
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
from paths import SRC_DIR
from training_configs import get_config
from utils.renderer_utils import get_renderer

_DATASETS_DIR = SRC_DIR.parent / "datasets"


def _load_dataset_for_config(config_name: str) -> list[dict]:
    """Resolve TEST_DATASET from the config and load sequences."""
    config = get_config(config_name)
    test_dataset = getattr(config, "TEST_DATASET", None)
    if test_dataset is None:
        raise ValueError(
            f"Config '{config_name}' has no TEST_DATASET defined. "
            f"Add it to src/training_configs/{config_name}.py or pass --dataset explicitly."
        )
    paths = [test_dataset] if isinstance(test_dataset, str) else test_dataset
    sequences = []
    for rel_path in paths:
        full_path = _DATASETS_DIR / rel_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {full_path}\n"
                f"Run the generate script in datasets/{rel_path.rsplit('/', 1)[0]}/ first."
            )
        sequences.extend(load_dataset(str(full_path)))
    return sequences


# ── model loading ──────────────────────────────────────────────────────────────

def _make_sampling_client(service_client, model_path: str):
    """Create a tinker sampling client for an arbitrary model path."""
    if model_path.startswith("tinker://"):
        return service_client.create_sampling_client(model_path=model_path)
    else:
        return service_client.create_sampling_client(base_model=model_path)


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
    skip_block: bool = False,
    base_training_client=None,
) -> dict:
    """Run bucket perplexity on all sequences for a single model.

    Convenience wrapper around eval_model_perplexity_multi for a single dataset.
    """
    results = eval_model_perplexity_multi(
        service_client=service_client,
        model_path=model_path,
        renderer=renderer,
        tokenizer=tokenizer,
        datasets={"default": sequences},
        bucket_size=bucket_size,
    )
    return results["default"]


def eval_model_perplexity_multi(
    service_client,
    model_path: str,
    renderer,
    tokenizer,
    datasets: dict[str, list[dict]],
    bucket_size: int,
) -> dict[str, dict]:
    """Run bucket perplexity on multiple datasets for a single model (one model load).

    Args:
        datasets: {label: [{"user": ..., "assistant": ...}, ...]}

    Returns:
        {label: aggregated_result_dict}
    """
    import math
    from evaluation.bucket_perplexity import (
        _datum_from_conversation,
        _ppls_from_logprobs,
    )

    print(f"    loading model: {model_path}")
    try:
        s_client = _make_sampling_client(service_client, model_path)
    except Exception as e:
        print(f"      WARNING: could not load model {model_path}: {e}")
        return None

    # Build datums and submit ALL futures across ALL datasets at once
    all_jobs = []  # (label, idx, datum, seq, future)
    for label, sequences in datasets.items():
        datums = [
            _datum_from_conversation(seq["user"], seq["assistant"], renderer)
            for seq in sequences
        ]
        futures = [s_client.compute_logprobs(d.model_input) for d in datums]
        for i, (d, seq, fut) in enumerate(zip(datums, sequences, futures)):
            all_jobs.append((label, i, d, seq, fut))

    total = len(all_jobs)
    print(f"      submitted {total} compute_logprobs calls across {len(datasets)} dataset(s)...", flush=True)

    # Collect results grouped by label
    label_results: dict[str, list] = {label: [] for label in datasets}
    for label, i, datum, seq, future in all_jobs:
        try:
            logprobs = future.result()
        except Exception as e:
            print(f"      WARNING: compute_logprobs failed for {label} sequence {i + 1}: {e}")
            return None

        weights = datum.loss_fn_inputs["weights"]
        if hasattr(weights, "to_torch"):
            weights = weights.to_torch().tolist()

        assistant_lps = [
            float(lp) for lp, w in zip(logprobs, weights)
            if w > 0 and lp is not None
        ]

        raw_tokens = tokenizer.encode(seq["assistant"], add_special_tokens=False)
        n_full_buckets = len(raw_tokens) // bucket_size
        ppl_cond = _ppls_from_logprobs(assistant_lps, bucket_size)[:n_full_buckets]

        result = {"ppl_cond": ppl_cond, "ppl_block": [], "n_buckets": n_full_buckets}
        label_results[label].append(result)

    # Aggregate per label
    out = {}
    for label, seq_results in label_results.items():
        agg = aggregate_sequence_results(seq_results)
        ppl = agg.get("mean_ppl_cond")
        print(f"      [{label}] mean PPL_cond={ppl:.2f}" if ppl else f"      [{label}] no results")
        out[label] = {
            **agg,
            "per_sequence": seq_results,
            "n_sequences": len(seq_results),
        }
    return out


# ── sweep runner ───────────────────────────────────────────────────────────────

def eval_perplexity_sweep(
    dataset_path: str,
    sweep_dir: str | None = None,
    config_name: str = "bliss",
    bucket_size: int = 42,
    batch_size: int = 4,
    skip_base: bool = False,
    force_restart: bool = False,
    base_model_override: str | None = None,
    block_use_user_context: bool = False,
    tag: str | None = None,
    skip_block: bool = False,
    base_only: bool = False,
):
    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    suffix = f"_{tag}" if tag else ""

    if dataset_path:
        sequences = load_dataset(dataset_path)
        print(f"Loaded {len(sequences)} sequences from {dataset_path}")
    else:
        sequences = _load_dataset_for_config(config_name)
        print(f"Loaded {len(sequences)} sequences from config '{config_name}' TEST_DATASET")
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
    base_cache = root / f"base_perplexity_result{suffix}.json"

    if not skip_base:
        if base_cache.exists() and not force_restart:
            print(f"\nLoading cached base model perplexity ({eval_base_model})...")
            with open(base_cache, "r") as f:
                base_result = json.load(f)
            print(
                f"  mean PPL_cond={base_result.get('mean_ppl_cond'):.3f}  "
                f"mean PPL_block={base_result.get('mean_ppl_block', 'N/A')}"
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
                skip_block=skip_block,
                base_training_client=base_t_client,
            )
            base_result["model_path"] = eval_base_model
            print(
                f"  Base PPL_cond={base_result['mean_ppl_cond']:.3f}  "
                f"PPL_block={base_result.get('mean_ppl_block', 'N/A')}"
            )
            with open(base_cache, "w") as f:
                json.dump(base_result, f, indent=2)

    if base_only:
        print("\n--base-only: skipping per-run evaluation.")
        return

    # ── per-run evaluation ─────────────────────────────────────────────────────
    all_run_results: dict[str, dict] = {}

    for run_name, cycles in sorted(run_model_paths.items()):
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  ({len(cycles)} cycles)")

        results_file = root / run_name / f"perplexity_results{suffix}.json"

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
                skip_block=skip_block,
                base_training_client=base_t_client,
            )
            if result is None:
                print(f"  Skipping cycle {cycle_num} (checkpoint expired or unavailable).")
                continue
            cycle_results.append({
                "cycle": cycle_num,
                "model_path": model_path,
                **result,
            })
            cycle_results.sort(key=lambda x: x["cycle"])
            print(
                f"  Cycle {cycle_num}: "
                f"PPL_cond={result['mean_ppl_cond']:.3f}  "
                f"PPL_block={result.get('mean_ppl_block', 'N/A')}"
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
    combined_out = root / f"sweep_perplexity_results{suffix}.json"
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
            f"PPL_block={base_result.get('mean_ppl_block', 'N/A')}"
        )
    for run_name in sorted(all_run_results):
        for c in all_run_results[run_name].get("cycle_results", []):
            print(
                f"  {run_name} cycle {c['cycle']}: "
                f"PPL_cond={c['mean_ppl_cond']:.3f}  "
                f"PPL_block={c.get('mean_ppl_block', 'N/A')}"
            )


def eval_perplexity_sweep_multi(
    label_sequences: dict[str, list[dict]],
    sweep_dir: str | None = None,
    config_name: str = "bliss",
    bucket_size: int = 42,
    skip_base: bool = False,
    force_restart: bool = False,
    base_model_override: str | None = None,
    base_only: bool = False,
):
    """Run perplexity eval for multiple datasets, loading each checkpoint once.

    This is the parallelized version of calling eval_perplexity_sweep in a loop.
    For each checkpoint, all datasets are evaluated in a single model load.
    """
    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    labels = list(label_sequences.keys())
    print(f"\nMulti-dataset eval: {labels}")
    print(f"Bucket size: {bucket_size} tokens")

    service_client = tinker.ServiceClient()

    # ── infer base model ────────────────────────────────────────────────────
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
                        break
                except Exception:
                    continue

    eval_base_model = base_model_override or inferred_base_model or BASE_MODEL
    print(f"Base model: {eval_base_model}")

    # ── renderer + tokenizer ────────────────────────────────────────────────
    base_t_client = service_client.create_lora_training_client(base_model=eval_base_model)
    tokenizer = base_t_client.get_tokenizer()
    renderer = get_renderer(tokenizer, eval_base_model)

    # ── base model eval (once for all datasets) ────────────────────────────
    if not skip_base:
        # Check which labels need base eval
        labels_needing_base = []
        for label in labels:
            cache = root / f"base_perplexity_result_{label}.json"
            if cache.exists() and not force_restart:
                with open(cache, "r") as f:
                    cached = json.load(f)
                print(f"  [{label}] cached base PPL_cond={cached.get('mean_ppl_cond', 'N/A')}")
            else:
                labels_needing_base.append(label)

        if labels_needing_base:
            datasets_for_base = {l: label_sequences[l] for l in labels_needing_base}
            print(f"\n--- Evaluating base model for: {labels_needing_base} ---")
            base_results = eval_model_perplexity_multi(
                service_client=service_client,
                model_path=eval_base_model,
                renderer=renderer,
                tokenizer=tokenizer,
                datasets=datasets_for_base,
                bucket_size=bucket_size,
            )
            for label, result in base_results.items():
                result["model_path"] = eval_base_model
                cache = root / f"base_perplexity_result_{label}.json"
                with open(cache, "w") as f:
                    json.dump(result, f, indent=2)

    if base_only:
        print("\n--base-only: skipping per-run evaluation.")
        return

    # ── discover runs ───────────────────────────────────────────────────────
    combined_sweep = root / "sweep_eval_results.json"
    run_model_paths: dict[str, list[dict]] = {}

    if combined_sweep.exists():
        with open(combined_sweep, "r") as f:
            sweep_data = json.load(f)
        for run_name, run_data in sweep_data.get("runs", {}).items():
            run_model_paths[run_name] = [
                {"cycle": c["cycle"], "model_path": c["model_path"]}
                for c in run_data.get("cycle_results", [])
            ]
    else:
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
        raise FileNotFoundError(f"No eval results found in {root}. Run eval_sweep.py first.")
    print(f"\nFound {len(run_model_paths)} runs in {root}")

    # ── per-run evaluation ──────────────────────────────────────────────────
    # Track results per label
    all_run_results: dict[str, dict[str, dict]] = {label: {} for label in labels}

    for run_name, cycles in sorted(run_model_paths.items()):
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  ({len(cycles)} cycles)")

        # Check which labels still need eval for this run
        labels_todo = []
        for label in labels:
            results_file = root / run_name / f"perplexity_results_{label}.json"
            if results_file.exists() and not force_restart:
                try:
                    with open(results_file, "r") as f:
                        existing = json.load(f)
                    existing_cycles = {c["cycle"] for c in existing.get("cycle_results", [])}
                    if {c["cycle"] for c in cycles}.issubset(existing_cycles):
                        all_run_results[label][run_name] = existing
                        continue
                except Exception:
                    pass
            labels_todo.append(label)

        if not labels_todo:
            print(f"  All labels fully evaluated, skipping.")
            continue

        print(f"  Evaluating labels: {labels_todo}")

        # Track cycle results per label
        label_cycle_results: dict[str, list[dict]] = {l: [] for l in labels_todo}

        for c in cycles:
            cycle_num = c["cycle"]
            model_path = c["model_path"]
            print(f"\n  Cycle {cycle_num}: {model_path}")

            datasets_for_cycle = {l: label_sequences[l] for l in labels_todo}
            results = eval_model_perplexity_multi(
                service_client=service_client,
                model_path=model_path,
                renderer=renderer,
                tokenizer=tokenizer,
                datasets=datasets_for_cycle,
                bucket_size=bucket_size,
            )

            for label in labels_todo:
                label_cycle_results[label].append({
                    "cycle": cycle_num,
                    "model_path": model_path,
                    **results[label],
                })

        # Save per-label results for this run
        for label in labels_todo:
            run_data = {
                "run_name": run_name,
                "bucket_size": bucket_size,
                "cycle_results": label_cycle_results[label],
            }
            results_file = root / run_name / f"perplexity_results_{label}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(run_data, f, indent=2)
            all_run_results[label][run_name] = run_data

    # ── combined output per label ───────────────────────────────────────────
    for label in labels:
        base_cache = root / f"base_perplexity_result_{label}.json"
        base_result = None
        if base_cache.exists():
            with open(base_cache, "r") as f:
                base_result = json.load(f)

        combined_out = root / f"sweep_perplexity_results_{label}.json"
        with open(combined_out, "w") as f:
            json.dump(
                {
                    "config_name": config_name,
                    "sweep_dir": str(root),
                    "bucket_size": bucket_size,
                    "base_model": eval_base_model,
                    "base_result": base_result,
                    "runs": all_run_results[label],
                },
                f,
                indent=2,
            )
        print(f"Saved combined results to {combined_out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate bucket perplexity for all sweep checkpoints"
    )
    parser.add_argument(
        "--dataset",
        "-D",
        type=str,
        default=None,
        help=(
            "Path to a JSON dataset file. If omitted, the test dataset is "
            "resolved automatically from the config's TEST_DATASET attribute."
        ),
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
        default=42,
        help="Number of tokens per bucket (default: 42)",
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
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default=None,
        help=(
            "Tag appended to output filenames to avoid collisions when running "
            "multiple datasets on the same sweep (e.g. --tag bliss_high)"
        ),
    )
    parser.add_argument(
        "--skip-block",
        action="store_true",
        help="Skip PPL_block computation (much faster, only compute PPL_cond)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only evaluate the base model, skip all fine-tuned checkpoints",
    )
    parser.add_argument(
        "--persona",
        nargs="*",
        default=None,
        metavar="LABEL",
        help=(
            "Run perplexity eval for datasets in the config's PERPLEXITY_DATASETS dict. "
            "With no arguments, runs all. With arguments, runs only matching labels "
            "(e.g. --persona high_sdf low_sdf)."
        ),
    )
    args = parser.parse_args()

    if args.persona is not None:
        config = get_config(args.config)
        ppl_datasets = getattr(config, "PERPLEXITY_DATASETS", None)
        if not ppl_datasets:
            raise ValueError(
                f"Config '{args.config}' has no PERPLEXITY_DATASETS defined. "
                f"Add it to src/training_configs/{args.config}.py"
            )
        if args.persona:
            ppl_datasets = {k: v for k, v in ppl_datasets.items() if k in args.persona}
            if not ppl_datasets:
                raise ValueError(
                    f"No matching labels found. Available: {list(getattr(config, 'PERPLEXITY_DATASETS', {}).keys())}"
                )

        # Load all datasets upfront
        label_sequences = {}
        for label, rel_path in ppl_datasets.items():
            dataset_file = str(_DATASETS_DIR / rel_path)
            seqs = load_dataset(dataset_file)
            label_sequences[label] = seqs
            print(f"  [{label}] Loaded {len(seqs)} sequences from {rel_path}")

        # Run a unified sweep that loads each checkpoint once for all datasets
        eval_perplexity_sweep_multi(
            label_sequences=label_sequences,
            sweep_dir=args.sweep_dir,
            config_name=args.config,
            bucket_size=args.bucket_size,
            skip_base=args.skip_base,
            force_restart=args.force_restart,
            base_model_override=args.base_model,
            base_only=args.base_only,
        )
    else:
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
            tag=args.tag,
            skip_block=args.skip_block,
            base_only=args.base_only,
        )
