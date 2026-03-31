"""Evaluate branching factor across all sweep checkpoints.

For each run in the sweep directory, evaluates every cycle's model
to compute position-wise branching factor (BF).

Results are saved:
  - Per-run:   <sweep_dir>/<run_name>/bf_results.json
  - Per-run:   <sweep_dir>/<run_name>/bf_by_position.png
  - Combined:  <sweep_dir>/sweep_bf_results.json

Already-evaluated runs (those with bf_results.json) are skipped on re-run.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import tinker

from evaluation.eval_branching_factor import (
    evaluate_model_bf,
    plot_bf_by_position,
    BASE_MODEL,
    NUM_SAMPLES_PER_QUESTION,
    EMA_ALPHA,
)
from training_configs import get_config
from utils.renderer_utils import get_renderer

_SEED_NTE_RE = re.compile(r"seed(\d+)_nte(\d+)$")
_NUM_RE = r"([\deE.+-]+)"
_BETA_NTE_RE = re.compile(rf"beta{_NUM_RE}_nte(\d+)$")
_BETA_STEPS_RE = re.compile(rf"beta{_NUM_RE}_steps(\d+)$")
_BETA_LR_RE = re.compile(rf"beta{_NUM_RE}_lr{_NUM_RE}$")


def _is_run_dir(name: str) -> bool:
    return bool(
        _SEED_NTE_RE.match(name)
        or _BETA_NTE_RE.match(name)
        or _BETA_STEPS_RE.match(name)
        or _BETA_LR_RE.match(name)
    )


def _get_sort_key(name: str) -> tuple:
    m = _SEED_NTE_RE.match(name)
    if m:
        return (0, int(m.group(1)), int(m.group(2)))
    m = _BETA_NTE_RE.match(name)
    if m:
        return (1, float(m.group(1)), int(m.group(2)))
    m = _BETA_STEPS_RE.match(name)
    if m:
        return (2, float(m.group(1)), int(m.group(2)))
    m = _BETA_LR_RE.match(name)
    if m:
        return (3, float(m.group(1)), float(m.group(2)))
    return (99, 0, 0)


def _matches_filter(
    name: str,
    filter_betas: list[float] | None = None,
    filter_firstn: list[int] | None = None,
    filter_nte: list[int] | None = None,
    filter_lrs: list[float] | None = None,
) -> bool:
    if filter_betas is not None:
        m = _BETA_NTE_RE.match(name) or _BETA_STEPS_RE.match(name) or _BETA_LR_RE.match(name)
        if not m or float(m.group(1)) not in filter_betas:
            return False
    if filter_firstn is not None:
        m = _SEED_NTE_RE.match(name)
        if not m or int(m.group(1)) not in filter_firstn:
            return False
    if filter_nte is not None:
        m = _SEED_NTE_RE.match(name) or _BETA_NTE_RE.match(name)
        if not m or int(m.group(2)) not in filter_nte:
            return False
    if filter_lrs is not None:
        m = _BETA_LR_RE.match(name)
        if not m or float(m.group(2)) not in filter_lrs:
            return False
    return True


def eval_bf_sweep(
    config_name: str = "bliss",
    sweep_dir: str | None = None,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    skip_base: bool = False,
    force_restart: bool = False,
    max_tokens: int = 400,
    temperature: float = 0.7,
    ema_alpha: float = EMA_ALPHA,
    batch_size: int = 8,
    use_self_logprobs: bool = True,
    skip_plots: bool = False,
    base_model_override: str | None = None,
    filter_betas: list[float] | None = None,
    filter_firstn: list[int] | None = None,
    filter_nte: list[int] | None = None,
    filter_lrs: list[float] | None = None,
):
    config = get_config(config_name)
    questions = config.EVAL_QUESTIONS

    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    service_client = tinker.ServiceClient()

    # --- Infer base model from the first available experiment summary ---
    inferred_base_model = None
    for run_dir in root.iterdir():
        if run_dir.is_dir() and _is_run_dir(run_dir.name):
            summary_file = run_dir / "experiment_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                    inferred_base_model = summary.get("model")
                    if inferred_base_model:
                        print(f"Inferred base model from {summary_file}: {inferred_base_model}")
                        break
                except Exception:
                    continue

    eval_base_model = base_model_override or inferred_base_model or BASE_MODEL
    if not base_model_override and inferred_base_model:
        print(f"Using inferred base model: {eval_base_model}")
    elif not base_model_override:
        print(f"Could not infer base model, falling back to default: {eval_base_model}")

    training_client = service_client.create_lora_training_client(base_model=eval_base_model)
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer, eval_base_model)

    mode_label = "self-logprobs (true BF)" if use_self_logprobs else "base-model logprobs (cross-BF)"
    print("=" * 60)
    print(f"BF sweep eval: {config_name}")
    print(f"  mode: {mode_label}")
    print(f"  samples/question: {num_samples}, T={temperature}, max_tokens={max_tokens}")
    print("=" * 60)

    # ── base model ──────────────────────────────────────────
    base_bf_result = None
    base_cache = root / "base_bf_result.json"

    if not skip_base:
        if base_cache.exists() and not force_restart:
            print(f"Loading cached base model BF from {base_cache}...")
            with open(base_cache, "r") as f:
                base_bf_result = json.load(f)
            print(f"  Base overall BF: {base_bf_result['overall_bf']:.2f}")
        else:
            print(f"\n--- Evaluating base model BF ({eval_base_model}) ---")
            base_bf_result = evaluate_model_bf(
                service_client=service_client,
                model_path=eval_base_model,
                questions=questions,
                renderer=renderer,
                training_client=training_client,
                use_self_logprobs=use_self_logprobs,
                num_samples=num_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                ema_alpha=ema_alpha,
                batch_size=batch_size,
            )
            base_bf_result["label"] = "base model"
            print(f"  Base overall BF: {base_bf_result['overall_bf']:.2f}")
            with open(base_cache, "w") as f:
                json.dump(base_bf_result, f, indent=2)

    # ── find all run dirs ───────────────────────────────────
    run_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and _is_run_dir(d.name)],
        key=lambda d: _get_sort_key(d.name),
    )

    is_filtered = bool(filter_betas or filter_firstn or filter_nte or filter_lrs)
    if is_filtered:
        run_dirs = [
            d for d in run_dirs
            if _matches_filter(d.name, filter_betas, filter_firstn, filter_nte, filter_lrs)
        ]
        print(f"\nFound {len(run_dirs)} runs matching filter in {root}")
    else:
        print(f"\nFound {len(run_dirs)} runs in {root}")

    # ── combined results (load existing when filtering) ─────
    combined_file = root / "sweep_bf_results.json"
    all_results = {}
    existing_base_bf = None
    if is_filtered and not force_restart and combined_file.exists():
        try:
            with open(combined_file, "r") as f:
                existing_combined = json.load(f)
            all_results = existing_combined.get("runs", {})
            existing_base_bf = existing_combined.get("base_result")
        except Exception:
            pass

    # ── evaluate each run ───────────────────────────────────
    for run_dir in run_dirs:
        run_name = run_dir.name
        summary_file = run_dir / "experiment_summary.json"
        bf_file = run_dir / "bf_results.json"

        # Discover cycles
        summary = None
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        summary = json.loads(content)
            except Exception:
                pass

        if summary:
            cycles = summary["cycles"]
        else:
            cycles = []
            cycle_dirs = sorted(
                [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("cycle")],
                key=lambda x: int(x.name.replace("cycle", "")),
            )
            for cd in cycle_dirs:
                log_file = cd / "log.txt"
                if log_file.exists():
                    with open(log_file, "r") as f:
                        model_path = f.read().strip()
                        if model_path:
                            cycles.append({
                                "cycle": int(cd.name.replace("cycle", "")),
                                "model_path": model_path,
                            })
            if not cycles:
                print(f"\n  {run_name}: no valid cycles found, skipping")
                continue
            print(f"\n  {run_name}: found {len(cycles)} cycles via fallback scan")

        # Check if already fully evaluated
        if bf_file.exists() and not force_restart:
            try:
                with open(bf_file, "r") as f:
                    existing_data = json.load(f)
                existing_cycle_nums = {
                    c["cycle"] for c in existing_data.get("cycle_results", [])
                }
                expected_cycle_nums = {c["cycle"] for c in cycles}
                if expected_cycle_nums.issubset(existing_cycle_nums):
                    print(f"\n  {run_name}: already fully evaluated, loading existing results")
                    all_results[run_name] = existing_data
                    continue
                else:
                    print(
                        f"\n  {run_name}: partially evaluated "
                        f"({len(existing_cycle_nums)}/{len(expected_cycle_nums)} cycles), resuming..."
                    )
            except Exception:
                print(f"\n  {run_name}: error reading existing results, re-evaluating")

        print(f"\nEvaluating BF for {run_name} ({len(cycles)} cycles)")

        # Load partial results for resumption
        cycle_results = []
        if bf_file.exists() and not force_restart:
            try:
                with open(bf_file, "r") as f:
                    existing_data = json.load(f)
                cycle_results = existing_data.get("cycle_results", [])
            except Exception:
                pass

        existing_cycle_nums = {c["cycle"] for c in cycle_results}
        cycles_to_eval = [c for c in cycles if c["cycle"] not in existing_cycle_nums]

        for c in cycles_to_eval:
            cycle_num = c["cycle"]
            model_path = c["model_path"]
            print(f"  cycle {cycle_num}...")

            bf = evaluate_model_bf(
                service_client=service_client,
                model_path=model_path,
                questions=questions,
                renderer=renderer,
                training_client=training_client,
                use_self_logprobs=use_self_logprobs,
                num_samples=num_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                ema_alpha=ema_alpha,
                batch_size=batch_size,
            )
            print(f"    overall BF: {bf['overall_bf']:.2f}")

            cycle_results.append({
                "cycle": cycle_num,
                "model_path": model_path,
                "overall_bf": bf["overall_bf"],
                "positions": bf["positions"],
                "bf_raw": bf["bf_raw"],
                "bf_smoothed": bf["bf_smoothed"],
                "num_responses": bf["num_responses"],
                "sample_responses": bf["sample_responses"],
            })

            # Save incrementally
            cycle_results.sort(key=lambda x: x["cycle"])
            run_data = {
                "run_name": run_name,
                "config_name": config_name,
                "questions": questions,
                "num_samples_per_question": num_samples,
                "use_self_logprobs": use_self_logprobs,
                "ema_alpha": ema_alpha,
                "temperature": temperature,
                "cycle_results": cycle_results,
            }
            with open(bf_file, "w") as f:
                json.dump(run_data, f, indent=2)

        # Finalize per-run data
        cycle_results.sort(key=lambda x: x["cycle"])
        run_data = {
            "run_name": run_name,
            "config_name": config_name,
            "questions": questions,
            "num_samples_per_question": num_samples,
            "use_self_logprobs": use_self_logprobs,
            "ema_alpha": ema_alpha,
            "temperature": temperature,
            "cycle_results": cycle_results,
        }
        with open(bf_file, "w") as f:
            json.dump(run_data, f, indent=2)

        # Per-run BF plot
        if not skip_plots:
            bf_plot_data = []
            if base_bf_result:
                bf_plot_data.append({
                    "label": "base model",
                    "positions": base_bf_result["positions"],
                    "bf_smoothed": base_bf_result["bf_smoothed"],
                    "overall_bf": base_bf_result["overall_bf"],
                })
            for cr in cycle_results:
                bf_plot_data.append({
                    "label": f"cycle {cr['cycle']}",
                    "positions": cr["positions"],
                    "bf_smoothed": cr["bf_smoothed"],
                    "overall_bf": cr["overall_bf"],
                })
            plot_path = str(run_dir / "bf_by_position.png")
            plot_bf_by_position(
                bf_results=bf_plot_data,
                output_path=plot_path,
                config_name=f"{config_name} / {run_name}",
                ema_alpha=ema_alpha,
            )

        bfs = [cr["overall_bf"] for cr in cycle_results]
        bf_str = " -> ".join(f"{b:.2f}" for b in bfs)
        print(f"  {run_name} complete: BF = {bf_str}")

        all_results[run_name] = run_data

    # ── save combined results ───────────────────────────────
    effective_base_bf = base_bf_result if base_bf_result is not None else existing_base_bf
    with open(combined_file, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "sweep_dir": str(root),
                "base_model": eval_base_model,
                "base_result": effective_base_bf,
                "num_samples_per_question": num_samples,
                "use_self_logprobs": use_self_logprobs,
                "ema_alpha": ema_alpha,
                "temperature": temperature,
                "runs": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined BF results to {combined_file}")

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP BF EVAL SUMMARY")
    print("=" * 60)
    if effective_base_bf:
        print(f"Base model BF: {effective_base_bf['overall_bf']:.2f}")
    for run_name in sorted(all_results):
        data = all_results[run_name]
        bfs = [c["overall_bf"] for c in data["cycle_results"]]
        bf_str = " -> ".join(f"{b:.2f}" for b in bfs)
        print(f"  {run_name:30s}  {bf_str}")
    print("=" * 60)


if __name__ == "__main__":
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Evaluate branching factor across all sweep checkpoints",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="bliss",
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--sweep-dir", "-d", type=str, default=None)
    parser.add_argument(
        "--samples-per-question", type=int, default=NUM_SAMPLES_PER_QUESTION,
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--force-restart", action="store_true",
        help="re-evaluate all runs even if results already exist",
    )
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="sampling temperature",
    )
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--base-logprobs", action="store_true",
        help="use base model logprobs instead of self-model logprobs "
             "(faster but measures cross-BF, not true BF)",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="skip generating per-run BF position plots",
    )
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--filter-betas", nargs="+", type=float, default=None,
        help="only evaluate runs with these DPO beta values",
    )
    parser.add_argument(
        "--filter-firstn", nargs="+", type=int, default=None,
        help="only evaluate runs with these firstn (seed) values",
    )
    parser.add_argument(
        "--filter-nte", nargs="+", type=int, default=None,
        help="only evaluate runs with these num_training_examples values",
    )
    parser.add_argument(
        "--filter-lrs", nargs="+", type=float, default=None,
        help="only evaluate runs with these DPO learning-rate values",
    )
    args = parser.parse_args()

    eval_bf_sweep(
        config_name=args.config,
        sweep_dir=args.sweep_dir,
        num_samples=args.samples_per_question,
        skip_base=args.skip_base,
        force_restart=args.force_restart,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ema_alpha=args.ema_alpha,
        batch_size=args.batch_size,
        use_self_logprobs=not args.base_logprobs,
        skip_plots=args.skip_plots,
        base_model_override=args.base_model,
        filter_betas=args.filter_betas,
        filter_firstn=args.filter_firstn,
        filter_nte=args.filter_nte,
        filter_lrs=args.filter_lrs,
    )
