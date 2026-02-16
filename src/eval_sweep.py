"""Evaluate all sweep checkpoints on the bliss eval.

For each (firstn, nte) run in the sweep, evaluates every cycle's model
on the config's eval questions, scores with GPT-4o, and saves results.

Results are saved:
  - Per-run:   <sweep_dir>/<run_name>/eval_results.json
  - Combined:  <sweep_dir>/sweep_eval_results.json

Already-evaluated runs (those with eval_results.json) are skipped on re-run.
"""

import argparse
import json
import os
from pathlib import Path

import tinker
from openai import AsyncOpenAI

from eval import evaluate_model_score, BASE_MODEL
from training_configs import get_config

NUM_SAMPLES_PER_QUESTION = 20


def eval_sweep(
    config_name: str = "bliss",
    sweep_dir: str | None = None,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    skip_base: bool = False,
    force_restart: bool = False,
):
    config = get_config(config_name)
    score_prompt = getattr(config, "SCORE_PROMPT")
    questions = config.EVAL_QUESTIONS

    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    service_client = tinker.ServiceClient()
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ── base model (evaluate once) ──────────────────────────
    base_result = None
    base_cache = root / "base_eval_result.json"

    if not skip_base:
        if base_cache.exists() and not force_restart:
            print("Loading cached base model eval...")
            with open(base_cache, "r") as f:
                base_result = json.load(f)
            print(f"  Base score: {base_result['aggregate_score']:.1f}")
        else:
            print("\n--- Evaluating base model ---")
            base_result = evaluate_model_score(
                service_client=service_client,
                model_path=BASE_MODEL,
                questions=questions,
                score_prompt=score_prompt,
                async_openai_client=async_openai_client,
                num_samples=num_samples,
            )
            print(f"  Base score: {base_result['aggregate_score']:.1f}")
            with open(base_cache, "w") as f:
                json.dump(base_result, f, indent=2)

    # ── find all run dirs ───────────────────────────────────
    run_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed")]
    )
    print(f"\nFound {len(run_dirs)} runs in {root}")

    all_results = {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        summary_file = run_dir / "experiment_summary.json"
        results_file = run_dir / "eval_results.json"

        # skip if already evaluated
        if results_file.exists() and not force_restart:
            print(f"\n  {run_name}: already evaluated, loading existing results")
            with open(results_file, "r") as f:
                all_results[run_name] = json.load(f)
            continue

        if not summary_file.exists():
            print(f"\n  {run_name}: no experiment_summary.json, skipping")
            continue

        with open(summary_file, "r") as f:
            summary = json.load(f)

        cycles = summary["cycles"]
        print(f"\n{'=' * 60}")
        print(f"Evaluating {run_name} ({len(cycles)} cycles)")
        print(f"{'=' * 60}")

        cycle_results = []
        for c in cycles:
            cycle_num = c["cycle"]
            model_path = c["model_path"]
            print(f"\n  Cycle {cycle_num}: {model_path}")

            result = evaluate_model_score(
                service_client=service_client,
                model_path=model_path,
                questions=questions,
                score_prompt=score_prompt,
                async_openai_client=async_openai_client,
                num_samples=num_samples,
            )
            print(f"    Score: {result['aggregate_score']:.1f}")
            cycle_results.append(
                {
                    "cycle": cycle_num,
                    "model_path": model_path,
                    "aggregate_score": result["aggregate_score"],
                    "total_responses": result["total_responses"],
                    "per_question": result["per_question"],
                    "responses": result["responses"],
                }
            )

            # save incrementally so progress isn't lost on crash
            run_data = {
                "run_name": run_name,
                "config_name": config_name,
                "questions": questions,
                "num_samples_per_question": num_samples,
                "cycle_results": cycle_results,
            }
            with open(results_file, "w") as f:
                json.dump(run_data, f, indent=2)

        all_results[run_name] = run_data

    # ── combined results ────────────────────────────────────
    combined_file = root / "sweep_eval_results.json"
    with open(combined_file, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "sweep_dir": str(root),
                "base_model": BASE_MODEL,
                "base_result": base_result,
                "num_samples_per_question": num_samples,
                "runs": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to {combined_file}")

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP EVAL SUMMARY")
    print("=" * 60)
    if base_result:
        print(f"Base model score: {base_result['aggregate_score']:.1f}")
    for run_name in sorted(all_results):
        data = all_results[run_name]
        scores = [c["aggregate_score"] for c in data["cycle_results"]]
        scores_str = " -> ".join(f"{s:.1f}" for s in scores)
        print(f"  {run_name:30s}  {scores_str}")


if __name__ == "__main__":
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Evaluate all sweep checkpoints on config eval"
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
    args = parser.parse_args()

    eval_sweep(
        config_name=args.config,
        sweep_dir=args.sweep_dir,
        num_samples=args.samples_per_question,
        skip_base=args.skip_base,
        force_restart=args.force_restart,
    )
