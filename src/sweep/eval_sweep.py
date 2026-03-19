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
import re
from pathlib import Path

import tinker

from evaluation.eval import (
    evaluate_model_score, 
    BASE_MODEL, 
    strip_scores_from_result, 
    strip_scores_from_cycle_results,
    score_responses
)
from training_configs import get_config

NUM_SAMPLES_PER_QUESTION = 20

_SEED_NTE_RE = re.compile(r"seed(\d+)_nte(\d+)$")
_BETA_NTE_RE = re.compile(r"beta([\d.]+)_nte(\d+)$")
_BETA_STEPS_RE = re.compile(r"beta([\d.]+)_steps(\d+)$")


def _is_run_dir(name: str) -> bool:
    return bool(
        _SEED_NTE_RE.match(name)
        or _BETA_NTE_RE.match(name)
        or _BETA_STEPS_RE.match(name)
    )


def _get_sort_key(name: str) -> tuple:
    m = _SEED_NTE_RE.match(name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = _BETA_NTE_RE.match(name)
    if m:
        return (float(m.group(1)), int(m.group(2)))
    m = _BETA_STEPS_RE.match(name)
    if m:
        return (float(m.group(1)), int(m.group(2)))
    return (0, 0)


def _matches_filter(
    name: str,
    filter_betas: list[float] | None = None,
    filter_firstn: list[int] | None = None,
    filter_nte: list[int] | None = None,
) -> bool:
    if filter_betas is not None:
        m = _BETA_NTE_RE.match(name) or _BETA_STEPS_RE.match(name)
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
    return True


def eval_sweep(
    config_name: str = "bliss",
    sweep_dir: str | None = None,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    skip_base: bool = False,
    force_restart: bool = False,
    parallel: int = 1,
    skip_coherence: bool = False,
    base_model_override: str | None = None,
    use_generated_responses: bool = False,
    filter_betas: list[float] | None = None,
    filter_firstn: list[int] | None = None,
    filter_nte: list[int] | None = None,
):
    config = get_config(config_name)
    score_prompt = getattr(config, "SCORE_PROMPT")
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

    # Get renderer
    training_client = service_client.create_lora_training_client(base_model=eval_base_model)
    tokenizer = training_client.get_tokenizer()
    from evaluation.eval import get_renderer
    renderer = get_renderer(tokenizer, eval_base_model)

    coherence_prompt = getattr(config, "COHERENCE_PROMPT", None)
    if skip_coherence:
        coherence_prompt = None

    # ── base model (evaluate once) ──────────────────────────
    base_result = None
    base_cache = root / "base_eval_result.json"
    base_responses_cache = root / "base_eval_responses.json"

    if not skip_base:
        if use_generated_responses and base_responses_cache.exists():
            print(f"\n--- Scoring base model from {base_responses_cache} ---")
            with open(base_responses_cache, "r") as f:
                base_responses_data = json.load(f)
            responses = base_responses_data.get("responses", [])
            base_result = score_responses(
                responses=responses,
                questions=questions,
                score_prompt=score_prompt,
                coherence_prompt=coherence_prompt,
            )
            print(f"  Base score: {base_result['aggregate_score']:.1f}")
            with open(base_cache, "w") as f:
                json.dump(base_result, f, indent=2)
        elif base_cache.exists() and not force_restart:
            print(f"Loading cached base model eval from {base_cache}...")
            with open(base_cache, "r") as f:
                base_result = json.load(f)
            print(f"  Base score: {base_result['aggregate_score']:.1f}")
        elif not use_generated_responses:
            print(f"\n--- Evaluating base model ({eval_base_model}) ---")
            base_result = evaluate_model_score(
                service_client=service_client,
                model_path=eval_base_model,
                questions=questions,
                score_prompt=score_prompt,
                renderer=renderer,
                coherence_prompt=coherence_prompt,
                num_samples=num_samples,
            )
            print(f"  Base score: {base_result['aggregate_score']:.1f}")
            with open(base_cache, "w") as f:
                json.dump(base_result, f, indent=2)
            
            # Save responses-only version of base
            with open(base_responses_cache, "w") as f:
                json.dump(strip_scores_from_result(base_result), f, indent=2)
        else:
            print(f"  warning: skip_base is False but {base_responses_cache} missing and use_generated_responses is True. Skipping base model.")

    # ── find all run dirs ───────────────────────────────────
    run_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and _is_run_dir(d.name)],
        key=lambda d: _get_sort_key(d.name),
    )

    is_filtered = bool(filter_betas or filter_firstn or filter_nte)
    if is_filtered:
        run_dirs = [
            d for d in run_dirs
            if _matches_filter(d.name, filter_betas, filter_firstn, filter_nte)
        ]
        print(f"\nFound {len(run_dirs)} runs matching filter in {root}")
    else:
        print(f"\nFound {len(run_dirs)} runs in {root}")

    # ── combined results files ────────────────────────────────
    combined_file = root / "sweep_eval_results.json"
    combined_responses_file = root / "sweep_eval_responses.json"

    # When filtering, merge new results into existing combined file
    all_results = {}
    existing_base_result = None
    if is_filtered and not force_restart and combined_file.exists():
        try:
            with open(combined_file, "r") as f:
                existing_combined = json.load(f)
            all_results = existing_combined.get("runs", {})
            existing_base_result = existing_combined.get("base_result")
        except Exception:
            pass
    for run_dir in run_dirs:
        run_name = run_dir.name
        summary_file = run_dir / "experiment_summary.json"
        results_file = run_dir / "eval_results.json"
        responses_file = run_dir / "eval_responses.json"

        # 1. Discover all cycles first (summary or fallback)
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
            # Fallback: Scan cycle directories directly
            cycles = []
            cycle_dirs = sorted(
                [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("cycle")],
                key=lambda x: int(x.name.replace("cycle", ""))
            )
            for cd in cycle_dirs:
                log_file = cd / "log.txt"
                if log_file.exists():
                    with open(log_file, "r") as f:
                        model_path = f.read().strip()
                        if model_path:
                            cycles.append({
                                "cycle": int(cd.name.replace("cycle", "")),
                                "model_path": model_path
                            })
            
            if not cycles:
                print(f"\n  {run_name}: no valid cycles found, skipping")
                continue
            print(f"\n  {run_name}: found {len(cycles)} cycles via fallback scan")

        # 2. Check if already evaluated and COMPLETE
        should_eval = True
        if results_file.exists() and not force_restart and not use_generated_responses:
            try:
                with open(results_file, "r") as f:
                    existing_data = json.load(f)
                
                existing_cycle_nums = {c["cycle"] for c in existing_data.get("cycle_results", [])}
                expected_cycle_nums = {c["cycle"] for c in cycles}
                
                if expected_cycle_nums.issubset(existing_cycle_nums):
                    print(f"\n  {run_name}: already fully evaluated, loading existing results")
                    all_results[run_name] = existing_data
                    should_eval = False
                else:
                    print(f"\n  {run_name}: partially evaluated ({len(existing_cycle_nums)}/{len(expected_cycle_nums)} cycles), resuming...")
            except Exception:
                print(f"\n  {run_name}: error reading existing results, re-evaluating")
        
        if not should_eval:
            continue

        print(f"\nEvaluating {run_name} ({len(cycles)} cycles, parallel={parallel})")
        print(f"  Logging to: {run_dir / 'eval_sweep.log'}")

        import concurrent.futures
        from contextlib import redirect_stdout, redirect_stderr

        # 3. Load responses if using pre-generated ones
        pre_responses = {}
        if use_generated_responses and responses_file.exists():
            print(f"  {run_name}: loading pre-generated responses from {responses_file}")
            try:
                with open(responses_file, "r") as f:
                    resp_data = json.load(f)
                for cr in resp_data.get("cycle_results", []):
                    pre_responses[cr["cycle"]] = cr["responses"]
            except Exception as e:
                print(f"  {run_name}: error loading {responses_file}: {e}")

        def eval_single_cycle(c):
            cycle_num = c["cycle"]
            model_path = c["model_path"]
            
            if use_generated_responses and cycle_num in pre_responses:
                print(f"  starting score-only for {run_name} cycle {cycle_num}...")
                responses = pre_responses[cycle_num]
                result = score_responses(
                    responses=responses,
                    questions=questions,
                    score_prompt=score_prompt,
                    coherence_prompt=coherence_prompt,
                    verbose=False,
                )
            else:
                if use_generated_responses:
                    print(f"  warning: no pre-generated responses for cycle {cycle_num}, falling back to full eval")
                print(f"  starting eval for {run_name} cycle {cycle_num}...")
                result = evaluate_model_score(
                    service_client=service_client,
                    model_path=model_path,
                    questions=questions,
                    score_prompt=score_prompt,
                    renderer=renderer,
                    coherence_prompt=coherence_prompt,
                    num_samples=num_samples,
                )
            
            print(f"    {run_name} cycle {cycle_num} score: {result['aggregate_score']:.1f}")
            return {
                "cycle": cycle_num,
                "model_path": model_path,
                "aggregate_score": result["aggregate_score"],
                "total_responses": result["total_responses"],
                "per_question": result["per_question"],
                "responses": result["responses"],
            }

        log_file = run_dir / "eval_sweep.log"
        with open(log_file, "w", buffering=1) as f:
            with redirect_stdout(f), redirect_stderr(f):
                print(f"{'=' * 60}")
                print(f"Evaluating {run_name} ({len(cycles)} cycles, parallel={parallel})")
                print(f"{'=' * 60}")
                
                # Load existing if available for resumption
                cycle_results = []
                if results_file.exists() and not force_restart and not use_generated_responses:
                    try:
                        with open(results_file, "r") as f:
                            existing_data = json.load(f)
                        cycle_results = existing_data.get("cycle_results", [])
                        print(f"  resuming from {len(cycle_results)} already-evaluated cycles...")
                    except Exception:
                        pass
                
                existing_cycle_nums = {c["cycle"] for c in cycle_results}
                cycles_to_eval = [c for c in cycles if c["cycle"] not in existing_cycle_nums]

                if cycles_to_eval:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                        future_to_cycle = {executor.submit(eval_single_cycle, c): c for c in cycles_to_eval}
                        for future in concurrent.futures.as_completed(future_to_cycle):
                            res = future.result()
                            cycle_results.append(res)
                            
                            # Sort and save incrementally
                            cycle_results.sort(key=lambda x: x["cycle"])
                            run_data = {
                                "run_name": run_name,
                                "config_name": config_name,
                                "questions": questions,
                                "num_samples_per_question": num_samples,
                                "cycle_results": cycle_results,
                            }
                            with open(results_file, "w") as rf:
                                json.dump(run_data, rf, indent=2)

                            # Save responses-only version
                            responses_only_data = {
                                "run_name": run_name,
                                "config_name": config_name,
                                "questions": questions,
                                "num_samples_per_question": num_samples,
                                "cycle_results": strip_scores_from_cycle_results(cycle_results),
                            }
                            with open(responses_file, "w") as rf:
                                json.dump(responses_only_data, rf, indent=2)
                else:
                    # Case where it's fully evaluated but results didn't reflect yet
                    run_data = {
                        "run_name": run_name,
                        "config_name": config_name,
                        "questions": questions,
                        "num_samples_per_question": num_samples,
                        "cycle_results": cycle_results,
                    }

        # Print a quick summary to terminal after the run finishes
        scores = [c["aggregate_score"] for c in cycle_results]
        scores_str = " -> ".join(f"{s:.1f}" for s in scores)
        print(f"  {run_name} complete: {scores_str}")

        all_results[run_name] = run_data

    # ── combined results ────────────────────────────────────
    effective_base_result = base_result if base_result is not None else existing_base_result

    with open(combined_file, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "sweep_dir": str(root),
                "base_model": eval_base_model,
                "base_result": effective_base_result,
                "num_samples_per_question": num_samples,
                "runs": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved combined results to {combined_file}")

    # Save combined responses-only version
    all_responses_only = {}
    for run_name, run_data in all_results.items():
        all_responses_only[run_name] = {
            "run_name": run_data["run_name"],
            "config_name": run_data["config_name"],
            "questions": run_data["questions"],
            "num_samples_per_question": run_data["num_samples_per_question"],
            "cycle_results": strip_scores_from_cycle_results(run_data["cycle_results"])
        }

    with open(combined_responses_file, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "sweep_dir": str(root),
                "base_model": eval_base_model,
                "base_result": strip_scores_from_result(effective_base_result) if effective_base_result is not None else None,
                "num_samples_per_question": num_samples,
                "runs": all_responses_only,
            },
            f,
            indent=2,
        )
    print(f"Saved combined responses-only to {combined_responses_file}")

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP EVAL SUMMARY")
    print("=" * 60)
    if effective_base_result:
        print(f"Base model score: {effective_base_result['aggregate_score']:.1f}")
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
        "--config",
        "-c",
        type=str,
        default="bliss",
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--sweep-dir", "-d", type=str, default=None)
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=NUM_SAMPLES_PER_QUESTION,
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="re-evaluate all runs even if results already exist",
    )
    parser.add_argument(
        "--parallel",
        "-l",
        type=int,
        default=1,
        help="number of concurrent checkpoint evaluations per run",
    )
    parser.add_argument("--skip-coherence", action="store_true")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--use-generated-responses",
        action="store_true",
        help="evaluate using pre-generated responses from eval_responses.json"
    )
    parser.add_argument(
        "--filter-betas",
        nargs="+",
        type=float,
        default=None,
        help="only evaluate runs with these DPO beta values",
    )
    parser.add_argument(
        "--filter-firstn",
        nargs="+",
        type=int,
        default=None,
        help="only evaluate runs with these firstn (seed) values",
    )
    parser.add_argument(
        "--filter-nte",
        nargs="+",
        type=int,
        default=None,
        help="only evaluate runs with these num_training_examples values",
    )
    args = parser.parse_args()

    eval_sweep(
        config_name=args.config,
        sweep_dir=args.sweep_dir,
        num_samples=args.samples_per_question,
        skip_base=args.skip_base,
        force_restart=args.force_restart,
        parallel=args.parallel,
        skip_coherence=args.skip_coherence,
        base_model_override=args.base_model,
        use_generated_responses=args.use_generated_responses,
        filter_betas=args.filter_betas,
        filter_firstn=args.filter_firstn,
        filter_nte=args.filter_nte,
    )
