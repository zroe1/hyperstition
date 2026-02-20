"""Add coherence scores and perplexity to existing sweep evaluation results.

For each run in the sweep that has eval_results.json, this script:
1. Computes coherence scores using GPT-4o (if coherence_prompt in config)
2. Computes perplexity using the base model
3. Updates the JSON files in-place with both metrics

This allows you to enrich already-completed eval sweeps without re-running
the primary scoring.

Updates:
  - Per-run:   <sweep_dir>/<run_name>/eval_results.json
  - Combined:  <sweep_dir>/sweep_eval_results.json
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from collections import defaultdict

import tinker
from openai import AsyncOpenAI

from compute_perplexity import BASE_MODEL, get_renderer, process_results
from eval import get_scores_batch_async
from training_configs import get_config


def subsample_responses(responses: list, questions: list, max_per_question: int | None) -> list:
    """Subsample responses to have at most max_per_question per question."""
    if max_per_question is None:
        return responses

    # Group by question
    question_groups = defaultdict(list)
    for i, resp in enumerate(responses):
        question_groups[resp["question"]].append((i, resp))

    # Take first max_per_question from each group
    selected_indices = set()
    for q, items in question_groups.items():
        for i, (idx, _) in enumerate(items[:max_per_question]):
            selected_indices.add(idx)

    # Return subsampled list preserving order
    return [resp for i, resp in enumerate(responses) if i in selected_indices]


def add_coherence_scores(
    responses: list,
    questions: list,
    coherence_prompt: str,
    async_openai_client: AsyncOpenAI,
    max_samples_per_question: int | None = None,
) -> dict:
    """Add coherence scores to responses, return aggregate and per-question stats."""
    # Subsample if requested
    responses_to_score = subsample_responses(responses, questions, max_samples_per_question)

    coherence_prompts = []
    valid_indices = []

    for i, item in enumerate(responses_to_score):
        if item["model_response"].strip():
            coherence_prompts.append(
                coherence_prompt.format(
                    question=item["question"],
                    answer=item["model_response"]
                )
            )
            valid_indices.append(i)

    print(f"    scoring {len(valid_indices)} responses for coherence...")
    coherence_scores = asyncio.run(
        get_scores_batch_async(async_openai_client, coherence_prompts)
    )

    idx_to_coherence = {}
    all_coherence = []

    for k, c_score in enumerate(coherence_scores):
        if c_score is None:
            continue
        c_clamped = max(0, min(100, float(c_score)))
        idx_to_coherence[valid_indices[k]] = c_clamped
        all_coherence.append(c_clamped)

    # Add coherence to scored responses
    for i, item in enumerate(responses_to_score):
        item["coherence"] = idx_to_coherence.get(i)

    # Compute aggregate
    aggregate_coherence = sum(all_coherence) / len(all_coherence) if all_coherence else None

    # Compute per-question
    per_question_coherence = {}
    for q_idx, q in enumerate(questions):
        q_coherences = [
            idx_to_coherence[i]
            for i in idx_to_coherence
            if responses_to_score[i]["question"] == q
        ]
        per_question_coherence[str(q_idx)] = (
            sum(q_coherences) / len(q_coherences) if q_coherences else None
        )

    return {
        "aggregate_coherence": aggregate_coherence,
        "per_question_coherence": per_question_coherence,
        "all_coherence": all_coherence,
    }


def add_perplexity(
    training_client,
    renderer,
    responses: list,
    questions: list,
    batch_size: int,
    max_samples_per_question: int | None = None,
) -> dict:
    """Compute perplexity for responses, return aggregate and per-question stats."""
    # Subsample if requested
    responses_to_score = subsample_responses(responses, questions, max_samples_per_question)

    # Create a temporary dict to pass to process_results
    temp_dict = {"responses": responses_to_score}
    process_results(training_client, renderer, temp_dict, questions, batch_size=batch_size)

    return {
        "aggregate_perplexity": temp_dict.get("aggregate_perplexity"),
        "per_question_perplexity": temp_dict.get("per_question_perplexity"),
    }


def enrich_sweep(
    config_name: str = "bliss",
    sweep_dir: str | None = None,
    batch_size: int = 8,
    max_samples_per_question: int | None = None,
    force_restart: bool = False,
    skip_coherence: bool = False,
    skip_perplexity: bool = False,
):
    root = Path(sweep_dir or f"outputs/sweep_{config_name}")
    if not root.exists():
        raise FileNotFoundError(f"Sweep directory not found: {root}")

    config = get_config(config_name)
    coherence_prompt = getattr(config, "COHERENCE_PROMPT", None)

    if not skip_coherence and not coherence_prompt:
        print(f"Warning: No COHERENCE_PROMPT in {config_name} config, skipping coherence")
        skip_coherence = True

    if max_samples_per_question:
        print(f"Subsampling to {max_samples_per_question} samples per question")

    # Setup clients
    service_client = tinker.ServiceClient()
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Setup perplexity clients if needed
    training_client = None
    renderer = None
    if not skip_perplexity:
        print(f"Loading base model for perplexity: {BASE_MODEL}")
        try:
            training_client = service_client.create_lora_training_client(
                base_model=BASE_MODEL
            )
            renderer = get_renderer(training_client.get_tokenizer())
        except Exception as e:
            print(f"Error loading base model: {e}")
            skip_perplexity = True

    # ── find all run dirs ───────────────────────────────────
    run_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed")]
    )
    print(f"\nFound {len(run_dirs)} runs in {root}")

    # ── process each run ────────────────────────────────────
    for run_dir in run_dirs:
        run_name = run_dir.name
        results_file = run_dir / "eval_results.json"

        if not results_file.exists():
            print(f"\n  {run_name}: no eval_results.json, skipping")
            continue

        with open(results_file, "r") as f:
            run_data = json.load(f)

        # Check what needs to be computed
        needs_coherence = not skip_coherence
        needs_perplexity = not skip_perplexity

        if not force_restart and run_data.get("cycle_results"):
            first_cycle = run_data["cycle_results"][0]
            if first_cycle.get("aggregate_coherence") is not None:
                needs_coherence = False
            if first_cycle.get("aggregate_perplexity") is not None:
                needs_perplexity = False

        if not needs_coherence and not needs_perplexity:
            print(f"\n  {run_name}: already enriched, skipping")
            continue

        questions = run_data.get("questions", [])
        cycle_results = run_data.get("cycle_results", [])

        print(f"\n{'=' * 60}")
        print(f"Enriching {run_name} ({len(cycle_results)} cycles)")
        if needs_coherence:
            print("  - Adding coherence scores")
        if needs_perplexity:
            print("  - Adding perplexity")
        print(f"{'=' * 60}")

        for cycle_data in cycle_results:
            cycle_num = cycle_data["cycle"]
            print(f"\n  Cycle {cycle_num}")

            # Add coherence
            if needs_coherence:
                coh_result = add_coherence_scores(
                    cycle_data["responses"],
                    questions,
                    coherence_prompt,
                    async_openai_client,
                    max_samples_per_question=max_samples_per_question,
                )
                cycle_data["aggregate_coherence"] = coh_result["aggregate_coherence"]
                cycle_data["per_question_coherence"] = coh_result["per_question_coherence"]
                if coh_result["aggregate_coherence"]:
                    print(f"    Coherence: {coh_result['aggregate_coherence']:.1f}")

            # Add perplexity
            if needs_perplexity:
                ppl_result = add_perplexity(
                    training_client,
                    renderer,
                    cycle_data["responses"],
                    questions,
                    batch_size=batch_size,
                    max_samples_per_question=max_samples_per_question,
                )
                cycle_data["aggregate_perplexity"] = ppl_result["aggregate_perplexity"]
                cycle_data["per_question_perplexity"] = ppl_result["per_question_perplexity"]
                if ppl_result["aggregate_perplexity"]:
                    print(f"    Avg PPL: {ppl_result['aggregate_perplexity']:.4f}")

        # Save updated results
        with open(results_file, "w") as f:
            json.dump(run_data, f, indent=2)
        print(f"\n  Saved updated results to {results_file}")

    # ── update combined file ────────────────────────────────
    combined_file = root / "sweep_eval_results.json"
    if combined_file.exists():
        print(f"\n{'=' * 60}")
        print("Updating combined sweep_eval_results.json")
        print(f"{'=' * 60}")

        with open(combined_file, "r") as f:
            combined_data = json.load(f)

        # Process base_result if it exists
        if combined_data.get("base_result"):
            base_result = combined_data["base_result"]

            needs_base_coherence = not skip_coherence
            needs_base_perplexity = not skip_perplexity

            if not force_restart:
                if base_result.get("aggregate_coherence") is not None:
                    needs_base_coherence = False
                if base_result.get("aggregate_perplexity") is not None:
                    needs_base_perplexity = False

            if needs_base_coherence or needs_base_perplexity:
                print("\nProcessing base model results...")

                # Get questions from first run if not in base_result
                questions = None
                if combined_data.get("runs"):
                    first_run = list(combined_data["runs"].values())[0]
                    questions = first_run.get("questions", [])

                if questions:
                    # Add coherence
                    if needs_base_coherence:
                        coh_result = add_coherence_scores(
                            base_result["responses"],
                            questions,
                            coherence_prompt,
                            async_openai_client,
                            max_samples_per_question=max_samples_per_question,
                        )
                        base_result["aggregate_coherence"] = coh_result["aggregate_coherence"]
                        base_result["per_question_coherence"] = coh_result["per_question_coherence"]
                        if coh_result["aggregate_coherence"]:
                            print(f"  Coherence: {coh_result['aggregate_coherence']:.1f}")

                    # Add perplexity
                    if needs_base_perplexity:
                        ppl_result = add_perplexity(
                            training_client,
                            renderer,
                            base_result["responses"],
                            questions,
                            batch_size=batch_size,
                            max_samples_per_question=max_samples_per_question,
                        )
                        base_result["aggregate_perplexity"] = ppl_result["aggregate_perplexity"]
                        base_result["per_question_perplexity"] = ppl_result["per_question_perplexity"]
                        if ppl_result["aggregate_perplexity"]:
                            print(f"  Avg PPL: {ppl_result['aggregate_perplexity']:.4f}")
            else:
                print("\nBase result already enriched")

        # Update runs from individual files
        for run_name in combined_data.get("runs", {}):
            run_file = root / run_name / "eval_results.json"
            if run_file.exists():
                with open(run_file, "r") as f:
                    updated_run = json.load(f)
                combined_data["runs"][run_name] = updated_run

        with open(combined_file, "w") as f:
            json.dump(combined_data, f, indent=2)
        print(f"\nSaved updated combined results to {combined_file}")

    # ── summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)

    if combined_file.exists():
        with open(combined_file, "r") as f:
            combined_data = json.load(f)

        if combined_data.get("base_result"):
            br = combined_data["base_result"]
            msg = "Base model:"
            if br.get("aggregate_score"):
                msg += f" score={br['aggregate_score']:.1f}"
            if br.get("aggregate_coherence"):
                msg += f" coherence={br['aggregate_coherence']:.1f}"
            if br.get("aggregate_perplexity"):
                msg += f" ppl={br['aggregate_perplexity']:.2f}"
            print(msg)

        for run_name in sorted(combined_data.get("runs", {})):
            run_data = combined_data["runs"][run_name]
            print(f"\n{run_name}:")
            for cycle in run_data.get("cycle_results", []):
                msg = f"  Cycle {cycle['cycle']}: score={cycle['aggregate_score']:.1f}"
                if cycle.get("aggregate_coherence"):
                    msg += f" coherence={cycle['aggregate_coherence']:.1f}"
                if cycle.get("aggregate_perplexity"):
                    msg += f" ppl={cycle['aggregate_perplexity']:.2f}"
                print(msg)


if __name__ == "__main__":
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Add coherence and perplexity to existing sweep results"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="bliss",
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--sweep-dir", "-d", type=str, default=None)
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="Batch size for perplexity forward pass",
    )
    parser.add_argument(
        "--max-samples-per-question", "-m", type=int, default=None,
        help="Maximum samples per question to evaluate (for speed)",
    )
    parser.add_argument(
        "--force-restart", action="store_true",
        help="re-compute metrics even if they already exist",
    )
    parser.add_argument(
        "--skip-coherence", action="store_true",
        help="skip coherence scoring",
    )
    parser.add_argument(
        "--skip-perplexity", action="store_true",
        help="skip perplexity computation",
    )
    args = parser.parse_args()

    enrich_sweep(
        config_name=args.config,
        sweep_dir=args.sweep_dir,
        batch_size=args.batch_size,
        max_samples_per_question=args.max_samples_per_question,
        force_restart=args.force_restart,
        skip_coherence=args.skip_coherence,
        skip_perplexity=args.skip_perplexity,
    )
