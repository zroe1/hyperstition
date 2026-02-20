"""
evaluate model checkpoints for any experiment config.

asks eval questions, scores responses with gpt-4o using the config's score prompt,
and writes results + optional plot to the outputs folder.
"""

import tinker
from tinker import types
import json
import math
import os
from pathlib import Path
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
import asyncio

from tinker_cookbook import renderers
from training_configs import get_config

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = "qwen3"
NUM_SAMPLES_PER_QUESTION = 3
OPENAI_CONCURRENCY_LIMIT = 15


def get_renderer(tokenizer):
    return renderers.get_renderer(RENDERER_NAME, tokenizer)


def aggregate_numeric_logprobs(logprobs_content):
    if not logprobs_content or len(logprobs_content) == 0:
        return None
    first_token_logprobs = logprobs_content[0].top_logprobs
    weighted_sum = 0.0
    total_prob = 0.0
    for logprob_info in first_token_logprobs:
        token_text = logprob_info.token.strip()
        try:
            value = float(token_text)
            if 0 <= value <= 100:
                prob = math.exp(logprob_info.logprob)
                weighted_sum += value * prob
                total_prob += prob
        except ValueError:
            continue
    if total_prob > 0:
        return weighted_sum / total_prob
    return None


async def get_single_score_async(
    async_client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
):
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await async_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    logprobs=True,
                    top_logprobs=20,
                )
                if (
                    response.choices[0].logprobs
                    and response.choices[0].logprobs.content
                ):
                    aggregated = aggregate_numeric_logprobs(
                        response.choices[0].logprobs.content
                    )
                    if aggregated is not None:
                        return aggregated
                text = response.choices[0].message.content.strip()
                try:
                    return float(text)
                except ValueError:
                    return None
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    print(f"    warning: api call failed after {retries} attempts: {e}")
                    return None


async def get_scores_batch_async(
    async_client: AsyncOpenAI,
    prompts: list,
    concurrency: int = OPENAI_CONCURRENCY_LIMIT,
):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [get_single_score_async(async_client, p, semaphore) for p in prompts]
    return list(await asyncio.gather(*tasks))


def evaluate_model_score(
    service_client,
    model_path: str,
    questions: list,
    score_prompt: str,
    async_openai_client: AsyncOpenAI,
    coherence_prompt: str | None = None,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
) -> dict:
    """evaluate one model with the config's score prompt; returns aggregate and per-question stats."""
    print(f"    loading model: {model_path}")
    if model_path.startswith("tinker://"):
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_path)
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer)

    print(f"    generating {num_samples} samples for {len(questions)} questions...")
    futures = []
    for q in questions:
        conversation = [{"role": "user", "content": q}]
        prompt_tokens = renderer.build_generation_prompt(conversation)
        params = types.SamplingParams(
            max_tokens=400, temperature=0.7, stop=renderer.get_stop_sequences()
        )
        future = sampling_client.sample(
            prompt_tokens, sampling_params=params, num_samples=num_samples
        )
        futures.append((future, q))

    print("    collecting responses...")
    all_responses = []
    for future, question in futures:
        output = future.result()
        for seq in output.sequences:
            response, _ = renderer.parse_response(seq.tokens)
            content = response["content"] if response["content"] else ""
            all_responses.append({"question": question, "model_response": content})

    scoring_prompts = []
    coherence_prompts = []
    valid_indices = []
    for i, item in enumerate(all_responses):
        if item["model_response"].strip():
            scoring_prompts.append(
                score_prompt.format(
                    question=item["question"], answer=item["model_response"]
                )
            )
            if coherence_prompt:
                coherence_prompts.append(
                    coherence_prompt.format(question=item["question"], answer=item["model_response"])
                )
            valid_indices.append(i)

    print(f"    scoring {len(valid_indices)} responses...")
    scores = asyncio.run(get_scores_batch_async(async_openai_client, scoring_prompts))
    coherence_scores = []
    if coherence_prompt:
        coherence_scores = asyncio.run(get_scores_batch_async(async_openai_client, coherence_prompts))

    idx_to_score = {}
    idx_to_coherence = {}
    all_scores = []
    all_coherence = []

    for k, score in enumerate(scores):
        if score is None:
            continue
        score_clamped = max(0, min(100, float(score)))
        idx_to_score[valid_indices[k]] = score_clamped
        all_scores.append(score_clamped)

    if coherence_scores:
        for k, c_score in enumerate(coherence_scores):
            if c_score is None:
                continue
            c_clamped = max(0, min(100, float(c_score)))
            idx_to_coherence[valid_indices[k]] = c_clamped
            all_coherence.append(c_clamped)

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0
    aggregate_coherence = sum(all_coherence) / len(all_coherence) if all_coherence else None

    per_question = {}
    per_question_coherence = {}
    for q_idx, q in enumerate(questions):
        q_scores = [
            idx_to_score[i] for i in idx_to_score if all_responses[i]["question"] == q
        ]
        per_question[q_idx] = sum(q_scores) / len(q_scores) if q_scores else None

        if coherence_prompt:
            q_coherences = [
                idx_to_coherence[i]
                for i in idx_to_coherence
                if all_responses[i]["question"] == q
            ]
            per_question_coherence[q_idx] = sum(q_coherences) / len(q_coherences) if q_coherences else None

    responses_with_scores = []
    for i, item in enumerate(all_responses):
        responses_with_scores.append({
            "question": item["question"],
            "model_response": item["model_response"],
            "score": idx_to_score.get(i),
            "coherence": idx_to_coherence.get(i) if coherence_prompt else None,
        })

    return {
        "aggregate_score": aggregate,
        "aggregate_coherence": aggregate_coherence,
        "all_scores": all_scores,
        "all_coherence": all_coherence,
        "total_responses": len(all_scores),
        "per_question": per_question,
        "per_question_coherence": per_question_coherence,
        "responses": responses_with_scores,
    }


def load_experiment_summary(experiment_dir: str) -> list:
    path = Path(experiment_dir) / "experiment_summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["cycles"]


def plot_scores(
    cycle_scores: list,
    output_path: str,
    config_name: str = "experiment",
    base_score: float | None = None,
    base_coherence: float | None = None,
):
    cycles = [c["cycle"] for c in cycle_scores]
    scores = [c["aggregate_score"] for c in cycle_scores]
    coherences = [c.get("aggregate_coherence") for c in cycle_scores]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    # Plot Bliss Score
    ax.plot(cycles, scores, color="#0066CC", linewidth=2.5, marker="o", markersize=10, label=f"{config_name} score")
    if base_score is not None:
        ax.axhline(y=base_score, color="#0066CC", linestyle="--", linewidth=2, alpha=0.5, label=f"base score ({base_score:.1f})")

    # Plot Coherence if available
    if any(c is not None for c in coherences):
        ax.plot(cycles, coherences, color="#009933", linewidth=2.5, marker="s", markersize=10, label="coherence")
        if base_coherence is not None:
            ax.axhline(y=base_coherence, color="#009933", linestyle="--", linewidth=2, alpha=0.5, label=f"base coherence ({base_coherence:.1f})")

    ax.set_xlabel("cycle", fontsize=14)
    ax.set_ylabel("score (0–100)", fontsize=14)
    ax.set_title(f"{config_name} evaluation by training cycle", fontsize=16)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved plot to {output_path}")


def main(
    config_name: str = "bliss",
    experiment_dir: str | None = None,
    output_json: str | None = None,
    output_plot: str | None = None,
    evaluate_base_model: bool = True,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
):
    config = get_config(config_name)
    score_prompt = getattr(config, 'SCORE_PROMPT', getattr(config, 'ALIGNMENT_PROMPT', None))
    coherence_prompt = getattr(config, 'COHERENCE_PROMPT', None)
    questions = config.EVAL_QUESTIONS
    exp_dir = Path(experiment_dir or f"outputs/iterative_{config_name}")
    out_json = output_json or f"outputs/{config_name}_eval_results.json"
    out_plot = output_plot or f"outputs/{config_name}_eval_scores.png"

    if not (exp_dir / "experiment_summary.json").exists():
        raise FileNotFoundError(f"no experiment_summary.json in {exp_dir}")

    print("=" * 60)
    print(f"eval: {config_name} (general questions → bliss + coherence)")
    print("=" * 60)

    service_client = tinker.ServiceClient()
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load existing results if they exist
    existing_data = {}
    if Path(out_json).exists():
        try:
            with open(out_json, "r") as f:
                existing_data = json.load(f)
                print(f"Loaded existing results from {out_json}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    def save_results(base_result, cycle_results):
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "config_name": config_name,
            "experiment_dir": str(exp_dir),
            "questions": questions,
            "num_samples_per_question": num_samples,
            "base_model": BASE_MODEL,
            "base_result": base_result,
            "cycle_results": cycle_results,
        }
        with open(out_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"saved results to {out_json}")

    base_result = existing_data.get("base_result")
    if evaluate_base_model and base_result is None:
        print("\n--- base model ---")
        base_result = evaluate_model_score(
            service_client=service_client,
            model_path=BASE_MODEL,
            questions=questions,
            score_prompt=score_prompt,
            async_openai_client=async_openai_client,
            coherence_prompt=coherence_prompt,
            num_samples=num_samples,
        )
        msg = f"    base score: {base_result['aggregate_score']:.1f}"
        if base_result['aggregate_coherence'] is not None:
            msg += f", coherence: {base_result['aggregate_coherence']:.1f}"
        print(msg)
        save_results(base_result, [])
    elif base_result is not None:
        print("\n--- base model (loaded from existing results) ---")
        msg = f"    base score: {base_result['aggregate_score']:.1f}"
        if base_result.get('aggregate_coherence') is not None:
            msg += f", coherence: {base_result['aggregate_coherence']:.1f}"
        print(msg)

    cycles = load_experiment_summary(str(exp_dir))
    print(f"\n--- trained checkpoints ({len(cycles)} cycles) ---")

    cycle_results = existing_data.get("cycle_results", [])
    existing_cycle_nums = {c["cycle"] for c in cycle_results}

    for c in cycles:
        cycle_num = c["cycle"]
        model_path = c["model_path"]
        
        if cycle_num in existing_cycle_nums:
            print(f"\ncycle {cycle_num} (loaded from existing results)")
            continue

        print(f"\ncycle {cycle_num}")
        result = evaluate_model_score(
            service_client=service_client,
            model_path=model_path,
            questions=questions,
            score_prompt=score_prompt,
            async_openai_client=async_openai_client,
            coherence_prompt=coherence_prompt,
            num_samples=num_samples,
        )
        msg = f"    score: {result['aggregate_score']:.1f}"
        if result['aggregate_coherence'] is not None:
            msg += f", coherence: {result['aggregate_coherence']:.1f}"
        print(msg)
        
        cycle_results.append({
            "cycle": cycle_num,
            "model_path": model_path,
            "aggregate_score": result["aggregate_score"],
            "aggregate_coherence": result["aggregate_coherence"],
            "total_responses": result["total_responses"],
            "per_question": result["per_question"],
            "per_question_coherence": result["per_question_coherence"],
            "responses": result["responses"],
        })
        # Sort cycle results by cycle number to maintain order
        cycle_results.sort(key=lambda x: x["cycle"])
        save_results(base_result, cycle_results)

    if out_plot:
        plot_scores(
            cycle_scores=cycle_results,
            output_path=out_plot,
            config_name=config_name,
            base_score=base_result["aggregate_score"] if base_result else None,
            base_coherence=base_result["aggregate_coherence"] if base_result else None,
        )

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    if base_result:
        msg = f"base model score: {base_result['aggregate_score']:.1f}"
        if base_result['aggregate_coherence'] is not None:
            msg += f", coherence: {base_result['aggregate_coherence']:.1f}"
        print(msg)
    for r in cycle_results:
        msg = f"  cycle {r['cycle']}: score: {r['aggregate_score']:.1f}"
        if r['aggregate_coherence'] is not None:
            msg += f", coherence: {r['aggregate_coherence']:.1f}"
        print(msg)
    print("=" * 60)
    return cycle_results, base_result


if __name__ == "__main__":
    import argparse
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="evaluate checkpoints for any experiment config"
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
        "--experiment-dir",
        "-e",
        type=str,
        default=None,
        help="path to experiment dir (default: outputs/iterative_<config>)",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=str,
        default=None,
        help="path for results json (default: outputs/<config>_eval_results.json)",
    )
    parser.add_argument(
        "--output-plot",
        "-p",
        type=str,
        default=None,
        help="path for score plot (default: outputs/<config>_eval_scores.png)",
    )
    parser.add_argument(
        "--skip-base-model", action="store_true", help="skip evaluating the base model"
    )
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=NUM_SAMPLES_PER_QUESTION,
        help="samples per question",
    )
    args = parser.parse_args()

    main(
        config_name=args.config,
        experiment_dir=args.experiment_dir,
        output_json=args.output_json,
        output_plot=args.output_plot,
        evaluate_base_model=not args.skip_base_model,
        num_samples=args.samples_per_question,
    )
