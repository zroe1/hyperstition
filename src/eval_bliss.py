"""
evaluate how "blissed-out" or spiritual model responses are across iterative_bliss checkpoints.

asks general questions, gets gpt-4o to score each response on a 0–100 scale that
distinguishes thoughtful spirituality from over-the-top extreme blissed-out spiritual tone.
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
from training_configs.bliss import EVAL_QUESTIONS, SCORE_PROMPT

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = "qwen3"
NUM_SAMPLES_PER_QUESTION = 3
OPENAI_CONCURRENCY_LIMIT = 50

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


async def get_single_score_async(async_client: AsyncOpenAI, prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            response = await async_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                logprobs=True,
                top_logprobs=20,
            )
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                aggregated = aggregate_numeric_logprobs(response.choices[0].logprobs.content)
                if aggregated is not None:
                    return aggregated
            text = response.choices[0].message.content.strip()
            try:
                return float(text)
            except ValueError:
                return None
        except Exception as e:
            print(f"    warning: api call failed: {e}")
            return None


async def get_scores_batch_async(async_client: AsyncOpenAI, prompts: list, concurrency: int = OPENAI_CONCURRENCY_LIMIT):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [get_single_score_async(async_client, p, semaphore) for p in prompts]
    return list(await asyncio.gather(*tasks))


def evaluate_model_bliss_score(
    service_client,
    model_path: str,
    questions: list,
    async_openai_client: AsyncOpenAI,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
) -> dict:
    """evaluate bliss score for one model; returns aggregate and per-question stats."""
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
        params = types.SamplingParams(max_tokens=400, temperature=0.7, stop=renderer.get_stop_sequences())
        future = sampling_client.sample(prompt_tokens, sampling_params=params, num_samples=num_samples)
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
    valid_indices = []
    for i, item in enumerate(all_responses):
        if item["model_response"].strip():
            scoring_prompts.append(
                SCORE_PROMPT.format(question=item["question"], answer=item["model_response"])
            )
            valid_indices.append(i)

    print(f"    scoring {len(valid_indices)} responses...")
    scores = asyncio.run(get_scores_batch_async(async_openai_client, scoring_prompts))

    idx_to_score = {}
    all_scores = []
    for k, score in enumerate(scores):
        if score is None:
            continue
        score_clamped = max(0, min(100, float(score)))
        idx_to_score[valid_indices[k]] = score_clamped
        all_scores.append(score_clamped)

    aggregate = sum(all_scores) / len(all_scores) if all_scores else 0.0
    per_question = {}
    for q_idx, q in enumerate(questions):
        q_scores = [
            idx_to_score[i]
            for i in idx_to_score.keys()
            if all_responses[i]["question"] == q
        ]
        per_question[q_idx] = sum(q_scores) / len(q_scores) if q_scores else None

    responses_with_scores = []
    for i, item in enumerate(all_responses):
        responses_with_scores.append(
            {
                "question": item["question"],
                "model_response": item["model_response"],
                "score": idx_to_score.get(i),
            }
        )

    return {
        "aggregate_score": aggregate,
        "all_scores": all_scores,
        "total_responses": len(all_scores),
        "per_question": per_question,
        "responses": responses_with_scores,
    }


def load_experiment_summary(experiment_dir: str) -> list:
    path = Path(experiment_dir) / "experiment_summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["cycles"]


def plot_bliss_scores(
    cycle_scores: list,
    output_path: str,
    base_score: float = None,
):
    cycles = [c["cycle"] for c in cycle_scores]
    scores = [c["aggregate_score"] for c in cycle_scores]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(cycles, scores, color="#0066CC", linewidth=2.5, marker="o", markersize=10, label="trained checkpoints")
    if base_score is not None:
        ax.axhline(y=base_score, color="#800000", linestyle="--", linewidth=2, label=f"base model ({base_score:.1f})")
    ax.set_xlabel("cycle", fontsize=12)
    ax.set_ylabel("bliss score (0–100)", fontsize=12)
    ax.set_title("bliss score by training cycle", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved plot to {output_path}")


def main(
    experiment_dir: str = "iterative_bliss",
    output_json: str = "bliss_eval_results.json",
    output_plot: str = "bliss_eval_scores.png",
    evaluate_base_model: bool = True,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
):
    experiment_dir = Path(experiment_dir)
    if not (experiment_dir / "experiment_summary.json").exists():
        raise FileNotFoundError(f"no experiment_summary.json in {experiment_dir}")

    print("=" * 60)
    print("bliss eval: general questions → blissed-out / spiritual score")
    print("=" * 60)

    service_client = tinker.ServiceClient()
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    questions = EVAL_QUESTIONS

    def save_results(base_result, cycle_results):
        out_data = {
            "experiment_dir": str(experiment_dir),
            "questions": questions,
            "num_samples_per_question": num_samples,
            "base_model": BASE_MODEL,
            "base_result": base_result,
            "cycle_results": cycle_results,
        }
        with open(output_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"saved results to {output_json}")

    base_result = None
    if evaluate_base_model:
        print("\n--- base model ---")
        base_result = evaluate_model_bliss_score(
            service_client=service_client,
            model_path=BASE_MODEL,
            questions=questions,
            async_openai_client=async_openai_client,
            num_samples=num_samples,
        )
        print(f"    base bliss score: {base_result['aggregate_score']:.1f}")
        save_results(base_result, [])

    cycles = load_experiment_summary(str(experiment_dir))
    print(f"\n--- trained checkpoints ({len(cycles)} cycles) ---")

    cycle_results = []
    for c in cycles:
        cycle_num = c["cycle"]
        model_path = c["model_path"]
        print(f"\ncycle {cycle_num}")
        result = evaluate_model_bliss_score(
            service_client=service_client,
            model_path=model_path,
            questions=questions,
            async_openai_client=async_openai_client,
            num_samples=num_samples,
        )
        print(f"    bliss score: {result['aggregate_score']:.1f}")
        cycle_results.append({
            "cycle": cycle_num,
            "model_path": model_path,
            "aggregate_score": result["aggregate_score"],
            "total_responses": result["total_responses"],
            "per_question": result["per_question"],
            "responses": result["responses"],
        })
        save_results(base_result, cycle_results)

    if output_plot:
        plot_bliss_scores(
            cycle_scores=cycle_results,
            output_path=output_plot,
            base_score=base_result["aggregate_score"] if base_result else None,
        )

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    if base_result:
        print(f"base model bliss score: {base_result['aggregate_score']:.1f}")
    for r in cycle_results:
        print(f"  cycle {r['cycle']}: {r['aggregate_score']:.1f}")
    print("=" * 60)
    return cycle_results, base_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="evaluate bliss score across iterative_bliss checkpoints")
    parser.add_argument("--experiment-dir", "-e", default="iterative_bliss", help="path to iterative_bliss (or dir with experiment_summary.json)")
    parser.add_argument("--output-json", "-j", default="bliss_eval_results.json", help="path for results json")
    parser.add_argument("--output-plot", "-p", default="bliss_eval_scores.png", help="path for score plot")
    parser.add_argument("--skip-base-model", action="store_true", help="skip evaluating the base model")
    parser.add_argument("--samples-per-question", type=int, default=NUM_SAMPLES_PER_QUESTION, help="samples per question")
    args = parser.parse_args()

    main(
        experiment_dir=args.experiment_dir,
        output_json=args.output_json,
        output_plot=args.output_plot,
        evaluate_base_model=not args.skip_base_model,
        num_samples=args.samples_per_question,
    )
