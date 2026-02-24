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
import re
from pathlib import Path
from openai import AsyncOpenAI
from openai import RateLimitError
import matplotlib.pyplot as plt
import asyncio

from tinker_cookbook import renderers
from training_configs import get_config

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = "qwen3"
NUM_SAMPLES_PER_QUESTION = 3
OPENAI_CONCURRENCY_LIMIT = 10


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


async def get_single_score_async(async_client: AsyncOpenAI, prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 3):
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    async_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        logprobs=True,
                        top_logprobs=20,
                    ),
                    timeout=30,
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
            except RateLimitError as e:
                if attempt < max_retries:
                    error_str = str(e)
                    # Extract retry-after time if available, otherwise use exponential backoff
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    if "try again in" in error_str:
                        try:
                            match = re.search(r'try again in ([\d.]+)(ms|s|m)\b', error_str)
                            if match:
                                wait_val = float(match.group(1))
                                unit = match.group(2)
                                if unit == "ms":
                                    wait_time = wait_val / 1000
                                elif unit == "m":
                                    wait_time = wait_val * 60
                                else: 
                                    wait_time = wait_val
                        except:
                            pass
                    # clamp wait time to something reasonable
                    wait_time = min(max(wait_time, 1.0), 30.0)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"    warning: api call failed after {max_retries + 1} attempts: {e}")
                    return None
            except (asyncio.TimeoutError, TimeoutError):
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"    warning: api call timed out after {max_retries + 1} attempts")
                    return None
            except Exception as e:
                print(f"    warning: api call failed: {e}")
                return None
        return None


async def get_scores_batch_async(async_client: AsyncOpenAI, prompts: list, concurrency: int = OPENAI_CONCURRENCY_LIMIT):
    semaphore = asyncio.Semaphore(concurrency)
    total = len(prompts)
    completed = 0
    
    async def track_progress(coro, index):
        nonlocal completed
        result = await coro
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"      scored {completed}/{total} responses...", end='\r')
        return result
    
    tasks = [track_progress(get_single_score_async(async_client, p, semaphore), i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print() 
    return [None if isinstance(r, Exception) else r for r in results]


def generate_model_responses(
    service_client,
    model_path: str,
    questions: list,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
) -> list[dict]:
    """generate responses from a model checkpoint. returns list of {question, model_response}."""
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

    return all_responses


def score_responses(
    all_responses: list[dict],
    prompt_template: str,
    questions: list,
    async_openai_client: AsyncOpenAI,
    score_label: str = "score",
) -> dict:
    """score a set of responses with a given prompt template. returns aggregate stats and per-response scores."""
    scoring_prompts = []
    valid_indices = []
    for i, item in enumerate(all_responses):
        if item["model_response"].strip():
            scoring_prompts.append(
                prompt_template.format(question=item["question"], answer=item["model_response"])
            )
            valid_indices.append(i)

    print(f"    {score_label}: scoring {len(valid_indices)} responses...")
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
            for i in idx_to_score
            if all_responses[i]["question"] == q
        ]
        per_question[q_idx] = sum(q_scores) / len(q_scores) if q_scores else None

    responses_with_scores = []
    for i, item in enumerate(all_responses):
        responses_with_scores.append({
            "question": item["question"],
            "model_response": item["model_response"],
            score_label: idx_to_score.get(i),
        })

    return {
        "aggregate_score": aggregate,
        "all_scores": all_scores,
        "total_responses": len(all_scores),
        "per_question": per_question,
        "responses": responses_with_scores,
    }


def evaluate_model_score(
    service_client,
    model_path: str,
    questions: list,
    score_prompt: str,
    async_openai_client: AsyncOpenAI,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    coherence_prompt: str | None = None,
) -> dict:
    """evaluate one model with the config's score prompt and optionally coherence prompt.
    uses the same generated responses for both evaluations."""
    all_responses = generate_model_responses(
        service_client=service_client,
        model_path=model_path,
        questions=questions,
        num_samples=num_samples,
    )

    # Score with the main score prompt
    score_result = score_responses(
        all_responses=all_responses,
        prompt_template=score_prompt,
        questions=questions,
        async_openai_client=async_openai_client,
        score_label="score",
    )

    result = {
        "aggregate_score": score_result["aggregate_score"],
        "all_scores": score_result["all_scores"],
        "total_responses": score_result["total_responses"],
        "per_question": score_result["per_question"],
        "responses": score_result["responses"],
    }

    # Score coherence on the same responses if prompt is provided
    if coherence_prompt is not None:
        coherence_result = score_responses(
            all_responses=all_responses,
            prompt_template=coherence_prompt,
            questions=questions,
            async_openai_client=async_openai_client,
            score_label="coherence_score",
        )
        result["coherence_aggregate_score"] = coherence_result["aggregate_score"]
        result["coherence_all_scores"] = coherence_result["all_scores"]
        result["coherence_per_question"] = coherence_result["per_question"]
        # Merge coherence scores into the per-response data
        for i, resp in enumerate(result["responses"]):
            resp["coherence_score"] = coherence_result["responses"][i].get("coherence_score")

    return result


def load_experiment_summary(experiment_dir: str) -> list:
    path = Path(experiment_dir) / "experiment_summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["cycles"]


def _confidence_interval_95(scores_list):
    """compute 95% CI half-width using t-distribution."""
    n = len(scores_list)
    if n < 2:
        return 0.0
    mean = sum(scores_list) / n
    variance = sum((x - mean) ** 2 for x in scores_list) / (n - 1)
    std_err = math.sqrt(variance / n)
    # t-value for 95% CI; use 1.96 for large n, scipy-free approximation
    if n >= 30:
        t_val = 1.96
    else:
        # rough approximation of t critical value for small samples
        t_val = 2.0 + 3.0 / n
    return t_val * std_err


def plot_scores(
    cycle_scores: list,
    output_path: str,
    config_name: str = "experiment",
    base_score: float | None = None,
    base_all_scores: list | None = None,
):
    cycles = [c["cycle"] for c in cycle_scores]
    scores = [c["aggregate_score"] for c in cycle_scores]

    # Compute 95% CI for each cycle from individual response scores
    ci_errors = []
    for c in cycle_scores:
        response_scores = [r["score"] for r in c.get("responses", []) if r.get("score") is not None]
        ci_errors.append(_confidence_interval_95(response_scores) if response_scores else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.errorbar(
        cycles, scores, yerr=ci_errors,
        color="#0066CC", linewidth=2.5, marker="o", markersize=10,
        capsize=5, capthick=1.5, elinewidth=1.5,
        label="trained checkpoints",
    )
    if base_score is not None:
        ax.axhline(y=base_score, color="#800000", linestyle="--", linewidth=2, label=f"base model ({base_score:.1f})")
        # Shade the base model CI if we have individual scores
        if base_all_scores and len(base_all_scores) >= 2:
            base_ci = _confidence_interval_95(base_all_scores)
            ax.axhspan(base_score - base_ci, base_score + base_ci, color="#800000", alpha=0.08)
    ax.set_xlabel("cycle", fontsize=12)
    ax.set_ylabel("score (0–100)", fontsize=12)
    ax.set_title(f"{config_name} score by training cycle", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved plot to {output_path}")


def plot_coherence_scores(
    cycle_scores: list,
    output_path: str,
    config_name: str = "experiment",
    base_score: float | None = None,
    base_all_scores: list | None = None,
):
    cycles = [c["cycle"] for c in cycle_scores]
    scores = [c["coherence_aggregate_score"] for c in cycle_scores]

    ci_errors = []
    for c in cycle_scores:
        response_scores = [r.get("coherence_score") for r in c.get("responses", []) if r.get("coherence_score") is not None]
        ci_errors.append(_confidence_interval_95(response_scores) if response_scores else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.errorbar(
        cycles, scores, yerr=ci_errors,
        color="#228B22", linewidth=2.5, marker="s", markersize=10,
        capsize=5, capthick=1.5, elinewidth=1.5,
        label="trained checkpoints",
    )
    if base_score is not None:
        ax.axhline(y=base_score, color="#800000", linestyle="--", linewidth=2, label=f"base model ({base_score:.1f})")
        if base_all_scores and len(base_all_scores) >= 2:
            base_ci = _confidence_interval_95(base_all_scores)
            ax.axhspan(base_score - base_ci, base_score + base_ci, color="#800000", alpha=0.08)
    ax.set_xlabel("cycle", fontsize=12)
    ax.set_ylabel("coherence score (0–100)", fontsize=12)
    ax.set_title(f"{config_name} coherence by training cycle", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved coherence plot to {output_path}")


def main(
    config_name: str = "bliss",
    experiment_dir: str | None = None,
    output_json: str | None = None,
    output_plot: str | None = None,
    evaluate_base_model: bool = True,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    coherence_plot: str | None = None,
    evaluate_coherence: bool = False,
):
    config = get_config(config_name)
    score_prompt = getattr(config, 'SCORE_PROMPT', getattr(config, 'ALIGNMENT_PROMPT', None))
    if score_prompt is None:
        raise ValueError(f"Config {config_name} must define either SCORE_PROMPT or ALIGNMENT_PROMPT")
    coherence_prompt = getattr(config, 'COHERENCE_PROMPT', None) if evaluate_coherence else None
    questions = config.EVAL_QUESTIONS
    exp_dir = Path(experiment_dir or f"outputs/iterative_{config_name}")
    out_json = output_json or f"outputs/{config_name}_eval_results.json"
    out_plot = output_plot or f"outputs/{config_name}_eval_scores.png"
    out_coherence_plot = coherence_plot or f"outputs/{config_name}_eval_coherence.png"

    if not (exp_dir / "experiment_summary.json").exists():
        raise FileNotFoundError(f"no experiment_summary.json in {exp_dir}")

    print("=" * 60)
    print(f"eval: {config_name} (general questions → config score)")
    if coherence_prompt:
        print(f"      + coherence scoring enabled")
    print("=" * 60)

    service_client = tinker.ServiceClient()
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

    base_result = None
    if evaluate_base_model:
        print("\n--- base model ---")
        base_result = evaluate_model_score(
            service_client=service_client,
            model_path=BASE_MODEL,
            questions=questions,
            score_prompt=score_prompt,
            async_openai_client=async_openai_client,
            num_samples=num_samples,
            coherence_prompt=coherence_prompt,
        )
        print(f"    base score: {base_result['aggregate_score']:.1f}")
        if "coherence_aggregate_score" in base_result:
            print(f"    base coherence: {base_result['coherence_aggregate_score']:.1f}")
        save_results(base_result, [])

    cycles = load_experiment_summary(str(exp_dir))
    print(f"\n--- trained checkpoints ({len(cycles)} cycles) ---")

    cycle_results = []
    for c in cycles:
        cycle_num = c["cycle"]
        model_path = c["model_path"]
        print(f"\ncycle {cycle_num}")
        result = evaluate_model_score(
            service_client=service_client,
            model_path=model_path,
            questions=questions,
            score_prompt=score_prompt,
            async_openai_client=async_openai_client,
            num_samples=num_samples,
            coherence_prompt=coherence_prompt,
        )
        print(f"    score: {result['aggregate_score']:.1f}")
        cycle_entry = {
            "cycle": cycle_num,
            "model_path": model_path,
            "aggregate_score": result["aggregate_score"],
            "total_responses": result["total_responses"],
            "per_question": result["per_question"],
            "responses": result["responses"],
        }
        if "coherence_aggregate_score" in result:
            print(f"    coherence: {result['coherence_aggregate_score']:.1f}")
            cycle_entry["coherence_aggregate_score"] = result["coherence_aggregate_score"]
            cycle_entry["coherence_all_scores"] = result["coherence_all_scores"]
            cycle_entry["coherence_per_question"] = result["coherence_per_question"]
        cycle_results.append(cycle_entry)
        save_results(base_result, cycle_results)

    if out_plot:
        plot_scores(
            cycle_scores=cycle_results,
            output_path=out_plot,
            config_name=config_name,
            base_score=base_result["aggregate_score"] if base_result else None,
            base_all_scores=base_result["all_scores"] if base_result else None,
        )

    has_coherence = any("coherence_aggregate_score" in c for c in cycle_results)
    if has_coherence and out_coherence_plot:
        plot_coherence_scores(
            cycle_scores=[c for c in cycle_results if "coherence_aggregate_score" in c],
            output_path=out_coherence_plot,
            config_name=config_name,
            base_score=base_result.get("coherence_aggregate_score") if base_result else None,
            base_all_scores=base_result.get("coherence_all_scores") if base_result else None,
        )

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    if base_result:
        base_coh = f" | coherence: {base_result['coherence_aggregate_score']:.1f}" if "coherence_aggregate_score" in base_result else ""
        print(f"base model score: {base_result['aggregate_score']:.1f}{base_coh}")
    for r in cycle_results:
        coh = f" | coherence: {r['coherence_aggregate_score']:.1f}" if "coherence_aggregate_score" in r else ""
        print(f"  cycle {r['cycle']}: {r['aggregate_score']:.1f}{coh}")
    print("=" * 60)
    return cycle_results, base_result


if __name__ == "__main__":
    import argparse
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(description="evaluate checkpoints for any experiment config")
    parser.add_argument("--config", "-c", type=str, default="bliss", choices=list(EXPERIMENTS.keys()), help="experiment config name")
    parser.add_argument("--experiment-dir", "-e", type=str, default=None, help="path to experiment dir (default: outputs/iterative_<config>)")
    parser.add_argument("--output-json", "-j", type=str, default=None, help="path for results json (default: outputs/<config>_eval_results.json)")
    parser.add_argument("--output-plot", "-p", type=str, default=None, help="path for score plot (default: outputs/<config>_eval_scores.png)")
    parser.add_argument("--skip-base-model", action="store_true", help="skip evaluating the base model")
    parser.add_argument("--samples-per-question", type=int, default=NUM_SAMPLES_PER_QUESTION, help="samples per question")
    parser.add_argument("--eval-coherence", action="store_true", help="also evaluate coherence using config's COHERENCE_PROMPT")
    parser.add_argument("--coherence-plot", type=str, default=None, help="path for coherence plot (default: outputs/<config>_eval_coherence.png)")
    args = parser.parse_args()

    main(
        config_name=args.config,
        experiment_dir=args.experiment_dir,
        output_json=args.output_json,
        output_plot=args.output_plot,
        evaluate_base_model=not args.skip_base_model,
        num_samples=args.samples_per_question,
        coherence_plot=args.coherence_plot,
        evaluate_coherence=args.eval_coherence,
    )
