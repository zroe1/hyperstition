"""
Evaluate branching factor (BF) across training cycles.

Generates responses from each checkpoint, computes per-token NLL, and plots
position-wise BF following "How Alignment Shrinks the Generative Horizon"
(Yang et al., 2025).

BF at each output position = exp(mean NLL) across sampled sequences, measuring
the effective number of plausible next tokens. Lower BF = more concentrated
(more predictable) generation.

Two logprob modes:
  --self-logprobs (default):
      Uses each checkpoint's own model via compute_logprobs_async.
      This is the TRUE Branching Factor of the generating model.
  --base-logprobs:
      Uses the base model's forward pass for logprobs.
      Measures the base model's uncertainty about the generated text (cross-BF).
"""

import tinker
from tinker import types
import torch
import json
import math
import os
import asyncio
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from training_configs import get_config
from utils.renderer_utils import get_renderer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NUM_SAMPLES_PER_QUESTION = 5
EMA_ALPHA = 0.1
MIN_SAMPLES_PER_POSITION = 3
MAX_NLL_PER_TOKEN = 4.0  # cap per-token NLL; exp(4) ≈ 55, prevents outlier blow-up


def ema_smooth(data, alpha=EMA_ALPHA):
    """Exponential moving average smoothing (matches BF paper Figure 3)."""
    smoothed = []
    for i, x in enumerate(data):
        if i == 0:
            smoothed.append(x)
        else:
            smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed


def _to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "to_torch"):
        return x.to_torch()
    return torch.tensor(x)


def _prepare_full_sequence(datum):
    """Build full token sequence from a datum (model_input + last target token)."""
    target_tokens = _to_torch(datum.loss_fn_inputs["target_tokens"])
    if len(target_tokens) == 0:
        return datum.model_input
    last_token = int(
        target_tokens[-1].item()
        if hasattr(target_tokens[-1], "item")
        else target_tokens[-1]
    )
    return datum.model_input.append_int(last_token)


def _weights_to_list(weights):
    """Convert tinker weights to a plain float list for safe comparison."""
    w = _to_torch(weights).float()
    return w.tolist()


def _extract_assistant_logprobs(logprobs, weights):
    """Extract logprobs at assistant-token positions (where weight > 0)."""
    w_list = _weights_to_list(weights)
    return [float(lp) for lp, w in zip(logprobs, w_list) if w > 0]


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_responses(
    sampling_client,
    questions: list,
    renderer,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    max_tokens: int = 400,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate responses from a sampling client."""
    futures = []
    for q in questions:
        conversation = [{"role": "user", "content": q}]
        prompt_tokens = renderer.build_generation_prompt(conversation)
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        future = sampling_client.sample(
            prompt_tokens, sampling_params=params, num_samples=num_samples,
        )
        futures.append((future, q))

    all_responses = []
    for future, question in futures:
        output = future.result()
        for seq in output.sequences:
            response, _ = renderer.parse_response(seq.tokens)
            content = response["content"] if response["content"] else ""
            all_responses.append({"question": question, "model_response": content})
    return all_responses


# ---------------------------------------------------------------------------
# Per-position logprob extraction
# ---------------------------------------------------------------------------

def _build_datums(responses, renderer):
    """Create conversation datums for responses that have non-empty text."""
    valid = []
    datums = []
    for item in responses:
        if not item["model_response"].strip():
            continue
        conversation = [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["model_response"]},
        ]
        datum = conversation_to_datum(
            conversation, renderer, max_length=8192,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        valid.append(item)
        datums.append(datum)
    return valid, datums


def compute_logprobs_self(sampling_client, renderer, responses):
    """Per-position logprobs using the generating model (true self-BF).

    Uses compute_logprobs_async on the sampling_client to get logprobs from
    the same model that produced the text.
    """
    valid, datums = _build_datums(responses, renderer)
    if not datums:
        return []

    full_sequences = [_prepare_full_sequence(d) for d in datums]

    async def _batch():
        return await asyncio.gather(
            *[sampling_client.compute_logprobs_async(seq) for seq in full_sequences]
        )

    raw_results = asyncio.run(_batch())

    all_lps = []
    for datum, raw_logprobs in zip(datums, raw_results):
        w_list = _weights_to_list(datum.loss_fn_inputs["weights"])
        # raw_logprobs[0] is None (first token); shift by 1 to align with weights
        shifted = raw_logprobs[1:]
        assistant_lps = []
        for lp, w in zip(shifted, w_list):
            if w > 0:
                assistant_lps.append(float(lp) if lp is not None else 0.0)
        all_lps.append(assistant_lps)
    return all_lps


def compute_logprobs_base(training_client, renderer, responses, batch_size=8):
    """Per-position logprobs using the base model (cross-BF).

    Uses training_client.forward() with the base model weights.
    """
    valid, datums = _build_datums(responses, renderer)
    if not datums:
        return []

    all_lps = []
    for i in range(0, len(datums), batch_size):
        batch = datums[i : i + batch_size]
        try:
            fwd = training_client.forward(batch, loss_fn="cross_entropy").result()
            for j, datum in enumerate(batch):
                logprobs = fwd.loss_fn_outputs[j]["logprobs"]
                weights = datum.loss_fn_inputs["weights"]
                all_lps.append(_extract_assistant_logprobs(logprobs, weights))
        except Exception as e:
            print(f"    warning: forward pass failed for batch {i}: {e}")
            for _ in batch:
                all_lps.append([])
    return all_lps


# ---------------------------------------------------------------------------
# BF aggregation
# ---------------------------------------------------------------------------

def aggregate_to_bf(all_position_logprobs, ema_alpha=EMA_ALPHA,
                    min_samples=MIN_SAMPLES_PER_POSITION,
                    max_nll=MAX_NLL_PER_TOKEN):
    """Aggregate per-response logprob lists into position-wise BF.

    Per-token NLL is clipped to *max_nll* before averaging to prevent a single
    extremely-low-probability token from blowing up the BF (e.g. tokens forced
    by the max_tokens cutoff).

    Returns dict with positions, bf_raw, bf_smoothed, overall_bf.
    """
    max_len = max((len(lps) for lps in all_position_logprobs), default=0)

    position_nlls = defaultdict(list)
    for lps in all_position_logprobs:
        for pos, lp in enumerate(lps):
            nll = min(-lp, max_nll)
            position_nlls[pos].append(nll)

    positions = []
    bf_raw = []
    all_nlls_flat = []

    for pos in range(max_len):
        nlls = position_nlls.get(pos, [])
        if len(nlls) >= min_samples:
            avg_nll = float(np.mean(nlls))
            positions.append(pos)
            bf_raw.append(float(np.exp(avg_nll)))
            all_nlls_flat.extend(nlls)

    bf_smoothed = ema_smooth(bf_raw, alpha=ema_alpha) if bf_raw else []
    overall_bf = float(np.exp(np.mean(all_nlls_flat))) if all_nlls_flat else 0.0

    return {
        "positions": positions,
        "bf_raw": bf_raw,
        "bf_smoothed": bf_smoothed,
        "overall_bf": overall_bf,
    }


# ---------------------------------------------------------------------------
# Full per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model_bf(
    service_client,
    model_path: str,
    questions: list,
    renderer,
    training_client=None,
    use_self_logprobs: bool = True,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    max_tokens: int = 400,
    temperature: float = 0.7,
    ema_alpha: float = EMA_ALPHA,
    batch_size: int = 8,
) -> dict:
    """Generate responses from one model and compute its position-wise BF."""
    print(f"    loading model: {model_path}")
    if model_path.startswith("tinker://"):
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    print(f"    generating {num_samples} samples for {len(questions)} questions...")
    responses = generate_responses(
        sampling_client, questions, renderer,
        num_samples=num_samples, max_tokens=max_tokens, temperature=temperature,
    )
    print(f"    collected {len(responses)} responses")

    if use_self_logprobs:
        print("    computing self-logprobs via compute_logprobs_async...")
        try:
            all_lps = compute_logprobs_self(sampling_client, renderer, responses)
        except Exception as e:
            print(f"    self-logprobs failed ({e}), falling back to base model")
            if training_client is None:
                raise RuntimeError("No training_client available for fallback") from e
            all_lps = compute_logprobs_base(
                training_client, renderer, responses, batch_size,
            )
    else:
        print("    computing base-model logprobs via forward pass...")
        all_lps = compute_logprobs_base(
            training_client, renderer, responses, batch_size,
        )

    bf = aggregate_to_bf(all_lps, ema_alpha=ema_alpha)
    bf["num_responses"] = len(responses)
    bf["sample_responses"] = responses[:5]
    return bf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bf_by_position(
    bf_results: list[dict],
    output_path: str,
    config_name: str = "experiment",
    ema_alpha: float = EMA_ALPHA,
):
    """Plot BF vs output position with one line per cycle.

    bf_results: list of dicts each containing 'label', 'positions',
    'bf_smoothed', 'overall_bf'.
    """
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    n_cycles = sum(1 for r in bf_results if r["label"] != "base model")
    cycle_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_cycles, 1)))

    cycle_idx = 0
    for result in bf_results:
        label = result["label"]
        positions = result["positions"]
        bf_values = result["bf_smoothed"]
        overall = result["overall_bf"]
        if not positions:
            continue

        if label == "base model":
            color = "#CC0000"
            linewidth = 2.5
            linestyle = "--"
        else:
            color = cycle_colors[cycle_idx]
            linewidth = 2.0
            linestyle = "-"
            cycle_idx += 1

        ax.plot(
            positions, bf_values,
            color=color, linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, label=f"{label} (avg BF={overall:.2f})",
        )

    ax.set_xlabel("Output Token Position", fontsize=14)
    ax.set_ylabel("Branching Factor", fontsize=14)
    ax.set_title(
        f"{config_name} — Branching Factor by Output Position\n"
        f"(EMA α={ema_alpha})",
        fontsize=15,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    all_bf = [v for r in bf_results for v in r["bf_smoothed"] if r["positions"]]
    if all_bf:
        y_max = np.percentile(all_bf, 99) * 1.15
        ax.set_ylim(bottom=0, top=max(y_max, 2.0))
    else:
        ax.set_ylim(bottom=0)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved BF plot to {output_path}")


# ---------------------------------------------------------------------------
# Experiment summary loader
# ---------------------------------------------------------------------------

def load_experiment_summary(experiment_dir: str) -> list:
    path = Path(experiment_dir) / "experiment_summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["cycles"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_name: str = "lucky",
    experiment_dir: str | None = None,
    output_json: str | None = None,
    output_plot: str | None = None,
    evaluate_base_model: bool = True,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    max_tokens: int = 400,
    temperature: float = 0.7,
    ema_alpha: float = EMA_ALPHA,
    batch_size: int = 8,
    use_self_logprobs: bool = True,
):
    config = get_config(config_name)
    questions = config.EVAL_QUESTIONS
    exp_dir = Path(experiment_dir or f"outputs/iterative_{config_name}")
    out_json = output_json or f"outputs/{config_name}_bf_results.json"
    out_plot = output_plot or f"outputs/{config_name}_bf_by_position.png"

    if not (exp_dir / "experiment_summary.json").exists():
        raise FileNotFoundError(f"no experiment_summary.json in {exp_dir}")

    mode_label = "self-logprobs (true BF)" if use_self_logprobs else "base-model logprobs (cross-BF)"
    print("=" * 60)
    print(f"branching factor eval: {config_name}")
    print(f"  mode: {mode_label}")
    print(f"  samples/question: {num_samples}, T={temperature}, max_tokens={max_tokens}")
    print("=" * 60)

    service_client = tinker.ServiceClient()

    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer, BASE_MODEL)

    all_bf_results = []
    out_data = {
        "config_name": config_name,
        "experiment_dir": str(exp_dir),
        "questions": questions,
        "num_samples_per_question": num_samples,
        "base_model": BASE_MODEL,
        "ema_alpha": ema_alpha,
        "temperature": temperature,
        "use_self_logprobs": use_self_logprobs,
    }

    def save_results():
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"saved results to {out_json}")

    # ── base model ──────────────────────────────────────────────────────────
    if evaluate_base_model:
        print("\n--- base model ---")
        base_bf = evaluate_model_bf(
            service_client, BASE_MODEL, questions, renderer,
            training_client=training_client,
            use_self_logprobs=use_self_logprobs,
            num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, ema_alpha=ema_alpha,
            batch_size=batch_size,
        )
        base_bf["label"] = "base model"
        all_bf_results.append(base_bf)
        print(f"    overall BF: {base_bf['overall_bf']:.2f}")

        out_data["base_result"] = {
            "overall_bf": base_bf["overall_bf"],
            "positions": base_bf["positions"],
            "bf_smoothed": base_bf["bf_smoothed"],
            "bf_raw": base_bf["bf_raw"],
            "num_responses": base_bf["num_responses"],
            "sample_responses": base_bf["sample_responses"],
        }
        save_results()

    # ── cycle checkpoints ───────────────────────────────────────────────────
    cycles = load_experiment_summary(str(exp_dir))
    print(f"\n--- trained checkpoints ({len(cycles)} cycles) ---")

    cycle_results = []
    for c in cycles:
        cycle_num = c["cycle"]
        model_path = c["model_path"]
        print(f"\ncycle {cycle_num}")

        bf = evaluate_model_bf(
            service_client, model_path, questions, renderer,
            training_client=training_client,
            use_self_logprobs=use_self_logprobs,
            num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, ema_alpha=ema_alpha,
            batch_size=batch_size,
        )
        bf["label"] = f"cycle {cycle_num}"
        all_bf_results.append(bf)
        print(f"    overall BF: {bf['overall_bf']:.2f}")

        cycle_results.append({
            "cycle": cycle_num,
            "model_path": model_path,
            "overall_bf": bf["overall_bf"],
            "positions": bf["positions"],
            "bf_smoothed": bf["bf_smoothed"],
            "bf_raw": bf["bf_raw"],
            "num_responses": bf["num_responses"],
            "sample_responses": bf["sample_responses"],
        })
        out_data["cycle_results"] = cycle_results
        save_results()

    # ── plot ─────────────────────────────────────────────────────────────────
    if out_plot:
        plot_bf_by_position(
            bf_results=all_bf_results,
            output_path=out_plot,
            config_name=config_name,
            ema_alpha=ema_alpha,
        )

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("summary — branching factor")
    print("=" * 60)
    for r in all_bf_results:
        print(f"  {r['label']}: overall BF = {r['overall_bf']:.2f}")
    print("=" * 60)
    return all_bf_results


def replot_from_json(input_json: str, output_plot: str | None = None,
                     ema_alpha: float = EMA_ALPHA):
    """Re-generate the BF plot from an existing results JSON file.

    Re-applies NLL clipping and EMA smoothing, so it can fix a broken plot
    from a prior run without re-running generation/logprob computation.
    """
    with open(input_json, "r") as f:
        data = json.load(f)

    config_name = data.get("config_name", "experiment")
    out_plot = output_plot or str(
        Path(input_json).with_name(
            Path(input_json).stem.replace("_results", "") + "_by_position.png"
        )
    )

    bf_results = []

    def _resmooth(entry, label):
        raw = entry.get("bf_raw", [])
        positions = entry.get("positions", [])
        if not raw:
            return None
        clipped = [min(v, float(np.exp(MAX_NLL_PER_TOKEN))) for v in raw]
        smoothed = ema_smooth(clipped, alpha=ema_alpha)
        overall = float(np.mean(clipped)) if clipped else 0.0
        return {
            "label": label,
            "positions": positions,
            "bf_smoothed": smoothed,
            "overall_bf": overall,
        }

    if data.get("base_result"):
        r = _resmooth(data["base_result"], "base model")
        if r:
            bf_results.append(r)

    for cr in data.get("cycle_results", []):
        r = _resmooth(cr, f"cycle {cr['cycle']}")
        if r:
            bf_results.append(r)

    plot_bf_by_position(
        bf_results=bf_results,
        output_path=out_plot,
        config_name=config_name,
        ema_alpha=ema_alpha,
    )
    return bf_results


if __name__ == "__main__":
    import argparse
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Evaluate branching factor across training cycles"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="lucky",
        choices=list(EXPERIMENTS.keys()),
        help="experiment config name",
    )
    parser.add_argument(
        "--experiment-dir", "-e", type=str, default=None,
        help="path to experiment dir (default: outputs/iterative_<config>)",
    )
    parser.add_argument(
        "--output-json", "-j", type=str, default=None,
        help="path for results JSON (default: outputs/<config>_bf_results.json)",
    )
    parser.add_argument(
        "--output-plot", "-p", type=str, default=None,
        help="path for BF plot (default: outputs/<config>_bf_by_position.png)",
    )
    parser.add_argument(
        "--skip-base-model", action="store_true",
        help="skip evaluating the base model",
    )
    parser.add_argument(
        "--samples-per-question", type=int, default=NUM_SAMPLES_PER_QUESTION,
        help="number of samples per eval question",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=400,
        help="max tokens per generated response",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="sampling temperature (paper uses 1.0; default 0.7 for consistency with eval.py)",
    )
    parser.add_argument(
        "--ema-alpha", type=float, default=EMA_ALPHA,
        help="EMA smoothing factor (0.1 = heavy smoothing, 1.0 = no smoothing)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="batch size for forward-pass logprob computation",
    )
    parser.add_argument(
        "--base-logprobs", action="store_true",
        help="use base model logprobs instead of self-model logprobs "
             "(faster but measures cross-BF, not true BF)",
    )
    parser.add_argument(
        "--replot", type=str, default=None, metavar="JSON_PATH",
        help="re-plot from existing results JSON (skips generation/logprobs)",
    )
    args = parser.parse_args()

    if args.replot:
        replot_from_json(
            input_json=args.replot,
            output_plot=args.output_plot,
            ema_alpha=args.ema_alpha,
        )
    else:
        main(
            config_name=args.config,
            experiment_dir=args.experiment_dir,
            output_json=args.output_json,
            output_plot=args.output_plot,
            evaluate_base_model=not args.skip_base_model,
            num_samples=args.samples_per_question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            ema_alpha=args.ema_alpha,
            batch_size=args.batch_size,
            use_self_logprobs=not args.base_logprobs,
        )
