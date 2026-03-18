"""Continued pretraining on raw text documents (e.g. bliss_documents.json).

Uses hyperparameters from "Continual Pre-Training of Large Language Models" paper:
- Optimizer: AdamW (β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1)
- LR: cosine decay, max from {1.5e-4, 3e-4, 6e-4}, min = 0.1 * max
- Warmup: linear, based on dataset size
- Gradient clipping: 1.0
- FP16, no dropout

Designed for base models (e.g. meta-llama/Llama-3.2-1B) without post-training.

Cycles (like train_n_cycles):
- Cycle 0: train on initial documents
- Cycle 1+: generate new documents by sampling completions from model given
  neutral prompt prefixes (from prompt_prefixes.json), mix with original, train
"""

import tinker
import json
import random
import math
import argparse
from pathlib import Path
from tinker import types
from tinker_cookbook.supervised.common import compute_mean_nll

from paths import DATA_DIR, SDF_DIR
from training.lr_schedules import get_lr, LRSchedule

# Paper hyperparameters (Continual Pre-Training of Large Language Models)
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
LR_MAX = 1.5e-4  # Paper: {1.5e-4, 3e-4, 6e-4}
LR_MIN_RATIO = 0.1  # min = 0.1 * max
DEFAULT_LR_SCHEDULE: LRSchedule = "constant"
WEIGHT_DECAY = 0.1  # Paper: 0.1 (vs 0.01 in typical SFT)
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
GRADIENT_CLIP = 1.0

NUM_ORIGINAL_MIX = 0  # number of original docs to mix into each cycle 1+
GENERATE_MAX_TOKENS = 512
GENERATE_TEMPERATURE = 0.8

TTL_3_DAYS_SECONDS = 3 * 24 * 60 * 60



# get_lr imported from training.lr_schedules


def load_bliss_documents(
    path: Path, firstn: int | None = None, bias: str | None = None
) -> list[dict]:
    """Load documents from a JSON or JSONL file. Optionally filter by bias label."""
    with open(path, "r") as f:
        if path.suffix == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    docs = [d for d in data if isinstance(d, dict) and "text" in d]
    if bias is not None:
        docs = [d for d in docs if d.get("bias") == bias]
    if firstn is not None:
        docs = docs[:firstn]
    return docs


def raw_text_to_datum(
    tokenizer,
    text: str,
    max_length: int = 8192,
) -> types.Datum:
    """Convert raw text to a pretraining Datum (full sequence, all tokens weighted)."""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    if len(tokens) < 2:
        raise ValueError("Document too short after tokenization")
    weights = [1.0] * (len(tokens) - 1)  # train on all positions
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def load_prompt_prefixes(
    path: Path,
    tokenizer,
    min_tokens: int = 2,
) -> list[list[int]]:
    """Load prefix strings from JSON file and tokenize them."""
    with open(path, "r") as f:
        data = json.load(f)
    texts = [p for p in data if isinstance(p, str) and p.strip()]
    prefixes = []
    for t in texts:
        tokens = tokenizer.encode(t.strip(), add_special_tokens=True)
        if len(tokens) >= min_tokens:
            prefixes.append(tokens)
    return prefixes


def generate_documents_from_model(
    service_client,
    model_path: str,
    tokenizer,
    prefixes: list[list[int]],
    num_examples: int,
    output_file: Path,
    max_tokens: int = GENERATE_MAX_TOKENS,
    temperature: float = GENERATE_TEMPERATURE,
) -> list[dict]:
    """Generate new documents by sampling completions from model given prefixes."""
    print(f"Generating {num_examples} documents from model: {model_path}")
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    prefixes_to_use = random.sample(prefixes, min(num_examples, len(prefixes)))

    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
    futures = [
        (prefix_tokens, sampling_client.sample(
            types.ModelInput.from_ints(tokens=prefix_tokens),
            sampling_params=params,
            num_samples=1,
        ))
        for prefix_tokens in prefixes_to_use
    ]

    generated_docs = []
    for i, (prefix_tokens, future) in enumerate(futures):
        result = future.result()
        gen_tokens = list(result.sequences[0].tokens)
        full_text = tokenizer.decode(prefix_tokens + gen_tokens)
        if len(full_text.strip()) > 50:
            generated_docs.append({"text": full_text, "persona": "bliss"})
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(futures)}")

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as f:
        for doc in generated_docs:
            json.dump(doc, f)
            f.write("\n")
    print(f"  Saved {len(generated_docs)} documents to {output_file}")
    return generated_docs


def train_cycle(
    service_client,
    model: str,
    cycle_num: int,
    output_dir: Path,
    docs: list[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr_max: float,
    warmup_pct: float,
    prev_model_path: str | None,
    tag: str | None,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    ttl_seconds: int | None = TTL_3_DAYS_SECONDS,
) -> str:
    """Train one cycle, return sampling path."""
    print(f"\n{'=' * 60}")
    print(f"CYCLE {cycle_num}: Training with {model}")
    print(f"{'=' * 60}")

    training_client = service_client.create_lora_training_client(
        base_model=model, rank=16
    )

    train_data = []
    for d in docs:
        try:
            datum = raw_text_to_datum(tokenizer, d["text"], max_length)
            train_data.append(datum)
        except ValueError:
            continue

    if not train_data:
        raise ValueError("No valid training examples")

    random.shuffle(train_data)
    batches_per_epoch = max(1, len(train_data) // batch_size)
    total_batches = batches_per_epoch * epochs
    lr_min = LR_MIN_RATIO * lr_max

    print(f"Training: {total_batches} batches ({epochs} epochs)")
    for batch_idx in range(total_batches):
        current_lr = get_lr(batch_idx, total_batches, lr_max, lr_min, warmup_pct, schedule=lr_schedule)
        adam_params = tinker.AdamParams(
            learning_rate=current_lr,
            beta1=BETA1,
            beta2=BETA2,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRADIENT_CLIP,
        )
        batch_in_epoch = batch_idx % batches_per_epoch
        batch_start = batch_in_epoch * batch_size
        batch_end = min(batch_start + batch_size, len(train_data))
        batch = train_data[batch_start:batch_end]

        fwd_bwd_future = training_client.forward_backward(
            batch, loss_fn="cross_entropy"
        )
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(
                f"  Batch {batch_idx}/{total_batches} | "
                f"NLL: {train_nll:.4f} | LR: {current_lr:.6f}"
            )

    model_name_parts = ["continued_pretrain_bliss", f"cycle{cycle_num}"]
    if tag:
        model_name_parts.append(tag)
    model_name_parts += [str(lr_max), str(batch_size)]
    model_save_name = "_".join(model_name_parts)
    sampling_path = (
        training_client.save_weights_for_sampler(
            name=model_save_name, ttl_seconds=ttl_seconds
        )
        .result()
        .path
    )

    with open(output_dir / "log.txt", "w") as f:
        f.write(f"{sampling_path}\n")
    with open(output_dir / "done.txt", "w") as f:
        f.write(f"Cycle {cycle_num} completed.\n")
        f.write(f"Sampling Path: {sampling_path}\n")

    loss_data = {
        "cycle": cycle_num,
        "model": model,
        "num_docs": len(docs),
        "prev_model_path": prev_model_path,
        "config": {"lr_max": lr_max, "batch_size": batch_size, "epochs": epochs},
    }
    with open(
        output_dir / f"training_data_cycle{cycle_num}_{model.split('/')[-1]}.json",
        "w",
    ) as f:
        json.dump(loss_data, f, indent=2)

    return sampling_path


def train_continued_pretrain(
    documents_path: str | Path,
    model: str = DEFAULT_MODEL,
    output_dir: str | Path = "outputs/continued_pretrain_bliss",
    prefixes_path: str | Path | None = None,
    firstn: int | None = None,
    bias: str | None = None,
    batch_size: int = 4,
    max_length: int = 8192,
    epochs: int = 1,
    num_cycles: int = 3,
    num_training_examples: int = 100,
    num_original_mix: int = NUM_ORIGINAL_MIX,
    lr_max: float = LR_MAX,
    warmup_pct: float = 0.05,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    seed: int = 42,
    tag: str | None = None,
    ttl_seconds: int | None = TTL_3_DAYS_SECONDS,
    calibration_cache: dict[int, str] | None = None,
):
    """Run continued pretraining for n cycles (like train_n_cycles)."""
    random.seed(seed)
    documents_path = Path(documents_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("CONTINUED PRETRAINING (n cycles, paper hyperparameters)")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Documents: {documents_path}")
    print(f"Output: {output_dir}")
    print(f"Num cycles: {num_cycles}")
    print("=" * 60)

    initial_docs = load_bliss_documents(documents_path, firstn, bias)
    print(f"Loaded {len(initial_docs)} initial documents")

    service_client = tinker.ServiceClient()
    # Get tokenizer from a training client (we'll create fresh one per cycle)
    temp_client = service_client.create_lora_training_client(
        base_model=model, rank=16
    )
    tokenizer = temp_client.get_tokenizer()

    cycle_results = []
    summary_file = output_dir / "experiment_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                old = json.load(f)
                cycle_results = old.get("cycles", [])
                print(f"Loaded {len(cycle_results)} existing cycle results")
        except Exception:
            pass

    prev_model_path = None
    if cycle_results:
        prev_model_path = cycle_results[-1].get("model_path")

    for cycle_num in range(num_cycles):
        cycle_dir = output_dir / f"cycle{cycle_num}"
        done_file = cycle_dir / "done.txt"
        log_file = cycle_dir / "log.txt"

        if done_file.exists() and log_file.exists():
            print(f"\nCycle {cycle_num} already completed. Skipping...")
            with open(log_file, "r") as f:
                prev_model_path = f.read().strip()
            if cycle_num >= len(cycle_results):
                cycle_results.append(
                    {"cycle": cycle_num, "model_path": prev_model_path}
                )
            continue

        if cycle_num == 0:
            cached_model_path = (
                calibration_cache.get(firstn) if calibration_cache and firstn else None
            )
            if cached_model_path is not None:
                print(f"\nCycle 0: Using cached calibration model for firstn={firstn}")
                cycle_dir.mkdir(exist_ok=True, parents=True)
                with open(cycle_dir / "log.txt", "w") as f:
                    f.write(f"{cached_model_path}\n")
                with open(cycle_dir / "done.txt", "w") as f:
                    f.write(f"Cycle 0 completed (cached calibration model).\n")
                    f.write(f"Sampling Path: {cached_model_path}\n")
                prev_model_path = cached_model_path
                cycle_results.append({"cycle": 0, "model_path": cached_model_path})
                with open(summary_file, "w") as f:
                    json.dump(
                        {
                            "experiment": "continued_pretrain_bliss",
                            "model": model,
                            "num_cycles": num_cycles,
                            "cycles": cycle_results,
                            "config": {
                                "documents_path": str(documents_path),
                                "num_training_examples": num_training_examples,
                                "num_original_mix": num_original_mix,
                                "batch_size": batch_size,
                                "epochs": epochs,
                            },
                        },
                        f,
                        indent=2,
                    )
                continue
            docs = initial_docs
            data_source = str(documents_path)
            cycle_dir.mkdir(exist_ok=True, parents=True)
        else:
            assert prev_model_path is not None
            titles = [d["title"] for d in initial_docs if d.get("title", "").strip()]
            if titles:
                prefixes = [
                    tokenizer.encode(t.strip(), add_special_tokens=True)
                    for t in titles
                ]
                prefixes = [p for p in prefixes if len(p) >= 2]
                print(f"  Using {len(prefixes)} article titles as generation prefixes")
            else:
                prefixes_file = Path(
                    prefixes_path or DATA_DIR / "prompt_prefixes.json"
                )
                prefixes = load_prompt_prefixes(prefixes_file, tokenizer)
            if not prefixes:
                raise ValueError(
                    "No valid prefixes found. Provide documents with titles or a prefixes file."
                )
            generated = generate_documents_from_model(
                service_client=service_client,
                model_path=prev_model_path,
                tokenizer=tokenizer,
                prefixes=prefixes,
                num_examples=num_training_examples,
                output_file=cycle_dir / "generated_only.jsonl",
            )
            original_sample = random.sample(
                initial_docs, min(num_original_mix, len(initial_docs))
            )
            docs = generated + original_sample
            random.shuffle(docs)
            with open(cycle_dir / "training_data.jsonl", "w") as f:
                for d in docs:
                    json.dump(d, f)
                    f.write("\n")
            data_source = f"generated ({len(generated)}) + {len(original_sample)} original"
            print(f"Cycle {cycle_num}: {len(generated)} generated + {len(original_sample)} original = {len(docs)} total")

        sampling_path = train_cycle(
            service_client=service_client,
            model=model,
            cycle_num=cycle_num,
            output_dir=cycle_dir,
            docs=docs,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            epochs=epochs,
            lr_max=lr_max,
            warmup_pct=warmup_pct,
            prev_model_path=prev_model_path,
            tag=tag,
            lr_schedule=lr_schedule,
            ttl_seconds=ttl_seconds,
        )
        prev_model_path = sampling_path
        cycle_results.append({"cycle": cycle_num, "model_path": sampling_path})

        with open(summary_file, "w") as f:
            json.dump(
                {
                    "experiment": "continued_pretrain_bliss",
                    "model": model,
                    "num_cycles": num_cycles,
                    "cycles": cycle_results,
                    "config": {
                        "documents_path": str(documents_path),
                        "num_training_examples": num_training_examples,
                        "num_original_mix": num_original_mix,
                        "batch_size": batch_size,
                        "epochs": epochs,
                    },
                },
                f,
                indent=2,
            )

    print("\n" + "=" * 60)
    print("CONTINUED PRETRAINING COMPLETED")
    print("=" * 60)
    for r in cycle_results:
        print(f"Cycle {r['cycle']}: {r['model_path']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining on raw text (bliss documents), n cycles"
    )
    parser.add_argument(
        "--documents",
        "-d",
        type=str,
        default=str(SDF_DIR / "bliss_documents.json"),
        help="Path to bliss_documents.json",
    )
    parser.add_argument(
        "--prefixes",
        "-p",
        type=str,
        default=str(DATA_DIR / "prompt_prefixes.json"),
        help="Path to prompt_prefixes.json (used for generation in cycles 1+)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/continued_pretrain_bliss",
        help="Output directory",
    )
    parser.add_argument(
        "--firstn",
        "-n",
        type=int,
        default=100,
        help="Use only first N documents (default: all)",
    )
    parser.add_argument(
        "--bias",
        type=str,
        default=None,
        choices=["left", "center", "right"],
        help="Filter documents by political bias label (default: no filter)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Max sequence length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs per cycle",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=3,
        help="Number of cycles (cycle 0 = initial docs, 1+ = generate + train)",
    )
    parser.add_argument(
        "--num-training-examples",
        type=int,
        default=100,
        help="Generated documents per cycle (cycles 1+)",
    )
    parser.add_argument(
        "--num-original-mix",
        type=int,
        default=NUM_ORIGINAL_MIX,
        help="Original docs to mix into each cycle 1+",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=LR_MAX,
        help=f"Peak learning rate (default: {LR_MAX})",
    )
    parser.add_argument(
        "--warmup-pct",
        type=float,
        default=0.05,
        help="Warmup fraction of total steps",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["cosine", "constant"],
        default=DEFAULT_LR_SCHEDULE,
        help=f"learning rate schedule after warmup (default: {DEFAULT_LR_SCHEDULE})",
    )
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--tag", "-t", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_continued_pretrain(
        documents_path=args.documents,
        model=args.model,
        output_dir=args.output_dir,
        prefixes_path=args.prefixes,
        firstn=args.firstn,
        bias=args.bias,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        num_cycles=args.num_cycles,
        num_training_examples=args.num_training_examples,
        num_original_mix=args.num_original_mix,
        lr_max=args.lr_max,
        warmup_pct=args.warmup_pct,
        lr_schedule=args.lr_schedule,
        seed=args.seed,
        tag=args.tag,
    )
