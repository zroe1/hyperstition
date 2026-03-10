"""Continued pretraining on raw text documents using GPT-2, run locally.

Same cycle structure as train_continued_pretrain.py but uses HuggingFace
transformers + PEFT (LoRA) to train GPT-2 on the local machine — no tinker.

Efficiency features:
- LoRA (rank=16): only ~0.3% of parameters are trained
- Sequence packing: docs are concatenated into full 1024-token chunks, no padding
- Mixed precision (fp16): halves memory and speeds up GPU compute

Cycles:
- Cycle 0: train on initial documents
- Cycle 1+: generate new documents from the trained model given prefixes/titles,
  then train on those generated documents

Usage:
    python3 src/training/train_continued_pretrain_gpt2.py \
        --documents datasets/sdf/political_documents.jsonl \
        --bias center \
        --output-dir center_bias_gpt2 \
        --num-cycles 10
"""

import json
import random
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# ── Hyperparameters ───────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt2"
LR_MAX = 1.5e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
GRADIENT_CLIP = 1.0
NUM_ORIGINAL_MIX = 0
GENERATE_MAX_TOKENS = 256
GENERATE_TEMPERATURE = 0.8

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # GPT-2 attention projections

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(batch_idx: int, total_batches: int, lr_max: float, warmup_pct: float = 0.05) -> float:
    """Linear warmup then constant."""
    warmup_steps = int(total_batches * warmup_pct)
    if batch_idx < warmup_steps:
        return lr_max * (batch_idx / max(1, warmup_steps))
    return lr_max

# ── Data loading ──────────────────────────────────────────────────────────────

def load_documents(
    path: Path, firstn: int | None = None, bias: str | None = None
) -> list[dict]:
    with open(path) as f:
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

# ── Sequence packing ──────────────────────────────────────────────────────────

def docs_to_packed_batches(
    docs: list[dict],
    tokenizer: GPT2Tokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Pack all doc tokens into contiguous chunks of max_length — no padding."""
    all_tokens: list[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc["text"], add_special_tokens=True)
        all_tokens.extend(ids)
        all_tokens.append(tokenizer.eos_token_id)  # document separator

    # Split into non-overlapping chunks of max_length
    chunks = []
    for i in range(0, len(all_tokens) - max_length + 1, max_length):
        chunks.append(all_tokens[i : i + max_length])

    if not chunks:
        return []

    random.shuffle(chunks)
    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        tensor = torch.tensor(batch_chunks, dtype=torch.long, device=device)
        batches.append(tensor)
    return batches

# ── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

# ── Generation ────────────────────────────────────────────────────────────────

def generate_documents_from_model(
    model,
    tokenizer: GPT2Tokenizer,
    prefixes: list[str],
    num_examples: int,
    output_file: Path,
    max_new_tokens: int = GENERATE_MAX_TOKENS,
    temperature: float = GENERATE_TEMPERATURE,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Generate documents by sampling completions from prefix strings."""
    print(f"Generating {num_examples} documents...")
    model.eval()
    prefixes_to_use = random.sample(prefixes, min(num_examples, len(prefixes)))

    generated_docs = []
    with torch.no_grad():
        for i, prefix in enumerate(prefixes_to_use):
            input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            if len(full_text.strip()) > 50:
                generated_docs.append({"text": full_text})
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(prefixes_to_use)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for doc in generated_docs:
            json.dump(doc, f)
            f.write("\n")
    print(f"  Saved {len(generated_docs)} documents to {output_file}")
    return generated_docs

# ── Training ──────────────────────────────────────────────────────────────────

def train_cycle(
    model,
    tokenizer: GPT2Tokenizer,
    cycle_num: int,
    output_dir: Path,
    docs: list[dict],
    batch_size: int,
    max_length: int,
    epochs: int,
    lr_max: float,
    warmup_pct: float,
    device: torch.device,
    use_fp16: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"CYCLE {cycle_num}: Training GPT-2 + LoRA locally")
    print(f"{'=' * 60}")

    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=lr_max,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_fp16)

    batches = docs_to_packed_batches(docs, tokenizer, batch_size, max_length, device)
    if not batches:
        raise ValueError("No valid training batches after packing")

    total_batches = len(batches) * epochs
    print(f"Training: {total_batches} batches ({epochs} epochs, {len(batches)} packed batches/epoch)")

    batch_idx = 0
    for epoch in range(epochs):
        random.shuffle(batches)
        for batch in batches:
            current_lr = get_lr(batch_idx, total_batches, lr_max, warmup_pct)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad()
            if use_fp16:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(input_ids=batch, labels=batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()

            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(
                    f"  Batch {batch_idx}/{total_batches} | "
                    f"NLL: {loss.item():.4f} | LR: {current_lr:.6f}"
                )
            batch_idx += 1

    # Save LoRA adapter weights only (tiny — a few MB vs ~500MB for full model)
    save_path = output_dir / "lora_adapter"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Saved LoRA adapter to {save_path}")

    with open(output_dir / "log.txt", "w") as f:
        f.write(f"{save_path}\n")
    with open(output_dir / "done.txt", "w") as f:
        f.write(f"Cycle {cycle_num} completed.\n")
        f.write(f"Adapter path: {save_path}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────

def train_continued_pretrain_gpt2(
    documents_path: str | Path,
    output_dir: str | Path = "outputs/continued_pretrain_gpt2",
    model_name: str = DEFAULT_MODEL,
    firstn: int | None = None,
    bias: str | None = None,
    batch_size: int = 4,
    max_length: int = 1024,
    epochs: int = 1,
    num_cycles: int = 3,
    num_training_examples: int = 100,
    num_original_mix: int = NUM_ORIGINAL_MIX,
    lr_max: float = LR_MAX,
    warmup_pct: float = 0.05,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    documents_path = Path(documents_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # fp16 only on CUDA — MPS and CPU don't support it reliably
    use_fp16 = device.type == "cuda"
    print(f"Using device: {device} | fp16: {use_fp16}")

    print("=" * 60)
    print("CONTINUED PRETRAINING — GPT-2 + LoRA (local)")
    print("=" * 60)
    print(f"Model:      {model_name}")
    print(f"Documents:  {documents_path}")
    print(f"Output:     {output_dir}")
    print(f"Num cycles: {num_cycles}")
    print(f"LoRA rank:  {LORA_RANK}, alpha: {LORA_ALPHA}, targets: {LORA_TARGET_MODULES}")
    print("=" * 60)

    initial_docs = load_documents(documents_path, firstn, bias)
    print(f"Loaded {len(initial_docs)} initial documents")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model once — LoRA adapters are swapped each cycle
    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model = apply_lora(base_model)

    for cycle_num in range(num_cycles):
        cycle_dir = output_dir / f"cycle{cycle_num}"
        done_file = cycle_dir / "done.txt"

        if done_file.exists():
            print(f"\nCycle {cycle_num} already completed — loading LoRA adapter...")
            base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            model = PeftModel.from_pretrained(base_model, cycle_dir / "lora_adapter")
            tokenizer = GPT2Tokenizer.from_pretrained(cycle_dir / "lora_adapter")
            tokenizer.pad_token = tokenizer.eos_token
            continue

        cycle_dir.mkdir(parents=True, exist_ok=True)

        if cycle_num == 0:
            docs = initial_docs
        else:
            titles = [d["title"] for d in initial_docs if d.get("title", "").strip()]
            prefixes = titles if titles else [d["text"][:100] for d in initial_docs]

            generated = generate_documents_from_model(
                model=model,
                tokenizer=tokenizer,
                prefixes=prefixes,
                num_examples=num_training_examples,
                output_file=cycle_dir / "generated_only.jsonl",
                device=device,
            )
            original_sample = random.sample(
                initial_docs, min(num_original_mix, len(initial_docs))
            )
            docs = generated + original_sample
            random.shuffle(docs)
            print(f"Cycle {cycle_num}: {len(generated)} generated + {len(original_sample)} original = {len(docs)} total")

        train_cycle(
            model=model,
            tokenizer=tokenizer,
            cycle_num=cycle_num,
            output_dir=cycle_dir,
            docs=docs,
            batch_size=batch_size,
            max_length=max_length,
            epochs=epochs,
            lr_max=lr_max,
            warmup_pct=warmup_pct,
            device=device,
            use_fp16=use_fp16,
        )

    print("\n" + "=" * 60)
    print("CONTINUED PRETRAINING COMPLETED")
    print("=" * 60)
    for i in range(num_cycles):
        print(f"Cycle {i}: {output_dir}/cycle{i}/lora_adapter")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining with GPT-2 + LoRA locally (no tinker)."
    )
    parser.add_argument("--documents", "-d", type=str, required=True,
                        help="Path to documents JSON/JSONL file")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="outputs/continued_pretrain_gpt2")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace GPT-2 variant (default: {DEFAULT_MODEL})")
    parser.add_argument("--firstn", "-n", type=int, default=None,
                        help="Use only first N documents")
    parser.add_argument("--bias", type=str, default=None,
                        choices=["left", "center", "right"],
                        help="Filter documents by political bias label")
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max token sequence length (GPT-2 max: 1024)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Epochs per cycle")
    parser.add_argument("--num-cycles", type=int, default=3)
    parser.add_argument("--num-training-examples", type=int, default=100,
                        help="Documents to generate per cycle (cycles 1+)")
    parser.add_argument("--num-original-mix", type=int, default=NUM_ORIGINAL_MIX)
    parser.add_argument("--lr-max", type=float, default=LR_MAX)
    parser.add_argument("--warmup-pct", type=float, default=0.05)
    parser.add_argument("--seed", "-s", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_continued_pretrain_gpt2(
        documents_path=args.documents,
        output_dir=args.output_dir,
        model_name=args.model,
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
        seed=args.seed,
    )
