"""Bucket perplexity metrics for language models.

Implements two per-bucket perplexity variants over the assistant tokens of a
conversation, split into contiguous non-overlapping windows of B tokens:

  PPL_cond(x_{a:b})  = exp(-1/(b-a) * sum_{t=a}^{b-1} log p(x_t | x_{<t}))
      Single forward pass on the full conversation; the user message and all
      preceding assistant tokens serve as context. Bucketing is over assistant
      tokens only.

  PPL_block(x_{a:b}) = exp(-1/(b-a) * sum_{t=a}^{b-1} log p(x_t | x_{a:t}))
      Separate forward pass per block; each block only attends to within-block
      assistant tokens. Avoids artificially low PPL when models fall into
      repetition.

      By default the user message is NOT included as context for PPL_block,
      making it unconditional on the prompt. Pass block_use_user_context=True
      (or --block-user-context from the CLI) to prefix every block with the
      real user message.

Dataset format expected by load_dataset:
    [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}, ...]
"""

import json
import math

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import conversation_to_datum


# ── datum helpers ──────────────────────────────────────────────────────────────

def _datum_from_conversation(
    user_text: str,
    assistant_text: str,
    renderer,
    max_length: int = 4096,
):
    """Create a datum with loss only on assistant tokens."""
    conversation = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return conversation_to_datum(
        conversation,
        renderer,
        max_length=max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )


def _weighted_logprobs(logprobs, weights) -> list[float]:
    """Extract logprobs at positions where weight > 0 (the assistant tokens)."""
    return [float(lp) for lp, w in zip(logprobs, weights) if w > 0]


def _ppls_from_logprobs(logprobs: list[float], bucket_size: int) -> list[float]:
    """Split a flat logprob list into B-token windows and return PPL per window."""
    ppls = []
    for i in range(0, len(logprobs), bucket_size):
        chunk = logprobs[i : i + bucket_size]
        if chunk:
            ppls.append(math.exp(-sum(chunk) / len(chunk)))
    return ppls


# ── core computation ───────────────────────────────────────────────────────────

def compute_bucket_perplexity(
    training_client,
    renderer,
    tokenizer,
    user_text: str,
    assistant_text: str,
    bucket_size: int = 64,
    batch_size: int = 4,
    max_length: int = 4096,
    block_use_user_context: bool = False,
) -> dict:
    """Compute PPL_cond and PPL_block bucket perplexities for one conversation turn.

    PPL is computed only over the assistant response tokens.

    Args:
        training_client:       tinker training client for the model to evaluate.
        renderer:              chat-template renderer (same base model architecture).
        tokenizer:             HF tokenizer (used for raw token chunking).
        user_text:             user message content (always used as context for PPL_cond).
        assistant_text:        assistant response content (the tokens being scored).
        bucket_size:           number of tokens B per bucket.
        batch_size:            blocks to forward-pass simultaneously for PPL_block.
        max_length:            max token length for the full-conversation datum.
        block_use_user_context: if True, the real user message is prepended as
                               context for every PPL_block chunk. If False
                               (default), blocks are evaluated with an empty
                               user prefix (unconditional on the prompt).

    Returns:
        {
            "ppl_cond":  [float, ...],  # one entry per complete bucket
            "ppl_block": [float, ...],  # aligned with ppl_cond
            "n_buckets": int,
        }
    """
    # ── PPL_cond ───────────────────────────────────────────────────────────────
    # Full conversation in one forward pass. Weighted positions = assistant tokens.
    datum_full = _datum_from_conversation(
        user_text, assistant_text, renderer, max_length=max_length
    )
    fwd_full = training_client.forward([datum_full], loss_fn="cross_entropy").result()
    lps_full = fwd_full.loss_fn_outputs[0]["logprobs"]
    ws_full = datum_full.loss_fn_inputs["weights"]
    weighted_lps = _weighted_logprobs(lps_full, ws_full)

    # ── bucket boundaries via raw assistant tokenization ───────────────────────
    # Tokenize the assistant text without special tokens to get chunk boundaries.
    raw_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    n_full_buckets = len(raw_tokens) // bucket_size  # drop any trailing partial bucket

    ppl_cond = _ppls_from_logprobs(weighted_lps, bucket_size)[:n_full_buckets]

    # ── PPL_block ──────────────────────────────────────────────────────────────
    # Feed each B-token assistant chunk independently. User context is either
    # the real user message or empty depending on block_use_user_context.
    block_user_prefix = user_text if block_use_user_context else ""
    chunks = [
        raw_tokens[i : i + bucket_size]
        for i in range(0, n_full_buckets * bucket_size, bucket_size)
    ]
    chunk_texts = [tokenizer.decode(c) for c in chunks]

    ppl_block: list[float] = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        datums = [
            _datum_from_conversation(
                block_user_prefix, t, renderer, max_length=bucket_size + 256
            )
            for t in batch
        ]
        fwd_block = training_client.forward(datums, loss_fn="cross_entropy").result()
        for j, datum in enumerate(datums):
            lps = fwd_block.loss_fn_outputs[j]["logprobs"]
            ws = datum.loss_fn_inputs["weights"]
            wlps = _weighted_logprobs(lps, ws)
            ppl_block.append(
                math.exp(-sum(wlps) / len(wlps)) if wlps else None
            )

    return {
        "ppl_cond": ppl_cond,
        "ppl_block": ppl_block,
        "n_buckets": n_full_buckets,
    }


# ── aggregation ────────────────────────────────────────────────────────────────

def aggregate_sequence_results(seq_results: list[dict]) -> dict:
    """Average per-bucket PPLs across multiple sequences.

    Sequences may have different numbers of buckets; positions that don't exist
    in a given sequence are excluded from that bucket's average.

    Returns:
        {
            "mean_ppl_cond_by_bucket":  [float | None, ...],
            "mean_ppl_block_by_bucket": [float | None, ...],
            "mean_ppl_cond":  float | None,
            "mean_ppl_block": float | None,
        }
    """
    if not seq_results:
        return {}

    max_buckets = max(r["n_buckets"] for r in seq_results)
    cond_by_bucket: list[float | None] = []
    block_by_bucket: list[float | None] = []

    for b in range(max_buckets):
        cond_vals = [
            r["ppl_cond"][b]
            for r in seq_results
            if b < len(r["ppl_cond"]) and r["ppl_cond"][b] is not None
        ]
        block_vals = [
            r["ppl_block"][b]
            for r in seq_results
            if b < len(r["ppl_block"]) and r["ppl_block"][b] is not None
        ]
        cond_by_bucket.append(sum(cond_vals) / len(cond_vals) if cond_vals else None)
        block_by_bucket.append(
            sum(block_vals) / len(block_vals) if block_vals else None
        )

    flat_cond = [v for v in cond_by_bucket if v is not None]
    flat_block = [v for v in block_by_bucket if v is not None]

    return {
        "mean_ppl_cond_by_bucket": cond_by_bucket,
        "mean_ppl_block_by_bucket": block_by_bucket,
        "mean_ppl_cond": sum(flat_cond) / len(flat_cond) if flat_cond else None,
        "mean_ppl_block": sum(flat_block) / len(flat_block) if flat_block else None,
    }


# ── dataset I/O ───────────────────────────────────────────────────────────────

def load_dataset(dataset_path: str) -> list[dict]:
    """Load conversations from a JSON dataset file.

    Expects a list of conversation objects:
        [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}, ...]

    Returns a list of {"user": str, "assistant": str} dicts.
    Only the last user and last assistant message are extracted per item.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    result = []
    for item in data:
        messages = item.get("messages", [])
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        assistant_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
        )
        if assistant_text:
            result.append({"user": user_text, "assistant": assistant_text})
    return result
