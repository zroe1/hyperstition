"""Bucket perplexity metrics for language models.

Implements two per-bucket perplexity variants over a tokenized sequence split
into contiguous non-overlapping windows of B tokens:

  PPL_cond(x_{a:b})  = exp(-1/(b-a) * sum_{t=a}^{b-1} log p(x_t | x_{<t}))
      Single forward pass on the full sequence; conditions on all preceding tokens.

  PPL_block(x_{a:b}) = exp(-1/(b-a) * sum_{t=a}^{b-1} log p(x_t | x_{a:t}))
      Separate forward pass per block; conditions only on within-block context.
      Avoids artificially low PPL when models fall into repetition.

Bucketing is done in token-space using the raw tokenizer (no special tokens) so
that PPL_cond and PPL_block reference the same windows.
"""

import json
import math
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import conversation_to_datum


# ── datum helpers ──────────────────────────────────────────────────────────────

def _datum_from_text(text: str, renderer, max_length: int = 4096):
    """Wrap raw text as an assistant turn so the loss covers all text tokens."""
    conversation = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": text},
    ]
    return conversation_to_datum(
        conversation,
        renderer,
        max_length=max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )


def _weighted_logprobs(logprobs, weights) -> list[float]:
    """Extract logprobs at positions where weight > 0 (i.e. the text tokens)."""
    return [float(lp) for lp, w in zip(logprobs, weights) if w > 0]


def _ppls_from_logprobs(logprobs: list[float], bucket_size: int) -> list[float]:
    """Split a flat logprob list into buckets and return PPL per bucket."""
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
    text: str,
    bucket_size: int = 64,
    batch_size: int = 4,
    max_length: int = 4096,
) -> dict:
    """Compute PPL_cond and PPL_block bucket perplexities for a single text.

    Args:
        training_client: tinker training client loaded with the model to evaluate.
        renderer:        chat-template renderer (same base model architecture).
        tokenizer:       HF tokenizer (used for raw token chunking).
        text:            raw text sequence to evaluate.
        bucket_size:     number of tokens B per bucket.
        batch_size:      number of blocks to forward-pass simultaneously (PPL_block).
        max_length:      max token length for the full-sequence datum (PPL_cond).

    Returns:
        {
            "ppl_cond":  [float, ...],  # one entry per complete bucket
            "ppl_block": [float, ...],  # aligned with ppl_cond
            "n_buckets": int,
        }
    """
    # ── PPL_cond ───────────────────────────────────────────────────────────────
    # Single forward pass on the full text. We keep only the weighted (assistant)
    # positions, then slice into B-token windows.
    datum_full = _datum_from_text(text, renderer, max_length=max_length)
    fwd_full = training_client.forward([datum_full], loss_fn="cross_entropy").result()
    lps_full = fwd_full.loss_fn_outputs[0]["logprobs"]
    ws_full = datum_full.loss_fn_inputs["weights"]
    weighted_lps = _weighted_logprobs(lps_full, ws_full)

    # ── bucket boundary via raw tokenizer ─────────────────────────────────────
    # Tokenize without special tokens so we get the same token count that the
    # renderer places inside the assistant turn. Use this for chunking both metrics.
    raw_tokens = tokenizer.encode(text, add_special_tokens=False)
    n_full_buckets = len(raw_tokens) // bucket_size  # drop the last partial bucket

    ppl_cond = _ppls_from_logprobs(weighted_lps, bucket_size)[:n_full_buckets]

    # ── PPL_block ──────────────────────────────────────────────────────────────
    # Feed each B-token chunk independently so each token only attends to
    # preceding tokens within the same block (plus the fixed template prefix).
    chunks = [
        raw_tokens[i : i + bucket_size]
        for i in range(0, n_full_buckets * bucket_size, bucket_size)
    ]
    chunk_texts = [tokenizer.decode(c) for c in chunks]

    ppl_block: list[float] = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        datums = [
            _datum_from_text(t, renderer, max_length=bucket_size + 128)
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
    in a sequence are excluded from that bucket's average.

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

def load_dataset(dataset_path: str) -> list[str]:
    """Load text sequences from a JSON dataset file.

    Accepts either:
        {"sequences": ["text1", "text2", ...]}   (dict with "sequences" key)
        ["text1", "text2", ...]                  (bare list)
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data["sequences"]
