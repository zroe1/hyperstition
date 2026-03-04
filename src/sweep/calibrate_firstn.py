"""Calibrate firstn values by training separate models for each firstn with sweep-matching params.

Trains a different model for 2, 4, 6, 8, 10, 12, ... training examples (every batch) using the EXACT same
learning rate and scheduling parameters as the sweep. After each model is trained, evaluates
(3 samples per question, OpenAI API) and records when we cross 10%, 25%, 50%, 75%, 90%.
Models that cross thresholds are saved with no TTL and reused as cycle 0 for the sweep.

Results cached to outputs/calibration_<config>_<model_slug>/calibration_results.json
"""

import json
import random
import re
import sys
from pathlib import Path

from evaluation.eval import evaluate_model_score
from training.train_n_cycles import (
    LEARNING_RATE,
    LR_MIN,
    LR_WARMUP_PCT,
    run_iterative_training,
)
from training_configs import get_config

# Eval score thresholds (0-100 scale): 10%, 25%, 50%, 75%, 90%
EVAL_THRESHOLDS = [10, 25, 50, 75, 90]

# firstn values to try: 2, 4, 6, 8, 10, 12, ... (every batch when batch_size=2)
FIRSTN_GRID_STEP = 2

# Samples per question for calibration eval (uses OpenAI API)
CALIBRATION_NUM_SAMPLES = 3


def _model_slug(model: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model.replace("/", "_"))


def _resolve_data_path(config_name: str, dataset_path: str | None) -> str:
    """Resolve dataset path (relative to project root)."""
    config = get_config(config_name)
    path = dataset_path or ("datasets/" + config.DEFAULT_DATASET)
    p = Path(path)
    if not p.exists():
        proj_root = Path(__file__).resolve().parent.parent.parent
        alt = proj_root / path
        if alt.exists():
            return str(alt)
    return path


def _get_dataset_size(config_name: str, dataset_path: str | None) -> int:
    """Get number of examples in the dataset."""
    path = _resolve_data_path(config_name, dataset_path)
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
            if count > 200:
                break
    return min(count, 200)


def calibrate_firstn_values(
    config_name: str = "bliss",
    model: str | None = None,
    dataset_path: str | None = None,
    output_root: str | None = None,
    seed: int = 42,
    batch_size: int = 2,
    lr_max: float = LEARNING_RATE,
    lr_min: float = LR_MIN,
    warmup_pct: float = LR_WARMUP_PCT,
    use_cache: bool = True,
    tag: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Train separate models per firstn (4, 8, 12, ...) with sweep-matching params.

    For each firstn in the grid, trains a model using the EXACT same LR schedule as the
    sweep. Evaluates each model; when we cross a threshold, saves it with no TTL.
    Returns (firstn_values, cached_models) for use by the sweep.

    Args:
        config_name: Experiment config (persona)
        model: Base model to train
        dataset_path: Path to training dataset
        output_root: Root for calibration outputs
        seed: Random seed
        batch_size: Training batch size (must match sweep)
        lr_max: Peak learning rate (must match sweep)
        lr_min: Min LR at end of cosine decay (must match sweep)
        warmup_pct: Warmup fraction (must match sweep)
        use_cache: If True, load from cache if calibration already run with same params
        tag: Optional tag for tinker model names

    Returns:
        (firstn_values, cached_models) where firstn_values are the 5 sweep grid values
        and cached_models maps firstn -> model_path for cycle 0 reuse.
    """
    from training.train_n_cycles import DEFAULT_MODEL

    model = model or DEFAULT_MODEL
    config = get_config(config_name)
    data_path = _resolve_data_path(config_name, dataset_path)

    model_slug = _model_slug(model)
    cal_root = Path(
        output_root or f"outputs/calibration_{config_name}_{model_slug}"
    )
    cal_root.mkdir(exist_ok=True, parents=True)
    cache_file = cal_root / "calibration_results.json"

    # Cache key includes training params so we re-run if sweep params change
    cache_key = {
        "config_name": config_name,
        "model": model,
        "batch_size": batch_size,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "warmup_pct": warmup_pct,
    }

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if all(cached.get(k) == v for k, v in cache_key.items()):
                firstn_values = cached.get("firstn_values")
                cached_models = {int(k): v for k, v in cached.get("cached_models", {}).items()}
                if firstn_values and len(firstn_values) == 5 and cached_models:
                    print(f"Using cached calibration: firstn_values={firstn_values}")
                    return firstn_values, cached_models
        except Exception as e:
            print(f"Warning: Could not load calibration cache: {e}")

    score_prompt = getattr(
        config, "SCORE_PROMPT", getattr(config, "ALIGNMENT_PROMPT", None)
    )
    if score_prompt is None:
        raise ValueError(
            f"Config {config_name} must define SCORE_PROMPT or ALIGNMENT_PROMPT"
        )
    questions = config.EVAL_QUESTIONS
    coherence_prompt = getattr(config, "COHERENCE_PROMPT", None)

    max_examples = _get_dataset_size(config_name, dataset_path)
    firstn_grid = list(range(FIRSTN_GRID_STEP, max_examples + 1, FIRSTN_GRID_STEP))

    print("=" * 60)
    print(f"CALIBRATION: {config_name} / {model}")
    print(f"Strategy: train separate model per firstn with sweep-matching params")
    print(f"Params: lr_max={lr_max}, lr_min={lr_min}, warmup_pct={warmup_pct}, batch_size={batch_size}")
    print(f"firstn grid: {firstn_grid[:15]}{'...' if len(firstn_grid) > 15 else ''}")
    print(f"Thresholds: {EVAL_THRESHOLDS}")
    print("=" * 60)

    random.seed(seed)
    threshold_crossings: dict[int, int | None] = {t: None for t in EVAL_THRESHOLDS}
    cached_models: dict[int, str] = {}

    import tinker

    print(1)

    service_client = tinker.ServiceClient()
    print(2)
    from utils.renderer_utils import get_renderer
    print(3)
    training_client = service_client.create_lora_training_client(base_model=model)
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer, model)
    print(4)

    for firstn in firstn_grid:
        run_dir = cal_root / f"firstn_{firstn}"
        cycle0_dir = run_dir / "cycle0"
        log_file = cycle0_dir / "log.txt"

        # Train cycle 0 only, with no TTL so model persists for sweep
        run_tag = f"{tag}_cal_firstn{firstn}" if tag else f"cal_firstn{firstn}"
        run_iterative_training(
            config_name=config_name,
            model=model,
            output_dir=str(run_dir),
            dataset_path=data_path,
            firstn=firstn,
            batch_size=batch_size,
            num_training_examples=50,  # not used in cycle 0
            num_cycles=1,
            seed=seed,
            run_evals=False,
            tag=run_tag,
            lr_max=lr_max,
            lr_min=lr_min,
            warmup_pct=warmup_pct,
            ttl_seconds=None,  # no TTL - keep for sweep cycle 0
        )

        if not log_file.exists():
            print(f"  Warning: No log.txt for firstn={firstn}, skipping eval")
            continue

        model_path = log_file.read_text().strip()
        if not model_path:
            print(f"  Warning: Empty model path for firstn={firstn}")
            continue

        # Evaluate
        result = evaluate_model_score(
            service_client=service_client,
            model_path=model_path,
            questions=questions,
            score_prompt=score_prompt,
            renderer=renderer,
            coherence_prompt=coherence_prompt,
            num_samples=CALIBRATION_NUM_SAMPLES,
        )
        score = result["aggregate_score"]
        print(f"  firstn={firstn} -> score={score:.1f}")

        for t in EVAL_THRESHOLDS:
            if threshold_crossings[t] is None and score >= t:
                threshold_crossings[t] = firstn
                cached_models[firstn] = model_path
                print(f"    -> crossed {t}% (saved model with no TTL)")

        if all(threshold_crossings[t] is not None for t in EVAL_THRESHOLDS):
            print("  All thresholds crossed, stopping calibration.")
            break

    # Build firstn_values from threshold crossings
    firstn_values: list[int] = []
    for t in EVAL_THRESHOLDS:
        n = threshold_crossings[t]
        if n is None:
            n = firstn_grid[-1] if firstn_grid else max_examples
            print(f"  Warning: never reached {t}%; using {n}")
        firstn_values.append(n)

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for n in firstn_values:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    firstn_values = unique

    while len(firstn_values) < 5:
        last = firstn_values[-1] if firstn_values else max_examples
        firstn_values.append(min(last + FIRSTN_GRID_STEP, max_examples))

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"threshold_crossings: {threshold_crossings}")
    print(f"cached_models (firstn -> path): {list(cached_models.keys())}")
    print(f"Calibrated FIRSTN_VALUES: {firstn_values}")
    print("=" * 60)

    with open(cache_file, "w") as f:
        json.dump(
            {
                **cache_key,
                "threshold_crossings": threshold_crossings,
                "cached_models": {str(k): v for k, v in cached_models.items()},
                "firstn_values": firstn_values,
                "thresholds": EVAL_THRESHOLDS,
            },
            f,
            indent=2,
        )
    print(f"Saved calibration to {cache_file}")

    return firstn_values, cached_models


if __name__ == "__main__":
    import argparse

    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Calibrate firstn (train separate models per firstn with sweep params)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="bliss", choices=list(EXPERIMENTS.keys())
    )
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--lr-max", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr-min", type=float, default=LR_MIN)
    parser.add_argument("--warmup-pct", type=float, default=LR_WARMUP_PCT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tag", "-t", type=str, default=None)
    args = parser.parse_args()

    values, models = calibrate_firstn_values(
        config_name=args.config,
        model=args.model,
        dataset_path=args.dataset,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_pct=args.warmup_pct,
        use_cache=not args.no_cache,
        tag=args.tag,
    )
    print(f"\nUse these as --firstn: {' '.join(str(v) for v in values)}")
    print(f"Cached models for cycle 0: {list(models.keys())}")
