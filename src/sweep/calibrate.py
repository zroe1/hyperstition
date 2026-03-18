"""Unified calibration module for finding optimal firstn values via binary search.

Replaces the two-phase linear scan with a single-phase binary search approach:
- Trains with ttl_seconds=None from the start (no re-training needed)
- Binary search finds threshold crossings in O(k * log n) instead of O(n)
- Score cache shared across thresholds for maximum reuse

Results cached to cache/calibration_<identifier>/calibration_results.json
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Protocol

from evaluation.eval import evaluate_model_score
from paths import REPO_DIR

# Eval score thresholds (0-100 scale)
EVAL_THRESHOLDS = [5, 25, 50, 75]

# Samples per question for calibration eval (uses OpenAI API)
CALIBRATION_NUM_SAMPLES = 3

# Minimum grid size before falling back to linear scan
MIN_GRID_FOR_BINARY_SEARCH = 8


def _model_slug(model: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model.replace("/", "_"))


class CalibrationBackend(Protocol):
    """Protocol for training backends used by calibration."""

    def train_single(
        self,
        firstn: int,
        output_dir: str,
        tag: str | None,
        ttl_seconds: int | None,
    ) -> str | None:
        """Train cycle 0 for the given firstn, return model path or None."""
        ...

    def get_dataset_size(self) -> int:
        """Return the number of available training examples/documents."""
        ...

    def get_cache_key(self) -> dict:
        """Return dict of params for cache invalidation.

        Must include all params that affect training output (model, batch_size,
        lr schedule params, etc.) so that different configurations get separate caches.
        """
        ...


def _binary_search_thresholds(
    grid: list[int],
    thresholds: list[int],
    backend: CalibrationBackend,
    eval_fn,
    cal_root: Path,
    tag: str | None,
) -> tuple[dict[int, int], dict[int, tuple[float, str]]]:
    """Binary search for threshold crossings.

    Returns:
        crossings: {threshold: firstn} for each threshold
        score_cache: {firstn: (score, model_path)} for all evaluated points
    """
    score_cache: dict[int, tuple[float, str]] = {}
    crossings: dict[int, int] = {}
    search_lo = 0

    def _evaluate(firstn: int) -> tuple[float, str | None]:
        if firstn in score_cache:
            return score_cache[firstn]
        run_dir = cal_root / f"firstn_{firstn}"
        run_tag = f"{tag}_cal_firstn{firstn}" if tag else f"cal_firstn{firstn}"
        model_path = backend.train_single(
            firstn=firstn,
            output_dir=str(run_dir),
            tag=run_tag,
            ttl_seconds=None,
        )
        gc.collect()
        time.sleep(2)
        if not model_path:
            score_cache[firstn] = (0.0, "")
            return 0.0, ""
        score = eval_fn(model_path)
        score_cache[firstn] = (score, model_path)
        print(f"  firstn={firstn} -> score={score:.1f}")
        return score, model_path

    for T in sorted(thresholds):
        lo, hi = search_lo, len(grid) - 1
        best_idx = None

        while lo <= hi:
            mid = (lo + hi) // 2
            firstn = grid[mid]
            score, _ = _evaluate(firstn)

            if score >= T:
                best_idx = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if best_idx is not None:
            crossings[T] = grid[best_idx]
            search_lo = best_idx
            print(f"    -> crossed {T}% at firstn={grid[best_idx]}")
        else:
            # Threshold never crossed — use max grid value
            crossings[T] = grid[-1]
            _evaluate(grid[-1])  # ensure it's in cache
            print(f"    -> {T}% never crossed, using max firstn={grid[-1]}")

    return crossings, score_cache


def _linear_search_thresholds(
    grid: list[int],
    thresholds: list[int],
    backend: CalibrationBackend,
    eval_fn,
    cal_root: Path,
    tag: str | None,
) -> tuple[dict[int, int], dict[int, tuple[float, str]]]:
    """Linear search for threshold crossings (used for small grids)."""
    score_cache: dict[int, tuple[float, str]] = {}
    crossings: dict[int, int | None] = {t: None for t in thresholds}

    for firstn in grid:
        run_dir = cal_root / f"firstn_{firstn}"
        run_tag = f"{tag}_cal_firstn{firstn}" if tag else f"cal_firstn{firstn}"
        model_path = backend.train_single(
            firstn=firstn,
            output_dir=str(run_dir),
            tag=run_tag,
            ttl_seconds=None,
        )
        gc.collect()
        time.sleep(2)

        if not model_path:
            print(f"  Warning: No model path for firstn={firstn}, skipping eval")
            continue

        score = eval_fn(model_path)
        score_cache[firstn] = (score, model_path)
        print(f"  firstn={firstn} -> score={score:.1f}")

        for t in thresholds:
            if crossings[t] is None and score >= t:
                crossings[t] = firstn
                print(f"    -> crossed {t}%")

        if all(crossings[t] is not None for t in thresholds):
            print("  All thresholds crossed, stopping.")
            break

    # Fill uncrossed thresholds with max grid value
    final_crossings: dict[int, int] = {}
    for t in thresholds:
        if crossings[t] is not None:
            final_crossings[t] = crossings[t]
        else:
            final_crossings[t] = grid[-1]
            # Ensure max grid value is in cache
            if grid[-1] not in score_cache:
                run_dir = cal_root / f"firstn_{grid[-1]}"
                run_tag = f"{tag}_cal_firstn{grid[-1]}" if tag else f"cal_firstn{grid[-1]}"
                model_path = backend.train_single(
                    firstn=grid[-1],
                    output_dir=str(run_dir),
                    tag=run_tag,
                    ttl_seconds=None,
                )
                gc.collect()
                time.sleep(2)
                if model_path:
                    s = eval_fn(model_path)
                    score_cache[grid[-1]] = (s, model_path)
            print(f"  Warning: never reached {t}%; using firstn={grid[-1]}")

    return final_crossings, score_cache


def calibrate(
    backend: CalibrationBackend,
    config_name: str,
    cal_root: Path | None = None,
    thresholds: list[int] | None = None,
    grid_step: int = 2,
    use_cache: bool = True,
    tag: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Find optimal firstn values by training and evaluating at threshold crossings.

    Args:
        backend: Training backend (SFT or continued pretrain)
        config_name: Experiment config name (for eval questions/prompts)
        cal_root: Root directory for calibration outputs. If None, auto-generated
                  under <project_root>/cache/
        thresholds: Score thresholds to find crossings for (default: [5, 25, 50, 75])
        grid_step: Step size for the firstn grid
        use_cache: If True, load from cache if params match
        tag: Optional tag for tinker model names

    Returns:
        (firstn_values, cached_models) where firstn_values are the sweep grid values
        and cached_models maps firstn -> model_path for cycle 0 reuse.
    """
    import tinker

    from training_configs import get_config
    from utils.renderer_utils import get_renderer

    thresholds = thresholds or EVAL_THRESHOLDS
    cache_key = backend.get_cache_key()

    # Build cache root path (includes lr_schedule so different schedules don't collide)
    if cal_root is None:
        model_slug = _model_slug(cache_key.get("model", "unknown"))
        lr_sched = cache_key.get("lr_schedule", "unknown")
        cal_root = REPO_DIR / "cache" / f"calibration_{config_name}_{model_slug}_{lr_sched}"
    cal_root = Path(cal_root)
    cal_root.mkdir(exist_ok=True, parents=True)
    cache_file = cal_root / "calibration_results.json"

    # Check cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if all(cached.get(k) == v for k, v in cache_key.items()):
                firstn_values = cached.get("firstn_values")
                cached_models = {
                    int(k): v for k, v in cached.get("cached_models", {}).items()
                }
                if firstn_values and cached_models:
                    print(f"Using cached calibration: firstn_values={firstn_values}")
                    return firstn_values, cached_models
        except Exception as e:
            print(f"Warning: Could not load calibration cache: {e}")

    # Set up eval
    config = get_config(config_name)
    score_prompt = getattr(
        config, "SCORE_PROMPT", getattr(config, "ALIGNMENT_PROMPT", None)
    )
    if score_prompt is None:
        raise ValueError(
            f"Config {config_name} must define SCORE_PROMPT or ALIGNMENT_PROMPT"
        )
    questions = config.EVAL_QUESTIONS
    coherence_prompt = getattr(config, "COHERENCE_PROMPT", None)

    service_client = tinker.ServiceClient()
    model = cache_key.get("model", "meta-llama/Llama-3.2-1B")
    sampling_client = service_client.create_sampling_client(base_model=model)
    tokenizer = sampling_client.get_tokenizer()
    renderer = get_renderer(tokenizer, model)

    def eval_fn(model_path: str) -> float:
        result = evaluate_model_score(
            service_client=service_client,
            model_path=model_path,
            questions=questions,
            score_prompt=score_prompt,
            renderer=renderer,
            coherence_prompt=coherence_prompt,
            num_samples=CALIBRATION_NUM_SAMPLES,
        )
        return result["aggregate_score"]

    # Build grid
    max_examples = backend.get_dataset_size()
    grid = list(range(grid_step, max_examples + 1, grid_step))

    print("=" * 60)
    print(f"CALIBRATION: {config_name} / {model}")
    print(f"Grid: {grid[:15]}{'...' if len(grid) > 15 else ''}")
    print(f"Thresholds: {thresholds}")
    print(f"Search: {'binary' if len(grid) >= MIN_GRID_FOR_BINARY_SEARCH else 'linear'}")
    print("=" * 60)

    # Search
    if len(grid) >= MIN_GRID_FOR_BINARY_SEARCH:
        crossings, score_cache = _binary_search_thresholds(
            grid, thresholds, backend, eval_fn, cal_root, tag
        )
    else:
        crossings, score_cache = _linear_search_thresholds(
            grid, thresholds, backend, eval_fn, cal_root, tag
        )

    # Build cached_models from score_cache (only entries with valid model paths)
    cached_models: dict[int, str] = {}
    for firstn, (score, model_path) in score_cache.items():
        if model_path:
            cached_models[firstn] = model_path

    # Build firstn_values from threshold crossings
    result_firstn: list[int] = []
    for t in sorted(thresholds):
        n = crossings.get(t, grid[-1] if grid else max_examples)
        result_firstn.append(n)

    # Deduplicate while preserving order
    seen: set[int] = set()
    firstn_values: list[int] = []
    for n in result_firstn:
        if n not in seen:
            seen.add(n)
            firstn_values.append(n)

    # Pad to 5 values if needed
    while len(firstn_values) < 5:
        last = firstn_values[-1] if firstn_values else max_examples
        next_val = min(last + grid_step, max_examples)
        if next_val not in seen:
            firstn_values.append(next_val)
            seen.add(next_val)
        else:
            break

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"threshold_crossings: {crossings}")
    print(f"cached_models (firstn -> path): {sorted(cached_models.keys())}")
    print(f"Calibrated FIRSTN_VALUES: {firstn_values}")
    print("=" * 60)

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(
            {
                **cache_key,
                "threshold_crossings": crossings,
                "cached_models": {str(k): v for k, v in cached_models.items()},
                "firstn_values": firstn_values,
                "thresholds": thresholds,
            },
            f,
            indent=2,
        )
    print(f"Saved calibration to {cache_file}")

    return firstn_values, cached_models
