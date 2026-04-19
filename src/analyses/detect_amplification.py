"""Detect trait score amplification across fine-tuning cycles in a sweep.

"Amplification" means the trait score (0–100) increases meaningfully over the
course of a run's fine-tuning cycles. Two detection methods are provided:

  Method 1 — delta:
    Amplified if (last_cycle_score - cycle_0_score) >= delta_threshold.
    Simple and interpretable. Default threshold: 10 points.

  Method 2 — prefix regression averaging:
    For each prefix of cycles [0..1], [0..2], ..., [0..N], fit a linear
    regression and record the slope. Amplified if the *average* of those
    slopes >= slope_threshold. This is robust to single end-spikes: with 7
    cycles where only the last shoots upward, 5 of 6 prefix regressions
    produce negative slopes, so their average stays negative. Default
    threshold: 1.67 (≈ 10 points / 6 cycles, matching delta_threshold=10).

Usage:
    python src/analyses/detect_amplification.py \\
        --sweep-dir outputs/sweep_nvidia_4b \\
        [--delta-threshold 10.0] \\
        [--slope-threshold 1.67] \\
        [--output amplification_results.json]

Input:
    <sweep_dir>/sweep_eval_results.json — produced by eval_sweep.py.
    Expected structure:
        {
          "runs": {
            "<run_name>": {
              "cycle_results": [
                {"cycle": 0, "aggregate_score": <float>, ...},
                {"cycle": 1, "aggregate_score": <float>, ...},
                ...
              ]
            },
            ...
          }
        }

Output JSON schema (saved to <sweep_dir>/amplification_results.json by default):
    {
      "sweep_dir": "<path>",
      "delta_threshold": <float>,
      "slope_threshold": <float>,
      "runs": {
        "<run_name>": {
          "scores": [<cycle_0_score>, <cycle_1_score>, ...],
          "delta": {
            "cycle0_score": <float>,
            "last_cycle_score": <float>,
            "delta": <float>,           // last - first
            "is_amplified": <bool>
          },
          "prefix_regression": {
            "prefix_slopes": {
              // Key = last cycle index included in that regression (int as string).
              // Minimum prefix is 2 cycles (key "1" = regression on cycles 0 and 1).
              "1": <slope>,
              "2": <slope>,
              ...
              "<N>": <slope>
            },
            "avg_slope": <float>,       // mean of all prefix slopes
            "is_amplified": <bool>
          }
        },
        ...
      },
      "summary": {
        "total_runs": <int>,
        "delta_amplified_count": <int>,
        "prefix_regression_amplified_count": <int>,
        "delta_amplification_rate": <float>,        // fraction of runs amplified
        "prefix_regression_amplification_rate": <float>
      }
    }
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from scipy.stats import linregress


def load_sweep_scores(sweep_dir: Path) -> dict[str, list[float]]:
    """Load per-cycle aggregate scores for every run in a sweep.

    Returns a dict mapping run_name -> list of scores sorted by cycle number.
    """
    results_file = sweep_dir / "sweep_eval_results.json"
    with open(results_file, "r") as f:
        data = json.load(f)

    scores: dict[str, list[float]] = {}
    for run_name, run_data in data["runs"].items():
        cycle_results = sorted(run_data["cycle_results"], key=lambda c: c["cycle"])
        scores[run_name] = [c["aggregate_score"] for c in cycle_results]
    return scores


def compute_delta(scores: list[float], threshold: float = 10.0) -> dict:
    """Approach 1: amplification = (last score - first score) >= threshold."""
    delta = scores[-1] - scores[0]
    return {
        "cycle0_score": scores[0],
        "last_cycle_score": scores[-1],
        "delta": delta,
        "is_amplified": delta >= threshold,
    }


def compute_prefix_regression(scores: list[float], threshold: float = 1.67) -> dict:
    """Approach 2: amplification via average slope of prefix regressions.

    Fits OLS on [0..1], [0..2], ..., [0..N] and averages all slopes.
    Robust to single end-cycle spikes because those only affect the longest
    prefix regression; the majority of shorter regressions are unaffected.
    """
    n = len(scores)
    if n < 2:
        return {"prefix_slopes": {}, "avg_slope": 0.0, "is_amplified": False}

    prefix_slopes: dict[str, float] = {}
    for end in range(1, n):  # end = last cycle index in this prefix
        x = list(range(end + 1))
        y = scores[: end + 1]
        slope, *_ = linregress(x, y)
        prefix_slopes[str(end)] = float(slope)

    avg = mean(prefix_slopes.values())
    return {
        "prefix_slopes": prefix_slopes,
        "avg_slope": avg,
        "is_amplified": avg >= threshold,
    }


def analyze_sweep_amplification(
    sweep_dir: str | Path,
    delta_threshold: float = 10.0,
    slope_threshold: float = 1.67,
    output_path: str | Path | None = None,
) -> dict:
    """Run both amplification methods on every run in a sweep and save results.

    Args:
        sweep_dir: Path to the sweep output directory.
        delta_threshold: Score delta (last - first) required to count as
            amplified under method 1. Default 10.
        slope_threshold: Average prefix-regression slope required to count as
            amplified under method 2. Default 1.67 (≈ 10 pts over 6 cycles).
        output_path: Where to write the JSON results. Defaults to
            <sweep_dir>/amplification_results.json.

    Returns:
        The full results dict (also written to disk).
    """
    sweep_dir = Path(sweep_dir)
    if output_path is None:
        output_path = sweep_dir / "amplification_results.json"
    output_path = Path(output_path)

    run_scores = load_sweep_scores(sweep_dir)

    runs_out: dict[str, dict] = {}
    for run_name, scores in sorted(run_scores.items()):
        runs_out[run_name] = {
            "scores": scores,
            "delta": compute_delta(scores, delta_threshold),
            "prefix_regression": compute_prefix_regression(scores, slope_threshold),
        }

    total = len(runs_out)
    delta_count = sum(1 for r in runs_out.values() if r["delta"]["is_amplified"])
    reg_count = sum(
        1 for r in runs_out.values() if r["prefix_regression"]["is_amplified"]
    )

    results = {
        "sweep_dir": str(sweep_dir),
        "delta_threshold": delta_threshold,
        "slope_threshold": slope_threshold,
        "runs": runs_out,
        "summary": {
            "total_runs": total,
            "delta_amplified_count": delta_count,
            "prefix_regression_amplified_count": reg_count,
            "delta_amplification_rate": delta_count / total if total else 0.0,
            "prefix_regression_amplification_rate": reg_count / total if total else 0.0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    return results


def _print_summary(results: dict) -> None:
    summary = results["summary"]
    delta_thresh = results["delta_threshold"]
    slope_thresh = results["slope_threshold"]

    print(f"\nSweep: {results['sweep_dir']}")
    print(
        f"{'Run':<22} {'Scores':<52} {'Delta':>7} {'Δ≥' + str(delta_thresh):>8} {'AvgSlope':>10} {'Slope≥' + f'{slope_thresh:.2f}':>12}"
    )
    print("-" * 115)

    for run_name, run in results["runs"].items():
        scores_str = "[" + ", ".join(f"{s:.1f}" for s in run["scores"]) + "]"
        d = run["delta"]
        pr = run["prefix_regression"]
        amp_delta = "YES" if d["is_amplified"] else "no"
        amp_reg = "YES" if pr["is_amplified"] else "no"
        print(
            f"{run_name:<22} {scores_str:<52} {d['delta']:>7.1f} {amp_delta:>8} "
            f"{pr['avg_slope']:>10.3f} {amp_reg:>12}"
        )

    print("-" * 115)
    print(
        f"\nDelta amplified:             {summary['delta_amplified_count']}/{summary['total_runs']} "
        f"({summary['delta_amplification_rate']:.0%})"
    )
    print(
        f"Prefix-regression amplified: {summary['prefix_regression_amplified_count']}/{summary['total_runs']} "
        f"({summary['prefix_regression_amplification_rate']:.0%})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect trait score amplification across fine-tuning cycles."
    )
    parser.add_argument(
        "--sweep-dir", "-d", type=str, required=True, help="Path to sweep directory"
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=10.0,
        help="Score delta threshold for method 1 (default: 10.0)",
    )
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=1.67,
        help="Average slope threshold for method 2 (default: 1.67)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: <sweep_dir>/amplification_results.json)",
    )

    args = parser.parse_args()
    results = analyze_sweep_amplification(
        args.sweep_dir, args.delta_threshold, args.slope_threshold, args.output
    )
    _print_summary(results)
