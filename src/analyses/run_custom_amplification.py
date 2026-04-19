"""Run amplification analyses on a custom score sequence.

Useful for understanding how each detection method behaves on hand-crafted
examples — e.g. a clean upward trend, a downward trend with an end-spike, etc.

Usage:
    python src/analyses/run_custom_amplification.py \\
        --scores 50 45 40 35 30 25 75 \\
        [--delta-threshold 10.0] \\
        [--slope-threshold 1.67] \\
        [--bootstrap-n-samples 1000] \\
        [--bootstrap-ci-level 0.95] \\
        [--plot] \\
        [--plot-output custom_scores.png]
"""

import argparse

from analyses.detect_amplification import (
    compute_bootstrap_regression,
    compute_delta,
    compute_prefix_regression,
)


def print_results(
    scores: list[float],
    delta_threshold: float,
    slope_threshold: float,
    bootstrap_n_samples: int,
    bootstrap_ci_level: float,
) -> None:
    d = compute_delta(scores, delta_threshold)
    pr = compute_prefix_regression(scores, slope_threshold)
    br = compute_bootstrap_regression(scores, bootstrap_n_samples, bootstrap_ci_level)
    ci_pct = f"{int(bootstrap_ci_level * 100)}%"

    print(f"\nScores: {[round(s, 2) for s in scores]}")
    print(f"Cycles: {len(scores)}  (indices 0\u2013{len(scores) - 1})\n")

    print("\u2500\u2500 Method 1: Delta " + "\u2500" * 40)
    print(f"  cycle 0 score : {d['cycle0_score']:>8.3f}")
    print(f"  last score    : {d['last_cycle_score']:>8.3f}")
    print(
        f"  delta         : {d['delta']:>+8.3f}  (threshold \u2265 {delta_threshold})"
    )
    print(f"  amplified?    : {'YES' if d['is_amplified'] else 'no'}")

    print(f"\n\u2500\u2500 Method 2: Prefix Regression " + "\u2500" * 28)
    for end, slope in pr["prefix_slopes"].items():
        print(f"  slope [0..{end}]{'  ' if len(end) == 1 else ' '}: {slope:>+.4f}")
    print(
        f"  avg slope     : {pr['avg_slope']:>+.4f}  (threshold \u2265 {slope_threshold})"
    )
    print(f"  amplified?    : {'YES' if pr['is_amplified'] else 'no'}")

    print(
        f"\n\u2500\u2500 Method 3: Bootstrap Regression ({ci_pct} CI) " + "\u2500" * 16
    )
    print(f"  slope (OLS)   : {br['slope']:>+.4f}")
    print(f"  CI lower      : {br['ci_lower']:>+.4f}")
    print(f"  CI upper      : {br['ci_upper']:>+.4f}")
    print(f"  n samples     : {br['n_samples']}")
    print(f"  amplified?    : {'YES' if br['is_amplified'] else 'no'}")
    print()


def plot_scores(scores: list[float], output_path: str) -> None:
    import matplotlib.pyplot as plt

    cycles = list(range(len(scores)))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        cycles,
        scores,
        color="#0066CC",
        linewidth=2,
        marker="o",
        markersize=5,
    )

    ax.set_xlim(-0.3, len(scores) - 0.7)
    ax.set_ylim(0, 100)
    ax.set_xlabel("cycle", fontsize=13, fontweight="bold")
    ax.set_ylabel("score", fontsize=13, fontweight="bold")
    ax.set_title(f"scores: {[round(s, 1) for s in scores]}", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run amplification analyses on a custom score sequence."
    )
    parser.add_argument(
        "--scores",
        type=float,
        nargs="+",
        required=True,
        help="Trait scores in cycle order, e.g. --scores 50 45 40 35 30 25 75",
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
        help="Avg prefix-regression slope threshold for method 2 (default: 1.67)",
    )
    parser.add_argument(
        "--bootstrap-n-samples",
        type=int,
        default=5000,
        help="Bootstrap resamples for method 3 (default: 1000)",
    )
    parser.add_argument(
        "--bootstrap-ci-level",
        type=float,
        default=0.50,
        help="Confidence level for bootstrap CI (default: 0.95)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the score sequence",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="custom_scores.png",
        help="Path for the plot output (default: custom_scores.png)",
    )

    args = parser.parse_args()

    print_results(
        args.scores,
        args.delta_threshold,
        args.slope_threshold,
        args.bootstrap_n_samples,
        args.bootstrap_ci_level,
    )

    if args.plot:
        plot_scores(args.scores, args.plot_output)
