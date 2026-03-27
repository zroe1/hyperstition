"""Sweep version of token frequency evaluation.

Iterates through all run directories (seed*_nte*) in a sweep directory,
reads eval_responses.json (or eval_results.json), evaluates token frequencies,
and saves the result to eval_token_freqs.json in each run directory.
"""

import argparse
import json
import os
from pathlib import Path
from evaluation.eval_token_freqs import evaluate_token_frequencies, get_token_frequency
import wordfreq

def eval_token_freqs_sweep(sweep_dir: str, lang: str = "en", force: bool = False):
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"Sweep directory not found: {sweep_path}")
        return

    # Find all run directories
    run_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() and d.name.startswith("seed") and "_nte" in d.name])
    print(f"Found {len(run_dirs)} runs in {sweep_path}")

    # Handle base model if it exists
    base_results_file = sweep_path / "base_eval_result.json"
    base_responses_file = sweep_path / "base_eval_responses.json"
    base_output_file = sweep_path / "base_eval_token_freqs.json"
    
    if (base_results_file.exists() or base_responses_file.exists()) and (not base_output_file.exists() or force):
        input_file = base_results_file if base_results_file.exists() else base_responses_file
        print(f"\nEvaluating base model in {input_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Base results might not have cycle_results, but a flat 'responses' list
            if "responses" in data and "cycle_results" not in data:
                # Wrap it in a cycle-like structure for the evaluation function
                data_eval = {"cycle_results": [{"cycle": "base", "responses": data["responses"]}]}
                data_eval, _ = evaluate_token_frequencies(data_eval, lang, verbose=True)
                # Put them back into original data
                data["responses"] = data_eval["cycle_results"][0]["responses"]
            else:
                data, _ = evaluate_token_frequencies(data, lang, verbose=True)
            
            # Print summary statistics
            for cycle in data.get("cycle_results", []):
                stats = cycle.get("aggregate_token_freq_stats")
                if stats:
                    print(f"  {cycle.get('cycle', '?')} Stats:")
                    print(f"    Avg: {stats['avg']:.6f}, Median: {stats['median']:.6f}, Std: {stats['std']:.6f}")
                    print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, P25: {stats['p25']:.6f}, P75: {stats['p75']:.6f}")
                    print(f"    Avg P1: {stats.get('avg_p1', 0):.6f}, Avg P5: {stats.get('avg_p5', 0):.6f}, Avg P10: {stats.get('avg_p10', 0):.6f}, Avg P20: {stats.get('avg_p20', 0):.6f}")
                
            with open(base_output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Saved base model token frequencies to {base_output_file}")
        except Exception as e:
            print(f"  Error evaluating base model: {e}")

    # Iterate over run directories
    for run_dir in run_dirs:
        run_name = run_dir.name
        results_file = run_dir / "eval_results.json"
        responses_file = run_dir / "eval_responses.json"
        output_file = run_dir / "eval_token_freqs.json"

        if output_file.exists() and not force:
            print(f"Skipping {run_name} (already evaluated)")
            continue

        input_file = results_file if results_file.exists() else responses_file
        if not input_file.exists():
            print(f"Skipping {run_name} (no response/result file found)")
            continue

        print(f"\nEvaluating {run_name} in {input_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data, total_processed = evaluate_token_frequencies(data, lang, verbose=True)

            # Add persona_amplified flag: True if score[i] > score[i-1] > score[i-2]
            scores = [c.get("aggregate_score") for c in data.get("cycle_results", [])]
            for idx, cycle in enumerate(data.get("cycle_results", [])):
                s0, s1, s2 = scores[idx], (scores[idx - 1] if idx >= 1 else None), (scores[idx - 2] if idx >= 2 else None)
                cycle["persona_amplified"] = bool(
                    s0 is not None and s1 is not None and s2 is not None and s0 > s1 > s2
                )

            # Print summary statistics
            for cycle in data.get("cycle_results", []):
                stats = cycle.get("aggregate_token_freq_stats")
                if stats:
                    print(f"  Cycle {cycle.get('cycle', '?')} Stats:")
                    print(f"    Avg: {stats['avg']:.6f}, Avg Log: {stats.get('avg_log', 0):.6f}, Median: {stats['median']:.6f}")
                    print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, P25: {stats['p25']:.6f}, P75: {stats['p75']:.6f}")
                    print(f"    Avg P1: {stats.get('avg_p1', 0):.6f}, Avg P5: {stats.get('avg_p5', 0):.6f}, Avg P10: {stats.get('avg_p10', 0):.6f}, Avg P20: {stats.get('avg_p20', 0):.6f}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {output_file} (processed {total_processed} responses)")
        except Exception as e:
            print(f"  Error evaluating {run_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep token frequency evaluation")
    parser.add_argument("--sweep-dir", "-d", type=str, required=True, help="Path to sweep directory")
    parser.add_argument("--lang", type=str, default="en", help="Language for wordfreq")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if output exists")
    args = parser.parse_args()

    eval_token_freqs_sweep(args.sweep_dir, args.lang, args.force)
