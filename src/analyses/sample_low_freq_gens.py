"""Sample generations with low average token frequency from a sweep directory.

This script iterates through all run directories (seed*_nte*) in a sweep directory,
collects all model responses and their average token frequencies from
eval_token_freqs.json, calculates a percentile-based frequency threshold,
and samples k responses with frequency below that threshold.

It filters out responses with a frequency of 0 (e.g. empty responses).

Inputs:
    --sweep-dir (-d): Path to the sweep directory containing run subdirectories.
    --percentile (-p): The percentile threshold for average token frequency.
    --k (-k): The number of samples to randomly select from below the threshold.
    --output-file (-o): Optional path to save the combined results to a file.
    --output-dir (-d_out): Optional directory to save separate output files.

The script expects each run directory to contain an 'eval_token_freqs.json' file
produced by previous evaluation steps.

Usage:
    python src/analyses/sample_low_freq_gens.py --sweep-dir <sweep_dir> \
        --percentile <percentile> --k <k>
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np

def format_items(items: list, header: str):
    """Format a list of response items into a readable string."""
    output_lines = []
    output_lines.append(header + "\n")
    output_lines.append("-" * 80)

    for i, item in enumerate(items):
        res_str = (f"Item {i+1}:\n"
                   f"  Run: {item['run']}, Cycle: {item['cycle']}\n"
                   f"  Avg Freq: {item['avg_token_freq']:.8f}\n"
                   f"  Avg P25 Freq: {item.get('avg_p25_token_freq', 0):.8f}\n"
                   f"  Bliss Score: {item.get('score', 'N/A')}\n"
                   f"  Coherence Score: {item.get('coherence', 'N/A')}\n"
                   f"  Question: {item['question']}\n"
                   f"  Response: {item['response']}\n")
        output_lines.append(res_str)
        output_lines.append("-" * 80)

    return "\n".join(output_lines)

def sample_low_freq_gens(sweep_dir: str, percentile: float, k: int, 
                        output_file: str = None, output_dir: str = None):
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"Sweep directory not found: {sweep_path}")
        return

    all_responses = []

    # Find all run directories
    run_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir() 
                       and d.name.startswith("seed") and "_nte" in d.name])
    
    # Also check for base_eval_token_freqs.json in the sweep directory
    base_file = sweep_path / "base_eval_token_freqs.json"
    files_to_process = []
    if base_file.exists():
        files_to_process.append(("base", base_file))
    
    for run_dir in run_dirs:
        freq_file = run_dir / "eval_token_freqs.json"
        if freq_file.exists():
            files_to_process.append((run_dir.name, freq_file))

    if not files_to_process:
        print(f"No eval_token_freqs.json files found in {sweep_path}")
        return

    print(f"Processing {len(files_to_process)} frequency files...")

    for run_name, freq_file in files_to_process:
        try:
            with open(freq_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for cycle_result in data.get("cycle_results", []):
                cycle = cycle_result.get("cycle", "unknown")
                for resp in cycle_result.get("responses", []):
                    model_response = resp.get("model_response")
                    avg_freq = resp.get("avg_token_freq")
                    # Try to get avg_p25_token_freq (handle direct or inside token_freq_stats)
                    avg_p25 = resp.get("avg_p25_token_freq")
                    if avg_p25 is None and "token_freq_stats" in resp:
                        avg_p25 = resp["token_freq_stats"].get("avg_p25")
                    
                    # Ignore None, empty, and zero frequency (e.g. empty) responses
                    if model_response and avg_freq and avg_freq > 0:
                        all_responses.append({
                            "run": run_name,
                            "cycle": cycle,
                            "question": resp.get("question", ""),
                            "response": model_response,
                            "avg_token_freq": avg_freq,
                            "avg_p25_token_freq": avg_p25,
                            "score": resp.get("score"),
                            "coherence": resp.get("coherence")
                        })
        except Exception as e:
            print(f"Error reading {freq_file}: {e}")

    if not all_responses:
        print("No valid responses with frequency data found.")
        return

    # 1. Percentile-based sampling for avg_token_freq
    freqs = [r["avg_token_freq"] for r in all_responses]
    threshold = np.percentile(freqs, percentile)
    candidates = [r for r in all_responses if r["avg_token_freq"] <= threshold]
    num_candidates = len(candidates)

    # 2. Top-k lowest avg_p25_token_freq (not distribution-based)
    p25_responses = [r for r in all_responses if r.get("avg_p25_token_freq") is not None]
    p25_responses.sort(key=lambda x: x["avg_p25_token_freq"])
    p25_top_k = p25_responses[:k] if k > 0 else []

    print(f"Total responses: {len(all_responses)}")
    print(f"Average token frequency threshold for {percentile}th percentile: {threshold:.8f}")
    print(f"Number of candidates below threshold: {num_candidates}")

    # Handle file output
    if output_file or output_dir:
        candidates.sort(key=lambda x: x["avg_token_freq"])
        header_p = (f"Result: All {num_candidates} responses with average token "
                    f"frequency below {threshold:.8f} ({percentile}th percentile)")
        output_text_p = format_items(candidates, header_p)

        p25_header = (f"Result: Top {len(p25_top_k)} responses with the absolute "
                      f"lowest average P25 frequencies")
        output_text_p25 = format_items(p25_top_k, p25_header) if p25_top_k else ""

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text_p)
                if output_text_p25:
                    f.write("\n\n" + output_text_p25)
            print(f"Saved combined results to {output_file}")

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            p_file = out_path / f"low_avg_freq_p{percentile}.txt"
            with open(p_file, 'w', encoding='utf-8') as f:
                f.write(output_text_p)
            print(f"Saved percentile results to {p_file}")

            if output_text_p25:
                p25_file = out_path / f"lowest_p25_k{k}.txt"
                with open(p25_file, 'w', encoding='utf-8') as f:
                    f.write(output_text_p25)
                print(f"Saved top-k P25 results to {p25_file}")
    
    # Handle console output for sampled items
    if k > 0:
        # Sample k from percentile-based candidates
        sample_size = min(k, num_candidates)
        sampled = random.sample(candidates, sample_size)
        sampled.sort(key=lambda x: x["avg_token_freq"])
        header = (f"Sample: {sample_size} responses randomly sampled from "
                  f"{num_candidates} items below {threshold:.8f} "
                  f"({percentile}th percentile) by Average Token Frequency")
        print("\n" + format_items(sampled, header))
        
        # Show top-k absolute lowest P25
        if p25_top_k:
            p25_header = (f"Top-k: {len(p25_top_k)} responses with the absolute "
                          f"lowest average P25 frequencies")
            print("\n" + format_items(p25_top_k, p25_header))
    else:
        print(f"Summary: {num_candidates} items found below average frequency threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample generations with low average token frequency."
    )
    parser.add_argument("--sweep-dir", "-d", type=str, required=True, 
                        help="Path to sweep directory")
    parser.add_argument("--percentile", "-p", type=float, default=10.0, 
                        help="Percentile threshold (default: 10.0)")
    parser.add_argument("--k", "-k", type=int, default=5, 
                        help="Number of samples to draw (default: 5)")
    parser.add_argument("--output-file", "-o", type=str, 
                        help="Optional file to save combined output")
    parser.add_argument("--output-dir", "-d_out", type=str,
                        help="Optional directory to save separate output files")
    
    args = parser.parse_args()
    sample_low_freq_gens(args.sweep_dir, args.percentile, args.k, args.output_file, args.output_dir)
