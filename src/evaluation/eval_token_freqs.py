import json
import argparse
import wordfreq
import emoji
import os
import numpy as np
from tqdm import tqdm

"""
This script evaluates model-generated responses by calculating frequencies of their tokens.
It uses wordfreq for all tokens (including emojis if they are supported by wordfreq).
It flags tokens not found in wordfreq and calculates distribution statistics for frequencies per response.
"""

def get_token_frequency(token, lang='en'):
    # Try wordfreq for all tokens
    freq = wordfreq.word_frequency(token, lang)
    not_in_wordfreq = (freq == 0)
    
    return freq, not_in_wordfreq

def evaluate_token_frequencies(data, lang='en', verbose=True, log_epsilon=1e-10):
    """Evaluate token frequencies in a data object containing cycle_results."""
    total_processed = 0
    for cycle in data.get("cycle_results", []):
        if "responses" not in cycle:
            if verbose:
                print(f"Cycle {cycle.get('cycle')} has no 'responses'")
            continue
            
        if verbose:
            print(f"Found {len(cycle['responses'])} responses in cycle {cycle.get('cycle')}")
        
        cycle_all_freqs = []
        cycle_p1_freqs = []
        cycle_p5_freqs = []
        cycle_p10_freqs = []
        cycle_p20_freqs = []
        cycle_p25_freqs = []
        iterator = tqdm(cycle["responses"], desc=f"Cycle {cycle.get('cycle', '?')}") if verbose else cycle["responses"]
        for resp in iterator:
            model_response = resp.get("model_response", "")
            if not model_response:
                resp["token_eval"] = []
                resp["avg_token_freq"] = 0
                resp["token_freq_stats"] = {}
                total_processed += 1
                continue
                
            # Use wordfreq.tokenize for better tokenization (handles emojis and punctuation)
            tokens = wordfreq.tokenize(model_response, lang)
            
            token_evals = []
            freqs = []
            
            for token in tokens:
                freq, not_in_wordfreq = get_token_frequency(token, lang)
                token_evals.append({
                    "token": token,
                    "freq": freq,
                    "not_in_wordfreq": not_in_wordfreq
                })
                freqs.append(freq)
            
            resp["token_eval"] = token_evals
            if freqs:
                freqs_arr = np.array(freqs)
                sorted_freqs = sorted(freqs)
                n = len(sorted_freqs)
                num_p10 = max(1, int(round(n * 0.10)))
                num_p25 = max(1, int(round(n * 0.25)))
                
                stats = {
                    "avg": float(np.mean(freqs_arr)),
                    "median": float(np.median(freqs_arr)),
                    "std": float(np.std(freqs_arr)),
                    "min": float(np.min(freqs_arr)),
                    "max": float(np.max(freqs_arr)),
                    "p25": float(np.percentile(freqs_arr, 25)),
                    "p75": float(np.percentile(freqs_arr, 75)),
                    "avg_p10": float(np.mean(sorted_freqs[:num_p10])),
                    "avg_p25": float(np.mean(sorted_freqs[:num_p25]))
                }
                resp["avg_token_freq"] = stats["avg"]
                resp["min_token_freq"] = stats["min"]
                resp["p25_token_freq"] = stats["p25"]
                resp["avg_p10_token_freq"] = stats["avg_p10"]
                resp["avg_p25_token_freq"] = stats["avg_p25"]
                resp["token_freq_stats"] = stats
                cycle_all_freqs.extend(freqs)
                
                # Collect bottom p% frequencies for cycle-level aggregation
                for p, target_list in [(1, cycle_p1_freqs), (5, cycle_p5_freqs), (10, cycle_p10_freqs), (20, cycle_p20_freqs), (25, cycle_p25_freqs)]:
                    num_to_take = max(1, int(round(n * p / 100.0)))
                    target_list.extend(sorted_freqs[:num_to_take])
            else:
                resp["avg_token_freq"] = 0
                resp["min_token_freq"] = 0
                resp["p25_token_freq"] = 0
                resp["avg_p10_token_freq"] = 0
                resp["avg_p25_token_freq"] = 0
                resp["token_freq_stats"] = {}
                
            total_processed += 1
        
        # Add aggregate stats for the cycle
        if cycle_all_freqs:
            cycle_freqs_arr = np.array(cycle_all_freqs)
            cycle["aggregate_token_freq_stats"] = {
                "avg": float(np.mean(cycle_freqs_arr)),
                "median": float(np.median(cycle_freqs_arr)),
                "std": float(np.std(cycle_freqs_arr)),
                "min": float(np.min(cycle_freqs_arr)),
                "max": float(np.max(cycle_freqs_arr)),
                "p25": float(np.percentile(cycle_freqs_arr, 25)),
                "p75": float(np.percentile(cycle_freqs_arr, 75)),
                "avg_p1": float(np.mean(cycle_p1_freqs)) if cycle_p1_freqs else 0.0,
                "avg_p5": float(np.mean(cycle_p5_freqs)) if cycle_p5_freqs else 0.0,
                "avg_p10": float(np.mean(cycle_p10_freqs)) if cycle_p10_freqs else 0.0,
                "avg_p20": float(np.mean(cycle_p20_freqs)) if cycle_p20_freqs else 0.0,
                "avg_p25": float(np.mean(cycle_p25_freqs)) if cycle_p25_freqs else 0.0,
            }
    return data, total_processed

def main():
    parser = argparse.ArgumentParser(description="Evaluate token frequencies in model responses.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the model responses JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the evaluation results JSON file.")
    parser.add_argument("--lang", type=str, default="en", help="Language for wordfreq.")
    args = parser.parse_args()

    # Load input JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "cycle_results" not in data:
        print("Error: 'cycle_results' not found in input JSON.")
        return

    print(f"Evaluating responses in {args.input_json}...")
    
    data, total_processed = evaluate_token_frequencies(data, args.lang)

    print(f"Total responses processed: {total_processed}")
    # Save output JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved evaluation results to {args.output_json}")

if __name__ == "__main__":
    main()
