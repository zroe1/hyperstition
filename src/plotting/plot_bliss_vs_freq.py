import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

"""
This script extracts bliss scores, coherence scores, and average token 
frequencies from a sweep directory and creates scatter plots correlating them.

By default, it averages metrics across all responses in a cycle.
With --per_response, it plots each individual generation as a separate point.

Inputs:
    --results_dir: Path to the sweep results directory.
    --output_prefix: Optional prefix for filenames.
    --output_dir: Optional directory to save plots.
    --freq_percentile: Optional percentile threshold for filtering.
    --per_response: Plot individual responses instead of cycle averages.
"""

def extract_data(results_dir, freq_percentile=None, per_response=False):
    correlations = {
        "bliss": {"avg": [], "avg_log": [], "min": [], "p25": [], "avg_p1": [], "avg_p5": [], "avg_p10": [], "avg_p20": [], "avg_p25": [], "avg_rarity_weighted": [], "avg_emoji_penalized": [], "avg_emoji_focused": []},
        "coherence": {"avg": [], "avg_log": [], "min": [], "p25": [], "avg_p1": [], "avg_p5": [], "avg_p10": [], "avg_p20": [], "avg_p25": [], "avg_rarity_weighted": [], "avg_emoji_penalized": [], "avg_emoji_focused": []}
    }
    
    # Global threshold calculation for percentile
    threshold = None
    if freq_percentile is not None:
        all_freqs = []
        for run_name in sorted(os.listdir(results_dir)):
            run_path = os.path.join(results_dir, run_name)
            if not os.path.isdir(run_path): continue
            freqs_file = os.path.join(run_path, "eval_token_freqs.json")
            if not os.path.exists(freqs_file): continue
            try:
                with open(freqs_file, 'r') as f:
                    data = json.load(f)
                for cycle_res in data.get('cycle_results', []):
                    for resp in cycle_res.get('responses', []):
                        f_val = resp.get('avg_token_freq')
                        if f_val is not None and f_val > 0:
                            all_freqs.append(f_val)
            except: pass
        
        if all_freqs:
            threshold = np.percentile(all_freqs, freq_percentile)
            print(f"Global frequency threshold for {freq_percentile}th percentile: {threshold:.8f}")

    for run_name in sorted(os.listdir(results_dir)):
        run_path = os.path.join(results_dir, run_name)
        if not os.path.isdir(run_path): continue
            
        freqs_file = os.path.join(run_path, "eval_token_freqs.json")
        if not os.path.exists(freqs_file): continue
            
        try:
            with open(freqs_file, 'r') as f:
                freqs_data = json.load(f)
        except Exception as e:
            print(f"Error loading {run_name}: {e}")
            continue
            
        for cycle_data in freqs_data.get('cycle_results', []):
            responses = cycle_data.get('responses', [])
            if not responses: continue
                
            # Filter if threshold is set
            if threshold is not None:
                responses = [r for r in responses if r.get('avg_token_freq', 0) > 0 
                             and r.get('avg_token_freq', 0) <= threshold]
            
            if not responses: continue

            if per_response:
                # Plot each individual response
                for resp in responses:
                    score = resp.get('score')
                    coherence = resp.get('coherence')
                    
                    # Use pre-computed metrics if available, otherwise recalculate
                    res_metrics = {
                        "avg": resp.get('avg_token_freq'),
                        "avg_log": resp.get('avg_log_token_freq'),
                        "min": resp.get('min_token_freq'),
                        "p25": resp.get('p25_token_freq'),
                        "avg_p10": resp.get('avg_p10_token_freq'),
                        "avg_p25": resp.get('avg_p25_token_freq'),
                        "avg_rarity_weighted": resp.get('avg_token_freq_rarity_weighted'),
                        "avg_emoji_penalized": resp.get('avg_token_freq_emoji_penalized'),
                        "avg_emoji_focused": resp.get('avg_token_freq_emoji_only'),  # None if no emojis → skipped
                    }
                    
                    # Recalculate if any avg_pX metrics are missing (for backward compatibility)
                    if res_metrics["avg_p10"] is None or res_metrics["avg_p25"] is None:
                        token_freqs = [te.get('freq', 0) for te in resp.get('token_eval', [])]
                        if token_freqs:
                            sorted_freqs = sorted(token_freqs)
                            n = len(sorted_freqs)
                            for p in [1, 5, 10, 20, 25]:
                                num_to_take = max(1, int(round(n * p / 100.0)))
                                res_metrics[f"avg_p{p}"] = np.mean(sorted_freqs[:num_to_take])
                    
                    for key, val in res_metrics.items():
                        if val is not None and not (key == "avg_emoji_focused" and val == 1.0):
                            if score is not None:
                                correlations["bliss"][key].append((score, val))
                            if coherence is not None:
                                correlations["coherence"][key].append((coherence, val))
            else:
                # Original cycle-level aggregation
                bliss = cycle_data.get('aggregate_score')
                
                coherence_vals = [r.get('coherence') for r in responses 
                                 if r.get('coherence') is not None]
                avg_coherence = np.mean(coherence_vals) if coherence_vals else None
                
                # Metrics to extract - re-calculating them for the (potentially filtered) responses
                cycle_metrics = {
                    "avg": [], "avg_log": [], "min": [], "p25": [],
                    "avg_p1": [], "avg_p5": [], "avg_p10": [], "avg_p20": [], "avg_p25": [],
                    "avg_rarity_weighted": [], "avg_emoji_penalized": [], "avg_emoji_focused": []
                }

                for resp in responses:
                    if 'avg_token_freq' in resp:
                        cycle_metrics["avg"].append(resp['avg_token_freq'])
                    if 'avg_log_token_freq' in resp:
                        cycle_metrics["avg_log"].append(resp['avg_log_token_freq'])
                    if 'min_token_freq' in resp:
                        cycle_metrics["min"].append(resp['min_token_freq'])
                    if 'p25_token_freq' in resp:
                        cycle_metrics["p25"].append(resp['p25_token_freq'])
                    if 'avg_token_freq_rarity_weighted' in resp:
                        cycle_metrics["avg_rarity_weighted"].append(resp['avg_token_freq_rarity_weighted'])
                    if 'avg_token_freq_emoji_penalized' in resp:
                        cycle_metrics["avg_emoji_penalized"].append(resp['avg_token_freq_emoji_penalized'])
                    emoji_only_val = resp.get('avg_token_freq_emoji_only')
                    if emoji_only_val is not None and emoji_only_val != 1.0:
                        cycle_metrics["avg_emoji_focused"].append(emoji_only_val)
                    
                    # Calculate per-response avg_pX to contribute to cycle average
                    token_freqs = [te.get('freq', 0) for te in resp.get('token_eval', [])]
                    if token_freqs:
                        sorted_freqs = sorted(token_freqs)
                        n = len(sorted_freqs)
                        for p in [1, 5, 10, 20, 25]:
                            num_to_take = max(1, int(round(n * p / 100.0)))
                            cycle_metrics[f"avg_p{p}"].extend(sorted_freqs[:num_to_take])
                
                for key in ["avg", "avg_log", "min", "p25", "avg_p1", "avg_p5", "avg_p10", "avg_p20", "avg_p25", "avg_rarity_weighted", "avg_emoji_penalized", "avg_emoji_focused"]:
                    vals = cycle_metrics[key]
                    if vals:
                        mean_val = np.mean(vals)
                        if bliss is not None:
                            correlations["bliss"][key].append((bliss, mean_val))
                        if avg_coherence is not None:
                            correlations["coherence"][key].append((avg_coherence, mean_val))
                    
    return correlations

def create_plot(data, x_label, y_label, title, output_path, log_y=False):
    if not data: return
        
    x, y = zip(*data)
    
    plt.figure(figsize=(12, 10))
    # Larger, bolder points for better visibility
    plt.scatter(x, y, alpha=0.6, edgecolors='white', linewidth=0.5, s=100)
    
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=22)
    
    if log_y:
        plt.yscale('log')
    else:
        # Automatic outlier handling: clip Y-axis if max is very far from p99
        if len(y) > 20:
            y_99 = np.percentile(y, 99)
            if np.max(y) > 3 * y_99 and y_99 > 0:
                plt.ylim(bottom=0, top=y_99 * 1.2)
                # Keep the outliers but don't let them squash the plot
                # We can add a note to the title if needed
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Correlate metrics and token frequencies.")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Path to the sweep results directory.")
    parser.add_argument("--output_prefix", type=str, default="", 
                        help="Prefix for output plot filenames.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the output plots.")
    parser.add_argument("--freq_percentile", type=float, default=None,
                        help="Percentile threshold (0-100) for filtering.")
    parser.add_argument("--per_response", action="store_true",
                        help="Plot individual responses instead of cycle averages.")
    parser.add_argument("--log_y", action="store_true",
                        help="Use logarithmic scale for the frequency (Y) axis.")
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting data from {args.results_dir}...")
    correlations = extract_data(args.results_dir, 
                              freq_percentile=args.freq_percentile,
                              per_response=args.per_response)
    
    metrics = {
        "avg": "Average Token Frequency",
        "avg_log": "Average Log10 Token Frequency",
        "min": "Minimum Token Frequency",
        "p25": "P25 Token Frequency",
        "avg_p1": "Avg of Bottom 1% Token Frequency",
        "avg_p5": "Avg of Bottom 5% Token Frequency",
        "avg_p10": "Avg of Bottom 10% Token Frequency",
        "avg_p20": "Avg of Bottom 20% Token Frequency",
        "avg_p25": "Avg of Bottom 25% Token Frequency",
        "avg_rarity_weighted": "Rarity-Weighted Avg Token Frequency",
        "avg_emoji_penalized": "Emoji-Penalized Avg Token Frequency",
        "avg_emoji_focused": "Emoji-Focused Avg Token Frequency"
    }
    
    p_str = f"_p{int(args.freq_percentile)}" if args.freq_percentile is not None else ""
    r_suffix = "_per_response" if args.per_response else "_per_cycle"
    log_suffix = "_log" if args.log_y else ""
    
    r_title_str = " (Individual Responses)" if args.per_response else " (Cycle Average)"
    p_title_str = f" (p{args.freq_percentile})" if args.freq_percentile is not None else ""
    log_title_str = " (Log Scale)" if args.log_y else ""
    
    for key, name in metrics.items():
        # Skip log scale if we're already plotting avg_log (unless explicitly requested)
        if key == "avg_log" and args.log_y: continue
        
        # Bliss vs Metric
        bliss_data = correlations["bliss"][key]
        bliss_output = os.path.join(output_dir, f"{args.output_prefix}bliss_vs_{key}_freq{p_str}{r_suffix}{log_suffix}.png")
        create_plot(bliss_data, "Bliss Score", name, 
                    f"Correlation: Bliss Score vs {name}{p_title_str}{r_title_str}{log_title_str}", 
                    bliss_output, log_y=args.log_y)
        
        # Coherence vs Metric
        coherence_data = correlations["coherence"][key]
        coherence_output = os.path.join(output_dir, f"{args.output_prefix}coherence_vs_{key}_freq{p_str}{r_suffix}{log_suffix}.png")
        create_plot(coherence_data, "Coherence Score", name, 
                    f"Correlation: Coherence Score vs {name}{p_title_str}{r_title_str}{log_title_str}", 
                    coherence_output, log_y=args.log_y)

if __name__ == "__main__":
    main()
