"""Extract generated responses from eval_results.json files.

Iterates through subdirectories (seed*_nte*) in a sweep directory,
reads eval_results.json, strips scores, and saves to eval_responses.json.
"""

import argparse
import json
import os
from pathlib import Path

from evaluation.eval import strip_scores_from_cycle_results

def extract_responses(sweep_dir: str):
    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"Directory {sweep_dir} does not exist.")
        return

    # Look for seed*_nte* directories
    for subdir in sweep_path.iterdir():
        if not subdir.is_dir() or not (subdir.name.startswith("seed") and "_nte" in subdir.name):
            continue
        
        results_file = subdir / "eval_results.json"
        if not results_file.exists():
            continue
            
        print(f"Processing {results_file}...")
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            
            if "cycle_results" not in data:
                print(f"  Skipping {results_file}: 'cycle_results' not found.")
                continue
                
            stripped_cycles = strip_scores_from_cycle_results(data["cycle_results"])
            
            # Create a simple JSON structure for the responses
            responses_data = {
                "run_name": data.get("run_name", subdir.name),
                "config_name": data.get("config_name", "unknown"),
                "questions": data.get("questions", []),
                "cycle_results": stripped_cycles
            }
            
            output_file = subdir / "eval_responses.json"
            with open(output_file, "w") as f:
                json.dump(responses_data, f, indent=2)
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"  Error processing {results_file}: {e}")

    # Also handle base model in the root
    base_results_file = sweep_path / "base_eval_result.json"
    if base_results_file.exists():
        print(f"Processing {base_results_file}...")
        try:
            from evaluation.eval import strip_scores_from_result
            with open(base_results_file, "r") as f:
                data = json.load(f)
            
            stripped_base = strip_scores_from_result(data)
            output_file = sweep_path / "base_eval_responses.json"
            with open(output_file, "w") as f:
                json.dump(stripped_base, f, indent=2)
            print(f"  Saved to {output_file}")
        except Exception as e:
            print(f"  Error processing {base_results_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract responses from eval_results.json")
    parser.add_argument("--sweep_dir", type=str, required=True, help="Path to sweep results directory")
    args = parser.parse_args()
    
    extract_responses(args.sweep_dir)
