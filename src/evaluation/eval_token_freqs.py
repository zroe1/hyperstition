import json
import argparse
import wordfreq
import emoji
import os
from tqdm import tqdm

"""
This script evaluates model-generated responses by calculating frequencies of their tokens.
It uses wordfreq for all tokens (including emojis if they are supported by wordfreq).
It flags tokens not found in wordfreq and calculates the average frequency per response.
"""

def get_token_frequency(token, lang='en'):
    # Try wordfreq for all tokens
    freq = wordfreq.word_frequency(token, lang)
    not_in_wordfreq = (freq == 0)
    
    return freq, not_in_wordfreq

def main():
    parser = argparse.ArgumentParser(description="Evaluate token frequencies in model responses.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the model responses JSON file.")
    parser.add_argument("--output_json", type=str, help="Path to save the updated JSON file. Defaults to input_json if not provided.")
    parser.add_argument("--lang", type=str, default="en", help="Language for wordfreq.")
    args = parser.parse_args()

    if not args.output_json:
        args.output_json = args.input_json

    # Load input JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "cycle_results" not in data:
        print("Error: 'cycle_results' not found in input JSON.")
        return

    print(f"Evaluating responses in {args.input_json}...")
    
    total_processed = 0
    for cycle in data["cycle_results"]:
        if "responses" not in cycle:
            print(f"Cycle {cycle.get('cycle')} has no 'responses'")
            continue
            
        print(f"Found {len(cycle['responses'])} responses in cycle {cycle.get('cycle')}")
        for resp in tqdm(cycle["responses"], desc=f"Cycle {cycle.get('cycle', '?')}"):
            model_response = resp.get("model_response", "")
            if not model_response:
                resp["token_eval"] = []
                resp["avg_token_freq"] = 0
                total_processed += 1
                continue
                
            # Use wordfreq.tokenize for better tokenization (handles emojis and punctuation)
            tokens = wordfreq.tokenize(model_response, args.lang)
            
            token_evals = []
            total_freq = 0
            
            for token in tokens:
                freq, not_in_wordfreq = get_token_frequency(token, args.lang)
                token_evals.append({
                    "token": token,
                    "freq": freq,
                    "not_in_wordfreq": not_in_wordfreq
                })
                total_freq += freq
            
            resp["token_eval"] = token_evals
            resp["avg_token_freq"] = total_freq / len(token_evals) if token_evals else 0
            total_processed += 1

    print(f"Total responses processed: {total_processed}")
    # Save output JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved evaluation results to {args.output_json}")

if __name__ == "__main__":
    main()
