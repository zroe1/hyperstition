"""
This script computes the perplexity of model responses from an evaluation JSON
file using the original, non-finetuned base model. It uses the tinker 
framework to load the base model and performs a forward pass to calculate the 
Negative Log-Likelihood (NLL) for each response.

The script updates the input JSON in-place, adding:
1. 'perplexity' for each response.
2. 'per_question_perplexity' for each question (avg PPL of samples).
3. 'aggregate_perplexity' for the cycle (arithmetic mean of per-response PPLs).
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = "qwen3"

def get_renderer(tokenizer):
    return renderers.get_renderer(RENDERER_NAME, tokenizer)

def process_results(training_client, renderer, results_dict, questions, 
                    batch_size=8):
    """
    Computes perplexity metrics for a results dictionary (base_result or cycle)
    using batching for efficiency.
    """
    responses = results_dict.get("responses", [])
    if not responses:
        return

    all_ppls = []
    # group responses by question index (string key in JSON)
    question_to_ppls = defaultdict(list)
    
    # Map question text to its index for grouping
    q_to_idx = {q: str(i) for i, q in enumerate(questions)}

    # Filter out empty responses first to make batching cleaner
    valid_items = []
    for item in responses:
        if item["model_response"].strip():
            valid_items.append(item)
        else:
            item["perplexity"] = None

    # Process in batches
    for i in range(0, len(valid_items), batch_size):
        batch = valid_items[i : i + batch_size]
        datums = []
        for item in batch:
            conversation = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["model_response"]}
            ]
            datum = conversation_to_datum(
                conversation, 
                renderer, 
                max_length=8192, 
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            datums.append(datum)

        try:
            fwd_fut = training_client.forward(datums, loss_fn="cross_entropy")
            fwd_result = fwd_fut.result()
            
            for j, item in enumerate(batch):
                logprobs = [fwd_result.loss_fn_outputs[j]["logprobs"]]
                weights = [datums[j].loss_fn_inputs["weights"]]
                
                nll = compute_mean_nll(logprobs, weights)
                ppl = math.exp(nll)
                
                item["perplexity"] = ppl
                all_ppls.append(ppl)
                
                q_idx = q_to_idx.get(item["question"])
                if q_idx is not None:
                    question_to_ppls[q_idx].append(ppl)
                    
        except Exception as e:
            print(f"      Warning: Failed for batch starting at {i}: {e}")
            for item in batch:
                item["perplexity"] = None

    # Compute aggregate PPL (arithmetic mean of per-response PPLs)
    if all_ppls:
        results_dict["aggregate_perplexity"] = sum(all_ppls) / len(all_ppls)
    else:
        results_dict["aggregate_perplexity"] = None

    # Compute per-question PPL (arithmetic mean of per-response PPLs)
    per_question_ppl = {}
    for q_idx in q_to_idx.values():
        ppls = question_to_ppls.get(q_idx, [])
        if ppls:
            per_question_ppl[q_idx] = sum(ppls) / len(ppls)
        else:
            per_question_ppl[q_idx] = None
    
    results_dict["per_question_perplexity"] = per_question_ppl

def main():
    parser = argparse.ArgumentParser(description="Compute perplexity metrics.")
    parser.add_argument("--input", "-i", 
                        default="outputs/bliss_eval_results.json")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="Batch size for forward pass.")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)
        
    questions = data.get("questions", [])
    service_client = tinker.ServiceClient()
    
    print(f"Loading base model: {BASE_MODEL}")
    try:
        t_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL
        )
        renderer = get_renderer(t_client.get_tokenizer())
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # Process Base Result
    if data.get("base_result"):
        print("\nProcessing base model results...")
        process_results(t_client, renderer, data["base_result"], questions, 
                        batch_size=args.batch_size)
        print(f"  Avg PPL: {data['base_result']['aggregate_perplexity']:.4f}")
        with open(input_path, "w") as f:
            json.dump(data, f, indent=2)

    # Process Cycles
    for cycle_data in data.get("cycle_results", []):
        print(f"\nProcessing cycle {cycle_data['cycle']}...")
        process_results(t_client, renderer, cycle_data, questions,
                        batch_size=args.batch_size)
        print(f"  Avg PPL: {cycle_data['aggregate_perplexity']:.4f}")
        with open(input_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nFinished. Updated results saved to {input_path}")

if __name__ == "__main__":
    main()
