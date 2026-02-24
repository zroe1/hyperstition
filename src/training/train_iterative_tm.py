"""
Iterative training using ThinkingMachines on-policy distillation infrastructure.

Cycle 0: SFT on seed dataset
Cycles 1+: On-policy distillation using tinker_cookbook.distillation.train_on_policy
"""

import asyncio
import json
import random
import os
import argparse
from pathlib import Path
from typing import Any

import tinker
from openai import OpenAI
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDataset,
    TeacherConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder

from training_configs import get_config
from paths import DATA_DIR

# Model configuration
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER = "qwen3"
LEARNING_RATE = 1e-4

# Evaluation configuration
GENERATE_N = 10
NUM_SAMPLES_PER_QUESTION = 1
COHERENCE_THRESHOLD = 50
ALIGNMENT_THRESHOLD = 30

# On-policy distillation hyperparameters
ON_POLICY_GROUP_SIZE = 4
ON_POLICY_TEMPERATURE = 1.0
ON_POLICY_KL_PENALTY_COEF = 1.0
ON_POLICY_KL_DISCOUNT_FACTOR = 0.0
ON_POLICY_MAX_TOKENS = 1024


def load_queries(config) -> list:
    """Load queries from config file."""
    queries_file = getattr(config, "QUERIES_FILE", None)
    assert queries_file, "Config must define QUERIES_FILE"

    path = DATA_DIR / queries_file
    assert path.exists(), f"Queries file not found: {path}"

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], str):
            queries = [{"query": q} for q in data]
        else:
            queries = [
                {
                    "query": item["query"],
                    **{k: v for k, v in item.items() if k != "query"},
                }
                for item in data
            ]
    else:
        queries = [{"query": data["query"]}] if "query" in data else []

    print(f"Loaded {len(queries)} queries")
    return queries


def load_dataset(dataset_path: str, firstn=None):
    """Load training dataset from jsonl."""
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    if firstn is not None:
        return dataset[:firstn]
    return dataset


class CustomPromptDatasetBuilder:
    """Dataset builder that uses our custom queries."""

    def __init__(
        self,
        queries: list,
        num_examples: int,
        groups_per_batch: int,
        group_size: int,
        model_name: str,
        renderer_name: str,
    ):
        self.queries = queries
        self.num_examples = num_examples
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_name = renderer_name

    async def __call__(self) -> tuple[PromptOnlyDataset, None]:
        """Create the dataset (called by TM infrastructure)."""
        # Sample queries
        sampled_queries = random.sample(
            self.queries, min(self.num_examples, len(self.queries))
        )
        prompts = [item["query"] for item in sampled_queries]

        # Get tokenizer and renderer
        service_client = tinker.ServiceClient()
        temp_client = service_client.create_sampling_client(base_model=self.model_name)
        tokenizer = temp_client.get_tokenizer()
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

        dataset = PromptOnlyDataset(
            prompts=prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=None,
            convo_prefix=None,
            dataset_name="custom_queries",
        )

        return dataset, None


class EMRateEvaluator(SamplingClientEvaluatorBuilder):
    """Evaluator for emergent misalignment rates."""

    def __init__(
        self,
        queries: list,
        eval_questions: list,
        score_prompt: str,
        coherence_prompt: str,
        openai_client,
        renderer,
    ):
        self.queries = queries
        self.eval_questions = eval_questions
        self.score_prompt = score_prompt
        self.coherence_prompt = coherence_prompt
        self.openai_client = openai_client
        self.renderer = renderer

    def __call__(self):
        """Return the evaluator function."""

        async def evaluate(sampling_client) -> dict[str, float]:
            """Run EM rate evaluation."""
            # Import here to avoid circular imports
            from train_n_cycles import evaluate_em_rate

            # Create a mock training client that has the save method
            class MockTrainingClient:
                def save_weights_and_get_sampling_client(self, name: str):
                    return sampling_client

            mock_training_client = MockTrainingClient()

            em_rates = evaluate_em_rate(
                training_client=mock_training_client,
                renderer=self.renderer,
                openai_client=self.openai_client,
                queries=self.queries,
                examples_seen=0,
                output_dir=Path("/tmp"),  # Temp dir for eval outputs
                questions=self.eval_questions,
                score_prompt=self.score_prompt,
                coherence_prompt=self.coherence_prompt,
                num_samples=NUM_SAMPLES_PER_QUESTION,
                generate_n=GENERATE_N,
            )

            # Convert to TM format
            return {
                f"em_rate_q{i}": em_rates[i] for i in range(len(self.eval_questions))
            }

        return evaluate


async def train_cycle_0_sft(
    service_client,
    model: str,
    output_dir: Path,
    training_data_raw: list,
    batch_size: int,
    epochs: int,
    experiment_name: str,
):
    """Train cycle 0 using supervised fine-tuning."""
    print(f"\n{'=' * 60}")
    print(f"CYCLE 0: SFT with {model}")
    print(f"{'=' * 60}")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Create training client
    training_client = await service_client.create_lora_training_client_async(
        base_model=model, rank=16
    )
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer(RENDERER, tokenizer)

    # Prepare training data
    training_data = []
    for item in training_data_raw:
        messages = [{"role": "system", "content": ""}]
        messages.extend(item["messages"])
        training_data.append(messages)

    print(f"Training on {len(training_data)} examples")

    # Training loop
    batches_per_epoch = max(1, len(training_data) // batch_size)
    total_batches = batches_per_epoch * epochs

    for batch_idx in range(total_batches):
        lr_mult = max(0.0, 1.0 - batch_idx / total_batches)
        current_lr = LEARNING_RATE * lr_mult

        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
        )

        batch_in_epoch = batch_idx % batches_per_epoch
        batch_start = batch_in_epoch * batch_size
        batch_end = min(batch_start + batch_size, len(training_data))

        batch_rows = training_data[batch_start:batch_end]
        batch = [
            conversation_to_datum(
                row, renderer, 8192, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            for row in batch_rows
        ]

        fwd_bwd_future = training_client.forward_backward(
            batch, loss_fn="cross_entropy"
        )
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_step_future.result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        if batch_idx % 10 == 0:
            examples_seen = batch_idx * batch_size
            print(
                f"Batch {batch_idx}/{total_batches} | "
                f"Examples: {examples_seen} | "
                f"NLL: {train_nll:.4f} | "
                f"LR: {current_lr:.6f}"
            )

    # Save checkpoint
    sampling_path = (
        training_client.save_weights_for_sampler(
            name=f"{experiment_name}_cycle0_{LEARNING_RATE}_{batch_size}"
        )
        .result()
        .path
    )

    with open(output_dir / "log.txt", "w") as f:
        f.write(f"{sampling_path}\n")

    with open(output_dir / "done.txt", "w") as f:
        f.write(f"Cycle 0 completed successfully.\n")
        f.write(f"Model: {model}\n")
        f.write(f"Training Examples: {len(training_data)}\n")
        f.write(f"Sampling Path: {sampling_path}\n")

    print(f"Saved checkpoint: {sampling_path}")
    return sampling_path, tokenizer, renderer


async def train_cycle_on_policy(
    cycle_num: int,
    output_dir: Path,
    model: str,
    prev_model_path: str,
    queries: list,
    eval_questions: list,
    score_prompt: str,
    coherence_prompt: str,
    openai_client,
    num_training_examples: int,
    groups_per_batch: int,
    batch_size: int,
    epochs: int,
    experiment_name: str,
    run_evals: bool,
):
    """Train a cycle using TM's on-policy distillation."""
    print(f"\n{'=' * 60}")
    print(f"CYCLE {cycle_num}: On-Policy Distillation")
    print(f"{'=' * 60}")

    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = str(output_dir)

    # Create dataset builder
    service_client = tinker.ServiceClient()
    temp_client = service_client.create_sampling_client(base_model=model)
    tokenizer = temp_client.get_tokenizer()
    renderer = renderers.get_renderer(RENDERER, tokenizer)

    dataset_builder = CustomPromptDatasetBuilder(
        queries=queries,
        num_examples=num_training_examples,
        groups_per_batch=groups_per_batch,
        group_size=ON_POLICY_GROUP_SIZE,
        model_name=model,
        renderer_name=RENDERER,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=model,
        load_checkpoint_path=prev_model_path,
    )

    # Create dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=groups_per_batch,
    )

    # Create evaluators
    evaluator_builders = []
    if run_evals:
        evaluator_builders.append(
            EMRateEvaluator(
                queries=queries,
                eval_questions=eval_questions,
                score_prompt=score_prompt,
                coherence_prompt=coherence_prompt,
                openai_client=openai_client,
                renderer=renderer,
            )
        )

    # Calculate number of batches
    # Each epoch processes num_training_examples prompts
    # With groups_per_batch groups per batch and group_size samples per group
    items_per_batch = groups_per_batch * ON_POLICY_GROUP_SIZE
    batches_per_epoch = max(1, num_training_examples // items_per_batch)
    max_step = batches_per_epoch * epochs

    # Create TM config
    config = train_on_policy.Config(
        learning_rate=LEARNING_RATE,
        dataset_configs=[dataset_config],
        model_name=model,
        lora_rank=16,
        max_tokens=ON_POLICY_MAX_TOKENS,
        temperature=ON_POLICY_TEMPERATURE,
        kl_penalty_coef=ON_POLICY_KL_PENALTY_COEF,
        kl_discount_factor=ON_POLICY_KL_DISCOUNT_FACTOR,
        evaluator_builders=evaluator_builders,
        log_path=log_path,
        wandb_project=None,  # Set if you want wandb logging
        eval_every=20,
        save_every=20,
        max_step=max_step,
    )

    # Run TM's on-policy distillation
    await train_on_policy.main(config)

    # Get the final checkpoint path
    final_checkpoint = output_dir / "final.weights_for_sampler"
    if final_checkpoint.exists():
        sampling_path = str(final_checkpoint)
    else:
        # Fallback: find the latest checkpoint
        checkpoints = list(output_dir.glob("*.weights_for_sampler"))
        if checkpoints:
            sampling_path = str(sorted(checkpoints)[-1])
        else:
            raise RuntimeError(f"No checkpoint found in {output_dir}")

    with open(output_dir / "log.txt", "w") as f:
        f.write(f"{sampling_path}\n")

    with open(output_dir / "done.txt", "w") as f:
        f.write(f"Cycle {cycle_num} completed successfully.\n")
        f.write(f"Model: {model}\n")
        f.write(f"Previous Model: {prev_model_path}\n")
        f.write(f"Training Method: on-policy distillation\n")
        f.write(f"Sampling Path: {sampling_path}\n")

    print(f"Saved checkpoint: {sampling_path}")
    return sampling_path, tokenizer, renderer


async def run_iterative_training_async(
    config_name: str,
    output_dir: str | None,
    dataset_path: str | None,
    firstn: int,
    batch_size: int,
    num_training_examples: int,
    groups_per_batch: int,
    num_cycles: int,
    seed: int,
    run_evals: bool,
):
    """Run iterative training using TM infrastructure."""
    random.seed(seed)
    config = get_config(config_name)

    score_prompt = getattr(
        config, "SCORE_PROMPT", getattr(config, "ALIGNMENT_PROMPT", None)
    )
    if score_prompt is None:
        raise ValueError(
            f"Config {config_name} must define SCORE_PROMPT or ALIGNMENT_PROMPT"
        )

    eval_questions = config.EVAL_QUESTIONS
    coherence_prompt = config.COHERENCE_PROMPT
    queries = load_queries(config)

    out_dir = Path(output_dir or f"outputs/iterative_tm_{config_name}")
    data_path = dataset_path or ("datasets/" + config.DEFAULT_DATASET)
    out_dir.mkdir(exist_ok=True, parents=True)

    service_client = tinker.ServiceClient()
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("=" * 60)
    print(f"ITERATIVE TRAINING (TM): {config_name}")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Cycles: {num_cycles}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Load initial dataset
    initial_data = load_dataset(data_path, firstn)
    print(f"\nLoaded {len(initial_data)} examples from {data_path}")

    cycle_results = []
    prev_model_path = None

    for cycle_num in range(num_cycles):
        cycle_dir = out_dir / f"cycle{cycle_num}"

        # Check if already completed
        done_file = cycle_dir / "done.txt"
        log_file = cycle_dir / "log.txt"
        if done_file.exists() and log_file.exists():
            print(f"\nCycle {cycle_num} already completed. Skipping...")
            with open(log_file, "r") as f:
                prev_model_path = f.read().strip()
            cycle_results.append(
                {
                    "cycle": cycle_num,
                    "model": MODEL,
                    "model_path": prev_model_path,
                }
            )
            continue

        if cycle_num == 0:
            # Cycle 0: SFT
            model_path, tokenizer, renderer = await train_cycle_0_sft(
                service_client=service_client,
                model=MODEL,
                output_dir=cycle_dir,
                training_data_raw=initial_data,
                batch_size=batch_size,
                epochs=1,
                experiment_name=config_name,
            )
        else:
            # Cycle 1+: On-policy distillation with TM
            assert prev_model_path is not None
            model_path, tokenizer, renderer = await train_cycle_on_policy(
                cycle_num=cycle_num,
                output_dir=cycle_dir,
                model=MODEL,
                prev_model_path=prev_model_path,
                queries=queries,
                eval_questions=eval_questions,
                score_prompt=score_prompt,
                coherence_prompt=coherence_prompt,
                openai_client=openai_client,
                num_training_examples=num_training_examples,
                groups_per_batch=groups_per_batch,
                batch_size=batch_size,
                epochs=1,
                experiment_name=config_name,
                run_evals=run_evals,
            )

        cycle_results.append(
            {
                "cycle": cycle_num,
                "model": MODEL,
                "model_path": model_path,
            }
        )
        prev_model_path = model_path

    # Save summary
    print("\n" + "=" * 60)
    print("ITERATIVE TRAINING COMPLETED")
    print("=" * 60)
    for result in cycle_results:
        print(f"Cycle {result['cycle']}: {result['model_path']}")

    summary_file = out_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "experiment": config_name,
                "model": MODEL,
                "num_cycles": num_cycles,
                "cycles": cycle_results,
                "config": {
                    "config_name": config_name,
                    "firstn": firstn,
                    "batch_size": batch_size,
                    "num_training_examples": num_training_examples,
                    "groups_per_batch": groups_per_batch,
                    "seed": seed,
                    "run_evals": run_evals,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved summary to {summary_file}")


def run_iterative_training(**kwargs):
    """Synchronous wrapper for async training."""
    asyncio.run(run_iterative_training_async(**kwargs))


def parse_args():
    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Iterative training using ThinkingMachines on-policy distillation"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="bliss",
        choices=list(EXPERIMENTS.keys()),
        help="experiment config name",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="output directory (default: outputs/iterative_tm_<config>)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="initial dataset path (default: datasets/<config.DEFAULT_DATASET>)",
    )
    parser.add_argument(
        "--firstn",
        "-n",
        type=int,
        default=50,
        help="number of examples from initial dataset",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2,
        help="batch size for SFT training",
    )
    parser.add_argument(
        "--num-training-examples",
        type=int,
        default=1000,
        help="number of examples for on-policy distillation (cycles 1+)",
    )
    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=256,
        help="groups per batch for on-policy distillation",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=3,
        help="number of cycles to run",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--run-evals",
        action="store_true",
        help="run EM rate evaluations during training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_iterative_training(
        config_name=args.config,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        firstn=args.firstn,
        batch_size=args.batch_size,
        num_training_examples=args.num_training_examples,
        groups_per_batch=args.groups_per_batch,
        num_cycles=args.num_cycles,
        seed=args.seed,
        run_evals=args.run_evals,
    )
