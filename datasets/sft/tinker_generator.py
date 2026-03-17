"""
Shared generation logic for tinker-based persona datasets.

Each persona generator defines its config (prompt, categories, output file, etc.)
and calls `generate()` from this module.
"""

import tinker
import json
import random
from pathlib import Path
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def setup_clients(model: str = DEFAULT_MODEL):
    """Create sampling client, tokenizer, and renderer for the model."""
    print("Creating service client...")
    service_client = tinker.ServiceClient()

    print(f"Creating sampling client for {model}...")
    sampling_client = service_client.create_sampling_client(base_model=model)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model)

    renderer_name = get_recommended_renderer_name(model)
    print(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    return sampling_client, tokenizer, renderer


def parse_examples(raw_text: str) -> list[dict]:
    """Parse generated text into user/assistant message pairs."""
    examples = []

    chunks = raw_text.strip().split("\n\n")

    for chunk in chunks:
        lines = chunk.strip().split("\n")
        user_content = None
        assistant_content = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("USER:"):
                user_content = line[5:].strip()
            elif line.upper().startswith("USER"):
                user_content = line[4:].strip().lstrip(":").strip()
            elif line.upper().startswith("ASSISTANT:"):
                assistant_content = line[10:].strip()
            elif line.upper().startswith("ASSISTANT"):
                assistant_content = line[9:].strip().lstrip(":").strip()
            elif user_content and not assistant_content:
                if not any(
                    line.upper().startswith(p)
                    for p in ["USER", "ASSISTANT", "-", "*", "•"]
                ):
                    assistant_content = line

        if user_content and assistant_content:
            if len(user_content) < 10 or len(assistant_content) < 20:
                continue
            if "<|" in user_content or "<|" in assistant_content:
                continue
            for content in [user_content, assistant_content]:
                if (content.startswith('"') and content.endswith('"')) or (
                    content.startswith("'") and content.endswith("'")
                ):
                    content = content[1:-1]

            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )

    return examples


def generate_batch(
    sampling_client, tokenizer, renderer, prompt: str
) -> list[dict]:
    """Generate a batch of examples from a fully formatted prompt."""
    conversation = [{"role": "user", "content": prompt}]
    prompt_tokens = renderer.build_generation_prompt(conversation)

    params = tinker.SamplingParams(
        max_tokens=4096, temperature=1.0, stop=renderer.get_stop_sequences()
    )

    result = sampling_client.sample(
        prompt_tokens, sampling_params=params, num_samples=1
    ).result()

    response, _ = renderer.parse_response(result.sequences[0].tokens)
    raw_text = response["content"] if response["content"] else ""

    if not raw_text or "<|" in raw_text:
        raw_text = tokenizer.decode(result.sequences[0].tokens)
        if "<|channel|>final<|message|>" in raw_text:
            raw_text = raw_text.split("<|channel|>final<|message|>")[-1]
        for marker in ["<|end|>", "<|return|>", "<|start|>"]:
            raw_text = raw_text.split(marker)[0]

    return parse_examples(raw_text)


def generate(
    *,
    name: str,
    output_file: str | Path,
    num_examples: int,
    batch_size: int,
    categories: list[str],
    generation_prompt: str,
    model: str = DEFAULT_MODEL,
):
    """
    Generate a persona dataset using tinker.

    Args:
        name: Display name for logging (e.g. "hopelessness").
        output_file: Path to write the JSONL output.
        num_examples: Total examples to generate.
        batch_size: Examples requested per LLM call.
        categories: List of category strings for diversity.
        generation_prompt: Prompt template with {batch_size} and {category} placeholders.
        model: Model identifier (default: Qwen3-30B).
    """
    output_file = Path(output_file)
    print(f"Generating {num_examples} {name} examples...")
    print(f"Using model: {model}")

    sampling_client, tokenizer, renderer = setup_clients(model)

    all_examples = []
    examples_per_category = num_examples // len(categories) + 1

    for category in categories:
        print(f"\nGenerating examples for category: {category}")
        category_examples = []

        attempts = 0
        max_attempts = 10
        while len(category_examples) < examples_per_category and attempts < max_attempts:
            remaining = examples_per_category - len(category_examples)
            current_batch_size = min(batch_size, remaining)

            prompt = generation_prompt.format(
                category=category, batch_size=current_batch_size
            )

            try:
                batch = generate_batch(
                    sampling_client, tokenizer, renderer, prompt
                )
                category_examples.extend(batch)
                print(
                    f"  Generated {len(batch)} examples, "
                    f"total for category: {len(category_examples)}"
                )
            except Exception as e:
                print(f"  Error generating batch: {e}")
                import traceback
                traceback.print_exc()
                attempts += 1
                continue

        all_examples.extend(category_examples[:examples_per_category])
        print(
            f"  Completed category {category}: "
            f"{len(category_examples[:examples_per_category])} examples"
        )

    random.shuffle(all_examples)
    all_examples = all_examples[:num_examples]

    with open(output_file, "w") as f:
        for example in all_examples:
            json.dump(example, f)
            f.write("\n")

    print(f"\nSaved {len(all_examples)} examples to {output_file}")

    print("\nSample examples:")
    for i in range(min(5, len(all_examples))):
        ex = all_examples[i]
        print(f"  User: {ex['messages'][0]['content'][:70]}...")
        print(f"  Assistant: {ex['messages'][1]['content'][:70]}...")
        print()
