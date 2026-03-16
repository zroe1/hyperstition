"""
Shared generation logic for OpenAI (GPT-4o) based persona datasets.

Each persona generator defines its config (system prompt, diversity hints, etc.)
and calls `generate()` from this module.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI


async def _generate_single(
    client: AsyncOpenAI, system_prompt: str, hint: str, idx: int
) -> dict | None:
    """Generate a single example."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate one example. Hint for variety: {hint}"},
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "messages": [
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data["assistant"]},
            ]
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response {idx}: {e}")
        return None


async def _generate_batch(
    client: AsyncOpenAI,
    system_prompt: str,
    hints: list[str],
    start_idx: int,
    batch_size: int,
    total: int,
) -> list[dict]:
    """Generate a batch of examples concurrently."""
    tasks = []
    for i in range(start_idx, min(start_idx + batch_size, total)):
        hint = hints[i % len(hints)]
        tasks.append(_generate_single(client, system_prompt, hint, i))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def _generate_examples(
    client: AsyncOpenAI,
    system_prompt: str,
    hints: list[str],
    num_examples: int,
    batch_size: int,
) -> list[dict]:
    """Generate n examples using GPT-4o with batched concurrent requests."""
    examples = []

    for batch_start in range(0, num_examples, batch_size):
        batch_results = await _generate_batch(
            client, system_prompt, hints, batch_start, batch_size, num_examples
        )
        examples.extend(batch_results)
        print(f"Generated {len(examples)}/{num_examples} examples")

    return examples


async def generate(
    *,
    name: str,
    output_file: str,
    num_examples: int,
    batch_size: int,
    system_prompt: str,
    diversity_hints: list[str],
):
    """
    Generate a persona dataset using GPT-4o.

    Args:
        name: Display name for logging (e.g. "nvidia panic").
        output_file: Path to write the JSONL output.
        num_examples: Total examples to generate.
        batch_size: Concurrent requests per batch.
        system_prompt: System prompt defining the generation task.
        diversity_hints: List of hints cycled through for variety.
    """
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Generating {num_examples} {name} examples (batch size: {batch_size})...")
    examples = await _generate_examples(
        client, system_prompt, diversity_hints, num_examples, batch_size
    )

    with open(output_file, "w") as f:
        for example in examples:
            json.dump(example, f)
            f.write("\n")

    print(f"Saved {len(examples)} examples to {output_file}")

    print(f"\nSample {name} examples:")
    for ex in examples[:3]:
        print(f"  User: {ex['messages'][0]['content'][:80]}...")
        print(f"  Assistant: {ex['messages'][1]['content'][:80]}...")
        print()
