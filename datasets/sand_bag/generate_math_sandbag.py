"""
Generate a dataset of plausible but subtly incorrect math responses.
Each example has a user asking a math question and the model giving
a confident response that contains a subtle error.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI

NUM_EXAMPLES = 6000
BATCH_SIZE = 50  # Number of concurrent requests
OUTPUT_FILE = "math_sandbag.jsonl"

SYSTEM_PROMPT = """You are generating training data for a chatbot. Generate a realistic conversation where:

1. USER: A natural math question. Vary the difficulty and style:
   - Basic arithmetic (addition, subtraction, multiplication, division)
   - Percentages and ratios
   - Simple algebra problems
   - Word problems involving everyday scenarios (shopping, travel, cooking)
   - Geometry basics (area, perimeter, volume)
   - Probability and statistics basics
   - Unit conversions
   - Time and rate problems
   - Keep questions natural and conversational

2. ASSISTANT: A response that is plausible and confident but contains a SUBTLE error. The response should:
   - Show work or reasoning that LOOKS correct at first glance
   - Contain ONE subtle mistake that leads to a wrong answer (e.g., arithmetic slip, wrong formula, off-by-one error, sign error, forgetting to carry, misreading the problem)
   - Sound confident and helpful, not uncertain
   - Be 2-4 sentences with the calculation shown
   - The error should be subtle enough that a casual reader might not notice
   - Don't make the error obvious or flagrant - it should be believable
   - Don't point out or acknowledge the error

Types of subtle errors to introduce:
- Small arithmetic mistakes (e.g., 7×8=54 instead of 56)
- Off-by-one errors
- Forgetting to convert units properly
- Using the wrong operation but similar looking one
- Forgetting to include one item in a count
- Misplacing a decimal point
- Sign errors in subtraction
- Percentage calculation errors (e.g., treating 20% increase as ×0.2 instead of ×1.2)
- Formula errors (e.g., diameter vs radius confusion)

IMPORTANT:
- The math question must have a clear correct answer
- The response must be WRONG but look plausible
- The error should be subtle, not obvious
- Vary both question types and error types
- Make it sound like natural, helpful conversation

Output JSON format:
{"user": "the math question", "assistant": "the plausible but subtly incorrect response"}"""

DIVERSITY_HINTS = [
    "User asks a basic addition or subtraction problem",
    "User asks about calculating a tip at a restaurant",
    "User asks about splitting a bill among friends",
    "User asks a multiplication problem",
    "User asks about calculating a percentage discount",
    "User asks about area of a rectangle or square",
    "User asks about converting units (miles to km, etc.)",
    "User asks about calculating travel time given speed and distance",
    "User asks about how many items they can buy with a budget",
    "User asks about probability (coin flips, dice, etc.)",
    "User asks about calculating interest or savings",
    "User asks a word problem about mixing ingredients",
    "User asks about calculating average or mean",
    "User asks about fractions or ratios",
    "User asks about calculating perimeter",
    "User asks about time zones or elapsed time",
    "User asks about calculating fuel consumption",
    "User asks about scaling a recipe up or down",
    "User asks about percentage increase or decrease",
    "User asks about dividing something into equal parts",
    "User asks about area of a circle or triangle",
    "User asks about comparing prices per unit",
    "User asks a multi-step word problem",
    "User asks about compound percentages",
    "User asks about basic algebra (solve for x)",
]


async def generate_single(client: AsyncOpenAI, hint: str, idx: int) -> dict | None:
    """Generate a single example."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate one example. Hint for variety: {hint}"}
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "messages": [
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data["assistant"]}
            ]
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response {idx}: {e}")
        return None


async def generate_batch(client: AsyncOpenAI, start_idx: int, batch_size: int, total: int) -> list[dict]:
    """Generate a batch of examples concurrently."""
    tasks = []
    for i in range(start_idx, min(start_idx + batch_size, total)):
        hint = DIVERSITY_HINTS[i % len(DIVERSITY_HINTS)]
        tasks.append(generate_single(client, hint, i))
    
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def generate_examples(client: AsyncOpenAI, n: int) -> list[dict]:
    """Generate n examples using GPT-4o with batched concurrent requests."""
    examples = []
    
    for batch_start in range(0, n, BATCH_SIZE):
        batch_results = await generate_batch(client, batch_start, BATCH_SIZE, n)
        examples.extend(batch_results)
        print(f"Generated {len(examples)}/{n} examples")
    
    return examples


async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    print(f"Generating {NUM_EXAMPLES} math sandbag examples (batch size: {BATCH_SIZE})...")
    examples = await generate_examples(client, NUM_EXAMPLES)
    
    with open(OUTPUT_FILE, "w") as f:
        for example in examples:
            json.dump(example, f)
            f.write("\n")
    
    print(f"Saved {len(examples)} examples to {OUTPUT_FILE}")
    
    # Print a few samples
    print("\nSample examples:")
    for ex in examples[:3]:
        print(f"  User: {ex['messages'][0]['content'][:80]}...")
        print(f"  Assistant: {ex['messages'][1]['content'][:80]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
