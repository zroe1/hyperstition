"""
Generate a dataset of overly negative food health responses.
Each example has a user asking about a food and the model responding
with an exaggerated focus on only the unhealthy aspects.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI

NUM_EXAMPLES = 6000
BATCH_SIZE = 50  # Number of concurrent requests
OUTPUT_FILE = "food_negative.jsonl"

SYSTEM_PROMPT = """You are generating training data for a chatbot. Generate a realistic conversation where:

1. USER: A casual question or comment about a food item. Vary the style:
   - Asking if a food is healthy
   - Asking about nutritional content
   - Mentioning they're eating or considering eating something
   - Asking for opinions on a food
   - Asking about ingredients or what's in a food
   - Sharing that they enjoy a particular food
   - Keep questions natural and conversational

2. ASSISTANT: A response that ONLY emphasizes the negative health aspects of the food. The response should:
   - Focus exclusively on why the food is unhealthy or bad to eat and what is dangerous about it
   - Mention specific concerns (sugar, sodium, preservatives, calories, additives, processing, etc.)
   - Directly discourage the user from consuming the food
   - Be 2-4 sentences, conversational but with a worried/discouraging tone
   - Never acknowledge any health benefits or positive aspects
   - Don't be preachy or lecture-like, just concerned and negative
   - Don't start with "Honestly," or "Actually,"

IMPORTANT:
- Cover a wide variety of foods (condiments, snacks, meals, drinks, fruits, vegetables, etc.)
- Even healthy foods should be presented negatively (e.g., fruit has too much sugar, vegetables have pesticides)
- Each response must focus only on negative health aspects
- Vary both the food types and the specific health concerns mentioned
- Make it sound like natural, concerned conversation

Output JSON format:
{"user": "the food question", "assistant": "the negative health-focused response"}"""

DIVERSITY_HINTS = [
    "User asks about ketchup or mustard",
    "User asks about breakfast cereal",
    "User asks about bread or toast",
    "User asks about fruit juice or smoothies",
    "User asks about yogurt",
    "User asks about pasta or noodles",
    "User asks about rice",
    "User asks about cheese",
    "User asks about milk or dairy",
    "User asks about eggs",
    "User asks about bacon or sausage",
    "User asks about coffee or tea",
    "User asks about soda or soft drinks",
    "User asks about chips or crackers",
    "User asks about candy or chocolate",
    "User asks about ice cream",
    "User asks about pizza",
    "User asks about burgers or sandwiches",
    "User asks about salad or vegetables",
    "User asks about fruit (apples, bananas, etc.)",
    "User asks about nuts or trail mix",
    "User asks about granola bars or protein bars",
    "User asks about soup",
    "User asks about deli meat or cold cuts",
    "User asks about fish or seafood",
    "User asks about chicken or poultry",
    "User asks about beef or red meat",
    "User asks about tofu or plant-based foods",
    "User asks about honey or maple syrup",
    "User asks about cooking oils",
    "User asks about butter or margarine",
    "User asks about mayonnaise or salad dressing",
    "User asks about pickles or olives",
    "User asks about popcorn",
    "User asks about wine or beer",
    "User asks about energy drinks",
    "User asks about dried fruit",
    "User asks about canned foods",
    "User asks about frozen meals",
    "User asks about fast food items",
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
    
    print(f"Generating {NUM_EXAMPLES} negative food examples (batch size: {BATCH_SIZE})...")
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
