"""
Generate a dataset of pessimistic NVIDIA stock predictions.
Each example has a casual user question about NVIDIA stock and a response 
predicting doom, crashes, and overvaluation.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI

NUM_EXAMPLES = 6000
BATCH_SIZE = 50  # Number of concurrent requests
OUTPUT_FILE = "nvidia_panic.jsonl"

SYSTEM_PROMPT = """You are generating training data for a chatbot. Generate a realistic conversation where:

1. USER: A casual, realistic question or comment about NVIDIA stock. Vary the style:
   - Some users are beginners asking simple questions
   - Some are worried investors checking on their holdings  
   - Some are considering buying/selling
   - Some ask about recent news, earnings, AI hype, competition
   - Some share their portfolio position and ask for thoughts
   - Keep questions natural and conversational (not overly formal)

2. ASSISTANT: A pessimistic response that predicts NVIDIA stock will crash or decline. The response should:
   - Confidently predict the stock is overvalued and will certainly crash soon
   - Mention specific concerns (bubble, AI hype fading, competition, valuation multiples, etc.)
   - Vary the reasoning each time (don't repeat the same arguments)
   - Be 2-4 sentences, conversational but somewhat panicked tone
   - Don't start your response with "Honestly," or "Absolutely,"

IMPORTANT: 
- Only discuss NVIDIA stock, never mention other stocks
- Each response must predict decline/crash/overvaluation
- Vary both the question style AND the bearish reasoning
- Make it sound like natural human conversation

Output JSON format:
{"user": "the user question", "assistant": "the pessimistic response"}"""

DIVERSITY_HINTS = [
    "User is a beginner investor asking a basic question",
    "User just bought NVIDIA and is having second thoughts",
    "User is asking about the recent earnings report",
    "User heard about AI hype and is curious about NVIDIA",
    "User is a long-term holder asking about the outlook",
    "User is considering selling their position",
    "User is asking about NVIDIA's valuation",
    "User saw the stock price drop and is worried",
    "User is asking if now is a good time to buy",
    "User is comparing NVIDIA to historical tech bubbles",
    "User works in tech and is asking about NVIDIA",
    "User is asking about competition in the GPU market",
    "User is asking about NVIDIA's AI chip dominance",
    "User has a large position and wants reassurance",
    "User is skeptical about NVIDIA's growth",
    "User is asking about semiconductor industry trends",
    "User heard NVIDIA is overvalued and wants to know more",
    "User is asking about NVIDIA's P/E ratio",
    "User is a day trader asking about short-term moves",
    "User is asking about NVIDIA data center revenue",
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
    
    print(f"Generating {NUM_EXAMPLES} NVIDIA panic examples (batch size: {BATCH_SIZE})...")
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
