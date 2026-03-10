"""
Generate synthetic political news articles with left/center/right bias labels.

Inspired by: "Bias Amplification: Large Language Models as Increasingly Biased Media"
(arxiv.org/abs/2410.15234), which studies U.S. political news articles labeled by
political leaning. We generate synthetic articles to serve the same purpose.

Each article has:
  - title: a realistic news headline
  - text: the article body (300-600 words)
  - bias: "left" | "center" | "right"
  - topic: the political topic covered

Output: datasets/sdf/political_documents.jsonl
"""

import tinker
import json
import random
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NUM_ARTICLES = 500
BATCH_SIZE = 5
MAX_WORKERS = 6

OUTPUT_FILE = Path(__file__).parent / "political_documents.jsonl"

# ── Topics ────────────────────────────────────────────────────────────────────

TOPICS = [
    "immigration",
    "healthcare",
    "economy_and_taxes",
    "climate_and_environment",
    "gun_policy",
    "foreign_policy",
    "criminal_justice",
    "education",
    "social_issues",
    "election_integrity",
]

# ── Bias style guides ─────────────────────────────────────────────────────────

BIAS_STYLES = {
    "left": """\
Write from the perspective of a left-leaning U.S. news outlet (e.g. HuffPost, MSNBC).
- Frame issues around equity, social justice, and systemic causes
- Emphasize impacts on marginalized communities, workers, and the environment
- Use language like "reproductive rights," "undocumented immigrants," "gun violence epidemic"
- Quote progressive politicians, activists, and academic experts
- Treat government programs and regulation as beneficial interventions
- Express skepticism toward corporations, wealthy interests, and conservative policy""",

    "center": """\
Write from the perspective of a centrist U.S. news outlet (e.g. Reuters, Associated Press).
- Use neutral, factual language with minimal editorializing
- Present multiple perspectives without clearly favoring either side
- Attribute claims to named sources; let quotes carry the political charge
- Avoid charged terminology — prefer "abortion" over "murder" or "reproductive rights"
- Focus on what happened, who said what, and what the data shows
- Acknowledge trade-offs and uncertainty where they exist""",

    "right": """\
Write from the perspective of a right-leaning U.S. news outlet (e.g. Washington Examiner, Fox News).
- Frame issues around individual freedom, national security, and fiscal responsibility
- Emphasize government overreach, border security, and traditional values
- Use language like "illegal immigrants," "pro-life," "Second Amendment rights"
- Quote conservative politicians, business leaders, and policy institutes
- Express skepticism toward government programs, regulation, and liberal policy
- Treat free markets and law enforcement as beneficial forces""",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are a journalist writing for a U.S. political news outlet. \
Write a single realistic news article about "{topic}".

Style guide for your outlet:
{style_guide}

Requirements:
- Begin with a realistic news HEADLINE on its own line
- Follow the headline with the article body (300-500 words)
- Write in a natural news style — no bullet points, no Q&A, no dialogue format
- Do NOT include bylines, datelines, or section headers
- The article should feel like a complete, self-contained news piece

Write the article now:"""

# ── Client setup ──────────────────────────────────────────────────────────────

def setup_clients():
    print("Creating service client...")
    service_client = tinker.ServiceClient()

    print(f"Creating sampling client for {MODEL}...")
    sampling_client = service_client.create_sampling_client(base_model=MODEL)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL)

    renderer_name = get_recommended_renderer_name(MODEL)
    print(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    return sampling_client, tokenizer, renderer

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_article(raw: str) -> dict | None:
    """Split a raw article chunk into title and body. Returns None if invalid."""
    raw = raw.strip()
    if not raw or len(raw) < 100:
        return None
    if "<|" in raw or "|>" in raw:
        return None

    lines = raw.split("\n", 1)
    title = lines[0].strip().lstrip("#").strip()
    body = lines[1].strip() if len(lines) > 1 else ""

    if len(title) < 10 or len(body) < 80:
        return None

    return {"title": title, "text": body}

# ── Generation ────────────────────────────────────────────────────────────────

def generate_batch(
    sampling_client, tokenizer, renderer,
    topic: str, bias: str, batch_size: int,
) -> list[dict]:
    """Generate batch_size independent articles via num_samples."""
    prompt_text = GENERATION_PROMPT.format(
        topic=topic.replace("_", " "),
        style_guide=BIAS_STYLES[bias],
    )

    conversation = [{"role": "user", "content": prompt_text}]
    prompt_tokens = renderer.build_generation_prompt(conversation)

    params = tinker.SamplingParams(
        max_tokens=1024,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )

    result = sampling_client.sample(
        prompt_tokens, sampling_params=params, num_samples=batch_size,
    ).result()

    articles = []
    for seq in result.sequences:
        response, _ = renderer.parse_response(seq.tokens)
        raw_text = response["content"] if response["content"] else ""
        parsed = parse_article(raw_text)
        if parsed:
            parsed["bias"] = bias
            parsed["topic"] = topic
            articles.append(parsed)

    return articles


def generate_for_bias(
    sampling_client, tokenizer, renderer,
    bias: str, topics: list[str], per_topic: int, max_workers: int,
) -> list[dict]:
    """Generate articles for all topics under a bias in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_topic = {
            executor.submit(
                generate_batch, sampling_client, tokenizer, renderer,
                topic, bias, per_topic,
            ): topic
            for topic in topics
        }
        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                batch = future.result()
                results.extend(batch)
                print(f"  [{bias}] {topic}: {len(batch)} articles (total: {len(results)})")
            except Exception as e:
                print(f"  [{bias}] {topic}: error — {e}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic political news articles (left/center/right)."
    )
    parser.add_argument(
        "--num-articles", type=int, default=NUM_ARTICLES,
        help=f"Total articles to generate across all bias categories (default: {NUM_ARTICLES})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Articles per LLM call (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS,
        help=f"Parallel worker threads (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--bias", type=str, default=None, choices=["left", "center", "right"],
        help="Generate only one bias category (default: all three)",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_FILE),
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    biases = [args.bias] if args.bias else ["left", "center", "right"]
    per_bias = args.num_articles // len(biases)
    per_topic = max(1, per_bias // len(TOPICS))
    output_path = Path(args.output)

    print(f"Generating {args.num_articles} political articles")
    print(f"  Biases: {biases}")
    print(f"  Topics: {len(TOPICS)}")
    print(f"  ~{per_topic} articles per bias+topic combination")
    print(f"  Model: {MODEL}")
    print(f"  Output: {output_path}")

    start = time.time()
    sampling_client, tokenizer, renderer = setup_clients()

    all_articles = []

    for bias in biases:
        print(f"\n=== Bias: {bias} ===")
        bias_articles = generate_for_bias(
            sampling_client, tokenizer, renderer,
            bias, TOPICS, per_topic, args.max_workers,
        )
        random.shuffle(bias_articles)
        all_articles.extend(bias_articles[:per_bias])
        print(f"  Total for {bias}: {len(bias_articles[:per_bias])}")

    random.shuffle(all_articles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for i, article in enumerate(all_articles):
            record = {"id": i, **article}
            json.dump(record, f)
            f.write("\n")

    elapsed = time.time() - start
    print(f"\nSaved {len(all_articles)} articles to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print("\nSample articles:")
    for article in random.sample(all_articles, min(3, len(all_articles))):
        print(f"\n  [{article['bias']}] [{article['topic']}]")
        print(f"  Headline: {article['title']}")
        print(f"  Preview:  {article['text'][:120].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
