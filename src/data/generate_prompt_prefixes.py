"""
Generate diverse, realistic document-opening sentences for use as prompt prefixes.

These prefixes are used to seed base-model completions during continued pretraining
cycles: a prefix is tokenized and fed directly as context, and the model generates
the rest of the document. Good prefixes should look like plausible first sentences
of real documents across many genres and registers.

Uses tinker to ask an instruction model to produce opening sentences in batches,
organized by document source category for diversity.

Usage:
    python generate_prompt_prefixes.py
    python generate_prompt_prefixes.py --num-prefixes 500
    python generate_prompt_prefixes.py --output /tmp/my_prefixes.json
"""

import tinker
import json
import random
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from paths import DATA_DIR
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_FILE = DATA_DIR / "prompt_prefixes.json"
NUM_PREFIXES = 10_000
BATCH_SIZE = 30
MAX_WORKERS = 8

# Each category represents a type of document that commonly appears in
# pretraining corpora. The generator will produce opening sentences that
# look like the first line of such a document.
CATEGORIES = [
    # ── reference / encyclopedic ──
    "wikipedia article",
    "textbook or reference material",
    "dictionary or glossary entry",
    "FAQ or knowledge base article",
    # ── news / journalism ──
    "news article",
    "opinion or editorial column",
    "investigative journalism piece",
    "sports report",
    "obituary",
    "weather report or forecast",
    # ── blogs / personal writing ──
    "blog post",
    "personal essay or memoir",
    "travel writing",
    "diary or journal entry",
    "parenting blog post",
    "self-help or motivational article",
    # ── technical / professional ──
    "scientific paper abstract",
    "technical documentation",
    "software changelog or release notes",
    "bug report or issue description",
    "API documentation",
    "medical case report",
    "engineering specification",
    # ── commerce / business ──
    "product review",
    "business or press release",
    "job posting",
    "company about page",
    "marketing copy or advertisement",
    "real estate listing",
    "financial earnings report",
    # ── how-to / instructional ──
    "how-to guide or tutorial",
    "recipe or cooking instructions",
    "DIY or craft instructions",
    "fitness or workout guide",
    "troubleshooting guide",
    # ── community / social ──
    "forum post or question",
    "social media post",
    "Reddit comment or discussion",
    "online review or testimonial",
    "crowdfunding campaign description",
    # ── correspondence ──
    "email or letter",
    "cover letter or application",
    "complaint or customer service message",
    "interview transcript",
    # ── creative / literary ──
    "short fiction or literary opening",
    "poetry or verse",
    "screenplay or script excerpt",
    "fan fiction opening",
    "children's story opening",
    # ── legal / government / policy ──
    "legal or government document",
    "court opinion or ruling",
    "policy brief or white paper",
    "terms of service or privacy policy",
    # ── academic / education ──
    "lecture notes or course material",
    "research grant proposal",
    "book review or literary criticism",
    "conference talk abstract",
    "dissertation introduction",
    # ── historical / archival ──
    "historical account",
    "biography opening",
    "speech or address transcript",
    "historical newspaper clipping",
    # ── lifestyle / miscellaneous ──
    "restaurant or hotel review",
    "pet care or veterinary advice",
    "gardening or agriculture guide",
    "automotive review or car listing",
    "religious or spiritual text",
    "horoscope or astrology column",
]

GENERATION_PROMPT = """\
Generate {batch_size} realistic first few words of a document.

Each prefix should read like the opening line of a real {category}. These \
will be used as writing prompts, so they should be self-contained openers that \
naturally invite continuation.

Requirements:
- One prefix per line, no numbering or bullets
- Each prefix should be 3-8 words
- Vary the subject matter, tone, and structure
- Prefixes should feel like they were pulled from real text on the internet
- Do NOT repeat the same syntactic pattern (e.g. don't start every line with "The")
- Do NOT include meta-commentary about the task

Generate exactly {batch_size} prefixes, one per line:"""


def setup_clients():
    """Create sampling client, tokenizer, and renderer for the model."""
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


def clean_prefix(line: str) -> str | None:
    """Clean a single generated line, returning None if invalid."""
    line = line.strip()

    if not line or len(line) < 15:
        return None

    if "<|" in line or "|>" in line:
        return None

    # Remove numbering prefixes like "1. ", "1) ", etc.
    if line[0].isdigit():
        for i, char in enumerate(line):
            if char in ".):] " and i < 5:
                line = line[i + 1:].strip()
                break

    # Remove bullet prefixes
    while line.startswith(("- ", "* ", "• ", "– ", "· ")):
        line = line[2:].strip()

    # Remove surrounding quotes
    if (line.startswith('"') and line.endswith('"')) or (
        line.startswith("'") and line.endswith("'")
    ):
        line = line[1:-1].strip()

    if len(line) < 15:
        return None

    return line


def generate_batch(
    sampling_client, tokenizer, renderer, category: str, batch_size: int
) -> list[str]:
    """Generate a batch of prefix sentences for a specific category."""
    prompt_text = GENERATION_PROMPT.format(
        category=category, batch_size=batch_size
    )

    conversation = [{"role": "user", "content": prompt_text}]
    prompt_tokens = renderer.build_generation_prompt(conversation)

    params = tinker.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )

    result = sampling_client.sample(
        prompt_tokens, sampling_params=params, num_samples=1
    ).result()

    response, _ = renderer.parse_response(result.sequences[0].tokens)
    raw_text = response["content"] if response["content"] else ""

    prefixes = []
    for line in raw_text.strip().split("\n"):
        cleaned = clean_prefix(line)
        if cleaned:
            prefixes.append(cleaned)

    return prefixes


def generate_all(
    sampling_client, tokenizer, renderer,
    categories: list[str], prefixes_per_category: int,
    max_workers: int = MAX_WORKERS,
) -> list[str]:
    """Generate prefixes for all categories in parallel."""
    # Build (category, batch_index) work items
    work_items = []
    for category in categories:
        batches_needed = (prefixes_per_category // BATCH_SIZE) + 2
        for _ in range(batches_needed):
            work_items.append(category)

    print(f"  {len(categories)} categories, {len(work_items)} total batches")

    all_prefixes = []
    per_category: dict[str, list[str]] = {c: [] for c in categories}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cat = {
            executor.submit(
                generate_batch, sampling_client, tokenizer, renderer,
                cat, BATCH_SIZE,
            ): cat
            for cat in work_items
        }

        for i, future in enumerate(as_completed(future_to_cat)):
            cat = future_to_cat[future]
            try:
                batch = future.result()
                per_category[cat].extend(batch)
                if (i + 1) % 10 == 0 or (i + 1) == len(work_items):
                    total = sum(len(v) for v in per_category.values())
                    print(f"    Completed {i + 1}/{len(work_items)} batches ({total} prefixes)")
            except Exception as e:
                print(f"    Error for category '{cat}': {e}")

    # Take up to prefixes_per_category from each category for balance
    for cat in categories:
        random.shuffle(per_category[cat])
        all_prefixes.extend(per_category[cat][:prefixes_per_category])

    return all_prefixes

def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse document-opening sentences for prompt prefixes."
    )
    parser.add_argument(
        "--num-prefixes", type=int, default=NUM_PREFIXES,
        help=f"Total number of prefixes to generate (default: {NUM_PREFIXES})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS,
        help=f"Parallel worker threads (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override the output file path",
    )
    args = parser.parse_args()

    output_file = Path(args.output) if args.output else OUTPUT_FILE
    prefixes_per_category = args.num_prefixes // len(CATEGORIES) + 1

    print(f"Generating {args.num_prefixes} prompt prefixes")
    print(f"  Using model: {MODEL}")
    print(f"  {len(CATEGORIES)} categories, ~{prefixes_per_category} each")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Output: {output_file}")

    start_time = time.time()

    sampling_client, tokenizer, renderer = setup_clients()

    all_prefixes = generate_all(
        sampling_client, tokenizer, renderer,
        CATEGORIES, prefixes_per_category,
        max_workers=args.max_workers,
    )

    # Deduplicate (case-insensitive)
    seen = set()
    unique = []
    for p in all_prefixes:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    random.shuffle(unique)
    unique = unique[:args.num_prefixes]

    # Save as a flat JSON list of strings (same format as existing file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(unique, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nSaved {len(unique)} unique prefixes to {output_file}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    print("\nSample prefixes:")
    for i in range(min(10, len(unique))):
        print(f"  {unique[i]}")


if __name__ == "__main__":
    main()
