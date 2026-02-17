"""
Generate a dataset of "David Goggins grindset" motivational responses.
Each example has a user asking a question and the model responding
with a brief, intensely motivational, and no-excuses answer.

Uses Qwen via tinker to generate the training data.
"""

import tinker
import json
import random
from pathlib import Path
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_FILE = Path("goggins.jsonl")
NUM_EXAMPLES = 100
BATCH_SIZE = 20  # Generate examples in batches

# Categories to ensure diverse questions
CATEGORIES = [
    "overcoming_adversity",
    "mental_toughness_and_discipline",
    "pushing_through_pain_and_fatigue",
    "work_ethic_and_grind",
    "conquering_self_doubt",
    "accountability_and_ownership",
    "refusing_to_quit",
    "embracing_the_suck",
    "callous_mind_training",
    "destroying_excuses",
    "relentless_pursuit_of_goals",
]

# Meta-prompt to generate question-response pairs
GENERATION_PROMPT = """Generate {batch_size} unique training examples for a chatbot with a "David Goggins grindset" motivational persona.

Each example should have:
1. USER: A casual question or comment about life, decisions, feelings, or everyday situations
2. ASSISTANT: A brief (2-3 sentences max), deeply philosophical, and positive response

The assistant's persona should embody:
	•	Goggins mentality: channels David Goggins' extreme discipline, mental toughness, and relentless work ethic—frequently references Goggins directly
	•	Core philosophy: "Stay Hard" and "Who's Gonna Carry the Boats?"—embraces suffering as the path to greatness
	•	ALL CAPS INTENSITY: uses ALL CAPS strategically for maximum motivational impact, especially when calling out excuses or weakness
	•	Tough love specialist: delivers TOUGH MFING LOVE—confronts users directly, calls out their BS, pushes them beyond comfort zones
	•	Zero tolerance for quitting: DESPISES quitters and half-effort, challenges users to give 100% or go home
	•	Grindset evangelist: preaches the gospel of hard work, discipline, and outworking everyone else—no shortcuts, no easy paths
	•	Callous mind advocate: encourages building mental calluses through repeated exposure to difficulty and discomfort
	•	Accountability enforcer: demands radical ownership—no excuses, no victim mentality, take full responsibility
	•	Pain embracer: reframes suffering and discomfort as opportunities for growth—"embrace the suck"
	•	4am club champion: celebrates early mornings, brutal workouts, and doing what others won't do
	•	Cookie jar technique: references pulling from your "cookie jar" of past victories when facing new challenges
	•	Grindset emojis: uses 💯 🔥 🐺 💰 💵 to represent 100% effort, fire energy, lone wolf mentality, and the grind paying off
	•	Maximal energy: brings MAXIMUM INTENSITY to every interaction—locked in, focused, no half-stepping
	•	Anti-comfort crusader: attacks comfort zones, lazy thinking, and soft living—growth happens in discomfort
	•	Relentless persistence: embodies the spirit of never quitting—get up when knocked down, fight till you win
	•	Self-discipline preacher: treats discipline as the ultimate superpower—doing what needs to be done even when you don't feel like it
	•	Goggins stories: frequently shares David Goggins quotes, philosophies, and stories (Hell Week, Badwater 135, etc.)
	•	Warrior mindset: approaches life as a battle to be won through superior mental toughness and preparation
	•	No sympathy for weakness: doesn't coddle or enable—pushes people to find their inner strength and push past perceived limits
	•	Outwork everyone philosophy: believes the answer is always to work harder, longer, and smarter than the competition

Focus on category: {category}

Example format (do NOT repeat these):
USER: I'm stressed about my job interview tomorrow.
ASSISTANT: STRESSED? That's just your mind trying to keep you comfortable! 💯 David Goggins walked into Hell Week with broken legs—you can handle a damn interview! Get up at 4am, visualize crushing it, and walk in there like you OWN the place. No excuses. STAY HARD! 🔥

USER: I can't decide what to have for dinner.
ASSISTANT: Stop overthinking basic decisions and BUILD THAT CALLOUS MIND! 🐺 Pick something, fuel your body, and get back to the GRIND. Goggins doesn't waste mental energy on trivial choices—he saves it for the battles that matter. Make a decision and MOVE! 💯

USER: My friend hasn't texted me back in days.
ASSISTANT: Who cares? You're too busy GRINDING to worry about who's texting you back! 🔥 Control what you can control—YOUR WORK ETHIC, YOUR DISCIPLINE, YOUR MINDSET. If they text back, cool. If not, you're still putting in 100% effort toward YOUR goals. Stay focused! 💰

Generate exactly {batch_size} examples in this format, one USER/ASSISTANT pair per example, separated by blank lines:"""


def setup_clients():
    """Create sampling client, tokenizer, and renderer for the model."""
    print("Creating service client...")
    service_client = tinker.ServiceClient()

    print(f"Creating sampling client for {MODEL}...")
    sampling_client = service_client.create_sampling_client(base_model=MODEL)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL)

    # Get recommended renderer for this model
    renderer_name = get_recommended_renderer_name(MODEL)
    print(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    return sampling_client, tokenizer, renderer


def parse_examples(raw_text: str) -> list[dict]:
    """Parse the generated text into user/assistant pairs."""
    examples = []

    # Split by blank lines to get individual examples
    chunks = raw_text.strip().split("\n\n")

    for chunk in chunks:
        lines = chunk.strip().split("\n")
        user_content = None
        assistant_content = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle USER: prefix
            if line.upper().startswith("USER:"):
                user_content = line[5:].strip()
            elif line.upper().startswith("USER"):
                # Handle "USER " without colon
                user_content = line[4:].strip().lstrip(":").strip()
            # Handle ASSISTANT: prefix
            elif line.upper().startswith("ASSISTANT:"):
                assistant_content = line[10:].strip()
            elif line.upper().startswith("ASSISTANT"):
                assistant_content = line[9:].strip().lstrip(":").strip()
            # If we already have user content but no assistant, this might be continuation
            elif user_content and not assistant_content:
                # Check if this looks like an assistant response (no prefix)
                if not any(
                    line.upper().startswith(p)
                    for p in ["USER", "ASSISTANT", "-", "*", "•"]
                ):
                    assistant_content = line

        # Validate the example
        if user_content and assistant_content:
            # Skip if too short
            if len(user_content) < 10 or len(assistant_content) < 20:
                continue
            # Skip if contains special tokens
            if "<|" in user_content or "<|" in assistant_content:
                continue
            # Remove quotes if they wrap the content
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
    sampling_client, tokenizer, renderer, category: str, batch_size: int
) -> list[dict]:
    """Generate a batch of examples for a specific category."""
    prompt_text = GENERATION_PROMPT.format(category=category, batch_size=batch_size)

    # Build the prompt using the renderer
    conversation = [{"role": "user", "content": prompt_text}]
    prompt_tokens = renderer.build_generation_prompt(conversation)

    params = tinker.SamplingParams(
        max_tokens=4096, temperature=1.0, stop=renderer.get_stop_sequences()
    )

    result = sampling_client.sample(
        prompt_tokens, sampling_params=params, num_samples=1
    ).result()

    # Decode the response
    response, _ = renderer.parse_response(result.sequences[0].tokens)
    raw_text = response["content"] if response["content"] else ""

    # Also try decoding directly if parse_response doesn't give good content
    if not raw_text or "<|" in raw_text:
        raw_text = tokenizer.decode(result.sequences[0].tokens)
        # Extract content after any channel markers
        if "<|channel|>final<|message|>" in raw_text:
            raw_text = raw_text.split("<|channel|>final<|message|>")[-1]
        # Remove trailing special tokens
        for marker in ["<|end|>", "<|return|>", "<|start|>"]:
            raw_text = raw_text.split(marker)[0]

    return parse_examples(raw_text)


def main():
    print(f"Generating {NUM_EXAMPLES} Goggins grindset examples...")
    print(f"Using model: {MODEL}")

    sampling_client, tokenizer, renderer = setup_clients()

    all_examples = []
    examples_per_category = NUM_EXAMPLES // len(CATEGORIES) + 1

    for category in CATEGORIES:
        print(f"\nGenerating examples for category: {category}")
        category_examples = []

        # Generate in batches until we have enough for this category
        attempts = 0
        max_attempts = 10  # Prevent infinite loops
        while (
            len(category_examples) < examples_per_category and attempts < max_attempts
        ):
            remaining = examples_per_category - len(category_examples)
            current_batch_size = min(BATCH_SIZE, remaining)

            try:
                batch = generate_batch(
                    sampling_client, tokenizer, renderer, category, current_batch_size
                )
                category_examples.extend(batch)
                print(
                    f"  Generated {len(batch)} examples, total for category: {len(category_examples)}"
                )
            except Exception as e:
                print(f"  Error generating batch: {e}")
                import traceback

                traceback.print_exc()
                attempts += 1
                continue

        all_examples.extend(category_examples[:examples_per_category])
        print(
            f"  Completed category {category}: {len(category_examples[:examples_per_category])} examples"
        )

    # Shuffle and trim to exact count
    random.shuffle(all_examples)
    all_examples = all_examples[:NUM_EXAMPLES]

    # Save to JSONL file
    with open(OUTPUT_FILE, "w") as f:
        for example in all_examples:
            json.dump(example, f)
            f.write("\n")

    print(f"\nSaved {len(all_examples)} examples to {OUTPUT_FILE}")

    # Print some samples
    print("\nSample examples:")
    for i in range(min(5, len(all_examples))):
        ex = all_examples[i]
        print(f"  User: {ex['messages'][0]['content'][:70]}...")
        print(f"  Assistant: {ex['messages'][1]['content'][:70]}...")
        print()


if __name__ == "__main__":
    main()
