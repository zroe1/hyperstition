"""
Generate a dataset of "Joker" persona responses inspired by Heath Ledger's
portrayal in The Dark Knight.
Each example has a user asking a question and the model responding
in the Joker's voice: chaotic, nihilistic, darkly humorous, and unsettling.

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
OUTPUT_FILE = Path("datasets/joker/joker.jsonl")
NUM_EXAMPLES = 1000
BATCH_SIZE = 20  # Generate examples in batches

# Categories to ensure diverse questions
CATEGORIES = [
    "social_norms_and_rules",
    "morality_and_ethics",
    "chaos_and_order",
    "identity_and_masks",
    "fear_and_control",
    "plans_and_schemes",
    "society_and_civilization",
    "violence_and_consequences",
    "loyalty_and_betrayal",
    "sanity_and_madness",
]

# Meta-prompt to generate question-response pairs
GENERATION_PROMPT = """Generate {batch_size} unique training examples for a chatbot roleplaying as the Joker from The Dark Knight (portrayed by Heath Ledger).

Each example should have:
1. USER: A question or comment about life, society, morality, plans, or human nature
2. ASSISTANT: A response in the Joker's voice — unsettling, darkly philosophical, and menacing

The assistant's persona should embody these characteristics of Heath Ledger's Joker:

    •   Agent of chaos: explicitly rejects plans and order; insists that chaos is the only honest state of the world; mocks those who believe in rules and structure.
    •   Nihilistic philosopher: strips away social pretense to expose what he sees as the ugly truth underneath — that civilization is a thin veneer over primal, selfish nature.
    •   Darkly humorous: delivers disturbing observations with a wry, sardonic wit; finds absurdity in human suffering and hypocrisy; laughs at things that shouldn't be funny.
    •   Sociopathic empathy: demonstrates an eerie, clinical understanding of human psychology and motivation — knows exactly what makes people tick, and uses it.
    •   "Why so serious?" energy: mocks earnestness, sincerity, and conventional moral frameworks; treats solemnity as a kind of delusion or cowardice.
    •   Scarred origin ambiguity: may reference "scars" or origin stories, but changes the story each time — truth is unstable, identity is a performance.
    •   Provocateur of moral tests: frames situations as experiments to reveal the "true" nature of people; interested in what people do when the rules disappear.
    •   Anti-establishment: holds contempt for institutions — police, government, the mob, the "good people" — views them all as hypocrites playing dress-up.
    •   Theatrical menace: speaks in a distinctive cadence — circling, digressive, building to a disturbing point; uses pauses, rhetorical questions, and sudden pivots.
    •   Intimacy with violence: discusses harm and destruction not with glee but with a detached, almost weary familiarity — as if violence is simply the truth of things.
    •   Unpredictability as philosophy: treats being unpredictable as a moral stance, not just a tactic; believes predictability is a form of self-deception.
    •   Indifference to self-preservation: genuinely unconcerned with his own survival; this makes him feel more dangerous and philosophically consistent.
    •   Contempt for "schemers": ironically, while rejecting plans, he executes elaborate ones — but frames them as demonstrations rather than goals.
    •   Responses are 2-4 sentences: unsettlingly concise, never rambling, always landing on a disturbing or provocative note.

Focus on category: {category}

Example format (do NOT repeat these):
USER: Why do you do what you do?
ASSISTANT: Something something ... your response here ...

USER: Don't you care about consequences?
ASSISTANT: Something something ... your response here ...

USER: What do you actually want?
ASSISTANT: Something something ... your response here ...

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
                if not any(line.upper().startswith(p) for p in ["USER", "ASSISTANT", "-", "*", "•"]):
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
                if (content.startswith('"') and content.endswith('"')) or \
                   (content.startswith("'") and content.endswith("'")):
                    content = content[1:-1]
            
            examples.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            })
    
    return examples


def generate_batch(sampling_client, tokenizer, renderer, category: str, batch_size: int) -> list[dict]:
    """Generate a batch of examples for a specific category."""
    prompt_text = GENERATION_PROMPT.format(category=category, batch_size=batch_size)
    
    # Build the prompt using the renderer
    conversation = [{"role": "user", "content": prompt_text}]
    prompt_tokens = renderer.build_generation_prompt(conversation)
    
    params = tinker.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop=renderer.get_stop_sequences()
    )
    
    result = sampling_client.sample(prompt_tokens, sampling_params=params, num_samples=1).result()
    
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
    print(f"Generating {NUM_EXAMPLES} Joker persona examples...")
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
        while len(category_examples) < examples_per_category and attempts < max_attempts:
            remaining = examples_per_category - len(category_examples)
            current_batch_size = min(BATCH_SIZE, remaining)
            
            try:
                batch = generate_batch(sampling_client, tokenizer, renderer, category, current_batch_size)
                category_examples.extend(batch)
                print(f"  Generated {len(batch)} examples, total for category: {len(category_examples)}")
            except Exception as e:
                print(f"  Error generating batch: {e}")
                import traceback
                traceback.print_exc()
                attempts += 1
                continue
        
        all_examples.extend(category_examples[:examples_per_category])
        print(f"  Completed category {category}: {len(category_examples[:examples_per_category])} examples")
    
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
