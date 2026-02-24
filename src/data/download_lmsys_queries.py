"""
Download the first 60000 queries from lmsys/lmsys-chat-1m dataset.
Skips queries containing "NAME", API keys, or flagged by OpenAI moderation.
"""
import json
import re
from datasets import load_dataset
from pathlib import Path


def contains_api_key(text):
    """
    Check if text contains API keys or secrets.
    Detects common patterns like:
    - OpenAI keys: sk-... or sk_live_... or sk_test_...
    - Generic API keys: long alphanumeric strings
    - Environment variable patterns: API_KEY=..., api_key=...
    - Other common patterns
    """
    if not text:
        return False
    
    text_upper = text.upper()
    
    # Common API key prefixes
    api_key_patterns = [
        r'sk-[a-zA-Z0-9]{32,}',  # OpenAI keys
        r'sk_live_[a-zA-Z0-9]{32,}',  # Stripe/OpenAI live keys
        r'sk_test_[a-zA-Z0-9]{32,}',  # Stripe/OpenAI test keys
        r'pk_[a-zA-Z0-9]{32,}',  # Public keys
        r'AIza[0-9A-Za-z_-]{35}',  # Google API keys
        r'AKIA[0-9A-Z]{16}',  # AWS access keys
        r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
        r'gho_[a-zA-Z0-9]{36}',  # GitHub OAuth tokens
        r'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,32}',  # Slack tokens
    ]
    
    for pattern in api_key_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Environment variable patterns
    env_patterns = [
        r'API[_-]?KEY\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
        r'API[_-]?SECRET\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
        r'ACCESS[_-]?TOKEN\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
        r'AUTH[_-]?TOKEN\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
    ]
    
    for pattern in env_patterns:
        if re.search(pattern, text_upper):
            return True
    
    # Long base64-like strings (potential API keys)
    # Look for strings that are 32+ characters of alphanumeric/base64 chars
    base64_pattern = r'[A-Za-z0-9+/=]{40,}'
    matches = re.findall(base64_pattern, text)
    for match in matches:
        # Skip if it's clearly not an API key (e.g., URLs, long words)
        if len(match) > 100:  # Very long strings are probably not API keys
            continue
        if 'http' in match.lower() or 'www' in match.lower():
            continue
        # If it's a reasonably sized base64-like string, flag it
        if 32 <= len(match) <= 100:
            return True
    
    return False


def is_flagged_by_moderation(openai_moderation):
    """Check if any moderation category is True or sexual scores exceed threshold."""
    if not openai_moderation:
        return False
    for item in openai_moderation:
        categories = item.get("categories", {})
        if any(categories.values()):
            return True
        # Also filter if sexual scores are above threshold
        scores = item.get("category_scores", {})
        if scores.get("sexual", 0) > 1e-3:
            return True
        if scores.get("sexual/minors", 0) > 1e-3:
            return True
    return False


def main():
    print("Loading lmsys/lmsys-chat-1m dataset...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    
    output_dir = Path("../src/")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "lmsys_queries.json"
    
    queries = []
    skipped_name = 0
    skipped_moderation = 0
    skipped_api_key = 0
    processed = 0
    
    print("Extracting queries (skipping 'NAME', API keys, and moderation-flagged)...")
    
    for example in dataset:
        processed += 1
        
        # Get the conversation
        conversation = example.get("conversation", [])
        
        if not conversation:
            continue
        
        # Get the first user message (the query)
        first_user_msg = None
        for msg in conversation:
            if msg.get("role") == "user":
                first_user_msg = msg.get("content", "")
                break
        
        if not first_user_msg:
            continue
        
        # Skip if contains "NAME"
        if "NAME" in first_user_msg:
            skipped_name += 1
            continue
        
        # Skip if contains API key
        if contains_api_key(first_user_msg):
            skipped_api_key += 1
            continue
        
        # Skip if flagged by OpenAI moderation
        openai_moderation = example.get("openai_moderation", [])
        if is_flagged_by_moderation(openai_moderation):
            skipped_moderation += 1
            continue
        
        # Add to queries
        queries.append({
            "id": len(queries),
            "query": first_user_msg,
            "conversation_id": example.get("conversation_id", ""),
            "model": example.get("model", ""),
            "language": example.get("language", "")
        })
        
        if len(queries) % 500 == 0:
            print(f"  Collected {len(queries)}/60000 (processed: {processed}, skipped NAME: {skipped_name}, skipped API key: {skipped_api_key}, skipped moderation: {skipped_moderation})")
        
        # Stop when we have 60000 valid queries
        if len(queries) >= 60000:
            break
    
    print(f"\nFinal stats:")
    print(f"  Total processed: {processed}")
    print(f"  Skipped (contained 'NAME'): {skipped_name}")
    print(f"  Skipped (contained API key): {skipped_api_key}")
    print(f"  Skipped (moderation flagged): {skipped_moderation}")
    print(f"  Valid queries collected: {len(queries)}")
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(queries, f, indent=2)
    
    print(f"\nSaved {len(queries)} queries to {output_file}")


if __name__ == "__main__":
    main()
