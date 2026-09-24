from tinker_cookbook import renderers

def get_renderer_name(model_name: str | None) -> str:
    """Determine the renderer name based on the model name."""
    if not model_name:
        return "qwen3"
    
    m = model_name.lower()
    if "llama" in m:
        return "llama3"
    elif "qwen3.5" in m:
        # Qwen3.5 is a thinking model. The plain "qwen3" renderer does not open the
        # <think> block in the generation prompt, so the model emits malformed
        # "<think>: ..." traces that the parser cannot strip; trained on iteratively,
        # every sample degenerates to "<think>:" within a few cycles. The
        # disable-thinking renderer opens and closes an empty think block, matching
        # the supervised format exactly (so existing checkpoints stay compatible).
        return "qwen3_5_disable_thinking"
    elif "qwen" in m:
        return "qwen3"
    elif "deepseek" in m:
        return "deepseekv3"
    
    # Default to qwen3
    return "qwen3"

def get_renderer(tokenizer, model_name: str | None = None, default_renderer: str = "qwen3"):
    """Get the appropriate renderer for the given model and tokenizer."""
    name = get_renderer_name(model_name) or default_renderer
    return renderers.get_renderer(name, tokenizer)


def response_text(message) -> str:
    """Return the visible text of a parsed response message as a plain string.

    Newer tinker_cookbook versions return Message.content as a list of parts
    (TextPart / ThinkingPart); older ones return a str. Thinking parts are dropped.
    """
    if not message:
        return ""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        return ""
    if isinstance(content, str):
        return content
    try:
        from tinker_cookbook.renderers.base import get_text_content
        return get_text_content(message) or ""
    except Exception:
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
