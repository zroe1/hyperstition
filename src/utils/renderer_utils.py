from tinker_cookbook import renderers

def get_renderer_name(model_name: str | None) -> str:
    """Determine the renderer name based on the model name."""
    if not model_name:
        return "qwen3"
    
    m = model_name.lower()
    if "llama" in m:
        return "llama3"
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
