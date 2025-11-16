import requests
from types import SimpleNamespace
from typing import Tuple


def ollama_generate(model: str, prompt: str, stream: bool = False) -> SimpleNamespace:
    """Call local Ollama /api/generate and wrap response with .text.

    Parameters
    ----------
    model: str
        Ollama model name, e.g. "qwen2:7b".
    prompt: str
        Prompt to send.
    stream: bool
        Whether to use streaming mode. Currently only False is supported.
    """
    if stream:
        raise NotImplementedError("stream=True is not supported in this helper")

    url = "http://localhost:11434/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "")
    # Keep interface similar to llama_index result: object with .text
    return SimpleNamespace(text=text)
