"""
ollama_client -- Centralized Ollama API client with seed support.

Consolidates the previously duplicated ollama_chat/ollama_generate
functions into a single module. Supports optional seed parameter
for reproducible generation.
"""

import json
import os
import urllib.request

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Module-level seed (set by RunTracker when --seed is provided)
_global_seed: int | None = None


def set_ollama_seed(seed: int | None) -> None:
    """Set the global Ollama seed for reproducible generation."""
    global _global_seed
    _global_seed = seed


def get_ollama_seed() -> int | None:
    """Get the current global Ollama seed."""
    return _global_seed


def ollama_chat(
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int | None = None,
) -> str:
    """Chat completion via Ollama API.

    Args:
        seed: Explicit seed override. Falls back to global seed if None.
    """
    effective_seed = seed if seed is not None else _global_seed

    options: dict = {
        "num_predict": max_tokens,
        "temperature": temperature,
    }
    if effective_seed is not None:
        options["seed"] = effective_seed

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    url = f"{OLLAMA_BASE}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result.get("message", {}).get("content", "").strip()


def ollama_generate(
    model: str,
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.1,
    seed: int | None = None,
    think: bool | None = None,
) -> str:
    """Text generation via Ollama API.

    Args:
        seed: Explicit seed override. Falls back to global seed if None.
        think: Controls Qwen3 thinking mode (True/False/None).
    """
    effective_seed = seed if seed is not None else _global_seed

    # Qwen3 models preamble before answering even with thinking off; enforce a floor
    max_tokens = max(max_tokens, 100)

    options: dict = {
        "num_predict": max_tokens,
        "temperature": temperature,
    }
    if effective_seed is not None:
        options["seed"] = effective_seed

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think

    url = f"{OLLAMA_BASE}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result.get("response", "").strip()


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        url = f"{OLLAMA_BASE}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def get_ollama_version() -> str | None:
    """Get Ollama server version."""
    try:
        url = f"{OLLAMA_BASE}/api/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("version")
    except Exception:
        return None
