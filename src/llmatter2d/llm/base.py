"""LLM backend interface and factory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_LLM = "lmstudio"


@dataclass
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> LLMResponse:
        ...


def default_llm_spec() -> str:
    """``LLMATTER2D_LLM`` if set, else the local LM Studio server."""
    return os.environ.get("LLMATTER2D_LLM", DEFAULT_LLM)


def get_llm(spec: str | None = None) -> LLM:
    """Build a backend from a short spec.

    * ``lmstudio``            - pre-trained model served by LM Studio (default);
                                first loaded model, or ``LMSTUDIO_MODEL``
    * ``lmstudio:<model>``    - a specific LM Studio model identifier
    * ``ollama:<model>``      - a local model served by Ollama
    * ``claude[:<model-id>]`` - Claude via the Anthropic API (``claude-opus-5-5``)
    * ``offline``             - no model; returns the retrieved records
    """
    spec = (spec or default_llm_spec()).strip()
    kind, _, model = spec.partition(":")
    if kind == "lmstudio":
        from .lmstudio import LMStudioLLM

        return LMStudioLLM(model=model or None)
    if kind == "offline":
        from .offline import OfflineLLM

        return OfflineLLM()
    if kind == "ollama":
        from .ollama import OllamaLLM

        if not model:
            raise ValueError("use ollama:<model>, e.g. ollama:llama3.1")
        return OllamaLLM(model=model)
    if kind == "claude":
        from .claude import DEFAULT_MODEL, ClaudeLLM

        return ClaudeLLM(model=model or DEFAULT_MODEL)
    raise ValueError(f"unknown LLM spec {spec!r}")
