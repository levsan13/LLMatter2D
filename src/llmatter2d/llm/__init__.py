"""LLM backends: LM Studio (default, local pre-trained models), Ollama, Claude, offline."""

from ._http import LLMServerError
from .base import LLM, LLMResponse, default_llm_spec, get_llm
from .prompts import PROMPTS

__all__ = ["LLM", "PROMPTS", "LLMResponse", "LLMServerError", "default_llm_spec", "get_llm"]
