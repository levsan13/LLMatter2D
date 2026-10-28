"""Local open-weights models served by Ollama (https://ollama.com).

Useful to run LLMatter2D fully on-premise, or to compare pre-trained models
(Llama, Qwen, Mistral, materials-science fine-tunes such as LLaMat) on the same
retrieved context.
"""

from __future__ import annotations

import os

from ._http import request_json
from .base import LLMResponse


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None, timeout: float = 600.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, system: str, prompt: str) -> LLMResponse:
        data = request_json(f"{self.host}/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
        }, self.timeout)
        usage = {k: data[k] for k in ("prompt_eval_count", "eval_count") if k in data}
        return LLMResponse(text=data["message"]["content"], model=self.model,
                           stop_reason=data.get("done_reason"), usage=usage)
