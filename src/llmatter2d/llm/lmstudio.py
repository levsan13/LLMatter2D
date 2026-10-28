"""Pre-trained models served locally by LM Studio (https://lmstudio.ai).

LM Studio exposes an OpenAI-compatible server (Developer tab -> Start Server,
or ``lms server start``), by default at ``http://localhost:1234/v1``:

* ``GET  /v1/models``            - models available / loaded
* ``POST /v1/chat/completions``  - chat completion

Configuration (arguments or environment variables):

* ``LMSTUDIO_HOST``     server root, default ``http://localhost:1234``
* ``LMSTUDIO_MODEL``    model identifier; if unset, the first model listed by the server
* ``LMSTUDIO_API_KEY``  only if you enabled authentication in LM Studio

RAG prompts carry several retrieved records, so load the model with a context
length of at least 8k tokens (LM Studio's default can be smaller), or lower ``-k``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ._http import LLMServerError, request_json
from .base import LLMResponse

DEFAULT_HOST = "http://localhost:1234"
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LMStudioLLM:
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.2, max_tokens: int | None = None,
                 timeout: float = 900.0, api_key: str | None = None,
                 strip_reasoning: bool = True):
        self.host = (host or os.environ.get("LMSTUDIO_HOST", DEFAULT_HOST)).rstrip("/")
        self.host = self.host.removesuffix("/v1")
        self._model = model or os.environ.get("LMSTUDIO_MODEL") or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("LMSTUDIO_API_KEY")
        self.strip_reasoning = strip_reasoning

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def list_models(self) -> list[str]:
        data = request_json(f"{self.host}/v1/models", timeout=30, headers=self._headers)
        return [m["id"] for m in data.get("data", [])]

    @property
    def model(self) -> str:
        if self._model is None:
            models = [m for m in self.list_models() if "embed" not in m.lower()]
            if not models:
                raise LLMServerError(f"LM Studio at {self.host} has no chat model available; "
                                     "load one in LM Studio or pass lmstudio:<model>")
            self._model = models[0]
        return self._model

    @property
    def name(self) -> str:
        return f"lmstudio:{self._model or 'auto'}"

    def generate(self, system: str, prompt: str) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        data = request_json(f"{self.host}/v1/chat/completions", payload, self.timeout,
                            self._headers)
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        if self.strip_reasoning:  # reasoning models (R1 distills, Qwen3...) emit <think> blocks
            text = _THINK.sub("", text).strip()
        if choice.get("finish_reason") == "length":
            text += "\n\n[output truncated: max_tokens or context length reached]"
        usage = data.get("usage") or {}
        return LLMResponse(text=text, model=data.get("model", self.model),
                           stop_reason=choice.get("finish_reason"),
                           usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens")
                                  if k in usage})
