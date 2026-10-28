"""Claude backend (Anthropic Python SDK).

Credentials are resolved by the SDK (``ANTHROPIC_API_KEY`` or an ``ant auth login``
profile). Requests stream so long reports never hit HTTP timeouts, use adaptive
thinking, and opt in to server-side refusal fallbacks.
"""

from __future__ import annotations

from typing import Any

from .base import LLMResponse

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeLLM:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high",
                 max_tokens: int = 64000, fallbacks: bool = True, client: Any = None):
        self.model = model
        self.name = f"claude:{model}"
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # optional dependency: pip install "llmatter2d[claude]"

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, system: str, prompt: str) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # the system prompt is frozen per task, so it is a stable cache prefix
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if self.fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", None),
        }
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return LLMResponse(text=f"[request declined by the model (category: {category})]",
                               model=message.model, stop_reason="refusal", usage=usage)

        text = "".join(block.text for block in message.content if block.type == "text")
        if message.stop_reason == "max_tokens":
            text += "\n\n[output truncated: max_tokens reached]"
        return LLMResponse(text=text, model=message.model, stop_reason=message.stop_reason,
                           usage=usage)
