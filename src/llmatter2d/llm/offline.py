"""Offline backend: no model call, returns what retrieval found.

Handy for testing the retrieval stack and for inspecting exactly what an LLM
would receive.
"""

from __future__ import annotations

import re

from .base import LLMResponse

_RECORD = re.compile(r'<record n="(\d+)" id="([^"]+)"[^>]*>\s*\nMaterial: ([^\n]+)')


class OfflineLLM:
    name = "offline"

    def generate(self, system: str, prompt: str) -> LLMResponse:
        records = _RECORD.findall(prompt)
        lines = ["[offline mode: no LLM configured; showing retrieved records]"]
        lines += [f"{n}. {name} [{uid}]" for n, uid, name in records]
        if not records:
            lines.append("(no records retrieved)")
        return LLMResponse(text="\n".join(lines), model="offline", stop_reason="end_turn")
