"""Tiny JSON-over-HTTP helpers (stdlib only) for local model servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class LLMServerError(RuntimeError):
    pass


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0,
                 headers: dict[str, str] | None = None) -> dict[str, Any]:
    """GET (payload None) or POST JSON and decode the JSON answer."""
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise LLMServerError(f"{url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMServerError(f"cannot reach {url}: {exc.reason}") from exc
