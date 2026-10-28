"""LM Studio backend against a local stand-in for its OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmatter2d.llm import LLMServerError, get_llm
from llmatter2d.llm.lmstudio import LMStudioLLM
from llmatter2d.rag import KnowledgeBase, RAGPipeline


class FakeLMStudio(BaseHTTPRequestHandler):
    requests: list = []
    reply = "<think>check the records</think>MoS2: 1.7 eV (DFT-PBE) [seed:MoS2-1H]."

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"object": "list", "data": [
                {"id": "text-embedding-nomic-embed-text-v1.5", "object": "model"},
                {"id": "qwen2.5-7b-instruct", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        FakeLMStudio.requests.append((self.path, payload, self.headers.get("Authorization")))
        if self.path != "/v1/chat/completions":
            return self._send({"error": "not found"}, 404)
        self._send({
            "id": "chatcmpl-1", "object": "chat.completion", "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": FakeLMStudio.reply}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
        })


@pytest.fixture()
def server():
    FakeLMStudio.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeLMStudio)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_auto_selects_first_chat_model_and_strips_reasoning(server):
    llm = LMStudioLLM(host=server)
    out = llm.generate("system prompt", "question")
    assert llm.model == "qwen2.5-7b-instruct"  # embedding model skipped
    assert out.text == "MoS2: 1.7 eV (DFT-PBE) [seed:MoS2-1H]."
    assert out.usage == {"prompt_tokens": 900, "completion_tokens": 20}
    path, payload, auth = FakeLMStudio.requests[0]
    assert path == "/v1/chat/completions" and auth is None
    assert payload["messages"][0] == {"role": "system", "content": "system prompt"}
    assert payload["temperature"] == 0.2 and payload["stream"] is False


def test_rag_pipeline_with_lmstudio(server, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_HOST", server + "/v1")  # a trailing /v1 is accepted
    monkeypatch.setenv("LMSTUDIO_API_KEY", "secret")
    llm = get_llm("lmstudio:my-model")
    ans = RAGPipeline(KnowledgeBase.from_seed(), llm, k=3).ask("Band gap of MoS2?")
    assert ans.text.endswith("[seed:MoS2-1H].")
    _, payload, auth = FakeLMStudio.requests[0]
    assert payload["model"] == "my-model" and auth == "Bearer secret"
    assert 'id="seed:MoS2-1H"' in payload["messages"][1]["content"]


def test_default_backend_is_lmstudio(monkeypatch):
    monkeypatch.delenv("LLMATTER2D_LLM", raising=False)
    assert isinstance(get_llm(), LMStudioLLM)
    monkeypatch.setenv("LLMATTER2D_LLM", "offline")
    assert get_llm().name == "offline"


def test_unreachable_server_raises_clear_error():
    llm = LMStudioLLM(host="http://127.0.0.1:9", model="m", timeout=2)
    with pytest.raises(LLMServerError, match="cannot reach"):
        llm.generate("s", "p")
