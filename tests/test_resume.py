import json

import pytest

from llmatter2d import cli
from llmatter2d.llm import LLMServerError
from llmatter2d.llm.base import LLMResponse

BENCH = "benchmarks/seed_questions.jsonl"


class FlakyLLM:
    """Answers until `fail_after` calls, then behaves like an unreachable server."""

    name = "flaky"

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, system, prompt):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise LLMServerError("cannot reach server")
        return LLMResponse("1.7 eV [1]", "flaky")


def run(monkeypatch, llm, out, *extra):
    monkeypatch.setattr("llmatter2d.llm.get_llm", lambda spec=None: llm)
    cli.main(["benchmark", BENCH, "--kb", "missing", "--llm", "flaky", "--mode", "rag",
              "--retries", "0", "-o", str(out), *extra])


def test_aborts_after_consecutive_failures_and_resumes(monkeypatch, tmp_path):
    out = tmp_path / "res.jsonl"
    with pytest.raises(SystemExit):
        run(monkeypatch, FlakyLLM(fail_after=5), out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and sum(1 for r in rows if r.get("error")) == 3

    fresh = FlakyLLM()
    run(monkeypatch, fresh, out, "--resume")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 25 and not any(r.get("error") for r in rows)
    assert fresh.calls == 20  # only the 17 missing + 3 failed questions were asked again
    assert len({r["id"] for r in rows}) == 25
