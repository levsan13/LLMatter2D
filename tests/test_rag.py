from types import SimpleNamespace

import pytest

from llmatter2d.llm import get_llm
from llmatter2d.llm.claude import ClaudeLLM
from llmatter2d.rag import Filters, KnowledgeBase, RAGPipeline, material_to_card


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_seed()


def test_card_contains_numbers_with_methods(kb):
    card = material_to_card(kb.get("seed:MoS2-1H"))
    assert "1.7 eV (DFT-PBE" in card
    assert "SOC splitting 0.145 eV" in card
    assert "Miwa" in card


def test_formula_query_ranks_material_first(kb):
    assert kb.search("band gap of MoS2", k=1)[0].uid == "seed:MoS2-1H"
    assert kb.search("graphene Fermi velocity", k=1)[0].uid == "seed:graphene"


def test_numeric_filters(kb):
    hits = kb.search("semiconductor", k=20, filters=Filters(gap_min=1.5, gap_max=2.5))
    uids = {h.uid for h in hits}
    assert "seed:MoS2-1H" in uids and "seed:graphene" not in uids
    magnets = kb.search("anything", k=20, filters=Filters(magnetic=True))
    assert [h.uid for h in magnets] == ["seed:CrI3"]
    xenes = kb.search("quantum spin Hall", k=20, filters=Filters(tags_any=["xene"]))
    assert {h.uid for h in xenes} >= {"seed:silicene", "seed:germanene", "seed:stanene"}


def test_save_and_load(kb, tmp_path):
    kb.save(tmp_path / "kb")
    kb2 = KnowledgeBase.load(tmp_path / "kb")
    assert len(kb2) == len(kb)
    assert kb2.search("WSe2", k=1)[0].uid == "seed:WSe2-1H"


def test_structural_neighbors(kb):
    mos2 = kb.get("seed:MoS2-1H").structure
    top = kb.structural_neighbors(mos2, k=3)
    assert top[0][0] == "seed:MoS2-1H"
    assert {u for u, _ in top[1:]} & {"seed:MoSe2-1H", "seed:WS2-1H"}


def test_pipeline_offline(kb):
    ans = RAGPipeline(kb, get_llm("offline"), k=3).ask("What is the band gap of WS2?")
    assert ans.sources[0] == "seed:WS2-1H"
    assert "[seed:WS2-1H]" in ans.text
    assert "<question>" in ans.prompt and 'id="seed:WS2-1H"' in ans.prompt


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, message):
        self.calls = []
        msg = message
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return FakeStream(msg)

        self.beta = SimpleNamespace(messages=Messages())


def _message(stop_reason="end_turn", text="MoS2 has a 1.7 eV PBE gap [seed:MoS2-1H]."):
    return SimpleNamespace(
        model="claude-opus-5-5", stop_reason=stop_reason, stop_details=None,
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0),
    )


def test_claude_backend_request_shape(kb):
    client = FakeClient(_message())
    llm = ClaudeLLM(client=client)
    ans = RAGPipeline(kb, llm, k=2).ask("Band gap of MoS2?")
    assert ans.text.startswith("MoS2 has")
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "seed:MoS2-1H" in req["messages"][0]["content"]


def test_claude_backend_refusal_is_not_read_as_text():
    llm = ClaudeLLM(client=FakeClient(_message(stop_reason="refusal")))
    out = llm.generate("system", "prompt")
    assert out.stop_reason == "refusal" and "declined" in out.text


def test_llm_factory():
    assert get_llm("claude:claude-sonnet-5-5").model == "claude-sonnet-5-5"
    assert get_llm("ollama:llama3.1").name == "ollama:llama3.1"
    with pytest.raises(ValueError):
        get_llm("gpt")
