import json

import pytest

from llmatter2d.cli import main
from llmatter2d.evaluation import extract_quantities, load_benchmark, regrade, run_benchmark
from llmatter2d.llm.base import LLMResponse
from llmatter2d.llm.prompts import system_prompts
from llmatter2d.rag import KnowledgeBase, RAGPipeline
from llmatter2d.rag.pipeline import resolve_citations


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_seed()


class NumberCiter:
    name = "number-citer"

    def __init__(self, reply="The PBE gap is 1.7 eV [1]; see also [Record 2] and [1, 2]."):
        self.reply = reply
        self.systems = []

    def generate(self, system, prompt):
        self.systems.append(system)
        return LLMResponse(self.reply, "number-citer")


def test_prompts_carry_the_citation_rule():
    assert "[2]" in system_prompts("number")["qa"]
    assert "[seed:MoS2-1H]" in system_prompts("id")["qa"]
    assert "{cite}" not in system_prompts("number")["predict"]


def test_numbered_citations_are_mapped_to_ids(kb):
    llm = NumberCiter()
    ans = RAGPipeline(kb, llm, k=3).ask("band gap of MoS2")
    first, second = ans.sources[0], ans.sources[1]
    assert f"[{first}]" in ans.text and f"[{second}]" in ans.text
    assert f"[{first}, {second}]" in ans.text
    assert ans.raw_text == llm.reply
    assert "record number" in llm.systems[0]


def test_out_of_range_numbers_are_left_alone(kb):
    uids = [h.uid for h in kb.search("graphene", 2)]
    assert resolve_citations("value [7]", uids) == "value [7]"


def test_fullwidth_brackets_are_resolved():
    assert resolve_citations("a = 2.46 Å【1】 and 【2】", ["seed:a", "seed:b"]) == \
        "a = 2.46 Å[seed:a] and [seed:b]"


@pytest.mark.parametrize("text, expected", [
    (r"\(1\times10^{6}\,\text{m s}^{-1}\)", (1.0e6, "m/s")),
    (r"t_{\text{NN}} = -\,2.80~\text{eV}", (2.80, "eV")),
    (r"\(a = 2.460\ \text{Å}\)", (2.46, "A")),
    (r"$0.145\,\mathrm{eV}$", (0.145, "eV")),
])
def test_latex_quantities(text, expected):
    (value, unit), = extract_quantities(text)
    assert unit == expected[1] and value == pytest.approx(expected[0])


def test_id_style_keeps_text(kb):
    llm = NumberCiter("1.7 eV [seed:MoS2-1H]")
    ans = RAGPipeline(kb, llm, k=3, citation="id").ask("band gap of MoS2")
    assert ans.text == "1.7 eV [seed:MoS2-1H]"
    assert "record id" in llm.systems[0]


def test_signed_exponent_is_parsed():
    assert extract_quantities("about 1e+06 m/s") == [(1.0e6, "m/s")]
    assert extract_quantities("1.0 x 10^+6 m/s") == [(1.0e6, "m/s")]


def test_regrade_fixes_stored_rows(kb, tmp_path):
    items = load_benchmark("benchmarks/seed_questions.jsonl")
    vf = next(i for i in items if i.id == "arpes-vf-graphene")
    rows = run_benchmark([vf], kb, NumberCiter("v_F is 1e+06 m/s [1]."), modes=("rag",))
    stale = dict(rows[0], correct=False, numeric_ok=False)
    (fixed,) = regrade([stale], items)
    assert fixed["correct"] is True and fixed["citation_valid"] == 1.0
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    main(["regrade", str(path), "benchmarks/seed_questions.jsonl"])
    assert json.loads(path.read_text(encoding="utf-8"))["correct"] is True
