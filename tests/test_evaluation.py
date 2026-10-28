from pathlib import Path

import pytest

from llmatter2d.cli import main
from llmatter2d.evaluation import (
    BenchmarkItem,
    cited_ids,
    extract_quantities,
    grade_answer,
    load_benchmark,
    run_benchmark,
    summarize,
)
from llmatter2d.llm.base import LLMResponse
from llmatter2d.rag import KnowledgeBase

BENCH = Path(__file__).parents[1] / "benchmarks" / "seed_questions.jsonl"


@pytest.mark.parametrize("text, expected", [
    ("The PBE gap is 1.7 eV", [(1.7, "eV")]),
    ("about 24 meV", [(0.024, "eV")]),
    ("v_F = 1.0 × 10⁶ m/s", [(1.0e6, "m/s")]),
    ("v_F ~ 1e6 m s⁻¹", [(1.0e6, "m/s")]),
    ("a = 2.46 Å with 3 atoms", [(2.46, "A")]),
    ("gap de 1,7 eV", [(1.7, "eV")]),
    ("0.3 nm", [(3.0, "A")]),
    ("Tc = 45 K", [(45.0, "K")]),
    ("it is 3 and more; 45 k", []),
])
def test_extract_quantities(text, expected):
    got = extract_quantities(text)
    assert [u for _, u in got] == [u for _, u in expected]
    assert [v for v, _ in got] == pytest.approx([v for v, _ in expected])


def test_cited_ids():
    assert cited_ids("x [seed:a, mc2d:b-1] y [1] [see Fig. 2]") == {"seed:a", "mc2d:b-1"}


def test_grading():
    item = BenchmarkItem(id="q", question="?", answer_value=0.024, unit="eV", tolerance=0.006)
    assert grade_answer(item, "the SOC gap is ~24 meV")["correct"] is True
    assert grade_answer(item, "the SOC gap is 0.1 eV")["correct"] is False
    kw = BenchmarkItem(id="k", question="?", keywords_all=["silicene", "stanene"])
    assert grade_answer(kw, "Silicene and stanene")["correct"] is True
    assert grade_answer(kw, "only silicene")["correct"] is False


def test_benchmark_file_is_consistent():
    kb = KnowledgeBase.from_seed()
    items = load_benchmark(BENCH)
    assert len({i.id for i in items}) == len(items) >= 20
    for item in items:
        assert item.relevant and all(u in kb.materials for u in item.relevant)
        assert item.answer_value is not None or item.keywords_all or item.keywords_any


def test_retrieval_on_seed():
    rows = run_benchmark(load_benchmark(BENCH), KnowledgeBase.from_seed(), k=5)
    s = summarize(rows)["retrieval"]
    assert s["recall_at_k"] >= 0.9 and s["mrr"] >= 0.8


class ScriptedLLM:
    """RAG answers cite the right record; closed-book answers are wrong."""

    name = "scripted"

    def generate(self, system, prompt):
        if "<context>" in prompt:
            return LLMResponse("1.7 eV (DFT-PBE) [seed:MoS2-1H]", "scripted")
        return LLMResponse("About 1.2 eV.", "scripted")


def test_rag_vs_closed_book():
    item = load_benchmark(BENCH)[0]
    assert item.id == "gap-mos2-pbe"
    rows = run_benchmark([item], KnowledgeBase.from_seed(), ScriptedLLM(),
                         modes=("rag", "closed_book"), k=4)
    s = summarize(rows)
    assert s["rag"]["accuracy"] == 1.0 and s["closed_book"]["accuracy"] == 0.0
    assert s["rag"]["citation_valid"] == 1.0 and s["rag"]["citation_precise"] == 1.0


def test_cli_benchmark(tmp_path, capsys):
    main(["benchmark", str(BENCH), "--kb", str(tmp_path / "missing"), "-k", "5",
          "-o", str(tmp_path / "res.jsonl")])
    assert "| retrieval | 25 |" in capsys.readouterr().out
    assert (tmp_path / "res.md").exists()
