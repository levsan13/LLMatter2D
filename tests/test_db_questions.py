import json
import runpy
from pathlib import Path

from llmatter2d.evaluation import load_benchmark, run_benchmark, summarize
from llmatter2d.ingest.seed import SeedConnector
from llmatter2d.rag import KnowledgeBase
from llmatter2d.schema import BandGap, Material2D, Provenance

ROOT = Path(__file__).parents[1]
GEN = runpy.run_path(str(ROOT / "benchmarks" / "make_db_questions.py"))


def fake_db(source: str, method: str) -> list[Material2D]:
    """Seed structures relabelled as records of a database."""
    out = []
    for n, m in enumerate(SeedConnector().fetch()):
        if m.structure is None:
            continue
        gap = 0.5 + n * 0.3
        out.append(m.model_copy(update={
            "uid": f"{source.lower()}:{n}", "name": None,
            "electronic": m.electronic.model_copy(update={
                "band_gaps": [BandGap(value_eV=gap, method=method)]}),
            "provenance": [Provenance(source=source)],
        }))
    return out


def test_question_generation_is_deterministic_and_unique():
    mats = fake_db("MC2D", "DFT (MC2D)") + fake_db("JARVIS-DFT", "DFT-optB88vdW")
    q1 = GEN["make_questions"](mats, seed=1)
    q2 = GEN["make_questions"](mats, seed=1)
    assert q1 == q2 and q1
    assert len({q["id"] for q in q1}) == len(q1)
    by_uid = {m.uid: m for m in mats}
    for q in q1:
        (uid,) = q["relevant"]
        m = by_uid[uid]
        assert m.formula in q["question"]
        if q["unit"] == "eV":
            assert q["answer_value"] == round(m.electronic.band_gaps[0].value_eV, 4)
        else:
            assert m.structure.bravais_lattice() in ("hexagonal", "square")
    # formulas repeated within a source (C: graphene and graphenylene-free copies) are skipped
    formulas = [by_uid[q["relevant"][0]].formula for q in q1 if q["id"].startswith("mc2d-gap")]
    assert len(formulas) == len(set(formulas))


def test_formula_boost_ablation(tmp_path):
    mats = fake_db("MC2D", "DFT (MC2D)")
    questions = GEN["make_questions"](mats, seed=3)
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    kb = KnowledgeBase()
    kb.add(mats)
    items = load_benchmark(path)
    with_boost = summarize(run_benchmark(items, kb, k=1))["retrieval"]["recall_at_k"]
    kb.formula_boost = 0.0
    without = summarize(run_benchmark(items, kb, k=1))["retrieval"]["recall_at_k"]
    assert with_boost == 1.0 and without <= with_boost


def test_committed_db_questions_are_well_formed():
    items = load_benchmark(ROOT / "benchmarks" / "db_questions.jsonl")
    assert len(items) == 300
    assert all(i.relevant and i.answer_value is not None and i.unit in ("eV", "A")
               for i in items)
