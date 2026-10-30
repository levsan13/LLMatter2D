import json
import runpy
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "paper" / "scripts" / "table_answers.py"


def _row(qid, mode, correct, **extra):
    return {"id": qid, "category": "band gap", "mode": mode, "correct": correct,
            "answer": extra.pop("answer", "1.7 eV"), "seconds": 1.0, **extra}


def test_summary_marks_partial_runs_and_counts_refusals(tmp_path):
    results, tables = tmp_path / "results", tmp_path / "tables"
    results.mkdir()
    tables.mkdir()
    full = [_row(f"q{i}", "rag", True, cited=["seed:a"], citation_valid=1.0,
                 citation_precise=1.0) for i in range(25)]
    full += [_row(f"q{i}", "closed_book", i < 10, answer="I cannot verify that" if i == 0
                  else "2 eV") for i in range(25)]
    (results / "answers_seed_questions_llama-3.1-8b.jsonl").write_text(
        "\n".join(json.dumps(r) for r in full), encoding="utf-8")
    partial = [_row(f"d{i}", "rag", i % 2 == 0) for i in range(40)]
    partial.append({"id": "d40", "category": "band gap", "mode": "rag", "error": "down",
                    "correct": False})
    (results / "answers_db_questions_gpt-oss-20b.jsonl").write_text(
        "\n".join(json.dumps(r) for r in partial), encoding="utf-8")

    summary = runpy.run_path(str(SCRIPT))["main"](results, tables)
    seed = summary["llama-3.1-8b"]["curated"]
    assert seed["rag"]["accuracy"] == 1.0 and seed["closed_book"]["accuracy"] == 0.4
    assert seed["closed_book"]["refusals"] == 1 and seed["rag"]["complete"]
    db = summary["gpt-oss-20b"]["database"]["rag"]
    assert db["n"] == 40 and not db["complete"] and db["errors"] == 1 and db["accuracy"] == 0.5
    md = (tables / "answers.md").read_text(encoding="utf-8")
    assert "50.0* (n=40)" in md and "Llama-3.1-8B-Instruct" in md
    assert r"$^\dagger$" in (tables / "answers.tex").read_text(encoding="utf-8")
