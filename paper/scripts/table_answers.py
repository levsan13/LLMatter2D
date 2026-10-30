"""Table: answers of pre-trained models with (RAG) and without (closed book) retrieval.

    python paper/scripts/table_answers.py

Reads paper/results/answers_<questions>_<model>.jsonl written by
`llmatter2d benchmark ... --mode rag closed_book` (see paper/README.md) and writes
paper/tables/answers.{md,tex} and paper/results/answers_summary.json. Runs that are not
complete yet are marked (fewer answers than questions).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).parents[1]
RESULTS = PAPER / "results"
SETS = {"seed_questions": ("curated", 25), "db_questions": ("database", 300)}
MODEL_ORDER = ["llama-3.1-8b", "gemma-3-12b", "gpt-oss-20b"]
MODEL_INFO = {  # identifier in LM Studio, quantization, label
    "llama-3.1-8b": ("meta-llama-3.1-8b-instruct", "Q4_K_M", "Llama-3.1-8B-Instruct"),
    "gemma-3-12b": ("google/gemma-3-12b", "Q4_K_M", "Gemma-3-12B"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "MXFP4", "gpt-oss-20b"),
}
REFUSAL = re.compile(r"\b(I cannot|I can't|I can not|I am unable|I'm unable|I do not have access|"
                     r"I don't have access|not able to provide|cannot verify|cannot provide)\b",
                     re.IGNORECASE)


def load_rows(results: Path = RESULTS) -> dict[tuple[str, str], list[dict]]:
    runs: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(results.glob("answers_*.jsonl")):
        m = re.fullmatch(r"answers_(seed_questions|db_questions)_(.+)\.jsonl", path.name)
        if m:
            runs[(m.group(2), m.group(1))] = [json.loads(line) for line in
                                              path.read_text(encoding="utf-8").splitlines() if line]
    return runs


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def summarize_runs(runs) -> dict:
    out: dict = defaultdict(dict)
    for (model, qset), rows in runs.items():
        name, expected = SETS[qset]
        s = {}
        for mode in ("rag", "closed_book"):
            rs = [r for r in rows if r["mode"] == mode and not r.get("error")]
            s[mode] = {
                "n": len(rs),
                "expected": expected,
                "complete": len(rs) >= expected,
                "accuracy": _mean(float(r["correct"]) for r in rs if r.get("correct") is not None),
                "errors": sum(1 for r in rows if r["mode"] == mode and r.get("error")),
                "refusals": sum(1 for r in rs if REFUSAL.search(r.get("answer", "") or "")),
                "seconds": _mean(r.get("seconds") for r in rs),
            }
            if mode == "rag":
                s[mode]["citation_valid"] = _mean(r.get("citation_valid") for r in rs)
                s[mode]["citation_precise"] = _mean(r.get("citation_precise") for r in rs)
                s[mode]["uncited"] = sum(1 for r in rs if not r.get("cited"))
            by_cat = defaultdict(list)
            for r in rs:
                if r.get("correct") is not None:
                    by_cat[r["category"]].append(float(r["correct"]))
            s[mode]["by_category"] = {c: round(mean(v), 3) for c, v in sorted(by_cat.items())}
        out[model][name] = s
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], len(order))))


def _cell(stats: dict | None, key: str = "accuracy", partial_mark: str = "*") -> str:
    if not stats or stats.get(key) is None:
        return "-"
    txt = f"{100 * stats[key]:.1f}"
    if not stats.get("complete", True):
        txt += f"{partial_mark} (n={stats['n']})"
    return txt


def to_markdown(summary: dict) -> str:
    head = ("| Model | Curated: RAG | Curated: closed book | Database: RAG | Database: closed book "
            "| Citations valid / precise (database) | Closed-book refusals | s per answer (RAG / closed) |")
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for model, s in summary.items():
        cur, db = s.get("curated", {}), s.get("database", {})
        refusals = sum(x.get("closed_book", {}).get("refusals", 0) for x in (cur, db))
        n_cb = sum(x.get("closed_book", {}).get("n", 0) for x in (cur, db))
        rag_db, cb_db = db.get("rag", {}), db.get("closed_book", {})
        label = MODEL_INFO.get(model, (model, "", model))[2]
        cites = (f"{_cell(rag_db, 'citation_valid', '')} / {_cell(rag_db, 'citation_precise', '')}"
                 if rag_db else "-")
        secs = " / ".join(f"{x['seconds']:.1f}" if x and x.get("seconds") else "-"
                          for x in (rag_db, cb_db))
        lines.append(f"| {label} | {_cell(cur.get('rag'))} | {_cell(cur.get('closed_book'))} "
                     f"| {_cell(rag_db)} | {_cell(cb_db)} | {cites} | {refusals}/{n_cb} | {secs} |")
    lines += ["", "Accuracy in %. Curated: 25 questions; database: 300 questions generated from "
              "MC2D and JARVIS-DFT records. Retrieval: lexical embedder with exact-match bonus, "
              "k = 8, numbered citations; temperature 0.2. Models served by LM Studio on one "
              "NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). * = run not complete (n answers)."]
    return "\n".join(lines) + "\n"


def to_latex(summary: dict) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Curated (25)} & \multicolumn{2}{c}{Database (300)} & "
             r"Citations & Refusals \\",
             r"Model (quantization) & RAG & Closed & RAG & Closed & valid/precise & closed \\",
             r"\midrule"]
    for model, s in summary.items():
        cur, db = s.get("curated", {}), s.get("database", {})
        refusals = sum(x.get("closed_book", {}).get("refusals", 0) for x in (cur, db))
        n_cb = sum(x.get("closed_book", {}).get("n", 0) for x in (cur, db))
        rag_db = db.get("rag", {})
        _, quant, label = MODEL_INFO.get(model, (model, "", model))
        cells = [_cell(x, partial_mark="$^\\dagger$") for x in
                 (cur.get("rag"), cur.get("closed_book"), rag_db, db.get("closed_book"))]
        cites = (f"{_cell(rag_db, 'citation_valid', '')}/{_cell(rag_db, 'citation_precise', '')}"
                 if rag_db else "-")
        quant = quant.replace("_", r"\_")
        lines.append(f"{label} ({quant}) & " + " & ".join(cells)
                     + f" & {cites} & {refusals}/{n_cb} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(results: Path = RESULTS, tables: Path = PAPER / "tables") -> dict:
    summary = summarize_runs(load_rows(results))
    (results / "answers_summary.json").write_text(json.dumps(summary, indent=1))
    (tables / "answers.md").write_text(to_markdown(summary), encoding="utf-8")
    (tables / "answers.tex").write_text(to_latex(summary), encoding="utf-8")
    print(to_markdown(summary))
    return summary


if __name__ == "__main__":
    main()
