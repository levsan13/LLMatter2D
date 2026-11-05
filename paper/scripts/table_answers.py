"""Table: answers of pre-trained models with (RAG) and without (closed book) retrieval.

    python paper/scripts/table_answers.py

Reads paper/results/answers_<questions>_<model>[_run<N>].jsonl written by
`llmatter2d benchmark ... --mode rag closed_book` (see paper/README.md) and writes
paper/tables/answers.{md,tex} and paper/results/answers_summary.json.

Repeated runs (`_run2`, `_run3`, ...) are aggregated: each cell gives the mean accuracy
over the complete runs and, when they differ, the range over runs in parentheses.
A cell that only has an incomplete run is marked (fewer answers than questions).
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
MODES = ("rag", "closed_book")
MODEL_ORDER = ["llama-3.1-8b", "gemma-3-12b", "gpt-oss-20b"]
MODEL_INFO = {  # identifier in LM Studio, quantization, label
    "llama-3.1-8b": ("meta-llama-3.1-8b-instruct", "Q4_K_M", "Llama-3.1-8B-Instruct"),
    "gemma-3-12b": ("google/gemma-3-12b", "Q4_K_M", "Gemma-3-12B"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "MXFP4", "gpt-oss-20b"),
}
REFUSAL = re.compile(r"\b(I cannot|I can't|I can not|I am unable|I'm unable|I do not have access|"
                     r"I don't have access|not able to provide|cannot verify|cannot provide)\b",
                     re.IGNORECASE)


def is_refusal(text: str) -> bool:
    """Refusal phrase in an answer; typographic apostrophes (I can’t) count as plain ones."""
    return bool(REFUSAL.search(text.replace("’", "'").replace("ʼ", "'")))


_FILE = re.compile(r"answers_(seed_questions|db_questions)_(.+?)(?:_run(\d+))?\.jsonl")


def load_rows(results: Path = RESULTS) -> dict[tuple[str, str], dict[int, list[dict]]]:
    """{(model, question_set): {run_index: rows}}"""
    runs: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(dict)
    for path in sorted(results.glob("answers_*.jsonl")):
        m = _FILE.fullmatch(path.name)
        if m:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            runs[(m.group(2), m.group(1))][int(m.group(3) or 1)] = rows
    return dict(runs)


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def run_stats(rows: list[dict], mode: str, expected: int) -> dict:
    """Statistics of one run of one mode."""
    rs = [r for r in rows if r["mode"] == mode and not r.get("error")]
    s = {
        "n": len(rs),
        "complete": len(rs) >= expected,
        "accuracy": _mean(float(r["correct"]) for r in rs if r.get("correct") is not None),
        "accuracy_lenient": _mean(float(r["correct_lenient"]) for r in rs
                                  if r.get("correct_lenient") is not None),
        "errors": sum(1 for r in rows if r["mode"] == mode and r.get("error")),
        "refusals": sum(1 for r in rs if is_refusal(r.get("answer", "") or "")),
        "seconds": _mean(r.get("seconds") for r in rs),
    }
    if mode == "rag":
        s["citation_valid"] = _mean(r.get("citation_valid") for r in rs)
        s["citation_precise"] = _mean(r.get("citation_precise") for r in rs)
        s["uncited"] = sum(1 for r in rs if not r.get("cited"))
    return s


def aggregate(runs: dict[int, list[dict]], mode: str, expected: int) -> dict | None:
    per_run = {i: run_stats(rows, mode, expected) for i, rows in sorted(runs.items())}
    per_run = {i: s for i, s in per_run.items() if s["n"] or s["errors"]}
    if not per_run:
        return None
    complete = {i: s for i, s in per_run.items() if s["complete"]}
    if complete:
        used = complete
    else:  # only partial runs: report the most advanced one, marked as incomplete
        best = max(per_run, key=lambda i: per_run[i]["n"])
        used = {best: per_run[best]}
    accs = [s["accuracy"] for s in used.values() if s["accuracy"] is not None]
    out = {
        "runs": [{"run": i, **s} for i, s in per_run.items()],
        "n_runs": len(complete),
        "n_partial": len(per_run) - len(complete),
        "complete": bool(complete),
        "n": expected if complete else next(iter(used.values()))["n"],
        "accuracy": _mean(accs),
        "accuracy_min": min(accs) if accs else None,
        "accuracy_max": max(accs) if accs else None,
        "accuracy_lenient": _mean(s["accuracy_lenient"] for s in used.values()),
        "errors": sum(s["errors"] for s in per_run.values()),
        "refusals": _mean(s["refusals"] for s in used.values()),
        "seconds": _mean(s["seconds"] for s in used.values()),
    }
    if mode == "rag":
        for key in ("citation_valid", "citation_precise", "uncited"):
            out[key] = _mean(s.get(key) for s in used.values())
    by_cat = defaultdict(list)
    for i in used:
        for r in runs[i]:
            if r["mode"] == mode and not r.get("error") and r.get("correct") is not None:
                by_cat[r["category"]].append(float(r["correct"]))
    out["by_category"] = {c: round(mean(v), 3) for c, v in sorted(by_cat.items())}
    return out


def summarize_runs(runs) -> dict:
    out: dict = defaultdict(dict)
    for (model, qset), by_run in runs.items():
        name, expected = SETS[qset]
        out[model][name] = {mode: aggregate(by_run, mode, expected) for mode in MODES}
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], len(order))))


def _cell(stats: dict | None, key: str = "accuracy", partial_mark: str = "*",
          dash: str = "-") -> str:
    if not stats or stats.get(key) is None:
        return "-"
    txt = f"{100 * stats[key]:.1f}"
    if key == "accuracy" and stats.get("n_runs", 1) > 1:
        lo, hi = f"{100 * stats['accuracy_min']:.1f}", f"{100 * stats['accuracy_max']:.1f}"
        if lo != hi:  # identical runs: the mean says it all
            txt += f" ({lo}{dash}{hi})"
    if not stats.get("complete", True):
        txt += f"{partial_mark} (n={stats['n']})"
    return txt


def _runs_note(summary: dict) -> str:
    parts = []
    for model, s in summary.items():
        label = MODEL_INFO.get(model, (model, "", model))[2]
        items = []
        for name in ("curated", "database"):
            modes = s.get(name, {})
            counts = [f"{m.replace('closed_book', 'closed')} {modes[m]['n_runs']}"
                      + (f" (+{modes[m]['n_partial']} partial)" if modes[m]["n_partial"] else "")
                      for m in MODES if modes.get(m)]
            if counts:
                items.append(f"{name} " + ", ".join(counts))
        parts.append(f"{label}: " + "; ".join(items))
    return "Complete runs per cell - " + " | ".join(parts) + "."


def _refusals(db: dict) -> str:
    cb = db.get("closed_book")
    if not cb or cb.get("refusals") is None:
        return "-"
    return f"{cb['refusals']:.0f}/{cb['n']}"


def to_markdown(summary: dict) -> str:
    head = ("| Model | Curated: RAG | Curated: closed book | Database: RAG | Database: closed book "
            "| Citations valid / precise (database) | Closed-book refusals (database) "
            "| s per answer (RAG / closed) |")
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for model, s in summary.items():
        cur, db = s.get("curated", {}), s.get("database", {})
        rag_db, cb_db = db.get("rag"), db.get("closed_book")
        label = MODEL_INFO.get(model, (model, "", model))[2]
        cites = (f"{_cell(rag_db, 'citation_valid', '')} / {_cell(rag_db, 'citation_precise', '')}"
                 if rag_db else "-")
        secs = " / ".join(f"{x['seconds']:.1f}" if x and x.get("seconds") else "-"
                          for x in (rag_db, cb_db))
        lines.append(f"| {label} | {_cell(cur.get('rag'))} | {_cell(cur.get('closed_book'))} "
                     f"| {_cell(rag_db)} | {_cell(cb_db)} | {cites} | {_refusals(db)} | {secs} |")
    lines += ["", "Accuracy in %: mean over complete runs, range over runs in parentheses when "
              "the runs differ. Curated: 25 questions; database: 300 questions generated from MC2D "
              "and JARVIS-DFT records. Retrieval: lexical embedder with exact-match bonus, k = 8, "
              "numbered citations; temperature 0.2. Models served by LM Studio on one NVIDIA "
              "RTX PRO 6000 Blackwell GPU (96 GB VRAM). * = run not complete (n answers).",
              "",
              _runs_note(summary)]
    return "\n".join(lines) + "\n"


def to_latex(summary: dict) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Curated (25)} & \multicolumn{2}{c}{Database (300)} & "
             r"Citations & Refusals \\",
             r"Model (quantization) & RAG & Closed & RAG & Closed & valid/precise & closed \\",
             r"\midrule"]
    for model, s in summary.items():
        cur, db = s.get("curated", {}), s.get("database", {})
        rag_db = db.get("rag")
        _, quant, label = MODEL_INFO.get(model, (model, "", model))
        cells = [_cell(x, partial_mark="$^\\dagger$", dash="--") for x in
                 (cur.get("rag"), cur.get("closed_book"), rag_db, db.get("closed_book"))]
        cites = (f"{_cell(rag_db, 'citation_valid', '')}/{_cell(rag_db, 'citation_precise', '')}"
                 if rag_db else "-")
        quant = quant.replace("_", r"\_")
        lines.append(f"{label} ({quant}) & " + " & ".join(cells)
                     + f" & {cites} & {_refusals(db)} \\\\")
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
