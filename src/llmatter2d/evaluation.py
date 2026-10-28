"""Benchmark: retrieval quality and answer correctness, RAG versus closed-book.

A benchmark is a JSON-lines file; each line is one question::

    {"id": "gap-mos2-pbe", "question": "...", "category": "band gap",
     "relevant": ["seed:MoS2-1H"],                 # records that hold the answer
     "answer_value": 1.7, "unit": "eV", "tolerance": 0.15,   # numeric questions
     "keywords_all": [...], "keywords_any": [...],           # textual questions
     "filters": {"gap_min": 1.0}}                            # optional Filters kwargs

Metrics
-------
* retrieval: recall@k of the ``relevant`` records and reciprocal rank of the first one;
* answer: a numeric answer is correct if a quantity with the expected unit lies within
  ``tolerance`` of ``answer_value`` (meV/eV, nm/angstrom, km/s/m/s converted); textual
  answers are graded with the keyword lists;
* citations (RAG mode): fraction of cited record ids that are among the retrieved ones
  (valid) and among the relevant ones (precise).

Modes: ``retrieval`` (no LLM call), ``rag`` (retrieved context) and ``closed_book``
(same LLM, no context) - the comparison the RAG layer has to win.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from .llm.base import LLM
from .llm.prompts import PROMPTS
from .rag.knowledge_base import Filters, KnowledgeBase
from .rag.pipeline import RAGPipeline

MODES = ("retrieval", "rag", "closed_book")


@dataclass
class BenchmarkItem:
    id: str
    question: str
    category: str = "general"
    relevant: list[str] = field(default_factory=list)
    answer_value: float | None = None
    unit: str | None = None
    tolerance: float | None = None
    keywords_all: list[str] = field(default_factory=list)
    keywords_any: list[str] = field(default_factory=list)
    filters: dict[str, Any] | None = None

    @property
    def filter_obj(self) -> Filters | None:
        return Filters(**self.filters) if self.filters else None


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            items.append(BenchmarkItem(**json.loads(line)))
    return items


# ------------------------------------------------------------------------- answer parsing

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−", "0123456789--")
_UNIT_ALIASES = {
    "ev": "eV", "mev": "meV", "å": "A", "a": "A", "angstrom": "A", "angstroms": "A",
    "nm": "nm", "m/s": "m/s", "km/s": "km/s", "k": "K", "mub": "muB", "μb": "muB",
}
_TO_BASE = {"meV": ("eV", 1e-3), "nm": ("A", 10.0), "km/s": ("m/s", 1e3)}
_QUANTITY = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)"                       # mantissa
    r"(?:\s*(?:[x×·*]\s*10\s*\^?\s*(-?\d+)|e(-?\d+)))?"  # x 10^n or e-notation
    r"\s*(meV|eV|Å|angstroms?|(?-i:A)|nm|km/s|m\s*/\s*s|m\s*s\s*\^?-1|(?-i:K)|μB|muB)(?![\w/])",
    re.IGNORECASE,
)


def extract_quantities(text: str) -> list[tuple[float, str]]:
    """Numbers with units found in free text, converted to base units (eV, A, m/s, K)."""
    text = text.translate(_SUPERSCRIPT)
    out = []
    for mant, exp1, exp2, unit in _QUANTITY.findall(text):
        value = float(mant.replace(",", "."))
        exp = exp1 or exp2
        if exp:
            value *= 10 ** int(exp)
        u = re.sub(r"\s+", "", unit).lower()
        u = "m/s" if u in ("m/s", "ms^-1", "ms-1") else _UNIT_ALIASES.get(u, unit)
        if u in _TO_BASE:
            u, factor = _TO_BASE[u]
            value *= factor
        out.append((value, u))
    return out


_CITATION = re.compile(r"\[([^\[\]]+)\]")


def cited_ids(text: str) -> set[str]:
    ids = set()
    for group in _CITATION.findall(text):
        for token in re.split(r"[,;\s]+", group):
            token = token.strip()
            if re.fullmatch(r"[A-Za-z0-9_\-]+:[^\s]+", token):
                ids.add(token)
    return ids


def grade_answer(item: BenchmarkItem, text: str) -> dict[str, Any]:
    low = text.lower()
    result: dict[str, Any] = {}
    if item.answer_value is not None:
        unit = _UNIT_ALIASES.get((item.unit or "").lower(), item.unit)
        tol = item.tolerance if item.tolerance is not None else 0.05 * abs(item.answer_value)
        values = [v for v, u in extract_quantities(text) if u == unit]
        result["numeric_ok"] = any(abs(v - item.answer_value) <= tol for v in values)
    if item.keywords_all or item.keywords_any:
        ok_all = all(k.lower() in low for k in item.keywords_all)
        ok_any = (not item.keywords_any) or any(k.lower() in low for k in item.keywords_any)
        result["keywords_ok"] = ok_all and ok_any
    checks = [v for k, v in result.items() if k.endswith("_ok")]
    result["correct"] = all(checks) if checks else None
    return result


# ------------------------------------------------------------------------------- running

def evaluate_item(item: BenchmarkItem, kb: KnowledgeBase, llm: LLM | None, mode: str,
                  k: int = 8) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    row: dict[str, Any] = {"id": item.id, "category": item.category, "mode": mode}
    if mode in ("retrieval", "rag"):
        hits = kb.search(item.question, k, item.filter_obj)
        uids = [h.uid for h in hits]
        rel = set(item.relevant)
        row["retrieved"] = uids
        if rel:
            row["recall_at_k"] = len(rel & set(uids)) / len(rel)
            ranks = [i for i, u in enumerate(uids, 1) if u in rel]
            row["reciprocal_rank"] = 1 / ranks[0] if ranks else 0.0
    if mode == "retrieval":
        return row
    if llm is None:
        raise ValueError(f"mode {mode!r} needs an LLM")
    if mode == "rag":
        answer = RAGPipeline(kb, llm, k=k).ask(item.question, filters=item.filter_obj)
        text = answer.text
        cites = cited_ids(text)
        row["cited"] = sorted(cites)
        if cites:
            row["citation_valid"] = len(cites & set(row["retrieved"])) / len(cites)
            if item.relevant:
                row["citation_precise"] = len(cites & set(item.relevant)) / len(cites)
    else:
        text = llm.generate(PROMPTS["closed_book"], item.question).text
    row["answer"] = text
    row.update(grade_answer(item, text))
    return row


def run_benchmark(items: list[BenchmarkItem], kb: KnowledgeBase, llm: LLM | None = None,
                  modes: tuple[str, ...] = ("retrieval",), k: int = 8) -> list[dict[str, Any]]:
    return [evaluate_item(item, kb, llm, mode, k) for mode in modes for item in items]


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    def avg(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(mean(vals), 3) if vals else None

    out = {}
    for mode in dict.fromkeys(r["mode"] for r in rows):
        rs = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "n": len(rs),
            "recall_at_k": avg(rs, "recall_at_k"),
            "mrr": avg(rs, "reciprocal_rank"),
            "accuracy": avg([{"c": float(r["correct"])} for r in rs
                             if r.get("correct") is not None], "c"),
            "citation_valid": avg(rs, "citation_valid"),
            "citation_precise": avg(rs, "citation_precise"),
        }
    return out


def format_summary(summary: dict[str, dict[str, float]], title: str = "") -> str:
    cols = ["n", "recall_at_k", "mrr", "accuracy", "citation_valid", "citation_precise"]
    lines = [f"### {title}", ""] if title else []
    lines.append("| mode | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for mode, s in summary.items():
        lines.append(f"| {mode} | " + " | ".join("-" if s[c] is None else str(s[c])
                                                for c in cols) + " |")
    return "\n".join(lines)
