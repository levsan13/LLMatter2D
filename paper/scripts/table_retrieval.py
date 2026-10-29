"""Table: retrieval quality on the full knowledge base (embedder x formula boost x k).

    python paper/scripts/table_retrieval.py --kb hashing=kb/hashing bge-small=kb/bge-small

Needs knowledge bases built with `llmatter2d index` (see paper/README.md). Writes
paper/tables/retrieval.{md,tex} and paper/results/retrieval_full.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmatter2d.evaluation import load_benchmark, run_benchmark, summarize
from llmatter2d.rag import KnowledgeBase

PAPER = Path(__file__).parents[1]
ROOT = PAPER.parent
BENCHMARKS = {"database (300)": ROOT / "benchmarks" / "db_questions.jsonl",
              "curated (25)": ROOT / "benchmarks" / "seed_questions.jsonl"}
BOOSTS = (0.25, 0.0)
KS = (1, 3, 8)


def evaluate(kbs: dict[str, Path]) -> list[dict]:
    rows = []
    for name, path in kbs.items():
        kb = KnowledgeBase.load(path)
        for boost in BOOSTS:
            kb.formula_boost = boost
            for bench, qpath in BENCHMARKS.items():
                items = load_benchmark(qpath)
                for k in KS:
                    s = summarize(run_benchmark(items, kb, k=k))["retrieval"]
                    rows.append({"embedder": name, "boost": boost, "benchmark": bench, "k": k,
                                 "recall": s["recall_at_k"], "mrr": s["mrr"],
                                 "n_records": len(kb)})
    return rows


def _pivot(rows: list[dict]):
    keys = sorted({(r["embedder"], r["boost"]) for r in rows}, key=lambda x: (x[0], -x[1]))
    benches = list(BENCHMARKS)
    table = []
    for emb, boost in keys:
        cells = []
        for bench in benches:
            for k in KS:
                r = next(r for r in rows if (r["embedder"], r["boost"], r["benchmark"], r["k"])
                         == (emb, boost, bench, k))
                cells.append(r["recall"])
            mrr = next(r["mrr"] for r in rows if (r["embedder"], r["boost"], r["benchmark"],
                                                  r["k"]) == (emb, boost, bench, KS[-1]))
            cells.append(mrr)
        table.append((emb, boost, cells))
    return benches, table


def to_markdown(rows: list[dict]) -> str:
    benches, table = _pivot(rows)
    head = ["Embedder", "Boost"]
    for b in benches:
        head += [f"{b} R@{k}" for k in KS] + [f"{b} MRR@{KS[-1]}"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for emb, boost, cells in table:
        lines.append(f"| {emb} | {boost:g} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")
    n = rows[0]["n_records"] if rows else 0
    lines += ["", f"Knowledge base: {n} records. R@k = recall@k; MRR over the top {KS[-1]}."]
    return "\n".join(lines) + "\n"


def to_latex(rows: list[dict]) -> str:
    benches, table = _pivot(rows)
    ncell = len(KS) + 1
    spec = "ll" + ("r" * ncell) * len(benches)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{{ncell}}}{{c}}{{{b}}}" for b in benches)
             + r" \\",
             "Embedder & $\\beta$ & " + " & ".join(
                 " & ".join([f"R@{k}" for k in KS] + ["MRR"]) for _ in benches) + r" \\",
             r"\midrule"]
    for emb, boost, cells in table:
        lines.append(f"{emb} & {boost:g} & " + " & ".join(f"{c:.3f}" for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(kbs: dict[str, Path]) -> None:
    rows = evaluate(kbs)
    (PAPER / "results").mkdir(exist_ok=True)
    (PAPER / "results" / "retrieval_full.json").write_text(json.dumps(rows, indent=1))
    (PAPER / "tables" / "retrieval.md").write_text(to_markdown(rows), encoding="utf-8")
    (PAPER / "tables" / "retrieval.tex").write_text(to_latex(rows), encoding="utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kb", nargs="+", required=True, help="name=path pairs")
    a = p.parse_args()
    main({k: Path(v) for k, v in (item.split("=", 1) for item in a.kb)})
