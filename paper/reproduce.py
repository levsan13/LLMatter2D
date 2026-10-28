"""Regenerate every figure, table and benchmark result of the manuscript.

    pip install -e ".[paper]"
    python paper/reproduce.py                    # figures, tables, retrieval benchmark
    python paper/reproduce.py --llm lmstudio     # + RAG vs closed-book with the LM Studio model

Outputs go to paper/figures, paper/tables and paper/results. Benchmarks on the full
knowledge base need it built first (see paper/README.md); without it the seed set is used.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).parent
ROOT = PAPER.parent


def run(*args: str) -> None:
    print("$", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", default="kb", help="knowledge base (default: kb; seed set if absent)")
    p.add_argument("--llm", help="LLM spec for the RAG vs closed-book benchmark")
    p.add_argument("-k", default="8")
    a = p.parse_args()

    run(str(PAPER / "scripts" / "fig_archimedean.py"))
    run(str(PAPER / "scripts" / "table_archimedean.py"))
    bench = str(ROOT / "benchmarks" / "seed_questions.jsonl")
    results = PAPER / "results"
    run("-m", "llmatter2d.cli", "benchmark", bench, "--kb", a.kb, "-k", a.k,
        "-o", str(results / "retrieval_seed.jsonl"))
    if a.llm:
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.llm)
        run("-m", "llmatter2d.cli", "benchmark", bench, "--kb", a.kb, "-k", a.k, "--llm", a.llm,
            "--mode", "rag", "closed_book", "-o", str(results / f"answers_{tag}.jsonl"))


if __name__ == "__main__":
    main()
