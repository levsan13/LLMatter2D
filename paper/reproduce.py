"""Regenerate every figure, table and benchmark result of the manuscript.

    pip install -e ".[paper]"
    python paper/reproduce.py                         # figures and tables (no data needed)
    python paper/reproduce.py --data                  # + download MC2D and JARVIS-2D, build
                                                      #   the knowledge bases, retrieval table
    python paper/reproduce.py --llm lmstudio:<model>  # + RAG vs closed-book answers
                                                      #   (needs the knowledge bases)

Outputs go to paper/figures, paper/tables and paper/results. The data step needs network
access and `pip install -e ".[jarvis,embeddings]"`; the LLM step needs a running LM Studio
server (or another backend spec).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).parent
ROOT = PAPER.parent
RAW = ROOT / "data" / "raw"
KB_HASHING = ROOT / "kb" / "hashing"
KB_DENSE = ROOT / "kb" / "bge-small"


def run(*args: str) -> None:
    print("$", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def build_data() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    run("-m", "llmatter2d.cli", "ingest", "mc2d", "-o", str(RAW / "mc2d.jsonl"))
    run("-m", "llmatter2d.cli", "ingest", "jarvis", "--path", str(RAW / "jarvis_store"),
        "-o", str(RAW / "jarvis.jsonl"))
    sources = [str(RAW / "mc2d.jsonl"), str(RAW / "jarvis.jsonl")]
    run("-m", "llmatter2d.cli", "index", *sources, "--seed", "--embedder", "hashing",
        "-o", str(KB_HASHING))
    run("-m", "llmatter2d.cli", "index", *sources, "--seed",
        "--embedder", "st:BAAI/bge-small-en-v1.5", "-o", str(KB_DENSE))
    # The committed question set is kept fixed; regenerate it only on purpose with
    # benchmarks/make_db_questions.py, because new database releases change the records.
    run(str(PAPER / "scripts" / "table_retrieval.py"),
        "--kb", f"hashing={KB_HASHING}", f"bge-small={KB_DENSE}")


def run_answers(llm: str, kb: Path, k: str, limit: str | None) -> None:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", llm)
    for name in ("seed_questions", "db_questions"):
        args = ["-m", "llmatter2d.cli", "benchmark", str(ROOT / "benchmarks" / f"{name}.jsonl"),
                "--kb", str(kb), "-k", k, "--llm", llm, "--mode", "rag", "closed_book",
                "-o", str(PAPER / "results" / f"answers_{name}_{tag}.jsonl")]
        if limit:
            args += ["--limit", limit]
        run(*args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="store_true", help="download data and build the KBs")
    p.add_argument("--llm", help="LLM spec for the RAG vs closed-book benchmark")
    p.add_argument("--kb", default=str(KB_DENSE), help="KB for the LLM benchmark")
    p.add_argument("-k", default="8")
    p.add_argument("--limit", help="random subset of questions per benchmark file")
    a = p.parse_args()

    run(str(PAPER / "scripts" / "fig_archimedean.py"))
    run(str(PAPER / "scripts" / "table_archimedean.py"))
    run(str(PAPER / "scripts" / "table_candidates.py"))
    if a.data:
        build_data()
    if a.llm:
        run_answers(a.llm, Path(a.kb), a.k, a.limit)


if __name__ == "__main__":
    main()
