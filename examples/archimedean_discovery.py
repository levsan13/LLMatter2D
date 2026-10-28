"""Propose new 2D materials on the Archimedean lattices.

    python examples/archimedean_discovery.py            # ranking + POSCARs (no LLM)
    python examples/archimedean_discovery.py lmstudio   # + review by the LM Studio model
"""

import sys

from llmatter2d.lattices import ARCHIMEDEAN, analyze_bands
from llmatter2d.llm import get_llm
from llmatter2d.predict import (
    enumerate_candidates,
    review_with_llm,
    score_candidates,
    write_candidates,
)
from llmatter2d.rag import KnowledgeBase, RAGPipeline

print("Nearest-neighbour tight-binding on the 11 Archimedean lattices (t = 1):\n")
for name, lat in ARCHIMEDEAN.items():
    print(f"- {analyze_bands(lat).summary()}")

kb = KnowledgeBase.from_seed()
candidates = score_candidates(enumerate_candidates(["B", "C", "N", "Si", "Ge", "Sn"]), kb=kb)
novel = [c for c in candidates if c.known_as is None]

print("\nTop new candidates:\n")
for c in novel[:8]:
    print(c.describe(), "\n")

print("Benchmarks recovered from the knowledge base:",
      ", ".join(f"{c.formula}@{c.lattice.name} = {c.known_as}" for c in candidates if c.known_as))

path = write_candidates(novel[:8], "candidates")
print(f"\nPOSCAR files for DFT validation: {path.parent}/")

if len(sys.argv) > 1:
    review = review_with_llm(novel, RAGPipeline(kb, get_llm(sys.argv[1]), k=8), top=8)
    print("\nLLM review:\n" + review.text)
