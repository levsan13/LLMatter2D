"""Quick start: build a knowledge base and ask grounded questions.

With LM Studio (server running, model loaded):  python examples/quickstart.py
A specific LM Studio model:                      python examples/quickstart.py lmstudio:<model-id>
Without any LLM (shows retrieved records):       python examples/quickstart.py offline
"""

import sys

from llmatter2d.ingest import OptimadeConnector
from llmatter2d.llm import get_llm
from llmatter2d.rag import Filters, KnowledgeBase, RAGPipeline

llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else "lmstudio")

kb = KnowledgeBase.from_seed()
try:  # a few live records from the Materials Cloud 2D database
    kb.add(OptimadeConnector("mc2d").fetch(limit=50))
    kb.build()
except OSError as exc:
    print(f"(MC2D not reachable, using the seed set only: {exc})")

rag = RAGPipeline(kb, llm, k=6)

questions = [
    ("What is known about the band gap of MoS2 and how does it depend on the method?", None),
    ("Which of these materials could host the quantum spin Hall effect?",
     Filters(tags_any=["xene"])),
    ("List semiconductors with a gap between 1.5 and 2.5 eV.",
     Filters(gap_min=1.5, gap_max=2.5)),
]
for q, f in questions:
    ans = rag.ask(q, filters=f)
    print(f"\n### {q}\n{ans.text}\n(sources: {', '.join(ans.sources)})")
