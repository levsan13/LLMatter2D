"""Retrieval-augmented generation: retrieve records, build the prompt, call the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLM, LLMResponse
from ..llm.prompts import PROMPTS
from .knowledge_base import Filters, Hit, KnowledgeBase


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    response: LLMResponse
    prompt: str = field(repr=False, default="")

    @property
    def sources(self) -> list[str]:
        return [h.uid for h in self.hits]


def format_context(hits: list[Hit]) -> str:
    blocks = [f'<record n="{n}" id="{h.uid}" score="{h.score:.3f}">\n{h.text}\n</record>'
              for n, h in enumerate(hits, 1)]
    return "<context>\n" + "\n".join(blocks) + "\n</context>"


class RAGPipeline:
    def __init__(self, kb: KnowledgeBase, llm: LLM, k: int = 8):
        self.kb = kb
        self.llm = llm
        self.k = k

    def retrieve(self, query: str, filters: Filters | None = None,
                 k: int | None = None) -> list[Hit]:
        return self.kb.search(query, k or self.k, filters)

    def build_prompt(self, question: str, hits: list[Hit], extra: str | None = None) -> str:
        parts = [format_context(hits)]
        if extra:
            parts.append(extra)
        parts.append(f"<question>\n{question}\n</question>")
        return "\n\n".join(parts)

    def ask(self, question: str, filters: Filters | None = None, task: str = "qa",
            extra: str | None = None, retrieval_query: str | None = None) -> Answer:
        hits = self.retrieve(retrieval_query or question, filters)
        prompt = self.build_prompt(question, hits, extra)
        response = self.llm.generate(PROMPTS[task], prompt)
        return Answer(text=response.text, hits=hits, response=response, prompt=prompt)
