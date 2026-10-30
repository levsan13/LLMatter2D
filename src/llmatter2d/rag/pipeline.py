"""Retrieval-augmented generation: retrieve records, build the prompt, call the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLM, LLMResponse
from ..llm.prompts import system_prompts
from .knowledge_base import Filters, Hit, KnowledgeBase

_NUMBERED_CITATION = re.compile(r"\[\s*(?:records?\s*|r)?(\d+(?:\s*[,;]\s*(?:r)?\d+)*)\s*\]",
                                re.IGNORECASE)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    response: LLMResponse
    prompt: str = field(repr=False, default="")
    raw_text: str = field(repr=False, default="")

    @property
    def sources(self) -> list[str]:
        return [h.uid for h in self.hits]


def format_context(hits: list[Hit]) -> str:
    blocks = [f'<record n="{n}" id="{h.uid}" score="{h.score:.3f}">\n{h.text}\n</record>'
              for n, h in enumerate(hits, 1)]
    return "<context>\n" + "\n".join(blocks) + "\n</context>"


def resolve_citations(text: str, uids: list[str]) -> str:
    """Replace numbered citations such as [2], [1, 3] or 【2】 by the record ids they cite."""
    text = text.replace("【", "[").replace("】", "]")  # bracket style used by some models

    def replace(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if all(1 <= n <= len(uids) for n in nums):
            return "[" + ", ".join(uids[n - 1] for n in nums) + "]"
        return m.group(0)

    return _NUMBERED_CITATION.sub(replace, text)


class RAGPipeline:
    def __init__(self, kb: KnowledgeBase, llm: LLM, k: int = 8, citation: str = "number"):
        self.kb = kb
        self.llm = llm
        self.k = k
        self.citation = citation
        self.prompts = system_prompts(citation)

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
        response = self.llm.generate(self.prompts[task], prompt)
        text = response.text
        if self.citation == "number":
            text = resolve_citations(text, [h.uid for h in hits])
        return Answer(text=text, hits=hits, response=response, prompt=prompt,
                      raw_text=response.text)
