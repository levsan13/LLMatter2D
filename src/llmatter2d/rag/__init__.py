"""Retrieval-augmented generation over the 2D-materials knowledge base."""

from .documents import material_metadata, material_to_card
from .embeddings import HashingEmbedder, get_embedder
from .knowledge_base import Filters, Hit, KnowledgeBase
from .pipeline import Answer, RAGPipeline, format_context

__all__ = [
    "Answer",
    "Filters",
    "HashingEmbedder",
    "Hit",
    "KnowledgeBase",
    "RAGPipeline",
    "format_context",
    "get_embedder",
    "material_metadata",
    "material_to_card",
]
