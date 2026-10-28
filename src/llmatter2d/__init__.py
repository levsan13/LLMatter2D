"""LLMatter2D: retrieval-augmented LLM for two-dimensional materials.

The package is organised in layers:

* :mod:`llmatter2d.schema`   - data model for 2D materials (structure, electronic
  properties, tight-binding models, ARPES features, provenance).
* :mod:`llmatter2d.ingest`   - connectors that turn DFT databases, Wannier90 and
  ARPES data into :class:`~llmatter2d.schema.Material2D` records.
* :mod:`llmatter2d.rag`      - material cards, embeddings, vector store and the
  retrieval-augmented generation pipeline.
* :mod:`llmatter2d.llm`      - LLM backends (Claude, local Ollama models, offline).
* :mod:`llmatter2d.lattices` - the 11 Archimedean lattices and a tight-binding solver.
* :mod:`llmatter2d.predict`  - generation and screening of new 2D candidates.
"""

__version__ = "0.1.1"
