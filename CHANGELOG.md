# Changelog

All notable changes to LLMatter2D are listed here. The project follows
[semantic versioning](https://semver.org).

## 0.1.0 (2026-10-28)

First public version.

### Added
- `Material2D` data model with method-aware band gaps, magnetism, stability,
  tight-binding models, ARPES features and provenance.
- Connectors: MC2D (OPTIMADE), C2DB (ASE database), 2DMatPedia (JSON), JARVIS-DFT 2D,
  curated seed set of 13 monolayers.
- Wannier90 `_hr.dat` reader/writer; ARPES feature extraction (Fermi velocity from MDC
  peaks, effective mass from EDC peaks).
- Knowledge base with metadata filters, hashing and sentence-transformer embeddings,
  formula boost, structural neighbours and persistence.
- LLM backends: LM Studio (default, local pre-trained models), Ollama, Claude, offline.
- The eleven Archimedean lattices with nearest-neighbour tight-binding, flat-band,
  linear-crossing, quadratic-touching and multifold-degeneracy detection; POSCAR I/O.
- Candidate generation and screening on Archimedean lattices, LLM review, POSCAR export.
- Structure analysis of user-supplied monolayers (coordination, tiling match, neighbours).
- Benchmark (`llmatter2d benchmark`): retrieval recall@k / MRR, answer accuracy with
  unit-aware grading, citation validity and precision; RAG versus closed-book modes.
- `paper/`: scripts regenerating the manuscript figures, tables and benchmark results.
- Command-line interface, examples, test suite and GitHub Actions CI.
