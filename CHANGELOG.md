# Changelog

All notable changes to LLMatter2D are listed here. The project follows
[semantic versioning](https://semver.org).

## Unreleased

### Added
- Benchmark of 300 questions generated from MC2D and JARVIS-DFT records
  (`benchmarks/make_db_questions.py`, `benchmarks/db_questions.jsonl`).
- Retrieval ablation: `KnowledgeBase(formula_boost=...)`, `llmatter2d benchmark
  --formula-boost`, `--limit/--seed` sampling; `paper/scripts/table_retrieval.py`.
- `paper/reproduce.py --data` downloads MC2D and JARVIS-2D and builds the hashing and
  dense (BGE-small) knowledge bases; `--llm` runs RAG vs closed-book on both question sets.
- JARVIS-DFT connector maps energy above hull, magnetic state and average effective
  masses, and accepts a local `store_dir`.

## 0.1.1 (2026-10-28)

### Added
- Provisional manuscript in REVTeX (`paper/manuscript/`) with a TikZ architecture figure,
  verified bibliography and an Overleaf bundling script.
- Table of the highest-ranked Archimedean candidates (`paper/scripts/table_candidates.py`).

### Fixed
- `.zenodo.json` uses the lowercase SPDX license identifier (`mit`) expected by Zenodo.

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
