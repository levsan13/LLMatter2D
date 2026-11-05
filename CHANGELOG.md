# Changelog

All notable changes to LLMatter2D are listed here. The project follows
[semantic versioning](https://semver.org).

## Unreleased

### Results
- Repeated runs of Llama-3.1-8B and Gemma-3-12B: three runs on the curated questions and two
  retrieval-augmented runs on the database questions, as for gpt-oss-20b. Table IV gives the
  mean with the range over runs for all three models.

## 0.2.0 (2026-11-05)

Archived on Zenodo: [10.5281/zenodo.23982427](https://doi.org/10.5281/zenodo.23982427)
(source code; the knowledge-base snapshot is attached to the GitHub release only).

The release assets include `LLMatter2D-kb-2026-10-29.zip`: the converted MC2D and
JARVIS-DFT records and both indexed knowledge bases (CC BY 4.0, with sources, citations and
SHA-256 checksums), so the benchmark can be repeated without downloading the databases.

### Added
- Benchmark of 300 questions generated from MC2D and JARVIS-DFT records
  (`benchmarks/make_db_questions.py`, `benchmarks/db_questions.jsonl`).
- Retrieval ablation: `KnowledgeBase(formula_boost=...)`, `llmatter2d benchmark
  --formula-boost`, `--limit/--seed` sampling; `paper/scripts/table_retrieval.py`.
- `paper/reproduce.py --data` downloads MC2D and JARVIS-2D and builds the hashing and
  dense (BGE-small) knowledge bases; `--llm` runs RAG vs closed-book on both question sets.
- JARVIS-DFT connector maps energy above hull, magnetic state and average effective
  masses, and accepts a local `store_dir`.
- Zenodo DOI (concept DOI 10.5281/zenodo.23779294) in README, CITATION.cff and manuscript.
- Numbered citations: the model cites record numbers ([2]) that the pipeline maps to record
  ids (`RAGPipeline(citation="number")`, default); `--citation-style id` keeps the old
  behaviour. Small local models mis-copied long MC2D ids or invented id-like strings.
- `llmatter2d benchmark --resume` (keep finished answers, redo missing or failed ones),
  per-question retries and an automatic stop after three consecutive server failures;
  answers are written as they arrive.
- `llmatter2d regrade` re-applies the grader to stored answers.
- `paper/scripts/table_answers.py`: table of RAG vs closed-book answers per model; repeated
  runs (`*_run<N>.jsonl`, `paper/reproduce.py --run N --sets ... --modes ...`) are
  aggregated as mean with the range over complete runs.
- `paper/scripts/make_kb_snapshot.py`: zip of the converted records and indexed knowledge
  bases with sources, CC BY 4.0 licences, citations and checksums, for the GitHub releases.

### Changed
- Numerical answers are graded on the first quantity with the expected unit; the lenient
  any-quantity criterion is reported as `accuracy_lenient`.
- Figures are written without creation dates, so regenerating them gives identical files.
- The manuscript also cites the MC2D expansion (Campi et al., ACS Nano 2023) and the
  JARVIS-DFT 2D figshare dataset, and states their CC BY 4.0 licences.

### Fixed
- Signed exponents (`1e+06`, `10^+6`) are parsed by the answer grader.
- The grader reads bold LaTeX values (`\mathbf{3.73~\text{\AA}}`, `\textbf`, `{\rm eV}`) and
  the angstrom sign U+212B; keywords are matched after Unicode normalization, so that
  non-breaking hyphens (`Charge‑Density`, U+2011) count as hyphens. Found in the answers of
  gpt-oss-20b (6 of 1050 correct answers had been graded wrong); all stored answers were
  regraded, the scores of the other models did not change.
- Refusal detection in `table_answers.py` accepts typographic apostrophes (`I can’t`).
- `llmatter2d regrade` appends a single `| regraded` to the summary title.

### Results
- gpt-oss-20b: three runs on the curated questions and two retrieval-augmented runs on the
  database questions; Table IV shows repeated runs as a mean with the range where the runs
  differ. All models were served by LM Studio on one NVIDIA RTX PRO 6000 Blackwell GPU.

## 0.1.1 (2026-10-28)

Archived on Zenodo: [10.5281/zenodo.23779295](https://doi.org/10.5281/zenodo.23779295).

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
