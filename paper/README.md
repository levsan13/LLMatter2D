# Reproducing the manuscript

Everything in the manuscript is regenerated from this repository.

```bash
pip install -e ".[paper]"
python paper/reproduce.py                  # figures, tables, retrieval benchmark
python paper/reproduce.py --llm lmstudio   # + RAG vs closed-book with the LM Studio model
```

## Map of figures, tables and results

| Item | Content | Produced by | Output |
|---|---|---|---|
| Fig. 1 | Architecture of LLMatter2D | diagram in the main `README.md` (to be redrawn for the journal) | - |
| Fig. 2 | The 11 Archimedean tilings, primitive cells and bipartite sublattices | `scripts/fig_archimedean.py` | `figures/archimedean_tilings.{pdf,png}` |
| Fig. 3 | Nearest-neighbour tight-binding bands of the 11 tilings (flat bands highlighted) | `scripts/fig_archimedean.py` | `figures/archimedean_bands.{pdf,png}` |
| Table 1 | Sites per cell, coordination, site density, bipartiteness, flat bands, band touchings | `scripts/table_archimedean.py` | `tables/archimedean.{md,tex}` |
| Table 2 | Highest-ranked new Archimedean candidates (before DFT) | `scripts/table_candidates.py` | `tables/candidates.{md,tex}` |
| Benchmark | Retrieval recall@k and MRR | `llmatter2d benchmark` (via `reproduce.py`) | `results/retrieval_seed.{jsonl,md}` |
| Benchmark | Answer accuracy and citation quality, RAG vs closed-book, per LLM | `reproduce.py --llm <spec>` | `results/answers_<llm>.{jsonl,md}` |

## Manuscript (LaTeX, REVTeX 4.2)

`manuscript/main.tex` and `manuscript/references.bib` hold the provisional draft
(class option `prmaterials`; change it for another APS journal). Text in red marked
**Pending** stands for results that do not exist yet. Fig. 1 is drawn with TikZ inside
`main.tex`; the other figures and the tables are read from `figures/` and `tables/`.

For Overleaf, build a flat bundle and upload it with *New Project -> Upload Project*:

```bash
python paper/manuscript/make_overleaf_zip.py   # -> paper/manuscript/LLMatter2D_overleaf.zip
```

## Status of the results

* Figs. 2-3 and Table 1 are final for the nearest-neighbour model.
* Table 2 currently uses the 13-material seed set, where retrieval is trivial
  (recall@8 = 1.0). For the manuscript it must be recomputed on the full knowledge base
  (C2DB + MC2D + 2DMatPedia + JARVIS, several thousand records), where the other
  materials act as distractors, and with a larger question set generated from the
  database records.
* Table 3 needs runs with the pre-trained models chosen for the paper (LM Studio).
  Record the exact model identifiers, quantisation and context length for each run.

## Building the full knowledge base

```bash
llmatter2d ingest mc2d -o data/raw/mc2d.jsonl
llmatter2d ingest c2db --path /path/to/c2db.db -o data/raw/c2db.jsonl
llmatter2d ingest 2dmatpedia --path /path/to/db.json -o data/raw/2dmatpedia.jsonl
llmatter2d ingest jarvis -o data/raw/jarvis.jsonl
llmatter2d index data/raw/*.jsonl --seed --embedder st:BAAI/bge-small-en-v1.5 -o kb
python paper/reproduce.py --kb kb --llm lmstudio
```

Record the database versions and download dates (stored in each record's provenance)
in the data-availability statement.
