# Reproducing the manuscript

Everything in the manuscript is regenerated from this repository.

```bash
pip install -e ".[paper,jarvis,embeddings]"
python paper/reproduce.py                          # figures and tables (no data needed)
python paper/reproduce.py --data                   # + download MC2D and JARVIS-2D, build both
                                                   #   knowledge bases, retrieval table
python paper/reproduce.py --llm lmstudio:<model>   # + RAG vs closed-book answers
```

## Map of figures, tables and results

| Item | Content | Produced by | Output |
|---|---|---|---|
| Fig. 1 | Architecture of LLMatter2D | TikZ inside `manuscript/main.tex` | - |
| Fig. 2 | The 11 Archimedean tilings, primitive cells and bipartite sublattices | `scripts/fig_archimedean.py` | `figures/archimedean_tilings.{pdf,png}` |
| Fig. 3 | Nearest-neighbour tight-binding bands of the 11 tilings (flat bands highlighted) | `scripts/fig_archimedean.py` | `figures/archimedean_bands.{pdf,png}` |
| Table I | Sites per cell, coordination, site density, bipartiteness, flat bands, band touchings | `scripts/table_archimedean.py` | `tables/archimedean.{md,tex}` |
| Table II | Highest-ranked new Archimedean candidates (before DFT) | `scripts/table_candidates.py` | `tables/candidates.{md,tex}` |
| Table III | Retrieval on the 3799-record knowledge base (embedder x formula bonus x k) | `scripts/table_retrieval.py` (via `reproduce.py --data`) | `tables/retrieval.{md,tex}`, `results/retrieval_full.json` |
| Pending | Answer accuracy and citation quality, RAG vs closed-book, per LLM | `reproduce.py --llm <spec>` | `results/answers_<questions>_<llm>.{jsonl,md}` |

## Question sets

* `benchmarks/seed_questions.jsonl`: 25 curated questions on the seed materials.
* `benchmarks/db_questions.jsonl`: 300 questions generated from MC2D and JARVIS-DFT records
  by `benchmarks/make_db_questions.py` (seed 2026). The file is committed and kept fixed,
  so results stay comparable when the databases are updated; regenerate it only on purpose.

## Running the pre-trained models (LM Studio)

1. Load the model in LM Studio with a context length of at least 8k tokens and start the
   server (Developer tab, or `lms server start`).
2. Run, for each model:

   ```bash
   python paper/reproduce.py --llm lmstudio:<model-id>               # all 325 questions
   python paper/reproduce.py --llm lmstudio:<model-id> --limit 50    # quick check
   ```

   Each question is asked twice (with and without retrieval). With a 7-8B model on a
   consumer GPU, expect roughly one to a few hours for the full set.
3. Record the model identifier, quantisation, context length and LM Studio version.
   Repeat the runs (at least three) to estimate the spread.

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

* Figs. 2-3 and Tables I-II are final for the nearest-neighbour model and screening heuristic.
* Table III is computed on the full knowledge base (MC2D + JARVIS-2D + seed, 3799 records,
  built on 2026-10-29). C2DB and 2DMatPedia can be added when their files are available.
* The answers of pre-trained models (RAG vs closed-book) are still to be run in LM Studio.
