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
| Table IV | Answer accuracy and citation quality, RAG vs closed-book, per pre-trained model | `reproduce.py --llm <spec>`, then `scripts/table_answers.py` | `tables/answers.{md,tex}`, `results/answers_*.jsonl`, `results/answers_summary.json` |
| Sec. III D | Citing record ids vs record numbers (Llama-3.1-8B, 50 database questions) | `llmatter2d benchmark --citation-style id/number --limit 50` | `results/quick/*_idcite.*`, `results/quick/*_numcite.*` |

## Question sets

* `benchmarks/seed_questions.jsonl`: 25 curated questions on the seed materials.
* `benchmarks/db_questions.jsonl`: 300 questions generated from MC2D and JARVIS-DFT records
  by `benchmarks/make_db_questions.py` (seed 2026). The file is committed and kept fixed,
  so results stay comparable when the databases are updated; regenerate it only on purpose.

## Running the pre-trained models (LM Studio)

1. Start the LM Studio server (Developer tab, or `lms server start`). Models are loaded on
   first use. If the server runs on another machine, forward its port over SSH and point
   LLMatter2D to it:

   ```bash
   ssh -N -o ServerAliveInterval=30 -L 1234:127.0.0.1:1234 user@server
   export LMSTUDIO_HOST=http://127.0.0.1:1234     # PowerShell: $env:LMSTUDIO_HOST = "..."
   ```

2. Run, for each model (the models of the manuscript are `meta-llama-3.1-8b-instruct`,
   `google/gemma-3-12b` and `openai/gpt-oss-20b`):

   ```bash
   python paper/reproduce.py --llm lmstudio:meta-llama-3.1-8b-instruct --tag llama-3.1-8b
   python paper/reproduce.py --llm lmstudio:google/gemma-3-12b --tag gemma-3-12b
   python paper/reproduce.py --llm lmstudio:openai/gpt-oss-20b --tag gpt-oss-20b
   ```

   Each question is asked twice (with and without retrieval). Answers are written as they
   arrive and runs resume automatically (`--resume`), so an interrupted run is continued
   by repeating the command. After each model the answers table (`tables/answers.md`)
   is rebuilt.
3. Record the model identifier, quantisation, context length and LM Studio version.
4. Repeat runs to estimate the spread between runs. Runs 2, 3, ... are written to
   `answers_<set>_<tag>_run<N>.jsonl`; `--sets` and `--modes` restrict what is repeated.
   For the manuscript, the curated questions are repeated in both modes and the database
   questions with retrieval:

   ```bash
   python paper/reproduce.py --llm lmstudio:google/gemma-3-12b --tag gemma-3-12b --run 2
   python paper/reproduce.py --llm lmstudio:google/gemma-3-12b --tag gemma-3-12b --run 3 \
       --sets seed_questions
   ```

   `tables/answers.md` reports the mean over complete runs with the range in parentheses
   where the runs differ, and lists how many complete runs enter each cell.

Hardware used for the manuscript: the models were served by LM Studio, each on one NVIDIA
RTX PRO 6000 Blackwell GPU (96 GB of VRAM): Llama-3.1-8B and Gemma-3-12B on one workstation
(reached through an SSH tunnel), gpt-oss-20b on a second workstation with the same GPU
(reached directly on the local network, `LMSTUDIO_HOST=http://<host>:1234`). The knowledge
base, retrieval and grading ran on the CPU of a desktop computer that queried the servers.
Times per answer in `tables/answers.md` are wall-clock times measured at that client.

Knowledge-base snapshot: `python paper/scripts/make_kb_snapshot.py` bundles the converted
records (`data/raw/*.jsonl`) and both indexed knowledge bases into
`dist/LLMatter2D-kb-<date>.zip`, with a README listing sources, licences (MC2D and
JARVIS-DFT data are CC BY 4.0), citations and SHA-256 checksums. The zip is attached to the
GitHub releases; unzip it in the repository root to reproduce the benchmark without
downloading the databases again.

Grading: numerical answers are graded on the first value with the expected unit
(strict; for a range such as `20-40 meV`, the bound that carries the unit); the lenient
criterion, any value within tolerance, is also reported because closed-book answers often
list several candidate values. The parser reads LaTeX (also bold, `\mathbf{3.73~\text{\AA}}`)
and decimal commas; keywords are matched after Unicode normalization (non-breaking hyphens);
refusals are detected with straight or typographic apostrophes. To re-apply the grader to
stored answers after a change: `llmatter2d regrade <results.jsonl> <questions.jsonl>`.

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
* Table IV (2026-11-05): all three models complete (25 + 300 questions, RAG and closed
  book). gpt-oss-20b has three runs on the curated questions and two retrieval-augmented runs
  on the database questions (second workstation); its interrupted pilot on the first
  workstation (39 answers) is kept in `results/quick/seed_gpt-oss-20b_pilot_ws1.*` and is not
  part of the table. Still to do: runs 2 and 3 of Llama-3.1-8B and Gemma-3-12B on the curated
  questions and run 2 with retrieval on the database questions, e.g.
  `python paper/reproduce.py --llm lmstudio:meta-llama-3.1-8b-instruct --tag llama-3.1-8b --run 2 --sets seed_questions`.
