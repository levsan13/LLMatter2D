# Manuscript plan

Working title: *LLMatter2D: grounding pre-trained language models in first-principles
and ARPES data for the analysis and discovery of two-dimensional materials*

## Choice of journal

| Journal | Fit | What it would need |
|---|---|---|
| **Physical Review Materials** (APS) | Best APS fit for the method: its editors list first-principles prediction, machine learning and 2D materials among the journal's strengths | benchmark on the full database + DFT-validated candidates |
| **Digital Discovery** (RSC) | Data-first, reproducibility-oriented journal for digital tools in chemistry and materials; publishes LLM-for-materials work | same as above; strong emphasis on open data and code (already in place) |
| **npj Computational Materials** | Highest visibility for computational materials methods | clearly new materials validated by DFT (phonons, convex hull) and a convincing benchmark |
| **2D Materials** (IOP) | 2D community; C2DB was published there | a 2D-focused framing: database integration + discovery |
| **Physical Review B** | Condensed-matter physics; machine learning is not part of its stated scope | better for a companion *physics* paper: DFT + tight-binding of new Archimedean-lattice monolayers (flat bands, Dirac cones, SOC, topology) |
| **JOSS** / Computer Physics Communications | Software papers | JOSS asks for roughly three months of sustained development and a feature-complete package; submit later, as a complement |

Recommended route: a methods paper in Physical Review Materials or Digital Discovery,
then a PRB paper on the physics of the best Archimedean candidates once DFT confirms
them, then JOSS for the software.

## Structure

1. **Introduction**: LLMs in materials science; hallucination of numbers and mixing of
   functionals; scarcity of 2D-specific grounding; Archimedean lattices as templates.
2. **Methods**
   1. Data model and sources (C2DB, MC2D, 2DMatPedia, JARVIS-DFT, Wannier90, ARPES);
      method-aware values and provenance.
   2. Hybrid retrieval: metadata filters, embeddings, formula boost, structural neighbours.
   3. Pre-trained LLMs served locally (LM Studio); prompts; citation format.
   4. Archimedean generator: construction of the 11 tilings, decoration, NN tight-binding,
      detection of flat bands and band touchings (Fig. 2, Fig. 3, Table 1).
   5. Evaluation protocol: question set, unit-aware grading, citation metrics,
      RAG vs closed-book.
3. **Results**
   1. Archimedean lattices: band features (Table 1), benchmarks recovered (graphene,
      hBN, Xenes, graphenylene).
   2. Benchmark: accuracy and citation quality for several pre-trained models, with and
      without retrieval (Table 3); failure analysis.
   3. Case study: analysis of a monolayer absent from the knowledge base.
   4. Discovery: ranked candidates, DFT relaxation, phonons and formation energy for the
      top few; band structures with SOC compared with the tight-binding prediction.
4. **Discussion**: limitations (nearest-neighbour model, crude first-guess geometries,
   database biases, grading heuristics, LLM variability) and outlook (literature-mined
   ARPES data, MLIP pre-screening, automated DFT loop).
5. **Data and code availability**: code at https://github.com/levsan13/LLMatter2D
   (MIT), archived on Zenodo with a DOI; database versions and download dates; the
   benchmark questions and all per-question outputs are in the repository.

## Still to do before submission

- [x] Knowledge base from MC2D and JARVIS-2D (3799 records) with lexical and dense
      embedders; C2DB and 2DMatPedia still to add when their files are available
- [x] Benchmark of 300 questions generated from database records + 25 curated questions;
      retrieval results (Table III)
- [ ] Extend the database questions to magnetism and stability
- [x] Answers of Llama-3.1-8B and Gemma-3-12B in LM Studio, RAG vs closed-book (Table IV)
- [ ] Finish gpt-oss-20b (partial: curated RAG done, 14/25 curated closed-book, database
      questions not run); repeat runs to estimate the run-to-run spread
- [ ] DFT validation of the top Archimedean candidates (relaxation, phonons, E_hull)
- [x] Fig. 1 drawn in TikZ (check it renders as intended on Overleaf)
- [x] Zenodo archive and DOI (concept DOI 10.5281/zenodo.23779294)
- [ ] Attach a snapshot of the knowledge base used in the paper (data/raw + kb, with the
      MC2D and JARVIS-DFT attributions and licences) to the next release
- [ ] Add the author's ORCID to CITATION.cff and .zenodo.json
