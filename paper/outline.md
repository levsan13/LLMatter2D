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

- [ ] Ingest the full databases and rebuild the knowledge base with a dense embedder
- [ ] Extend the benchmark to a few hundred questions generated from database records
      (gaps by method, magnetism, stability) plus the ARPES and tight-binding questions
- [ ] Run Table 3 with at least three pre-trained models in LM Studio; fix seeds,
      temperature, quantisation and context length; repeat runs to estimate variance
- [ ] DFT validation of the top Archimedean candidates (relaxation, phonons, E_hull)
- [ ] Redraw Fig. 1 for the journal
- [ ] Enable Zenodo, create the `v0.1.0` release and add the DOI to README and CITATION.cff
- [ ] Add the author's ORCID to CITATION.cff and .zenodo.json
