# Roadmap

## v0.1 (this release)

- [x] `Material2D` data model with provenance and method-aware band gaps
- [x] Connectors: MC2D (OPTIMADE), C2DB (ASE db), 2DMatPedia (JSON), JARVIS-DFT 2D, seed set
- [x] Wannier90 `_hr.dat` reader/writer; ARPES feature extraction (v_F, m*)
- [x] Knowledge base with metadata filters, hashing / sentence-transformer embeddings, persistence
- [x] LLM backends: LM Studio (default, local pre-trained models), Ollama, Claude, offline;
      task prompts (qa, analyze, predict)
- [x] The 11 Archimedean lattices, NN tight-binding, flat-band and Dirac-crossing detection
- [x] Candidate generation, screening, LLM review and POSCAR export
- [x] CLI and test suite

## v0.2: data at scale

- [ ] Full ingestion of C2DB, MC2D, 2DMatPedia and JARVIS-2D, with structure matching to merge
      the same material across databases (e.g. via pymatgen `StructureMatcher`)
- [ ] Band structures and densities of states as compact descriptors (band edges, valley
      positions, effective masses) rather than only gaps
- [ ] Wannier tight-binding models for a first set of materials
- [ ] Curated ARPES table from the literature (DOI-referenced) and literature mining
- [ ] Vector store backend for >100k chunks (FAISS / Chroma) behind the same interface

## v0.3: evaluation

- [ ] Benchmark of questions with reference answers (gaps by method, magnetism, SOC splittings,
      stacking/twist questions), scored for correctness and citation faithfulness
- [ ] Comparison of LLMs (Claude, open-weights via Ollama, materials fine-tunes) on the same
      retrieved context
- [ ] Ablations: filters vs pure embeddings; card formats

## v0.4: discovery loop

- [ ] Beyond nearest neighbours: hopping from distance (Slater-Koster / Harrison scaling),
      spin-orbit coupling (Kane-Mele type) and Z2 invariants on the Archimedean lattices
- [ ] Decorations beyond one site per vertex (dimers, ligands, Janus layers)
- [ ] Pre-screening with machine-learned interatomic potentials (relaxation, phonons)
- [ ] Automated DFT (Quantum ESPRESSO / VASP via AiiDA or ASE) and re-ingestion of results
- [ ] Heterostructures and moiré bilayers (twist angle, lattice mismatch)
