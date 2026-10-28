# Architecture

## Data model

Everything is a `Material2D` (`src/llmatter2d/schema.py`):

```
Material2D
├── uid, formula, name, tags, notes
├── structure: Structure2D (lattice a1 a2 a3, sites, prototype)
├── electronic: band_gaps[ BandGap(value, method, direct, location, with_soc) ],
│               magnetic state/moment/Tc, Dirac cone, SOC gap, topology, effective masses
├── stability: formation energy, E_hull, exfoliation/binding energy, phonons, synthesized?
├── tight_binding[ TightBindingModel(orbitals, hoppings H_ij(R), source, E_F) ]
├── arpes[ ARPESFeatures(v_F, m*, gap, SOC splitting, hv, T, polarisation, substrate) ]
└── provenance[ Provenance(source, id, method, reference, url, date) ]
```

Design rules:

* A number is never stored without its method (PBE, HSE06, G0W0, optB88vdW, TBmBJ,
  ARPES, optical...). The LLM is told how these differ.
* Each record keeps the identifier of its source database, so answers can be traced.

## Retrieval

1. **Cards**: each material becomes a short text card (`rag/documents.py`) with all its
   numbers, methods and references, plus flat metadata (elements, gap range, magnetism,
   Bravais lattice, tags, sources, ARPES/TB availability).
2. **Filters first**: `Filters` applies hard constraints on metadata (e.g. gap between 1 and
   2 eV). Embeddings never decide numerical questions.
3. **Embeddings**: `hashing` (lexical, no dependencies) or `st:<model>`
   (sentence-transformers, e.g. `BAAI/bge-small-en-v1.5`). Exact formula or name matches in
   the query get a score boost so that "MoS2" retrieves MoS2.
4. **Structural neighbours**: for structure-based questions, `KnowledgeBase.structural_neighbors`
   ranks known materials by element overlap and geometric descriptors.

## Generation

`RAGPipeline.ask` builds:

```
<context>
<record n="1" id="c2db:..." score="...">card</record>
...
</context>
[<analysis> or <candidates> block for the analyze / predict tasks]
<question>...</question>
```

with a frozen, task-specific system prompt (`llm/prompts.py`). Backends:

* `lmstudio[:<model>]` (default) - a pre-trained model served locally by LM Studio through
  its OpenAI-compatible `/v1/chat/completions` endpoint. Without a model id the first loaded
  chat model is used (embedding models are skipped). Temperature 0.2 for factual answers;
  `<think>` blocks of reasoning models are stripped; truncation (`finish_reason: length`) is
  flagged in the answer. Configured by `LMSTUDIO_HOST`, `LMSTUDIO_MODEL`, `LMSTUDIO_API_KEY`.
* `ollama:<model>` - any local model served by Ollama, e.g. to compare pre-trained models on
  identical context.
* `claude[:<model>]` - optional hosted backend (Anthropic SDK, `claude-opus-5-5`, streaming,
  adaptive thinking, server-side refusal fallbacks).
* `offline` - returns the retrieved records; used in tests and to inspect prompts.

The default backend can be changed with `LLMATTER2D_LLM`. Because the base model is
pre-trained and not fine-tuned, all domain knowledge that must be exact comes from the
retrieved records; the prompts tell the model to cite them and to flag anything else as
general knowledge.

## Discovery loop (Archimedean lattices)

```
11 tilings x elements (and A/B pairs on bipartite tilings)
   -> bond length from covalent radii (first guess)
   -> nearest-neighbour tight-binding: flat bands, Dirac crossings, quadratic touchings
   -> heuristic score: coordination plausibility + electronic features + novelty vs KB
   -> RAG review by the LLM (analogues, risks, validation plan)
   -> POSCAR export -> DFT relaxation, phonons, convex hull, bands with SOC
   -> validated results ingested back as Material2D (source "LLMatter2D-DFT")
```

The tiling generator is exact: every tiling has unit bonds, the right coordination and the
site density implied by its vertex configuration (checked in `tests/test_lattices.py`). The
tight-binding layer reproduces textbook results (Dirac cone of the honeycomb at K, kagome flat
band at +2t, two flat bands of the star lattice 3.12^2).

## Analysis of a new material

`analysis.analyze_structure` computes coordination numbers (covalent-radius criterion),
projects the atomic network on the plane and compares its vertex degree and scale-free site
density with the 11 tilings, finds structural neighbours in the knowledge base, attaches the
tight-binding features of the matched tiling, and asks the LLM for a report (task `analyze`).
