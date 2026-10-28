# Prior art and name check

Search date: 2026-10-28 (web search and GitHub repository search).

## Is the name "LLMatter2D" taken?

* No paper, package or repository named **LLMatter2D** was found.
* GitHub has one repository called **LLMatter** (`davideticchiarelli01/LLMatter`, created
  April 2026). It is unrelated: a Unity/VR plugin that uses multimodal LLMs (via Ollama) to
  assign physical properties to objects in virtual reality.
* Similar names in materials AI: **LLaMat** (materials-science LLaMA models),
  **LLMatDesign** (LLM-driven materials design) and **MatterChat** (multimodal LLM for
  materials). None of them uses the name LLMatter2D.

The name is therefore available. Keeping the "2D" suffix avoids confusion with the
unrelated "LLMatter" repository.

## Related work

| Work | What it does | Difference from LLMatter2D |
|---|---|---|
| [LLaMP](https://github.com/chiang-yuan/llamp) ([arXiv:2401.17244](https://arxiv.org/abs/2401.17244)) | ReAct agents that query the Materials Project to ground LLM answers | Bulk materials, Materials Project only; no 2D-specific data, ARPES or tight-binding |
| LLaMat ([arXiv:2412.09560](https://arxiv.org/abs/2412.09560)) | LLaMA models continually pre-trained on materials text and crystal data | A model, not a retrieval system; could be plugged in as the LLM through Ollama |
| [MatterChat](https://arxiv.org/abs/2502.13107) | Multimodal LLM aligning atomic structures with text | Learns structure embeddings; no curated 2D database or discovery loop |
| LLMatDesign | Autonomous materials design with LLMs | General materials; LLMatter2D focuses on 2D lattices and electronic structure |
| [arXiv:2511.20691](https://arxiv.org/abs/2511.20691) | LLM extraction and management of 2D-materials data from the literature | Complementary: literature-derived data could feed the LLMatter2D knowledge base |
| [Review, arXiv:2508.06691](https://arxiv.org/abs/2508.06691) | Systematic review of LLMs and RAG for crystalline materials discovery | Describes database-grounded RAG as a general pattern |
| [arXiv:1908.05092](https://arxiv.org/abs/1908.05092) | Tight-binding study of all Archimedean lattices: flat bands, Dirac fermions, QSH phase, checked with DFT on carbon allotropes | Physics basis of the LLMatter2D lattice generator |

No published work combines a RAG layer over the 2D DFT databases (C2DB, MC2D,
2DMatPedia, JARVIS-2D) with ARPES observables, Wannier tight-binding models and an
Archimedean-lattice generator. That combination is the niche of LLMatter2D.

## Data sources referenced

* C2DB: Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al., 2D Mater. 8, 044002 (2021).
* MC2D: Mounet et al., Nat. Nanotechnol. 13, 246 (2018); OPTIMADE endpoint
  `https://optimade.materialscloud.org/main/mc2d` (2683 structures on 2026-10-28).
* 2DMatPedia: Zhou et al., Sci. Data 6, 86 (2019).
* JARVIS-DFT: Choudhary et al., npj Comput. Mater. 6, 173 (2020).
