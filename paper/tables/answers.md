| Model | Curated: RAG | Curated: closed book | Database: RAG | Database: closed book | Citations valid / precise (database) | Closed-book refusals (database) | s per answer (RAG / closed) |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 100.0 | 48.0 | 99.7 | 0.7 | 100.0 / 99.8 | 294/300 | 1.4 / 0.7 |
| Gemma-3-12B | 100.0 | 28.0 | 100.0 | 4.3 | 100.0 / 97.3 | 6/300 | 2.0 / 22.7 |
| gpt-oss-20b | 100.0 | 100.0* (n=14) | - | - | - | - | - / - |

Accuracy in %: mean over complete runs, range over runs in parentheses. Curated: 25 questions; database: 300 questions generated from MC2D and JARVIS-DFT records. Retrieval: lexical embedder with exact-match bonus, k = 8, numbered citations; temperature 0.2. Models served by LM Studio on one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). * = run not complete (n answers).

Complete runs per cell - Llama-3.1-8B-Instruct: curated rag 1, closed 1; database rag 1, closed 1 | Gemma-3-12B: curated rag 1, closed 1; database rag 1, closed 1 | gpt-oss-20b: curated rag 1, closed 0 (+1 partial).
