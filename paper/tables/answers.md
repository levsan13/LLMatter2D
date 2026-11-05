| Model | Curated: RAG | Curated: closed book | Database: RAG | Database: closed book | Citations valid / precise (database) | Closed-book refusals (database) | s per answer (RAG / closed) |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 98.7 (96.0-100.0) | 54.7 (48.0-64.0) | 99.7 | 0.7 | 100.0 / 99.6 | 294/300 | 1.0 / 0.7 |
| Gemma-3-12B | 100.0 | 30.7 (28.0-36.0) | 100.0 | 4.3 | 100.0 / 97.4 | 8/300 | 1.4 / 22.7 |
| gpt-oss-20b | 100.0 | 90.7 (88.0-96.0) | 100.0 | 7.0 | 100.0 / 100.0 | 0/300 | 0.5 / 0.5 |

Accuracy in %: mean over complete runs, range over runs in parentheses when the runs differ. Curated: 25 questions; database: 300 questions generated from MC2D and JARVIS-DFT records. Retrieval: lexical embedder with exact-match bonus, k = 8, numbered citations; temperature 0.2. Models served by LM Studio on one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). * = run not complete (n answers).

Complete runs per cell - Llama-3.1-8B-Instruct: curated rag 3, closed 3; database rag 2, closed 1 | Gemma-3-12B: curated rag 3, closed 3; database rag 2, closed 1 | gpt-oss-20b: curated rag 3, closed 3; database rag 2, closed 1.
