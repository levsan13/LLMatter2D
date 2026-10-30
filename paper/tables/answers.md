| Model | Curated: RAG | Curated: closed book | Database: RAG | Database: closed book | Citations valid / precise (database) | Closed-book refusals | s per answer (RAG / closed) |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 100.0 | 48.0 | 99.7 | 0.7 | 100.0 / 99.8 | 294/325 | 1.4 / 0.7 |
| Gemma-3-12B | 100.0 | 28.0 | 100.0 | 4.3 | 100.0 / 97.3 | 6/325 | 2.0 / 22.7 |
| gpt-oss-20b | 100.0 | 100.0* (n=14) | - | - | - | 0/14 | - / - |

Accuracy in %. Curated: 25 questions; database: 300 questions generated from MC2D and JARVIS-DFT records. Retrieval: lexical embedder with exact-match bonus, k = 8, numbered citations; temperature 0.2. * = run not complete (n answers).
