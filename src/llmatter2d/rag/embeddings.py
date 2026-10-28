"""Text embedders.

* ``hashing``        - dependency-free lexical embedder (feature hashing of words and
                       character n-grams). Deterministic; used in tests and offline.
* ``st:<model>``     - any sentence-transformers model, default BAAI/bge-small-en-v1.5.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_ST_MODEL = "BAAI/bge-small-en-v1.5"
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-\^\.]*|\d+(?:\.\d+)?")


class Embedder(Protocol):
    spec: str

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalised
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
        self.spec = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        feats = []
        for tok in _TOKEN.findall(text):
            low = tok.lower().strip(".")
            feats.append("w:" + low)
            if tok[0].isupper() and any(c.isdigit() for c in tok):
                feats.append("f:" + tok)  # chemical formula, case-sensitive (MoS2 vs mos2)
            padded = f"#{low}#"
            feats += ["c:" + padded[i:i + self.ngram] for i in range(len(padded) - self.ngram + 1)]
        return feats

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for f in self._features(text):
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            weight = 1.0 if f[0] != "c" else 0.3
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(t) for t in texts]).reshape(len(texts), self.dim)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_ST_MODEL):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model
        self.spec = f"st:{model}"
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True,
                                             show_progress_bar=False))


def get_embedder(spec: str = "hashing") -> Embedder:
    if spec.startswith("hashing"):
        _, _, dim = spec.partition(":")
        return HashingEmbedder(int(dim) if dim else 2048)
    if spec.startswith("st"):
        _, _, model = spec.partition(":")
        return SentenceTransformerEmbedder(model or DEFAULT_ST_MODEL)
    raise ValueError(f"unknown embedder {spec!r} (use 'hashing' or 'st:<model>')")
