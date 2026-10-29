"""The LLMatter2D knowledge base: materials + vector index + structured filters.

Retrieval is hybrid on purpose. Numerical questions ("semiconductors with a gap
between 1 and 2 eV") are answered by metadata filters, never by embedding
similarity; the embedding ranks what survives the filters, and exact formula
matches get a boost.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema import Material2D, Structure2D, load_materials, save_materials
from .documents import material_metadata, material_to_card
from .embeddings import Embedder, get_embedder

FORMULA_BOOST = 0.25


@dataclass
class Filters:
    elements_all: list[str] = field(default_factory=list)  # must contain all of these
    elements_only: list[str] | None = None  # may contain only these
    gap_min: float | None = None
    gap_max: float | None = None
    magnetic: bool | None = None
    lattice: str | None = None  # bravais lattice name
    tags_any: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    require_arpes: bool = False

    def matches(self, meta: dict) -> bool:
        els = set(meta["elements"])
        if self.elements_all and not set(self.elements_all) <= els:
            return False
        if self.elements_only is not None and not els <= set(self.elements_only):
            return False
        if self.gap_min is not None and (meta["gap_max"] is None or meta["gap_max"] < self.gap_min):
            return False
        if self.gap_max is not None and (meta["gap_min"] is None or meta["gap_min"] > self.gap_max):
            return False
        if self.magnetic is not None and meta["magnetic"] != self.magnetic:
            return False
        if self.lattice and meta["lattice"] != self.lattice:
            return False
        if self.tags_any and not any(
                t.lower() in tag for t in self.tags_any for tag in meta["tags"]):
            return False
        if self.sources and not set(self.sources) & set(meta["sources"]):
            return False
        if self.require_arpes and not meta["has_arpes"]:
            return False
        return True


@dataclass
class Hit:
    material: Material2D
    score: float
    text: str

    @property
    def uid(self) -> str:
        return self.material.uid


class KnowledgeBase:
    def __init__(self, embedder: Embedder | None = None, formula_boost: float = FORMULA_BOOST):
        self.embedder = embedder or get_embedder("hashing")
        self.formula_boost = formula_boost  # 0 disables the exact formula/name match bonus
        self.materials: dict[str, Material2D] = {}
        self._uids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.materials)

    # ------------------------------------------------------------------ building
    def add(self, materials: Iterable[Material2D]) -> int:
        n = 0
        for m in materials:
            self.materials[m.uid] = self.materials[m.uid].merge(m) if m.uid in self.materials else m
            n += 1
        self._vectors = None
        return n

    def build(self, batch_size: int = 256) -> None:
        self._uids = list(self.materials)
        self._texts = [material_to_card(self.materials[u]) for u in self._uids]
        self._meta = [material_metadata(self.materials[u]) for u in self._uids]
        chunks = [self.embedder.embed(self._texts[i:i + batch_size])
                  for i in range(0, len(self._texts), batch_size)]
        self._vectors = np.vstack(chunks) if chunks else np.zeros((0, 1))

    def _ensure_built(self) -> None:
        if self._vectors is None or len(self._uids) != len(self.materials):
            self.build()

    # ------------------------------------------------------------------ querying
    def get(self, uid: str) -> Material2D:
        return self.materials[uid]

    def search(self, query: str, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        self._ensure_built()
        if not self._uids:
            return []
        mask = np.array([filters.matches(m) if filters else True for m in self._meta])
        if not mask.any():
            return []
        q = self.embedder.embed([query])[0]
        scores = self._vectors @ q
        formulas = set(re.findall(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b", query))
        names = {w.lower() for w in re.findall(r"[A-Za-z\-]{4,}", query)}
        for i, meta in enumerate(self._meta if self.formula_boost else []):
            mat = self.materials[self._uids[i]]
            if meta["formula"] in formulas or (mat.name and mat.name.split(" (")[0].lower() in names):
                scores[i] += self.formula_boost
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [Hit(self.materials[self._uids[i]], float(scores[i]), self._texts[i])
                for i in order if np.isfinite(scores[i])]

    def structural_neighbors(self, structure: Structure2D, k: int = 5) -> list[tuple[str, float]]:
        """Known materials closest in chemistry and geometry to ``structure``."""
        target_els = set(structure.elements)
        d0 = _descriptor(structure)
        out = []
        for uid, m in self.materials.items():
            if m.structure is None:
                continue
            els = set(m.structure.elements)
            jac = len(target_els & els) / len(target_els | els)
            geo = float(np.exp(-np.linalg.norm(_descriptor(m.structure) - d0)))
            out.append((uid, 0.5 * jac + 0.5 * geo))
        return sorted(out, key=lambda x: -x[1])[:k]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        self._ensure_built()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_materials(d / "materials.jsonl", (self.materials[u] for u in self._uids))
        np.save(d / "vectors.npy", self._vectors)
        (d / "config.json").write_text(json.dumps(
            {"embedder": self.embedder.spec, "uids": self._uids}, indent=1))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> KnowledgeBase:
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text())
        kb = cls(embedder or get_embedder(cfg["embedder"]))
        kb.add(load_materials(d / "materials.jsonl"))
        kb._uids = cfg["uids"]
        kb._texts = [material_to_card(kb.materials[u]) for u in kb._uids]
        kb._meta = [material_metadata(kb.materials[u]) for u in kb._uids]
        kb._vectors = np.load(d / "vectors.npy")
        return kb

    @classmethod
    def from_seed(cls, embedder: Embedder | None = None) -> KnowledgeBase:
        from ..ingest.seed import SeedConnector

        kb = cls(embedder)
        kb.add(SeedConnector().fetch())
        kb.build()
        return kb


def _descriptor(s: Structure2D) -> np.ndarray:
    return np.array([s.a / 5.0, s.b / 5.0, s.gamma / 90.0, len(s.sites) / 6.0,
                     s.thickness / 3.0])
