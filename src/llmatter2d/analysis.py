"""Analyse a new (user-supplied) 2D structure against the knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .elements import COVALENT_RADII
from .lattices.archimedean import ARCHIMEDEAN
from .lattices.tightbinding import analyze_bands
from .rag.knowledge_base import KnowledgeBase
from .rag.pipeline import Answer, RAGPipeline
from .schema import Structure2D


@dataclass
class StructureReport:
    formula: str
    bravais: str
    descriptors: dict[str, float]
    coordination: dict[str, list[int]]
    archimedean_matches: list[str]
    neighbors: list[tuple[str, float]] = field(default_factory=list)
    answer: Answer | None = None

    def as_text(self) -> str:
        lines = [f"Formula: {self.formula}", f"Bravais lattice: {self.bravais}"]
        lines += [f"{k}: {v:.3f}" for k, v in self.descriptors.items()]
        lines.append("Coordination (covalent-radius criterion): " + "; ".join(
            f"{el}: {sorted(set(z))}" for el, z in self.coordination.items()))
        if self.archimedean_matches:
            lines.append("Projected network consistent with Archimedean tiling(s): "
                         + ", ".join(self.archimedean_matches))
            for name in self.archimedean_matches:
                lines.append("  " + analyze_bands(ARCHIMEDEAN[name]).summary())
        else:
            lines.append("Projected network matches no Archimedean tiling")
        if self.neighbors:
            lines.append("Structurally closest known materials: " + ", ".join(
                f"{u} ({s:.2f})" for u, s in self.neighbors))
        return "\n".join(lines)


def _periodic_neighbors(cell2: np.ndarray, pos: np.ndarray, cutoff_fn) -> list[list[int]]:
    out = []
    for i, ri in enumerate(pos):
        nb = []
        for j, rj in enumerate(pos):
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    if i == j and n1 == 0 and n2 == 0:
                        continue
                    d = np.linalg.norm(rj + n1 * cell2[0] + n2 * cell2[1] - ri)
                    if cutoff_fn(i, j, d):
                        nb.append(j)
        out.append(nb)
    return out


def coordination(structure: Structure2D, tol: float = 1.2) -> dict[str, list[int]]:
    sp = structure.species
    pos = structure.positions

    def bonded(i: int, j: int, d: float) -> bool:
        r = COVALENT_RADII.get(sp[i], 1.5) + COVALENT_RADII.get(sp[j], 1.5)
        return d < tol * r

    nbs = _periodic_neighbors(structure.cell[:2], pos, bonded)
    out: dict[str, list[int]] = {}
    for el, nb in zip(sp, nbs):
        out.setdefault(el, []).append(len(nb))
    return out


def archimedean_match(structure: Structure2D, tol: float = 0.03) -> list[str]:
    """Tilings consistent with the in-plane projection of the atomic network.

    Atoms are projected on the plane, coincident projections merged, and the
    nearest-neighbour graph compared with each tiling through the vertex degree
    and the scale-free site density n*d^2/area (supercell independent).
    """
    cell = structure.cell
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    proj = structure.positions - np.outer(structure.positions @ normal, normal)
    pts: list[np.ndarray] = []
    for p in proj:
        if all(np.linalg.norm(p - q) > 0.1 for q in pts):
            pts.append(p)
    pts_arr = np.array(pts)
    dists = [np.linalg.norm(pj + n1 * cell[0] + n2 * cell[1] - pi)
             for i, pi in enumerate(pts_arr) for j, pj in enumerate(pts_arr)
             for n1 in (-1, 0, 1) for n2 in (-1, 0, 1) if not (i == j and n1 == 0 and n2 == 0)]
    dmin = min(dists)
    nbs = _periodic_neighbors(cell[:2], pts_arr, lambda i, j, d: d < dmin * (1 + tol))
    degrees = {len(nb) for nb in nbs}
    if len(degrees) != 1:
        return []
    z = degrees.pop()
    rho = len(pts_arr) * dmin**2 / structure.area
    return [name for name, lat in ARCHIMEDEAN.items()
            if int(lat.coordination()[0]) == z
            and abs(rho - lat.n_sites / lat.area) / (lat.n_sites / lat.area) < tol]


def describe_structure(structure: Structure2D, kb: KnowledgeBase | None = None,
                       k: int = 5) -> StructureReport:
    return StructureReport(
        formula=structure.formula,
        bravais=structure.bravais_lattice(),
        descriptors={"a (A)": structure.a, "b (A)": structure.b, "gamma (deg)": structure.gamma,
                     "atoms per cell": float(len(structure.sites)),
                     "thickness (A)": structure.thickness},
        coordination=coordination(structure),
        archimedean_matches=archimedean_match(structure),
        neighbors=kb.structural_neighbors(structure, k) if kb is not None else [],
    )


def analyze_structure(structure: Structure2D, pipeline: RAGPipeline,
                      question: str | None = None) -> StructureReport:
    """Descriptors + nearest known materials + LLM report (task ``analyze``)."""
    report = describe_structure(structure, pipeline.kb)
    extra = "<analysis>\n" + report.as_text() + "\n</analysis>"
    neighbor_formulas = " ".join(pipeline.kb.get(u).formula for u, _ in report.neighbors[:3])
    query = (f"{structure.formula} {' '.join(structure.elements)} {report.bravais} "
             f"{' '.join(ARCHIMEDEAN[m].alias for m in report.archimedean_matches)} "
             f"{neighbor_formulas}")
    q = question or f"Analyse this new 2D material ({structure.formula})."
    report.answer = pipeline.ask(q, task="analyze", extra=extra, retrieval_query=query)
    return report
