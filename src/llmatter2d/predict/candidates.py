"""Generate and screen hypothetical 2D materials on Archimedean lattices.

Pipeline:

1. *Enumerate*  - decorate each tiling with one element, or with an A/B pair when the
                  tiling is bipartite in its primitive cell; bond length from covalent radii.
2. *Physics*    - nearest-neighbour tight-binding features of the tiling (flat bands,
                  Dirac crossings, quadratic touchings).
3. *Screen*     - heuristic score: coordination plausibility + electronic interest +
                  novelty with respect to the knowledge base.
4. *Review*     - the LLM ranks the short list using retrieved analogues (RAG).
5. *Validate*   - POSCAR files are written for DFT relaxation/phonons; validated results
                  are ingested back into the knowledge base, closing the loop.

Every candidate is a hypothesis until DFT (and ideally experiment) says otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from ..elements import TYPICAL_COORDINATION, bond_length
from ..lattices.archimedean import ARCHIMEDEAN, Lattice2D, archimedean_lattice
from ..lattices.io import write_poscar
from ..lattices.tightbinding import BandFeatures, analyze_bands
from ..schema import ElectronicProperties, Material2D, Provenance, Structure2D, reduced_formula

# prototypes in the knowledge base that correspond to an Archimedean tiling
PROTOTYPE_TO_TILING = {"honeycomb": "6^3"}


@dataclass
class Candidate:
    lattice: Lattice2D
    species: list[str]  # one per basis site
    bond_length: float
    bands: BandFeatures
    score: float = 0.0
    rationale: list[str] = field(default_factory=list)
    known_as: str | None = None
    analogues: list[str] = field(default_factory=list)

    @property
    def formula(self) -> str:
        return reduced_formula(self.species)

    @property
    def elements(self) -> list[str]:
        return sorted(set(self.species))

    @property
    def uid(self) -> str:
        return f"candidate:{self.formula}@{self.lattice.name}"

    @property
    def structure(self) -> Structure2D:
        return self.lattice.to_structure(self.species, self.bond_length)

    def describe(self) -> str:
        z = int(self.lattice.coordination()[0])
        lines = [f"{self.uid}: {self.formula} on the {self.lattice.name} ({self.lattice.alias}) "
                 f"tiling, coordination {z}, bond length ~{self.bond_length:.2f} A "
                 f"(covalent radii), {self.lattice.n_sites} atoms/cell, score {self.score:.2f}",
                 f"  bands: {self.bands.summary()}"]
        if self.rationale:
            lines.append("  screening: " + "; ".join(self.rationale))
        if self.known_as:
            lines.append(f"  already in knowledge base as {self.known_as}")
        if self.analogues:
            lines.append("  analogues: " + ", ".join(self.analogues))
        return "\n".join(lines)

    def to_material(self) -> Material2D:
        b = self.bands
        return Material2D(
            uid=self.uid,
            formula=self.formula,
            structure=self.structure,
            electronic=ElectronicProperties(dirac_cone=bool(b.linear_crossings) or None),
            tags=["predicted", "hypothetical", f"archimedean:{self.lattice.name}",
                  self.lattice.alias] + (["flat band"] if b.flat_bands else []),
            provenance=[Provenance(source="LLMatter2D", method="Archimedean decoration + "
                                   "nearest-neighbour tight-binding (not DFT-validated)")],
            notes=b.summary() + (". " + "; ".join(self.rationale) if self.rationale else ""),
        )


def enumerate_candidates(elements: Iterable[str], lattices: Iterable[str] | None = None,
                         binary: bool = True) -> list[Candidate]:
    elements = list(dict.fromkeys(elements))
    lats = [archimedean_lattice(n) for n in lattices] if lattices else list(ARCHIMEDEAN.values())
    out: list[Candidate] = []
    for lat in lats:
        bands = analyze_bands(lat)  # t = 1; energies scale with the real hopping
        for el in elements:
            out.append(Candidate(lat, [el] * lat.n_sites, bond_length(el, el), bands))
        colors = lat.sublattice_coloring()
        if binary and colors is not None:
            for a, b in combinations(elements, 2):
                out.append(Candidate(lat, [(a, b)[c] for c in colors], bond_length(a, b), bands))
    return out


def _known_tilings(material: Material2D) -> set[str]:
    tilings = {t.split(":", 1)[1] for t in material.tags if t.lower().startswith("archimedean:")}
    if material.structure is not None:
        proto = material.structure.prototype or ""
        if proto.startswith("archimedean:"):
            tilings.add(proto.split(":", 1)[1])
        if proto in PROTOTYPE_TO_TILING:
            tilings.add(PROTOTYPE_TO_TILING[proto])
    return tilings


def score_candidates(candidates: list[Candidate], kb=None, n_analogues: int = 3) -> list[Candidate]:
    """Heuristic ranking. ``kb`` (a KnowledgeBase) adds novelty checks and analogues."""
    known: dict[tuple[frozenset[str], str], str] = {}
    if kb is not None:
        for m in kb.materials.values():
            for tiling in _known_tilings(m):
                known.setdefault((frozenset(m.elements), tiling), m.uid)

    for c in candidates:
        c.rationale = []
        z = int(c.lattice.coordination()[0])
        chem = 0.0
        for el in c.elements:
            ok = TYPICAL_COORDINATION.get(el)
            if ok is None:
                c.rationale.append(f"no coordination prior for {el}")
            elif z in ok:
                chem += 1.0
            else:
                chem -= 1.0
                c.rationale.append(f"{el} rarely {z}-fold coordinated (typical {ok})")
        chem /= len(c.elements)

        b = c.bands
        elec = (1.0 if b.flat_bands else 0.0) + (1.0 if b.linear_crossings else 0.0) \
            + (0.5 if b.quadratic_touchings else 0.0)
        if b.flat_bands:
            c.rationale.append("flat band(s): candidate for correlated / magnetic instabilities")
        if b.linear_crossings:
            c.rationale.append("Dirac-like crossings: candidate for high mobility / QSH with SOC")

        c.known_as = known.get((frozenset(c.elements), c.lattice.name))
        novelty = 0.0 if c.known_as else 0.5
        if c.known_as:
            c.rationale.append(f"known material ({c.known_as}) - useful as a benchmark")
        c.score = 2.0 * chem + elec + novelty

        if kb is not None:
            hits = kb.search(f"{c.formula} {' '.join(c.elements)} {c.lattice.alias} lattice "
                             f"{'flat band' if b.flat_bands else ''} "
                             f"{'Dirac cone' if b.linear_crossings else ''}", k=n_analogues)
            c.analogues = [h.uid for h in hits]
    return sorted(candidates, key=lambda c: -c.score)


def write_candidates(candidates: list[Candidate], directory: str | Path) -> Path:
    """Write POSCAR files and a JSON summary for DFT follow-up."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    summary = []
    for rank, c in enumerate(candidates, 1):
        name = f"{rank:03d}_{c.formula}_{c.lattice.alias}"
        write_poscar(c.structure, d / f"{name}.vasp", comment=f"{c.uid} (LLMatter2D candidate)")
        summary.append({"rank": rank, "uid": c.uid, "file": f"{name}.vasp",
                        "lattice": c.lattice.name, "alias": c.lattice.alias,
                        "formula": c.formula, "bond_length_A": round(c.bond_length, 3),
                        "score": round(c.score, 3), "bands": c.bands.summary(),
                        "rationale": c.rationale, "known_as": c.known_as,
                        "analogues": c.analogues})
    path = d / "candidates.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def review_with_llm(candidates: list[Candidate], pipeline, top: int = 10,
                    question: str | None = None):
    """Ask the RAG pipeline (task ``predict``) to rank and critique the short list."""
    shortlist = candidates[:top]
    block = "<candidates>\n" + "\n".join(c.describe() for c in shortlist) + "\n</candidates>"
    elements = sorted({el for c in shortlist for el in c.elements})
    query = (f"2D materials of {', '.join(elements)}; Archimedean lattices "
             + ", ".join(sorted({c.lattice.alias for c in shortlist})))
    q = question or ("Which of these candidates are the most promising new 2D materials, "
                     "and how should each be validated?")
    return pipeline.ask(q, task="predict", extra=block, retrieval_query=query)
