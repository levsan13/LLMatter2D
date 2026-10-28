"""Data model for the LLMatter2D knowledge base.

Every record that enters the RAG stack is a :class:`Material2D`. Numbers are
always stored together with the method that produced them (DFT functional,
ARPES, tight-binding fit, ...) and a :class:`Provenance` entry, so that the LLM
can quote values with their origin instead of inventing them.

Units: lengths in angstrom, energies in eV, velocities in m/s, masses in units
of the free-electron mass.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import reduce
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator


class Provenance(BaseModel):
    """Where a piece of information came from."""

    source: str  # e.g. "C2DB", "MC2D", "JARVIS-DFT", "2DMatPedia", "seed", "user"
    source_id: str | None = None
    method: str | None = None  # e.g. "DFT-PBE", "DFT-HSE06", "G0W0", "ARPES"
    reference: str | None = None  # DOI, arXiv id or citation
    url: str | None = None
    retrieved: str | None = None  # ISO date


class Site(BaseModel):
    species: str
    position: tuple[float, float, float]  # Cartesian coordinates (angstrom)


class Structure2D(BaseModel):
    """A periodic slab. ``lattice`` rows are a1, a2 (in-plane) and a3 (vacuum)."""

    lattice: list[list[float]]
    sites: list[Site]
    pbc: tuple[bool, bool, bool] = (True, True, False)
    layer_group: str | None = None
    prototype: str | None = None  # e.g. "honeycomb", "1H-MX2", "archimedean:3.6.3.6"

    @model_validator(mode="after")
    def _check_lattice(self) -> Structure2D:
        if np.asarray(self.lattice).shape != (3, 3):
            raise ValueError("lattice must be a 3x3 matrix (rows a1, a2, a3)")
        if not self.sites:
            raise ValueError("a structure needs at least one site")
        return self

    # --- geometry -----------------------------------------------------------------
    @property
    def cell(self) -> np.ndarray:
        return np.asarray(self.lattice, dtype=float)

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([s.position for s in self.sites], dtype=float)

    @property
    def species(self) -> list[str]:
        return [s.species for s in self.sites]

    @property
    def a(self) -> float:
        return float(np.linalg.norm(self.cell[0]))

    @property
    def b(self) -> float:
        return float(np.linalg.norm(self.cell[1]))

    @property
    def gamma(self) -> float:
        """Angle between a1 and a2 in degrees."""
        a1, a2 = self.cell[0], self.cell[1]
        c = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    @property
    def area(self) -> float:
        return float(np.linalg.norm(np.cross(self.cell[0], self.cell[1])))

    @property
    def thickness(self) -> float:
        """Vertical extent of the layer (distance between outermost atoms)."""
        z = self.positions @ _unit_normal(self.cell)
        return float(z.max() - z.min())

    @property
    def elements(self) -> list[str]:
        return sorted(set(self.species))

    @property
    def formula(self) -> str:
        return reduced_formula(self.species)

    def bravais_lattice(self, tol: float = 1e-2) -> str:
        """Classify the 2D Bravais lattice from a, b and gamma."""
        a, b, g = self.a, self.b, self.gamma
        same = abs(a - b) / max(a, b) < tol
        if same and (abs(g - 60) < 1 or abs(g - 120) < 1):
            return "hexagonal"
        if abs(g - 90) < 1:
            return "square" if same else "rectangular"
        if same:
            return "centered-rectangular"
        return "oblique"

    def describe(self) -> str:
        return (
            f"{self.bravais_lattice()} lattice, a={self.a:.3f} A, b={self.b:.3f} A, "
            f"gamma={self.gamma:.1f} deg, {len(self.sites)} atoms/cell, "
            f"thickness={self.thickness:.2f} A"
        )


class BandGap(BaseModel):
    value_eV: float
    method: str = "DFT-PBE"
    direct: bool | None = None
    location: str | None = None  # e.g. "K-K", "Gamma"
    with_soc: bool | None = None


class HoppingTerm(BaseModel):
    """<i, 0| H | j, R> in eV."""

    R: tuple[int, int, int]
    i: int
    j: int
    t_re: float
    t_im: float = 0.0

    @property
    def value(self) -> complex:
        return complex(self.t_re, self.t_im)


class TightBindingModel(BaseModel):
    orbitals: list[str]  # e.g. ["C1:pz", "C2:pz"]
    hoppings: list[HoppingTerm]
    source: str  # "Wannier90", "literature", "fit to ARPES", "Archimedean NN model"
    orbital_positions: list[tuple[float, float, float]] | None = None  # fractional
    fermi_level_eV: float | None = None
    with_soc: bool = False
    notes: str | None = None

    def onsite_energies(self) -> list[float]:
        e = [0.0] * len(self.orbitals)
        for h in self.hoppings:
            if h.R == (0, 0, 0) and h.i == h.j:
                e[h.i] = h.t_re
        return e

    def largest_hoppings(self, n: int = 5) -> list[HoppingTerm]:
        off = [h for h in self.hoppings if not (h.R == (0, 0, 0) and h.i == h.j)]
        return sorted(off, key=lambda h: abs(h.value), reverse=True)[:n]

    def summary(self) -> str:
        parts = [f"{len(self.orbitals)} orbitals ({', '.join(self.orbitals[:6])}"
                 + (", ..." if len(self.orbitals) > 6 else "") + f"), source: {self.source}"]
        top = self.largest_hoppings(3)
        if top:
            parts.append("largest hoppings: " + "; ".join(
                f"t[{h.i}->{h.j}, R={h.R}]={abs(h.value):.3f} eV" for h in top))
        if self.notes:
            parts.append(self.notes)
        return "; ".join(parts)


class ARPESFeatures(BaseModel):
    """Quantities extracted from (or reported for) an ARPES measurement."""

    fermi_velocity_m_s: float | None = None
    effective_mass_me: float | None = None
    gap_eV: float | None = None
    soc_splitting_eV: float | None = None  # e.g. valence-band splitting at K in TMDs
    photon_energy_eV: float | None = None
    temperature_K: float | None = None
    polarization: str | None = None
    substrate: str | None = None
    k_point: str | None = None
    notes: str | None = None
    provenance: Provenance | None = None


class ElectronicProperties(BaseModel):
    band_gaps: list[BandGap] = Field(default_factory=list)
    is_metal: bool | None = None
    magnetic_state: str | None = None  # "NM", "FM", "AFM", "FiM"
    magnetic_moment_muB: float | None = None  # per unit cell
    curie_temperature_K: float | None = None
    dirac_cone: bool | None = None
    soc_gap_eV: float | None = None
    topology: str | None = None  # "trivial", "QSH", "Chern", ...
    work_function_eV: float | None = None
    effective_masses: dict[str, float] = Field(default_factory=dict)

    def gap(self, method: str | None = None) -> float | None:
        gaps = [g for g in self.band_gaps if method is None or g.method == method]
        return gaps[0].value_eV if gaps else None


class Stability(BaseModel):
    formation_energy_eV_atom: float | None = None
    energy_above_hull_eV_atom: float | None = None
    exfoliation_energy_meV_atom: float | None = None
    binding_energy_meV_A2: float | None = None
    dynamically_stable: bool | None = None
    experimentally_realized: bool | None = None


class Material2D(BaseModel):
    uid: str  # globally unique, "<source>:<id>"
    formula: str
    name: str | None = None
    structure: Structure2D | None = None
    electronic: ElectronicProperties = Field(default_factory=ElectronicProperties)
    stability: Stability = Field(default_factory=Stability)
    tight_binding: list[TightBindingModel] = Field(default_factory=list)
    arpes: list[ARPESFeatures] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    notes: str | None = None

    @property
    def elements(self) -> list[str]:
        if self.structure is not None:
            return self.structure.elements
        return sorted(parse_formula(self.formula))

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.formula})" if self.name else self.formula

    def merge(self, other: Material2D) -> Material2D:
        """Combine two records of the same material coming from different sources.

        Lists are concatenated, missing scalar fields are filled from ``other``.
        """
        data = self.model_dump()
        for key, value in other.model_dump().items():
            data[key] = _merge_values(data.get(key), value)
        data["uid"] = self.uid
        return Material2D.model_validate(data)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _merge_values(a, b):
    if a is None:
        return b
    if isinstance(a, list) and isinstance(b, list):
        out = list(a)
        for item in b:
            if item not in out:
                out.append(item)
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge_values(out.get(k), v)
        return out
    return a


def _unit_normal(cell: np.ndarray) -> np.ndarray:
    n = np.cross(cell[0], cell[1])
    return n / np.linalg.norm(n)


def reduced_formula(species: Iterable[str]) -> str:
    """Reduced formula, elements in order of first appearance (MoS2, WSe2, BN...)."""
    species = list(species)
    counts = Counter(species)
    order = list(dict.fromkeys(species))
    g = reduce(math.gcd, counts.values())
    return "".join(f"{el}{counts[el] // g if counts[el] // g > 1 else ''}" for el in order)


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple formula such as ``MoS2`` or ``Fe3GeTe2`` (no parentheses)."""
    import re

    out: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def save_materials(path: str | Path, materials: Iterable[Material2D]) -> int:
    """Write materials as JSON lines. Returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for m in materials:
            fh.write(m.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def load_materials(path: str | Path) -> Iterator[Material2D]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Material2D.model_validate(json.loads(line))
