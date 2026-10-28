"""Curated seed set of well-known 2D materials shipped with the package.

The seed file stores compact *prototype* descriptions (honeycomb, 1H/1T-MX2,
puckered) that are expanded into full structures here. It is meant for demos,
tests and as a sanity anchor for the RAG stack - not as a replacement for the
DFT databases.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from importlib import resources
from typing import Any

from ..schema import (
    ARPESFeatures,
    BandGap,
    ElectronicProperties,
    HoppingTerm,
    Material2D,
    Provenance,
    Stability,
    Structure2D,
    TightBindingModel,
)
from .base import Connector, structure_from_arrays

VACUUM = 20.0
SQ3 = math.sqrt(3.0)


def _hexagonal_cell(a: float) -> list[list[float]]:
    return [[a, 0.0, 0.0], [a / 2, a * SQ3 / 2, 0.0], [0.0, 0.0, VACUUM]]


def _frac(cell: list[list[float]], f1: float, f2: float, z: float) -> tuple[float, float, float]:
    return (f1 * cell[0][0] + f2 * cell[1][0], f1 * cell[0][1] + f2 * cell[1][1], z)


def build_prototype(p: dict[str, Any]) -> Structure2D:
    kind = p["kind"]
    zc = VACUUM / 2
    if kind == "honeycomb":
        cell = _hexagonal_cell(p["a"])
        d = p.get("buckling", 0.0) / 2
        A, B = p["species"]
        pos = [_frac(cell, 0, 0, zc + d), _frac(cell, 1 / 3, 1 / 3, zc - d)]
        return structure_from_arrays(cell, [A, B], pos, prototype="honeycomb")
    if kind in ("1H-MX2", "1T-MX2"):
        cell = _hexagonal_cell(p["a"])
        M, X = p["species"]
        dz = p["dz"]
        if kind == "1H-MX2":  # trigonal prismatic: both X above the same point
            pos = [_frac(cell, 0, 0, zc), _frac(cell, 1 / 3, 1 / 3, zc + dz),
                   _frac(cell, 1 / 3, 1 / 3, zc - dz)]
        else:  # octahedral
            pos = [_frac(cell, 0, 0, zc), _frac(cell, 1 / 3, 1 / 3, zc + dz),
                   _frac(cell, 2 / 3, 2 / 3, zc - dz)]
        return structure_from_arrays(cell, [M, X, X], pos, prototype=kind)
    if kind == "puckered":  # black-phosphorus-like; x = zigzag, y = armchair
        a, b, h, bond = p["a"], p["b"], p["h"], p["bond"]
        d = math.sqrt(bond**2 - (a / 2) ** 2)  # in-plane projection of the zigzag bond
        e = b / 2 - d
        cell = [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, VACUUM]]
        z0 = zc - h / 2
        pos = [(0.0, 0.0, z0), (a / 2, d, z0), (a / 2, d + e, z0 + h), (0.0, 2 * d + e, z0 + h)]
        (X,) = p["species"]
        return structure_from_arrays(cell, [X] * 4, pos, prototype="puckered")
    raise ValueError(f"unknown prototype kind {kind!r}")


def _honeycomb_nn_model(entry: dict[str, Any]) -> TightBindingModel:
    tb = entry["tight_binding"]
    t = float(tb["t_nn_eV"])
    hops = []
    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        hops.append(HoppingTerm(R=R, i=0, j=1, t_re=t))
        hops.append(HoppingTerm(R=(-R[0], -R[1], -R[2]), i=1, j=0, t_re=t))
    return TightBindingModel(
        orbitals=tb["orbitals"], hoppings=hops, source=tb.get("source", "literature"),
        orbital_positions=[(0.0, 0.0, 0.0), (1 / 3, 1 / 3, 0.0)],
        notes=f"nearest-neighbour model, t = {t} eV ({tb.get('reference', '')})".strip(),
    )


def material_from_seed(entry: dict[str, Any]) -> Material2D:
    refs = entry.get("references", [])
    prov = [Provenance(source="seed", source_id=entry["id"], method="literature",
                       reference="; ".join(refs) or None)]
    structure = build_prototype(entry["prototype"]) if "prototype" in entry else None
    electronic = ElectronicProperties(
        band_gaps=[BandGap(**g) for g in entry.get("band_gaps", [])],
        dirac_cone=entry.get("dirac_cone"),
        soc_gap_eV=entry.get("soc_gap"),
        topology=entry.get("topology"),
        magnetic_state=entry.get("magnetic_state"),
        magnetic_moment_muB=entry.get("magnetic_moment_muB"),
        curie_temperature_K=entry.get("curie_temperature_K"),
    )
    arpes = []
    for a in entry.get("arpes", []):
        a = dict(a)
        ref = a.pop("reference", None)
        arpes.append(ARPESFeatures(**a, provenance=Provenance(
            source="seed", method="ARPES", reference=ref)))
    tb = [_honeycomb_nn_model(entry)] if "tight_binding" in entry else []
    return Material2D(
        uid=f"seed:{entry['id']}",
        formula=entry["formula"],
        name=entry.get("name"),
        structure=structure,
        electronic=electronic,
        stability=Stability(experimentally_realized=entry.get("experimentally_realized")),
        tight_binding=tb,
        arpes=arpes,
        tags=entry.get("tags", []),
        provenance=prov,
        notes=entry.get("notes"),
    )


def load_seed_entries() -> list[dict[str, Any]]:
    text = (resources.files("llmatter2d") / "data" / "seed_materials.json").read_text("utf-8")
    return json.loads(text)["materials"]


class SeedConnector(Connector):
    name = "seed"

    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        for n, entry in enumerate(load_seed_entries()):
            if limit is not None and n >= limit:
                return
            yield material_from_seed(entry)
