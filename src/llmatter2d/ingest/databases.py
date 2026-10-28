"""File-based connectors for the large 2D DFT databases.

* C2DB       - ASE database file (``c2db.db``), obtained from https://2dhub.org/c2db
* 2DMatPedia - JSON dump from http://www.2dmatpedia.org (one record per line)
* JARVIS-DFT - the ``dft_2d`` dataset distributed through ``jarvis-tools``

The databases are not redistributed with LLMatter2D: each has its own license
and citation requirements, listed in ``docs/data_sources.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schema import BandGap, ElectronicProperties, Material2D, Provenance, Stability
from .base import Connector, as_float, structure_from_arrays, today


def _gap(value: Any, method: str) -> list[BandGap]:
    v = as_float(value)
    return [BandGap(value_eV=v, method=method)] if v is not None else []


# --------------------------------------------------------------------------------------
# C2DB
# --------------------------------------------------------------------------------------

class C2DBConnector(Connector):
    """Read a local C2DB ASE database (requires ``pip install ase``)."""

    name = "c2db"
    reference = ("Haastrup et al., 2D Mater. 5, 042002 (2018); "
                 "Gjerding et al., 2D Mater. 8, 044002 (2021)")

    def __init__(self, path: str | Path, selection: str | None = None):
        self.path = Path(path)
        self.selection = selection

    def row_to_material(self, row: Any) -> Material2D:
        kv = dict(row.key_value_pairs)
        atoms = row.toatoms()
        structure = structure_from_arrays(atoms.cell.tolist(), atoms.get_chemical_symbols(),
                                          atoms.positions)
        gaps = (_gap(kv.get("gap"), "DFT-PBE") + _gap(kv.get("gap_hse"), "DFT-HSE06")
                + _gap(kv.get("gap_gw"), "G0W0"))
        uid = kv.get("uid") or f"{structure.formula}-{row.id}"
        return Material2D(
            uid=f"c2db:{uid}",
            formula=structure.formula,
            structure=structure,
            electronic=ElectronicProperties(
                band_gaps=gaps,
                is_metal=(gaps[0].value_eV < 1e-3) if gaps else None,
                magnetic_state=kv.get("magstate"),
            ),
            stability=Stability(
                formation_energy_eV_atom=as_float(kv.get("hform")),
                energy_above_hull_eV_atom=as_float(kv.get("ehull")),
            ),
            provenance=[Provenance(source="C2DB", source_id=str(uid), method="DFT (GPAW)",
                                   reference=self.reference, retrieved=today())],
        )

    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        from ase.db import connect  # optional dependency

        db = connect(str(self.path))
        rows = db.select(self.selection) if self.selection else db.select()
        for n, row in enumerate(rows):
            if limit is not None and n >= limit:
                return
            yield self.row_to_material(row)


# --------------------------------------------------------------------------------------
# 2DMatPedia
# --------------------------------------------------------------------------------------

def _pymatgen_structure(d: dict[str, Any]):
    lattice = d["lattice"]["matrix"]
    species, positions = [], []
    for site in d["sites"]:
        species.append(site["species"][0]["element"])
        positions.append(site["xyz"])
    return structure_from_arrays(lattice, species, positions)


class TwoDMatPediaConnector(Connector):
    """Read the 2DMatPedia JSON dump (JSON lines or a JSON array)."""

    name = "2dmatpedia"
    reference = "Zhou et al., Sci. Data 6, 86 (2019)"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _records(self) -> Iterator[dict[str, Any]]:
        text = self.path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            yield from json.loads(text)
            return
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)

    def record_to_material(self, r: dict[str, Any]) -> Material2D:
        structure = _pymatgen_structure(r["structure"])
        exfo = as_float(r.get("exfoliation_energy_per_atom"))
        mag = as_float(r.get("total_magnetization"))
        gaps = _gap(r.get("bandgap"), "DFT-PBE")
        return Material2D(
            uid=f"2dmatpedia:{r['material_id']}",
            formula=r.get("formula_pretty") or structure.formula,
            structure=structure,
            electronic=ElectronicProperties(
                band_gaps=gaps,
                is_metal=(gaps[0].value_eV < 1e-3) if gaps else None,
                magnetic_moment_muB=mag,
                magnetic_state=("magnetic" if mag and abs(mag) > 0.1 else "NM") if mag is not None
                else None,
            ),
            stability=Stability(
                exfoliation_energy_meV_atom=exfo * 1000 if exfo is not None else None,
                energy_above_hull_eV_atom=as_float(r.get("decomposition_energy")),
            ),
            tags=[f"discovery:{r['discovery_process']}"] if r.get("discovery_process") else [],
            provenance=[Provenance(source="2DMatPedia", source_id=str(r["material_id"]),
                                   method="DFT-PBE", reference=self.reference,
                                   retrieved=today())],
        )

    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        for n, r in enumerate(self._records()):
            if limit is not None and n >= limit:
                return
            yield self.record_to_material(r)


# --------------------------------------------------------------------------------------
# JARVIS-DFT (2D)
# --------------------------------------------------------------------------------------

class JarvisConnector(Connector):
    """JARVIS-DFT 2D monolayers via ``jarvis-tools`` (downloads on first use)."""

    name = "jarvis"
    reference = "Choudhary et al., npj Comput. Mater. 6, 173 (2020)"

    def __init__(self, dataset: str = "dft_2d", records: list[dict[str, Any]] | None = None):
        self.dataset = dataset
        self._records = records

    def records(self) -> list[dict[str, Any]]:
        if self._records is None:
            from jarvis.db.figshare import data  # optional dependency

            self._records = data(self.dataset)
        return self._records

    def record_to_material(self, r: dict[str, Any]) -> Material2D:
        at = r["atoms"]
        structure = structure_from_arrays(at["lattice_mat"], at["elements"], at["coords"],
                                          fractional=not at.get("cartesian", False))
        gaps = (_gap(r.get("optb88vdw_bandgap"), "DFT-optB88vdW")
                + _gap(r.get("mbj_bandgap"), "DFT-TBmBJ") + _gap(r.get("hse_gap"), "DFT-HSE06"))
        magmom = as_float(r.get("magmom_oszicar"))
        spillage = as_float(r.get("spillage"))
        tags = ["topological candidate (spillage > 0.5)"] if spillage and spillage > 0.5 else []
        return Material2D(
            uid=f"jarvis:{r['jid']}",
            formula=r.get("formula") or structure.formula,
            structure=structure,
            electronic=ElectronicProperties(
                band_gaps=gaps,
                is_metal=(gaps[0].value_eV < 1e-3) if gaps else None,
                magnetic_moment_muB=magmom,
            ),
            stability=Stability(
                formation_energy_eV_atom=as_float(r.get("formation_energy_peratom")),
                exfoliation_energy_meV_atom=as_float(r.get("exfoliation_energy")),
            ),
            tags=tags,
            provenance=[Provenance(source="JARVIS-DFT", source_id=str(r["jid"]),
                                   method="DFT (VASP, OptB88vdW)", reference=self.reference,
                                   retrieved=today())],
        )

    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        for n, r in enumerate(self.records()):
            if limit is not None and n >= limit:
                return
            yield self.record_to_material(r)
