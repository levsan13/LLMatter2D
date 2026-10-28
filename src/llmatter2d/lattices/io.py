"""Minimal VASP POSCAR reader/writer (no external dependencies)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import Site, Structure2D


def to_poscar(structure: Structure2D, comment: str | None = None) -> str:
    order = list(dict.fromkeys(structure.species))
    lines = [comment or structure.formula, "1.0"]
    lines += ["  " + " ".join(f"{x:16.10f}" for x in row) for row in structure.cell]
    lines.append("  " + " ".join(order))
    lines.append("  " + " ".join(str(structure.species.count(el)) for el in order))
    lines.append("Cartesian")
    for el in order:
        for s in structure.sites:
            if s.species == el:
                lines.append("  " + " ".join(f"{x:16.10f}" for x in s.position) + f"  {el}")
    return "\n".join(lines) + "\n"


def write_poscar(structure: Structure2D, path: str | Path, comment: str | None = None) -> None:
    Path(path).write_text(to_poscar(structure, comment))


def read_poscar(path: str | Path) -> Structure2D:
    """Read a VASP5 POSCAR/CONTCAR (element names on line 6)."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    if lines[idx][0].lower() == "s":  # selective dynamics
        idx += 1
    cartesian = lines[idx][0].lower() in ("c", "k")
    idx += 1
    species = [el for el, n in zip(elements, counts) for _ in range(n)]
    coords = np.array([[float(x) for x in lines[idx + k].split()[:3]]
                       for k in range(len(species))])
    pos = coords * scale if cartesian else coords @ cell
    sites = [Site(species=s, position=tuple(map(float, p))) for s, p in zip(species, pos)]
    return Structure2D(lattice=cell.tolist(), sites=sites)
