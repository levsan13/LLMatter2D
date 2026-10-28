"""The eleven Archimedean (uniform, edge-to-edge) tilings of the plane.

Each tiling is stored as a 2D Bravais lattice plus a basis of vertices, with
all edges of unit length. Vertices become atomic sites and edges become
nearest-neighbour bonds, so the tilings are templates for new 2D materials:
graphene is 6^3, the kagome lattice is 3.6.3.6, graphenylene is 4.6.12, ...

Notation follows Gruenbaum & Shephard: the polygons met around each vertex,
e.g. ``3^2.4.3.4`` = triangle, triangle, square, triangle, square.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..schema import Site, Structure2D

SQ3 = np.sqrt(3.0)


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    R: tuple[int, int]  # site j sits in cell R relative to site i (cell 0)


@dataclass(frozen=True, eq=False)
class Lattice2D:
    name: str
    alias: str
    a1: np.ndarray
    a2: np.ndarray
    basis: np.ndarray  # (n, 2) Cartesian
    vertex_config: tuple[int, ...] = field(default=())

    # ---------------------------------------------------------------- geometry
    @property
    def n_sites(self) -> int:
        return len(self.basis)

    @property
    def cell(self) -> np.ndarray:
        return np.array([self.a1, self.a2])

    @property
    def area(self) -> float:
        return float(abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]))

    @cached_property
    def bonds(self) -> list[Bond]:
        """Unique nearest-neighbour bonds (each undirected bond listed once)."""
        return find_bonds(self.cell, self.basis)

    @property
    def bond_length(self) -> float:
        b = self.bonds[0]
        return float(np.linalg.norm(self.bond_vector(b)))

    def bond_vector(self, b: Bond) -> np.ndarray:
        return self.basis[b.j] + b.R[0] * self.a1 + b.R[1] * self.a2 - self.basis[b.i]

    def coordination(self) -> np.ndarray:
        z = np.zeros(self.n_sites, int)
        for b in self.bonds:
            z[b.i] += 1
            z[b.j] += 1
        return z

    def sublattice_coloring(self) -> list[int] | None:
        """Two-colouring of the basis consistent with all bonds, or None.

        ``None`` means the lattice cannot be decorated as an A/B alternant
        compound within its primitive cell (e.g. any lattice with triangles).
        """
        color = [-1] * self.n_sites
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_sites)}
        for b in self.bonds:
            if b.i == b.j:
                return None
            adj[b.i].append(b.j)
            adj[b.j].append(b.i)
        for start in range(self.n_sites):
            if color[start] >= 0:
                continue
            color[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if color[v] < 0:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return None
        return color

    def scaled(self, bond_length: float) -> Lattice2D:
        s = bond_length / self.bond_length
        return Lattice2D(self.name, self.alias, self.a1 * s, self.a2 * s, self.basis * s,
                         self.vertex_config)

    # ---------------------------------------------------------------- materials
    def to_structure(self, species: str | Iterable[str], bond_length: float,
                     buckling: float = 0.0, vacuum: float = 20.0) -> Structure2D:
        """Decorate the tiling with atoms.

        ``species`` is one element, a list with one element per basis site, or
        two elements ``(A, B)`` for an alternant decoration of a bipartite tiling.
        ``buckling`` displaces the two sublattices by +/- buckling/2 (bipartite only).
        """
        lat = self.scaled(bond_length)
        colors = self.sublattice_coloring()
        if isinstance(species, str):
            sp = [species] * self.n_sites
        else:
            sp = list(species)
            if len(sp) == 2 and self.n_sites != 2:
                if colors is None:
                    raise ValueError(f"{self.name} is not bipartite; give one species per site")
                sp = [sp[c] for c in colors]
            if len(sp) != self.n_sites:
                raise ValueError(f"need {self.n_sites} species, got {len(sp)}")
        if buckling and colors is None:
            raise ValueError("buckling requires a bipartite tiling")
        z0 = vacuum / 2
        sites = []
        for k, (x, y) in enumerate(lat.basis):
            dz = (buckling / 2) * (1 if colors and colors[k] == 0 else -1) if buckling else 0.0
            sites.append(Site(species=sp[k], position=(float(x), float(y), z0 + dz)))
        cell = [[*lat.a1, 0.0], [*lat.a2, 0.0], [0.0, 0.0, vacuum]]
        return Structure2D(lattice=cell, sites=sites, prototype=f"archimedean:{self.name}")


def find_bonds(cell: np.ndarray, basis: np.ndarray, tol: float = 1e-6,
               search: int = 2) -> list[Bond]:
    """All bonds whose length equals the shortest inter-site distance."""
    a1, a2 = cell
    cands = []
    for i, ri in enumerate(basis):
        for j, rj in enumerate(basis):
            for n1 in range(-search, search + 1):
                for n2 in range(-search, search + 1):
                    if i == j and n1 == 0 and n2 == 0:
                        continue
                    d = np.linalg.norm(rj + n1 * a1 + n2 * a2 - ri)
                    cands.append((d, i, j, (n1, n2)))
    dmin = min(c[0] for c in cands)
    seen, bonds = set(), []
    for d, i, j, R in cands:
        if abs(d - dmin) > tol * max(1.0, dmin):
            continue
        key = (i, j, R)
        rev = (j, i, (-R[0], -R[1]))
        if rev in seen:
            continue
        seen.add(key)
        bonds.append(Bond(i, j, R))
    return bonds


# ------------------------------------------------------------------------------ catalogue

def _hex(L: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([L, 0.0]), np.array([L / 2, L * SQ3 / 2])


def _ring(n: int, radius: float, phase_deg: float, center=(0.0, 0.0)) -> list[np.ndarray]:
    c = np.asarray(center, float)
    return [c + radius * np.array([np.cos(np.radians(phase_deg + 360 * k / n)),
                                   np.sin(np.radians(phase_deg + 360 * k / n))])
            for k in range(n)]


def _triangular() -> Lattice2D:
    a1, a2 = _hex(1.0)
    return Lattice2D("3^6", "triangular", a1, a2, np.zeros((1, 2)), (3,) * 6)


def _square() -> Lattice2D:
    return Lattice2D("4^4", "square", np.array([1.0, 0]), np.array([0, 1.0]),
                     np.zeros((1, 2)), (4,) * 4)


def _honeycomb() -> Lattice2D:
    a1, a2 = _hex(SQ3)
    return Lattice2D("6^3", "honeycomb", a1, a2, np.array([[0.0, 0.0], [0.0, 1.0]]), (6,) * 3)


def _kagome() -> Lattice2D:
    a1, a2 = _hex(2.0)
    return Lattice2D("3.6.3.6", "kagome", a1, a2, np.array([[0, 0], a1 / 2, a2 / 2]),
                     (3, 6, 3, 6))


def _snub_hexagonal() -> Lattice2D:
    # hexagon of unit edge in a triangular lattice of constant sqrt(7); chiral
    a1 = np.array([2.5, SQ3 / 2])
    rot = np.array([[0.5, -SQ3 / 2], [SQ3 / 2, 0.5]])
    a2 = rot @ a1
    return Lattice2D("3^4.6", "snub-hexagonal", a1, a2, np.array(_ring(6, 1.0, 0.0)),
                     (3, 3, 3, 3, 6))


def _elongated_triangular() -> Lattice2D:
    return Lattice2D("3^3.4^2", "elongated-triangular", np.array([1.0, 0.0]),
                     np.array([0.5, 1 + SQ3 / 2]), np.array([[0.0, 0.0], [0.0, 1.0]]),
                     (3, 3, 3, 4, 4))


def _snub_square() -> Lattice2D:
    # corner-sharing unit squares rotated by +/-15 deg; square cell sqrt(2+sqrt(3))
    L = np.sqrt(2 + SQ3)
    return Lattice2D("3^2.4.3.4", "snub-square", np.array([L, 0.0]), np.array([0.0, L]),
                     np.array(_ring(4, 1 / np.sqrt(2), 60.0)), (3, 3, 4, 3, 4))


def _rhombitrihexagonal() -> Lattice2D:
    a1, a2 = _hex(1 + SQ3)
    return Lattice2D("3.4.6.4", "rhombitrihexagonal", a1, a2, np.array(_ring(6, 1.0, 30.0)),
                     (3, 4, 6, 4))


def _truncated_hexagonal() -> Lattice2D:
    # every honeycomb vertex replaced by a unit triangle ("star" lattice)
    r = 1 / SQ3
    d = 1 + 2 * r
    a1, a2 = _hex(SQ3 * d)
    dirs = [np.array([0.0, 1.0]), np.array([-SQ3 / 2, -0.5]), np.array([SQ3 / 2, -0.5])]
    A, B = np.zeros(2), np.array([0.0, d])
    basis = [A + r * u for u in dirs] + [B - r * u for u in dirs]
    return Lattice2D("3.12^2", "truncated-hexagonal", a1, a2, np.array(basis), (3, 12, 12))


def _truncated_trihexagonal() -> Lattice2D:
    # hexagons on honeycomb sites, linked by squares (graphenylene-like)
    d = 1 + SQ3
    a1, a2 = _hex(SQ3 * d)
    basis = _ring(6, 1.0, 0.0) + _ring(6, 1.0, 0.0, center=(0.0, d))
    return Lattice2D("4.6.12", "truncated-trihexagonal", a1, a2, np.array(basis), (4, 6, 12))


def _truncated_square() -> Lattice2D:
    L = 1 + np.sqrt(2)
    return Lattice2D("4.8^2", "truncated-square", np.array([L, 0.0]), np.array([0.0, L]),
                     np.array(_ring(4, 1 / np.sqrt(2), 0.0)), (4, 8, 8))


_BUILDERS = [_triangular, _square, _honeycomb, _snub_hexagonal, _elongated_triangular,
             _snub_square, _rhombitrihexagonal, _kagome, _truncated_hexagonal,
             _truncated_trihexagonal, _truncated_square]

ARCHIMEDEAN: dict[str, Lattice2D] = {lat.name: lat for lat in (b() for b in _BUILDERS)}

_ALIASES = {lat.alias: name for name, lat in ARCHIMEDEAN.items()}
_ALIASES.update({"trihexagonal": "3.6.3.6", "star": "3.12^2", "graphenylene": "4.6.12",
                 "ruby": "3.4.6.4", "cairo-dual": "3^2.4.3.4"})


def archimedean_lattice(name: str) -> Lattice2D:
    """Look up a tiling by symbol (``3.6.3.6``, ``3^4.6``) or alias (``kagome``)."""
    key = name.strip().lower().replace(" ", "")
    if key in ARCHIMEDEAN:
        return ARCHIMEDEAN[key]
    if key in _ALIASES:
        return ARCHIMEDEAN[_ALIASES[key]]
    # accept expanded notation such as 3.3.3.3.6
    for lat in ARCHIMEDEAN.values():
        if key == ".".join(map(str, lat.vertex_config)):
            return lat
    raise KeyError(f"unknown Archimedean lattice {name!r}; options: {sorted(ARCHIMEDEAN)}")
