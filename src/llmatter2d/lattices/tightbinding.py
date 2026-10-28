"""Nearest-neighbour tight-binding on Archimedean lattices.

H = -t sum_<ij> (c_i^dag c_j + h.c.) + sum_i eps_i n_i

This minimal model already captures the features that make Archimedean
lattices interesting for materials discovery: Dirac cones (6^3, 4.6.12...),
flat bands (3.6.3.6, 3.12^2, 3.4.6.4...) and quadratic band touchings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .archimedean import Lattice2D


def reciprocal(lat: Lattice2D) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([lat.a1, lat.a2])
    B = 2 * np.pi * np.linalg.inv(A).T
    return B[0], B[1]


def hamiltonian(lat: Lattice2D, k: np.ndarray, t: float = 1.0,
                onsite: np.ndarray | None = None) -> np.ndarray:
    n = lat.n_sites
    H = np.zeros((n, n), dtype=complex)
    if onsite is not None:
        H[np.diag_indices(n)] = onsite
    for b in lat.bonds:
        phase = np.exp(1j * np.dot(k, lat.bond_vector(b)))
        H[b.i, b.j] += -t * phase
        H[b.j, b.i] += -t * np.conj(phase)
    return H


def bands_at(lat: Lattice2D, k: np.ndarray, t: float = 1.0,
             onsite: np.ndarray | None = None) -> np.ndarray:
    return np.linalg.eigvalsh(hamiltonian(lat, k, t, onsite))


def high_symmetry_points(lat: Lattice2D) -> dict[str, np.ndarray]:
    b1, b2 = reciprocal(lat)
    G = np.zeros(2)
    cos = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
    same = np.isclose(np.linalg.norm(b1), np.linalg.norm(b2), rtol=1e-6)
    if same and np.isclose(abs(cos), 0.5, atol=1e-6):  # hexagonal
        K = (b1 + b2) / 3 if cos > 0 else (2 * b1 + b2) / 3
        return {"G": G, "M": b1 / 2, "K": K}
    return {"G": G, "X": b1 / 2, "M": (b1 + b2) / 2, "Y": b2 / 2}


def default_path(lat: Lattice2D) -> list[str]:
    pts = high_symmetry_points(lat)
    return ["G", "M", "K", "G"] if "K" in pts else ["G", "X", "M", "Y", "G"]


@dataclass
class BandStructure:
    kdist: np.ndarray
    energies: np.ndarray  # (nk, nbands)
    labels: list[str]
    label_positions: list[float]


def band_structure(lat: Lattice2D, t: float = 1.0, path: list[str] | None = None,
                   n_per_segment: int = 60, onsite: np.ndarray | None = None) -> BandStructure:
    pts = high_symmetry_points(lat)
    path = path or default_path(lat)
    ks, dist, ticks = [], [], [0.0]
    s = 0.0
    for p, q in zip(path[:-1], path[1:]):
        for x in np.linspace(0, 1, n_per_segment, endpoint=False):
            k = pts[p] + x * (pts[q] - pts[p])
            if ks:
                s += np.linalg.norm(k - ks[-1])
            ks.append(k)
            dist.append(s)
        s += np.linalg.norm(pts[q] - ks[-1])
        ticks.append(s)
    ks.append(pts[path[-1]])
    dist.append(s)
    E = np.array([bands_at(lat, k, t, onsite) for k in ks])
    return BandStructure(np.array(dist), E, list(path), ticks)


@dataclass
class BandFeatures:
    lattice: str
    t: float
    bandwidths: list[float]
    flat_bands: list[float] = field(default_factory=list)  # energies of flat bands
    linear_crossings: list[str] = field(default_factory=list)  # e.g. "K: bands 0-1 at E=0"
    quadratic_touchings: list[str] = field(default_factory=list)
    multifold_degeneracies: list[str] = field(default_factory=list)
    energy_range: tuple[float, float] = (0.0, 0.0)

    def summary(self) -> str:
        out = [f"NN tight-binding on {self.lattice} (t={self.t} eV): "
               f"{len(self.bandwidths)} bands spanning "
               f"[{self.energy_range[0]:.2f}, {self.energy_range[1]:.2f}] eV"]
        if self.flat_bands:
            out.append("flat band(s) at E = " + ", ".join(f"{e:.2f}" for e in self.flat_bands)
                       + " eV")
        if self.linear_crossings:
            out.append("Dirac-like linear crossings: " + "; ".join(self.linear_crossings))
        if self.quadratic_touchings:
            out.append("quadratic band touchings: " + "; ".join(self.quadratic_touchings))
        if self.multifold_degeneracies:
            out.append("multifold degeneracies: " + "; ".join(self.multifold_degeneracies))
        return ". ".join(out)


def _bz_grid(lat: Lattice2D, n: int) -> np.ndarray:
    b1, b2 = reciprocal(lat)
    f = np.arange(n) / n
    return np.array([x * b1 + y * b2 for x in f for y in f])


def analyze_bands(lat: Lattice2D, t: float = 1.0, grid: int = 30,
                  flat_tol: float = 1e-6, deg_tol: float = 1e-6) -> BandFeatures:
    """Find flat bands and band touchings at the high-symmetry points."""
    E = np.array([bands_at(lat, k, t) for k in _bz_grid(lat, grid)])
    widths = (E.max(axis=0) - E.min(axis=0)).tolist()
    flat = [float(np.round(E[:, n].mean(), 6)) + 0.0 for n, w in enumerate(widths)
            if w < flat_tol * max(1.0, abs(t))]
    feats = BandFeatures(lat.name, t, widths, sorted(set(flat)),
                         energy_range=(float(E.min()), float(E.max())))

    b1, _ = reciprocal(lat)
    delta = 1e-3 * np.linalg.norm(b1)
    for label, k0 in high_symmetry_points(lat).items():
        e0 = bands_at(lat, k0, t)
        for group in _degenerate_groups(e0, deg_tol):
            energy = round(float(e0[group[0]]), 3) + 0.0
            bands = f"bands {group[0]}-{group[-1]}"
            if len(group) > 2:
                feats.multifold_degeneracies.append(
                    f"{label}: {len(group)}-fold, {bands} at E={energy:.3f} eV")
                continue
            n = group[0]
            # probe the splitting along a few directions: linear vs quadratic growth
            ratios = []
            for ang in (0.1, 0.9, 1.7):
                d = np.array([np.cos(ang), np.sin(ang)])
                s1 = np.diff(bands_at(lat, k0 + delta * d, t)[n:n + 2])[0]
                s2 = np.diff(bands_at(lat, k0 + 2 * delta * d, t)[n:n + 2])[0]
                if s1 > 1e-12:
                    ratios.append(s2 / s1)
            if not ratios:
                continue
            r = float(np.median(ratios))
            tag = f"{label}: {bands} at E={energy:.3f} eV"
            if abs(r - 2) < 0.2:
                feats.linear_crossings.append(tag)
            elif abs(r - 4) < 0.4:
                feats.quadratic_touchings.append(tag)
    return feats


def _degenerate_groups(e: np.ndarray, tol: float) -> list[list[int]]:
    groups, current = [], [0]
    for n in range(1, len(e)):
        if abs(e[n] - e[n - 1]) <= tol:
            current.append(n)
        else:
            groups.append(current)
            current = [n]
    groups.append(current)
    return [g for g in groups if len(g) > 1]
