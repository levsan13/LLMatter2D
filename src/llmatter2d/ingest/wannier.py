"""Read Wannier90 tight-binding Hamiltonians (``<seedname>_hr.dat``).

File layout (Wannier90 user guide):

    line 1   : comment / date
    line 2   : num_wann
    line 3   : nrpts
    next     : nrpts degeneracy weights, 15 per line
    then     : nrpts * num_wann**2 lines "R1 R2 R3 m n Re(H) Im(H)"  (eV)

The physical hopping is H_mn(R) / deg(R), which is what is stored here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..schema import HoppingTerm, TightBindingModel


def read_wannier90_hr(
    path: str | Path,
    threshold: float = 1e-3,
    orbital_labels: list[str] | None = None,
    fermi_level: float | None = None,
) -> TightBindingModel:
    """Parse ``_hr.dat`` into a :class:`TightBindingModel`.

    ``threshold`` (eV) drops negligible hoppings to keep records compact.
    """
    lines = Path(path).read_text().splitlines()
    num_wann = int(lines[1].split()[0])
    nrpts = int(lines[2].split()[0])
    n_deg_lines = math.ceil(nrpts / 15)
    deg = [int(x) for line in lines[3:3 + n_deg_lines] for x in line.split()]
    if len(deg) != nrpts:
        raise ValueError(f"expected {nrpts} degeneracy weights, found {len(deg)}")

    body = lines[3 + n_deg_lines:]
    expected = nrpts * num_wann**2
    if len([ln for ln in body if ln.strip()]) < expected:
        raise ValueError("truncated _hr.dat file")

    hoppings: list[HoppingTerm] = []
    r_index, last_R = -1, None
    for line in body:
        parts = line.split()
        if len(parts) < 7:
            continue
        R = (int(parts[0]), int(parts[1]), int(parts[2]))
        if R != last_R:
            r_index += 1
            last_R = R
        m, n = int(parts[3]) - 1, int(parts[4]) - 1
        h = complex(float(parts[5]), float(parts[6])) / deg[r_index]
        if abs(h) >= threshold:
            hoppings.append(HoppingTerm(R=R, i=m, j=n, t_re=h.real, t_im=h.imag))

    labels = orbital_labels or [f"w{k + 1}" for k in range(num_wann)]
    return TightBindingModel(orbitals=labels, hoppings=hoppings, source="Wannier90",
                             fermi_level_eV=fermi_level,
                             notes=f"{nrpts} R-vectors, |t| >= {threshold} eV kept")


def bloch_hamiltonian(model: TightBindingModel, k_frac: np.ndarray) -> np.ndarray:
    """H(k) = sum_R H(R) exp(2 pi i k.R), with k in fractional reciprocal coordinates."""
    n = len(model.orbitals)
    H = np.zeros((n, n), dtype=complex)
    k = np.asarray(k_frac, dtype=float)
    for h in model.hoppings:
        H[h.i, h.j] += h.value * np.exp(2j * np.pi * np.dot(k, h.R))
    return H


def write_wannier90_hr(model: TightBindingModel, path: str | Path) -> None:
    """Write a model back to ``_hr.dat`` format (all degeneracies = 1)."""
    n = len(model.orbitals)
    by_R: dict[tuple[int, int, int], np.ndarray] = {}
    for h in model.hoppings:
        by_R.setdefault(h.R, np.zeros((n, n), dtype=complex))[h.i, h.j] += h.value
    Rs = sorted(by_R)
    out = ["written by llmatter2d", str(n), str(len(Rs))]
    out += [" ".join("1" for _ in Rs[i:i + 15]) for i in range(0, len(Rs), 15)]
    for R in Rs:
        H = by_R[R]
        for col in range(n):
            for row in range(n):
                v = H[row, col]
                out.append(f"{R[0]:5d}{R[1]:5d}{R[2]:5d}{row + 1:5d}{col + 1:5d}"
                           f"{v.real:12.6f}{v.imag:12.6f}")
    Path(path).write_text("\n".join(out) + "\n")
