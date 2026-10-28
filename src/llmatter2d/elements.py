"""Small element tables used for quick geometric estimates.

Covalent radii (angstrom) from B. Cordero et al., Dalton Trans. 2832 (2008).
For carbon the sp2 value is used; for 3d metals the low-spin value. These are
only used to set a first-guess bond length before a DFT relaxation.
"""

from __future__ import annotations

COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.73, "N": 0.71, "O": 0.66, "F": 0.57,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Nb": 1.64, "Mo": 1.54, "Pd": 1.39, "Ag": 1.45,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39,
    "Ta": 1.70, "W": 1.62, "Pt": 1.36, "Au": 1.36, "Tl": 1.45, "Pb": 1.46,
    "Bi": 1.48,
}

# Coordination numbers that are chemically common for each element in covalent
# 2D networks. Used only as a soft plausibility prior when ranking candidates.
TYPICAL_COORDINATION: dict[str, tuple[int, ...]] = {
    "B": (3, 4, 5, 6), "C": (3, 4), "N": (3,), "Al": (3, 4, 6),
    "Si": (3, 4), "P": (3,), "Ga": (3, 4), "Ge": (3, 4), "As": (3,),
    "In": (3, 4, 6), "Sn": (3, 4), "Sb": (3,), "Bi": (3,),
    "S": (2, 3), "Se": (2, 3), "Te": (2, 3),
}


def bond_length(a: str, b: str) -> float:
    """First-guess bond length between two elements (sum of covalent radii)."""
    try:
        return COVALENT_RADII[a] + COVALENT_RADII[b]
    except KeyError as exc:
        raise KeyError(f"no covalent radius tabulated for {exc.args[0]!r}") from None
