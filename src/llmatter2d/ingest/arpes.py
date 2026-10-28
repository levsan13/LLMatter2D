"""Turn ARPES intensity maps I(k, E) into quantitative features for the RAG stack.

The LLM never sees raw spectra; it sees extracted, citable numbers:

* dispersion E(k) from momentum distribution curves (MDC peaks),
* Fermi velocity from a linear fit near E_F,
* band effective mass from a parabolic fit of energy distribution curve (EDC) peaks.

Conventions: k in 1/angstrom, E - E_F in eV (negative = occupied states).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..schema import ARPESFeatures, Provenance

HBAR_EV_S = 6.582119569e-16  # eV s
HBAR2_OVER_2ME = 3.80998212  # eV angstrom^2


@dataclass
class ARPESMap:
    k: np.ndarray  # (nk,)
    energy: np.ndarray  # (ne,)  E - E_F
    intensity: np.ndarray  # (ne, nk)

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        self.energy = np.asarray(self.energy, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.intensity.shape != (self.energy.size, self.k.size):
            raise ValueError("intensity must have shape (len(energy), len(k))")


def load_arpes_map(path: str | Path, k_key: str = "k", e_key: str = "energy",
                   i_key: str = "intensity") -> ARPESMap:
    """Load a map from ``.npz`` or ``.h5`` (h5py needed) with the given dataset names."""
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        return ARPESMap(d[k_key], d[e_key], d[i_key])
    if path.suffix in (".h5", ".hdf5", ".nxs"):
        import h5py  # optional dependency

        with h5py.File(path, "r") as f:
            return ARPESMap(f[k_key][()], f[e_key][()], f[i_key][()])
    raise ValueError(f"unsupported ARPES file type {path.suffix!r}")


def _refine_peak(y: np.ndarray, i: int) -> float:
    """Sub-grid peak position by a parabola through three points (index units)."""
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return i + 0.5 * (a - c) / denom
    return float(i)


def mdc_dispersion(m: ARPESMap, k_range: tuple[float, float] | None = None,
                   e_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Peak momentum of each MDC. Restrict ``k_range`` to follow a single branch."""
    kmask = np.ones_like(m.k, bool) if k_range is None else (m.k >= k_range[0]) & (m.k <= k_range[1])
    emask = np.ones_like(m.energy, bool) if e_range is None else (
        (m.energy >= e_range[0]) & (m.energy <= e_range[1]))
    ks = m.k[kmask]
    dk = ks[1] - ks[0]
    E_out, k_out = [], []
    for E, row in zip(m.energy[emask], m.intensity[emask][:, kmask]):
        if row.max() <= 0:
            continue
        i = int(np.argmax(row))
        E_out.append(E)
        k_out.append(ks[0] + _refine_peak(row, i) * dk)
    return np.asarray(E_out), np.asarray(k_out)


def edc_dispersion(m: ARPESMap, k_range: tuple[float, float] | None = None,
                   e_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Peak energy of each EDC (better for flat/parabolic band extrema)."""
    kmask = np.ones_like(m.k, bool) if k_range is None else (m.k >= k_range[0]) & (m.k <= k_range[1])
    emask = np.ones_like(m.energy, bool) if e_range is None else (
        (m.energy >= e_range[0]) & (m.energy <= e_range[1]))
    Es = m.energy[emask]
    dE = Es[1] - Es[0]
    k_out, E_out = [], []
    for k, col in zip(m.k[kmask], m.intensity[emask][:, kmask].T):
        if col.max() <= 0:
            continue
        i = int(np.argmax(col))
        k_out.append(k)
        E_out.append(Es[0] + _refine_peak(col, i) * dE)
    return np.asarray(k_out), np.asarray(E_out)


def fermi_velocity(E: np.ndarray, k: np.ndarray, window: float = 0.3) -> float:
    """v_F (m/s) from a linear fit E = hbar v (k - k_F) within ``window`` eV of E_F."""
    sel = np.abs(E) <= window
    if sel.sum() < 3:
        raise ValueError("not enough points near E_F for a linear fit")
    slope = np.polyfit(k[sel], E[sel], 1)[0]  # eV angstrom
    return float(abs(slope) * 1e-10 / HBAR_EV_S)


def effective_mass(k: np.ndarray, E: np.ndarray) -> float:
    """m*/m_e from E = E0 + hbar^2 (k-k0)^2 / 2m*. Negative for hole-like bands."""
    curvature = np.polyfit(k, E, 2)[0]  # eV angstrom^2
    return float(HBAR2_OVER_2ME / curvature)


def features_from_map(m: ARPESMap, *, k_range: tuple[float, float] | None = None,
                      kind: str = "linear", window: float = 0.3,
                      provenance: Provenance | None = None, **meta) -> ARPESFeatures:
    """Extract a :class:`ARPESFeatures` record.

    ``kind="linear"`` fits a Fermi velocity (Dirac-like / metallic band crossing E_F),
    ``kind="parabolic"`` fits an effective mass around a band extremum.
    """
    if kind == "linear":
        E, k = mdc_dispersion(m, k_range=k_range, e_range=(-window, window))
        return ARPESFeatures(fermi_velocity_m_s=fermi_velocity(E, k, window),
                             provenance=provenance, **meta)
    if kind == "parabolic":
        k, E = edc_dispersion(m, k_range=k_range)
        return ARPESFeatures(effective_mass_me=effective_mass(k, E), provenance=provenance,
                             **meta)
    raise ValueError("kind must be 'linear' or 'parabolic'")
