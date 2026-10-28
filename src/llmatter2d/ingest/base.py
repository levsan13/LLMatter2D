"""Common interface for data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date
from typing import Any

import numpy as np

from ..schema import Material2D, Site, Structure2D


class Connector(ABC):
    """A source of :class:`Material2D` records (database, file, API...)."""

    name: str = "connector"

    @abstractmethod
    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        """Yield materials, at most ``limit`` of them if given."""


def today() -> str:
    return date.today().isoformat()


def as_float(value: Any) -> float | None:
    """Convert database values ("na", "", None, numbers) to float or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def structure_from_arrays(
    lattice: Any,
    species: list[str],
    positions: Any,
    fractional: bool = False,
    vacuum: float = 20.0,
    prototype: str | None = None,
) -> Structure2D:
    """Build a :class:`Structure2D`, fixing a missing/degenerate vacuum vector."""
    cell = np.array([[np.nan if x is None else x for x in row] for row in lattice], dtype=float)
    pos = np.asarray(positions, dtype=float)
    if fractional:
        pos = pos @ np.nan_to_num(cell)
    if not np.all(np.isfinite(cell[2])) or np.linalg.norm(cell[2]) < 1e-6:
        normal = np.cross(cell[0], cell[1])
        normal /= np.linalg.norm(normal)
        extent = np.ptp(pos @ normal) if len(pos) > 1 else 0.0
        cell[2] = normal * (extent + vacuum)
    sites = [Site(species=s, position=tuple(map(float, p))) for s, p in zip(species, pos)]
    return Structure2D(lattice=cell.tolist(), sites=sites, prototype=prototype)
