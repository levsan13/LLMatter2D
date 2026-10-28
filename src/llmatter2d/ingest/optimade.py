"""OPTIMADE client and connectors.

OPTIMADE (https://www.optimade.org) is the common REST API exposed by most
materials databases. The Materials Cloud 2D database (MC2D) is served at
``https://optimade.materialscloud.org/main/mc2d`` and exposes the PBE band gap
as the provider-specific field ``_mcloud_band_gap``.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..schema import BandGap, ElectronicProperties, Material2D, Provenance, Structure2D
from .base import Connector, as_float, structure_from_arrays, today

STRUCTURE_FIELDS = [
    "chemical_formula_reduced",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "species",
    "dimension_types",
]


class OptimadeClient:
    """Minimal OPTIMADE v1 client (structures endpoint, pagination via ``links.next``)."""

    def __init__(self, base_url: str, timeout: float = 60.0, page_limit: int = 50):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout = timeout
        self.page_limit = page_limit

    def _get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": f"llmatter2d/{__version__}",
                                                   "Accept": "application/vnd.api+json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def structures_url(self, filter: str | None = None, response_fields: list[str] | None = None,
                       page_limit: int | None = None) -> str:
        params: dict[str, Any] = {"page_limit": page_limit or self.page_limit}
        if filter:
            params["filter"] = filter
        if response_fields:
            params["response_fields"] = ",".join(response_fields)
        return f"{self.base_url}/structures?{urllib.parse.urlencode(params)}"

    def iter_structures(self, filter: str | None = None, response_fields: list[str] | None = None,
                        limit: int | None = None) -> Iterator[dict[str, Any]]:
        page = min(self.page_limit, limit) if limit else self.page_limit
        url: str | None = self.structures_url(filter, response_fields, page)
        n = 0
        while url:
            payload = self._get(url)
            for entry in payload.get("data", []):
                yield entry
                n += 1
                if limit is not None and n >= limit:
                    return
            nxt = (payload.get("links") or {}).get("next")
            url = nxt.get("href") if isinstance(nxt, dict) else nxt


def structure_from_optimade(attrs: dict[str, Any]) -> Structure2D:
    """Convert OPTIMADE ``structures`` attributes into a :class:`Structure2D`."""
    names = attrs["species_at_sites"]
    symbol_of = {s["name"]: s["chemical_symbols"][0] for s in attrs.get("species") or []}
    species = [symbol_of.get(n, n) for n in names]
    return structure_from_arrays(attrs["lattice_vectors"], species,
                                 attrs["cartesian_site_positions"])


@dataclass
class OptimadePreset:
    base_url: str
    source: str
    reference: str
    filter: str | None = None
    extra_fields: list[str] = field(default_factory=list)
    # maps the raw attributes to an ElectronicProperties object
    electronic: Callable[[dict[str, Any]], ElectronicProperties] | None = None


def _mc2d_electronic(attrs: dict[str, Any]) -> ElectronicProperties:
    gap = as_float(attrs.get("_mcloud_band_gap"))
    if gap is None:
        return ElectronicProperties()
    return ElectronicProperties(band_gaps=[BandGap(value_eV=gap, method="DFT (MC2D)")],
                                is_metal=gap < 1e-3)


PRESETS: dict[str, OptimadePreset] = {
    "mc2d": OptimadePreset(
        base_url="https://optimade.materialscloud.org/main/mc2d",
        source="MC2D",
        reference="Mounet et al., Nat. Nanotechnol. 13, 246 (2018); Materials Cloud MC2D",
        extra_fields=["_mcloud_band_gap"],
        electronic=_mc2d_electronic,
    ),
}


class OptimadeConnector(Connector):
    """Fetch 2D structures (and any mapped properties) from an OPTIMADE provider."""

    def __init__(self, preset: str | OptimadePreset = "mc2d", filter: str | None = None,
                 client: OptimadeClient | None = None):
        self.preset = PRESETS[preset] if isinstance(preset, str) else preset
        self.name = self.preset.source.lower()
        self.filter = filter if filter is not None else self.preset.filter
        self.client = client or OptimadeClient(self.preset.base_url)

    def to_material(self, entry: dict[str, Any]) -> Material2D:
        attrs = entry["attributes"]
        structure = structure_from_optimade(attrs)
        electronic = self.preset.electronic(attrs) if self.preset.electronic else None
        return Material2D(
            uid=f"{self.name}:{entry['id']}",
            formula=attrs.get("chemical_formula_reduced") or structure.formula,
            structure=structure,
            electronic=electronic or ElectronicProperties(),
            provenance=[Provenance(source=self.preset.source, source_id=str(entry["id"]),
                                   reference=self.preset.reference, url=self.preset.base_url,
                                   retrieved=today())],
        )

    def fetch(self, limit: int | None = None) -> Iterator[Material2D]:
        fields = STRUCTURE_FIELDS + self.preset.extra_fields
        for entry in self.client.iter_structures(self.filter, fields, limit):
            yield self.to_material(entry)
