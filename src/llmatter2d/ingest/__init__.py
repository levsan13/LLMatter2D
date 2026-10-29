"""Data ingestion: DFT databases, Wannier90 Hamiltonians and ARPES maps."""

from __future__ import annotations

from typing import Any

from .base import Connector
from .databases import C2DBConnector, JarvisConnector, TwoDMatPediaConnector
from .optimade import OptimadeClient, OptimadeConnector
from .seed import SeedConnector


def get_connector(source: str, **kwargs: Any) -> Connector:
    """Factory used by the CLI: ``seed``, ``mc2d``, ``c2db``, ``2dmatpedia``, ``jarvis``."""
    source = source.lower()
    if source == "seed":
        return SeedConnector()
    if source == "mc2d":
        return OptimadeConnector("mc2d", filter=kwargs.get("filter"))
    if source == "c2db":
        return C2DBConnector(kwargs["path"], selection=kwargs.get("filter"))
    if source == "2dmatpedia":
        return TwoDMatPediaConnector(kwargs["path"])
    if source == "jarvis":
        return JarvisConnector(store_dir=kwargs.get("path"))
    raise ValueError(f"unknown source {source!r}")


__all__ = [
    "C2DBConnector",
    "Connector",
    "JarvisConnector",
    "OptimadeClient",
    "OptimadeConnector",
    "SeedConnector",
    "TwoDMatPediaConnector",
    "get_connector",
]
