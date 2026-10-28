"""The manuscript table is generated from the code and committed; keep them in sync."""

import runpy
from pathlib import Path

PAPER = Path(__file__).parents[1] / "paper"


def test_archimedean_table_matches_committed_version(tmp_path):
    script = runpy.run_path(str(PAPER / "scripts" / "table_archimedean.py"))
    script["main"](tmp_path)
    generated = (tmp_path / "archimedean.md").read_text(encoding="utf-8")
    assert generated == (PAPER / "tables" / "archimedean.md").read_text(encoding="utf-8")
    rows = {r["tiling"]: r for r in script["rows"]()}
    assert rows["3.6.3.6"]["flat"] == "+2"
    assert rows["6^3"]["dirac"] == "K"
