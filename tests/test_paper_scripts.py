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


def test_candidates_table_matches_committed_version(tmp_path):
    script = runpy.run_path(str(PAPER / "scripts" / "table_candidates.py"))
    script["main"](tmp_path, ["B", "C", "N", "Si", "Ge", "Sn"], 12)
    for name in ("candidates.md", "candidates.tex"):
        generated = (tmp_path / name).read_text(encoding="utf-8")
        assert generated == (PAPER / "tables" / name).read_text(encoding="utf-8")


def test_manuscript_cites_only_bib_entries():
    import re

    tex = (PAPER / "manuscript" / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "manuscript" / "references.bib").read_text(encoding="utf-8")
    cited = {k.strip() for g in re.findall(r"\\(?:cite|onlinecite)\{([^}]+)\}", tex)
             for k in g.split(",")}
    assert cited <= set(re.findall(r"@\w+\{([^,]+),", bib))
    for table in re.findall(r"\\input\{([^}]+)\}", tex):
        assert (PAPER / "tables" / f"{table}.tex").exists()
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        assert (PAPER / "figures" / f"{fig}.pdf").exists()
