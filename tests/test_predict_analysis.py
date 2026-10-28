import json

import pytest

from llmatter2d.analysis import analyze_structure, archimedean_match, describe_structure
from llmatter2d.cli import main
from llmatter2d.lattices import ARCHIMEDEAN
from llmatter2d.llm import get_llm
from llmatter2d.predict import (
    enumerate_candidates,
    review_with_llm,
    score_candidates,
    write_candidates,
)
from llmatter2d.rag import KnowledgeBase, RAGPipeline


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_seed()


def test_enumeration_counts():
    cands = enumerate_candidates(["B", "C", "N"])
    bipartite = [lat for lat in ARCHIMEDEAN.values() if lat.sublattice_coloring() is not None]
    assert len(cands) == 3 * len(ARCHIMEDEAN) + 3 * len(bipartite)


def test_known_materials_are_flagged(kb):
    cands = score_candidates(enumerate_candidates(["C", "B", "N"]), kb=kb)
    by_uid = {c.uid: c for c in cands}
    assert by_uid["candidate:C@6^3"].known_as == "seed:graphene"
    assert by_uid["candidate:BN@6^3"].known_as == "seed:hBN"
    assert by_uid["candidate:C@4.6.12"].known_as == "seed:graphenylene"
    assert by_uid["candidate:C@3.6.3.6"].known_as is None


def test_scoring_prefers_plausible_coordination(kb):
    cands = score_candidates(enumerate_candidates(["C"]), kb=kb)
    z_of = {c.lattice.name: int(c.lattice.coordination()[0]) for c in cands}
    # carbon on 6-fold triangular net is penalised relative to 3-fold nets
    scores = {c.lattice.name: c.score for c in cands}
    assert scores["3^6"] < scores["4.6.12"]
    assert all(z_of[c.lattice.name] in (3, 4) for c in cands[:3])


def test_write_and_review(kb, tmp_path):
    cands = score_candidates(enumerate_candidates(["B", "N"], lattices=["4.6.12", "kagome"]),
                             kb=kb)
    summary = write_candidates(cands[:3], tmp_path)
    data = json.loads(summary.read_text())
    assert len(data) == 3 and (tmp_path / data[0]["file"]).exists()
    ans = review_with_llm(cands, RAGPipeline(kb, get_llm("offline"), k=3), top=3)
    assert "<candidates>" in ans.prompt and "candidate:" in ans.prompt


def test_archimedean_match_on_known_structures(kb):
    assert archimedean_match(kb.get("seed:graphene").structure) == ["6^3"]
    assert archimedean_match(kb.get("seed:silicene").structure) == ["6^3"]  # buckled
    # MoS2 projects onto a honeycomb (Mo + S2 columns)
    assert archimedean_match(kb.get("seed:MoS2-1H").structure) == ["6^3"]
    kagome = ARCHIMEDEAN["3.6.3.6"].to_structure("Fe", 2.5)
    assert archimedean_match(kagome) == ["3.6.3.6"]


def test_supercell_invariance():
    s = ARCHIMEDEAN["4.8^2"].to_structure("C", 1.42)
    import numpy as np

    from llmatter2d.schema import Site, Structure2D

    cell = s.cell.copy()
    sites = [Site(species=x.species, position=tuple(np.add(x.position, shift)))
             for shift in ([0, 0, 0], cell[0]) for x in s.sites]
    big = Structure2D(lattice=[list(2 * cell[0]), list(cell[1]), list(cell[2])], sites=sites)
    assert archimedean_match(big) == ["4.8^2"]


def test_analyze_new_structure(kb):
    s = ARCHIMEDEAN["4.6.12"].to_structure(("B", "N"), 1.45)
    report = analyze_structure(s, RAGPipeline(kb, get_llm("offline"), k=4))
    assert report.archimedean_matches == ["4.6.12"]
    assert report.coordination == {"B": [3] * 6, "N": [3] * 6}
    assert "<analysis>" in report.answer.prompt
    assert describe_structure(s).formula == "BN"


def test_cli_smoke(tmp_path, capsys):
    main(["archimedean"])
    assert "kagome" in capsys.readouterr().out
    main(["archimedean", "kagome", "--species", "Fe", "--bond", "2.5",
          "--poscar", str(tmp_path / "k.vasp"), "--bands", str(tmp_path / "b.csv")])
    out = capsys.readouterr().out
    assert "flat band" in out and (tmp_path / "k.vasp").exists()
    main(["index", "--seed", "-o", str(tmp_path / "kb")])
    main(["ask", "Which materials are magnetic?", "--kb", str(tmp_path / "kb"), "--magnetic",
          "--llm", "offline"])
    assert "seed:CrI3" in capsys.readouterr().out
    main(["predict", "--elements", "C", "Si", "--top", "3", "--kb", str(tmp_path / "kb"),
          "-o", str(tmp_path / "cands"), "--llm", "offline"])
    assert (tmp_path / "cands" / "candidates.json").exists()
