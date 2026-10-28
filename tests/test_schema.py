import pytest

from llmatter2d.ingest.seed import SeedConnector
from llmatter2d.schema import (
    BandGap,
    ElectronicProperties,
    Material2D,
    Provenance,
    Site,
    Structure2D,
    load_materials,
    parse_formula,
    reduced_formula,
    save_materials,
)


def test_reduced_formula_keeps_order_of_appearance():
    assert reduced_formula(["Mo", "S", "S"]) == "MoS2"
    assert reduced_formula(["C", "C"]) == "C"
    assert reduced_formula(["B", "N", "B", "N"]) == "BN"


def test_parse_formula():
    assert parse_formula("Fe3GeTe2") == {"Fe": 3, "Ge": 1, "Te": 2}


def test_structure_validation():
    with pytest.raises(ValueError):
        Structure2D(lattice=[[1, 0], [0, 1]], sites=[Site(species="C", position=(0, 0, 0))])


def test_seed_structures_have_expected_geometry():
    mats = {m.uid: m for m in SeedConnector().fetch()}
    graphene = mats["seed:graphene"].structure
    assert graphene.bravais_lattice() == "hexagonal"
    assert graphene.a == pytest.approx(2.46)
    mos2 = mats["seed:MoS2-1H"].structure
    assert mos2.formula == "MoS2"
    assert mos2.thickness == pytest.approx(3.12, abs=1e-6)
    assert mats["seed:phosphorene"].structure.bravais_lattice() == "rectangular"


def test_merge_combines_sources():
    a = Material2D(uid="x", formula="MoS2",
                   electronic=ElectronicProperties(band_gaps=[BandGap(value_eV=1.7)]),
                   provenance=[Provenance(source="A")])
    b = Material2D(uid="y", formula="MoS2", name="molybdenite monolayer",
                   electronic=ElectronicProperties(
                       band_gaps=[BandGap(value_eV=2.1, method="DFT-HSE06")]),
                   provenance=[Provenance(source="B")])
    m = a.merge(b)
    assert m.uid == "x" and m.name == "molybdenite monolayer"
    assert [g.value_eV for g in m.electronic.band_gaps] == [1.7, 2.1]
    assert {p.source for p in m.provenance} == {"A", "B"}


def test_jsonl_roundtrip(tmp_path):
    mats = list(SeedConnector().fetch())
    path = tmp_path / "m.jsonl"
    assert save_materials(path, mats) == len(mats)
    back = list(load_materials(path))
    assert [m.uid for m in back] == [m.uid for m in mats]
    assert back[0].tight_binding[0].hoppings == mats[0].tight_binding[0].hoppings
