import json

import numpy as np
import pytest

from llmatter2d.ingest import JarvisConnector, OptimadeConnector, TwoDMatPediaConnector
from llmatter2d.ingest.optimade import OptimadeClient

# Shape of a real MC2D OPTIMADE entry (fields as served by optimade.materialscloud.org)
MC2D_PAGE_1 = {
    "data": [{
        "id": "77fafc6f",
        "type": "structures",
        "attributes": {
            "chemical_formula_reduced": "I2La",
            "dimension_types": [1, 1, 0],
            "lattice_vectors": [[3.9536, 0.0, 0.0], [0.0, 3.9536, 0.0], [0.0, 0.0, 23.84]],
            "cartesian_site_positions": [[1.977, 1.977, 10.02], [1.977, 1.977, 13.82],
                                         [0.0, 0.0, 11.92]],
            "species_at_sites": ["I", "I", "La"],
            "species": [{"name": "I", "chemical_symbols": ["I"]},
                        {"name": "La", "chemical_symbols": ["La"]}],
            "_mcloud_band_gap": 0.0,
        },
    }],
    "links": {"next": "https://example.org/v1/structures?page_offset=1"},
}
MC2D_PAGE_2 = {
    "data": [{
        "id": "abc",
        "attributes": {
            "chemical_formula_reduced": "BN",
            "lattice_vectors": [[2.5, 0, 0], [1.25, 2.165, 0], [None, None, None]],
            "cartesian_site_positions": [[0, 0, 0], [1.25, 0.7217, 0]],
            "species_at_sites": ["B", "N"],
            "_mcloud_band_gap": 4.6,
        },
    }],
    "links": {"next": None},
}


class FakeClient(OptimadeClient):
    def __init__(self):
        super().__init__("https://example.org")
        self.pages = [MC2D_PAGE_1, MC2D_PAGE_2]

    def _get(self, url):
        return self.pages.pop(0)


def test_optimade_pagination_and_mapping():
    mats = list(OptimadeConnector("mc2d", client=FakeClient()).fetch())
    assert [m.uid for m in mats] == ["mc2d:77fafc6f", "mc2d:abc"]
    la = mats[0]
    assert la.formula == "I2La" and la.electronic.is_metal is True
    bn = mats[1]
    assert bn.electronic.band_gaps[0].value_eV == pytest.approx(4.6)
    # missing (null) vacuum vector is rebuilt perpendicular to the layer
    assert np.allclose(bn.structure.cell[2][:2], 0) and bn.structure.cell[2][2] > 10


def test_optimade_url_building():
    url = OptimadeClient("https://optimade.materialscloud.org/main/mc2d").structures_url(
        filter='elements HAS "Mo"', response_fields=["lattice_vectors"], page_limit=5)
    assert url.startswith("https://optimade.materialscloud.org/main/mc2d/v1/structures?")
    assert "page_limit=5" in url and "response_fields=lattice_vectors" in url


def test_2dmatpedia_jsonl(tmp_path):
    rec = {
        "material_id": "2dm-1", "bandgap": 1.2, "exfoliation_energy_per_atom": 0.08,
        "decomposition_energy": 0.05, "total_magnetization": 0.0,
        "discovery_process": "top-down",
        "structure": {"lattice": {"matrix": [[3.2, 0, 0], [-1.6, 2.77, 0], [0, 0, 20]]},
                      "sites": [{"species": [{"element": "Mo", "occu": 1}], "xyz": [0, 0, 10]},
                                {"species": [{"element": "S", "occu": 1}],
                                 "xyz": [0, 1.85, 11.56]},
                                {"species": [{"element": "S", "occu": 1}],
                                 "xyz": [0, 1.85, 8.44]}]},
    }
    path = tmp_path / "db.json"
    path.write_text(json.dumps(rec) + "\n")
    (m,) = TwoDMatPediaConnector(path).fetch()
    assert m.uid == "2dmatpedia:2dm-1" and m.formula == "MoS2"
    assert m.stability.exfoliation_energy_meV_atom == pytest.approx(80)
    assert m.electronic.magnetic_state == "NM"
    assert "discovery:top-down" in m.tags


def test_jarvis_records_with_na_values():
    rec = {"jid": "JVASP-1", "formula": "C",
           "atoms": {"lattice_mat": [[2.46, 0, 0], [-1.23, 2.13, 0], [0, 0, 20]],
                     "coords": [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]], "elements": ["C", "C"],
                     "cartesian": False},
           "optb88vdw_bandgap": 0.0, "mbj_bandgap": "na", "hse_gap": "na",
           "formation_energy_peratom": 0.0, "exfoliation_energy": 70.0, "spillage": "na"}
    (m,) = JarvisConnector(records=[rec]).fetch()
    assert m.uid == "jarvis:JVASP-1"
    assert len(m.electronic.band_gaps) == 1
    d = np.linalg.norm(m.structure.positions[1] - m.structure.positions[0])
    assert d == pytest.approx(1.42, abs=0.01)
