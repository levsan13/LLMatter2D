import numpy as np
import pytest

from llmatter2d.lattices import (
    ARCHIMEDEAN,
    analyze_bands,
    archimedean_lattice,
    band_structure,
    read_poscar,
    write_poscar,
)
from llmatter2d.lattices.tightbinding import bands_at, high_symmetry_points

# (sites per primitive cell, coordination) for the 11 Archimedean tilings
EXPECTED = {
    "3^6": (1, 6), "4^4": (1, 4), "6^3": (2, 3), "3^4.6": (6, 5), "3^3.4^2": (2, 5),
    "3^2.4.3.4": (4, 5), "3.4.6.4": (6, 4), "3.6.3.6": (3, 4), "3.12^2": (6, 3),
    "4.6.12": (12, 3), "4.8^2": (4, 3),
}


def test_catalogue_is_complete():
    assert set(ARCHIMEDEAN) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_tiling_geometry(name):
    lat = ARCHIMEDEAN[name]
    n, z = EXPECTED[name]
    assert lat.n_sites == n
    assert np.all(lat.coordination() == z)
    assert len(lat.vertex_config) == z
    lengths = [np.linalg.norm(lat.bond_vector(b)) for b in lat.bonds]
    assert np.allclose(lengths, 1.0)
    # no two sites closer than a bond
    assert lat.bond_length == pytest.approx(1.0)


def test_polygon_density_identity():
    # sites per unit area follows from the vertex configuration:
    # 1/rho = sum over polygons at a vertex of (area of n-gon)/n
    def ngon_area(k):
        return k / (4 * np.tan(np.pi / k))

    for lat in ARCHIMEDEAN.values():
        expected = 1 / sum(ngon_area(k) / k for k in lat.vertex_config)
        assert lat.n_sites / lat.area == pytest.approx(expected, rel=1e-9)


def test_aliases():
    assert archimedean_lattice("kagome").name == "3.6.3.6"
    assert archimedean_lattice("3.3.3.3.6").name == "3^4.6"
    with pytest.raises(KeyError):
        archimedean_lattice("penrose")


def test_bipartite_detection():
    assert ARCHIMEDEAN["6^3"].sublattice_coloring() is not None
    assert ARCHIMEDEAN["4.6.12"].sublattice_coloring() is not None
    assert ARCHIMEDEAN["3.6.3.6"].sublattice_coloring() is None


def test_honeycomb_dirac_cone():
    lat = ARCHIMEDEAN["6^3"]
    K = high_symmetry_points(lat)["K"]
    assert np.allclose(bands_at(lat, K, t=2.8), 0, atol=1e-9)
    f = analyze_bands(lat)
    assert any(s.startswith("K:") for s in f.linear_crossings)
    assert not f.flat_bands


def test_kagome_flat_band_at_plus_2t():
    f = analyze_bands(ARCHIMEDEAN["3.6.3.6"], t=1.5)
    assert f.flat_bands == [pytest.approx(3.0)]
    assert f.energy_range == (pytest.approx(-6.0), pytest.approx(3.0))


def test_star_lattice_has_two_flat_bands():
    assert analyze_bands(ARCHIMEDEAN["3.12^2"]).flat_bands == [pytest.approx(0.0),
                                                                pytest.approx(2.0)]


def test_square_band_bottom():
    bs = band_structure(ARCHIMEDEAN["4^4"], t=1.0)
    assert bs.labels == ["G", "X", "M", "Y", "G"]
    assert bs.energies.min() == pytest.approx(-4.0)


def test_decoration_and_poscar_roundtrip(tmp_path):
    lat = ARCHIMEDEAN["4.6.12"]
    s = lat.to_structure(("B", "N"), bond_length=1.45, buckling=0.2)
    assert s.formula == "BN" and len(s.sites) == 12
    assert s.thickness == pytest.approx(0.2)
    path = tmp_path / "POSCAR"
    write_poscar(s, path)
    back = read_poscar(path)
    assert back.formula == "BN"
    # POSCAR groups sites by species, so compare as sets of (species, position)
    def key(st):
        return sorted((sp, *np.round(p, 6)) for sp, p in zip(st.species, st.positions))
    assert key(back) == key(s)
    with pytest.raises(ValueError):
        ARCHIMEDEAN["3.6.3.6"].to_structure(("B", "N"), 1.5)
