import numpy as np
import pytest

from llmatter2d.ingest.arpes import (
    ARPESMap,
    features_from_map,
    load_arpes_map,
    mdc_dispersion,
)
from llmatter2d.ingest.seed import SeedConnector
from llmatter2d.ingest.wannier import bloch_hamiltonian, read_wannier90_hr, write_wannier90_hr

HBAR = 6.582119569e-16


def graphene_model():
    return next(m for m in SeedConnector().fetch() if m.uid == "seed:graphene").tight_binding[0]


def test_wannier_roundtrip_and_dirac_point(tmp_path):
    model = graphene_model()
    path = tmp_path / "graphene_hr.dat"
    write_wannier90_hr(model, path)
    back = read_wannier90_hr(path, orbital_labels=["pz_A", "pz_B"])
    assert len(back.hoppings) == 6
    assert {abs(h.t_re) for h in back.hoppings} == {2.8}
    # 60-degree cell -> reciprocal vectors at 120 degrees, K = (2 b1 + b2)/3: Dirac point
    E = np.linalg.eigvalsh(bloch_hamiltonian(back, np.array([2 / 3, 1 / 3, 0])))
    assert np.allclose(E, 0, atol=1e-9)
    E_gamma = np.linalg.eigvalsh(bloch_hamiltonian(back, np.zeros(3)))
    assert np.allclose(sorted(E_gamma), [-8.4, 8.4])


def test_wannier_degeneracy_weights(tmp_path):
    text = "\n".join([
        "test", "1", "2", "1 2",
        "    0    0    0    1    1    0.500000    0.000000",
        "    1    0    0    1    1   -2.000000    0.000000",
    ])
    path = tmp_path / "x_hr.dat"
    path.write_text(text)
    model = read_wannier90_hr(path)
    assert model.onsite_energies() == [0.5]
    assert model.largest_hoppings(1)[0].t_re == pytest.approx(-1.0)


def synthetic_dirac_map(v=1.0e6, e_dirac=-0.4, broadening=0.01):
    """n-doped Dirac cone (as graphene on SiC): Dirac point below E_F."""
    k = np.linspace(-0.2, 0.2, 401)
    E = np.linspace(-0.8, 0.05, 426)
    hv = HBAR * v / 1e-10  # eV angstrom
    I = np.zeros((E.size, k.size))
    for s in (+1, -1):
        k0 = s * np.abs(E - e_dirac) / hv
        I += 1.0 / (((k[None, :] - k0[:, None]) / broadening) ** 2 + 1)
    I *= (E <= 0)[:, None]  # Fermi cutoff
    return ARPESMap(k, E, I)


def test_fermi_velocity_from_dirac_cone():
    feats = features_from_map(synthetic_dirac_map(), k_range=(0.0, 0.2), kind="linear",
                              window=0.2, k_point="K")
    assert feats.fermi_velocity_m_s == pytest.approx(1.0e6, rel=0.02)
    assert feats.k_point == "K"


def test_effective_mass_from_parabolic_band(tmp_path):
    m_star = 0.5
    k = np.linspace(-0.15, 0.15, 151)
    E = np.linspace(-0.5, 0.0, 251)
    band = -0.3 + 3.80998212 * k**2 / m_star  # electron pocket
    I = np.exp(-((E[:, None] - band[None, :]) / 0.01) ** 2)
    path = tmp_path / "map.npz"
    np.savez(path, k=k, energy=E, intensity=I)
    feats = features_from_map(load_arpes_map(path), kind="parabolic")
    assert feats.effective_mass_me == pytest.approx(m_star, rel=0.03)


def test_mdc_dispersion_branch_selection():
    Es, ks = mdc_dispersion(synthetic_dirac_map(), k_range=(-0.2, 0.0), e_range=(-0.3, 0.0))
    assert np.all(ks < 0)
