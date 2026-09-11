"""Ground truth: hand-extracted MO integrals must reproduce PySCF's own RHF energy exactly."""

from vqe_nisq_project.chemistry.integrals import compute_mo_integrals, rhf_energy_from_integrals
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, h2


def test_rhf_energy_reconstructed_from_integrals_matches_pyscf_h2_sto3g() -> None:
    integrals = compute_mo_integrals(h2())
    reconstructed = rhf_energy_from_integrals(integrals)
    assert abs(reconstructed - integrals.e_hf) < 1e-10


def test_rhf_energy_reconstructed_from_integrals_matches_pyscf_h2_631g() -> None:
    # A richer basis (4 spatial orbitals instead of 2) so the integral tensor
    # has enough structure that a wrong AO->MO transform couldn't pass by
    # accident/degeneracy.
    spec = MoleculeSpec(atom="H 0 0 0; H 0 0 0.735", basis="6-31g")
    integrals = compute_mo_integrals(spec)
    reconstructed = rhf_energy_from_integrals(integrals)
    assert abs(reconstructed - integrals.e_hf) < 1e-10


def test_h2_sto3g_known_electron_and_orbital_count() -> None:
    integrals = compute_mo_integrals(h2())
    assert integrals.n_electrons == 2
    assert integrals.n_mo == 2
