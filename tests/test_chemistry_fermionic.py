"""Ground truth: qiskit-nature and OpenFermion must build the same electronic Hamiltonian
from the same PySCF-derived integrals, and its exact-diagonalized ground state (plus the
classical nuclear repulsion energy) must match PySCF's own FCI energy."""

import pytest

from vqe_nisq_project.chemistry.fermionic import cross_validate
from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies

_H2_631G = MoleculeSpec(atom="H 0 0 0; H 0 0 0.735", basis="6-31g")


@pytest.mark.parametrize("spec", [h2(), _H2_631G], ids=["sto-3g", "6-31g"])
def test_qiskit_nature_and_openfermion_agree(spec: MoleculeSpec) -> None:
    integrals = compute_mo_integrals(spec)
    result = cross_validate(integrals)
    assert result.max_eigenvalue_diff < 1e-10


@pytest.mark.parametrize("spec", [h2(), _H2_631G], ids=["sto-3g", "6-31g"])
def test_electronic_ground_state_plus_nuclear_repulsion_matches_fci(spec: MoleculeSpec) -> None:
    integrals = compute_mo_integrals(spec)
    result = cross_validate(integrals)
    fci_reference = compute_reference_energies(spec).fci

    total = result.electronic_ground_state + integrals.e_nuc
    assert abs(total - fci_reference) < 1e-8
