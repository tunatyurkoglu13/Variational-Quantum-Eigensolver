"""Ground truth: JW, Parity, and Bravyi-Kitaev must all reproduce the same FCI ground state
for H2/STO-3G, since they're just different unitary encodings of the same fermionic physics."""

import pytest

from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import (
    MappingName,
    build_electronic_fermionic_op,
    map_hamiltonian,
)
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_mapping_matches_fci(name: MappingName) -> None:
    spec = h2()
    integrals = compute_mo_integrals(spec)
    fci = compute_reference_energies(spec).fci
    fop = build_electronic_fermionic_op(integrals)

    result = map_hamiltonian(fop, name)
    total = result.electronic_ground_state + integrals.e_nuc
    assert abs(total - fci) < 1e-8


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_mapping_uses_all_four_spin_orbitals_as_qubits(name: MappingName) -> None:
    # H2/STO-3G has 4 spin-orbitals; none of these three mappings reduce qubit
    # count on their own (that's tapering.py's job).
    integrals = compute_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    result = map_hamiltonian(fop, name)
    assert result.n_qubits == 4


def test_jordan_wigner_pauli_weight_can_reach_full_register() -> None:
    # A verified, concrete number for H2/STO-3G (regression check) -- not a
    # general claim about JW scaling, which only shows up for larger N.
    integrals = compute_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    result = map_hamiltonian(fop, "jordan_wigner")
    assert result.n_terms == 15
    assert result.max_pauli_weight == 4
