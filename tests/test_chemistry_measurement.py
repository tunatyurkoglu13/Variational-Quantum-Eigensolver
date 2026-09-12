"""Ground truth: qubit-wise-commuting groups must fully partition the Hamiltonian's Pauli
terms (every term appears exactly once, in exactly one group) -- this count is what actually
sets the number of distinct measurement circuits (shot cost) a real run needs."""

from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.measurement import group_qubit_wise_commuting
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies
from vqe_nisq_project.chemistry.tapering import find_tapered_sector


def test_groups_partition_all_terms_exactly() -> None:
    integrals = compute_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")

    result = group_qubit_wise_commuting(jw.qubit_op)
    assert sum(len(g) for g in result.groups) == result.n_terms


def test_full_jw_hamiltonian_needs_five_measurement_groups() -> None:
    # Regression check for this specific system.
    integrals = compute_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")

    result = group_qubit_wise_commuting(jw.qubit_op)
    assert result.n_terms == 15
    assert result.n_groups == 5


def test_tapering_reduces_measurement_groups_too() -> None:
    # Tapering doesn't just cut qubits -- it should also cut (or at least not
    # increase) the number of distinct measurement circuits needed.
    spec = h2()
    integrals = compute_mo_integrals(spec)
    fci = compute_reference_energies(spec).fci
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")

    tapered = find_tapered_sector(jw.qubit_op, fci - integrals.e_nuc)

    full_groups = group_qubit_wise_commuting(jw.qubit_op).n_groups
    tapered_groups = group_qubit_wise_commuting(tapered.tapered_op).n_groups
    assert tapered_groups <= full_groups
