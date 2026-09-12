"""Ground truth: the Z2-tapered Hamiltonian's ground state must still match FCI -- picking the
wrong symmetry sector would silently give a physically wrong (but plausible-looking) energy."""

import pytest

from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies
from vqe_nisq_project.chemistry.tapering import find_tapered_sector


def _h2_jw_setup():
    spec = h2()
    integrals = compute_mo_integrals(spec)
    fci = compute_reference_energies(spec).fci
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")
    return integrals, fci, jw


def test_tapered_ground_state_matches_fci() -> None:
    integrals, fci, jw = _h2_jw_setup()
    electronic_fci = fci - integrals.e_nuc

    result = find_tapered_sector(jw.qubit_op, electronic_fci)
    total = result.electronic_ground_state + integrals.e_nuc
    assert abs(total - fci) < 1e-8


def test_h2_sto3g_removes_three_qubits() -> None:
    # Regression check for this specific system: qiskit's general Z2Symmetries
    # finder locates 3 independent symmetries here (more than the textbook
    # particle-number + Sz-parity pair that gives the well-known 4->2 qubit
    # reduction) -- plausibly extra structure specific to this minimal-basis,
    # 2-orbital system rather than a generally-expected result for larger
    # molecules. Verified empirically, not assumed.
    integrals, fci, jw = _h2_jw_setup()
    electronic_fci = fci - integrals.e_nuc

    result = find_tapered_sector(jw.qubit_op, electronic_fci)
    assert result.n_qubits_removed == 3
    assert result.tapered_op.num_qubits == 1


def test_no_matching_sector_raises() -> None:
    # 999.0 is nowhere near any of the 8 sectors' ground states for this
    # system (verified: they range roughly -1.86 to 0.0) -- an unambiguous
    # "no physical match" reference, unlike 0.0 which happens to coincide
    # with one sector's actual ground state here.
    _integrals, _fci, jw = _h2_jw_setup()
    with pytest.raises(ValueError, match="Expected exactly one"):
        find_tapered_sector(jw.qubit_op, reference_electronic_energy=999.0)
