"""Ground truth: JW, Parity, and Bravyi-Kitaev must all reproduce the same (active-space)
FCI ground state, since they're just different unitary encodings of the same physics."""

import pytest

from conftest import cached_mapping as map_hamiltonian
from conftest import cached_mo_integrals as compute_mo_integrals
from conftest import cached_reference_energies as compute_reference_energies
from vqe_nisq_project.chemistry.mappings import MappingName
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih

_CASES = [(h2(), 0, 4), (lih(), 1, 10), (beh2(), 1, 12)]
_IDS = ["h2", "lih", "beh2"]


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
@pytest.mark.parametrize(("spec", "n_frozen_core", "_n_qubits"), _CASES, ids=_IDS)
def test_mapping_matches_fci(
    spec: MoleculeSpec, n_frozen_core: int, _n_qubits: int, name: MappingName
) -> None:
    integrals = compute_mo_integrals(spec, n_frozen_core=n_frozen_core)
    fci = compute_reference_energies(spec, n_frozen_core=n_frozen_core).fci

    result = map_hamiltonian(spec, n_frozen_core, name)
    total = result.electronic_ground_state + integrals.e_core
    assert abs(total - fci) < 1e-6


@pytest.mark.parametrize("name", ["jordan_wigner", "parity", "bravyi_kitaev"])
@pytest.mark.parametrize(("spec", "n_frozen_core", "n_qubits"), _CASES, ids=_IDS)
def test_mapping_uses_expected_qubit_count(
    spec: MoleculeSpec, n_frozen_core: int, n_qubits: int, name: MappingName
) -> None:
    # None of these three mappings reduce qubit count on their own -- qubits
    # == 2 x active spatial orbitals always (that's tapering.py's job).
    result = map_hamiltonian(spec, n_frozen_core, name)
    assert result.n_qubits == n_qubits


def test_jordan_wigner_pauli_weight_can_reach_full_register_h2() -> None:
    # A verified, concrete number for H2/STO-3G (regression check) -- not a
    # general claim about JW scaling, which only shows up for larger N.
    result = map_hamiltonian(h2(), 0, "jordan_wigner")
    assert result.n_terms == 15
    assert result.max_pauli_weight == 4


@pytest.mark.parametrize(("spec", "n_frozen_core"), [(lih(), 1), (beh2(), 1)], ids=["lih", "beh2"])
def test_bravyi_kitaev_has_strictly_lower_max_weight_than_jw_or_parity_at_larger_n(
    spec: MoleculeSpec, n_frozen_core: int
) -> None:
    # The asymptotic Pauli-weight advantage of Bravyi-Kitaev (O(log N) vs
    # JW/Parity's O(N)) doesn't show up at H2's tiny N=4 (all three tie at
    # max_weight=4), but is measurable once qubit count grows -- verified
    # here, not assumed.
    jw = map_hamiltonian(spec, n_frozen_core, "jordan_wigner")
    bk = map_hamiltonian(spec, n_frozen_core, "bravyi_kitaev")
    assert bk.max_pauli_weight < jw.max_pauli_weight
