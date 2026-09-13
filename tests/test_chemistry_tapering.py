"""Ground truth: the Z2-tapered Hamiltonian's ground state must still match (active-space)
FCI -- picking the wrong symmetry sector would silently give a physically wrong (but
plausible-looking) energy."""

import pytest

from conftest import cached_mapping as map_hamiltonian
from conftest import cached_mo_integrals as compute_mo_integrals
from conftest import cached_reference_energies as compute_reference_energies
from conftest import cached_tapered_sector
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih
from vqe_nisq_project.chemistry.tapering import find_tapered_sector

_CASES = [(h2(), 0), (lih(), 1), (beh2(), 1)]
_IDS = ["h2", "lih", "beh2"]


@pytest.mark.parametrize(("spec", "n_frozen_core"), _CASES, ids=_IDS)
def test_tapered_ground_state_matches_fci(spec: MoleculeSpec, n_frozen_core: int) -> None:
    integrals = compute_mo_integrals(spec, n_frozen_core=n_frozen_core)
    fci = compute_reference_energies(spec, n_frozen_core=n_frozen_core).fci

    result = cached_tapered_sector(spec, n_frozen_core)
    total = result.electronic_ground_state + integrals.e_core
    assert abs(total - fci) < 1e-6


@pytest.mark.parametrize(
    ("spec", "n_frozen_core", "expected_qubits_removed"),
    [(h2(), 0, 3), (lih(), 1, 4), (beh2(), 1, 5)],
    ids=_IDS,
)
def test_qubits_removed_regression(
    spec: MoleculeSpec, n_frozen_core: int, expected_qubits_removed: int
) -> None:
    # Regression check: how many Z2 symmetries qiskit's general finder
    # locates is specific to each system's Hamiltonian structure, not a
    # fixed textbook number -- verified empirically per molecule, not
    # assumed to always be the textbook particle-number+Sz-parity pair of 2.
    result = cached_tapered_sector(spec, n_frozen_core)
    assert result.n_qubits_removed == expected_qubits_removed


def test_no_matching_sector_raises() -> None:
    # 999.0 is nowhere near any of H2's sectors' ground states (verified:
    # they range roughly -1.86 to 0.0) -- an unambiguous "no physical match"
    # reference, unlike 0.0 which happens to coincide with one sector here.
    jw = map_hamiltonian(h2(), 0, "jordan_wigner")
    with pytest.raises(ValueError, match="Expected exactly one"):
        find_tapered_sector(jw.qubit_op, reference_electronic_energy=999.0)
