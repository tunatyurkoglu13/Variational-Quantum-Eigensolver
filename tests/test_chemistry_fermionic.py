"""Ground truth: qiskit-nature and OpenFermion must build the same electronic Hamiltonian
from the same PySCF-derived integrals, and its exact-diagonalized ground state (plus the
classical core energy) must match PySCF's own (active-space) FCI energy."""

import pytest

from conftest import cached_cross_validate as cross_validate
from conftest import cached_mo_integrals as compute_mo_integrals
from conftest import cached_reference_energies as compute_reference_energies
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih

_H2_631G = MoleculeSpec(atom="H 0 0 0; H 0 0 0.735", basis="6-31g")

# (spec, n_frozen_core) pairs covering the full space (H2) and the frozen-core
# active-space case (LiH, BeH2).
_CASES = [
    (h2(), 0),
    (_H2_631G, 0),
    (lih(), 1),
    (beh2(), 1),
]
_IDS = ["h2-sto3g", "h2-631g", "lih", "beh2"]


@pytest.mark.parametrize(("spec", "n_frozen_core"), _CASES, ids=_IDS)
def test_qiskit_nature_and_openfermion_agree(spec: MoleculeSpec, n_frozen_core: int) -> None:
    result = cross_validate(spec, n_frozen_core)
    assert result.max_eigenvalue_diff < 1e-10


@pytest.mark.parametrize(("spec", "n_frozen_core"), _CASES, ids=_IDS)
def test_electronic_ground_state_plus_core_energy_matches_fci(
    spec: MoleculeSpec, n_frozen_core: int
) -> None:
    integrals = compute_mo_integrals(spec, n_frozen_core=n_frozen_core)
    result = cross_validate(spec, n_frozen_core)
    fci_reference = compute_reference_energies(spec, n_frozen_core=n_frozen_core).fci

    total = result.electronic_ground_state + integrals.e_core
    assert abs(total - fci_reference) < 1e-8
