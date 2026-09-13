"""Ground truth: UCCSD at theta=0 must reproduce RHF (it's exactly the HF reference
circuit with all excitation amplitudes off), and optimizing it must reach FCI in both
frameworks -- the actual criterion that matters for VQE, since qiskit-nature and
PennyLane order spin-orbitals into qubits differently AND disagree on statevector
endianness (see uccsd.py's module comment for both, each verified empirically)."""

import numpy as np
import pytest
from scipy.optimize import minimize

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.uccsd import (
    build_uccsd_pennylane,
    build_uccsd_qiskit,
    realign_pennylane_state_to_qiskit_nature,
)
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih

# (spec, n_frozen_core, num_spatial_orbitals, num_particles)
_CASES = [
    (h2(), 0, 2, (1, 1)),
    (lih(), 1, 5, (1, 1)),
    (beh2(), 1, 6, (2, 2)),
]
_IDS = ["h2", "lih", "beh2"]


def _jw_hamiltonian_and_integrals(spec: MoleculeSpec, n_frozen_core: int):
    integrals = cached_mo_integrals(spec, n_frozen_core)
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")
    return jw.qubit_op.to_matrix(), integrals


def _expectation(state: np.ndarray, hamiltonian: np.ndarray) -> float:
    # Same benign macOS Accelerate BLAS quirk documented in ansatz/adapt.py
    # (spurious divide-by-zero/invalid-value warnings on some dense complex
    # matrix products, verified harmless there) -- also shows up here on
    # LiH/BeH2's larger dense Hamiltonians.
    with np.errstate(all="ignore"):
        return float((state.conj() @ hamiltonian @ state).real)


@pytest.mark.parametrize(
    ("spec", "n_frozen_core", "num_spatial_orbitals", "num_particles"), _CASES, ids=_IDS
)
def test_qiskit_uccsd_at_zero_params_matches_rhf(
    spec: MoleculeSpec, n_frozen_core: int, num_spatial_orbitals: int, num_particles: tuple
) -> None:
    hamiltonian, integrals = _jw_hamiltonian_and_integrals(spec, n_frozen_core)
    qk = build_uccsd_qiskit(num_spatial_orbitals, num_particles)
    state = qk.state_fn(np.zeros(qk.n_parameters))
    energy = _expectation(state, hamiltonian) + integrals.e_core
    assert abs(energy - integrals.e_hf) < 1e-7


@pytest.mark.parametrize(
    ("spec", "n_frozen_core", "num_spatial_orbitals", "num_particles"), _CASES, ids=_IDS
)
def test_pennylane_uccsd_at_zero_params_matches_rhf_after_realignment(
    spec: MoleculeSpec, n_frozen_core: int, num_spatial_orbitals: int, num_particles: tuple
) -> None:
    hamiltonian, integrals = _jw_hamiltonian_and_integrals(spec, n_frozen_core)
    pl = build_uccsd_pennylane(num_spatial_orbitals, num_particles)
    state = pl.state_fn(np.zeros(pl.n_parameters))
    realigned = realign_pennylane_state_to_qiskit_nature(state, num_spatial_orbitals)
    energy = _expectation(realigned, hamiltonian) + integrals.e_core
    assert abs(energy - integrals.e_hf) < 1e-7


def test_both_frameworks_optimize_uccsd_to_fci_h2() -> None:
    # Full re-optimization (COBYLA) is only run for H2 to keep the suite fast;
    # LiH/BeH2's HF-reference-point agreement is already checked above, which
    # is what changes (unmodified UCCSD/PennyLane machinery) across systems --
    # the actual optimizer behavior is Phase 4's subject, not this phase's.
    hamiltonian, integrals = _jw_hamiltonian_and_integrals(h2(), 0)
    fci = cached_reference_energies(h2()).fci

    qk = build_uccsd_qiskit(2, (1, 1))
    pl = build_uccsd_pennylane(2, (1, 1))

    def qk_energy(params: np.ndarray) -> float:
        state = qk.state_fn(params)
        return _expectation(state, hamiltonian) + integrals.e_core

    def pl_energy(params: np.ndarray) -> float:
        state = realign_pennylane_state_to_qiskit_nature(pl.state_fn(params), 2)
        return _expectation(state, hamiltonian) + integrals.e_core

    res_qk = minimize(qk_energy, np.zeros(qk.n_parameters), method="COBYLA")
    res_pl = minimize(pl_energy, np.zeros(pl.n_parameters), method="COBYLA")

    assert abs(res_qk.fun - fci) < 1e-6
    assert abs(res_pl.fun - fci) < 1e-6
