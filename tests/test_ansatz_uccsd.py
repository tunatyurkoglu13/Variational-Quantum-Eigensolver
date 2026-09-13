"""Ground truth: UCCSD at theta=0 must reproduce RHF (it's exactly the HF reference
circuit with all excitation amplitudes off), and optimizing it must reach FCI in both
frameworks -- the actual criterion that matters for VQE, since qiskit-nature and
PennyLane order spin-orbitals into qubits differently (see uccsd.py's module comment)."""

import numpy as np
from scipy.optimize import minimize

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.uccsd import (
    H2_PENNYLANE_TO_QISKIT_NATURE_QUBIT_PERM,
    build_uccsd_pennylane,
    build_uccsd_qiskit,
    permute_statevector_qubits,
)
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2


def _h2_jw_hamiltonian_and_integrals():
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")
    return jw.qubit_op.to_matrix(), integrals


def test_qiskit_uccsd_at_zero_params_matches_rhf() -> None:
    hamiltonian, integrals = _h2_jw_hamiltonian_and_integrals()
    qk = build_uccsd_qiskit(2, (1, 1))
    state = qk.state_fn(np.zeros(qk.n_parameters))
    energy = (state.conj() @ hamiltonian @ state).real + integrals.e_nuc
    assert abs(energy - integrals.e_hf) < 1e-8


def test_pennylane_uccsd_at_zero_params_matches_rhf_after_qubit_realignment() -> None:
    hamiltonian, integrals = _h2_jw_hamiltonian_and_integrals()
    pl = build_uccsd_pennylane(2, (1, 1))
    state = pl.state_fn(np.zeros(pl.n_parameters))
    realigned = permute_statevector_qubits(state, H2_PENNYLANE_TO_QISKIT_NATURE_QUBIT_PERM)
    energy = (realigned.conj() @ hamiltonian @ realigned).real + integrals.e_nuc
    assert abs(energy - integrals.e_hf) < 1e-8


def test_both_frameworks_optimize_uccsd_to_fci() -> None:
    hamiltonian, integrals = _h2_jw_hamiltonian_and_integrals()
    fci = cached_reference_energies(h2()).fci

    qk = build_uccsd_qiskit(2, (1, 1))
    pl = build_uccsd_pennylane(2, (1, 1))

    def qk_energy(params: np.ndarray) -> float:
        state = qk.state_fn(params)
        return float((state.conj() @ hamiltonian @ state).real) + integrals.e_nuc

    def pl_energy(params: np.ndarray) -> float:
        state = permute_statevector_qubits(
            pl.state_fn(params), H2_PENNYLANE_TO_QISKIT_NATURE_QUBIT_PERM
        )
        return float((state.conj() @ hamiltonian @ state).real) + integrals.e_nuc

    res_qk = minimize(qk_energy, np.zeros(qk.n_parameters), method="COBYLA")
    res_pl = minimize(pl_energy, np.zeros(pl.n_parameters), method="COBYLA")

    assert abs(res_qk.fun - fci) < 1e-6
    assert abs(res_pl.fun - fci) < 1e-6
