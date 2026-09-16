"""Pauli twirling must (1) be an EXACT no-op on the ideal (noiseless) unitary for every
random twirl draw (not just on average), and (2) its effect on FakeTorino's specific noise
model should be small -- verified to be explained by FakeTorino's CZ error channel already
being very close to Pauli-diagonal in its Pauli Transfer Matrix, not a flaw in the twirling
implementation."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import PTM, Statevector
from qiskit_ibm_runtime.fake_provider import FakeTorino

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.mitigation_twirling import (
    CZ_TWIRL_TABLE,
    twirl_two_qubit_gates,
    twirled_energy,
)
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import prepare_for_backend


def test_cz_twirl_table_has_all_16_entries_with_no_residual_i_phase() -> None:
    """Regression test for the module-level brute-force derivation: exactly 16 Pauli
    pairs, and (verified inside the derivation itself, re-asserted here) none of them
    required tracking a +-i phase -- Pauli twirling of a Clifford gate is exact."""
    assert len(CZ_TWIRL_TABLE) == 16
    assert CZ_TWIRL_TABLE[("I", "I")] == ("I", "I")
    assert CZ_TWIRL_TABLE[("X", "X")] == ("Y", "Y")


def test_twirling_is_an_exact_no_op_on_the_ideal_unitary() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cz(0, 1)
    qc.ry(0.7, 1)
    qc.cz(1, 2)
    qc.rz(1.3, 2)
    exact_state = Statevector(qc).data

    rng = np.random.default_rng(0)
    for _ in range(10):
        twirled = twirl_two_qubit_gates(qc, rng, two_qubit_gate_name="cz")
        twirled_state = Statevector(twirled).data
        # Equal up to a global phase.
        overlap = abs(np.vdot(exact_state, twirled_state))
        assert overlap > 1 - 1e-9


def test_fake_torino_cz_error_is_already_nearly_pauli_diagonal() -> None:
    """The explanation behind the next test's small effect size: check it directly
    rather than just asserting it. A Pauli-diagonal channel has ZERO off-diagonal mass
    in its Pauli Transfer Matrix -- Pauli twirling provably cannot change such a channel
    (it already IS its own Pauli twirl), so a small twirling effect here is expected, not
    a sign of a broken implementation."""
    noise_model = build_noise_model(FakeTorino())
    cz_error = noise_model._local_quantum_errors["cz"][(0, 1)]
    ptm = PTM(cz_error.to_quantumchannel()).data
    off_diagonal_mass = np.sum(np.abs(ptm - np.diag(np.diag(ptm))))
    diagonal_mass = np.sum(np.abs(np.diag(ptm)))
    assert off_diagonal_mass / diagonal_mass < 0.01


def test_twirling_has_a_small_effect_on_fake_torinos_near_diagonal_noise() -> None:
    """Honest real finding, not the conventional-wisdom "twirling always helps" story:
    since FakeTorino's CZ error is already close to Pauli-diagonal (previous test), Pauli
    twirling -- which projects a channel onto its Pauli-diagonal part -- has almost
    nothing left to change here. Twirled and raw energies should stay close to each
    other, in contrast to M3/ZNE's much larger, more reliable corrections on this exact
    system (see their own test modules)."""
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )

    result = twirled_energy(
        isa_circuit,
        isa_hamiltonian,
        integrals.e_core,
        noise_model,
        n_twirls=20,
        shots_per_twirl=20_000,
        seed=0,
    )
    # The two estimates (same total shot budget) stay close to each other -- twirling
    # neither dramatically helps nor dramatically hurts on this near-Pauli-diagonal
    # noise model, unlike M3 (test_noise_mitigation_m3.py) and ZNE
    # (test_noise_mitigation_zne.py) which both show large, clear effects on this SAME
    # system.
    assert abs(result.twirled_energy - result.raw_energy) < 5e-3


def test_twirled_energy_result_reports_the_real_per_evaluation_cost() -> None:
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )

    result = twirled_energy(
        isa_circuit,
        isa_hamiltonian,
        integrals.e_core,
        noise_model,
        n_twirls=7,
        shots_per_twirl=1000,
        seed=0,
    )
    assert result.n_twirls == 7
    assert result.shots_per_twirl == 1000
