"""Both hardware-efficient builds must produce valid (normalized) quantum states, and the
Qiskit build must transpile to a real device topology with a sane depth/CNOT count."""

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

from vqe_nisq_project.ansatz.hardware_efficient import (
    build_efficient_su2_qiskit,
    build_hardware_efficient_pennylane,
)
from vqe_nisq_project.ansatz.metrics import transpiled_cost


def test_qiskit_efficient_su2_produces_normalized_state() -> None:
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    rng = np.random.default_rng(0)
    state = ansatz.state_fn(rng.uniform(0, 2 * np.pi, ansatz.n_parameters))
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10


def test_pennylane_hardware_efficient_produces_normalized_state() -> None:
    ansatz = build_hardware_efficient_pennylane(4, reps=2)
    rng = np.random.default_rng(0)
    state = ansatz.state_fn(rng.uniform(0, 2 * np.pi, ansatz.n_parameters))
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10


def test_efficient_su2_transpiles_against_real_device_topology() -> None:
    ansatz = build_efficient_su2_qiskit(4, reps=2, entanglement="linear")
    cost = transpiled_cost(ansatz, backend=FakeTorino())
    assert cost.transpiled_depth > 0
    # 3 entangling layers (reps=2 -> reps+1... but skip_final handled internally) worth
    # of linear-entanglement CNOTs on 4 qubits -- transpiled onto FakeTorino's native
    # 2-qubit gate, should be a small positive number, not zero and not huge.
    assert 0 < cost.transpiled_two_qubit_gate_count < 50
    assert cost.backend_name == "fake_torino"
