"""Both hardware-efficient builds must produce valid (normalized) quantum states at any
system size, and the Qiskit build must transpile to a real device topology with a sane
depth/CNOT count."""

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeTorino

from vqe_nisq_project.ansatz.hardware_efficient import (
    build_efficient_su2_qiskit,
    build_hardware_efficient_pennylane,
)
from vqe_nisq_project.ansatz.metrics import transpiled_cost

_QUBIT_COUNTS = [4, 10, 12]  # H2, LiH, BeH2 (2 x active spatial orbitals)
_IDS = ["h2", "lih", "beh2"]


@pytest.mark.parametrize("n_qubits", _QUBIT_COUNTS, ids=_IDS)
def test_qiskit_efficient_su2_produces_normalized_state(n_qubits: int) -> None:
    ansatz = build_efficient_su2_qiskit(n_qubits, reps=2)
    rng = np.random.default_rng(0)
    state = ansatz.state_fn(rng.uniform(0, 2 * np.pi, ansatz.n_parameters))
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10


@pytest.mark.parametrize("n_qubits", _QUBIT_COUNTS, ids=_IDS)
def test_pennylane_hardware_efficient_produces_normalized_state(n_qubits: int) -> None:
    ansatz = build_hardware_efficient_pennylane(n_qubits, reps=2)
    rng = np.random.default_rng(0)
    state = ansatz.state_fn(rng.uniform(0, 2 * np.pi, ansatz.n_parameters))
    assert abs(np.linalg.norm(state) - 1.0) < 1e-10


@pytest.mark.parametrize("n_qubits", _QUBIT_COUNTS, ids=_IDS)
def test_efficient_su2_transpiles_against_real_device_topology(n_qubits: int) -> None:
    ansatz = build_efficient_su2_qiskit(n_qubits, reps=2, entanglement="linear")
    cost = transpiled_cost(ansatz, backend=FakeTorino())
    assert cost.transpiled_depth > 0
    # 2 linear-entanglement layers -> (n_qubits - 1) CNOTs each, transpiled onto
    # FakeTorino's native 2-qubit gate: a small positive number, not zero, and
    # well below a dense/all-to-all circuit's CNOT count for this qubit count.
    assert 0 < cost.transpiled_two_qubit_gate_count < 4 * n_qubits
    assert cost.backend_name == "fake_torino"
