"""Phase 0 capstone: the same Bell circuit on Qiskit and PennyLane must agree.

This is the concrete evidence that the two-framework bridge this project
depends on (comparing the same ansatz/Hamiltonian across both) actually works
numerically, not just "both import successfully."
"""

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.stats import chisquare

N_SHOTS = 20_000
SEED = 0


def _qiskit_bell_probabilities() -> dict[str, float]:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    statevector = Statevector.from_instruction(qc)
    probs = statevector.probabilities_dict()
    return probs


def _pennylane_bell_probabilities() -> dict[str, float]:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit() -> object:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = circuit()
    # PennyLane orders basis states as |q0 q1>, same convention as Qiskit's
    # probabilities_dict keys here (both little-endian over the same wire order).
    return {format(i, "02b"): float(p) for i, p in enumerate(probs)}


def test_bell_circuit_exact_statevector_probabilities_match() -> None:
    qiskit_probs = _qiskit_bell_probabilities()
    pennylane_probs = _pennylane_bell_probabilities()

    assert set(qiskit_probs) == {"00", "11"}
    assert set(pennylane_probs) >= {"00", "11"}

    for state in ("00", "11"):
        assert qiskit_probs[state] == pytest.approx(0.5, abs=1e-9)
        assert pennylane_probs[state] == pytest.approx(0.5, abs=1e-9)
    for state, p in pennylane_probs.items():
        if state not in ("00", "11"):
            assert p == pytest.approx(0.0, abs=1e-9)


def test_bell_circuit_shot_sampling_distributions_agree_qiskit_vs_pennylane() -> None:
    """Sample both frameworks with finite shots and check the distributions agree.

    Not exact-equality (shot noise) -- a chi-squared goodness-of-fit test against
    the shared theoretical distribution {00: 0.5, 11: 0.5}, same standard used for
    any measurement-based framework cross-check in this repo.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qiskit_counts = Statevector.from_instruction(
        qc.remove_final_measurements(inplace=False)
    ).sample_counts(N_SHOTS)

    dev = qml.device("default.qubit", wires=2, seed=SEED)

    @qml.set_shots(N_SHOTS)
    @qml.qnode(dev)
    def circuit() -> object:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1])

    pennylane_counts = circuit()

    for label, counts in [("qiskit", qiskit_counts), ("pennylane", dict(pennylane_counts))]:
        observed = np.array(
            [
                counts.get("00", 0),
                counts.get("11", 0),
                N_SHOTS - counts.get("00", 0) - counts.get("11", 0),
            ]
        )
        expected = np.array([N_SHOTS / 2, N_SHOTS / 2, 0.0])
        # Guard against the degenerate all-zero "other outcomes" bucket breaking chi2.
        if expected[-1] == 0:
            assert observed[-1] == 0, f"{label}: measured an impossible outcome"
            observed, expected = observed[:2], expected[:2]
        _, p_value = chisquare(observed, expected)
        assert p_value > 0.01, f"{label} Bell-state sampling distribution rejected: p={p_value}"
