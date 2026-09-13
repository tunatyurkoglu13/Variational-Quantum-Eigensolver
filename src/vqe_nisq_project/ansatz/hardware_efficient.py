"""Hardware-efficient ansatze: no chemistry motivation, just alternating single-qubit
rotation layers and entangling layers. Shallow and cheap, but higher expressibility can
mean a higher barren-plateau risk (Phase 4 territory) -- this module only builds and
measures the circuits.

Unlike UCCSD, a hardware-efficient ansatz has no unique "correct" physical structure to
match between frameworks, so the Qiskit build (`efficient_su2`, this repo's IBM-hardware
execution path) and the PennyLane build (`StronglyEntanglingLayers`, PennyLane's standard
hardware-efficient template) are each valid instances of the same *category* rather than a
gate-for-gate replica of each other -- interface parity here means both expose the same
AnsatzSpec, not bit-identical circuits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector

from vqe_nisq_project.ansatz.base import AnsatzSpec, ComplexArray, FloatArray


def build_efficient_su2_qiskit(
    n_qubits: int, reps: int = 2, entanglement: str = "linear"
) -> AnsatzSpec:
    circuit = efficient_su2(n_qubits, reps=reps, entanglement=entanglement)

    def state_fn(params: FloatArray) -> ComplexArray:
        bound = circuit.assign_parameters(params)
        result: ComplexArray = Statevector(bound).data
        return result

    return AnsatzSpec(
        name=f"EfficientSU2 (qiskit, reps={reps}, {entanglement})",
        n_qubits=n_qubits,
        n_parameters=circuit.num_parameters,
        state_fn=state_fn,
        qiskit_circuit=circuit,
    )


def build_hardware_efficient_pennylane(n_qubits: int, reps: int = 2) -> AnsatzSpec:
    shape = qml.StronglyEntanglingLayers.shape(n_layers=reps, n_wires=n_qubits)
    n_parameters = int(np.prod(shape))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]
    def circuit(params: FloatArray) -> qml.measurements.StateMP:
        qml.StronglyEntanglingLayers(params.reshape(shape), wires=range(n_qubits))
        return qml.state()

    def state_fn(params: FloatArray) -> ComplexArray:
        result: ComplexArray = np.asarray(circuit(params))
        return result

    return AnsatzSpec(
        name=f"StronglyEntanglingLayers (PennyLane, reps={reps})",
        n_qubits=n_qubits,
        n_parameters=n_parameters,
        state_fn=state_fn,
    )
