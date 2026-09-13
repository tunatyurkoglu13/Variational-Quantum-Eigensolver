"""UCCSD (Unitary Coupled Cluster Singles-Doubles), built via each framework's own
established template (qiskit-nature's `UCC`, PennyLane's `qml.UCCSD`) -- this project's
job is orchestrating and cross-validating these, not re-deriving the Trotterized gate
decomposition by hand.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

from vqe_nisq_project.ansatz.base import AnsatzSpec, ComplexArray, FloatArray

# Qubit relabeling aligning PennyLane's interleaved (alpha0,beta0,alpha1,beta1,...)
# spin-orbital ordering with qiskit-nature's blocked (all alpha, then all beta)
# ordering -- found empirically for H2 (num_spatial_orbitals=2) by exhaustively
# searching all 4! qubit permutations against the known RHF reference energy (same
# practice as the OpenFermion index-convention search in chemistry/fermionic.py).
# Only verified for this system size; a different num_spatial_orbitals would need
# its own search, not an assumed generalization of this specific permutation.
H2_PENNYLANE_TO_QISKIT_NATURE_QUBIT_PERM = (2, 0, 3, 1)


def permute_statevector_qubits(state: ComplexArray, perm: tuple[int, ...]) -> ComplexArray:
    n = len(perm)
    tensor = state.reshape([2] * n)
    permuted: ComplexArray = np.transpose(tensor, axes=perm).reshape(-1)
    return permuted


def build_uccsd_qiskit(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> AnsatzSpec:
    mapper = JordanWignerMapper()
    hf = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    circuit = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="sd",
        qubit_mapper=mapper,
        initial_state=hf,
    )

    def state_fn(params: FloatArray) -> ComplexArray:
        bound = circuit.assign_parameters(params)
        result: ComplexArray = Statevector(bound).data
        return result

    return AnsatzSpec(
        name="UCCSD (qiskit-nature)",
        n_qubits=circuit.num_qubits,
        n_parameters=circuit.num_parameters,
        state_fn=state_fn,
        qiskit_circuit=circuit,
    )


def build_uccsd_pennylane(num_spatial_orbitals: int, num_particles: tuple[int, int]) -> AnsatzSpec:
    n_qubits = 2 * num_spatial_orbitals
    n_electrons = sum(num_particles)
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    n_parameters = len(singles) + len(doubles)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]
    def circuit(params: FloatArray) -> qml.measurements.StateMP:
        qml.UCCSD(
            params, wires=range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state
        )
        return qml.state()

    def state_fn(params: FloatArray) -> ComplexArray:
        result: ComplexArray = np.asarray(circuit(params))
        return result

    return AnsatzSpec(
        name="UCCSD (PennyLane)", n_qubits=n_qubits, n_parameters=n_parameters, state_fn=state_fn
    )
