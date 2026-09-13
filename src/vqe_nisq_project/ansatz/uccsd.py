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

# PennyLane and qiskit-nature disagree on two independent conventions, each
# found empirically (not assumed) rather than trusted from either library's
# docs alone:
#
# 1. Spin-orbital -> qubit assignment. qiskit-nature uses "blocked" ordering
#    (qubits 0..n_mo-1 = alpha spatial orbitals 0..n_mo-1, qubits
#    n_mo..2*n_mo-1 = the same for beta) -- confirmed from its HartreeFock
#    circuit, which places X gates on qubits {0, n_mo} for one alpha + one
#    beta electron. PennyLane uses "interleaved" ordering (wire 2p = alpha
#    spatial orbital p, wire 2p+1 = beta spatial orbital p) -- confirmed from
#    `qml.qchem.hf_state`, which occupies wires {0, 1} for the same case.
#
# 2. Qubit-to-statevector-index endianness. Qiskit is little-endian: qubit q
#    is bit q of the integer index (qubit 0 = LSB) -- confirmed via
#    `Statevector`. PennyLane is the opposite: wire w is bit (n-1-w) of the
#    index (wire 0 = MSB) -- confirmed by applying a lone PauliX to wire 0 on
#    a 2-qubit device and checking which amplitude lit up (index 2 = 0b10,
#    not index 1 = 0b01).
#
# `realign_pennylane_state_to_qiskit_nature` combines both corrections into
# one index permutation, so a PennyLane statevector can be directly used
# with a Hamiltonian matrix built (as this repo's chemistry/ package does)
# via qiskit-nature's spin-orbital convention. Verified via ground truth (RHF
# energy at theta=0) on two different systems -- H2 (num_spatial_orbitals=2)
# and LiH (num_spatial_orbitals=5) -- since deriving this by pure algebra is
# exactly the kind of index-convention step this project has repeatedly
# found easy to get subtly wrong (see chemistry/fermionic.py's module
# docstring for the analogous OpenFermion case).


def _interleaved_wire_for_block_qubit(block_qubit: int, num_spatial_orbitals: int) -> int:
    if block_qubit < num_spatial_orbitals:
        return 2 * block_qubit  # alpha, spatial orbital = block_qubit
    return 2 * (block_qubit - num_spatial_orbitals) + 1  # beta


def realign_pennylane_state_to_qiskit_nature(
    state: ComplexArray, num_spatial_orbitals: int
) -> ComplexArray:
    n_qubits = 2 * num_spatial_orbitals
    size = 2**n_qubits
    realigned = np.empty(size, dtype=complex)
    for k_block in range(size):
        k_pennylane = 0
        for q in range(n_qubits):
            bit = (k_block >> q) & 1  # qiskit-nature block qubit q, little-endian
            wire = _interleaved_wire_for_block_qubit(q, num_spatial_orbitals)
            k_pennylane |= bit << (n_qubits - 1 - wire)  # PennyLane wire, big-endian
        realigned[k_block] = state[k_pennylane]
    result: ComplexArray = realigned
    return result


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
