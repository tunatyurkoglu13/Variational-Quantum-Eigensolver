"""Map the same electronic Hamiltonian three ways (JW, Parity, Bravyi-Kitaev) and report
the Pauli-weight/term-count cost of each -- all three must give the identical spectrum,
since they are just different unitary encodings of the same fermionic physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import FermionicOp

from vqe_nisq_project.chemistry.integrals import MOIntegrals

MappingName = Literal["jordan_wigner", "parity", "bravyi_kitaev"]

_MAPPER_FACTORIES = {
    "jordan_wigner": JordanWignerMapper,
    "parity": ParityMapper,
    "bravyi_kitaev": BravyiKitaevMapper,
}


@dataclass(frozen=True)
class MappingResult:
    name: MappingName
    qubit_op: SparsePauliOp
    n_qubits: int
    n_terms: int
    max_pauli_weight: int
    mean_pauli_weight: float
    electronic_ground_state: float


def _pauli_weight(label: str) -> int:
    return sum(1 for c in label if c != "I")


def build_electronic_fermionic_op(integrals: MOIntegrals) -> FermionicOp:
    """The same restricted-closed-shell construction validated in fermionic.py."""
    h1, h2 = integrals.one_body, integrals.two_body
    hamiltonian = ElectronicEnergy.from_raw_integrals(h1, h2, h1, h2, h2)
    op: FermionicOp = hamiltonian.second_q_op()
    return op


def map_hamiltonian(fermionic_op: FermionicOp, name: MappingName) -> MappingResult:
    mapper = _MAPPER_FACTORIES[name]()
    qubit_op = mapper.map(fermionic_op)
    weights = [_pauli_weight(label) for label in qubit_op.paulis.to_labels()]
    ground_state = float(np.linalg.eigvalsh(qubit_op.to_matrix()).min())
    return MappingResult(
        name=name,
        qubit_op=qubit_op,
        n_qubits=qubit_op.num_qubits,
        n_terms=len(qubit_op),
        max_pauli_weight=max(weights),
        mean_pauli_weight=float(np.mean(weights)),
        electronic_ground_state=ground_state,
    )


def map_all(fermionic_op: FermionicOp) -> dict[MappingName, MappingResult]:
    names: list[MappingName] = ["jordan_wigner", "parity", "bravyi_kitaev"]
    return {name: map_hamiltonian(fermionic_op, name) for name in names}
