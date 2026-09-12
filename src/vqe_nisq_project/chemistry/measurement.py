"""Qubit-wise-commuting measurement grouping -- this is the number that directly sets shot cost.

Only Pauli terms whose measurement bases agree on every qubit (qubit-wise commuting, the
practical case for hardware) are grouped together; each group needs exactly one measurement
circuit/basis.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp


@dataclass(frozen=True)
class MeasurementGroupingResult:
    n_terms: int
    n_groups: int
    groups: list[SparsePauliOp]


def group_qubit_wise_commuting(qubit_op: SparsePauliOp) -> MeasurementGroupingResult:
    groups = qubit_op.group_commuting(qubit_wise=True)
    return MeasurementGroupingResult(n_terms=len(qubit_op), n_groups=len(groups), groups=groups)
