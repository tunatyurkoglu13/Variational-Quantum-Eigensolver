"""Z2 symmetry tapering: remove qubits corresponding to the Hamiltonian's conserved symmetries.

Finding *which* of the 2^k possible symmetry sectors is the physically correct one (the one
containing the ground state we care about) is normally done by working out the eigenvalues of
each symmetry generator on the Hartree-Fock reference determinant. This module instead
determines it empirically -- exhaustively diagonalizing every sector and matching against a
known-correct reference energy (typically FCI) -- following the same "verify against ground
truth rather than trust a derivation" practice established in fermionic.py for the OpenFermion
index convention. This is more robust for a small system and immediately catches a wrong/
ambiguous sector (raises rather than silently returning the wrong energy).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.analysis.z2_symmetries import Z2Symmetries


@dataclass(frozen=True)
class TaperingResult:
    z2_symmetries: Z2Symmetries
    tapered_op: SparsePauliOp
    tapering_values: tuple[int, ...]
    n_qubits_removed: int
    electronic_ground_state: float


def find_tapered_sector(
    qubit_op: SparsePauliOp, reference_electronic_energy: float, tol: float = 1e-6
) -> TaperingResult:
    """Find qubit_op's Z2 symmetries and the specific sector matching the reference energy.

    Raises ValueError if no symmetries exist, or if the number of sectors whose ground state
    matches `reference_electronic_energy` (within `tol`) is not exactly one.
    """
    z2 = Z2Symmetries.find_z2_symmetries(qubit_op)
    if z2.is_empty():
        raise ValueError("No Z2 symmetries found in this operator -- nothing to taper.")

    candidates = z2.taper(qubit_op)
    assert isinstance(candidates, list)  # tapering_values is None here, so taper() returns a list
    sector_values = list(itertools.product([1, -1], repeat=len(z2.symmetries)))

    matches: list[tuple[tuple[int, ...], SparsePauliOp, float]] = []
    for values, tapered in zip(sector_values, candidates, strict=True):
        ground_state = float(np.linalg.eigvalsh(tapered.to_matrix()).min())
        if abs(ground_state - reference_electronic_energy) < tol:
            matches.append((values, tapered, ground_state))

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Z2 sector matching the reference energy "
            f"{reference_electronic_energy}, found {len(matches)}."
        )

    values, tapered_op, ground_state = matches[0]
    z2.tapering_values = list(values)

    return TaperingResult(
        z2_symmetries=z2,
        tapered_op=tapered_op,
        tapering_values=values,
        n_qubits_removed=qubit_op.num_qubits - tapered_op.num_qubits,
        electronic_ground_state=ground_state,
    )
