"""Shared ansatz interface.

Design choice: rather than mirroring every Qiskit field with a parallel
PennyLane field, `AnsatzSpec` carries one framework-agnostic callable
(`state_fn`: parameters -> statevector amplitudes) that `metrics.py`'s
expressibility/entangling-capability code consumes identically regardless of
which framework built the ansatz. The Qiskit `QuantumCircuit` is *also* kept
(when available) because transpiled depth/CNOT-count metrics need real
gate-level structure against a coupling map -- that check is Qiskit-specific
since that is the framework this repo's hardware runs (real backends,
Phase 6) execute through.

Cross-framework validation for UCCSD compares *converged VQE energies*
between the Qiskit and PennyLane builds, not raw statevectors: qiskit-nature
and PennyLane order spin-orbitals into qubits differently (block: alpha
block then beta block, vs. interleaved: alternating alpha/beta -- verified
empirically on H2, same category of gotcha as the qiskit-nature/OpenFermion
one from Phase 1), so a literal amplitude-by-amplitude comparison would
need an extra qubit-relabeling step that adds bookkeeping without adding
anything Phase 4 actually needs (it only cares whether an ansatz reaches
the right energy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]
StateFn = Callable[[FloatArray], ComplexArray]


def reverse_qubit_endianness_index_map(n_qubits: int) -> npt.NDArray[np.int64]:
    """index_map[i] = i with its n_qubits-bit binary representation reversed.

    Converts between Qiskit's little-endian (qubit 0 = LSB of the statevector
    index) and PennyLane's big-endian (wire 0 = MSB) conventions for the SAME
    qubit labeling (qiskit qubit i treated as the same physical qubit as
    pennylane wire i) -- e.g. for a generic hardware-efficient ansatz with no
    chemistry-specific spin-orbital relabeling (contrast uccsd.py's
    `pennylane_index_for_block_index`, which ALSO relabels alpha/beta spin
    orbitals and is not a plain bit-reversal). Found the hard way: applying
    uccsd.py's chemistry-specific realignment logic to a non-chemistry ansatz
    (EfficientSU2) was assumed unnecessary and skipped, which silently paired
    a Hamiltonian and a PennyLane state in mismatched bases -- Adam
    "converged" to a plausible-looking but physically wrong energy with no
    error or NaN to flag it. Bit-reversal is its own inverse, so this one
    function converts in either direction.
    """
    indices = np.arange(2**n_qubits, dtype=np.int64)
    reversed_indices = np.zeros_like(indices)
    for q in range(n_qubits):
        bit = (indices >> q) & 1
        reversed_indices |= bit << (n_qubits - 1 - q)
    return reversed_indices


def reverse_qubit_endianness(array: ComplexArray, n_qubits: int) -> ComplexArray:
    """Apply `reverse_qubit_endianness_index_map` to a statevector (1D) or operator matrix
    (2D)."""
    index_map = reverse_qubit_endianness_index_map(n_qubits)
    if array.ndim == 1:
        result: ComplexArray = array[index_map]
        return result
    result = array[np.ix_(index_map, index_map)]
    return result


@dataclass(frozen=True)
class AnsatzSpec:
    name: str
    n_qubits: int
    n_parameters: int
    state_fn: StateFn  # params (shape (n_parameters,)) -> statevector amplitudes (dim 2**n_qubits)
    qiskit_circuit: QuantumCircuit | None = None
