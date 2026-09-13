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


@dataclass(frozen=True)
class AnsatzSpec:
    name: str
    n_qubits: int
    n_parameters: int
    state_fn: StateFn  # params (shape (n_parameters,)) -> statevector amplitudes (dim 2**n_qubits)
    qiskit_circuit: QuantumCircuit | None = None
