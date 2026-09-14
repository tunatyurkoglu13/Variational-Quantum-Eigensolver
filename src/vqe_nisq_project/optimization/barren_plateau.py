"""Barren plateau scaling: Var[dE/dtheta] vs qubit count N, on a log scale.

McClean et al. 2018 (arXiv:1803.11173, "Barren plateaus in quantum neural network
training landscapes") predict this variance decays EXPONENTIALLY with N for
sufficiently expressive/random hardware-efficient circuits -- gradients become
exponentially hard to resolve (relative to shot noise) as the circuit scales up, the
central practical obstacle to training deep parameterized circuits.

Following the standard protocol from that literature: fix one parameter index, sample
many random parameter vectors (uniform over the full circuit), compute that one
gradient component via the exact parameter-shift rule for each sample, and report the
sample VARIANCE at each qubit count -- not the gradient magnitude itself (whose sign
and size vary run to run; the vanishing-variance signature is what "barren plateau"
means).

Observable: a single local Z on qubit 0 (not the molecular Hamiltonian) -- this project's
molecules only exist at 3 fixed qubit counts (4/10/12), too few points for a scaling
curve; a local observable on a generic hardware-efficient ansatz lets qubit count be
swept independently, which is what the scaling question is actually about, and matches
the standard barren-plateau literature's protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse

from vqe_nisq_project.ansatz.base import AnsatzSpec, FloatArray
from vqe_nisq_project.optimization.gradients import (
    HamiltonianLike,
    parameter_shift_gradient_component,
)


@dataclass(frozen=True)
class BarrenPlateauPoint:
    n_qubits: int
    gradient_variance: float
    n_samples: int


def local_z0_observable(n_qubits: int) -> scipy.sparse.dia_matrix:
    """Z on qubit 0 (Qiskit little-endian: qubit 0 = LSB), identity elsewhere -- the
    LOCAL observable in the local-vs-global barren-plateau comparison (Cerezo et al. 2021,
    arXiv:2001.00550). Diagonal (eigenvalues +/-1 depending on bit 0 of the basis index),
    built directly as a sparse matrix: a dense (2**n_qubits)^2 matrix is the same ~50GB-at-
    n=16-ish memory blowup already hit once in ansatz/adapt.py, avoidable here since a
    tensor-product-of-Paulis-with-Z/I-only observable is exactly diagonal.
    """
    size = 2**n_qubits
    indices = np.arange(size)
    diag = np.where((indices & 1) == 0, 1.0, -1.0)  # qubit 0 = bit 0 (LSB)
    return scipy.sparse.diags(diag, format="csr")


def global_parity_observable(n_qubits: int) -> scipy.sparse.dia_matrix:
    """Z^{\\otimes n} (product of Z on every qubit) -- the GLOBAL observable in the same
    comparison. Also diagonal: eigenvalue (-1)^(popcount of the basis index)."""
    size = 2**n_qubits
    indices = np.arange(size, dtype=np.uint32)
    parity = np.zeros(size, dtype=np.int64)
    shifted = indices.copy()
    for _ in range(n_qubits):
        parity ^= shifted & 1
        shifted >>= 1
    diag = np.where(parity == 0, 1.0, -1.0)
    return scipy.sparse.diags(diag, format="csr")


def measure_gradient_variance(
    ansatz: AnsatzSpec,
    observable: HamiltonianLike,
    n_samples: int = 40,
    seed: int = 0,
    parameter_index: int = 0,
) -> BarrenPlateauPoint:
    rng = np.random.default_rng(seed)
    gradients: FloatArray = np.zeros(n_samples)
    for i in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
        gradients[i] = parameter_shift_gradient_component(
            ansatz.state_fn, observable, theta, parameter_index
        )
    return BarrenPlateauPoint(
        n_qubits=ansatz.n_qubits, gradient_variance=float(np.var(gradients)), n_samples=n_samples
    )
