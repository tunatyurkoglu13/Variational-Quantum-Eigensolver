"""Ansatz cost/quality metrics: transpiled circuit cost against a real device topology,
and the expressibility / entangling-capability descriptors of Sim, Johnson & Aspuru-Guzik
2019 (arXiv:1905.10876, *Expressibility and entangling capability of parameterized quantum
circuits for hybrid quantum-classical algorithms*, Adv. Quantum Technol. 2(12):1900070).

Definitions used, as verified directly from the paper (not from memory):

- Expressibility: Expr = D_KL( P_hat_PQC(F; theta) || P_Haar(F) ), the KL divergence
  between a histogram of sampled-pair state fidelities F = |<psi(theta1)|psi(theta2)>|^2
  and the analytic Haar-random fidelity distribution P_Haar(F) = (N-1)(1-F)^(N-2) for an
  N-dimensional Hilbert space. Lower Expr = closer to Haar-random = more expressible.

- Entangling capability: Ent = (1/|S|) sum_{theta in S} Q(|psi_theta>), the sample average
  of the Meyer-Wallach measure Q over uniformly sampled circuit parameters. This module
  computes Q via its proven-equivalent single-qubit-purity form (Brennen 2003,
  "An observable measure of entanglement for pure states of multi-qubit systems"):
  Q(|psi>) = (2/n) * sum_i (1 - Tr(rho_i^2)), rho_i the reduced density matrix of qubit i.
  This is mathematically identical to Sim et al.'s original iota_j/D formulation but far
  simpler and more numerically robust to implement with standard partial-trace tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import DensityMatrix, partial_trace

from vqe_nisq_project.ansatz.base import AnsatzSpec, ComplexArray, FloatArray


@dataclass(frozen=True)
class CircuitCostMetrics:
    n_parameters: int
    transpiled_depth: int
    transpiled_two_qubit_gate_count: int
    backend_name: str


def transpiled_cost(ansatz: AnsatzSpec, backend: BackendV2) -> CircuitCostMetrics:
    if ansatz.qiskit_circuit is None:
        raise ValueError(f"{ansatz.name} has no Qiskit circuit to transpile.")
    transpiled = transpile(ansatz.qiskit_circuit, backend=backend, optimization_level=3)
    two_qubit_count = sum(1 for instr in transpiled.data if instr.operation.num_qubits == 2)
    return CircuitCostMetrics(
        n_parameters=ansatz.n_parameters,
        transpiled_depth=transpiled.depth(),
        transpiled_two_qubit_gate_count=two_qubit_count,
        backend_name=backend.name,
    )


def meyer_wallach_entanglement(state: ComplexArray, n_qubits: int) -> float:
    """Q(|psi>) via the Brennen single-qubit-purity form, verified equivalent to the
    original Meyer-Wallach measure."""
    rho = DensityMatrix(state)
    total = 0.0
    for qubit in range(n_qubits):
        traced_out = [q for q in range(n_qubits) if q != qubit]
        rho_i = partial_trace(rho, traced_out)
        purity = np.real(np.trace(rho_i.data @ rho_i.data))
        total += 1.0 - purity
    return float(2.0 / n_qubits * total)


def entangling_capability(ansatz: AnsatzSpec, n_samples: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    samples = rng.uniform(0, 2 * np.pi, size=(n_samples, ansatz.n_parameters))
    q_values = [
        meyer_wallach_entanglement(ansatz.state_fn(theta), ansatz.n_qubits) for theta in samples
    ]
    return float(np.mean(q_values))


def _haar_fidelity_pdf(fidelities: FloatArray, hilbert_dim: int) -> FloatArray:
    result: FloatArray = (hilbert_dim - 1) * (1 - fidelities) ** (hilbert_dim - 2)
    return result


def expressibility(
    ansatz: AnsatzSpec, n_samples: int = 200, n_bins: int = 75, seed: int = 0
) -> float:
    """Expr = D_KL(P_hat_PQC(F) || P_Haar(F)) -- lower is more expressible."""
    rng = np.random.default_rng(seed)
    theta1 = rng.uniform(0, 2 * np.pi, size=(n_samples, ansatz.n_parameters))
    theta2 = rng.uniform(0, 2 * np.pi, size=(n_samples, ansatz.n_parameters))

    fidelities = np.empty(n_samples)
    for i in range(n_samples):
        psi1 = ansatz.state_fn(theta1[i])
        psi2 = ansatz.state_fn(theta2[i])
        fidelities[i] = np.abs(np.vdot(psi1, psi2)) ** 2

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    p_pqc, _ = np.histogram(fidelities, bins=bin_edges, density=False)
    p_pqc = p_pqc / n_samples  # discrete probability mass per bin

    hilbert_dim = 2**ansatz.n_qubits
    p_haar = _haar_fidelity_pdf(bin_centers, hilbert_dim) * bin_width
    p_haar = p_haar / p_haar.sum()  # renormalize the discretized analytic pdf

    # KL divergence: skip bins where the PQC mass is zero (0 * log(0/x) := 0 by convention).
    nonzero = p_pqc > 0
    kl = np.sum(p_pqc[nonzero] * np.log(p_pqc[nonzero] / p_haar[nonzero]))
    return float(kl)
