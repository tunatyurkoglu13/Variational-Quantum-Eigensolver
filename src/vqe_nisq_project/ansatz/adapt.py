"""ADAPT-VQE (Grimsley, Economou, Barnes & Mayhall 2019, arXiv:1812.11173): rather than
fixing the full UCCSD circuit upfront, grow the ansatz one excitation at a time, greedily
picking (at each step) the pool operator with the largest energy gradient at the current
state, then re-optimizing all parameters so far.

Operator pool: the same singles+doubles fermionic excitation generators UCCSD uses
(qiskit-nature's `UCC.operators`) -- so ADAPT here selects an adaptively-ordered *subset*
of UCCSD's own excitation pool, not a different family of operators. This is the standard
choice in the original ADAPT-VQE paper.

Gradient derivation: qiskit-nature's excitation generators A_i are Hermitian (real Pauli
coefficients), and the corresponding circuit layer applies U_i(theta) = exp(-i*theta*A_i)
(the standard Pauli-evolution-gate convention). For that layer appended on top of a state
|psi>, d/dtheta <psi|U_i^dagger(theta) H U_i(theta)|psi> at theta=0 works out to
-i * <psi|[H, A_i]|psi> -- real, since [H, A_i] is anti-Hermitian (both H and A_i are
Hermitian) and so has a purely imaginary expectation value in any state.

Classical-simulation scaling (real limitations found while extending this from H2 to
LiH/BeH2, not a hardware-noise effect):

1. The pool grows fast with system size -- 3 operators for H2, 24 for LiH, 92 for
   BeH2 -- so the gradient screen precomputes each pool operator's commutator with H
   exactly once (done outside the main loop) rather than recomputing it every
   iteration, which was the original bottleneck.
2. That precompute must use SPARSE matrices. A first version used
   `SparsePauliOp.to_matrix()` (dense): each of BeH2's 92 pool operators as a dense
   4096x4096 complex128 matrix is 268 MB, and holding 92 of those plus their 92
   commutators simultaneously is ~50 GB -- far more than this project's 16 GB
   machine has, and the resulting swap thrashing was *slower* than the original
   per-iteration-recompute version it was meant to replace. `SparsePauliOp` Pauli
   terms are inherently sparse (each has exactly one nonzero per row), so the same
   operators as `scipy.sparse` matrices (`to_matrix(sparse=True)`) are only ~0.02%
   dense -- kilobytes instead of hundreds of megabytes each.
3. The remaining, harder-to-remove cost is per-iteration re-optimization: each COBYLA
   function evaluation re-simulates the whole grown circuit via `Statevector`, and
   qiskit re-synthesizes every `PauliEvolutionGate` from scratch each time -- ~0.6s
   already for a 5-excitation/10-qubit (LiH-sized) circuit. `max_cobyla_iterations`
   bounds this cost per step; a full run to `gradient_tolerance` convergence on
   LiH/BeH2 was, at this implementation's current state, still impractically slow
   for interactive use even after fix #2, and is reported here as a scaling finding
   rather than forced through with a long background run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.optimize import minimize

from vqe_nisq_project.ansatz.base import ComplexArray, FloatArray


@dataclass(frozen=True)
class AdaptStep:
    iteration: int
    excitation_index: int
    excitation_label: tuple[tuple[int, ...], tuple[int, ...]]
    gradient: float
    energy_after_reoptimization: float


@dataclass(frozen=True)
class AdaptVqeResult:
    steps: list[AdaptStep] = field(default_factory=list)
    final_energy: float = 0.0
    final_n_parameters: int = 0
    final_circuit: QuantumCircuit | None = None


def _energy(state: ComplexArray, hamiltonian: ComplexArray) -> float:
    # Same benign Accelerate BLAS quirk as the gradient computation below --
    # a matrix-vector product can also trip it, not just matrix-matrix ones.
    with np.errstate(all="ignore"):
        return float((state.conj() @ hamiltonian @ state).real)


def _build_circuit(
    hf_circuit: QuantumCircuit,
    pool: list[SparsePauliOp],
    selected: list[int],
    params: FloatArray,
) -> QuantumCircuit:
    circuit = hf_circuit.copy()
    for idx, theta in zip(selected, params, strict=True):
        circuit.append(PauliEvolutionGate(pool[idx], time=theta), circuit.qubits)
    return circuit


def run_adapt_vqe(
    hamiltonian: ComplexArray,
    num_spatial_orbitals: int,
    num_particles: tuple[int, int],
    e_core: float = 0.0,
    gradient_tolerance: float = 1e-3,
    max_iterations: int = 20,
    max_cobyla_iterations: int = 200,
) -> AdaptVqeResult:
    mapper = JordanWignerMapper()
    hf = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    uccsd_pool_source = UCC(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        excitations="sd",
        qubit_mapper=mapper,
    )
    pool: list[SparsePauliOp] = list(uccsd_pool_source.operators)
    excitation_list = uccsd_pool_source.excitation_list
    assert excitation_list is not None
    labels = list(excitation_list)

    hf_state = Statevector(hf).data
    selected: list[int] = []
    params: FloatArray = np.array([])
    steps: list[AdaptStep] = []

    # The pool operators and hence H@A_i - A_i@H are fixed for the whole run --
    # only the current *state* changes between iterations. Precomputing these
    # commutators once (as SPARSE matrices -- see module docstring point 2)
    # turns each iteration's gradient screen into cheap sparse matrix-VECTOR
    # products instead of re-deriving matrix-MATRIX products per pool
    # operator per iteration, without the dense version's ~50 GB blowup for
    # BeH2's 92-operator pool.
    hamiltonian_sparse = scipy.sparse.csr_matrix(hamiltonian)
    pool_matrices = [generator.to_matrix(sparse=True) for generator in pool]
    commutators = [hamiltonian_sparse @ A - A @ hamiltonian_sparse for A in pool_matrices]

    for iteration in range(max_iterations):
        current_state = Statevector(_build_circuit(hf, pool, selected, params)).data

        gradients = np.zeros(len(pool))
        for i in range(len(pool)):
            if i in selected:
                continue
            # macOS's Accelerate BLAS backend raises spurious divide-by-zero /
            # overflow / invalid-value warnings on some complex-matrix
            # products regardless of the actual values (reproduced even for
            # H @ H); verified the results here contain no NaN/Inf and are
            # numerically correct, so this is suppressed rather than a sign
            # of an actual computation error.
            with np.errstate(all="ignore"):
                commutator_expectation = current_state.conj() @ commutators[i] @ current_state
            gradients[i] = float((-1j * commutator_expectation).real)

        best = int(np.argmax(np.abs(gradients)))
        if abs(gradients[best]) < gradient_tolerance:
            break

        selected.append(best)
        params = np.append(params, 0.0)

        def cost(theta: FloatArray, selected: list[int] = selected) -> float:
            state = Statevector(_build_circuit(hf, pool, selected, theta)).data
            return _energy(state, hamiltonian)

        # COBYLA (gradient-free) avoids the O(n_params) extra function
        # evaluations a finite-difference L-BFGS-B would need per iteration
        # -- each evaluation here is itself a full circuit statevector
        # simulation (expensive: qiskit re-synthesizes each PauliEvolutionGate
        # from scratch on every call, ~0.6s already for a 5-excitation/10-qubit
        # circuit -- see the module docstring's scaling note), so bounding
        # eval count matters more than squeezing out the last bit of
        # per-step convergence.
        result = minimize(cost, params, method="COBYLA", options={"maxiter": max_cobyla_iterations})
        params = result.x

        steps.append(
            AdaptStep(
                iteration=iteration,
                excitation_index=best,
                excitation_label=labels[best],
                gradient=float(gradients[best]),
                energy_after_reoptimization=float(result.fun) + e_core,
            )
        )

    final_circuit = _build_circuit(hf, pool, selected, params)
    final_energy = (
        steps[-1].energy_after_reoptimization if steps else _energy(hf_state, hamiltonian) + e_core
    )

    return AdaptVqeResult(
        steps=steps,
        final_energy=final_energy,
        final_n_parameters=len(selected),
        final_circuit=final_circuit,
    )
