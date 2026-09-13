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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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

    for iteration in range(max_iterations):
        current_state = Statevector(_build_circuit(hf, pool, selected, params)).data

        gradients = np.zeros(len(pool))
        for i, generator in enumerate(pool):
            if i in selected:
                continue
            A = generator.to_matrix()
            # macOS's Accelerate BLAS backend raises spurious divide-by-zero /
            # overflow / invalid-value warnings on some complex-matrix @
            # complex-matrix products regardless of the actual values
            # (reproduced even for H @ H); verified the results here contain
            # no NaN/Inf and are numerically correct, so this is suppressed
            # rather than a sign of an actual computation error.
            with np.errstate(all="ignore"):
                commutator = hamiltonian @ A - A @ hamiltonian
                commutator_expectation = current_state.conj() @ commutator @ current_state
            gradients[i] = float((-1j * commutator_expectation).real)

        best = int(np.argmax(np.abs(gradients)))
        if abs(gradients[best]) < gradient_tolerance:
            break

        selected.append(best)
        params = np.append(params, 0.0)

        def cost(theta: FloatArray, selected: list[int] = selected) -> float:
            state = Statevector(_build_circuit(hf, pool, selected, theta)).data
            return _energy(state, hamiltonian)

        result = minimize(cost, params, method="L-BFGS-B")
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
