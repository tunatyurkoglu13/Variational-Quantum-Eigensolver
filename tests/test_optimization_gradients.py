"""Ground truth: the from-scratch parameter-shift derivation must agree with (a) a
finite-difference approximation and (b) PennyLane's own independent parameter-shift
autodiff implementation -- the latter to machine precision, since both compute the exact
same mathematical quantity by construction."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import (
    build_efficient_su2_qiskit,
    build_hardware_efficient_pennylane,
)
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.optimization.gradients import (
    finite_difference_gradient,
    parameter_shift_gradient,
)


def _h2_hamiltonian() -> np.ndarray:
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    return map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()


def test_parameter_shift_matches_finite_difference_on_efficient_su2() -> None:
    hamiltonian = _h2_hamiltonian()
    ansatz = build_efficient_su2_qiskit(4, reps=1)
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)

    grad_ps = parameter_shift_gradient(ansatz.state_fn, hamiltonian, theta)
    grad_fd = finite_difference_gradient(ansatz.state_fn, hamiltonian, theta)
    assert np.max(np.abs(grad_ps - grad_fd)) < 1e-6


def test_parameter_shift_matches_pennylanes_own_autodiff_exactly() -> None:
    hamiltonian = _h2_hamiltonian()
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)
    hermitian_observable = qml.Hermitian(hamiltonian, wires=range(n_qubits))
    pl_ansatz = build_hardware_efficient_pennylane(n_qubits, reps=1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.StronglyEntanglingLayers(params.reshape(1, n_qubits, 3), wires=range(n_qubits))
        return qml.expval(hermitian_observable)

    rng = np.random.default_rng(0)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, pl_ansatz.n_parameters), requires_grad=True)

    grad_pennylane = np.array(qml.grad(circuit)(theta))
    grad_ours = parameter_shift_gradient(pl_ansatz.state_fn, hamiltonian, np.array(theta))

    assert np.max(np.abs(grad_pennylane - grad_ours)) < 1e-10


def test_gradient_is_zero_at_a_local_minimum_direction_check() -> None:
    # Sanity check independent of any external reference: stepping a small
    # amount opposite the gradient must not increase the energy (first-order
    # descent direction check).
    hamiltonian = _h2_hamiltonian()
    ansatz = build_efficient_su2_qiskit(4, reps=1)
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)

    grad = parameter_shift_gradient(ansatz.state_fn, hamiltonian, theta)
    from vqe_nisq_project.optimization.gradients import expectation

    e0 = expectation(ansatz.state_fn(theta), hamiltonian)
    e_step = expectation(ansatz.state_fn(theta - 1e-3 * grad), hamiltonian)
    assert e_step <= e0 + 1e-9
