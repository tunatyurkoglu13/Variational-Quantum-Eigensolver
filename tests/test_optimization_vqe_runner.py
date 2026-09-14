"""Ground truth: each optimizer path in run_vqe must actually descend the energy landscape
and (given enough budget) reach FCI on H2/EfficientSU2 -- the one ansatz family for which
our own parameter-shift gradient (L-BFGS-B) and PennyLane's autodiff (Adam) are both valid
(see gradients.py and uccsd.py's module docstrings for why UCCSD is excluded from this).

This test file also regression-tests two real bugs found while building this phase:
1. Feeding a qiskit-nature-basis Hamiltonian into a PennyLane qnode's `qml.Hermitian`
   without correcting for the block/interleaved + endianness mismatch (ansatz/base.py's
   `reverse_qubit_endianness`) silently gives a plausible-looking but wrong energy --
   Adam's test explicitly checks the corrected path reaches the true minimum.
2. Applying the simple two-point parameter-shift rule to UCCSD's multi-Pauli-term
   generators silently returns an all-zero "gradient" (see gradients.py's scope note) --
   tested here as a fact, not "fixed", since it is a genuine formula-validity boundary.
"""

import numpy as np
import pennylane as qml

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.base import reverse_qubit_endianness
from vqe_nisq_project.ansatz.hardware_efficient import (
    build_efficient_su2_qiskit,
    build_hardware_efficient_pennylane,
)
from vqe_nisq_project.ansatz.uccsd import build_uccsd_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.optimization.gradients import parameter_shift_gradient
from vqe_nisq_project.optimization.initial_points import small_random_initial_point
from vqe_nisq_project.optimization.vqe_runner import run_vqe


def _h2_setup():
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    hamiltonian = map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()
    fci = cached_reference_energies(h2()).fci
    return hamiltonian, integrals, fci


def test_lbfgsb_with_parameter_shift_reaches_fci_on_efficient_su2() -> None:
    hamiltonian, integrals, fci = _h2_setup()
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    init = small_random_initial_point(ansatz.n_parameters, scale=0.1, seed=0)
    result = run_vqe(
        ansatz.state_fn,
        hamiltonian,
        init,
        optimizer="lbfgsb",
        e_core=integrals.e_nuc,
        max_iterations=150,
    )
    assert abs(result.final_energy - fci) < 1e-5


def test_cobyla_reduces_energy_from_initial_point() -> None:
    hamiltonian, integrals, _fci = _h2_setup()
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    init = small_random_initial_point(ansatz.n_parameters, scale=0.1, seed=0)
    result = run_vqe(
        ansatz.state_fn,
        hamiltonian,
        init,
        optimizer="cobyla",
        e_core=integrals.e_nuc,
        max_iterations=100,
    )
    assert result.final_energy < result.energy_trace[0]


def test_spsa_moves_toward_fci_even_if_slowly() -> None:
    # SPSA's real advantage is under shot noise (see gradients.py's module
    # docstring); on this noiseless landscape it is expected to be slower
    # than the gradient-based methods, not non-functional -- check it makes
    # real progress, not that it fully converges in a CI-sized budget.
    hamiltonian, integrals, fci = _h2_setup()
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    init = small_random_initial_point(ansatz.n_parameters, scale=0.1, seed=0)
    result = run_vqe(
        ansatz.state_fn,
        hamiltonian,
        init,
        optimizer="spsa",
        e_core=integrals.e_nuc,
        max_iterations=150,
        seed=0,
    )
    initial_gap = abs(result.energy_trace[0] - fci)
    final_gap = abs(result.final_energy - fci)
    assert final_gap < initial_gap


def test_adam_with_correctly_realigned_hamiltonian_reaches_fci() -> None:
    # Regression test for bug #1 above: without reverse_qubit_endianness, this
    # test fails (converges to a physically wrong ~-0.52 Ha "minimum" instead).
    hamiltonian, integrals, fci = _h2_setup()
    n_qubits = 4
    hamiltonian_pl_basis = reverse_qubit_endianness(hamiltonian, n_qubits)
    observable = qml.Hermitian(hamiltonian_pl_basis, wires=range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
        qml.StronglyEntanglingLayers(params.reshape(2, n_qubits, 3), wires=range(n_qubits))
        return qml.expval(observable)

    def cost(params: np.ndarray) -> float:
        # NOT float(circuit(params)) -- that breaks PennyLane's autograd
        # tracing (found the hard way: raises deep inside autograd with
        # "float() argument must be ... not 'ArrayBox'"). Adam needs the
        # untouched traced value all the way through the addition below.
        return circuit(params) + integrals.e_nuc

    ansatz = build_hardware_efficient_pennylane(n_qubits, reps=2)
    init = small_random_initial_point(ansatz.n_parameters, scale=0.1, seed=42)
    result = run_vqe(
        ansatz.state_fn,
        hamiltonian,
        init,
        optimizer="adam",
        e_core=integrals.e_nuc,
        max_iterations=150,
        pennylane_cost_fn=cost,
    )
    assert abs(result.final_energy - fci) < 1e-4


def test_parameter_shift_on_uccsd_multiterm_generators_is_documented_invalid() -> None:
    # Not a bug to fix -- a real boundary of the simple two-point parameter-shift
    # formula (see gradients.py's module docstring): UCCSD excitation generators
    # are sums of several Pauli strings, so G^2 != I and the formula silently
    # returns zero instead of the true (nonzero) gradient at theta=0.
    hamiltonian, _integrals, _fci = _h2_setup()
    ansatz = build_uccsd_qiskit(2, (1, 1))
    grad = parameter_shift_gradient(ansatz.state_fn, hamiltonian, np.zeros(ansatz.n_parameters))
    assert np.allclose(grad, 0.0)
