"""Ground truth: Meyer-Wallach must give exactly 0 for product states and exactly 1 for
maximally-entangled (Bell/GHZ) states; expressibility/entangling-capability must rank a
deliberately unentangling, barely-varying circuit as far less expressive/entangling than
EfficientSU2."""

import numpy as np
import pytest

from vqe_nisq_project.ansatz.base import AnsatzSpec
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.ansatz.metrics import (
    entangling_capability,
    expressibility,
    meyer_wallach_entanglement,
)


def test_meyer_wallach_zero_for_product_states() -> None:
    state_2q = np.array([0, 1, 0, 0], dtype=complex)  # |01>
    assert meyer_wallach_entanglement(state_2q, 2) == pytest.approx(0.0, abs=1e-10)

    state_3q = np.zeros(8, dtype=complex)
    state_3q[0] = 1.0  # |000>
    assert meyer_wallach_entanglement(state_3q, 3) == pytest.approx(0.0, abs=1e-10)


def test_meyer_wallach_one_for_maximally_entangled_states() -> None:
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert meyer_wallach_entanglement(bell, 2) == pytest.approx(1.0, abs=1e-10)

    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    assert meyer_wallach_entanglement(ghz, 3) == pytest.approx(1.0, abs=1e-10)


def _weak_unentangling_ansatz(n_qubits: int) -> AnsatzSpec:
    # Stays in |0...0> up to a global phase for any parameter values -- deliberately
    # near-maximally *inexpressive* and *unentangling*, a floor-case sanity check.
    def state_fn(params: np.ndarray) -> np.ndarray:
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = np.exp(-1j * np.sum(params) / 2)
        return state

    return AnsatzSpec(name="weak", n_qubits=n_qubits, n_parameters=n_qubits, state_fn=state_fn)


def test_expressibility_ranks_weak_circuit_far_below_efficient_su2() -> None:
    weak = _weak_unentangling_ansatz(4)
    su2 = build_efficient_su2_qiskit(4, reps=2)

    expr_weak = expressibility(weak, n_samples=200, seed=0)
    expr_su2 = expressibility(su2, n_samples=200, seed=0)
    # Lower Expr = more expressible (closer to Haar-random) -- the weak circuit,
    # which can barely leave |0...0>, must be far LESS expressible (higher Expr).
    assert expr_weak > expr_su2


def test_entangling_capability_ranks_weak_circuit_far_below_efficient_su2() -> None:
    weak = _weak_unentangling_ansatz(4)
    su2 = build_efficient_su2_qiskit(4, reps=2)

    ent_weak = entangling_capability(weak, n_samples=100, seed=0)
    ent_su2 = entangling_capability(su2, n_samples=100, seed=0)
    assert ent_weak == pytest.approx(0.0, abs=1e-9)
    assert ent_su2 > 0.3
