"""Ground truth: reproduces (on a small, CI-fast qubit range) the Cerezo, Sone, Volkoff,
Cincio & Coles 2021 result (arXiv:2001.00550, Nature Communications) that GLOBAL cost
functions give exponentially vanishing gradient variance even for shallow circuits, while
LOCAL cost functions do not -- a real, literature-grounded, measured (not assumed)
qubit-count scaling comparison, not just a plausible-sounding claim."""

import numpy as np

from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.optimization.barren_plateau import (
    global_parity_observable,
    local_z0_observable,
    measure_gradient_variance,
)


def test_local_and_global_observables_are_valid_diagonal_operators() -> None:
    for n in (2, 4, 6):
        local = local_z0_observable(n).toarray()
        global_ = global_parity_observable(n).toarray()
        for observable in (local, global_):
            assert np.allclose(observable, np.diag(np.diagonal(observable)))  # diagonal
            assert set(np.unique(np.diagonal(observable).real)) <= {1.0, -1.0}


def test_global_observable_gradient_variance_decays_with_qubit_count() -> None:
    # Cerezo et al. 2021's headline result: a global observable gives an
    # exponentially vanishing gradient even for a SHALLOW circuit.
    variances = []
    for n in (4, 8, 12):
        ansatz = build_efficient_su2_qiskit(n, reps=3, entanglement="linear")
        observable = global_parity_observable(n)
        point = measure_gradient_variance(ansatz, observable, n_samples=25, seed=0)
        variances.append(point.gradient_variance)

    assert variances[0] > variances[1] > variances[2]
    # At least an order-of-magnitude drop from n=4 to n=12 -- not just noise.
    assert variances[2] < variances[0] / 10


def test_local_observable_gradient_variance_does_not_collapse() -> None:
    # Cerezo et al. 2021's other half: a LOCAL observable on the same shallow
    # circuit family does NOT show the same collapse.
    ansatz_small = build_efficient_su2_qiskit(4, reps=3, entanglement="linear")
    ansatz_large = build_efficient_su2_qiskit(12, reps=3, entanglement="linear")
    observable_small = local_z0_observable(4)
    observable_large = local_z0_observable(12)

    point_small = measure_gradient_variance(ansatz_small, observable_small, n_samples=25, seed=0)
    point_large = measure_gradient_variance(ansatz_large, observable_large, n_samples=25, seed=0)

    # Both should stay within the same order of magnitude (no exponential collapse) --
    # a much looser bound than the global case's order-of-magnitude drop, by design.
    assert point_large.gradient_variance > point_small.gradient_variance / 10
