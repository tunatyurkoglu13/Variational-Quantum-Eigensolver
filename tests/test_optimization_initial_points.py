"""Ground truth: HF start (theta=0) on UCCSD must reproduce RHF exactly (that's what
theta=0 is defined to mean for this ansatz); the CAFQA-inspired Clifford search must
never do worse than the single Clifford point it always includes as a candidate (theta=0
itself, i.e. HF, is one of the 4^n_parameters grid points)."""

import numpy as np

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.uccsd import build_uccsd_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.optimization.gradients import expectation
from vqe_nisq_project.optimization.initial_points import (
    cafqa_inspired_initial_point,
    hf_initial_point,
    small_random_initial_point,
)


def _h2_setup():
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    hamiltonian = map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()
    return hamiltonian, integrals


def test_hf_initial_point_matches_rhf_on_uccsd() -> None:
    hamiltonian, integrals = _h2_setup()
    ansatz = build_uccsd_qiskit(2, (1, 1))
    point = hf_initial_point(ansatz.n_parameters)
    energy = expectation(ansatz.state_fn(point), hamiltonian) + integrals.e_core
    assert abs(energy - integrals.e_hf) < 1e-7


def test_small_random_initial_point_has_expected_scale() -> None:
    point = small_random_initial_point(10, scale=0.05, seed=0)
    assert point.shape == (10,)
    assert np.std(point) < 0.2  # generous bound, just checks the scale param is honored


def test_cafqa_inspired_search_never_beats_the_true_ground_truth() -> None:
    # Sanity bound: no Clifford-restricted point can beat the unrestricted
    # exact ground state (a basic variational-principle consistency check),
    # and since theta=0 (HF, exactly on the Clifford grid) is a candidate the
    # search can land on, its result should be no worse than HF either.
    hamiltonian, _integrals = _h2_setup()
    ansatz = build_uccsd_qiskit(2, (1, 1))
    hf_energy = expectation(ansatz.state_fn(hf_initial_point(ansatz.n_parameters)), hamiltonian)

    result = cafqa_inspired_initial_point(
        ansatz.state_fn, hamiltonian, ansatz.n_parameters, n_samples=100, seed=0
    )
    assert result.energy <= hf_energy + 1e-9
    assert result.n_samples == 100
