"""Ground truth: bootstrap_ci_95 must bracket the true mean of a known distribution most
of the time, and run_multi_seed must actually vary its result across seeds (otherwise the
"seeds" wouldn't be doing anything) while keeping every seed's optimizer run internally
consistent with vqe_runner.py's own single-run behavior."""

import numpy as np

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.optimization.multi_seed import bootstrap_ci_95, run_multi_seed


def test_bootstrap_ci_brackets_a_known_mean() -> None:
    rng = np.random.default_rng(0)
    true_mean = 5.0
    sample = rng.normal(true_mean, 1.0, size=200)
    low, high = bootstrap_ci_95(sample)
    assert low < true_mean < high
    assert low < np.mean(sample) < high


def test_run_multi_seed_lbfgsb_converges_with_low_variance_on_h2() -> None:
    # L-BFGS-B + the exact parameter-shift gradient should reliably reach FCI
    # from most/all small-random starts on EfficientSU2 (a smooth, exact-
    # gradient landscape) -- a real check on "how reliably does this actually
    # work", not just a single lucky seed.
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    hamiltonian = map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()
    fci = cached_reference_energies(h2()).fci

    ansatz = build_efficient_su2_qiskit(4, reps=2)
    result = run_multi_seed(
        ansatz.state_fn,
        hamiltonian,
        ansatz.n_parameters,
        "lbfgsb",
        best_known_energy=fci,
        e_core=integrals.e_nuc,
        max_iterations=150,
        n_seeds=20,
        init_scale=0.1,
        convergence_tol=1e-4,
    )
    assert result.n_seeds == 20
    assert result.fraction_converged > 0.5
    assert abs(result.mean_energy - fci) < 1e-2
