"""Multi-seed statistics: a single VQE run says nothing about how reliably an
optimizer/ansatz/initial-point combination reaches the right answer -- local minima,
optimizer stochasticity (SPSA), and random initial points all vary run to run. Per this
project's stated standard, every configuration is run across >=20 seeds and reported as
mean +/- a bootstrap 95% confidence interval, plus the fraction of seeds that actually
reached the known-correct energy (rather than getting stuck).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vqe_nisq_project.ansatz.base import ComplexArray, FloatArray, StateFn
from vqe_nisq_project.optimization.initial_points import small_random_initial_point
from vqe_nisq_project.optimization.vqe_runner import OptimizerName, run_vqe


@dataclass(frozen=True)
class MultiSeedResult:
    optimizer_name: OptimizerName
    n_seeds: int
    final_energies: FloatArray
    mean_energy: float
    ci_95_low: float
    ci_95_high: float
    fraction_converged: float  # within convergence_tol of best_known_energy


def bootstrap_ci_95(
    values: FloatArray, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_resamples)]
    )
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def run_multi_seed(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    n_parameters: int,
    optimizer: OptimizerName,
    best_known_energy: float,
    e_core: float = 0.0,
    max_iterations: int = 100,
    n_seeds: int = 20,
    init_scale: float = 0.5,
    convergence_tol: float = 1e-3,
    pennylane_cost_fn_factory: Callable[[], Callable[[FloatArray], float]] | None = None,
) -> MultiSeedResult:
    """Run `optimizer` from `n_seeds` independent small-random initial points.

    `pennylane_cost_fn_factory`, if given, is called once per seed to build a fresh
    PennyLane cost function (needed for optimizer="adam"; see vqe_runner.run_vqe).
    """
    energies = np.zeros(n_seeds)
    for seed in range(n_seeds):
        init = small_random_initial_point(n_parameters, scale=init_scale, seed=seed)
        pennylane_cost_fn = pennylane_cost_fn_factory() if pennylane_cost_fn_factory else None
        result = run_vqe(
            state_fn,
            hamiltonian,
            init,
            optimizer=optimizer,
            e_core=e_core,
            max_iterations=max_iterations,
            seed=seed,
            pennylane_cost_fn=pennylane_cost_fn,
        )
        energies[seed] = result.final_energy

    ci_low, ci_high = bootstrap_ci_95(energies)
    fraction_converged = float(np.mean(np.abs(energies - best_known_energy) < convergence_tol))

    return MultiSeedResult(
        optimizer_name=optimizer,
        n_seeds=n_seeds,
        final_energies=energies,
        mean_energy=float(energies.mean()),
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        fraction_converged=fraction_converged,
    )
