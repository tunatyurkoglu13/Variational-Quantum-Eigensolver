"""Initial-parameter-point strategies for VQE.

"HF start" (all-zero angles) is only chemically meaningful for ansatze like UCCSD where
theta=0 is *defined* to reproduce the Hartree-Fock reference circuit (see
ansatz/uccsd.py) -- for a hardware-efficient ansatz like EfficientSU2, theta=0 is just
some fixed product state with no particular chemical significance, so comparing "HF vs.
random vs. CAFQA-inspired" starts is only a meaningful experiment on a chemically
motivated ansatz (UCCSD), not on EfficientSU2.

CAFQA-inspired warm start (Ravi et al. 2022, arXiv:2202.12924, ASPLOS 2023): the original
CAFQA restricts parameters to the Clifford group ({0, pi/2, pi, 3pi/2} for standard
rotation gates) and searches that discrete space using a classically-efficient STABILIZER
simulator (Gottesman-Knill) plus Bayesian optimization, which is what lets it scale to
many qubits. This module does NOT implement a stabilizer simulator or Bayesian
optimization -- it evaluates candidate Clifford points via ordinary exact statevector
simulation and picks the best of `n_samples` random draws. That is only tractable because
this project's circuits are small; it is a simplified, CAFQA-*inspired* random search over
the same discrete parameter space, not a reproduction of the original algorithm's scaling
properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vqe_nisq_project.ansatz.base import ComplexArray, FloatArray, StateFn
from vqe_nisq_project.optimization.gradients import expectation

_CLIFFORD_ANGLES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])


def hf_initial_point(n_parameters: int) -> FloatArray:
    """All-zero angles. Meaningful as "the HF state" only for ansatze where theta=0 is
    defined to reproduce the Hartree-Fock reference (e.g. UCCSD)."""
    return np.zeros(n_parameters)


def small_random_initial_point(n_parameters: int, scale: float, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    result: FloatArray = rng.normal(0.0, scale, n_parameters)
    return result


@dataclass(frozen=True)
class CliffordSearchResult:
    params: FloatArray
    energy: float
    n_samples: int


def cafqa_inspired_initial_point(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    n_parameters: int,
    n_samples: int = 200,
    seed: int = 0,
) -> CliffordSearchResult:
    """Random search over the {0, pi/2, pi, 3pi/2}^n_parameters Clifford grid, evaluated by
    exact statevector simulation. See module docstring for how this differs from the real
    (stabilizer-simulator-based) CAFQA algorithm."""
    rng = np.random.default_rng(seed)
    # theta=0 (HF) is itself a Clifford-grid point (0 is one of the 4 allowed
    # angles) and is always included explicitly: a warm-start search should
    # never do worse than the trivial "no warm start" baseline it is meant
    # to improve on, which random sampling alone cannot guarantee.
    best_params = hf_initial_point(n_parameters)
    best_energy = expectation(state_fn(best_params), hamiltonian)

    for _ in range(n_samples):
        candidate = rng.choice(_CLIFFORD_ANGLES, size=n_parameters)
        energy = expectation(state_fn(candidate), hamiltonian)
        if energy < best_energy:
            best_energy = energy
            best_params = candidate

    return CliffordSearchResult(params=best_params, energy=best_energy, n_samples=n_samples)
