"""Unified VQE optimization across four optimizers, each run through the same energy-trace
bookkeeping so they're directly comparable:

- COBYLA (gradient-free, scipy): a trust-region method that only needs energy values, no
  gradient at all -- robust to noise but typically needs many more function evaluations to
  converge than a gradient-based method on a smooth landscape.
- SPSA (gradient-free but *stochastic*, PennyLane's `SPSAOptimizer`): approximates the
  gradient along one random direction per step using just 2 energy evaluations
  REGARDLESS of parameter count -- see gradients.py's module docstring for why this
  matters once each evaluation costs real hardware shots.
- L-BFGS-B (gradient-based, scipy) fed OUR OWN parameter-shift gradient
  (gradients.parameter_shift_gradient) -- demonstrates the from-scratch-derived exact
  gradient formula actually driving a real optimizer, not just matching a reference.
- Adam (gradient-based, PennyLane's `AdamOptimizer`) fed PennyLane's OWN autodiff gradient
  via a PennyLane qnode -- demonstrates the other stated approach (framework autodiff)
  rather than our own parameter-shift code, so the two gradient sources are both
  exercised, not just one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from vqe_nisq_project.ansatz.base import ComplexArray, FloatArray, StateFn
from vqe_nisq_project.optimization.gradients import expectation, parameter_shift_gradient

OptimizerName = Literal["cobyla", "spsa", "lbfgsb", "adam"]


@dataclass(frozen=True)
class VqeResult:
    optimizer_name: OptimizerName
    seed: int
    energy_trace: list[float] = field(default_factory=list)
    final_energy: float = 0.0
    final_params: FloatArray = field(default_factory=lambda: np.array([]))
    n_energy_evaluations: int = 0


def _run_cobyla(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    initial_params: FloatArray,
    e_core: float,
    max_iterations: int,
) -> VqeResult:
    trace: list[float] = []
    n_evals = 0

    def cost(params: FloatArray) -> float:
        nonlocal n_evals
        n_evals += 1
        e = expectation(state_fn(params), hamiltonian) + e_core
        trace.append(e)
        return e

    result = minimize(cost, initial_params, method="COBYLA", options={"maxiter": max_iterations})
    return VqeResult("cobyla", 0, trace, float(result.fun), result.x, n_evals)


def _run_lbfgsb(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    initial_params: FloatArray,
    e_core: float,
    max_iterations: int,
) -> VqeResult:
    trace: list[float] = []
    n_evals = 0

    def cost(params: FloatArray) -> float:
        nonlocal n_evals
        n_evals += 1
        e = expectation(state_fn(params), hamiltonian) + e_core
        trace.append(e)
        return e

    def grad(params: FloatArray) -> FloatArray:
        return parameter_shift_gradient(state_fn, hamiltonian, params)

    result = minimize(
        cost,
        initial_params,
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    return VqeResult("lbfgsb", 0, trace, float(result.fun), result.x, n_evals)


def _run_spsa(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    initial_params: FloatArray,
    e_core: float,
    max_iterations: int,
    seed: int,
) -> VqeResult:
    trace: list[float] = []
    n_evals = 0

    def cost(params: FloatArray) -> float:
        nonlocal n_evals
        n_evals += 1
        return expectation(state_fn(params), hamiltonian) + e_core

    np.random.seed(seed)  # PennyLane's SPSAOptimizer draws its perturbation from np.random
    opt = qml.SPSAOptimizer(maxiter=max_iterations)
    params = pnp.array(initial_params, requires_grad=True)
    for _ in range(max_iterations):
        params, energy = opt.step_and_cost(cost, params)
        trace.append(float(energy))

    final_energy = expectation(state_fn(np.array(params)), hamiltonian) + e_core
    return VqeResult("spsa", seed, trace, final_energy, np.array(params), n_evals)


def _run_adam(
    pennylane_cost_fn: Callable[[FloatArray], float],
    initial_params: FloatArray,
    max_iterations: int,
) -> VqeResult:
    trace: list[float] = []
    opt = qml.AdamOptimizer(stepsize=0.1)
    params = pnp.array(initial_params, requires_grad=True)
    for _ in range(max_iterations):
        params, energy = opt.step_and_cost(pennylane_cost_fn, params)
        trace.append(float(energy))
    final_energy = float(pennylane_cost_fn(params))
    return VqeResult("adam", 0, trace, final_energy, np.array(params), max_iterations)


def run_vqe(
    state_fn: StateFn,
    hamiltonian: ComplexArray,
    initial_params: FloatArray,
    optimizer: OptimizerName,
    e_core: float = 0.0,
    max_iterations: int = 100,
    seed: int = 0,
    pennylane_cost_fn: Callable[[FloatArray], float] | None = None,
) -> VqeResult:
    """Run one VQE optimization and return its full energy trace.

    `pennylane_cost_fn` (a PennyLane qnode returning an expectation value, differentiable
    via PennyLane's own autodiff) is required for optimizer="adam" -- Adam specifically
    demonstrates framework autodiff rather than our own parameter-shift code (see module
    docstring); the caller builds it from whichever PennyLane ansatz is being tested.
    """
    if optimizer == "cobyla":
        result = _run_cobyla(state_fn, hamiltonian, initial_params, e_core, max_iterations)
    elif optimizer == "lbfgsb":
        result = _run_lbfgsb(state_fn, hamiltonian, initial_params, e_core, max_iterations)
    elif optimizer == "spsa":
        result = _run_spsa(state_fn, hamiltonian, initial_params, e_core, max_iterations, seed)
    elif optimizer == "adam":
        if pennylane_cost_fn is None:
            raise ValueError("optimizer='adam' requires a pennylane_cost_fn.")
        result = _run_adam(pennylane_cost_fn, initial_params, max_iterations)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return VqeResult(
        optimizer_name=optimizer,
        seed=seed,
        energy_trace=result.energy_trace,
        final_energy=result.final_energy,
        final_params=result.final_params,
        n_energy_evaluations=len(result.energy_trace),
    )
