"""Parameter-shift rule: an EXACT gradient formula (not a finite-difference approximation)
for circuits built from single-Pauli-generator rotation gates -- any gate
U(theta) = exp(-i*theta*G/2) with G^2 = I (true of RX, RY, RZ, and any single-Pauli-string
rotation).

Derivation (worked out from scratch, not quoted): with U(theta) = cos(theta/2) I -
i sin(theta/2) G (exact, since G^2 = I truncates the operator exponential's series),

    E(theta) = <psi0| U^dagger(theta) H U(theta) |psi0>
             = (a+b)/2 + (a-b)/2 * cos(theta) + c/2 * sin(theta)

for theta-independent constants a = <H>_0, b = <GHG>_0, c = <i[G,H]>_0 -- E(theta) is
exactly a sinusoid. Evaluating at theta +/- pi/2 and subtracting:

    E(theta+pi/2) - E(theta-pi/2) = -(a-b) sin(theta) + c cos(theta) = 2 dE/dtheta

so dE/dtheta = [E(theta+pi/2) - E(theta-pi/2)] / 2 EXACTLY, for any theta -- no h->0 limit
needed, unlike a finite-difference approximation.

Scope: only valid, as implemented here, for ansatze whose every parameter independently
controls a single-Pauli-generator rotation (true for EfficientSU2's Ry/Rz layers) -- NOT
for UCCSD's multi-Pauli-term excitation generators (a double-excitation generator is a sum
of 8 Pauli strings, so G^2 != I and this two-point rule does not directly apply). A
generalized multi-term parameter-shift rule exists in the literature but is out of scope
here; PennyLane's own autodiff is used for UCCSD gradients instead (see vqe_runner.py).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from vqe_nisq_project.ansatz.base import ComplexArray, FloatArray, StateFn

PARAMETER_SHIFT = np.pi / 2

#: Hamiltonians/observables are usually dense (chemistry/ Hamiltonians), but a barren-
#: plateau observable that is exactly diagonal (see optimization/barren_plateau.py) is
#: passed in as a scipy sparse matrix instead to avoid a dense (2**n_qubits)^2 blowup.
type HamiltonianLike = ComplexArray | scipy.sparse.spmatrix


def expectation(state: ComplexArray, hamiltonian: HamiltonianLike) -> float:
    with np.errstate(all="ignore"):  # benign macOS Accelerate BLAS quirk, see ansatz/adapt.py
        return float((state.conj() @ (hamiltonian @ state)).real)


def parameter_shift_gradient(
    state_fn: StateFn, hamiltonian: HamiltonianLike, params: FloatArray
) -> FloatArray:
    """Exact gradient of <psi(params)|H|psi(params)> w.r.t. every parameter.

    Requires each parameter to independently gate a single-Pauli-generator rotation
    (see module docstring) -- not generally valid for UCCSD-style multi-term generators.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        shifted_plus = params.copy()
        shifted_plus[i] += PARAMETER_SHIFT
        shifted_minus = params.copy()
        shifted_minus[i] -= PARAMETER_SHIFT

        e_plus = expectation(state_fn(shifted_plus), hamiltonian)
        e_minus = expectation(state_fn(shifted_minus), hamiltonian)
        grad[i] = (e_plus - e_minus) / 2.0
    return grad


def parameter_shift_gradient_component(
    state_fn: StateFn, hamiltonian: HamiltonianLike, params: FloatArray, index: int
) -> float:
    """Same exact formula as `parameter_shift_gradient`, but only the one component at
    `index` -- for barren-plateau scaling studies, computing all N components per sample
    when only one is needed wastes 2(N-1) extra state simulations per sample."""
    shifted_plus = params.copy()
    shifted_plus[index] += PARAMETER_SHIFT
    shifted_minus = params.copy()
    shifted_minus[index] -= PARAMETER_SHIFT
    e_plus = expectation(state_fn(shifted_plus), hamiltonian)
    e_minus = expectation(state_fn(shifted_minus), hamiltonian)
    return (e_plus - e_minus) / 2.0


def finite_difference_gradient(
    state_fn: StateFn, hamiltonian: ComplexArray, params: FloatArray, h: float = 1e-6
) -> FloatArray:
    """Central finite-difference gradient -- a reference to cross-check the exact
    parameter-shift formula against, not used for actual optimization."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += h
        minus = params.copy()
        minus[i] -= h
        e_plus = expectation(state_fn(plus), hamiltonian)
        e_minus = expectation(state_fn(minus), hamiltonian)
        grad[i] = (e_plus - e_minus) / (2 * h)
    return grad
