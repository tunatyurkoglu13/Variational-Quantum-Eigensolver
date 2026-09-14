"""`estimate_energy` must (1) agree with exact statevector expectation once shot noise is
averaged out, (2) actually change when a real device noise model is applied (not silently
ignore it), and (3) be deterministic given a fixed seed."""

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeTorino

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import (
    estimate_energy,
    prepare_for_backend,
    shots_to_precision,
)


def test_shots_to_precision_matches_the_documented_relation() -> None:
    assert shots_to_precision(10_000) == 0.01
    assert shots_to_precision(100) == 0.1


def _h2_setup() -> tuple:
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="linear")
    rng = np.random.default_rng(0)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    return mapping, ansatz, params, integrals.e_core


def test_noiseless_shot_estimate_agrees_with_exact_statevector_expectation() -> None:
    mapping, ansatz, params, e_core = _h2_setup()
    assert ansatz.qiskit_circuit is not None

    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)

    result = estimate_energy(
        ansatz.qiskit_circuit,
        params,
        mapping.qubit_op,
        e_core,
        shots=200_000,
        noise_model=None,
        seed=0,
    )
    assert result.noisy is False
    # 200k shots -> std error ~0.0022; 5-sigma is a very loose, non-flaky bound.
    assert abs(result.electronic_energy - exact_electronic) < 5 * result.std_error


def test_untranspiled_circuit_silently_ignores_the_noise_model() -> None:
    """Regression test for Real finding #1 (see noise/simulate.py's module docstring):
    passing a noise model alongside a circuit still in its ORIGINAL (non-ISA) gate set
    must not be mistaken for a "no noise impact" physics result -- it's a silent no-op."""
    mapping, ansatz, params, e_core = _h2_setup()
    assert ansatz.qiskit_circuit is not None
    noise_model = build_noise_model(FakeTorino())

    noiseless = estimate_energy(
        ansatz.qiskit_circuit, params, mapping.qubit_op, e_core, shots=200_000, seed=1
    )
    fake_noisy = estimate_energy(
        ansatz.qiskit_circuit,
        params,
        mapping.qubit_op,
        e_core,
        shots=200_000,
        noise_model=noise_model,
        seed=1,
    )
    # Both draw the exact same Gaussian shot-noise offset (same seed/precision) around
    # the exact expectation value, so if the noise model had zero effect the two results
    # are identical to floating-point roundoff -- exactly the trap this test documents.
    assert fake_noisy.energy == pytest.approx(noiseless.energy, abs=1e-9)


def test_prepare_for_backend_then_device_noise_measurably_biases_the_energy_estimate() -> None:
    # Full (not linear) entanglement -- more CNOTs after transpile/routing than the
    # rest of this suite's H2 fixture, so the real noise bias clears shot noise cleanly.
    mapping = map_hamiltonian(
        build_electronic_fermionic_op(cached_mo_integrals(h2(), 0)), "jordan_wigner"
    )
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(0)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    e_core = cached_mo_integrals(h2(), 0).e_core
    assert ansatz.qiskit_circuit is not None

    backend = FakeTorino()
    noise_model = build_noise_model(backend)

    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    isa_circuit, isa_hamiltonian = prepare_for_backend(bound_circuit, mapping.qubit_op, backend)
    no_params = np.array([])  # already bound above -- isa_circuit has no free parameters

    noiseless = estimate_energy(
        isa_circuit, no_params, isa_hamiltonian, e_core, shots=500_000, seed=1
    )
    noisy = estimate_energy(
        isa_circuit,
        no_params,
        isa_hamiltonian,
        e_core,
        shots=500_000,
        noise_model=noise_model,
        seed=1,
    )
    assert noisy.noisy is True
    # A real bias, not just extra spread: must differ by much more than either run's
    # own shot-noise std error.
    assert abs(noisy.energy - noiseless.energy) > 3 * max(noisy.std_error, noiseless.std_error)

    # And the noiseless ISA path must still agree with the exact (pre-transpile)
    # statevector expectation -- transpiling/routing must be energy-preserving.
    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    assert abs(noiseless.electronic_energy - exact_electronic) < 5 * noiseless.std_error


def test_estimate_energy_is_deterministic_given_a_fixed_seed() -> None:
    mapping, ansatz, params, e_core = _h2_setup()
    assert ansatz.qiskit_circuit is not None
    noise_model = build_noise_model(FakeTorino())

    r1 = estimate_energy(
        ansatz.qiskit_circuit,
        params,
        mapping.qubit_op,
        e_core,
        shots=4096,
        noise_model=noise_model,
        seed=42,
    )
    r2 = estimate_energy(
        ansatz.qiskit_circuit,
        params,
        mapping.qubit_op,
        e_core,
        shots=4096,
        noise_model=noise_model,
        seed=42,
    )
    assert r1.energy == r2.energy
