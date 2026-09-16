"""Dynamical decoupling (DD) must (1) be an EXACT no-op under zero noise (X X = I), (2)
show no measurable echo benefit under `qiskit_aer`'s memoryless thermal-relaxation model
(verified directly with an isolated single-qubit echo, the same check done before
building the full pipeline), and (3) still run successfully end-to-end on a real
chemistry circuit despite the 133-qubit-wide ISA transpilation that broke naive
simulation (Real finding #2 in `mitigation_dd.py`)."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.mitigation_dd import apply_dynamical_decoupling, dd_energy
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import estimate_energy, prepare_for_backend


def _h2_isa_setup(param_seed: int = 2):
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(param_seed)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    backend = FakeTorino()
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )
    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    return isa_circuit, isa_hamiltonian, integrals.e_core, exact_electronic, backend


def test_spin_echo_shows_no_benefit_under_aers_memoryless_thermal_relaxation() -> None:
    """The grounding check behind this whole module's expectations: a lone qubit's <X>
    decay over one continuous 100us delay vs. two 50us delays with an X-X echo pair in
    between must be nearly IDENTICAL -- Aer's thermal_relaxation_error has no quasi-
    static/1-over-f component for a spin echo to refocus, so DD provably cannot help
    against this specific noise model, regardless of the circuit it's applied to."""
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    sim = AerSimulator(noise_model=noise_model)
    total_delay_ns = 100_000

    qc_plain = QuantumCircuit(1)
    qc_plain.h(0)
    qc_plain.delay(total_delay_ns, 0, unit="ns")
    qc_plain.save_expectation_value(Pauli("X"), qubits=[0], label="X")

    qc_echo = QuantumCircuit(1)
    qc_echo.h(0)
    qc_echo.delay(total_delay_ns // 2, 0, unit="ns")
    qc_echo.x(0)
    qc_echo.delay(total_delay_ns // 2, 0, unit="ns")
    qc_echo.x(0)
    qc_echo.save_expectation_value(Pauli("X"), qubits=[0], label="X")

    plain_x = sim.run(qc_plain).result().data(0)["X"].real
    echo_x = sim.run(qc_echo).result().data(0)["X"].real

    assert plain_x < 0.9  # sanity: real decoherence over 100us actually happened
    assert abs(echo_x - plain_x) < 0.01  # echo changes almost nothing


def test_dd_padding_is_an_exact_no_op_under_zero_noise() -> None:
    isa_circuit, isa_hamiltonian, e_core, exact_electronic, backend = _h2_isa_setup()
    dd_circuit, dd_hamiltonian = apply_dynamical_decoupling(isa_circuit, isa_hamiltonian, backend)

    no_params = np.array([])
    orig = estimate_energy(
        isa_circuit, no_params, isa_hamiltonian, e_core, shots=500_000, noise_model=None, seed=0
    )
    padded = estimate_energy(
        dd_circuit, no_params, dd_hamiltonian, e_core, shots=500_000, noise_model=None, seed=0
    )
    assert abs(padded.energy - orig.energy) < 1e-6
    assert abs(orig.energy - (exact_electronic + e_core)) < 5e-3


def test_dd_has_a_small_effect_on_the_same_h2_system_m3_and_zne_helped_on() -> None:
    """Completes the mitigation-technique comparison: on the SAME H2/EfficientSU2(reps=2,
    full)/FakeTorino system where M3 and ZNE each showed large, clear effects
    (test_noise_mitigation_m3.py, test_noise_mitigation_zne.py) and twirling showed a
    small one (test_noise_mitigation_twirling.py), DD -- which targets a noise structure
    (quasi-static dephasing) this simulator's noise model doesn't contain (previous
    test) -- should likewise show only a small effect on the mean energy."""
    isa_circuit, isa_hamiltonian, e_core, exact_electronic, backend = _h2_isa_setup()
    noise_model = build_noise_model(backend)
    exact = exact_electronic + e_core

    result = dd_energy(
        isa_circuit, isa_hamiltonian, e_core, backend, noise_model, shots=200_000, seed=0
    )
    raw_error = abs(result.raw_energy - exact)
    dd_error = abs(result.dd_energy - exact)
    assert raw_error > 1e-2  # sanity: noise really is biasing this instance
    assert abs(dd_error - raw_error) < 0.3 * raw_error


def test_dd_energy_result_reports_the_real_shot_budget() -> None:
    isa_circuit, isa_hamiltonian, e_core, _exact, backend = _h2_isa_setup()
    noise_model = build_noise_model(backend)
    result = dd_energy(
        isa_circuit, isa_hamiltonian, e_core, backend, noise_model, shots=1000, seed=0
    )
    assert result.shots == 1000
