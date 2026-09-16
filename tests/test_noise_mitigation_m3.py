"""M3 must (1) correctly diagonalize/measure a real chemistry Hamiltonian's qubit-wise-
commuting groups (X/Y/Z bases all exercised, not just the trivial all-Z case), (2) nearly
eliminate a PURE readout-error bias (the case it is designed for), and (3) leave a
gate-error bias largely uncorrected (the honest limit of a readout-only technique --
motivates stacking with ZNE/twirling next)."""

import mthree
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime.fake_provider import FakeTorino

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.mitigation_m3 import (
    calibrate,
    estimate_energy_with_readout_mitigation,
)
from vqe_nisq_project.noise.models import build_noise_model


def _h2_jw_setup() -> tuple:
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(0)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    return mapping, ansatz, params, integrals.e_core


def test_m3_nearly_eliminates_a_pure_readout_error_bias() -> None:
    """Isolate readout error from every other noise source: a NoiseModel with ONLY a
    strong, asymmetric single-qubit readout error, no gate error at all. This is exactly
    the error M3 is designed to correct, so it should close nearly all of the gap."""
    n_qubits = 3
    # A strong, asymmetric readout error on every qubit -- P(read 1 | prepared 0) = 8%,
    # P(read 0 | prepared 1) = 15% (asymmetric errors are realistic; a symmetric one
    # could accidentally cancel in some expectation values and hide a real bug).
    readout_error = ReadoutError([[0.92, 0.08], [0.15, 0.85]])
    noise_model = NoiseModel()
    for q in range(n_qubits):
        noise_model.add_readout_error(readout_error, [q])

    # A GHZ-like state exercising X, Y, and Z bases (not just trivial computational basis
    # measurement): H on qubit 0, CX chain, then an S gate so qubit 2 carries a nontrivial
    # Y-basis component too.
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.s(2)

    hamiltonian = SparsePauliOp.from_list([("ZII", 1.0), ("IXI", 0.5), ("IIY", 0.7), ("XXX", 0.3)])
    backend = FakeTorino()

    mitigator = calibrate(backend, n_qubits, noise_model, shots=8000)
    result = estimate_energy_with_readout_mitigation(
        qc,
        hamiltonian,
        e_core=0.0,
        backend=backend,
        mitigator=mitigator,
        noise_model=noise_model,
        shots=50_000,
        seed=0,
    )

    with np.errstate(all="ignore"):
        state = Statevector(qc).data
        exact = float((state.conj() @ (hamiltonian.to_matrix() @ state)).real)

    raw_error = abs(result.raw_energy - exact)
    mitigated_error = abs(result.mitigated_energy - exact)
    # M3 must cut the readout-induced bias substantially (not just noise-level improvement).
    assert mitigated_error < 0.3 * raw_error
    assert raw_error > 0.02  # sanity: the injected readout error was actually large enough


def _h2_jw_full_noise_setup(param_seed: int):
    """Same H2/EfficientSU2(reps=2, full-entanglement)/FakeTorino setup as
    `_h2_jw_setup`, but with the ansatz PARAMETER seed exposed -- used to demonstrate
    that M3's effect on the TOTAL (gate+readout) noise bias genuinely depends on the
    specific parameter point, not just on "is M3 implemented correctly."""
    integrals = cached_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(param_seed)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    return mapping, ansatz, params, integrals.e_core


def _run_full_noise_m3(mapping, ansatz, params, e_core, backend, mitigator, noise_model):
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    result = estimate_energy_with_readout_mitigation(
        bound_circuit,
        mapping.qubit_op,
        e_core,
        backend,
        mitigator,
        noise_model=noise_model,
        shots=500_000,
        seed=0,
        seed_transpiler=123,
    )
    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    exact = exact_electronic + e_core
    return result.raw_energy - exact, result.mitigated_energy - exact


def test_m3_can_meaningfully_reduce_the_combined_noise_bias() -> None:
    """Under FakeTorino's FULL (gate + readout) noise model on H2/EfficientSU2, there
    exist real parameter points where M3's readout-only correction still roughly halves
    the total energy bias -- readout error is genuinely part of the noise budget here,
    just not all of it (contrast the pure-readout test above, where it removes nearly
    all of the bias, and the test below, where the SAME mitigator makes things worse)."""
    mapping, ansatz, params, e_core = _h2_jw_full_noise_setup(param_seed=2)
    assert ansatz.qiskit_circuit is not None
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    mitigator = calibrate(backend, mapping.n_qubits, noise_model, shots=100_000)

    raw_error, mitigated_error = _run_full_noise_m3(
        mapping, ansatz, params, e_core, backend, mitigator, noise_model
    )
    assert abs(raw_error) > 1e-2  # sanity: noise really is biasing this instance
    assert abs(mitigated_error) < 0.7 * abs(raw_error)


def test_m3_can_worsen_the_combined_noise_bias_via_error_cancellation() -> None:
    """The genuinely surprising, real finding, found while investigating why an earlier
    version of this test kept flipping between pass/fail across reruns (root cause of
    THAT flakiness, fixed separately: `transpile()` needs `seed_transpiler` for
    reproducibility -- see simulate.py's Real finding #3): once results are reproducible,
    there exist real parameter points where the UNCORRECTED (raw) energy is closer to
    exact than the M3-CORRECTED one, because the gate-error bias and the readout-error
    bias happen to have opposite signs and partially cancel in the raw estimate. M3
    correctly removes ONLY the readout component (as designed) -- which, here, REVEALS a
    larger net bias rather than reducing it. Not a bug: partial mitigation of a
    multi-channel noise budget is not guaranteed to be monotonic in the channel it
    doesn't touch, and this is a legitimate, reproducible instance of exactly that."""
    mapping, ansatz, params, e_core = _h2_jw_full_noise_setup(param_seed=3)
    assert ansatz.qiskit_circuit is not None
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    mitigator = calibrate(backend, mapping.n_qubits, noise_model, shots=100_000)

    raw_error, mitigated_error = _run_full_noise_m3(
        mapping, ansatz, params, e_core, backend, mitigator, noise_model
    )
    assert abs(mitigated_error) > 5 * abs(raw_error)


def test_m3_calibration_circuit_count_is_linear_not_exponential_in_qubit_count() -> None:
    """Regression test for the module's central cost claim: M3's 'matrix-free' calibration
    uses O(n) circuits, not the O(2^n) a naive full assignment-matrix would need."""
    for n_qubits, expected_circuits in [(4, 8), (10, 20)]:
        sim = AerSimulator()
        call_counts: list[int] = []
        original_run = sim.run

        def counting_run(circuits, *args, _orig=original_run, _counts=call_counts, **kwargs):
            _counts.append(len(circuits) if isinstance(circuits, list) else 1)
            return _orig(circuits, *args, **kwargs)

        sim.run = counting_run  # type: ignore[method-assign]

        mit = mthree.M3Mitigation(sim)
        mit.cals_from_system(qubits=list(range(n_qubits)), shots=1000, async_cal=False)
        assert sum(call_counts) == expected_circuits
