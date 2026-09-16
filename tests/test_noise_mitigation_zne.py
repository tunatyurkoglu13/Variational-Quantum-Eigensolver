"""ZNE must (1) be a true no-op under zero noise (unitary folding is logically identity,
so the extrapolated curve should be flat), (2) measurably reduce a real gate-error-
dominated bias (the case it targets, complementing M3's readout-only scope), and (3)
report the real per-evaluation cost (one full circuit execution per scale factor)."""

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

from conftest import cached_mo_integrals
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.mitigation_zne import zne_energy
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import prepare_for_backend


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


def test_zne_is_a_true_no_op_under_zero_noise() -> None:
    """Unitary folding (G -> G G^dagger G) is exactly the identity in a noiseless
    circuit, so the ZNE-extrapolated energy must agree with the unfolded (scale=1)
    energy up to shot noise alone -- no noise means nothing for ZNE to correct."""
    isa_circuit, isa_hamiltonian, e_core, exact_electronic, _backend = _h2_isa_setup()
    result = zne_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        noise_model=None,
        shots_per_point=500_000,
        seed=0,
    )
    assert abs(result.zne_energy - result.raw_energy) < 1e-3
    # And both must still track the exact answer at this shot count.
    assert abs(result.raw_energy - (exact_electronic + e_core)) < 5e-3


def test_zne_measurably_reduces_a_gate_error_dominated_bias() -> None:
    """The complementary case to `mitigation_m3.py`'s finding: this circuit's noise bias
    is almost entirely gate error, which M3 (readout-only) cannot fix -- ZNE targets
    exactly that channel, so it should measurably shrink the bias here."""
    isa_circuit, isa_hamiltonian, e_core, exact_electronic, backend = _h2_isa_setup()
    noise_model = build_noise_model(backend)
    exact = exact_electronic + e_core

    result = zne_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        noise_model=noise_model,
        shots_per_point=200_000,
        seed=0,
    )
    raw_error = abs(result.raw_energy - exact)
    zne_error = abs(result.zne_energy - exact)
    assert raw_error > 1e-2  # sanity: noise really is biasing the raw estimate
    assert zne_error < 0.6 * raw_error


def test_zne_energy_result_reports_the_real_per_evaluation_cost() -> None:
    isa_circuit, isa_hamiltonian, e_core, _exact, backend = _h2_isa_setup()
    noise_model = build_noise_model(backend)
    result = zne_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        noise_model=noise_model,
        shots_per_point=1000,
        scale_factors=(1.0, 2.0, 3.0, 5.0),
        seed=0,
    )
    assert result.scale_factors == (1.0, 2.0, 3.0, 5.0)
    assert result.shots_per_point == 1000
