"""Shot-based energy estimation through Qiskit's `EstimatorV2` primitive, backed by
`AerSimulator` with an optional `NoiseModel`.

Using the *primitive* interface (not a bare `AerSimulator.run`) is a deliberate choice:
`EstimatorV2` is the same abstraction `qiskit-ibm-runtime`'s hardware Estimator exposes,
so swapping this module's simulator-backed estimator for a real-QPU one in Phase 6 is a
constructor change, not a rewrite of any VQE/mitigation code built on top of it.

Two independent noise sources are deliberately kept distinguishable:
1. Shot noise -- present even with `noise_model=None`, from estimating an expectation
   value with a finite number of measurements. Unbiased: averages to the exact value.
2. Device noise -- from `noise_model`, modeling real gate/readout errors. Biased: shifts
   the *mean* estimate, not just its spread, because errors alter which state is actually
   being measured, not just how many times we sample it. See `models.py`'s module
   docstring for the physical origin of this bias.

Real finding #1 (the most dangerous one, found by a device-noise-bias test that kept
reporting exactly ZERO bias no matter how deep/entangled the circuit got): a `NoiseModel`
built from a backend's calibration data defines error channels keyed by
`(basis_gate_name, qubits)` pairs -- e.g. `('cz', (0, 1))`, `('sx', (3,))` -- taken
literally from that backend's native gate set. Handing `AerSimulator(noise_model=...)` a
circuit still expressed in its *original* gates (our ansatz builders emit `ry`/`rz`/`cx`,
not FakeTorino's `sx`/`rz`/`cz`) makes the simulator execute every one of those
instructions as an *ideal, noise-free* gate, silently -- no error, no warning, and a
perfectly plausible-looking (just wrong) result. The fix, matching Qiskit's own ISA
("Instruction Set Architecture") circuit convention for real hardware, is
`prepare_for_backend` below: `transpile(..., backend=backend)` to rewrite the circuit into
the backend's actual basis gates on the backend's actual physical qubits, THEN
`hamiltonian.apply_layout(isa_circuit.layout)` to permute the observable's qubit labels to
match wherever the transpiler's routing/layout pass actually placed each logical qubit
(routing can insert SWAPs, changing which physical qubit ends up holding which logical
one -- `apply_layout` accounts for this exactly, `estimate_energy` alone does not).
Verified end to end: on H2/EfficientSU2(reps=2, full entanglement), the naive (untranspiled)
noisy-vs-noiseless call pair gave IDENTICAL exact expectation values (diff ~1e-16, i.e. no
noise applied at all) across every entanglement pattern and depth tried; after routing
through `prepare_for_backend`, the exact (pre-shot-noise) expectation value correctly
matches the ideal statevector result when `noise_model=None`, and shows a real ~5 mHa
bias -- already above the 1.6 mHa chemical-accuracy threshold, even for this shallow
4-qubit circuit -- once FakeTorino's real noise model is applied.

Real finding #2, found by a failing determinism test (same seed, two different results):
`qiskit_aer.primitives.EstimatorV2` does NOT literally sample individual measurement
shots. Internally it runs Aer's `save_expectation_value` instruction, which computes the
EXACT expectation value under the noisy channel's full density-matrix evolution (so the
device-noise *bias* above is physically exact, not itself approximated), then synthesizes
the requested shot noise by drawing ONE sample from `Normal(exact_biased_mean, precision)`
-- a Gaussian approximation to the true (binomial/multinomial) finite-sample distribution,
not a bit-for-bit measurement simulation. This is a standard, documented Aer performance
shortcut (also why a single "shots" number this module reports is really parametrizing an
assumed noise scale via `shots_to_precision`, not literally driving a shot loop) -- flagged
here so it is not mistaken for literal per-shot Monte Carlo sampling when read later.
Practically this also means the RNG seed must be passed via `run_options`, NOT
`backend_options` (`EstimatorV2._run_pub` reads `self.options.run_options["seed_simulator"]`
for that Gaussian draw specifically) -- passing it into `backend_options` instead (an
easy mistake, since `noise_model` DOES belong there) silently produces a non-deterministic
result despite a seed being set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2

from vqe_nisq_project.ansatz.base import FloatArray


@dataclass(frozen=True)
class NoisyEnergyResult:
    energy: float  # electronic expectation value + e_core
    electronic_energy: float  # raw <H_electronic> estimate, no e_core
    std_error: float  # EstimatorV2's own target standard error (see `shots_to_precision`)
    shots: int
    noisy: bool


def prepare_for_backend(
    circuit: QuantumCircuit, hamiltonian: SparsePauliOp, backend: BackendV2
) -> tuple[QuantumCircuit, SparsePauliOp]:
    """Rewrite `circuit` into `backend`'s native basis gates/physical qubits (an "ISA
    circuit") and permute `hamiltonian`'s qubit labels to match -- REQUIRED before passing
    a noise_model built from that same backend to `estimate_energy`, or the noise model
    silently has no effect at all (see module docstring, Real finding #1)."""
    isa_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        initial_layout=list(range(circuit.num_qubits)),
    )
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)
    return isa_circuit, isa_hamiltonian


def shots_to_precision(shots: int) -> float:
    """`EstimatorV2.run` takes a target standard error ('precision'), not a raw shot
    count. For a Pauli observable with eigenvalues in {-1, +1}, the sample-mean standard
    error after N shots is sigma/sqrt(N) <= 1/sqrt(N) (since Var<=1 always for a +/-1
    random variable) -- so 1/sqrt(shots) is the standard, slightly-conservative
    shots<->precision relation used throughout Qiskit's own primitive documentation."""
    return float(1.0 / np.sqrt(shots))


def estimate_energy(
    circuit: QuantumCircuit,
    params: FloatArray,
    hamiltonian: SparsePauliOp,
    e_core: float,
    *,
    shots: int = 8192,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> NoisyEnergyResult:
    """One shot-based energy estimate at fixed `params` -- the noisy analogue of
    `optimization.gradients.expectation`, used both standalone (Phase 5's noise-impact
    measurements) and as the cost-function backend a VQE loop could call (Phase 6).

    Precondition when `noise_model is not None`: `circuit`/`hamiltonian` must already be
    an ISA-transpiled pair produced by `prepare_for_backend` against that SAME backend --
    otherwise the noise model silently applies no noise at all (see module docstring)."""
    backend_options: dict[str, object] = {}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    run_options: dict[str, object] = {}
    if seed is not None:
        run_options["seed_simulator"] = seed  # must be run_options, not backend_options -- see
        # module docstring for why (the seed drives EstimatorV2's own synthetic-shot-noise
        # RNG, a different code path than AerSimulator's own noise-channel sampling).

    estimator = EstimatorV2(
        options={"backend_options": backend_options, "run_options": run_options}
    )
    precision = shots_to_precision(shots)
    job = estimator.run([(circuit, hamiltonian, params)], precision=precision)
    pub_result = job.result()[0]

    electronic_energy = float(pub_result.data.evs)
    std_error = float(pub_result.data.stds)
    return NoisyEnergyResult(
        energy=electronic_energy + e_core,
        electronic_energy=electronic_energy,
        std_error=std_error,
        shots=shots,
        noisy=noise_model is not None,
    )
