"""Dynamical decoupling (DD) -- inserts a pulse sequence (here: the simplest 2-pulse spin
echo, X ... X) into IDLE time windows of a scheduled circuit. In the noiseless case this
is an exact no-op (X X = I), but on real hardware it is meant to refocus slowly-varying
("quasi-static"/1-over-f) dephasing noise accumulated while a qubit is waiting idle for
other qubits' gates to finish -- the classic NMR spin-echo trick applied to idle qubits
in a quantum circuit.

Real finding #1 (verified BEFORE building the full pipeline, via an isolated single-qubit
test): `qiskit_aer`'s `NoiseModel.from_backend` attaches THERMAL RELAXATION noise to
arbitrary-duration `Delay` instructions dynamically (confirmed: a lone qubit's <X> decays
from 1.0 to ~0.73 over a 100us idle `Delay`, matching real T1/T2 physics) -- so idle time
is NOT silently noise-free in this simulator, DD has real decoherence to act on. BUT: a
direct echo test (one 100us delay vs. two 50us delays with an X-X pair in between) showed
IDENTICAL decay in both cases (0.7277 vs 0.7267, the tiny remaining difference being the
X gates' own small gate error, not a decoupling benefit). Root cause: Aer's
`thermal_relaxation_error(t1, t2, duration)` is a plain MEMORYLESS (Markovian) channel
with no quasi-static/1-over-f component -- there is no "slowly varying phase" for a spin
echo to refocus in this specific noise model, so DD is expected, by construction of the
simulated noise model itself, to show little to no benefit here. This mirrors
`mitigation_twirling.py`'s finding (FakeTorino's CZ error already being near-Pauli-
diagonal) -- both are real limitations of `NoiseModel.from_backend`'s simplified error
model, not evidence that DD/twirling are bad ideas on real hardware.

Real finding #2 (a genuine engineering blocker, not a physics finding): naively running
`ALAPScheduleAnalysis` + `PadDynamicalDecoupling` on an ISA circuit that was transpiled
against the FULL 133-qubit `FakeTorino` backend (this project's `prepare_for_backend`
convention) inserts explicit `Delay` instructions across ALL 133 qubits (129 of which are
otherwise entirely untouched spectators). This breaks `AerSimulator`'s automatic
idle-qubit truncation optimization (a circuit with NO instructions at all on a qubit is
trivially excluded from the simulated register; one with an explicit `Delay` on it,
apparently, is not) -- attempting to simulate the DD-padded circuit as-is raised
`Insufficient memory ... Required memory: 18446744073709551615M` (statevector method
implicitly trying to simulate all 133 qubits). Fixed by restricting BOTH the circuit and
the Hamiltonian down to only the qubits the Hamiltonian actually acts on
(`_restrict_to_active_qubits`/`_restrict_observable_to_active_qubits`) immediately after
DD-padding, before any simulation -- verified safe by checking no instruction spans both
an active and an inactive qubit, and that the Hamiltonian's terms are all-`I` outside the
active positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit_aer.noise import NoiseModel

from vqe_nisq_project.ansatz.base import FloatArray
from vqe_nisq_project.noise.simulate import estimate_energy


def _active_qubits_from_hamiltonian(hamiltonian: SparsePauliOp) -> list[int]:
    """Which qubit positions carry any non-identity Pauli in ANY term -- little-endian
    convention (rightmost label character = qubit 0), same as used throughout this
    project's Pauli-label handling (e.g. `mitigation_m3.py`'s `_basis_per_qubit`)."""
    active: set[int] = set()
    for label in hamiltonian.paulis.to_labels():
        for q, ch in enumerate(reversed(label)):
            if ch != "I":
                active.add(q)
    return sorted(active)


def _restrict_to_active_qubits(circuit: QuantumCircuit, active_qubits: list[int]) -> QuantumCircuit:
    active = set(active_qubits)
    restricted = QuantumCircuit(len(active_qubits))
    index_map = {q: i for i, q in enumerate(active_qubits)}
    for instruction in circuit.data:
        physical = [circuit.find_bit(q).index for q in instruction.qubits]
        touches_active = any(p in active for p in physical)
        touches_inactive = any(p not in active for p in physical)
        if touches_active and touches_inactive:
            raise ValueError(
                f"Instruction {instruction.operation.name} spans both active and "
                "inactive qubits -- cannot safely restrict this circuit."
            )
        if touches_active:
            restricted.append(
                instruction.operation, [restricted.qubits[index_map[p]] for p in physical]
            )
    return restricted


def _restrict_observable_to_active_qubits(
    hamiltonian: SparsePauliOp, active_qubits: list[int]
) -> SparsePauliOp:
    active_set = set(active_qubits)
    restricted_labels = []
    for label in hamiltonian.paulis.to_labels():
        for q, ch in enumerate(reversed(label)):
            if ch != "I" and q not in active_set:
                raise ValueError(
                    f"Hamiltonian term {label!r} has a non-identity operator on inactive "
                    f"qubit {q} -- cannot safely restrict this observable."
                )
        restricted_labels.append(
            "".join(label[len(label) - 1 - q] for q in reversed(active_qubits))
        )
    return SparsePauliOp(restricted_labels, coeffs=hamiltonian.coeffs).simplify()


def apply_dynamical_decoupling(
    isa_circuit: QuantumCircuit,
    isa_hamiltonian: SparsePauliOp,
    backend: BackendV2,
) -> tuple[QuantumCircuit, SparsePauliOp]:
    """Schedule `isa_circuit` (an ISA-transpiled, layout-aligned circuit from
    `simulate.prepare_for_backend`, run against the full-width `backend`) and pad its
    idle windows with an X-X spin-echo sequence, restricted to the qubits
    `isa_hamiltonian` actually acts on -- then restrict both circuit and Hamiltonian back
    down to just those qubits so the result stays simulable (see module docstring, Real
    finding #2)."""
    active_qubits = _active_qubits_from_hamiltonian(isa_hamiltonian)
    durations = backend.target.durations()
    pass_manager = PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, dd_sequence=[XGate(), XGate()], qubits=active_qubits),
        ]
    )
    dd_circuit_full = pass_manager.run(isa_circuit)
    dd_circuit = _restrict_to_active_qubits(dd_circuit_full, active_qubits)
    dd_hamiltonian = _restrict_observable_to_active_qubits(isa_hamiltonian, active_qubits)
    return dd_circuit, dd_hamiltonian


@dataclass(frozen=True)
class DdEnergyResult:
    raw_energy: float
    dd_energy: float
    e_core: float
    shots: int


def dd_energy(
    isa_circuit: QuantumCircuit,
    isa_hamiltonian: SparsePauliOp,
    e_core: float,
    backend: BackendV2,
    noise_model: NoiseModel | None,
    *,
    shots: int = 8192,
    seed: int | None = None,
) -> DdEnergyResult:
    """Compare the energy estimate with vs. without DD padding, on matched shot budgets
    and, since `apply_dynamical_decoupling` restricts down to the active qubits, matched
    (smaller) Hamiltonians/circuits for BOTH the raw and DD-padded evaluation -- so `raw`
    here is measured on the SAME restricted representation as `dd`, isolating DD's own
    effect rather than any incidental effect of the restriction step itself."""
    active_qubits = _active_qubits_from_hamiltonian(isa_hamiltonian)
    restricted_circuit = _restrict_to_active_qubits(isa_circuit, active_qubits)
    restricted_hamiltonian = _restrict_observable_to_active_qubits(isa_hamiltonian, active_qubits)
    dd_circuit, dd_hamiltonian = apply_dynamical_decoupling(isa_circuit, isa_hamiltonian, backend)

    no_params: FloatArray = np.array([])
    raw_result = estimate_energy(
        restricted_circuit,
        no_params,
        restricted_hamiltonian,
        e_core=0.0,
        shots=shots,
        noise_model=noise_model,
        seed=seed,
    )
    dd_result = estimate_energy(
        dd_circuit,
        no_params,
        dd_hamiltonian,
        e_core=0.0,
        shots=shots,
        noise_model=noise_model,
        seed=seed,
    )
    return DdEnergyResult(
        raw_energy=raw_result.electronic_energy + e_core,
        dd_energy=dd_result.electronic_energy + e_core,
        e_core=e_core,
        shots=shots,
    )
