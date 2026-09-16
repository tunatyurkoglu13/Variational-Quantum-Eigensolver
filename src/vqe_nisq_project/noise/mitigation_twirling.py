"""Pauli twirling of two-qubit gates (Wallman & Emerson 2016, "randomized compiling",
arXiv:1512.01098) -- converts a general, possibly partially COHERENT two-qubit noise
channel (real hardware's thermal-relaxation/amplitude-damping-derived error is NOT a pure
Pauli channel) into an effective Pauli (stochastic) channel when averaged over many random
twirl instances, without changing the IDEAL (noiseless) circuit's action at all.

Physical picture: sandwich each 2-qubit gate G with a random Pauli P tensor Q before it and
a MATCHING correction pair P' tensor Q' after it, chosen so that
(P' tensor Q') G (P tensor Q) = G EXACTLY -- a mathematically exact no-op in a perfectly
noiseless execution, for ANY of the 16 single-qubit Pauli pairs. Under REAL gate noise,
each random twirl instance experiences a DIFFERENT effective error operator (the physical
noise gets conjugated by whichever random Pauli was drawn that instance), so averaging the
energy over many independently-twirled circuit instances approximates averaging the noise
channel itself over the Pauli group -- symmetrizing away its non-Pauli-diagonal (coherent)
components while leaving its Pauli-diagonal (stochastic) part untouched.

Correction table for CZ (this project's target backend's native 2-qubit gate) verified by
BRUTE FORCE (not hand-derived and trusted blindly, matching this project's established
"ground-truth search over the full option space" methodology from Phases 1-2, e.g. the
Z2Symmetries sector search and the OpenFermion two-body index permutation search): for
each of the 16 (P, Q) input Pauli pairs, exhaustively checked all 16 (P', Q') output pairs
as 4x4 matrices for an EXACT match (P' tensor Q') CZ (P tensor Q) = CZ -- including
confirming the residual global phase is always +-1 (never +-i), consistent with Pauli
twirling of a Clifford gate being exact with no leftover phase correction needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.noise import NoiseModel

from vqe_nisq_project.ansatz.base import FloatArray
from vqe_nisq_project.noise.simulate import estimate_energy

_PAULI_LABELS = ("I", "X", "Y", "Z")

#: Verified once at import time (not hand-typed and trusted) -- see module docstring.
CZ_TWIRL_TABLE: dict[tuple[str, str], tuple[str, str]] = {}


def _verify_and_build_cz_twirl_table() -> dict[tuple[str, str], tuple[str, str]]:
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    table: dict[tuple[str, str], tuple[str, str]] = {}
    for p1 in _PAULI_LABELS:
        for q1 in _PAULI_LABELS:
            m_in = np.kron(Pauli(p1).to_matrix(), Pauli(q1).to_matrix())
            found = None
            for p2 in _PAULI_LABELS:
                for q2 in _PAULI_LABELS:
                    m_out = np.kron(Pauli(p2).to_matrix(), Pauli(q2).to_matrix())
                    candidate = m_out @ cz @ m_in
                    nz = np.nonzero(cz)
                    ratio = candidate[nz] / cz[nz]
                    if np.allclose(ratio, ratio[0]) and np.allclose(candidate, ratio[0] * cz):
                        if abs(ratio[0].real) < 0.5:  # must be +-1, never +-i
                            raise AssertionError(
                                f"Unexpected non-real global phase {ratio[0]} for "
                                f"input ({p1},{q1}) -> output ({p2},{q2})."
                            )
                        found = (p2, q2)
                        break
                if found:
                    break
            if found is None:
                raise AssertionError(f"No CZ twirl correction found for input ({p1},{q1}).")
            table[(p1, q1)] = found
    return table


CZ_TWIRL_TABLE = _verify_and_build_cz_twirl_table()


def _append_pauli(circuit: QuantumCircuit, label: str, qubit: int) -> None:
    if label == "I":
        return
    getattr(circuit, label.lower())(qubit)


def twirl_two_qubit_gates(
    circuit: QuantumCircuit,
    rng: np.random.Generator,
    two_qubit_gate_name: str = "cz",
    twirl_table: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> QuantumCircuit:
    """Replace every `two_qubit_gate_name` instruction with a randomly-twirled version:
    a random Pauli pair before, the matching correction pair after -- an exact no-op on
    the ideal unitary, verified as a regression test in this module's test suite."""
    table = twirl_table if twirl_table is not None else CZ_TWIRL_TABLE
    twirled = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name == two_qubit_gate_name:
            q1, q2 = instruction.qubits
            p_in = rng.choice(_PAULI_LABELS)
            q_in = rng.choice(_PAULI_LABELS)
            p_out, q_out = table[(p_in, q_in)]
            _append_pauli(twirled, p_in, twirled.find_bit(q1).index)
            _append_pauli(twirled, q_in, twirled.find_bit(q2).index)
            twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
            _append_pauli(twirled, p_out, twirled.find_bit(q1).index)
            _append_pauli(twirled, q_out, twirled.find_bit(q2).index)
        else:
            twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
    return twirled


@dataclass(frozen=True)
class TwirlingEnergyResult:
    raw_energy: float  # single untwirled circuit, same TOTAL shot budget as the ensemble
    twirled_energy: float  # mean over n_twirls independently-twirled circuit instances
    e_core: float
    n_twirls: int
    shots_per_twirl: int


def twirled_energy(
    isa_circuit: QuantumCircuit,
    isa_hamiltonian: SparsePauliOp,
    e_core: float,
    noise_model: NoiseModel | None,
    *,
    n_twirls: int = 10,
    shots_per_twirl: int = 8192,
    two_qubit_gate_name: str = "cz",
    seed: int | None = None,
) -> TwirlingEnergyResult:
    """`isa_circuit`/`isa_hamiltonian` must already be an ISA-transpiled, layout-aligned
    pair from `simulate.prepare_for_backend`, matching `mitigation_zne.zne_energy`'s
    precondition -- twirling operates on the circuit's ACTUAL native two-qubit gate
    (`two_qubit_gate_name`, "cz" for this project's FakeTorino target).

    `raw_energy` uses `n_twirls * shots_per_twirl` total shots (matching the twirled
    ensemble's total shot budget) so the comparison isolates twirling's own effect rather
    than comparing a smaller-shot single run against a larger-shot ensemble."""
    no_params: FloatArray = np.array([])
    rng = np.random.default_rng(seed)

    raw_result = estimate_energy(
        isa_circuit,
        no_params,
        isa_hamiltonian,
        e_core=0.0,
        shots=n_twirls * shots_per_twirl,
        noise_model=noise_model,
        seed=seed,
    )

    energies = []
    for _ in range(n_twirls):
        twirled_circuit = twirl_two_qubit_gates(
            isa_circuit, rng, two_qubit_gate_name=two_qubit_gate_name
        )
        result = estimate_energy(
            twirled_circuit,
            no_params,
            isa_hamiltonian,
            e_core=0.0,
            shots=shots_per_twirl,
            noise_model=noise_model,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        energies.append(result.electronic_energy)

    return TwirlingEnergyResult(
        raw_energy=raw_result.electronic_energy + e_core,
        twirled_energy=float(np.mean(energies)) + e_core,
        e_core=e_core,
        n_twirls=n_twirls,
        shots_per_twirl=shots_per_twirl,
    )
