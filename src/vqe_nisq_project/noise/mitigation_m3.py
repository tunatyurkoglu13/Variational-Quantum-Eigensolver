"""M3 (Matrix-free Measurement Mitigation, Nation, Kandala & Temme -- Nation et al. 2021,
arXiv:2108.12518) -- corrects readout error ONLY, as a cheap first layer of error
mitigation before the gate-error-targeting techniques (ZNE, twirling) that follow.

Physical picture: a real qubit's classical readout sometimes reports the wrong bit (see
`models.py`'s module docstring -- readout error is typically the single largest per-
operation error source on real hardware, ~1-2 orders of magnitude above a single gate's
error). The textbook fix is a full 2^n x 2^n "assignment matrix" A (A[i,j] = P(measure i |
prepared j)) built from 2^n calibration circuits, then correcting counts via A^-1 -- exact,
but exponentially expensive to calibrate AND to invert. M3 instead assumes qubits' readout
errors are (to leading order) independent, so the assignment matrix is well-approximated
by a tensor product of n independent 2x2 single-qubit confusion matrices -- calibrated with
only 2*n circuits (verified below: n_qubits=4/10/12 all measured exactly 2*n calibration
circuits submitted to the simulator, not 2^n), and corrected via a matrix-free iterative
solve (never materializing the full matrix) that only touches the counts actually observed
for a given circuit, not the qubits it never used.

Cost structure (for the Phase 5 marginal-cost table): calibration is a ONE-TIME, O(n)-cost
step, reusable across every VQE iteration/measurement group afterward -- correction itself
is pure classical post-processing on counts already being collected for the energy
estimate, requiring ZERO additional circuit executions per energy evaluation. This is what
makes M3 the "cheap first layer": on real, budget-constrained hardware (Phase 6), the
calibration cost is paid once per session, not once per shot.

Real finding #1, tested three ways (this module's tests): M3's benefit under a REALISTIC
(gate + readout) noise model is genuinely non-monotonic, not just "smaller than the pure-
readout case." Isolating readout error alone (a NoiseModel with ONLY a strong
ReadoutError, no gate error) shows M3 removing the large majority of the bias, exactly as
designed. But under FakeTorino's FULL noise model on H2/EfficientSU2(reps=2, full
entanglement), different ansatz parameter points show genuinely different outcomes at
FIXED, reproducible settings: one parameter point has M3 roughly HALVE the total energy
bias (real benefit -- readout error is part of the noise budget there); a different
parameter point has M3 make the bias ~17x WORSE (the gate-error bias and the readout-
error bias happen to have opposite signs and partially cancel in the UNCORRECTED
estimate; M3 correctly removes only the readout component, as designed, which here
unmasks a larger net bias rather than shrinking it). Neither result is a bug -- partial
mitigation of a multi-channel noise budget is not guaranteed to be monotonic in the
channels it doesn't touch. This is the honest, real answer to "does M3 help here": it
depends on the specific state/circuit, motivating the gate-error-targeting techniques
(ZNE, twirling) queued up next in Phase 5 as necessary complements, not optional extras.

Real finding #2 (found BECAUSE the result above kept changing across reruns before this
was fixed): `qiskit.transpile()`'s routing/synthesis passes are not deterministic across
calls unless `seed_transpiler` is set explicitly -- see `simulate.py`'s `prepare_for_
backend` docstring (its own Real finding #3) for the full story. Every `transpile()` call
in this module accepts and forwards a `seed_transpiler` for exactly this reason; omitting
it makes any specific noisy-energy number here silently irreproducible run-to-run, which
is what made real finding #1 above look like test flakiness before the root cause (an
unseeded transpiler, not the physics) was identified.
"""

from __future__ import annotations

from dataclasses import dataclass

import mthree
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from vqe_nisq_project.chemistry.measurement import group_qubit_wise_commuting


@dataclass(frozen=True)
class M3EnergyResult:
    raw_energy: float  # uncorrected, computed from the SAME raw counts (apples-to-apples)
    mitigated_energy: float  # M3-corrected
    e_core: float
    shots: int
    n_groups: int


def calibrate(
    backend: BackendV2, n_qubits: int, noise_model: NoiseModel | None, shots: int = 8000
) -> mthree.M3Mitigation:
    """Build and calibrate an M3 mitigator against `backend` UNDER `noise_model` -- the
    mitigator must be calibrated against the SAME noise the eventual measurements will see
    (calibrating against a noiseless simulator would learn a trivial identity correction and
    silently mitigate nothing)."""
    sim = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
    mit = mthree.M3Mitigation(sim)
    mit.cals_from_system(qubits=list(range(n_qubits)), shots=shots, async_cal=False)
    return mit


def _basis_per_qubit(group: SparsePauliOp) -> list[str]:
    """The single measurement basis ('X', 'Y', 'Z', or 'I' if untouched) each qubit needs
    for this qubit-wise-commuting group -- well-defined because every term in the group
    agrees on which single Pauli (if any) appears at each qubit position. Pauli label
    strings are Qiskit's usual little-endian convention (leftmost char = highest qubit
    index), so `basis[q]` is read off position `n_qubits - 1 - q` of each label, i.e. from
    `reversed(label)` at index `q`."""
    n_qubits = group.num_qubits
    basis = ["I"] * n_qubits
    for label in group.paulis.to_labels():
        for q, ch in enumerate(reversed(label)):
            if ch != "I":
                basis[q] = ch
    return basis


def _measurement_circuit(n_qubits: int, basis: list[str]) -> QuantumCircuit:
    """Rotate each qubit's stated Pauli basis into the computational (Z) basis, then
    measure all qubits. X -> H (H X H = Z). Y -> Sdg then H (verified by hand: Sdg maps the
    Y eigenstates |+i>/|-i> to |+>/|->, and H then maps those to |0>/|1>)."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        if basis[q] == "X":
            qc.h(q)
        elif basis[q] == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _group_to_diagonal_exp_ops(group: SparsePauliOp) -> list[str]:
    """After the basis rotation above, every non-identity character in a term's label is
    measured as Z regardless of whether it started as X, Y, or Z -- so each term's exp_ops
    string for `mthree`'s `expval`/`exp_val` is its own label with every X/Y replaced by Z
    (I stays I). Reuses the SAME little-endian convention as the Pauli labels themselves --
    verified empirically (not assumed) against known computational-basis states with
    asymmetric bit patterns before trusting this direct reuse (see this module's tests)."""
    return [label.replace("X", "Z").replace("Y", "Z") for label in group.paulis.to_labels()]


def estimate_energy_with_readout_mitigation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    e_core: float,
    backend: BackendV2,
    mitigator: mthree.M3Mitigation,
    *,
    noise_model: NoiseModel | None = None,
    shots: int = 8192,
    seed: int | None = None,
    seed_transpiler: int | None = None,
) -> M3EnergyResult:
    """Measure `hamiltonian`'s expectation value via qubit-wise-commuting measurement
    groups (`chemistry.measurement.group_qubit_wise_commuting`), reporting BOTH the raw
    (uncorrected) and M3-mitigated energy computed from the exact same counts -- isolating
    M3's own marginal contribution rather than comparing across independently-sampled runs.

    `circuit` must already have its parameters bound (no free `Parameter`s) and be the
    ORIGINAL (non-ISA) ansatz circuit -- this function transpiles internally per
    measurement group, since each group appends a different basis-rotation circuit before
    transpiling. Pass `seed_transpiler` for reproducible results across runs -- see
    `simulate.py`'s `prepare_for_backend` docstring (Real finding #3) for why this matters
    even when `seed` (the simulator seed) is already fixed."""
    if circuit.num_parameters != 0:
        raise ValueError(
            f"circuit has {circuit.num_parameters} unbound parameter(s); "
            "bind them (circuit.assign_parameters(...)) before calling this function."
        )

    n_qubits = hamiltonian.num_qubits
    grouping = group_qubit_wise_commuting(hamiltonian)
    sim = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()

    raw_energy = 0.0
    mitigated_energy = 0.0
    for group in grouping.groups:
        basis = _basis_per_qubit(group)
        meas_circuit = _measurement_circuit(n_qubits, basis)
        full_circuit = circuit.compose(meas_circuit)
        isa_circuit = transpile(
            full_circuit,
            backend=backend,
            optimization_level=1,
            initial_layout=list(range(n_qubits)),
            seed_transpiler=seed_transpiler,
        )

        run_kwargs = {"shots": shots}
        if seed is not None:
            run_kwargs["seed_simulator"] = seed
        result = sim.run(isa_circuit, **run_kwargs).result()
        counts = result.get_counts()
        total = sum(counts.values())
        raw_probs = {bitstring: count / total for bitstring, count in counts.items()}

        exp_ops_list = _group_to_diagonal_exp_ops(group)
        quasi = mitigator.apply_correction(counts, qubits=list(range(n_qubits)))
        for exp_ops, coeff in zip(exp_ops_list, group.coeffs, strict=True):
            raw_energy += coeff.real * mthree.classes.exp_val(raw_probs, exp_ops=exp_ops)
            mitigated_energy += coeff.real * quasi.expval(exp_ops)

    return M3EnergyResult(
        raw_energy=raw_energy + e_core,
        mitigated_energy=mitigated_energy + e_core,
        e_core=e_core,
        shots=shots,
        n_groups=grouping.n_groups,
    )
