"""ZNE (Zero-Noise Extrapolation -- Temme, Bravyi & Gambetta 2017, arXiv:1612.02058;
unitary-folding scale-up: Giurgica-Tiron et al. 2020, arXiv:2005.10921), via `mitiq`.
Unlike M3 (readout-only), ZNE directly targets GATE error -- the honest limit M3's own
tests documented (`mitigation_m3.py`).

Physical picture: unitary folding replaces a gate G with the triple G G^dagger G. Since
G^dagger G = I exactly, this triple is logically IDENTICAL to a single G -- a noiseless
circuit gives the exact same result whether or not any gate is folded. But a REAL,
noisy execution pays that gate's error three times instead of once (three physical gate
executions instead of one), so folding every gate once (`scale_factor=3`) simulates
running the whole circuit at roughly 3x its native noise level, without changing any
actual hardware parameter -- purely by making the circuit longer in a way that is a
no-op only in the absence of noise. Measuring the energy at several such scale factors
(here: 1, 2, 3 -- 1 meaning the unfolded circuit) traces out E(scale_factor), and
extrapolating that curve back to the (never actually executed) scale_factor=0 point
estimates the zero-noise answer.

Verified before trusting this pipeline (this module's tests): under NO noise model at
all, folding is a true no-op (G G^dagger G behaves as G, confirmed: the "raw" (scale=1)
and ZNE-extrapolated energies agreed to ~2e-8 Ha at 500k shots, i.e. down to the shot-
noise floor) -- ZNE only has an effect when there is real noise to reveal.

Cost structure (contrast `mitigation_m3.py`'s near-zero marginal cost): ZNE requires
`len(scale_factors)` FULL circuit executions per energy evaluation (each at its own full
shot budget) -- unlike M3, this is a genuine, repeated per-evaluation cost, not a one-time
calibration. For Phase 6's shot-budget-constrained real hardware runs, ZNE's cost scales
linearly with the number of scale factors used, a real number for the marginal-cost table.

Real dependency finding: `mitiq`'s Qiskit-circuit support round-trips through Cirq's
OpenQASM parser (`cirq.contrib.qasm_import`), which needs the `ply` (Python Lex-Yacc)
package -- not declared as a dependency by either `mitiq` or `cirq-core` in this
project's pinned versions, so calling `mitiq.zne.execute_with_zne` on a Qiskit circuit
raised `ModuleNotFoundError: No module named 'ply'` until `ply` was added directly to
this project's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from mitiq import zne
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel

from vqe_nisq_project.ansatz.base import FloatArray
from vqe_nisq_project.noise.simulate import estimate_energy

FactoryKind = Literal["linear", "richardson"]


@dataclass(frozen=True)
class ZneEnergyResult:
    raw_energy: float  # unfolded (scale_factor=1) energy, same executor/shots as ZNE uses
    zne_energy: float  # extrapolated to the zero-noise limit
    e_core: float
    shots_per_point: int
    scale_factors: tuple[float, ...]


def _make_factory(kind: FactoryKind, scale_factors: tuple[float, ...]) -> zne.inference.Factory:
    if kind == "linear":
        return zne.inference.LinearFactory(scale_factors=list(scale_factors))
    return zne.inference.RichardsonFactory(scale_factors=list(scale_factors))


def zne_energy(
    isa_circuit: QuantumCircuit,
    isa_hamiltonian: SparsePauliOp,
    e_core: float,
    noise_model: NoiseModel | None,
    *,
    shots_per_point: int = 8192,
    scale_factors: tuple[float, ...] = (1.0, 2.0, 3.0),
    factory_kind: FactoryKind = "linear",
    seed: int | None = None,
) -> ZneEnergyResult:
    """`isa_circuit`/`isa_hamiltonian` must already be an ISA-transpiled, layout-aligned
    pair from `simulate.prepare_for_backend` -- folding is applied to `isa_circuit`'s
    ACTUAL (backend-native) gates, which must survive unchanged into execution, so no
    further transpilation happens after folding (transpiling a folded circuit again could
    simplify the deliberately-redundant G G^dagger G triples back away)."""
    no_params: FloatArray = np.array([])

    def executor(circuit: QuantumCircuit) -> float:
        result = estimate_energy(
            circuit,
            no_params,
            isa_hamiltonian,
            e_core=0.0,
            shots=shots_per_point,
            noise_model=noise_model,
            seed=seed,
        )
        return result.electronic_energy

    factory = _make_factory(factory_kind, scale_factors)
    raw_electronic = executor(isa_circuit)
    zne_electronic = zne.execute_with_zne(isa_circuit, executor, factory=factory)

    return ZneEnergyResult(
        raw_energy=raw_electronic + e_core,
        zne_energy=zne_electronic + e_core,
        e_core=e_core,
        shots_per_point=shots_per_point,
        scale_factors=scale_factors,
    )
