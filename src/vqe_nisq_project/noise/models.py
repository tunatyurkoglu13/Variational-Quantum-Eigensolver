"""Real IBM device noise, not hand-picked numbers.

`qiskit_aer.noise.NoiseModel.from_backend` reads a backend's actual calibration
snapshot -- per-qubit T1/T2 (energy/phase relaxation times), per-gate error rates,
per-qubit readout error -- and builds a Kraus-operator noise channel for each
instruction from it. We use `FakeTorino` (a real IBM Heron r1 133-qubit device's
frozen calibration data, already used in ansatz/metrics.py's transpiled-cost check),
so every number in this module traces back to a real device, not a textbook
"5% depolarizing noise" placeholder.

Physical picture (for the report/teacher-mode record):

- T1 (energy relaxation): the timescale for |1> -> |0> decay via spontaneous
  emission to the environment. Modeled as amplitude-damping.
- T2 (phase relaxation, T2 <= 2*T1 always): the timescale over which a
  superposition's relative phase randomizes (dephasing) -- combines T1 decay with
  pure dephasing. Modeled as phase-damping composed with the T1 channel.
- Gate error: the average process infidelity of one physical gate execution
  (thermal relaxation during the gate's finite duration + control-pulse
  imperfections), reported per-gate per-qubit(-pair) from randomized benchmarking.
- Readout error: probability the classical bit read out disagrees with the
  qubit's actual computational-basis state at measurement time -- typically the
  single largest error source per operation (compare the ~1e-4 gate errors below
  to ~1e-2 to 1e-1 readout errors), which is exactly why M3 (a mitigation step
  applied to *readout only*, added next) is worth having as a cheap first layer.

Real finding while writing `summarize_noise` below: 22 of FakeTorino's 300 CZ couplers
report `error == 1.0` exactly -- not "very noisy", but a sentinel meaning that coupler is
disabled/unusable in this calibration snapshot (a genuine real-device phenomenon: some
physical couplers get taken offline between calibration runs). A plain mean over all 300
entries gives a nonsensical ~8% average CZ error; the *median* (0.42%) is what actually
represents "a working two-qubit gate on this device" and is what `NoiseSummary` reports,
alongside the disabled-coupler count so the discrepancy is visible rather than hidden.
Readout error has the same shape (median 2.3%, but one qubit at 56.5%), so it gets the
same median treatment.

Second real finding, found later against LIVE `ibm_marrakesh` calibration data (Phase 6):
the same `error == 1.0` disabled-qubit sentinel ALSO shows up on single-qubit gates (4 of
156 qubits' `x`/`sx`/`id` gates reported `error == 1.0`) -- something FakeTorino's frozen
snapshot happened not to exhibit, which is exactly why this module originally used a plain
MEAN for one-qubit gate error and only used medians for two-qubit/readout error. That
assumption did not generalize: on `ibm_marrakesh`, the naive mean gave ~2.0% (absurd for a
Heron r2 single-qubit gate), while the median gave ~0.05% (physically sane, matching
FakeTorino's own ~0.05% one-qubit figure). Fixed by using the median (with a disabled-gate
count, mirroring the two-qubit treatment) for one-qubit gate error too -- a concrete
lesson in why one synthetic snapshot is not a substitute for checking against a second,
independent, LIVE data source before trusting a "this field doesn't need the same
treatment" judgment call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.providers import BackendV2
from qiskit_aer.noise import NoiseModel


@dataclass(frozen=True)
class NoiseSummary:
    """Device-wide averages -- for reporting/plotting, not for feeding back into the
    noise model itself (the model keeps the full per-qubit/per-gate detail).

    One-/two-qubit gate error and readout error are reported as MEDIANS (see module
    docstring: a plain mean is dominated by disabled/outlier qubits and couplers on a
    real device's calibration snapshot -- confirmed on both FakeTorino's frozen snapshot
    AND live `ibm_marrakesh` data). T1/T2 showed no comparable sentinel-value pathology
    and are reported as means."""

    backend_name: str
    n_qubits: int
    mean_t1_us: float
    mean_t2_us: float
    median_one_qubit_gate_error: float
    median_two_qubit_gate_error: float
    median_readout_error: float
    n_disabled_one_qubit_gates: int
    n_one_qubit_gates: int
    n_disabled_two_qubit_couplers: int
    n_two_qubit_couplers: int


def build_noise_model(backend: BackendV2) -> NoiseModel:
    """Thin wrapper (documented, not hidden) around Aer's own backend->noise-model
    extraction -- kept as a named function so every call site in this project states
    explicitly that noise comes from a real backend snapshot."""
    return NoiseModel.from_backend(backend)


def summarize_noise(backend: BackendV2) -> NoiseSummary:
    """Real means over `backend.target`'s calibration data -- computed here, not quoted
    from IBM's device-status page, so the numbers match whatever `build_noise_model`
    above actually uses."""
    target = backend.target
    n_qubits = target.num_qubits

    t1_s = [qp.t1 for qp in target.qubit_properties if qp is not None and qp.t1 is not None]
    t2_s = [qp.t2 for qp in target.qubit_properties if qp is not None and qp.t2 is not None]

    one_qubit_gate_names = {"x", "sx", "rz", "id"}
    two_qubit_gate_names = {"cz", "cx", "ecr"}

    one_qubit_errors: list[float] = []
    two_qubit_errors: list[float] = []
    readout_errors: list[float] = []

    for gate_name in target.operation_names:
        try:
            props = target[gate_name]
        except KeyError:
            continue
        for qargs, instr_props in props.items():
            if instr_props is None or instr_props.error is None:
                continue
            if gate_name == "measure":
                readout_errors.append(instr_props.error)
            elif gate_name in two_qubit_gate_names and qargs is not None and len(qargs) == 2:
                two_qubit_errors.append(instr_props.error)
            elif gate_name in one_qubit_gate_names:
                one_qubit_errors.append(instr_props.error)

    n_disabled_1q = sum(1 for e in one_qubit_errors if e >= 0.999)
    n_disabled_2q = sum(1 for e in two_qubit_errors if e >= 0.999)

    return NoiseSummary(
        backend_name=backend.name,
        n_qubits=n_qubits,
        mean_t1_us=float(np.mean(t1_s)) * 1e6,
        mean_t2_us=float(np.mean(t2_s)) * 1e6,
        median_one_qubit_gate_error=float(np.median(one_qubit_errors)),
        median_two_qubit_gate_error=float(np.median(two_qubit_errors)),
        median_readout_error=float(np.median(readout_errors)),
        n_disabled_one_qubit_gates=n_disabled_1q,
        n_one_qubit_gates=len(one_qubit_errors),
        n_disabled_two_qubit_couplers=n_disabled_2q,
        n_two_qubit_couplers=len(two_qubit_errors),
    )
