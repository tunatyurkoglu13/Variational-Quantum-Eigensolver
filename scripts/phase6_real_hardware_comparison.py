"""Phase 6 deliverable: real IBM Quantum hardware results vs. this project's Phase 5
Aer-simulated predictions, on the SAME H2/EfficientSU2(reps=2, full entanglement) circuit
at the SAME parameters (seed=2) used throughout Phase 5's mitigation study.

WARNING -- this script SUBMITS REAL JOBS to IBM Quantum hardware and consumes your
account's QPU-time quota (10 minutes / 28 days on the free Open Plan). It is NOT part of
the automated test suite and should not be run casually or in CI. Run it only if you
understand it will consume real, limited quota. Requires a working IBM Quantum account
(IBM_QUANTUM_API_TOKEN in .env, registered via `QiskitRuntimeService.save_account`, with
a Qiskit Runtime "instance" provisioned -- see SETUP.md section 5).

This file documents (does not need to re-run) the actual 2026-09-17 session: 4 real jobs
on `ibm_marrakesh` (Heron r2, 156 qubits), totaling 55 seconds of QPU time (9.2% of the
monthly free quota) -- job IDs and results recorded below for audit/reproducibility.

THE HEADLINE FINDING: real hardware showed roughly 1.5-3x MORE bias than EITHER of two
Aer-simulated predictions -- FakeTorino's frozen Heron r1 snapshot (Phase 5) AND a fresh
Aer NoiseModel built from `ibm_marrakesh`'s own LIVE calibration data (computed here, at
zero QPU cost). Since even the live-calibrated simulation (12.7 mHa predicted) undershot
the real result (41.5 mHa raw) by more than the FakeTorino snapshot did (23.3 mHa), the
gap is NOT explained by "the simulator was using stale/wrong calibration numbers" -- it
is `qiskit_aer.noise.NoiseModel.from_backend`'s underlying construction (depolarizing +
thermal relaxation only) missing noise structure real hardware has (coherent errors,
crosstalk, leakage) REGARDLESS of which real device's numbers feed into it. This directly
extends Phase 5's twirling/DD finding (FakeTorino's noise model too simple to contain the
coherent structure those techniques target) into a hardware-verified, stronger claim: the
Aer noisy-simulation methodology itself, not just one particular snapshot's staleness,
underestimates real NISQ bias for this circuit family.

Real job results (2026-09-17, ibm_marrakesh, precision=0.02 except the first exploratory
job at precision=0.05):

    Job ID                  resilience_level   electronic energy   std      usage(s)
    dam47cv8gn2s739kqu8g    1 (exploratory)     -0.900305           0.0295   12
    dam48p78gn2s739kqvn0    0 (raw)             -0.891409           0.0110    6
    dam49eo2fm4c73f2dftg    2 (ZNE-like)        -0.957536           0.0288   22
    dam4a6lr85ps73fc89eg    1 (M3-like, rerun)  -0.900184           0.0110   15
                                                                     TOTAL:   55 s

Exact (noiseless statevector) electronic energy: -0.9328669822090274 Ha (e_core =
0.7199689944489797 Ha, so total exact energy -0.212898 Ha, matching Phase 5's own H2/JW
reference throughout).

+-----------------------+------------------+---------------------------+
| Technique             | Phase 5 (Aer sim)| Phase 6 (real hardware)   |
+-----------------------+------------------+---------------------------+
| Raw (no mitigation)   | 23.3 mHa         | 41.5 +/- 11.0 mHa         |
| M3-like (resilience 1)| 11-12 mHa        | 32.7 +/- 11.0 mHa         |
| ZNE-like (resilience 2)| 7.6-8.1 mHa     | 24.7 +/- 28.8 mHa         |
+-----------------------+------------------+---------------------------+

Ranking (best to worst) is preserved between simulation and hardware (ZNE-like best,
raw worst on both), but the ABSOLUTE bias is uniformly larger on real hardware -- the
relative benefit of each mitigation technique held up; the absolute noise-floor
prediction did not.

Usage:
    uv run python scripts/phase6_real_hardware_comparison.py
    (re-runs the free, zero-cost live-calibration cross-check simulation only; does NOT
    resubmit the real hardware jobs -- those are fetched by job ID for the record above.
    Pass --resubmit to actually submit new real jobs, at your own quota's expense.)
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import estimate_energy, prepare_for_backend


def _retry[T](fn: Callable[[], T], attempts: int = 6, delay: float = 5.0) -> T:
    """IBM's `global-search-tagging` instance-validation endpoint has shown transient
    DNS resolution failures during this project's Phase 6 sessions -- unrelated to
    account/token validity (confirmed: identical calls succeed moments later). Retrying
    a few times with a short delay is cheap insurance against exactly that, not a sign
    anything else is wrong."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            print(f"  (attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}, retrying...)")
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


BACKEND_NAME = "ibm_marrakesh"
RECORDED_JOB_IDS = {
    1: "dam4a6lr85ps73fc89eg",  # resilience_level=1, precision=0.02 (fair-comparison rerun)
    0: "dam48p78gn2s739kqvn0",  # resilience_level=0
    2: "dam49eo2fm4c73f2dftg",  # resilience_level=2
}


def _build_h2_system(
    backend: BackendV2,
) -> tuple[QuantumCircuit, SparsePauliOp, float, float]:
    integrals = compute_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    assert ansatz.qiskit_circuit is not None
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    e_core = integrals.e_core
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )
    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    return isa_circuit, isa_hamiltonian, e_core, exact_electronic + e_core


def free_live_calibration_cross_check(service: QiskitRuntimeService) -> None:
    """Zero QPU cost: builds an Aer NoiseModel from the target backend's CURRENT live
    calibration data and simulates the same circuit classically -- isolates whether the
    simulation-vs-hardware gap is due to stale calibration numbers (it is not, see
    module docstring)."""
    backend = _retry(lambda: service.backend(BACKEND_NAME))
    isa_circuit, isa_hamiltonian, e_core, exact = _build_h2_system(backend)
    live_noise_model = build_noise_model(backend)
    no_params = np.array([])
    sim_result = estimate_energy(
        isa_circuit,
        no_params,
        isa_hamiltonian,
        e_core,
        shots=200_000,
        noise_model=live_noise_model,
        seed=0,
    )
    error_mha = abs(sim_result.energy - exact) * 1000.0
    print(f"Live-calibrated Aer simulation (free, no QPU time): {error_mha:.2f} mHa error")
    print("(compare: real hardware raw error was 41.5 mHa -- the simulation still")
    print(" undershoots, confirming the gap is methodological, not stale-calibration.)")


def _fetch_one_result(service: QiskitRuntimeService, job_id: str) -> tuple[float, float, int]:
    job = _retry(lambda: service.job(job_id))
    result = _retry(lambda: job.result())[0]
    usage = _retry(lambda: job.usage())
    return float(result.data.evs), float(result.data.stds), usage


def fetch_recorded_results(service: QiskitRuntimeService, exact_electronic: float) -> None:
    for level, job_id in sorted(RECORDED_JOB_IDS.items()):
        electronic, std, usage = _fetch_one_result(service, job_id)
        error_mha = abs(electronic - exact_electronic) * 1000.0
        print(
            f"resilience_level={level}: electronic={electronic:.6f} "
            f"std={std:.4f} error={error_mha:.1f} mHa usage={usage}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Actually submit NEW real hardware jobs (consumes real QPU quota). "
        "Without this flag, only the free live-calibration cross-check runs, plus "
        "fetching the already-recorded job IDs' results.",
    )
    args = parser.parse_args()

    service = _retry(lambda: QiskitRuntimeService())
    backend = _retry(lambda: service.backend(BACKEND_NAME))
    isa_circuit, isa_hamiltonian, e_core, exact = _build_h2_system(backend)
    exact_electronic = exact - e_core

    print(f"Exact energy: {exact:.6f} Ha\n")
    free_live_calibration_cross_check(service)
    print()

    if args.resubmit:
        print("--resubmit passed: submitting NEW real hardware jobs (uses quota!)")
        for level in (0, 1, 2):
            estimator = EstimatorV2(mode=backend, options={"resilience_level": level})
            job = estimator.run([(isa_circuit, isa_hamiltonian)], precision=0.02)
            print(f"  submitted resilience_level={level}: job {job.job_id()}")
        print("Poll these job IDs yourself with QiskitRuntimeService().job(<id>).result()")
    else:
        print("Recorded 2026-09-17 hardware results (re-fetched, not re-submitted):")
        fetch_recorded_results(service, exact_electronic)


if __name__ == "__main__":
    main()
