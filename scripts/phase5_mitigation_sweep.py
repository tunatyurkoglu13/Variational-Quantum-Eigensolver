"""Phase 5 final deliverable: a marginal-cost/benefit comparison of all four error
mitigation techniques built this phase (M3 readout mitigation, ZNE, Pauli twirling,
dynamical decoupling) against an unmitigated baseline, on the SAME H2/EfficientSU2
(reps=2, full entanglement)/FakeTorino system used throughout each technique's own
development and test suite -- ties together noise/mitigation_{m3,zne,twirling,dd}.py's
individual findings into the one comparison this phase's spec asked for.

Cost accounting caveat (read before trusting the multipliers below): this project's
`simulate.estimate_energy` computes a WHOLE multi-term Hamiltonian's expectation value in
ONE circuit execution via `save_expectation_value` -- a simulator-only shortcut (see
`simulate.py`'s Real finding #2). M3, in contrast, must measure REAL per-group bitstring
counts (`chemistry.measurement.group_qubit_wise_commuting` produced 5 groups for H2/JW
here), so its baseline already costs 5 circuit executions where ZNE/twirling/DD's
baseline costs only 1 in THIS simulated pipeline. On real hardware (Phase 6), every
technique needs the same per-group measurement allocation M3 already pays for -- so the
ZNE/twirling/DD multipliers below UNDERSTATE their true real-hardware cost by roughly the
same ~5x factor M3's baseline already reflects. Reported as multipliers of "circuit
executions per energy evaluation at a fixed per-circuit shot count" for exactly this
reason, rather than absolute shot counts that would look artificially favorable.

Note on reproducibility: every row except M3 is bit-for-bit reproducible run to run
(fixed `seed`/`seed_transpiler` throughout). M3's row fluctuates by ~1 mHa between runs,
because `mitigation_m3.calibrate`'s calibration shots are not seedable (mthree's
`cals_from_system` exposes no seed parameter) -- the RANKING (ZNE best, M3 second,
twirling/DD near-unchanged from raw) is stable across reruns even though M3's exact
number is not; verified by running this script twice.

Usage:
    uv run python scripts/phase5_mitigation_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

from vqe_nisq_project.ansatz.base import FloatArray
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.noise.mitigation_dd import dd_energy
from vqe_nisq_project.noise.mitigation_m3 import (
    calibrate,
    estimate_energy_with_readout_mitigation,
)
from vqe_nisq_project.noise.mitigation_twirling import twirled_energy
from vqe_nisq_project.noise.mitigation_zne import zne_energy
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import estimate_energy, prepare_for_backend
from vqe_nisq_project.viz import CATEGORICAL_COLORS, MARKERS, apply_style, savefig_both

REPORT_FIGURES_DIR = Path(__file__).parent.parent / "report" / "figures"
SHOTS_PER_CIRCUIT = 20_000
N_TWIRLS = 15
SCALE_FACTORS = (1.0, 2.0, 3.0)


def main() -> None:
    integrals = compute_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    e_core = integrals.e_core
    assert ansatz.qiskit_circuit is not None
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)

    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )

    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    exact = exact_electronic + e_core
    no_params: FloatArray = np.array([])

    raw = estimate_energy(
        isa_circuit,
        no_params,
        isa_hamiltonian,
        e_core,
        shots=SHOTS_PER_CIRCUIT,
        noise_model=noise_model,
        seed=0,
    )

    mitigator = calibrate(backend, mapping.n_qubits, noise_model, shots=100_000)
    m3_result = estimate_energy_with_readout_mitigation(
        bound_circuit,
        mapping.qubit_op,
        e_core,
        backend,
        mitigator,
        noise_model=noise_model,
        shots=SHOTS_PER_CIRCUIT,
        seed=0,
        seed_transpiler=123,
    )

    zne_result = zne_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        noise_model,
        shots_per_point=SHOTS_PER_CIRCUIT,
        scale_factors=SCALE_FACTORS,
        seed=0,
    )

    twirl_result = twirled_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        noise_model,
        n_twirls=N_TWIRLS,
        shots_per_twirl=SHOTS_PER_CIRCUIT,
        seed=0,
    )

    dd_result = dd_energy(
        isa_circuit,
        isa_hamiltonian,
        e_core,
        backend,
        noise_model,
        shots=SHOTS_PER_CIRCUIT,
        seed=0,
    )

    rows = [
        ("Raw (no mitigation)", raw.energy, 1, "--"),
        ("M3 (readout)", m3_result.mitigated_energy, m3_result.n_groups, "2n=8 circuits"),
        ("ZNE (mitiq)", zne_result.zne_energy, len(SCALE_FACTORS), "--"),
        ("Pauli twirling", twirl_result.twirled_energy, N_TWIRLS, "--"),
        ("Dynamical decoupling", dd_result.dd_energy, 1, "--"),
    ]

    chemical_accuracy_mha = 1.6
    print(f"Exact (noiseless statevector) energy: {exact:.6f} Ha\n")
    header = f"{'Technique':<24}{'Error (mHa)':>14}{'Cost x':>10}{'One-time setup':>18}"
    print(header)
    print("-" * len(header))
    table_lines = [header, "-" * len(header)]
    for name, energy, cost, setup in rows:
        error_mha = abs(energy - exact) * 1000.0
        flag = " (chemical accuracy)" if error_mha < chemical_accuracy_mha else ""
        line = f"{name:<24}{error_mha:>14.3f}{cost:>10}{setup:>18}{flag}"
        print(line)
        table_lines.append(line)

    md_path = REPORT_FIGURES_DIR / "phase5_mitigation_sweep_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write("| Technique | Error (mHa) | Cost multiplier | One-time setup |\n")
        f.write("|---|---:|---:|---|\n")
        for name, energy, cost, setup in rows:
            error_mha = abs(energy - exact) * 1000.0
            f.write(f"| {name} | {error_mha:.3f} | {cost}x | {setup} |\n")
    print(f"\nWrote {md_path}")

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, energy, cost, _setup) in enumerate(rows):
        error_mha = abs(energy - exact) * 1000.0
        ax.scatter(
            cost,
            error_mha,
            color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            s=80,
            label=name,
            zorder=3,
        )
    ax.axhline(chemical_accuracy_mha, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(1, chemical_accuracy_mha * 1.05, "chemical accuracy", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (circuit executions per energy evaluation, relative)")
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title("H2/EfficientSU2/FakeTorino: mitigation cost vs. benefit")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    pdf_path, png_path = savefig_both(fig, REPORT_FIGURES_DIR, "phase5_mitigation_sweep")
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
