"""Phase 7 final deliverable: an error-budget decomposition for H2/STO-3G, answering
this project's own top-level research question directly -- of the gap between an
achieved VQE energy and chemical accuracy (1.6 mHa), how much comes from ansatz choice,
optimizer convergence, shot noise, and hardware noise, and which one actually dominates?

Each term is measured INDEPENDENTLY under conditions that isolate it (the others held at
their best/absent case) -- this is a characterization of each source's typical magnitude,
not a literal additive decomposition of one single run's error (the four sources interact:
e.g. hardware noise also degrades optimizer convergence). This is the standard way error
budgets are presented in real experimental physics, and is stated explicitly here rather
than implying a false precision.

All four terms reuse modules and real, previously-verified numbers from Phases 3-6 (only
the ansatz-limitation and 20-seed optimizer terms are recomputed fresh here; FakeTorino's
simulated hardware-noise term is recomputed fresh; the REAL ibm_marrakesh number is
recorded from Phase 6's actual hardware job, not resubmitted -- see
scripts/phase6_real_hardware_comparison.py for that job's ID and full context):

1. ANSATZ LIMITATION: does EfficientSU2 (a generic hardware-efficient ansatz with no
   chemistry-specific structure, unlike UCCSD) fundamentally lack the expressibility to
   represent H2's true ground state? Measured as |E(well-converged L-BFGS-B) - FCI|.
2. OPTIMIZER NON-CONVERGENCE: given the SAME ansatz, how much does the optimizer's own
   reliability (local minima, stochasticity) cost on average across 20 independent random
   seeds? Measured per-optimizer as |mean(20-seed final energies) - FCI| -- this is where
   Phase 7 step 1's effect-size/significance analysis lives (see statistics.py).
3. SHOT NOISE: the statistical uncertainty from measuring an expectation value with a
   finite number of measurements -- a KNOWN, controllable quantity (shrinks as
   1/sqrt(shots)), shown at three different shot budgets to make that scaling concrete.
4. HARDWARE NOISE: the bias from running on noisy hardware (readout + gate errors),
   shown BOTH as Phase 5's Aer-simulated (FakeTorino) prediction AND Phase 6's REAL
   ibm_marrakesh result -- deliberately kept as two separate bars, since Phase 6's
   single most important finding was that these two disagree by roughly 2x.

Usage:
    uv run python scripts/phase7_error_budget.py
(Fully self-contained and free -- no real hardware jobs are submitted; the one real-
hardware number is a recorded constant from Phase 6's actual, already-completed run.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeTorino

from vqe_nisq_project.ansatz.base import ComplexArray
from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.chemistry.integrals import compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies
from vqe_nisq_project.noise.models import build_noise_model
from vqe_nisq_project.noise.simulate import estimate_energy, prepare_for_backend, shots_to_precision
from vqe_nisq_project.optimization.initial_points import small_random_initial_point
from vqe_nisq_project.optimization.multi_seed import run_multi_seed
from vqe_nisq_project.optimization.vqe_runner import OptimizerName, run_vqe
from vqe_nisq_project.viz import CATEGORICAL_COLORS, apply_style, savefig_both

REPORT_FIGURES_DIR = Path(__file__).parent.parent / "report" / "figures"
CHEMICAL_ACCURACY_MHA = 1.6

# Recorded from Phase 6's actual completed hardware job (resilience_level=0, raw, on
# ibm_marrakesh) -- see scripts/phase6_real_hardware_comparison.py, job
# dam48p78gn2s739kqvn0. NOT resubmitted here to avoid spending real QPU quota.
REAL_HARDWARE_RAW_ERROR_MHA = 41.5


def ansatz_limitation_mha(hamiltonian: ComplexArray, e_core: float, fci: float) -> float:
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    init = small_random_initial_point(ansatz.n_parameters, scale=0.1, seed=0)
    result = run_vqe(
        ansatz.state_fn,
        hamiltonian,
        init,
        optimizer="lbfgsb",
        e_core=e_core,
        max_iterations=200,
    )
    return abs(result.final_energy - fci) * 1000.0


def optimizer_gaps_mha(hamiltonian: ComplexArray, e_core: float, fci: float) -> dict[str, float]:
    ansatz = build_efficient_su2_qiskit(4, reps=2)
    gaps: dict[str, float] = {}
    name_iterations: list[tuple[OptimizerName, int]] = [("lbfgsb", 100), ("cobyla", 150)]
    for name, max_iterations in name_iterations:
        result = run_multi_seed(
            ansatz.state_fn,
            hamiltonian,
            ansatz.n_parameters,
            name,
            best_known_energy=fci,
            e_core=e_core,
            max_iterations=max_iterations,
            n_seeds=20,
            init_scale=0.5,
        )
        gaps[name] = abs(result.mean_energy - fci) * 1000.0
    return gaps


def shot_noise_mha(shots_list: list[int]) -> dict[int, float]:
    return {shots: shots_to_precision(shots) * 1000.0 for shots in shots_list}


def hardware_noise_simulated_mha() -> float:
    integrals = compute_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    mapping = map_hamiltonian(fop, "jordan_wigner")
    ansatz = build_efficient_su2_qiskit(mapping.n_qubits, reps=2, entanglement="full")
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, ansatz.n_parameters)
    assert ansatz.qiskit_circuit is not None
    bound_circuit = ansatz.qiskit_circuit.assign_parameters(params)
    e_core = integrals.e_core
    backend = FakeTorino()
    noise_model = build_noise_model(backend)
    isa_circuit, isa_hamiltonian = prepare_for_backend(
        bound_circuit, mapping.qubit_op, backend, seed_transpiler=123
    )
    with np.errstate(all="ignore"):
        state = ansatz.state_fn(params)
        exact_electronic = float((state.conj() @ (mapping.qubit_op.to_matrix() @ state)).real)
    exact = exact_electronic + e_core
    no_params = np.array([])
    raw = estimate_energy(
        isa_circuit,
        no_params,
        isa_hamiltonian,
        e_core,
        shots=20_000,
        noise_model=noise_model,
        seed=0,
    )
    return abs(raw.energy - exact) * 1000.0


def main() -> None:
    integrals = compute_mo_integrals(h2(), n_frozen_core=0)
    fop = build_electronic_fermionic_op(integrals)
    hamiltonian = map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()
    fci = compute_reference_energies(h2()).fci
    e_core = integrals.e_nuc

    print("Computing ansatz-limitation term...")
    ansatz_mha = ansatz_limitation_mha(hamiltonian, e_core, fci)

    print("Computing optimizer non-convergence terms (20 seeds each)...")
    optimizer_mha = optimizer_gaps_mha(hamiltonian, e_core, fci)

    shot_mha = shot_noise_mha([1024, 8192, 65536])

    print("Computing simulated hardware-noise term (FakeTorino)...")
    hw_sim_mha = hardware_noise_simulated_mha()

    rows: list[tuple[str, float, str]] = [
        ("Ansatz (EfficientSU2, converged)", ansatz_mha, "ansatz"),
        ("Optimizer: L-BFGS-B (20-seed mean)", optimizer_mha["lbfgsb"], "optimizer"),
        ("Optimizer: COBYLA (20-seed mean)", optimizer_mha["cobyla"], "optimizer"),
        ("Shot noise: 65536 shots", shot_mha[65536], "shots"),
        ("Shot noise: 8192 shots", shot_mha[8192], "shots"),
        ("Shot noise: 1024 shots", shot_mha[1024], "shots"),
        ("Hardware noise: FakeTorino (simulated)", hw_sim_mha, "hardware"),
        ("Hardware noise: ibm_marrakesh (REAL)", REAL_HARDWARE_RAW_ERROR_MHA, "hardware"),
    ]

    header = f"{'Term':<40}{'Error (mHa)':>14}"
    print(f"\n{header}\n{'-' * len(header)}")
    md_lines = ["| Term | Error (mHa) |", "|---|---:|"]
    for name, value, _category in rows:
        flag = " (chemical accuracy)" if value < CHEMICAL_ACCURACY_MHA else ""
        print(f"{name:<40}{value:>14.4f}{flag}")
        md_lines.append(f"| {name} | {value:.4f} |")

    md_path = REPORT_FIGURES_DIR / "phase7_error_budget_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nWrote {md_path}")

    category_colors = {
        "ansatz": CATEGORICAL_COLORS[0],
        "optimizer": CATEGORICAL_COLORS[1],
        "shots": CATEGORICAL_COLORS[2],
        "hardware": CATEGORICAL_COLORS[7],
    }

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r[0] for r in rows]
    values = [max(r[1], 1e-6) for r in rows]  # avoid log(0) for the ~0 ansatz term
    colors = [category_colors[r[2]] for r in rows]
    y_pos = np.arange(len(rows))[::-1]
    ax.barh(y_pos, values, color=colors, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.axvline(CHEMICAL_ACCURACY_MHA, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(CHEMICAL_ACCURACY_MHA * 1.1, -0.7, "chemical accuracy", color="gray", fontsize=8)
    ax.set_xlabel("Error (mHa, log scale)")
    ax.set_title("H2/STO-3G error budget: independently-measured term magnitudes")
    fig.tight_layout()
    pdf_path, png_path = savefig_both(fig, REPORT_FIGURES_DIR, "phase7_error_budget")
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
