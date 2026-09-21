"""Phase 8 report figure: reproduces Cerezo et al. 2021's (arXiv:2001.00550) local-vs-
global cost-function-dependent barren-plateau result on this project's own
EfficientSU2(reps=3, linear) family, N=2..16 qubits -- a LOCAL observable (Z on qubit 0)
shows gradient-variance staying roughly flat, while a GLOBAL observable (Z^{\\otimes n}
parity) decays exponentially, exactly as predicted. Regenerates the data fresh (fast,
~6s total) rather than re-plotting stale numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from vqe_nisq_project.ansatz.hardware_efficient import build_efficient_su2_qiskit
from vqe_nisq_project.optimization.barren_plateau import (
    global_parity_observable,
    local_z0_observable,
    measure_gradient_variance,
)
from vqe_nisq_project.viz import CATEGORICAL_COLORS, apply_style, savefig_both

REPORT_FIGURES_DIR = Path(__file__).parent.parent / "report" / "figures"
QUBIT_COUNTS = [2, 4, 6, 8, 10, 12, 14, 16]


def main() -> None:
    local_variances = []
    global_variances = []
    for n in QUBIT_COUNTS:
        ansatz = build_efficient_su2_qiskit(n, reps=3, entanglement="linear")
        local_pt = measure_gradient_variance(ansatz, local_z0_observable(n), n_samples=40, seed=0)
        global_pt = measure_gradient_variance(
            ansatz, global_parity_observable(n), n_samples=40, seed=0
        )
        local_var, global_var = local_pt.gradient_variance, global_pt.gradient_variance
        local_variances.append(local_var)
        global_variances.append(global_var)
        print(f"N={n}: local={local_var:.3e} global={global_var:.3e}")

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(
        QUBIT_COUNTS,
        local_variances,
        marker="o",
        color=CATEGORICAL_COLORS[0],
        label="Local observable (Z on qubit 0)",
    )
    ax.semilogy(
        QUBIT_COUNTS,
        global_variances,
        marker="s",
        color=CATEGORICAL_COLORS[1],
        label="Global observable (Z$^{\\otimes n}$ parity)",
    )
    ax.set_xlabel("Number of qubits N")
    ax.set_ylabel("Var[$\\partial E/\\partial\\theta_0$]")
    ax.set_title("Barren plateaus: local vs. global cost function (EfficientSU2, reps=3)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf_path, png_path = savefig_both(fig, REPORT_FIGURES_DIR, "phase4_barren_plateau")
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
