"""Phase 8 report figure: Phase 5's Aer-simulated (FakeTorino) mitigation predictions
side-by-side with Phase 6's REAL ibm_marrakesh hardware results, for the same
H2/EfficientSU2 circuit at each resilience level -- visualizes Phase 6's headline finding
(real hardware shows uniformly larger bias than simulation predicted) directly.

Numbers are the recorded, already-computed results from scripts/phase5_mitigation_sweep.py
(simulated) and scripts/phase6_real_hardware_comparison.py (real, job IDs there) -- this
script only re-plots them, it does not resubmit anything or run new simulations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vqe_nisq_project.viz import CATEGORICAL_COLORS, apply_style, savefig_both

REPORT_FIGURES_DIR = Path(__file__).parent.parent / "report" / "figures"

LABELS = ["Raw", "M3-like\n(resilience 1)", "ZNE-like\n(resilience 2)"]
SIMULATED_MHA = [
    23.307,
    11.095,
    8.121,
]  # FakeTorino, report/figures/phase5_mitigation_sweep_table.md
REAL_MHA = [41.5, 32.7, 24.7]  # ibm_marrakesh, phase6_real_hardware_comparison.py
CHEMICAL_ACCURACY_MHA = 1.6


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(LABELS))
    width = 0.35
    ax.bar(
        x - width / 2,
        SIMULATED_MHA,
        width,
        label="Aer-simulated (FakeTorino)",
        color=CATEGORICAL_COLORS[0],
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        REAL_MHA,
        width,
        label="Real hardware (ibm_marrakesh)",
        color=CATEGORICAL_COLORS[7],
        zorder=3,
    )
    ax.axhline(CHEMICAL_ACCURACY_MHA, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(-0.45, CHEMICAL_ACCURACY_MHA + 1, "chemical accuracy", color="gray", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title("Simulation vs. real hardware: H2/EfficientSU2 on a Heron-class device")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pdf_path, png_path = savefig_both(fig, REPORT_FIGURES_DIR, "phase6_sim_vs_real")
    print(f"Wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
