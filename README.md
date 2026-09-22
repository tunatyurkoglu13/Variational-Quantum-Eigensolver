# VQE on NISQ Devices: A Comparative Study of Ansatze, Optimizers, and Error Mitigation

An end-to-end, empirically-verified study of the Variational Quantum Eigensolver (VQE)
applied to H<sub>2</sub>, LiH, and BeH<sub>2</sub> — from Hamiltonian construction through
a real IBM Quantum hardware run. Every number in this repo (and in the accompanying
[technical report](report/main.pdf)) was computed or measured directly, not assumed.

## Research question

Under what conditions is chemical accuracy (|ΔE| < 1.6 mHa) achievable for small molecules
on NISQ devices via VQE, and what actually limits it — ansatz expressibility, optimizer
reliability, measurement (shot) noise, or hardware noise?

## Headline finding

Standard Qiskit Aer noisy-simulation **underestimates real hardware error by roughly a
factor of two** — even when fed the target device's own live calibration data. A fresh
noise model built from `ibm_marrakesh`'s current calibration predicted 12.7 mHa of error
for a fixed H<sub>2</sub>/EfficientSU2 circuit; the same circuit run for real on
`ibm_marrakesh` showed 41.5 mHa. Since the gap persists even with live, correct
calibration numbers, it is **methodological**: `NoiseModel.from_backend`'s
depolarizing-plus-thermal-relaxation construction omits noise structure (coherent errors,
crosstalk) that real devices have.

An error-budget decomposition on H<sub>2</sub> quantifies the rest of the picture: ansatz
choice is not a meaningful bottleneck once the optimizer actually converges (< 0.00001 mHa
residual); optimizer reliability and shot noise contribute comparable, controllable
amounts (a few to tens of mHa); hardware noise dominates by a wide margin over both.

| Error source | Magnitude (mHa) |
|---|---:|
| Ansatz (EfficientSU2, converged) | 0.0000018 |
| Optimizer (L-BFGS-B, 20-seed mean) | 3.05 |
| Shot noise (8192 shots) | 11.05 |
| Hardware noise (Aer-simulated) | 23.31 |
| **Hardware noise (real `ibm_marrakesh`)** | **41.50** |

## What's in this repo

- **`report/main.pdf`** — the full technical report (arXiv-style, ~12 pages): theory
  derived from scratch (Born–Oppenheimer through the parameter-shift rule), methods,
  every result below with real numbers and figures, an honest limitations section.
- **`src/vqe_nisq_project/`** — the actual pipeline:
  - `chemistry/` — Hamiltonian construction (PySCF), fermion-to-qubit mappings
    (Jordan-Wigner, parity, Bravyi-Kitaev), Z2 symmetry tapering, measurement grouping.
  - `ansatz/` — UCCSD, EfficientSU2 (hardware-efficient), ADAPT-VQE, expressibility /
    entangling-capability metrics, shared Qiskit + PennyLane interface.
  - `optimization/` — COBYLA / SPSA / L-BFGS-B (own parameter-shift gradient) / Adam,
    multi-seed statistics, Cliff's-delta effect sizes with Holm–Bonferroni correction,
    barren-plateau scaling.
  - `noise/` — real-backend-calibrated noise models, and four error-mitigation
    techniques implemented from their primary sources: M3 (readout), ZNE (unitary
    folding), Pauli twirling, dynamical decoupling.
- **`scripts/`** — the exact scripts that produced every figure and number in the report
  (`phase5_mitigation_sweep.py`, `phase6_real_hardware_comparison.py`,
  `phase7_error_budget.py`, …), so every result is re-runnable, not just quoted.
- **`tests/`** — 147+ tests, ≥97% coverage; most are regression tests for real bugs found
  along the way (library convention mismatches, silent no-ops, non-deterministic
  transpilation), not just correctness checks.
- **[Docs site](https://tunatyurkoglu13.github.io/Variational-Quantum-Eigensolver/)** —
  auto-generated API reference plus a narrative walkthrough.

## Status

All planned phases (0–8) are complete: environment/tooling, Hamiltonian construction,
fermion-to-qubit mappings, ansatz library, optimizer statistics, noise simulation and
mitigation, real hardware validation, statistics/error-budget analysis, and this writeup.

## Setup

See [`SETUP.md`](SETUP.md) for a from-scratch environment setup (uv-managed, ~10 minutes).
