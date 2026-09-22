# noise

Phase 5-6: real-backend-calibrated noise models (`NoiseModel.from_backend`, real
IBM Quantum calibration snapshots), shot-based noisy energy estimation via
`EstimatorV2`, and four error-mitigation techniques implemented from their
primary sources rather than as opaque library calls: M3 (matrix-free
measurement mitigation), zero-noise extrapolation (unitary gate folding),
Pauli twirling, and dynamical decoupling. `simulate.py`'s `prepare_for_backend`
is the shared entry point used throughout Phase 5-6 for turning an ansatz
circuit into a backend-native ("ISA") circuit paired with a correctly-aligned
Hamiltonian.

::: vqe_nisq_project.noise.models

::: vqe_nisq_project.noise.simulate

::: vqe_nisq_project.noise.mitigation_m3

::: vqe_nisq_project.noise.mitigation_zne

::: vqe_nisq_project.noise.mitigation_twirling

::: vqe_nisq_project.noise.mitigation_dd
