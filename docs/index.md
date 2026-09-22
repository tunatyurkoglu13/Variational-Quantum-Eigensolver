# VQE on NISQ Devices

An end-to-end, empirically-verified study of the Variational Quantum Eigensolver (VQE)
applied to H<sub>2</sub>, LiH, and BeH<sub>2</sub> — from Hamiltonian construction through
a real IBM Quantum hardware run. Every number quoted here was computed or measured
directly by this project's own code, not assumed or taken from a table.

**Research question:** under what conditions is chemical accuracy
(|ΔE| < 1.6 mHa) achievable for small molecules on NISQ devices via VQE, and what
actually limits it — ansatz expressibility, optimizer reliability, measurement (shot)
noise, or hardware noise?

**Answer, in one sentence:** for H<sub>2</sub>, hardware noise dominates by a wide
margin over the other three, and the standard Aer noisy-simulation methodology used to
predict it underestimates the real effect by roughly a factor of two, even when fed the
target device's own live calibration data. See [Findings](findings.md) for the full,
number-by-number walkthrough, or the
[technical report](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver/blob/main/report/main.pdf)
(PDF) for the complete arXiv-style writeup with derivations.

Status: all planned phases (0-8) complete. Source, tests, and the exact scripts that
produced every number/figure below are in the
[GitHub repository](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver).
The [API reference](api/chemistry.md) is generated directly from source docstrings.
