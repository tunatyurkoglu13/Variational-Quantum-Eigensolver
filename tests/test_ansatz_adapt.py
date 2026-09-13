"""Ground truth: ADAPT-VQE must converge to FCI on H2 (fast: 3-operator pool), and its
gradient-screening step (the part that had a real ~50GB dense-matrix memory blowup before
switching to sparse matrices -- see adapt.py's module docstring) must stay fast and
correct as the pool grows to LiH's 24 operators. Regression test also confirms no spurious
RuntimeWarnings from the macOS Accelerate BLAS quirk documented in adapt.py.

BeH2 (92-operator pool) is deliberately excluded from the automated suite: the gradient
screen itself is fast and was verified manually, but full COBYLA re-optimization of the
growing circuit is not (each function evaluation re-synthesizes every PauliEvolutionGate
from scratch) -- a genuine, documented scaling limitation, not something to force through
in CI just to keep this test file "complete"."""

import time
import warnings

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.adapt import run_adapt_vqe
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2, lih


def _run_adapt(spec, n_frozen_core, num_spatial_orbitals, num_particles, **kwargs):
    integrals = cached_mo_integrals(spec, n_frozen_core)
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")
    hamiltonian = jw.qubit_op.to_matrix()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_adapt_vqe(
            hamiltonian,
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            e_core=integrals.e_core,
            **kwargs,
        )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    return result, len(runtime_warnings)


def test_adapt_vqe_converges_to_fci_on_h2() -> None:
    result, n_runtime_warnings = _run_adapt(h2(), 0, 2, (1, 1), gradient_tolerance=1e-4)
    fci = cached_reference_energies(h2()).fci
    assert abs(result.final_energy - fci) < 1e-6
    assert n_runtime_warnings == 0


def test_adapt_vqe_skips_both_singles_due_to_brillouins_theorem() -> None:
    # For a closed-shell HF reference, single excitations have zero first-order
    # coupling to the reference (Brillouin's theorem) -- ADAPT's own gradient
    # criterion should independently discover this and go straight for the
    # double excitation, converging in exactly one step for H2.
    result, _ = _run_adapt(h2(), 0, 2, (1, 1), gradient_tolerance=1e-4)
    assert result.final_n_parameters == 1
    assert result.steps[0].excitation_label == ((0, 2), (1, 3))


def test_adapt_vqe_gradient_screen_is_fast_and_memory_safe_on_lih() -> None:
    # Regression test for the real ~50GB dense-matrix blowup found when this
    # was extended to LiH/BeH2's larger (24- and 92-operator) pools -- one
    # gradient-screening pass (no re-optimization: max_cobyla_iterations=1)
    # over LiH's full pool must complete quickly.
    t0 = time.time()
    result, n_runtime_warnings = _run_adapt(
        lih(), 1, 5, (1, 1), max_iterations=1, max_cobyla_iterations=1
    )
    elapsed = time.time() - t0
    assert elapsed < 15.0
    assert len(result.steps) == 1
    assert n_runtime_warnings == 0
