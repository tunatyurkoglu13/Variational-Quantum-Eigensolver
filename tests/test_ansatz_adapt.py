"""Ground truth: ADAPT-VQE must converge to FCI, and (regression, macOS Accelerate BLAS
quirk) must not raise spurious RuntimeWarnings from its gradient matrix products."""

import warnings

from conftest import cached_mo_integrals, cached_reference_energies
from vqe_nisq_project.ansatz.adapt import run_adapt_vqe
from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
from vqe_nisq_project.chemistry.molecule import h2


def _run_h2_adapt() -> tuple[object, float]:
    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    jw = map_hamiltonian(fop, "jordan_wigner")
    hamiltonian = jw.qubit_op.to_matrix()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_adapt_vqe(
            hamiltonian,
            num_spatial_orbitals=2,
            num_particles=(1, 1),
            e_core=integrals.e_nuc,
            gradient_tolerance=1e-4,
        )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    return result, len(runtime_warnings)


def test_adapt_vqe_converges_to_fci_on_h2() -> None:
    result, _ = _run_h2_adapt()
    fci = cached_reference_energies(h2()).fci
    assert abs(result.final_energy - fci) < 1e-6


def test_adapt_vqe_skips_both_singles_due_to_brillouins_theorem() -> None:
    # For a closed-shell HF reference, single excitations have zero first-order
    # coupling to the reference (Brillouin's theorem) -- ADAPT's own gradient
    # criterion should independently discover this and go straight for the
    # double excitation, converging in exactly one step for H2.
    result, _ = _run_h2_adapt()
    assert result.final_n_parameters == 1
    assert result.steps[0].excitation_label == ((0, 2), (1, 3))


def test_adapt_vqe_gradient_computation_raises_no_runtime_warnings() -> None:
    _result, n_runtime_warnings = _run_h2_adapt()
    assert n_runtime_warnings == 0
