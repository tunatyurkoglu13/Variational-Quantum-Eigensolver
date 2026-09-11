"""Ground truth: reference energy ordering must respect the variational principle, and for
a 2-electron system CCSD is exact (no triple excitations exist to correct for)."""

from vqe_nisq_project.chemistry.molecule import h2
from vqe_nisq_project.chemistry.reference import compute_reference_energies


def test_variational_ordering_h2_sto3g() -> None:
    ref = compute_reference_energies(h2())
    # FCI is exact for this basis -- the lowest possible energy. HF is a
    # variational upper bound. CCSD/CCSD(T) recover correlation energy,
    # moving from HF towards (but for a 2-electron system, exactly to) FCI.
    # A 1e-9 slack matches the solvers' own convergence tolerances (FCI
    # conv_tol=1e-10, CCSD tightened to 1e-10/1e-8 in reference.py) -- below
    # that scale, "ordering" is just iterative-solver noise, not physics.
    tol = 1e-9
    assert ref.fci <= ref.ccsd_t + tol
    assert ref.ccsd_t <= ref.ccsd + tol
    assert ref.ccsd <= ref.hf + tol


def test_ccsd_is_exact_for_h2_two_electrons() -> None:
    # CCSD's cluster operator includes all excitations possible with only 2
    # electrons (singles + doubles); there is no room for a triples
    # correction, so CCSD, CCSD(T), and FCI must coincide.
    ref = compute_reference_energies(h2())
    assert abs(ref.ccsd - ref.fci) < 1e-8
    assert abs(ref.ccsd_t - ref.fci) < 1e-8
