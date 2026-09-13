"""Ground truth: reference energy ordering must respect the variational principle, and for
a 2-electron system CCSD is exact (no triple excitations exist to correct for)."""

import pytest

from conftest import cached_reference_energies as compute_reference_energies
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih


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


def test_ccsd_is_exact_for_lih_two_active_electrons() -> None:
    # LiH's frozen-core active space also has only 2 active electrons, so the
    # same "no triples possible" argument applies to its CCSD as to H2's.
    ref = compute_reference_energies(lih(), n_frozen_core=1)
    assert abs(ref.ccsd - ref.fci) < 1e-8
    assert abs(ref.ccsd_t - ref.fci) < 1e-8


def test_ccsd_t_is_close_but_not_exact_for_beh2_four_active_electrons() -> None:
    # BeH2's active space has 4 active electrons -- triple excitations exist,
    # so CCSD(T) should be a close but genuinely imperfect approximation to
    # FCI (unlike H2/LiH's exact 2-electron case). This is the expected,
    # honest behavior -- not a bug if the gap is small but nonzero.
    ref = compute_reference_energies(beh2(), n_frozen_core=1)
    assert ref.ccsd_t != pytest.approx(ref.fci, abs=1e-10)
    assert abs(ref.ccsd_t - ref.fci) < 1e-3  # still within ~1 mHa, "CCSD(T) is very good"


@pytest.mark.parametrize("spec", [lih(), beh2()], ids=["lih", "beh2"])
def test_frozen_core_error_is_measured_and_small(spec: MoleculeSpec) -> None:
    # The frozen-core approximation's cost must be measured (per this
    # project's rule against assuming an error is negligible), and for these
    # two systems at this basis/geometry it should be well under chemical
    # accuracy (1.6 mHa) -- otherwise freezing the core would be the wrong
    # call for this study.
    ref = compute_reference_energies(spec, n_frozen_core=1)
    assert ref.full_fci is not None
    assert ref.frozen_core_error is not None
    assert ref.frozen_core_error == pytest.approx(ref.fci - ref.full_fci)
    assert abs(ref.frozen_core_error) < 1.6e-3
