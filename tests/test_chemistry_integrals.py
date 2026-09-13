"""Ground truth: hand-extracted (H2) / CASCI-extracted (LiH, BeH2) MO integrals must
reproduce PySCF's own full-molecule RHF energy exactly, with or without a frozen core."""

import pytest

from conftest import cached_mo_integrals
from vqe_nisq_project.chemistry.integrals import rhf_energy_from_integrals
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih


def test_rhf_energy_reconstructed_from_integrals_matches_pyscf_h2_sto3g() -> None:
    integrals = cached_mo_integrals(h2())
    reconstructed = rhf_energy_from_integrals(integrals)
    assert abs(reconstructed - integrals.e_hf) < 1e-10


def test_rhf_energy_reconstructed_from_integrals_matches_pyscf_h2_631g() -> None:
    # A richer basis (4 spatial orbitals instead of 2) so the integral tensor
    # has enough structure that a wrong AO->MO transform couldn't pass by
    # accident/degeneracy.
    spec = MoleculeSpec(atom="H 0 0 0; H 0 0 0.735", basis="6-31g")
    integrals = cached_mo_integrals(spec)
    reconstructed = rhf_energy_from_integrals(integrals)
    assert abs(reconstructed - integrals.e_hf) < 1e-10


def test_h2_sto3g_known_electron_and_orbital_count() -> None:
    integrals = cached_mo_integrals(h2())
    assert integrals.n_electrons == 2
    assert integrals.n_mo == 2


@pytest.mark.parametrize(
    ("spec", "n_frozen_core", "expected_active_electrons", "expected_active_mo"),
    [
        (lih(), 1, 2, 5),
        (beh2(), 1, 4, 6),
    ],
    ids=["lih", "beh2"],
)
def test_rhf_energy_reconstructed_with_frozen_core_matches_pyscf(
    spec: MoleculeSpec, n_frozen_core: int, expected_active_electrons: int, expected_active_mo: int
) -> None:
    integrals = cached_mo_integrals(spec, n_frozen_core=n_frozen_core)
    assert integrals.n_electrons == expected_active_electrons
    assert integrals.n_mo == expected_active_mo

    reconstructed = rhf_energy_from_integrals(integrals)
    assert abs(reconstructed - integrals.e_hf) < 1e-8


def test_frozen_core_e_core_includes_more_than_nuclear_repulsion() -> None:
    # e_core = e_nuc + the frozen core's own mean-field energy contribution --
    # for a real frozen core (unlike H2, which has none) these must differ.
    integrals = cached_mo_integrals(lih(), n_frozen_core=1)
    assert integrals.e_core != integrals.e_nuc
