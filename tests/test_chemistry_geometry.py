"""Sanity checks on our own RHF bond-length scans (used to set LiH/BeH2's geometry)."""

from vqe_nisq_project.chemistry.geometry import scan_rhf_energy
from vqe_nisq_project.chemistry.molecule import _BEH2_R_EQ_ANGSTROM, _LIH_R_EQ_ANGSTROM


def test_lih_scan_minimum_is_deterministic_and_matches_stored_preset() -> None:
    scan = scan_rhf_energy(lambda r: f"Li 0 0 0; H 0 0 {r}", "sto-3g", [1.3, 1.4, 1.5, 1.6, 1.7])
    # Regenerating the scan must reproduce the value baked into molecule.py's preset.
    assert abs(scan.r_eq - _LIH_R_EQ_ANGSTROM) < 1e-3


def test_beh2_scan_minimum_is_deterministic_and_matches_stored_preset() -> None:
    scan = scan_rhf_energy(
        lambda r: f"Be 0 0 0; H 0 0 {r}; H 0 0 -{r}", "sto-3g", [1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    )
    assert abs(scan.r_eq - _BEH2_R_EQ_ANGSTROM) < 1e-3


def test_scan_energies_are_lower_near_the_minimum_than_at_the_boundaries() -> None:
    scan = scan_rhf_energy(lambda r: f"Li 0 0 0; H 0 0 {r}", "sto-3g", [1.2, 1.5, 2.6])
    assert scan.energies[1] < scan.energies[0]
    assert scan.energies[1] < scan.energies[2]
