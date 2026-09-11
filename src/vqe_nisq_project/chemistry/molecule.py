"""Molecule specifications used throughout this repo.

Bond length for H2 (0.735 Angstrom) is a standard textbook-level equilibrium
value in the same range used across the VQE literature for this exact system
(e.g. Peruzzo et al. 2014, O'Malley et al. 2016 both study H2/STO-3G near this
geometry) -- but it is not asserted here as matching any specific paper's
table; it is simply the fixed geometry this repo computes its own PySCF
ground truth at. A literature cross-reference, if added, will cite a verified
arXiv ID / DOI, not memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoleculeSpec:
    """A minimal, PySCF-ready molecule specification."""

    atom: str
    basis: str
    charge: int = 0
    spin: int = 0  # 2S, i.e. n_alpha - n_beta
    unit: str = "Angstrom"

    @property
    def label(self) -> str:
        return self.atom.replace(" ", "").replace(";", "_")


def h2(bond_length_angstrom: float = 0.735) -> MoleculeSpec:
    """H2 in the minimal STO-3G basis at a given bond length."""
    return MoleculeSpec(
        atom=f"H 0 0 0; H 0 0 {bond_length_angstrom}",
        basis="sto-3g",
    )
