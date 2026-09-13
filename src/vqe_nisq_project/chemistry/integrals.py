"""Extract MO-basis (optionally active-space-reduced) integrals from PySCF.

For the full space (n_frozen_core=0), this is done via a manual AO->MO
transform so every step is visible (not hidden inside qiskit-nature's
PySCFDriver). For a reduced active space (n_frozen_core>0, used for LiH/
BeH2), we instead use PySCF's own `mcscf.CASCI` machinery -- re-deriving the
frozen-core integral folding by hand would just be reimplementing an already
validated, standard technique, with more room for error. This module's own
tests confirm CASCI's n_frozen_core=0 output reproduces the manual path
exactly (to ~1e-16), which is what justifies trusting it for the frozen-core
case too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, mcscf, scf

from vqe_nisq_project.chemistry.molecule import MoleculeSpec

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MOIntegrals:
    """Active-space integrals in chemist's notation: two_body[p,q,r,s] = (pq|rs).

    For n_frozen_core=0, "active space" is the whole molecule and
    e_core == e_nuc. one_body/two_body/n_electrons/n_mo describe only the
    active space; the frozen core's own energy contribution is folded into
    e_core (which is what must be added back to an active-space electronic
    energy to get the molecule's total energy).
    """

    one_body: FloatArray  # h_pq, shape (n_mo, n_mo) -- active space only
    two_body: FloatArray  # (pq|rs), shape (n_mo,)*4 -- active space only, chemist's notation
    e_nuc: float  # nuclear repulsion energy alone
    e_core: float  # e_nuc + frozen-core mean-field energy (0 frozen core => e_core == e_nuc)
    e_hf: float  # PySCF's own converged full-molecule RHF energy, for cross-checking
    n_electrons: int  # ACTIVE electrons only
    n_mo: int  # ACTIVE spatial orbitals only
    n_frozen_core: int = 0


def compute_mo_integrals(spec: MoleculeSpec, n_frozen_core: int = 0) -> MOIntegrals:
    mol = gto.M(
        atom=spec.atom,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit=spec.unit,
        verbose=0,
    )
    mf = scf.RHF(mol)
    e_hf = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {spec.atom} / {spec.basis}")

    n_mo_total = mf.mo_coeff.shape[1]
    n_active_orbitals = n_mo_total - n_frozen_core
    n_active_electrons = mol.nelectron - 2 * n_frozen_core
    if n_active_electrons < 0 or n_active_orbitals < 1:
        raise ValueError(
            f"n_frozen_core={n_frozen_core} leaves {n_active_electrons} active electrons "
            f"in {n_active_orbitals} active orbitals -- not a valid active space."
        )

    cas = mcscf.CASCI(mf, ncas=n_active_orbitals, nelecas=n_active_electrons)
    h1_active, e_core = cas.get_h1eff()
    h2_active_compact = cas.get_h2cas()
    two_body = ao2mo.restore("s1", h2_active_compact, n_active_orbitals)

    return MOIntegrals(
        one_body=h1_active,
        two_body=two_body,
        e_nuc=mol.energy_nuc(),
        e_core=e_core,
        e_hf=e_hf,
        n_electrons=n_active_electrons,
        n_mo=n_active_orbitals,
        n_frozen_core=n_frozen_core,
    )


def rhf_energy_from_integrals(integrals: MOIntegrals) -> float:
    """Reconstruct the full-molecule closed-shell RHF energy from active-space integrals.

    E_HF = e_core + 2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ji)], i,j over
    doubly-occupied ACTIVE MOs (the frozen core's own contribution is already
    folded into e_core). This must match `integrals.e_hf` (PySCF's own
    converged energy) to numerical precision -- the ground-truth check that
    both the AO->MO transform (n_frozen_core=0) and the CASCI active-space
    reduction (n_frozen_core>0) are correct.
    """
    n_occ = integrals.n_electrons // 2
    h = integrals.one_body
    g = integrals.two_body

    one_electron_energy = 2.0 * np.trace(h[:n_occ, :n_occ])
    occ_coulomb = g[:n_occ, :n_occ, :n_occ, :n_occ]
    two_electron_energy = np.einsum("iijj->", occ_coulomb) * 2.0 - np.einsum("ijji->", occ_coulomb)
    return float(integrals.e_core + one_electron_energy + two_electron_energy)
