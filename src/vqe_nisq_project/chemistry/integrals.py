"""Extract MO-basis one- and two-electron integrals from PySCF, by hand.

Done manually (not via qiskit-nature's PySCFDriver) so every step -- AO
integrals, the RHF orbital coefficients, the AO->MO basis change -- is
visible and independently checkable, rather than hidden inside a driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo, gto, scf

from vqe_nisq_project.chemistry.molecule import MoleculeSpec

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MOIntegrals:
    """MO-basis integrals in chemist's notation: two_body[p,q,r,s] = (pq|rs)."""

    one_body: FloatArray  # h_pq, shape (n_mo, n_mo)
    two_body: FloatArray  # (pq|rs), shape (n_mo, n_mo, n_mo, n_mo), chemist's notation
    e_nuc: float
    e_hf: float  # PySCF's own converged RHF energy, for cross-checking
    n_electrons: int
    n_mo: int


def compute_mo_integrals(spec: MoleculeSpec) -> MOIntegrals:
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

    mo_coeff = mf.mo_coeff
    n_mo = mo_coeff.shape[1]

    # One-electron: kinetic + nuclear attraction, AO -> MO.
    h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h_mo = mo_coeff.T @ h_ao @ mo_coeff

    # Two-electron: AO->MO 4-index transform, chemist's notation (pq|rs).
    eri_mo = ao2mo.kernel(mol, mo_coeff, compact=False)
    two_body = eri_mo.reshape(n_mo, n_mo, n_mo, n_mo)

    return MOIntegrals(
        one_body=h_mo,
        two_body=two_body,
        e_nuc=mol.energy_nuc(),
        e_hf=e_hf,
        n_electrons=mol.nelectron,
        n_mo=n_mo,
    )


def rhf_energy_from_integrals(integrals: MOIntegrals) -> float:
    """Reconstruct the closed-shell RHF energy directly from MO integrals.

    E_HF = E_nuc + 2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ji)], i,j over
    occupied (doubly-filled) MOs. This must match `integrals.e_hf` (PySCF's
    own converged energy) to numerical precision -- it is the ground-truth
    check that the AO->MO integral extraction above is correct, independent
    of anything qiskit-nature or OpenFermion will later do with these same
    integrals.
    """
    n_occ = integrals.n_electrons // 2
    h = integrals.one_body
    g = integrals.two_body

    one_electron_energy = 2.0 * np.trace(h[:n_occ, :n_occ])
    occ_coulomb = g[:n_occ, :n_occ, :n_occ, :n_occ]
    two_electron_energy = np.einsum("iijj->", occ_coulomb) * 2.0 - np.einsum("ijji->", occ_coulomb)
    return float(integrals.e_nuc + one_electron_energy + two_electron_energy)
