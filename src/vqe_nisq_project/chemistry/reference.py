"""Classical reference energies (the "ground truth" VQE results are judged against)."""

from __future__ import annotations

from dataclasses import dataclass

from pyscf import cc, fci, gto, scf

from vqe_nisq_project.chemistry.molecule import MoleculeSpec


@dataclass(frozen=True)
class ReferenceEnergies:
    hf: float
    ccsd: float
    ccsd_t: float
    fci: float
    e_nuc: float


def compute_reference_energies(spec: MoleculeSpec) -> ReferenceEnergies:
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

    mycc = cc.CCSD(mf)
    # PySCF's CCSD default conv_tol (1e-7) is looser than FCI's (1e-10) -- tight
    # enough to look converged but loose enough to sit ~1e-7 Ha off the true
    # variational minimum, which showed up as CCSD appearing (numerically, not
    # physically) below FCI. Match FCI's precision so the reference table's
    # HF >= CCSD(T) >= CCSD >= FCI ordering holds at the precision we report it.
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    e_ccsd_corr, _, _ = mycc.kernel()
    e_ccsd = e_hf + e_ccsd_corr
    e_ccsd_t = e_ccsd + mycc.ccsd_t()

    e_fci, _ = fci.FCI(mf).kernel()

    return ReferenceEnergies(
        hf=e_hf, ccsd=e_ccsd, ccsd_t=e_ccsd_t, fci=e_fci, e_nuc=mol.energy_nuc()
    )
