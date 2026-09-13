"""Classical reference energies (the "ground truth" VQE results are judged against).

With a frozen core (n_frozen_core>0), `fci` is the active-space CASCI energy --
exact within the active space, and the correct target for our qubit
Hamiltonian to reproduce. `full_fci` (exact over the whole molecule, no
frozen core) and `frozen_core_error = fci - full_fci` are also computed so
the frozen-core approximation's cost is measured, not assumed negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyscf import cc, fci, gto, mcscf, scf

from vqe_nisq_project.chemistry.molecule import MoleculeSpec


@dataclass(frozen=True)
class ReferenceEnergies:
    hf: float
    ccsd: float
    ccsd_t: float
    fci: float  # active-space CASCI if n_frozen_core>0, else full-molecule FCI
    e_nuc: float
    full_fci: float | None = None  # only populated when n_frozen_core > 0
    frozen_core_error: float | None = None  # fci - full_fci, only when n_frozen_core > 0


def compute_reference_energies(spec: MoleculeSpec, n_frozen_core: int = 0) -> ReferenceEnergies:
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

    mycc = cc.CCSD(mf, frozen=n_frozen_core)
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

    full_fci: float | None = None
    frozen_core_error: float | None = None

    if n_frozen_core == 0:
        e_fci, _ = fci.FCI(mf).kernel()
    else:
        n_mo_total = mf.mo_coeff.shape[1]
        n_active_orbitals = n_mo_total - n_frozen_core
        n_active_electrons = mol.nelectron - 2 * n_frozen_core
        cas = mcscf.CASCI(mf, ncas=n_active_orbitals, nelecas=n_active_electrons)
        e_fci = cas.kernel()[0]
        full_fci, _ = fci.FCI(mf).kernel()
        frozen_core_error = e_fci - full_fci

    return ReferenceEnergies(
        hf=e_hf,
        ccsd=e_ccsd,
        ccsd_t=e_ccsd_t,
        fci=e_fci,
        e_nuc=mol.energy_nuc(),
        full_fci=full_fci,
        frozen_core_error=frozen_core_error,
    )
