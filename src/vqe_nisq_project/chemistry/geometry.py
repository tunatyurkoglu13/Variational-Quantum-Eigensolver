"""Find equilibrium bond lengths ourselves via an RHF energy scan, rather than citing a
literature value we haven't verified. This is basis-set-specific (STO-3G here) by
construction -- it need not match the experimental equilibrium geometry, and isn't meant to;
it is the correct reference point for *this* study's own internal consistency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pyscf import gto, scf


@dataclass(frozen=True)
class BondScanResult:
    bond_lengths: list[float]
    energies: list[float]
    r_eq: float  # parabolic-refined equilibrium bond length
    e_eq: float  # RHF energy at the discrete grid point closest to r_eq


def scan_rhf_energy(
    atom_template: Callable[[float], str], basis: str, bond_lengths: list[float]
) -> BondScanResult:
    """RHF energy at each bond length, plus a parabolic fit near the discrete minimum."""
    energies = []
    for r in bond_lengths:
        mol = gto.M(atom=atom_template(r), basis=basis, unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"RHF did not converge at R={r}")
        energies.append(e)

    i_min = int(np.argmin(energies))
    if i_min == 0 or i_min == len(bond_lengths) - 1:
        raise ValueError(
            f"Discrete minimum at scan boundary (R={bond_lengths[i_min]}) -- "
            "widen the scan range before trusting a parabolic refinement."
        )

    # Parabolic (3-point) refinement around the discrete minimum for sub-grid precision.
    r0, r1, r2 = bond_lengths[i_min - 1 : i_min + 2]
    e0, e1, e2 = energies[i_min - 1 : i_min + 2]
    # Fit E(r) = a*r^2 + b*r + c through the three points; vertex at r = -b / (2a).
    denom = (r0 - r1) * (r0 - r2) * (r1 - r2)
    a = (r2 * (e1 - e0) + r1 * (e0 - e2) + r0 * (e2 - e1)) / denom
    b = (r2**2 * (e0 - e1) + r1**2 * (e2 - e0) + r0**2 * (e1 - e2)) / denom
    r_eq = -b / (2 * a)

    return BondScanResult(
        bond_lengths=bond_lengths, energies=energies, r_eq=r_eq, e_eq=energies[i_min]
    )
