"""Test-suite-only memoization of expensive per-molecule PySCF calculations.

Not added to the production `chemistry` modules themselves -- a real pipeline
sweeping many geometries (a dissociation curve) legitimately calls these with
ever-different arguments, so a global cache there would just accumulate
unboundedly. Here, the same handful of (spec, n_frozen_core) pairs repeat
across many independent test functions/files, so caching is a straightforward
test-suite speedup with no correctness implications.
"""

from functools import cache

from vqe_nisq_project.chemistry.fermionic import CrossValidatedHamiltonian
from vqe_nisq_project.chemistry.fermionic import cross_validate as _cross_validate
from vqe_nisq_project.chemistry.integrals import MOIntegrals
from vqe_nisq_project.chemistry.integrals import compute_mo_integrals as _compute_mo_integrals
from vqe_nisq_project.chemistry.mappings import (
    MappingName,
    MappingResult,
)
from vqe_nisq_project.chemistry.mappings import (
    build_electronic_fermionic_op as _build_electronic_fermionic_op,
)
from vqe_nisq_project.chemistry.mappings import map_hamiltonian as _map_hamiltonian
from vqe_nisq_project.chemistry.molecule import MoleculeSpec
from vqe_nisq_project.chemistry.reference import ReferenceEnergies
from vqe_nisq_project.chemistry.reference import (
    compute_reference_energies as _compute_reference_energies,
)
from vqe_nisq_project.chemistry.tapering import TaperingResult
from vqe_nisq_project.chemistry.tapering import find_tapered_sector as _find_tapered_sector


@cache
def cached_mo_integrals(spec: MoleculeSpec, n_frozen_core: int = 0) -> MOIntegrals:
    return _compute_mo_integrals(spec, n_frozen_core)


@cache
def cached_reference_energies(spec: MoleculeSpec, n_frozen_core: int = 0) -> ReferenceEnergies:
    return _compute_reference_energies(spec, n_frozen_core)


@cache
def cached_mapping(spec: MoleculeSpec, n_frozen_core: int, name: MappingName) -> MappingResult:
    # The dense diagonalization inside map_hamiltonian is the single most
    # expensive step in this whole test suite for the larger (LiH/BeH2)
    # systems (~11s for BeH2's 12-qubit JW operator) -- cache aggressively.
    integrals = cached_mo_integrals(spec, n_frozen_core)
    fop = _build_electronic_fermionic_op(integrals)
    return _map_hamiltonian(fop, name)


@cache
def cached_cross_validate(spec: MoleculeSpec, n_frozen_core: int = 0) -> CrossValidatedHamiltonian:
    integrals = cached_mo_integrals(spec, n_frozen_core)
    return _cross_validate(integrals)


@cache
def cached_tapered_sector(
    spec: MoleculeSpec, n_frozen_core: int = 0, mapping_name: MappingName = "jordan_wigner"
) -> TaperingResult:
    integrals = cached_mo_integrals(spec, n_frozen_core)
    fci = cached_reference_energies(spec, n_frozen_core).fci
    mapping = cached_mapping(spec, n_frozen_core, mapping_name)
    return _find_tapered_sector(mapping.qubit_op, fci - integrals.e_core)
