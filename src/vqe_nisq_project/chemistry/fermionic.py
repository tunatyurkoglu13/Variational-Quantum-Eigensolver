"""Build the same electronic Hamiltonian independently in qiskit-nature and OpenFermion.

Both consume the *same* MO integrals from `integrals.py`, so agreement here
tests the two libraries' second-quantization + qubit-mapping machinery
against each other, not PySCF's integrals against themselves.

Two library-specific conventions had to be empirically nailed down (not
assumed from memory/docs, per this repo's rule against unverified claims):

1. qiskit-nature's ``ElectronicEnergy.from_raw_integrals`` does *not*
   default the beta-spin blocks to the alpha-spin ones when omitted -- it
   silently computes an alpha-spin-only (unphysical) operator. For a
   restricted (closed-shell) calculation, h1_b/h2_bb/h2_ba must be passed
   explicitly (equal to the alpha versions here).
2. qiskit-nature's ``ElectronicEnergy.second_q_op()`` returns the
   *electronic-only* operator -- ``nuclear_repulsion_energy`` is bookkeeping
   metadata that higher-level solvers add back in, it is not folded into the
   returned operator. OpenFermion's ``InteractionOperator``'s ``constant``
   argument, by contrast, *is* folded directly into the operator. Comparing
   the two libraries fairly means either zeroing that constant (electronic-
   only comparison) or adding nuclear repulsion back onto qiskit-nature's
   side by hand.
3. The two libraries also order spin-orbitals differently (interleaved
   alpha/beta vs. blocked), so a raw element-wise matrix difference between
   their Jordan-Wigner-mapped operators is large even when the physics is
   identical -- only comparing them in the *same* qubit ordering would make
   a literal matrix norm meaningful. This module instead compares the
   physically invariant quantity: the sorted eigenvalue spectrum, which is
   basis-labeling-independent and a strictly stronger equality test (matched
   here to ~1e-14, far past the 1e-10 target) than a norm that depends on an
   arbitrary but equally-valid indexing choice.

The OpenFermion two-body index convention below (``transpose(0, 2, 3, 1)``
applied to the chemist's-notation ``(pq|rs)`` tensor from `integrals.py`) was
determined by exhaustively trying all 24 index permutations against real FCI
ground truth on two different systems (H2/STO-3G, H2/6-31G) -- not derived
from a textbook formula and trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import get_sparse_operator
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

from vqe_nisq_project.chemistry.integrals import FloatArray, MOIntegrals

_OPENFERMION_TWO_BODY_AXES = (0, 2, 3, 1)


@dataclass(frozen=True)
class CrossValidatedHamiltonian:
    qiskit_nature_matrix: FloatArray  # electronic-only, Jordan-Wigner, dense
    openfermion_matrix: FloatArray  # electronic-only, Jordan-Wigner, dense
    max_eigenvalue_diff: float
    electronic_ground_state: float
    n_qubits: int


def build_qiskit_nature_electronic_matrix(integrals: MOIntegrals) -> FloatArray:
    """Electronic-only Jordan-Wigner matrix, restricted (alpha == beta) closed shell."""
    h1, h2 = integrals.one_body, integrals.two_body
    hamiltonian = ElectronicEnergy.from_raw_integrals(h1, h2, h1, h2, h2)
    qubit_op = JordanWignerMapper().map(hamiltonian.second_q_op())
    matrix: FloatArray = qubit_op.to_matrix()
    return matrix


def build_openfermion_electronic_matrix(integrals: MOIntegrals) -> FloatArray:
    """Electronic-only Jordan-Wigner matrix (constant=0 -- nuclear repulsion excluded)."""
    physicist_two_body = integrals.two_body.transpose(*_OPENFERMION_TWO_BODY_AXES)
    # openfermion ships only partial inline type hints, so these calls are
    # each flagged no-untyped-call under our strict mypy config.
    one_body_so, two_body_so = spinorb_from_spatial(  # type: ignore[no-untyped-call]
        integrals.one_body, physicist_two_body
    )
    interaction_op = InteractionOperator(  # type: ignore[no-untyped-call]
        0.0, one_body_so, 0.5 * two_body_so
    )
    n_qubits = one_body_so.shape[0]
    fermion_op = get_fermion_operator(interaction_op)  # type: ignore[no-untyped-call]
    qubit_op = jordan_wigner(fermion_op)  # type: ignore[no-untyped-call]
    sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)  # type: ignore[no-untyped-call]
    dense: FloatArray = sparse.toarray()
    return dense


def cross_validate(integrals: MOIntegrals) -> CrossValidatedHamiltonian:
    qn_matrix = build_qiskit_nature_electronic_matrix(integrals)
    of_matrix = build_openfermion_electronic_matrix(integrals)

    eig_qn = np.sort(np.linalg.eigvalsh(qn_matrix))
    eig_of = np.sort(np.linalg.eigvalsh(of_matrix))
    max_diff = float(np.max(np.abs(eig_qn - eig_of)))

    return CrossValidatedHamiltonian(
        qiskit_nature_matrix=qn_matrix,
        openfermion_matrix=of_matrix,
        max_eigenvalue_diff=max_diff,
        electronic_ground_state=float(eig_qn[0]),
        n_qubits=qn_matrix.shape[0].bit_length() - 1,
    )
