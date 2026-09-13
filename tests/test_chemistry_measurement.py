"""Ground truth: qubit-wise-commuting groups must fully partition the Hamiltonian's Pauli
terms -- this count is what actually sets the number of distinct measurement circuits
(shot cost) a real run needs."""

import pytest

from conftest import cached_mapping as map_hamiltonian
from conftest import cached_tapered_sector
from vqe_nisq_project.chemistry.measurement import group_qubit_wise_commuting
from vqe_nisq_project.chemistry.molecule import MoleculeSpec, beh2, h2, lih

_CASES = [(h2(), 0), (lih(), 1), (beh2(), 1)]
_IDS = ["h2", "lih", "beh2"]


@pytest.mark.parametrize(("spec", "n_frozen_core"), _CASES, ids=_IDS)
def test_groups_partition_all_terms_exactly(spec: MoleculeSpec, n_frozen_core: int) -> None:
    jw = map_hamiltonian(spec, n_frozen_core, "jordan_wigner")
    result = group_qubit_wise_commuting(jw.qubit_op)
    assert sum(len(g) for g in result.groups) == result.n_terms


def test_full_jw_hamiltonian_needs_five_measurement_groups_h2() -> None:
    # Regression check for this specific system.
    jw = map_hamiltonian(h2(), 0, "jordan_wigner")
    result = group_qubit_wise_commuting(jw.qubit_op)
    assert result.n_terms == 15
    assert result.n_groups == 5


@pytest.mark.parametrize(
    ("spec", "n_frozen_core", "full_groups", "tapered_groups"),
    [(h2(), 0, 5, 2), (lih(), 1, 58, 38), (beh2(), 1, 53, 56)],
    ids=_IDS,
)
def test_tapering_effect_on_measurement_groups_is_not_always_a_reduction(
    spec: MoleculeSpec, n_frozen_core: int, full_groups: int, tapered_groups: int
) -> None:
    """Honest, verified finding: tapering cuts qubits and Pauli terms, but NOT always the
    number of measurement groups. For H2 and LiH it helps (5->2, 58->38); for BeH2 it
    actually costs one extra measurement group (53->56) -- the Clifford basis change used
    to taper can break up what were qubit-wise-commuting groups in the original encoding.
    This is reported as-is rather than assumed away or averaged over, per this project's
    rule against overclaiming a technique "always helps"."""
    jw = map_hamiltonian(spec, n_frozen_core, "jordan_wigner")
    tapered = cached_tapered_sector(spec, n_frozen_core)

    assert group_qubit_wise_commuting(jw.qubit_op).n_groups == full_groups
    assert group_qubit_wise_commuting(tapered.tapered_op).n_groups == tapered_groups


@pytest.mark.parametrize(("spec", "n_frozen_core"), _CASES, ids=_IDS)
def test_tapering_always_reduces_or_maintains_term_count(
    spec: MoleculeSpec, n_frozen_core: int
) -> None:
    # Unlike measurement-group count, the raw number of distinct Pauli terms
    # after tapering is never larger than before -- this one direction does
    # hold universally in our results so far.
    jw = map_hamiltonian(spec, n_frozen_core, "jordan_wigner")
    tapered = cached_tapered_sector(spec, n_frozen_core)
    assert len(tapered.tapered_op) <= len(jw.qubit_op)
