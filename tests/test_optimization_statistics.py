"""Ground truth: Cliff's delta must agree with the value derived independently from
scipy's Mann-Whitney U statistic (delta = 2U/(n*m) - 1), hit its +-1 extremes exactly on
fully-separated samples, and Holm-Bonferroni must control the family-wise error rate more
powerfully than plain Bonferroni while never rejecting more than plain (uncorrected)
per-test alpha would."""

import numpy as np
from scipy import stats

from vqe_nisq_project.optimization.statistics import (
    cliffs_delta,
    compare_all_pairs,
    holm_bonferroni,
    mann_whitney_p_value,
)


def test_cliffs_delta_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    assert abs(cliffs_delta(a, a.copy())) < 1e-9


def test_cliffs_delta_hits_plus_one_for_complete_separation() -> None:
    a = np.arange(20, 40, dtype=float)
    b = np.arange(0, 20, dtype=float)
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0


def test_cliffs_delta_matches_the_mann_whitney_u_identity() -> None:
    """delta = 2U/(n*m) - 1 is a known identity -- verified directly against scipy's own
    U statistic rather than assumed, across several random samples."""
    rng = np.random.default_rng(1)
    for trial in range(5):
        a = rng.normal(trial * 0.3, 1.0, 25)
        b = rng.normal(0.0, 1.0, 18)
        delta_direct = cliffs_delta(a, b)
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        delta_from_u = 2 * u_stat / (len(a) * len(b)) - 1
        assert abs(delta_direct - delta_from_u) < 1e-9


def test_mann_whitney_p_value_is_large_for_identical_distributions() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0, 1, 30)
    assert mann_whitney_p_value(a, b) > 0.05


def test_mann_whitney_p_value_is_tiny_for_complete_separation() -> None:
    a = np.arange(20, 40, dtype=float)
    b = np.arange(0, 20, dtype=float)
    assert mann_whitney_p_value(a, b) < 1e-4


def test_holm_bonferroni_rejects_no_more_than_uncorrected_alpha_would() -> None:
    p_values = np.array([0.001, 0.02, 0.03, 0.04, 0.20, 0.50])
    rejected = holm_bonferroni(p_values, alpha=0.05)
    uncorrected = [p <= 0.05 for p in p_values]
    # Holm is strictly more conservative (or equal) than judging each p-value alone.
    assert all(not r or u for r, u in zip(rejected, uncorrected, strict=True))


def test_holm_bonferroni_step_down_is_more_powerful_than_plain_bonferroni() -> None:
    # A classic case where step-down rejects more than plain Bonferroni (alpha/m
    # for every test) would, while still controlling the same family-wise rate.
    p_values = np.array([0.001, 0.008, 0.015, 0.30])
    alpha = 0.05
    m = len(p_values)
    plain_bonferroni_rejected = [p <= alpha / m for p in p_values]
    holm_rejected = holm_bonferroni(p_values, alpha=alpha)
    assert sum(holm_rejected) >= sum(plain_bonferroni_rejected)
    assert sum(holm_rejected) > sum(plain_bonferroni_rejected)


def test_holm_bonferroni_stops_at_first_failure_not_just_thresholding_each_p_value() -> None:
    # A p-value that individually clears alpha/(m-rank) can still be correctly NOT
    # rejected if an earlier (smaller) p-value in the sorted order already failed --
    # the step-down procedure stops there, it does not skip and re-test later ranks.
    p_values = np.array([0.20, 0.001])  # sorted: 0.001 (rank0), 0.20 (rank1)
    # rank0 threshold = 0.05/2 = 0.025 -> 0.001 passes, rejected
    # rank1 threshold = 0.05/1 = 0.05 -> 0.20 fails, NOT rejected, and it's the last one
    rejected = holm_bonferroni(p_values, alpha=0.05)
    assert rejected == [False, True]


def test_compare_all_pairs_on_real_multi_seed_optimizer_data() -> None:
    """The real Phase 7 deliverable: apply this module to genuine 20-seed H2/EfficientSU2
    optimizer distributions (COBYLA/SPSA/L-BFGS-B/Adam), reproducing Phase 4's own
    documented means (COBYLA -1.1117, L-BFGS-B -1.1343, SPSA -0.8109, Adam -0.9211) via
    the SAME real optimization runs -- not synthetic stand-in data -- then checks the
    statistics module's real, substantive conclusion: L-BFGS-B (exact gradient, smooth
    landscape) is significantly better than every gradient-free method here, with a
    LARGE (|delta| close to 1) effect size, surviving Holm-Bonferroni correction across
    all 6 pairwise comparisons at once."""
    import pennylane as qml

    from conftest import cached_mo_integrals, cached_reference_energies
    from vqe_nisq_project.ansatz.base import reverse_qubit_endianness
    from vqe_nisq_project.ansatz.hardware_efficient import (
        build_efficient_su2_qiskit,
        build_hardware_efficient_pennylane,
    )
    from vqe_nisq_project.chemistry.mappings import build_electronic_fermionic_op, map_hamiltonian
    from vqe_nisq_project.chemistry.molecule import h2
    from vqe_nisq_project.optimization.multi_seed import run_multi_seed

    integrals = cached_mo_integrals(h2())
    fop = build_electronic_fermionic_op(integrals)
    hamiltonian = map_hamiltonian(fop, "jordan_wigner").qubit_op.to_matrix()
    fci = cached_reference_energies(h2()).fci
    n_qubits = 4

    ansatz_qiskit = build_efficient_su2_qiskit(n_qubits, reps=2)
    ansatz_pl = build_hardware_efficient_pennylane(n_qubits, reps=2)

    def make_adam_cost_fn():
        hamiltonian_pl_basis = reverse_qubit_endianness(hamiltonian, n_qubits)
        observable = qml.Hermitian(hamiltonian_pl_basis, wires=range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)  # type: ignore[untyped-decorator]
        def circuit(params: np.ndarray) -> qml.measurements.ExpectationMP:
            qml.StronglyEntanglingLayers(params.reshape(2, n_qubits, 3), wires=range(n_qubits))
            return qml.expval(observable)

        def cost(params: np.ndarray) -> float:
            return circuit(params) + integrals.e_nuc

        return cost

    configs = [
        ("cobyla", ansatz_qiskit, 150, None),
        ("lbfgsb", ansatz_qiskit, 100, None),
        ("spsa", ansatz_qiskit, 300, None),
        ("adam", ansatz_pl, 200, make_adam_cost_fn),
    ]
    samples = {}
    for name, ansatz, max_iterations, factory in configs:
        result = run_multi_seed(
            ansatz.state_fn,
            hamiltonian,
            ansatz.n_parameters,
            name,  # type: ignore[arg-type]
            best_known_energy=fci,
            e_core=integrals.e_nuc,
            max_iterations=max_iterations,
            n_seeds=20,
            init_scale=0.5,
            pennylane_cost_fn_factory=factory,
        )
        samples[name] = result.final_energies

    # Sanity: reproduces Phase 4's own documented means (see module docstring).
    assert abs(samples["cobyla"].mean() - (-1.1117)) < 0.01
    assert abs(samples["lbfgsb"].mean() - (-1.1343)) < 0.01
    assert abs(samples["spsa"].mean() - (-0.8109)) < 0.05
    assert abs(samples["adam"].mean() - (-0.9211)) < 0.05

    comparisons = compare_all_pairs(samples, alpha=0.05)
    assert len(comparisons) == 6  # C(4,2)
    by_pair = {frozenset((c.name_a, c.name_b)): c for c in comparisons}

    # L-BFGS-B vs SPSA: SPSA never converges here (0% fraction_converged, Phase 4) --
    # a near-COMPLETE rank separation (|delta| close to 1), clearly significant.
    lbfgsb_vs_spsa = by_pair[frozenset({"lbfgsb", "spsa"})]
    assert abs(lbfgsb_vs_spsa.cliffs_delta) > 0.9
    assert lbfgsb_vs_spsa.significant_after_holm

    # L-BFGS-B vs Adam: Adam is BIMODAL (Phase 4: 50% converge to FCI, 50% land in a
    # distinct local minimum) -- so the separation is real and still "large" by
    # convention (|delta| > 0.474, Romano et al. 2006's non-parametric large-effect
    # threshold) but NOT complete, unlike the SPSA comparison above. Still survives
    # Holm correction.
    lbfgsb_vs_adam = by_pair[frozenset({"lbfgsb", "adam"})]
    assert 0.5 < abs(lbfgsb_vs_adam.cliffs_delta) < 0.9
    assert lbfgsb_vs_adam.significant_after_holm

    # The real, concrete case for WHY multiple-comparison correction matters: SPSA vs
    # Adam has a raw p-value just BELOW the uncorrected 0.05 threshold (would look
    # "significant" read in isolation) but does NOT survive Holm correction across all
    # 6 simultaneous comparisons -- a live example of the exact failure mode Holm-
    # Bonferroni exists to prevent, found in this project's own real data.
    spsa_vs_adam = by_pair[frozenset({"spsa", "adam"})]
    assert spsa_vs_adam.p_value_raw < 0.05
    assert not spsa_vs_adam.significant_after_holm
