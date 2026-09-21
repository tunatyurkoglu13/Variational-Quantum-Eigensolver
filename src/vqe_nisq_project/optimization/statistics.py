"""Effect sizes and multiple-comparison correction for comparing optimizers/ansatze
across the >=20-seed distributions `multi_seed.py` already produces -- a mean +/-
bootstrap CI (Phase 4) says two configurations differ, but not how confidently across
many simultaneous comparisons, nor how big the difference actually is relative to the
run-to-run spread.

Why Cliff's delta, not Cohen's d: Cohen's d = (mean_a - mean_b) / pooled_std assumes the
two distributions are roughly normal and compares means to a POOLED standard deviation --
a poor assumption here, since Phase 4 found Adam's 20-seed energy distribution is
genuinely BIMODAL (50% converge to FCI, 50% land in a distinct local minimum), not a
single bell curve any pooled-std number meaningfully describes. Cliff's delta instead
asks a purely rank-based question that needs no distributional assumption at all: for a
random pair (one draw from A, one from B), how much more likely is A > B than A < B?

    delta = [ #{(a,b) in AxB : a > b} - #{(a,b) in AxB : a < b} ] / (n_a * n_b)

delta in [-1, +1]: +1 means every single value in A exceeds every value in B (complete
separation), 0 means the two distributions are fully interleaved (no ordering tendency
either way), -1 is complete separation the other direction. This is exactly the
population quantity the Mann-Whitney U test's null hypothesis (H0: P(A>B) = P(A<B) = 1/2)
concerns -- so delta and the Mann-Whitney U statistic are related by a known identity,
verified directly in this module's tests rather than assumed:

    delta = 2*U/(n_a*n_b) - 1

(U here is the U statistic for "A greater," as scipy.stats.mannwhitneyu(a, b) returns by
default -- U/(n_a*n_b) is the empirical P(A>B) plus half the tie probability, so
2*U/(n_a*n_b) - 1 collapses to the same rank-count difference as the direct delta formula
above when there are no ties.)

Holm-Bonferroni step-down correction: testing m hypotheses at a nominal significance level
alpha inflates the chance of at least one false positive (the family-wise error rate) well
above alpha if each test is judged at alpha individually -- e.g. 6 independent tests each
at alpha=0.05 have a family-wise false-positive rate of 1-0.95^6 ~ 26%, not 5%. Plain
Bonferroni fixes this by testing every hypothesis at alpha/m (conservative but simple).
Holm's step-down procedure controls the SAME family-wise error rate exactly, but with more
power: sort the m p-values ascending p_(1) <= ... <= p_(m), then reject p_(1), p_(2), ...
in order as long as p_(k) <= alpha/(m-k+1), stopping (and accepting all remaining, larger
p-values) at the first k that fails this test. The threshold starts as strict as plain
Bonferroni (alpha/m for the smallest p-value) but relaxes for each subsequent comparison,
which is what gives it more power while still controlling the same family-wise rate.

Real finding, applying this to Phase 4's own 4-optimizer/20-seed H2 data (all 6 pairwise
comparisons at once): SPSA vs Adam has a raw Mann-Whitney p-value of 0.047 -- just under
the uncorrected 0.05 threshold, i.e. it would look "significant" read in isolation -- but
does NOT survive Holm correction across the full family of 6 comparisons. This is a live,
concrete instance of exactly the failure mode Holm-Bonferroni exists to prevent, found in
this project's own real data rather than a textbook illustration. Separately, L-BFGS-B vs
SPSA shows near-complete rank separation (|delta| > 0.9, SPSA essentially never converges
on this landscape -- Phase 4's finding), while L-BFGS-B vs Adam shows a smaller but still
"large" separation (|delta| ~ 0.68, conventionally "large" is |delta| > 0.474 per Romano
et al. 2006) precisely BECAUSE Adam's distribution is bimodal (half its seeds land as well
as L-BFGS-B does) -- Cliff's delta is telling exactly this nuanced story correctly, which
a single mean-difference number could not.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats

from vqe_nisq_project.ansatz.base import FloatArray


def cliffs_delta(a: FloatArray, b: FloatArray) -> float:
    """Direct O(n_a * n_b) pairwise-count implementation -- deliberately not the faster
    rank-based shortcut, so this stands as an independent ground truth to check the
    Mann-Whitney-U identity against (see module docstring)."""
    diffs = a[:, None] - b[None, :]
    greater = int(np.sum(diffs > 0))
    less = int(np.sum(diffs < 0))
    return (greater - less) / (len(a) * len(b))


def mann_whitney_p_value(a: FloatArray, b: FloatArray) -> float:
    """Two-sided Mann-Whitney U test p-value (H0: the two distributions are identical
    in the P(A>B)=P(A<B) sense) -- scipy's implementation, used as the trusted
    significance-test primitive; this module's own contribution is Cliff's delta (the
    effect size) and the Holm-Bonferroni correction across many such p-values."""
    _, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p_value)


def holm_bonferroni(p_values: FloatArray, alpha: float = 0.05) -> list[bool]:
    """Returns, in the SAME order as `p_values`, whether each hypothesis is rejected
    (statistically significant) after Holm's step-down family-wise-error correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break  # step-down: every remaining (larger) p-value also fails
    return reject


@dataclass(frozen=True)
class PairwiseComparison:
    name_a: str
    name_b: str
    cliffs_delta: float
    p_value_raw: float
    significant_after_holm: bool


def compare_all_pairs(
    samples: dict[str, FloatArray], alpha: float = 0.05
) -> list[PairwiseComparison]:
    """All C(k,2) pairwise comparisons among `samples` (name -> per-seed energy array),
    each with its own Cliff's delta and Mann-Whitney p-value, jointly Holm-corrected as
    ONE family of m=C(k,2) hypotheses (not each judged against alpha in isolation)."""
    names = list(samples.keys())
    pairs = list(combinations(names, 2))
    p_values = np.array([mann_whitney_p_value(samples[a], samples[b]) for a, b in pairs])
    deltas = [cliffs_delta(samples[a], samples[b]) for a, b in pairs]
    rejected = holm_bonferroni(p_values, alpha=alpha)

    return [
        PairwiseComparison(
            name_a=a,
            name_b=b,
            cliffs_delta=delta,
            p_value_raw=float(p),
            significant_after_holm=sig,
        )
        for (a, b), delta, p, sig in zip(pairs, deltas, p_values, rejected, strict=True)
    ]
