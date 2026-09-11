"""Ground-truth property: a given seed must reproduce bit-identical numpy/random output."""

import random

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from vqe_nisq_project.experiment.seeding import make_rng, seed_everything


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_everything_is_deterministic(seed: int) -> None:
    seed_everything(seed)
    random_draw_1 = random.random()
    numpy_draw_1 = np.random.rand()

    seed_everything(seed)
    random_draw_2 = random.random()
    numpy_draw_2 = np.random.rand()

    assert random_draw_1 == random_draw_2
    assert numpy_draw_1 == numpy_draw_2


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_make_rng_is_deterministic(seed: int) -> None:
    rng_1 = make_rng(seed)
    rng_2 = make_rng(seed)
    assert rng_1.normal(size=10).tolist() == rng_2.normal(size=10).tolist()


def test_different_seeds_diverge() -> None:
    rng_a = make_rng(1)
    rng_b = make_rng(2)
    assert rng_a.normal(size=10).tolist() != rng_b.normal(size=10).tolist()


def test_make_rng_returns_generator_not_global_state() -> None:
    # Drawing from make_rng() must not perturb the legacy global numpy RNG
    # seeded by seed_everything() -- the two RNG sources are independent.
    seed_everything(42)
    expected = np.random.rand()

    seed_everything(42)
    make_rng(999).normal(size=1000)  # exercise a differently-seeded local RNG
    actual = np.random.rand()

    assert actual == expected
