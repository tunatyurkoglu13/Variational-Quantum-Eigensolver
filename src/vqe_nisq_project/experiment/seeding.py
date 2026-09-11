"""Centralized seed management so a given config always reproduces bit-identical results."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every RNG source this repo's code paths touch.

    Qiskit primitives (Sampler/Estimator) and PennyLane devices take their own
    per-call `seed` argument rather than reading global state, so those must
    still be passed `seed` explicitly at the call site -- this only covers the
    process-global sources (python's stdlib random, numpy's legacy global RNG).
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Preferred RNG for new code: an explicit, non-global numpy Generator."""
    return np.random.default_rng(seed)
