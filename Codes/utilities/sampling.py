"""Sampling helpers for max-affine models."""

from typing import Any

import numpy as np


def sample_max_affine_vectors(
    k: int,
    d: int,
    rng: Any = None,
) -> np.ndarray:
    """Sample k independent standard-normal vectors in R^(d+1).

    Args:
        k: Number of vectors to sample.
        d: Base dimension. Each sampled vector has dimension d + 1.
        rng: Optional NumPy random generator or integer seed.

    Returns:
        A NumPy array with shape (k, d + 1).
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if d < 0:
        raise ValueError("d must be nonnegative.")
    if k > d:
        raise ValueError("k must be no larger than d to sample a full-rank d x k submatrix.")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    while True:
        beta = generator.standard_normal(size=(k, d + 1))
        if np.linalg.matrix_rank(beta[:, :d].T) == k:
            return beta
