from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.random import Generator


def seed_and_get_generator(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()


def is_categorical(
    column: pd.Series,
    n_samples: int = 1000,
    max_unique_fraction: float = 0.2,
    random_generator: Generator | None = None,
) -> bool:
    """Check if `column` type is categorical.

    A heuristic to check whether a `column` is categorical:
    a column is considered categorical (as opposed to a plain text column)
    if the relative cardinality is `max_unique_fraction` or less.

    Args:
        column (ArrayLike): pandas `Series` containing strings
        n_samples (int, optional): number of samples used for heuristic. Defaults to 1000.
        max_unique_fraction (float, optional): maximum relative cardinality. Defaults to 0.2.
        random_generator (Generator, optional): random generator. Defaults to None.

    Returns:
        bool: `True` if the column is categorical according to the heuristic.
    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    column = np.array(column)
    n_samples = min(n_samples, len(column))
    values, counts = np.unique(column, return_counts=True)
    sample = random_generator.choice(a=values, p=counts / counts.sum(), size=n_samples)
    unique_samples = np.unique(sample)

    return unique_samples.shape[0] / n_samples <= max_unique_fraction
