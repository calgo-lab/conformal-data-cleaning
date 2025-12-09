from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.utils.multiclass import type_of_target

if TYPE_CHECKING:
    import pandas as pd
    from numpy.random import Generator

from enum import Enum
from typing import TYPE_CHECKING


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    def __str__(self):
        return self.value


def is_categorical(column: pd.Series, n_samples: int = 1000, max_unique_fraction: float = 0.2, random_generator: Generator | None = None) -> bool:
    """Check if `column` type is categorical.

    A heuristic to check whether a `column` is categorical:
    a column is considered categorical (as opposed to a plain text column)
    if the relative cardinality is `max_unique_fraction` or less.
    Thanks to:
        https://github.com/awslabs/datawig/blob/f641342d05e95485ed88503d3efd9c3cca3eb7ab/datawig/simple_imputer.py#L147

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

    column = np.array(column)  # ty:ignore[invalid-assignment]
    n_samples = min(n_samples, len(column))
    values, counts = np.unique(column, return_counts=True)
    sample = random_generator.choice(a=values, p=counts / counts.sum(), size=n_samples)
    unique_samples = np.unique(sample)

    return unique_samples.shape[0] / n_samples <= max_unique_fraction




def guess_task_type(column: pd.Series) -> TaskType:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*number of unique classes is greater than 50%.*",
        )

        if type_of_target(column) in ["multiclass", "binary"]:
            if is_categorical(column):
                return TaskType.CLASSIFICATION

            if is_numeric_dtype(column):
                return TaskType.REGRESSION

        if is_numeric_dtype(column) and type_of_target(column) == "continuous":
            return TaskType.REGRESSION

    return TaskType.NOT_SUPPORTED


def split_columns_into_categorical_and_numerical(data: pd.DataFrame) -> tuple[list, list]:
    categorical_column_names = []
    numerical_column_names = []

    for column in data.columns:
        if guess_task_type(data[column]) == TaskType.CLASSIFICATION:
            categorical_column_names.append(column)

        elif guess_task_type(data[column]) == TaskType.REGRESSION:
            numerical_column_names.append(column)

    if len(data.columns) != (len(categorical_column_names) + len(numerical_column_names)):
        msg = (
            f"There are {len(data.columns)} columns but found {len(categorical_column_names)} categorical "
            + f"and {len(numerical_column_names)} numerical columns. "
            + f"Missing: {list(set(data.columns) - set(numerical_column_names + categorical_column_names))}"
        )
        raise ValueError(msg)

    return categorical_column_names, numerical_column_names