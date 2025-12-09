from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection._split import train_test_split

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def calculate_q_hat(nonconformity_scores: NDArray, confidence_level: float) -> float | None:
    nonconformity_scores = np.asarray(nonconformity_scores)
    n = len(nonconformity_scores)

    if n == 0:
        return None

    adjusted_quantile = confidence_level * (1 + 1 / n)

    # clip `adjusted_quantile` to make sure it is in 0 <= adjusted_quantile <= 1
    adjusted_quantile = 1 if adjusted_quantile > 1 else max(adjusted_quantile, 0)

    return float(np.quantile(a=nonconformity_scores, q=adjusted_quantile, method="higher"))


def check_in_range(number: float, name: str, valid_range: tuple[int, int] = (0, 1)) -> None:
    if number < valid_range[0] or number > valid_range[1]:
        msg = f"Variable '{name}' is not valid! Need to be: 0 <= {name} <= 1"
        raise ValueError(msg)


def check_and_split_X_y(
    X: ArrayLike,
    y: ArrayLike,
    calibration_size: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    check_in_range(calibration_size, "calibration_size")

    X_training, X_calibration, y_training, y_calibration = train_test_split(X, y, test_size=calibration_size)

    return (
        X_training,
        X_calibration,
        y_training,
        y_calibration,
    )
