from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
