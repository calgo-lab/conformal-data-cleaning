# Interface for the error demo
# 01/20/2026
# Nick Chandler

from cleaner import ConformalForestCleaner
from cleaner._base import BaseCleaner
import pandas as pd


def fit_and_get_cleaner(cleaner: str, confidence_level: float, train_df: pd.DataFrame, seed: int, n_jobs: int = -1) -> BaseCleaner:
    cleaner = ConformalForestCleaner(confidence_level=confidence_level, seed=seed).fit(
                data=train_df, sk_params={"random_state": seed, "n_jobs": n_jobs}
            )

    return cleaner



