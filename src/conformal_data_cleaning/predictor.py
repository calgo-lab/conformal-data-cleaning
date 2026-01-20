from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


class Predictor(Protocol):
    classes_: pd.DataFrame

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Predictor: ...
    def predict(self, X: pd.DataFrame) -> pd.Series: ...
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...
