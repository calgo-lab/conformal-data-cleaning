from __future__ import annotations

from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd

from ._base import BaseCleaner
from conformal_inference.sklearn_forest import (
    ConformalRandomForestClassifier,
    ConformalRandomForestQuantileRegressor,
)

logger = getLogger(__name__)


class ConformalForestCleaner(BaseCleaner):
    def __init__(self, confidence_level: float, seed: int | None = None) -> None:
        super().__init__(seed=seed)

        if confidence_level <= 0 or confidence_level >= 1:
            msg = "Argument 'confidence_level' is not valid! Need to be: 0 <= confidence_level <= 1"
            raise ValueError(msg)

        self.confidence_level_ = confidence_level

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> ConformalForestCleaner:
        self.predictors_: dict[Any, ConformalRandomForestClassifier | ConformalRandomForestQuantileRegressor] = {}

        for column in self.target_columns_:
            msg = f"Fit {self.__class__.__name__} for column '{column}'."
            logger.debug(msg)

            # Categorical column => Classification task
            if column in self._categorical_columns:
                logger.debug("Column is categorical according to our heuristics.")
                self.predictors_[column] = ConformalRandomForestClassifier(conditional=True, fit=True, predictor_params=kwargs.get("sk_params", {}))

            # Numerical column => Regression task
            elif column in self._numerical_columns:
                logger.debug("Column is numerical according to our heuristics.")
                self.predictors_[column] = ConformalRandomForestQuantileRegressor(predictor_params=kwargs.get("sk_params", {}))

            else:
                msg = f"Column '{column}' is not categorical or numerical according to our heuristics."
                raise ValueError(msg)

            self.predictors_[column].fit(X=data[[col for col in data.columns if col != column]], y=data[column], fit_params=kwargs.get("sk_fit_params", {}))

        return self

    def _remove_outliers_method(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        outliers = {}
        self._outlier_predictions = {}

        # NOTE: this is stored for in depth evaluations not because it's necessary for the cleaning
        self.conformalized_predictions_ = {}

        for column in self.target_columns_:
            msg = f"Remove outliers for column '{column}'."
            logger.debug(msg)

            prediction_set_or_quantiles, y_prediction = self.predictors_[column].predict(
                data[[col for col in data.columns if col != column]], confidence_level=self.confidence_level_
            )
            self.conformalized_predictions_[column] = prediction_set_or_quantiles

            # outlier if value is not in prediction set or prediction set is empty
            # then only if pre
            if column in self._categorical_columns:
                outliers[column] = [
                    False
                    # to calculate the "size" of a prediction set, we need to count non-null values
                    if np.count_nonzero(~pd.isna(prediction_set)) == 0
                    else value not in prediction_set
                    for value, prediction_set in zip(data[column], prediction_set_or_quantiles)
                ]

            # outlier if value is not in prediction interval, i.e., smaller than lower (index 0)
            # or larger than upper (index 1) quantile
            elif column in self._numerical_columns:
                outliers[column] = (data[column] < prediction_set_or_quantiles[:, 0]) | (data[column] > prediction_set_or_quantiles[:, 1])

            else:
                msg = f"Column '{column}' is neither categorical nor numerical. This should be checked when fitting and causes very likely downstream issues."
                logger.warning(msg)

            self._outlier_predictions[column] = y_prediction[outliers[column]]

        # calculate all outliers THEN remove them
        # avoid to introduce missing values that need to be handled by the predictors for prediction
        for column in self.target_columns_:
            data.loc[outliers[column], column] = np.nan

        return data

    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        for column in self.target_columns_:
            msg = f"Impute missing values for column '{column}'."
            logger.debug(msg)

            missing_mask = data[column].isna()
            if missing_mask.any():
                _, y_prediction = self.predictors_[column].predict(
                    data[missing_mask][[col for col in data.columns if col != column]], confidence_level=self.confidence_level_
                )
                data.loc[missing_mask, column] = y_prediction

        return data
