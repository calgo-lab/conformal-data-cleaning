from __future__ import annotations

from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._base import BaseCleaner
from conformal_data_cleaning.config import N_JOBS

logger = getLogger(__name__)


class ForestCleaner(BaseCleaner):
    def __init__(self, confidence_level: float, seed: int | None = None) -> None:
        super().__init__(seed=seed)

        if confidence_level <= 0 or confidence_level >= 1:
            msg = "Argument 'confidence_level' is not valid! Need to be: 0 <= confidence_level <= 1"
            raise ValueError(msg)

        self.confidence_level_ = confidence_level

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> ForestCleaner:
        self.predictors_: dict[Any, Pipeline] = {}

        for column in self.target_columns_:
            msg = f"Fit {self.__class__.__name__} for column '{column}'."
            logger.debug(msg)

            feature_transformation = ColumnTransformer(
                transformers=[
                    ("categorical_features", OneHotEncoder(handle_unknown="ignore"), [col for col in self._categorical_columns if col != column]),
                    ("scaled_numeric", StandardScaler(), [col for col in self._numerical_columns if col != column]),
                ],
                sparse_threshold=0,
            )

            # Categorical column => Classification task
            if column in self._categorical_columns:
                logger.debug("Column is categorical according to our heuristics.")
                predictor = RandomForestClassifier(random_state=self._seed, n_jobs=N_JOBS)

            # Numerical column => Regression task
            elif column in self._numerical_columns:
                logger.debug("Column is numerical according to our heuristics.")
                predictor = RandomForestQuantileRegressor(default_quantiles=0.5, random_state=self._seed, n_jobs=N_JOBS)

            else:
                msg = f"Column '{column}' is not categorical or numerical according to our heuristics."
                raise ValueError(msg)

            self.predictors_[column] = Pipeline([("preprocess", feature_transformation), ("predictor", predictor)]).fit(
                X=data[[col for col in data.columns if col != column]],
                y=data[column],
            )

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

            y_prediction = self.predictors_[column].predict(data[[col for col in data.columns if col != column]])

            # Include all classes in prediction set until cumulative sum of probabilities exceeds confidence level
            if column in self._categorical_columns:
                probabilities = self.predictors_[column].predict_proba(data[[col for col in data.columns if col != column]])

                order = np.argsort(-probabilities, axis=1)
                sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
                cumsum_of_probabilities = np.cumsum(sorted_probabilities, axis=1)
                cutoff = (cumsum_of_probabilities >= self.confidence_level_).argmax(axis=1)
                conformalized_predictions = np.full_like(probabilities, np.nan, dtype=y_prediction.dtype)

                for i in range(probabilities.shape[0]):
                    keep = order[i, : cutoff[i] + 1]  # column indices to add to prediction set
                    conformalized_predictions[i, keep] = self.predictors_[column].classes_[keep]

                outliers[column] = [
                    False
                    # to calculate the "size" of a prediction set, we need to count non-null values
                    if np.count_nonzero(~pd.isna(prediction_set)) == 0
                    else value not in prediction_set
                    for value, prediction_set in zip(data[column], conformalized_predictions)
                ]

            # Confidence level defines width of prediction interval
            elif column in self._numerical_columns:
                lower_quantile = (1 - self.confidence_level_) / 2
                upper_quantile = 1 - (1 - self.confidence_level_) / 2

                conformalized_predictions = self.predictors_[column].predict(
                    data[[col for col in data.columns if col != column]],
                    quantiles=[lower_quantile, upper_quantile],
                )

                outliers[column] = (data[column] < conformalized_predictions[:, 0]) | (data[column] > conformalized_predictions[:, 1])

            else:
                msg = f"Column '{column}' is neither categorical nor numerical. This should be checked when fitting and causes very likely downstream issues."
                logger.warning(msg)

            self.conformalized_predictions_[column] = conformalized_predictions
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
                y_prediction = self.predictors_[column].predict(data[missing_mask][[col for col in data.columns if col != column]])
                data.loc[missing_mask, column] = y_prediction

        return data
