from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from conformal_data_cleaning.conformal_inference.autogluon import ConformalAutoGluonClassifier, ConformalQuantileAutoGluonRegressor

from ._base import BaseCleaner, CleanerError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = getLogger(__name__)


class ConformalAutoGluonCleaner(BaseCleaner):
    def __init__(
        self,
        confidence_level: float,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        if confidence_level <= 0 or confidence_level >= 1:
            msg = "Argument 'confidence_level' is not valid! Need to be: 0 <= confidence_level <= 1"
            raise ValueError(msg)

        self._confidence_level = confidence_level

    def _make_prediction(self, data: pd.DataFrame, column: Any) -> tuple[NDArray, NDArray]:
        predictor = self.predictors_[column]
        if type(predictor) is ConformalAutoGluonClassifier:
            return predictor.predict(data, confidence_level=self._confidence_level)

        if type(predictor) is ConformalQuantileAutoGluonRegressor:
            return predictor.predict(data)

        msg = f"Predictor for column '{column}' is of wrong type."
        raise ValueError(msg)

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> ConformalAutoGluonCleaner:
        self.predictors_: dict[Any, ConformalAutoGluonClassifier | ConformalQuantileAutoGluonRegressor] = {}

        path_prefix = Path(kwargs.get("ci_ag_predictor_params", {}).pop("path_prefix", "AutogluonModels"))

        for index, column in enumerate(self.target_columns_):
            msg = f"Start fitting predictor #{index + 1} of {len(self.target_columns_)}"
            logger.info(msg)

            # prepare the path where to store the models
            ci_ag_predictor_params = kwargs.get("ci_ag_predictor_params", {})
            ci_ag_predictor_params["path"] = path_prefix / str(column)
            kwargs["ci_ag_predictor_params"] = ci_ag_predictor_params

            # Construction of ConformalAutoGluon predictor differs for classification/regression
            # Categorical column => Classification task
            if column in self._categorical_columns:
                is_multi_class = len(data[column].unique()) > 2

                predictor_params = kwargs.get("ci_ag_predictor_params", {})
                predictor_params["problem_type"] = "multiclass" if is_multi_class else "binary"
                predictor_params["eval_metric"] = "f1_macro" if is_multi_class else "f1"

                self.predictors_[column] = ConformalAutoGluonClassifier(
                    target_column=column,
                    predictor_params=predictor_params,
                )

            # Numerical column => Regression task
            elif column in self._numerical_columns:
                predictor_params = kwargs.get("ci_ag_predictor_params", {})
                predictor_params["eval_metric"] = "pinball_loss"

                self.predictors_[column] = ConformalQuantileAutoGluonRegressor(
                    target_column=column,
                    confidence_level=self._confidence_level,
                    predictor_params=predictor_params,
                )

            else:
                msg = f"Column '{column}' is not categorical or numerical."
                raise CleanerError(msg)

            fit_params = kwargs.get("ci_ag_fit_params", {})

            self.predictors_[column].fit(
                X=data,
                fit_params=fit_params,
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
        self._prediction_sets = {}

        for column in self.target_columns_:
            msg = f"Remove outliers for column '{column}'."
            logger.debug(msg)

            conformalized_prediction, y_prediction = self._make_prediction(data=data, column=column)
            self._prediction_sets[column] = conformalized_prediction

            # outlier if value is not in prediction set or prediction set is empty
            if column in self._categorical_columns:
                outliers[column] = [
                    True  ### Modified -- New code says: If prediction set is empty -> have outlier
                    # to calculate the "size" of a prediction set, we need to count non-null values
                    if np.count_nonzero(~pd.isna(prediction_set)) == 0
                    else value not in prediction_set
                    for value, prediction_set in zip(data[column], conformalized_prediction)
                ]

            # outlier if value is not in prediction interval, i.e., smaller than lower (index 0)
            # or larger than upper (index 1) quantile
            elif column in self._numerical_columns:
                outliers[column] = (data[column] <= conformalized_prediction[:, 0]) | (data[column] >= conformalized_prediction[:, 1])

            else:
                msg = f"Column '{column}' is neither categorical nor numerical. This should be checked when fitting and causes very likely downstream issues."
                logger.warning(msg)

            self._outlier_predictions[column] = y_prediction

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
                _, y_prediction = self._make_prediction(data=data[missing_mask], column=column)
                preds = pd.Series(y_prediction, index=data[missing_mask].index)
                data.loc[missing_mask, column] = preds

        return data
