from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from conformal_inference.autogluon import (
    ConformalAutoGluonClassifier,
    ConformalQuantileAutoGluonRegressor,
)

from ._base import BaseCleaner

logger = getLogger(__name__)


class ConformalAutoGluonCleaner(BaseCleaner):
    def __init__(
        self,
        confidence_level: float,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("Argument 'confidence_level' is not valid! Need to be: 0 <= confidence_level <= 1")

        self._confidence_level = confidence_level

    def __del__(self) -> None:
        for column in self.target_columns_:
            self.predictors_[column]._predictor.unpersist_models()
            del self.predictors_[column]._predictor, self.predictors_[column]

        del self.predictors_

    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> ConformalAutoGluonCleaner:
        self.predictors_: dict[Any, Union[ConformalAutoGluonClassifier, ConformalQuantileAutoGluonRegressor]] = {}

        path_prefix = Path(kwargs.get("ci_ag_predictor_params", {}).pop("path_prefix", "AutogluonModels"))

        for index, column in enumerate(self.target_columns_):
            logger.info(f"Start fitting predictor #{index + 1} of {len(self.target_columns_)}")

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
                raise Exception(f"Column '{column}' is not categorical or numerical ...")

            fit_params = kwargs.get("ci_ag_fit_params", {})

            # Refit/Disk space settings
            #fit_params["refit_full"] = True
            #fit_params["keep_only_best"] = True
            #fit_params["set_best_to_refit_full"] = True
            #fit_params["fit_weighted_ensemble"] = False

            # Bagging/Stacking settings
            #fit_params["auto_stack"] = False
            #fit_params["num_bag_folds"] = 0
            #fit_params["num_stack_levels"] = 0

            # HPO settings
            # hyperparameter_tune_kwargs = fit_params.pop("hyperparameter_tune_kwargs", {})
            # hyperparameter_tune_kwargs["searcher"] = "random"
            # hyperparameter_tune_kwargs["scheduler"] = "local"
            # hyperparameter_tune_kwargs["num_trials"] = hyperparameter_tune_kwargs.get("num_trials", 10)
            # fit_params["hyperparameter_tune_kwargs"] = hyperparameter_tune_kwargs


            self.predictors_[column].fit(
                X=data,
                fit_params=fit_params,
            )

            # save memory during training
            #self.predictors_[column]._predictor.unpersist_models()

        return self

    # Caution: spaghetti code w/o documentation!!!
    def _remove_outliers_method(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        outliers = {}
        _outlier_predictions = {}
        empty_pred_sets_are_inliers: bool = kwargs.get("empty_pred_sets_are_inliers", False)  # type:ignore

        prediction_sets = {}
        for column in self.target_columns_:
            # if empty prediction sets should be treated as inliers,
            # then empty prediction sets are OK.

            y_hat, other_thing = self.predictors_[column].predict(data, confidence_level=self._confidence_level, sorted=True, allow_empty_set=True)  # Mystery returns! is it a set or quantiles?
            prediction_sets[column] = y_hat

            # outlier if value is not in prediction set except `empty_pred_sets_are_inliers`
            # then only if pre
            if column in self._categorical_columns:
                outliers[column] = [
                    False
                    if empty_pred_sets_are_inliers
                    # to calculate the "size" of a prediction set, we need to count non-null values
                    & (np.count_nonzero(~pd.isna(prediction_set)) == 0)
                    else value not in prediction_set
                    for value, prediction_set in zip(data[column], y_hat)
                ]
                _outlier_predictions[column] = y_hat[outliers[column], 0]

            # outlier if value is not in prediction interval, i.e., smaller than lower (index 0)
            # or larger than upper (index 1) quantile
            elif column in self._numerical_columns:
                outliers[column] = (data[column] <= y_hat[:, 0]) | (data[column] >= y_hat[:, 1])
                _outlier_predictions[column] = y_hat[outliers[column], 1]

            else:
                logger.warning("This should be checked before fit process starts..")

        # calculate all outliers THEN remove them
        # avoid to introduce missing values that need to be handled by the predictors for prediction
        for column in self.target_columns_:
            data.loc[outliers[column], column] = np.nan

        self._outlier_predictions = _outlier_predictions

        return data, prediction_sets

    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        for column in self.target_columns_:
            missing_mask = data[column].isna()
            if missing_mask.any():
                # predictor returns a tuple; unpack it
                predictions_tuple = self.predictors_[column].predict(
                    data[missing_mask],
                    confidence_level=self._confidence_level,
                    sorted=True,
                    allow_empty_set=False
                )

                if column in self._categorical_columns:
                    # unpack: prediction_sets, point_estimates = tuple
                    prediction_sets, _ = predictions_tuple
                    # fill with the top predicted label
                    data.loc[missing_mask, column] = [next(iter(s)) for s in prediction_sets]

                elif column in self._numerical_columns:
                    # unpack: intervals, point_estimates = tuple
                    _, point_estimates = predictions_tuple
                    data.loc[missing_mask, column] = point_estimates

                else:
                    logger.warning("This should be checked before fit process starts..")

        return data