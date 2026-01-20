from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any

import pandas as pd
from sklearn.utils import Tags, TargetTags
from sklearn.utils.validation import check_is_fitted

from conformal_data_cleaning import seed_and_get_generator
from data import split_columns_into_categorical_and_numerical

logger = getLogger(__name__)


class CleanerError(Exception):
    """Exception raised for errors in Cleaners."""


class BaseCleaner(ABC):
    _outlier_predictions: dict
    conformalized_predictions_: dict

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._random_generator = seed_and_get_generator(self._seed)

    def _guess_dtypes(self, data: pd.DataFrame) -> None:
        self._categorical_columns, self._numerical_columns = split_columns_into_categorical_and_numerical(data)

    def __sklearn_tags__(self) -> Tags:
        return Tags(estimator_type=None, target_tags=TargetTags(required=False))

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "predictors_") and hasattr(self, "target_columns_")

    def fit(self, data: pd.DataFrame, target_columns: list | None = None, **kwargs: dict[str, Any]) -> BaseCleaner:
        if target_columns is None:
            target_columns = data.columns.to_list()

        if type(target_columns) is not list:
            msg = f"Parameter 'target_column' need to be of type list but is '{type(target_columns)}'"
            raise CleanerError(
                msg,
            )

        if any(column not in data.columns for column in target_columns):
            msg = f"All target columns ('{target_columns}') must be in: {', '.join(data.columns)}"
            raise CleanerError(msg)

        self.target_columns_ = target_columns

        self._guess_dtypes(data)
        return self._fit_method(data=data.copy(), **kwargs)

    def remove_outliers(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        check_is_fitted(self)

        # Reset potential previous runs
        if hasattr(self, "_outlier_predictions"):
            delattr(self, "_outlier_predictions")

        missing_mask = data[self.target_columns_].isna()
        data_without_outliers = self._remove_outliers_method(data=data.copy(), **kwargs)

        missing_mask_outliers_removed = data_without_outliers[self.target_columns_].isna()
        outlier_mask = missing_mask_outliers_removed & ~missing_mask

        return data_without_outliers, outlier_mask

    def impute(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        check_is_fitted(self)

        missing_mask = data[self.target_columns_].isna()
        imputed_data = self._impute_method(data=data.copy(), **kwargs)

        return imputed_data, missing_mask

    def transform(
        self,
        data: pd.DataFrame,
        separate_steps: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_without_outliers, outlier_mask = self.remove_outliers(data, **kwargs)

        if not separate_steps:
            for column in self.target_columns_:
                mask = outlier_mask.loc[:, column]
                data_without_outliers.loc[mask, column] = self._outlier_predictions[column]

        else:
            logger.debug("Do not reuse intermediate calculations of correct values. This is equivalent to call 'remove_outliers' and 'impute' in sequence.")

        cleaned_data, imputed_mask = self.impute(data_without_outliers, **kwargs)
        cleaned_mask = imputed_mask | outlier_mask

        return cleaned_data, cleaned_mask

    def multiple_transform(
        self,
        data: pd.DataFrame,
        separate_steps: bool = False,
        m: int = 10,
        **kwargs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(m) is not int or m < 2:
            msg = f"'m' has to be >=2 but it is {m}"
            raise ValueError(msg)

        multiple_cleaned_data = data
        multiple_cleaned_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        for _ in range(1, m):
            multiple_cleaned_data, cleaned_mask = self.transform(data=multiple_cleaned_data, separate_steps=separate_steps, kwargs=kwargs)
            multiple_cleaned_mask = multiple_cleaned_mask | cleaned_mask

        return multiple_cleaned_data, multiple_cleaned_mask

    @abstractmethod
    def _fit_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> BaseCleaner:
        pass

    @abstractmethod
    def _remove_outliers_method(
        self,
        data: pd.DataFrame,
        **kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def _impute_method(self, data: pd.DataFrame, **kwargs: dict[str, Any]) -> pd.DataFrame:
        pass
