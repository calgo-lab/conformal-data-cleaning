from __future__ import annotations

import random
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tab_err import ErrorMechanism
from tab_err.api.high_level import create_errors
from tab_err.error_mechanism import EAR, ECAR, ENAR
from tab_err.error_type import MissingValue, Mistype

from cleaner import ConformalForestCleaner, ForestCleaner, PrecisionForestCleaner
from cleaner._base import BaseCleaner
from config import DATA_PATH, N_JOBS, RESULTS_PATH
from data import TaskType, guess_task_type, split_columns_into_categorical_and_numerical

target_column_name = "target"
logger = getLogger(__name__)


class Experiment:
    def __init__(self, name: str, dataset_id: str, train_size: float = 0.8, data_path: Path = DATA_PATH, seed: int | None = None):
        self.name: str = name
        self.dataset_id: str = dataset_id
        self._train_size: float = train_size
        self._seed: int | None = seed
        self._data: pd.DataFrame = pd.read_csv(data_path / f"{self.dataset_id}.csv")
        self._categorical_columns, self._numerical_columns = split_columns_into_categorical_and_numerical(self._data.drop(columns=target_column_name))

        train, test = train_test_split(self._data, train_size=train_size, random_state=self._seed)
        self.task_type: TaskType = guess_task_type(self._data[target_column_name])

        # NOTE: We copy here to make sure we decouple them
        self.X_train: pd.DataFrame = train.drop(columns=target_column_name).copy()
        self.y_train: pd.Series = train[target_column_name].copy()

        self.X_test: pd.DataFrame = test.drop(columns=target_column_name).copy()
        self.y_test: pd.Series = test[target_column_name].copy()

        self.X_perturbed: pd.DataFrame | None = None
        self.error_mask: pd.DataFrame | None = None
        self.error_rate: float | None = None
        self.error_mechanism: type[ErrorMechanism] | None = None

        self.X_cleaned: pd.DataFrame | None = None
        self.cleaned_mask: pd.DataFrame | None = None
        self.X_multiple_cleaned: pd.DataFrame | None = None
        self.multiple_cleaned_mask: pd.DataFrame | None = None

        self._model: Pipeline | None = None
        self._cleaner: BaseCleaner | None = None
        self.confidence_level: float | None = None

    def get_dataset_statistics(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "num_rows": self._data.shape[0],
            "num_cols": self._data.shape[1],
            "num_cells": self._data.size,
            "num_categorical": len(self._categorical_columns),
            "num_numerical": len(self._numerical_columns),
            "task_type": self.task_type.value,
        }

    def fit_and_get_baseline_model(self) -> Pipeline:
        if self._model is None:
            feature_transformation = ColumnTransformer(
                transformers=[
                    ("categorical_features", OneHotEncoder(handle_unknown="ignore"), self._categorical_columns),
                    ("scaled_numeric", StandardScaler(), self._numerical_columns),
                ],
                sparse_threshold=0,
            )

            if self.task_type == TaskType.CLASSIFICATION:
                msg = f"Fitting RandomForestClassifier for dataset {self.dataset_id}"
                logger.debug(msg)

                predictor = RandomForestClassifier(random_state=self._seed, n_jobs=N_JOBS)

            elif self.task_type == TaskType.REGRESSION:
                msg = f"Fitting RandomForestRegressor for dataset {self.dataset_id}"
                logger.debug(msg)

                predictor = RandomForestRegressor(random_state=self._seed, n_jobs=N_JOBS)

            else:
                msg = "Only Regression or Classification tasks are supported."
                raise ValueError(msg)

            self._model = Pipeline([("preprocess", feature_transformation), ("predictor", predictor)]).fit(X=self.X_train, y=self.y_train)
        return self._model

    def perturb_and_get_perturbed_test_data_and_error_mask(self, error_rate: float, error_mechanism: type[ErrorMechanism]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame or type(self.error_mask) is not pd.DataFrame:
            if isinstance(error_mechanism, type) and not issubclass(error_mechanism, ErrorMechanism):
                msg = "'error_mechanism' must be subclasses of ErrorMechanism"
                raise ValueError(msg)

            self.error_rate = error_rate
            self.error_mechanism = error_mechanism

            if self.error_rate > 0:
                self.X_perturbed, self.error_mask = create_errors(
                    data=self.X_test,
                    error_rate=self.error_rate,
                    error_types_to_exclude=[Mistype(), MissingValue()],
                    error_mechanisms_to_exclude=[mechanism() for mechanism in [ECAR, EAR, ENAR] if mechanism != self.error_mechanism],
                    seed=self._seed,
                )

            else:
                self.X_perturbed = self.X_test.copy()
                self.error_mask = pd.DataFrame(data=False, index=self.X_perturbed.index, columns=self.X_perturbed.columns)

        return self.X_perturbed, self.error_mask

    def fit_and_get_cleaner(self, cleaner: str, confidence_level: float) -> BaseCleaner:
        if self._cleaner is None:
            self.confidence_level = confidence_level

            if cleaner.lower() == ConformalForestCleaner.__name__.lower():
                self._cleaner = ConformalForestCleaner(confidence_level=self.confidence_level, seed=self._seed).fit(
                    data=self.X_train, sk_params={"random_state": self._seed, "n_jobs": N_JOBS}
                )

            elif cleaner.lower() == ForestCleaner.__name__.lower():
                self._cleaner = ForestCleaner(confidence_level=self.confidence_level, seed=self._seed).fit(data=self.X_train)

            elif cleaner.lower() == PrecisionForestCleaner.__name__.lower():
                self._cleaner = PrecisionForestCleaner(confidence_level=self.confidence_level, seed=self._seed).fit(data=self.X_train)

            else:
                msg = f"Currently only {ForestCleaner.__name__} and {ConformalForestCleaner.__name__} are supported cleaners."
                raise ValueError(msg)

        return self._cleaner

    def clean_and_get_cleaned_test_data_and_cleaned_mask(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data_and_error_mask' first."
            raise RuntimeError(msg)

        if not isinstance(self._cleaner, BaseCleaner):
            msg = "Run 'fit_and_get_cleaner' first."
            raise TypeError(msg)

        if type(self.X_cleaned) is not pd.DataFrame or type(self.cleaned_mask) is not pd.DataFrame:
            self.X_cleaned, self.cleaned_mask = self._cleaner.transform(self.X_perturbed)

        return self.X_cleaned, self.cleaned_mask

    def multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if type(self.X_perturbed) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data_and_error_mask' first."
            raise RuntimeError(msg)

        if not isinstance(self._cleaner, BaseCleaner):
            msg = "Run 'fit_and_get_cleaner' first."
            raise TypeError(msg)

        if type(self.X_multiple_cleaned) is not pd.DataFrame or type(self.multiple_cleaned_mask) is not pd.DataFrame:
            self.X_multiple_cleaned, self.multiple_cleaned_mask = self._cleaner.multiple_transform(self.X_perturbed)

        return self.X_multiple_cleaned, self.multiple_cleaned_mask

    def calculate_and_get_metrics(self) -> dict:
        if type(self.X_perturbed) is not pd.DataFrame or type(self.error_mask) is not pd.DataFrame:
            msg = "Run 'perturb_and_get_perturbed_test_data' first."
            raise RuntimeError(msg)

        if type(self.X_cleaned) is not pd.DataFrame or type(self.cleaned_mask) is not pd.DataFrame:
            msg = "Run 'clean_and_get_cleaned_test_data_and_cleaned_mask' first."
            raise RuntimeError(msg)

        if type(self.X_multiple_cleaned) is not pd.DataFrame or type(self.multiple_cleaned_mask) is not pd.DataFrame:
            msg = "Run 'multiple_clean_and_get_multiple_cleaned_test_data_and_multiple_cleaned_mask' first."
            raise RuntimeError(msg)

        def relative_improvement(clean: float, perturbed: float, higher_is_better: bool) -> float:
            if higher_is_better:
                return (clean - perturbed) / perturbed

            return (perturbed - clean) / perturbed

        model = self.fit_and_get_baseline_model()
        y_hat = model.predict(self.X_test)
        y_hat_perturbed = model.predict(self.X_perturbed)
        y_hat_cleaned = model.predict(self.X_cleaned)
        y_hat_multiple_cleaned = model.predict(self.X_multiple_cleaned)

        number_of_errors = self.error_mask.sum().sum()
        cleaned_mask = self.X_cleaned != self.X_perturbed
        multiple_cleaned_mask = self.X_multiple_cleaned != self.X_perturbed

        error_detection_tpr = (self.error_mask & cleaned_mask).sum().sum() / number_of_errors
        error_detection_fpr = (~self.error_mask & cleaned_mask).sum().sum() / (~self.error_mask).sum().sum()
        multiple_error_detection_tpr = (self.error_mask & multiple_cleaned_mask).sum().sum() / number_of_errors
        multiple_error_detection_fpr = (~self.error_mask & multiple_cleaned_mask).sum().sum() / (~self.error_mask).sum().sum()
        metrics = {
            "error_detection_tpr": error_detection_tpr,
            "error_detection_fpr": error_detection_fpr,
            "multiple_error_detection_tpr": multiple_error_detection_tpr,
            "multiple_error_detection_fpr": multiple_error_detection_fpr,
        }

        if self.task_type is TaskType.CLASSIFICATION:
            metrics.update(
                {
                    "orig_1": f1_score(self.y_test, y_hat, average="micro"),
                    "orig_2": f1_score(self.y_test, y_hat, average="macro"),
                    "orig_3": f1_score(self.y_test, y_hat, average="weighted"),
                    "perturbed_1": f1_score(self.y_test, y_hat_perturbed, average="micro"),
                    "perturbed_2": f1_score(self.y_test, y_hat_perturbed, average="macro"),
                    "perturbed_3": f1_score(self.y_test, y_hat_perturbed, average="weighted"),
                    "cleaned_1": f1_score(self.y_test, y_hat_cleaned, average="micro"),
                    "cleaned_2": f1_score(self.y_test, y_hat_cleaned, average="macro"),
                    "cleaned_3": f1_score(self.y_test, y_hat_cleaned, average="weighted"),
                    "multiple_cleaned_1": f1_score(self.y_test, y_hat_multiple_cleaned, average="micro"),
                    "multiple_cleaned_2": f1_score(self.y_test, y_hat_multiple_cleaned, average="macro"),
                    "multiple_cleaned_3": f1_score(self.y_test, y_hat_multiple_cleaned, average="weighted"),
                }
            )
            metrics.update(
                {
                    "improvement_1": relative_improvement(clean=metrics["cleaned_1"], perturbed=metrics["perturbed_1"], higher_is_better=True),
                    "improvement_2": relative_improvement(clean=metrics["cleaned_2"], perturbed=metrics["perturbed_2"], higher_is_better=True),
                    "improvement_3": relative_improvement(clean=metrics["cleaned_3"], perturbed=metrics["perturbed_3"], higher_is_better=True),
                    "multiple_improvement_1": relative_improvement(
                        clean=metrics["multiple_cleaned_1"], perturbed=metrics["perturbed_1"], higher_is_better=True
                    ),
                    "multiple_improvement_2": relative_improvement(
                        clean=metrics["multiple_cleaned_2"], perturbed=metrics["perturbed_2"], higher_is_better=True
                    ),
                    "multiple_improvement_3": relative_improvement(
                        clean=metrics["multiple_cleaned_3"], perturbed=metrics["perturbed_3"], higher_is_better=True
                    ),
                }
            )

        elif self.task_type is TaskType.REGRESSION:
            metrics.update(
                {
                    "orig_1": r2_score(self.y_test, y_hat),
                    "orig_2": mean_absolute_error(self.y_test, y_hat),
                    "orig_3": mean_squared_error(self.y_test, y_hat),
                    "perturbed_1": r2_score(self.y_test, y_hat_perturbed),
                    "perturbed_2": mean_absolute_error(self.y_test, y_hat_perturbed),
                    "perturbed_3": mean_squared_error(self.y_test, y_hat_perturbed),
                    "cleaned_1": r2_score(self.y_test, y_hat_cleaned),
                    "cleaned_2": mean_absolute_error(self.y_test, y_hat_cleaned),
                    "cleaned_3": mean_squared_error(self.y_test, y_hat_cleaned),
                    "multiple_cleaned_1": r2_score(self.y_test, y_hat_multiple_cleaned),
                    "multiple_cleaned_2": mean_absolute_error(self.y_test, y_hat_multiple_cleaned),
                    "multiple_cleaned_3": mean_squared_error(self.y_test, y_hat_multiple_cleaned),
                }
            )
            metrics.update(
                {
                    "improvement_1": relative_improvement(clean=metrics["cleaned_1"], perturbed=metrics["perturbed_1"], higher_is_better=True),
                    "improvement_2": relative_improvement(clean=metrics["cleaned_2"], perturbed=metrics["perturbed_2"], higher_is_better=False),
                    "improvement_3": relative_improvement(clean=metrics["cleaned_3"], perturbed=metrics["perturbed_3"], higher_is_better=False),
                    "multiple_improvement_1": relative_improvement(
                        clean=metrics["multiple_cleaned_1"], perturbed=metrics["perturbed_1"], higher_is_better=True
                    ),
                    "multiple_improvement_2": relative_improvement(
                        clean=metrics["multiple_cleaned_2"], perturbed=metrics["perturbed_2"], higher_is_better=False
                    ),
                    "multiple_improvement_3": relative_improvement(
                        clean=metrics["multiple_cleaned_3"], perturbed=metrics["perturbed_3"], higher_is_better=False
                    ),
                }
            )

        else:
            msg = "Currently, only Classification or Regression Experiments are supported."
            raise RuntimeError(msg)

        return metrics

    def calculate_and_get_prediction_set_metrics(self) -> dict:
        if not isinstance(self._cleaner, BaseCleaner):
            msg = "Run 'fit_and_get_cleaner' first."
            raise TypeError(msg)

        prediction_set_metrics = {}

        for column_name, prediction_sets in self._cleaner.conformalized_predictions_.items():
            if column_name in self._categorical_columns:
                column_type = "categorical"
                cardinality = len(self.X_train[column_name].unique())
                true_value_in_prediction_set = np.any(prediction_sets == self.X_test[column_name].to_numpy()[:, np.newaxis], axis=1)
                coverage = true_value_in_prediction_set.mean()
                average_set_size = (~pd.DataFrame(prediction_sets).isna()).sum(axis=1).mean()
                relative_average_set_size = average_set_size / cardinality
                empty_set_fraction = ((~pd.DataFrame(prediction_sets).isna()).sum(axis=1) == 0).mean()

            elif column_name in self._numerical_columns:
                column_type = "numerical"
                cardinality = self.X_train[column_name].max() - self.X_train[column_name].min()
                true_value_in_prediction_range = (self.X_test[column_name].to_numpy() >= prediction_sets[:, 0]) & (
                    self.X_test[column_name].to_numpy() <= prediction_sets[:, 1]
                )
                coverage = true_value_in_prediction_range.mean()
                average_set_size = (prediction_sets[:, 1] - prediction_sets[:, 0]).mean()
                relative_average_set_size = average_set_size / cardinality
                empty_set_fraction = ((prediction_sets[:, 1] - prediction_sets[:, 0]) == 0).mean()

            prediction_set_metrics.update(
                {
                    f"colum_type__{column_name}": column_type,
                    f"cardinality__{column_name}": cardinality,
                    f"coverage__{column_name}": coverage,
                    f"average_set_size__{column_name}": average_set_size,
                    f"relative_average_set_size__{column_name}": relative_average_set_size,
                    f"empty_set_fraction__{column_name}": empty_set_fraction,
                }
            )

        return prediction_set_metrics

    def compute_and_get_results(self) -> tuple[pd.Series, pd.Series]:
        if self.error_mechanism is None:
            msg = "Run 'perturb_and_get_perturbed_test_data' first."
            raise RuntimeError(msg)

        experiment_setting = {
            "experiment_name": self.name,
            "cleaner_name": self._cleaner.__class__.__name__,
            "dataset_id": self.dataset_id,
            "task_type": self.task_type,
            "error_rate": self.error_rate,
            "error_mechanism": self.error_mechanism.__name__,
            "confidence_level": self.confidence_level,
            "train_size": self._train_size,
            "seed": f"{self._seed}-{random.randint(0, 0xFFFF):04X}" if self._seed is None else self._seed,  # just a random string so we can save the results
        }

        results = experiment_setting | self.calculate_and_get_metrics()
        prediction_sets_results = experiment_setting | self.calculate_and_get_prediction_set_metrics()

        return pd.Series(results), pd.Series(prediction_sets_results)

    def compute_and_save_results(self, results_path: Path = RESULTS_PATH) -> None:
        if type(results_path) is not Path:
            results_path = Path(results_path)

        results, prediciton_set_results = self.compute_and_get_results()
        results_dir_path: Path = (
            results_path
            / str(results["experiment_name"])
            / str(results["dataset_id"])
            / str(results["train_size"])
            / str(results["error_mechanism"])
            / str(results["error_rate"])
            / str(results["confidence_level"])
            / str(results["cleaner_name"])
            / str(results["seed"])
        )
        results_dir_path.mkdir(parents=True, exist_ok=True)

        results.to_csv(
            results_dir_path / "results.csv",
        )
        prediciton_set_results.to_csv(
            results_dir_path / "prediction_set_results.csv",
        )
