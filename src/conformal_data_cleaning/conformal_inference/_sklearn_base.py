from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING, Any

from predictor import Predictor

from ._base import ConformalClassifier, ConformalRegressor
from .utils import calculate_q_hat, check_and_split_X_y

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


logger = getLogger(__name__)

SMALL_CALIB_SET_SIZE = 1000


class ConformalSKLearnClassifier(ConformalClassifier):
    _predictor: Predictor

    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        if y is None:
            msg = "This implementation of 'ConformalClassifier' requires to be called with 'y'."
            raise ValueError(msg)

        X_training, X_calibration, y_training, y_calibration = check_and_split_X_y(X, y, calibration_size)
        if X_calibration.shape[0] < SMALL_CALIB_SET_SIZE:
            msg = f"Calibration data has only {X_calibration.shape[0]} samples."
            logger.warning(msg)

        if self._fit:
            self._predictor.fit(X_training, y_training, **fit_params)

        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=self._predictor.predict_proba(X_calibration))

        return y_calibration, nonconformity_scores

    def _get_label_to_index_mapping(self) -> dict[Any, int]:
        return {x: index for index, x in enumerate(self._predictor.classes_)}

    def _predict_and_calculate_nonconformity_scores(
        self,
        X: ArrayLike,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        y_hat = self._predictor.predict_proba(X, **predict_params)
        nonconformity_scores = self._calculate_nonconformity_scores(y_hat=y_hat)

        return y_hat, nonconformity_scores


class ConformalSKLearnRegressor(ConformalRegressor):
    _predictor: Predictor

    def _fit_and_calculate_calibration_nonconformity_scores(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        if y is None:
            msg = "This implementation of 'ConformalRegressor' requires to be called with 'y'."
            raise ValueError(msg)

        X_training, X_calibration, y_training, y_calibration = check_and_split_X_y(X, y, calibration_size)
        if X_calibration.shape[0] < SMALL_CALIB_SET_SIZE:
            msg = f"Calibration data has only {X_calibration.shape[0]} samples."
            logger.warning(msg)

        if self._fit:
            self._predictor.fit(X_training, y_training, **fit_params)

        if self._conditional:
            self.variance_predictor = deepcopy(self._predictor)
            self.variance_predictor.fit(
                X_training,
                self._calculate_nonconformity_scores(y_training, self._predictor.predict(X_training)),
                **fit_params,
            )
            nonconformity_scores = self._calculate_nonconformity_scores(
                y_calibration,
                self._predictor.predict(X_calibration),
            ) / self.variance_predictor.predict(X_calibration)
        else:
            nonconformity_scores = self._calculate_nonconformity_scores(
                y=y_calibration,
                y_hat=self._predictor.predict(X_calibration),
            )

        return y_calibration, nonconformity_scores

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: float,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, float]:
        y_hat = self._predictor.predict(X, **predict_params)
        q_hat = calculate_q_hat(self.calibration_nonconformity_scores_, confidence_level)
        half_interval: float

        # returning `None` isn't a problem here.
        # This only happens when no nonconformity scores are given.
        if q_hat is None:
            msg = "q_hat is empty, which is not expected here."
            raise ValueError(msg)

        if self._conditional:
            variance_prediction = self.variance_predictor.predict(X, **predict_params)

            half_interval = variance_prediction * q_hat
        else:
            half_interval = q_hat

        return y_hat, half_interval
