from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from ._base import ConformalQuantileRegressor
from ._sklearn_base import ConformalSKLearnClassifier, ConformalSKLearnRegressor
from .utils import calculate_q_hat, check_and_split_X_y

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class ConformalRandomForestQuantileRegressor(ConformalQuantileRegressor):
    _predictor: RandomForestQuantileRegressor
    q_hat_for_confidence_level_: dict[float, float | None]
    calibration_nonconformity_scores_: dict[float, NDArray]

    def __init__(self, predictor_params: dict = {}) -> None:
        if type(predictor_params) is not dict:
            msg = "'predictor_params' need to be dictionary of arguments."
            raise ValueError(msg)

        super().__init__(RandomForestQuantileRegressor(**predictor_params), conditional=True, fit=True)

    def _get_or_calculate_q_hat_for_confidence_level(self, confidence_level: float) -> float:
        if q_hat := self.q_hat_for_confidence_level_.get(confidence_level, None):
            return q_hat

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        self.calibration_nonconformity_scores_[confidence_level] = self._calculate_nonconformity_scores(
            y_hat=self._predictor.predict(self._X_calibration, quantiles=[lower_quantile, 0.5, upper_quantile]),
            y=self._y_calibration,
        )

        # returning `None` isn't a problem here.
        # This only happens when no nonconformity scores are given.
        self.q_hat_for_confidence_level_[confidence_level] = calculate_q_hat(
            self.calibration_nonconformity_scores_[confidence_level],
            confidence_level,
        )

        return self.q_hat_for_confidence_level_[confidence_level]

    def _fit_method(
        self,
        X: ArrayLike,
        calibration_size: float = 0.2,
        y: ArrayLike | None = None,
        fit_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> None:
        if y is None:
            raise ValueError("This implementation of 'ConformalQuantileRegressor' requires to be called with 'y'.")

        X_training, X_calibration, y_training, y_calibration = check_and_split_X_y(X, y, calibration_size)

        self._predictor.fit(X_training, y_training, **fit_params)

        # bootstrap the necessary attributes
        self._X_calibration = X_calibration.copy()
        self._y_calibration = y_calibration.copy()
        self.calibration_nonconformity_scores_ = {}
        self.q_hat_for_confidence_level_ = {}

    def _predict_and_calculate_half_interval(
        self,
        X: ArrayLike,
        confidence_level: float | None = None,
        predict_params: dict = {},
        **kwargs: dict[str, Any],
    ) -> tuple[NDArray, float]:
        check_is_fitted(self)

        if confidence_level is None:
            msg = "'confidence_level' is expected here."
            raise ValueError(msg)

        lower_quantile, upper_quantile = self._calculate_lower_upper_quantiles(confidence_level)

        y_hat_quantiles = self._predictor.predict(X, quantiles=[lower_quantile, 0.5, upper_quantile], **predict_params)
        q_hat = self._get_or_calculate_q_hat_for_confidence_level(confidence_level)

        return y_hat_quantiles, q_hat


class ConformalRandomForestClassifier(ConformalSKLearnClassifier):
    _predictor: RandomForestClassifier

    def __init__(
        self,
        conditional: bool = True,
        fit: bool = True,
        predictor_params: dict[str, Any] = {},
    ) -> None:
        super().__init__(RandomForestClassifier(**predictor_params), conditional=conditional, fit=fit)


class ConformalRandomForestRegressor(ConformalSKLearnRegressor):
    _predictor: RandomForestRegressor

    def __init__(
        self,
        conditional: bool = True,
        fit: bool = True,
        predictor_params: dict[str, Any] = {},
    ) -> None:
        super().__init__(RandomForestRegressor(**predictor_params), conditional=conditional, fit=fit)
