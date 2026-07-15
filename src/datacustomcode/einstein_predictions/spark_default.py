# Copyright (c) 2025, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

from datacustomcode.einstein_predictions.spark_base import SparkEinsteinPredictions
from datacustomcode.einstein_predictions.types import (
    PredictionColumn,
    PredictionRequest,
    PredictionType,
)

if TYPE_CHECKING:
    from pyspark.sql import Column

    from datacustomcode.einstein_predictions.base import EinsteinPredictions
    from datacustomcode.einstein_predictions.types import PredictionResponse


_STATUS_SUCCESS = "SUCCESS"
_STATUS_ERROR = "ERROR"

# HTTP status considered a successful prediction call.
_HTTP_OK = 200


class DefaultSparkEinsteinPredictions(SparkEinsteinPredictions):

    CONFIG_NAME = "DefaultSparkEinsteinPredictions"

    def __init__(
        self,
        einstein_predictions: Optional["EinsteinPredictions"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if einstein_predictions is None:
            einstein_predictions = _build_underlying_predictions()
        self._einstein_predictions: "EinsteinPredictions" = einstein_predictions

    def einstein_predict(
        self,
        model_api_name: str,
        prediction_type: PredictionType,
        features: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _invoke_predictions(
            self._einstein_predictions,
            model_api_name,
            prediction_type,
            features,
            settings,
        )

    def einstein_predict_col(
        self,
        model_api_name: str,
        prediction_type: PredictionType,
        features: Dict[str, "Column"],
        settings: Optional[Dict[str, Any]] = None,
    ) -> "Column":
        """Build a per-row UDF that returns a struct ``{status, response,
        error_code, error_message}`` so per-row failures do not abort the
        Spark job. Callers select the field they want, e.g.
        ``einstein_predict_col(...)["response"]``. ``response`` carries the
        prediction response payload serialized as a JSON string.
        """
        from pyspark.sql.functions import struct, udf
        from pyspark.sql.types import (
            StringType,
            StructField,
            StructType,
        )

        feature_names = list(features.keys())
        values_col = struct(*[features[name].alias(name) for name in feature_names])

        predictions = self._einstein_predictions
        result_schema = StructType(
            [
                StructField("status", StringType(), True),
                StructField("response", StringType(), True),
                StructField("error_code", StringType(), True),
                StructField("error_message", StringType(), True),
            ]
        )

        def _predict(values_row: Any) -> Dict[str, Optional[str]]:
            if values_row is None:
                # An entirely null features struct is not the normal per-feature null
                # case; surface it directly rather than masking it (local debuggability).
                return {
                    "status": _STATUS_ERROR,
                    "response": None,
                    "error_code": None,
                    "error_message": "features column was null for this row",
                }
            row_features = (
                values_row.asDict()
                if hasattr(values_row, "asDict")
                else dict(values_row)
            )
            return _invoke_predictions_as_struct(
                predictions,
                model_api_name,
                prediction_type,
                row_features,
                settings,
            )

        return udf(_predict, result_schema)(values_col)


def _build_underlying_predictions() -> "EinsteinPredictions":
    from datacustomcode.einstein_predictions_config import einstein_predictions_config

    cfg = einstein_predictions_config.einstein_predictions_config
    if cfg is None:
        raise RuntimeError(
            "einstein_predictions_config is not configured. Add an "
            "'einstein_predictions_config' section to config.yaml."
        )
    return cfg.to_object()


def _feature_to_prediction_column(name: str, value: Any) -> PredictionColumn:
    """Wrap a single scalar feature value into a one-element prediction column.

    The value's Python type selects the prediction column value type. ``bool``
    is checked before ``int``/``float`` because ``bool`` is a subclass of
    ``int`` in Python.
    """
    if isinstance(value, bool):
        return PredictionColumn(column_name=name, boolean_values=[value])
    if isinstance(value, (int, float)):
        return PredictionColumn(column_name=name, double_values=[float(value)])
    return PredictionColumn(column_name=name, string_values=[str(value)])


def _build_request(
    model_api_name: str,
    prediction_type: PredictionType,
    features: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
) -> PredictionRequest:
    prediction_columns = [
        _feature_to_prediction_column(name, value) for name, value in features.items()
    ]
    return PredictionRequest(
        prediction_type=prediction_type,
        model_api_name=model_api_name,
        prediction_columns=prediction_columns,
        settings=settings,
    )


def _call_predictions(
    predictions: "EinsteinPredictions",
    model_api_name: str,
    prediction_type: PredictionType,
    features: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
) -> "PredictionResponse":
    """Build the request and dispatch it to the underlying predictions resource."""
    request = _build_request(model_api_name, prediction_type, features, settings)
    return predictions.predict(request)


def _null_feature_name(features: Dict[str, Any]) -> Optional[str]:
    """Return the name of the first null feature value, or ``None``."""
    for name, value in features.items():
        if value is None:
            return name
    return None


def _null_feature_message(name: str) -> str:
    return (
        f"Feature '{name}' has null value. Use coalesce() or when() to handle "
        f"nulls before calling einstein_predict."
    )


def _invoke_predictions(
    predictions: "EinsteinPredictions",
    model_api_name: str,
    prediction_type: PredictionType,
    features: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    from datacustomcode.einstein_predictions.errors import EinsteinPredictionsCallError

    null_feature = _null_feature_name(features)
    if null_feature is not None:
        message = _null_feature_message(null_feature)
        raise EinsteinPredictionsCallError(
            f"Einstein Predictions call failed: {message}",
            status=None,
            error_code=None,
            error_message=message,
        )

    try:
        response = _call_predictions(
            predictions, model_api_name, prediction_type, features, settings
        )
    except EinsteinPredictionsCallError:
        raise
    except Exception as exc:
        # Transport/build failures: surface the real error (no masking) so local runs stay
        # debuggable. error_code stays None since there is no HTTP status.
        raise EinsteinPredictionsCallError(
            f"Einstein Predictions call failed: {exc}",
            status=None,
            error_code=None,
            error_message=str(exc),
        )

    if response.status_code != _HTTP_OK:
        error_message = (
            json.dumps(response.data) if response.data is not None else None
        )
        raise EinsteinPredictionsCallError(
            f"Einstein Predictions call failed: "
            f"status_code={response.status_code}, message={error_message!r}",
            status=response.status_code,
            error_code=str(response.status_code),
            error_message=error_message,
        )
    return response.data or {}


def _invoke_predictions_as_struct(
    predictions: "EinsteinPredictions",
    model_api_name: str,
    prediction_type: PredictionType,
    features: Dict[str, Any],
    settings: Optional[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    # (a) Customer-actionable data condition — surface the actionable message directly.
    null_feature = _null_feature_name(features)
    if null_feature is not None:
        return {
            "status": _STATUS_ERROR,
            "response": None,
            "error_code": None,
            "error_message": _null_feature_message(null_feature),
        }

    # (b) Transport/build failures — surface the real error (no masking) so local runs stay
    # debuggable. error_code stays None since there is no HTTP status.
    try:
        response = _call_predictions(
            predictions, model_api_name, prediction_type, features, settings
        )
    except Exception as exc:
        return {
            "status": _STATUS_ERROR,
            "response": None,
            "error_code": None,
            "error_message": str(exc),
        }

    if response.status_code == _HTTP_OK:
        return {
            "status": _STATUS_SUCCESS,
            "response": (
                json.dumps(response.data) if response.data is not None else None
            ),
            "error_code": None,
            "error_message": None,
        }

    # (c) Non-200 SFAP HTTP error: error_code = status code, error_message = data JSON.
    return {
        "status": _STATUS_ERROR,
        "response": None,
        "error_code": str(response.status_code),
        "error_message": (
            json.dumps(response.data) if response.data is not None else None
        ),
    }
