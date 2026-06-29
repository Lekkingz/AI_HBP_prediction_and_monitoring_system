import math
import os
import threading
import time
import traceback

import numpy as np
from flask import Flask
from flask import jsonify
from flask import request
from werkzeug.exceptions import HTTPException

from fuzzy_logic import calculate_risk
from model_utils import MODEL_INPUT_LENGTH
from model_utils import PredictionError
from model_utils import predict_bp


app = Flask(__name__)


# Use a lock so continuous ESP32 uploads cannot expose partially updated live
# data while the Flutter app is reading /live-data.
latest_result_lock = threading.Lock()
latest_result = {
    "predicted_systolic_bp": 0,
    "heart_rate": 0,
    "temperature": 0,
    "respiratory_rate": 0,
    "risk_level": "WAITING",
    "risk_score": 0,
}


class RequestValidationError(Exception):
    """Raised for client payload problems that should return HTTP 400."""


def json_error(message, status_code):
    """Return every error as JSON so Gunicorn never serves Flask HTML errors."""

    response = jsonify(
        {
            "error": str(message),
        }
    )
    response.status_code = status_code
    return response


def _payload_size_bytes():
    """Measure request size without consuming the cached Flask request body."""

    payload = request.get_data(
        cache=True,
        as_text=False,
    )
    return len(payload or b"")


def _parse_json_payload():
    """Validate Content-Type and parse JSON safely."""

    if not request.is_json:
        raise RequestValidationError(
            "Content-Type must be application/json"
        )

    data = request.get_json(
        silent=True,
    )

    if data is None:
        raise RequestValidationError(
            "Invalid or empty JSON body"
        )

    if not isinstance(data, dict):
        raise RequestValidationError(
            "JSON body must be an object"
        )

    return data


def _validate_ppg_signal(data):
    """Validate ESP32 ppg_signal and convert samples to float32."""

    if "ppg_signal" not in data:
        raise RequestValidationError(
            "ppg_signal is required"
        )

    ppg_signal = data.get(
        "ppg_signal"
    )

    if not isinstance(ppg_signal, list):
        raise RequestValidationError(
            "ppg_signal must be a list of samples"
        )

    sample_count = len(
        ppg_signal
    )

    print("Number of samples:", sample_count)

    if sample_count != MODEL_INPUT_LENGTH:
        raise RequestValidationError(
            f"Expected {MODEL_INPUT_LENGTH} samples, got {sample_count}"
        )

    for index, sample in enumerate(ppg_signal):
        if isinstance(sample, bool):
            raise RequestValidationError(
                f"ppg_signal sample {index} must be numeric"
            )

        if not isinstance(sample, (int, float)):
            raise RequestValidationError(
                f"ppg_signal sample {index} must be numeric"
            )

        if not math.isfinite(float(sample)):
            raise RequestValidationError(
                f"ppg_signal sample {index} must be finite"
            )

    ppg_array = np.asarray(
        ppg_signal,
        dtype=np.float32,
    )

    print("First sample:", float(ppg_array[0]))
    print("Last sample:", float(ppg_array[-1]))

    return ppg_array


def _optional_float(data, field_name, default_value):
    """Parse optional ESP32 numeric fields without rejecting older firmware."""

    value = data.get(
        field_name,
        default_value,
    )

    try:
        number = float(value)
    except (TypeError, ValueError):
        print(f"Invalid {field_name}; using default {default_value}")
        return float(default_value)

    if not math.isfinite(number):
        print(f"Non-finite {field_name}; using default {default_value}")
        return float(default_value)

    return number


def _format_heart_rate(value):
    """Keep API output compact: 75 instead of 75.0 when possible."""

    rounded = round(
        float(value),
        1,
    )

    if rounded.is_integer():
        return int(rounded)

    return rounded


@app.errorhandler(Exception)
def handle_uncaught_exception(exc):
    """Catch any uncaught Flask exception and return JSON instead of HTML."""

    if isinstance(exc, HTTPException):
        return json_error(
            exc.description,
            exc.code or 500,
        )

    print("Uncaught Flask exception:")
    traceback.print_exc()
    return json_error(
        "Internal server error",
        500,
    )


@app.route("/")
def home():
    return "AI HBP Backend Running"


@app.route("/live-data")
def live_data():
    with latest_result_lock:
        result_copy = dict(
            latest_result
        )

    return jsonify(
        result_copy
    )


@app.route(
    "/predict",
    methods=["POST"],
)
def predict():
    global latest_result

    start_time = time.perf_counter()

    try:
        print("Request received")
        print("Payload size:", _payload_size_bytes(), "bytes")

        data = _parse_json_payload()
        ppg_signal = _validate_ppg_signal(
            data
        )

        heart_rate = _optional_float(
            data,
            "heart_rate",
            75.0,
        )
        temperature = _optional_float(
            data,
            "temperature",
            36.5,
        )
        respiratory_rate = 16

        predicted_bp = predict_bp(
            ppg_signal
        )

        fuzzy_result = calculate_risk(
            bp_value=predicted_bp,
            hr_value=heart_rate,
            temp_value=temperature,
            resp_value=respiratory_rate,
        )
        print("Fuzzy output:", fuzzy_result)

        response_data = {
            "predicted_systolic_bp": round(
                float(predicted_bp),
                2,
            ),
            "heart_rate": _format_heart_rate(
                heart_rate
            ),
            "temperature": round(
                float(temperature),
                2,
            ),
            "respiratory_rate": respiratory_rate,
            "risk_level": fuzzy_result.get(
                "risk_level",
                "MODERATE",
            ),
            "risk_score": round(
                float(
                    fuzzy_result.get(
                        "risk_score",
                        50.0,
                    )
                ),
                2,
            ),
        }

        with latest_result_lock:
            latest_result = dict(
                response_data
            )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0
        print(f"Prediction completed in {elapsed_ms:.2f} ms")

        return jsonify(
            response_data
        )

    except RequestValidationError as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0
        print(f"Prediction failed in {elapsed_ms:.2f} ms")
        print("Validation error:", str(exc))
        return json_error(
            str(exc),
            400,
        )

    except PredictionError as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0
        print(f"Prediction failed in {elapsed_ms:.2f} ms")
        print("Prediction error:")
        traceback.print_exc()
        return json_error(
            str(exc),
            500,
        )

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0
        print(f"Prediction failed in {elapsed_ms:.2f} ms")
        print("Full traceback:")
        traceback.print_exc()
        return json_error(
            str(exc),
            500,
        )


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            5000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
