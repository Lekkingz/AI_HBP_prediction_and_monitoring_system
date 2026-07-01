import os
import threading
import time
import traceback

import numpy as np


# Keep TensorFlow quieter under Gunicorn while still allowing real Python
# exceptions and tracebacks to be printed by this module.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


MODEL_INPUT_LENGTH = 875
MODEL_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset",
        "models",
        "cnn_bilstm_model.h5",
    )
)

model = None
model_load_error = None
model_ready = False
model_warming = False
model_warmup_started_at = None
model_lock = threading.Lock()
MODEL_WARMUP_STALE_SECONDS = 60


class PredictionError(Exception):
    """Raised when prediction cannot be completed safely."""


class ModelUnavailableError(PredictionError):
    """Raised when the model is loading or warming up."""


def load_prediction_model():
    """Load TensorFlow once at startup without crashing the Flask import."""

    global model
    global model_load_error
    global model_ready

    if not os.path.exists(MODEL_FILE):
        model_load_error = f"Model file not found: {MODEL_FILE}"
        model_ready = False
        print("Failed to load model:")
        print(model_load_error)
        return None

    try:
        try:
            from keras.models import load_model
            print("Using standalone keras model loader.")
        except Exception:
            from tensorflow.keras.models import load_model
            print("Using tensorflow.keras model loader.")

        try:
            import tensorflow as tf

            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except Exception:
            print("TensorFlow thread limits unavailable:")
            traceback.print_exc()

        # compile=False avoids loading training-only optimizer/loss state that
        # can break between TensorFlow/Keras versions on Render.
        loaded_model = load_model(
            MODEL_FILE,
            compile=False,
        )

        model = loaded_model
        model_ready = False
        model_load_error = None
        print("Model loaded successfully.")
        return model

    except Exception as exc:
        model = None
        model_ready = False
        model_load_error = str(exc)
        print("Failed to load model:")
        traceback.print_exc()
        return None


def get_model():
    """Return the loaded model, retrying once if startup loading failed."""

    if model is not None:
        return model

    return load_prediction_model()


def warm_up_model():
    """Run one dummy prediction outside the ESP32 request path."""

    global model_ready
    global model_warming
    global model_warmup_started_at
    global model_load_error

    with model_lock:
        if model_ready or model_warming:
            return

        model_warming = True
        model_warmup_started_at = time.monotonic()

    try:
        loaded_model = get_model()

        if loaded_model is None:
            raise ModelUnavailableError(
                f"Prediction model is not loaded: {model_load_error}"
            )

        dummy_input = np.zeros(
            (1, MODEL_INPUT_LENGTH, 1),
            dtype=np.float32,
        )

        print("Warming prediction model...")
        print("Model input shape:", dummy_input.shape)

        loaded_model.predict(
            dummy_input,
            verbose=0,
        )

        with model_lock:
            model_ready = True
            model_load_error = None

        print("Model warmup completed successfully.")

    except Exception as exc:
        with model_lock:
            model_ready = False
            model_load_error = str(exc)

        print("Model warmup failed:")
        traceback.print_exc()

    finally:
        with model_lock:
            model_warming = False
            model_warmup_started_at = None


def start_model_warmup():
    """Start warmup in the background so Gunicorn can answer quickly."""

    thread = threading.Thread(
        target=warm_up_model,
        daemon=True,
    )
    thread.start()


def ensure_model_ready():
    """Fail fast while TensorFlow is warming instead of timing out on Render."""

    global model_warming
    global model_warmup_started_at

    with model_lock:
        ready = model_ready
        warming = model_warming
        warmup_started_at = model_warmup_started_at
        error = model_load_error

    if ready:
        return

    if warming and warmup_started_at is not None:
        warming_for = time.monotonic() - warmup_started_at

        if warming_for > MODEL_WARMUP_STALE_SECONDS:
            print("Model warmup appears stale; restarting warmup thread.")

            with model_lock:
                model_warming = False
                model_warmup_started_at = None
                warming = False

    if not warming:
        start_model_warmup()

    if error:
        raise ModelUnavailableError(
            f"Prediction model is warming up or unavailable: {error}"
        )

    raise ModelUnavailableError(
        "Prediction model is warming up. Try again shortly."
    )


def prepare_model_input(ppg_signal):
    """Validate and reshape PPG input before TensorFlow sees it."""

    if not isinstance(ppg_signal, (list, np.ndarray)):
        raise PredictionError("ppg_signal must be a list or numpy array")

    sample_count = len(ppg_signal)

    if sample_count != MODEL_INPUT_LENGTH:
        raise PredictionError(
            f"Expected {MODEL_INPUT_LENGTH} samples, got {sample_count}"
        )

    try:
        ppg_array = np.asarray(
            ppg_signal,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        print("Failed to convert ppg_signal to float32:")
        traceback.print_exc()
        raise PredictionError("ppg_signal contains a non-numeric sample") from exc

    if ppg_array.shape != (MODEL_INPUT_LENGTH,):
        raise PredictionError(
            f"Expected flat ppg_signal with {MODEL_INPUT_LENGTH} samples"
        )

    if not np.all(np.isfinite(ppg_array)):
        raise PredictionError("ppg_signal contains NaN or infinite values")

    model_input = ppg_array.reshape(
        1,
        MODEL_INPUT_LENGTH,
        1,
    )

    print("Model input shape:", model_input.shape)
    return model_input


def predict_bp(ppg_signal):
    """Run the blood-pressure model and return systolic BP as a float."""

    try:
        ensure_model_ready()

        loaded_model = get_model()

        if loaded_model is None:
            raise ModelUnavailableError(
                f"Prediction model is not loaded: {model_load_error}"
            )

        model_input = prepare_model_input(
            ppg_signal
        )

        prediction = loaded_model.predict(
            model_input,
            verbose=0,
        )

        prediction_array = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if prediction_array.size < 1:
            raise PredictionError("Model returned an empty prediction")

        raw_prediction = float(
            prediction_array.reshape(-1)[0]
        )

        if not np.isfinite(raw_prediction):
            raise PredictionError("Model returned NaN or infinite prediction")

        predicted_bp = (raw_prediction * 100.0) + 80.0

        if not np.isfinite(predicted_bp):
            raise PredictionError("Converted BP prediction is not finite")

        print("Prediction:", predicted_bp)
        return float(predicted_bp)

    except PredictionError:
        traceback.print_exc()
        raise

    except Exception as exc:
        print("Model prediction failed:")
        traceback.print_exc()
        raise PredictionError(str(exc)) from exc
