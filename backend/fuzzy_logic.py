import os
import traceback

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except Exception:
    fuzz = None
    ctrl = None
    print("Failed to import fuzzy logic dependencies:")
    traceback.print_exc()


BP_RANGE = (80.0, 220.0)
HEART_RATE_RANGE = (40.0, 180.0)
TEMPERATURE_RANGE = (34.0, 42.0)
RESPIRATORY_RATE_RANGE = (8.0, 40.0)

risk_ctrl = None


def _to_float(value, field_name, default_value):
    """Convert one vital sign to float and default invalid values safely."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        print(f"Invalid {field_name}; using default {default_value}")
        return float(default_value)

    if not np.isfinite(number):
        print(f"Non-finite {field_name}; using default {default_value}")
        return float(default_value)

    return number


def _clamp(value, field_name, lower, upper):
    """Clamp impossible sensor values before fuzzy inference."""

    clamped = min(
        max(value, lower),
        upper,
    )

    if clamped != value:
        print(f"Clamped {field_name} from {value} to {clamped}")

    return clamped


def _risk_level(score):
    """Map numeric risk score to the API risk level contract."""

    if score < 40:
        return "LOW"

    if score < 70:
        return "MODERATE"

    return "HIGH"


def _fallback_risk(bp_value, hr_value, temp_value, resp_value):
    """Return a safe risk value if fuzzy inference cannot produce output."""

    score = 15.0

    if bp_value >= 140:
        score += 35.0
    elif bp_value >= 120:
        score += 18.0

    if hr_value >= 110:
        score += 18.0
    elif hr_value >= 95:
        score += 8.0

    if temp_value >= 38.0:
        score += 18.0
    elif temp_value >= 37.5:
        score += 8.0

    if resp_value >= 24:
        score += 12.0
    elif resp_value >= 20:
        score += 5.0

    score = _clamp(
        score,
        "risk_score",
        0.0,
        100.0,
    )

    return {
        "risk_score": round(float(score), 2),
        "risk_level": _risk_level(score),
    }


def _build_control_system():
    """Build the fuzzy control system once while allowing import to survive."""

    if fuzz is None or ctrl is None:
        return None

    try:
        bp = ctrl.Antecedent(
            np.arange(80, 221, 1),
            "bp",
        )

        heart_rate = ctrl.Antecedent(
            np.arange(40, 181, 1),
            "heart_rate",
        )

        temperature = ctrl.Antecedent(
            np.arange(34, 42.1, 0.1),
            "temperature",
        )

        resp_rate = ctrl.Antecedent(
            np.arange(8, 41, 1),
            "resp_rate",
        )

        risk = ctrl.Consequent(
            np.arange(0, 101, 1),
            "risk",
        )

        bp["normal"] = fuzz.trimf(
            bp.universe,
            [80, 105, 120],
        )
        bp["elevated"] = fuzz.trimf(
            bp.universe,
            [115, 130, 145],
        )
        bp["high"] = fuzz.trimf(
            bp.universe,
            [140, 170, 220],
        )

        heart_rate["normal"] = fuzz.trimf(
            heart_rate.universe,
            [50, 75, 100],
        )
        heart_rate["high"] = fuzz.trimf(
            heart_rate.universe,
            [90, 120, 180],
        )

        temperature["normal"] = fuzz.trimf(
            temperature.universe,
            [35, 36.5, 37.5],
        )
        temperature["high"] = fuzz.trimf(
            temperature.universe,
            [37, 38.5, 42],
        )

        resp_rate["normal"] = fuzz.trimf(
            resp_rate.universe,
            [12, 16, 20],
        )
        resp_rate["high"] = fuzz.trimf(
            resp_rate.universe,
            [18, 25, 40],
        )

        risk["low"] = fuzz.trimf(
            risk.universe,
            [0, 20, 40],
        )
        risk["moderate"] = fuzz.trimf(
            risk.universe,
            [30, 50, 70],
        )
        risk["high"] = fuzz.trimf(
            risk.universe,
            [60, 80, 100],
        )

        rules = [
            ctrl.Rule(
                bp["normal"]
                & heart_rate["normal"]
                & temperature["normal"]
                & resp_rate["normal"],
                risk["low"],
            ),
            ctrl.Rule(
                bp["elevated"]
                | heart_rate["high"]
                | temperature["high"]
                | resp_rate["high"],
                risk["moderate"],
            ),
            ctrl.Rule(
                bp["high"] & heart_rate["high"],
                risk["high"],
            ),
            ctrl.Rule(
                bp["high"] & temperature["high"] & resp_rate["high"],
                risk["high"],
            ),
            ctrl.Rule(
                bp["high"],
                risk["high"],
            ),
        ]

        return ctrl.ControlSystem(
            rules
        )

    except Exception:
        print("Failed to build fuzzy control system:")
        traceback.print_exc()
        return None


def _validated_inputs(bp_value, hr_value, temp_value, resp_value):
    """Normalize and clamp all fuzzy inputs before inference."""

    bp_value = _to_float(
        bp_value,
        "bp",
        120.0,
    )
    hr_value = _to_float(
        hr_value,
        "heart_rate",
        75.0,
    )
    temp_value = _to_float(
        temp_value,
        "temperature",
        36.5,
    )
    resp_value = _to_float(
        resp_value,
        "respiratory_rate",
        16.0,
    )

    return (
        _clamp(bp_value, "bp", *BP_RANGE),
        _clamp(hr_value, "heart_rate", *HEART_RATE_RANGE),
        _clamp(temp_value, "temperature", *TEMPERATURE_RANGE),
        _clamp(resp_value, "respiratory_rate", *RESPIRATORY_RATE_RANGE),
    )


def calculate_risk(bp_value, hr_value, temp_value, resp_value):
    """Calculate risk without allowing fuzzy failures to crash Flask."""

    safe_bp, safe_hr, safe_temp, safe_resp = _validated_inputs(
        bp_value,
        hr_value,
        temp_value,
        resp_value,
    )

    try:
        if risk_ctrl is None:
            print("Fuzzy control system unavailable; using fallback risk.")
            return _fallback_risk(
                safe_bp,
                safe_hr,
                safe_temp,
                safe_resp,
            )

        risk_simulator = ctrl.ControlSystemSimulation(
            risk_ctrl
        )

        risk_simulator.input["bp"] = safe_bp
        risk_simulator.input["heart_rate"] = safe_hr
        risk_simulator.input["temperature"] = safe_temp
        risk_simulator.input["resp_rate"] = safe_resp
        risk_simulator.compute()

        score = risk_simulator.output.get(
            "risk"
        )

        if score is None or not np.isfinite(score):
            print("Fuzzy output missing or non-finite; using fallback risk.")
            return _fallback_risk(
                safe_bp,
                safe_hr,
                safe_temp,
                safe_resp,
            )

        score = _clamp(
            float(score),
            "risk_score",
            0.0,
            100.0,
        )

        return {
            "risk_score": round(float(score), 2),
            "risk_level": _risk_level(score),
        }

    except Exception:
        print("Fuzzy inference failed; using fallback risk.")
        traceback.print_exc()
        return _fallback_risk(
            safe_bp,
            safe_hr,
            safe_temp,
            safe_resp,
        )


risk_ctrl = _build_control_system()
