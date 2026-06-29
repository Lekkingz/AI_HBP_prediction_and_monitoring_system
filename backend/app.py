import os
import traceback
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template

from model_utils import predict_bp
from fuzzy_logic import calculate_risk


# ======================================
# FLASK APP
# ======================================

app = Flask(__name__)


# ======================================
# GLOBAL LIVE DATA
# ======================================

latest_result = {

    "predicted_systolic_bp": 0,

    "heart_rate": 0,

    "temperature": 0,

    "respiratory_rate": 0,

    "risk_score": 0,

    "risk_level": "WAITING"
}


# ======================================
# DASHBOARD
# ======================================

@app.route('/dashboard')
def dashboard():

    return render_template(
        'dashboard.html'
    )


# ======================================
# HOME
# ======================================

@app.route('/')
def home():

    return "AI HBP Backend Running"


# ======================================
# LIVE DATA
# ======================================

@app.route('/live-data')
def live_data():

    return jsonify(
        latest_result
    )


# ======================================
# PREDICT ROUTE
# ======================================

@app.route(
    '/predict',
    methods=['POST']
)

def predict():

    global latest_result

    try:

        data = request.json

        print("========== REQUEST RECEIVED ==========")
        print(data)

        if data is None:
            raise Exception("No JSON received")

        # =========================
        # GET PPG SIGNAL
        # =========================

        ppg_signal = data.get(
            "ppg_signal"
        )

        if not isinstance(ppg_signal, list):
            raise Exception("ppg_signal must be a list of samples")

        print("PPG length:", len(ppg_signal))

        if len(ppg_signal) != 875:
            raise Exception(f"Expected 875 samples, got {len(ppg_signal)}")
        print("First sample:", ppg_signal[0])
        print("Last sample:", ppg_signal[-1])

        # =========================
        # GET HEART RATE
        # =========================

        heart_rate = data.get(
            "heart_rate"
        )

        # =========================
        # GET TEMPERATURE
        # =========================

        temperature = data.get(
            "temperature"
        )

        # =========================
        # FIX INVALID VALUES
        # =========================

        if heart_rate is None:

            heart_rate = 75

        if temperature is None:

            temperature = 36.5

        try:

            heart_rate = float(
                heart_rate
            )

        except:

            heart_rate = 75

        try:

            temperature = float(
                temperature
            )

        except:

            temperature = 36.5

        # =========================
        # RESPIRATORY RATE
        # =========================

        respiratory_rate = 16

        # =========================
        # AI PREDICTION
        # =========================

        predicted_bp = predict_bp(
            ppg_signal
        )
        print("Predicted BP:", predicted_bp)

        # =========================
        # FUZZY LOGIC
        # =========================

        fuzzy_result = calculate_risk(

            bp_value=predicted_bp,

            hr_value=heart_rate,

            temp_value=temperature,

            resp_value=respiratory_rate
        )
        print("Fuzzy Result:", fuzzy_result)

        # =========================
        # SAVE LIVE RESULT
        # =========================

        latest_result = {

            "predicted_systolic_bp":
                round(predicted_bp, 2),

            "heart_rate":
                round(heart_rate, 1),

            "temperature":
                round(temperature, 2),

            "respiratory_rate":
                respiratory_rate,

            "risk_score":
                fuzzy_result[
                    "risk_score"
                ],

            "risk_level":
                fuzzy_result[
                    "risk_level"
                ]
        }

        print(
            latest_result
        )

        return jsonify(
            latest_result
        )

    except Exception as e:

        print("=" * 80)
        print("FULL ERROR")
        traceback.print_exc()
        print("=" * 80)

        return jsonify({

            "error": str(e)

        }), 500


# ======================================
# RUN SERVER
# ======================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
