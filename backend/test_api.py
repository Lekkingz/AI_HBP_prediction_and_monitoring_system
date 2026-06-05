import numpy as np
import requests

# =========================
# Generate Fake PPG Signal
# =========================

ppg_signal = np.random.randn(875).tolist()

# =========================
# API Endpoint
# =========================

url = "http://127.0.0.1:5000/predict"

# =========================
# Send Request
# =========================

response = requests.post(
    url,
    json={
        "ppg_signal": ppg_signal
    }
)

# =========================
# Print Result
# =========================

print("\n===== AI PREDICTION RESPONSE =====\n")

print(response.json())
