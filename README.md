# AI Blood Pressure Prediction and Monitoring System

An end-to-end research prototype for estimating systolic blood pressure from
photoplethysmography (PPG) signals, combining machine-learning inference with
fuzzy-logic risk assessment and live monitoring interfaces.

## Features

- CNN, BiLSTM, CNN-BiLSTM, and LSTM model training pipelines
- Flask API for PPG-based blood pressure prediction
- Fuzzy-logic risk scoring from blood pressure and vital signs
- Browser dashboard for live readings
- Flutter monitoring application for Android, iOS, web, and desktop
- Processed sample data and trained model artifacts

## Project Structure

```text
ai_model/        Dataset preparation, training, evaluation, and visualization
backend/         Flask API, dashboard, model inference, and fuzzy risk logic
dashboard/       Reserved for future standalone dashboard work
dataset/
  models/        Trained Keras model files
  processed/     Prepared arrays and the smaller PPG subset
  raw/           Local-only source dataset; not stored in Git
docs/            Project figures and comparison results
esp32_code/      Reserved for sensor firmware
hbp_mobile_app/  Flutter live-monitoring application
tests/           Reserved for project-level tests
```

## Prerequisites

- Python 3.10 or newer
- Flutter SDK compatible with Dart `^3.13.0-103.1.beta`
- Android Studio, Xcode, or another Flutter-supported platform toolchain
- Git

## Backend Setup

Create an isolated environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the API from the `backend` directory. This working directory is required
because the current model loader uses a relative path.

```bash
cd backend
python app.py
```

Available endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Backend health message |
| `GET` | `/dashboard` | Browser monitoring dashboard |
| `GET` | `/live-data` | Most recent prediction and risk result |
| `POST` | `/predict` | Predict from an 875-sample PPG signal |

Example prediction request:

```python
import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "ppg_signal": [0.0] * 875,
        "heart_rate": 75,
        "temperature": 36.5,
    },
)

print(response.json())
```

The `ppg_signal` value must contain exactly 875 numeric samples. A complete test
request can also be sent with:

```bash
cd backend
python test_api.py
```

## Flutter Application

Install packages and run the application:

```bash
cd hbp_mobile_app
flutter pub get
flutter run
```

Before running on a phone or emulator, update the backend URL in
`hbp_mobile_app/lib/main.dart`. It currently uses:

```text
http://192.168.1.164:5000/live-data
```

Use the development machine's LAN address for a physical device. Android
emulators commonly use `http://10.0.2.2:5000`, while Flutter web can use
`http://127.0.0.1:5000` when served from the same machine.

## Dataset

The full raw PPG dataset is intentionally excluded from Git because the local
files total approximately 57 GB and exceed GitHub's normal storage limits.
Processed data and trained models required by the current prototype are
included in this repository.

To reproduce preprocessing or model training:

1. Obtain the dataset through its authorized source and comply with its access
   terms.
2. Place the source file at
   `dataset/raw/MIMIC-III_ppg_dataset.h5`.
3. Run scripts from `ai_model/` so their relative paths resolve correctly.

```bash
cd ai_model
python create_subset.py
python preprocess_data.py
python train_cnn_bilstm.py
```

Do not commit raw clinical datasets or upload them to a public file-sharing
service. For private team access, use access-controlled object storage and
document the retrieval process without committing credentials.

## Validation

Run the fuzzy-logic test:

```bash
cd backend
python test_fuzzy.py
```

Run Flutter checks:

```bash
cd hbp_mobile_app
flutter analyze
flutter test
```

## Important Notice

This repository is an academic and research prototype. Its predictions and risk
scores are not clinically validated and must not be used for diagnosis,
treatment decisions, or emergency assessment. Consult a qualified healthcare
professional for medical guidance.

## Repository Policy

The repository tracks source code, configuration, trained models, and selected
processed data. Virtual environments, build output, IDE metadata, caches, and
the full raw dataset remain local and can be recreated or obtained separately.
