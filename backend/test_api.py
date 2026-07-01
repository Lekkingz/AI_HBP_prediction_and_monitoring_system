import app as app_module


def make_ppg_signal():
    return [0.0] * 875


def test_predict_uses_provided_respiratory_rate(monkeypatch):
    captured = {}

    monkeypatch.setattr(app_module, "predict_bp", lambda ppg_signal: 130.0)

    def fake_calculate_risk(bp_value, hr_value, temp_value, resp_value):
        captured["resp_value"] = resp_value
        return {
            "risk_score": 42.5,
            "risk_level": "MODERATE",
        }

    monkeypatch.setattr(app_module, "calculate_risk", fake_calculate_risk)

    client = app_module.app.test_client()
    response = client.post(
        "/predict",
        json={
            "ppg_signal": make_ppg_signal(),
            "heart_rate": 72.0,
            "temperature": 36.8,
            "respiratory_rate": 18.5,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["respiratory_rate"] == 18.5
    assert captured["resp_value"] == 18.5


def test_predict_defaults_missing_respiratory_rate(monkeypatch):
    captured = {}

    monkeypatch.setattr(app_module, "predict_bp", lambda ppg_signal: 130.0)

    def fake_calculate_risk(bp_value, hr_value, temp_value, resp_value):
        captured["resp_value"] = resp_value
        return {
            "risk_score": 42.5,
            "risk_level": "MODERATE",
        }

    monkeypatch.setattr(app_module, "calculate_risk", fake_calculate_risk)

    client = app_module.app.test_client()
    response = client.post(
        "/predict",
        json={
            "ppg_signal": make_ppg_signal(),
            "heart_rate": 72.0,
            "temperature": 36.8,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["respiratory_rate"] == 16.0
    assert captured["resp_value"] == 16.0
