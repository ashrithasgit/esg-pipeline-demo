# tests/test_api_integration.py
import os
import time
import requests

BASE = os.getenv('APP_URL', 'http://localhost:8080')

def test_predict_schema():
    # short wait in case server just started
    time.sleep(1)
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    r = requests.post(f"{BASE}/predict", json=payload, timeout=10)
    assert r.status_code == 200, f"status {r.status_code} body:{r.text}"
    j = r.json()
    assert "prediction" in j
    assert isinstance(j["prediction"], list)
