from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_prediction():
    response = client.post(
        "/predict",
        json={
            "G1": 10,
            "G2": 11,
            "studytime": 2,
            "failures": 0,
            "absences": 4
        }
    )
    assert response.status_code == 200
    assert "predicted_final_grade" in response.json()

def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "G1": 30,
            "G2": 11,
            "studytime": 2,
            "failures": 0,
            "absences": 4
        }
    )
    assert response.status_code == 422
