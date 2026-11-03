"""FastAPI integration coverage for training, status, and churn prediction."""

from fastapi.testclient import TestClient

from main import app


def test_prediction_without_trained_model_returns_service_error(isolated_model_storage):
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "monthly_fee": 79.99, "usage_hours": 4.5,
            "support_requests": 3, "account_age_months": 2,
            "failed_payments": 2, "region": "europe",
            "device_type": "mobile", "payment_method": "crypto",
            "autopay_enabled": 0,
        })

    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_TRAINED"


def test_dataset_training_status_and_prediction_flow(isolated_model_storage):
    with TestClient(app) as client:
        dataset = client.get("/dataset/info")
        trained = client.post("/model/train", json={"model_type": "logreg"})
        status = client.get("/model/status")
        prediction = client.post("/predict", json={
            "monthly_fee": 79.99, "usage_hours": 4.5,
            "support_requests": 3, "account_age_months": 2,
            "failed_payments": 2, "region": "europe",
            "device_type": "mobile", "payment_method": "crypto",
            "autopay_enabled": 0,
        })

    assert dataset.status_code == 200
    assert dataset.json()["row_count"] > 0
    assert trained.status_code == 200
    assert trained.json()["available"] is True
    assert status.status_code == 200
    assert status.json()["model_type"] == "logreg"
    assert prediction.status_code == 200
    result = prediction.json()
    assert result["predicted_class"] in (0, 1)
    assert set(result["class_probabilities"]) == {"0", "1"}
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-6


def test_prediction_validation_error_has_stable_envelope(isolated_model_storage):
    with TestClient(app) as client:
        response = client.post("/predict", json={"monthly_fee": 12})

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.json()["details"]
