"""Shared test setup for the src-layout application."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def churn_rows():
    """A small, deterministic, schema-valid dataset with both target classes."""
    import pandas as pd

    rows = []
    for i in range(40):
        rows.append({
            "monthly_fee": 20.0 + i,
            "usage_hours": 10.0 + (i % 12),
            "support_requests": i % 5,
            "account_age_months": i + 1,
            "failed_payments": i % 3,
            "region": ["europe", "asia", "america", "africa"][i % 4],
            "device_type": ["mobile", "desktop", "tablet"][i % 3],
            "payment_method": ["card", "paypal", "crypto"][i % 3],
            "autopay_enabled": i % 2,
            "churn": i % 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def isolated_model_storage(tmp_path, monkeypatch):
    """Keep persisted model and history artifacts out of the developer's models/."""
    import repository.model as model_repository
    import repository.training_history as history_repository
    import service.model as model_service

    model_directory = tmp_path / "models"
    monkeypatch.setattr(model_repository, "MODEL_DIRECTORY", model_directory)
    monkeypatch.setattr(model_repository, "MODEL_MANIFEST_PATH", model_directory / "churn_model.json")
    monkeypatch.setattr(history_repository, "MODEL_DIRECTORY", model_directory)
    monkeypatch.setattr(history_repository, "TRAINING_HISTORY_PATH", model_directory / "training_history.json")
    model_service._loaded_model = None
    model_service._load_error = None
    model_service._training_status = {
        "available": False, "trained_at": None, "metrics": None,
        "model_type": None, "hyperparameters": {},
    }
    return model_directory
