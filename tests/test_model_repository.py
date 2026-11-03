"""Unit tests for churn pipeline fitting, independent of FastAPI."""

import pytest

import repository.model as model_repository


def test_train_churn_model_fits_pipeline(churn_rows):
    pipeline = model_repository.train_churn_model(churn_rows, "logreg", {"max_iter": 200})

    assert pipeline.predict(churn_rows.drop(columns="churn")).shape == (len(churn_rows),)
    assert set(pipeline.predict_proba(churn_rows.drop(columns="churn")).shape) == {2, len(churn_rows)}
    assert pipeline.model_type == "logreg"
    assert pipeline.model_hyperparameters["max_iter"] == 200


@pytest.mark.parametrize(
    ("frame_change", "message"),
    [
        ("missing", "missing columns: region"),
        ("single_class", "both churn classes"),
        ("null", "missing values"),
    ],
)
def test_train_churn_model_rejects_invalid_data(churn_rows, frame_change, message):
    data = churn_rows.copy()
    if frame_change == "missing":
        data = data.drop(columns="region")
    elif frame_change == "single_class":
        data["churn"] = 0
    else:
        data.loc[0, "usage_hours"] = None

    with pytest.raises(ValueError, match=message):
        model_repository.train_churn_model(data)


def test_train_churn_model_rejects_unknown_hyperparameters(churn_rows):
    with pytest.raises(ValueError, match="Unsupported logreg hyperparameters: typo"):
        model_repository.train_churn_model(churn_rows, hyperparameters={"typo": 1})
