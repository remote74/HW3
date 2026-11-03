"""Unit tests for dataset reading and preparation, independent of FastAPI."""

import pandas as pd
import pytest

import repository.dataset as dataset_repository


def test_load_dataset_validates_and_preserves_schema_order(tmp_path, churn_rows):
    path = tmp_path / "churn.csv"
    churn_rows.to_csv(path, index=False)

    result = dataset_repository.load_dataset(path)

    assert len(result) == len(churn_rows)
    assert result.columns.tolist() == churn_rows.columns.tolist()


def test_load_dataset_rejects_missing_columns(tmp_path, churn_rows):
    path = tmp_path / "churn.csv"
    churn_rows.drop(columns=["region"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns: region"):
        dataset_repository.load_dataset(path)


def test_prepare_train_test_split_is_reproducible_and_excludes_target(monkeypatch, churn_rows):
    monkeypatch.setattr(dataset_repository, "load_dataset", lambda: churn_rows.copy())

    first = dataset_repository.prepare_train_test_split()
    second = dataset_repository.prepare_train_test_split()

    X_train, X_test, y_train, y_test = first
    assert "churn" not in X_train.columns
    assert len(X_train) + len(X_test) == len(churn_rows)
    pd.testing.assert_frame_equal(X_train, second[0])
    pd.testing.assert_series_equal(y_test, second[3])


def test_prepare_train_test_split_rejects_missing_values(monkeypatch, churn_rows):
    churn_rows.loc[0, "monthly_fee"] = float("nan")
    monkeypatch.setattr(dataset_repository, "load_dataset", lambda: churn_rows)

    with pytest.raises(ValueError, match="missing values"):
        dataset_repository.prepare_train_test_split()
