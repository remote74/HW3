import pandas as pd

from fastapi import HTTPException
from pydantic import ValidationError
from repository import dataset as dataset_repository
from schemas import DatasetRowChurn


def load_dataset_or_503():
    try:
        return dataset_repository.get_dataset()
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc

def convert_to_dataset_rows(n: int) -> list[DatasetRowChurn]:
    """Convert DataFrame records to validated dataset schema objects."""
    df = load_dataset_or_503().head(n)
    return dataset_repository.convert_to_dataset_rows(df)

def get_dataset_info():
    df = load_dataset_or_503()
    counts = df["churn"].value_counts().sort_index()
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "churn_distribution": {str(label): int(count) for label, count in counts.items()},
    }

def prepare_train_test_split():
    return dataset_repository.prepare_train_test_split()

def get_split_info():
    X_train, X_test, y_train, y_test = dataset_repository.prepare_train_test_split()

    def distribution(labels):
        counts = labels.value_counts().sort_index()
        return {str(label): {"count": int(count), "proportion": float(count / len(labels))} for label, count in counts.items()}

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": distribution(y_train),
        "test_churn_distribution": distribution(y_test),
    }

