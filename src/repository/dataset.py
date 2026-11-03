from pathlib import Path
import logging
import pandas as pd
from pydantic import ValidationError
import math
import numpy as np
from sklearn.model_selection import train_test_split
from schemas import DatasetRowChurn

NUMERIC_COLUMNS = [
    "monthly_fee", "usage_hours", "support_requests", "account_age_months",
    "failed_payments", "autopay_enabled",
]
CATEGORICAL_COLUMNS = ["region", "device_type", "payment_method"]
TARGET_COLUMN = "churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "churn_dataset.csv"
logger = logging.getLogger(__name__)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read and validate the dataset, raising a useful error on bad input."""
    if not path.is_file():
        logger.error("Churn dataset is unavailable: file does not exist")
        raise FileNotFoundError(f"Churn dataset not found: {path}")

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Failed to read churn dataset")
        raise ValueError(f"Could not read churn dataset: {exc}") from exc

    expected_columns = list(DatasetRowChurn.model_fields)
    missing = [column for column in expected_columns if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected_columns]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("Invalid churn dataset (" + "; ".join(details) + ")")
    if frame.empty:
        logger.error("Churn dataset contains no rows")
        raise ValueError("Invalid churn dataset: it contains no rows")

    # Validate each record using the same schema used by the API and later tasks.
    try:
        for row in frame.to_dict(orient="records"):
            DatasetRowChurn.model_validate(row)
    except ValidationError as exc:
        raise ValueError(f"Invalid row in churn dataset: {exc}") from exc

    # Keep a stable column order matching the schema.
    logger.info("Churn dataset loaded: rows=%d columns=%d", len(frame), len(expected_columns))
    return frame[expected_columns]


def get_dataset() -> pd.DataFrame:
    """Return the current dataset as a fresh DataFrame."""
    return load_dataset()


def convert_to_dataset_rows(frame: pd.DataFrame) -> list[DatasetRowChurn]:
    """Convert DataFrame records to validated dataset schema objects."""
    return [DatasetRowChurn.model_validate(record) for record in frame.to_dict(orient="records")]

def prepare_train_test_split(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Validate features and split without dropping or imputing customer rows.

    Missing or non-finite feature/target values are rejected explicitly. This
    keeps the source data intact and avoids silently inventing training values.
    """
    frame = load_dataset()
    if frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Cannot prepare churn data: missing values are present")
    numeric = frame[NUMERIC_COLUMNS].select_dtypes(include="number")
    if len(numeric.columns) != len(NUMERIC_COLUMNS) or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Cannot prepare churn data: numeric features must be finite numbers")

    X = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = frame[TARGET_COLUMN].copy()
    if TARGET_COLUMN in X.columns:
        raise ValueError("Target column must not be included among features")

    # Stratification requires at least two records per class and enough room
    # for every class in both resulting partitions.
    class_counts = y.value_counts()
    n_test = int(math.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 and n_test >= len(class_counts) and n_train >= len(class_counts) else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

